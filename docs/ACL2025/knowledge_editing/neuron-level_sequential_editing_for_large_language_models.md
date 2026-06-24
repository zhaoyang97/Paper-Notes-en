---
title: >-
  [Paper Note] Neuron-Level Sequential Editing for Large Language Models
description: >-
  [ACL2025][Knowledge Editing][NSE] This work proposes the NSE method for sequential model editing in LLMs, which prevents model collapse through weights rewinding, mitigates model forgetting via activation-based neuron-level selective weight updates, and improves the success rate of large-scale knowledge updates through iterative multi-layer editing.
tags:
  - "ACL2025"
  - "Knowledge Editing"
  - "NSE"
  - "Sequential Model Editing"
  - "Neuron Selection"
  - "Weights Rewinding"
  - "Iterative Multi-Layer Editing"
date: 2026-05-08
content_hash: 1f5224f3d95ab271
---

# Neuron-Level Sequential Editing for Large Language Models

**Conference**: ACL2025  
**arXiv**: [2410.04045](https://arxiv.org/abs/2410.04045)  
**Code**: [GitHub](https://github.com/jianghoucheng/NSE)  
**Area**: Model Editing / Knowledge Updating / LLM  
**Keywords**: NSE, Sequential Model Editing, Neuron Selection, Weights Rewinding, Iterative Multi-Layer Editing  

## TL;DR

This work proposes the NSE method for sequential model editing in LLMs, which prevents model collapse through weights rewinding, mitigates model forgetting via activation-based neuron-level selective weight updates, and improves the success rate of large-scale knowledge updates through iterative multi-layer editing.

## Background & Motivation

### Background
LLMs store vast amounts of factual knowledge during pre-training. However, real-world knowledge constantly evolves, rendering internal knowledge outdated or incorrect. Since full retraining is prohibitively expensive, model editing (modifying specific knowledge without retraining) has emerged as a research hotspot.

### Challenges in Sequential Editing
Existing model editing methods (such as ROME, MEMIT) perform well in single-round editing. However, practical applications require consecutive multiple rounds of editing:

**Model Forgetting**: Cumulative parameter updates cause the model to forget previously edited knowledge.

**Model Failure**: Excessive edits impair the model's ability to generate coherent text, even leading to gibberish outputs.

### Why Existing Methods Fail
- **ROME/MEMIT**: Directly modify FFN layer weights, where parameter changes accumulate over editing rounds, eventually leading to catastrophic degradation.
- **Memory-based Methods (GRACE)**: Retain original parameters but store increments, leading to extremely high storage demands as the number of edits grows.
- **Key Challenge**: Modifying enough parameters is necessary to edit knowledge, but excessive modifications destroy the model's functionality.

## Method

### Preliminaries

#### Autoregressive Language Models
In an $L$-layer Transformer, the hidden state of the $l$-th layer is defined as:
$$\mathbf{h}_t^l = \mathbf{h}_t^{l-1} + \mathbf{a}_t^l + \mathbf{v}_t^l$$

The FFN layer output is $\mathbf{v}_t^l = \mathbf{W}_{\text{out}}^l \sigma(\mathbf{W}_{\text{in}}^l \gamma(\mathbf{h}_t^{l-1} + \mathbf{a}_t^l))$.

#### Knowledge Storage Perspective
The linear operation of the FFN layer can be viewed as a key-value store:
- **Key** $\mathbf{k}_i^l$: The activation output of the last token of the input subject in the $l$-th layer FFN.
- **Value** $\mathbf{v}_i^l$: The output processed by $\mathbf{W}_{\text{out}}^l$.

The objective of editing is to introduce a new key-value pair $(\mathbf{K}_1, \mathbf{V}_1)$ while retaining the old key-value pairs $(\mathbf{K}_0, \mathbf{V}_0)$:
$$\Delta^* = \arg\min_\Delta \left(\|(\mathbf{W}+\Delta)\mathbf{K}_1 - \mathbf{V}_1\|^2 + \|(\mathbf{W}+\Delta)\mathbf{K}_0 - \mathbf{V}_0\|^2\right)$$

### NSE Method

#### 1. Weights Rewinding for Value Computation

**Problem**: In sequential editing, computing the target value $\mathbf{z}_i$ using the currently updated model parameters $f_{\theta_t}$ leads to severe degradation. Cumulative parameter changes shift the value computation.

**Solution**: Always use the original model parameters $f_{\theta_0}$ to compute the target value:
$$\mathbf{z}_i = \mathbf{h}_i^l + \arg\min_{\delta_i}\left(-\log \mathbb{P}_{f_{\theta_0}(\mathbf{h}_i^l += \delta_i)}[o_i|(s_i, r_i)]\right)$$

**Implementation Details**:
- Only the $\mathbf{W}_{\text{out}}$ weight matrices that need updating must be saved, avoiding the need to store the entire model.
- The original weights are used only during the computation of $\mathbf{z}_i$.
- $\delta_i$ is optimized via gradient descent to maximize the probability of the model outputting the target $o_i$.

#### 2. Neuron-Level Weights Updating

**Core Idea**: Rather than modifying the entire weight matrix, only a small subset of "influential neurons" is selectively modified.

**Neuron Selection**: A score $\mathbf{Q}_i = |\mathbf{k}_i|$ is computed based on the activation $\mathbf{k}_i$. The minimum subset of neurons $\mathcal{I}$ is selected such that their cumulative score reaches a predefined percentage $p$ of the total score:
$$\mathcal{I} = \arg\min_{\mathcal{I} \subseteq \{1,...,N\}} |\mathcal{I}| \quad \text{s.t.} \quad \sum_{j \in \mathcal{I}} \mathbf{Q}_{ij} \geq p \times \sum_{j=1}^N \mathbf{Q}_{ij}$$

**Batch Editing**: For multiple knowledge facts in a batch, neuron scores from all samples are aggregated to perform a unified selection.

**Sub-matrix Update**: Only the sub-matrix corresponding to the selected neurons is updated:
$$\hat{\Delta}^* = \hat{\mathbf{R}} \hat{\mathbf{K}}_1^T \hat{\mathbf{C}}^{-1}$$

where $\hat{\mathbf{R}} = \mathbf{V}_1 - \hat{\mathbf{W}} \hat{\mathbf{K}}_1$, and $\hat{\mathbf{C}} = \hat{\mathbf{K}}_0\hat{\mathbf{K}}_0^T + \hat{\mathbf{K}}_1\hat{\mathbf{K}}_1^T$.

**Knowledge Accumulation**: After each round of editing, the newly edited knowledge is added to $\mathbf{K}_0\mathbf{K}_0^T$, becoming the "old knowledge" for the next round.

#### 3. Iterative Multi-Layer Editing

**Limitations of MEMIT**: MEMIT evenly distributes the residual $\delta_i$ across multiple layers ($\mathbf{v}_i^l += \frac{\delta_i}{l_0 - l + 1}$). However, some knowledge is hard to edit, and fitting errors lead to editing failures.

**Iterative Strategy**:
1. Execute one round of multi-layer editing.
2. Check the residual $\|\mathbf{z}_i - \mathbf{h}_i^{l_0}\|^2$ for each sample.
3. If $< \alpha$: Successful edit; if $> \alpha$: Unsuccessful.
4. Form a new batch with the unsuccessful samples and repeat the multi-layer editing.
5. Continue until all samples succeed or the maximum number of iterations is reached.

## Experiments

### Experimental Setup
- **Models**: GPT2-XL (1.5B), GPT-J (6B), Llama3 (8B)
- **Datasets**: Counterfact, ZsRE (2,000 edits each)
- **Batch size**: 100 (editing 100 knowledge facts simultaneously per round)
- **Evaluation Metrics**: Efficacy (success rate), Generalization, Specificity, Fluency, Consistency
- **Baselines**: FT-L, FT-W, MEND, ROME, MEMIT, GRACE

### Main Results (Llama3, Counterfact)

| Method | Efficacy↑ | Generalization↑ | Specificity↑ | Fluency↑ | Consistency↑ |
|------|-----------|----------------|-------------|---------|-------------|
| MEMIT | 65.65 | 64.65 | 51.56 | 437.43 | 6.58 |
| ROME | 64.40 | 61.42 | 49.44 | 449.06 | 3.31 |
| GRACE | 90.72 | 0.09 | 87.23 | 632.43 | 23.79 |
| **NSE** | **96.14** | **78.42** | **87.66** | **632.72** | **30.20** |

Key Findings:

1. **Overall Lead**: NSE outperforms all baselines on almost all metrics, providing an average gain of approximately 30.33% on Llama3.
2. **Preservation of Generation Ability**: Fluency (632.72) and Consistency (30.20) are significantly better than parameter modification methods, yielding a gain of over 40.75%.
3. **Fatal Flaw of GRACE**: Generalization is extremely low (0.09) because it does not modify parameters, failing when the phrasing of the prompt is changed.
4. **Degradation of MEMIT/ROME**: Performance drops sharply as the number of editing rounds increases (especially in Specificity and Fluency).

### Results on Different Models

| Model | NSE Efficacy | NSE Specificity | MEMIT Efficacy | MEMIT Specificity |
|------|-------------|-----------------|----------------|-------------------|
| GPT2-XL | 96.80 | 72.10 | 94.70 | 60.50 |
| GPT-J | 99.55 | 78.96 | 98.55 | 63.64 |
| Llama3 | 96.14 | 87.66 | 65.65 | 51.56 |

NSE maintains outstanding performance across all models, with the most pronounced advantage on Llama3.

### Impact of Batch Size
- MEMIT experiences a sharp performance drop when batch size = 10 (which implies more editing rounds).
- NSE remains stable across various batch sizes, achieving an average gain of 45.60% when batch size = 10.

### Ablation Study (Llama3, Counterfact)

| Variant | Eff. | Gen. | Spe. | Flu. | Consis. |
|------|------|------|------|------|---------|
| NSE (Full) | 96.14 | 78.42 | 87.66 | 632.72 | 30.20 |
| w/o weights rewinding | 98.90↑ | 91.18↑ | **76.60↓↓** | **625.65↓** | 32.30↑ |
| w/o neuron update | 96.00 | 77.13↓ | 87.68 | - | - |

Key Findings:
- **Weights Rewinding is Crucial**: Removing it drops Specificity by 11.06, indicating that computing values with updated weights causes drift.
- Although Efficacy and Generalization increase, it comes at the expense of Specificity and Fluency (overfitting to new knowledge).
- **Stable Contribution of Neuron Selection**: Removing it leads to minor decreases across all metrics.

### General Ability Testing
When evaluating the edited models on general benchmarks such as MMLU and CMMLU, the model edited by NSE maintains its original general abilities, whereas the general abilities of models edited by ROME and MEMIT experience a significant drop.

## Highlights & Insights

1. **Core Insight of Weights Rewinding**: Restores original weights as "anchors" during sequential editing, resolving model collapse caused by cumulative parameter changes that contaminate value computation.
2. **Elegance of Neuron Selection**: Sparse, activation-based selection (typically modifying only a fraction of neurons) balances precise editing with the preservation of model capabilities.
3. **Adaptability of Iterative Editing**: Automatically adjusts to the difficulty of editing different knowledge instances using a residual threshold.
4. **Synergy of Three Components**: Weights rewinding to prevent collapse + neuron selection to mitigate forgetting + iterative editing to boost success rate — a highly interconnected design.

## Limitations & Future Work

1. Needs to store original weight matrices $\mathbf{W}_{\text{out}}$, which increases storage overhead.
2. Iterative multi-layer editing increases computational time.
3. Hyperparameters (selection percentage $p$, residual threshold $\alpha$) need fine-tuning for different models.
4. Experiments are verified only on factual knowledge editing (subject-relation-object) and are not extended to more complex scenarios (such as behavior modification).
5. As the number of edits continuously grows, the long-term effectiveness still requires larger-scale validation.

## Related Work & Insights

- **Knowledge Localization**: Causal tracing to identify critical layers (Meng et al., 2022), treating FFNs as key-value stores.
- **Parameter Modification Methods**: ROME (single-layer editing), MEMIT (multi-layer editing), Fine-tuning.
- **Parameter Preservation Methods**: GRACE (memory-based), MEND (hypernetwork).
- **Continual Learning**: Catastrophic forgetting, Elastic Weight Consolidation (EWC).
- **Neuron Analysis**: Knowledge neurons, information storage in FFNs.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Evaluation | ⭐⭐⭐⭐ |

> NSE is a solid methodology paper. Its three technical components each address a core problem in sequential editing (collapse, forgetting, and failure) with designs that are simple, intuitive, and highly reproducible. The experiments are comprehensively validated across 3 models $\times$ 2 datasets with clear ablation analysis. While the idea of weights rewinding seems straightforward, the underlying insight (the contamination effect of cumulative updates on value computation) is profound and critical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](../../ICML2026/knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)

</div>

<!-- RELATED:END -->
