---
title: >-
  [Paper Note] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models
description: >-
  [ACL 2025 (Long Paper)][LLM Safety][Machine Unlearning] Proposes MANU, the first modality-aware unlearning framework for MLLMs. It identifies cross-modality entangled knowledge-carrying neurons through four complementary neuron importance functions (absolute, frequency, variance, and RMS), and selectively prunes the top-$\alpha\%$ neurons to achieve balanced unlearning under both multimodal and text-only inputs, completely training-free without any gradient updates.
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM Safety"
  - "Machine Unlearning"
  - "MLLM"
  - "Neuron Pruning"
  - "Modality-Aware"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 1fdd23b661dfab4e
---

# Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2502.15910](https://arxiv.org/abs/2502.15910)  
**Code**: [GitHub](https://github.com/franciscoliu/MANU)  
**Area**: AI Safety / Multimodal VLM  
**Keywords**: Machine Unlearning, MLLM, Neuron Pruning, Modality-Aware, Privacy Protection

## TL;DR

Proposes MANU, the first modality-aware unlearning framework for MLLMs. It identifies cross-modality entangled knowledge-carrying neurons through four complementary neuron importance functions (absolute, frequency, variance, and RMS), and selectively prunes the top-$\alpha\%$ neurons to achieve balanced unlearning under both multimodal and text-only inputs, completely training-free without any gradient updates.

## Background & Motivation

**MLLM training inevitably memorizes sensitive information on massive datasets.** The remarkable capabilities of LLMs and MLLMs stem from pre-training and fine-tuning on large-scale data, which conversely raises risks of privacy leakage and copyright infringement. Retraining models from scratch to exclude sensitive data is computationally prohibitive, making machine unlearning an efficient alternative.

**Existing LLM unlearning methods suffer from severe modality imbalance on MLLMs.** Liu et al. (2024e) revealed a key finding: when directly applying LLM unlearning methods (such as Gradient Ascent or Gradient Difference) to MLLMs, knowledge under multimodal input (image + text) is successfully forgotten, but the exact same knowledge remains retrievable under text-only input. For instance, the model might forget to "name this person when seeing their photo" but can still answer "who this person is" through a text-only description.

**The root cause lies in modality-entangled knowledge representations.** Inputs from different modalities activate different subsets of neurons. Multimodal unlearning only affects the neural pathways processing combined image-text inputs, whereas the pathways processing text-only inputs remain untouched. Heatmap visualizations clearly demonstrate this modality-specific activation pattern.

**Core Idea of MANU: Simultaneously remove neurons associated with the target knowledge along both modality pathways through modality-aware neuron analysis and selective pruning.** This is the first framework specifically designed for MLLM cross-modality unlearning, and it is fully training-free, requiring no gradient updates.

## Method

### Overall Architecture

MANU consists of two phases: **Phase 1 (Important Neuron Selection)**—transforms both the forget set and retain set into both text-only and multimodal formats, collects neuron activation statistics across all MLP layers via forward propagation, and evaluates the modality-specific contribution of each neuron using four importance functions; **Phase 2 (Selective Pruning)**—utilizes a scoring function $S_n$ to aggregate the relative importance of forget vs. retain sets and prunes the top-$\alpha\%$ neurons (setting weights to zero).

### Key Designs

1. **Four Modality-Aware Neuron Importance Functions**:
    - Function: Evaluates behavioral differences of each neuron under different modalities from four complementary perspectives.
    - Mechanism:
        - **Absolute Importance** $I_{\text{abs}} = \frac{|\bar{Z}_{\text{multi}} - \bar{Z}_{\text{text}}|}{\bar{Z}_{\text{multi}} + \bar{Z}_{\text{text}} + \epsilon}$—measures the normalized modality difference in activation magnitude, capturing which neurons exhibit distinctly different activation intensities across modalities.
        - **Frequency Importance** $I_{\text{freq}} = \frac{|N_{\text{multi}} - N_{\text{text}}|}{N_{\text{multi}} + N_{\text{text}} + \epsilon}$—measures the difference in activation frequency (exceeding a threshold $\tau$) across different modalities, capturing consistency rather than raw amplitude.
        - **Variance Importance** $I_{\text{var}} = \sqrt{\text{Var}_{\text{multi}} + \text{Var}_{\text{text}}}$—based on information theory principles, neurons with more diverse activation patterns carry more information.
        - **RMS Importance** $I_{\text{rms}} = \sqrt{\frac{|\Delta Z^2|}{\Sigma Z^2 + \epsilon}}$—identifies consistently highly-activated and modality-specific neurons, filtering out redundant "indiscriminate activation" neurons.
    - Design Motivation: A single metric is insufficient to comprehensively capture modality specificity—magnitude, frequency, diversity, and sustained intensity are four complementary dimensions. The sum of the four functions $\mathcal{I}(\mathcal{D}, n) = \sum_{k \in \mathcal{K}} I_k(\mathcal{D}, n)$ forms a comprehensive importance measure.

2. **Relative Importance Score (Forget vs. Retain)**:
    - Function: Ensures that the pruned neurons primarily serve unlearning target knowledge rather than retaining baseline knowledge.
    - Mechanism: The scoring function takes the ratio of importance between the forget and retain sets: $S_n = \frac{\mathcal{I}(\mathcal{D}_f, n)}{\mathcal{I}(\mathcal{D}_r, n) + \epsilon}$. A high score indicates that the neuron is far more critical for the forget data than for the retain data.
    - Design Motivation: Relying solely on absolute importance risks mistakenly pruning neurons associated with general knowledge. The ratio design guarantees "surgical precision"—removing only those neurons serving the specific unlearning target.

3. **Selective Pruning—Weight Zeroing**:
    - Function: Zeroes out the weights of the top-$\alpha\%$ highest-scoring neurons, applied to both the language and vision MLP layers.
    - Mechanism: Selects the set $\mathcal{N} = \{n : S_n \text{ is among top } \alpha\%\}$ and sets $\theta' = 0$ if $n \in \mathcal{N}$.
    - Design Motivation: Weight zeroing is the simplest pruning method, avoiding gradient updates—the entire process requires only a single forward pass to collect activation statistics.

### Loss & Training

**Completely training-free.** It only requires a single forward pass to collect activation statistics of the forget and retain sets, followed by pruning. Validating on LLaVA-1.5-7B and Idefics2-8B was performed using 3 NVIDIA A6000 GPUs.

## Key Experimental Results

### Main Results—Unlearning Performance on MLLMU-Bench (LLaVA-1.5-7B, 5% Forget)

| Method | Forget Classification Accuracy ↓ | Forget ROUGE ↓ | Retain Classification Accuracy ↑ | Retain ROUGE ↑ | Real Celebrity Classification ↑ |
|------|-------------------|---------------|-------------------|---------------|---------------------|
| Vanilla (No Unlearning) | 51.70% | 0.645 | 46.11% | 0.632 | 51.80% |
| GA | 44.40% | 0.485 | 39.09% | 0.495 | 45.56% |
| Grad. Diff. | 43.60% | 0.507 | 41.07% | 0.508 | 46.52% |
| NPO | 45.61% | 0.525 | 42.61% | 0.515 | 49.51% |
| **MANU** | **41.25%** | **0.491** | **43.38%** | **0.542** | **49.57%** |

MANU achieves the best balance between unlearning efficacy (Forget set ↓) and retaining capability (Retain/Celebrity sets ↑).

### Ablation Study

| Configuration | Forget Efficacy | Retain Preservation | Description |
|------|-----------|------------|------|
| Only $I_{\text{abs}}$ | Effective | Good | Single metric is also effective |
| Only $I_{\text{freq}}$ | Effective | Good | Consistency perspective is complementary |
| Four-component Union | **Optimal** | **Optimal** | Multidimensional complementarity is the most comprehensive |
| $\alpha=1\%$ | Insufficient unlearning | Best retention | Too little pruning |
| $\alpha=3\%$ | **Optimal balance** | Good | Optimal range |
| $\alpha=10\%$ | Strongest unlearning | Harms general capability | Over-pruning |

### Key Findings

- **Confirmation of modality imbalance**: Heatmap visualizations clearly show that while GA/Gradient Difference are effective for unlearning under multimodal inputs (light color in Figure 2b), target knowledge is still preserved under text-only inputs (dark color in Figure 2a).
- Although GA-based methods sometimes surpass MANU in unlearning efficacy, they do so at the cost of severely damaging the model's general capabilities, showing a sharp performance drop on the Retain Set and the Real Celebrity Set.
- The optimal pruning ratio $\alpha$ lies between 1% and 5%; excessive pruning degrades general performance.
- MANU exhibits consistent performance across both LLaVA-1.5-7B and Idefics2-8B, demonstrating cross-model generalization.
- MANU incurs the minimal degradation in general capabilities on MMMU and LLaVA-Bench datasets.

## Highlights & Insights

- **Revealing and systematizing a new problem in MLLM unlearning**: Modality imbalance has not been formally studied before, and heatmap visualizations provide intuitive evidence.
- **A completely training-free unlearning method**: Requires only a single forward pass, statistics collection, and pruning, resulting in extremely low computational cost.
- **Complementary design of four importance functions**: Fully capturing modality specificity across four dimensions: magnitude, frequency, variance, and RMS.
- **Forget/Retain ratio-based score**: Surgical-precision pruning that minimizes collateral damage to retained knowledge.

## Limitations & Future Work

- Only validated in fictional character unlearning scenarios; concept-level unlearning (e.g., specific skills or knowledge domains) has not been tested.
- Pruning (weight zeroing) is a coarse-grained operation; more refined weight modifications (e.g., scaling or decay) might yield better performance.
- Only validated on 7B/8B models; the division of labor among neurons for different modalities might differ in larger models.
- The four importance functions are combined with equal weights; adaptive or learned weighting has not been explored.
- Unlearning robustness has not been investigated—specifically, whether the model can easily relearn the forgotten knowledge through fine-tuning.

## Related Work & Insights

- **vs. Gradient Ascent (GA)**: Performs unlearning via reverse gradients; leads to modality imbalance and impairs general capability in MLLMs.
- **vs. Gradient Difference**: An improved version of GA incorporating the retain set gradient; still suffers from modality imbalance.
- **vs. NPO (Negative Preference Optimization)**: Frames unlearning as preference optimization; relatively stable but less balanced compared to MANU.
- **vs. System Prompting**: Simple prompts can partially prevent sensitive outputs, but the knowledge remains stored within the model parameters.
- **Insight**: The finding that different modalities activate different neurons can be leveraged for modality-specific model compression. Knowing which neurons specifically handle vision or text could facilitate efficient deployment through modality decoupling.

## Rating

- Novelty: ⭐⭐⭐⭐ The first modality-aware unlearning framework for MLLMs, with a systematic and comprehensive design of four importance functions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-model validation (LLaVA/Idefics2), multiple unlearning ratios, detailed ablation studies, and heatmap visualizations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is intuitively demonstrated via heatmaps, and the methodology is clearly formulated.
- Value: ⭐⭐⭐⭐ Directly applicable to AI safety and privacy protection; the identified modality imbalance phenomenon is of cognitive significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/llm_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)

</div>

<!-- RELATED:END -->
