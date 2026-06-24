---
title: >-
  [Paper Note] TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing
description: >-
  [ICLR 2026][Knowledge Editing][Unstructured Knowledge] This paper proposes **TangleScore**, an intrinsic difficulty metric determined solely by the "model + knowledge sample" independent of specific editing algorithms, to measure how "hard" a piece of knowledge is to modify. Based on this, it designs **PIPE** (a two-stage editing framework that purges old knowledge before imprinting new knowledge), improving generalization performance by an average of 6.49% across four LLMs o…
tags:
  - "ICLR 2026"
  - "Knowledge Editing"
  - "Unstructured Knowledge"
  - "Editing Difficulty Quantization"
  - "Knowledge Unlearning"
  - "Generalization"
date: 2026-05-08
content_hash: ee15718ab7df6322
---

# TangleScore: Tangle-Guided Purge and Imprint for Unstructured Knowledge Editing

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TbLkgJCGfc](https://openreview.net/forum?id=TbLkgJCGfc)  
**Code**: https://github.com/famoustourist/TangleScore  
**Area**: Knowledge Editing / LLM  
**Keywords**: Knowledge Editing, Unstructured Knowledge, Editing Difficulty Quantization, Knowledge Unlearning, Generalization

## TL;DR
This paper proposes **TangleScore**, an intrinsic difficulty metric determined solely by the "model + knowledge sample" independent of specific editing algorithms, to measure how "hard" a piece of knowledge is to modify. Based on this, it designs **PIPE** (a two-stage editing framework that purges old knowledge before imprinting new knowledge), improving generalization performance by an average of 6.49% across four LLMs of varying scales and two unstructured editing benchmarks.

## Background & Motivation

**Background**: Knowledge editing is a mainstream approach to replace expensive retraining by performing lightweight corrections of outdated or incorrect facts within LLMs. Locate-then-edit methods such as ROME, MEMIT, and AlphaEdit are well-established for **structured knowledge** (subject-verb-object triples), where they precisely modify specific layer weights to inject new facts.

**Limitations of Prior Work**: However, approximately 80% of real-world knowledge is **unstructured**—free-form text and information-dense paragraphs rather than clean triples. When structured methods are applied to unstructured scenarios, a significant gap emerges between **accuracy (memorizing the target answer) and generalization (answering correctly under paraphrased queries)**. Worse, knowledge-wise analysis reveals that this performance collapse is **not random**; it shows a consistent pattern across different editing methods and models, where specific samples remain "uneditable."

**Key Challenge**: The authors attribute the root cause to a **mismatch between editing perturbation intensity and the coupling degree between the "target knowledge and the model."** When the model has a deep-rooted internal dependence on old knowledge, existing methods merely "paste" the new answer over it without truly overwriting the internal representations of the old knowledge. Consequently, the model binds the new knowledge to a fixed response pattern through rote memorization, which fails once the query is paraphrased. The problem is that current methods treat all samples equally, applying the same editing intensity, leading to "under-editing" for hard samples and "over-editing" for simple ones.

**Goal**: (1) To identify a metric that can **quantify in advance** the editing difficulty of a piece of knowledge; (2) To use this metric to **adaptively adjust** editing intensity—applying stronger intervention for hard samples and more restraint for easy ones.

**Key Insight**: The authors observe a commonality in failure cases: "the model output remains strongly tethered to the original knowledge." They hypothesize the existence of an **intrinsic edit-resistance** determined only by the base model and the knowledge itself, independent of the editing algorithm used, which is strongly correlated with post-edit generalization performance.

**Core Idea**: The proposed **TangleScore** quantifies this "knowledge-model entanglement/edit-resistance." It then drives a two-stage editing framework, **PIPE** (Purge-then-Imprint), which adaptively decides "how aggressively to forget and how heavily to imprint" based on the difficulty score.

## Method

### Overall Architecture

PIPE (Purge-Imprint Patch Editing) decomposes "editing a piece of research" into two steps modulated by difficulty. Given an edit sample $(x_e, y_e)$, it **first calculates the TangleScore** to judge the edit-resistance of the knowledge. A higher TangleScore indicates tighter entanglement, resulting in a **higher purge rate**. Subsequently, a **knowledge purge function** uses gradient ascent to actively "forget" the model's dependence on the old key vectors. After purging, a **knowledge imprint function** writes in the new knowledge, with the TangleScore similarly modulating the weight between "learning new vs. preserving old." The two stages are jointly optimized to complete the edit end-to-end. The entire method only edits the selected 7th layer.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Knowledge to Edit<br/>(xe, ye)"] --> B["TangleScore<br/>Quantifying Intrinsic Difficulty"]
    B --> C["Adaptive Purge Rate<br/>Higher Difficulty, Brisk Purge"]
    C --> D["Knowledge Purge Function<br/>Gradient Ascent identifies Old Key"]
    D --> E["Knowledge Imprint Function<br/>Write New Knowledge + Preserve Stability"]
    B -.->|σ(TS) Modulates α| E
    E --> F["Joint Optimization<br/>Edited Model fϕ"]
```

### Key Designs

**1. TangleScore: Quantifying intrinsic edit-resistance via "Internal Representation Shift ÷ Output Semantic Gap"**

This is the foundation of the work, addressing the pain point that "existing methods cannot identify hard-to-edit knowledge and apply a one-size-fits-all approach." The metric must be determined solely by the base model + knowledge sample, independent of the algorithm, and predictive of generalization. It combines two parts. First is the **internal representation shift** $D_{semantic}$: the original and new knowledge are fed into the model with custom prompts; the hidden representations $r_{old}, r_{new}$ are average-pooled to calculate cosine distance $D_{semantic} = 1 - \frac{r_{old}\cdot r_{new}}{\|r_{old}\|\,\|r_{new}\|}$. Larger values indicate greater internal vibration and resistance when injecting new knowledge. Second is the **semantic gap of the output response**: comparing the difference in model answers before and after editing. Instead of KL divergence, which is sensitive to token-level mismatches, the authors use the differentiable and stable **Optimal Transport Sinkhorn distance** to perform global semantic alignment in the embedding space. TangleScore is defined as the ratio:

$$\text{TANGLESCORE}(M) = \frac{D_{semantic}}{\text{Sinkhorn}(\text{Ans}_{old}, \text{Ans}_{new})}.$$

This **ratio form** characterizes the "transition rate of the model response from old to new answers"—the numerator is the internal representation cost, and the denominator is the actual distance achieved at the output. High values indicate higher "inertia" and difficulty. Experiments confirm its key properties: the distribution remains stable before and after editing (proving it is an **intrinsic attribute**) and metrics like ROUGE/BERT-Score **monotonically decrease** as TangleScore increases (proving it predicts generalization).

**2. TangleScore-Driven Adaptive Purge Rate: Aggressive for Hard Samples, Restrained for Simple Ones**

The next step is integrating difficulty into editing intensity. Uniform strategies either leave residues in hard samples or cause overfitting in simple ones. The purge rate $PR$ varies exponentially with TangleScore:

$$PR = \lambda_{max}^{\gamma\,\text{TANGLESCORE}},\quad \gamma = \frac{\log\lambda_{max}}{\log\lambda_{min}},$$

where $\lambda_{max}=0.001$ and $\lambda_{min}=0.0001$ are determined empirically. High TangleScore (hard) maps to a higher $PR$ for aggressive forgetting; low TangleScore (easy) maps to a lower $PR$ to avoid overfitting.

**3. Bounded Knowledge Purge Function: Stable "Forgetting" of Old Keys**

The authors use a gradient ascent approach inspired by unlearning to raise the loss of old information. However, direct gradient ascent on MSE causes numerical instability as gradients $\nabla_{\hat y}L_{MSE}=2(\hat y - y)$ are proportional to error. PIPE uses a **bounded function that penalizes "proximity to old knowledge" rather than "encouraging large deviation"**: by clamping the absolute difference to $[0,1]$ to ensure controlled gradients, the purge loss for the $i$-th knowledge is:

$$L_{purge} = \sum_{i=1}^{u} PR \cdot \big(\text{Clamp}(1 - |f_\phi(h_{q,i}) - \tilde k_i^{orig}|)\big)^2,$$

where $f_\phi(h_{q,i})$ is the current key and $\tilde k_i^{orig}$ is the old key. Gradients saturate when predictions are far from the old key, ensuring stable and clean unlearning.

**4. TangleScore-Modulated Knowledge Imprint Function: Writing New Facts while Preserving Capabilities**

The core challenge is "learning the new without destroying the old." The consistency loss $L_{consistency}$ weights "knowledge imprinting" against "stability preservation" per token:

$$L_{consistency} = \sum_{i=1}^{u}\sum_{j=1}^{n-1}\big[\alpha\,\|f_\phi(h_{q,i,j}) - k_{q,i,j}\|^2 + (1-\alpha)\,\|f_\phi(h_{q,i,j}) - f_\theta(h_{q,i,j})\|^2\big],$$

The weight $\alpha$ is dynamically modulated by TangleScore via a sigmoid: $\alpha = \sigma(\text{TANGLESCORE})$. Hard samples (high TS) drive $\alpha\to1$ to prioritize learning; simple samples (low TS) drive $\alpha\to0$ to preserve existing capabilities. Combined with the learning loss $L_{learn}=\sum_i \|f_\phi(h_{q,i}) - \tilde k_i\|^2$, it forms $L_{imprint}=L_{consistency}+L_{learn}$.

### Loss & Training
To achieve end-to-end editing, PIPE jointly optimizes the purge and imprint stages:

$$f_\phi = \arg\min_\phi (L_{purge} + L_{imprint}).$$

Editing is applied specifically to the 7th layer (validated via ablation).

## Key Experimental Results

### Main Results: Unstructured Knowledge Editing (UNKEBench)

Results are shown for original questions / paraphrased questions. FC indicates FactScore (multi-hop understanding), and MMLU measures general capability preservation.

| Model / Method | BERT-Score | ROUGE-L | FactScore | MMLU |
|------|------|------|------|------|
| LLaMA3-8B · UNKE | 97.41 / 90.06 | 97.86 / 77.72 | 41.56 | 29.45 |
| LLaMA3-8B · AnyEdit | 98.62 / 91.56 | 95.15 / 79.60 | 48.48 | 28.52 |
| **LLaMA3-8B · PIPE (Ours)** | **98.64 / 91.71** | **98.44 / 84.07** | **50.91** | **29.51** |
| LLaMA2-7B · UNKE | 96.56 / 90.73 | 95.34 / 76.11 | 40.99 | 29.07 |
| **LLaMA2-7B · PIPE (Ours)** | **98.57 / 91.73** | **97.39 / 78.47** | **50.12** | 29.65 |
| Qwen2.5-7B · UNKE | 96.84 / 89.92 | 93.64 / 73.89 | 40.12 | 31.28 |
| **Qwen2.5-7B · PIPE (Ours)** | **97.42 / 91.76** | **97.05 / 80.18** | **42.47** | **31.78** |

Structured methods (ROME/MEMIT/RECT/AlphaEdit) score only around 40 on ROUGE-L for UNKEBench. PIPE outperforms UNKE/AnyEdit primarily in **generalization to paraphrased questions (Gain in the 78→84 range)** and **FactScore**, while maintaining MMLU scores. The paper reports an average generalization **Gain** of **6.49%**.

### Structured Knowledge Editing (KEBench)

| Method | Ori-Acc | Para-Acc | Src-Acc | Tgt-Acc |
|------|------|------|------|------|
| UNKE (Prev. SOTA) | 93.59 | 85.34 | 89.28 | 62.56 |
| **PIPE (Ours)** | **95.89** | **88.23** | **94.47** | **70.11** |

### Key Findings
- **TangleScore is an intrinsic attribute**: Its distribution remains nearly identical before and after editing, confirming it is an inherent property of the model-knowledge pair.
- **Difficulty strongly correlates with generalization**: ROUGE and BERT-Score decrease monotonically as TangleScore increases—hard samples are precisely where previous methods fail to generalize.
- **Purge effectively suppresses old knowledge**: Probabilistic analysis shows that after PIPE editing, the model is significantly less likely to output old information.
- **Stability comes from bounded loss**: The use of clamping in the purge function is critical to prevent gradient explosion during the unlearning phase.

## Highlights & Insights
- **Quantifying "editing difficulty" as a pre-computable intrinsic value**: TangleScore is decoupled from editing algorithms, serving as a diagnostic tool to determine necessary intensity before editing.
- **Effective ratio-based difficulty definition**: By using a ratio between internal representation shifts and output semantic gaps, it captures the "transition rate," reflecting both internal resistance and external change.
- **Dual-modulation by a single signal**: Both the purge rate and imprint weight $\alpha$ are driven by TangleScore, ensuring a self-consistent "aggressive forgetting + intensive learning" logic for hard samples.
- **Decoupled Purge-and-Imprint approach**: By separating the overwriting of old knowledge from the writing of new facts, it addresses the core issue in unstructured editing where old knowledge often persists beneath the new answer.

## Limitations & Future Work
- PIPE does not yet support **continuous/sequential knowledge update streams** and was tested in controlled settings.
- Currently limited to **text editing**, lacking support for multimodal updates.
- Validation is needed on **larger models and more diverse datasets**.
- TangleScore incurs additional overhead due to hidden representation and Sinkhorn distance calculations.

## Related Work & Insights
- **vs UNKE**: UNKE updates all parameters in a single layer, but treats all samples with equal intensity, leading to poor generalization on hard samples; PIPE wins via adaptive intensity and explicit purging.
- **vs AnyEdit**: AnyEdit focuses on decomposing long paragraphs into blocks; PIPE focuses on understanding knowledge difficulty. The two approaches are orthogonal and could be combined.
- **vs ROME / MEMIT / AlphaEdit**: These structured-first methods suffer significant performance drops on unstructured text; PIPE provides a unified framework effective for both.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[CVPR 2026\] Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](../../CVPR2026/knowledge_editing/attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](../../ACL2025/knowledge_editing/chainedit_propagating_ripple_effects_in_llm.md)
- [\[ICLR 2026\] SUIT: Knowledge Editing with Subspace-Aware Key-Value Mappings](suit_knowledge_editing_with_subspace-aware_key-value_mappings.md)
- [\[ICLR 2026\] MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll.md)

</div>

<!-- RELATED:END -->
