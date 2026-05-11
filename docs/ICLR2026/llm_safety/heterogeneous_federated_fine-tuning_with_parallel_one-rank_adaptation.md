---
title: >-
  [Paper Note] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation
description: >-
  [ICLR 2026][LLM Safety][Federated Fine-Tuning] This paper proposes Fed-PLoRA, a framework that replaces multi-rank LoRA with multiple parallel one-rank modules (PLoRA). Via a Select-N-Fold strategy—selecting $N$ modules…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Federated Fine-Tuning"
  - "LoRA"
  - "Heterogeneous Rank"
  - "Initialization Noise"
  - "Aggregation Noise"
date: 2026-05-08
content_hash: 575d560621e51a5d
---

# Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation

**Conference**: ICLR 2026
**arXiv**: [2602.16936](https://arxiv.org/abs/2602.16936)
**Code**: [GitHub](https://github.com/TNI-playground/Fed-PLoRA)
**Area**: AI Safety
**Keywords**: Federated Fine-Tuning, LoRA, Heterogeneous Rank, Initialization Noise, Aggregation Noise

## TL;DR
This paper proposes Fed-PLoRA, a framework that replaces multi-rank LoRA with multiple parallel one-rank modules (PLoRA). Via a Select-N-Fold strategy—selecting $N$ modules for training and folding the remainder into frozen weights—it achieves zero initialization noise and minimal aggregation noise for heterogeneous federated fine-tuning, outperforming existing methods across 6 LLMs and multiple tasks.

## Background & Motivation

**Background**: Federated fine-tuning (FFT) leverages LoRA to collaboratively fine-tune LLMs across distributed clients while preserving data privacy. However, heterogeneous client resources lead to different LoRA ranks, causing dimension mismatches during initialization and aggregation.

**Limitations of Prior Work**: (1) FLoRA: randomly reinitializes LoRA each round, introducing large initialization noise; (2) HETLoRA: truncates the global LoRA, discarding information beyond the low-rank subspace and introducing aggregation bias; (3) FlexLoRA: uses SVD reconstruction, introducing decomposition errors. All existing methods face an irreconcilable trade-off between initialization noise and aggregation noise.

**Key Challenge**: When the global model rank $R$ exceeds a client's rank $r_i$, the client cannot fully inherit global information (initialization noise), and aggregating separately trained adapters is imperfect (aggregation noise).

**Key Insight**: Decompose multi-rank LoRA into multiple parallel one-rank modules. Each module is independent, allowing clients to select a subset for training and fold the remainder into frozen weights, achieving zero initialization noise.

## Method

### Overall Architecture
PLoRA: $\Delta W = \sum_{j=1}^{R} B_{(j)}A_{(j)}$, mathematically equivalent to standard LoRA but with independent modules. Select-N-Fold: client $i$ selects $r_i$ modules for training and folds the remaining modules into the pretrained weights for freezing. Aggregation: independent averaging along the rank dimension.

### Key Designs

1. **PLoRA (Parallel One-Rank Adaptation)**:

    - Function: Decomposes a rank-$R$ LoRA into $R$ parallel rank-1 modules.
    - Mechanism: $\Delta W_{\text{PLoRA}} = \sum_{j=1}^R B_{(j)}A_{(j)} = \sum_{j=1}^R B_{[:,j]}A_{[j,:]} = BA = \Delta W_{\text{LoRA}}$
    - Design Motivation: Mathematical equivalence with modular independence naturally enables subset selection.

2. **Select-N-Fold Strategy**:

    - Function: Each client randomly selects $r_i$ PLoRA modules for training; the remaining modules are folded into frozen weights.
    - Mechanism: $\mathcal{W}_i^t = \mathcal{W}^0 + \sum_{j \notin \mathcal{K}_i^t} B_{(j)}^{t-1}A_{(j)}^{t-1}$, with training performed on $\mathcal{W}_i^t$.
    - Design Motivation: Folding preserves information from unselected modules, yielding zero initialization noise. Random selection ensures all modules are updated in expectation.

3. **Noise Analysis**:

    - Initialization noise: $\mathcal{N}_{\text{Init}}^t = 0$ (global information perfectly preserved).
    - Aggregation noise upper bound: $\leq \sum_{j=1}^R \frac{1}{|\mathcal{Q}_{(j)}^t|}\sum_i \|B_{i,(j)}^t - \bar{B}_{(j)}^t\|_2 + \|A_{i,(j)}^t - \bar{A}_{(j)}^t\|_2$
    - Cosine similarity analysis demonstrates that modules converge across clients during training, progressively tightening the upper bound.

### Loss & Training
- Standard federated fine-tuning pipeline (broadcast → local training → aggregation).
- 10% client participation per round.
- SGD/AdamW as local optimizers.

## Key Experimental Results

### Main Results (Llama-1B, Natural Instructions)

| Method | IID Accuracy | non-IID Accuracy | Initialization Noise |
|--------|-------------|-----------------|---------------------|
| FedIT (homogeneous) | 66.88 | 61.28 | 0 |
| FLoRA | Medium | Medium | High (random re-init) |
| FlexLoRA | Medium | Medium | Medium (truncation + SVD error) |
| HETLoRA | Medium | Medium | Medium (truncation) |
| **Fed-PLoRA** | **Highest** | **Highest** | **0** |

### Multi-Model / Multi-Task Validation

| Model | Task | Fed-PLoRA vs. Best Baseline |
|-------|------|-----------------------------|
| BERT-base | GLUE | Outperforms |
| Llama-3.1-8B | Financial NLP | Outperforms |
| Qwen3-4B | Instruction Following | Outperforms |
| Mistral-7B | Medical QA | Outperforms |

### Key Findings
- Cosine similarity heatmaps show that PLoRA modules at the same rank index converge across clients after training (high diagonal values), while modules at different rank indices remain independent (low off-diagonal values), indicating that each rank captures distinct knowledge while clients converge.
- Fed-PLoRA demonstrates a larger advantage in non-IID settings, suggesting that zero initialization noise is especially critical under data heterogeneity.
- Communication, computation, and memory overhead are comparable to existing methods, introducing no additional cost.

## Highlights & Insights
- **Zero Initialization Noise**: By folding rather than truncating or reinitializing, Fed-PLoRA perfectly preserves global information—a simple yet fundamental solution to heterogeneous FFT.
- **Modular Independence of PLoRA**: Although mathematically equivalent to standard LoRA, the modular independence enables subset selection and independent aggregation naturally—an architectural trick that yields systematic improvement.
- **Unified Noise Analysis Framework**: The paper provides a unified analysis of initialization noise and aggregation noise for FLoRA, FlexLoRA, HETLoRA, and Fed-PLoRA, clearly characterizing the strengths and weaknesses of each approach.

## Limitations & Future Work
- Random module selection may be suboptimal; importance- or gradient-based selection strategies could be more effective.
- The folding operation adds $O(dk(R-r_i))$ computation per round, which, while much smaller than training cost, is non-zero.
- Downlink communication incurs $O((d+k)(R-r_i))$ additional overhead compared to HETLoRA and FlexLoRA.
- Experiments are limited to LoRA applied to self-attention layers; the effect of applying PLoRA to FFN layers remains unexplored.

## Related Work & Insights
- **vs. FLoRA**: FLoRA achieves zero aggregation noise but suffers from large initialization noise; Fed-PLoRA achieves zero initialization noise with small aggregation noise, yielding overall superior performance.
- **vs. HETLoRA**: HETLoRA truncates high-rank components, discarding information; Fed-PLoRA folds them, retaining all information.
- **vs. Standard LoRA / FedIT**: Fed-PLoRA is equivalent to FedIT in the homogeneous setting and outperforms all baselines in the heterogeneous setting.

## Rating
- Novelty: ⭐⭐⭐⭐ The PLoRA decomposition combined with the Select-N-Fold strategy is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 models, multiple domain tasks, IID/non-IID settings, and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ The noise analysis framework is clear and comparisons are fair.
- Value: ⭐⭐⭐⭐ Directly applicable to practical heterogeneous federated fine-tuning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)

</div>

<!-- RELATED:END -->
