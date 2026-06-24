---
title: >-
  [Paper Note] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training
description: >-
  [NeurIPS 2025][LLM Efficiency][critical batch size] This paper proposes a branched training method to directly measure the critical batch size (CBS) empirically, finding that CBS grows rapidly in early training before plateauing and is independent of model scale. Based on this insight, a batch size warmup strategy is designed that achieves equivalent or superior training loss with 43% fewer gradient steps.
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "critical batch size"
  - "large-batch training"
  - "learning rate scaling"
  - "batch size warmup"
  - "gradient noise scale"
  - "OLMo"
date: 2026-05-08
content_hash: fab625b3c6e5bd7e
---

# Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training

**Conference**: NeurIPS 2025
**arXiv**: [2505.23971](https://arxiv.org/abs/2505.23971)  
**Code**: None  
**Area**: LLM Efficiency / Training Optimization
**Keywords**: critical batch size, large-batch training, learning rate scaling, batch size warmup, gradient noise scale, OLMo

## TL;DR

This paper proposes a branched training method to directly measure the critical batch size (CBS) empirically, finding that CBS grows rapidly in early training before plateauing and is independent of model scale. Based on this insight, a batch size warmup strategy is designed that achieves equivalent or superior training loss with 43% fewer gradient steps.

## Background & Motivation

**Background**: Large-batch training improves LLM training throughput by increasing data parallelism and is a core requirement for large-scale pretraining. The critical batch size (CBS), defined as the largest batch size that does not significantly degrade token efficiency, is a key concept for balancing efficiency and performance.

**Limitations of Prior Work**: McCandlish et al. (2018) proposed the gradient noise scale as a proxy for CBS, which was adopted by GPT-3 and subsequent works. However, this approach relies on two strong assumptions: (a) the use of SGD, and (b) well-conditioned gradients.

**Key Challenge**: Practical LLM training uses the Adam optimizer, and theoretical analysis suggests Adam should follow a square-root scaling rule rather than a linear one. Furthermore, the simplified gradient noise scale $\mathcal{B}_{\text{simple}}$ is only equivalent to the true CBS under the assumption that the Hessian is an identity matrix.

**Goal**: How can CBS be measured directly without relying on strong assumptions? How does CBS evolve during training? How can CBS information be leveraged to formulate practical large-batch training strategies?

**Key Insight**: Branched training is used to empirically approximate CBS directly, avoiding the unreliability introduced by indirect proxies.

**Core Idea**: CBS is a dynamic quantity that grows from near zero to a plateau value during training, an insight that naturally motivates a batch size warmup strategy.

## Method

### Overall Architecture

From any checkpoint during training, multiple branches are forked, each using a different batch size (k times the base BS). After training for a fixed Δ tokens, the resulting losses are compared, and the largest batch size whose loss does not increase significantly is identified as the local CBS at that checkpoint.

### Key Designs

1. **Branched Training for CBS Measurement**

    - **Function**: Starting from a checkpoint, train for Δ=2B tokens at different batch size multipliers k and compare the final loss.
    - **Mechanism**: Directly observe the loss response to batch size rather than inferring it indirectly from gradient statistics.
    - **Design Motivation**: Avoids the two strong assumptions of the gradient noise approach (SGD + well-conditioned gradients), requiring only a weak assumption—if the loss recovers within Δ tokens, that batch size will not degrade performance over longer training.

2. **Local Recovery Assumption**

    - **Function**: Assumes that if a batch size B can recover to a loss close to that of a smaller batch size within a window of Δ tokens, it will not degrade over longer training.
    - **Mechanism**: Reduces global CBS measurement to a local one, requiring only a small amount of additional training.
    - **Design Motivation**: Avoids the need to run complete training for each batch size as in prior work.

3. **CBS Detection Criterion**

    - **Function**: Uses a tolerance of ε=0.01 to determine whether loss increases significantly, and an exponential moving average (α=0.5) to smooth loss noise.
    - **Mechanism**: Based on the loss-vs-k curve, identifies the inflection point k* at which loss begins to rise; the corresponding $B^* = k^* \times B_{\text{base}}$ is taken as the CBS.
    - **Design Motivation**: Pretraining loss is noisy and requires smoothing; the tolerance prevents false positives due to noise.

4. **Batch Size Warmup Strategy**

    - **Function**: Training begins with BS=1024; whenever the measured CBS exceeds 2× the current BS, the BS is doubled and the learning rate is adjusted according to the square-root rule.
    - **Mechanism**: Since CBS grows rapidly in early training, the BS is increased incrementally to track CBS growth, thereby enjoying the throughput benefits of large batches without sacrificing performance.
    - **Design Motivation**: CBS starting near zero means large batches cannot be used at the outset, but CBS quickly grows to several thousand, allowing large batches for most of the training.

### Loss & Training

- Learning rate scaling rule: Square-root scaling for Adam, $\eta' = \sqrt{B'/B} \cdot \eta$ (rather than linear scaling for SGD).
- BS doubling schedule: Based on CBS measurements, doubled to 2048 at 168B tokens and to 4096 at 503B tokens.
- Applied on top of the original cosine learning rate schedule, compatible with existing training pipelines.
- Mid-training phase: Linear decay from the final pretraining checkpoint to LR=0 over 50B tokens.

## Key Experimental Results

### Main Results

| Method | Pretraining Loss ↓ | Mid-training Loss ↓ | Gradient Step Savings ↑ |
|---|---|---|---|
| Batch Size Warmup (Ours) | **2.5891** | **2.5433** | 43% |
| Fixed Small Batch (BS=1024) | 2.6057 | 2.5486 | 0% |
| Fixed Large Batch (BS=4096) | 2.5962 | 2.5506 | 75% |

### Ablation Study / OOD Evaluation

| Method | Task BPB (PT/MT) ↓ | C4 Loss (PT/MT) ↓ | Pile Loss (PT/MT) ↓ |
|---|---|---|---|
| BS Warmup | 1.0316 / **1.0076** | **2.8049** / 2.7597 | 2.1916 / 2.1521 |
| Small Batch Baseline | **1.0112** / 0.9999 | 2.8196 / 2.7622 | 2.2073 / 2.1471 |
| Large Batch Baseline | 1.0571 / 1.0193 | 2.8107 / 2.7658 | 2.1996 / 2.1586 |

### Key Findings

1. **CBS Dynamics**: CBS is near zero at initialization, grows rapidly within the first 50k steps, then plateaus at approximately 4096 documents (each 4096 tokens long). Both the 1B and 7B models exhibit the same qualitative trend.
2. **Unreliability of Gradient Noise**: The gradient noise scale substantially underestimates CBS on both models (by several orders of magnitude), and its qualitative trend does not match the empirical CBS (particularly for the 7B model), making it an unreliable proxy.
3. **Effectiveness of Warmup**: The BS warmup strategy achieves slightly better loss than the small-batch baseline after both pretraining and mid-training, while using 43% fewer gradient steps. The fixed large-batch baseline saves 75% of steps but at the cost of degraded loss.
4. **CBS Scale-Independence**: The CBS curves for 1B and 7B are qualitatively consistent, supporting prior findings that CBS scales primarily with data volume rather than model size.
5. **Theoretical Derivation**: If the local CBS grows as $t^{1/2}$, the global fixed CBS can be shown to scale as $\sqrt{T}$, consistent with prior work.

## Highlights & Insights

- **CBS Is Dynamic, Not Constant**: This is the paper's most central insight. CBS is small at the start of training, so using a fixed large batch in the early phase exceeds CBS and degrades loss; for most of training, CBS is large enough to safely use large batches. The warmup strategy precisely exploits this pattern.
- **Simplicity and Reliability of the Method**: Branched training requires no additional assumptions and measures CBS with only a small amount of forked training, making it more trustworthy than gradient noise approaches.
- **Square-Root Scaling for Adam**: The paper provides empirical validation of the batch size–learning rate relationship under Adam, consistent with the theoretical analysis of Malladi et al. (2022) and constituting an important correction to practice (GPT-3 used linear scaling).
- **Cross-Scale Consistency Has Significant Implications**: CBS measured at 1B can directly guide training configurations at 7B and beyond, substantially reducing measurement cost.

## Limitations & Future Work

1. **Additional Cost of Branched Training**: Although cheaper than full training, multiple branches must still be run for each checkpoint. Future work could explore online CBS measurement methods.
2. **Sensitivity to Window Size Δ**: Larger Δ values may yield higher CBS estimates; the effect of Δ has not been systematically analyzed.
3. **Power-of-Two Doubling Only**: BS can only be doubled, resulting in coarse alignment with the true CBS; finer-grained adjustments may further improve efficiency.
4. **Limited Scale Coverage**: Validation is limited to 1B and 7B models; behavior at 70B+ scales remains to be confirmed.
5. **Only 608B Tokens of Pretraining**: Full OLMo training uses 4T tokens; the current experiments cover approximately 15%, and CBS behavior over longer training remains to be verified.
6. **Manual Selection of Doubling Thresholds**: Doubling points are currently chosen by manual inspection of plots; a systematic methodology for threshold selection is lacking.

## Related Work & Insights

- **McCandlish et al. (2018)**: Proposed the gradient noise scale for estimating CBS, adopted by GPT-3, but shown here to be unreliable.
- **Zhang et al. (2024)**: Found that CBS scales primarily with data volume ($\propto \sqrt{D}$); this paper provides a consistent explanation from the perspective of local CBS.
- **Malladi et al. (2022)**: Argued via SDE analysis that Adam should use square-root scaling; this paper provides empirical validation.
- **Smith et al. (2018)**: Explored replacing learning rate decay with batch size increases; conceptually related, but this paper ensures BS never exceeds CBS.
- **Inspiration**: The proposed method can be extended to other settings where optimal batch size must be determined (e.g., fine-tuning, RLHF); CBS-aware schedulers could become a standard component of training pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The method is remarkably simple (branched training with loss comparison), yet the insights are profound (CBS is dynamic + gradient noise is unreliable); a simple approach resolves an important problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic measurement across multiple checkpoints × multiple batch size multipliers × two model scales, with OOD evaluation and mid-training validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear motivation, explicitly stated assumptions, and highly informative figures; a model example of a methodology paper.
- **Value**: ⭐⭐⭐⭐ — Directly actionable for LLM pretraining practice. The warmup strategy is non-invasive and integrates seamlessly with existing training pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](../../ICLR2026/llm_efficiency/fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](../../ACL2025/llm_efficiency/tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ICML 2026\] MineDraft: A Framework for Batch Parallel Speculative Decoding](../../ICML2026/llm_efficiency/minedraft_a_framework_for_batch_parallel_speculative_decoding.md)
- [\[NeurIPS 2025\] Constant Bit-Size Transformers Are Turing Complete](constant_bit-size_transformers_are_turing_complete.md)
- [\[ICLR 2026\] Reasoning Language Model Inference Serving Unveiled: An Empirical Study](../../ICLR2026/llm_efficiency/reasoning_language_model_inference_serving_unveiled_an_empirical_study.md)

</div>

<!-- RELATED:END -->
