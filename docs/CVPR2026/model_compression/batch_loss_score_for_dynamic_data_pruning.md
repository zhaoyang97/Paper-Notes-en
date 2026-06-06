---
title: >-
  [Paper Note] Batch Loss Score for Dynamic Data Pruning
description: >-
  [CVPR 2026][Model Compression][dynamic data pruning] This paper proposes Batch Loss Score (BLS), a method that estimates sample importance using only the mean batch loss — which is universally available — rather than per…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "dynamic data pruning"
  - "batch loss"
  - "EMA"
  - "training efficiency"
  - "sample importance"
date: 2026-05-08
content_hash: 38780b070d4cac49
---

# Batch Loss Score for Dynamic Data Pruning

**Conference**: CVPR 2026
**arXiv**: [2604.04681](https://arxiv.org/abs/2604.04681)  
**Code**: [https://github.com/mrazhou/BLS](https://github.com/mrazhou/BLS)  
**Area**: Training Efficiency / Data Pruning
**Keywords**: dynamic data pruning, batch loss, EMA, training efficiency, sample importance

## TL;DR

This paper proposes Batch Loss Score (BLS), a method that estimates sample importance using only the mean batch loss — which is universally available — rather than per-sample loss, which is difficult to obtain in practice. Grounded in a signal-processing perspective via EMA-based low-pass filtering, BLS offers theoretical guarantees and can be integrated into existing dynamic pruning frameworks with just 3 lines of code.

## Background & Motivation

Dynamic data pruning accelerates deep learning training by skipping less informative samples. Per-sample loss is the most intuitive importance measure, yet obtaining it in practice poses significant obstacles: standard training pipelines are highly optimized for computing the mean batch loss, and recovering individual losses from the aggregated scalar is non-trivial. For complex objective functions (e.g., multi-component detection losses), defining and isolating a per-sample scalar requires deep task-specific knowledge and invasive code modifications.

**Core insight of BLS**: Although per-sample loss is hard to access, the mean batch loss is ubiquitous. By maintaining an EMA score for each sample — updated only when that sample appears in the current batch — sample importance can be inferred indirectly.

## Method

### Overall Architecture

Each sample is associated with a score $s_i(t)$, updated via EMA whenever sample $i$ appears in batch $B_t$: $s_i(t) = \alpha \cdot s_i(t-1) + (1-\alpha) \cdot L(B_t, t)$. BLS serves as a transparent proxy, replacing per-sample loss within existing pruning frameworks.

### Key Designs

1. **Signal Decomposition and Filtering**: From a single sample's perspective, the mean batch loss equals a scaled signal ($\frac{1}{B} l_i(t)$, the contribution of sample $i$) plus batch-composition noise (the loss contributions of the remaining $B-1$ samples). EMA acts as a first-order IIR low-pass filter, attenuating high-frequency batch-composition noise while preserving low-frequency persistent loss trends.

2. **Frequency Separation Assumption**: The high-frequency fluctuations of batch-composition noise (caused by random sampling at each step) are substantially higher than the evolution frequency of the scaled per-sample loss (driven by slow parameter updates), making low-pass filtering effective.

3. **Seamless Proxy Integration**: BLS serves as a plug-and-play replacement for per-sample loss; downstream pruning algorithms remain entirely agnostic to the source of the scores, requiring no modifications to core scheduling logic or hyperparameters. Integration requires only 3 lines of code, compared to 33+ lines of invasive modification in InfoBatch.

### Loss & Training

BLS does not alter the training loss; it only affects sample selection. The EMA decay factor $\alpha$ controls the filtering characteristics: larger $\alpha$ yields stronger noise suppression at the cost of slower response.

### Theoretical Guarantees

From a signal decomposition perspective, the mean batch loss for a batch containing sample $i$ can be decomposed into a scaled signal $\frac{1}{B} l_i(t)$ and batch-composition noise $\frac{1}{B}\sum_{j \neq i} l_j(t)$. The frequency separation assumption states that the high-frequency fluctuations of batch-composition noise far exceed the evolution frequency of the scaled per-sample loss. EMA acts as a first-order IIR low-pass filter $H_\alpha$ with impulse response $h[n] = (1-\alpha)\alpha^n u[n]$ and frequency response $|H(e^{j\omega})| = \frac{1-\alpha}{\sqrt{1-2\alpha\cos(\omega)+\alpha^2}}$, which is maximized at $\omega = 0$, thereby filtering out high-frequency noise while retaining low-frequency trends.

## Key Experimental Results

### Main Results

| Dataset / Task | Method | Pruning Ratio | Performance | Notes |
|---|---|---|---|---|
| ToCa (3M, zero-shot captioning) | BLS-SeTa | 32% | CIDEr 71.2 | ≈ SeTa 71.5 |
| MJ+ST (15M, text recognition) | BLS-SeTa | 33% | IIIT5k 96.2% | ≈ Full 96.1% |
| CIFAR10 | BLS-InfoBatch | 30% | 95.5% | ≈ Full 95.6% |

BLS acts as a transparent proxy within both InfoBatch and SeTa frameworks, requiring only 3 lines of code (vs. 33+ lines of invasive modification in InfoBatch). Downstream pruning algorithms remain fully agnostic to the score source, with no changes needed to core scheduling logic or hyperparameters.

### Key Findings

- BLS is validated across 14 datasets, 11 tasks, and 18 models, achieving lossless pruning of 20%–50% of samples.
- When used as a proxy replacement for per-sample loss, performance is comparable to or better than the original methods.
- BLS is particularly well-suited for complex scenarios (multi-component losses, large-scale data) where per-sample loss is difficult to obtain.
- BLS is initialized with the mean loss of the first batch and subsequently updated only when a sample appears in the current batch.

## Highlights & Insights

- A signal-processing perspective (low-pass filtering) provides rigorous theoretical grounding for BLS.
- The minimalist 3-line implementation substantially lowers the barrier to adoption.
- Decoupling "sample scoring" from "sample selection" enables combination with any loss-based pruning strategy.
- The frequency separation assumption is intuitively clear and empirically validated.

## Limitations & Future Work

- The EMA decay factor $\alpha$ requires task-specific tuning.
- BLS may be less accurate in the very early stages of training, before sufficient score accumulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using batch loss as a proxy for per-sample loss is a novel idea.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Signal-processing theoretical analysis is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 datasets, 11 tasks, 18 models.
- **Practical Value**: ⭐⭐⭐⭐⭐ — 3 lines of code; extremely high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)
- [\[CVPR 2026\] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)
- [\[CVPR 2026\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)

</div>

<!-- RELATED:END -->
