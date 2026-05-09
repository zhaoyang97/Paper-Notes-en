---
title: >-
  [Paper Note] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning
description: >-
  [CVPR 2026][Medical Imaging][domain-incremental learning] This paper proposes the Residual SODAP framework, which jointly addresses representation adaptation (via α-entmax sparse prompt selection with residual aggregation) and classifier preservation (via statistical pseudo-feature replay and knowledge distillation) for domain-incremental learning without task IDs or data buffers, achieving state-of-the-art performance on three benchmarks: DR, Skin Cancer, and CORe50.
tags:
  - CVPR 2026
  - Medical Imaging
  - domain-incremental learning
  - prompt pool
  - α-entmax
  - pseudo-feature replay
  - drift detection
  - uncertainty weighting
date: 2026-05-08
content_hash: e34556e4a113cdc8
---

# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)
**Code**: None
**Area**: Continual Learning / Prompt-based CL
**Keywords**: domain-incremental learning, prompt pool, α-entmax, pseudo-feature replay, drift detection, uncertainty weighting

## TL;DR
This paper proposes the Residual SODAP framework, which jointly addresses representation adaptation (via α-entmax sparse prompt selection with residual aggregation) and classifier preservation (via statistical pseudo-feature replay and knowledge distillation) for domain-incremental learning without task IDs or data buffers, achieving state-of-the-art performance on three benchmarks: DR, Skin Cancer, and CORe50.

## Background & Motivation

### Root Cause

**Root Cause**: **State of the Field**: Existing prompt-based continual learning (PCL) methods suffer from two critical limitations: (1) suboptimal prompt selection — top-$k$ hard selection is non-differentiable with limited expressiveness, while softmax soft selection, though differentiable, accumulates noise by assigning non-zero weights to irrelevant prompts; (2) neglect of classifier-level forgetting — existing PCL methods focus primarily on prompt/prompt-pool design for improved representation adaptation, yet cross-composition diagnostic experiments reveal that unstable classifier decision boundaries are the dominant source of forgetting in domain-incremental learning.

### Starting Point

**Paper Goals**: How can a prompt-based CL framework simultaneously achieve high-quality representation adaptation and classifier-level knowledge preservation under strict constraints — no task IDs and no past data storage — to mitigate catastrophic forgetting?

## Method

### Overall Architecture
Four core components operate jointly on a frozen ViT backbone: (1) α-entmax sparse prompt selection with residual aggregation; (2) statistics-based pseudo-feature replay for classifier knowledge preservation; (3) prompt-usage-based domain drift detection (PUDD); and (4) uncertainty-weighted multi-objective optimization.

### Key Designs
1. **α-entmax Residual Prompt Selection**: The query is augmented via a memory bank (CLS token + global context + memory retrieval signal), and sparse prompt selection is performed in a bottleneck space using α-entmax ($\alpha = 1.5$), which automatically assigns exactly zero weight to irrelevant prompts. The prompt pool is partitioned into a frozen set $\mathcal{F}$ and an active set $\mathcal{A}$: the frozen set retains prior knowledge while the active set performs residual adaptation ($p_{out} = p_\mathcal{F} + 0.1 \cdot p_\mathcal{A}$).

2. **Statistical Knowledge Preservation**: At the end of each stage, class-level feature means and variances ($\mu_c, \sigma_c^2$) are stored using the Welford online algorithm. In the subsequent stage, classifier decision boundaries are preserved through two complementary paths: (a) real-feature distillation (aligning teacher and student head outputs via KL divergence on current data) and (b) pseudo-feature replay (sampling pseudo-features from $\mathcal{N}(\mu_c, \text{diag}(\sigma_c^2))$ and distilling them).

3. **PUDD Drift Detection**: Domain drift is detected by monitoring changes in prompt selection patterns, combining selection entropy variation (a z-score computed over short-term fluctuations) and usage-set variation (IoU between the current prompt usage set and a sliding-window history). The drift score $D$ proportionally determines the prompt pool expansion size.

4. **Uncertainty Weighting**: A log-variance $s_i$ is learned for each of the five loss terms (CE, real distillation, pseudo replay, diversity, norm) to enable automatic balancing: $\mathcal{L}_{total} = \sum_i (e^{-s_i}\mathcal{L}_i + s_i)$.

### Loss & Training
Five loss terms are automatically balanced via uncertainty weighting. Auxiliary losses include a diversity loss (penalizing similarity among frequently co-activated prompts) and a norm regularization (constraining active prompt magnitudes to serve as residuals). Optimization uses AdamW with lr = 1e-3, cosine schedule, 100 epochs, and early stopping with patience 5.

## Key Experimental Results

| Benchmark | Method | AvgACC↑ | AvgF↓ |
|-----------|--------|---------|-------|
| DR | OS-Prompt++ | 0.769 | 0.113 |
| DR | Coda-Prompt | 0.688 | 0.140 |
| DR | **Residual SODAP** | **0.850** | **0.047** |
| Skin Cancer | OS-Prompt++ | 0.725 | 0.063 |
| Skin Cancer | **Residual SODAP** | **0.760** | 0.031 |
| CORe50 (11-stage) | DER++ | 0.994 | 0.061 |
| CORe50 (11-stage) | **Residual SODAP** | **0.995** | **0.003** |

### Ablation Study
- Removing the Query Enhancer degrades AvgACC by 4.2 pp, highlighting its critical role in reliable prompt selection.
- Removing the diversity loss reduces AvgACC by 3.2 pp and increases AvgF by 2.5 pp, confirming its dual role in preventing prompt collapse and retaining prior knowledge.
- Real distillation and pseudo replay each contribute 1.5–2.2 pp accuracy gains independently.
- An accuracy–forgetting trade-off exists across component configurations; the full model resides at the optimal point on this trade-off curve.

## Highlights & Insights
- **The backbone × classifier cross-composition diagnostic analysis clearly exposes classifier-level forgetting — a previously overlooked problem in PCL** — and constitutes a highly compelling motivation.
- α-entmax elegantly resolves the dilemma between top-$k$ (non-differentiable) and softmax (noise accumulation) selection, achieving both exact zero weights and differentiability.
- The statistical pseudo-feature replay is extremely lightweight: storing only per-class means and variances and sampling from a Gaussian suffices to replay past representations.
- Uncertainty weighting eliminates the need for manual tuning of the five loss coefficients.
- A forgetting rate of only 0.003 on the 11-stage CORe50 benchmark demonstrates remarkable stability under long-sequence domain drift.

## Limitations & Future Work
- Validation is limited to the domain-incremental learning (DIL) setting; extension to class-incremental learning (CIL) has not been explored.
- The Gaussian assumption underlying pseudo-feature replay may fail when true feature distributions are non-Gaussian.
- PUDD introduces numerous hyperparameters (window size, threshold, $D_{max}$, etc.); while loss weights require no manual tuning, other hyperparameters are added in their place.
- The prompt pool expands continuously (60 → 84 → 94), leading to linear parameter growth under long-term deployment.

## Related Work & Insights
- **OS-Prompt++**: A PCL method that lacks a classifier preservation mechanism; achieves AvgACC of 0.769 vs. 0.850 on DR.
- **Coda-Prompt**: Prompt learning with orthogonality regularization; achieves only 0.688 AvgACC on DR.
- **DER++**: Requires a replay buffer storing past data; even with data storage, it underperforms the proposed data-free approach.
- **Online EWC**: A classical regularization-based method with AvgF of 0.174, far worse than the proposed method's 0.047.

## Related Work & Insights
- The insight of "classifier-level forgetting" generalizes beyond PCL — any CL method employing a shared classifier may suffer from this issue.
- The α-entmax sparse selection mechanism is applicable to other scenarios requiring subset selection from large pools, such as MoE routing.
- The statistical pseudo-feature replay paradigm is transferable to any privacy-sensitive setting where data storage is prohibited.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The joint framework combining classifier preservation and prompt adaptation is novel, though individual components (α-entmax, KD, uncertainty weighting) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks, comprehensive ablations, cross-composition diagnostics, and prompt visualization analyses are all included.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation analysis (Fig. 1) is persuasive, and the method description is detailed with mathematical rigor.
- **Value**: ⭐⭐⭐⭐ The work has direct practical value for data-free domain-incremental learning in medical imaging.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[CVPR 2026\] Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning](active_inference_for_micro-gesture_recognition_efe-guided_temporal_sampling_and_.md)

<!-- RELATED:END -->
