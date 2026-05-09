---
title: >-
  [Paper Note] Towards Efficient General Feature Prediction in Masked Skeleton Modeling
description: >-
  [ICCV 2025][Video Understanding][masked skeleton modeling] This paper proposes GFP (General Feature Prediction), a framework that elevates the reconstruction target in masked skeleton modeling from low-level joint coordinates to multi-scale high-level semantic feature prediction. Coupled with a lightweight Target Generation Network and an information maximization constraint, GFP achieves a 6.2× training speedup while attaining state-of-the-art performance.
tags:
  - ICCV 2025
  - Video Understanding
  - masked skeleton modeling
  - high-level semantic prediction
  - target generation network
  - self-supervised learning
  - action recognition
date: 2026-05-08
content_hash: 3c16f2af6e4cf6fa
---

# Towards Efficient General Feature Prediction in Masked Skeleton Modeling

**Conference**: ICCV 2025  
**arXiv**: [2509.03609](https://arxiv.org/abs/2509.03609)  
**Code**: None  
**Area**: Skeleton Action Recognition / Self-Supervised Learning  
**Keywords**: masked skeleton modeling, high-level semantic prediction, target generation network, self-supervised learning, action recognition

## TL;DR

This paper proposes GFP (General Feature Prediction), a framework that elevates the reconstruction target in masked skeleton modeling from low-level joint coordinates to multi-scale high-level semantic feature prediction. Coupled with a lightweight Target Generation Network and an information maximization constraint, GFP achieves a 6.2× training speedup while attaining state-of-the-art performance.

## Background & Motivation

- Masked skeleton modeling follows the MAE paradigm, randomly masking joints and reconstructing missing coordinates.
- Existing methods suffer from two critical issues:
  1. **Heavy decoder computation**: A 90% masking ratio combined with a Transformer decoder produces extremely long decoding sequences (750 low-level targets), resulting in slow training.
  2. **Lack of semantic supervision**: Low-level coordinate reconstruction lacks high-level spatiotemporal semantic guidance, creating a semantic gap with downstream tasks.
- Although S-JEPA uses model-generated features as targets, its patch-level prediction requires a large EMA encoder (28.32G FLOPs) and converges very slowly (requiring 1200 epochs).
- **Core Idea**: Replace the large number of low-level targets (750) with fewer high-level semantic targets (251), simultaneously improving feature quality and training efficiency.

## Method

### Overall Architecture

GFP establishes a bidirectional learning paradigm between an encoder–decoder architecture and a Target Generation Network (TGN). The encoder processes visible joint features; the decoder progressively predicts multi-scale high-level features (from short-term motion patterns to global action semantics); and the TGN provides online supervision through consistency learning. Variance–covariance regularization prevents representation collapse.

### Key Designs

1. **High-Level Feature Prediction**: Instead of conventional per-patch coordinate reconstruction, a hierarchical prediction objective is designed:

    - The learning target is elevated from $\mathcal{L}_p = \frac{1}{N}\|f(E_N) - X_e\|^2_F$ (low-level) to $\mathcal{L}_p = \frac{1}{M}\|f(E_N) - g(X)\|^2_F$ (high-level).
    - Decoder inputs are progressively downsampled along the temporal axis via temporal average pooling (kernel sizes $t_1, t_2, \ldots$), building a pyramid structure through cascaded Transformer decoders.
    - Four hierarchical feature levels: $t_1=5$ (5-frame local motion), $t_2=10$ (10-frame mid-term dynamics), $t_3=30$ (30-frame long-range cycles), and global semantics.
    - The total number of prediction targets is reduced from 750 to **251**, and decoder FLOPs drop from 17.70G to **1.57G**.

2. **Target Generation Network (TGN)**: A lightweight multi-MLP structure that provides online supervision at each hierarchical level.

    - Each level employs an independent MLP: a 3-layer MLP (512 hidden units) for local semantic extraction and a 3-layer MLP (2048 hidden units) for global semantic extraction.
    - TGN takes motion features (inter-frame differences) as input, $X_e = X_e[1:] - X_e[:-1]$, avoiding the bias introduced by sharing the same input as the encoder.
    - Computational cost is only 0.64G FLOPs, compared to 28.32G for S-JEPA's EMA encoder.
    - Bidirectional learning: the decoder predicts TGN-generated targets, while TGN simultaneously adapts to the decoder's feature representations.

3. **Information Maximization Constraint**: Prevents the bidirectional learning from collapsing to a trivial solution.

    - **Variance regularization**: Ensures that the batch variance of each feature dimension exceeds a threshold $\gamma=1$:
    $$\mathcal{L}_{var} = \frac{1}{C_t}\sum_{i=1}^{C_t}\max(0, \gamma - \sqrt{\text{Var}(Z_{t_g}[:,i])})$$
    - **Covariance regularization**: Drives the covariance between feature dimensions toward zero, eliminating redundancy:
    $$\mathcal{L}_{cov} = \frac{1}{C_t}\sum_{i \neq j}[\text{Cov}(Z_{t_g})]^2_{i,j}$$
    - Inspired by VICReg's information-maximizing representation learning.

### Loss & Training

- Total loss: $\mathcal{L}_{total} = \lambda \mathcal{L}_{pred} + \mathcal{L}_{reg}$
- Regularization loss: $\mathcal{L}_{reg} = \sum_{j \in \mathcal{J}}(\alpha \mathcal{L}_{cov}(Z_{t_j}) + \beta \mathcal{L}_{var}(Z_{t_j}))$
- Hyperparameters: $\lambda=5,\ \alpha=5,\ \beta=1$
- Pre-training for 400 epochs; AdamW ($\beta_1=0.9,\ \beta_2=0.95$, weight decay 0.05)
- Learning rate: cosine annealing after 20 warmup epochs ($1\text{e-}3 \to 5\text{e-}4$)
- Encoder: 8 layers, 8-head attention with 256-dim embeddings and 1024-dim FFN
- Motion-aware masking strategy with segment length $l=4$

## Key Experimental Results

### Main Results (Tables)

**NTU-60 Skeleton Action Recognition (comparison with MAE-based methods, single RTX 4090):**

| Method | Target Type | Enc FLOPs | Dec FLOPs | TGN FLOPs | Training Time | Speedup | x-sub | x-view |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SkeletonMAE | Joint | 1.97G | 17.70G | - | 20h27m | 1× | 74.8 | 77.7 |
| MAMP | Motion | 1.97G | 17.70G | - | 20h27m | 1× | 84.9 | 89.1 |
| S-JEPA | Patch-level | 1.97G | 17.70G | 28.32G | 90h57m | 0.2× | 85.3 | 89.8 |
| **GFP** | **Hierarchical** | **1.97G** | **1.57G** | **0.64G** | **3h14m** | **6.2×** | **85.9** | **92.0** |

**NTU-120 + PKU-MMD II:**

| Method | NTU-120 x-sub | NTU-120 x-setup | PKU-II x-sub |
|:---:|:---:|:---:|:---:|
| MAMP | 78.6 | 79.1 | 53.8 |
| S-JEPA | 79.6 | 79.9 | 53.5 |
| **GFP** | **79.1** | **80.3** | **56.2** |

### Ablation Study (Tables)

**TGN Input Ablation (NTU-60):**

| TGN Input | x-sub | x-view |
|:---:|:---:|:---:|
| Joint coordinates | 85.0 | 90.9 |
| Masked joints | 84.2 | 90.3 |
| **Motion features (frame differences)** | **85.9** | **92.0** |

**Semi-Supervised Learning (NTU-60, 1%/10% labels):**

| Method | x-sub 1% | x-sub 10% | x-view 1% | x-view 10% |
|:---:|:---:|:---:|:---:|:---:|
| HaLP | 46.6 | 72.6 | 48.7 | 77.1 |
| USDRL | 57.3 | 80.2 | 60.7 | 84.0 |
| MAMP | 66.0 | 88.0 | 68.7 | 91.5 |
| S-JEPA | 67.5 | 88.4 | 69.1 | 91.4 |
| **GFP** | **71.8** | **88.7** | **72.9** | **92.1** |

### Key Findings

- **Significant training efficiency gains**: GFP completes pre-training in only 3h14m, **6.2×** faster than SkeletonMAE and **28×** faster than S-JEPA.
- High-level semantic targets outperform low-level reconstruction targets: removing TGN and directly predicting low-level targets (e.g., flattened vectors) leads to a notable performance drop.
- **Complementarity of hierarchical features**: Progressively removing global or local objectives both lead to performance degradation (Figure 3).
- Motion features (frame differences) are optimal as TGN input; masked inputs are detrimental.
- Action retrieval improvements are particularly pronounced: GFP outperforms MAMP by 17.1% on x-view (87.1 vs. 70.0), validating the value of high-level semantic targets for global representation learning.
- TGN architecture is robust: variants with 2/3/4 MLP layers show negligible performance differences.

## Highlights & Insights

- Precise problem diagnosis: the paper identifies computational redundancy and semantic insufficiency of low-level reconstruction as the current bottleneck, and addresses both issues directly and effectively.
- The hierarchical target design (5-frame → 10-frame → 30-frame → global) elegantly balances local motion details and global action understanding.
- Incorporating VICReg's information maximization constraint cleanly resolves the collapse problem in joint training.
- The lightweight TGN (0.64G FLOPs, 3-layer MLP) is far more efficient than S-JEPA's EMA encoder (28.32G FLOPs).

## Limitations & Future Work

- The hierarchical granularity (5/10/30 frames) is manually specified; adaptive determination strategies remain unexplored.
- Validation is limited to skeleton data and has not been extended to masked modeling on RGB video.
- The computational overhead of covariance regularization under large batch sizes warrants attention.
- Comparisons with methods that jointly combine contrastive learning and masked modeling are absent.

## Related Work & Insights

- This work demonstrates the principle that "**target quality matters more than quantity**" in masked modeling.
- The paradigm shift from low-level reconstruction to high-level semantic prediction is generalizable to masked modeling in other modalities.
- The lightweight online target generation scheme of TGN offers a viable alternative to large EMA-updated encoders.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of replacing low-level reconstruction with high-level semantic prediction is clear and effective, though the core components (pyramid decoding, VICReg) are adapted from prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple tasks (recognition, retrieval, semi-supervised learning), and comprehensive ablations (target type, TGN input, architecture, projector depth).
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is well articulated; efficiency comparisons are intuitive (Table 1 includes both FLOPs and training time).
- **Value**: ⭐⭐⭐⭐⭐ The combination of 6.2× speedup and state-of-the-art performance offers substantial practical value and establishes a new paradigm for skeleton self-supervised learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](../../ICLR2026/video_understanding/online_time_series_prediction_using_feature_adjustment.md)
- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)

</div>

<!-- RELATED:END -->
