---
title: >-
  [Paper Note] Self-Supervised Any-Point Tracking by Contrastive Random Walks
description: >-
  [ECCV 2024][Video Understanding][Tracking Any Point] Proposes GMRW (Global Matching Random Walk) which combines a global matching Transformer architecture with a contrastive random walk self-supervised objective, achieving robust "Tracking Any Point" (TAP) performance without annotations for the first time, and designs label warping data augmentation to prevent the Transformer from learning shortcut solutions.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Tracking Any Point"
  - "Self-Supervised Learning"
  - "Contrastive Random Walk"
  - "Global Matching Transformer"
  - "Cycle Consistency"
date: 2026-05-08
content_hash: 361389d4fec0d05a
---

# Self-Supervised Any-Point Tracking by Contrastive Random Walks

**Conference**: ECCV 2024  
**arXiv**: [2409.16288](https://arxiv.org/abs/2409.16288)  
**Code**: [https://ayshrv.com/gmrw](https://ayshrv.com/gmrw)  
**Area**: Video Understanding (Point Tracking / Self-Supervised Learning)  
**Keywords**: Tracking Any Point, Self-Supervised Learning, Contrastive Random Walk, Global Matching Transformer, Cycle Consistency

## TL;DR

Proposes GMRW (Global Matching Random Walk) which combines a global matching Transformer architecture with a contrastive random walk self-supervised objective, achieving robust "Tracking Any Point" (TAP) performance without annotations for the first time, and designs label warping data augmentation to prevent the Transformer from learning shortcut solutions.

## Background & Motivation

**Tracking Any Point (TAP)** is an emerging video understanding task: given a query position of an arbitrary physical point in a video, the goal is to predict its trajectory and visibility across all frames. Unlike optical flow (which only handles local motion between adjacent frames) or semantic tracking (which might match to the same category but a different physical point), TAP requires precise tracking of the exact same physical point.

Limitations of Prior Work:
- **Supervised methods** (TAP-Net, PIPs, TAPIR, CoTracker) rely on GT trajectories from synthetic data for training, and annotation data is limited.
- **Self-supervised optical flow methods** (unsupervised versions of RAFT, ARFlow) can only predict short-range motion, and long-term tracking accumulates errors through chained propagation.
- **Diffusion feature methods** (DIFT) are suitable for semantic correspondence but not physical point tracking.
- **Early contrastive random walks** (CRW) operate on coarse-grained patches (7×7), yielding extremely low spatial precision.

**Core Motivation**: Utilise the "all-to-all comparison" capability of global matching Transformers (GMFlow architecture) to define the transition matrix of a random walk, achieving precise self-supervised point tracking on high resolution (64×64 grid).

## Method

### Overall Architecture

1. A CNN extracts features for each frame and adds positional encodings.
2. Pairwise frames pass through a 6-layer self-attention/cross-attention/FFN global matching Transformer, outputting correlation features $F_t$, $F_{t+1}$.
3. The transition matrix $A_t^{t+1} = \text{softmax}(F_t F_{t+1}^\top / \tau)$ is computed.
4. Self-supervised training is conducted using a contrastive random walk loss (cycle consistency).
5. During inference, the expected coordinate calculated from the transition matrix is used as the trajectory prediction.

### Key Designs

1. **Global Matching Transition Matrix**:

    - A GMFlow-style Transformer architecture is adopted for global frame-to-frame matching.
    - Compared to early CRW (7×7 coarse-grained patches), GMRW operates on a 64×64 resolution, significantly improving spatial precision.
    - The "all-to-all comparison" mechanism allows the model to consider a massive number of matching hypotheses simultaneously, generating richer contrastive learning signals.
    - The transition matrix $A_t^{t+1}$ is directly used as the probabilistic transition matrix of the random walk, eliminating the need for additional coarse-to-fine matching steps.

2. **Label Warping to Prevent Shortcut Solutions**:

    - A classic shortcut in cycle-consistency training: the model ignores visual elements and matches solely based on positional encodings (since positions do not change, returning to the origin is a trivial solution).
    - Tang et al. proposed using different crop augmentations for forward/backward cycles, but this is ineffective for Transformers—the Transformer has enough global self-attention layers to "undo" spatial transformations.
    - **Innovation of GMRW**: Instead of warping features, it warps the labels.
    - Different spatial transformations $T^f$, $T^b$ are applied to the palindromic sequence $[T^f(I_1), T^f(I_2), T^b(I_1)]$.
    - The loss becomes $\mathcal{L}_{crw} = \mathcal{L}_{CE}(A_s, T_f^b(I))$, where $T_f^b(I)$ is the transformed identity matrix.
    - The model must find the correct correspondences across different spatial transformations and cannot "cheat" using positional encodings.

3. **Sampling Stride Strategy**:

    - Feature sampling with different strides (stride $s \in \{1, 2, 4\}$) is achieved by upsampling the original images.
    - $s=1$ achieves the highest spatial accuracy (but is computationally heavy), while $s=4$ is faster but coarser.
    - $s=2$ is used during training, and the model can flexibily switch strides during evaluation.

### Loss & Training

Contrastive random walk loss (CRW) + optional smoothness loss:

$$\mathcal{L}_{total} = \mathcal{L}_{crw} + \lambda_s \mathcal{L}_{smooth}$$

- **CRW Loss**: Maximises the probability of the random walk returning to the starting point in a palindromic sequence.
  $$\mathcal{L}_{crw} = \mathcal{L}_{CE}(A_t^{t+1} A_{t+1}^t, I)$$
- **Smoothness Loss**: Edge-aware second-order derivative regularisation that promotes spatial smoothness of the motion field.
- Training Data: Kubric synthetic dataset (38,325 videos), **without using any annotations**.
- An "in the wild" version is also trained on Kinetics-400.

## Key Experimental Results

### Main Results

TAPVid benchmark test (Kubric and DAVIS, comparison of self-supervised methods):

| Method | Type | Kubric AJ↑ | Kubric δ_avg↑ | DAVIS AJ↑ | DAVIS δ_avg↑ |
|------|------|-----------|---------------|----------|--------------|
| CRW-C | Self-supervised | 31.4 | 48.1 | 7.7 | 13.5 |
| CRW-D | Self-supervised | 35.8 | 52.4 | 23.6 | 38.0 |
| DIFT-D | Self-supervised | 41.6 | 59.8 | 29.7 | 48.2 |
| FlowWalk-C | Self-supervised | 49.4 | 66.7 | 35.2 | 51.4 |
| ARFlow-C | Self-supervised | 52.3 | 68.1 | 35.0 | 51.8 |
| **GMRW-C** | **Self-supervised** | **54.2** | **72.4** | **41.8** | **60.9** |
| TAP-Net | Supervised | 65.4 | 77.7 | 38.4 | 53.1 |
| TAPIR | Supervised | 84.7 | 92.1 | 61.3 | 73.6 |

GMRW leads comprehensively among self-supervised methods, exceeding FlowWalk by 4.8 points on Kubric AJ and 6.6 points on DAVIS AJ. It even surpasses the supervised TAP-Net on DAVIS (41.8 vs 38.4 AJ).

### Ablation Study

Analysis of model variants (AJ/δ_avg/OA on Kubric + DAVIS):

| Configuration | Kubric AJ↑ | Kubric δ_avg↑ | DAVIS AJ↑ | DAVIS δ_avg↑ | Description |
|------|-----------|---------------|----------|--------------|------|
| Supervised Oracle | 63.7 | 83.2 | 39.1 | 59.6 | Upper bound |
| Base CRW | 25.4 | 39.3 | 10.4 | 19.2 | w/o label warping |
| + Label Warping | 45.3 | 62.2 | 32.1 | 48.9 | **+19.9 AJ, core contribution** |
| + Smoothness | 49.0 | 66.7 | 33.0 | 50.5 | +3.7 |
| + Train stride s=2 | 47.7 | 65.6 | 34.5 | 52.1 | Slight difference |
| Eval stride s=4 | 37.8 | 53.8 | 23.5 | 38.4 | Coarse stride performance drop |
| Eval stride s=1 | 54.2 | 72.4 | 41.8 | 60.9 | Optimal with finest stride |
| Kinetics Training | 47.5 | 65.0 | 34.6 | 52.6 | Also trainable on real videos |

### Key Findings

- **Label Warping is the most critical contribution**: this single component improves AJ from 25.4 to 45.3 (+19.9); without it, the Transformer easily learns a shortcut solution.
- **Evaluation stride has a immense impact on performance**: the AJ gap between $s=1$ and $s=4$ on Kubric is 16.4, demonstrating that fine-grained matching is indeed crucial.
- **Training on Kubric generalises well to real DAVIS videos**: although the training data is synthetic, the feature correspondence capability transfers well.
- **Training on Kinetics achieves similar results** (AJ 34.6 vs. 34.5 trained on Kubric evaluated on DAVIS), indicating that the method does not rely on synthetic data.
- **A gap still remains between self-supervised and supervised methods**: there is still a noticeable performance gap compared to TAPIR (84.7 Kubric AJ), but considering the complete lack of annotations, the results are highly encouraging.
- **Visibility prediction is achieved through a cycle consistency threshold**: if the displacement error of the forward-backward loop exceeds $\tau_{cyc}=3$ pixels, the point is marked as occluded.

## Highlights & Insights

- **Simple and efficient self-supervised solution**: The overall method is conceptually simple—global matching + random walk + label warping, without requiring complex multi-stage training or coarse-to-fine matching.
- **Highly inspiring label warping**: Data augmentation strategies that worked historically for CNNs fail on Transformers because the global receptive field of the Transformer can easily "invert" spatial transformations. Warping the labels instead of features is an elegant solution.
- **Dual advantages of global matching**: (1) High spatial accuracy (64×64 vs. early 7×7), (2) considering a vast number of paths per iteration, providing richer gradient signals.
- Visibility detection requires no additional modules, directly reusing the cycle consistency check.

## Limitations & Future Work

- The gap with supervised SOTAs (TAPIR/CoTracker) remains significant, especially in occlusion handling and long-term tracking.
- Only single-scale matching is used (the original GMFlow design supports dual-scale); incorporating multi-scale might yield further improvements.
- The computational overhead of evaluation stride $s=1$ is high: global matching on a 64×64 resolution translates to a 4096×4096 attention matrix.
- The potential of training on large-scale unlabeled videos (e.g., using YouTube data) has not been explored.
- Cycle consistency is naturally unable to handle "disappearance and reappearance" occlusion patterns.

## Related Work & Insights

- CRW (Contrastive Random Walk) is the theoretical foundation of this work; GMRW upgrades it from coarse-grained CNNs to high-precision global Transformers.
- The global matching architecture of GMFlow is creatively applied to self-supervised learning, whereas its original design was for supervised optical flow estimation.
- Label warping may also inspire other tasks using Transformers + cycle consistency (e.g., self-supervised depth estimation, 3D correspondence learning).
- Proves that the trend of "self-supervised approaching supervised baselines" also holds true in the field of point tracking.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of global matching and CRW is novel, and label warping cleverly resolves Transformer-specific issues)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four TAPVid benchmarks + comparisons with various self-supervised/supervised baselines + complete ablations)
- Writing Quality: ⭐⭐⭐⭐ (The methodology is clearly described, but the paper is relatively short)
- Value: ⭐⭐⭐⭐ (The self-supervised point tracking direction shows potential, but is still some distance away from practical application)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)
- [\[CVPR 2025\] ETAP: Event-based Tracking of Any Point](../../CVPR2025/video_understanding/etap_event-based_tracking_of_any_point.md)
- [\[ECCV 2024\] Local All-Pair Correspondence for Point Tracking](local_all-pair_correspondence_for_point_tracking.md)
- [\[ICLR 2026\] TAPTRv3: Spatial and Temporal Context Foster Robust Tracking of Any Point in Long Video](../../ICLR2026/video_understanding/taptrv3_spatial_and_temporal_context_foster_robust_tracking_of_any_point_in_long.md)
- [\[ECCV 2024\] TimeCraft: Navigate Weakly-Supervised Temporal Grounded Video Question Answering via Bi-directional Reasoning](timecraft_navigate_weakly-supervised_temporal_grounded_video_question_answering_.md)

</div>

<!-- RELATED:END -->
