---
title: >-
  [Paper Note] TC-Stereo: Temporally Consistent Stereo Matching
description: >-
  [ECCV 2024][3D Vision][Stereo Matching] Proposed TC-Stereo, which achieves temporally consistent stereo matching through temporal disparity completion for good initialization, temporal state fusion to maintain hidden state coherence, and dual-space (disparity + disparity gradient) iterative refinement to improve ill-posed regions.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Stereo Matching"
  - "Temporal Consistency"
  - "Disparity Completion"
  - "Dual-Space Optimization"
  - "Video Stereo"
date: 2026-05-08
content_hash: 75c651a23e3fbcf5
---

# TC-Stereo: Temporally Consistent Stereo Matching

**Conference**: ECCV 2024  
**arXiv**: [2407.11950](https://arxiv.org/abs/2407.11950)  
**Code**: [GitHub](https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching)  
**Area**: 3D Vision  
**Keywords**: Stereo Matching, Temporal Consistency, Disparity Completion, Dual-Space Optimization, Video Stereo

## TL;DR

Proposed TC-Stereo, which achieves temporally consistent stereo matching through temporal disparity completion for good initialization, temporal state fusion to maintain hidden state coherence, and dual-space (disparity + disparity gradient) iterative refinement to improve ill-posed regions.

## Background & Motivation

### Background

**Background**: Existing stereo matching methods perform independent inference frame-by-frame, leading to temporal inconsistency. The inconsistency stems from two main sources: (1) each frame searches for disparity globally from scratch, resulting in large update steps and high variability; (2) inherent ambiguities in ill-posed regions (e.g., occlusions, reflections) lead to unstable outputs under frame-to-frame appearance changes. This work addresses these issues by localizing the search range and improving ill-posed regions.

### Proposed Solution

**Goal**: ### Overall Architecture

TC-Stereo processes stereo video sequences online: (1) Temporal Disparity Completion (TDC)—generates initial dense disparity from semi-dense disparity projected from preceding frames; (2) Temporal State Fusion—fuses the current completed state with the historical refined state; (3) Dual-Space Iterative Refinement—alternately optimizes in the disparity space and disparity gradient space.   


## Method

### Overall Architecture

TC-Stereo processes stereo video sequences online: (1) Temporal Disparity Completion (TDC)—generates initial dense disparity from semi-dense disparity projected from preceding frames; (2) Temporal State Fusion—fuses the current completed state with the historical refined state; (3) Dual-Space Iterative Refinement—alternately optimizes in the disparity space and disparity gradient space.

### Key Designs

**Source of Semi-Dense Disparity**: The first frame is obtained from the cost volume via winner-take-all filtering with a confidence threshold; subsequent frames project the preceding disparity map to the current frame view using camera poses, which naturally produces holes in non-overlapping regions.

**Temporal Disparity Completion**: A lightweight encoder-decoder network that takes the encoder's context features, a semi-dense disparity, and a sparsity mask as inputs, and outputs a dense disparity and state features.

**Temporal State Fusion**: Uses a GRU-like module to fuse the current state $c^t$ and the historical hidden state $h_{N-1}^{t-1}$, addressing two challenges: (1) the historical state only encodes the information of the preceding viewpoint; (2) there are no historical states in non-overlapping regions.

**Dual-Space Refinement**: In the disparity space, a multi-level GRU is used to update the disparity (similar to RAFT-Stereo); in the gradient space, the disparity is converted to a gradient field and refined—leveraging the prior that real-world depths are typically locally smooth. Through gradient-guided disparity propagation, neighboring disparities are propagated to the current pixel based on the local plane assumption: $\hat{d}_n = d + (u_n-u)\frac{\partial d}{\partial u} + (v_n-v)\frac{\partial d}{\partial v}$, followed by a softmax weighted sum to obtain the final disparity.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{cv} + \mathcal{L}_{disp} + \mathcal{L}_{grad}$$

- $\mathcal{L}_{cv}$: Cost volume contrastive loss, maximizing similarity at the ground truth disparity and enforcing a margin
- $\mathcal{L}_{disp}$: L1 disparity loss over three stages (completion, refinement, and propagation), using exponentially decaying weights
- $\mathcal{L}_{grad}$: L1 loss of the disparity gradient after gradient-space refinement and propagation

## Key Experimental Results

### TartanAir Ablation Study


### Main Results

| Setting | Iters | Temporal | State Fusion | TDC | Dual Space | ALL >1px↓ | OCC >1px↓ | ALL \|Δd\|↓ | OCC \|Δd\|↓ |
|------|------|----------|----------|-----|--------|-----------|-----------|-------------|-------------|
| RAFT-Stereo (5iter) | 5 | ✗ | ✗ | ✗ | ✗ | 8.04% | 34.91% | 0.35 | 1.15% |
| RAFT-Stereo (32iter) | 32 | ✗ | ✗ | ✗ | ✗ | 6.02% | 25.89% | 0.28 | 0.93 |
| +Temporal+Fusion | 5 | ✓ | ✓ | ✗ | ✗ | 6.98% | 29.89% | 0.25 | 0.86 |
| +Temporal+Fusion+TDC | 5 | ✓ | ✓ | ✓ | ✗ | 6.10% | 25.97% | 0.23 | 0.78 |
| +Temporal+Fusion+Dual Space | 5 | ✓ | ✓ | ✗ | ✓ | 6.28% | 25.89% | 0.21 | 0.74 |
| **TC-Stereo (Full)** | 5 | ✓ | ✓ | ✓ | ✓ | **5.98%** | **24.67%** | **0.21** | **0.71** |

### KITTI 2015 Leaderboard

TC-Stereo ranks second while achieving better efficiency compared to other SOTA methods.

### Key Findings

- Achieves better accuracy and consistency with only 5 iterations than RAFT-Stereo with 32 iterations
- Temporal jitter |Δd| in occluded regions is reduced from 1.15 to 0.71, representing a 38% improvement
- TDC and dual-space refinement are complementary in improving occluded regions: TDC provides a good initialization, while dual-space refinement enhances details in ill-posed regions
- Directly using the state of the preceding frame for initialization (similar to XR-Stereo) degrades performance, demonstrating the necessity of state fusion

## Highlights & Insights

1. **Dual-space refinement serves as the core innovation**: It constrains smoothness in the gradient space and iteratively propagates it globally, which is particularly effective for reflection and occlusion areas
2. Temporal disparity completion restricts the search range from global to local, leading to smaller and more stable update steps
3. Two hierarchical temporal consistency evaluation metrics are designed (absolute difference |Δd| and error divergence Relu(Δe)), providing a more comprehensive evaluation

## Limitations & Future Work

- Requires camera poses as input
- Robustness to extremely fast motions or large occlusion changes needs further improvement
- Dual-space refinement introduces a certain amount of computational overhead

## Related Work & Insights

Incorporating video information into stereo matching has become increasingly popular (e.g., Dynamic-Stereo, TemporalStereo). The proposed disparity gradient space refinement strategy could be extended to other tasks such as monocular depth estimation.

## Rating

- Novelty: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](../../ICCV2025/3d_vision/stereo_any_video_temporally_consistent_stereo_matching.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](../../CVPR2025/3d_vision/defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[ECCV 2024\] TCC-Det: Temporarily Consistent Cues for Weakly-Supervised 3D Detection](tcc-det_temporarily_consistent_cues_for_weakly-supervised_3d_detection.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[ECCV 2024\] MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)

</div>

<!-- RELATED:END -->
