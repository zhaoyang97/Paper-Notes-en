---
title: >-
  [Paper Note] BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation
description: >-
  [AAAI2026][Video Understanding][event camera] This paper proposes the Bidirectional Adaptive Temporal Correlation (BAT) framework, which converts temporally dense motion cues from event cameras into spatially dense cues, achieving high-accuracy event-based optical flow estimation and ranking first on the DSEC-Flow benchmark.
tags:
  - AAAI2026
  - Video Understanding
  - event camera
  - optical flow
  - bidirectional temporal correlation
  - deformable attention
  - event-based vision
date: 2026-05-08
content_hash: 871920318ce6911e
---

# BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation

**Conference**: AAAI2026
**arXiv**: [2503.03256](https://arxiv.org/abs/2503.03256)
**Code**: [gangweix/BAT](https://github.com/gangweix/BAT)
**Area**: Video Understanding
**Keywords**: event camera, optical flow, bidirectional temporal correlation, deformable attention, event-based vision

## TL;DR

This paper proposes the Bidirectional Adaptive Temporal Correlation (BAT) framework, which converts temporally dense motion cues from event cameras into spatially dense cues, achieving high-accuracy event-based optical flow estimation and ranking first on the DSEC-Flow benchmark.

## Background & Motivation

Event cameras continuously capture asynchronous brightness changes with high dynamic range and high temporal resolution, making them well suited for optical flow estimation under fast motion and challenging illumination. However, event data exhibits **spatial sparsity**, which limits the effectiveness of directly applying image-based optical flow frameworks (e.g., RAFT). The representative method E-RAFT constructs a correlation volume between only two event representations, yielding insufficient motion cues; TMA extends this to multi-frame forward correlation volumes but ignores backward temporal motion cues, leaving a performance bottleneck.

Core insight: both events and motion are continuous and fine-grained in time. **Both forward and backward temporal motion cues are essential for accurate optical flow estimation.** Moreover, motion may vary non-uniformly over time, necessitating adaptive aggregation of bidirectional motion cues to ensure temporal consistency.

## Core Problem

1. Event data is spatially sparse and the resulting correlation volumes carry insufficient motion information — how can richer motion cues be obtained?
2. Existing methods exploit only unidirectional (forward) temporal correlation — how can bidirectional temporal cues be fully leveraged?
3. Non-uniform motion causes linear sampling to introduce inconsistent motion features — how can consistent motion information be adaptively aggregated?

## Method

### Overall Architecture

The voxel grid representations of the reference and target event streams are each divided into $N$ groups. After extracting features from the resulting $2N$ groups, bidirectional temporal correlation is computed. A spatially adaptive temporal motion aggregation module then fuses the motion features, and a ConvGRU iteratively updates the optical flow estimate.

### 1. Event Representation and Feature Extraction

- Event streams are converted to voxel grids $\bm{V} \in \mathbb{R}^{B \times H_0 \times W_0}$ using bilinear kernels to discretize along the temporal dimension.
- The voxel grids of the two event streams are each divided into $N$ groups along the temporal dimension, each group containing $B/N$ time bins.
- All $2N$ groups are processed by a shared-weight feature extraction network (6 residual blocks), yielding features $\bm{F}_n \in \mathbb{R}^{D \times H \times W}$.
- $\bm{F}_N$ serves as the reference frame feature and $\bm{F}_{2N}$ as the target frame feature.

### 2. Bidirectional Temporal Correlation (BTC)

Given the current flow estimate $\bm{f}$ and a linear motion assumption, the inter-group flow is derived as $\bm{df} = \bm{f}/N$:

- **Forward temporal correlation**: correlations between the reference feature $\bm{F}_N$ and subsequent features $\bm{F}_{N+j}$ ($j=1,...,N$) are computed via warping followed by dot-product sampling over a local grid.
- **Backward temporal correlation**: correlations between the reference feature $\bm{F}_N$ and preceding features $\bm{F}_{N-j}$ ($j=1,...,N-1$) are computed analogously.

This yields $N$ forward and $N-1$ backward correlation maps. Backward correlation is particularly effective for handling occlusions caused by objects leaving the target frame.

### 3. Adaptive Temporal Sampling (ATS)

Prior work treats the sampling radius $r$ as a fixed hyperparameter. BAT introduces a learnable scale factor $\alpha$ to obtain an adaptive sampling radius $lr = \alpha \cdot r$, allowing the optimal sampling range to be learned during training and maintaining temporal consistency.

### 4. Spatially Adaptive Temporal Motion Aggregation (SATMA)

This module addresses inconsistent motion features arising from non-uniform motion:

- Correlation features are encoded by a MotionEncoder into motion features $\bm{M}_j^{fwd}$ / $\bm{M}_j^{bwd}$.
- The target motion feature $\bm{M}_N^{fwd}$ is concatenated with neighboring motion features and passed through convolution + Sigmoid to generate a spatial attention map $\bm{A}_{spa}$.
- **Deformable attention** aggregates relevant information from the target motion feature for each neighboring motion feature: sparse sampling offsets are predicted, keys and values are sampled, and attention is computed.
- Fusion: $\bm{M}_j^{fuse} = \bm{A}_{spa} \odot \bm{M}_j^{agg} + \bm{M}_j^{fwd}$.
- The fused forward and backward motion features are jointly fed into the ConvGRU to update the optical flow.

### 5. Loss & Training

Following RAFT, an $l_1$ loss with exponentially increasing weights over $K$ iterations is adopted: $\mathcal{L} = \sum_{i=1}^{K} \gamma^{K-i} \|\bm{f}^i - \bm{f}^{gt}\|_1$.

## Key Experimental Results

| Benchmark | Metric | BAT | 2nd Best | Gain |
|-----------|--------|-----|----------|------|
| DSEC-Flow | 1PE↓ | **7.715** | IDNet 10.069 | 23.4% |
| DSEC-Flow | EPE↓ | **0.655** | IDNet 0.719 | 8.9% |
| MVSEC dt=4 | EPE↓ | **0.53** | TMA 0.70 | 24.3% |
| MVSEC dt=4 | %Out↓ | **0.71** | TMA 1.08 | 34.3% |
| MVSEC dt=1 | EPE↓ | **0.21** | TMA 0.25 | 16.0% |

Ablation Study (DSEC-Flow, 1PE metric):

- Baseline (TMA, N=3): 9.123
- +BTC: 8.279 (backward correlation yields significant improvement)
- +BTC+ATS: 8.179 (adaptive sampling provides further gains)
- Full (BAT): **7.715** (SATMA effectively suppresses inconsistent motion)

Attention type comparison: deformable attention (7.715) > dense attention (8.049) > spatial-reduction attention (8.731).

Future flow prediction: using only past events, BAT (bwd corr) achieves 1PE=33.026, substantially outperforming E-RAFT warm-start at 85.378.

## Highlights & Insights

1. **Bidirectional temporal correlation** is a novel design that converts temporally dense cues into spatially dense cues, directly addressing the core bottleneck of spatial sparsity in event data.
2. **Future flow prediction** capability is distinctive: predicting future optical flow from past events alone has significant value for real-time applications such as autonomous driving and UAVs.
3. **Occlusion handling**: backward temporal correlation naturally benefits scenarios where objects move out of the target frame.
4. The use of deformable attention in SATMA is both efficient and focused on relevant motion features.
5. BAT achieves substantial state-of-the-art results on both major benchmarks, DSEC-Flow and MVSEC.

## Limitations & Future Work

- In scenes with **rapid motion variation** (e.g., severe camera shake), backward and forward temporal motion cues may diverge significantly, limiting the benefit of backward correlation.
- Inter-frame flow is derived under a linear motion assumption, which may be insufficiently accurate for highly nonlinear motion.
- The temporal grouping number $N$ is a fixed hyperparameter ($N=3$); a more flexible dynamic grouping strategy may yield further improvements.
- Unsupervised/self-supervised training paradigms are not explored; the method relies on ground-truth flow annotations.

## Related Work & Insights

| Method | Temporal Cues | Motion Aggregation | Future Prediction |
|--------|---------------|--------------------|-------------------|
| E-RAFT | Two-frame correlation | None | Not supported |
| TMA | Forward multi-frame correlation | Simple concatenation | Not supported |
| IDNet | No correlation volume; iterative deblurring | — | Not supported |
| **BAT** | **Bidirectional multi-frame correlation** | **SATMA (deformable attn)** | **Supported** |

Distinction from VideoFlow: VideoFlow simultaneously estimates multi-frame bidirectional optical flow, whereas BAT aggregates bidirectional temporal motion cues into the target frame and focuses on single-frame optical flow estimation.

The "temporally dense → spatially dense" conversion paradigm is generalizable to other event camera tasks (e.g., depth estimation, scene flow). The adaptive sampling radius design can be adopted in other correlation-volume-based tasks. The deformable attention + spatial attention fusion mechanism in SATMA has broad utility for handling temporal inconsistency. The future flow prediction capability warrants further validation in downstream tasks such as collision avoidance and motion planning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of bidirectional temporal correlation and adaptive aggregation is effective and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons on two benchmarks, detailed ablations, and future prediction experiments.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and the method is systematically described.
- Value: ⭐⭐⭐⭐ — State-of-the-art on DSEC-Flow; future flow prediction has practical application prospects.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](../../ICCV2025/video_understanding/unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[CVPR 2026\] LAOF: Robust Latent Action Learning with Optical Flow Constraints](../../CVPR2026/video_understanding/laof_robust_latent_action_learning_with_optical_flow_constraints.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)
- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](lifelong_domain_adaptive_3d_human_pose_estimation.md)

<!-- RELATED:END -->
