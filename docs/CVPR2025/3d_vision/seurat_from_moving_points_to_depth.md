---
title: >-
  [Paper Note] Seurat: From Moving Points to Depth
description: >-
  [CVPR 2025][3D Vision][Point Trajectory Depth Estimation] This paper proposes Seurat, a monocular video depth estimation method based on 2D point trajectories. By analyzing the motion patterns of tracked points using spatial and temporal Transformers to infer depth changes over time, Seurat achieves zero-shot generalization to real-world scenes while being trained solely on synthetic data.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Trajectory Depth Estimation"
  - "Monocular Video Depth"
  - "Transformer"
  - "Zero-shot Generalization"
  - "Temporal Depth Ratio"
date: 2026-05-08
content_hash: 416d50984a5ad26a
---

# Seurat: From Moving Points to Depth

**Conference**: CVPR 2025  
**arXiv**: [2504.14687](https://arxiv.org/abs/2504.14687)  
**Code**: [https://seurat-cvpr.github.io](https://seurat-cvpr.github.io)  
**Area**: 3D Vision / Depth Estimation  
**Keywords**: Point Trajectory Depth Estimation, Monocular Video Depth, Transformer, Zero-shot Generalization, Temporal Depth Ratio

## TL;DR

This paper proposes Seurat, a monocular video depth estimation method based on 2D point trajectories. By analyzing the motion patterns of tracked points using spatial and temporal Transformers to infer depth changes over time, Seurat achieves zero-shot generalization to real-world scenes while being trained solely on synthetic data.

## Background & Motivation

**Background**: Monocular depth estimation (MDE) has made massive progress recently with the help of large-scale training. Methods such as MiDaS, DPT, and DepthPro perform exceptionally well on single-frame depth estimation, but they lack temporal consistency when applied to videos, leading to depth flickering. Recent video depth estimation methods (e.g., DepthCrafter, ChronoDepth) improve this issue through temporal modeling, but still rely on large-scale annotated datasets and strong feature backbones.

**Limitations of Prior Work**: (1) Single-frame MDE methods focus on representing relative spatial depth within a frame, ignoring the temporal depth change information. (2) Existing methods rely heavily on strong pre-trained feature backbones (such as DINOv2 and Stable Diffusion) and require large amounts of annotated data. (3) Depth estimation for dynamic objects remains challenging, as most methods assume static scenes.

**Key Challenge**: Monocular video contains rich temporal depth cues (projected points converge as objects move away), but existing methods fail to effectively exploit this purely motion-based geometric information.

**Goal**: Design a lightweight method that infers depth changes relying solely on 2D point trajectories, without requiring image feature backbones, multi-view or stereo setups, or ground-truth annotations.

**Key Insight**: The authors draw inspiration from structured light 3D scanning—where 3D structures are inferred through the deformation of projected patterns. Similarly, changes in the trajectory patterns of tracked points in a video encode depth information. For example, as an object moves away from the camera, its surface tracked points will converge and cluster in the image plane.

**Core Idea**: Transform the depth estimation problem from "guessing depth by looking at images" to "inferring depth ratios by analyzing point trajectory motion patterns," using a Transformer to learn the mapping from point density changes to depth changes.

## Method

### Overall Architecture

Given a monocular video, a off-the-shelf point tracking model (LocoTrack or CoTracker) is first used to extract 2D trajectories and occlusion states. The trajectories are then processed in two branches: the support trajectory branch (using uniformly grid-sampled points) captures global scene motion; and the query trajectory branch handles the user's query points of interest, obtaining global motion context from the support branch via cross-attention. Both branches independently output depth ratio predictions. Slide-window inference is used, and the final metric depth is reconstructed by integrating a single-frame metric depth model.

### Key Designs

1. **Decoupled Design of Support/Query Trajectories**:

    - Function: Decouples global scene motion modeling from query point depth prediction to prevent query point distribution bias from affecting depth estimation.
    - Mechanism: Support trajectories are obtained by uniform grid sampling on the image, processed by alternating temporal and spatial Transformer layers to capture global motion dynamics. Query trajectories are processed independently, obtaining global motion information from the support branch through a cross-attention mechanism. Both branches have independent regression heads outputting depth ratios. The trajectory of each query point is processed independently to prevent the query point distribution from influencing predictions.
    - Design Motivation: The distribution of query points defined by users or datasets is often non-uniform (biased towards salient objects). If directly involved in global motion modeling, it introduces distribution bias. The decoupled design ensures global motion understanding is unaffected by query distribution.

2. **Sliding Window Prediction and Log-Ratio Depth Loss**:

    - Function: Decomposes complex depth prediction for long videos into simpler depth change predictions within short temporal windows.
    - Mechanism: Divides the video into overlapping windows (length $W$, stride $S$), predicting the log depth ratio $\ell_{i,t}^w = \log(d_{i,t}^w / d_{i,0}^w)$ relative to the starting frame of the window. The key design is that query trajectories persist across windows, while support trajectories are reinitialized for each window (since support points are more likely to stay within the frame). During inference, the global depth ratio is obtained by aggregating the log depth ratios: $\hat{r}_{i,t} = \exp(\hat{\ell}_{i,t_k}^k + \sum_{w=0}^{k-1}\hat{\ell}_{i,S}^w)$. Training uses an L1 loss.
    - Design Motivation: Directly processing the entire video causes unstable depth predictions due to the complexity of long-range motion. Depth changes within short windows are more consistent and manageable, and the log-ratio formulation makes the model invariant to absolute depth scales.

3. **Fusion of Depth Ratios and Metric Depth**:

    - Function: Converts predicted temporal depth ratios into final metric depth values.
    - Mechanism: For each visible subsequence of each query trajectory, the scale factor $s_{i,t} = \frac{\text{median}_{t' \in \mathcal{S}_{i,t}}(d_{\text{MDE}}(p_{i,t'}))}{\text{median}_{t' \in \mathcal{S}_{i,t}}(\hat{r}_{i,t'})}$ is calculated, followed by $\hat{d}_{i,t} = s_{i,t} \cdot \hat{r}_{i,t}$. Robustness is improved by matching medians rather than matching frame-by-frame. This can be used in conjunction with various MDE models (such as ZoeDepth, DepthPro, etc.).
    - Design Motivation: The proposed method predicts the rate of depth change (temporal relative depth) and requires an "anchor" to obtain the metric scale. Leveraging the advantages of existing MDE models in spatial depth estimation results in a complementary design—our method provides temporal consistency, while the MDE model provides spatial scale.

### Loss & Training

An L1 loss is used to supervise the log depth ratio predictions within the window, and loss is applied to both query and support trajectories. Training is conducted solely on the synthetic dataset PointOdyssey without using any pre-trained feature backbones. An iterative prediction mechanism (inspired by works like RAFT) is adopted, feeding current predictions back into the model for multiple refinement stages.

## Key Experimental Results

### Main Results

Quantitative results on the TAPVid-3D benchmark (per-trajectory depth scaling, using CoTracker):

| Depth Estimation Method | Aria 3D-AJ↑ | DriveTrack 3D-AJ↑ | PStudio 3D-AJ↑ | Average 3D-AJ↑ | Average TC↓ |
|------------|------------|-------------------|----------------|------------|---------|
| ZoeDepth (per-frame) | 16.5 | 9.5 | 11.8 | 12.6 | 0.48 |
| DepthPro (per-frame) | 11.3 | 5.4 | 6.7 | 7.8 | 1.40 |
| DepthCrafter (video) | 15.1 | 8.4 | 11.1 | 11.5 | 0.35 |
| ChronoDepth (video) | 11.0 | 6.1 | 3.0 | 6.7 | 3.39 |
| **Seurat (Ours)** | **25.1** | **11.6** | **17.3** | **18.0** | **0.05** |

### Ablation Study

| Configuration | Explanation |
|------|------|
| Theoretical formula calculation only | Requires accurate rotation orientation information, infeasible in real environments (validated in Table 4) |
| No decoupling of support/query branches | Query point distribution bias affects global motion modeling, degrading performance |
| Full sequence processed together | Complex motion in long videos leads to unstable depth prediction |
| Sliding window prediction | Depth changes within short windows are more manageable, significantly improving performance |

### Key Findings

- Seurat significantly outperforms all baselines on the average 3D-AJ metric (18.0 vs the second best 12.6) and exhibits extremely strong temporal consistency (TC) (0.05 vs 0.35).
- Compared to the strong frame-by-frame MDE method (DepthPro), Seurat's 3D-AJ is 2.3 times higher, demonstrating the massive potential of depth estimation using pure motion cues.
- Training solely on synthetic data generalizes well to various real-world scenarios such as driving, first-person view, and deformable objects.
- Operating without reliance on pre-trained feature backbones is a significant advantage, proving that depth information embedded in point trajectory motion patterns is extremely rich.

## Highlights & Insights

- **Minimalist Depth Cues**: Depth changes can be inferred solely from the motion trajectories of 2D points without image textures, semantic features, or stereo disparity. This reveals that the geometric information embedded in motion is severely underestimated.
- **Ingenious Analogy to Structured Light**: Analogizing tracked points in video to pattern projections of structured light is both intuitive and theoretically grounded (projection area analysis in Section 3.2).
- **Zero-Shot Generalization Ability**: Trained purely on synthetic data but demonstrating strong performance in diverse real-world scenes, indicating that the depth cues in point trajectory motion patterns possess high domain invariance.
- **Complementary Design**: The fusion strategy of temporal depth ratio + spatial metric depth can be adapted to other temporal vision tasks.

## Limitations & Future Work

- Relies heavily on the quality of point tracking models; depth estimation fails if tracking fails.
- Local rigidity assumptions may hold poorly in scenarios with drastic deformations.
- Sliding window accumulation may introduce long-range drift errors.
- Currently only predicts depth on sparse points without extension to dense depth maps.
- Lack of comparison with 3D tracking methods like SpatialTracker on 2D tracking accuracy.

## Related Work & Insights

- **vs DepthCrafter/ChronoDepth**: These video depth methods rely on large-scale annotation and strong feature backbones, whereas the proposed method does not and significantly outperforms them in 3D-AJ.
- **vs SpatialTracker**: SpatialTracker utilizes MDE to assist 2D tracking (2D $\rightarrow$ 3D $\rightarrow$ refinement $\rightarrow$ 2D). This work reverses this paradigm, mining 3D geometric information directly from 2D trajectories.
- **vs TrackTo4D**: Similarly reconstructs 3D from trajectories, but introduces additional dynamic constraints and low-dimensional basis assumptions, which this work does not require.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The perspective of inferring depth from pure motion trajectories is very novel, and the structured light analogy is extremely inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively compared on TAPVid-3D, though details of the ablation studies are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivation is clear, and the transition from intuition to mathematical motivation is seamless.
- Value: ⭐⭐⭐⭐⭐ Opens up a new paradigm for depth estimation, and its zero-shot generalization capabilities make it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2026\] Moving Border Ownership for Event-based Motion Segmentation](../../CVPR2026/3d_vision/moving_border_ownership_for_event-based_motion_segmentation.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](sharpdepth_sharpening_metric_depth_predictions_using_diffusion_distillation.md)

</div>

<!-- RELATED:END -->
