---
title: >-
  [Paper Note] AllTracker: Efficient Dense Point Tracking at High Resolution
description: >-
  [ICCV 2025][Video Understanding][dense point tracking] AllTracker reformulates point tracking as a multi-frame long-range optical flow problem…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "dense point tracking"
  - "optical flow"
  - "long-range correspondence"
  - "high-resolution tracking"
  - "recurrent network"
date: 2026-05-08
content_hash: 0486218d7e8f7bfd
---

# AllTracker: Efficient Dense Point Tracking at High Resolution

**Conference**: ICCV 2025  
**arXiv**: [2506.07310](https://arxiv.org/abs/2506.07310)  
**Code**: [https://alltracker.github.io](https://alltracker.github.io)  
**Area**: Video Understanding / Dense Point Tracking  
**Keywords**: dense point tracking, optical flow, long-range correspondence, high-resolution tracking, recurrent network

## TL;DR
AllTracker reformulates point tracking as a multi-frame long-range optical flow problem, iteratively refining correspondence estimates on low-resolution grids via 2D convolutions and pixel-aligned temporal attention, followed by upsampling. With only 16M parameters, it achieves state-of-the-art accuracy and enables high-resolution (768×1024) dense tracking of all pixels at speeds approaching optical flow methods.

## Background & Motivation

Estimating long-range trajectories of arbitrary 2D points in image sequences is a fundamental problem in computer vision. Optical flow methods (e.g., RAFT, SEA-RAFT) estimate per-pixel motion between consecutive frames, but chaining instantaneous flow into long-range trajectories accumulates drift and requires careful handling of occlusions. Directly computing "long-range optical flow" between a reference frame and a distant frame avoids drift but becomes increasingly difficult as temporal gaps grow due to changes in viewpoint, illumination, and scene geometry.

Recent dedicated point trackers (PIPs, TAP, CoTracker, etc.) leverage learned multi-frame temporal priors to reduce drift and track through occlusions, achieving notable progress. However, these methods sacrifice spatial awareness for temporal awareness and can only track sparse point sets. Recent attempts at "dense" point tracking (DTF, DELTA) fall short of the latest sparse trackers in accuracy and struggle with high-resolution inputs.

The core insight of this paper is that learnable multi-frame temporal priors and high-resolution spatial awareness can be jointly achieved — by simply reformulating point tracking as a multi-frame long-range optical flow problem. This enables dense correspondence fields at high resolution, as in optical flow methods, while supporting occlusion-aware tracking, as in point trackers.

## Method

### Overall Architecture
Given a video and a query frame index, AllTracker outputs a $T \times H \times W \times 4$ tensor: per-pixel flow offset (2 channels), visibility, and confidence at every frame. A sliding window strategy is adopted (window length $S=16$, stride $S/2$). Within each window: (1) a CNN encoder extracts low-resolution features → (2) a multi-scale 4D correlation volume is constructed → (3) tracking estimates are initialized → (4) a recurrent refinement module iteratively updates estimates → (5) results are upsampled to full resolution.

### Key Designs

1. **ConvNeXt-Tiny Encoder**:

    - Function: Compresses video frames into low-resolution feature maps ($H/8 \times W/8$).
    - Mechanism: Uses the first 3 blocks of a pretrained ConvNeXt-Tiny (12.72M parameters), modifying the stride-2 in the third block to stride-1 (by bicubic interpolation of 2×2 kernels to 3×3), and outputs 256-dimensional features.
    - Design Motivation: Low-resolution features enable fast 2D convolutional message passing, with spatial precision recovered via upsampling. Using stride 8 instead of CoTracker3's stride 4 significantly reduces memory consumption.

2. **Multi-Scale 4D Correlation Volume**:

    - Function: Captures appearance-based tracking cues.
    - Mechanism: Feature maps from each frame are average-pooled into a feature pyramid (5 scales: {1,2,4,8,16}), followed by dot-product cross-correlation between the query frame's feature map and each timestep's pyramid. Each tracked pixel obtains a heatmap at each scale and timestep. During iterative refinement, a $(2R+1)^2$ patch ($R=4$) is extracted around the current estimate, flattened into a vector $\mathbf{q}$ of dimension $L \cdot (2R+1)^2 = 5 \times 81 = 405$.
    - Design Motivation: Multi-scale correlation provides a strong inductive bias for feature matching and accelerates training.

3. **Alternating Spatial-Temporal Recurrent Refinement Module (Core)**:

    - Function: Iteratively updates motion, visibility, and confidence estimates for all pixels across all timesteps.
    - Mechanism: Per-pixel, per-timestep inputs include feature vector $\mathbf{f}$ (256D), visibility/confidence $\mathbf{v,c}$ (2D), motion estimate $\mathbf{m}$ (2D), and correlation vector $\mathbf{q}$ (405D), totaling 665 channels. The module alternates between:
        - **Spatial block**: A 2D ConvNeXt block propagating information across spatial dimensions.
        - **Temporal block**: Pixel-aligned Transformer attention applied only along the temporal axis ($S=16$ frames), executed in parallel for all pixels.
      Since all tensors are aligned to the query frame, pixel-aligned attention naturally corresponds to attention between corresponding pixels.
    - Design Motivation: Spatial message passing via simple 2D convolutions is efficient and effective, avoiding global attention. Temporal attention learns motion priors and handles occlusions. Iterative refinement progressively improves estimates via residual updates $\mathbf{x}_{\text{new}} = \mathbf{x}_{\text{old}} + \delta\mathbf{x}$.

4. **Pixel-Shuffle Upsampling**:

    - Function: Restores full resolution from 1/8 resolution.
    - Mechanism: The recurrent module additionally decodes pixel-shuffle weights, which are applied to the visibility, confidence, and motion maps.
    - Design Motivation: This technique, known from RAFT but underexplored in the point tracking literature, is key to enabling high-resolution dense tracking.

### Loss & Training
- **Tracking loss**: $L_{\text{track}} = \alpha \sum_k^K \gamma^{K-k} (\mathbb{1}_{\text{occ}}/5 + \mathbb{1}_{\text{vis}}) \|P_k - \hat{P}\|_1$, with $\gamma=0.8$ upweighting later refinement steps and $\alpha=0.05$.
- **Visibility loss**: Binary cross-entropy matching ground-truth binary labels.
- **Confidence loss**: Binary cross-entropy with target indicating whether the estimated position is within 12 pixels of ground truth.
- Two-stage training: 200K steps on Kubric (lr=5e-4), followed by 400K steps on a mixed dataset (lr=1e-5).
- The mixed dataset jointly samples optical flow data (FlyingChairs, FlyingThings3D, AutoFlow, Spring, VIPER, etc.) and point tracking data (Kubric, PointOdyssey, etc.) with uniform sampling.

## Key Experimental Results

### Main Results: Average $\delta_{avg}$ across 9 benchmarks (384×512 resolution)

| Method | Params | BADJA | Davis | Kinetics | RGB-Stack | RoboTAP | Avg |
|--------|--------|-------|-------|----------|-----------|---------|-----|
| RAFT | 5.3M | 23.7 | 48.5 | 64.3 | 82.8 | 72.2 | 48.3 |
| SEA-RAFT | 19.7M | 23.9 | 48.7 | 64.3 | 85.7 | 67.6 | 48.7 |
| CoTracker3-Kub | 25.4M | 47.5 | 77.4 | 70.6 | 83.4 | 77.2 | 64.5 |
| CoTracker3 | 25.4M | 48.3 | 77.1 | 71.8 | 84.2 | 81.6 | 65.0 |
| **AllTracker** | **16.5M** | **51.5** | 76.3 | **72.3** | **90.0** | **83.4** | **66.1** |

### High-Resolution Results (768×1024)

| Method | Params | BADJA | Davis | RGB-Stack | Avg |
|--------|--------|-------|-------|-----------|-----|
| CoTracker3 | 25.4M | 49.8 | 79.6 | 77.9 | 66.9 |
| AllTracker-Tiny | 6.3M | 51.6 | 79.1 | 87.4 | 67.5 |
| **AllTracker** | **16.5M** | **53.6** | **80.6** | **90.6** | **69.5** |

### Key Findings
- AllTracker surpasses CoTracker3 by 1.1 points on the 9-benchmark average with 16.5M parameters (65% of CoTracker3's parameter count).
- The advantage grows at higher resolution: at 768×1024, AllTracker achieves an average of 69.5 vs. CoTracker3's 66.9 (+2.6); AllTracker's performance scales steadily with resolution while CoTracker3 saturates beyond 448×768.
- AllTracker-Tiny (only 6.3M parameters) already surpasses the full CoTracker3 at 768×1024 (67.5 vs. 66.9).
- The largest gains appear on RGB-Stacking (90.6 vs. 77.9), demonstrating effective utilization of broad spatial context.
- Joint training on optical flow and point tracking data is critical: training on Kubric alone yields an average of 64.8, which improves to 66.1 (+1.3) with optical flow data included.
- In real-time mode at 512×512, the model achieves 57.9 FPS with a $\delta_{avg}$ of 62.6.
- Average Jaccard: AllTracker 68.9 vs. CoTracker3 63.1; occlusion accuracy: 91.5 vs. 89.3.

## Highlights & Insights
- The core contribution is the problem reformulation as multi-frame long-range optical flow rather than sparse point tracking, which naturally enables (1) dense tracking as a direct output and (2) joint training on both optical flow and point tracking data.
- The strategy of low-resolution refinement with pixel-shuffle upsampling is key to practical high-resolution dense tracking — a technique known from RAFT but underappreciated in the point tracking literature.
- Pixel-aligned temporal attention (cost proportional only to window length $S=16$) is far more efficient than global attention.
- The effectiveness of mixed optical flow training data validates the value of cross-task data utilization.

## Limitations & Future Work
- Additional training data from CroHD (surveillance) and DriveTrack (driving) datasets did not reliably improve performance, suggesting room for improvement in data balancing or model capacity.
- The sliding window strategy incurs approximately a 3.5-point accuracy loss in real-time mode (66.1→62.6).
- The method has not been combined with CoTracker3's pseudo-label bootstrapping scheme, which may further improve performance.

## Related Work & Insights
- **vs. CoTracker3**: CoTracker3 is the previous state-of-the-art sparse point tracker that propagates spatial information via virtual points. AllTracker replaces this with 2D convolutions — simpler and more efficient — while natively supporting full-pixel dense tracking.
- **vs. DELTA**: A concurrent dense tracking work that approximates global spatial attention with sparse anchor tokens, incurring high memory consumption and lower accuracy compared to AllTracker.
- **vs. SEA-RAFT**: AllTracker inherits SEA-RAFT's low-resolution estimation and upsampling strategy, extending it from 2 frames to a 16-frame window and adding temporal attention for occlusion-aware tracking.

## Rating
- Novelty: ⭐⭐⭐⭐ The problem reformulation is elegant and concise; known techniques are combined to yield new capabilities.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 9 benchmarks across 3 resolutions with speed comparisons and AJ/occlusion metrics — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Design choices are clearly articulated and the relationship to prior work is well organized.
- Value: ⭐⭐⭐⭐⭐ Brings dense long-range point tracking to practical applicability and unifies the optical flow and point tracking paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/video_understanding/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)

</div>

<!-- RELATED:END -->
