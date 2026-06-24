---
title: >-
  [Paper Note] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction
description: >-
  [ICCV 2025][3D Vision][Dynamic 3D Reconstruction] This paper proposes Dynamic Point Maps (DPM), extending DUSt3R's viewpoint-invariant point maps into a spatiotemporal-invariant representation that jointly controls viewpoint and time. By predicting only four sets of point maps in a feed-forward manner, DPM simultaneously addresses multiple 4D tasks including depth estimation, scene flow, motion segmentation, and 3D object tracking.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Dynamic 3D Reconstruction"
  - "Point Map Representation"
  - "Scene Flow"
  - "Motion Segmentation"
  - "DUSt3R"
date: 2026-05-08
content_hash: f5488521a0dac3f3
---

# Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2503.16318](https://arxiv.org/abs/2503.16318)  
**Code**: [Project Page](https://www.robots.ox.ac.uk/~vgg/research/dynamic-point-maps/)  
**Area**: 3D Vision
**Keywords**: Dynamic 3D Reconstruction, Point Map Representation, Scene Flow, Motion Segmentation, DUSt3R

## TL;DR

This paper proposes Dynamic Point Maps (DPM), extending DUSt3R's viewpoint-invariant point maps into a spatiotemporal-invariant representation that jointly controls viewpoint and time. By predicting only four sets of point maps in a feed-forward manner, DPM simultaneously addresses multiple 4D tasks including depth estimation, scene flow, motion segmentation, and 3D object tracking.

## Background & Motivation

The key breakthrough of DUSt3R lies in introducing the concept of viewpoint-invariant point maps: given two images, each pixel is mapped to a 3D point in a unified reference frame. This elegant representation reduces various tasks—camera intrinsic/extrinsic estimation, 3D reconstruction, 2D matching—to point map prediction.

**However, DUSt3R cannot handle dynamic scenes.** When moving objects are present, fixing the viewpoint reference frame is insufficient, as the 3D position of the same physical point differs across time, violating viewpoint invariance: $P_1(\pi_1)(\boldsymbol{u}_1) \neq P_2(\pi_1)(\boldsymbol{u}_2)$.

**Limitations of MonST3R**: MonST3R directly applies DUSt3R to dynamic scenes, but due to the lack of temporal invariance, it cannot directly predict corresponding 3D points. It must rely on an optical flow network to establish temporal correspondences, increasing system complexity while being restricted to visible pixels and handling occlusion and disocclusion poorly.

The authors' core insight is that **invariance in dynamic scenes requires simultaneously fixing both viewpoint and time.** Predicting two sets of point maps per image—corresponding to 3D positions at two timestamps—restores spatiotemporal invariance while preserving motion information. This constitutes the minimal design sufficient to solve all 4D tasks.

Specifically:
- $P_1(t_1, \pi_1)$, $P_1(t_2, \pi_1)$: 3D positions of pixels in image 1 at times $t_1$ and $t_2$
- $P_2(t_1, \pi_1)$, $P_2(t_2, \pi_1)$: 3D positions of pixels in image 2 at times $t_1$ and $t_2$
- Point maps at the same timestamp restore invariance: $P_1(t_1, \pi_1)(\boldsymbol{u}_1) = P_2(t_1, \pi_1)(\boldsymbol{u}_2)$
- The difference across timestamps directly yields scene flow: $P_1(t_2, \pi_1) - P_1(t_1, \pi_1)$

## Method

### Overall Architecture

Building upon DUSt3R's ViT encoder-decoder backbone, one additional prediction head is added per image, yielding four heads in total that respectively predict four sets of point maps $P_i(t_j, \pi_1)$, $i,j \in \{1,2\}$. Each point map consists of a 3-channel coordinate map and a 1-channel confidence map. All point maps are expressed in the reference frame $\pi_1$ of the first image.

### Key Designs

1. **Dynamic Point Maps Representation**:

    - **Function**: Defines a minimal and complete point map representation for dynamic scenes.
    - **Mechanism**: Each image is mapped to two sets of point maps (one per timestamp), yielding four sets in total. Temporally invariant point map pairs directly establish cross-view correspondences (matching $P_1(t_1, \pi_1)$ with $P_2(t_1, \pi_1)$), while temporally varying pairs directly yield scene flow ($P_i(t_2, \pi_1) - P_i(t_1, \pi_1)$). Consistency of the two temporal point maps for static regions naturally produces motion segmentation.
    - **Design Motivation**: This is the natural and minimal extension of DUSt3R to 4D scenes. MonST3R effectively predicts only 2 of the 4 sets ($P_1(t_1, \pi_1)$ and $P_2(t_2, \pi_1)$); the missing cross-temporal point maps prevent direct derivation of scene flow and motion correspondences. DPM completes this design space.

2. **Network Architecture Extension**:

    - **Function**: Augments the DUSt3R backbone with two additional prediction heads.
    - **Mechanism**: $\{P_i(t_j, \pi_1)\}_{i,j \in \{1,2\}} = \Phi(I_1, I_2)$. All four heads share the same Transformer encoder-decoder backbone; each head predicts $(P_i(t_j, \pi_1), C_i(t_j, \pi_1))$ (point map + confidence). The two new heads are initialized with the weights of DUSt3R's original heads, approximating static reconstruction at the start of training.
    - **Design Motivation**: Minimizes architectural modifications while fully leveraging DUSt3R's pretrained knowledge. The additional parameter count is negligible since only two heads are added while the backbone is shared.

3. **Mixed-Data Training**:

    - **Function**: Trains the DPM predictor on a mixture of synthetic and real data.
    - **Mechanism**: The Kubric pipeline is used to generate the synthetic dataset MOVi-G (with complex camera trajectories and dynamic objects), providing complete ground truth for all four point map sets. Waymo real-world data is incorporated, with LiDAR used to generate dynamic point map ground truth. For datasets lacking dynamic ground truth, cross-temporal point map supervision is omitted or scenes are treated as static. A total of seven datasets are mixed for training.
    - **Design Motivation**: Complete dynamic ground truth is only available from synthetic data and LiDAR, but mixing real video data (even with static supervision only) improves generalization.

### Loss & Training

A confidence-calibrated regression loss is employed:
$$L_{\text{conf}}(\hat{P}, P) = \frac{1}{HW} \sum_{i=1}^{HW} C_i L_{\text{reg}}(\hat{P}, P, i) - \alpha \log C_i$$

where $L_{\text{reg}}$ is a scale-normalized per-pixel regression loss that allows predictions up to an arbitrary scale factor. All four point map sets are stacked and optimized jointly. Training resolutions are $(512, 288)$ and $(512, 336)$.

## Key Experimental Results

### Main Results

**Depth Estimation (2-View, Abs Rel↓)**:

| Dataset | DPM | MonST3R | Gain |
|--------|-----|---------|------|
| Sintel | **0.321** | 0.347 | 7.5% |
| Point Odyssey | **0.059** | 0.065 | 9.2% |
| Kubric | **0.078** | 0.166 | 53% |
| KITTI (crop) | **0.052** | 0.069 | 24.6% |

**Dynamic Reconstruction (Relative Point Cloud Error $L_{\text{rel}}$↓)**:

| Dataset | Point Map | DPM | MonST3R |
|--------|------|-----|---------|
| Kubric-G | $P_1(t_1)$ | **0.057** | 0.163 |
| Kubric-G | $P_2(t_1)$ (cross-temporal) | **0.071** | 0.265 |
| Kubric-G | $P_1(t_2)$ (cross-temporal) | **0.079** | 0.346 |
| Waymo | $P_1(t_1)$ | **0.068** | 0.197 |

**Scene Flow 3D EPE↓**:

| Dataset | DPM | MonST3R | RAFT-3D (requires depth GT) |
|--------|-----|---------|-------------------|
| Kubric-G Forward | **0.104** | 0.334 | 4.067 |
| Waymo Forward | **0.051** | 0.161 | 0.150 |
| Waymo Backward | **0.053** | 0.135 | 0.145 |

### Ablation Study

| Configuration | Kubric $L_{\text{rel}}$ $P_1(t_1)$↓ | Kubric $L_{\text{rel}}$ $P_1(t_2)$↓ | Notes |
|------|-------------------------------------|-------------------------------------|------|
| MonST3R (2 point maps only) | 0.163 | 0.346 | Cross-temporal prediction degrades severely |
| DPM (4 point maps) | **0.057** | **0.079** | Cross-temporal prediction remains stable |
| Object Tracking RPE rot↓ | DPM: 33.7° | MonST3R: 56.1° | Rotation error reduced by 40% |

### Key Findings

1. DPM substantially outperforms MonST3R on cross-temporal prediction—on Kubric-G, $P_2(t_1)$ error is reduced by 73% (0.265→0.071), demonstrating the necessity of explicitly modeling the temporal dimension.
2. DPM using only RGB input surpasses RAFT-3D, which requires depth ground truth, on Waymo scene flow (0.051 vs. 0.150).
3. MonST3R's scene flow EPE on Kubric-G reaches 4.067 (via RAFT-3D), indicating that 2D optical flow warping fails under complex camera motion.
4. DPM requires no additional optical flow model, resulting in a simpler and more efficient pipeline than MonST3R.
5. A 40% reduction in rotation error for object tracking demonstrates the value of explicit temporally invariant point maps for rigid body motion estimation.

## Highlights & Insights

1. **Conceptual Contribution**: DPM is a general representational concept rather than merely a specific method. It explicitly identifies that invariance in dynamic scenes requires joint control of both viewpoint and time, and this design space analysis provides a theoretical foundation for subsequent work.
2. **Minimal Design Principle**: Four sets of point maps constitute the minimal sufficient set for solving all 4D tasks—multiple tasks (depth, scene flow, matching, segmentation, tracking) are all reduced to simple point comparisons or difference operations.
3. **Minimal Architectural Modification**: Only two heads are added, fully reusing DUSt3R pretraining, demonstrating that good representational design matters more than complex architectural improvements.
4. The dependency on optical flow networks is eliminated, unifying the 4D reconstruction pipeline.

## Limitations & Future Work

1. Reliance on synthetic training data: the object motion patterns in MOVi-G are limited, which may affect generalization to complex real-world scenes.
2. Pairwise processing: DPM processes only two frames at a time; long video sequences require bundle adjustment.
3. Rotation error remains relatively high for large object motions (33.7°), indicating that cross-temporal 3D understanding remains challenging.
4. Only $T=2$ timestamps are currently supported; extension to multiple timestamps would yield a more general framework.
5. Handling of occluded and disoccluded regions still has room for improvement.

## Related Work & Insights

- **DUSt3R / MASt3R**: The direct foundation of DPM, demonstrating the strong generality of point map representations → DPM extends them to 4D.
- **MonST3R**: Applies DUSt3R to dynamic scenes but incompletely → DPM completes the representation space.
- **CUT3R / Stereo4D**: Concurrent works with different emphases, but neither explores the full invariance design space.
- **Shape of Motion**: Reconstructs dynamic scenes by fitting 3D Gaussian trajectories, but requires expensive test-time optimization → DPM is a purely feed-forward solution.
- Insight: Good representational design—rather than larger models—is the key leverage point in 3D/4D vision.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ DPM is conceptually elegant; the minimal design approach that addresses multiple tasks simultaneously is highly inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple tasks including depth, scene flow, and object tracking, though evaluation on real-world data could be more comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The design is naturally derived from invariance analysis, with clear and rigorous logic.
- **Value**: ⭐⭐⭐⭐⭐ Provides a unified representational foundation for dynamic 3D vision with potentially broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] V-DPM: 4D Video Reconstruction with Dynamic Point Maps](../../CVPR2026/3d_vision/v-dpm_4d_video_reconstruction_with_dynamic_point_maps.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](../../NeurIPS2025/3d_vision/d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)

</div>

<!-- RELATED:END -->
