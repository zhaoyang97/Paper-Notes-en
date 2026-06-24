---
title: >-
  [Paper Note] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation
description: >-
  [ICCV 2025][Autonomous Driving][4D scene generation] This paper proposes DiST-4D, the first feed-forward 4D driving scene generation framework. By disentangling temporal prediction (DiST-T) and spatial novel view synthesis (DiST-S) into two separate diffusion processes, with metric depth serving as a geometric bridge, the method achieves state-of-the-art temporal video generation (FVD 22.67) and spatial NVS (FID 10.12) on nuScenes simultaneously…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "4D scene generation"
  - "spatiotemporal disentangled diffusion"
  - "metric depth"
  - "novel view synthesis"
date: 2026-05-08
content_hash: c47556fbae42d565
---

# DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation

**Conference**: ICCV 2025
**arXiv**: [2503.15208](https://arxiv.org/abs/2503.15208)  
**Code**: [https://royalmelon0505.github.io/DiST-4D](https://royalmelon0505.github.io/DiST-4D)  
**Area**: Autonomous Driving
**Keywords**: 4D scene generation, spatiotemporal disentangled diffusion, metric depth, novel view synthesis, autonomous driving

## TL;DR
This paper proposes DiST-4D, the first feed-forward 4D driving scene generation framework. By disentangling temporal prediction (DiST-T) and spatial novel view synthesis (DiST-S) into two separate diffusion processes, with metric depth serving as a geometric bridge, the method achieves state-of-the-art temporal video generation (FVD 22.67) and spatial NVS (FID 10.12) on nuScenes simultaneously, without any per-scene optimization.

## Background & Motivation
Generative models for autonomous driving must produce large-scale synthetic data for training and evaluating perception and planning systems. The ideal goal is to generate dynamic 4D scenes renderable at **arbitrary times and locations**. However, existing approaches suffer from fundamental limitations:

**Temporal generation methods** (MagicDrive, Vista, etc.): can predict future video but are tied to predefined trajectories and cannot synthesize novel views.

**Implicit reconstruction methods** (3DGS, NeRF, PVG, etc.): excel at NVS but require per-scene optimization at high computational cost.

**Feed-forward NVS methods** (STORM, FreeVS, etc.): enable fast NVS but lack temporal extrapolation capability.

**Hybrid methods** (DreamDrive, MagicDrive3D): attempt to bridge both worlds but still inherit the limitations of per-scene optimization.

The core challenge is to find a 4D representation that simultaneously encodes the geometric information needed for NVS and can be learned and predicted by a generative model. This paper selects **per-frame metric depth** as the central representation, because it: (1) satisfies both requirements above; (2) demonstrates strong generalization for NVS (as validated by ViewCrafter); and (3) has intrinsic practical value in autonomous driving (annotation, perception training).

## Method

### Overall Architecture
DiST-4D consists of two disentangled diffusion branches:
- **DiST-T (Temporal Generation)**: Given one frame of historical multi-camera observations and control signals (BEV map, trajectory, 3D bounding boxes), it generates future multi-camera RGB-D video sequences.
- **DiST-S (Spatial NVS)**: Given the generated RGB-D frames, it projects point clouds to a novel viewpoint to form a sparse conditioning map, which a diffusion model then completes into a dense RGB-D output.

### Key Designs

1. **Metric Depth Curation Pipeline**:

    - Function: Provides high-quality dense metric depth pseudo-GT for training.
    - Mechanism:
      1. Multi-frame LiDAR point cloud aggregation (dynamic objects removed via 3D bounding boxes).
      2. MVS network (multi-view stereo) reconstructs the static scene point cloud.
      3. 2D semantic segmentation filters sky and dynamic objects.
      4. LiDAR depth and MVS depth are fused as prompts.
      5. A generative depth completion network refines the result into a dense metric depth map.
    - Design Motivation: LiDAR alone is sparse and incomplete (distant buildings, elevated structures), while standalone depth estimation networks lack sufficient accuracy for direct use as GT.

2. **DiST-T (Temporal RGB-D Generation)**:

    - Function: Predicts future multi-view RGB-D video from historical frames and control signals.
    - Mechanism:
        - A pretrained 3D VAE encodes RGB and depth videos separately into a compressed latent space.
        - An STDiT (spatiotemporal DiT)-based diffusion model includes multi-view blocks (cross-camera interaction) and spatiotemporal blocks.
        - Control signals (BEV map $M$, 3D boxes $B$, trajectory $A$, camera pose $P$) are injected via a ControlNet-style branch.
        - Training formulation: $\{Z_t^I, Z_t^D\} = \mathcal{G}_\theta(\{S_t, I_{ref}\})$
    - Progressive training: low-resolution RGB → low-resolution RGB-D → high-resolution (424×800) RGB-D, with a mixed frame-length strategy.
    - Rectified flow with v-prediction loss is employed.

3. **DiST-S (Spatial NVS) + Self-Supervised Cycle Consistency**:

    - Function: Projects RGB-D from existing viewpoints to arbitrary novel views.
    - Mechanism:
        - Multi-camera RGB-D is converted to a point cloud → projected to the target viewpoint → yielding a sparse conditioning map.
        - A Stable Video Diffusion-based UNet completes the sparse condition into a dense RGB-D output.
        - Input channels are expanded from 8 to 16 (RGB + Depth dual-modal), and output channels from 4 to 8.
    - **Self-Supervised Cycle Consistency (SCC)**:
        - Stage 1: Train on original trajectory data (projection from adjacent ±2 frames).
        - Stage 2: Randomly generate offset trajectories (lateral ±3 m), synthesize RGB-D at the offset viewpoint using the trained DiST-S, then project back to the original viewpoint as self-supervised training pairs.
        - Formulation: $\{Z_{tgt}^I, Z_{tgt}^D\} = \mathcal{F}_\theta(\{Z_{cond}^I, Z_{cond}^D\})$
    - Design Motivation: Real driving data has limited trajectory diversity; SCC compensates for out-of-distribution viewpoints by constructing virtual trajectories with cycle constraints.

### Loss & Training
Both DiST-T and DiST-S use a v-prediction-based MSE loss. DiST-T adopts a three-stage progressive training strategy. DiST-S follows a two-stage training procedure (original trajectories → fine-tuning with SCC). The lateral offset in SCC is $\tau \in [-3\text{m}, +3\text{m}]$.

## Key Experimental Results

### Main Results — Temporal Generation

| Method | Multi-view | Video | Depth | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| MagicDrive | ✓ | ✓ | ✗ | 16.20 | 217.94 |
| MagicDriveDiT | ✓ | ✓ | ✗ | 20.91 | 94.84 |
| Drive-WM | ✓ | ✓ | ✗ | 15.80 | 122.70 |
| Vista* | ✓ | ✓ | ✗ | 13.97 | 112.65 |
| UniScene | ✓ | ✓ | ✗ | 6.45 | 71.94 |
| **DiST-T (Ours)** | ✓ | ✓ | **✓** | **6.83** | **22.67** |

FVD substantially outperforms all methods (22.67 vs. the next best 71.94). DiST-T generates depth sequences alongside video.

### Main Results — Spatial NVS

| Method | FID(±1m)↓ | FVD(±1m)↓ | FID(±2m)↓ | FID(±4m)↓ |
|---|---|---|---|---|
| StreetGaussian | 32.12 | 153.45 | 43.24 | 67.44 |
| OmniRe | 31.48 | 152.01 | 43.31 | 67.36 |
| FreeVS* | 51.26 | 431.99 | 62.04 | 77.14 |
| DiST-4D (Ours) | 20.64 | 130.98 | 25.08 | 33.56 |
| DiST-4D (+SCC) | 16.40 | 112.86 | 19.50 | 25.16 |
| **DiST-S (+SCC)** | **10.12** | **45.14** | **12.97** | **17.57** |

DiST-S+SCC achieves an FID 68% lower than the best reconstruction method OmniRe, without any per-scene optimization.

### Ablation Study

| Configuration | FID-1m↓ | FID-2m↓ | Notes |
|------|---------|---------|------|
| (a) RGB only (no depth) | 26.33 | 33.54 | Depth is critical for NVS |
| (b) + Depth (no valid mask) | 30.31 | 32.80 | Mask is key for sparse projection |
| (c) + Depth + Mask (no data aug.) | 26.19 | 29.69 | Augmentation improves by ~2.5% |
| **DiST-S (full)** | **25.51** | **27.75** | All components contribute |
| + SCC | ~20% reduction | ~20% reduction | SCC substantially improves quality |

The valid mask contributes most (15.8% FID reduction); SCC further reduces FID by ~20%.

### Key Findings
- Reference depth input has a slight adverse effect on RGB prediction (DiST-T Ours-D has slightly higher FID than Ours), yet depth generation quality surpasses dedicated depth estimation methods.
- Downstream task evaluation (UniAD detection/segmentation/planning): performance on generated video closely approaches that on original GT data.
- Generated depth surpasses SurroundDepth and M2Depth when evaluated against multi-frame LiDAR GT.
- The performance gap between reconstruction methods and DiST widens as viewpoint offset increases, demonstrating DiST's superior generalization.

## Highlights & Insights
- The core idea of "metric depth as a spatiotemporal bridge" is both clear and effective: depth simultaneously serves as a geometric constraint for temporal generation and a projection basis for spatial NVS.
- The Self-Supervised Cycle Consistency (SCC) strategy elegantly addresses the limited trajectory diversity in training data.
- The metric depth curation pipeline has independent value; the combination of LiDAR + MVS + depth completion is broadly reusable.
- This work achieves the first feed-forward joint spatiotemporal 4D scene generation, establishing a new paradigm for autonomous driving simulation.

## Limitations & Future Work
- Pseudo-GT depth quality still has room for improvement, particularly for distant buildings.
- The inference overhead of two diffusion models (DiST-T + DiST-S) is substantial.
- Validation is currently limited to nuScenes (700 training clips).
- Incorporating reference depth into DiST-T slightly degrades RGB quality; joint modeling of RGB and depth warrants further investigation.
- Future work may extend the framework to embodied intelligence and robotics.

## Related Work & Insights
- The evolution of the MagicDrive series (MagicDrive → MagicDriveDiT → MagicDrive3D) clearly illustrates the progression of the field.
- ViewCrafter and FreeVS validate the effectiveness of depth projection as an NVS condition; DiST-4D systematizes this approach.
- The SCC strategy draws inspiration from the success of cycle consistency in image-to-image translation, and is well-adapted to driving NVS.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First feed-forward 4D driving scene generation; the spatiotemporal disentanglement + metric depth bridge framework is excellently designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers temporal generation, spatial NVS, downstream tasks, depth quality, and ablations, though validation is limited to nuScenes.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with thorough method descriptions and intuitive comparison tables.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for driving scene simulation with high practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving](../../ICLR2026/autonomous_driving/worldsplat_gaussian-centric_feed-forward_4d_scene_generation_for_autonomous_driv.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[ICCV 2025\] 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads](4dsegstreamer_streaming_4d_panoptic_segmentation_via_dual_threads.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](../../CVPR2025/autonomous_driving/prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)

</div>

<!-- RELATED:END -->
