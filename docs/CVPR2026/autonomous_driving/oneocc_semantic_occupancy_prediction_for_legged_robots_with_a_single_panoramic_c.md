---
title: >-
  [Paper Note] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera
description: >-
  [CVPR 2026][Autonomous Driving][Semantic Scene Completion] This paper proposes OneOcc, a vision-only panoramic semantic occupancy prediction framework for legged/humanoid robots. Through dual-projection fusion, dual-grid voxelization, gait displacement compensation, and a hierarchical mixture-of-experts decoder, OneOcc achieves 360° semantic scene completion using only a single panoramic camera, surpassing LiDAR baselines on both real quadruped and simulated humanoid datasets.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Semantic Scene Completion
  - Panoramic Camera
  - Legged Robots
  - Voxel Occupancy Prediction
  - Gait Compensation
date: 2026-05-08
content_hash: 367460aa065f69a5
---

# OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera

**Conference**: CVPR 2026
**arXiv**: [2511.03571](https://arxiv.org/abs/2511.03571)
**Code**: Available
**Area**: Autonomous Driving
**Keywords**: Semantic Scene Completion, Panoramic Camera, Legged Robots, Voxel Occupancy Prediction, Gait Compensation

## TL;DR

This paper proposes OneOcc, a vision-only panoramic semantic occupancy prediction framework for legged/humanoid robots. Through dual-projection fusion, dual-grid voxelization, gait displacement compensation, and a hierarchical mixture-of-experts decoder, OneOcc achieves 360° semantic scene completion using only a single panoramic camera, surpassing LiDAR baselines on both real quadruped and simulated humanoid datasets.

## Background & Motivation

### 1. State of the Field

Semantic Scene Completion (SSC) aims to predict complete 3D voxel semantics from partial observations and has become a core task in autonomous driving. From LiDAR-based methods (SSCNet → LMSCNet → SCPNet) to vision-based methods (MonoScene → VoxFormer → OccFormer), SSC has made substantial progress on **wheeled platforms**. However, all mainstream SSC methods assume forward-facing pinhole/fisheye sensors paired with stable wheeled-chassis motion, making them difficult to transfer to legged robot scenarios.

### 2. Limitations of Prior Work

Legged/humanoid robots face three major challenges: **(1)** Gait jitter — impulsive foot strikes and minor roll/pitch perturbations during agile locomotion corrupt the feature-to-voxel mapping and temporal consistency; **(2)** 360° omnidirectional perception — situational awareness across all directions is required on complex terrain, not merely a forward field of view; **(3)** Payload and power constraints — legged platforms operate under much tighter sensor and computational budgets than autonomous vehicles.

### 3. Root Cause

Existing SSC systems are designed for wheeled platforms and rely on forward-facing pinhole sensors and stable motion assumptions. Legged platforms instead require a **single-sensor panoramic** solution that must simultaneously handle **annular distortion and seam artifacts** inherent to panoramic imagery, as well as **feature-to-voxel mapping phase errors** induced by gait motion.

### 4. Paper Goals

Design a lightweight, vision-only, gait-jitter-robust 360° semantic occupancy prediction framework for legged/humanoid robots, and establish an evaluation benchmark for this setting.

### 5. Starting Point

The work exploits the **dual-projection properties** of panoramic cameras (raw annular image vs. equirectangular unrolled image) and **dual-coordinate voxelization** (Cartesian vs. cylindrical), leveraging their complementarity to address panoramic distortion and near/far-field imbalance. A learnable displacement compensation module eliminates gait-induced errors prior to feature lifting.

### 6. Core Idea

Four plug-and-play modules operate synergistically: DP-ER (dual-projection encoder fusion) preserves annular continuity and grid alignment; BGV (dual-grid voxelization) balances near-field precision and far-field azimuthal continuity; GDC (gait displacement compensation) corrects phase errors before feature lifting; AMoE-3D (hierarchical attentive mixture-of-experts) performs scale-adaptive anisotropic 3D fusion.

## Method

### Overall Architecture

The OneOcc pipeline proceeds as follows:

1. **Calibrated Unrolling**: The raw panoramic annular image is unrolled into an equirectangular (ER) image via the Taylor polynomial camera model.
2. **Dual-Projection Encoder DP-ER**: Two parallel 2D encoders process the raw annular image and the ER image respectively, producing multi-scale features at strides {1/4, 1/8, 1/16}.
3. **Gait Displacement Compensation GDC** (optional): A 2D displacement $\Delta_s = (dx, dy)$ is regressed at each scale and projection branch to correct sampling coordinates before feature lifting.
4. **Dual-Grid Voxelization BGV**: Features are bilinearly sampled and lifted into 3D in both Cartesian and cylindrical voxel spaces; a precomputed Po2Ca cross-grid index injects polar-coordinate context into the Cartesian grid.
5. **Hierarchical AMoE-3D Decoder**: A three-level depthwise-separable 3D UNet in which each level employs an attentive-MoE fusion module to aggregate multi-scale evidence.
6. **Segmentation Head**: A $1{\times}1{\times}1$ convolution outputs per-voxel semantic logits with deep supervision.

### Key Designs

#### DP-ER Dual-Projection Fusion

- **Function**: Processes the raw annular image and the unrolled ER image in parallel.
- **Mechanism**: The ER image preserves azimuthal continuity and is amenable to convolution; the raw annular image retains native geometry and fine-grained texture. The two branches provide complementary coverage of the equatorial–polar resolution/receptive-field trade-off.
- **Design Motivation**: The equatorial region of a panoramic camera dominates motion priors, whereas the polar region suffers from severe distortion. Taylor-based unrolling respects PAL optics, preserving annular continuity in ER space and local texture in the raw space, thereby providing more stable cues for subsequent voxel lifting.

#### BGV Dual-Grid Voxelization

- **Function**: Voxel centroids are defined simultaneously in a Cartesian grid $(x, y, z)$ and a cylindrical grid $(r, \varphi, z)$; dual-projection features are bilinearly sampled and lifted to 3D in each grid independently.
- **Mechanism**: Cartesian voxels accurately represent near-field contact geometry (foot placement, obstacles), while the azimuth angle $\varphi$ of cylindrical voxels aligns linearly with the horizontal axis of the panoramic image, preserving annular continuity and reducing far-field aliasing. Polar-coordinate features are resampled onto the Cartesian grid via precomputed cross-grid indices and concatenated.
- **Design Motivation**: Safety-critical decisions for legged robots simultaneously require near-field precision (safe footing) and far-field context (loop closure/scene layout); fusing dual grids balances near and far evidence.

#### GDC Gait Displacement Compensation

- **Function**: At each scale and projection branch, a 2D displacement $\Delta_s = (dx, dy)$ is regressed via global average pooling followed by a zero-initialized linear layer, correcting sampling coordinates before voxel lifting.
- **Mechanism**: Phase errors caused by gait impacts, if corrected after voxel lifting, are already contaminated by voxel quantization. Compensation in 2D sampling coordinates prior to lifting avoids quantization loss at lower computational cost.
- **Design Motivation**: Zero initialization ensures the module is equivalent to an identity transformation at the start of training (i.e., $\Delta_s \approx 0$ when no jitter is present), leaving baseline performance unaffected. Integer indexing is also upgraded to bilinear sampling, reducing projection aliasing.

#### AMoE-3D Hierarchical Attentive Mixture of Experts

- **Function**: At each of the three levels of the 3D UNet, dual-path volumetric saliency (channel attention + spatial attention) combined with gradient-energy-gated mixture of experts achieves scale-aware anisotropic fusion.
- **Mechanism**: Channel gate $A_c$ and spatial gate $A_s$ perform channel-wise and spatial-wise selection respectively. GradEnergy3D then computes the energy of 3D gradients along each axis; a softmax gate selects $K$ Conv-GELU-Conv experts (with $1{\times}1{\times}1$ kernels) for weighted aggregation.
- **Design Motivation**: Panoramic scenes are highly anisotropic (strong azimuthal variation vs. weak vertical structure, large near/far scale disparity). Gradient-energy gating amplifies high-contrast structures (vehicles, poles) at class boundaries while suppressing overfitting on large homogeneous regions (road surface), improving the stability of foot-placement decisions.

### Loss & Training

The total loss follows the MonoScene framework:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{SCAL}^{sem} + \mathcal{L}_{SCAL}^{geo} + \mathcal{L}_{FP}$$

- **Cross-entropy $\mathcal{L}_{CE}$**: Computed over valid voxels with class re-weighting.
- **Scene-Class Affinity Loss (SCAL)**: Applied in both semantic and geometric variants to constrain scene-level statistics.
- **Frustum Proportion Loss $\mathcal{L}_{FP}$**: Constrains per-frustum class proportions.
- The **relation loss is intentionally omitted**, as its co-occurrence priors over-smooth azimuthal boundaries and suppress small near-field classes in the panoramic legged setting.
- Deep supervision is applied at three resolution strides: {1, 2, 4}.

## Key Experimental Results

### Main Results

**Table 1: Semantic Scene Completion on the QuadOcc Validation Set** (real quadruped robot campus scenes, 6 semantic classes, $64{\times}64{\times}8$ grid)

| Method | Input | vehicle | pedestrian | road | building | vegetation | terrain | mIoU |
|--------|-------|---------|-----------|------|----------|-----------|---------|------|
| SSCNet | LiDAR | 0.00 | 0.04 | 44.34 | 16.06 | 22.41 | 4.77 | 14.60 |
| LMSCNet | LiDAR | 0.88 | 0.20 | 57.02 | 16.45 | 24.39 | 11.70 | **18.44** |
| OccFormer | Panoramic | 0.29 | 0.37 | 49.46 | 10.36 | 15.00 | 2.64 | 13.02 |
| MonoScene | Panoramic | 8.15 | 1.59 | 55.66 | 12.88 | 26.10 | 10.78 | 19.19 |
| SGN† | Panoramic+LiDAR | 11.62 | 2.47 | 53.06 | 15.60 | 25.67 | 9.91 | 19.72 |
| **OneOcc** | **Panoramic** | **12.16** | **2.86** | **54.41** | **16.03** | **24.91** | **13.01** | **20.56** |

**Table 2: Semantic Scene Completion on Human360Occ (H3O)** (CARLA-simulated humanoid 360°, 10 semantic classes)

| Method | Input | In-domain mIoU | Cross-domain mIoU |
|--------|-------|----------------|-------------------|
| VoxFormer-S | Panoramic + predicted depth | 11.09 | 10.63 |
| VoxFormer-S | Panoramic + GT depth | 15.44 | 14.82 |
| SGN-T | Panoramic + predicted depth | 28.64 | 20.02 |
| OccFormer | Panoramic | 24.88 | 20.87 |
| MonoScene | Panoramic | 33.46 | 24.15 |
| **OneOcc** | **Panoramic** | **37.29 (+3.83)** | **32.23 (+8.08)** |

### Ablation Study

**Table 3: Per-Module Ablation on QuadOcc**

| Variant | GDC | DP-ER | BGV | AMoE-3D | mIoU | Gain |
|---------|-----|-------|-----|---------|------|------|
| Q0 baseline | ✗ | ✗ | ✗ | ✗ | 19.19 | — |
| Q1 +GDC | ✓ | ✗ | ✗ | ✗ | 19.58 | +0.39 |
| Q2 +DP-ER | ✓ | ✓ | ✗ | ✗ | 19.89 | +0.31 |
| Q3 +BGV | ✓ | ✓ | ✓ | ✗ | 20.30 | +0.41 |
| Q4 +AMoE-3D (full) | ✓ | ✓ | ✓ | ✓ | **20.56** | +0.26 |

The gains from all four modules are additive: GDC stabilizes sampling → DP-ER supplies complementary cues → BGV reduces discretization bias → AMoE-3D sharpens boundaries and contact regions.

**Illumination Robustness** (QuadOcc day/dusk/night mIoU):

| Method | Day | Dusk | Night |
|--------|-----|------|-------|
| LMSCNet (LiDAR) | 17.33 | 18.91 | 13.40 |
| MonoScene (Vision) | 18.58 | 15.14 | 14.20 |
| **OneOcc** | **21.15** | **19.86** | 13.50 |

### Key Findings

1. **Vision-only surpasses LiDAR**: OneOcc with a single panoramic camera (20.56 mIoU) outperforms the best LiDAR method LMSCNet (18.44) by +11.5%, demonstrating that task-aligned panoramic fusion design can bridge the modality gap.
2. **Strong cross-domain generalization**: Under the H3O cross-city setting, OneOcc (32.23) substantially outperforms MonoScene (24.15) by +8.08 mIoU (+33.5% relative), indicating that the distortion-aware priors in DP-ER and AMoE-3D effectively mitigate distribution shift.
3. **Significant gains on rare classes**: On QuadOcc, vehicle IoU improves from 8.15 to 12.16 and pedestrian IoU from 1.59 to 2.86, reflecting the benefits of BGV's near/far-field balance and AMoE-3D's boundary sharpening.
4. **Moderate efficiency suitable for deployment**: 101.76M parameters; FP32 inference latency of 69.93 ms (14.3 FPS) on an RTX 4090; 52.84 ms (18.9 FPS) in mixed precision; peak GPU memory of 1.49 GB.

## Highlights & Insights

1. **Precise problem formulation**: The paper is the first to systematically define "panoramic semantic occupancy prediction for legged robots," clearly articulating the three mismatches in transferring from wheeled to legged platforms (motion pattern, field-of-view coverage, payload constraints).
2. **Zero-initialization design in GDC**: Borrowing ControlNet's zero-conv strategy, the module degrades to an identity mapping in the absence of jitter, ensuring training stability without degrading baseline performance.
3. **Gradient-energy-gated MoE**: Using 3D gradient energy as the expert routing signal is physically interpretable — high-gradient regions (class boundaries) activate fine-grained experts, while low-gradient regions (flat ground) follow simpler pathways.
4. **Complementary dual-coordinate systems**: The azimuth angle of cylindrical coordinates is naturally aligned with the horizontal axis of a panoramic image; this geometric insight means BGV genuinely exploits the intrinsic structure of panoramic imaging rather than merely substituting one grid for another.
5. **Comprehensive data contribution**: The simultaneous release of a real-world (QuadOcc) and a simulated (H3O) benchmark supports both in-domain and cross-domain evaluation.

## Limitations & Future Work

1. **Calibration dependency**: OneOcc assumes accurate calibration with bounded drift; in long-duration deployment, accumulated calibration drift may introduce errors. The authors suggest incorporating online extrinsic self-calibration.
2. **Degraded nighttime performance**: Night-scene mIoU (13.50) falls below MonoScene (14.20), as vision-only panoramic methods are limited by the sensor's dynamic range under extremely low illumination.
3. **Simplified GDC formulation**: Only a global 2D translation is regressed; rotational or spatially varying distortions are not modeled, which may be insufficient for complex multi-DOF gaits.
4. **Temporal information not exploited**: The current method is single-frame; inter-frame temporal consistency and the periodic regularity of gait cycles are not utilized.
5. **Sim-to-real transfer**: H3O is built on CARLA simulation; the domain gap has not been systematically evaluated.

## Related Work & Insights

- **MonoScene** serves as the direct baseline (Q0); OneOcc incrementally adds four modules atop it, achieving a +1.37 mIoU improvement.
- The cylindrical discretization in **Cylinder3D** is inherited and extended by BGV into a dual-grid fusion scheme.
- The zero-initialization strategy from **ControlNet** is adopted in GDC to ensure harmless initialization of the plug-in module.
- **MoE for 3D perception** (e.g., Point-MoE) inspires AMoE-3D, but gradient energy replaces conventional routing.
- The work meaningfully complements panoramic occupancy perception research (Humanoid Occupancy, OmniHD-Scenes) and advances this emerging direction.

## Rating

⭐⭐⭐⭐ A systematic contribution that precisely defines a new problem, presents a complete method, and contributes dual datasets. The design motivation behind each of the four modules is clear and their gains are additive. However, the individual gain per module is modest (0.26–0.41 mIoU), and nighttime performance remains a weakness.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2occ_resilient_3d_semantic_occupancy_prediction_f.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

<!-- RELATED:END -->
