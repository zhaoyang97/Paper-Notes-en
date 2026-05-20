---
title: >-
  [Paper Note] TT-Occ: Test-Time 3D Occupancy Prediction
description: >-
  [CVPR2026][Autonomous Driving][3D occupancy prediction] This paper proposes TT-Occ, a training-free test-time 3D occupancy prediction framework that integrates vision foundation models (VFMs) at inference time to increme…
tags:
  - "CVPR2026"
  - "Autonomous Driving"
  - "3D occupancy prediction"
  - "test-time"
  - "3D Gaussian Splatting"
  - "vision foundation models"
  - "self-supervised"
  - "open-vocabulary"
date: 2026-05-08
content_hash: 3c5cc8cd4af8487d
---

# TT-Occ: Test-Time 3D Occupancy Prediction

**Conference**: CVPR2026
**arXiv**: [2503.08485](https://arxiv.org/abs/2503.08485)  
**Code**: [Xian-Bei/TT-Occ](https://github.com/Xian-Bei/TT-Occ)  
**Area**: Autonomous Driving / 3D Occupancy Prediction
**Keywords**: 3D occupancy prediction, test-time, 3D Gaussian Splatting, vision foundation models, self-supervised, open-vocabulary

## TL;DR

This paper proposes TT-Occ, a training-free test-time 3D occupancy prediction framework that integrates vision foundation models (VFMs) at inference time to incrementally construct, refine, and voxelize temporally-aware 3D Gaussians. TT-Occ surpasses all self-supervised methods requiring extensive training on both Occ3D-nuScenes and nuCraft benchmarks.

## Background & Motivation

**Importance of 3D Occupancy Prediction**: 3D occupancy prediction requires accurately identifying regions occupied by objects of specific categories as well as free space, which is critical for collision-free trajectory planning and reliable navigation in autonomous driving.

**High Annotation Cost of Supervised Methods**: Existing fully supervised approaches rely heavily on dense per-frame 3D annotations, which are extremely costly to obtain in dynamic driving scenarios (covering an 80m range per frame).

**High Training Cost of Self-Supervised Methods**: Although self-supervised methods reduce annotation costs, their training overhead remains substantial — for instance, SelfOcc requires approximately 384 GPU hours (8 GPUs for 2 days) at 0.4m resolution on Occ3D-nuScenes.

**Poor Generalization**: Once trained, adapting to finer resolutions (e.g., 0.2m on nuCraft) or novel object categories demands extensive retraining, limiting flexibility.

**The Rise of VFMs Changes the Landscape**: 3D vision foundation models such as VGGT and MapAnything provide reliable multi-view geometry, while REX-Omni supports open-vocabulary semantic reasoning — all accessible directly at test time without task-specific training.

**Core Problem**: Given that geometric and semantic information no longer needs to be learned by a network, is training an occupancy prediction model still necessary? TT-Occ answers in the negative.

## Method

### Overall Architecture: Lift-Track-Voxelize

TT-Occ follows a three-stage Lift-Track-Voxelize pipeline, offering two variants: a LiDAR-based variant (TT-OccLiDAR) and a camera-only variant (TT-OccCamera).

#### Step 1: Lift — Lifting Geometry and Semantics into 3D Gaussians

- **Modality-Specific Initialization**:
    - **TT-OccLiDAR**: Sparse LiDAR points are directly initialized as 3D Gaussians, inheriting accurate spatial positions from real-world measurements.
    - **TT-OccCamera**: A 3D vision foundation model (VGGT/MapAnything) estimates dense depth maps from multi-view RGB inputs; multi-view triangulation is applied to resolve scale ambiguity.
- **VFM Semantics**: An open-vocabulary segmentation model (OpenSeeD/GroundingSAM2/REX-Omni) extracts semantic maps from $M$ surround-view images, which are fused into 3D via visibility-weighted projection: $\mathbf{m}_i = \frac{1}{M}\sum_{m=1}^{M}\mathbb{I}_m(\boldsymbol{\mu}_i)\mathcal{M}_m(\text{Proj}(\boldsymbol{\mu}_i;\mathbf{K}_m,\mathbf{E}_m))$
- **Voxel-Aware Simplification**: Scale parameters are constrained with sigmoid (rather than exponential) to prevent excessive growth; redundant Gaussians within the same voxel are pruned and their semantic probabilities are merged.

#### Step 2: Track — Tracking Dynamic Gaussians

Fast-moving objects (vehicles, pedestrians) are typically only partially observed, and online optimization of 3D Gaussians introduces severe trailing artifacts.

- **TT-OccLiDAR**: A learning-free scene flow estimation approach — LiDAR points are projected onto segmentation masks to associate instances → DBSCAN denoising → cross-frame cluster matching based on spatial/shape similarity → ICP-based 3D motion flow estimation.
- **TT-OccCamera**: RAFT is used to estimate optical flow; ego-motion flow is subtracted to obtain residual dynamic flow → thresholding yields a dynamic mask identifying moving regions → corresponding 3D Gaussians are excluded from static accumulation (a compromise strategy to avoid noise amplification from 3D back-projection).

#### Step 3: Voxelize — Gaussian Voxelization

- **Differentiable Optimization**: Gaussian parameters are refined at test time via a color consistency loss.
- **Trilateral Radial Basis Function (TRBF) Smoothing**: An optional denoising module that periodically smooths Gaussian parameters by jointly leveraging spatial, color, and semantic affinity: $\mathcal{K}(i,j) = \mathcal{K}_{\boldsymbol{\mu}}(i,j)\cdot\mathcal{K}_{\mathbf{c}}(i,j)\cdot\mathcal{K}_{\mathbf{m}}(i,j)$
- **Flexible-Resolution Voxelization**: Semantic probabilities are aggregated into a discrete occupancy grid weighted by spatial proximity, supporting arbitrary user-specified resolutions.

### Loss & Training

- Color consistency loss (projecting 3D Gaussians back to the image plane via differentiable rendering)
- Sky region masking to exclude invalid areas

## Key Experimental Results

### Main Results

**Occ3D-nuScenes (0.4m resolution)**:

| Method | Input | Pretraining | mIoU |
|---|---|---|---|
| SelfOcc (CVPR'24) | C | ~384 GPU hrs | 10.54 |
| GaussianTR (CVPR'25) | C | Yes | 11.70 |
| VEON-LiDAR (ECCV'24) | C&L | Yes | 15.14 |
| **TT-OccCamera** | **C** | **None** | **16.70** |
| RenderOcc (ICRA'24) | C | Yes (sparse 3D GT) | 23.93 |
| **TT-OccLiDAR** | **C&L** | **None** | **27.41** |
| BEVFormer (ECCV'22) | C | Yes (dense 3D GT) | 26.88 |

**nuCraft High-Resolution (0.2m resolution)**:

| Method | Pretraining Time | mIoU |
|---|---|---|
| SelfOcc† | 384 hrs | 2.22 |
| TT-OccCamera | 0 | 5.95 |
| TT-OccLiDAR | 0 | 10.92 |

### Ablation Study

| Configuration | TT-OccLiDAR mIoU | TT-OccCamera mIoU |
|---|---|---|
| A: Baseline (single-frame direct splatting) | 7.3 | 4.2 |
| B: + Covariance-aware voxelization | 18.3 (+11.0) | 8.5 (+4.3) |
| C: + Inheriting historical Gaussians (no tracking) | 23.5 (+5.2) | 14.1 (+5.6) |
| D: + Dynamic Gaussian tracking | 25.6 (+2.1) | 14.1 (+0.0) |

### Key Findings

1. **Zero-Training Outperforms Trained Methods**: TT-OccLiDAR (27.41 mIoU) surpasses even RenderOcc (23.93), which is trained with sparse 3D GT; TT-OccCamera (16.70) outperforms VEON-LiDAR (15.14), trained with LiDAR supervision.
2. **Strong Resolution Adaptability**: On the nuCraft high-resolution setting, SelfOcc drops sharply from 10.54 to 2.22, while TT-Occ adapts without any retraining.
3. **RayIoU Validates Geometric Quality**: TT-OccCamera improves over SelfOcc by 30.8% on RayIoU@4; TT-OccLiDAR improves by 115%.
4. **Modular VFM Design**: REX-Omni yields the strongest semantics; MapAnything depth outperforms VGGT due to metric-scale depth estimation; the framework supports plug-and-play replacement of VFM components.
5. **Dynamic Tracking Eliminates Trailing Artifacts**: Without tracking, IoU for dynamic classes (bus, pedestrian) degrades significantly and recovers markedly with tracking.
6. **Memory Efficiency**: Peak GPU memory is 5.6 GB for the LiDAR variant and 9.9 GB for the camera variant, both under 10 GB.

## Highlights & Insights

- **Paradigm Innovation**: This is the first work to demonstrate that integrating VFMs at test time can fully replace training a dense occupancy decoder, shifting 3D occupancy prediction from a *training paradigm* to an *inference paradigm*.
- **Exceptional Flexibility**: Supports arbitrary voxel resolutions, open-vocabulary semantic queries, and plug-and-play VFM component replacement.
- **Zero Training Cost**: Eliminates hundreds of GPU hours of pretraining; inference runs directly on the validation set.
- **Temporally-Aware Gaussians**: The Lift-Track-Voxelize pipeline enables online incremental scene reconstruction with dynamic-static decomposition to eliminate trailing artifacts.
- **TRBF Smoothing**: Extends bilateral filtering to a trilateral kernel that performs adaptive denoising jointly over spatial, color, and semantic dimensions.

## Limitations & Future Work

- **Dependence on VFM Quality**: Performance is upper-bounded by the capabilities of the VFMs employed; with the weaker GroundingSAM2 semantics, mIoU drops to 21.3.
- **Limited Dynamic Handling for Camera Variant**: Unlike the LiDAR variant, the camera-only variant cannot accumulate dynamic objects and can only exclude dynamic regions.
- **Inference Speed**: Semantic segmentation (OpenSeeD) accounts for 28.5%–77.9% of total runtime; the camera variant additionally requires depth estimation, triangulation, and other steps.
- **Camera Variant Degrades at Long Range**: Due to occlusion and depth resolution limitations, the camera-only variant achieves inferior geometric accuracy at long distances compared to the LiDAR variant.
- **Open-Vocabulary Sensitivity to Prompt Quality**: VFM semantic segmentation relies on text prompts; while predefined label sets are used on standard benchmarks, prompt design affects performance in real open-world scenarios.

## Related Work & Insights

- **Fully Supervised Occupancy Prediction**: BEVFormer, CTF-Occ, RenderOcc — rely on dense/sparse 3D annotations.
- **Self-Supervised Occupancy Prediction**: SelfOcc (SDF + multi-view stereo), OccNeRF (photometric consistency), GaussianOcc/GaussianTR (3DGS representation), LangOcc/VEON (open-vocabulary).
- **3D Reconstruction for Driving Scenes**: OmniRe, Street Gaussians, DrivingGaussian, HUGS — depend on external priors (HD maps, GT bounding boxes) for offline per-scene reconstruction; by contrast, TT-Occ performs online inference using only raw sensor streams.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first training-free test-time 3D occupancy prediction framework; a paradigm-level contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, two modality variants, multi-VFM combination ablations, and RayIoU evaluation; comprehensive overall.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured; the Lift-Track-Voxelize framework is intuitive and easy to follow.
- Value: ⭐⭐⭐⭐⭐ — Challenges the necessity of training occupancy models in the VFM era, with far-reaching implications for autonomous driving perception paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)

</div>

<!-- RELATED:END -->
