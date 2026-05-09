---
title: >-
  [Paper Note] RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation
description: >-
  [ICCV 2025][3D Vision][roadside collaborative perception] RoCo-Sim is proposed as the first simulation framework for roadside collaborative perception. By integrating extrinsic parameter optimization, occlusion-aware 3D asset placement, DepthSAM-based depth modeling, and style-transfer post-processing, it generates multi-view consistent simulation data from single images, achieving over 83% improvement in roadside 3D detection performance.
tags:
  - ICCV 2025
  - 3D Vision
  - roadside collaborative perception
  - simulation data
  - foreground editing
  - 3D detection
  - multi-view consistency
date: 2026-05-08
content_hash: 66e1998c42b3c2f8
---

# RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation

**Conference**: ICCV 2025
**arXiv**: [2503.10410](https://arxiv.org/abs/2503.10410)
**Code**: [https://github.com/duyuwen-duen/RoCo-Sim](https://github.com/duyuwen-duen/RoCo-Sim)
**Area**: 3D Vision / Collaborative Perception
**Keywords**: roadside collaborative perception, simulation data, foreground editing, 3D detection, multi-view consistency

## TL;DR
RoCo-Sim is proposed as the first simulation framework for roadside collaborative perception. By integrating extrinsic parameter optimization, occlusion-aware 3D asset placement, DepthSAM-based depth modeling, and style-transfer post-processing, it generates multi-view consistent simulation data from single images, achieving over 83% improvement in roadside 3D detection performance.

## Background & Motivation

**Background**: Roadside collaborative perception enhances vehicles' environmental understanding by enabling multiple roadside units to share perceptual data. Existing methods focus on model architecture design but perform poorly on recent datasets.

**Limitations of Prior Work**: (1) Calibration of fixed roadside cameras is difficult and drifts over time; (2) prolonged intervals without passing vehicles result in sparse information density; (3) ensuring multi-view annotation consistency is challenging and data collection costs are high. Existing simulation methods (NeRF/3DGS) fail to reconstruct scenes under the sparse fixed viewpoints typical of roadside setups, while diffusion-based methods cannot guarantee multi-view consistency.

**Key Challenge**: Roadside perception models lack sufficient high-quality training data, and existing simulation methods are ill-suited for fixed-viewpoint roadside scenes.

**Goal**: To build the first roadside simulation framework that generates multi-view consistent, information-dense synthetic training data from single real images.

**Key Insight**: Rather than adopting the scene reconstruction paradigm of NeRF/3DGS, the paper establishes a 3D-to-2D mapping via a 3D asset library, rendering digital vehicles onto real backgrounds — ensuring 3D consistency without requiring multi-view training data.

**Core Idea**: Extrinsic optimization ensures accurate projection → an occlusion-aware sampler places assets appropriately in 3D space → DepthSAM models foreground-background occlusion relationships → style-transfer post-processing ensures photorealism. The entire pipeline can be deployed to new scenes without any training.

## Method

### Overall Architecture
(1) Camera Extrinsic Optimizer refines roadside camera extrinsics → (2) MOAS determines digital asset placement positions in 3D space → (3) DepthSAM models per-frame foreground-background depth relationships → (4) 3D assets are rendered onto 2D backgrounds with style-transfer post-processing. Edits are performed in 3D space and automatically propagated to all viewpoints to guarantee consistency.

### Key Designs

1. **Camera Extrinsic Optimizer**:

    - **Function**: Reduces calibration errors in fixed roadside cameras.
    - **Mechanism**: Mathematically models the 3D-to-2D projection process and iteratively minimizes extrinsic errors via an optimization algorithm. Provides a convenient calibration tool applicable to various roadside cameras.
    - **Design Motivation**: Roadside camera extrinsics are prone to drift due to installation offsets and environmental disturbances; accurate extrinsics are a prerequisite for correct simulation rendering.

2. **Multi-View Occlusion-Aware Sampler (MOAS)**:

    - **Function**: Places virtual vehicles in 3D space in a physically plausible manner, ensuring multi-view occlusion consistency.
    - **Mechanism**: Samples 3D positions within road regions for asset placement, accounting for occlusion relationships across multiple camera viewpoints to ensure that visibility and occlusion states of the same virtual vehicle are consistent across views.
    - **Design Motivation**: Naive random placement leads to physically implausible artifacts such as floating or interpenetrating objects, and may produce contradictory occlusion states across views.

3. **DepthSAM**:

    - **Function**: Models foreground-background depth relationships from single-frame fixed-viewpoint images.
    - **Mechanism**: Combines depth estimation and SAM segmentation to establish precise occlusion relationships for each frame. During rendering, 3D assets are correctly occluded in regions blocked by real foreground objects, maintaining multi-view consistency.
    - **Design Motivation**: Roadside scenes contain streetlights, guardrails, and other structures that partially occlude vehicles; ignoring occlusion results in unrealistic simulation and inconsistent annotations.

### Loss & Training
RoCo-Sim is a simulation framework and does not involve training. The generated data is used to train downstream 3D detection models such as BEVHeight.

## Key Experimental Results

### Main Results

| Dataset | Method | AP70 | Gain |
|--------|------|------|------|
| RCooper-Intersection | BEVHeight + RoCo-Sim | Best | **+83.74%** vs. SOTA |
| TUMTraf-V2X | BEVHeight + RoCo-Sim | Best | **+83.12%** vs. SOTA |
| RCooper (extrinsic opt. only) | BEVHeight | Significant | +62.55% |

### Ablation Study

| Component | AP70 Change | Note |
|------|----------|------|
| Extrinsic optimization | +62.55% | Most critical single improvement |
| + Simulation data | Further gain | More data → better performance |
| More simulated vehicles/image | Consistent gain | Increased information density |
| Style transfer (rain/night) | Improved robustness | Scene diversity |

### Key Findings
- Extrinsic optimization is the single largest performance gain factor (+62.55%), indicating that calibration quality issues in roadside datasets are extremely severe.
- Performance correlates positively with the volume of simulation data and the number of simulated vehicles per image, with no observed saturation.
- The gains from data simulation far exceed those from algorithmic improvements — in roadside perception, data quality is the current bottleneck.

## Highlights & Insights
- **A case study of data-driven approaches outperforming model innovation**: Extrinsic optimization combined with simulation data alone substantially surpasses carefully designed model architectures, demonstrating that data quality is the key bottleneck in roadside perception.
- **3D asset rendering pipelines are better suited to roadside scenes than NeRF/diffusion-based generation**: they require zero training, guarantee multi-view consistency, and produce annotations automatically.
- The paper delivers a complete roadside simulation toolchain (calibration → placement → rendering → post-processing) with strong engineering utility.

## Limitations & Future Work
- The diversity of the 3D asset library limits the richness of simulated scenarios.
- The rendering-to-real domain gap may affect transfer performance; style transfer cannot fully eliminate this discrepancy.
- Only foreground elements (e.g., vehicles) are simulated; background modifications such as roads or buildings are not supported.
- Simulation trajectory generation for dynamic scenes is based on predefined rules rather than traffic model-driven generation.

## Related Work & Insights
- **vs. CARLA**: CARLA relies on fully manual scene simulation, which is time-consuming and lacks realism; RoCo-Sim overlays 3D assets onto real backgrounds.
- **vs. MagicDrive/DriveDreamer**: Diffusion-based methods do not guarantee multi-view consistency and require training; RoCo-Sim requires zero training.
- **vs. ChatSim/OmniRe**: NeRF/3DGS-based methods require multi-view training data and are not applicable to the fixed, sparse viewpoints of roadside cameras.

## Rating
- Novelty: ⭐⭐⭐⭐ First simulation framework for roadside collaborative perception, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Substantial improvements on two real-world datasets with comprehensive component ablations.
- Writing Quality: ⭐⭐⭐⭐ Rich and intuitive visualizations of the simulation pipeline.
- Value: ⭐⭐⭐⭐⭐ Performance gains exceeding 83% demonstrate strong practical engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs](../../CVPR2026/3d_vision/sope_spherical_coordinate-based_positional_embedding_for_enhancing_spatial_perce.md)
- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)

<!-- RELATED:END -->
