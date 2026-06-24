---
title: >-
  [Paper Note] VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving
description: >-
  [CVPR 2025][Autonomous Driving][Self-Supervised Pre-training] This paper proposes VisionPAD, a vision-centric self-supervised pre-training framework. It replaces volume rendering with anchor-based 3D Gaussian Splatting to reconstruct multi-view images, and introduces self-supervised voxel velocity estimation combined with multi-frame photometric consistency constraints to learn motion cues and 3D geometry. Completely independent of LiDAR depth supervision…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Self-Supervised Pre-training"
  - "3D Gaussian Splatting"
  - "Voxel Velocity Estimation"
  - "Photometric Consistency"
  - "Autonomous Driving Perception"
date: 2026-05-08
content_hash: 332edbcdf17034d1
---

# VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving

**Conference**: CVPR 2025  
**arXiv**: [2411.14716](https://arxiv.org/abs/2411.14716)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Self-Supervised Pre-training, 3D Gaussian Splatting, Voxel Velocity Estimation, Photometric Consistency, Autonomous Driving Perception

## TL;DR

This paper proposes VisionPAD, a vision-centric self-supervised pre-training framework. It replaces volume rendering with anchor-based 3D Gaussian Splatting to reconstruct multi-view images, and introduces self-supervised voxel velocity estimation combined with multi-frame photometric consistency constraints to learn motion cues and 3D geometry. Completely independent of LiDAR depth supervision, it significantly outperforms existing pre-training methods on three downstream tasks: 3D detection, occupancy prediction, and map segmentation.

## Background & Motivation

**Background**: Vision-centric autonomous driving perception methods extract BEV and occupancy features from multi-view images and have achieved outstanding performance in various downstream tasks. Pre-training is a crucial strategy for scaling downstream applications, but the acquisition of high-quality 3D annotations (such as occupancy and 3D bounding boxes) is extremely costly.

**Limitations of Prior Work**: Recent rendering-based pre-training methods (such as UniPAD) use volume rendering to reconstruct multi-view depth and images, but heavily rely on explicit depth supervision projected from LiDAR to learn 3D geometry. When supervised solely by images, UniPAD performs poorly (with NDS even dropping by 0.2), limiting its application in camera-only systems. Furthermore, volume rendering can only sample a limited number of rays per iteration, making it difficult to reconstruct fine-grained details in high-resolution images.

**Key Challenge**: Vision-only pre-training requires simultaneous learning of appearance, 3D geometry, and motion information. However, inferring this information from RGB images without explicit depth supervision is highly challenging. In addition, the computational cost of volume rendering is sensitive to resolution, restricting the richness of supervision signals.

**Goal**: To design a pre-training framework that can effectively learn 3D geometry and motion representations relying solely on multi-frame multi-view images.

**Key Insight**: 3D Gaussian Splatting (3D-GS) is based on splat rasterization, making its computational cost insensitive to resolution, which allows rendering higher-resolution images to provide richer supervision signals. Voxel velocity can be self-supervised learned through temporal consistency, and photometric consistency can constrain cross-frame relationships using rendered depth and relative poses.

**Core Idea**: To replace volume rendering with 3D Gaussian Splatting for image reconstruction pre-training. Self-supervised motion learning is achieved through voxel velocity prediction and warping, and self-supervised geometry learning is achieved via photometric consistency loss. These three components work synergistically to form a pre-training scheme that is entirely free of depth annotations.

## Method

### Overall Architecture

VisionPAD utilizes a general visual perception network as the backbone. It takes multi-frame multi-view images as input, generating voxel features $\mathbf{V} \in \mathbb{R}^{X \times Y \times Z \times C}$ through 2D feature extraction and view transformation. The pre-training consists of four modules: (1) a voxel construction module to extract 3D volumetric representations; (2) a 3D-GS decoder to convert voxel features into Gaussian primitives and render multi-view images and depth maps of the current frame; (3) a voxel velocity estimation module to predict voxel-wise velocity and warp features to adjacent frames, supervised by self-supervised reconstruction of adjacent frame images; (4) a photometric consistency module utilizing the rendered depth and relative poses for cross-frame projection constraints.

### Key Designs

1. **Anchor-based 3D-GS Decoder**:

    - Function: Efficiently renders high-resolution multi-view images and depth maps from voxel features.
    - Mechanism: Each voxel center is treated as an anchor, and an MLP is used to predict the attributes of multiple Gaussian primitives (offset, spherical harmonics, opacity, scale, rotation). The rendering formula is $\mathbf{C}(p) = \sum_{i \in K} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_i)$, and the depth map is similarly obtained via alpha-blending of distance values: $\mathbf{D}(p) = \sum_{i \in K} d_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$. Gaussian filtering is implemented: $\tanh$ is used to predict opacity, and low-confidence Gaussians with values $<0$ are discarded to reduce computational cost.
    - Design Motivation: Volume rendering requires sampling along rays and is sensitive to resolution. In contrast, 3D-GS is based on parallel rasterized projection, where the impact of resolution is minimal, allowing it to render higher-resolution images under the same computational budget. Ablation studies demonstrate that simply replacing the decoder with 3D-GS yields NDS improvements, and Gaussian filtering further boosts NDS by 0.6.

2. **Self-Supervised Voxel Velocity Estimation**:

    - Function: Learns voxel-wise motion information without any motion annotations, encoding dynamic/static distinctions in object representations.
    - Mechanism: An auxiliary velocity head is attached after voxel features to predict voxel-wise velocity vectors in the world coordinate system. Voxel flow is approximated using the temporal interval between frames (velocity $\times$ time difference), and the current frame voxels are warped to neighboring frame positions via GridSample. The neighboring multi-view images are then rendered using the 3D-GS decoder, supervised by the corresponding ground-truth neighboring images. Crucially, during backpropagation, only the velocity head parameters are updated, guiding the network to focus on learning discriminative motion features.
    - Design Motivation: Traditional methods rely on LiDAR to obtain motion information. This design leverages the self-consistent property that "if velocity prediction is correct, the warped voxels should be able to reconstruct the neighboring frames" for self-supervision. Ablation shows an increase of 1.2 points in mAP after adding velocity estimation.

3. **Multi-frame Photometric Consistency**:

    - Function: Leverages the rendered depth maps and known camera poses to achieve self-supervised 3D geometry learning.
    - Mechanism: Inspired by self-supervised depth estimation, the depth map rendered from the current frame $\mathbf{D}_t = \text{3DGS}(\mathbf{V}_t, \mathbf{K}_t, \mathbf{T}_t)$ is used to reproject neighboring images $\mathbf{I}_{t'}$ to the current frame view $\mathbf{I}_{t' \to t} = \mathbf{I}_{t'}\langle \text{proj}(\mathbf{D}_t, \mathbf{T}_{t \to t'}, \mathbf{K}) \rangle$. The photometric consistency loss combines SSIM and L1: $\mathcal{L}_{pc} = \alpha(1 - \text{SSIM}(\mathbf{I}_t, \mathbf{I}_{t' \to t})) + (1-\alpha)\|\mathbf{I}_t - \mathbf{I}_{t' \to t}\|_1$.
    - Design Motivation: A correct depth map is required to make the reprojected image consistent with the target frame. This constraint forces the model to learn precise 3D geometry. Ablation proves this to be the most contributing component (NDS +2.4, mAP +4.4).

### Loss & Training

The total pre-training loss is $\mathcal{L} = \omega_1 \mathcal{L}_{img} + \omega_2 \mathcal{L}_{vel} + \omega_3 \mathcal{L}_{pc}$, where $\omega_1=0.5, \omega_2=1, \omega_3=1$. Both $\mathcal{L}_{img}$ and $\mathcal{L}_{vel}$ are L1 reconstruction losses. Pre-training is conducted for 12 epochs using the AdamW optimizer with a learning rate of $2 \times 10^{-4}$ and a batch size of 4. During the fine-tuning stage, the official downstream model configurations are used without modification. Data augmentations include random scaling/rotation and partial input masking.

## Key Experimental Results

### Main Results

3D Object Detection (nuScenes val):

| Method | Pre-train Modality | NDS↑ | mAP↑ |
|------|-----------|------|------|
| UVTR (baseline) | - | 48.8 | 39.2 |
| UVTR + UniPAD (C only) | C | 48.6 (-0.2) | 40.5 (+0.7) |
| UVTR + UniPAD (C+L) | C+L | 50.2 (+1.4) | 42.8 (+3.6) |
| **UVTR + VisionPAD** | **C only** | **49.7 (+0.9)** | **41.2 (+2.0)** |
| **UVTR + VisionPAD** | **C+L** | **50.4 (+1.6)** | **43.1 (+3.9)** |

Semantic Occupancy Prediction (Occ3D val) / Map Segmentation:

| Method | Occ mIoU↑ | Map Lane IoU↑ |
|------|-----------|-------------|
| UVTR (baseline) | 30.1 | 15.0 |
| UVTR + UniPAD | 31.0 (+0.9) | 16.3 (+1.3) |
| **UVTR + VisionPAD** | **35.4 (+5.4)** | **20.4 (+5.4)** |

### Ablation Study

| Config | NDS | mAP | Description |
|------|-----|-----|------|
| UVTR baseline | 22.8 | 19.4 | - |
| + UniPAD (Vol. Rend.) | 22.3 (-0.5) | 18.3 (-1.1) | Volume rendering with pure image supervision is harmful |
| + 3DGS Decoder | 22.8 (+0.0) | 18.2 (-1.2) | Replaced with 3DGS |
| + Gaussian Filtering | 23.4 (+0.6) | 18.9 (-0.5) | Filter low-opacity Gaussians |
| + Velocity Estimation | 23.6 (+0.8) | 20.1 (+0.7) | Motion cues |
| + Photometric Consistency | 26.0 (+3.2) | 24.5 (+5.1) | Highly contributing component |
| **Full VisionPAD** | **27.3 (+4.5)** | **26.5 (+7.1)** | Synergistic combination of all components |

### Key Findings

- Photometric consistency is the most important component (contributing NDS +2.4, mAP +4.4 individually), demonstrating that cross-frame geometry constraints are vital for vision-only pre-training.
- Under pure image supervision, UniPAD's volume rendering pre-training leads to performance degradation (NDS -0.5), whereas VisionPAD achieves a significant improvement of +4.5 NDS.
- Data efficiency experiments show that the advantages of VisionPAD are even more pronounced when using only 25% of the fine-tuning data (+6 mAP), proving that pre-training is highly valuable in data-scarce scenarios.
- One anchor per voxel is sufficient; increasing to 2/3/4 anchors per voxel actually degrades performance.

## Highlights & Insights

- First to apply 3D-GS to autonomous driving pre-training, breaking the resolution bottleneck of volume rendering by utilizing its resolution-insensitive properties.
- The self-supervised design of voxel velocity estimation is highly elegant—the self-consistency of "if the velocity is correct, the warped voxels should be able to reconstruct the neighboring frames" provides free motion supervision signals, and updating only the velocity head parameters avoids interfering with backbone representation learning.
- Migrating photometric consistency from self-supervised depth estimation to pre-training is a natural and effective integration, as the rendered depth map itself is a byproduct of pre-training.

## Limitations & Future Work

- Experiments are only validated on the nuScenes dataset, lacking cross-dataset generalization validation.
- Photometric consistency assumes a static scene, so dynamic object regions can introduce erroneous gradients (although velocity estimation partly mitigates this).
- The 3D-GS decoder introduces extra MLPs and parameters for Gaussian primitive prediction, which increases the memory overhead during pre-training.
- Future directions: (1) Incorporate temporal forecasting (similar to ViDAR) to further utilize motion information; (2) Expand pre-training to larger-scale unlabeled driving video data; (3) Explore photometric consistency variants capable of handling dynamic objects.

## Related Work & Insights

- **vs UniPAD**: UniPAD relies on volume rendering with LiDAR depth supervision, and its performance in pure image mode is poor. VisionPAD eliminates dependence on LiDAR entirely via 3D-GS and self-supervised geometric constraints, outperforming UniPAD by +2.5 mAP under the same settings.
- **vs ViDAR**: ViDAR uses a Transformer to predict future frames and render depth (still requiring LiDAR supervision). VisionPAD's voxel velocity estimation introduces temporal information from a different perspective and is fully self-supervised.
- **vs Self-Supervised Depth Estimation**: This paper generalizes Monodepth-style photometric consistency from separate depth models to a perception pre-training framework, representing an elegant cross-domain technology migration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of 3D-GS for pre-training, self-supervised velocity estimation, and photometric consistency is novel, though individual components have known precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive across three downstream tasks, detailed ablations, and data efficiency analysis, but restricted to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear method description with progressive ablation logic; comparison with UniPAD is maintained throughout the text.
- **Value**: ⭐⭐⭐⭐⭐ — Eliminating autonomous driving pre-training's reliance on LiDAR has significant practical implications, and the method is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](../../CVPR2026/autonomous_driving/dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[CVPR 2025\] UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)
- [\[CVPR 2025\] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](spatiotemporal_decoupling_for_efficient_vision-based_occupancy_forecasting.md)
- [\[ICLR 2026\] To View Transform or Not to View Transform: NeRF-based Pre-training Perspective](../../ICLR2026/autonomous_driving/to_view_transform_or_not_to_view_transform_nerf-based_pre-training_perspective.md)

</div>

<!-- RELATED:END -->
