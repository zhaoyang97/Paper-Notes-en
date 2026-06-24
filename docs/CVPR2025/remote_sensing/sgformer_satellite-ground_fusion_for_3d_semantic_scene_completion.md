---
title: >-
  [Paper Note] SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion
description: >-
  [CVPR 2025][Remote Sensing][Semantic Scene Completion] This work introduces satellite imagery to the 3D Semantic Scene Completion (SSC) task for the first time, proposing a dual-branch framework named SGFormer. By utilizing ground-view guided satellite feature correction and adaptive fusion strategies, it effectively addresses the scene incompleteness issue caused by visual occlusions.
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "Semantic Scene Completion"
  - "Satellite-Ground Fusion"
  - "Bird's-Eye View"
  - "Deformable Attention"
  - "Adaptive Weight"
date: 2026-05-08
content_hash: 5aee6efeede59946
---

# SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion

**Conference**: CVPR 2025  
**arXiv**: [2503.16825](https://arxiv.org/abs/2503.16825)  
**Code**: [GitHub](https://github.com/gxytcrc/SGFormer)  
**Area**: Remote Sensing  
**Keywords**: Semantic Scene Completion, Satellite-Ground Fusion, Bird's-Eye View, Deformable Attention, Adaptive Weight

## TL;DR

This work introduces satellite imagery to the 3D Semantic Scene Completion (SSC) task for the first time, proposing a dual-branch framework named SGFormer. By utilizing ground-view guided satellite feature correction and adaptive fusion strategies, it effectively addresses the scene incompleteness issue caused by visual occlusions.

## Background & Motivation

3D Semantic Scene Completion (SSC) aims to predict the occupancy status and semantic category of each voxel in a scene, which is a crucial task for autonomous driving and robot navigation. Although existing camera-based methods have made progress, they face the following fundamental bottlenecks:

- **Non-unique correspondence of 3D-2D projection**: Multiple 3D voxels correspond to overlapping regions of 2D images, leading to semantic ambiguity and radial artifacts.
- **Severe visual occlusion**: The ground camera has a limited perspective, and the occluded areas lack global perspective information, making it difficult to recover the complete scene.
- **Difficulty in long-range area prediction**: Ground-only methods lack long-range global perspective, affecting planning and decision-making.
- **Uncertainty in depth estimation**: Even with depth information, it can only process visible areas.

Satellite imagery offers several advantages: low-cost and widely available; BEV perspective naturally fits the horizontal layout of urban scenes; wide coverage can effectively supplement the blind spots of ground views. However, incorporating satellite imagery into SSC faces two major challenges: alignment and information quality.

## Method

### Overall Architecture

SGFormer adopts a dual-branch architecture: the ground branch utilizes EfficientNet-B7 to extract features and transforms 2D features to 3D voxel space via depth guidance; the satellite branch uses ResNet-50 to extract features and projects them into the BEV space. The features from both branches are merged through an adaptive fusion module, and subsequently output voxel-level category predictions via a semantic head.

### Key Design 1: Ground-View Guided Satellite Feature Correction — Addressing Satellite-Ground View Misalignment

**Function**: Utilize the compressed features of the ground branch to guide the feature learning of the satellite branch, addressing the feature misalignment issue caused by satellite image localization noise and top-down occlusion.

**Mechanism**: First, the 3D voxel features of the ground branch $\mathbf{F}_g^{3D}$ are compressed into BEV features $\mathbf{F}_g^{BEV}$ along the z-axis via max pooling. Then, they are combined with a learnable BEV query $\mathbf{Q}_{bev}$ to form hybrid features $v_{hybrid}$, warming up the BEV queries through a deformable self-attention mechanism:

$$\mathbf{Q}_{bev} = \text{DA}(\mathbf{Q}_{bev}, \mathbf{p}_{bev}, v_{hybrid})$$

The warmed-up query then retrieves information from the satellite features $\mathbf{F}_s^{2D}$ via deformable cross-attention.

**Design Motivation**: Localization noise in satellite images and top-down occlusions cause feature-level inconsistencies. Warming up BEV queries through ground-view information allows the offset layer to predict more appropriate sampling offsets in cross-attention, achieving more precise satellite feature extraction.

### Key Design 2: Adaptive Dual-Path Fusion Module — Balancing Satellite and Ground Feature Contributions

**Function**: Dynamically predict channel-domain and spatial-domain weights to adaptively balance the contributions of both perspectives across different regions and scales.

**Mechanism**: Channel and spatial attention paths are designed. The channel path computes channel weights $\mathbf{W}_c \in \mathbb{R}^{D}$ in the 3D space, and the spatial path computes spatial weights $\mathbf{W}_s \in \mathbb{R}^{H \times W}$ in the BEV space. Both are combined with the MLP output to obtain the final fusion weights:

$$\mathbf{W}_a = \text{MLP}(\mathbf{F}'_c) \oplus C(\mathbf{F}'_c) \oplus S(\mathbf{F}'_c)$$

The fused features are formulated as $\mathbf{F}_f = \mathbf{W}_a \cdot \mathbf{F}'_g + (1 - \mathbf{W}_a) \cdot \mathbf{F}'_s$.

**Design Motivation**: The satellite perspective excels at large-scale scene layouts (roads, buildings), while the ground perspective excels at details of small and dynamic objects. Adaptive weights allow the network to flexibly select the optimal information source across different regions and semantic categories. A probability network is also introduced to identify valuable voxels to enhance learning efficiency.

### Key Design 3: Uncertainty-Guided Feature Refinement — Focusing on High-Uncertainty Regions

**Function**: Perform fine-grained refinement only on high-uncertainty voxels to improve efficiency.

**Mechanism**: The fused features are first projected into coarse semantic predictions $\mathbf{L}_{coarse}$. The entropy of each voxel is calculated, and the top-k high-entropy voxels are selected to resample features from the ground features $\mathbf{F}_g^{2D}$ via deformable cross-attention.

**Design Motivation**: Avoid dense refinement operations on all voxels, thereby concentrating computational resources on the most critical areas.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem} + \mathcal{L}_{ce} + \lambda_{bev} \mathcal{L}_{bev} + \lambda_{co} \mathcal{L}_{co}$$

Including weighted cross-entropy loss, scene-class relation loss (geometric and semantic), BEV auxiliary loss, and coarse prediction loss.

## Key Experimental Results

### Main Results: SemanticKITTI Semantic Scene Completion

| Method | IoU (SC) | mIoU (SSC) | Input |
|------|----------|------------|------|
| MonoScene | 34.16 | 6.06 | Monocular |
| TPVFormer | 34.25 | 11.26 | Multi-camera |
| VoxFormer | 44.15 | 12.35 | Monocular + Depth |
| SurroundOcc | 34.72 | 11.86 | Multi-camera |
| **SGFormer** | **45.67** | **13.72** | Monocular + Satellite |

### SSCBench-KITTI-360 Dataset

| Method | IoU | mIoU |
|------|-----|------|
| VoxFormer | 42.95 | 12.20 |
| OccFormer | 40.83 | 13.46 |
| **SGFormer** | **44.32** | **14.58** |

### Ablation Study

| Setting | IoU | mIoU |
|------|-----|------|
| Ground-branch only | 44.15 | 12.35 |
| + Satellite branch (w/o correction) | 44.78 | 13.01 |
| + Ground-guided correction | 45.12 | 13.39 |
| + Adaptive fusion | **45.67** | **13.72** |

### Key Findings

- The introduction of the satellite branch improves IoU by approximately **1.5 points** and mIoU by approximately **1.4 points**.
- The ground-guided correction strategy contributes **0.38 points** to mIoU, demonstrating the critical importance of the alignment issue.
- Satellite imagery provides the most significant improvement in large object categories (roads, buildings), while showing limited improvement on small objects.
- Adaptive fusion performs better than simple concatenation or fixed-weight fusion.

## Highlights & Insights

1. **First to introduce satellite imagery into the SSC task**, opening up an entirely new research direction where the two orthogonal views (satellite and ground) naturally complement each other.
2. **Ground-guided satellite feature correction** is elegantly designed, leveraging existing ground information to warm up BEV queries to mitigate the alignment issue.
3. **Adaptive dual-path fusion** weighs the two perspectives across both channel and spatial dimensions, offering greater flexibility than fixed fusion strategies.
4. Satellite imagery serves as a low-cost, lightweight, and widely available complementary information source, carrying high practical deployment value.

## Limitations & Future Work

- Satellite imagery is typically pre-collected and cannot reflect dynamic scene changes (e.g., temporarily parked vehicles).
- The resolution of satellite imagery is limited, which offers little help for small objects (e.g., traffic signs, pedestrians).
- The satellite image alignment issue caused by GPS localization noise still needs further resolution.
- Validation is only conducted on the KITTI series datasets; the generalization to larger-scale and more diverse scenes remains to be tested.
- Future work could explore temporal update mechanisms for satellite imagery.

## Related Work & Insights

- **VoxFormer**: An SSC method that uses depth to initialize sparse proposals, upon which the SGFormer ground branch is based.
- **BEVFormer**: A framework that leverages Transformers to aggregate multi-view information into the BEV space.
- **SG-BEV / SNAP**: Pioneering works that introduce satellite images into BEV segmentation and 2D map construction.

## Rating

⭐⭐⭐⭐ — Pioneering work that introduces satellite imagery to the SSC task for the first time, characterized by a clear problem definition and a reasonable technical pipeline. The dual-branch design and adaptive fusion strategy effectively mitigate the challenges of cross-view fusion. Although the performance gains are modest (mIoU ~1.4), it opens up a promising new direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semantic-Adaptive Diffusion for Dynamic Spatiotemporal Fusion](../../CVPR2026/remote_sensing/semantic-adaptive_diffusion_for_dynamic_spatiotemporal_fusion.md)
- [\[CVPR 2025\] Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes](dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)
- [\[CVPR 2025\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[CVPR 2025\] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)

</div>

<!-- RELATED:END -->
