---
title: >-
  [Paper Note] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots
description: >-
  [CVPR2026][Autonomous Driving][Panoramic occupancy prediction] This paper introduces PanoMMOcc, the first panoramic multimodal semantic occupancy prediction dataset for quadruped robots, along with the VoxelHound framework. By incorporating a Vertical Jitter Compensation (VJC) module and a Multimodal Information Prompt Fusion (MIPF) module, VoxelHound achieves 23.34% mIoU under a four-modality setup (panoramic RGB + thermal + polarization + LiDAR), surpassing existing methods by +4.16%.
tags:
  - CVPR2026
  - Autonomous Driving
  - Panoramic occupancy prediction
  - multimodal fusion
  - quadruped robots
  - semantic occupancy
  - BEV perception
date: 2026-05-08
content_hash: 18cfef7a638f421b
---

# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**Conference**: CVPR2026
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)
**Code**: [PanoMMOcc](https://github.com/) (coming soon)
**Area**: Autonomous Driving
**Keywords**: Panoramic occupancy prediction, multimodal fusion, quadruped robots, semantic occupancy, BEV perception

## TL;DR

This paper introduces PanoMMOcc, the first panoramic multimodal semantic occupancy prediction dataset for quadruped robots, along with the VoxelHound framework. By incorporating a Vertical Jitter Compensation (VJC) module and a Multimodal Information Prompt Fusion (MIPF) module, VoxelHound achieves 23.34% mIoU under a four-modality setup (panoramic RGB + thermal + polarization + LiDAR), surpassing existing methods by +4.16%.

## Background & Motivation

1. **Panoramic perception requirements**: Panoramic images provide 360° blind-spot-free visual coverage, which is critical for mobile agents in dynamic unstructured environments. However, existing occupancy prediction methods are primarily designed for multi-camera narrow-FoV setups in autonomous driving.
2. **Challenges of quadruped platforms**: Compared to wheeled platforms, quadruped robots inherently suffer from low viewpoints, frequent self-occlusion, and strong ego-motion caused by gait dynamics—challenges that existing methods do not address.
3. **Limitations of RGB-only sensing**: Relying solely on the RGB modality yields insufficient robustness under illumination changes, low-texture regions, and long-range perception scenarios, necessitating multimodal sensor fusion.
4. **Dataset gap**: Existing panoramic datasets focus on 2D visual tasks and lack 3D occupancy annotations; existing occupancy benchmarks target autonomous driving and cover neither panoramic imaging nor quadruped platforms.
5. **Insufficient fusion strategies**: Common multimodal fusion approaches (simple concatenation or addition) treat heterogeneous sensor contributions indiscriminately, diluting geometric consistency and introducing cross-modal interference.
6. **Gait-induced jitter**: Quadruped locomotion causes vertical body oscillations, leading to spatial misalignment of captured image features and degrading the stability of BEV transformation.

## Method

### Overall Architecture: VoxelHound

The model takes four modalities as input: panoramic RGB image $\mathcal{I}^{pal}$, thermal image $\mathcal{I}^{th}$, polarization image $\mathcal{I}^{pol}$, and LiDAR point cloud $\mathcal{P}$. Each modality is processed by an independent encoder for feature extraction, projected into a unified BEV space for fusion, and the final output is a 3D semantic occupancy prediction $\mathbf{O} \in \mathbb{R}^{X \times Y \times Z}$.

### Multimodal Fusion Network

- **Camera branch**: Each of the three image modalities is processed by a ResNet-18 backbone to extract multi-scale features, aggregated via FPN, and then transformed to BEV features $\mathbf{F}_c^m \in \mathbb{R}^{C_m \times H \times W}$ via a 2D-to-BEV view transformation.
- **LiDAR branch**: The point cloud is voxelized and processed by sparse 3D convolutions (stride=8) to extract hierarchical geometric features, which are collapsed into BEV features $\mathbf{F}_l \in \mathbb{R}^{C_l \times H \times W}$.
- **Fusion branch**: The fused features are enhanced by a SECOND-FPN BEV encoder for multi-scale spatial representation. The occupancy head reshapes BEV channels into vertical bins, producing a 64×64×16 voxel prediction.

### Vertical Jitter Compensation Module (VJC)

Inserted between the image encoder and the BEV view transformation:
1. Mean pooling along the width dimension to obtain vertical structure $\mathbf{F}_v \in \mathbb{R}^{C \times H}$.
2. Two-layer Conv1D + ReLU to encode vertical features.
3. Adaptive average pooling + linear layer to predict the global vertical offset $\Delta h$.
4. Normalization into grid coordinate offsets to construct a displacement sampling grid.
5. Bilinear grid sampling to obtain the compensated feature $\mathbf{F}_{comp}$.

### Multimodal Information Prompt Fusion Module (MIPF)

Adopts an asymmetric fusion principle of "geometry-dominant, semantics-supplementary":
1. Each modality is projected to a shared embedding space via 1×1 convolution.
2. Global average pooling + MLP over image modalities generates compact semantic prompts $\mathbf{p}_m$.
3. LiDAR BEV features serve as Query, while modality prompts serve as Key/Value in prompt attention.
4. Residual modulation: $\mathbf{F}_f = \tilde{\mathbf{F}}_l + \sigma(\gamma(\mathbf{F}_{attn})) \odot \tilde{\mathbf{F}}_l$, ensuring geometric structure remains the primary representational basis.

### Loss & Training

$$\mathcal{L}_{occ} = \mathcal{L}_{ce} + \mathcal{L}_{ls} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem}$$

Comprising cross-entropy loss, Lovász-Softmax loss, and geometric/semantic affinity losses.

## Key Experimental Results

### Dataset: PanoMMOcc

- Unitree Go2 quadruped robot equipped with a panoramic camera (360°×70° FoV, 2048×2048), MID360 LiDAR, thermal camera (640×512), and polarization camera (1224×1024).
- 54 sequences @10Hz, 40 seconds per sequence, totaling 21,600 frames; 42 sequences annotated with 12 semantic classes.
- Voxel space: 64×64×16, resolution 0.4m, spatial range [-12.8, 12.8]m (xy) × [-2.4, 4.0]m (z).
- 30 sequences for training / 12 for testing, covering six scene types: campus, urban, residential, green space, rural, and forest.

### Main Results (mIoU %)

| Method | Modality | mIoU |
|--------|----------|------|
| MonoScene | C | 8.94 |
| EFFOcc-C | C | 4.47 |
| EFFOcc-L | L | 18.77 |
| EFFOcc-T | C+L | 19.18 |
| C-CONet | C | 3.79 |
| M-CONet | C+L | 4.68 |
| **VoxelHound** | C | 5.79 |
| **VoxelHound** | C+T+P | 6.14 |
| **VoxelHound** | C+L | 22.87 |
| **VoxelHound** | **C+L+T+P** | **23.34** |

- Full-modality VoxelHound outperforms the strongest competing method EFFOcc-T by **+4.16% mIoU**.
- Compared to camera-only MonoScene: **+14.40% mIoU**.
- Thermal and polarization modalities yield notable nighttime improvements: nighttime mIoU increases from 3.52% (C) to 4.07% (C+T+P).

### Ablation Study

| VJC | MIPF | mIoU |
|-----|------|------|
| ✗ | ✗ | 22.74 |
| ✓ | ✗ | 22.92 |
| ✗ | ✓ | 23.14 |
| ✓ | ✓ | **23.34** |

- VJC contributes +0.18%, MIPF contributes +0.40%, and their combination yields an additional +0.20%.
- Optimal hidden channel dimension for VJC: $C_{hd}=64$; optimal MIPF settings: $C_{pd}=8$, $C_{nh}=8$.

## Highlights & Insights

1. **Pioneering contribution**: The first panoramic multimodal occupancy prediction dataset and framework for quadruped robots, filling the gap at the intersection of panoramic occupancy and legged platforms.
2. **Elegant VJC design**: The module estimates vertical offsets via lightweight 1D convolutions for grid-sampling-based compensation, with minimal parameter overhead (+0.04M).
3. **Asymmetric MIPF fusion**: Image modalities are compressed into compact prompts to avoid dense spatial cross-attention; the "geometry-dominant, semantics-supplementary" design is well aligned with sensor characteristics.
4. **Four-modality sensing**: Integrating panoramic RGB, thermal, polarization, and LiDAR—four complementary modalities—with polarization imaging introduced to occupancy prediction for the first time.

## Limitations & Future Work

1. **Low absolute performance**: The best mIoU of 23.34% remains limited, with some categories completely undetected (bicycle=0.00%, pedestrian=0.00%), indicating severe deficiency in small-object perception.
2. **Marginal VJC gain**: VJC alone contributes only +0.18%, suggesting limited compensation for gait-induced jitter; more sophisticated temporal modeling may be required.
3. **Small dataset scale**: The 21.6k frames are considerably fewer than driving datasets such as nuScenes, and generalization capability remains to be validated.
4. **Evaluation on proprietary dataset only**: VoxelHound is not validated on existing public occupancy benchmarks, leaving cross-dataset generalization unknown.
5. **Nighttime degradation with full modality**: Nighttime C+L+T+P (18.68%) underperforms C+L (19.17%), suggesting that thermal and polarization modalities introduce noise under certain conditions.

## Related Work & Insights

- **vs. MonoScene / EFFOcc / CONet**: These methods are designed for autonomous driving with pinhole multi-camera setups, and are ill-suited for panoramic imaging on quadruped platforms. VoxelHound substantially outperforms them in the quadruped scenario under full-modality fusion.
- **vs. QuadOcc**: QuadOcc is also a panoramic occupancy dataset for quadruped platforms, but is limited to RGB-only, 6 semantic classes, and 64×64×8 voxels. PanoMMOcc extends to four modalities, 12 classes, and 64×64×16 voxels.
- **Existing multimodal fusion**: Most approaches apply simple concatenation or symmetric fusion of Camera+LiDAR. The asymmetric prompt attention in MIPF more effectively exploits cross-modal complementarity.
- **Panoramic perception**: Prior work in this domain primarily addresses 2D semantic segmentation or BEV mapping. This paper is the first to extend panoramic vision to 3D occupancy prediction.

## Rating

- Novelty: ⭐⭐⭐⭐ (First quadruped panoramic multimodal occupancy dataset and framework; VJC and MIPF designs are well-motivated)
- Experimental Thoroughness: ⭐⭐⭐ (Complete ablations, but low absolute performance and absence of cross-dataset experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with rich figures and tables)
- Value: ⭐⭐⭐⭐ (Opens a new direction for panoramic occupancy on quadruped robots; dataset and benchmark hold long-term research value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[AAAI 2026\] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](../../AAAI2026/autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)

</div>

<!-- RELATED:END -->
