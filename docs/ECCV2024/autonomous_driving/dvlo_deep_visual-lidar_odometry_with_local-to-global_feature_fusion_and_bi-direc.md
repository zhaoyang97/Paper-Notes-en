---
title: >-
  [Paper Note] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment
description: >-
  [ECCV2024][Autonomous Driving][Visual-LiDAR Odometry] A clustering-based Local-to-Global fusion network, DVLO, is proposed to address the data structure inconsistency between vision and LiDAR through bi-directional structure alignment (image-to-pseudo-point-cloud + point-cloud-to-pseudo-image), achieving state-of-the-art (SOTA) performance on both the KITTI odometry and FlyingThings3D scene flow tasks.
tags:
  - "ECCV2024"
  - "Autonomous Driving"
  - "Visual-LiDAR Odometry"
  - "Multi-Modal Fusion"
  - "Clustering-based Fusion"
  - "Bi-Directional Structure Alignment"
  - "Scene Flow"
date: 2026-05-08
content_hash: eae08e73ffb69d52
---

# DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment

**Conference**: ECCV2024  
**arXiv**: [2403.18274](https://arxiv.org/abs/2403.18274)  
**Code**: [IRMVLab/DVLO](https://github.com/IRMVLab/DVLO)  
**Area**: Autonomous Driving  
**Keywords**: Visual-LiDAR Odometry, Multi-Modal Fusion, Clustering-based Fusion, Bi-Directional Structure Alignment, Scene Flow

## TL;DR

A clustering-based Local-to-Global fusion network, DVLO, is proposed to address the data structure inconsistency between vision and LiDAR through bi-directional structure alignment (image-to-pseudo-point-cloud + point-cloud-to-pseudo-image), achieving state-of-the-art (SOTA) performance on both the KITTI odometry and FlyingThings3D scene flow tasks.

## Background & Motivation

Visual-LiDAR odometry is a fundamental task in autonomous driving and SLAM, requiring the estimation of relative pose transformations from consecutive frames. Images provide fine-grained texture details, while point clouds offer rich geometric information, making them highly complementary. However, the core difficulty of visual-LiDAR fusion lies in the **inherent inconsistency of their data structures**:

- Image pixels are **regular and dense** 2D grids
- LiDAR point clouds are **unordered and sparse** 3D point sets

Limitations of prior work:

1. **CNN-based fusion**: The receptive field is limited by the kernel size, preventing the establishment of global correspondences.
2. **Attention-based fusion**: Although it enables global interaction, the quadratic computational complexity leads to excessive inference time.
3. **Single-level fusion**: Performs only global or local fusion, failing to preserve both fine-grained details and global context simultaneously.

## Core Problem

How to design an efficient multi-modal fusion strategy that both preserves **local fine-grained pixel-to-point correspondences** and enables **global information interaction**, while simultaneously resolving the data structure inconsistency between images and point clouds?

## Method

### Overall Architecture

DVLO comprises four core modules: hierarchical feature extraction, Local Fuser (local fusion), Global Fuser (global fusion), and iterative pose estimation.

### 1. Hierarchical Feature Extraction

**Point Cloud Feature Extraction**: The LiDAR point cloud is projected into a pseudo-image ($64 \times 1800$) via cylindrical projection, defined by the formula:

$$u = \arctan2(y/x) / \Delta\theta, \quad v = \arcsin(z / \sqrt{x^2+y^2+z^2}) / \Delta\phi$$

Each 2D position is filled with its corresponding raw 3D coordinates, converting the data into a pseudo-image structure while preserving 3D geometric information. A hierarchical convolutional network is then used to extract multi-scale point cloud features $F_P \in \mathbb{R}^{H_P \times W_P \times D}$.

**Image Feature Extraction**: The camera image (padded to $384 \times 1280$) is processed using a convolutional feature pyramid to extract multi-scale image features $F_I \in \mathbb{R}^{H_I \times W_I \times C}$.

### 2. Local Fuser: Clustering-based Local Fusion

This is the most core innovation of this work. Inspired by Context Clusters, the authors propose the first **clustering-based multi-modal fusion module**, which does not rely on CNNs or Transformers.

**Image-to-Point Structure Alignment**: Reshapes the image features $F_I$ into a pseudo-point set $F_{pp} \in \mathbb{R}^{M \times C}$ ($M = H_I \times W_I$), aligning the image data structure with that of the LiDAR point cloud.

**Pseudo-point Clustering**: LiDAR points are projected onto the image plane to obtain 2D coordinates serving as **cluster centers**. Center features $F_c$ are obtained via bilinear interpolation, and pseudo-points are assigned to the nearest center based on the **cosine similarity** between the center features and pseudo-point features, forming $N$ clusters.

**Local Feature Aggregation**: Within each cluster, pseudo-point features are dynamically aggregated based on similarity:

$$F_L^i = \frac{1}{X}\left(F_c^i + \sum_{j=1}^{k} \text{sigmoid}(\alpha s_{ij} + \beta) \cdot F_{pp}^j\right)$$

where $\alpha, \beta$ are learnable parameters, and $s_{ij}$ is the similarity score. The dimension of the locally fused features $F_L$ matches the original number of LiDAR points.

### 3. Global Fuser: Adaptive Global Fusion

Since local fusion has a limited receptive field, a global adaptive fusion mechanism is introduced.

**Point-to-Image Structure Alignment**: Convert the point cloud into a pseudo-image structure via cylindrical projection.

**Adaptive Fusion**: Adaptive weights $A_L, A_P$ are generated for the local fusion features $F_L$ and point cloud features $F_P$ respectively via MLP + Sigmoid, followed by weighted fusion:

$$F_G = \frac{A_P \odot F_P + A_L \odot F_L}{A_P + A_L}$$

### 4. Iterative Pose Estimation

An Attentive Cost Volume is used to correlate the globally fused features of two consecutive frames at the coarsest layer to generate embedding features $E$. Following weighting by a learnable mask, fully connected (FC) layers regress the rotation quaternion $q$ and translation vector $t$, which are then iteratively refined layer by layer.

### 5. Loss & Training

A multi-level supervised loss is used, with learnable scalars $k_x, k_q$ dynamically balancing the translation and rotation errors at each layer:

$$\mathcal{L}^l = \|t_{gt} - t^l\| \exp(-k_x) + k_x + \|q_{gt} - q^l\|_2 \exp(-k_q) + k_q$$

## Key Experimental Results

### KITTI Odometry

- **Training set**: Sequences 00-06; **Test set**: Sequences 07-10
- Average test set $t_{rel}$: **0.82%**, $r_{rel}$: **0.41°/100m**
- Compared with vision-only SOTA (Cho et al.): $t_{rel}$ decreased by 63.4%, $r_{rel}$ decreased by 43.8%
- Compared with LiDAR-only SOTA (EfficientLO): $t_{rel}$ decreased by 4.9%, and rotation error remained equivalent.
- Compared with multi-modal SOTA (H-VLO, which used more training data 00-08): $t_{rel}$ decreased by 47.0%
- Compared with the traditional multi-modal method (PL-LOAM): Full-sequence average $t_{rel}$ decreased by 28.7% (0.67 vs 0.94)

### FlyingThings3D Scene Flow

- EPE2D: **1.69** (CamLiRAFT: 1.73)
- EPE3D: **0.048** (CamLiRAFT: 0.049)
- Outperforms CamLiRAFT (which is specifically designed for scene flow) in both 2D and 3D metrics.

### Inference Efficiency

- Inference time: **98.5ms**, making it the only multi-modal method that satisfies the 10Hz LiDAR real-time requirement (< 100ms).
- Compared with Attention-based methods (183.76ms): Requires only about half the inference time.
- Comparable to CNN-based methods (87.24ms) but with significantly higher accuracy.

### Ablation Study

| Configuration | Average Test Set $t_{rel}$ | Average Test Set $r_{rel}$ |
|------|---------------------|---------------------|
| Global Fuser Only | 0.93 | 0.47 |
| Local Fuser Only | 1.00 | 0.50 |
| **Full DVLO** | **0.82** | **0.41** |

## Highlights & Insights

1. **First clustering-based multi-modal fusion method**: Neither CNN nor Transformer, presenting a paradigm shift for fusion.
2. **Bi-directional structure alignment**: Simultaneously performs image-to-pseudo-point and point-to-pseudo-image structural conversions to maximize cross-modal complementarity.
3. **Local-to-Global hierarchical design**: Preserves fine-grained pixel-to-point correspondences locally while enabling large-receptive-field information interaction globally.
4. **High efficiency and real-time capability**: The 98.5ms inference time meets the 10Hz real-time constraint, outperforming all other multi-modal methods.
5. **Strong generalization**: The fusion module can be directly transferred to scene flow estimation tasks and surpasses dedicated SOTAs.

## Limitations & Future Work

1. **Evaluation limited to KITTI**: Lacks evaluation on larger-scale datasets such as nuScenes and Waymo.
2. **Monocular camera limitation**: Uses only a monocular left camera, leaving the depth-information gains of stereo vision unexplored.
3. **Lack of a mapping backend**: Currently acts only as an odometry frontend without integrated loop closure or a complete SLAM system.
4. **Fixed clustering strategy**: Relies on nearest-neighbor assignment without exploring the potential of soft assignment or differentiable clustering.
5. **Information loss in cylindrical projection**: The projection resolution of distant points decreases, potentially limiting performance in long-range scenes.

## Related Work & Insights

| Method | Modality | Fusion Strategy | KITTI 07-10 $t_{rel}$ | Inference Time |
|------|------|---------|----------------------|---------|
| EfficientLO | LiDAR | No Fusion | 0.86| — |
| H-VLO | Vision+LiDAR | CNN Fusion | 1.36 | — |
| TransLO | LiDAR | Transformer | 0.99 | — |
| PL-LOAM | Vision+LiDAR | Traditional | — | 200ms |
| **DVLO (Ours)** | **Vision+LiDAR** | **Clustering Local-to-Global** | **0.82** | **98.5ms** |

The core advantage of DVLO lies in achieving both leading accuracy and the fastest inference speed, balancing performance and efficiency effectively.

## Inspirations & Connections

1. **Clustering as a Fusion Primitive**: Context Clusters has shown that clustering can serve as a visual backbone. This work extends it to multi-modal fusion, a paradigm that deserves further exploration in more multi-modal tasks (e.g., 3D detection, BEV perception).
2. **Bi-directional Structure Alignment Concept**: When dealing with heterogeneous data fusion, mutual bi-directional alignment is more effective than unidirectional projection. This concept can be extended to fusion scenarios like radar-camera and thermal-RGB.
3. **Hierarchical Fusion Strategy**: The sequence of local-then-global fusion can be applied to other tasks that require balancing fine-grained details with global context.

## Rating
- Novelty: ⭐⭐⭐⭐ (First clustering-based multi-modal fusion; the bi-directional structure alignment is highly innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (KITTI odometry + FlyingThings3D scene flow + detailed ablation studies, though lacks evaluation on additional datasets)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete mathematical formulations, and intuitive illustrations)
- Value: ⭐⭐⭐⭐ (Strong real-time performance and excellent generalization, offering great reference value for multi-modal fusion research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamVLO: Streaming Visual-LiDAR Odometry with Cumulative Drift Compensation](../../CVPR2026/autonomous_driving/streamvlo_streaming_visual-lidar_odometry_with_cumulative_drift_compensation.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[CVPR 2026\] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization](../../CVPR2026/autonomous_driving/leader_lidar_relocalization.md)
- [\[CVPR 2025\] ZeroVO: Visual Odometry with Minimal Assumptions](../../CVPR2025/autonomous_driving/zerovo_visual_odometry_with_minimal_assumptions.md)
- [\[ICCV 2025\] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](../../ICCV2025/autonomous_driving/mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)

</div>

<!-- RELATED:END -->
