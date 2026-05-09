---
title: >-
  [Paper Note] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization
description: >-
  [CVPR 2026][Autonomous Driving][Scene categorization] This paper proposes a method for outdoor scene categorization using LiDAR panoramic depth maps and reflectance maps as CNN inputs. The authors construct the large-scale MPO outdoor 3D dataset (6 scene categories, 34,200 frames), and address the ring topology of panoramic images via Horizontal Circular Convolution (HCC) and Row-Wise Max Pooling (RWMP). The proposed multimodal fusion approach achieves 97.47% classification accuracy.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Scene categorization
  - LiDAR panorama
  - CNN
  - multimodal fusion
  - outdoor environment understanding
date: 2026-05-08
content_hash: ff6f738276db5e28
---

# Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization

**Conference**: CVPR 2026
**arXiv**: [2603.12663](https://arxiv.org/abs/2603.12663)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Scene categorization, LiDAR panorama, CNN, multimodal fusion, outdoor environment understanding

## TL;DR

This paper proposes a method for outdoor scene categorization using LiDAR panoramic depth maps and reflectance maps as CNN inputs. The authors construct the large-scale MPO outdoor 3D dataset (6 scene categories, 34,200 frames), and address the ring topology of panoramic images via Horizontal Circular Convolution (HCC) and Row-Wise Max Pooling (RWMP). The proposed multimodal fusion approach achieves 97.47% classification accuracy.

## Background & Motivation

Autonomous robots and vehicles must understand the type of environment they operate in (coastline, forest, parking lot, residential area, urban district, etc.) to enable context-aware decision-making, such as automatically reducing speed in residential zones. Conventional approaches rely on GPS coordinates to retrieve semantic labels from pre-built maps; however, such maps suffer from update latency and spatio-temporal discrepancies with the actual scene.

RGB cameras are widely used for scene recognition, yet outdoor environments pose a fundamental challenge: **dynamic illumination variation** (day/night/overcast/rain) renders visual features unreliable. 3D LiDAR provides an illumination-robust alternative by simultaneously measuring **depth (geometric information)** and **reflectance (photometric information)** — two naturally aligned modalities that are unaffected by ambient lighting.

Nevertheless, existing LiDAR-based research predominantly focuses on localization and mapping (e.g., SLAM), and work on **semantic scene categorization** remains scarce. The KITTI dataset includes four scene-level annotations but is primarily designed for object detection and tracking, making it unsuitable for scene categorization. Furthermore, how to effectively exploit the **ring topology** of LiDAR panoramas (360° continuous, with left and right boundaries connected) is an open problem — standard CNN zero-padding disrupts this continuity at the boundaries.

The starting point of this paper is: **project 3D point clouds into panoramic depth and reflectance maps, design CNNs adapted to the ring topology, and fuse both modalities for outdoor scene categorization**.

## Method

### Overall Architecture

1. LiDAR 3D point clouds are converted to 2D panoramic depth maps and reflectance maps via cylindrical projection.
2. A single-modality CNN adapted to panoramic image properties is designed, based on a modified VGG-11.
3. Four multimodal fusion strategies are explored: Softmax Average, Adaptive Fusion, Early Fusion, and Late Fusion.

### Key Designs

1. **Horizontal Circular Convolution (HCC)**: The horizontal direction of a panoramic image is 360°-continuous (left and right boundaries are connected), whereas standard convolution applies zero-padding, introducing spurious edges at the boundaries. HCC replaces horizontal padding with **circular padding** — convolution kernels that overflow the left boundary wrap around from the right, and vice versa. This preserves the ring-topology consistency of the panorama during both forward computation and backward gradient propagation. The design motivation is to eliminate the corrupting effect of zero-padding on boundary features.

2. **Row-Wise Max Pooling (RWMP)**: Due to yaw motion during vehicle driving and the LiDAR mounting angle, scene content undergoes primarily **horizontal translation** in the panoramic image. RWMP takes the maximum value across each row of every feature map after the last convolutional layer and before the fully connected layers, yielding a row-dimensional vector. This effectively compresses the horizontal dimension with translation invariance:

    $\text{RWMP}(f_{c,h,:}) = \max_{w} f_{c,h,w}$

   The motivation is to make the CNN invariant to horizontal displacements caused by vehicle yaw.

3. **Multimodal Fusion Strategies**:

    - **Softmax Average**: Depth and reflectance models are trained independently; at test time, the softmax probabilities of both models are averaged arithmetically.
    - **Adaptive Fusion**: A gating network $g(\cdot)$ learns adaptive weights $(w_d, w_r)$ from intermediate features of both models and produces a weighted fusion of their probability outputs.
    - **Early Fusion**: Depth and reflectance are concatenated into a 2-channel input, enabling joint learning of cross-modal features at the pixel level from the first convolutional layer onward.
    - **Late Fusion**: Two independent convolutional streams extract features separately and are merged at the fully connected layer, learning cross-modal interactions at a higher semantic level.

### Loss & Training

- Cross-entropy loss with SGD optimization (lr=1e-4, momentum=0.9)
- L2 regularization (weight decay=5e-4) and 50% Dropout
- Data augmentation: horizontal flipping and random circular horizontal shift (equivalent to yaw rotation of the panorama)
- Evaluation via 10-fold cross-validation
- Training conducted on an NVIDIA GTX Titan X GPU

## Key Experimental Results

### MPO Dataset Statistics

| Dataset | LiDAR Model | Resolution | Categories | Total Frames | Data Size |
|--------|-----------|--------|--------|--------|--------|
| Sparse MPO | Velodyne HDL-32e | 2166×32 | 6 classes × 10 areas | 34,200 | 59.23 GB |
| Dense MPO | FARO Focus 3D S120 | 5140×1757 | 6 classes × 7 areas | 650 | 242 GB |

### Main Results

| Method | Input | Overall Accuracy (%) |
|------|------|--------------|
| LBP+SVM | Depth | 92.00 |
| ResNet-20 | Depth | 95.66 |
| VGG (baseline) | Depth | 97.18 |
| VGG+RWMP+HCC | Reflectance | 95.92 |
| **Late Fusion** | **Depth + Reflectance** | **97.47** |
| Early Fusion | Depth + Reflectance | 97.27 |

### Ablation Study

| Configuration | Depth Acc. (%) | Reflectance Acc. (%) | Notes |
|------|-------------|----------------|------|
| VGG baseline | 97.18 | 94.75 | No modification |
| VGG+HCC | 96.89 | 95.45 | Circular convolution alone yields marginal gain on depth but improves reflectance |
| VGG+RWMP | 97.11 | 95.74 | Horizontal invariance is effective |
| VGG+RWMP+HCC | 96.92 | **95.92** | Best for reflectance; indicates reflectance is more sensitive to horizontal shift |

### Key Findings

- **The depth modality alone is highly discriminative**: baseline VGG on depth achieves 97.18%, indicating that geometric structure is a powerful cue for scene categorization.
- **Reflectance is complementary**: reflectance outperforms depth on the Forest and ParkingOut categories; multimodal fusion improves overall performance.
- **HCC+RWMP primarily benefits the reflectance modality**: depth maps have clear geometric structure and are less sensitive to horizontal translation, whereas reflectance maps contain finer texture patterns that benefit more from explicit rotation-invariance design.
- **Late Fusion achieves the best performance** at 97.47%, since the optimal single-modality architectures differ between depth and reflectance; Late Fusion allows each branch to retain its own optimal structure.
- Coast and Forest are the most frequently confused categories, as coastal vegetation regions are visually similar to forests.

## Highlights & Insights

- The **ring topology of panoramic images** is elegantly exploited: HCC and RWMP are precisely tailored to the physical properties of 360° panoramas.
- The paper **introduces the first large-scale LiDAR dataset specifically designed for outdoor scene categorization** (MPO), filling the gap left by datasets such as KITTI.
- It is demonstrated that LiDAR reflectance can serve as a substitute for RGB imagery to provide photometric information while remaining entirely robust to illumination changes.
- Grad-CAM visualizations reveal interesting patterns: the model relies on horizontally extended structural features (e.g., building contour lines) to distinguish Urban from Residential scenes.

## Limitations & Future Work

- Only VGG-11 is evaluated; more modern architectures (e.g., ViT, PointNet++) are not explored, leaving potentially significant room for improvement.
- The Sparse MPO resolution is very low (2166×32), discarding a large amount of fine-grained information.
- Only 6 coarse-grained categories are defined; practical applications require finer-grained scene categorization (e.g., intersections, roundabouts, construction zones).
- No controlled comparison with RGB-based methods under challenging conditions (e.g., night or rainy scenes with LiDAR vs. RGB) is provided.
- Data collection is limited to Fukuoka, Japan; generalization to other geographic regions and climate conditions remains unverified.

## Related Work & Insights

- This work contrasts with RGB-based scene categorization methods (e.g., Places/Places2 by Zhou et al.), as the LiDAR-based approach inherently addresses illumination variation.
- RWMP is adapted from the panoramic rotation-invariance work of Shi et al., and the underlying idea is transferable to other ring-structured data.
- The Adaptive Fusion strategy (gating network from Mees et al.) is applicable to broader multi-sensor fusion tasks.

## Rating

- **Novelty**: ⭐⭐⭐ The core idea (LiDAR panorama + scene categorization) is original, though the CNN design is relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines, detailed ablations, robustness analysis, and visualizations are all well covered.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and figures are informative, though some notation could be made more concise.
- **Value**: ⭐⭐⭐ Practical value for autonomous driving scene understanding is clear, but the methodology is somewhat conventional for the deep learning era.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)

</div>

<!-- RELATED:END -->
