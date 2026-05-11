---
title: >-
  [Paper Note] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization
description: >-
  [CVPR 2026][Autonomous Driving][Outdoor scene categorization] This paper uses panoramic depth and reflectance images derived from 3D LiDAR point clouds as CNN inputs, constructs a large-scale outdoor scene categorization dataset (MPO), and proposes two architectural improvements—Horizontal Circular Convolution (HCC) and Row-Wise Max Pooling (RWMP)—to achieve high-accuracy classification (up to 97.87%) across six outdoor scene categories, substantially outperforming traditional handcrafted feature methods.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Outdoor scene categorization
  - LiDAR panoramic image
  - multimodal fusion
  - CNN
  - "depth & reflectance"
date: 2026-05-08
content_hash: 555e4a2a81709402
---

# Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization

**Conference**: CVPR 2026
**arXiv**: [2603.12663](https://arxiv.org/abs/2603.12663)
**Code**: N/A
**Area**: Autonomous Driving / Scene Understanding
**Keywords**: Outdoor scene categorization, LiDAR panoramic image, multimodal fusion, CNN, depth & reflectance

## TL;DR
This paper uses panoramic depth and reflectance images derived from 3D LiDAR point clouds as CNN inputs, constructs a large-scale outdoor scene categorization dataset (MPO), and proposes two architectural improvements—Horizontal Circular Convolution (HCC) and Row-Wise Max Pooling (RWMP)—to achieve high-accuracy classification (up to 97.87%) across six outdoor scene categories, substantially outperforming traditional handcrafted feature methods.

## Background & Motivation
1. **Background**: Autonomous robots and vehicles require environment understanding for navigation and decision-making. Place categorization—determining the semantic category of the agent's current location—is a critical component of this capability.
2. **Limitations of Prior Work**: Conventional approaches rely primarily on RGB cameras, which suffer from severe illumination variations between day and night and from occlusions caused by pedestrians and vehicles, leading to unstable visual features. Existing 3D datasets (e.g., KITTI) target localization and mapping tasks and offer limited scene category annotations (only 4 classes).
3. **Key Challenge**: RGB images are sensitive to illumination changes, whereas depth and reflectance data from LiDAR are inherently illumination-robust. Nevertheless, large-scale outdoor scene categorization datasets and dedicated CNN architectures tailored for LiDAR data remain absent.
4. **Goal**: (1) Construct a large-scale multimodal LiDAR outdoor scene categorization dataset; (2) design a CNN architecture suited to panoramic LiDAR images; (3) explore optimal fusion strategies for the depth and reflectance modalities.
5. **Key Insight**: The authors observe that LiDAR panoramic images have a cyclic horizontal structure (the left and right boundaries are physically contiguous). Standard zero-padding breaks this continuity at the borders, and vehicle yaw motion causes scene features to shift substantially along the horizontal axis.
6. **Core Idea**: Horizontal circular convolution preserves the cyclic structure of panoramic images; row-wise max pooling provides rotation invariance; and depth–reflectance multimodal fusion further boosts categorization accuracy.

## Method

### Overall Architecture
The input is a 3D LiDAR point cloud, which is projected cylindrically into 2D panoramic depth and reflectance images (resolution $384 \times 32$). These images are fed individually or jointly into a CNN to classify six outdoor scene categories: coast, forest, indoor parking, outdoor parking, residential area, and urban area.

### Key Designs

1. **MPO Dataset Construction**:

    - **Function**: Provides a large-scale multimodal LiDAR benchmark for outdoor place categorization.
    - **Mechanism**: A Velodyne HDL-32e LiDAR mounted on a vehicle roof is driven at 30–50 km/h across 10 areas in Fukuoka City, covering six scene categories, yielding 34,200 panoramic scans. Each scan contains both a depth image and a reflectance image, totaling 59.23 GB. A high-resolution Dense MPO variant is also constructed using a FARO Focus 3D S120 scanner (650 scans).
    - **Design Motivation**: Existing datasets either use only RGB imagery (e.g., Places) or provide limited category annotations for 3D data (KITTI has only 4 classes). A dedicated large-scale LiDAR scene categorization dataset is lacking.

2. **Horizontal Circular Convolution (HCC)**:

    - **Function**: Preserves the cyclic structure of panoramic images during convolution.
    - **Mechanism**: Zero-padding in standard convolution is replaced by circular padding: pixels from the right boundary of the image are used to fill the left-side padding region, and vice versa. This allows the convolutional kernel to extract correct neighborhood features at image borders. Both forward computation and backward gradient propagation follow this circular data flow.
    - **Design Motivation**: A 360° panoramic image is inherently cyclic; its left and right boundaries are spatially contiguous. Zero-padding introduces discontinuities at the borders, degrading feature extraction in those regions.

3. **Row-Wise Max Pooling (RWMP)**:

    - **Function**: Achieves translation invariance along the horizontal direction.
    - **Mechanism**: A RWMP layer is inserted between the final convolutional layer and the first fully connected layer. It takes the maximum value across each row of every feature map, producing a column vector. As a result, the output is identical regardless of horizontal rotation, as long as the same visual concepts appear at the same row (i.e., the same elevation angle).
    - **Design Motivation**: Vehicle yaw motion and LiDAR mounting angle cause visual concepts in panoramic images to shift substantially in the horizontal direction, a variation to which standard CNNs are not invariant.

### Multimodal Fusion Strategies

Four strategies for fusing depth and reflectance images are investigated:

- **Softmax Average**: Two independent unimodal models are trained separately; at test time, their softmax probability vectors are averaged and the class with the highest probability is selected. This strategy achieves the best performance (97.87%).
- **Adaptive Fusion**: A gating network is added on top of Softmax Average to adaptively estimate per-model weights from intermediate features. Performance is slightly lower due to insufficient training samples for the gating network.
- **Early Fusion**: Depth and reflectance images are concatenated into a two-channel input and trained end-to-end. Performance is relatively poor, attributed to gradient vanishing.
- **Late Fusion**: Two convolutional streams extract features independently and are merged at the fully connected layer. Performance gains are limited.

### Loss & Training
Cross-entropy loss is used with an SGD optimizer (learning rate $10^{-4}$, momentum 0.9), batch size 64, $L_2$ regularization (coefficient $5 \times 10^{-4}$), and 50% dropout. Early stopping is applied (training halts if validation loss does not decrease for 10 consecutive epochs). Data augmentation includes horizontal flipping and random horizontal circular shifts.

## Key Experimental Results

### Main Results (Single-Modality Classification Accuracy, %)

| Modality | Method | Coast | Forest | ParkingIn | ParkingOut | Residential | Urban | Overall |
|----------|--------|-------|--------|-----------|------------|-------------|-------|---------|
| Depth | LBP+SVM | 84.25 | 94.93 | 96.41 | 86.86 | 94.58 | 92.71 | 92.00 |
| Depth | VGG (baseline) | 92.73 | 97.26 | 99.94 | 94.23 | 98.35 | 99.20 | **97.18** |
| Reflectance | VGG+RWMP+HCC | 91.83 | 98.20 | 91.45 | 95.16 | 97.99 | 98.27 | **95.92** |
| Multimodal | Softmax Average | — | — | — | — | — | — | **97.87** |

### Ablation Study (Effect of HCC and RWMP)

| Configuration | Depth Acc. | Reflectance Acc. | Notes |
|---------------|-----------|-----------------|-------|
| VGG baseline | 97.18% | 94.75% | Baseline |
| VGG + RWMP | 97.11% | 95.74% | Row pooling only |
| VGG + HCC | 96.89% | 95.45% | Circular convolution only |
| VGG + RWMP + HCC | 96.92% | **95.92%** | Combined |

### Key Findings
- The depth modality (97.18%) generally outperforms the reflectance modality (95.92%), though reflectance holds an advantage on the Forest and ParkingOut categories.
- HCC and RWMP yield more substantial improvements for the reflectance modality (+1.17%) but provide limited or marginally negative gains for depth, suggesting that depth features are inherently less sensitive to horizontal shifts.
- Softmax Average is both the simplest and most effective fusion strategy, improving over the best single modality by 0.69%.
- Grad-CAM visualizations show that HCC+RWMP enables the model to extract features uniformly at image boundaries, eliminating the feature attenuation at borders observed in standard CNNs.
- In rotation invariance tests, the HCC+RWMP combination produces a flatter accuracy curve, whereas baseline VGG shows accuracy degradation at 90° and 270° rotations.

## Highlights & Insights
- **The design of HCC is highly intuitive**: The cyclic structure of panoramic images is well-known prior knowledge, yet few prior works explicitly exploit it at the CNN level. This idea transfers directly to any task involving panoramic or spherical images.
- **Complementarity of depth and reflectance**: The two modalities capture different visual cues—depth encodes geometric structure (building contours, road geometry) while reflectance encodes material properties (vegetation, road surface texture). This complementarity explains why simple probability averaging serves as an effective fusion mechanism.
- **Grad-CAM analysis reveals the model's decision logic**: The coast category relies on horizon-line features (central region), the residential category relies on building features in the forward and backward directions, and the forest category relies on distributed texture features.

## Limitations & Future Work
- Training and evaluation rely solely on Sparse MPO; Dense MPO is underutilized due to its small size.
- The six-category taxonomy is coarse-grained; finer-grained classification (e.g., distinguishing subtypes of urban areas) remains unexplored.
- Early Fusion and Late Fusion underperform; more advanced attention-based fusion mechanisms (e.g., Transformers) may yield improvements.
- Data augmentation is limited to horizontal flipping and circular shifts; more complex augmentation strategies are not explored.
- Generalization to data collected in other cities or countries is not validated.

## Related Work & Insights
- **vs. Places/Places2**: The Places dataset trains CNNs on RGB scene images; this paper uses LiDAR panoramic images and is more robust to illumination changes.
- **vs. KITTI**: KITTI provides only 4 scene categories and is primarily designed for driving tasks; MPO offers 6 categories specifically for scene categorization.
- **vs. Song et al. (SUN RGB-D)**: SUN performs indoor scene categorization by concatenating RGB and depth CNN features; this paper focuses on outdoor LiDAR-based scenes.

## Rating
- **Novelty**: ⭐⭐⭐ — The circular convolution and row pooling ideas are concise and effective, but technically straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons across multiple model variants, exploration of multimodal fusion strategies, rotation invariance analysis, and Grad-CAM visualizations are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is well-structured with systematic experimental design and insightful visual analysis.
- **Value**: ⭐⭐⭐ — The dataset contribution is meaningful, but the research topic is relatively niche with limited broader impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)

</div>

<!-- RELATED:END -->
