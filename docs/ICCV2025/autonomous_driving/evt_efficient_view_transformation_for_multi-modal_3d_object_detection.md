---
title: >-
  [Paper Note] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection
description: >-
  [ICCV 2025][Autonomous Driving][BEV representation] This paper proposes EVT, a framework that achieves efficient LiDAR-guided view transformation via Adaptive Sampling and Adaptive Projection (ASAP), combined with group-wise mixed query selection and geometry-aware cross-attention, attaining state-of-the-art performance of 75.3% NDS on the nuScenes test set at real-time inference speed.
tags:
  - ICCV 2025
  - Autonomous Driving
  - BEV representation
  - LiDAR-Camera fusion
  - view transformation
  - multi-modal 3D detection
  - real-time inference
date: 2026-05-08
content_hash: 52ad0836b4b99c04
---

# EVT: Efficient View Transformation for Multi-Modal 3D Object Detection

**Conference**: ICCV 2025
**arXiv**: [2411.10715](https://arxiv.org/abs/2411.10715)
**Code**: N/A
**Area**: Autonomous Driving / 3D Object Detection
**Keywords**: BEV representation, LiDAR-Camera fusion, view transformation, multi-modal 3D detection, real-time inference

## TL;DR

This paper proposes EVT, a framework that achieves efficient LiDAR-guided view transformation via Adaptive Sampling and Adaptive Projection (ASAP), combined with group-wise mixed query selection and geometry-aware cross-attention, attaining state-of-the-art performance of 75.3% NDS on the nuScenes test set at real-time inference speed.

## Background & Motivation

Multi-modal sensor fusion under BEV representation has become the dominant paradigm for 3D object detection. Existing explicit fusion methods fall into two categories:

**Depth-based VT methods**: Lift image features into BEV space via per-pixel depth estimation, but are sensitive to depth errors and exhibit poor robustness.

**Query-based VT methods**: Extract features at predefined 3D points via attention mechanisms, but incur high computational cost, and the fixed sampling point locations cannot precisely align with object regions.

Both categories lack geometric guidance, leading to **ray-directional misalignment**—non-target information is captured along the ray direction, degrading BEV representation quality. Furthermore, existing query initialization strategies and cross-attention mechanisms fail to fully exploit object geometry, limiting detection performance.

## Method

### Overall Architecture

EVT consists of three core components:
- An image backbone and a LiDAR backbone extract multi-scale perspective-view features $\{PV_j\}$ and BEV LiDAR features $BEV_{lidar}$, respectively.
- The ASAP module transforms image features into BEV space and fuses them with LiDAR features.
- An improved query-based detection head (group-wise mixed query selection + geometry-aware cross-attention) predicts 3D bounding boxes.

### Key Designs

1. **Adaptive Sampling (AS)**: Instead of using predefined 3D points, LiDAR features are used to generate optimal sampling heights for each BEV grid cell. Specifically, $N_h$ height values $\{Z_i\}$ are generated from $BEV_{lidar}$ via convolution to construct 3D sampling points $P=\{(X,Y,Z_i)\}$, which are projected onto the image plane to sample multi-scale features $f_i^j$. These are then aggregated with adaptive weights $W_{as}$:

$$BEV_{as}(u,v) = \sum_{j=1}^{N_s}\sum_{i=1}^{N_h} W_{as}(j,i) \cdot f_i^j$$

The weights are generated from LiDAR features via softmax, focusing sampling on highly relevant image regions.

2. **Adaptive Projection (AP)**: LiDAR features are used to generate a per-grid adaptive kernel $K_{ap} \in \mathbb{R}^{C \times C}$, which performs a per-channel linear projection to refine the BEV feature map output from AS:

$$BEV_{camera}(u,v) = BEV_{as}(u,v) \times K_{ap}$$

AP leverages LiDAR spatial information to effectively eliminate ray-directional misalignment, particularly handling occlusion and empty 3D space. The fused feature $BEV_{fuse}$ is obtained via concatenation followed by convolution.

3. **Group-wise Mixed Query Selection**: Object categories are divided into 6 groups (e.g., car; truck + construction vehicle; etc.). A heatmap is predicted per group, and top-k keypoints are selected as query positions. Unlike prior methods, query features are initialized with **group-shared learnable parameters**—queries within the same group share parameters to capture group-level common attributes, as opposed to DINO's per-instance parameters or TransFusion's direct sampling. Experiments confirm that group-shared initialization outperforms instance-level initialization.

4. **Geometry-aware Cross-Attention**:

    - **Corner-aware Sampling**: After initial offsets are generated from query features, sampling points are relocated to the four corners of the bounding box via geometric transformation and aligned according to the heading angle, involving a rotation matrix and bounding box dimensions $(l, w, \theta)$.
    - **Position-aware Feature Mixing**: Positional vectors $e_i$ based on sinusoidal positional encoding are added to sampled features to construct position-aware features $G$. Adaptive channel mixing (dynamic weights $W_c$) and adaptive spatial mixing (dynamic weights $W_s$) are applied sequentially, with the query updated via a residual connection.

### Loss & Training

- Heatmap prediction: Gaussian Focal Loss
- Classification: Focal Loss
- Regression: L1 Loss
- Query denoising strategy adopted
- 8× RTX 3090, batch size 16, 10 epochs with CBGS
- AdamW optimizer, learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-2}$, cyclic learning rate schedule

## Key Experimental Results

### Main Results

| Method | Modality | NDS (val) | mAP (val) | NDS (test) | mAP (test) |
|--------|----------|-----------|-----------|------------|------------|
| TransFusion-L | L | 70.1 | 65.1 | 70.2 | 65.5 |
| EVT-L (Ours) | L | **71.7** | **66.4** | **72.1** | **67.7** |
| BEVFusion | LC | 71.4 | 68.5 | 72.9 | 70.2 |
| DeepInteraction | LC | 72.6 | 69.9 | 73.4 | 70.8 |
| CMT | LC | 72.9 | 70.3 | 74.1 | 72.0 |
| SparseFusion | LC | 72.8 | 70.4 | 73.8 | 72.0 |
| UniTR | LC | 73.3 | 70.5 | 74.5 | 70.9 |
| FusionFormer | LCT | 74.1 | 71.4 | 75.1 | 72.6 |
| **EVT (Ours)** | **LC** | **74.6** | **72.1** | **75.3** | **72.6** |

EVT achieves 75.3% NDS and 72.6% mAP on the nuScenes test set, surpassing all prior methods. The 3.2% NDS gain over LiDAR-only EVT-L demonstrates effective utilization of camera data.

### Ablation Study

**ASAP module ablation** (ResNet-50 backbone):

| Setting | LiDAR | Camera | AS | AP | NDS | mAP | FPS |
|---------|-------|--------|----|----|-----|-----|-----|
| (a) LiDAR-only | ✓ | | | | 71.7 | 66.4 | 12.1 |
| (b) +Vanilla VT | ✓ | ✓ | | | 72.7 | 69.1 | 8.5 |
| (c) +AS | ✓ | ✓ | ✓ | | 73.5 | 70.6 | 8.5 |
| (d) +ASAP | ✓ | ✓ | ✓ | ✓ | **74.1** | **71.1** | 8.3 |

ASAP improves over vanilla VT by 1.4% NDS and 2.0% mAP with only 3 ms additional latency.

**Query initialization strategy ablation** (NDS % at 1 and 6 decoder layers):

| Strategy | 1 Layer | 6 Layers |
|----------|---------|----------|
| (a) Fully learnable | 56.8 | 69.6 |
| (b) Fully heatmap-based | 70.4 | 70.7 |
| (c) Mixed + group-shared | 69.8 | **71.7** |

Group-wise mixed initialization significantly outperforms other strategies across multiple decoder layers.

### Key Findings

- ASAP leverages LiDAR guidance to adaptively position sampling points in object-relevant regions, effectively resolving ray-directional misalignment.
- Group-shared query initialization outperforms per-instance parameterization in multi-layer decoders by better capturing intra-group common attributes.
- Corner-aware sampling combined with position-aware feature mixing contributes a joint gain of 1.2% NDS and 1.2% mAP.
- Geometry-aware cross-attention is generalizable; integrating it into StreamPETR also yields improvement (+1.3% NDS @24 epochs).

## Highlights & Insights

- **Elegant design philosophy**: LiDAR features guide the image-to-BEV transformation without relying on error-prone depth estimation or expensive transformer encoders.
- **Clear problem formulation and resolution**: The ray-directional misalignment problem is explicitly defined and effectively addressed by ASAP, supported by BEV feature map visualizations.
- **Strong efficiency**: 8.3 FPS with ResNet-50 backbone and 4.9 FPS with V2-99, indicating practical real-time deployment potential.
- **Transferability of geometry-aware attention**: The proposed attention mechanism can be directly applied to other detectors.

## Limitations & Future Work

- Evaluation is limited to nuScenes; validation on other datasets (e.g., Waymo) is absent.
- Only single-frame data is used; temporal information is not exploited (FusionFormer incorporates temporal fusion).
- ASAP relies on the quality of LiDAR features and may degrade in LiDAR-sparse regions.
- Adaptation to camera-only settings is not discussed.

## Related Work & Insights

- TransFusion's sequential cross-attention design inspired the multi-modal fusion framework.
- DINO's query initialization strategy is improved by replacing per-instance parameters with group-shared ones.
- AdaMixer's feature decoding approach is used for comparison, revealing its incompatibility with corner-aware sampling.
- The extension experiment on StreamPETR motivates the general applicability of geometry-aware attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Both ASAP and geometry-aware cross-attention present clear innovations; the analysis of ray-directional misalignment is well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive with visualization evidence for each component, though multi-dataset validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Equations are clear, figures are intuitive, and the paper is well-structured.
- **Value**: ⭐⭐⭐⭐ The method is practical with a strong efficiency–accuracy trade-off, and the state-of-the-art results are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](../../CVPR2026/autonomous_driving/ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[AAAI 2026\] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](../../AAAI2026/autonomous_driving/fq-petr_fully_quantized_position_embedding_transformation_fo.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)

</div>

<!-- RELATED:END -->
