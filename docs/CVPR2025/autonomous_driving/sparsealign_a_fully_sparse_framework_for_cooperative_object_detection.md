---
title: >-
  [Paper Note] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection
description: >-
  [CVPR 2025][Autonomous Driving][Cooperative Perception] SparseAlign proposes the first fully sparse framework for cooperative object detection. By resolving the problems of center feature missing and isolated convolution domains via coordinate-expandable sparse convolution, it outperforms dense BEV-based state-of-the-art methods while reducing communication bandwidth by 98%.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Cooperative Perception"
  - "Sparse Detection"
  - "LiDAR"
  - "Object Detection"
  - "Communication Efficiency"
date: 2026-05-08
content_hash: 305c59fc35f71c52
---

# SparseAlign: A Fully Sparse Framework for Cooperative Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.12982](https://arxiv.org/abs/2503.12982)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Cooperative Perception, Sparse Detection, LiDAR, Object Detection, Communication Efficiency

## TL;DR

SparseAlign proposes the first fully sparse framework for cooperative object detection. By resolving the problems of center feature missing and isolated convolution domains via coordinate-expandable sparse convolution, it outperforms dense BEV-based state-of-the-art methods while reducing communication bandwidth by 98%.

## Background & Motivation

Cooperative perception enhances the field of view and mitigates occlusions by sharing sensory information among multiple agents, which is crucial for autonomous driving safety. Existing cooperative object detection methods primarily operate on dense BEV feature maps, posing two core issues:

- **Computational complexity grows quadratically with perception range**: The size of the BEV feature map is proportional to the perception distance, making long-range detection computationally expensive.
- **High communication bandwidth consumption**: Sharing dense BEV feature maps requires substantial transmission resources.

Fully sparse frameworks exploit the sparsity of point clouds, where computational complexity scales linearly with the number of points. However, building a competitive fully sparse framework faces two technical challenges:

- **Center Feature Missing (CFM)**: LiDAR points are typically absent in the central regions of objects, whereas the center points are most critical for object representation.
- **Isolated Convolution Fields (ICF)**: In long-range areas, points scanned by different laser beams show poor connectivity, leading to mutually isolated voxel blocks, which prevents the receptive field from expanding.

## Method

### Overall Architecture

SparseAlign consists of an enhanced sparse 3D backbone (SUNet), a query-based temporal context learning module (TAM), a pose alignment module (PAM), and a spatial alignment module (SAM). All agents share network weights and broadcast Object Query features as CPM.

### Key Designs

**1. Coordinate-Expandable Sparse Convolution (CEC) resolving CFM and ICF**

- **Function**: Concurrently addresses both center feature missing and isolated convolution fields, constructing an effective fully sparse 3D backbone.
- **Mechanism**: Employs CEC in the $4\times$ and $8\times$ downsampling layers of 3D sparse convolution to expand voxel connectivity and increase receptive field coverage. It uses CEC coordinate expansion on 2D BEV sparse features to ensure that the center locations of all scanned objects are covered by features.
- **Design Motivation**: The receptive field of standard sparse convolution only covers LiDAR points of a single vehicle (as shown in Fig. 3c). After CEC expansion, it can cover the points of neighboring vehicles (as shown in Fig. 3d), enabling occluded and distant objects to aggregate neighborhood information.

**2. Pose-Agnostic Neighborhood Graph Feature Matching (PAM)**

- **Function**: Corrects relative pose errors among cooperative agents without relying on the accuracy of initial poses.
- **Mechanism**: Embeds the **relative geometric features** of its $K=8$ nearest neighbors for each detection bounding box (relative orientation $\nu_a$, relative edge orientation $\epsilon_a$, Euclidean distance $\epsilon_d$, and neighbor dimensions $\nu_{dim}$). These features are pose-independent relative quantities. After aggregating them via self-attention, the Hungarian algorithm is used to match two sets of BBoxes, followed by Pose Graph Optimization (PGO) for alignment.
- **Design Motivation**: Existing methods require small initial pose errors to achieve correct matching. The relative features in the proposed method are inherently independent of the global coordinate system, enabling robust matching even under large pose errors.

**3. Spatial Alignment Module (SAM) for fusing Sparse Queries**

- **Function**: Precisely fuses the sparse query features of cooperative agents into the ego-vehicle coordinate system.
- **Mechanism**: First uses an MLP conditioned on the rotation matrix $R$ to perform feature space transformation $F^c = MLP([F^c; F^R])$. It then aligns the rotated cooperative query coordinates to the nearest ego grid points. Finally, it aggregates features from $Q^c \cup Q^e$ (including relative position encoding) via K-Nearest Neighbors (KNN), employing mean+max pooling to generate fused features.
- **Design Motivation**: The coordinates of sparse queries do not align with the grid after rotation, requiring special handling. KNN aggregation combined with position encoding can flexibly handle irregular point positions.

### Loss & Training

Focal Loss (foreground-background classification) + Smooth L1 Loss (BBox regression, including position offset, dimension, and CompassRose orientation encoding). CompassRose encodes orientation using 4 anchor angles, ensuring that at least one anchor can monotonically reach the target angle.

## Key Experimental Results

### Main Results: OPV2V Dataset

| Method | Communication Bandwidth (Mb)↓ | AP@0.5↑ | AP@0.7↑ |
|------|-----------|---------|---------|
| V2VNet (Dense BEV) | 72.08 | 0.917 | 0.822 |
| CoBEVT (Dense BEV) | 72.08 | 0.927 | 0.830 |
| V2X-ViT (Dense BEV) | 72.08 | 0.926 | 0.844 |
| **SparseAlign** | **~1.5** | **0.935** | **0.860** |

### Ablation Study: Contribution of Each Component (OPV2V)

| Component | AP@0.7 |
|------|--------|
| MinkUNet baseline | 0.790 |
| + CEC (Resolving ICF) | 0.825 |
| + CEC (Resolving CFM) | 0.842 |
| + TAM | 0.850 |
| + PAM + SAM | **0.860** |

### Key Findings

- SparseAlign outperforms all dense BEV methods while achieving a **98% reduction in communication bandwidth**.
- Addressing ICF and CFM using CEC yields improvements of 3.5% and 1.7% in AP@0.7, respectively.
- Achieves state-of-the-art performance on DairV2X (real-world dataset) and temporal alignment tasks (OPV2Vt/DairV2Xt).
- CompassRose orientation encoding improves AP by approximately 0.5% compared to standard sin/cos encoding.
- Free Space Augmentation effectively mitigates the ICF problem in long-range sparse areas.

## Highlights & Insights

1. **The first fully sparse framework to outperform dense BEV methods** is a significant breakthrough in the field of cooperative perception, proving the feasibility and superiority of sparse processing in multi-agent scenarios.
2. **Pose-agnostic graph matching** is an elegant engineering design that utilizes topological structures instead of absolute coordinates for cross-agent matching.
3. The 98% improvement in communication efficiency is highly significant for practical V2X deployment, as reducing the bandwidth from 72Mb to 1.5Mb enables support under existing cellular networks.

## Limitations & Future Work

- The current work only handles LiDAR cooperative detection, without expanding to camera fusion or semantic segmentation.
- CEC introduces minor computational overhead (though still far lower than dense methods).
- The robustness of matching under extreme sparsity (ultra-long range) remains to be validated.
- Future research can explore adaptive CEC expansion strategies and more efficient query compression schemes.

## Related Work & Insights

- **Relationship with V2X-ViT/CoBEVT**: These methods perform attention fusion on dense BEV representation. SparseAlign proves that sparse queries can achieve equal or even superior fusion performance.
- **Relationship with FPVRCNN**: FPVRCNN also utilizes sparse feature sharing, but the CEC backbone and SAM fusion in SparseAlign are more robust.
- **Insight**: In multi-agent cooperation, "less is more" — sparse queries are more efficient than "comprehensive" dense feature maps.

## Rating

⭐⭐⭐⭐

The first fully sparse framework to outperform dense BEV in cooperative detection, achieving a 98% communication bandwidth reduction. It systematically resolves the two core problems of sparse backbones, CFM and ICF. The pose-agnostic matching design of PAM is highly ingenious. This work holds substantial value for practical V2X deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](../../ECCV2024/autonomous_driving/fully_sparse_3d_occupancy_prediction.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)

</div>

<!-- RELATED:END -->
