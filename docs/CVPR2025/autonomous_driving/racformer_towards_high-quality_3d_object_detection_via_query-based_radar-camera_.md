---
title: >-
  [Paper Note] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion
description: >-
  [CVPR 2025][Autonomous Driving][Radar-Camera Fusion] This work proposes RaCFormer, a query-based radar-camera fusion framework. By simultaneously sampling features from both the image perspective and the BEV perspective, and incorporating modules such as circular query initialization, radar-aware depth prediction, and an implicit dynamic catcher, it achieves 64.9% mAP and 70.2% NDS on nuScenes.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Radar-Camera Fusion"
  - "3D Object Detection"
  - "BEV Perception"
  - "Query Mechanism"
date: 2026-05-08
content_hash: cc0e2fcc38986080
---

# RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion

**Conference**: CVPR 2025  
**arXiv**: [2412.12725](https://arxiv.org/abs/2412.12725)  
**Code**: [GitHub](https://github.com/cxmomo/RaCFormer)  
**Area**: Autonomous Driving  
**Keywords**: Radar-Camera Fusion, 3D Object Detection, BEV Perception, Query Mechanism

## TL;DR

This work proposes RaCFormer, a query-based radar-camera fusion framework. By simultaneously sampling features from both the image perspective and the BEV perspective, and incorporating modules such as circular query initialization, radar-aware depth prediction, and an implicit dynamic catcher, it achieves 64.9% mAP and 70.2% NDS on nuScenes.

## Background & Motivation

Current mainstream radar-camera fusion methods for 3D detection predominantly adopt the BEV fusion paradigm: converting image and radar features into BEV space individually before fusing them via concatenation or cross-attention. However, this paradigm faces three core bottlenecks:

1. **Camera BEV feature distortion**: The perspective-to-BEV transformation relies heavily on depth estimation. Inaccurate depth predictions lead to misaligned visual content in the BEV features.
2. **Radar BEV feature sparsity**: Millimeter-wave radar possesses limited spatial resolution, making the generated BEV features extremely sparse.
3. **Wasted perspective information**: While the original perspective-view image features are semantically rich and free of distortion, the BEV fusion paradigm fails to exploit these features.

The key observation is that query-based methods can leverage object queries initialized in 3D space as a medium to adaptively sample features from any projection perspective (perspective view and BEV), thereby bypassing issues related to feature density discrepancies and distortion. This motivates the authors to propose RaCFormer, a cross-view, cross-modal query-based fusion framework.

## Method

### Overall Architecture

RaCFormer consists of six core modules: an image encoder, a Pillar encoder, a radar-aware depth head, an LSS view transformer, an implicit dynamic catcher, and a Transformer decoder. The image encoder extracts multi-frame multi-camera features; the Pillar encoder processes radar point clouds and flattens them into BEV space; the depth head leverages radar data to enhance depth estimation; the LSS module generates camera BEV features; the implicit dynamic catcher captures temporal motion cues in the radar BEV space; and the Transformer decoder initializes queries in a circular distribution, progressively refining them through $L$ layers while using a ray-sampling module to extract features from both BEV and perspective-view images in each layer.

### Key Designs

#### Key Design 1: Linearly Increasing Circular Query Distribution

**Function**: Optimizing the initialization distribution of object queries in 3D space.

**Mechanism**: Traditional radial distribution (RayFormer) places queries uniformly along camera rays, leading to sparse queries in far-range areas. RaCFormer adopts a concentric-circle distribution, placing queries across $k$ concentric circles. The innermost circle contains $n$ queries, and each outward circle increases the quantity by a factor of $\alpha$, meaning the $k$-th circle contains $\alpha^{k-1} \times n$ queries. The total number of queries is formulated as:

$$N = \frac{\alpha^k - 1}{\alpha - 1} \times n, \quad \alpha \neq 1$$

**Design Motivation**: Far-range areas require more queries to cover a larger surface area. The linearly increasing strategy allows the query density to grow appropriately with distance, aligning with sensor projection principles. When $\alpha = 1$, this degenerates to a radial distribution.

#### Key Design 2: Radar-Aware Depth Prediction

**Function**: Leveraging radar depth information to improve the accuracy of perspective-to-BEV view transformation.

**Mechanism**: Conventional automotive radars have low vertical angular resolution, leading to large errors in $z$-coordinate estimation. RaCFormer sets $z_r$ of all radar points to 1. After projecting them to the image plane, it expands the vertical coordinate of each projected point to the full height $H$ of the image and discretizes the depth values. Additionally, RCS (Radar Cross Section) attributes and pixel position information are embedded, concatenated with the downsampled image features $C4$, and fed into the depth head.

**Design Motivation**: Setting a constant height maximizes the number of radar points that fall within the image’s field of view. Combining RCS and position embeddings provides more comprehensive radar-aware features, ultimately generating a more accurate depth probability distribution $D'$.

#### Key Design 3: Implicit Dynamic Catcher

**Function**: Capturing temporal motion elements from multi-frame radar BEV features.

**Mechanism**: Exploiting the Doppler effect of millimeter-wave radar for measuring moving object velocities, a ConvGRU is employed to accumulate hidden states across sequential frames:

$$h_t = \text{ConvGRU}(x_t, h_{t-1})$$
$$x'_t = \text{Conv2D}(h_t \oplus x_t)$$

where $x_t$ is the BEV features of the $t$-th frame, and $h_{t-1}$ is the hidden state of the previous frame.

**Design Motivation**: ConvGRU is proficient in processing sequential data and capturing spatial hierarchies. By accumulating hidden states of multi-frame radar BEV features, it implicitly models the temporal dynamics of moving targets, enhancing motion awareness.

### Loss & Training

A standard 3D object detection loss is adopted, including classification loss and regression loss (position, size, orientation, velocity, attribute), which resolves refined queries through classification and regression heads.

## Key Experimental Results

### Main Results

#### nuScenes Validation Set

| Method | Input | Backbone | mAP↑ | NDS↑ | mATE↓ | mAVE↓ |
|------|------|----------|------|------|-------|-------|
| StreamPETR | C | ResNet50 | 45.0 | 55.0 | 0.613 | 0.265 |
| RCBEVDet | C+R | ResNet50 | 45.3 | 56.8 | 0.486 | 0.220 |
| HyDRa | C+R | ResNet50 | 49.4 | 58.5 | 0.463 | 0.227 |
| **RaCFormer** | **C+R** | **ResNet50** | **54.1** | **61.3** | **0.478** | **0.208** |
| HyDRa | C+R | ResNet101 | 53.6 | 61.7 | 0.416 | 0.231 |
| **RaCFormer** | **C+R** | **ResNet101** | **57.3** | **63.0** | **0.476** | **0.213** |

#### nuScenes Test Set

RaCFormer achieves **64.9% mAP** and **70.2% NDS**, outperforming all other radar-camera fusion methods.

### Ablation Study

| Configuration | mAP | NDS |
|------|-----|-----|
| Baseline (BEV fusion) | 49.4 | 58.5 |
| + Query fusion | 51.2 | 59.8 |
| + Circular query initialization | 52.5 | 60.4 |
| + Radar-aware depth head | 53.3 | 60.9 |
| + Implicit dynamic catcher | 54.1 | 61.3 |

### Key Findings

- The query-based fusion paradigm outperforms the BEV fusion paradigm by approximately **+4.7 mAP**, validating the benefit of cross-view sampling.
- Circular query initialization yields a **+1.3 mAP** improvement over the radial distribution, showing significant enhancement in long-range detection.
- On the VoD dataset, RaCFormer achieves **54.4% mAP**, securing first place in the entire annotated region.

## Highlights & Insights

1. **Core Idea**: Stepping away from the conventional BEV fusion paradigm, this work returns to the query mechanism to sample features from arbitrary perspectives, effectively bypassing issues of uneven feature density and depth estimation distortion.
2. **Circular Query Distribution**: A simple yet effective geometric prior, where the linearly increasing query density naturally aligns with the uneven distribution of near-and-far objects in autonomous driving scenarios.
3. **Radar Depth Perception**: By adopting a preprocessing strategy of setting a default height and expanding it to the full image height, it creatively resolves the limitation of insufficient vertical angular resolution in automotive radar.

## Limitations & Future Work

- The method relies on high-quality tracklets from 3D object annotations, which may limit its performance in severely occluded scenarios.
- The hyperparameters $\alpha$ and $k$ in the circular query distribution require tuning for different scenarios.
- Future research could explore adaptive strategies for dynamically adjusting query distributions or extensions incorporating 4D radar fusion.

## Related Work & Insights

- **RayFormer**: The pioneer of radial ray query initialization, upon which RaCFormer improves by introducing a distance-adaptive circular distribution.
- **SparseBEV**: A fully sparse detection framework; RaCFormer benefits from incorporating its scale-adaptive self-attention mechanism.
- **HyDRa**: One of the strongest baselines; it uses a hybrid approach to fuse perspective-view and BEV features, yet remains limited by the BEV fusion paradigm.

## Rating

⭐⭐⭐⭐ — The method's design motivation is clear. Both the circular query distribution and the radar-aware depth prediction present simple and effective solutions to realistic challenges. The +4.7 mAP strength on nuScenes is substantial, backed by competitive results demonstrating first place on VoD. However, the core innovation leans toward engineering combinations rather than a fundamental breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2026\] RPGFusion: 4D Radar Prior-Guided Multi-Modal Fusion for 3D Detection](../../CVPR2026/autonomous_driving/rpgfusion_4d_radar_prior-guided_multi-modal_fusion_for_3d_detection.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)

</div>

<!-- RELATED:END -->
