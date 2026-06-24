---
title: >-
  [Paper Note] Unlocking Generalization Power in LiDAR Point Cloud Registration
description: >-
  [CVPR 2025][Autonomous Driving][point cloud registration] This work proposes the UGP framework, which significantly improves the generalization capability of LiDAR point cloud registration in cross-range and cross-dataset scenarios by eliminating cross-attention and introducing progressive self-attention and BEV feature fusion.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "point cloud registration"
  - "cross-range generalization"
  - "cross-dataset generalization"
  - "self-attention"
  - "BEV features"
date: 2026-05-08
content_hash: 2c9421b3f2d3400b
---

# Unlocking Generalization Power in LiDAR Point Cloud Registration

**Conference**: CVPR 2025  
**arXiv**: [2503.10149](https://arxiv.org/abs/2503.10149)  
**Code**: [GitHub](https://github.com/peakpang/UGP)  
**Area**: Autonomous Driving  
**Keywords**: point cloud registration, cross-range generalization, cross-dataset generalization, self-attention, BEV features

## TL;DR

This work proposes the UGP framework, which significantly improves the generalization capability of LiDAR point cloud registration in cross-range and cross-dataset scenarios by eliminating cross-attention and introducing progressive self-attention and BEV feature fusion.

## Background & Motivation

LiDAR point cloud registration is a core task in autonomous driving and SLAM. In real-world scenarios, there are two major variations between point cloud pairs: (1) cross-range variations, where distance differences between point clouds acquired at different speeds or times lead to changes in overlap rate and density distribution; (2) cross-dataset variations, where different environments or LiDAR types (e.g., 32-beam vs. 64-beam) result in discrepancies in data characteristics.

Existing State-of-the-Art (SOTA) methods (such as CoFiNet, GeoTransformer, and PARE) extensively utilize cross-attention to model geometric consistency between two frames. This, however, relies on an implicit assumption that the same structure has consistent representations across both frames. In cross-range and cross-dataset scenarios, the non-uniform density distribution of LiDAR point clouds invalidates this assumption, leading to severe degradation in generalization performance.

**Key Findings**: During training on KITTI@10m and testing on KITTI@20m, the cross-attention of GeoTransformer tends to match structures with similar densities (but wrong locations) rather than truly corresponding point pairs. After eliminating cross-attention, the matching focus shifts back to the correct regions.

Experimental evidence: On KITTI@40m, CoFiNet achieves a Registration Recall (RR) of only 1.4%, GeoTransformer achieves only 2.2%, and PARE scores 0%.

## Method

### Overall Architecture

UGP adopts a coarse-to-fine strategy: (1) projecting point clouds into BEV images; (2) extracting point features and image features using Point-Encoder (KPConv) and BEV-Encoder (ResNet), respectively; (3) fusing superpoint and BEV features based on indexing relationships; (4) performing coarse matching by extracting superpoint features using only progressive self-attention (without cross-attention); (5) obtaining fine matching and utilizing LGR to recover the rigid transformation.

### Key Designs

#### Key Design 1: Eliminating Cross-Attention

**Function**: Unlocking the network's cross-range and cross-dataset generalization capabilities.

**Mechanism**: Completely removing the cross-attention modules between the two frames, retaining only intra-frame self-attention. Features are extracted independently for each frame, and superpoint matching is based on feature similarity rather than explicit cross-frame interactions. The attention formula $e_{i,j} = \frac{(\mathbf{x}_i W^Q)(\mathbf{x}_j W^K + \mathbf{r}_{i,j} W^R)^T}{\sqrt{d_t}}$ is computed solely among superpoints within the same frame.

**Design Motivation**: The core flaw of cross-attention lies in the fact that the density of LiDAR point clouds decreases with distance, making the density representation of the same structure in two frames with different distances vastly different. Cross-attention is easily misled by regions with similar densities but different semantics. After its elimination, the network focuses on learning stable intra-frame feature representations, which are more invariant to range/dataset variations. This effect is directly validated by ablation experiments (Fig.2(d)).

#### Key Design 2: Progressive Self-Attention

**Function**: Reducing feature ambiguity in large-scale scenes and capturing multi-scale spatial structures.

**Mechanism**: Constraining the range of self-attention to local neighborhoods in the initial layers, and progressively expanding the attention range in subsequent layers. Each layer is configured with a different attention radius $r_l$, restricting superpoints in the $l$-th layer to interact only with superpoints within the radius $r_l$. The radius $r_l$ increases from small to large, forming a local-to-global attention cascade.

**Design Motivation**: Standard global self-attention allows each point to interact with all other points with equal weight, introducing feature ambiguity from distant, unrelated points. The progressive design enables the model to first learn fine-grained local geometric information and then gradually integrate global context, forming a more robust and consistent multi-scale representation.

#### Key Design 3: BEV Feature Fusion

**Function**: Introducing scene element-level semantic information (roads, corners, etc.) to reduce scene ambiguity.

**Mechanism**: Projecting 3D point clouds onto a Bird's-Eye View (BEV) plane, where the pixel coordinates are given by $u_i = \lfloor \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \cdot H \rfloor$, with filler values of 1. The BEV-Encoder (multi-layer ResNet + 2D max pooling) extracts patch features. Superpoint 3D geometric features are then concatenated and fused with 2D texture features based on the index mapping from superpoints to BEV patches.

**Design Motivation**: Pure point cloud backbones like KPConv struggle to establish correlations between local geometry and global context. BEV provides a global view of the point cloud, containing clear boundaries and texture features (e.g., road profiles). Such semantic information is crucial for reducing scene ambiguity and improving feature consistency.

### Loss & Training

Following the loss design of GeoTransformer, it contains superpoint matching loss and fine matching loss without introducing additional loss terms.

## Key Experimental Results

### Cross-Range Generalization (KITTI, Train on @10m)

| Method | @10m RR | @20m RR | @30m RR | @40m RR | mRR |
|------|---------|---------|---------|---------|-----|
| CoFiNet | 99.8 | 82.9 | 14.6 | 1.4 | 49.7 |
| GeoTrans | 99.8 | 7.5 | 3.2 | 2.2 | 28.2 |
| BUFFER | 99.8 | 98.6 | 93.5 | 61.2 | 88.3 |
| PARE | 99.8 | 1.8 | 0.0 | 0.0 | 25.4 |
| **UGP** | **99.8** | **99.3** | **96.8** | **82.0** | **94.5** |

### Cross-Range Generalization (nuScenes, Train on @10m)

| Method | @10m RR | @40m RR | mRR |
|------|---------|---------|-----|
| BUFFER | — | — | — |
| **UGP** | — | **72.3** | **91.4** |

### Key Findings

1. **Huge Advantage in Cross-Range Generalization**: On KITTI@40m, UGP achieves 82.0% RR, outperforming BUFFER by 20.8 percentage points (+34%) and CoFiNet by 80.6 percentage points. The mRR of 94.5% establishes a new SOTA.
2. **Direct Evidence of Eliminating Cross-Attention**: The mRR of GeoTrans w/o C (without cross-attention) improves from 28.2% to 87.7%, and CoFiNet w/o C improves from 49.7% to 58.8%. Simply removing cross-attention boosts generalization significantly.
3. **Cross-Dataset Generalization**: The average RR from nuScenes to KITTI reaches 90.9%, outperforming BUFFER by 6.2 percentage points.
4. **Effectiveness of BEV Features**: Compared to pure point cloud methods, the semantic information provided by BEV fusion further reduces scene ambiguity.

## Highlights & Insights

- **Counter-intuitive but Effective**: "Removing cross-attention" seemingly deprives the network of cross-frame information interaction capabilities, yet practically unlocks vast generalization potential. This reveals that the heavily relied-upon cross-attention in Transformer-based point cloud registration is indeed a generalization bottleneck.
- **In-depth Problem Analysis**: The root cause of cross-attention failure is uncovered from the perspective of LiDAR density distribution, which is theoretically sound and well-reasoned.
- **Simple yet Effective**: The core innovation lies in a "subtraction design" (removing cross-attention), complemented by progressive self-attention and BEV fusion, without introducing complex new modules.

## Limitations & Future Work

- **Loss of Information without Cross-Attention**: In same-range registration scenarios with consistent density and high overlap rates, UGP might perform slightly worse than methods with cross-attention.
- **BEV Projection Assumption**: The assumption of an approximately flat ground may lead to distortions in BEV projections for complex terrains (e.g., mountainous roads).
- **Unexplored Dynamic Objects**: The impact of moving objects in the scene on registration is not discussed.
- Future directions include adaptively deciding whether to use cross-attention and extending the work to multi-LiDAR sensor fusion.

## Related Work & Insights

- **GeoTransformer**: A registration method introducing cross-attention embedded with geometric structures; UGP demonstrates that its cross-attention acts as a bottleneck for generalization.
- **BUFFER**: A method using patch-wise feature extraction, which is robust to noise and occlusion but still lacks sufficient generalization.
- **Insights**: In tasks dependent on cross-instance interaction, eliminating cross-interaction could be a general strategy to enhance generalization.

## Rating

⭐⭐⭐⭐ — Deep core insights (cross-attention limiting generalization), impressive experimental results (mRR 94.5%), and comprehensive analysis. The simplicity of the "subtraction design" reflects a profound understanding of the core problem. Direct significance for autonomous driving safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](../../ICML2025/autonomous_driving/geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[CVPR 2025\] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization](superpc_a_single_diffusion_model_for_point_cloud_completion_upsampling_denoising.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[ICCV 2025\] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration](../../ICCV2025/autonomous_driving/mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)
- [\[CVPR 2025\] Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)

</div>

<!-- RELATED:END -->
