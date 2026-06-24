---
title: >-
  [Paper Note] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction
description: >-
  [CVPR 2025][Autonomous Driving][3D Occupancy Prediction] This paper proposes SDG-OCC, a multimodal 3D semantic occupancy prediction framework. By replacing the traditional LSS pipeline with a semantic and depth-guided view transformation (which utilizes LiDAR depth and image semantic segmentation masks to construct virtual points), and combining it with a fusion-to-occupancy-driven active distillation module, the method achieves SOTA performance on Occ3D-nuScenes while mainta…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Occupancy Prediction"
  - "Multimodal Fusion"
  - "BEV Transformation"
  - "Knowledge Distillation"
  - "LiDAR-Camera Fusion"
date: 2026-05-08
content_hash: 3e1d82ce37f4d01a
---

# SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction

**Conference**: CVPR 2025  
**arXiv**: [2507.17083](https://arxiv.org/abs/2507.17083)  
**Code**: [https://github.com/DzpLab/SDGOCC](https://github.com/DzpLab/SDGOCC)  
**Area**: Autonomous Driving / 3D Occupancy Prediction  
**Keywords**: 3D Occupancy Prediction, Multimodal Fusion, BEV Transformation, Knowledge Distillation, LiDAR-Camera Fusion

## TL;DR

This paper proposes SDG-OCC, a multimodal 3D semantic occupancy prediction framework. By replacing the traditional LSS pipeline with a semantic and depth-guided view transformation (which utilizes LiDAR depth and image semantic segmentation masks to construct virtual points), and combining it with a fusion-to-occupancy-driven active distillation module, the method achieves SOTA performance on Occ3D-nuScenes while maintaining real-time inference speed.

## Background & Motivation

**Background**: 3D semantic occupancy prediction simultaneously estimates the geometric structure and semantic categories of scene voxels, modeling the environment more comprehensively compared to 3D object detection and semantic segmentation. Multimodal methods leverage the complementarity of LiDAR and camera data: cameras provide rich semantic info but lack accurate depth, while LiDAR provides precise depth but is sparse.

**Limitations of Prior Work**: (1) Currently, lightweight methods primarily rely on the LSS (Lift-Splat-Shoot) pipeline for 2D-3D view transformation, but LSS depth estimation is inaccurate, resulting in extremely sparse BEV representations—where less than 50% of the BEV grids receive valid image features. (2) Reducing the depth interval can improve accuracy but significantly increases computational overhead. (3) Existing fusion methods directly concatenate LiDAR and image BEV features; however, due to feature misalignment caused by extrinsic calibration errors, the fusion performance is suboptimal. (4) Most multimodal methods involve massive computational costs and cannot run in real time.

**Key Challenge**: The LSS pipeline constructs a massive number of virtual points within a predefined depth range for each pixel, but most of these virtual points fall into empty BEV grids, causing redundant computations and low BEV utilization. Meanwhile, LiDAR, which can provide accurate depth priors, is underutilized.

**Goal**: To design a multimodal occupancy prediction framework that boosts accuracy while maintaining real-time inference speed, with the key being the improvement of 2D-3D view transformation and multimodal fusion mechanisms.

**Key Insight**: Using the sparse depth info of LiDAR point clouds as a prior, depth values are diffused within the same semantic category. Then, high-precision virtual point seeds are generated through bilinear discretization, which significantly reduces the number of redundant virtual points while improving BEV feature quality.

**Core Idea**: To use semantic segmentation masks to guide LiDAR depth diffusion to generate semi-dense depth maps, and construct precise virtual points in combination with bidirectional linear interpolation discretization, replacing the blind depth sampling of LSS. Meanwhile, multimodal knowledge is unidirectionally distilled into image features through a fusion-and-occupancy-driven active distillation mechanism.

## Method

### Overall Architecture

Multi-frame multi-view images and corresponding point clouds are input and processed by the image backbone and LiDAR backbone, respectively, to extract features. Image features pass through multi-task heads to generate semantic masks and depth distributions, which are then combined with the LiDAR depth map via SDG view transformation to generate the image BEV feature $F_{bev}^C$. Point cloud features are voxelized, encoded by SPVCNN, and compressed into the LiDAR BEV feature $F_{bev}^L$. The two BEV features are integrated via dynamic neighborhood feature fusion and occupancy-driven active distillation, and then fed into the occupancy prediction head to generate the final output.

### Key Designs

1. **SDG Selection Transformation (Semantic and Depth-Guided View Transformation)**:

    - **Function**: To replace the traditional LSS pipeline and construct more accurate and efficient BEV features using LiDAR depth and semantic information.
    - **Mechanism**: First, a multi-task head generates semantic segmentation masks and depth distributions. Then, the LiDAR point cloud is projected onto the image to obtain a sparse depth map, and local depth diffusion is performed within regions of the same semantic category (within a circle of radius $r$, calculated as:
    $$D_{\text{temp}}(i,j) = \frac{\sum_{(p,q)\in N(i,j)} D(p,q) \cdot \mathbb{I}[M(p,q)=M(i,j)]}{\sum_{(p,q)\in N(i,j)} \mathbb{I}[M(p,q)=M(i,j)]}$$
    ) to generate a semi-dense depth map. Next, bidirectional linear interpolation discretization is applied to generate precise virtual point seeds. Finally, the image texture feature $F_t$ and depth distribution weight $D_w$ are combined via outer product $F_t \otimes D_w$ to obtain the features of each virtual point, which are then processed by BEV pooling to generate the image BEV feature $F_{bev}^C$.
    - **Design Motivation**: LSS generates a large number of depth hypothesis points for each pixel, but most are invalid and the BEV utilization is low (<50%). Using LiDAR depth priors and semantic mask guidance can substantially reduce redundant virtual points while improving depth estimation accuracy.

2. **Dynamic Neighborhood Feature Fusion (Dynamic Neighborhood Feature Fusion)**:

    - **Function**: To resolve the spatial misalignment issue between LiDAR and image BEV features caused by extrinsic calibration errors.
    - **Mechanism**: The image feature is treated as the source (query) and the LiDAR feature as the cross (key/value). Neighborhood attention is used to extract features within a local patch around the corresponding pixel: $F_{\text{neighbor}} = \sigma(\frac{Q_s^i \cdot (K_c^{n(i)})^T + B(i, n(i))}{\sqrt{v}}) \cdot V_c^i$, where $n(i)$ represents a neighborhood of size $k$. Then, a gated attention mechanism dynamically adjusts the fusion weights: $F_{bev}^{fuse} = \sigma(\text{Conv}(f_{\text{Avg}}(F_{\text{neighbor}}))) \cdot F_{\text{neighbor}}$.
    - **Design Motivation**: Simple channel concatenation cannot handle the spatial misalignment between LiDAR and image BEV features. The neighborhood attention combined with the gating mechanism can implicitly address projection deviations and dynamically adjust fusion weights.

3. **Occupancy-Driven Active Distillation (Occupancy-Driven Active Distillation)**:

    - **Function**: To unidirectionally distill knowledge from multimodal fused features to pure image features, achieving faster inference.
    - **Mechanism**: The space is divided into Active Regions (AR, where both LiDAR and image features indicate occupancy) and Inactive Regions (IR, where only LiDAR features indicate occupancy). Since AR is usually much larger than IR, adaptive scaling is applied: $W_{I,i,j} = \alpha$ (for AR) or $\rho \times \beta$ (for IR), where $\rho = N_{AR}/N_{IR}$ prevents the distillation from being overly biased toward AR. The distillation loss is defined as $L_{\text{distill}} = \sum W_{i,j}(F_{bev}^{fuse} - F_{bev}^C)^2$. During training, both fusion and distillation are utilized (SDG-KL), while only the image branch is used during inference.
    - **Design Motivation**: The fusion model (SDG-Fusion) offers high accuracy but requires processing both LiDAR and image data during inference. Through active distillation, SDG-KL only needs image inputs during inference to achieve performance close to the fusion model, enabling real-time inference.

### Loss & Training

- SDG-Fusion: Classification loss (cross-entropy loss of the occupancy prediction head output)
- SDG-KL: Classification loss + distillation loss (region-weighted MSE)
- The auxiliary semantic segmentation task provides supervision signals for the multi-task heads
- Depth and semantic heads complement cross-task information via gated attention

## Key Experimental Results

### Main Results

Occ3D-nuScenes validation set:

| Method | Input | Backbone | mIoU | Inference Time (ms) |
|------|------|----------|------|-------------|
| FlashOcc | C | Swin-B | 43.52 | 909 |
| COTR | C | Swin-B | 46.2 | 840 |
| OCCFusion | C+L | R-101 | 46.79 | - |
| RadOcc-LC | C+L | Swin-B | 49.38 | 3333 |
| **SDG-KL** | C+L | R-50 | **50.16** | **83** |
| **SDG-Fusion** | C+L | R-50 | **51.66** | 133 |

SDG-Fusion with an R-50 backbone achieves 51.66 mIoU with an inference time of 133ms, and SDG-KL achieves 50.16 mIoU in 83ms, both outperforming other methods that use larger backbones.

### Ablation Study

Comparison of view transformations (as shown in Fig.2 of the paper):
- Valid pixels in the BEV feature map of LSS are <50%, which is highly sparse.
- BEV features generated by SDG view transformation exhibit density and distribution significantly closer to the Ground Truth.

Ablation on fusion methods verifies that neighborhood attention + the gating mechanism outperforms simple concatenation. The region-weighted strategy in active distillation effectively balances knowledge transfer between AR and IR regions.

### Key Findings

- Performing view transformation with 4x downsampled features (rather than higher downsampling ratios) achieves the best performance, as higher downsampling increases semantic and depth ambiguity of pixels.
- SDG view transformation not only improves the quality of BEV features but also reduces the number of virtual points, thereby accelerating inference.
- Through distillation, SDG-KL improves inference speed by 37% (133ms -> 83ms) at the cost of losing only about 1.5 mIoU.
- It also demonstrates comparable performance on the more challenging SurroundOcc-nuScenes dataset.

## Highlights & Insights

- Precisely pinpoints the core bottlenecks of the LSS pipeline—virtual point redundancy and BEV sparsity—and delivers an elegant solution.
- The idea of using semantic segmentation masks to guide depth diffusion is highly ingenious, based on the fact that the same semantic category typically possesses continuous depth values.
- The dual-mode design of SDG-Fusion / SDG-KL is highly practical—the fusion mode is used for high-precision scenarios, while the distillation mode is deployed for real-time scenarios.
- Outperforming methods utilizing Swin-B under an R-50 backbone demonstrates the effectiveness of the framework design.
- The adaptive weighting of AR/IR regions in active distillation is a valuable design detail reference.

## Limitations & Future Work

- The auxiliary semantic segmentation task introduces additional annotation requirements for view transformation (though LiDAR already has semantic labels).
- The diffusion radius $r$ in depth diffusion needs to be manually set and might require tuning for different scenarios.
- Exploration of temporal info is lacking; multi-frame fusion could potentially further improve performance.
- Inference in SDG-KL only utilizes the image branch, but training still requires LiDAR data.
- Robustness under extreme occlusion or adverse weather conditions has not been thoroughly verified.

## Related Work & Insights

- Compared to the BEVFusion series, SDG-OCC improves BEV quality from the source by refining the view transformation, rather than merely performing fusion in the BEV space.
- FlashOcc and FastOcc focus on lightweight pure-visual models, while SDG-KL achieves similar speeds with higher accuracy through distillation.
- CO-Occ uses KNN search to identify co-occurring voxels, whereas the neighborhood attention fusion of SDG-OCC is more flexible.
- It provides valuable insights for view transformation improvements and multimodal fusion strategies in BEV perception.

## Rating

- **Novelty**: 7/10 — The improvements in view transformation and distillation strategies are innovative in their respective areas, but the overall work leans towards incremental progress.
- **Experimental Thoroughness**: 8/10 — Comprehensive comparisons across multiple datasets with intuitive visualizations.
- **Writing Quality**: 7/10 — Clear methodology descriptions, though there are numerous notations and some equation layouts could be clearer.
- **Value**: 8/10 — The balance between real-time performance and accuracy holds practical significance for autonomous driving deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
