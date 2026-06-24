---
title: >-
  [Paper Note] 3D Dental Model Segmentation with Geometrical Boundary Preserving
description: >-
  [CVPR 2025][3D Vision][Dental Segmentation] This paper proposes CrossTooth, which utilizes selective downsampling based on curvature priors (increasing vertex density in boundary areas by 10-15%) and cross-modal boundary feature fusion with multi-view rendered images. It achieves 95.86% mIoU and 82.05% boundary IoU on the public 3DTeethSeg'22 dataset, outperforming the previous SOTA (ToothGroupNet) by 2.3% and 5.7% respectively.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dental Segmentation"
  - "Intraoral Scan"
  - "Selective Downsampling"
  - "Cross-modal Fusion"
  - "Boundary Preserving"
date: 2026-05-08
content_hash: 9d997c3e5182a633
---

# 3D Dental Model Segmentation with Geometrical Boundary Preserving

**Conference**: CVPR 2025  
**arXiv**: [2503.23702](https://arxiv.org/abs/2503.23702)  
**Code**: [https://github.com/XiShuFan/CrossTooth_CVPR2025](https://github.com/XiShuFan/CrossTooth_CVPR2025)  
**Area**: 3D Vision / Medical Image  
**Keywords**: Dental Segmentation, Intraoral Scan, Selective Downsampling, Cross-modal Fusion, Boundary Preserving

## TL;DR
This paper proposes CrossTooth, which utilizes selective downsampling based on curvature priors (increasing vertex density in boundary areas by 10-15%) and cross-modal boundary feature fusion with multi-view rendered images. It achieves 95.86% mIoU and 82.05% boundary IoU on the public 3DTeethSeg'22 dataset, outperforming the previous SOTA (ToothGroupNet) by 2.3% and 5.7% respectively.

## Background & Motivation
**Background**: 3D Intraoral Scan (IOS) meshes are widely used for digital dental diagnosis. Tooth segmentation is a critical preprocessing step, and deep learning methods have achieved high precision in segmenting tooth crown regions.

**Limitations of Prior Work**: **The segmentation accuracy at the crown-gingiva junction is significantly lower than average.** Reasons: (a) Existing downsampling methods (QEM/FPS/Voxel Grid) are uniform, discarding crucial geometric details at the junction. (b) 3D features based solely on coordinates and normal vectors struggle to fully express the subtle boundary areas. Physically, the junction exhibits distinctive negative curvature characteristics, and boundaries are much clearer in 2D rendered images.

**Key Challenge**: Downsampling is necessary (100K+ points to 16K points), but uniform downsampling does not distinguish between boundary and non-boundary regions, causing boundary information to be diluted.

**Goal**: To retain more geometric details of the boundary regions while keeping the downsampled point count at 16K, and introduce cross-modal information to enhance boundary features.

**Key Insight**: It is observed that (a) the junction exhibits distinctive negative curvature characteristics that can serve as a prior; (b) the intensity variation in 2D rendered images is clearer at the junction (strong shadow contrast between the tooth crown and gum under vertical parallel light).

**Core Idea**: Using curvature-weighted selective downsampling to preserve boundary vertices, combined with multi-view rendered images to provide cross-modal boundary features.

## Method

### Overall Architecture
CrossTooth consists of two branches: (1) **Point Cloud Branch**: Performs multi-scale encoder-decoder processing (based on Point Transformer) on the selectively downsampled intraoral scan point cloud (16K points, 6D features = coordinates + normal vectors); (2) **Image Branch**: Conducts semantic segmentation (based on PSPNet) on multi-view rendered images (96 views) to generate pixel-level segmentation results. Finally, image features are projected back to the point cloud using 2D-3D correspondences, and both features are fused with an MLP to output 17-class segmentation results.

### Key Designs

1. **Selective Downsampling**:

    - **Function**: Introduces curvature prior in QEM downsampling to preserve more vertices at the boundary regions.
    - **Mechanism**: Modifies the edge collapse cost function of QEM by multiplying negative curvature edges (junctions) with a large coefficient $k=10$ and positive curvature edges (crown top) with a small coefficient $k=1$. Consequently, the collapse cost of negative curvature edges is high, placing them later in the priority queue and preserving boundary vertices. Curvature is recomputed after each iteration.
    - **Design Motivation**: In intraoral scans, the crown-gum junction possesses distinctive negative curvature characteristics, which can serve as a natural prior without requiring extra annotations. Experiments demonstrate that the vertex density in boundary areas increases by 10-15%.

2. **Multi-View Image Rendering and Feature Extraction**:

    - **Function**: Uniformly places 96 virtual cameras in the hemisphere above the intraoral scan model, rendering 96 images using vertically downward white parallel light.
    - **Mechanism**: The parallel light produces a strong shading contrast at the crown-gingiva junction (the protruding crown forms a bright area, while the receding gingiva forms a shadow), making boundary details in the 2D images more discriminative than in the 3D point cloud. PSPNet is used to perform 17-class semantic segmentation on each rendered image ($C \times H \times W = 17 \times 1024 \times 1024$).
    - **Design Motivation**: Image downsampling naturally preserves edge info (due to the edge-aware nature of convolution operations), whereas point cloud downsampling tends to lose boundaries.

3. **Cross-modal Feature Fusion + Boundary-Aware Loss**:

    - **Function**: Establishes a 3D-to-2D mapping using camera parameters to project segmentation results of multi-view images back to the point cloud. Image features are encoded as one-hot vectors, concatenated with the last feature layer of the point cloud decoder, and fused using an MLP.
    - **Boundary Loss**: Additionally predicts a binary boundary mask (if over half of the $k=8$ nearest neighbors belong to a different class, the point is classified as a boundary point), and utilizes Contrastive Boundary Learning (CBL) loss to enforce the inter-class separability of boundary point features.
    - **Total Loss**: $L = L_{CE}(image, point) + L_{CBL}(point)$

### Loss & Training
- Dataset: 3DTeethSeg'22 (1800 dental models, split into 1440/360)
- Input: 16K points $\times$ 6D features + 96 rendered images
- 100 epochs, Adam, lr=1e-3, cosine decay, batch=4, RTX 3090

## Key Experimental Results

### Main Results: 3DTeethSeg'22 Dataset

| Method | mIoU (%) | Boundary IoU (%) |
|------|:---:|:---:|
| MeshSegNet | 66.13 | 40.13 |
| TSegNet | 57.24 | 27.04 |
| SimpSegNet | 88.45 | 59.95 |
| ToothGroupNet | 93.55 | 65.13 |
| DilatedSegNet | 91.44 | 62.70 |
| **CrossTooth** | **95.86** | **82.05** |

### Ablation Study

| Configuration | mIoU (%) | Boundary IoU (%) | Description |
|------|:---:|:---:|------|
| CrossTooth-point | 95.12 | 81.57 | Point cloud branch only |
| CrossTooth-pixel | 89.49 | - | Image branch only |
| CrossTooth (full) | **95.86** | **82.05** | Best with cross-modal fusion |

### Key Findings
- Selective downsampling increases the vertex density in boundary areas by 10-15% (reducing average distance by ~15%), which is the core contribution to the boost in boundary IoU—CrossTooth's boundary IoU is 16.9 percentage points higher than ToothGroupNet's.
- Image feature fusion yields an incremental gain of 0.7% mIoU and 0.5% boundary IoU—small but stable and effective, potentially limited by the simple MLP fusion strategy.
- The image-only branch (89.49% mIoU) is significantly lower than the point-only branch (95.12%), indicating that 3D geometry remains the primary source of information, with images playing a complementary role.
- The general effectiveness of selective downsampling and image fusion is also validated on more baselines such as TSGCNet/HiCANet and the 3D-IOSSeg dataset.
- Increasing the number of rendered images from 0 $\to$ 32 $\to$ 96 $\to$ 128 progressively improves performance, but at 128, the boundary IoU begins to decrease (excessive viewpoints introduce redundancy).

## Highlights & Insights
- **The selective downsampling concept is simple yet elegant**: By only modifying one hyperparameter—the cost coefficient of QEM—it allocates more "budget" to boundary regions without extra networks, annotations, or increasing the total point count. This approach can be generalized to other 3D processing tasks that require preserving local details within a limited budget.
- **Physical intuition in lighting design**: Choosing vertically downward parallel light to create the strongest contrast at the crown-gingiva junction reflects deep domain understanding.
- CrossTooth's FLOPs are only 5.05G (excluding PSPNet's 7.08G), representing the lowest computational cost among all methods (compared to TSGCNet's 174.85G), demonstrating that careful design is more important than simply stacking parameters.

## Limitations & Future Work
- Struggles with missing teeth/few-teeth scenarios—the correlation between global tooth information and boundary information is disrupted in the presence of missing teeth.
- Low extraction accuracy for wisdom teeth due to scarce training samples; few-shot learning could be considered.
- Cross-modal fusion is only performed in the final layer using a simple MLP concatenation; more fine-grained multi-level fusion (e.g., cross-modal attention at the encoder-decoder stage) could potentially improve results.
- Rendering 96 images introduces extra computation during inference; more efficient viewpoint selection strategies can be explored.

## Related Work & Insights
- **vs ToothGroupNet**: ToothGroupNet uses a two-stage strategy of tooth detection and region segmentation, achieving 93.55% mIoU but only 65.13% boundary IoU. CrossTooth directly targets the boundary challenges through selective downsampling, reaching 82.05% boundary IoU.
- **vs TSGCNet**: This dual-branched graph convolutional network processes coordinates and normal vectors separately but ignores the impact of downsampling on boundaries. CrossTooth resolves this issue at the data preprocessing stage.
- **vs 2DPASS (LiDAR segmentation)**: Also utilizes 2D-3D cross-modal fusion to boost segmentation performance, but 2DPASS targets autonomous driving scenarios, whereas CrossTooth is designed for the specific challenges of medical dental scenes.

## Rating
- Novelty: ⭐⭐⭐ Both selective downsampling and cross-modal fusion have precedents, but their combined application specifically targeting the dental boundary challenge is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two public datasets, 7+ baselines, complete ablations (number of images, downsampling methods).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the problem, rich visualizations.
- Value: ⭐⭐⭐ Application-oriented work with direct practical value for the digital dentistry community, though the generalizability of the method is somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)
- [\[CVPR 2025\] COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)
- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)
- [\[CVPR 2025\] Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [\[CVPR 2026\] Photo-Guided Tooth Segmentation on 3D Oral Scan Model](../../CVPR2026/3d_vision/photo-guided_tooth_segmentation_on_3d_oral_scan_model.md)

</div>

<!-- RELATED:END -->
