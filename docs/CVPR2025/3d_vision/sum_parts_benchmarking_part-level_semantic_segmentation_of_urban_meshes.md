---
title: >-
  [Paper Note] SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes
description: >-
  [CVPR 2025][3D Vision][Urban mesh semantic segmentation] Proposes SUM Parts, the first large-scale benchmark dataset for part-level semantic segmentation of urban textured meshes (covering $2.5\,\text{km}^2$ with 21 categories), featuring two types of labels (face annotation and texture pixel annotation), and develops an efficient interactive annotation tool combining 3D and 2D template matching.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Urban mesh semantic segmentation"
  - "part-level annotation"
  - "texture annotation"
  - "interactive annotation tool"
  - "benchmark dataset"
date: 2026-05-08
content_hash: 89bb7488f1975eb1
---

# SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes

**Conference**: CVPR 2025  
**arXiv**: [2503.15300](https://arxiv.org/abs/2503.15300)  
**Code**: [GitHub](https://tudelft3d.github.io/SUMParts/)  
**Area**: 3D Vision  
**Keywords**: Urban mesh semantic segmentation, part-level annotation, texture annotation, interactive annotation tool, benchmark dataset

## TL;DR

Proposes SUM Parts, the first large-scale benchmark dataset for part-level semantic segmentation of urban textured meshes (covering $2.5\,\text{km}^2$ with 21 categories), featuring two types of labels (face annotation and texture pixel annotation), and develops an efficient interactive annotation tool combining 3D and 2D template matching.

## Background & Motivation

- Semantic segmentation of urban scenes primarily focuses on images and point clouds, while textured mesh, as a richer spatial representation, has been severely overlooked.
- Existing mesh datasets mainly focus on small-scale indoor scenes, lacking fine-grained semantic labels for large-scale outdoor environments.
- There is a lack of part-level semantic segmentation in urban scene understanding, such as the subdivision of functional components like windows, chimneys, and road markings.
- Most existing 3D annotation methods only label mesh vertices or faces, ignoring the richer detail information in texture images.
- Textured meshes offer better resolution and completeness compared to point clouds, but lack corresponding part-level annotated datasets.
- Manual cost for annotating large-scale 3D scenes is extremely high, necessitating the development of more efficient interactive annotation tools.
- Inspired by the international CityGML standard, part-level segmentation is crucial for automated city modeling.
- Existing interactive annotation methods (such as SAM) show limited performance in 3D mesh texture scenes, especially regarding rotation and scale invariance.

## Method

### Overall Architecture

The SUM Parts system consists of two core modules: the **face-based annotation module** and the **texture-based annotation module**, which perform part-level semantic annotation on triangular mesh faces and texture pixels respectively. The annotation is based on textured meshes reconstructed from aerial surveys of Helsinki (with a ground sampling distance of approximately $7.5\,\text{cm}$), covering 40 tiles totaling around $2.5\,\text{km}^2$, and defines 13 face labels and 8 additional texture pixel labels (21 categories in total). The system also proposes a superpixel texture sampling strategy to map texture annotations to point clouds for semantic segmentation evaluation.

### Key Designs

**Design 1: Interactive 3D Protrusion Selection**
- **Function**: Interactively extracts non-planar structures (such as protrusions like chimneys and balconies) from over-segmented planar fragments.
- **Mechanism**: Protrusion extraction is formulated as a binary labeling problem $l^f = \{\text{support plane}, \text{protrusion}\}$. A dual graph $\mathcal{G}^f$ of candidate faces is constructed, designing a data term $D^f$ containing geometric features and a smoothness term $V^f$ based on the shrinking ball radius. The target function $E^f(l^f) = \sum_i D^f(l^f_i) + \lambda^f \sum_{\{i,j\}} V^f(l^f_i, l^f_j)$ is optimized via graph cut.
- **Design Motivation**: Traditional over-segmentation methods are prone to under-segmentation or over-segmentation in non-planar areas, sharp features, and small-scale structures; binary labeling based on graph cuts can more accurately separate protrusions from the support plane.

**Design 2: 3D Template Matching**
- **Function**: Achieves batch annotation by leveraging repetitive structures in urban scenes (such as window arrangements).
- **Mechanism**: After the user selects a template, structure-aware feature matching is performed separately for planar fragments and protrusions. Planar matching is based on feature vectors $\mathbf{F}^{(\text{seg})}$ such as geometric uniformity, spatial distribution, orientation, and sphericity; protrusion matching first decomposes the template into planar fragment seeds, then expands candidate regions through spatial and scale constraints, performing matching based on feature vectors $\mathbf{F}^{(\text{str})}$ such as compactness and surface complexity.
- **Design Motivation**: Urban architecture exhibits high repetitiveness; template matching significantly reduces manual interaction, improving annotation efficiency by approximately 1.73 times.

**Design 3: Interactive 2D Texture Selection and Template Matching**
- **Function**: Achieves rotation- and scale-invariant region selection and matching on texture images.
- **Mechanism**: SLIC is first used to generate superpixels, and a user click triggers local expansion (graph-cut optimization based on GMM Wasserstein distance), followed by GrabCut for pixel-level fine segmentation. 2D template matching utilizes regional structural features (shape index, shape regularity, contextual color features) instead of NCC to achieve rotation and scale invariance.
- **Design Motivation**: NCC performs poorly in 3D urban scenes with rotation and scale variations; matching methods based on regional structural features balance both robustness and efficiency.

### Loss & Training

The system utilizes an energy-function-based optimization framework: the face annotation energy $E^f$ consists of a data term (based on protrusion score $p_i = d_i + \omega_i \theta_i$) and a smoothness term (based on shrinking ball radius difference $R_{i,j}$); the texture annotation energy $E^s$ consists of a superpixel color similarity data term (Wasserstein distance) and a color difference smoothness term (CIEDE2000 color difference), both minimized via the graph cut algorithm.

## Key Experimental Results

### Main Results: Comparison of 3D Semantic Segmentation Methods (mIoU / mAcc)

| Method | Face mIoU | Face mAcc | Pixel mIoU | Pixel mAcc |
|------|-----------|-----------|------------|------------|
| PointNet | 15.1 | 22.0 | 2.6 | 9.8 |
| PointNet++ | 33.1 | 46.9 | 24.7 | 35.2 |
| SparseUNet | 60.5 | 71.7 | 34.5 | 45.1 |
| KPConv | 57.5 | 64.7 | 42.6 | 58.3 |
| PointNext | 65.3 | 77.2 | 44.7 | 57.6 |
| PointTransV3 | 59.1 | 70.2 | 38.0 | 54.1 |
| **PointVector** | **70.0** | **80.7** | **47.9** | **63.8** |

### Ablation Study: Impact of Sampling Strategies (Mean Face Labeling mIoU)

| Sampling Method | Face Avg. mIoU | Pixel Avg. mIoU |
|----------|---------------|-----------------|
| Face-centered | 48.0 | - |
| Random | 40.7 | 30.6 |
| Poisson-disk | 38.2 | 27.8 |
| Superpixel (Ours) | 44.0 | **31.5** |

### Key Findings
- PointVector achieves the best performance on both face annotation and pixel annotation tracks (mIoU of 70.0% and 47.9%).
- Face-centered sampling is optimal for face annotation because triangle density naturally adapts to geometric complexity; however, it is not suitable for texture annotation.
- The proposed superpixel texture sampling outperforms other sampling methods in pixel annotation.
- The interactive annotation tool speeds up face annotation by 1.73 times, with the intelligent interaction usage rate exceeding 80%.

## Highlights & Insights

1. **Pioneering Part-level Urban Mesh Benchmark**: Fills the gap in part-level semantic segmentation of urban textured meshes, with a 21-class labeling system conforming to the international CityGML standard.
2. **Bimodal Annotation Innovation**: Dual-track parallel of face and texture annotation, where texture annotation captures details that face annotation cannot represent (e.g., road markings).
3. **Unsupervised Interactive Annotation**: Does not rely on large-scale pre-trained data; achieves efficient annotation through geometric features and template matching, showing better generalization than deep learning methods like SAM.
4. **Practical Superpixel Sampling**: Bridges the gap between texture annotation and point cloud segmentation methods.

## Limitations & Future Work

- Relies on geometric accuracy and structural clarity, with limited effectiveness on topological errors or low-resolution meshes.
- Inapplicable to natural scenes (mountains) or complex irregular structures (palaces), relying heavily on planar and protrusion assumptions.
- Performance of texture annotation degrades on complex textures, cluttered backgrounds, shaded areas, and regions with low color contrast.
- The highest mIoU for the pixel annotation track is only 47.9%, leaving significant room for improvement.
- Future work can integrate foundation models (such as SAM2) to enhance interactive annotation efficiency or extend to cross-domain generalization across more cities.

## Related Work & Insights

- Unlike indoor mesh datasets such as ScanNet, this work focuses on part-level annotation in large-scale outdoor urban scenes.
- Directly promotes the automated modeling of CityGML LoD3.
- The superpixel sampling strategy can be generalized to other tasks that require mapping texture information to 3D representations.

## Rating

⭐⭐⭐⭐ — A high-quality benchmark work that fills an important gap in part-level segmentation of urban meshes; the annotation tool is exquisitely designed but its application scenarios are limited by regular architectural structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)
- [\[CVPR 2025\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](rewis3d_reconstruction_improves_weakly-supervised_semantic_segmentation.md)
- [\[CVPR 2025\] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)
- [\[ECCV 2024\] 3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](../../ECCV2024/3d_vision/3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)

</div>

<!-- RELATED:END -->
