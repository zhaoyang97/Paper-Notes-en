---
title: >-
  [Paper Note] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This work proposes Click-Gaussian, which learns a discriminative 3D feature field with two-level granularity (coarse/fine) and combines it with Global Feature-guided Learning (GFL) to address cross-view mask inconsistency. It achieves real-time interactive 3D Gaussian segmentation at only 10ms per click, which is 15-130 times faster than existing methods while significantly improving segmentation accuracy.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Interactive Segmentation"
  - "Feature Fields"
  - "Contrastive Learning"
  - "View Consistency"
date: 2026-05-08
content_hash: 3e42ace39311574e
---

# Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

**Conference**: ECCV 2024  
**arXiv**: [2407.11793](https://arxiv.org/abs/2407.11793)  
**Code**: [https://seokhunchoi.github.io/Click-Gaussian](https://seokhunchoi.github.io/Click-Gaussian) (Project Page)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Interactive Segmentation, Feature Fields, Contrastive Learning, View Consistency

## TL;DR

This work proposes Click-Gaussian, which learns a discriminative 3D feature field with two-level granularity (coarse/fine) and combines it with Global Feature-guided Learning (GFL) to address cross-view mask inconsistency. It achieves real-time interactive 3D Gaussian segmentation at only 10ms per click, which is 15-130 times faster than existing methods while significantly improving segmentation accuracy.

## Background & Motivation

3D Gaussian Splatting (3DGS) has gained significant attention in 3D scene manipulation due to its real-time rendering capability, where accurate segmentation of objects in scenes is a fundamental requirement for applications like editing and virtual reality. Existing 3DGS-based segmentation methods face several limitations of prior work:

*   **Time-consuming post-processing**: Methods like SAGA require extensive post-processing on noisy segmentation outputs to obtain clean results, which severely constrains the efficiency benefits of 3DGS real-time rendering.
*   **Difficulty in fine-grained segmentation**: Existing methods struggle to provide precise segmentation results, failing to meet the demands of fine-grained 3D scene manipulation.
*   **Cross-view mask inconsistency**: 2D segmentation results obtained independently from different views conflict with each other, hindering 3D feature learning.

**Key Challenge**: How to learn a feature field that is both discriminative in 3D space and view-consistent, without relying on time-consuming post-processing?

**Key Insight**: Design a two-level granularity feature field representation combined with a global feature-guided learning strategy to fundamentally eliminate the reliance on post-processing.

## Method

### Overall Architecture

Based on pre-trained 3DGS, Click-Gaussian augments each 3D Gaussian with a $D$-dimensional feature vector $\mathbf{f}_i \in \mathbb{R}^D$, split into coarse-level and fine-level features. Employing SAM to automatically generate 2D masks across all training views, it constructs two-level masks, trains the feature field via contrastive learning, and introduces GFL to ensure cross-view consistency. Once trained, user clicks can complete segmentation within 10ms.

### Key Designs

1.  **Two-level Granularity Feature Fields**:
    *   Each Gaussian's feature $\mathbf{f}_i$ is decomposed into a coarse-level feature $\mathbf{f}_i^c \in \mathbb{R}^{D^c}$ and a complementary part $\bar{\mathbf{f}}_i^c \in \mathbb{R}^{D-D^c}$.
    *   The coarse-level feature is directly $\mathbf{f}_i^c$, and the fine-level feature is acquired via concatenation as $\mathbf{f}_i^f = \mathbf{f}_i^c \oplus \bar{\mathbf{f}}_i^c$.
    *   **Granularity Prior**: The fine-level feature contains the coarse-level feature, embodying hierarchical dependencies in the real world—if A and B differ at the coarse level, then $a \subset A$ and $b \subset B$ are naturally also different at the fine level.
    *   In experiments, $D^c=12$ and $D=24$ are set, and both level features are computed in a single forward pass through the 3DGS rasterizer.

2.  **Contrastive Learning**:
    *   Positive contrastive loss: For pixel pairs within the same mask region, maximize the cosine similarity of their rendered features.
        $\mathcal{L}_{\text{pos}}^{\text{cont}} = -\frac{1}{|P_1||P_2|}\sum_l \sum_{p_1}\sum_{p_2} \mathbb{1}[M_{p_1}^l = M_{p_2}^l] \mathbf{S}^l(p_1, p_2)$
    *   Negative contrastive loss: For pixel pairs from different mask regions, constrain the cosine similarity not to exceed a threshold $\tau^l$ ($\tau^f=0.75$, $\tau^c=0.5$).
    *   A stop-gradient operation is applied to the fine-level negative contrastive loss to prevent coarse-level features from being mistakenly updated during fine-level differentiation.

3.  **Global Feature-guided Learning (GFL)**:
    *   **Global Feature Candidate Generation**: After a specified number of iterations, 2D feature maps are rendered for all training views, and average pooling is performed on each mask region to obtain the average feature set $\mathcal{F}^l$.
    *   **HDBSCAN Clustering**: HDBSCAN clustering is performed on $\mathcal{F}^l$ to yield $C^l$ global feature candidates $\tilde{\mathcal{F}}^l$. These cluster centers represent the most typical features in the entire scene, effectively smoothing out cross-view noise.
    *   **GFL Loss**: Guides each Gaussian feature close to its most probable global cluster while keeping it away from others.
        $\mathcal{L}_{\text{pos}}^{\text{GFL}} = -\frac{1}{N}\sum_l\sum_i \mathbb{1}[\tilde{\mathbf{S}}^l(i,c_i^l) > \tau^g] \tilde{\mathbf{S}}^l(i,c_i^l)$
    *   Clusters are updated periodically, with $\tau^g=0.9$.

### Loss & Training

Total loss function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cont}} + \lambda_1 \mathcal{L}_{\text{GFL}} + \lambda_2 \mathcal{L}_{\text{3D-norm}} + \lambda_3 \mathcal{L}_{\text{2D-norm}} + \lambda_4 \mathcal{L}_{\text{spatial}}$$

The regularization terms include:
*   **Hyperspherical Regularization** $\mathcal{L}_{\text{3D-norm}}$: Constrains the $L_2$-norm of each Gaussian feature to 1 to prevent overly large feature norms from dominating the rendering.
*   **Rendered Feature Regularization** $\mathcal{L}_{\text{2D-norm}}$: Ensures that multiple Gaussian features corresponding to a single pixel align in direction.
*   **Spatial Consistency Regularization** $\mathcal{L}_{\text{spatial}}$: Utilizes KD-tree queries for spatial nearest neighbors to enforce similarity between adjacent Gaussian features.

Hyperparameters: $\lambda_1=10.0$, $\lambda_2=0.2$, $\lambda_3=0.2$, $\lambda_4=0.5$. Training takes 3000 iterations, with GFL starting from the 2000th iteration, costing approximately 13 minutes in total (NVIDIA RTX A5000).

## Key Experimental Results

### Main Results

**LERF-Mask Dataset (mIoU %)**:

| Method | Coarse Avg | Fine Avg | All Avg |
|------|-----------|---------|---------|
| Gau-Group | 72.8 | 31.5 | 49.9 |
| OmniSeg3D | 79.4 | 46.9 | 60.9 |
| Feature3DGS | 65.6 | 63.5 | 63.4 |
| GARField | 80.9 | 71.4 | 75.2 |
| **Click-Gaussian** | **89.1** | **84.3** | **85.4** |

**SPIn-NeRF Dataset (mIoU %)**:

| Method | mIoU |
|------|------|
| MVSeg | 90.9 |
| SA3D | 92.4 |
| SAGA | 88.0 |
| **Click-Gaussian** | **94.0** |

### Ablation Study

| Configuration | All mIoU | Relative Change |
|------|---------|---------|
| w/o $\mathcal{L}_{\text{2D-norm}}$ | 83.2 | -2.6% |
| w/o $\mathcal{L}_{\text{3D-norm}}$ | 80.3 | -6.0% |
| w/o $\mathcal{L}_{\text{spatial}}$ | 82.0 | -4.0% |
| w/o $\mathcal{L}_{\text{GFL}}$ | 58.6 | **-31.3%** |
| w/o prior | 82.1 | -3.9% |
| **Full Model** | **85.4** | - |

### Key Findings

*   GFL is the most critical component, with performance dropping by 31.3% when removed, particularly in fine-level segmentation where it drops from 84.3 to 42.3.
*   The granularity prior boosts fine-level segmentation by 6 percentage points (78.3 $\to$ 84.3).
*   Hyperspherical regularization is particularly crucial for fine-level segmentation (74.1 vs 84.3).
*   Inference speed: ~10ms per click, which is 130 times faster than GARField and 15 times faster than SAGA.

## Highlights & Insights

1.  **Elegant granularity prior design**: Designing the fine-level feature as a superset of the coarse-level feature gracefully encodes real-world hierarchical dependencies.
2.  **GFL fundamentally resolves the cross-view consistency issue**: It provides stable supervision signals through global clustering rather than relying on noisy single-view masks.
3.  **High practicality**: The 10ms-level interactive response time makes true real-time 3D editing possible.
4.  **Clever use of stop gradient**: Applying stop-gradient to the coarse-level components during fine-level negative contrastive learning avoids gradient conflicts between coarse and fine features.

## Limitations & Future Work

1.  Dependence on the quality of pre-trained 3DGS; if a single Gaussian represents multiple objects with different semantics but similar colors, feature learning is hindered.
2.  Having only two levels of granularity lacks intermediate stages, which may require multiple interactions for complex scenes demanding multi-level segmentation.
3.  Lack of semantic understanding; learning features purely on appearance and space prevents open-vocabulary segmentation.
4.  Future work can consider extending the approach to dynamic scenes or 4D Gaussian segmentation.

## Related Work & Insights

*   Comparison with SAGA demonstrates that post-processing is a bottleneck, and learning discriminative features end-to-end is a superior direction.
*   The GFL concept can be generalized to other tasks requiring cross-view consistency (such as 3D editing and semantic understanding).
*   Inspiration from two-level granularity prior: Encoding explicit hierarchical dependencies in multi-scale learning is more effective than independent learning.

## Rating

*   Novelty: ⭐⭐⭐⭐ (Novel designs in GFL and granularity prior, while the overall framework is standard)
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Evaluated on multiple datasets, baseline models, comprehensive ablation studies, and application demonstrations)
*   Writing Quality: ⭐⭐⭐⭐ (Clear logic with well-formulated equations)
*   Value: ⭐⭐⭐⭐ (10ms interactive segmentation holds high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](../../CVPR2025/3d_vision/isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[ECCV 2024\] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)
- [\[ECCV 2024\] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally](flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Human Hair Reconstruction with Strand-Aligned 3D Gaussians](human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)

</div>

<!-- RELATED:END -->
