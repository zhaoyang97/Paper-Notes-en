---
title: >-
  [Paper Note] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Scene Segmentation] The authors propose Unified-Lift, an end-to-end, object-aware 2D-to-3D segmentation method based on 3DGS. By learning the association between a global object-level codebook and Gaussian-level features, it eliminates the dependency of existing methods on pre- and post-processing, significantly outperforming the state-of-the-art in multi-view consistent instance segmentation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Scene Segmentation"
  - "Gaussian Splatting"
  - "2D-to-3D Lifting"
  - "Object-level Codebook"
  - "End-to-End Segmentation"
date: 2026-05-08
content_hash: 72751038680d506b
---

# Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.14029](https://arxiv.org/abs/2503.14029)  
**Code**: [GitHub](https://github.com/Runsong123/Unified-Lift)  
**Area**: 3D Vision  
**Keywords**: 3D Scene Segmentation, Gaussian Splatting, 2D-to-3D Lifting, Object-level Codebook, End-to-End Segmentation

## TL;DR

The authors propose Unified-Lift, an end-to-end, object-aware 2D-to-3D segmentation method based on 3DGS. By learning the association between a global object-level codebook and Gaussian-level features, it eliminates the dependency of existing methods on pre- and post-processing, significantly outperforming the state-of-the-art in multi-view consistent instance segmentation.

## Background & Motivation

Lifting 2D segmentations from foundational models (such as SAM) to 3D radiance fields is an effective way to achieve 3D scene understanding. However, 2D instance segmentation lacks consistency across different views (the same object has different IDs from different angles) and suffers from under-/over-segmentation issues, posing conflicting supervision for the lifting process.

Existing methods fall into three categories: (1) End-to-end methods (e.g., Panoptic Lifting) which use Hungarian matching to obtain pseudo-labels but are sensitive to matching results; (2) Two-stage methods which establish cross-view correspondences in a preprocessing stage but suffer from error accumulation; (3) Contrastive learning + post-processing methods which employ contrastive learning to encode instance information into a feature field and then utilize HDBSCAN clustering to extract final segmentations, though clustering is sensitive to hyperparameters and introduces errors.

**Core Problem**: Can an end-to-end framework be designed to achieve accurate 3D scene segmentation without any pre- or post-processing? Unified-Lift addresses this question by introducing a learnable global object-level codebook and specialized learning strategies.

## Method

### Overall Architecture

Unified-Lift is built upon the 3DGS representation and consists of three components: (1) augment each Gaussian point with a learnable feature optimized via contrastive learning (Gaussian-level features); (2) introduce a global object-level codebook for segmentation prediction through association with Gaussian-level features (object-level understanding); (3) design an association learning module and a noisy label filtering module to achieve effective codebook learning. During inference, features are rendered directly $\rightarrow$ similarity with the codebook is computed $\rightarrow$ the codebook index with the highest similarity is taken as the instance ID, without requiring any post-processing.

### Key Designs 1: Object-level Codebook Representation

**Function**: Provide explicit object-level 3D scene understanding, replacing clustering-based post-processing.

**Mechanism**: Define a learnable codebook matrix $\mathbf{F}_{obj} \in \mathbb{R}^{L \times d}$, where each row corresponds to an object in the 3D scene. For the rendered Gaussian-level feature $\mathbf{F}_u$, the probability distribution is computed via softmax similarity: $\mathbf{P}_u = \text{softmax}(\text{sim}(\mathbf{F}_u, \mathbf{F}_{obj}))$, where $\text{sim}$ denotes the dot product. During inference, the index with the maximum probability is taken as the instance ID.

**Design Motivation**: Gaussian-level features only implicitly encode instance information, necessitating clustering post-processing. The codebook provides an explicit object-level representation, making segmentation prediction a simple matching between features and the codebook, thereby eliminating the hyperparameter sensitivity and error accumulation of clustering.

### Key Designs 2: Association Learning Module

**Function**: Generate multi-view consistent pseudo-labels and provide robust constraints for codebook optimization.

**Mechanism**: Two key improvements: (1) **Area-Aware ID Mapping**: The objective function in Hungarian matching is improved by removing the normalization term $1/|\Omega_j|$ used in Panoptic Lifting, allowing large-area segmentation masks to dominate the mapping process to enhance multi-view consistency; (2) **Concentration Constraint**: In addition to the cross-entropy loss $\mathcal{L}_{\text{class}}$ for sparsity, an L1 distance constraint is added to align the direction of the codebook features with their corresponding Gaussian-level features: $\mathcal{L}_{\text{concen}} = \frac{1}{|\Omega|} \sum \|\mathbf{F}_{obj}^{\Pi^*(K_u)} - \mathbf{F}_u / \|\mathbf{F}_u\|\|_1$.

**Design Motivation**: Normalization gives small-area segmentations a disproportionate impact on mapping, leading to multi-view inconsistency. Removing normalization allows large objects to dominate the mapping, while small objects are progressively and correctly associated through learning in subsequent training. The concentration constraint ensures that codebook features align in direction with Gaussian features, complementing the dot-product similarity metric of contrastive learning.

### Key Designs 3: Noisy Label Filtering Module

**Function**: Enhance robustness against 2D segmentation noise (under-/over-segmentation).

**Mechanism**: Utilize the learned Gaussian-level features to estimate an uncertainty map in a self-supervised manner. For each pixel, its Gaussian-level feature is compared for consistency with the features of other pixels within the same mask; inconsistent regions are designated as highly uncertain. Segmentation labels of highly uncertain regions are down-weighted or filtered during training.

**Design Motivation**: Segmentation masks produced by 2D models like SAM are imperfect (under-/over-segmented); using them directly as supervision introduces noisy gradients. Utilizing feature consistency for self-supervised uncertainty estimation allows identifying and filtering noisy labels without requiring additional annotations.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{\text{contra}} + \lambda_1 \mathcal{L}_{\text{class}} + \lambda_2 \mathcal{L}_{\text{concen}} + \mathcal{L}_{\text{photo}}$, where $\mathcal{L}_{\text{contra}}$ is the InfoNCE contrastive loss, $\mathcal{L}_{\text{class}}$ represents the cross-entropy sparsity constraint, $\mathcal{L}_{\text{concen}}$ represents the concentration constraint, and $\mathcal{L}_{\text{photo}}$ denotes the photometric loss of 3DGS.

## Key Experimental Results

### Main Results: LERF-Masked Dataset

| Method | mAP ↑ | mAP50 ↑ | mAP75 ↑ | Time (min) ↓ |
|------|-------|---------|---------|-----------|
| Panoptic Lifting | 36.7 | 60.3 | 36.2 | ~120 |
| Contrastive Lift | 52.3 | 72.6 | 56.2 | ~60 |
| SAGA (+ HDBSCAN) | 56.5 | 79.1 | 57.5 | ~30 |
| OmniSeg3D (+ HDBSCAN) | 61.3 | 82.0 | 65.2 | ~35 |
| **Unified-Lift** | **67.4** | **87.1** | **73.8** | **~20** |

### Replica Dataset

| Method | mPQ ↑ | mSQ ↑ | mRQ ↑ |
|------|-------|-------|-------|
| Panoptic Lifting | 33.7 | 66.8 | 47.4 |
| Contrastive Lift | 39.1 | 68.3 | 55.2 |
| OmniSeg3D | 45.7 | 71.5 | 62.9 |
| **Unified-Lift** | **52.3** | **74.5** | **69.1** |

### Ablation Study: Contributions of Each Component (LERF-Masked mAP)

| Configuration | mAP ↑ |
|------|-------|
| Contrastive Learning + HDBSCAN | 61.3 |
| + Baseline Codebook Strategy | 63.1 |
| + Area-Aware ID Mapping | 64.8 |
| + Concentration Constraint | 66.2 |
| + Noisy Label Filtering | **67.4** |

### Key Findings

- Unified-Lift outperforms all other methods in mAP by 6.1 points on LERF-Masked, while achieving the fastest inference speed (as it requires no clustering post-processing).
- Area-aware ID mapping significantly enhances multi-view consistency, with a particularly pronounced improvement in small-object segmentation.
- Excellent scalability is demonstrated on the Messy Rooms dataset (with over 500 objects per scene).
- Eliminating post-processing not only improves accuracy but also yields a cleaner inference pipeline with fewer hyperparameters.

## Highlights & Insights

1. **End-to-End Design**: The first 2D-to-3D instance segmentation lifting method that requires no pre- or post-processing, eliminating error accumulation and hyperparameter tuning.
2. **Codebook Replacing Clustering**: Using a learnable matrix to replace HDBSCAN clustering not only improves accuracy but also simplifies the workflow.
3. **Intuition of Area-Aware Mapping**: Eliminating a single normalization term significantly enhances multi-view consistency, which is simple yet highly effective.

## Limitations & Future Work

- The codebook size $L$ must be preset as the maximum number of objects in the scene, which requires estimation for unseen scenes.
- Joint optimization of contrastive learning and codebook learning may exhibit training instability.
- The performance depends heavily on the quality of SAM's 2D segmentation, which may be suboptimal for severely occluded or textureless objects.
- Currently, the method only handles instance segmentation; semantic category information is not yet addressed.

## Related Work & Insights

- **SAGA / OmniSeg3D**: State-of-the-art contrastive learning + clustering methods; this work demonstrates that the end-to-end codebook scheme comprehensively outperforms them in both accuracy and efficiency.
- **Panoptic Lifting**: An early end-to-end method whose Hungarian matching strategy is improved by the area-aware version introduced in this paper.
- **VQ-VAE / Codebook Learning**: The concept of discrete codebooks is introduced here to represent continuous features at the object level.

## Rating

⭐⭐⭐⭐ — The end-to-end design successfully eliminates the pain points of post-processing. The codebook representation is elegant, and the area-aware mapping is simple yet effective. It achieves significant performance gains and efficiency improvements across multiple datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)
- [\[CVPR 2025\] End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [\[ICML 2026\] EPS3D: End-to-End Feed-Forward 3D Panoptic Segmentation](../../ICML2026/3d_vision/eps3d_end-to-end_feed-forward_3d_panoptic_segmentation.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[ICLR 2026\] Mesh Splatting for End-to-end Multiview Surface Reconstruction](../../ICLR2026/3d_vision/mesh_splatting_for_end-to-end_multiview_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
