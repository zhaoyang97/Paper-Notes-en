---
title: >-
  [Paper Note] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering
description: >-
  [NeurIPS 2025][Visual Place Recognition] This paper proposes MutualVPR, a mutual learning framework that dynamically assigns scene category labels through feature-driven adaptive K-means clustering, addressing the supervision inconsistency problem in classification-based VPR methods caused by viewpoint variation and occlusion.
tags:
  - "NeurIPS 2025"
  - "Visual Place Recognition"
  - "Adaptive Clustering"
  - "Mutual Learning"
  - "Supervision Consistency"
  - "DINOv2"
date: 2026-05-08
content_hash: 9194384249f2e1ca
---

# MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2412.09199](https://arxiv.org/abs/2412.09199)  
**Code**: [Available](https://github.com/Gucci233/MutualVPR)  
**Area**: Visual Localization / Image Retrieval
**Keywords**: Visual Place Recognition, Adaptive Clustering, Mutual Learning, Supervision Consistency, DINOv2

## TL;DR

This paper proposes MutualVPR, a mutual learning framework that dynamically assigns scene category labels through feature-driven adaptive K-means clustering, addressing the supervision inconsistency problem in classification-based VPR methods caused by viewpoint variation and occlusion.

## Background & Motivation

**Visual Place Recognition (VPR)** requires matching query images to previously visited locations, serving as a core component for long-term localization in autonomous driving and robotics.

**Classification-based VPR** methods (e.g., CosPlace, EigenPlaces) partition the environment into geographic grids, assigning one category label per grid cell and training with classification losses. This avoids the expensive hard sample mining required by contrastive learning. However, these methods suffer from a severe **supervision inconsistency problem**:

**Viewpoint variation (a)**: Images captured at the same location from different orientations exhibit drastically different visual content yet are assigned the same label.

**Viewpoint variation (b)**: Visually similar images with different orientation labels are assigned to different categories.

**Occlusion**: Buildings, vehicles, and other occluders cause significant content discrepancy among images facing the same reference point (violating EigenPlaces' core assumption).

**CosPlace** relies on manually annotated orientation labels to partition categories → visual intra-class consistency is not guaranteed.

**EigenPlaces** uses SVD decomposition under the assumption that images facing the same reference point are visually similar → fails in occluded urban environments.

**Core Insight**: Effective VPR supervision should reflect **semantic similarity** rather than relying solely on spatial proximity or fixed orientation assumptions.

## Method

### Overall Architecture

MutualVPR consists of two alternating optimization phases — **feature extraction** and **adaptive clustering** (Figure 3):

- **Phase 1**: Updated clustering labels supervise feature learning.
- **Phase 2**: Improved features guide re-clustering.

The two phases co-evolve, progressively eliminating supervision inconsistencies.

### Key Designs

1. **Feature Encoder**:

    - Built on a DINOv2 ViT-B/14 backbone with frozen pretrained parameters.
    - Introduces a **MulConv Adapter**: a bottleneck structure with three parallel convolutional branches (different receptive fields) for multi-scale feature extraction.
    - GeM pooling + fully connected layer to reduce dimensionality to a 512-dimensional descriptor.
    - The adapter contribution is controlled by a learnable scaling factor $s$: $z_l = \text{MLP}(\text{LN}(z_l')) + s \cdot \text{Adapter}(\text{LN}(z_l')) + z_l'$

2. **Mutual Learning via Adaptive Clustering**:

    - **Initialization**: Coarse-grained location grids are defined by UTM coordinates as $x = \{\lfloor east/M \rfloor, \lfloor north/M \rfloor\}$.
    - **Intra-grid refinement**: Within each UTM region, iterative K-means clustering is applied based on learned descriptors.
    - Category definition: $C = \{e_i, n_j, h \mid e_i, n_j \in x, h \in K\}$, where $K$ controls the granularity of orientation partitioning.
    - **Core mechanism**: The feature encoder and clustering process **iteratively update each other**, preventing error accumulation from early-stage incorrect assignments.
    - Classification loss adopts **LMCL (Large Margin Cosine Loss)**.

3. **Multi-Angle Cropping Strategy**:

    - Panoramic images are cropped every 60° starting from different angles (0° and 30°) to generate training data.
    - This enhances semantic continuity, as adjacent crops naturally share semantic content.

### Loss & Training

- Image size 504×504, descriptor dimension 512, grid size $M=10$, number of clusters $K=3$.
- Feature extractor learning rate 1e-5, classifier learning rate 1e-2, Adam optimizer with cosine annealing.
- Training for 50 epochs with 10,000 iterations per epoch.
- All UTM classes are divided into 8 groups for rotation-based training; 1/5 are randomly selected for clustering each epoch.

## Key Experimental Results

### Main Results — Multi-Benchmark Comparison (Table 1)

| Method | Dim | Training Set | MSLS-val R@1 | Pitts30k R@1 | Tokyo24/7 R@1 | SF-XL-test R@1 |
|--------|-----|--------------|-------------|-------------|--------------|---------------|
| CosPlace | 512 | SF-XL | 84.4 | 89.6 | 76.5 | 64.8 |
| EigenPlaces | 512 | SF-XL | 88.1 | 92.3 | 84.8 | **83.8** |
| MixVPR | 4096 | GSV-Cities | 87.1 | 91.6 | 87.0 | 69.2 |
| SALAD+CM | 512+32 | MSLS+GSV | **90.4** | 90.9 | 92.8 | 78.4 |
| BoQ | 512 | GSV-Cities | 88.4 | **93.1** | 91.9 | 79.6 |
| **MutualVPR** | **512** | **SF-XL** | 89.2 | 90.9 | 92.4 | 80.8 |

### Occlusion Robustness — SF-XL-Occlusion (Table 2)

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| CosPlace | 32.9 | 43.4 | 46.1 | 48.7 |
| EigenPlaces | 36.8 | 51.8 | 56.6 | 59.2 |
| MixVPR | 30.3 | 35.5 | 38.2 | 44.7 |
| SALAD+CM | 40.8 | 53.7 | 58.3 | 61.3 |
| **MutualVPR** | **47.4** | **65.8** | **71.1** | **73.7** |

MutualVPR outperforms the second-best method (CricaVPR / SALAD+CM at 40.8%) by **6.6 percentage points** in R@1 under occlusion.

### Ablation Study — Effect of Cluster Number $K$ (Table 4)

| Backbone | K | Tokyo24/7 R@1 | SF-XL-test R@1 |
|----------|---|--------------|---------------|
| DINOv2 | 1 (no clustering) | 80.6 | 61.1 |
| DINOv2 | 3 | **91.1** | **77.0** |
| DINOv2 | 6 | 86.0 | 70.9 |
| ResNet50 | 1 | 68.3 | 52.1 |
| ResNet50 | 6 | **82.9** | **74.5** |

### Key Findings

- Performance is worst without clustering ($K=1$), validating the effectiveness of orientation-based category partitioning.
- The optimal $K$ is 3 for DINOv2 and 6 for ResNet50 — the optimal cluster count depends on both backbone and data characteristics.
- Adaptive clustering vs. fixed orientation labels: even without multi-angle cropping, MutualVPR (DINOv2, 0°) achieves R@1=91.1 vs. CosPlace (DINOv2, 0°) at R@1=90.2.
- The substantial margin under occlusion stems from two mechanisms: **adaptive supervision correction** (iteratively correcting initial erroneous assignments) and **intra-class semantic compactness** (semantically similar views are brought closer in feature space).

## Highlights & Insights

- **In-depth problem analysis**: t-SNE visualizations clearly demonstrate three types of supervision inconsistency.
- **Simple yet effective**: Alternating K-means clustering and feature learning alone resolves the core issue of classification-based VPR.
- **No orientation labels required**: The method relies entirely on visual semantic self-organization, offering greater flexibility than CosPlace's manual labels and EigenPlaces' SVD assumptions.
- Trained solely on SF-XL, MutualVPR achieves competitive performance with contrastive learning methods trained on GSV-Cities (a larger and cleaner dataset) across multiple benchmarks.

## Limitations & Future Work

- The cluster number $K$ is a fixed hyperparameter and does not adapt automatically to data characteristics.
- Convergence of the mutual learning process lacks theoretical guarantees.
- Validation is limited to 2D images (panoramic crops); other modalities such as 3D point clouds have not been explored.
- The optimal $K$ is sensitive to the choice of backbone, and an automatic selection strategy is absent.

## Related Work & Insights

- **Contrastive learning VPR** (NetVLAD, MixVPR, CricaVPR) relies on carefully curated training data to avoid supervision inconsistency.
- **Classification-based VPR** (CosPlace, EigenPlaces) is efficient but constrained by fixed label assignments.
- Mutual learning / self-training paradigms are widely used in semi-supervised learning; this paper cleverly applies them to label refinement in VPR.
- Inspiration: similar adaptive clustering strategies could be adopted in other retrieval tasks that rely on coarse-grained label supervision.

## Rating

- Novelty: ⭐⭐⭐⭐ — The mutual learning framework and adaptive clustering approach are concise and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark validation is comprehensive; the occlusion experiment is convincing; ablation study is complete.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear and visualizations are intuitive.
- Value: ⭐⭐⭐⭐ — Effectively addresses the core limitation of classification-based VPR with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](../../CVPR2025/others/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](../../ACL2025/others/adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)

</div>

<!-- RELATED:END -->
