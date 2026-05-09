---
title: >-
  [Paper Note] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval
description: >-
  [ICCV 2025][Hyperbolic Space] This paper presents the first learning paradigm for encoding user-defined multi-level visual hierarchies in hyperbolic space. It introduces an angle-based entailment contrastive loss to learn scene→object→part hierarchies without explicit hierarchy labels, and proposes an optimal-transport-based hierarchical retrieval evaluation metric.
tags:
  - ICCV 2025
  - Hyperbolic Space
  - Visual Hierarchy
  - Entailment Learning
  - Image Retrieval
  - Contrastive Loss
date: 2026-05-08
content_hash: bdc7067f62f0e41d
---

# Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval

**Conference**: ICCV 2025
**arXiv**: [2411.17490](https://arxiv.org/abs/2411.17490)
**Code**: None
**Area**: Image Retrieval / Representation Learning
**Keywords**: Hyperbolic Space, Visual Hierarchy, Entailment Learning, Image Retrieval, Contrastive Loss

## TL;DR

This paper presents the first learning paradigm for encoding user-defined multi-level visual hierarchies in hyperbolic space. It introduces an angle-based entailment contrastive loss to learn scene→object→part hierarchies without explicit hierarchy labels, and proposes an optimal-transport-based hierarchical retrieval evaluation metric.

## Background & Motivation

Humans organize world knowledge hierarchically, yet mainstream image understanding models (SimCLR, MoCo, CLIP, etc.) focus solely on **visual similarity** and fail to capture hierarchical semantic relationships. For instance, an urban street scene contains a hierarchy of building→skyscraper→window, yet similarity metrics cannot distinguish such containment relations.

Limitations of existing hierarchical learning methods:
- Most require predefined **explicit hierarchy labels** (e.g., the ImageNet label tree), which are costly and inflexible
- Methods such as HCL handle only simple two-level hierarchies for single-class images and cannot model complex part-based hierarchies in multi-object scenes
- Symmetric distances (inner product, cosine similarity) cannot express **asymmetric containment relations**

The advantage of hyperbolic space: its volume grows **exponentially** with radius, making it naturally suited for embedding tree-structured hierarchies.

## Method

### Overall Architecture

1. Bounding box annotations are used to define part-based image hierarchies (scene→object→part)
2. Hierarchies are decomposed into pairwise entailment relations (A→B denotes A contains B)
3. An angle-based entailment contrastive loss enforces these relations in hyperbolic space
4. Training uses only pairwise entailments, without the full hierarchy tree

### Key Designs

1. **Part-Based Image Hierarchy Definition**:

    - The full scene image occupies the top level of the hierarchy; bounding boxes within it form lower levels
    - A large bounding box that substantially encloses a smaller one (≥80% area overlap) constitutes an entailment relation
    - **Intra-image hierarchy**: containment rules are applied recursively to construct a tree (road scene → cyclist → bicycle → wheels)
    - **Cross-image hierarchy**: for each bounding box, $K$ same-category bounding boxes are sampled from other images, establishing entailments from full scenes to same-category objects across images
    - Design motivation: only bounding box and category annotations are required (no hierarchy annotations), enabling automatic construction of complex multi-level hierarchies

2. **Angle-Based Entailment Loss**:

    - Given an entailment pair $(x \to y)$, the exterior angles $\beta_1$ and $\alpha_2$ are maximized in hyperbolic space:
        - $\beta_1(\mathbf{x}, \mathbf{y}) = \pi - \text{ext}(\mathbf{x}, \mathbf{y})$
        - $\alpha_2(\mathbf{y}, \mathbf{x}) = \text{ext}(\mathbf{y}, \mathbf{x})$
    - The exterior angle is computed via the Lorentz model inner product: $\text{ext}(\mathbf{x}, \mathbf{y}) = \cos^{-1}\left(\frac{y_{\text{time}} + x_{\text{time}} c\langle\mathbf{x}, \mathbf{y}\rangle_{\mathbb{H}}}{\|\mathbf{x}_{\text{space}}\|\sqrt{(c\langle\mathbf{x}, \mathbf{y}\rangle_{\mathbb{H}})^2 - 1}}\right)$
    - A multi-positive InfoNCE loss handles the case where one parent corresponds to multiple children
    - Bidirectional loss: $L_{\text{angle}} = L^{p\to c}(\mathcal{D}, \beta_1) + L^{c\to p}(\mathcal{D}, \alpha_2)$
    - Design motivation: angular metrics provide an additional degree of freedom along the radial axis, allowing embeddings to distribute naturally along tree structures; bidirectional constraints yield greater training stability than unidirectional ones

3. **Hierarchical Retrieval Evaluation**:

    - For each query image $I$, the label distribution $\mathbf{h}_I$ of its hierarchy tree $\mathcal{H}_I$ is precomputed
    - The 1-D Wasserstein distance between the retrieval label distribution $\mathbf{r}_I$ and $\mathbf{h}_I$ is computed: $\text{OT}(\mathbf{h}_I, \mathbf{r}_I) = \text{Wasserstein}(\bar{\mathbf{h}}_I, \bar{\mathbf{r}}_I)$
    - A smaller distance indicates better hierarchical alignment
    - Design motivation: standard Recall@k ignores class distribution imbalance; the OT distance measures the alignment between retrieved results and the ground-truth hierarchical distribution

### Loss & Training

- AdamW optimizer, lr=2e-5, $(\beta_1, \beta_2)=(0.9, 0.999)$
- CLIP ViT effective batch size=640; MoCo-v2 batch size=1984
- Hyperbolic model uses a learnable curvature parameter with embedding dimension 128
- Fine-tuned on HierOpenImages (a hierarchical dataset constructed from OpenImages)
- Temperature parameter $\tau$ is learnable, initialized to 0.07

## Key Experimental Results

### Main Results (HierOpenImages Retrieval)

**Child-to-Parent Same-Category Retrieval (CLIP ViT)**:

| Model | Metric | Top-5 | Top-10 | Top-50 | Top-100 |
|---|---|---|---|---|---|
| CLIP | Cos Sim. | 53.04 | 51.69 | 47.87 | 45.79 |
| HCL | Cos Sim. | 55.48 | 54.81 | 52.23 | 50.67 |
| CLIP-euc† | Euc Ang.* | 75.63 | 74.65 | 72.25 | 70.52 |
| **CLIP-hyp†** | **Hyp Ang.*** | **77.28** | **75.91** | **72.85** | **70.94** |

**Hierarchical Retrieval Evaluation (Parent-to-Child, CLIP ViT)**:

| Model | Recall@150k↑ | Recall@250k↑ | OT@150k↓ | OT@250k↓ |
|---|---|---|---|---|
| CLIP | 66.63 | 86.57 | 21.31 | 23.79 |
| CLIP-euc† | 76.46 | 91.38 | 15.65 | 21.09 |
| **CLIP-hyp†** | **77.00** | **91.89** | **14.96** | **20.76** |

### Ablation Study (Effect of Cross-Image Entailment)

As shown in the Precision-Recall curves (Figure 4):
- Training with intra-image entailment alone already yields significant improvement
- Adding cross-image entailment further improves performance in the mid-to-high recall region, at the cost of a slight reduction in top-rank precision (due to the visual diversity introduced by cross-image relations)

**Out-of-Domain Generalization (LVIS Dataset, Child-to-Parent)**:

| Model | Top-5 | Top-10 | Top-50 |
|---|---|---|---|
| CLIP | 15.00 | 14.22 | 11.91 |
| CLIP-euc† | 28.45 | 26.76 | 22.65 |
| **CLIP-hyp†** | **28.84** | **27.02** | **22.87** |

**Zero-Shot Object Detection (VOC/COCO)**:

| Dataset | CLIP | CLIP-euc† | CLIP-hyp† |
|---|---|---|---|
| VOC | 87.3 | 93.6 | **94.2** |
| COCO | 63.6 | 73.4 | **73.9** |

### Key Findings

- Hyperbolic space consistently outperforms Euclidean space regardless of backbone (CLIP or MoCo)
- The Hyp Dist metric for HCL degrades after fine-tuning, indicating that its training scheme is ill-suited for complex visual hierarchies
- The radial structure of hyperbolic embeddings forms naturally: high-level concepts cluster near the origin while fine-grained or ambiguous crops are pushed toward the boundary
- Significant improvements are observed on unseen LVIS and VOC/COCO datasets, demonstrating the generalization of the learned hierarchical representations
- Even when only the angular entailment loss is applied (without distance constraints), the model forms a well-structured radial hierarchy

## Highlights & Insights

- **First work to encode complex multi-level visual hierarchies without explicit hierarchy labels**, requiring only bounding box and category annotations
- Pairwise entailment relations suffice to learn global tree structure — a finding of considerable theoretical significance
- The OT evaluation metric addresses the limitations of standard Recall metrics in hierarchical retrieval
- Out-of-domain generalization experiments are compelling, demonstrating the universality of the learned hierarchical representations
- Visualizations reveal an elegant embedding space structure: the hierarchy from harbor → boat parts is clearly delineated

## Limitations & Future Work

- Hierarchy construction depends on bounding box annotations; unannotated data cannot be utilized
- The 80% area overlap threshold is manually set and may require adjustment for different datasets
- Only the image encoder is used; the text encoder of CLIP is not exploited (HierOpenImages is not suitable for text-based evaluation)
- Label distributions are highly imbalanced (Figure 8), potentially leading to insufficient hierarchical learning for rare categories
- More complex hierarchical structures, such as those arising in 3D scene understanding, remain unexplored

## Related Work & Insights

- HCL is the most direct baseline but handles only simple two-level scene-object hierarchies
- ACCEPT proposes the angle-based entailment loss, which this paper extends to the multi-positive setting
- Poincaré Embeddings is a pioneering work in hyperbolic representation learning
- The central insight of this paper: **hierarchy is an asymmetric relation requiring asymmetric metrics** — angular metrics in hyperbolic space naturally provide this asymmetry

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First work to achieve label-free multi-level complex visual hierarchy learning in the image domain
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive in-domain and out-of-domain evaluation; additional backbones and datasets would strengthen the claims
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical and methodological foundations are well-developed, though the density of mathematical formulations may affect readability
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for visual hierarchy learning; the retrieval evaluation metric offers meaningful reference value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)

</div>

<!-- RELATED:END -->
