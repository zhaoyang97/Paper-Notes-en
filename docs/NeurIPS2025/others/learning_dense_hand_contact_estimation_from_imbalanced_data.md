---
title: >-
  [Paper Note] Learning Dense Hand Contact Estimation from Imbalanced Data
description: >-
  [NeurIPS 2025][hand contact estimation] This paper proposes the HACO framework, which addresses class imbalance via Balanced Contact Sampling (BCS) and spatial imbalance via a Vertex-level Class-Balanced Loss (VCB Loss).…
tags:
  - "NeurIPS 2025"
  - "hand contact estimation"
  - "data imbalance"
  - "class-balanced loss"
  - "large-scale training"
  - "ViT"
date: 2026-05-08
content_hash: aa3f548a6396952b
---

# Learning Dense Hand Contact Estimation from Imbalanced Data

**Conference**: NeurIPS 2025
**arXiv**: [2505.11152](https://arxiv.org/abs/2505.11152)
**Code**: [Available](https://github.com/dqj5182/HACO_RELEASE)
**Area**: Other
**Keywords**: hand contact estimation, data imbalance, class-balanced loss, large-scale training, ViT

## TL;DR

This paper proposes the HACO framework, which addresses class imbalance via Balanced Contact Sampling (BCS) and spatial imbalance via a Vertex-level Class-Balanced Loss (VCB Loss). HACO is the first dense hand contact estimation model trained across 14 datasets (655K images) and achieves state-of-the-art performance across diverse interaction scenarios.

## Background & Motivation

Hand contact estimation is essential for understanding human hand interactions. Recent datasets cover multiple interaction types, including hand-object, hand-hand, hand-scene, and hand-body. However, learning dense hand contact estimation effectively from these datasets poses two core challenges:

**Class Imbalance**: The majority of hand surface regions are not in contact. The non-contact-to-contact ratio is 2.7:1 in DexYCB, 19.5:1 in InterHand2.6M, and 21.7:1 in Decaf.

**Spatial Imbalance**: Hand contact is heavily concentrated at fingertips (due to the prevalence of fingertip-centric motions in motion capture datasets), making it difficult for models to generalize to contact at the palm, back of the hand, and other regions.

The authors' core insight is that these two types of imbalance require distinct strategies: class imbalance calls for a sampling strategy, while spatial imbalance demands careful loss function design.

## Method

### Overall Architecture

HACO is built on a ViT backbone initialized with HaMeR pretrained weights:

1. **Image Encoding**: RGB image → Patch Embedding → ViT → image features $\mathbf{F} \in \mathbb{R}^{1280 \times 16 \times 12}$
2. **Contact Decoding**: Contact Tokens serve as queries and interact with image features via self-attention and cross-attention Transformers
3. **Output**: Linear layer + Contact Initialization (learnable embedding, analogous to a residual connection) + Sigmoid → contact probabilities for 778 MANO vertices
4. **Multi-scale Supervision**: A regression matrix maps the 778-dimensional output to coarser representations of 336, 84, and 21 dimensions

### Key Designs

#### 1. Balanced Contact Sampling (BCS)

**Objective**: Mitigate class imbalance between contact and non-contact samples.

**Contact balance score** is defined as:

$$s_i = \frac{1}{V}(\mathbf{c}_i^\top (1 - \bar{\mathbf{c}}) - \mathbf{c}_i^\top \bar{\mathbf{c}})$$

where $\bar{\mathbf{c}} = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_i$ is the dataset-level mean contact probability. A higher score indicates that a sample's contact pattern deviates more from the dataset mean.

**Non-linear binning**: Samples are divided into $K$ bins using logarithmically spaced boundaries (curvature parameter $\beta = 5$), providing finer resolution in high-contact-score regions:

$$\tau_k = s_{\min} + (s_{\max} - s_{\min}) \cdot \frac{\log(1 + \beta \cdot x_k)}{\log(1 + \beta)}$$

Stratified resampling is then applied across bins to ensure equal sample counts per bin.

#### 2. Vertex-level Class-Balanced Loss (VCB Loss)

**Design Motivation**: Standard CB Loss assigns only two weights (contact/non-contact) to the entire hand, failing to distinguish between frequently-contacted regions (e.g., fingertips) and rarely-contacted regions (e.g., back of the hand).

**Core Idea**: An independent weight is computed for each vertex $v$:

$$\alpha_{y_v} = \frac{1}{E_n^{(y_v)}} = \frac{1 - \beta}{1 - \beta^{n_{y_v}}}$$

where $n_{y_v}$ denotes the number of occurrences of class $y$ at vertex $v$ in the dataset.

The resulting VCB Loss is:

$$\mathcal{L}_{\text{VCB}} = \frac{1}{|V|} \sum_{v \in V} \alpha_{y_v} \ell_{\text{BCE}}(y_v, p_v)$$

**Progressive weighting strategy**: During early training, only the global CB Loss is applied; the VCB component weight increases linearly with epoch and reaches its maximum at the final epoch, enabling a smooth transition from global to vertex-adaptive supervision.

#### 3. Auxiliary Losses

- **Regularization loss**: L1 distance between predicted contacts and the dataset mean, preventing excessive deviation
- **Smoothness loss**: Encourages the model to predict fewer, larger contact regions rather than fragmented ones

### Loss & Training

Total loss = VCB Loss (weight 1.0) + Regularization loss (weight 0.1) + Smoothness loss (weight 1.0)

Training configuration:
- Backbone: HaMeR pretrained ViT
- Optimizer: AdamW, lr = $10^{-5}$, batch size 24
- Learning rate decay: multiplied by 0.9 at epochs 5 and 10
- Training: 10 epochs on a single NVIDIA A6000 GPU
- Data augmentation: random scaling, cropping, rotation + low-resolution, noise, and blur augmentation

## Key Experimental Results

### Main Results (MOW Dataset, Hand-Object Contact)

| Method | Precision ↑ | Recall ↑ | F1-Score ↑ |
|--------|-------------|----------|------------|
| POSA | 0.134 | 0.128 | 0.101 |
| BSTRO | 0.204 | 0.126 | 0.112 |
| DECO | 0.246 | 0.235 | 0.197 |
| **HACO** | **0.525** | **0.607** | **0.522** |

HACO achieves an F1-score **2.65×** higher than DECO.

### Downstream Task Validation

| Contact Method | Task | Key Metric |
|----------------|------|------------|
| DeepContact (3D input) | Grasp optimization | F1=0.612, MPJPE=37.155 |
| **HACO (image input)** | Grasp optimization | **F1=0.666**, **MPJPE=36.520** |
| EasyHOI (original) | Hand-object reconstruction | MPVPE=21.254 |
| **EasyHOI + HACO** | Hand-object reconstruction | **MPVPE=21.093** |

### Ablation Study

| Strategy | Precision ↑ | Recall ↑ | F1-Score ↑ |
|----------|-------------|----------|------------|
| w/o BCS | 0.520 | 0.542 | 0.481 |
| **w/ BCS** | **0.525** | **0.607** | **0.522** |

| Loss Function | Precision ↑ | Recall ↑ | F1-Score ↑ |
|---------------|-------------|----------|------------|
| CE Loss | 0.530 | 0.294 | 0.348 |
| Focal Loss | 0.518 | 0.387 | 0.409 |
| CB Loss | 0.484 | 0.534 | 0.465 |
| **VCB Loss** | **0.525** | **0.607** | **0.522** |

### Key Findings

1. **BCS primarily improves Recall**: +12.0% (0.542→0.607), demonstrating that the sampling strategy effectively encourages the model to learn from contact-rich samples.
2. **VCB Loss consistently outperforms all other imbalance-handling methods**: including Focal Loss, CB Loss, Asymmetric Loss, and five others, improving F1 from 0.465 (best competitor, CB Loss) to 0.522.
3. **Data diversity is critical**: Each interaction type (HO/HH/HS/HB) contributes uniquely; only the full combination achieves optimal performance across all test sets.
4. **Image-only input surpasses 3D input**: HACO, using only RGB images, outperforms DeepContact—which relies on full 3D meshes—in 3D grasp optimization.

## Highlights & Insights

1. **First large-scale hand contact estimation model**: Integrates 14 datasets covering four interaction types—hand-object, hand-hand, hand-scene, and hand-body.
2. **Extends CB Loss from class-level to vertex-level**: VCB Loss is a simple yet effective contribution whose underlying idea is transferable to other spatially imbalanced prediction problems.
3. **Progressive training strategy**: The transition from global to local balancing is elegantly designed and empirically effective.
4. **Contact Initialization**: The learnable contact initialization embedding functions analogously to a residual connection, stabilizing training.

## Limitations & Future Work

1. Only right-hand contact is predicted; left-right hand symmetry is not addressed.
2. Fully non-contact hands are excluded from evaluation (Recall/F1 are undefined for purely non-contact samples), potentially inflating reported performance.
3. Contact annotation protocols may be inconsistent across the 14 datasets, potentially introducing label noise.
4. Temporal information is not exploited (e.g., contact consistency across video frames).
5. The four resolution levels used in multi-scale supervision are fixed and not adaptively selected.

## Related Work & Insights

- **DECO**: Expands contact data via crowdsourced annotation but does not address data imbalance.
- **BSTRO**: Uses a Transformer to predict human-scene contact directly from visual input, but is limited in data scale.
- **ContactOpt**: Optimizes hand-object pose using contact priors derived from 3D geometry rather than visual observations.
- **Insight**: The vertex-level weighting idea underlying VCB Loss can be generalized to spatial imbalance in other dense prediction tasks, such as boundary or small-object regions in semantic segmentation.

## Rating

- Novelty: ⭐⭐⭐ — BCS and VCB Loss are principled extensions of existing methods with moderate novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale training across 14 datasets, multi-scenario evaluation, downstream task validation, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and data analysis is thorough.
- Value: ⭐⭐⭐⭐ — Provides the first general-purpose hand contact estimation model and a practical solution for imbalanced data in this domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](../../CVPR2026/others/shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)
- [\[NeurIPS 2025\] Dense Associative Memory with Epanechnikov Energy](dense_associative_memory_with_epanechnikov_energy.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)

</div>

<!-- RELATED:END -->
