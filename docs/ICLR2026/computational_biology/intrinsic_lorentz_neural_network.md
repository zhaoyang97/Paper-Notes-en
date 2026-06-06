---
title: >-
  [Paper Note] Intrinsic Lorentz Neural Network
description: >-
  [ICLR 2026][Computational Biology][hyperbolic neural network] This paper proposes ILNN, a fully intrinsic hyperbolic neural network in which all operations are performed entirely within the Lorentz model…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "hyperbolic neural network"
  - "Lorentz model"
  - "intrinsic operations"
  - "batch normalization"
  - "geometric deep learning"
date: 2026-05-08
content_hash: ca00e658847476bf
---

# Intrinsic Lorentz Neural Network

**Conference**: ICLR 2026
**arXiv**: [2602.23981](https://arxiv.org/abs/2602.23981)  
**Code**: To be confirmed  
**Area**: Model Architecture / Hyperbolic Geometry
**Keywords**: hyperbolic neural network, Lorentz model, intrinsic operations, batch normalization, geometric deep learning

## TL;DR
This paper proposes ILNN, a fully intrinsic hyperbolic neural network in which all operations are performed entirely within the Lorentz model, eliminating the geometric inconsistencies introduced by Euclidean operations in existing methods. ILNN achieves state-of-the-art performance on image classification, genomics, and graph classification tasks.

## Background & Motivation
**Background**: Hyperbolic neural networks exploit the exponentially growing volume of hyperbolic space to represent hierarchical data. The Lorentz model has become the preferred choice over the Poincaré model due to its superior numerical stability.

**Limitations of Prior Work**: Existing hyperbolic networks "fall back" to Euclidean space for certain operations (e.g., tangent-space linear transformations, Euclidean batch normalization), resulting in geometric inconsistency.

**Key Challenge**: How can one design sufficiently expressive and numerically stable components while keeping all operations on the hyperbolic manifold?

**Goal**: Construct a fully intrinsic set of operations for the Lorentz space.

**Key Insight**: Replace affine transformations with signed distances to Lorentz hyperplanes, and realize intrinsic statistics via gyro-structures.

**Core Idea**: Point-to-hyperplane distance → fully connected layer; gyro-centering + gyro-scaling → BatchNorm.

## Method

### Overall Architecture
ILNN operates entirely on the Lorentz hyperboloid $\mathbb{L}_K^n$ (with $K<0$). All intermediate computations remain on the manifold, and the network degenerates to a standard Euclidean network in the limit $K \to 0$.

### Key Designs

1. **PLFC (Point-to-Hyperplane Lorentz FC)**:

    - Learns $m$ Lorentz hyperplanes and computes the signed distance from the input to each hyperplane as logits.
    - Applies a $\sinh$ mapping to recover spatial coordinates, with the time coordinate derived from the hyperboloid constraint.
    - Degenerates to a standard affine transformation $Wx + b$ as $K \to 0$.

2. **GyroLBN (Gyro-Lorentz Batch Normalization)**:

    - Gyro-centering: closed-form Lorentzian centroid (non-iterative Fréchet mean).
    - Gyro-scaling: Fréchet variance normalization.
    - Faster than GyroBN (closed-form vs. iterative) and more intrinsic than LBN (hyperbolic mean vs. Euclidean mean).

3. **Auxiliary Components**: Log-radius concatenation, Lorentz dropout, Lorentz activation, and gyro-bias.

## Key Experimental Results

### Image Classification

| Method | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| Euclidean ResNet-18 | 95.14% | 77.72% |
| HCNN-Lorentz | 95.14% | 78.07% |
| **ILNN** | **95.36%** | **78.41%** |

### Genomics (TEB) — MCC Metric

| Task | Euclidean | **ILNN** | Gain |
|------|-----------|----------|------|
| Processed Pseudogene | 60.66 | **70.26** | +9.6 |
| Unprocessed Pseudogene | 51.94 | **64.90** | +13.0 |

### Genomics (GUE)

| Task | Prev. SOTA | **ILNN** |
|------|-----------|----------|
| Covid Classification | 36.71 | **64.76** |
| Core Promoter (tata) | 79.87 | **83.90** |

### Graph Classification
Airport 96.03%, Cora 85.68%, PubMed 82.52% — all state-of-the-art.

### Key Findings
- Gains on CIFAR are modest (+0.2–0.7 pp), whereas genomics tasks yield substantial improvements (+10–28 pp).
- HCNN-S collapses on Covid classification (36.71), while ILNN remains robust — demonstrating the numerical stability conferred by intrinsic operations.
- GyroLBN outperforms both GyroBN and LBN in speed and accuracy.

## Highlights & Insights
- **Fully intrinsic design philosophy**: operating directly on the hyperboloid is strictly preferable to mapping to a tangent space and back.
- **Closed-form centroid replaces iterative solvers**: GyroLBN is both faster and more accurate.
- The **$K \to 0$ degeneration** property is elegant — a hyperbolic network should be no worse than its Euclidean counterpart in the flat limit.
- The large genomics gains suggest that hyperbolic representations are particularly well-suited to biological sequences.

## Limitations & Future Work
- Experiments are limited to ResNet-18; modern architectures such as ViT have not been evaluated.
- Absolute improvements on CIFAR are small.
- The curvature $K=-1$ is fixed; learnable curvature remains unexplored.
- Only classification tasks are validated.

## Related Work & Insights
- **vs. HCNN**: HCNN reverts to tangent space for certain operations, whereas ILNN is fully intrinsic throughout.
- **vs. Poincaré networks**: The Lorentz model offers better numerical stability, an advantage that ILNN further reinforces.
- The proposed approach may inspire hyperbolic embedding designs for large language models.

## Supplementary Technical Details

### Geometric Intuition Behind PLFC
In Euclidean space, a fully connected layer computes $y = Wx + b$, which is essentially a projection of the input onto multiple hyperplanes. PLFC makes this operation intrinsic: hyperplanes are defined on the Lorentz hyperboloid, and the hyperbolic distance from a point to each hyperplane serves as the feature. This distance naturally degenerates to Euclidean distance as $K \to 0$, ensuring compatibility.

### Why Are Genomics Gains So Large?
Gene sequences exhibit an inherent hierarchical structure (gene families → genes → exons → sequence motifs). Hyperbolic space can capture such exponentially growing hierarchies more effectively in finite dimensions, whereas Euclidean space suffers from severe representational crowding at low dimensionality. The collapse of HCNN-S on Covid classification may be attributable precisely to its mixed operations discarding critical hierarchical information.

### Lorentz vs. Poincaré Model
The Lorentz model uses an $(n+1)$-dimensional ambient space, with coordinates $(x_0, x_1, \ldots, x_n)$ satisfying $-x_0^2 + x_1^2 + \cdots + x_n^2 = 1/K$. Unlike the Poincaré ball model, which suffers from numerical instability near the boundary, the Lorentz model avoids numerical issues by computing the time coordinate $x_0$ analytically.

## Rating
- Novelty: ⭐⭐⭐⭐ — The fully intrinsic operation design is a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three task categories: image, genomics, and graph.
- Writing Quality: ⭐⭐⭐⭐ — Mathematics is clear; degeneration analysis is elegant.
- Value: ⭐⭐⭐⭐ — A significant advance for hyperbolic geometry in deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[ICML 2026\] Demystifying Multimodal Biomolecular Co-design with Intrinsic Geodesic Coupling](../../ICML2026/computational_biology/demystifying_multimodal_biomolecular_co-design_with_intrinsic_geodesic_coupling.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/computational_biology/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](../../ICML2026/computational_biology/neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)
- [\[ICML 2026\] CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences](../../ICML2026/computational_biology/conditionally_site-independent_neural_evolution_of_antibody_sequences.md)

</div>

<!-- RELATED:END -->
