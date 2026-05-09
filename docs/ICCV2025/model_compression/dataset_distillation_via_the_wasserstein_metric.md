---
title: >-
  [Paper Note] Dataset Distillation via the Wasserstein Metric
description: >-
  [ICCV 2025][Model Compression][Dataset Distillation] This paper proposes WMDD (Wasserstein Metric-based Dataset Distillation), which replaces MMD with Wasserstein barycenters for distribution matching and incorporates per-class BatchNorm regularization, achieving state-of-the-art dataset distillation performance on large-scale benchmarks including ImageNet-1K.
tags:
  - ICCV 2025
  - Model Compression
  - Dataset Distillation
  - Wasserstein Distance
  - Optimal Transport
  - Distribution Matching
  - BatchNorm Regularization
date: 2026-05-08
content_hash: 006081cd7f0ec7e9
---

# Dataset Distillation via the Wasserstein Metric

**Conference**: ICCV 2025
**arXiv**: [2311.18531](https://arxiv.org/abs/2311.18531)
**Code**: [https://github.com/Liu-Hy/WMDD](https://github.com/Liu-Hy/WMDD)
**Area**: Dataset Distillation / Model Compression
**Keywords**: Dataset Distillation, Wasserstein Distance, Optimal Transport, Distribution Matching, BatchNorm Regularization

## TL;DR

This paper proposes WMDD (Wasserstein Metric-based Dataset Distillation), which replaces MMD with Wasserstein barycenters for distribution matching and incorporates per-class BatchNorm regularization, achieving state-of-the-art dataset distillation performance on large-scale benchmarks including ImageNet-1K.

## Background & Motivation

Dataset distillation aims to synthesize a compact dataset such that models trained on it can approximate the performance of training on the full data, thereby substantially reducing computational cost. Existing methods fall into three categories:

**Performance matching** (e.g., DD, KIP): bilevel optimization, computationally expensive, difficult to scale to large datasets.

**Parameter matching** (e.g., DC, MTT): requires second-order gradient computation, leading to high memory demands.

**Distribution matching** (e.g., DM): computationally efficient but generally underperforms the above two categories.

**Key Challenge**: Distribution matching methods are efficient but insufficiently accurate. The bottleneck lies in MMD (Maximum Mean Discrepancy) as a distributional metric, which suffers from two issues: (1) practical implementations typically match only first-order moments (means), equivalent to a linear-kernel MMD that cannot distinguish higher-order moment differences; (2) using more expressive kernels (e.g., RBF) incurs prohibitive computational cost that prevents scaling to large datasets.

**Key Insight**: The Wasserstein distance from optimal transport theory naturally accounts for the geometric structure of distributions, and its barycenter preserves structural characteristics of the original distribution. This work computes the Wasserstein barycenter of per-class features in the representation space of a pretrained classifier, yielding a compact and geometrically faithful summary for efficient yet accurate distribution matching.

## Method

### Overall Architecture

WMDD proceeds in three steps: (1) extract features of the full dataset using a pretrained classifier; (2) compute the Wasserstein barycenter for each class, obtaining representative feature locations and weights; (3) optimize synthetic images via a feature matching loss and per-class BN regularization to align their features with the barycenter positions.

### Key Designs

1. **Wasserstein Barycenter Computation**: For each class with $n_k$ feature points, a Wasserstein barycenter supported on $m_k$ atoms is computed. The alternating optimization algorithm of [Cuturi & Doucet, 2014] is adopted:

   - **Weight optimization**: with positions fixed, solve a linear program for the optimal transport plan $\mathbf{T}$; use dual variables $\boldsymbol{\beta}$ as subgradients with respect to the weights and perform projected subgradient descent.
   - **Position optimization**: with weights fixed, the objective is quadratic in each synthetic point's position (Hessian $2w_j \mathbf{I}$), admitting a one-step Newton update: $\tilde{\mathbf{x}}_j \leftarrow \tilde{\mathbf{x}}_j - \frac{1}{w_j}\sum_i t_{ij}(\tilde{\mathbf{x}}_j - \mathbf{x}_i)$
   - Experiments show that only $K=10$ alternating iterations suffice to obtain high-quality synthetic data.

2. **Per-Class BatchNorm Regularization (PCBN)**: Conventional methods (e.g., SRe2L) align synthetic and real data statistics using global BN moments. However, feature distributions can vary substantially across classes, and global BN cannot provide class-differentiated guidance. PCBN independently computes and matches per-class mean and variance at each BN layer, and further incorporates the Wasserstein barycenter weights $w_{k,j}$ for weighted statistic computation.

3. **Joint Optimization Objective**:
   $$\mathcal{L}(\tilde{\mathbf{X}}) = \mathcal{L}_{\text{feature}}(\tilde{\mathbf{X}}) + \lambda \mathcal{L}_{\text{BN}}(\tilde{\mathbf{X}})$$
   where the feature loss is the sum of squared L2 distances from each synthetic image's feature to its corresponding barycenter atom, and $\lambda$ is the regularization coefficient.

### Loss & Training

Training follows a squeeze-and-recover paradigm. The squeeze stage pretrains the classifier; the recover stage optimizes synthetic images using the Adam optimizer, requiring only approximately 2,000 iterations on ImageNet-1K. The per-atom weights from the Wasserstein barycenter are retained for use in the subsequent Fast Knowledge Distillation (FKD) stage.

## Key Experimental Results

### Main Results

| Method | ImageNette 1IPC | ImageNette 10IPC | Tiny-IN 50IPC | ImageNet-1K 10IPC | ImageNet-1K 50IPC |
|--------|-----------------|------------------|---------------|-------------------|-------------------|
| Random | 23.5 | 47.7 | 16.8 | 3.6 | 15.3 |
| DM | 32.8 | 58.1 | 24.1 | - | - |
| SRe2L | 20.6 | 54.2 | 41.1 | 21.3 | 46.8 |
| G-VBSM | - | - | 47.6 | 31.4 | 51.8 |
| SCDD | - | - | 45.9 | 32.1 | 53.1 |
| **WMDD** | **40.2** | **64.8** | **59.4** | **38.2** | **57.6** |

At 100 IPC, WMDD achieves 87.1%, 61.0%, and 60.7% on the three benchmarks, respectively, approaching full-data training performance (89.9%, 63.5%, 63.1%).

### Ablation Study

| Feature Loss | Regularization | ImageNette | Tiny-IN | ImageNet-1K |
|--------------|----------------|------------|---------|-------------|
| Wasserstein | PCBN | **64.7** | **41.8** | **38.1** |
| CE | PCBN | 63.5 | 41.0 | 36.4 |
| Wasserstein | BN | 60.7 | 36.6 | 26.8 |
| CE | BN | 54.2 | 38.0 | 35.9 |

The combination of PCBN and Wasserstein matching consistently outperforms all other combinations across datasets, demonstrating that both designs are indispensable. Replacing the Wasserstein metric with MMD yields near-random performance on Tiny-IN and ImageNet-1K.

### Key Findings

- **Cross-architecture generalization**: Synthetic data distilled with ResNet-18 transfers well to ResNet-50/101 and ViT-Tiny/Small (with slight degradation on ViT).
- **Computational efficiency**: WMDD's per-iteration time is only 0.013s, comparable to SRe2L (0.015s) and far faster than DC (2.154s) and DM (1.965s).
- **Theoretical interpretation of Wasserstein vs. MMD**: The error bound of Wasserstein depends only on the Lipschitz constant, whereas the MMD bound depends on RKHS norms, which are difficult to control precisely in practice.

## Highlights & Insights

- Elegantly introduces optimal transport theory into dataset distillation, replacing simple mean matching with Wasserstein barycenters.
- The PCBN design is concise yet effective—per-class BN statistics should not be conflated across classes.
- Retains the computational efficiency of distribution matching while matching or surpassing bilevel optimization methods in performance.
- The additional overhead of computing barycenters in the embedding space is negligible (approximately 10 seconds total).

## Limitations & Future Work

- Performance depends on the quality of the pretrained classifier; biases in the classifier may be transferred to the synthetic data.
- Solving optimal transport for Wasserstein barycenters may become a bottleneck at extremely large class counts.
- The paper does not explore integration with generative models (e.g., GAN or Diffusion models).
- Cross-architecture generalization to data-hungry architectures such as ViTs remains an open avenue for improvement.

## Related Work & Insights

- The **SRe2L** family (squeeze–recover–relabel) serves as the most direct baseline; WMDD introduces Wasserstein matching into its recover stage.
- Sliced Wasserstein distance slightly accelerates computation but yields marginally lower performance (Table 5), indicating that full OT computation is critical for quality.
- The approach may generalize to other scenarios requiring distributional summaries, such as data sharing in federated learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically introducing OT theory into dataset distillation is a novel perspective.
- Theoretical Depth: ⭐⭐⭐⭐ — Provides error bound analysis comparing Wasserstein and MMD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, multiple IPC settings, cross-architecture evaluation, and efficiency analysis are all covered.
- Practicality: ⭐⭐⭐⭐⭐ — Excellent computational efficiency, scalable to ImageNet-1K.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[ICCV 2025\] Heavy Labels Out! Dataset Distillation with Label Space Lightening](heavy_labels_out_dataset_distillation_with_label_space_lightening.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)

<!-- RELATED:END -->
