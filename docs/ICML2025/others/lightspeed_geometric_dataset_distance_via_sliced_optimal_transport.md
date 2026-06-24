---
title: >-
  [Paper Note] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport
description: >-
  [ICML2025][Sliced Optimal Transport] Proposes s-OTDD (sliced optimal transport dataset distance), which maps label distributions to scalars via Moment Transform Projection (MTP) to achieve near-linear complexity dataset distance computation, running significantly faster than OTDD while achieving comparable performance.
tags:
  - "ICML2025"
  - "Sliced Optimal Transport"
  - "Dataset Distance"
  - "Moment Transform Projection"
  - "Wasserstein Distance"
  - "Transfer Learning"
date: 2026-05-08
content_hash: 545f5084c8b56ae6
---

# Lightspeed Geometric Dataset Distance via Sliced Optimal Transport

**Conference**: ICML2025  
**arXiv**: [2501.18901](https://arxiv.org/abs/2501.18901)  
**Code**: [hainn2803/s-OTDD](https://github.com/hainn2803/s-OTDD)  
**Area**: Optimal Transport / Dataset Distance  
**Keywords**: Sliced Optimal Transport, Dataset Distance, Moment Transform Projection, Wasserstein Distance, Transfer Learning

## TL;DR

Proposes s-OTDD (sliced optimal transport dataset distance), which maps label distributions to scalars via Moment Transform Projection (MTP) to achieve near-linear complexity dataset distance computation, running significantly faster than OTDD while achieving comparable performance.

## Background & Motivation

Dataset distance metrics are crucial in tasks such as transfer learning, domain adaptation, and data augmentation. Among existing methods, OTDD (Optimal Transport Dataset Distance) is based on a hierarchical optimal transport framework, which first computes label distances via an inner OT and then computes dataset distances via an outer OT. Although it possesses favorable theoretical properties, it suffers from high computational complexity:

- **OTDD (Exact)**: time complexity $\mathcal{O}(n^3 \log n + c^2(n_{\max}^3 \log n_{\max} + d))$, space $\mathcal{O}(n^2 + c^2)$
- **WTE** (Wasserstein Task Embedding): requires MDS + Wasserstein embedding, is computationally expensive, and is not a valid metric
- **CHSW**: requires embedding labels using MDS beforehand, still maintaining a quadratic dependence on the number of classes

Limitations of Prior Work: When the dataset size $n > 30000$ or the feature dimension $d$ is high, the aforementioned methods crash due to memory limits. The authors aim to design a **training-free, embedding-free** dataset distance that achieves near-linear computational complexity without depending on the number of classes.

## Method

### Mechanism: Projecting data points to one dimension

The key challenge of s-OTDD lies in: a data point $(x, y)$ contains features $x$ and a label $y$, where the label is represented as a conditional distribution $q_y(X) = q(X|Y=y)$ over the feature space. How can a "distribution" be projected into a scalar?

### 1. Moment Transform Projection (MTP)

MTP is the core innovation of this work, mapping the label distribution $q_y$ to a scalar in two steps:

**Step 1 — Feature Projection**: Uses a projection function $\mathcal{FP}_\theta: \mathcal{X} \to \mathbb{R}$ (such as the Radon transform $\theta^\top x$) to project the high-dimensional distribution onto one dimension.

**Step 2 — Scaled Moment**: Computes the $\lambda$-th order scaled moment of the one-dimensional projected distribution:

$$\mathcal{SM}_\lambda(\mu) = \int_{\mathbb{R}} \frac{x^\lambda}{\lambda!} f_\mu(x) dx$$

Scaling with $\lambda!$ prevents numerical explosion of high-order moments. Under the empirical distribution: $\mathcal{MTP}_{\lambda,\theta}(q_y) = \frac{1}{n_y} \sum_{i: y_i=y} \frac{(\theta^\top x_i)^\lambda}{\lambda!}$.

**Injectivity Guarantee**: When the moment generating function exists, MTP is injective under $\Lambda = \mathbb{N}$ (infinite orders); in the finite-order case, if the Hankel matrix is positive definite and the moments are bounded ($|m_{\theta,\mu,\lambda}| < CD^\lambda \lambda!$), injectivity is likewise guaranteed.

### 2. Data Point Projection

Combining feature projection with multiple MTPs:

$$\mathcal{DP}^k_{\psi,\theta,\lambda,\phi}(x, q_y) = \psi^{(1)} \mathcal{FP}_\theta(x) + \sum_{i=1}^{k} \psi^{(i+1)} \mathcal{MTP}_{\lambda^{(i)}, \phi}(q_y)$$

Where $\psi \in \mathbb{S}^k$ is a weight vector on the unit hypersphere, and $k$ is the number of moment orders used. In the experiments, $k=5$ is chosen.

### 3. s-OTDD Definition

$$\text{s-OTDD}_p^p(\mathcal{D}_1, \mathcal{D}_2) = \mathbb{E}_{(\psi,\theta,\lambda,\phi)} \left[ W_p^p \left( \mathcal{DP}^k \sharp P_{\mathcal{D}_1},\ \mathcal{DP}^k \sharp P_{\mathcal{D}_2} \right) \right]$$

The expectation is taken over the random projection parameters. The one-dimensional Wasserstein distance has a closed-form solution (via sorting), and the Monte Carlo estimation is approximated using $L$ sets of projections.

### Computational Complexity Comparison

| Method | Time Complexity | Space Complexity | Class Dependency |
|------|-----------|-----------|---------|
| OTDD (Exact) | $\mathcal{O}(n^3 \log n + c^2 n_{\max}^3)$ | $\mathcal{O}(n^2 + c^2)$ | Yes |
| OTDD (Gaussian) | $\mathcal{O}(n^3 \log n + c^2 d^3)$ | $\mathcal{O}(n^2 + c^2)$ | Yes |
| **s-OTDD** | $\mathcal{O}(L(n \log n + dn))$ | $\mathcal{O}(L(d+n))$ | **No** |

s-OTDD does not depend on the number of classes $c$, is friendly to imbalanced data, and supports distributed computing (projections can be precomputed independently for each dataset).

## Key Experimental Results

### 1. Correlation with OTDD (MNIST / CIFAR10 Subsets Comparison)

| Method | MNIST Spearman $\rho$ | CIFAR10 Spearman $\rho$ |
|------|----------------------|------------------------|
| OTDD (Gaussian) | ~0.90 | ~0.85 |
| WTE | ~0.88 | ~0.82 |
| CHSW (10K proj) | ~0.75 | ~0.70 |
| **s-OTDD (10K proj)** | **~0.90** | **~0.85** |

s-OTDD is highly correlated with OTDD (Exact), performing comparably to the Gaussian approximation and WTE.

### 2. Computation Time

When the dataset size exceeds 30K, OTDD/WTE/CHSW crash due to out-of-memory errors, whereas s-OTDD runs smoothly up to full datasets (60K MNIST, 50K CIFAR10). From MNIST to CIFAR10 (with increased dimensions), the increase in run-time for s-OTDD is significantly smaller than for other methods.

### 3. Transfer Learning Correlation

| Experiment | Method | Spearman $\rho$ | Pearson $r$ |
|------|------|----------------|------------|
| NIST Images | OTDD (Exact) | 0.40 | — |
| NIST Images | **s-OTDD** | **0.40** | — |
| Text Datasets | OTDD (Exact) | High | 0.48 |
| Text Datasets | **s-OTDD** | Slightly Lower | **0.48** |
| Split Tiny-ImageNet 224×224 | **s-OTDD** | Strong correlation | Strong correlation |
| Split Tiny-ImageNet 224×224 | OTDD | Infeasible | Infeasible |

On Tiny-ImageNet at 224×224 resolution, OTDD cannot run due to excessive computational costs, while s-OTDD remains functional.

### 4. Data Augmentation (Tiny-ImageNet → CIFAR10)

| Method | Sample Size | Projection Number | Spearman $\rho$ | Time Elapsed |
|------|--------|--------|----------------|------|
| OTDD (Exact) | 5K | — | -0.70 | ~74×10³s |
| **s-OTDD** | **50K** | 100K | **-0.74** | **~53×10³s** |

s-OTDD processes 10 times the amount of data, running faster and achieving higher correlation.

## Highlights & Insights

1. **Elegant Design of MTP**: Maps distributions to scalars using scaled moments, where the $\lambda!$ scaling prevents numerical explosion, theoretically guaranteeing injectivity (based on the Hamburger moment problem).
2. **Completely Decoupled from the Number of Classes**: Computational complexity does not depend on $c$, which is particularly friendly to datasets with a large number of classes or imbalanced distributions.
3. **Supports Distributed / Federated Learning**: The inverse CDF of projections can be precomputed independently for each dataset, and distances can be obtained by exchanging very little data.
4. **Monte Carlo Approximation Error of $\mathcal{O}(L^{-1/2})$**: Converges rapidly, with 10K projections already sufficient.
5. **s-OTDD Is a Valid Metric**: Satisfies metric axioms on $\mathcal{P}(\mathcal{X} \times \mathcal{P}(\mathcal{X}))$.

## Limitations & Future Work

1. **Choice of Moment Order $k$**: The experiments use $k=5$; excessively high orders may cause numerical overflow, and an adaptive selection strategy is currently lacking.
2. **Choice of Projection Type**: The effectiveness of Radon transform versus convolutional projection varies across different data types, requiring manual selection.
3. **Limited Capture of Non-linear Structures**: Linear projections may lose the non-linear manifold structure of the data.
4. **Unexplored Gradient Flows**: The paper mentions that exploring the gradient flow properties of s-OTDD is a direction for future work; currently, it does not support differentiable optimization.
5. **Only Evaluated on Images and Text**: Experiments on other modalities, such as graphs and time-series data, are missing.

## Related Work & Insights

- **OTDD** (Alvarez-Melis & Fusi, 2020): A dataset distance based on a hierarchical OT framework; this work serves as its accelerated version.
- **WTE** (Liu et al., 2025): MDS + Wasserstein embedding, efficient but not a valid metric.
- **CHSW** (Bonet et al., 2024): Slices Wasserstein on Cartan-Hadamard manifolds, requiring MDS preprocessing.
- **Sliced Wasserstein** (Bonneel et al., 2015): The theoretical foundation of s-OTDD, utilizing random projections and closed-form 1D OT solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ — The projection design of MTP mapping "distribution $\to$ scalar" is highly novel and theoretically sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers image, text, and large-scale experiments, including ablation studies and computation time analyses.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured and mathematically rigorous.
- Value: ⭐⭐⭐⭐ — Resolves the core scalability bottleneck of OTDD, offering strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/others/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](../../ECCV2024/others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ICLR 2026\] Mixed-Curvature Tree-Sliced Wasserstein Distance](../../ICLR2026/others/mixed-curvature_tree-sliced_wasserstein_distance.md)

</div>

<!-- RELATED:END -->
