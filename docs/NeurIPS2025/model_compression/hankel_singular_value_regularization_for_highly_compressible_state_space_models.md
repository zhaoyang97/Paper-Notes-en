---
title: >-
  [Paper Note] Hankel Singular Value Regularization for Highly Compressible State Space Models
description: >-
  [NeurIPS 2025][Model Compression][SSM compression] By regularizing the **Hankel singular value nuclear norm** of SSM layers during training to encourage rapid decay…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "SSM compression"
  - "Hankel singular values"
  - "balanced truncation"
  - "regularization"
  - "Long Range Arena"
date: 2026-05-08
content_hash: c5d5f1aca7055d94
---

# Hankel Singular Value Regularization for Highly Compressible State Space Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.22951](https://arxiv.org/abs/2510.22951)  
**Code**: [GitHub](https://github.com/Algopaul/hankelreg)  
**Area**: Model Compression / State Space Models
**Keywords**: SSM compression, Hankel singular values, balanced truncation, regularization, Long Range Arena

## TL;DR

By regularizing the **Hankel singular value nuclear norm** of SSM layers during training to encourage rapid decay, the trained model can be compressed to **10%** of its original order via balanced truncation with negligible accuracy loss. A block-diagonal rotation matrix parameterization reduces Gramian computation from $\mathcal{O}(n^3)$ to $\mathcal{O}(n^2)$.

## Background & Motivation

SSMs (e.g., S4/S5/Mamba) excel on long-sequence tasks, but inference cost scales with state dimension $n$. Compression requires reducing $n$, and the key factor is the decay rate of Hankel singular values.

**Core systems-theoretic perspective**: The sequence-to-sequence mapping of a linear time-invariant system is characterized by the Hankel operator, whose singular values $\sigma_1, \ldots, \sigma_n$ determine compressibility. The approximation error when truncating order $n$ to $r$ satisfies:

$$\|\hat{y}_k - y_k\|_{\ell_2} \leq 2\|u\|_{\ell_2} \sum_{i=r+1}^n \sigma_i$$

**Problem**: Hankel singular values of standardly trained SSMs decay slowly, causing sharp accuracy degradation upon direct truncation. Prior work (Ezoe et al.) requires retraining after truncation.

## Method

### Block-Diagonal Rotation Matrix Parameterization

The matrix $\mathbf{A}$ is parameterized as a block diagonal of $2 \times 2$ rotation matrices:

$$\mathbf{A}_i(\rho_i, \alpha_i) = \rho_i \begin{bmatrix} \cos(\alpha_i) & \sin(\alpha_i) \\ -\sin(\alpha_i) & \cos(\alpha_i) \end{bmatrix}$$

- **Stability guarantee**: $\rho_i$ is constrained to $[0, 1)$ via $\tanh$
- **Universality**: Proposition 1 proves this parameterization is dense in SSM space
- **Parameter count**: Linear in $n$, identical to the diagonal parameterization of S5

### Efficient Gramian Computation

Hankel singular values $\sigma_i = \sqrt{\lambda_i(\mathbf{P}\mathbf{Q})}$ require solving the Lyapunov equation:

$$\mathbf{A}\mathbf{P}\mathbf{A}^\top - \mathbf{P} + \mathbf{B}\mathbf{B}^\top = \mathbf{0}$$

The standard Bartels–Stewart algorithm costs $\mathcal{O}(n^3)$. Exploiting the block-diagonal structure, the equation decomposes into $q$ independent $2 \times 2$ Lyapunov equations and $q(q-1)/2$ independent $2 \times 2$ Sylvester equations ($q = n/2$), each solvable in parallel:

$$\mathbf{A}_i \mathbf{P}_{ij} \mathbf{A}_j^\top - \mathbf{P}_{ij} + \mathbf{B}_i \mathbf{B}_j^\top = \mathbf{0}$$

Total complexity is $\mathcal{O}(q^2) = \mathcal{O}(n^2)$, independent of sequence length $n_s$.

### Differentiable Regularizer

Although individual Hankel singular values are non-differentiable with respect to system matrices, Proposition 2 proves their **sum** is smooth. The nuclear norm regularizer is defined as:

$$\mathcal{R}_*(\boldsymbol{\sigma}^{(1)}, \ldots, \boldsymbol{\sigma}^{(L)}) = \sum_{\ell=1}^L \sum_{i=1}^n \sigma_i^{(\ell)}$$

Added to the training loss, this encourages singular values to concentrate on the leading components, promoting rapid decay.

### Associative Scan Acceleration

Using the property $\mathbf{A}(\mathbf{x}, \boldsymbol{\beta}) \mathbf{A}(\mathbf{y}, \boldsymbol{\gamma}) = \mathbf{A}(\mathbf{x} \odot \mathbf{y}, \boldsymbol{\beta} + \boldsymbol{\gamma})$, a new associative binary operation is defined that avoids explicit block matrix products, enabling the same $\mathcal{O}(\log(n_s) \cdot n)$ parallel scan as diagonal S5.

### Post-Training Compression

1. Solve the final Gramians $\mathbf{P}$, $\mathbf{Q}$
2. Apply balanced truncation to obtain a reduced system of order $r$
3. Layer-adaptive rank allocation: given a total budget $r_t$, bisection search allocates per-layer ranks $r^{(\ell)}$ to preserve equal proportions of energy across layers
4. Diagonalize the reduced system to restore efficient scanning

## Key Experimental Results

### Long Range Arena Accuracy Under Various Truncation Ratios

| Method | sCIFAR 60% | 70% | 80% | 90% | sMNIST 60% | 70% | 80% | 90% |
|--------|-----------|-----|-----|-----|-----------|-----|-----|-----|
| LAST | 62.93 | 36.66 | 17.35 | 11.19 | 95.11 | 89.17 | 62.37 | 27.67 |
| global | 28.91 | 13.62 | 11.12 | 10.47 | 91.67 | 83.32 | 52.52 | 21.94 |
| no reg. | 71.28 | 41.98 | 21.14 | 9.84 | 91.32 | 13.35 | 11.05 | 10.55 |
| **HSVR** | **81.84** | **81.75** | **81.37** | **51.08** | **99.45** | **99.22** | **98.90** | **86.95** |

| Method | IMDB 60% | 70% | 80% | 90% | PATH-X 60% | 70% | 80% |
|--------|---------|-----|-----|-----|-----------|-----|-----|
| LAST | 88.48 | 85.05 | 80.26 | 57.08 | 50.33 | 49.16 | 49.53 |
| no reg. | 71.45 | 71.04 | 51.32 | 50.00 | 56.09 | 50.39 | 50.16 |
| **HSVR** | **87.26** | **87.16** | **86.97** | **86.40** | **87.74** | **82.82** | **54.02** |

### Key Findings

- **sMNIST at 80% truncation**: HSVR retains 98.90% accuracy, while no-regularization collapses to 11.05% (near random)
- **IMDB at 90% truncation**: HSVR loses only ~1% (86.40 vs. 87.26), while no-regularization drops to 50%
- **PATH-X at 60% truncation**: HSVR achieves 87.74%, while all baselines degrade to chance level (~50%)
- Layer-adaptive rank allocation outperforms uniform allocation: the number of important states varies substantially across layers

## Highlights & Insights

- **Elegant bridge between systems theory and deep learning**: Compressibility of neural SSM layers is characterized through classical Hankel singular value analysis
- **Non-differentiable made differentiable**: Smoothness of the nuclear norm is established by exploiting the smoothness of singular value sums at crossing points — an analytically elegant technique
- **Training as compression preparation**: A single regularized training run enables post-hoc standard balanced truncation without retraining
- **$\mathcal{O}(n^2)$ Gramian algorithm**: A practical efficiency improvement that makes regularization feasible during training

## Limitations & Future Work

1. **Not applicable to pretrained models**: Regularized training from scratch is required; post-hoc compression of already-trained SSMs is not supported
2. **Limited to linear time-invariant systems**: Time-varying/input-dependent SSMs such as Mamba are not covered
3. Only unidirectional scanning is considered; results may differ for bidirectional SSMs (the standard configuration for sCIFAR/IMDB)
4. The nuclear norm regularization strength requires manual tuning and exhibits high task-specific sensitivity
5. Post-compression diagonalization is numerically unstable when eigenvalues are near-degenerate

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introduces classical systems theory into SSM compression with a distinctive perspective
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Contributions span parameterization design, Gramian algorithms, differentiability proofs, and associative scanning
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of all 5 LRA benchmarks, but lacks practical NLP tasks such as language modeling
- **Practicality**: ⭐⭐⭐ — The requirement for training from scratch limits real-world applicability
- **Overall**: ⭐⭐⭐⭐

## Related Work & Insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](../../ICML2026/model_compression/quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)

</div>

<!-- RELATED:END -->
