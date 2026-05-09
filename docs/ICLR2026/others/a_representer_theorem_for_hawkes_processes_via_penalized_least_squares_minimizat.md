---
title: >-
  [Paper Note] A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization
description: >-
  [ICLR 2026][Hawkes process] This paper establishes a novel representer theorem for estimating triggering kernels of linear multivariate Hawkes processes within the RKHS framework, proving that the optimal estimator admits a finite representation as a linear combination of equivalent kernels evaluated at data points, with all dual coefficients analytically equal to 1. This eliminates the need to solve a dual optimization problem, enabling efficient and scalable nonparametric estimation.
tags:
  - ICLR 2026
  - Hawkes process
  - representer theorem
  - RKHS
  - Fredholm integral equation
  - nonparametric estimation
date: 2026-05-08
content_hash: f827f52d8e555ec6
---

# A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization

**Conference**: ICLR 2026
**arXiv**: [2510.08916](https://arxiv.org/abs/2510.08916)
**Code**: None
**Area**: Point Processes / Kernel Methods
**Keywords**: Hawkes process, representer theorem, RKHS, Fredholm integral equation, nonparametric estimation

## TL;DR
This paper establishes a novel representer theorem for estimating triggering kernels of linear multivariate Hawkes processes within the RKHS framework, proving that the optimal estimator admits a finite representation as a linear combination of equivalent kernels evaluated at data points, with all dual coefficients analytically equal to 1. This eliminates the need to solve a dual optimization problem, enabling efficient and scalable nonparametric estimation.

## Background & Motivation

**Background**: Kernel methods enable nonparametric function estimation via RKHS, and representer theorems reduce infinite-dimensional optimization to finite-dimensional problems. Recent work has extended kernel methods to intensity function estimation for point processes; Bonnet & Sangnier (2025) derived a representer theorem for Hawkes processes but relied on discretization approximations.

**Limitations of Prior Work**: The approach of Bonnet & Sangnier requires solving a nonlinear optimization problem to obtain dual coefficients, whose dimensionality grows with the data size, making the method computationally infeasible at scale. Furthermore, it relies on discretized approximations of the likelihood/least-squares loss.

**Key Challenge**: The loss function for Hawkes processes involves integrals of the intensity over the observation domain and violates the independence assumptions underlying classical representer theorems. Exact variational analysis, in turn, leads to coupled systems of integral equations.

**Goal**: To establish a representer theorem for an exact, non-approximate penalized least-squares formulation for linear Hawkes processes, enabling scalable nonparametric triggering kernel estimation.

**Key Insight**: Leveraging path-integral representations and variational analysis, the paper derives exact variational equations directly in the continuous domain, revealing that the quadratic structure of the least-squares loss causes all dual coefficients to be automatically equal to 1.

**Core Idea**: The quadratic structure of the least-squares loss in RKHS causes all dual coefficients in the representer theorem for Hawkes processes to be analytically fixed at 1, eliminating the need for dual optimization.

## Method

### Overall Architecture
For a linear multivariate Hawkes process $\lambda_i(t) = \mu_i + \sum_j \int_0^t g_{ij}(t-s) dN_j(s)$, the triggering kernels $g_{ij}$ are estimated by minimizing a penalized least-squares loss within the RKHS. A representer theorem is derived via variational analysis, and equivalent kernels are approximated using random feature maps to yield an efficient estimator.

### Key Designs

1. **Novel Representer Theorem (Theorem 1)**:

    - **Function**: Proves that the optimal triggering kernel estimator admits a finite-dimensional representation with all dual coefficients equal to 1.
    - **Mechanism**: Setting the functional derivative of the penalized least-squares objective to zero yields a coupled system of Fredholm integral equations. The optimal estimator takes the form $\hat{g}_{ij}(s) = \sum_{n \in \mathcal{N}_i} h_j(s, t_n) - \hat{\mu}_i \int h_j(s,t) dt$, where the equivalent kernel $h_j$ is defined by a system of Fredholm integral equations of the second kind.
    - **Design Motivation**: The quadratic structure of the least-squares loss—unlike the nonlinearity of the log-likelihood—linearizes the variational equations, making the dual coefficients analytically equal to 1. This is the key theoretical insight.

2. **Closed-Form Construction of Equivalent Kernels (Proposition 3)**:

    - **Function**: Approximates the solution to the coupled Fredholm integral equations using degenerate kernels (random feature maps).
    - **Mechanism**: Using $M$ random feature maps with $k(s,s') = \phi(s)^\top \phi(s')$, the equivalent kernel takes the form $h_j(s,s') = \phi(s)^\top [(\frac{1}{\gamma}I_{MU} + \Xi)^{-1} \tilde{\phi}(s')]$, requiring only the inversion of a matrix of size $MU \times MU$, which is independent of the data size.
    - **Design Motivation**: This avoids the numerical errors of Riemann approximations, as all integrals can be computed in closed form under the random feature map.

3. **Efficient and Scalable Estimator**:

    - **Function**: Reduces the entire estimation procedure to additive matrix operations and a single matrix inversion.
    - **Mechanism**: The matrix $\Xi$ is constructed via incremental matrix additions over the data, and the matrix to be inverted has size $MU \times MU$ (where $M$ is the feature dimension and $U$ is the event dimension), independent of the total event count $N(T)$.
    - **Design Motivation**: In contrast to Bonnet & Sangnier, which requires solving an $O(N(T))$-dimensional nonlinear optimization, the core computational cost of the proposed method is decoupled from the data size.

### Loss & Training
Penalized least-squares loss with RKHS regularization $\frac{1}{\gamma}\sum \|g_{ij}\|_{\mathcal{H}_k}^2$. The baseline intensity $\mu_i$ also admits a closed-form solution (Proposition 2). Random Fourier features are used for kernel approximation.

## Key Experimental Results

### Main Results

**Prediction accuracy and computational efficiency on synthetic data:**

| Method | Prediction Error↓ | Computation Time↓ | Scalability |
|------|---------|---------|---------|
| Ours (RFF) | Competitive | **Significantly faster** | $O(M^3 U^3 + NM^2U)$ |
| Bonnet & Sangnier | Competitive | Slow (requires dual optimization) | $O(N^3)$ |

### Ablation Study

| Configuration | Description |
|------|------|
| Increasing $M$ (feature dimension) | Higher accuracy, increased computation |
| Increasing $\gamma$ (regularization) | Bias–variance tradeoff |
| Increasing $N$ (data size) | Efficiency advantage of the proposed method becomes more pronounced |

### Key Findings
- The theoretical result that all dual coefficients equal 1 is confirmed empirically; eliminating dual optimization yields substantial reductions in computation time.
- The random feature map approximation enables closed-form computation of all integrals, avoiding the error accumulation associated with discretization.
- On large-scale datasets, the proposed method is orders of magnitude faster than Bonnet & Sangnier while maintaining competitive prediction accuracy.
- The matrix to be inverted has size $MU \times MU$, independent of the data size, which is the key to scalability.

## Highlights & Insights
- **Hidden Structure of Least Squares**: The quadratic nature of the least-squares loss—rather than the nonlinearity of the log-likelihood—linearizes the variational equations, automatically setting the dual coefficients to 1. This is an elegant theoretical finding: the choice of loss function affects not only statistical properties but also the computational structure of the representer theorem.
- **Two-Stage Dimensionality Reduction**: The representer theorem reduces the problem from function space to a kernel expansion, and random feature maps further reduce the kernel expansion to a finite-dimensional vector. These two steps make otherwise intractable nonparametric estimation efficient.
- **Application of Integral Equation Theory**: The degenerate kernel method for Fredholm integral equations of the second kind represents a modern application of classical mathematics.

## Limitations & Future Work
- Applicable only to linear Hawkes processes (identity link function); inhibitory interactions cannot be modeled.
- The estimated triggering kernels are not guaranteed to be non-negative, potentially yielding negative conditional intensities.
- The approximation quality of random feature maps depends on the choice of feature dimension $M$.
- Theoretical analysis is restricted to RKHS kernels on a one-dimensional time domain; extensions to spatiotemporal settings remain to be explored.

## Related Work & Insights
- **vs. Bonnet & Sangnier 2025**: Their approach uses a discretized approximation of the least-squares/likelihood objective and obtains a representer theorem requiring dual optimization; the proposed method employs exact variational analysis and achieves a more elegant result that requires no dual optimization.
- **vs. Flaxman et al. 2017**: Flaxman et al. extend the representer theorem to point processes but only for univariate intensity functions; the present work handles triggering kernels in multivariate Hawkes processes, introducing the additional challenge of coupled integral equations.
- **vs. Classical Representer Theorems**: Classical representer theorems require optimization over dual coefficients; this work demonstrates that under certain loss functions the dual coefficients can be analytically fixed, representing a new member of the representer theorem family.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The finding that all dual coefficients equal 1 is an elegant theoretical contribution and constitutes the first representer theorem for non-approximate penalized least-squares estimation of Hawkes processes.
- Experimental Thoroughness: ⭐⭐⭐ Validation is limited to synthetic data; experiments on real-world datasets are absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and complete, though the high notational density limits readability.
- Value: ⭐⭐⭐⭐ Provides both theoretical foundations and an efficient algorithm for nonparametric estimation of Hawkes processes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation](revisiting_sharpness-aware_minimization_a_more_faithful_and_effective_implementa.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](../../NeurIPS2025/others/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[NeurIPS 2025\] Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes](../../NeurIPS2025/others/addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)

</div>

<!-- RELATED:END -->
