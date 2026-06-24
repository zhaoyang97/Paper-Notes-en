---
title: >-
  [Paper Note] Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances
description: >-
  [ICML2025][Optimization][Wasserstein Distance] This paper provides **sharp finite-sample statistical guarantees** (dimension-free, with a convergence rate of $n^{-1/(2p)}$) and **computational guarantees** (proving exact computation is NP-hard and proposing an efficient semidefinite relaxation SDR alongside a first-order algorithm) for the Kernel Max-Sliced (KMS) Wasserstein distance. Its superior performance is demonstrated in high-dimensional two-sample testing…
tags:
  - "ICML2025"
  - "Optimization"
  - "Wasserstein Distance"
  - "Kernel Methods"
  - "Semidefinite Relaxation"
  - "Sample Complexity"
  - "Non-parametric Hypothesis Testing"
  - "Curse of Dimensionality"
date: 2026-05-08
content_hash: f902d63048c1d935
---

# Statistical and Computational Guarantees of Kernel Max-Sliced Wasserstein Distances

**Conference**: ICML2025  
**arXiv**: [2405.15441](https://arxiv.org/abs/2405.15441)  
**Code**: Not released  
**Area**: Optimization / Optimal Transport  
**Keywords**: Wasserstein Distance, Kernel Methods, Semidefinite Relaxation, Sample Complexity, Non-parametric Hypothesis Testing, Curse of Dimensionality

## TL;DR

This paper provides **sharp finite-sample statistical guarantees** (dimension-free, with a convergence rate of $n^{-1/(2p)}$) and **computational guarantees** (proving exact computation is NP-hard and proposing an efficient semidefinite relaxation SDR alongside a first-order algorithm) for the Kernel Max-Sliced (KMS) Wasserstein distance. Its superior performance is demonstrated in high-dimensional two-sample testing, human activity detection, and generative modeling.

## Background & Motivation

- **Curse of Dimensionality**: The sample complexity of the classical Wasserstein distance grows exponentially with the data dimension $d$, which severely limits its application in high dimensions.
- **Dimension Reduction Strategies**: Sliced Wasserstein takes the mean over random one-dimensional projections, yielding limited information. Max-Sliced (MS) Wasserstein searches for the optimal **linear** projection direction but lacks discriminative power for data with non-linear structures.
- **Introduction of KMS Wasserstein**: Wang et al. (2022a) extended the projection function from linear mappings to the optimal **non-linear** mapping in an RKHS ball, defining the Kernel Max-Sliced Wasserstein distance. It degenerates to the MS Wasserstein when using a dot-product kernel and possesses universal approximation properties when using a Gaussian kernel.
- **Limitations of Prior Work**: (1) Statistical convergence rates rely on hard-to-verify regularity conditions such as the projection Poincaré inequality; (2) Computation relies solely on local optima from gradient descent, which lacks theoretical guarantees and is highly sensitive to initialization.

## Method

### Core Definition

Given an RKHS $\mathcal{H}$ and the corresponding kernel $K$, the KMS $p$-Wasserstein distance is defined as:

$$\mathcal{KMS}_p(\mu,\nu) = \max_{f\in\mathcal{H},\|f\|_{\mathcal{H}}\le 1} W_p(f_\#\mu, f_\#\nu)$$

where $f_\#\mu$ is the push-forward measure of $\mu$ under $f:\mathcal{B}\to\mathbb{R}$. Exploiting the reproducing property $f(x)=\langle f, K_x\rangle_{\mathcal{H}}$, KMS can be equivalently reformulated as embedding the data first via the feature map $\Phi(x)=K_x$ into $\mathcal{H}$, and then performing the MS Wasserstein in the Hilbert space:

$$\mathcal{KMS}_p(\mu,\nu) = \mathcal{MS}_p(\Phi_\#\mu, \Phi_\#\nu)$$

### Statistical Guarantees (Theorem 3.2)

**Condition**: Requires only that the kernel satisfies $\sqrt{K(x,x)}\le A$ (which is naturally satisfied by the Gaussian kernel).

- **One-sample**: $\Pr[\mathcal{KMS}_p(\hat\mu_n, \mu) \le \frac{1}{2}\Delta(n,\alpha)] \ge 1-\alpha$
- **Two-sample**: $\Pr[\mathcal{KMS}_p(\hat\mu_n, \hat\nu_n) \le \mathcal{KMS}_p(\mu,\nu) + \Delta(n,\alpha)] \ge 1-\alpha$

where the critical value is $\Delta(n,\alpha) = 4A(C+4\sqrt{\log(2/\alpha)})^{1/p} \cdot n^{-1/(2p)}$.

**Key Property**: The convergence rate of $O(n^{-1/(2p)})$ is **dimension-free** and is optimal in the worst case (coinciding with the classical $W_p$ for the one-dimensional dot-product kernel).

### Computational Guarantees

#### NP-hardness (Theorem 4.2)

Using the representer theorem, the functional optimization of KMS is transformed into a finite-dimensional problem (Eq. 9):

$$\max_{\omega\in\mathbb{R}^{2n},\|\omega\|_2=1} \min_{\pi\in\Gamma_n} \sum_{i,j}\pi_{i,j}(M_{i,j}^T\omega)^2$$

It is proved that this max-min problem can be reduced to the fair-PCA / fair beamforming problem in the worst case, thereby establishing that it is NP-hard.

#### Semidefinite Relaxation SDR

Let $S=\omega\omega^T$. Relaxing the rank-1 constraint yields a semidefinite program (SDP):

$$\max_{S\in\mathcal{S}_{2n}} F(S), \quad F(S)=\min_{\pi\in\Gamma_n}\sum_{i,j}\pi_{i,j}\langle M_{i,j}M_{i,j}^T, S\rangle$$

#### First-Order Inexact Mirror Ascent Algorithm (Algorithm 1)

- Resolves the inner OT problem using the Katyusha momentum stochastic gradient method to obtain an approximation $\hat\pi$, constructing the supergradient $v(S_k)=\sum_{i,j}\hat\pi_{i,j}M_{i,j}M_{i,j}^T$
- Performs von Neumann entropy mirror ascent on the spectrahedron $\mathcal{S}_{2n}$, where the update has a closed-form solution (matrix exponential)
- **Complexity** (Theorem 4.4): $\tilde{O}(n^2\delta^{-2}\cdot\max(n,\delta^{-1}))$, which is significantly superior to the $\tilde{O}(n^{6.5})$ complexity of interior-point methods.

#### Rank Bounds and Rank Reduction (Theorems 4.5 & 4.6)

- There exists an optimal solution for SDR with a rank of at most $k=1+\lfloor\sqrt{2n+9/4}-3/2\rfloor$, which is much smaller than the trivial upper bound of $2n$.
- A rank reduction algorithm based on the Hungarian method and null-space search is designed to project the approximate solution to a lower-rank space with a complexity of $O(n^5)$.
- Relaxation Gap (Theorem 4.7): $\varepsilon n^{-4}\cdot\text{Optval(SDR)}\le\text{Optval(KMS)}\le\text{Optval(SDR)}$

## Key Experimental Results

### Computational Efficiency and Solution Quality (Fig. 3)

| Method | Computation Time | Solution Quality (Estimated KMS $W_2$) |
|------|---------|------------------------|
| SDR-Efficient (Ours) | Moderate, strong advantage on large scale | Largest mean, smallest variance |
| BCD (Wang et al. 2022a) | Slightly faster on small scale, timeout on large scale | Sensitive to initialization, large variance |
| SDR-IPM (Interior Point Method) | Extremely slow even on small scale | Consistent with SDR-Efficient |

Test scenarios: 100-dimensional synthetic Gaussian, MNIST, CIFAR-10.

### High-Dimensional Two-Sample Testing (Fig. 4)

| Scenario | Best Method | KMS Performance |
|------|---------|---------|
| Gaussian Covariance Shift | MS (Linear Optimal) | Suboptimal |
| Gaussian Mixture Distribution Shift | **KMS** | Optimal |
| MNIST Distribution Abundance Shift | **KMS** | Optimal |
| CIFAR-10 Distribution Abundance Shift | **KMS** | Optimal |

### Human Activity Detection (Table 1)

| Method | Mean Detection Delay | Standard Deviation |
|------|-----------|-------|
| **KMS** | **11.4** | **5.56** |
| GSW | 12.9 | 6.4 |
| Sinkhorn Div | 16.5 | 4.4 |
| SW | 17.2 | 8.7 |
| MS | 17.8 | 9.2 |
| MMD | 50.6 | 39.5 |

### Generative Modeling FID Score (Fig. 6)

| Method | FID ↓ |
|------|------|
| **KMS Wasserstein** | **105.98** |
| MMD | 113.08 |
| Sinkhorn Div | 114.36 |
| SW | 113.92 |
| MS | 115.21 |
| GSW | 128.07 |

## Highlights & Insights

1. **Dual Statistical-Computational Guarantee**: Simultaneously establishing sharp statistical convergence rates and efficient computational schemes within a single framework is rare in the literature of OT variants.
2. **Dimension-Free Properties**: The convergence rate of $O(n^{-1/(2p)})$ depends solely on the sample size, completely circumventing the curse of dimensionality under extremely mild conditions (bounded kernel).
3. **NP-hardness and High-Quality Relaxation**: The paper first establishes the intractability of exact computation, and then proposes a convex relaxation with theoretical guarantees, including rank bounds and relaxation gaps.
4. **Practical First-Order Algorithm**: The computational complexity is reduced from $\tilde{O}(n^{6.5})$ for interior point methods to $\tilde{O}(n^2\delta^{-3})$, making medium-scale problems computationally tractable.
5. **Adaptability of Non-linear Projections**: KMS significantly outperforms the linear-projection-based MS in scenarios containing non-linear structures (e.g., mixture distributions, image data).

## Limitations & Future Work

- **Conservative Relaxation Gap**: The ratio provided by Theorem 4.7 contains an $n^{-4}$ factor, suggesting that the theoretical guarantee is pessimistic compared to the much better empirical performance.
- **Rank Reduction Overhead**: The $O(n^5)$ rank reduction algorithm remains a bottleneck for large-scale datasets.
- **Scope Limited to $p=2$**: The computational guarantees (SDR and algorithms) apply only to the 2-Wasserstein distance, leaving cases like $p=1$ unaddressed.
- **Kernel and Bandwidth Selection**: The experiments uniformly employ the Gaussian kernel with a median bandwidth heuristic, lacking theoretical guidance on kernel selection.
- **Limited Scale of High-Dimensional Experiments**: The sample sizes on MNIST/CIFAR-10 are limited by a computation budget (1 hour), leaving larger scales untested.
- **Power Comparison with MMD**: Although KMS is more flexible, MMD with an $O(n^2)$ computational complexity is far more efficient; the cost-performance trade-off in practical applications requires careful balance.

## Related Work & Insights

- **MS Wasserstein**: The linear projection version by Deshpande et al. (2019), of which KMS is a non-linear generalization.
- **Boedihardjo (2025)**: Provided dimension-free sharp bounds for MS Wasserstein; this work extends a similar approach to RKHS.
- **Kernel OT**: Zhang et al. (2019) considered kernel-mapped push-forward Wasserstein, but failed to obtain a sharp rate due to the curse of dimensionality.
- **SDR for OT**: Xie and Xie (2021) first applied SDR to MS $W_1$ but lacked theoretical guarantees and efficient algorithms.
- **Connection to Kernel PCA** (Remark 2.7): When $\nu=\delta_0$ and $p=2$, KMS degenerates to Kernel PCA, and can be viewed as a two-sample generalization of Kernel PCA.

## Rating

- Novelty: ⭐⭐⭐⭐ — The NP-hard proof, SDR rank bounds, and the first-order algorithm constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Extensive comparisons across three tasks (testing/detection/generative modeling) using both synthetic and real-world datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ — The structure of theory-algorithm-experiments is clear, and the theorems are stated precisely.
- Value: ⭐⭐⭐⭐ — Lays a solid foundation for the theory and computation of kernel optimal transport.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)
- [\[NeurIPS 2025\] Wasserstein Transfer Learning](../../NeurIPS2025/optimization/wasserstein_transfer_learning.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](../../NeurIPS2025/optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)

</div>

<!-- RELATED:END -->
