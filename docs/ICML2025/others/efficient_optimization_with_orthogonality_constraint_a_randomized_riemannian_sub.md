---
title: >-
  [Paper Note] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method
description: >-
  [ICML2025][Orthogonality-constrained optimization] A randomized Riemannian submanifold descent method (RSDM) is proposed. By restricting each update step to a randomized low-dimensional submanifold, the computational complexity of the retraction operation in orthogonality-constrained optimization is reduced from $O(np^2)$ to $O(r^3)$, while maintaining a total computational complexity that matches that of full-space Riemannian gradient descent.
tags:
  - "ICML2025"
  - "Orthogonality-constrained optimization"
  - "Riemannian optimization"
  - "Stiefel manifold"
  - "Randomized submanifold"
  - "Coordinate descent"
date: 2026-05-08
content_hash: 437c97e8509186c9
---

# Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method

**Conference**: ICML2025  
**arXiv**: [2505.12378](https://arxiv.org/abs/2505.12378)  
**Code**: [andyjm3/RSDM](https://github.com/andyjm3/RSDM)  
**Area**: Others/Optimization  
**Keywords**: Orthogonality-constrained optimization, Riemannian optimization, Stiefel manifold, Randomized submanifold, Coordinate descent

## TL;DR

A randomized Riemannian submanifold descent method (RSDM) is proposed. By restricting each update step to a randomized low-dimensional submanifold, the computational complexity of the retraction operation in orthogonality-constrained optimization is reduced from $O(np^2)$ to $O(r^3)$, while maintaining a total computational complexity that matches that of full-space Riemannian gradient descent.

## Background & Motivation

Orthogonality-constrained optimization $\min_{X \in \mathbb{R}^{n \times p}: X^\top X = I_p} F(X)$ is widely used in scenarios such as PCA, deep network training, and large model fine-tuning. Riemannian optimization is the standard framework for solving such problems, but its core operation, the retraction (mapping tangent space updates back to the manifold), has a computational cost of at least $O(np^2)$ (involving non-standard linear algebra operations such as QR decomposition, polar decomposition, and Cayley transform), which becomes a severe bottleneck in large-scale problems.

Existing coordinate descent methods (e.g., updating only a small number of rows/columns) either suffer from poor running time on GPUs due to excessive iterations, or involve subproblems that are difficult to solve. Infeasible methods avoid retractions but do not strictly maintain orthogonality constraints. This work is motivated by the question: **Is it possible to perform retraction only on a low-dimensional submanifold at each iteration to fundamentally reduce the cost while preserving convergence?**

## Method

### Core Framework: Orthogonal Group Parameterization + Randomized Submanifold

**Key Idea**: Utilizing the transitivity of the orthogonal group $\mathcal{O}(n)$ on the Stiefel manifold $\mathrm{St}(n,p)$, the update is parameterized as $X_{k+1} = U_k X_k$, where $U_k \in \mathcal{O}(n)$. Subsequently, $U_k$ is restricted to an $r$-dimensional submanifold defined by a random orthogonal matrix $P_k$:

$$U_k(Y) = P_k^\top \begin{bmatrix} Y & 0 \\ 0 & I_{n-r} \end{bmatrix} P_k, \quad Y \in \mathcal{O}(r)$$

Consequently, the original optimization on $\mathcal{O}(n)$ is transformed into a small-scale optimization on $\mathcal{O}(r)$, reducing the retraction cost from $O(np^2)$ to $O(r^3)$.

### Algorithm Flow (RSDM)

At each iteration:
1. **Sample** a random orthogonal matrix $P_k \in \mathcal{O}(n)$
2. **Compute submanifold gradient**: $\mathrm{grad}\,\tilde{F}_k(I_r) = P_k(r) \cdot \mathrm{grad}\,F_k(I_n) \cdot P_k(r)^\top$
3. **Submanifold update**: $Y_k = \mathrm{Retr}_{I_r}(-\eta\,\mathrm{grad}\,\tilde{F}_k(I_r))$ (only $r \times r$ retraction)
4. **Iterative update**: $X_{k+1} = U_k(Y_k) X_k$

### Two Sampling Strategies

| Strategy | Sampling Mode | Per-iteration Complexity | Characteristics |
|------|---------|------------|------|
| **Uniform Orthogonal Sampling (RSDM-O)** | Sample $P_k(r) \in \mathbb{R}^{r \times n}$ from Haar measure | $O(npr)$ | Tighter high-probability convergence bound |
| **Uniform Permutation Sampling (RSDM-P)** | Sample $r$ indices without replacement | $O(nr^2)$ | Faster sampling, but high-probability bound has an extra $\binom{n}{r}$ factor |

**Interesting Connection**: When $r=2$ and permutation sampling is used, RSDM degenerates into the classical Givens rotation coordinate descent. Thus, this method strictly generalizes Riemannian coordinate descent.

### Convergence Guarantees

**Expected Convergence** (consistent for both sampling strategies):

- Non-convex: $\min_i \mathbb{E}[\|\mathrm{grad}\,F(X_i)\|^2] \leq O\!\left(\frac{n^2}{r^2 k}\right)$
- Linear convergence under PL condition: $O\!\left(\exp\!\left(-\frac{r^2}{n^2}k\right)\right)$

**Total Complexity**: Iteration count $O(n^2 r^{-2} \epsilon^{-2})$ $\times$ per-step cost $O(nr^2)$ = $O(n^3 \epsilon^{-2})$, which matches the $O(np^2 \epsilon^{-2})$ complexity of RGD when $p = \Theta(n)$.

### Generalization Capability

The method can be directly generalized to quotient manifolds of the orthogonal group (such as Grassmann manifolds, Flag manifolds, etc.) because the algorithm is defined entirely on $\mathcal{O}(n)$.

## Key Experimental Results

### Main Results

Benchmarking is performed on the Procrustes problem and the PCA problem, with baselines including RGD, RCD, TSD, PCAL, and Landing.

| Experiment | Settings | RSDM Performance |
|------|------|----------|
| Procrustes | Various $(n,p)$ | Competitively comparable with the best baseline |
| PCA | Various $(n,p)$ | **Fastest convergence speed** |
| PCA $(2000,1500)$ | Varying $r$ | Robust to the choice of $r$ |
| PCA $(2000,1500)$ | Varying random seeds | Robust to randomness |

### Ablation Study

| Ablation Factor | Findings |
|---------|------|
| Submanifold dimension $r$ | Effective across small to large values of $r$; there exists an efficiency-accuracy trade-off |
| RSDM-O vs RSDM-P | Orthogonal sampling is more stable in practice, while permutation sampling is faster per step |
| Different retractions | QR-based by default; the method is insensitive to the type of retraction |
| Random seeds | The advantage of RSDM over RGD is stably maintained across different seeds |

### Key Findings

- In large-scale problems where $p$ is close to $n$ (where retraction cost dominates), RSDM shows a **significant runtime advantage** compared to RGD.
- **RSDM-P** is faster per step but requires more iterations; **RSDM-O** is slightly slower per step but achieves better convergence.
- The theoretically predicted $O(n^2/r^2)$ scaling relationship is validated in the experiments.

## Highlights & Insights

1. **Elegant Parameterization**: By leveraging the transitive action of the orthogonal group combined with block-diagonal embedding, the large-scale retraction problem on the manifold is perfectly decomposed into small-scale problems.
2. **Unified Framework**: Coordinate descent ($r=2$) is a special case, providing for the first time a continuous spectrum from coordinate descent to full-space gradient descent.
3. **Unified Analysis**: Orthogonal sampling and permutation sampling are completely consistent in expected convergence rates ($r(r-1)/n(n-1)$), with differences only reflected in the high-probability bounds.
4. **Scalability**: The method generalizes naturally to quotient manifolds such as Grassmann and Flag without requiring any algorithmic modifications.
5. **Practical Effectiveness**: PyTorch implementation, GPU-friendly, open-source code.

## Limitations & Future Work

1. **Total Complexity Matching Only When $p = \Theta(n)$**: When $p \ll n$, the retraction itself is not the bottleneck, and the advantage of RSDM diminishes.
2. **The $\binom{n}{r}$ Factor of Permutation Sampling in High-probability Bounds**: Theoretically, this is highly likely to be loose, but it cannot be improved at present.
3. **Selection of Submanifold Dimension $r$**: Lacks an adaptive strategy and currently relies on manual tuning.
4. **Second-order/Variance-reduced/Momentum Versions of RSDM are Not Discussed**, which could potentially accelerate convergence further.
5. **Lack of Large-scale Deep Learning Scenarios**: Experiments are mainly validated on synthetic tasks and PCA, lacking evaluations in large-scale deep learning scenarios (e.g., orthogonality-constrained fine-tuning of LLMs).

## Related Work & Insights

- **Textbooks on Stiefel Manifold Optimization**: Absil et al. (2008), Boumal (2023) — Methodological foundation of this work.
- **Coordinate Descent**: Shalit & Chechik (2014), Han et al. (2024a), Yuan (2023) — RSDM with $r=2$ is a special case.
- **Infeasible Methods**: Gao et al. (2019), Ablin & Peyré (2022) — Sacrificing orthogonality feasibility for efficiency.
- **Euclidean Randomized Subspace Methods**: Kozak et al. (2021) — Direct inspiration for RSDM; this work generalizes it to manifolds.
- **Orthogonality-constrained Fine-tuning**: Qiu et al. (2023), Liu et al. (2024) — Potential application scenarios.
- **Insights**: The idea of "decomposing a large problem into randomized subspaces" might be applicable to other manifolds (e.g., fixed-rank matrices, positive definite cones).

## Rating

- Novelty: ⭐⭐⭐⭐ — Orthogonal group parameterization + randomized submanifold is a novel and natural combination.
- Experimental Thoroughness: ⭐⭐⭐ — Well-validated on synthetic tasks, but lacks large-scale practical applications.
- Writing Quality: ⭐⭐⭐⭐⭐ — Complete theoretical proofs, clear framework, and includes unification analysis.
- Value: ⭐⭐⭐⭐ — Provides a scalable new tool for large-scale orthogonality-constrained optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](exploiting_similarity_for_computation_and_communication-efficient_decentralized_.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](../../AAAI2026/others/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](../../ACL2025/others/taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)

</div>

<!-- RELATED:END -->
