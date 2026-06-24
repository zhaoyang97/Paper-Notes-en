---
title: >-
  [Paper Note] Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization
description: >-
  [ICML 2025][Recommender Systems][Pairwise Comparisons] Provides the first theoretical recovery guarantee for non-convex matrix factorization based on pairwise comparison data in recommendation systems: proving that under a warm start condition, projected gradient descent converges exponentially to the true low-rank feature matrix with a nearly optimal sample complexity of $O(nr^2 \log n)$. The key technical contribution extends the matrix Bernstein inequality to the sampling…
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Pairwise Comparisons"
  - "Collaborative Filtering"
  - "Non-convex Matrix Factorization"
  - "Projected Gradient Descent"
  - "Convergence Guarantees"
date: 2026-05-08
content_hash: 6b403c077df5107a
---

# Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization

**Conference**: ICML 2025  
**arXiv**: [2502.20033](https://arxiv.org/abs/2502.20033)  
**Code**: Not publicly available  
**Area**: Recommendation Systems / Optimization Theory  
**Keywords**: Pairwise Comparisons, Collaborative Filtering, Non-convex Matrix Factorization, Projected Gradient Descent, Convergence Guarantees

## TL;DR

Provides the first theoretical recovery guarantee for non-convex matrix factorization based on pairwise comparison data in recommendation systems: proving that under a warm start condition, projected gradient descent converges exponentially to the true low-rank feature matrix with a nearly optimal sample complexity of $O(nr^2 \log n)$. The key technical contribution extends the matrix Bernstein inequality to the sampling matrix structure of pairwise comparisons.

## Background & Motivation

### Background

Classical recommendation systems learn preferences from user rating data via Matrix Completion: assuming the user-item utility matrix $X^* = U^* \Sigma^* V^{*T}$ is low-rank, the complete matrix is recovered from partially observed entries. Prior work has established a mature theoretical framework for this non-convex problem (Keshavan et al. 2010, Zheng & Lafferty 2016, Ge et al. 2016).

### Core Problem

However, **pairwise comparison data** is equally prevalent in practice: a user clicking one of the options implicitly indicates a preference for the chosen item over the unchosen one. Comparison data offers three major advantages over traditional ratings: (1) naturally eliminating user rating bias; (2) avoiding rating discretization issues; (3) comparing two items incurs less cognitive load than providing an absolute rating.

When user $u$ compares items $i$ and $j$, the probability of choosing $i$ is $g(x_{u,i}^* - x_{u,j}^*)$, where $g(\cdot)$ is the link function (e.g., sigmoid). The goal is to recover the feature matrix $Z^*$ of the low-rank utility matrix $X^*$ from sparse comparison data.

### Limitations of Prior Work

Park et al. (2015) and Negahban et al. (2018) provided theoretical guarantees for convex relaxation versions of this problem, but convex methods require optimizing the entire $n_1 \times n_2$ utility matrix, which is computationally expensive. In practice (e.g., BPR algorithm, Rendle et al. 2009), non-convex matrix factorization $X = UV^T$ is utilized to reduce parameters from $n_1 n_2$ to $(n_1+n_2)r$, which significantly improves computational efficiency. However, the non-convex formulation lacks theoretical guarantees, which Negahban et al. (2018) explicitly highlighted as an important open problem.

### Ours

**Provides the first theoretical recovery guarantee for the non-convex matrix factorization of comparison data**, filling this important theoretical gap.

## Method

### Overall Architecture

Problem modeling: Low-rank utility matrix $X^* = U^* \Sigma^* V^{*T}$ $\to$ Converted to symmetric form $Z^* = [U^*; V^*] \Sigma^{*1/2}$ $\to$ Defined negative log-likelihood loss $\mathcal{L}(Z)$ + regularization $\mathcal{R}(Z)$ to eliminate scale invariance $\to$ Projected gradient descent to recover $Z^*$

### Key Designs

#### 1. Problem Symmetrization and Regularization

**Function**: Converts asymmetric matrix factorization $X = UV^T$ into a symmetric form $Y = ZZ^T$, where $Z = [U; V] \in \mathbb{R}^{n \times r}$.

**Regularization Design**: Adds $\mathcal{R}(Z) = \|Z^T D Z\|_F^2$ ($D = \text{diag}(I_{n_1}, -I_{n_2})$) to eliminate scale invariance, making the objective function:

$$f(Z) = \mathcal{L}(Z) + \frac{\lambda}{4}\mathcal{R}(Z)$$

Setting $\lambda = \xi\gamma/4$ precisely eliminates the adverse terms brought by rotation invariance.

**Design Motivation**: The original problem is invariant to $Z \to ZR$ (rotation) and $(U, V) \to (UP^T, VP^{-1})$ (scale). These symmetries prevent the establishment of strong convexity. Regularization constrains the problem to a "balanced" solution space $\mathcal{H} = \{Z: \mathbf{1}^T V = 0\}$.

#### 2. Projected Gradient Descent (Algorithm 1)

**Algorithm Flow**:

$$Z_{t+1} = \mathcal{P}_\mathcal{H}(\mathcal{P}_\mathcal{C}(Z_t - \eta \nabla f(Z_t)))$$

Two-step projection:
1. $\mathcal{P}_\mathcal{C}$: Clips the $\ell_2$ norm of each row to $\beta = \frac{4}{3}\sqrt{\mu/n}\|Z_0\|_F$ to maintain incoherence.
2. $\mathcal{P}_\mathcal{H}$: Projects onto the subspace $\mathcal{H}$ to eliminate translation invariance.

**Design Motivation**: The loss function exhibits strong-convexity-like properties only within the region of incoherent matrices; projection ensures that iterations always stay within this region.

#### 3. New Techniques of Concentration Inequalities (Core Technical Contribution)

**Challenge**: In classical matrix completion, the sampling matrix $A$ corresponds to a single $(u, i)$ entry, which can exploit projection operator properties (Candès & Recht 2009) or bipartite graph structures. However, in comparison data, each data point corresponds to a $(u; i, j)$ triplet, resulting in a different sampling matrix structure:

$$A = \begin{bmatrix} 0 & e_u(e_i - e_j)^T \\ 0 & 0 \end{bmatrix}$$

This makes classical concentration results no longer applicable.

**Ours**: Starting from the Matrix Bernstein inequality, an auxiliary "dual sampling matrix" $B = e_u(e_i + e_j)^T$ is defined. Concentration bounds for $\|S_\mathcal{D} - \bar{S}\|_2$ and $\|B_\mathcal{D} - \bar{B}\|_2$ are established respectively, deriving strong convexity-like and smoothness-like properties (Lemma 5.1 and 5.2).

### Main Theorem (Theorem 4.1)

Assuming the number of samples $m \geq 10^7 (\mu r \kappa / \tau)^2 n \log(8n/\delta)$, and the initial point $Z_0$ is within the $\tau/50$-neighborhood of the true solution, with step size $\eta\alpha \leq 2.5 \times 10^{-6} (\tau/\mu r \kappa)^2$, then with probability $\geq 1-\delta$:

$$\|\Delta(Z_t)\|_F^2 \leq \left(1 - \frac{\alpha\eta}{4}\right)^t \|\Delta(Z_0)\|_F^2$$

which indicates **exponential convergence** to the true solution. The dependency of sample complexity on the problem scale is $O(nr^2 \log n)$, which is nearly optimal.

## Key Experimental Results

### Simulation Experiments (Synthetic Data)

| Settings | $(n_1, n_2)$ | Rank $r$ | Conclusion |
|------|-------------|--------|------|
| Different initializations ($\vartheta \in \{0.5, 1, 2\}$) | (200,300) and (2000,3000) | 3 | Linear convergence in all cases, verifying theoretical predictions |
| Different sample sizes $m$ | (200,300) and (2000,3000) | 3 | Linear convergence once $m$ exceeds the theoretical threshold; otherwise, error does not converge to zero |
| Different ranks $r \in \{2,...,6\}$ | (2000,3000) | Varies | Error increases with $r$, and sample requirements empirically scale as approximately $O(nr)$ |

### Key Findings

- **Warm start is unnecessary in practice**: While theory requires the initial point to be close to the true solution, randomized initialization also converges in simulations.
- **Projection is unnecessary in practice**: Ordinary gradient descent without projection also works.
- **Theoretical constants are conservative**: The theoretically required sample constant $c_0 = 10^7$ is much larger than the practically needed $c_0 \approx 1/4$.
- **Sample complexity could be tighter**: Experiments show that $m$ is actually linearly related to $r$ instead of $r^2$, suggesting potential for theoretical improvement.

## Highlights & Insights

- **Filling an important theoretical gap**: The BPR algorithm has been used in recommendation systems for over 15 years; this work is the first to establish theoretical guarantees for it, answering an open question raised by Negahban et al. (2018).
- **Dual sampling matrix technique**: Introducing $B = e_u(e_i + e_j)^T$ to handle the concentration problem of triplet structures is a key innovation in extending the matrix completion theoretical toolbox to comparison models.
- **The gap between theory and practice is a positive signal**: While the theory requires warm start and projections, they are not necessary in practice, suggesting that the practical landscape of the problem is more friendly than theoretical analysis indicates.

## Limitations & Future Work

- **Assumption of noiseless observations**: The current framework assumes observing the expected value of comparison probability $g(x_{u,i}^* - x_{u,j}^*)$ rather than binary outcomes. Extending this to noisy settings is an important direction.
- **Requirement of a warm start**: Theoretical analysis depends on initial points being near the true solution. Removing this assumption (e.g., proving all local minima are global minima) remains an open problem.
- **Experiments limited to synthetic data**: Lacks verification on real recommendation datasets (e.g., Netflix); the authors admit that publicly available pairwise comparison recommendation datasets are currently lacking.
- **Overly large constants**: The constant of $10^7$ in the sample complexity is far from what is required in practice. A more refined analysis could potentially improve this.

## Related Work & Insights

- **vs. Convex Methods (Negahban et al. 2018)**: Convex methods provide optimal sample complexity guarantees but require optimizing an $n_1 \times n_2$ matrix, raising computational intractability. In contrast, the non-convex method proposed here uses only $(n_1+n_2)r$ parameters, which is highly practical but lacked theoretical guarantees until now.
- **vs. Matrix Completion Theory (Zheng & Lafferty 2016)**: The proof strategy of this work is inspired by theirs (symmetrization + projected gradient + regularization), but the core concentration inequality requires entirely new techniques due to the fundamental differences in the sampling structures.
- **vs. Ge et al. (2016)**: They proved that the non-convex objective of matrix completion is free of spurious local minima. A similar analysis for comparison models remains an open question.

## Rating

- Novelty: ⭐⭐⭐⭐ Provides the first theoretical guarantee for the non-convex factorization of comparison data, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐ Verification is limited to synthetic simulations, lacking real-world dataset experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly presented theory with a well-organized proof structure.
- Value: ⭐⭐⭐⭐ Makes a significant contribution to the recommendation system theory community, though practical impact is limited by the noiseless assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations](recommendations_and_reporting_checklist_for_rigorous_transparent_human_baselines.md)
- [\[AAAI 2026\] Interpretable Reward Model via Sparse Autoencoder](../../AAAI2026/recommender/interpretable_reward_model_via_sparse_autoencoder.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[AAAI 2026\] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information](../../AAAI2026/recommender/generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)
- [\[ICLR 2026\] Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](../../ICLR2026/recommender/adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)

</div>

<!-- RELATED:END -->
