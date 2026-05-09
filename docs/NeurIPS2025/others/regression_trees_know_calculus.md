---
title: >-
  [Paper Note] Regression Trees Know Calculus
description: >-
  [NeurIPS 2025][regression trees] This paper reveals gradient information latent in piecewise-constant regression trees — by treating the difference in child-node means as a finite-difference analogue, it efficiently extracts gradient estimates, thereby importing differential tools such as Active Subspaces (AS) and Integrated Gradients (IG) into tree models, broadening both their interpretability and predictive improvement capabilities.
tags:
  - NeurIPS 2025
  - regression trees
  - gradient estimation
  - active subspace
  - integrated gradients
  - interpretability
date: 2026-05-08
content_hash: 578ef03b91e04a36
---

# Regression Trees Know Calculus

**Conference**: NeurIPS 2025
**arXiv**: [2405.13846](https://arxiv.org/abs/2405.13846)
**Code**: Unavailable
**Area**: Other
**Keywords**: regression trees, gradient estimation, active subspace, integrated gradients, interpretability

## TL;DR

This paper reveals gradient information latent in piecewise-constant regression trees — by treating the difference in child-node means as a finite-difference analogue, it efficiently extracts gradient estimates, thereby importing differential tools such as Active Subspaces (AS) and Integrated Gradients (IG) into tree models, broadening both their interpretability and predictive improvement capabilities.

## Background & Motivation

Regression trees — including random forests and gradient boosting — are among the most widely used tools in data science, approximating target functions with piecewise-constant representations. Because their primary design goal is to handle nonlinearity, interaction effects, and sharp discontinuities, the behavior of tree models when approximating smooth, differentiable functions has received comparatively little attention.

Existing interpretability tools for tree models include:
- **MDI (Mean Decrease in Impurity)**: allocates impurity reduction to features at each split, but is known to suffer from bias.
- **Feature Permutation**: measures performance degradation after shuffling a feature, but performs poorly when features are correlated.
- **SHAP**: a general attribution framework, but computationally expensive.

On the other hand, differentiable models such as neural networks and Gaussian processes enjoy rich gradient-based analysis tools:
- **Integrated Gradients (IG)**: integrates gradients along the path from a baseline to the target point to provide local attribution.
- **Active Subspace (AS)**: discovers important linear directions via eigendecomposition of the expected outer product of gradients.

These methods cannot be directly applied to tree models because the formal gradient of a piecewise-constant function is zero almost everywhere. The central insight of this paper is that, although the formal gradient of a tree is zero, **the difference in means across the two sides of a split node implicitly encodes gradient information**, which can be used to construct a gradient estimator analogous to finite differences.

## Method

### Overall Architecture

Each internal node of a regression tree of depth $K$ is treated as a finite-difference sample point from which an estimate of the partial derivative with respect to the splitting variable is extracted. Traversing the tree from root to leaf, gradient vectors are updated layer by layer, and a complete gradient estimate is ultimately associated with each leaf node. Building on this, estimators for integral-differential quantities such as the active subspace matrix and integrated gradients are further constructed.

### Key Designs

1. **Tree-Based Gradient Estimator (TBGE)**:

   For node $i$, which splits data along variable $\sigma_i$ into left and right child nodes with response-variable means $\hat{\mu}_l^i$ and $\hat{\mu}_r^i$ respectively, and with range $[l_{\sigma_i}^i, u_{\sigma_i}^i]$ along dimension $\sigma_i$, the estimator is defined as:

   $\gamma_i := \frac{2(\hat{\mu}_r^i - \hat{\mu}_l^i)}{u_{\sigma_i}^i - l_{\sigma_i}^i} \approx \frac{\partial f}{\partial x_{\sigma_i}}(\mathbf{x}), \quad \forall \mathbf{x} \in [\mathbf{l}^i, \mathbf{u}^i]$

   This is essentially the difference in child-node means divided by the node width — a direct analogue of classical finite differences. Each node can estimate only one partial derivative (along the splitting variable), but by composing estimates from nodes at different depths during a root-to-leaf traversal, a complete gradient vector $\mathbf{G}^i$ is assembled.

   **Design Motivation**: When nodes are sufficiently small and the function gradient varies smoothly, this finite-difference approximation is well justified. The computation requires only a single tree traversal, incurring the same complexity as standard prediction.

2. **Tree-Based Active Subspace (TBAS)**:

   Using the Partition-Based Estimator (PBE), the active subspace matrix is constructed by taking a weighted sum of outer products of gradient estimates over all nodes at the penultimate layer:

   $\hat{C}_\mu^f = \sum_{i \in \mathcal{N}_{K-1}} \mathbf{G}^i (\mathbf{G}^i)^\top \mu([\mathbf{l}^i, \mathbf{u}^i])$

   where $\mu([\mathbf{l}^i, \mathbf{u}^i])$ is the measure of node $i$ under $\mu$ (equal to the volume fraction under a uniform distribution). Eigendecomposition of this matrix yields important linear feature combinations.

   **Design Motivation**: No additional sampling or auxiliary models are required; the method directly reuses information already encoded in the tree structure. Moreover, since trees naturally tend to split on important variables, TBAS inherits a built-in sparsity inductive bias.

3. **Tree-Based Integrated Gradients (TBIG)**:

   Monte Carlo samples are drawn along the path between a baseline $\mathbf{x}^*$ and a target point $\mathbf{x}$, with TBGE substituted for the true gradient:

   $\hat{IG}(\mathbf{x}) = (\mathbf{x} - \mathbf{x}^*) \odot \frac{1}{M} \sum_{m=1}^{M} \tilde{\nabla}f(u_m \mathbf{x} + (1-u_m)\mathbf{x}^*)$

   **Design Motivation**: This transplants the local attribution method from neural networks into tree models, enabling visualization of feature importance for random forest classifiers.

### Theoretical Guarantees

**Theorem 4.1** (Gradient Estimation Consistency): When the number of splits $S(N)$ grows slower than $\log N$, the TBGE converges to the true gradient at a rate of $O_P(P(\log N)^{-1})$. Although the convergence rate is slow, the method benefits from the computational efficiency of trees.

**Theorem 4.2** (Integral Quantity Consistency): Under boundedness conditions, integral-differential quantities estimated by the PBE converge to their true values as the sample size tends to infinity.

## Key Experimental Results

### Main Results: Effect of TBAS Rotation on Prediction

| Dataset | Method | Depth-4 Single Tree | Depth-8 Single Tree | Depth-4 Random Forest |
|---------|--------|--------------------|--------------------|----------------------|
| kin40k | TBAS | **0.856** | **0.586** | **0.802** |
| kin40k | No Rotation (Id) | 0.963 | 0.862 | 0.954 |
| kin40k | PCA | 0.964 | 0.872 | 0.954 |
| keggu | TBAS | **0.194** | **0.078** | **0.161** |
| keggu | No Rotation (Id) | 0.350 | 0.121 | 0.344 |
| concrete | TBAS | **0.470** | **0.350** | **0.406** |
| concrete | No Rotation (Id) | 0.537 | 0.403 | 0.462 |

(Values are 100-fold RMSE; lower is better. TBAS significantly outperforms both the no-rotation and PCA-rotation baselines across multiple datasets.)

### Ablation Study: Active Subspace Estimation Efficiency Comparison

| Dimensionality | Method | Runtime Trend | Estimation Error Trend | Notes |
|---------------|--------|--------------|----------------------|-------|
| 2–4D | TBAS | Much faster | Comparable | Dominates most of the Pareto frontier |
| 2–4D | GP | Fast at small $N$, intractable at large $N$ | Better at small $N$ | Computation explodes beyond ~150 samples |
| 2–4D | PRA | Similar to GP | Similar to GP | Polynomial Ridge Approximation |
| 2–4D | DASM | Moderate | Moderate | Neural network-based method |
| 10–100D | TBAS | Far superior | Far superior | Sparsity inductive bias is pronounced |
| 10–100D | DASM | Only comparable method | Requires larger $N$ | GP/PRA infeasible |

### Key Findings

1. TBAS matches or outperforms all other rotation methods across 8 benchmark datasets, reducing RMSE by 10–50% on kin40k and keggu.
2. In low-dimensional (2–4D) settings, TBAS achieves accuracy comparable to UQ-specialized methods at orders-of-magnitude faster speeds.
3. In high-dimensional (10–100D) sparse settings, TBAS naturally favors coordinate-aligned active subspaces, substantially outperforming DASM.
4. TBIG successfully identifies discriminative pixel regions for digit classes on the MNIST classification task.

## Highlights & Insights

1. **Conceptual Elegance**: The work is grounded in a minimal observation — the difference in means across a split is a finite difference — yet this observation bridges tree models with the full toolkit of calculus-based analysis.
2. **Computational Efficiency**: Gradient estimation requires only a single tree traversal, with no additional data or auxiliary models needed.
3. **Bidirectional Value**: The contribution enriches both directions — importing differential tools into tree models (enhancing interpretability) and introducing tree models into the uncertainty quantification (UQ) literature (providing efficient estimators).
4. **Sparsity Inductive Bias**: The tree's splitting mechanism naturally favors important variables, endowing TBAS with sparsity without any additional regularization.

## Limitations & Future Work

- Theoretical analysis rests on simplifying assumptions (cyclic variable splitting, median splits) that diverge from practical algorithms such as CART.
- The gradient estimation convergence rate of $O((\log N)^{-1})$ is slow, making the method most suitable for scenarios where rough estimates suffice.
- In high dimensions, very deep trees are required to obtain high-accuracy gradient estimates.
- Extensions to classification tasks and categorical features remain at a preliminary stage.
- Since each layer overwrites the previous layer's gradient estimate, aggregating estimates across multiple layers may yield improved convergence rates.

## Related Work & Insights

- **Tree Interpretability**: SHAP, MDI, Feature Permutation
- **Gradient Attribution**: Integrated Gradients, Active Subspace Method
- **Active Subspace Estimation**: GP-based methods, PRA, DASM
- Inspiration: Can other non-differentiable models (e.g., KNN, kernel methods) benefit from a similar "implicit gradient" perspective? Applications of tree models in physics-informed machine learning (PIML) also merit exploration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A highly insightful contribution; the first work to reveal and exploit gradient estimates implicit in piecewise-constant regression trees.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation spans predictive improvement, low/high-dimensional active subspace estimation, and visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ — Notation is clear; the exposition proceeds systematically from intuition to theory to experiments.
- Value: ⭐⭐⭐⭐ — A bridging work connecting tree models with differential tools, opening several promising directions for future research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[NeurIPS 2025\] Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination](information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)

<!-- RELATED:END -->
