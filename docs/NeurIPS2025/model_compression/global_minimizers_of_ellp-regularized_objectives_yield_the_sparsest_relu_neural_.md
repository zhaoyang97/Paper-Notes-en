---
title: >-
  [Paper Note] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks
description: >-
  [NeurIPS 2025][Model Compression][ℓp regularization] This paper proves that, for single-hidden-layer ReLU networks, global minimizers of the $\ell^p$ ($0 < p < 1$) path norm correspond exactly to the **sparsest** data-in…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "ℓp regularization"
  - "ReLU networks"
  - "sparsity"
  - "global minimizers"
  - "network pruning"
date: 2026-05-08
content_hash: 25cfbc7d3265d766
---

# Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2505.21791](https://arxiv.org/abs/2505.21791)  
**Code**: None  
**Area**: Neural Network Sparsification / Optimization Theory
**Keywords**: ℓp regularization, ReLU networks, sparsity, global minimizers, network pruning

## TL;DR

This paper proves that, for single-hidden-layer ReLU networks, global minimizers of the $\ell^p$ ($0 < p < 1$) path norm correspond exactly to the **sparsest** data-interpolating networks, thereby recasting the combinatorial sparse interpolation problem as a continuously differentiable optimization task.

## Background & Motivation

### Importance of Sparse Networks

Overparameterized neural networks can interpolate a given dataset in many ways. A fundamental question is: among all feasible solutions, which should be preferred? **Sparse** networks (those with fewest parameters or neurons) are particularly valuable for the following reasons:
- **Computational efficiency**: lower storage and inference overhead
- **Generalization**: sparse networks empirically tend to generalize better
- **Interpretability**: fewer neurons yield more transparent model structures
- **Model compression**: provides a foundation for knowledge distillation and deployment

### Limitations of Prior Work

Existing sparsification strategies are largely **heuristic** and lack theoretical guarantees:

| Method Category | Representative Methods | Theoretical Guarantee |
|:---|:---|:---|
| Pruning | Lottery Ticket, Magnitude Pruning | No sparsest-solution guarantee |
| $\ell_1$ Regularization | Lasso, path norm minimization | Non-unique solutions; possibly non-sparsest |
| Structured Sparsity | Group Lasso, $\ell_{2,1}$ | Group-level sparsity, not finest-grained |
| Stochastic Gating | $\ell_0$ approximation | Approximate solutions, no exact guarantees |

Notably, $\ell_1$ path norm minimization, despite its apparent promotion of sparsity, has the following shortcomings in the neural network setting:
- Solutions are generally **non-unique**
- Certain $\ell_1$-optimal solutions may have **arbitrarily many** neurons
- The intuition "$\ell_1 =$ sparsity" therefore does not carry over to neural networks

### Paper Goals

This paper proposes a continuously differentiable regularization objective based on the $\ell^p$ ($0 < p < 1$) quasi-norm, whose global minimizers are **provably** the sparsest ReLU networks.

## Method

### Overall Architecture

#### Univariate Case (Input Dimension $d = 1$)

Consider a single-hidden-layer ReLU network of the form:

$$f_{\boldsymbol{\theta}}(x) = \sum_{k=1}^{K} v_k (w_k x + b_k)_+ + ax + c$$

The $\ell^p$ path norm minimization problem is defined as:

$$\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} |w_k v_k|^p \quad \text{s.t.} \quad f_{\boldsymbol{\theta}}(x_i) = y_i, \; i=1,\ldots,N$$

#### Multivariate Case (Input Dimension $d > 1$)

For ReLU networks mapping $\mathbb{R}^d \to \mathbb{R}$, an additional $\ell_\infty$ constraint is introduced:

$$\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} \|v_k \mathbf{w}_k\|_p^p \quad \text{s.t.} \quad f_{\boldsymbol{\theta}}(\mathbf{x}_i) = y_i, \; \|v_k \mathbf{w}_k\|_\infty \leq R$$

### Key Designs

#### Variational Reformulation (Proposition 3.1)

The network optimization problem is equivalently reformulated as a variational problem over continuous piecewise-linear (CPWL) functions:

$$\min_f V_p(f), \quad \text{s.t.} \quad f(x_i) = y_i$$

where $V_p(f)$ denotes the $p$-variation of the derivative of $f$. For a CPWL function with $K$ breakpoints:

$$V_p(f) = \sum_{k=1}^{K} |c_k|^p$$

with $c_k$ being the slope change at each breakpoint. This reformulation transforms the parameter-space optimization into a geometric problem over function space.

#### Geometric Characterization (Theorem 3.1)

The core theorem precisely characterizes the geometric behavior of optimal functions for $0 < p < 1$:

1. **Linear regions**: The function must be linear before $x_2$ and after $x_N$, and also between consecutive data points with alternating discrete curvature signs.
2. **Constant-curvature regions**: Between $m$ consecutive data points $x_i, \ldots, x_{i+m}$ sharing the same curvature sign, the function is convex (or concave) with at most $m-1$ breakpoints.
3. **Slope constraints**: The slope $u_j$ at each breakpoint satisfies $s_{i+j-1} \leq u_j \leq s_{i+j}$ (assuming positive curvature).

#### Uniqueness and Sparsity (Theorem 3.2)

For Lebesgue almost every $0 < p < 1$, the solution is **unique**. Moreover, there exists a data-dependent threshold $p^*$ such that for $0 < p < p^*$, the $\ell^p$-optimal solution coincides exactly with the sparsest ($\ell^0$-optimal) interpolating network.

In particular:
- Any $\ell^p$-optimal solution has at most $N-2$ active neurons.
- By contrast, $\ell^1$-optimal solutions may have arbitrarily many neurons.

#### Multivariate Extension (Theorem 4.1)

Using the Bauer maximum principle and properties of concave function minimization over polytopes:
- The optimization problem is reduced to minimizing a finite-dimensional concave function over a polytope.
- The optimal solution must be attained at an extreme point of the polytope.
- There exists $p^*$ such that $\ell^p$-$\ell^0$ equivalence holds.

### Loss & Training

This paper does not address specific training algorithms. The core contribution lies in establishing the correct form of the objective function. The $\ell^p$ path norm objective is continuous and almost everywhere differentiable, and can in principle be optimized via gradient descent (albeit non-convexly).

## Key Experimental Results

### Comparison of Theoretical Guarantees

This is a purely theoretical work. Core results are presented as theorems:

| Theorem | Dimension | Result | Condition |
|:---|:---|:---|:---|
| Theorem 3.1 | $d=1$ | Geometric characterization: optimal function is piecewise linear + locally convex/concave | $0 < p < 1$ |
| Corollary 3.1.1 | $d=1$ | $\ell^p$-optimal $\Rightarrow$ $\ell^1$-optimal; $\leq N-2$ neurons | $0 < p < 1$ |
| Theorem 3.2 | $d=1$ | Almost unique; $\exists\, p^*$ such that $\ell^p = \ell^0$ | No data assumptions required |
| Proposition 4.1 | Any $d$ | $\ell^0$-optimal solution has $\leq N$ active neurons | General position assumption |
| Theorem 4.1 | Any $d$ | $\exists\, p^*$ such that $\ell^p$-optimal $=$ $\ell^0$-optimal | Requires $\ell_\infty$ constraint |

### Sparsity Guarantees Across Regularization Strategies

| Regularization Strategy | Sparsest Guarantee | Neuron Count Upper Bound | Solution Uniqueness |
|:---|:---|:---|:---|
| $\ell_1$ path norm | **No** (arbitrarily many neurons possible) | None (unbounded in worst case) | Non-unique |
| $\ell_0$ path norm | Yes (by definition) | $N$ | Non-unique |
| **$\ell^p$ ($0 < p < 1$)** | **Yes (for sufficiently small $p$)** | $N-2$ ($d=1$) / $N$ (any $d$) | **Almost unique** |
| Weight decay ($\ell_2$) | No | No guarantee | — |

### Key Findings

1. **$\ell_1$ does not guarantee sparsity in neural networks**: Unlike finite-dimensional compressed sensing where $\ell_1$ recovers sparse solutions, $\ell_1$ path norm minimization in neural networks can yield solutions with arbitrarily many neurons, because the parameterization of CPWL functions is non-unique.

2. **$\ell^p$ ($0 < p < 1$) simultaneously minimizes $\ell^1$ and $\ell^0$**: Theorem 3.1 shows that $\ell^p$-optimal solutions are also $\ell^1$-optimal, thereby achieving both sparsity (few neurons) and norm control (bounded weights).

3. **From combinatorial to continuous optimization**: While $\ell^0$ minimization is NP-hard, this paper proves that minimizing the continuously differentiable $\ell^p$ objective exactly recovers the $\ell^0$ solution.

4. **Constraints are necessary in the multivariate case**: Contrary to claims by Peng et al. (2015), unconstrained $\ell^p$ minimization does not necessarily yield bounded solutions (this paper corrects an error in their proof); hence an explicit $\ell_\infty$ constraint is required in the multivariate setting.

## Highlights & Insights

- **Conceptual breakthrough**: The first rigorous proof that a continuous regularization objective can exactly recover the sparsest neural network.
- **Variational perspective**: Recasting neural network optimization as a variational problem over the space of CPWL functions yields strong geometric intuition.
- **Elegant use of the Bauer maximum principle**: Exploiting the extremal properties of concave functions on convex sets reduces an infinite-dimensional problem to the comparison of finitely many candidate solutions.
- **Correction of a prior error**: Identifies and corrects an incorrect claim in Peng et al. (2015) regarding the boundedness of $\ell^p$ minimization solutions.

## Limitations & Future Work

- Results are limited to **single-hidden-layer** ReLU networks; extension to deep networks is an important direction for future work.
- No method is provided for computing the threshold $p^*$; how to select $p$ in practice remains an open problem.
- Only **exact interpolation** is considered; the case with training error (approximate fitting) is not addressed.
- The $\ell_\infty$ constraint required in the multivariate case needs further study regarding practical implementation during training.
- Convergence of non-convex optimization: even with the correct objective, whether gradient descent can find the global optimum remains to be analyzed.

## Related Work & Insights

- **Debarre et al. (2022)**: Characterizes $\ell^1$ path norm optimal solutions but cannot guarantee sparsity.
- **Boursier & Flammarion (2023)**: Proves sparsity under $\ell^1$ + bias regularization with strong data assumptions.
- **Ergen & Pilanci (2021)**: Requires $d > N$ and whitened data assumptions.
- **Pilanci & Ergen (2020)**: Convex reformulation framework, which inspired the derivation of Lemma 4.1 in this paper.
- **Parhi & Nowak (2021, 2022)**: Representation theorems within a variational framework; the present work is a natural extension thereof.
- The results of this paper provide a theoretical foundation for designing new sparse training algorithms (e.g., reweighted $\ell^1$ methods).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First rigorous establishment of $\ell^p$-$\ell^0$ equivalence
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Sophisticated variational analysis and convex geometry arguments
- **Experimental Thoroughness**: ⭐⭐⭐ — Theoretical breakthrough, but algorithmic design requires follow-up work
- **Writing Quality**: ⭐⭐⭐⭐ — Geometric intuition is well presented, though mathematical details are dense
- **Overall Score**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](../../ICLR2026/model_compression/topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
