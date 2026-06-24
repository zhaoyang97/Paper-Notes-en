---
title: >-
  [Paper Note] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization
description: >-
  [NeurIPS 2025][Optimization][bilevel optimization] This paper introduces *set smoothness*, a novel structural property of set-valued mappings, proves that it holds naturally in the nonconvex-PŁ bilevel setting, and uses it to reveal hidden weak convexity/concavity structure in the hyper-objective. This yields the first computability guarantee for Clarke stationary points of nonsmooth hyper-objectives.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "bilevel optimization"
  - "set smoothness"
  - "Clarke subdifferential"
  - "weak convexity"
  - "zeroth-order methods"
date: 2026-05-08
content_hash: 84f3dfff998273b1
---

# Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2506.04587](https://arxiv.org/abs/2506.04587)  
**Code**: Unavailable  
**Area**: Optimization
**Keywords**: bilevel optimization, set smoothness, Clarke subdifferential, weak convexity, zeroth-order methods

## TL;DR

This paper introduces *set smoothness*, a novel structural property of set-valued mappings, proves that it holds naturally in the nonconvex-PŁ bilevel setting, and uses it to reveal hidden weak convexity/concavity structure in the hyper-objective. This yields the first computability guarantee for Clarke stationary points of nonsmooth hyper-objectives.

## Background & Motivation

Bilevel optimization (BLO) models hierarchical decision-making: a leader (upper level) optimizes its own objective subject to the constraint that a follower (lower level) solves its own optimization problem given the leader's decision. BLO finds broad applications in hyperparameter optimization, Stackelberg games, reinforcement learning, and interception games.

The standard reduction converts BLO into a single-level problem by defining the **hyper-objective** $\varphi_o(\mathbf{x}) = \min_{\mathbf{y} \in \mathcal{S}(\mathbf{x})} F(\mathbf{x}, \mathbf{y})$ (optimistic) or $\varphi_p(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{S}(\mathbf{x})} F(\mathbf{x}, \mathbf{y})$ (pessimistic), where $\mathcal{S}(\mathbf{x})$ denotes the lower-level solution set. When the lower level is strongly convex, $\mathcal{S}(\mathbf{x})$ is a singleton and $\varphi$ is smooth, so existing methods can locate stationary points. When the lower level admits multiple solutions, however, $\varphi$ is typically **nonsmooth**, and existing methods face two difficulties:

1. Either unrealistic assumptions are required (e.g., the penalized model $\sigma F + f$ satisfies PŁ in $\mathbf{y}$) to recover smoothness;
2. Or only the weaker Goldstein stationarity can be guaranteed, rather than Clarke stationarity.

**Core Problem: Under the general multi-solution lower-level setting, can Clarke hyper-stationary points be computed?**

The challenge chain is: multiple lower-level solutions → nonsmooth hyper-objective → Clarke stationary points of general Lipschitz functions are not computable → new verifiable structural conditions are needed.

## Method

### Overall Architecture

The technical roadmap is: (1) introduce set smoothness → (2) prove it implies weak convexity/concavity of the hyper-objective → (3) prove the lower-level solution mapping satisfies set smoothness under nonconvex-PŁ conditions → (4) exploit weak convexity together with a zeroth-order method to compute Clarke stationary points.

### Key Designs

1. **Set Smoothness (Definition 3)**: A set-valued mapping $\mathcal{Y}: \mathbb{R}^m \rightrightarrows \mathbb{R}^n$ is said to be $L$-smooth if, for all $\mathbf{x}_1, \mathbf{x}_2$, $\theta \in [0,1]$, and any $\mathbf{y} \in \mathcal{Y}(\theta\mathbf{x}_1 + (1-\theta)\mathbf{x}_2)$, there exist $\mathbf{y}_1 \in \mathcal{Y}(\mathbf{x}_1)$ and $\mathbf{y}_2 \in \mathcal{Y}(\mathbf{x}_2)$ such that:

    - Convex combination approximation: $\|\theta\mathbf{y}_1 + (1-\theta)\mathbf{y}_2 - \mathbf{y}\| \leq \frac{L}{2}\theta(1-\theta)\|\mathbf{x}_1 - \mathbf{x}_2\|^2$
    - Uniform branch selection: $\|\mathbf{y}_1 - \mathbf{y}_2\|^2 \leq L\|\mathbf{x}_1 - \mathbf{x}_2\|^2$

   Condition (4) is a natural generalization of scalar smoothness to set-valued mappings — the approximation error of convex combinations is second order. Condition (5) prevents trivial satisfaction via cross-branch pairing (Example 1 provides a counterexample). Together, set smoothness is equivalent to $\mathcal{Y}(\theta\mathbf{x}_1 + (1-\theta)\mathbf{x}_2) \subseteq \theta\mathcal{Y}(\mathbf{x}_1) + (1-\theta)\mathcal{Y}(\mathbf{x}_2) + O(\|\mathbf{x}_1 - \mathbf{x}_2\|^2)\mathbb{B}$.

2. **Set Smoothness Implies Weak Convexity (Theorem 1)**: If $\mathcal{Y}$ is $L_\mathcal{Y}$-smooth, $g$ is $M_g$-Lipschitz and $L_g$-smooth, then $\phi(\mathbf{x}) = \max_{\mathbf{y} \in \mathcal{Y}(\mathbf{x})} g(\mathbf{x}, \mathbf{y})$ is $\rho$-weakly convex with $\rho = M_g L_\mathcal{Y} + L_g(1 + L_\mathcal{Y})$. This is the central structural result — it reduces the regularity of a parametric value function to the smoothness of the constraint set mapping.

3. **Error Bound Implies Set Smoothness (Theorem 2)**: Under standard assumptions ($f$ smooth with Lipschitz second derivatives, solution set nonempty, closed, and convex, and satisfying the error bound $\text{dist}(\mathbf{y}, \mathcal{S}(\mathbf{x})) \leq \tau\|\nabla_\mathbf{y} f(\mathbf{x}, \mathbf{y})\|$), the lower-level solution mapping $\mathcal{S}$ is $L_\mathcal{S}$-smooth.

   The key proof technique is **residual backfilling**: naively projecting $\mathbf{y}$ onto the endpoint fibers $\mathcal{S}(\mathbf{x}_i)$ yields $\bar{\mathbf{y}}_i$, whose convex combination $\bar{\mathbf{y}}^\theta$ may differ from $\mathbf{y}$ by a first-order error (due to branch mismatch). The fix: first project $\bar{\mathbf{y}}^\theta$ onto the intermediate fiber to obtain $\hat{\mathbf{y}}$, then correct the endpoint selections using the residual $\mathbf{y} - \hat{\mathbf{y}}$: $\mathbf{y}_i = \Pi_{\mathcal{S}(\mathbf{x}_i)}(\bar{\mathbf{y}}_i + (\mathbf{y} - \hat{\mathbf{y}}))$. This eliminates the first-order branch mismatch, leaving only second-order remainder terms.

4. **Weak Convexity/Concavity of Hyper-objectives (Theorem 3)**: Combining Theorems 1 and 2, under Assumptions 1–2, the optimistic hyper-objective $\varphi_o$ is $\rho$-weakly concave and the pessimistic hyper-objective $\varphi_p$ is $\rho$-weakly convex.

### Algorithm & Convergence Guarantees

The paper employs an Inexact Zeroth-Order Method (IZOM, Algorithm 1) that approximates directional derivatives of the hyper-objective via finite differences. A subroutine $\mathcal{A}$ approximately evaluates $\varphi(\mathbf{x})$ by solving the inner optimization.

**Theorem 4**: For pessimistic BLO, with step size $\eta = \Theta(m^{-1/2}T^{-1/2})$, $\mathbb{E}[\|\nabla\varphi_{p,\gamma}(\bar{\mathbf{x}})\|^2] = O(\sqrt{m}/\sqrt{T})$. For optimistic BLO, $\mathbb{E}[\text{dist}(\mathbf{0}, \cup_{\mathbf{z} \in \mathbb{B}(\bar{\mathbf{x}}, \delta)} \partial\varphi_o(\mathbf{z}))^2] = O(\sqrt{m}/\sqrt{T})$ with $\delta = O(T^{-1/4})$.

## Key Experimental Results

This paper is primarily a theoretical contribution and contains no standard numerical experiment tables. Key theoretical comparisons are as follows.

### Method Comparison

| Method | Stationarity Guarantee | Lower-Level Strong Convexity Required | Penalized-Model PŁ Required |
|---|---|---|---|
| Implicit gradient descent | $\|\nabla\varphi\| \leq \epsilon$ | Yes | N/A |
| Kwon et al. penalty method | $\|\nabla\varphi\| \leq \epsilon$ | No | Yes |
| Chen et al. (Goldstein) | Goldstein stationarity | No | No |
| **Ours (Clarke)** | **Clarke stationarity** | **No** | **No** |

### Convergence Complexity

| Setting | Complexity | Stationarity Type |
|---|---|---|
| Pessimistic BLO | $O(\sqrt{m}\Delta_p / \sqrt{T})$ | Clarke (via Moreau envelope) |
| Optimistic BLO | $O(\sqrt{m}\Delta_o / \sqrt{T})$ | $(\varepsilon, O(T^{-1/4}))$-Clarke |

### Key Findings

- Clarke stationarity is strictly stronger than Goldstein stationarity: there exist convex Lipschitz functions that are Goldstein stationary at a point yet far from Clarke stationary.
- Lower-level constraints ($\mathbf{y} \in \mathcal{Y}$) can destroy weak convexity (Example 3 provides a concrete counterexample), whereas upper-level constraints do not.
- Set smoothness does not necessarily require error bound conditions (Example 2), suggesting broader sufficient conditions remain to be discovered.

## Highlights & Insights

- "Set smoothness" is an elegant concept of independent interest that generalizes classical scalar smoothness to set-valued mappings, providing a new tool for analyzing value functions of parametric optimization problems.
- The residual backfilling proof technique is elegant — it eliminates first-order branch-mismatch errors through a "project-then-correct" procedure.
- No off-the-shelf Moreau envelope machinery applies to the optimistic (weakly concave) setting; the authors develop a new convergence analysis based on a Brøndsted–Rockafellar-type approximation.
- The analysis provides a unified framework for both optimistic and pessimistic BLO.

## Limitations & Future Work

- The complexity bound carries a $\sqrt{m}$ factor ($m$ being the upper-level dimension), which is unfavorable for high-dimensional problems.
- Weak convexity may fail in the presence of lower-level constraints ($\mathbf{y} \in \mathcal{Y}$), necessitating new structural conditions.
- Faster algorithms exploiting the weak convexity/concavity structure (e.g., proximal-type methods) have yet to be developed.
- Applications of set smoothness to minimax optimization, set-valued optimization, and related areas remain to be explored.

## Related Work & Insights

- This work complements Chen et al. (2023, 2024) on BLO without lower-level strong convexity: their results guarantee only Goldstein stationarity, whereas this paper achieves the stronger Clarke stationarity.
- Set smoothness is related to, but stronger than, classical concepts in variational analysis (Lipschitz continuity, proto-differentiability, etc.), providing finer structural information.
- The Moreau envelope technique for weakly convex optimization (Davis & Drusvyatskiy, 2019) serves as the cornerstone of the algorithmic analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Set smoothness is a novel concept; this is the first proof that Clarke stationary points of nonsmooth hyper-objectives are computable.
- Experimental Thoroughness: ⭐⭐⭐ — Pure theoretical contribution; numerical validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — The logical chain (concept → structural property → algorithm → guarantee) is clear, though mathematical density is high.
- Value: ⭐⭐⭐⭐⭐ — Resolves a fundamental theoretical question in bilevel optimization; set smoothness has broad application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)

</div>

<!-- RELATED:END -->
