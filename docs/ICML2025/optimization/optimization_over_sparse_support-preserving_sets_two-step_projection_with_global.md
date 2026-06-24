---
title: >-
  [Paper Note] Optimization over Sparse Support-Preserving Sets: Two-Step Projection with Global Optimality Guarantees
description: >-
  [ICML 2025][Optimization][sparse optimization] An IHT algorithm with two-step projection (first hard thresholding, then convex projection) is proposed for sparse optimization problems with extra support-preserving constraints. Global objective value guarantees (without systematic error) are provided under RSC/RSS conditions, revealing a new trade-off between sparsity relaxation and suboptimality.
tags:
  - "ICML 2025"
  - "Optimization"
  - "sparse optimization"
  - "IHT"
  - "l0 constraint"
  - "support-preserving"
  - "two-step projection"
date: 2026-05-08
content_hash: 1bd7597afecf0de2
---

# Optimization over Sparse Support-Preserving Sets: Two-Step Projection with Global Optimality Guarantees

**Conference**: ICML 2025  
**arXiv**: [2506.08558](https://arxiv.org/abs/2506.08558)  
**Code**: None  
**Area**: Optimization  
**Keywords**: sparse optimization, IHT, l0 constraint, support-preserving, two-step projection

## TL;DR
An IHT algorithm with two-step projection (first hard thresholding, then convex projection) is proposed for sparse optimization problems with extra support-preserving constraints. Global objective value guarantees (without systematic error) are provided under RSC/RSS conditions, revealing a new trade-off between sparsity relaxation and suboptimality.

## Background & Motivation

### Background

**Background**: In sparse optimization, using $l_0$ constraints allows for precise sparsity control compared to convex relaxation. The IHT algorithm is a standard method and already enjoys global convergence guarantees.

**Limitations of Prior Work**: Extra constraints (budget constraints, norm constraints, etc.) are often required in practice. Existing algorithms either require closed-form projection over mixed constraints (which often does not exist) or only provide local guarantees.

**Key Challenge**: IHT with global guarantees is not applicable to mixed constraints, while methods applicable to mixed constraints only offer local guarantees. The gap between them urgently needs to be bridged.

**Goal**: To obtain IHT variants with global objective value guarantees under support-preserving constraints (such as $l_p$ balls).

**Key Insight**: Define "support-preserving sets" where the support of a $k$-sparse vector remains unchanged after projection (such as $l_p$ balls with $p \geq 1$).

**Core Idea**: Replace Euclidean projection on mixed constraints with a two-step projection: (1) Select the top-$k$ coordinates via hard thresholding; (2) Project onto the convex set. It is simple and widely applicable.

## Method

### Overall Architecture
Three algorithms: IHT-2SP (deterministic), HSG-HT-2SP (stochastic), and HZO-HT-2SP (zeroth-order). Core steps: Gradient Descent $\rightarrow$ Two-Step Projection (hard thresholding + convex projection).

### Key Designs
1. **Two-Step Projection Operator**: The first step retains the top-$k$ coordinates and sets the rest to zero, and the second step projects onto the convex set $\Gamma$. Since $\Gamma$ is support-preserving, the second step does not change the support. This avoids the requirement for closed-form projection under mixed constraints.
2. **Extension of the Three-Point Lemma**: The core technical contribution. It generalizes the classical three-point lemma to the non-convex two-step projection setting, enabling global objective value analysis. This tool also simplifies existing IHT analyses and possesses independent technical value.
3. **Sparsity-Suboptimality Trade-off**: The parameter $\rho$ controls the relaxation level. $R(w_T) \leq (1+2\rho)R(\bar{w}) + \varepsilon$, with the sparsity requirement $k = \Omega(\kappa_s^2\bar{k}/\rho^2)$.
4. **Zeroth-Order Improvement**: Eliminates the non-vanishing systematic error in de Vazelhes et al. (2022). This is the first convergence result for zeroth-order hard thresholding without systematic error.

### Loss & Training
Analyzed under RSC/RSS conditions. The deterministic version converges linearly to an approximate solution, while the stochastic and zeroth-order versions converge at similar rates.

## Key Experimental Results

### Comparison of Theoretical Results

| Algorithm | Extra Constraint | Guarantee | Systematic Error |
|------|---------|------|---------|
| Jain et al. 2014 (IHT) | None | Global | None |
| Lu 2015 / Beck & Hallak 2016 | Support-Preserving | Local Only | - |
| **IHT-2SP (Ours)** | **Support-Preserving** | **Global** | **None** |
| **HZO-HT-2SP (Ours)** | **Support-Preserving** | **Global** | $O(\mu)$ smoothing error only |

### Ablation Analysis

| Value of $\rho$ | Sparsity $k$ Requirement | Suboptimality Multiplier | Description |
|-------|-----------|-----------|------|
| Small $\rho$ | Large | $(1+2\rho)$ close to 1 | Close to exact but requires higher sparsity |
| Large $\rho$ | Small | $(1+2\rho)$ large | Sparser but suboptimality increases |

### Key Findings
- The two-step projection is equivalent to the exact projection on support-preserving sets but is computationally simpler.
- The extension of the three-point lemma simplifies the classical IHT analysis as a byproduct.
- The zeroth-order optimization without systematic error is a novel result even in the unconstrained setting.

## Highlights & Insights
- **The extension of the three-point lemma** is the core technical innovation, enabling the global analysis of non-convex two-step projections. This tool can be generalized to other non-convex projection problems.
- **High practicality**: Two-step projection avoids the need for closed-form projections and is applicable to various common constraints such as $l_p$ balls and simplexes.
- **The sparsity-suboptimality trade-off** is entirely new in the mixed-constraint setting, allowing flexible choices in practice according to requirements.

## Limitations & Future Work
- Limited to support-preserving sets; non-support-preserving constraints (general polyhedra) are not applicable.
- The RSC/RSS assumptions may not hold in deep learning scenarios.
- The sparsity relaxation $k/\rho^2$ can be excessively large in practice.
- Lack of large-scale experimental validation.

## Related Work & Insights
- **vs Jain et al. (2014)**: Classical IHT is unconstrained; ours extends to mixed constraints while preserving global optimality.
- **vs Lu (2015)**: Support-preserving constraints but only local convergence; ours is global.
- **vs de Vazelhes et al. (2022)**: Zeroth-order optimization has systematic error; ours eliminates it.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension of the three-point lemma and the trade-off are novel contributions.
- Experimental Thoroughness: ⭐⭐ Purely theoretical.
- Writing Quality: ⭐⭐⭐⭐ Table 1 is comprehensive.
- Value: ⭐⭐⭐⭐ Bridges the gap of global guarantees under mixed sparse constraints.
- Overall: ⭐⭐⭐⭐ Solid contribution to optimization theory.


## Supplementary Analysis

### Applicable Constraint Types
Support-preserving sets (Definition 2.3) include, but are not limited to:
- **$\ell_p$ balls** ($p \geq 1$): Including $\ell_1$ balls (Lasso constraints), $\ell_2$ balls (Ridge constraints), and $\ell_\infty$ balls.
- **Non-negative constraints + $\ell_p$ balls**: Commonly found in non-negative matrix factorization.
- **Simplex**: Probability distribution constraints.
- **Sign-free convex sets**: A general category defined by Lu (2015).
- **Intersections of the above**: As long as the intersection is also support-preserving.

### Computational Complexity
- Per-step cost of the two-step projection: hard thresholding $O(d \log k)$, convex projection $O(d)$ ($\ell_2$ ball) or $O(d \log d)$ ($\ell_1$ ball).
- The total complexity is comparable to standard IHT—the overhead of the extra projection step is negligible.
- The zeroth-order version requires $2d$ function evaluations per step (coordinate perturbation), making it suitable for scenarios where gradients are unavailable.

### Comparison with Regularization Methods
Our hard-constraint method has a fundamental difference from $\ell_1$-regularized sparse methods:
- Hard constraints directly control the sparsity $k$ without the need for parameter tuning.
- The sparsity of $\ell_1$ regularization is indirectly controlled by the regularization coefficient, making it impossible to specify precisely.
- In scenarios requiring exact sparsity (e.g., combinatorial optimization, sensor selection), hard constraints are more appropriate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[NeurIPS 2025\] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](../../NeurIPS2025/optimization/optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)
- [\[ICLR 2026\] Improving LLM-based Global Optimization with Search Space Partitioning](../../ICLR2026/optimization/improving_llm-based_global_optimization_with_search_space_partitioning.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](../../ICLR2026/optimization/strongly_convex_sets_in_riemannian_manifolds.md)

</div>

<!-- RELATED:END -->
