---
title: >-
  [Paper Note] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization
description: >-
  [ICLR 2026][Optimization & Theory][bilevel optimization] By reinterpreting the F2SA method as a forward difference approximation of the hyper-gradient, this work proposes the F2SA-p algorithmic family utilizing higher-order finite differences. Under high-order smoothness conditions, it improves the SFO complexity of stochastic bilevel optimization from $\tilde{\mathcal{O}}(\
tags:
  - ICLR 2026
  - Optimization & Theory
  - bilevel optimization
  - stochastic optimization
  - finite difference
  - hyper-gradient estimation
  - complexity lower bound
date: 2026-05-08
content_hash: 277d0cbef4fecdd0
---
# Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization

**Conference**: ICLR2026  
**arXiv**: [2509.02937](https://arxiv.org/abs/2509.02937)  
**Code**: [GitHub](https://github.com/TrueNobility303/F2BA)  
**Area**: Optimization  
**Keywords**: bilevel optimization, stochastic optimization, finite difference, hyper-gradient estimation, complexity lower bound

## TL;DR
By reinterpreting the F2SA method as a forward difference approximation of the hyper-gradient, this work proposes the F2SA-p algorithmic family utilizing higher-order finite differences. Under high-order smoothness conditions, it improves the SFO complexity of stochastic bilevel optimization from $\tilde{\mathcal{O}}(\epsilon^{-6})$ to $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$, and proves an $\Omega(\epsilon^{-4})$ lower bound indicating the method is nearly optimal for sufficiently large $p$.

## Background & Motivation
Bilevel optimization frequently arises in tasks such as meta-learning, hyperparameter tuning, adversarial training, and reinforcement learning. Its objective is $\min_{\bm{x}} f(\bm{x}, \bm{y}^*(\bm{x}))$, where $\bm{y}^*(\bm{x}) = \arg\min_{\bm{y}} g(\bm{x}, \bm{y})$. Existing methods are primarily categorized into two types:

1. **Hessian-vector product (HVP) based methods**: Such as BSA and stocBiO, which require a stochastic Hessian oracle and rely on stronger assumptions.
2. **Pure first-order methods**: Such as F2SA, which only require a stochastic gradient oracle, make weaker assumptions, and have been scaled to training 32B LLMs in practice.

While F2SA achieves a complexity upper bound of $\tilde{\mathcal{O}}(\epsilon^{-6})$ under standard SGD assumptions, a significant gap remains compared to the $\Omega(\epsilon^{-4})$ lower bound of single-level optimization. The core question is: **Can pure first-order methods achieve optimal complexity in bilevel optimization?**

## Core Problem
Under the stochastic bilevel optimization setting with a non-convex upper-level function and a strongly convex lower-level function, can high-order smoothness be leveraged to break the $\tilde{\mathcal{O}}(\epsilon^{-6})$ complexity bottleneck of existing pure first-order methods and approach the $\Omega(\epsilon^{-4})$ lower bound?

## Method

### Overall Architecture
This paper reinterprets the pure first-order method F2SA as a **forward difference approximation** of the hyper-gradient. From this perspective, higher-order finite difference schemes are substituted to reduce approximation error from $\mathcal{O}(\nu)$ to $\mathcal{O}(\nu^p)$, resulting in the F2SA-$p$ algorithmic family. This is accompanied by a matching $\Omega(\epsilon^{-4})$ lower bound, demonstrating near-optimality in high-order smooth regimes. The derivation revolves around a core object: the derivative of the optimal value function $\ell_\nu(\bm{x})$ of the perturbed lower-level problem with respect to the perturbation intensity $\nu$, which equals the hyper-gradient. As a theoretical work focusing on reinterpretation, algorithmic reduction, and complexity analysis, no architectural diagrams are provided.

### Key Designs

**1. Finite difference perspective: Translating hyper-gradient estimation into a numerical differentiation problem**

This serves as the starting point. The authors define the perturbed lower-level problem $g_\nu(\bm{x}, \bm{y}) = \nu f(\bm{x}, \bm{y}) + g(\bm{x}, \bm{y})$ and its optimal value function $\ell_\nu(\bm{x})$, proving that the hyper-gradient can be expressed as a mixed partial derivative $\frac{\partial^2}{\partial \nu \partial \bm{x}} \ell_\nu(\bm{x})\big|_{\nu=0} = \nabla \varphi(\bm{x})$. In this view, the gradient of the F2SA penalty function is essentially a forward difference $(\ell_\nu - \ell_0)/\nu$ approximating $\partial \ell_\nu / \partial \nu|_{\nu=0}$, naturally incurring a first-order error $\mathcal{O}(\nu)$. By identifying this relationship, hyper-gradient estimation is reframed as a numerical differentiation problem, revealing the root cause of the gap between F2SA and optimal complexity.

**2. F2SA-$p$: Reducing approximation error from $\mathcal{O}(\nu)$ to $\mathcal{O}(\nu^p)$ using high-order finite differences**

By switching to higher-order difference schemes, the error is reduced to $\mathcal{O}(\nu^p)$ (Lemma 3.1): $p=1$ corresponds to coefficients $\alpha_0=-1, \alpha_1=1$ (the original F2SA); $p=2$ corresponds to central difference $\alpha_{-1}=-1/2, \alpha_1=1/2$ (F2SA-2 with symmetric penalty). The algorithm utilizes a double-loop: the inner loop runs $K$ steps of SGD for each offset $j=-p/2, \ldots, p/2$ to solve the perturbed problem $g_{j\nu}(\bm{x}, \cdot)$ for approximate optimal solutions $\bm{y}_{j\nu}^*(\bm{x})$; the outer loop linearly combines gradients of these perturbation points using coefficients $\{\alpha_j\}$ into a hyper-gradient estimate $\Phi_t$, followed by a normalized gradient step $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$. The complexity upper bound is established via high-dimensional Faà di Bruno formula for the Lipschitzness of higher-order derivatives of $\ell_\nu$ w.r.t. $\nu$ (Lemma 3.2), yielding an SFO complexity of $\tilde{\mathcal{O}}\left(p \kappa^{9+2/p} \epsilon^{-4-2/p}\right)$. For $p = \Omega(\log\epsilon^{-1}/\log\log\epsilon^{-1})$, it simplifies to $\tilde{\mathcal{O}}(\kappa^9 \epsilon^{-4})$, matching HVP-based methods without requiring a Hessian oracle. F2SA-2, in particular, requires only two lower-level solves (same as F2SA) but reduces error to second-order, representing a nearly "free" improvement.

**3. Required high-order smoothness only in the lower-level direction: Weakening assumptions**

To achieve an error order of $\mathcal{O}(\nu^p)$, the authors only require high-order smoothness in the direction of the lower-level variable $\bm{y}$ (Assumption 2.5), meaning high-order derivatives of $f, g$ w.r.t. $\bm{y}$ are Lipschitz. This is significantly weaker than requiring joint $(\bm{x}, \bm{y})$ high-order smoothness. This assumption is practical; for instance, logistic regression tasks such as data hyper-cleaning and learn-to-regularize naturally satisfy this condition due to the infinite smoothness of softmax w.r.t. its input.

**4. Matching lower bound: Constructing separable instances reduced to single-level**

To demonstrate that $\epsilon^{-4}$ is the limit, the authors construct a fully separable bilevel instance ($f(\bm{x}, \bm{y}) \equiv f_{\bm{U}}(\bm{x})$, $g(\bm{x}, y)=\mu y^2/2$), reducing the bilevel problem to a single-level one and inheriting the $\Omega(\epsilon^{-4})$ lower bound from single-level optimization. This construction satisfies all high-order smoothness conditions, ensuring the lower bound and upper bound reside within the same assumption framework.

## Key Experimental Results
Logistic regression experiments for learn-to-regularize were performed on the 20 Newsgroups dataset (18,000 samples, 130,107 features), which naturally satisfies high-order smoothness.

### Main Results
- Methods compared: F2SA-p ($p \in \{2,3,5,8,10\}$) vs F2SA vs stocBiO vs MRBO vs VRBO.
- Parameters: Inner loop $K=10$, outer loop $T=1000$.
- The convergence speed of the F2SA-p series improved as $p$ increased, validating the theory.
- Appendix experiments on a 5-layer ReLU MLP demonstrate potential in non-smooth non-convex problems.

## Highlights & Insights
1. **Elegant theoretical insight**: Reinterpreting F2SA as finite difference provides a natural path for algorithmic advancement.
2. **Nearly free improvement**: F2SA-2 requires only 2 lower-level solves—identical to F2SA—while providing second-order error guarantees.
3. **Complete theoretical framework**: The upper bound $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$ and lower bound $\Omega(\epsilon^{-4})$ establish near-optimality in high-order smooth regimes.
4. **Tighter known bounds**: Improved $\kappa$ dependence for $p=1$ and correction of Hessian convergence bounds for $p=2$.
5. **Weaker assumptions**: Only requires high-order smoothness in the $\bm{y}$ direction, eliminating the need for joint smoothness or stochastic Hessian assumptions.

## Limitations & Future Work
1. A gap still exists between bounds for small $p$ (e.g., $p=1$); the optimal complexity for $p=1$ remains open.
2. The condition number $\kappa$ dependence has an $\Omega(\kappa^9)$ gap.
3. Experiments focus on convex lower-level problems; non-convex-non-convex structured bilevel optimization remains for future study.
4. No integration with variance reduction or momentum techniques yet.
5. High-order smoothness requirements limit application in certain deep learning contexts.

## Related Work & Insights

| Method | Smoothness Order | Complexity | Requires HVP |
|------|--------|--------|----------|
| BSA | 1st-order | $\tilde{\mathcal{O}}(\epsilon^{-6})$ SFO + $\tilde{\mathcal{O}}(\epsilon^{-4})$ HVP | Yes |
| stocBiO | 1st-order | $\tilde{\mathcal{O}}(\epsilon^{-4})$ | Yes |
| F2SA | 1st-order | $\tilde{\mathcal{O}}(\kappa^{12}\epsilon^{-6})$ | No |
| **Ours (F2SA-p)** | **$p$-th order** | **$\tilde{\mathcal{O}}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$** | **No** |
| Lower Bound | $p$-th order | $\Omega(\epsilon^{-4})$ | — |

Compared to Chayti & Jaggi (2024), which focused on finite differences in meta-learning, this work generalizes to general bilevel optimization. Unlike Huang et al. (2025), it requires only $\bm{y}$-direction high-order smoothness.

## Rating
- Novelty: ⭐⭐⭐⭐ — Finite difference perspective is elegant and natural.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are limited to convex-lower-level logistic regression.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure with a complete logical chain from insight to theory.
- Value: ⭐⭐⭐⭐ — Establishes significant results in bilevel theory and narrows the complexity gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](../../ICML2026/optimization/safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)

</div>

<!-- RELATED:END -->
