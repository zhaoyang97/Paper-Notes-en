---
title: >-
  [Paper Note] Learning Theory for Kernel Bilevel Optimization
description: >-
  [NeurIPS 2025][Optimization][Bilevel Optimization] This work establishes the first finite-sample generalization bounds for kernel bilevel optimization (KBO), proving that plug-in estimation errors for both the objective value and its gradient converge uniformly at the parametric rate $\mathcal{O}(1/\sqrt{m}+1/\sqrt{n})$, and applies this theory to analyze the statistical accuracy of bilevel gradient descent.
tags:
  - NeurIPS 2025
  - Optimization
  - Bilevel Optimization
  - Kernel Methods
  - Generalization Theory
  - RKHS
  - Empirical Processes
  - U-processes
  - Implicit Differentiation
date: 2026-05-08
content_hash: ab1a9e37f586520e
---

# Learning Theory for Kernel Bilevel Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2502.08457](https://arxiv.org/abs/2502.08457)
**Authors**: Fares El Khoury (INRIA/Université Grenoble Alpes), Edouard Pauwels (Toulouse School of Economics), Samuel Vaiter (CNRS/Université Côte d'Azur), Michael Arbel (INRIA/Université Grenoble Alpes)
**Code**: [fareselkhoury/KBO](https://github.com/fareselkhoury/KBO)
**Area**: Optimization
**Keywords**: Bilevel Optimization, Kernel Methods, Generalization Theory, RKHS, Empirical Processes, U-processes, Implicit Differentiation

## TL;DR

This work establishes the first finite-sample generalization bounds for kernel bilevel optimization (KBO), proving that plug-in estimation errors for both the objective value and its gradient converge uniformly at the parametric rate $\mathcal{O}(1/\sqrt{m}+1/\sqrt{n})$, and applies this theory to analyze the statistical accuracy of bilevel gradient descent.

## Background & Motivation

### State of the Field
Bilevel optimization is widely used in machine learning, encompassing hyperparameter tuning, meta-learning, inverse problems, and reinforcement learning. Its defining structure is that the outer objective implicitly depends on the solution of an inner optimization problem. Existing theoretical work has focused primarily on parametric settings (finite-dimensional spaces with strongly convex inner problems), analyzing generalization through algorithmic stability or stochastic bilevel optimization convergence.

### Limitations of Prior Work
- **Finite-dimensional assumptions limit expressivity**: Existing generalization analyses (Bao et al. 2021, Zhang et al. 2024) require both inner and outer parameters to lie in fixed-dimensional Euclidean spaces, restricting models to linear functions.
- **Gap in nonparametric settings**: When the inner variable lives in an infinite-dimensional function space such as an RKHS, the effective parameter dimension grows with the sample size (via the representer theorem), and existing frameworks cannot handle statistical analyses with varying parameter-space dimensionality.
- **Tension between strong convexity and expressivity**: Relaxing strong convexity (e.g., deep networks) destabilizes the inner solution, making generalization guarantees difficult; maintaining strong convexity requires nonparametric approaches such as kernel methods.
- **Lack of statistical analysis in existing kernel bilevel work**: Keerthi et al. (2006) and Kunapuli et al. (2008) use the representer theorem to reduce kernel bilevel problems to finite-dimensional ones, but do not study how sample size affects generalization.

### Root Cause
The goal is to overcome finite-dimensional limitations while preserving strong convexity—leveraging the rich expressivity and favorable mathematical structure of RKHSs to establish, for the first time, a learning theory for nonparametric bilevel optimization, including finite-sample generalization bounds.

## Method

### Problem Formulation (KBO)
Given a reproducing kernel Hilbert space $\mathcal{H}$ with kernel $K$, the outer and inner objectives are defined in expectation form:

$$L_{out}(\omega,h)=\mathbb{E}_{\mathbb{Q}}[\ell_{out}(\omega,h(x),y)], \quad L_{in}(\omega,h)=\mathbb{E}_{\mathbb{P}}[\ell_{in}(\omega,h(x),y)]+\frac{\lambda}{2}\|h\|_{\mathcal{H}}^2$$

The goal is $\min_{\omega\in\mathcal{C}} \mathcal{F}(\omega) := L_{out}(\omega, h_\omega^\star)$, where $h_\omega^\star = \arg\min_{h\in\mathcal{H}} L_{in}(\omega,h)$.

Five basic assumptions are imposed: (A) kernel measurability; (B) bounded kernel $K(x,x)\leq\kappa$; (C) compact output space $\mathcal{Y}$; (D) $C^3$-smooth loss functions; (E) convexity of the inner loss with respect to $v$.

### Key Theoretical Contribution 1: Implicit Differentiation in RKHS
The implicit function theorem is applied within the RKHS to derive an explicit expression for $\nabla\mathcal{F}(\omega)$:

$$\nabla\mathcal{F}(\omega) = \mathbb{E}_{\mathbb{Q}}[\partial_\omega\ell_{out}(\omega,h_\omega^\star(x),y)] + \mathbb{E}_{\mathbb{P}}[\partial_{\omega,v}^2\ell_{in}(\omega,h_\omega^\star(x),y)a_\omega^\star(x)]$$

where $a_\omega^\star$ is an adjoint function defined as the minimizer of a strongly convex quadratic objective $L_{adj}$ in $\mathcal{H}$. The central challenge lies in handling operators in infinite-dimensional space—the Hessian $\partial_h^2 L_{in}$ is an operator $\mathcal{H}\to\mathcal{H}$, and the cross-derivative $\partial_{\omega,h}^2 L_{in}$ is an operator $\mathcal{H}\to\mathbb{R}^d$—requiring Bochner integration theory and the Lebesgue dominated convergence theorem to interchange differentiation and expectation.

### Key Theoretical Contribution 2: Commutativity of Differentiation and Statistical Estimation
**Proposition 3.1**: The gradient $\nabla\hat{\mathcal{F}}(\omega)$ of the empirical objective $\hat{\mathcal{F}}$ equals exactly the plug-in estimator $\widehat{\nabla\mathcal{F}}(\omega)$ of the population gradient $\nabla\mathcal{F}(\omega)$. That is, "differentiate then estimate" and "estimate then differentiate" yield identical results. This commutativity is a structural property unique to RKHSs and does not hold in spaces such as $L^2$.

### Main Theorem: Uniform Generalization Bounds (Theorem 4.1)
For any compact subset $\Omega\subset\mathbb{R}^d$, under the five assumptions:

$$\mathbb{E}\left[\sup_{\omega\in\Omega}|\mathcal{F}(\omega)-\hat{\mathcal{F}}(\omega)|\right] \leq C\left(\frac{1}{\sqrt{m}}+\frac{1}{\sqrt{n}}\right)$$

$$\mathbb{E}\left[\sup_{\omega\in\Omega}\|\nabla\mathcal{F}(\omega)-\widehat{\nabla\mathcal{F}}(\omega)\|\right] \leq C\left(\frac{1}{\sqrt{m}}+\frac{1}{\sqrt{n}}\right)$$

where $C$ depends only on $\Omega$, dimension $d$, regularization parameter $\lambda$, kernel bound $\kappa$, and local upper bounds on the derivatives of the loss functions.

### Three-Step Proof Strategy
1. **Pointwise error decomposition**: $|\mathcal{F}-\hat{\mathcal{F}}|$ is decomposed into a combination of $\delta_\omega^{out}$ (outer-level sampling error) and $\partial_h\delta_\omega^{in}$ (inner-level Fréchet derivative error); the gradient error is further expanded into five terms.
2. **Empirical process control**: For real-valued terms (e.g., $\delta_\omega^{out}$, $\partial_\omega\delta_\omega^{out}$), classical maximal inequalities from empirical process theory are applied directly, with complexity controlled via the packing numbers of the function class.
3. **Degenerate U-process control**: For RKHS-valued terms involving $\partial_h$, the reproducing property $\|f\|_\mathcal{H}^2 = \langle f, f\rangle_\mathcal{H}$ is used to convert their squares into real-valued functions depending on pairs of samples, yielding a second-order degenerate U-process, to which the maximal inequality of Sherman (1994) is applied.

### Algorithmic Application: Generalization Guarantee for Bilevel Gradient Descent (Corollary 4.2)
For gradient descent on unconstrained KBO, $\omega_{t+1}=\omega_t-\eta\nabla\hat{\mathcal{F}}(\omega_t)$:

$$\mathbb{E}\left[\min_{i=0,\ldots,t}\|\nabla\mathcal{F}(\omega_i)\|\right] \leq \bar{c}\left(\frac{1}{\sqrt{m}}+\frac{1}{\sqrt{n}}+\frac{1}{\sqrt{t+1}}\right)$$

After sufficiently many iterations, the statistical error dominates: $\mathbb{E}[\limsup_{i\to\infty}\|\nabla\mathcal{F}(\omega_i)\|] \leq \bar{c}(1/\sqrt{m}+1/\sqrt{n})$.

### Multiple Roles of the Regularization Parameter $\lambda$
$\lambda$ simultaneously controls: (1) the strong convexity constant of the inner problem; (2) the boundedness and Lipschitz continuity of the inner solution; (3) the smoothness of the outer objective; (4) the modulus of continuity of the loss functions; and (5) the constants in the maximal inequalities for empirical processes. A large $\lambda$ facilitates optimization but introduces statistical bias; a small $\lambda$ reduces bias but makes generalization harder.

## Key Experimental Results

### Experimental Setup
Synthetic data from instrumental variable regression are used to validate the theory. The model is linearly parameterized as $f_\omega(t)=\omega^\top\phi(t)$, with a Gaussian kernel. The inner sample size $n$ varies from 100 to 5000 with $m=n$. The outer optimization uses gradient descent with backtracking line search, stopping when $\|\nabla\hat{\mathcal{F}}(\omega_i)\|\leq 10^{-5}$. The population objective is approximated using $10^6$ samples and 26,000 random Fourier features.

### Experiment 1: Convergence Rate of Generalization Error (Gaussian vs. Student-t Distributions)

| Metric | Distribution | Empirical Rate (log-log slope) | Theoretical Prediction |
|--------|--------------|-------------------------------|------------------------|
| $\|\mathcal{F}(\omega_0)-\hat{\mathcal{F}}(\omega_0)\|$ | Gaussian | $\approx -1/2$ | $O(n^{-1/2})$ |
| $\|\nabla\mathcal{F}(\omega_0)-\widehat{\nabla\mathcal{F}}(\omega_0)\|$ | Gaussian | $\approx -1/2$ | $O(n^{-1/2})$ |
| $\|\mathcal{F}(\omega_0)-\hat{\mathcal{F}}(\omega_0)\|$ | Student-t ($\nu=2.1$) | $\approx -1/2$ | $O(n^{-1/2})$ |
| $\|\nabla\mathcal{F}(\omega_0)-\widehat{\nabla\mathcal{F}}(\omega_0)\|$ | Student-t ($\nu=2.1$) | $\approx -1/2$ | $O(n^{-1/2})$ |

All curves exhibit slopes close to $-1/2$ on a log-log scale, confirming the theoretical $O(1/\sqrt{n})$ convergence rate. The 95% confidence intervals are very narrow.

### Experiment 2: Statistical Accuracy of Gradient Descent

| Metric | Description | Empirical Behavior | Theoretical Prediction |
|--------|-------------|-------------------|------------------------|
| $\|\nabla\mathcal{F}(\omega_T)\|$ | Population gradient norm at termination | $O(n^{-1/2})$ decay | Corollary 4.2 |
| $\min_{i\leq T}\|\nabla\mathcal{F}(\omega_i)\|$ | Minimum gradient norm along trajectory | $O(n^{-1/2})$ decay | Corollary 4.2 |

Results are averaged over 50 independent runs. Both Student-t distributions ($\nu\in\{2.1,2.5,2.9\}$) and the Gaussian distribution consistently exhibit the $n^{-1/2}$ rate. The balanced setting $m=n$ yields optimal optimization behavior.

## Highlights & Insights

- **Pioneer theoretical contribution**: This is the first work to establish finite-sample generalization bounds for nonparametric (kernel-based) bilevel optimization, bridging the theoretical gap between finite-dimensional analyses and infinite-dimensional settings.
- **Commutativity of differentiation and estimation**: The paper proves that in an RKHS, "differentiate first, then estimate statistically" is equivalent to "estimate first, then differentiate." This structural result does not hold in spaces such as $L^2$ and is a property unique to RKHSs.
- **Novel application of U-process techniques**: The reproducing property of RKHSs is cleverly exploited to convert infinite-dimensional error terms into second-order degenerate U-processes, avoiding the curse of dimensionality in covering number arguments.
- **Characterization of the statistics-optimization trade-off**: Corollary 4.2 clearly delineates the balance among sample sizes $(m,n)$ and iteration count $t$.
- **Strong agreement between theory and experiment**: Synthetic experiments precisely reproduce the theoretical $n^{-1/2}$ rate.

## Limitations & Future Work

- **Restricted to full-batch exact gradients**: Stochastic bilevel optimization (SGD, SABA, etc.) is not covered; full-batch computation is infeasible in large-scale practical applications.
- **Fixed regularization parameter $\lambda$**: The constant $C$ may diverge as $\lambda\to 0$, precluding generalization guarantees in the unregularized setting.
- **Requires smooth losses ($C^3$)**: The framework does not apply to non-smooth losses such as the hinge loss used in SVMs.
- **Non-tight constants**: The constant $C$ in the generalization bounds may be highly conservative; its optimality is not analyzed.
- **Only synthetic data experiments**: Validation on real-world datasets is absent.
- **Compact set $\Omega$ constraint**: $C$ may grow with the diameter of $\Omega$; global guarantees require additional assumptions.

## Related Work & Insights

- **Bao et al. (2021)**: Analyze finite-dimensional bilevel optimization via uniform stability, obtaining generalization rates of $O(T^\kappa/m)$, but are limited to fixed parameter dimensions and the error grows with iterations.
- **Zhang et al. (2024)**: Extend the analysis to stochastic bilevel optimization, achieving $O(1/\sqrt{m})$ generalization rates, but remain in the finite-dimensional setting.
- **Oymak et al. (2021)**: Obtain $O(1/\sqrt{m}+1/\sqrt{n})$ rates via Lipschitz outer losses and approximate inner solutions, but still operate in finite dimensions.
- **Meunier et al. (2024)**: Achieve minimax-optimal rates in instrumental variable regression via spectral filtering, but require quadratic losses and $\lambda\to 0$, making the approach inapplicable to general inner objectives.
- **Petrulionyte et al. (2024)**: Propose a functional bilevel optimization framework, but differentiation and estimation do not commute in $L^2$ spaces, and no finite-sample guarantees are provided.
- **Advantages of this work**: Covers general convex inner losses in the fixed-$\lambda$ regime; uses the reproducing property of RKHSs to circumvent infinite-dimensional difficulties; provides uniform bounds controlling both the objective value and the gradient simultaneously.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces empirical process and U-process theory into kernel bilevel optimization, establishing the first nonparametric generalization bounds.
- Experimental Thoroughness: ⭐⭐⭐ — Limited to synthetic instrumental variable regression experiments; confirms the theoretical rate but lacks real-world applications.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous and self-contained theoretical framework; proof strategy is clearly articulated; commutative diagrams and auxiliary illustrations are well-placed.
- Value: ⭐⭐⭐⭐ — Fills an important theoretical gap with solid technical contributions; practical impact awaits validation through future extensions to stochastic and large-scale settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)

</div>

<!-- RELATED:END -->
