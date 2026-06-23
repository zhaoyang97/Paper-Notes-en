---
title: >-
  [Paper Note] Distributionally Robust Linear Regression with Block Lewis Weights
description: >-
  [ICLR 2026][Optimization & Theory][Block Lewis Weights] This paper designs a second-order algorithm for "Empirical Group Distributionally Robust Least Squares" (i.e., min-max group regression) with an iterative complexity of only $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$. The key is using **block Lewis weights** to construct an optimal approximating el
tags:
  - ICLR 2026
  - Optimization & Theory
  - Block Lewis Weights
date: 2026-05-08
content_hash: 03e450dfbecf1067
---
# Distributionally Robust Linear Regression with Block Lewis Weights

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0r1KU3dlps](https://openreview.net/forum?id=0r1KU3dlps)  
**Code**: Jupyter notebook included with the paper (synthetic experiments)  
**Area**: optimization  
**Keywords**: Group Distributionally Robust Optimization, Min-max Regression, Block Lewis Weights, Accelerated Proximal Method, Quasi-self-concordance, $\ell_\infty/\ell_p$ Regression  

## TL;DR
This paper designs a second-order algorithm for "Empirical Group Distributionally Robust Least Squares" (i.e., min-max group regression) with an iterative complexity of only $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$. The key is using **block Lewis weights** to construct an optimal approximating ellipsoidal geometry, coupled with an accelerated proximal method, making the number of iterations nearly independent of the group count $m$.

## Background & Motivation

**Background**: In scenarios requiring interpretable models such as house price prediction, cross-industry wage estimation, or loan limit prediction, it is common to fit a single linear model $x\in\mathbb{R}^d$ to $m$ groups (data sources). The most direct "utilitarian" goal is to minimize the average squared error across groups (objective 1), but due to group heterogeneity, this averaging favors mainstream data sources and provides significantly worse predictions for minority groups. Consequently, recent work has shifted toward "egalitarian" group DRO objectives:

$$\min_{x\in\mathbb{R}^d}\ \max_{i\in[m]}\ \tfrac{1}{n_i}\|A_{S_i}x-b_{S_i}\|_2^2 .$$

**Limitations of Prior Work**: Although this objective is convex, standard methods are suboptimal. ① First-order methods (subgradient, Abernethy et al.) have convergence rates that **depend on the geometric condition number**—converging extremely slowly when the stacked matrix $A$ is ill-conditioned. ② The objective is the pointwise maximum of several norms, making it **non-smooth**; naive subgradients only achieve $1/\varepsilon^2$ iterative complexity. ③ Treating it as a min-max game using no-regret methods (smooth online convex optimization) still results in $1/\varepsilon^2$, which is no better. ④ Reformulating as an epigraph for Interior Point Methods (IPM) yields an iterative complexity of $O(\sqrt{m})$, which is super-linear when $m$ is large; furthermore, duplicating each group $k$ times keeps the value unchanged but causes complexity to explode to $\sqrt{mk}$.

**Key Challenge**: An ideal algorithm should be **geometry-independent** (unaffected by condition numbers) and have an iteration count **nearly independent of $m$** (preferably a low power of $\min\{\mathrm{rank}(A),m\}$), all while solving only one structured linear system per step. No existing method satisfies both.

**Goal**: Provide a second-order algorithm that uses $\tilde{O}(\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3})$ solves of "linear systems of the form $A^\top BA$ ($B$ is block-diagonal)" to approximate the $(1+\varepsilon)$-optimal solution, performing no worse than state-of-the-art results for $\ell_\infty$ regression when $n_i=1$.

**Core Idea**: **Geometry is everything**—treat the min-max objective as a norm on $\mathbb{R}^d$, calculate an optimal approximating ellipsoidal norm $\|\cdot\|_M$ using **block Lewis weights**, and then run an accelerated proximal (ball-oracle) method within this "correct geometry." This ensures $\|x_0-x^\star\|_M$ is bounded by $\sqrt{d}$, thereby replacing the $m$-dependence with $\min\{\mathrm{rank}(A),m\}$.

## Method

### Overall Architecture
The algorithm decomposes the difficult min-max problem into "repeatedly solving proximal subproblems under a specific ellipsoidal geometry $\|\cdot\|_M$." Three components are indispensable: (1) **Fast subproblem solver**: Transforming the non-smooth/high-order objective into a form with stable Hessians solvable via damped Newton within the $\|\cdot\|_M$ ball. (2) **Accelerated iterations**: Using a Monteiro–Svaiter type acceleration framework to chain subproblem solutions, yielding an outer convergence rate of $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$. (3) **Selecting the correct geometry $M$**: Using block Lewis weights to find an ellipsoid such that the distance $\|x_0-x^\star\|_M\lesssim\sqrt{d}$, which compresses the $m$-dependence into $\min\{\mathrm{rank}(A),m\}^{1/3}$.

```mermaid
flowchart LR
    A[Min-max Group Regression<br/>Non-smooth/Geometry Dependent] --> B[Select Geometry M<br/>Block Lewis Weights]
    B --> C[Proximal Subproblem O&#40;q&#41;<br/>Within M-ball]
    C --> D{p=∞ or 2≤p<∞}
    D -->|p=∞| E[Log-sum-exp Smoothing<br/>+ Quasi-self-concordance + Damped Newton]
    D -->|Finite p| F[ℓp Proximal Regularization<br/>+ Strong Convexity + Mirror Descent]
    E --> G[Monteiro–Svaiter Acceleration<br/>Chaining Subproblems]
    F --> G
    G --> H[(1+ε) Optimal Solution<br/>Õ&#40;min&#123;rank,m&#125;^1/3 ε^-2/3&#41;]
```

### Key Designs

**1. Using block Lewis weights to select the "correct ellipsoidal geometry," replacing $m$ with $\mathrm{rank}(A)$.** This is the geometric core. The outer acceleration rate is proportional to $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$, so the problem reduces to: choosing an ellipsoidal norm $\|\cdot\|_M$ that sandwiches the true loss norm $\|y\|_{G_p}=(\sum_i\|y_{S_i}\|_2^p)^{1/p}$ with minimal distortion $\triangle\ge 1$, such that $\|x-b\|_M\le \big(\sum_i\|A_{S_i}x-b_{S_i}\|_2^p\big)^{1/p}\le \triangle\,\|x-b\|_M$. Taking the naive $M=A^\top A$ yields a distortion of $m^{1/2-1/p}$ due to $\ell_2$-$\ell_p$ norm equivalence, degrading the rate to $m^{1/3}$. The key improvement: the loss itself is a **norm** on $\mathbb{R}^d$ whose unit ball is a symmetric convex body. By **John’s Theorem**, there exists a maximum volume inscribed ellipsoid such that distortion $\triangle\le\sqrt{d}$, implying $\|x_0-x^\star\|_M\lesssim\sqrt{d}$. Substituting this back yields $\min\{\mathrm{rank}(A),m\}^{1/3}$. The paper utilizes the **block Lewis weights algorithm by Manoj & Ovsiankin (2025)** to actually compute this diagonal weighting $W$, ensuring that $\|W^{1/2-1/p}(Ax-cb)\|_2$ and the $G_p$ loss are equivalent within a factor of $(2(\mathrm{rank}(A)+1))^{1/2-1/p}$.

**2. $p=\infty$ Robust Scenario: Log-sum-exp smoothing + quasi-self-concordance.** Since $f$ is the pointwise maximum of Euclidean norms, it is non-differentiable. The authors introduce a two-parameter smooth proxy:

$$\tilde f_{\beta,\delta}(x)=\beta\log\!\Big(\sum_{i=1}^m\exp\tfrac{\sqrt{\delta^2+\|A_{S_i}x-b_{S_i}\|_2^2}-\delta}{\beta}\Big),$$

which is a "softmax with temperature $\beta$" composed with an "inner function $\sqrt{\delta^2+\|\cdot\|_2^2}-\delta$." It can be proven that $|\tilde f_{\beta,\delta}-f|\le\beta\log m+\delta$. The challenge is ensuring the proxy satisfies **quasi-self-concordance** ($|\frac{d^3}{dt^3}f|\le\nu\|d\|\frac{d^2}{dt^2}f$), a sufficient condition for the Hessian-stable second-order solvers by Carmon et al. The paper proves a general **Composition Theorem** (Lemma C.3): a softmax composed with a family of quasi-self-concordant functions remains quasi-self-concordant. Proving the inner functions are individually $O(1/\delta)$-quasi-self-concordant makes $\tilde f_{\beta,\delta}$ quasi-self-concordant, allowing fixed-radius $r_q=\Theta(1/\varepsilon)$ subproblems to be solved quickly via damped Newton.

**3. $2\le p<\infty$ Interpolation Scenario: ℓp proximal regularization + norm strong convexity.** For the interpolation objective $\min_x\frac{1}{m}\sum_i(\frac{1}{n_i}\|A_{S_i}x-b_{S_i}\|_2^2)^{p/2}$, subproblems use regularization rather than hard radius: $\arg\min_x f(x)+\tilde p^p\|x-q\|_M^p$. This generalizes proximal problems in $\ell_p$ regression. The acceleration framework requires an **approximate stationary point**. The authors prove a **Strong Convexity Lemma** (Lemma D.3) for $\|y\|_2^p$, deriving strong convexity for $\|x-q\|_M^p$. This converts "function value near-optimality" into "parameter space near-optimality." Combined with local gradient Lipschitzness, an approximate stationary point is obtained via a slightly modified mirror descent solver.

**4. Monteiro–Svaiter Acceleration for $\varepsilon^{-2/3}$ and high precision.** Naive proximal iterations would only yield $\|x_0-x^\star\|_M/\varepsilon$. This paper adopts the Monteiro–Svaiter (2013) acceleration scheme refined by Carmon et al., compressing the rate to $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$. For finite $p$, repeating the process to halve the distance $\|x_t-x^\star\|_M$ yields **high-precision** ($\mathrm{polylog}(1/\varepsilon)$) solutions.

## Key Experimental Results

Experiments use a synthetic "heterogeneous group regression" setup ($d=10$, $100$ groups, $5$ adversarial). The Gram matrix of stacked $A$ has a condition number $\approx 10^5$. Most groups share geometry, while a few have high curvature in specific directions.

### Main Results (Iterative Complexity)

| Method | Iterative Behavior on Adversarial Instances |
|--------|-------------------------------------------|
| Subgradient | Drops briefly, then plateaus far from optimum |
| Smooth 1st-order (GD/HB/Nesterov) | Still plateaus above optimum even with tuned parameters |
| Ball-oracle (Euclidean, Naive) | Consistently reduces worst-group loss over outer iterations |
| Ball-oracle (Lewis-weight, Ours) | Consistently reduces loss, outperforming naive ball |
| Interior Point Method (IPM) | Fastest convergence and best final loss (expected, as solver uses IPM) |

### Ablation Study (Geometry vs. Acceleration)

| Variant | Distortion $\triangle$ | Iterative Complexity ($p=\infty$) | Description |
|---------|-------------------------|--------------------------------|-------------|
| Naive Geometry $M=A^\top A$ | $m^{1/2-1/p}$ | $m^{1/3}\varepsilon^{-2/3}$ | Only relies on norm equivalence |
| **Block Lewis Geometry (Ours)** | $\le\sqrt{d}$ | $\min\{\mathrm{rank}(A),m\}^{1/3}\varepsilon^{-2/3}$ | Computes optimal approximating ellipsoid |
| Naive Proximal (No Accel) | — | $\|x_0-x^\star\|_M/\varepsilon$ | Linear $1/\varepsilon$ |
| **+ Monteiro–Svaiter Accel** | — | $(\|x_0-x^\star\|_M/\varepsilon)^{2/3}$ | $2/3$ power dependency |

### Key Findings
- **1st-order methods are crippled by geometry**: Average gradients emphasize the majority groups' curvature, causing stagnant progress in the "sharp directions" of adversarial groups.
- **Lewis Geometry > Naive Ball**: Using Lewis weights for the trust region converges faster than Euclidean balls, validating the value of "choosing the right geometry."
- **IPM is strong for small $m$**: It achieves the best final accuracy, but its advantage is eroded as $m$ increases due to $O(\sqrt{m})$ complexity—this is where the $m^{1/3}$ theory becomes relevant ($\varepsilon \ge m^{-1/4}$).

## Highlights & Insights
- **Robust Regression as Geometric Selection**: The core insight is that min-max loss is a norm; its optimal approximating ellipsoid (John ellipsoid / Lewis weights) dictates the acceleration rate, reducing complexity from $\sqrt{m}$ (IPM) or $1/\varepsilon^2$ (1st-order) to the rank-dependent rate.
- **Copy-Invariance as a Signal**: The author uses the example where duplicating groups $k$ times keeps the objective same but blows up IPM complexity to $\sqrt{mk}$ as a convincing argument that iteration count should be independent of $m$.
- **General Theoretical Tools**: The composition theorem for quasi-self-concordant functions (Lemma C.3) and the strong convexity lemma for $\|y\|_2^p$ (Lemma D.3) are of independent interest for broader second-order optimization.

## Limitations & Future Work
- **Not High-Precision for $p=\infty$**: The $\varepsilon$ dependency for the robust objective is $\varepsilon^{-2/3}$ rather than $\text{polylog}(1/\varepsilon)$. Achieving high precision for $p=\infty$ remains an open problem.
- **Limited Improvement Range**: The advantage over log-barrier IPM only holds in the medium precision range $\varepsilon \ge m^{-1/4}$ and when $m \gg \mathrm{rank}(A)$.
- **Synthetic Evaluation**: Experiments are limited to $d=10$ and $m=100$ synthetic setups, lacking large-scale validation on real-world fairness or multi-source datasets.
- **Generalization**: The work is a first step toward achieving $\min\{\mathrm{rank}, m\}^{1/3}$ complexity for general Convex Quadratic Programming.

## Related Work & Insights
- **Lewis Weights Lineage**: This work builds on the intuition from Lee & Sidford (2019) and Jambulapati et al. (2022) that geometric understanding leads to reduced dimension-dependent complexity, extending the concept to block structures.
- **Group DRO & Fairness**: Aligns with the "egalitarian" fairness goals of Sagawa and Duchi.
- **Insight**: When a robust/min-max objective can be expressed as a norm, "spending a budget to compute the optimal approximating ellipsoid and then running second-order methods in that geometry" is a replicable strategy to replace $m$-dependence with $\mathrm{rank}$-dependence.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First application of block Lewis weights to group DRO regression, achieving $m$-independent complexity with strong theoretical lemmas.
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic experiments are well-explained but lack real-world data and scale.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression (systematically dismissing lower-order methods) and clear technical summaries.
- **Value**: ⭐⭐⭐⭐ Provides a geometry-independent and $m$-agnostic algorithm for fair linear regression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Distributionally Robust Optimization via Generative Ambiguity Modeling](distributionally_robust_optimization_via_generative_ambiguity_modeling.md)
- [\[ICLR 2026\] Byzantine-Robust Federated Learning with Learnable Aggregation Weights](byzantine-robust_federated_learning_with_learnable_aggregation_weights.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](../../NeurIPS2025/optimization/robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[ICLR 2026\] MuonBP: Faster Muon via Block-Periodic Orthogonalization](muonbp_faster_muon_via_block-periodic_orthogonalization.md)

</div>

<!-- RELATED:END -->
