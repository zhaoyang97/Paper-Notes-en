---
title: >-
  [Paper Note] High-Probability Bounds for the Last Iterate of Clipped SGD
description: >-
  [ICLR 2026][Optimization & Theory][Clipped-SGD] This paper provides the first high-probability convergence rate for the **last iterate** of Clipped-SGD under convex and smooth objectives with heavy-tailed noise (only finite $\alpha$-th moments, $\alpha\in(1,2]$). It also introduces a general technique to convert high-probability bounds into expectation bounds.
tags:
  - ICLR 2026
  - Optimization & Theory
  - Clipped-SGD
date: 2026-05-08
content_hash: cd0f5116f70b6d48
---
# High-Probability Bounds for the Last Iterate of Clipped SGD

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4sGEvpwyxN](https://openreview.net/forum?id=4sGEvpwyxN)  
**Code**: Not public (pure theory + numerical simulation)  
**Area**: optimization  
**Keywords**: Clipped-SGD, heavy-tailed noise, last iterate, high-probability convergence bound, stochastic optimization  

## TL;DR
This paper provides the first high-probability convergence rate for the **last iterate** of Clipped-SGD under convex and smooth objectives with heavy-tailed noise (only finite $\alpha$-th moments, $\alpha\in(1,2]$). It also introduces a general technique to convert high-probability bounds into expectation bounds.

## Background & Motivation
- **Background**: Gradient clipping has become a standard component for training LLMs. Theoretically, it counters heavy-tailed noise and provides high-probability convergence guarantees. For convex problems, the convergence theory for the average iterate of Clipped-SGD is well-established, reaching the optimal rate of $\tilde{O}(1/K^{(\alpha-1)/\alpha})$.
- **Limitations of Prior Work**: In practice, the **last iterate** $x_K$ is what is actually deployed (as storing all iterates for averaging is impractical). However, for convex and smooth objectives with heavy-tailed noise, the convergence of the last iterate of Clipped-SGD remains largely unstudied. Existing last-iterate results typically require stronger assumptions such as strong convexity, PL conditions, or noise symmetry.
- **Key Challenge**: Average iterates have elegant theory but are not used in practice; last iterates are used in practice but lack high-probability guarantees under heavy-tailed noise. A long-standing theoretical gap exists between the two.
- **Goal**: To provide a high-probability convergence rate for the last iterate of Clipped-SGD under the weak assumption of bounded $\alpha$-th moments ($\alpha\in(1,2]$, covering heavy-tailed cases with infinite variance), with only polylog dependence on the confidence parameter $\delta$ and without requiring prior knowledge of the total steps $K$.
- **Core Idea**: By combining **high-probability inductive analysis using a potential function**, **fine-grained scaling of the clipping threshold by $1/\sqrt{d_k}$**, and a **horizon-free failure budget $\delta_k\sim 1/k^2$**, the authors achieve the first high-probability bound for the last iterate. They further complement this with a conversion technique from "a.s. bounded updates" to "expectation bounds."

## Method

### Overall Architecture
The study focuses on Clipped-SGD with time-varying step sizes $\gamma_k$ and clipping thresholds $\lambda_k$:
$$x_{k+1}=x_k-\gamma_k\cdot\mathrm{clip}(\nabla f_{\xi_k}(x_k),\lambda_k),\quad \mathrm{clip}(x,\lambda):=\min\!\Big\{1,\tfrac{\lambda}{\|x\|}\Big\}x.$$
The analysis uses a Lyapunov potential function $\Phi_k$ as a framework to control the single-step descent with high probability. The change in the potential function is decomposed into several martingale-type sum terms, bounded term-by-term using Freedman/Bernstein inequalities, followed by a union bound. Induction is used to prove that $f(x_k)-f^*$ decays at $O(1/d_k)$ with high probability. This analysis is supported by three technical innovations and a high-probability-to-expectation converter.

```mermaid
flowchart TD
    A[Clipped-SGD: Time-varying γk, λk] --> B[Potential Function Φk = dk·(f-f*) + L‖x-x*‖²]
    B --> C[High-Prob Descent Lemma: Decompose martingale sums]
    C --> D[Freedman/Bernstein + Time-varying failure budget δt~1/t²]
    D --> E[Inductive proof: f-f* = O(1/dk) holds w.h.p.]
    E --> F[λk ~ 1/√dk scaling: Balance clipping bias/variance]
    F --> G[Theorem 1: Last-iterate high-prob rate]
    G --> H[Section 4.2: a.s. bounded update → expectation bound conversion]
```

### Key Designs

**1. Potential function-driven high-probability inductive proof: Upgrading "average iterate analysis" to "last iterate + high probability."** The authors adopt the potential function $\Phi_k=d_k(f(x_k)-f^*)+L\|x_k-x^*\|^2$, where $d_{k+1}=d_k+2\gamma_k L,\ d_0=0$. This potential function originates from Bansal & Gupta's analysis of gradient descent and was used by Taylor & Bach to derive expectation bounds for the last iterate. The difference here is the target of a **high-probability** bound, rewriting the descent lemma as:
$$\Phi_K\le\Phi_0-\sum_{k}2\gamma_kL\langle x_k-x^*,\theta_k\rangle-\sum_k d_k\gamma_k\langle\nabla f(x_k),\theta_k\rangle+\sum_k(d_{k+1}+1)L\gamma_k^2\|\theta_k\|^2,$$
where $\theta_k:=g_k-\nabla f(x_k)$ is the "error" of the clipped gradient $g_k=\mathrm{clip}(\nabla f_{\xi_k}(x_k),\lambda_k)$. The three martingale sums on the right are key: they are controlled using Freedman/Bernstein-type concentration inequalities paired with a **time-varying failure budget** $\delta_t\sim 1/t^2$ and a union bound. In the potential function, $f-f^*$ is weighted by $d_k$ (which grows with accumulated step sizes), making the control of the **last iterate** feasible.

**2. Fine-grained scaling of clipping threshold by $1/\sqrt{d_k}$: More tightly balancing clipping bias and variance under heavy tails.** While existing average-iterate analyses usually set $\lambda_k\propto 1/\gamma_k$, this paper uses:
$$\lambda_k=\frac{R_0}{80\gamma_k\ln^{1/2}\!\big(6(k+1)^2/\delta\big)}\cdot\min\!\Big\{\tfrac{1}{\sqrt{d_k}},1\Big\},$$
where the $\min\{1/\sqrt{d_k},1\}$ factor is the crucial addition. The motivation is: since induction proves $f(x_k)-f^*$ decays at $O(1/d_k)$ with high probability, by smoothness $\|\nabla f(x_k)\|\le\sqrt{2L(f(x_k)-f^*)}=O(1/\sqrt{d_k})$, meaning the true gradient itself is shrinking. Shrinking the clipping threshold synchronously allows for a more precise trade-off between the **bias** introduced by clipping (discarding true signal) and the **variance** (heavy-tailed tails), which is the core technical lever for achieving the last-iterate rate.

**3. Horizon-agnostic scheduling + $\delta_k\sim 1/k^2$ failure budget: Parameter selection without prior knowledge of $K$.** Both the step size $\gamma_k$ and threshold $\lambda_k$ are any-time, not depending on the total steps $K$, making them suitable for streaming or indefinite training scenarios where restart schemes cannot be used. Since the horizon is unknown, the authors cannot set the failure probability at each step to $\delta/K$. Instead, they use $\delta/k^2$ at step $k$, ensuring the total failure probability is bounded by $\delta$ via $\sum_k 1/k^2\le\pi^2/6$. The cost is the introduction of polylog factors like $\sim\ln^2(6(K+1)^2/\delta)$ and a parameter $\beta\ge(2+\alpha)/(3\alpha)$ in the exponent. Setting optimal $\beta=(2+\alpha)/(3\alpha)$ yields the last-iterate rate in Corollary 1: $\tilde O\big(1/K^{(2\alpha-2)/(3\alpha)}\big)$. Notably, while $\gamma_k, \lambda_k$ depend on $\delta$, this is a necessity for inductive arguments under heavy tails (to prove iterates remain bounded) rather than an artificial stabilization.

**4. General conversion from high probability to expectation: Using a.s. bounded updates to complete the expectation bound.** Because $\gamma_k, \lambda_k$ depend on $\delta$, expectation cannot be directly obtained via tail integration. The authors take a different path: the updates of Clipped-SGD are a.s. bounded by the clipping threshold, thus $\|x_K-x^*\|\le R_0+\sum_k\gamma_k\lambda_k\le KR_0$, implying $f(x_K)-f^*\le \tfrac{L}{2}\|x_K-x^*\|^2\le \tfrac{L R_0^2 K^2}{2}$ **with probability 1**. By combining the "good high-probability bound (6)" with the "coarse probability-1 bound (7)" and setting $\delta=1/K^3$, they obtain an expectation bound (Corollary 2) that recovers Taylor & Bach's (2019) result for $\alpha=2$ (up to log factors). This technique applies to any method with **a.s. bounded updates** (e.g., normalized SGD, SignSGD, Muon), making it a general tool beyond Clipped-SGD.

## Key Experimental Results
The paper is purely theoretical; experiments use numerical simulations to verify that the "last iterate outperforms the average iterate," reporting the 0.95 quantile and mean ± standard deviation across 1000 repetitions.

### Main Results

| Experiment | Objective Function | Noise Model | Dimension | Observation |
|------|----------|----------|------|------|
| Corrupted Gradient #1 | $f(x)=\ln(1+e^{\langle x,a\rangle})+\tfrac{\lambda}{2}\|x\|^2$ | Pareto, $\alpha\!\approx\!2$ (infinite >2.001 moments) | $d=100$ | Last iterate better than average |
| Corrupted Gradient #2 | $f(x)=\tfrac12\|x\|^2$ | Same as above | $d=100$ | Last iterate better than average |
| Statistical Learning | Expected logistic loss $\mathbb{E}[\ln(1+e^{-Y\langle x,Z\rangle})]$ | Student-t (2.001 DoF, infinite high moments) | $d=10$ | Last iterate better than average |

### Convergence Rate Comparison (Core Theoretical Results)

| Source | Convergence Type | Iterate | Noise Assumption | Rate |
|------|----------|----------|----------|------|
| Liu & Zhou (2024) | Expectation | Last | As.4, $\alpha\in(1,2]$ | $O\big(\tfrac{LR_0^2}{K^{2(\alpha-1)/\alpha}}+\tfrac{R_0\sigma}{K^{(\alpha-1)/\alpha}}\big)$ |
| Nguyen et al. (2023) | High-Prob | Average | As.4, $\alpha\in(1,2]$ | $\tilde O\big(\tfrac{LR_0^2}{K}+\tfrac{R_0\sigma}{K^{(\alpha-1)/\alpha}}\big)$ |
| **Ours** | **High-Prob** | **Last** | As.4, $\alpha\in(1,2]$ | $\tilde O\big(\tfrac{LR_0^2}{K}+\tfrac{D}{K^{2(\alpha-1)/3\alpha}}\big)$ |

Where $D:=\max\{R_0\sigma,\ L^{(\alpha-1)/(3\alpha-1)}R_0^{(4\alpha-2)/(3\alpha-1)}\sigma^{2\alpha/(3\alpha-1)},\ L^{1/3}R_0^{4/3}\sigma^{2/3}\}$.

### Key Findings
- The last iterate outperformed the average iterate in all three simulations. Crucially, the same schedule (optimized for the last iterate) was used for both, meaning the comparison was fair to the average iterate.
- As $\alpha\to 1$, results converge only to a finite neighborhood, consistent with lower bounds for the Lipschitz convex case—an inherent limitation of the heavy-tailed limit rather than a proof weakness.
- The expectation bound (Corollary 2, setting $\delta=1/K^3$) recovers the $O(1/K^{1/3})$ last-iterate rate of Taylor & Bach (2019) for $\alpha=2$ (up to log factors), validating that the conversion technique doesn't "falsely" stabilize dynamics through $\delta$-dependent parameters.
- Noise was constructed to have all moments $>2.001$ diverge (Pareto / Student-t), aiming at the hardest case for Assumption 4 at the $\alpha=2$ threshold, showing the advantage holds under true heavy-tailedness.

## Highlights & Insights
- **Filling a long-standing gap**: First to provide high-probability bounds for the last iterate of Clipped-SGD under convex-smooth objectives and heavy-tailed noise, aligning "theoretical objects of study" with "iterates used in practice."
- **The insight of $\lambda_k\sim 1/\sqrt{d_k}$ is elegant**: Feeding the dynamic fact that "gradients shrink with convergence" back into the clipping threshold design provides a refined bias-variance trade-off applicable to other clipping/normalization methods.
- **High-probability to expectation converter is a general-purpose tool**: It applies as long as updates are a.s. bounded, relevant for a wide class of methods including SignSGD, normalized SGD with momentum, and Muon.
- **Horizon-free**: Parameters do not require prior knowledge of $K$, making it naturally compatible with streaming/indefinite training and more engineering-friendly.
- **Reliable single runs**: High-probability bounds directly constrain the failure probability of a single training run, which is more practically informative than expectation bounds characterizing "average behavior."

## Limitations & Future Work
- **Rate is not yet optimal**: For $\alpha=2$, a polynomial gap exists between the last-iterate high-probability rate and the optimal expectation rate $1/\sqrt K$ (exponent $2(\alpha-1)/3\alpha$ vs $(\alpha-1)/\alpha$); whether this can be closed remains open.
- **Parameter dependence on $\delta$**: High-probability results require $\gamma_k, \lambda_k$ to depend on $\delta$ for the inductive proof; obtaining the same last-iterate bound with $\delta$-independent parameters is unresolved.
- **Expectation bound sacrifices horizon-free property**: Setting $\delta=1/K^3$ in Corollary 2 makes parameters horizon-dependent; achieving the same expectation rate with horizon-agnostic parameters is a stated open question.
- **Limited to convex + smooth**: Does not cover non-convex or generalized smooth settings closer to deep learning.
- **Limited experimental scale**: Verified only on synthetic convex objectives with $d\le100$; the advantage of the last iterate has not yet been tested on large-scale neural network training.

## Related Work & Insights
- **Heavy-tailed theory for average iterates**: Gorbunov et al. (2020), Sadiev et al. (2023), Nguyen et al. (2023), and Parletta et al. (2024/2025) proved that clipping allows average iterates to reach $\tilde O(1/K^{(\alpha-1)/\alpha})$. This paper extends the frontier to the last iterate.
- **Last-iterate theory**: Shamir & Zhang (2013), Jain et al. (2021), and Liu & Zhou (2024) explored last iterates under stronger noise assumptions. Sadiev et al. (2023) addressed heavy tails but required strong convexity/PL conditions; this work removes those extra structural assumptions.
- **Weaker noise spectra**: Eldowa & Paudice (2024) and Madden et al. (2024) used sub-Weibull tails to relax sub-Gaussianity, but still imply existence of all moments. This paper's $\alpha\in(1,2]$ model allows for infinite variance, a more aggressive heavy-tailed setting.
- **Potential function techniques**: Deriving from Bansal & Gupta's (2017) GD potential and Taylor & Bach's (2019) last-iterate expectation analysis, this paper "high-probabilities" these tools.
- **Insights**: The idea of "feedback-looping convergence progress into hyperparameters (clipping threshold)" and the "a.s. bounded update $\implies$ high-prob to expectation" paradigm are general tools reusable for other stochastic optimization methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof of high-prob last-iterate bounds for Clipped-SGD under convex-smooth heavy-tailed settings; $\lambda_k\sim1/\sqrt{d_k}$ and the conversion technique are valuable new tools.
- Experimental Thoroughness: ⭐⭐⭐ Pure theoretical work; numerical simulations verify last > average iterates, which is small-scale but appropriate.
- Writing Quality: ⭐⭐⭐⭐ Clear contributions, the three technical innovations in the proof sketch are well-explained, and the SOTA comparison table is comprehensive.
- Value: ⭐⭐⭐⭐ Aligns theoretical analysis with practical usage; the conversion technique is transferable to methods like SignSGD/Muon.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)
- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[NeurIPS 2025\] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](../../NeurIPS2025/optimization/from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)
- [\[ICLR 2026\] Birch SGD: A Tree Graph Framework for Local and Asynchronous SGD Methods](birch_sgd_a_tree_graph_framework_for_local_and_asynchronous_sgd_methods.md)

</div>

<!-- RELATED:END -->
