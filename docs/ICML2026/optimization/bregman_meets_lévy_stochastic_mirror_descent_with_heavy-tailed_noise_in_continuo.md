---
title: >-
  [Paper Note] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper introduces Lévy Mirror Flow (LMF)—a continuous-time SDE model for stochastic mirror descent driven by Lévy noise. It demonstrates that SMD maintains convergence guarantees even under heavy-tailed gradient noise with infinite variance (convex case: $O(\varepsilon^{-p/(p-1)})$, strongly convex case: $\tilde{O}
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 3fb6fb1ea6729e41
---
# Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time

**Conference**: ICML 2026  
**arXiv**: [2606.03769](https://arxiv.org/abs/2606.03769)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Heavy-tailed noise, Stochastic Mirror Descent, Lévy process, Convergence rate, Convex optimization  

## TL;DR

This paper introduces Lévy Mirror Flow (LMF)—a continuous-time SDE model for stochastic mirror descent driven by Lévy noise. It demonstrates that SMD maintains convergence guarantees even under heavy-tailed gradient noise with infinite variance (convex case: $O(\varepsilon^{-p/(p-1)})$, strongly convex case: $\tilde{O}(\varepsilon^{-1/(p-1)})$), and seamlessly transfers continuous-time results to discrete-time algorithms.

## Background & Motivation

**Background**: Stochastic Mirror Descent (SMD) and its variants are among the most classic first-order methods in convex stochastic optimization. The core idea is to replace Euclidean projection with non-Euclidean Bregman divergence to achieve near dimension-independent convergence guarantees in constrained optimization. Existing theoretical analyses are almost entirely built on the assumption of light-tailed (finite variance) gradient noise.

**Limitations of Prior Work**: Substantial empirical evidence suggests that gradient noise in deep neural network training exhibits a heavy-tailed distribution ($\alpha$-stable distribution), as reported across CNNs, LLMs, and reinforcement learning. When the variance of gradient noise is infinite, standard SGD may even diverge on one-dimensional quadratic functions. Existing continuous-time analysis (Stochastic Mirror Flow, SMF) only handles diffusion SDEs driven by Brownian noise, whose trajectories are continuous with Gaussian increments, failing completely to characterize the "heavy jumps" observed in heavy-tailed scenarios.

**Key Challenge**: Infinite variance caused by heavy-tailed noise renders the classical Itô formula (which relies on finite second moments) invalid. Furthermore, the Fenchel coupling function is only Lipschitz smooth rather than $C^2$, breaking standard stochastic analysis tools.

**Goal**: Establish a unified theoretical framework from continuous to discrete time to rigorously prove the convergence, concentration, and first-arrival time guarantees of SMD under heavy-tailed noise.

**Key Insight**: Replace the noise source of SMD from Brownian motion with a centered Lévy process ($p$-th moment finite, $1 < p \le 2$). The resulting SDE naturally allows for infinite variance and jump discontinuities of any size, providing a more faithful description of heavy-tailed training dynamics.

**Core Idea**: Use Lévy Mirror Flow (LMF) as a continuous-time proxy model for heavy-tailed SMD, develop a new weak Itô formula to handle non-$C^2$ convex functions, and establish a transparent characterization of convergence rates for backward discrete-time transfer.

## Method

### Overall Architecture

The input is a convex optimization problem $\min_{x \in X} f(x)$, where $X$ is a compact convex set. The optimizer accesses stochastic gradients $g_t = \nabla f(x_t) + U_t$ via a black-box gradient oracle, where the noise $U_t$ has only a finite $p$-th moment ($1 < p \le 2$) and potentially infinite variance. The method expands across two layers: (1) The continuous-time layer defines LMF and establishes theories for convergence, concentration, and first-arrival time; (2) The discrete-time layer analyzes three SMD variants—SDA, LMD, and SMD—proving that discrete bounds can be decomposed into "continuous-time terms + discretization terms."

### Key Designs

**1. Lévy Mirror Flow (LMF): Continuous-Time Proxy for Heavy-Tailed SMD**

Brownian-driven SMF trajectories are continuous with Gaussian increments, which cannot characterize the "large jumps" of heavy-tailed noise. LMF replaces Brownian noise in SMF with a Lévy process $L(t)$, defining the dual-space SDE $dY(t) = -\nabla f(X(t))dt + dL(t)$, with primal iterations $X(t) = Q(\eta(t)Y(t))$. The Lévy process is decomposed via Lévy-Itô decomposition into three parts: the diffusion component $M(t)$ (Brownian), the short-jump component $S(t)$ (bounded jumps), and the unbounded-jump component $U(t)$ ($p$-th moment finite but variance possibly infinite). Accordingly, noise intensity is split into a tame part $\sigma^2_{\text{tame}} = \sigma^2_0 + \sigma^2_{\text{short}}$ and a heavy part $\sigma^p_{\text{heavy}} = \sigma^p_{\text{long}}$. The advantage is that the contributions of light versus heavy tails are clearly visible in the convergence rate, and LMF emerges naturally as the scaling limit of SMD under heavy-tailed noise.

**2. Weak Itô Formula: Lévy-style Chain Rule for Lipschitz Smooth Convex Functions**

The technical core of the analysis is that the classical Itô lemma cannot be used—it requires second-order continuous differentiability, whereas the Fenchel coupling $F(q,y) = h(q) + h^*(y) - \langle y, q \rangle$ analyzed here is only Lipschitz smooth, not $C^2$. The authors handle this by first applying mollification to $F$, then deriving a "weak Itô formula" that holds only as an inequality. To handle the second-order jump terms brought by Lévy jumps, they use $p$-th moment finiteness instead of second moments to control unbounded jumps. Without this tool, the Lévy jump term could not be controlled in the evolution of the energy function $E(t) = F(q, \eta(t)Y(t))/\eta(t)$. To the authors' knowledge, this result is novel in stochastic analysis literature.

**3. Unified Continuous-Discrete Analysis: Additive Decomposition of Guarantees**

Continuous-time theory alone is insufficient; it must be applied to actual discrete algorithms. This work establishes convergence rates for SDA (Stochastic Dual Averaging), LMD (Lazy Mirror Descent), and SMD under the relative smoothness assumption $f(x') \le f(x) + \langle \nabla f(x), x'-x\rangle + LD(x',x)$. It proves that all discrete bounds can be decomposed into a "continuous-time term + $[f(x_1) - \min f]$ discretization term," the latter of which naturally vanishes with iterations. Relative smoothness is used instead of standard Lipschitz smoothness because it is naturally compatible with Bregman geometry and handles gradients that diverge at the boundaries of the constraint set.

### Loss & Training

This is a purely theoretical work and does not introduce new training objectives or algorithmic implementations. The analysis is built on two core assumptions: the constraint set $X$ is compact and convex; the gradient oracle noise satisfies the martingale difference condition and has a finite $p$-th moment ($1 < p \le 2$, variance may be infinite). The step size $\eta(t) \propto 1/t^{1/p}$ (convex) or constant (strongly convex) is chosen, directly linked to the heavy-tail index $p$.

## Key Experimental Results

### Main Results

| Setting | Algorithm | Convergence Rate | Remarks |
|------|------|--------|------|
| Convex + Continuous | LMF, $\eta(t) = 1/t^{1/p}$ | $O(1/t^{(p-1)/p})$ | Recovers $O(1/\sqrt{t})$ at $p=2$ |
| Convex + Discrete | SDA, $\eta_t = \beta/t^{1/p}$ | $O(1/T^{(p-1)/p})$ | Matches continuous-time rate |
| Strongly Convex + Continuous | LMF, Constant $\eta$ | Geometric convergence to $O(\delta^2_\eta)$ ball | $\delta^2_\eta$ scales with $\eta$ and noise |
| Strongly Convex + Discrete | SDA, Constant $\eta$ | $\tilde{O}(\varepsilon^{-1/(p-1)})$ to $\varepsilon$-opt | Better than ergodic rate $O(\varepsilon^{-p/(p-1)})$ |
| Rel. Strongly Convex + Discrete | LMD, $\gamma_t = \beta/t$ | $O(1/t^{p-1})$ ($p < 1+\beta\mu$) | Varies across three regimes based on $p$ and $\beta\mu$ |

### Key Experimental Results

| Guarantee Type | Continuous Theorem | Discrete Theorem | Key Quantity |
|----------|------------|------------|--------|
| Ergodic Convergence | Theorem 1 | Theorem 5 | Time average $\bar{X}(t)$ |
| Concentration | Theorem 2 | Theorem 7 | Occupation time ratio $\mu_T(B_\delta)$ |
| First-Arrival Time | Theorem 3 | Theorem 8 | $\tau_\delta = \inf\{t: \|X(t)-x^*\| \le \delta\}$ |
| Last-iterate Convergence | Theorem 4 | Theorem 6/9 | $E[\|x_t - x^*\|^2]$ |

### Key Findings

- Heavy-tailed noise causes the convergence rate to degrade from $O(1/\sqrt{t})$ to $O(1/t^{(p-1)/p})$; the degradation varies smoothly with $p$ and is entirely controlled by the heavy jump term $\sigma^p_{\text{heavy}}$.
- Despite large jump discontinuities and infinite variance in LMF trajectories, SMD still maintains convergence—the constraint mechanism of the Bregman structure effectively "absorbs" long jumps.
- Quantitative matching between discrete and continuous bounds validates the faithfulness of LMF as a proxy for heavy-tailed SMD.
- Numerical experiments verify that $f(\bar{x}_T)$ decays via power law on simple 2D strongly convex functions; heavy tails ($\alpha = p = 3/2$) converge slower than light tails ($\alpha = p = 2$) but still converge.

## Highlights & Insights

- **Generality of the Weak Itô Formula**: This tool is valuable for any future work requiring the analysis of Lévy-driven optimization; it extends the chain rule of Itô calculus from $C^2$ functions + Brownian motion to Lipschitz convex functions + Lévy processes.
- **Noise Decoupling Design**: Splitting Lévy noise into $\sigma_{\text{tame}}$ (light-tailed) and $\sigma_{\text{heavy}}$ (heavy-tailed) components allows independent tracking of noise contributions. This "diagnostic" analysis can be transferred to adaptive optimizer design (e.g., auto-adjusting $\eta \propto 1/t^{1/p}$ based on the tail index $p$).
- **"Additive Decomposition" from Continuous to Discrete**: The framework where "discrete bound = continuous-time term + discretization term" provides a systematic analysis paradigm—establishing clean guarantees in continuous time before handling discretization errors.

## Limitations & Future Work

- The discrete-time rate for the strongly convex case $\tilde{O}(\varepsilon^{-1/(p-1)})$ does not match the lower bound $\Omega(\varepsilon^{-p/[2(p-1)]})$ from Zhang et al.; the authors speculate this can be improved by changing the energy function.
- Theoretical assumptions require the constraint set $X$ to be compact and convex, and the gradient oracle noise must satisfy martingale difference conditions with finite $p$-th moments—not directly applicable to unconstrained or open-set optimization.
- Numerical experiments are only verified on simple 2D functions, lacking large-scale empirical evidence from actual deep learning training.
- Potential applications of LMF in sampling problems (e.g., Langevin dynamics on constrained spaces) remain unexplored.

## Related Work & Insights

- Nemirovski & Yudin’s classic mirror descent theory and the optimal lower bound $\Omega(t^{-(p-1)/p})$.
- Zhang et al. (2020) showed that SGD with fixed step sizes might diverge under heavy-tailed noise, whereas gradient clipping restores convergence.
- Şimşekli (2017) proposed fractional Langevin Monte Carlo, modeling SGD with $\alpha$-stable Lévy processes.
- Liu (2024) established similar convergence rates for SGD/Dual Averaging under unbounded variance.
- Insight: The noise decomposition (tame + heavy) of Lévy processes can be used to design adaptive gradient clipping thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)

</div>

<!-- RELATED:END -->
