---
title: >-
  [Paper Note] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper proposes Lévy Mirror Flow (LMF)—a continuous-time SDE model for Stochastic Mirror Descent driven by Lévy noise. It proves that SMD maintains convergence guarantees even under heavy-tailed gradient noise with infinite variance (convex case $O(\varepsilon^{-p/(p-1)})$, strongly convex case $\tilde{O}(\varepsil
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: ce744ec5e2c5e353
---
# Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time

**Conference**: ICML 2026  
**arXiv**: [2606.03769](https://arxiv.org/abs/2606.03769)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Heavy-tailed noise, Stochastic Mirror Descent, Lévy process, Convergence rate, Convex optimization  

## TL;DR

This paper proposes Lévy Mirror Flow (LMF)—a continuous-time SDE model for Stochastic Mirror Descent driven by Lévy noise. It proves that SMD maintains convergence guarantees even under heavy-tailed gradient noise with infinite variance (convex case $O(\varepsilon^{-p/(p-1)})$, strongly convex case $\tilde{O}(\varepsilon^{-1/(p-1)})$), and seamlessly transfers continuous-time results to discrete-time algorithms.

## Background & Motivation

**Background**: Stochastic Mirror Descent (SMD) and its variants are among the most classic first-order methods in convex stochastic optimization. The core idea is to replace Euclidean projection with Bregman divergence of non-Euclidean geometry, thereby obtaining near dimension-independent convergence guarantees in constrained optimization. Existing theoretical analyses are almost entirely established on the assumption of light-tailed (finite variance) gradient noise.

**Limitations of Prior Work**: Extensive empirical evidence indicates that gradient noise in deep neural network training exhibits heavy-tailed distributions ($\alpha$-stable distributions), reported from CNNs to LLMs and reinforcement learning. When the variance of gradient noise is infinite, standard SGD may even diverge on one-dimensional quadratic functions. Existing continuous-time analysis (Stochastic Mirror Flow, SMF) only handles diffusion SDEs driven by Brownian noise, whose trajectories are continuous with Gaussian increments, failing completely to characterize the "large jump" behavior in heavy-tailed scenarios.

**Key Challenge**: Infinite variance caused by heavy-tailed noise renders the classical Itô formula (which relies on finite second moments) invalid. Furthermore, the Fenchel coupling function is only Lipschitz smooth rather than $C^2$, breaking the standard stochastic analysis toolchain.

**Goal**: Establish a unified theoretical framework from continuous time to discrete time to rigorously prove the convergence, concentration, and hitting time guarantees of SMD under heavy-tailed noise.

**Key Insight**: Replace the noise source of SMD from Brownian motion with a centered Lévy process ($p$-th moment finite, $1 < p \le 2$). The resulting SDEs naturally allow for infinite variance and jump discontinuities of arbitrary size, describing heavy-tailed training dynamics more faithfully.

**Core Idea**: Use the Lévy-noise-driven Mirror Flow (LMF) as a continuous-time proxy model for heavy-tailed SMD. Develop a new Weak Itô formula to handle non-$C^2$ convex functions and establish a transparent characterization of convergence rates for backward discrete-time transfer.

## Method

### Overall Architecture

The input is a convex optimization problem $\min_{x \in X} f(x)$, where $X$ is a compact convex set. The optimizer obtains stochastic gradients $g_t = \nabla f(x_t) + U_t$ through a black-box gradient oracle, where the noise $U_t$ only has a finite $p$-th moment ($1 < p \le 2$), and the variance can be infinite. The method unfolds in two layers: (1) The continuous-time layer defines LMF and establishes theories for convergence, concentration, and hitting time; (2) The discrete-time layer analyzes three SMD variants (SDA, LMD, and SMD), proving that discrete bounds can be decomposed into a "continuous-time term + discretization term."

### Key Designs

**1. Lévy Mirror Flow (LMF): Continuous-Time Proxy for Heavy-Tailed SMD using Lévy Noise**

The Brownian-driven Stochastic Mirror Flow (SMF) has continuous trajectories and Gaussian increments, which cannot characterize the "large jumps" of heavy-tailed noise. LMF replaces the Brownian noise in SMF with a Lévy process $L(t)$, defining the dual space SDE $dY(t) = -\nabla f(X(t))dt + dL(t)$, with primal iterations $X(t) = Q(\eta(t)Y(t))$. The Lévy process is decomposed via the Lévy-Itô decomposition into three parts: a diffusion component $M(t)$ (Brownian), a short-jump component $S(t)$ (bounded jumps), and an unbounded-jump component $U(t)$ ($p$-th moment finite but variance potentially infinite). Correspondingly, the noise intensity is split into a tame part $\sigma^2_{\text{tame}} = \sigma^2_0 + \sigma^2_{\text{short}}$ and a heavy part $\sigma^p_{\text{heavy}} = \sigma^p_{\text{long}}$. This allows the convergence rate to clearly show the independent contributions of light and heavy tails. LMF emerges naturally as the scaling limit of SMD under heavy-tailed noise, accommodating both infinite variance and arbitrary large jumps.

**2. Weak Itô Formula: Supplementing the Chain Rule for Lipschitz Smooth Convex Functions**

The technical core of the analysis is that the classical Itô lemma is inapplicable here—it requires the function to be twice continuously differentiable, while the Fenchel coupling $F(q,y) = h(q) + h^*(y) - \langle y, q \rangle$ analyzed in this paper is only Lipschitz smooth and not $C^2$. The authors handle this by first applying mollification to $F$, and then deriving a "Weak Itô formula" that holds only in the form of an inequality. When handling the second-order jump terms brought by Lévy jumps, the control of unbounded jumps relies on the finite $p$-th moment of the noise rather than the second moment. Without this tool, the Lévy jump term could not be controlled in the evolution of the energy function $E(t) = F(q, \eta(t)Y(t))/\eta(t)$. To the authors' knowledge, this result is new in the stochastic analysis literature and serves any subsequent analysis of Lévy-driven optimization.

**3. Unified Continuous-Discrete Analysis Framework: "Additive Decomposition" for Discrete Algorithms**

Continuous-time theory alone is insufficient; it must be applied to actual discrete algorithms. This paper establishes convergence rates for three variants: SDA (Stochastic Dual Averaging), LMD (Lazy Mirror Descent), and SMD, under the relative smoothness assumption $f(x') \le f(x) + \langle \nabla f(x), x'-x\rangle + LD(x',x)$. It proves that all discrete bounds can be decomposed into a "continuous-time term + $[f(x_1) - \min f]$ discretization term," the latter of which disappears naturally with iterations. Relative smoothness is used instead of standard Lipschitz smoothness because it is naturally compatible with Bregman geometry and can handle cases where gradients diverge at the boundary of the constraint set (e.g., Poisson inverse problems, entropy-regularized optimal transport)—precisely where standard Lipschitz smoothness fails.

## Key Experimental Results

### Main Results

| Setting | Algorithm | Convergence Rate | Remarks |
|------|------|--------|------|
| Convex + Continuous | LMF, $\eta(t) = 1/t^{1/p}$ | $O(1/t^{(p-1)/p})$ | Degenerates to $O(1/\sqrt{t})$ for $p=2$ |
| Convex + Discrete | SDA, $\eta_t = \beta/t^{1/p}$ | $O(1/T^{(p-1)/p})$ | Matches continuous-time rate |
| Strongly Convex + Continuous | LMF, constant $\eta$ | Geometric convergence to $O(\delta^2_\eta)$ ball | $\delta^2_\eta$ scales with $\eta$ and noise intensity |
| Strongly Convex + Discrete | SDA, constant $\eta$ | $\tilde{O}(\varepsilon^{-1/(p-1)})$ to $\varepsilon$-opt | Better than ergodic rate $O(\varepsilon^{-p/(p-1)})$ |
| Rel. Str. Convex + Discrete | LMD, $\gamma_t = \beta/t$ | $O(1/t^{p-1})$ (if $p < 1+\beta\mu$) | Three cases based on $p$ vs $\beta\mu$ |

### Theoretical Guarantee Types

| Guarantee Type | Continuous Theorem | Discrete Theorem | Key Quantity |
|----------|------------|------------|--------|
| Ergodic Convergence | Theorem 1 | Theorem 5 | Time average $\bar{X}(t)$ |
| Concentration | Theorem 2 | Theorem 7 | Fraction of sojourn time $\mu_T(B_\delta)$ |
| Hitting Time | Theorem 3 | Theorem 8 | $\tau_\delta = \inf\{t: \|X(t)-x^*\| \le \delta\}$ |
| Last-iterate Convergence | Theorem 4 | Theorem 6/9 | $E[\|x_t - x^*\|^2]$ |

### Key Findings

- Heavy-tailed noise causes the convergence rate to degrade from $O(1/\sqrt{t})$ to $O(1/t^{(p-1)/p})$. The degradation varies smoothly with $p$ and is entirely controlled by the long-jump term $\sigma^p_{\text{heavy}}$.
- Despite LMF trajectories having arbitrary jump discontinuities and infinite variance, SMD still maintains convergence—the constraint mechanism of the Bregman structure effectively "absorbs" the long jumps.
- The quantitative matching of discrete-time and continuous-time bounds verifies the faithfulness of LMF as a proxy model for heavy-tailed SMD.
- Numerical experiments on a simple 2D strongly convex function verify that $f(\bar{x}_T)$ decays according to a power law; heavy tails ($\alpha = p = 3/2$) converge more slowly than light tails ($\alpha = p = 2$) but still converge.

## Highlights & Insights

- **Generality of the Weak Itô Formula**: This tool serves not only this paper but is of direct value to any subsequent work requiring analysis of Lévy-driven optimization—it generalizes the chain rule of Itô stochastic calculus from $C^2$ functions + Brownian motion to Lipschitz convex functions + Lévy processes.
- **Noise Decoupling Design**: Splitting Lévy noise into $\sigma_{\text{tame}}$ (light-tailed) and $\sigma_{\text{heavy}}$ (heavy-tailed) allows for independent tracking of each noise source's contribution. This "diagnostic" analysis approach can be transferred to adaptive optimizer design (e.g., automatically adjusting the learning rate $\eta \propto 1/t^{1/p}$ based on the noise tail index $p$).
- **"Additive Decomposition" from Continuous to Discrete**: All discrete bounds = continuous-time term + discretization term, providing a systematic analysis paradigm: establish clean convergence guarantees in continuous time first, then handle discretization errors.

## Limitations & Future Work

- The discrete-time rate for the strongly convex case $\tilde{O}(\varepsilon^{-1/(p-1)})$ does not match the lower bound $\Omega(\varepsilon^{-p/[2(p-1)]})$ from Zhang et al.; the authors speculate this can be improved by changing the energy function.
- Theoretical assumptions require the constraint set $X$ to be compact and convex, and require the gradient oracle noise to satisfy the martingale difference condition and have a finite $p$-th moment—this is not directly applicable to unconstrained or open-set optimization.
- Numerical experiments are only verified on simple 2D functions, lacking large-scale empirical evidence from actual deep learning training.
- Potential applications of LMF in sampling problems (e.g., Langevin dynamics on constrained spaces) have not been explored.

## Related Work & Insights

- Nemirovski & Yudin's classic Mirror Descent theory and its optimal lower bound $\Omega(t^{-(p-1)/p})$.
- Zhang et al. (2020) proved that SGD with a fixed step size may diverge under heavy-tailed noise, while gradient clipping can restore convergence.
- Şimşekli (2017) proposed fractional Langevin Monte Carlo, using $\alpha$-stable Lévy processes to model SGD.
- Liu (2024) established similar convergence rates for SGD/Dual Averaging under unbounded variance.
- Insight: The noise decomposition idea of Lévy processes (tame + heavy) can be used to design adaptive gradient clipping thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2025\] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)

</div>

<!-- RELATED:END -->
