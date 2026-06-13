---
title: >-
  [Paper Note] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time
description: >-
  [ICML 2026][Optimization][heavy-tailed noise] This paper proposes Lévy Mirror Flow (LMF)—a continuous-time SDE model for stochastic mirror descent driven by Lévy noise. It proves that despite heavy-tailed gradient noise…
tags:
  - "ICML 2026"
  - "Optimization"
  - "heavy-tailed noise"
  - "stochastic mirror descent"
  - "Lévy process"
  - "convergence rate"
  - "convex optimization"
date: 2026-05-08
content_hash: c20ffc187f541a64
---

# Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time

**Conference**: ICML 2026  
**arXiv**: [2606.03769](https://arxiv.org/abs/2606.03769)  
**Code**: None  
**Area**: optimization  
**Keywords**: heavy-tailed noise, stochastic mirror descent, Lévy process, convergence rate, convex optimization  

## TL;DR

This paper proposes Lévy Mirror Flow (LMF)—a continuous-time SDE model for stochastic mirror descent driven by Lévy noise. It proves that despite heavy-tailed gradient noise with infinite variance, SMD maintains convergence guarantees ($O(\varepsilon^{-p/(p-1)})$ for the convex case and $\tilde{O}(\varepsilon^{-1/(p-1)})$ for the strongly convex case), and seamlessly transfers continuous-time results to discrete-time algorithms.

## Background & Motivation

**Background**: Stochastic Mirror Descent (SMD) and its variants are among the most classic first-order methods in convex stochastic optimization. The core idea is to replace Euclidean projection with Bregman divergence of non-Euclidean geometry to obtain near dimension-independent convergence guarantees in constrained optimization. Existing theoretical analyses are almost entirely established on the assumption that gradient noise is light-tailed (finite variance).

**Limitations of Prior Work**: Extensive empirical evidence indicates that gradient noise in deep neural network training exhibits heavy-tailed distributions ($\alpha$-stable distributions), as reported in settings ranging from CNNs to LLMs and reinforcement learning. When the variance of gradient noise is infinite, standard SGD may even diverge on a one-dimensional quadratic function. Existing continuous-time analysis (Stochastic Mirror Flow, SMF) only handles diffusion SDEs driven by Brownian noise, which have continuous trajectories and Gaussian increments, completely failing to characterize "large jump" behaviors in heavy-tailed scenarios.

**Key Challenge**: The infinite variance caused by heavy-tailed noise renders the classic Itô formula (which relies on finite second moments) invalid. Furthermore, the Fenchel coupling function is only Lipschitz smooth rather than $C^2$, causing the standard stochastic analysis toolchain to break down.

**Goal**: Establish a unified theoretical framework from continuous time to discrete time to rigorously prove the convergence, concentration, and first-passage time guarantees of SMD under heavy-tailed noise.

**Key Insight**: Replace the noise source of SMD from Brownian motion with a centralized Lévy process ($p$-th moment finite, $1 < p \le 2$). The resulting SDE naturally allows for infinite variance and jump discontinuities of any size, more faithfully describing heavy-tailed training dynamics.

**Core Idea**: Use Lévy Mirror Flow (LMF) as a continuous-time proxy model for heavy-tailed SMD, develop a new Weak Itô Formula to handle non-$C^2$ convex functions, and establish transparent convergence rate characterizations for backward transfer to discrete time.

## Method

### Overall Architecture

The input is a convex optimization problem $\min_{x \in X} f(x)$, where $X$ is a compact convex set. The optimizer obtains stochastic gradients $g_t = \nabla f(x_t) + U_t$ via a black-box gradient oracle, where the noise $U_t$ only has a finite $p$-th moment ($1 < p \le 2$), and the variance can be infinite. The method unfolds in two layers: (1) The continuous-time layer defines LMF and establishes theories for convergence, concentration, and first-passage time; (2) The discrete-time layer analyzes three SMD variants—SDA, LMD, and SMD—proving that discrete bounds can be decomposed into "continuous-time terms + discretization terms."

### Key Designs

1. **Lévy Mirror Flow (LMF)**:

    - **Function**: Provides a continuous-time theoretical model for SMD under heavy-tailed noise.
    - **Mechanism**: Replaces the Brownian noise of classic Stochastic Mirror Flow with a Lévy process $L(t)$, defining the dual-space SDE $dY(t) = -\nabla f(X(t))dt + dL(t)$, with primal iterations $X(t) = Q(\eta(t)Y(t))$. The Lévy process is decomposed via Lévy-Itô into three parts: a diffusion component $M(t)$ (Brownian), a short-jump component $S(t)$ (bounded jumps), and an unbounded-jump component $U(t)$ ($p$-th moment finite but variance may be infinite). Noise intensity is split into $\sigma^2_{\text{tame}} = \sigma^2_0 + \sigma^2_{\text{short}}$ and $\sigma^p_{\text{heavy}} = \sigma^p_{\text{long}}$, clearly presenting the independent contributions of light-tailed and heavy-tailed parts in the convergence rate.
    - **Design Motivation**: Brownian-driven SMF trajectories are continuous with Gaussian increments, failing to capture the "large jump" behavior of heavy-tailed noise. LMF naturally emerges as the scaling limit of SMD under heavy-tailed noise and can simultaneously handle infinite variance and arbitrarily large jumps.

2. **Weak Itô Formula**:

    - **Function**: Establishes a stochastic calculus chain rule applicable to Lévy-driven processes for non-$C^2$ convex functions.
    - **Mechanism**: Classic Itô's Lemma requires a function to be second-order continuously differentiable, but the Fenchel coupling $F(q,y) = h(q) + h^*(y) - \langle y, q \rangle$ is only Lipschitz smooth. The authors first apply mollification to $F$ and then derive a Weak Itô Formula that holds only as an inequality. To handle second-order jump terms brought by Lévy jumps, the $p$-th moment finiteness of the noise is utilized instead of the second moment to control unbounded jumps.
    - **Design Motivation**: This is the technical core of the entire analysis—without this tool, Lévy jump terms cannot be controlled in the evolution of the energy function $E(t) = F(q, \eta(t)Y(t))/\eta(t)$. To the authors' knowledge, this result is entirely new in the stochastic analysis literature.

3. **Continuous-Discrete Unified Analysis Framework**:

    - **Function**: Seamlessly transfers continuous-time convergence guarantees to discrete-time algorithms.
    - **Mechanism**: Convergence rates are established for three discrete-time variants—SDA (Stochastic Dual Averaging), LMD (Lazy Mirror Descent), and SMD—under the assumption of relative smoothness $f(x') \le f(x) + \langle \nabla f(x), x'-x\rangle + LD(x',x)$. All discrete bounds can be decomposed into a "continuous-time term + a $[f(x_1) - \min f]$ discretization term," the latter of which naturally vanishes with iteration.
    - **Design Motivation**: Relative smoothness is naturally compatible with Bregman geometry and can handle cases where gradients diverge at the boundary of the constraint set (e.g., Poisson inverse problems, entropy-regularized optimal transport), where standard Lipschitz smoothness is inapplicable.

## Key Experimental Results

### Main Results

| Setting | Algorithm | Convergence Rate | Remarks |
|------|------|--------|------|
| Convex + Continuous | LMF, $\eta(t) = 1/t^{1/p}$ | $O(1/t^{(p-1)/p})$ | $p=2$ degenerates to classic $O(1/\sqrt{t})$ |
| Convex + Discrete | SDA, $\eta_t = \beta/t^{1/p}$ | $O(1/T^{(p-1)/p})$ | Matches continuous-time rate |
| Strongly Convex + Continuous | LMF, constant $\eta$ | Geometric convergence to $O(\delta^2_\eta)$ ball | $\delta^2_\eta$ scales with $\eta$ and noise intensity |
| Strongly Convex + Discrete | SDA, constant $\eta$ | $\tilde{O}(\varepsilon^{-1/(p-1)})$ to $\varepsilon$-opt | Superior to ergodic rate $O(\varepsilon^{-p/(p-1)})$ |
| Rel. Strongly Convex + Discrete | LMD, $\gamma_t = \beta/t$ | $O(1/t^{p-1})$ (if $p < 1+\beta\mu$) | Varies across three stages based on $p$ and $\beta\mu$ |

### Theoretical Guarantee Types

| Guarantee Type | Continuous-Time Theorem | Discrete-Time Theorem | Key Quantity |
|----------|------------|------------|--------|
| Ergodic Convergence | Theorem 1 | Theorem 5 | Time average $\bar{X}(t)$ |
| Concentration Inequality | Theorem 2 | Theorem 7 | Occupation time fraction $\mu_T(B_\delta)$ |
| First-Passage Time | Theorem 3 | Theorem 8 | $\tau_\delta = \inf\{t: \|X(t)-x^*\| \le \delta\}$ |
| Last-Iterate Convergence | Theorem 4 | Theorem 6/9 | $E[\|x_t - x^*\|^2]$ |

### Key Findings

- Heavy-tailed noise causes the convergence rate to degrade from $O(1/\sqrt{t})$ to $O(1/t^{(p-1)/p})$. The degradation varies smoothly with $p$ and is entirely controlled by the long-jump term $\sigma^p_{\text{heavy}}$.
- Despite LMF trajectories having arbitrarily large jump discontinuities and infinite variance, SMD still maintains convergence—the constraint mechanism of the Bregman structure effectively "absorbs" long jumps.
- The quantitative matching of discrete-time bounds with continuous-time bounds verifies the faithfulness of LMF as a proxy model for heavy-tailed SMD.
- Numerical experiments on a simple 2D strongly convex function verify that $f(\bar{x}_T)$ decays according to a power law; heavy tail ($\alpha = p = 3/2$) converges slower than light tail ($\alpha = p = 2$) but still converges.

## Highlights & Insights

- **Generality of the Weak Itô Formula**: This tool serves not only this paper but is of direct value to any subsequent work requiring the analysis of Lévy-driven optimization—it generalizes the chain rule of Itô stochastic calculus from $C^2$ functions + Brownian motion to Lipschitz convex functions + Lévy processes.
- **Noise Decoupling Design**: Splitting Lévy noise into $\sigma_{\text{tame}}$ (light-tailed) and $\sigma_{\text{heavy}}$ (heavy-tailed) allows independent tracking of each noise source's contribution in the convergence rate. This "diagnostic" analysis approach can be transferred to the design of adaptive optimizers (e.g., automatically adjusting the learning rate $\eta \propto 1/t^{1/p}$ based on the noise tail index $p$).
- **"Additive Decomposition" from Continuous to Discrete**: The paradigm of "Discrete Bound = Continuous-Time Term + Discretization Term" provides a systematic analysis framework: establish clean convergence guarantees in continuous time first, then handle discretization errors.

## Limitations & Future Work

- The discrete-time rate for the strongly convex case $\tilde{O}(\varepsilon^{-1/(p-1)})$ does not match the lower bound $\Omega(\varepsilon^{-p/[2(p-1)]})$ from Zhang et al.; the authors speculate this could be improved by changing the energy function.
- Theoretical assumptions require the constraint set $X$ to be compact and convex, and noise from the gradient oracle must satisfy martingale difference conditions and finite $p$-th moments—not directly applicable to unconstrained or open-set optimization.
- Numerical experiments were only verified on simple 2D functions, lacking large-scale empirical evidence from actual deep learning training.
- The potential application of LMF in sampling problems (such as Langevin dynamics on constrained spaces) has not been explored.

## Related Work & Insights

- Nemirovski & Yudin's classic mirror descent theory and its optimal lower bound $\Omega(t^{-(p-1)/p})$.
- Zhang et al. (2020) proved that SGD with fixed step sizes might diverge under heavy-tailed noise, while gradient clipping can restore convergence.
- Şimşekli (2017) proposed Fractional Langevin Monte Carlo, modeling SGD with $\alpha$-stable Lévy processes.
- Liu (2024) established similar convergence rates for SGD/Dual Averaging under unbounded variance.
- **Insight**: The noise decomposition idea of Lévy processes (tame + heavy) can be used to design adaptive gradient clipping thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Can Adaptive Gradient Methods Converge under Heavy-Tailed Noise? A Case Study of AdaGrad](can_adaptive_gradient_methods_converge_under_heavy-tailed_noise_a_case_study_of_.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)

</div>

<!-- RELATED:END -->
