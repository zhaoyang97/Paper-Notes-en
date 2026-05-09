---
title: >-
  [Paper Note] Non-Stationary Bandit Convex Optimization: A Comprehensive Study
description: >-
  [NeurIPS 2025][Optimization][Bandit Convex Optimization] This paper systematically studies bandit convex optimization (BCO) in non-stationary environments, proposes two algorithms (TEWA-SE and cExO), and establishes unified regret upper and lower bounds under three non-stationarity measures (number of switches $S$, total variation $\Delta$, and path length $P$), achieving minimax optimality in several settings.
tags:
  - NeurIPS 2025
  - Optimization
  - Bandit Convex Optimization
  - Non-Stationary Environments
  - Dynamic Regret
  - Adaptive Regret
  - Minimax Optimality
date: 2026-05-08
content_hash: 4efc89692a48c731
---

# Non-Stationary Bandit Convex Optimization: A Comprehensive Study

**Conference**: NeurIPS 2025
**arXiv**: [2506.02980](https://arxiv.org/abs/2506.02980)
**Code**: None
**Area**: Optimization
**Keywords**: Bandit Convex Optimization, Non-Stationary Environments, Dynamic Regret, Adaptive Regret, Minimax Optimality

## TL;DR

This paper systematically studies bandit convex optimization (BCO) in non-stationary environments, proposes two algorithms (TEWA-SE and cExO), and establishes unified regret upper and lower bounds under three non-stationarity measures (number of switches $S$, total variation $\Delta$, and path length $P$), achieving minimax optimality in several settings.

## Background & Motivation

Bandit convex optimization (BCO) is a fundamental sequential decision-making problem: a learner selects actions from a continuous domain and observes only a (possibly noisy) loss value at the queried point, without access to gradient information. In non-stationary environments, the optimal action varies over time, requiring algorithms that minimize *dynamic regret* — the gap relative to a time-varying comparator sequence.

Prior work has addressed different non-stationarity measures in a fragmented manner, with some focusing on total variation $\Delta$ of the loss functions and others on path length $P$. This paper aims to **systematically unify** these results by establishing a complete landscape of non-stationary regret bounds for BCO, covering both convex and strongly convex function classes.

## Method

### Overall Architecture

The paper establishes a chain of reductions among regret notions: adaptive regret $R^{\text{ada}}$ → switching regret $R^{\text{swi}}$ → dynamic regret $R^{\text{dyn}}$ and path-length regret $R^{\text{path}}$. It therefore suffices to analyze adaptive regret, from which bounds under all other measures are systematically derived.

### Key Designs

1. **TEWA-SE (Polynomial-Time Algorithm)**: Built on a *sleeping experts* framework. The core idea is to run multiple expert algorithms in parallel with different learning rates, aggregated via tilted exponential weighting. Each expert runs projected online gradient descent over its active lifetime. Key innovations include: (a) using a one-point gradient estimator $\mathbf{g}_t = (d/h) y_t \boldsymbol{\zeta}_t$ in place of exact gradients, where $h$ is the smoothing parameter and $\boldsymbol{\zeta}_t$ is sampled uniformly from the unit sphere; (b) designing a strongly convex surrogate loss $\ell_t^\eta(\mathbf{x}) = -\eta \mathbf{g}_t^\top(\mathbf{x}_t - \mathbf{x}) + \eta^2 G^2 \|\mathbf{x}_t - \mathbf{x}\|^2$, which exploits curvature to reduce the per-interval regret from $\mathcal{O}(\sqrt{|I|})$ to $\mathcal{O}(\log |I|)$; (c) using a geometric covering scheme to schedule experts, with learning rates assigned on an exponential grid, ensuring only $\mathcal{O}(\log T)$ active experts per round.

2. **cExO (Improved Bounds for Convex Losses)**: For general convex functions, TEWA-SE achieves only a suboptimal $T^{3/4}$ rate. cExO runs exponential weights over a discretized action space; the key modification is performing mirror descent on the *clipped simplex* $\tilde{\Delta} = \Delta(\mathcal{C}) \cap [\gamma, 1]^{|\mathcal{C}|}$, which prevents the algorithm from concentrating excessively on a single action and thus better adapts to environmental changes. Although not polynomial-time computable, it provides tighter bounds with respect to the dimension $d$.

3. **Minimax Lower Bounds**: For strongly convex functions, hard instance constructions establish an $\Omega(d\sqrt{ST} \wedge d^{2/3}\Delta^{1/3}T^{2/3})$ lower bound, matching the upper bounds of TEWA-SE. For path-length regret, an $\Omega((d^2 P)^{2/5} T^{3/5})$ lower bound is established, improving upon the prior $d\sqrt{PT}$ lower bound.

### Loss & Training

At each round, the learner selects an action $\mathbf{z}_t \in \Theta$ and observes $y_t = f_t(\mathbf{z}_t) + \xi_t$, with the objective of minimizing dynamic regret. The smoothing parameter $h = \min(\sqrt{d} \mathsf{B}^{-1/4}, r)$ controls the bias-variance tradeoff in gradient estimation. A Bandit-over-Bandit (BoB) framework yields parameter-free variants. Under strong convexity, TEWA-SE achieves adaptive regret $\mathcal{O}(d\sqrt{\mathsf{B}}/\alpha)$ without requiring knowledge of the strong convexity parameter $\alpha$.

## Key Experimental Results

### Main Results

This paper is a purely theoretical contribution with no empirical experiments. Main results are presented as theorems and corollaries.

| Setting | Measure | TEWA-SE | cExO | Lower Bound |
|---------|---------|---------|------|-------------|
| Strongly convex + known $S$ | $R^{\text{swi}}$ | $d\sqrt{ST}$ (optimal) | — | $d\sqrt{ST}$ |
| Strongly convex + known $\Delta$ | $R^{\text{dyn}}$ | $d^{2/3}\Delta^{1/3}T^{2/3}$ (optimal) | — | $d^{2/3}\Delta^{1/3}T^{2/3}$ |
| Convex + known $S$ | $R^{\text{swi}}$ | $\sqrt{d}S^{1/4}T^{3/4}$ | $d^{5/2}\sqrt{ST}$ (optimal) | $\sqrt{ST}$ |
| Convex + known $P$ | $R^{\text{path}}$ | $d^{2/5}P^{1/5}T^{4/5}$ | $d^{5/3}P^{1/3}T^{2/3}$ (new) | $(d^2 P)^{2/5}T^{3/5}$ |

### Ablation Study

| Configuration | Result | Remarks |
|---------------|--------|---------|
| Linear surrogate vs. strongly convex surrogate | $\mathcal{O}(\sqrt{|I|})$ vs. $\mathcal{O}(\log |I|)$ | Strongly convex surrogate is key to exploiting curvature |
| Clipped domain vs. unclipped | Adaptive regret vs. static regret | Clipping converts static regret into adaptive regret |
| Known parameters vs. BoB framework | Additional $d^{1/2}T^{3/4}$ overhead | The cost of parameter-free variants is acceptable |

### Key Findings

- Under strong convexity, TEWA-SE achieves simultaneous minimax optimality in all four parameters $d, T, S, \Delta$.
- The $T^{3/4}$ vs. $T^{1/2}$ gap between BCO and OCO for convex functions reflects a fundamental separation between zeroth-order and first-order information.
- The improved path-length lower bound ($T^{3/5}$ vs. $T^{1/2}$) exploits the assumption $P = o(T)$.

## Highlights & Insights

- The paper establishes a complete theoretical landscape of non-stationary regret for BCO and provides, for the first time, optimality guarantees in several settings.
- The chain of reductions among regret notions (Proposition 1) offers an elegant unifying framework, deriving all other regret bounds from adaptive regret.
- The surrogate loss design $\eta^2 G^2 \|\mathbf{x}_t - \mathbf{x}\|^2$ cleverly circumvents the tension between $\mathbb{E}[\|\mathbf{g}_t\|]$ and $\mathbb{E}[\|\mathbf{g}_t\|^2]$ conditions encountered in OCO.
- Combining sleeping experts with gradient clipping in the bandit setting is technically non-trivial.
- TEWA-SE adaptively exploits curvature without knowing the strong convexity parameter $\alpha$, inheriting the elegant property of MetaGrad.
- The novel reduction from path-length to switching regret in Proposition 1 relies on a simple geometric argument.

## Limitations & Future Work

- cExO is not a polynomial-time algorithm; designing an efficient optimal algorithm for convex functions remains an open problem.
- The dependence on dimension $d$ is not tight in the convex case (cExO achieves $d^{5/2}$), which may be prohibitive in high-dimensional settings.
- The optimal rate for path-length regret, particularly its dependence on $d$, remains undetermined, with a gap between upper and lower bounds.
- The parameter-free BoB framework incurs additional $T^{5/6}$ or $T^{3/4}$ overhead; whether this can be eliminated through more refined techniques warrants further investigation.
- As a purely theoretical work, the results lack numerical validation.
- The noise model assumes sub-Gaussian noise; extensions to heavy-tailed noise are an important direction.
- The tradeoff in one-point function value estimation (selection of smoothing parameter $h$) is difficult to tune precisely in practice.

## Related Work & Insights

- The tilted exponential weighting idea is inherited from the MetaGrad line of work and adapted to the bandit setting.
- The clipping technique in cExO draws from non-stationary handling methods in the multi-armed bandit (MAB) literature.
- This work provides a systematic baseline for non-stationary BCO research; future work may focus on closing the gap in the convex function setting.
- The one-point gradient estimation framework of Flaxman et al. plays a central role in BCO: all upper bounds in this paper ultimately rely on this estimator.
- A clear comparison with OCO literature is drawn: first-order information yields $\sqrt{\mathsf{B}}$ adaptive regret, while zeroth-order information only achieves $\mathsf{B}^{3/4}$.
- The proposed path-length lower bound $(d^2 P)^{2/5} T^{3/5}$ improves upon the prior $d\sqrt{PT}$ lower bound by exploiting the critical assumption $P = o(T)$.
- Non-stationary techniques from the MAB literature (change detection, discounting, sliding windows) do not directly extend to continuous action spaces.
- Open problems include: a polynomial-time optimal algorithm for convex functions, and the exact optimal rate for path-length regret.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Establishes the first systematic theoretical framework for non-stationary BCO.
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical; no empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear exposition, intuitive diagram of regret relationships, concise summary tables.
- **Value**: ⭐⭐⭐⭐ Lays a solid theoretical foundation and a complete optimality landscape for non-stationary bandit convex optimization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[NeurIPS 2025\] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)

<!-- RELATED:END -->
