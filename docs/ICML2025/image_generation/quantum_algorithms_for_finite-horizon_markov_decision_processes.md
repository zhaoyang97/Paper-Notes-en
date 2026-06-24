---
title: >-
  [Paper Note] Quantum Algorithms for Finite-horizon Markov Decision Processes
description: >-
  [ICML2025][Image Generation][Quantum algorithms] Four quantum value iteration algorithms (QVI-1/2/3/4) are proposed to achieve multi-dimensional quantum speedups in terms of the state space $S$, action space $A$, error $\epsilon$, and horizon $H$ under both the exact dynamics and generative model settings for finite-horizon time-varying MDPs, alongside proofs of asymptotically optimal quantum lower bounds.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Quantum algorithms"
  - "finite-horizon MDPs"
  - "quantum value iteration"
  - "quantum speedup"
  - "sample complexity lower bounds"
date: 2026-05-08
content_hash: d878d8d2711785d7
---

# Quantum Algorithms for Finite-horizon Markov Decision Processes

**Conference**: ICML2025  
**arXiv**: [2508.05712](https://arxiv.org/abs/2508.05712)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Quantum algorithms, finite-horizon MDPs, quantum value iteration, quantum speedup, sample complexity lower bounds

## TL;DR

Four quantum value iteration algorithms (QVI-1/2/3/4) are proposed to achieve multi-dimensional quantum speedups in terms of the state space $S$, action space $A$, error $\epsilon$, and horizon $H$ under both the exact dynamics and generative model settings for finite-horizon time-varying MDPs, alongside proofs of asymptotically optimal quantum lower bounds.

## Background & Motivation

### Background

**Background**: Markov Decision Processes (MDPs) serve as the core mathematical framework for reinforcement learning, but they suffer from a severe "curse of dimensionality" when the state or action spaces are extremely large. Quantum computing has demonstrated significant speedups over classical algorithms in tasks such as unstructured search (Grover) and optimization.

**Limitations of Prior Work**: 

### Limitations of Prior Work

**Limitations of Prior Work**: Previous quantum RL algorithms (Wang et al. 2021; Cherrat et al. 2023) were designed only for **infinite-horizon, time-invariant** MDPs and cannot handle finite-horizon scenarios where the value functions vary over time.

### Key Challenge

**Key Challenge**: Although Wiedemann et al. (2022) attempted to address the finite-horizon setting, their sample complexity scales exponentially with respect to $S$.

### Proposed Solution

**Proposed Solution**: The quantum walk approach by Naguleswaran et al. (2006) is only applicable to deterministic shortest path problems and cannot be generalized to general MDPs.

**Goal**: To design quantum algorithms that break the classical complexity lower bounds under both the **exact dynamics setting** and the **generative model setting** of finite-horizon, time-varying MDPs.

## Method

### Problem Formulation

A finite-horizon time-varying MDP is defined as a five-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \{P_h\}, \{r_h\}, H)$, where $S = |\mathcal{S}|$, $A = |\mathcal{A}|$, and $H$ is the time horizon. The value functions satisfy the Bellman recurrence relation:

$$V_h^*(s) = \max_{a \in \mathcal{A}} \left\{ r_h(s,a) + \sum_{s'} P_h(s'|s,a) V_{h+1}^*(s') \right\}$$

### Setting 1: Exact Dynamics (Section 3)

**QVI-1 — Quadratic Speedup over Action Space $A$**

- Core Idea: In the Bellman recurrence of classical value iteration, finding the maximum requires traversing all $A$ actions. QVI-1 replaces this step with **quantum maximum search** (Dürr & Høyer 1999), reducing the complexity from $O(A)$ to $O(\sqrt{A})$.
- Quantum Query Complexity: $\tilde{O}(S^2 \sqrt{A} H)$ (compared to $O(S^2 A H)$ in the classical case).
- Outputs the exact optimal strategy $\pi^*$ and $V_0^*$ with a success probability $\geq 1 - \delta$.

**QVI-2 — Additional Speedup over State Space $S$**

- Motivation: In many problems (such as Go), the state space is much larger than the action space, making $O(S^2)$ still a bottleneck.
- Core Innovation: Proposes the **QMEBO (Quantum Mean Estimation with Binary Oracles)** subroutine, which estimates $P_{h|s,a}^T V$ using $O(\sqrt{S}/\epsilon)$ queries, compared to $O(S)$ in the classical case.
- Key Steps of QMEBO: Converts a binary-encoded probability distribution $B_p$ into an amplitude-encoded unitary operator $\hat{U}_p$, prepares the quantum state, and extracts the mean via amplitude estimation.
- Quantum Query Complexity: $\tilde{O}\left(\frac{S^{1.5}\sqrt{A} H^3}{\epsilon}\right)$.
- Outputs an $\epsilon$-optimal policy satisfying $V_h^* - \epsilon \leq \hat{V}_h \leq V_h^{\hat{\pi}} \leq V_h^*$.

### Setting 2: Generative Model (Section 4)

When the environmental dynamics are unknown but can be sampled through a generative model:

**QVI-3 — Based on Hoeffding-type Quantum Mean Estimation**

- Replaces the classical sampling in RandomizedFiniteHorizonVI (Sidford et al. 2023) with quantum mean estimation QME1, while accelerating the traversal of the action space using quantum maximum search.
- Quantum Sample Complexity: $\tilde{O}\left(\frac{S\sqrt{A} H^3}{\epsilon}\right)$.

**QVI-4 — Based on Bernstein-type Quantum Mean Estimation**

- Utilizes the variance-adaptive property and monotonicity techniques of QME2 to achieve tighter estimates.
- Quantum Sample Complexity: $\tilde{O}\left(\frac{SA H^{2.5}}{\epsilon}\right)$.
- Although less optimal in $A$ compared to QVI-3, it exhibits a better dependence on $H$.

## Theoretical Results (Key Data)

### Complexity Comparison under Exact Dynamics Setting

| Objective | Classical Upper Bound | Classical Lower Bound | Quantum Upper Bound (Ours) |
|------|---------|---------|----------------|
| Exact $\pi^*, V_0^*$ | $O(S^2 A H)$ | $\Omega(S^2 A)$ | $\tilde{O}(S^2 \sqrt{A} H)$ — **QVI-1** |
| $\epsilon$-optimal $\pi^*, \{V_h^*\}$ | $O(S^2 A H)$ | $\Omega(S^2 A)$ | $\tilde{O}(S^{1.5}\sqrt{A}H^3/\epsilon)$ — **QVI-2** |

### Complexity Comparison under Generative Model Setting

| Objective | Classical Upper Bound | Classical Lower Bound | Quantum Upper Bound (Ours) | Quantum Lower Bound (Ours) |
|------|---------|---------|----------------|----------------|
| $\epsilon$-optimal $\{Q_h^*\}$ | $O(SAH^4/\epsilon^2)$ | $\Omega(SAH^3/\epsilon^2)$ | $\tilde{O}(SAH^{2.5}/\epsilon)$ — **QVI-4** | $\Omega(SAH^{1.5}/\epsilon)$ |
| $\epsilon$-optimal $\pi^*, \{V_h^*\}$ | $O(SAH^4/\epsilon^2)$ | $\Omega(SAH^3/\epsilon^2)$ | $\tilde{O}(S\sqrt{A}H^3/\epsilon)$ — **QVI-3** | $\Omega(S\sqrt{A}H^{1.5}/\epsilon)$ |

**Key Observation**: When $H$ is a constant, the quantum upper bounds of QVI-3 and QVI-4 match the quantum lower bounds (up to logarithmic factors), **proving asymptotic optimality**.

### Summary of Speedups

- **QVI-1**: Quadratic speedup in $A$ ($A \to \sqrt{A}$)
- **QVI-2**: Dual speedups in $A$ and $S$ ($S^2 \to S^{1.5}$, $A \to \sqrt{A}$)
- **QVI-3**: Quadratic speedup in $A$ + quadratic speedup in $\epsilon$ + horizon improvement ($A \to \sqrt{A}$, $\epsilon^2 \to \epsilon$)
- **QVI-4**: Quadratic speedup in $\epsilon$ + horizon improvement ($\epsilon^2 \to \epsilon$, $H^4 \to H^{2.5}$)

## Highlights & Insights

1. **The QMEBO subroutine is the core technical contribution**: Solves the problem of quantum mean estimation from binary oracles for the first time, bridging the limitation of prior QME which only supported amplitude encoding.
2. **Establishment of quantum lower bounds**: Not only provides upper bound algorithms but also proves quantum lower bounds under the generative model setting, indicating that QVI-3/4 cannot be fundamentally improved for constant $H$.
3. **New classical lower bounds**: As a byproduct, derives a new lower bound of $\Omega(SAH^3/\epsilon^2)$ for obtaining Q-values in the classical setting.
4. **Generalization of the monotonicity technique**: Generalizes the monotonicity technique used by Sidford et al. (2018) for infinite-horizon settings to finite-horizon and quantum settings.

## Limitations & Future Work

1. **High demands on quantum hardware**: The algorithms rely on fault-tolerant quantum computers and QRAM, making actual deployment difficult in the near term.
2. **Gap in $H$ dependence**: There is still a gap in $H$ between the quantum upper and lower bounds (such as $H^3$ in QVI-3 vs. $H^{1.5}$ in the lower bound); closing this gap remains an open problem.
3. **Deterministic policies only**: Does not cover modern RL settings like stochastic policies, continuous state spaces, or function approximation.
4. **No experimental validation**: Purely theoretical work, lacking experiments on quantum simulators or real quantum hardware.

## Related Work & Insights

- **Infinite-horizon quantum MDPs**: The approximate minimax-optimal quantum algorithm by Wang et al. (2021) is the most direct prior work.
- **Foundations of quantum search/estimation**: Grover search, Dürr-Høyer maximum search, and Montanaro QME serve as the sources for the quantum subroutines in this work.
- **Classical finite-horizon MDPs**: Classical algorithms and lower bounds from Li et al. (2020) and Sidford et al. (2018, 2023) serve as the baselines for comparison.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first comprehensive family of quantum algorithms + matching lower bounds for finite-horizon time-varying MDPs.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical, without experiments.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with logical organization of the two settings across four algorithms.
- Value: ⭐⭐⭐⭐ — A significant advancement in quantum RL theory, though practical impact must await mature quantum hardware.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] Recurrent Memory for Online Interdomain Gaussian Processes](../../NeurIPS2025/image_generation/recurrent_memory_for_online_interdomain_gaussian_processes.md)
- [\[NeurIPS 2025\] ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering](../../NeurIPS2025/image_generation/alebench_a_benchmark_for_longhorizon_objectivedriven_algorit.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](../../CVPR2025/image_generation/finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](../../NeurIPS2025/image_generation/fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)

</div>

<!-- RELATED:END -->
