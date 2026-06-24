---
title: >-
  [Paper Note] Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics
description: >-
  [ICML2025][Reinforcement Learning][Safety Certificate] This paper proposes a safety certificate design method based on invariance conditions in the probability space. It utilizes causal reinforcement learning to learn marginalized Q-functions from offline data with latent variables. This ensures long-term safety even when offline and online statistical distributions are inconsistent, and rigorously proves the persistent feasibility of safe actions.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Safety Certificate"
  - "Latent Variables"
  - "Partially Unidentifiable Dynamics"
  - "Causal Reinforcement Learning"
  - "Control Barrier Functions"
  - "Distribution Shift"
  - "Front-door Adjustment"
date: 2026-05-08
content_hash: 44b5bae54c0afa90
---

# Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics

**Conference**: ICML2025  
**arXiv**: [2506.17927](https://arxiv.org/abs/2506.17927)  
**Code**: None  
**Area**: Safe RL / Safe Control  
**Keywords**: Safety Certificate, Latent Variables, Partially Unidentifiable Dynamics, Causal Reinforcement Learning, Control Barrier Functions, Distribution Shift, Front-door Adjustment

## TL;DR

This paper proposes a safety certificate design method based on invariance conditions in the probability space. It utilizes causal reinforcement learning to learn marginalized Q-functions from offline data with latent variables. This ensures long-term safety even when offline and online statistical distributions are inconsistent, and rigorously proves the persistent feasibility of safe actions.

## Background & Motivation

In scenarios such as autonomous driving, systems are often affected by **unobservable latent variables** $W_t$ (e.g., road slickness, pedestrian intentions). These latent variables pose two major challenges:

**Partially unidentifiable dynamics**: The complete state $(X_t, W_t)$ is inaccessible, which makes the underlying transition kernel $\mathcal{P}(X_{t+1}, W_{t+1} | X_t, W_t, U_t)$ completely unidentifiable.

**Offline-to-online distribution shift**: The behavior policy $\pi^b$ in offline data depends on latent variables (e.g., human drivers brake harder when noticing wet roads), leading to $P_{\text{offline}}(X_{t+1}|X_t,U_t) \neq P_{\text{online}}(X_{t+1}|X_t,U_t)$.

Existing safe control methods (e.g., Control Barrier Functions (CBF), Lyapunov methods) typically assume:

- Fully known system dynamics or a perfect simulator
- Fully observable states
- Distributional consistency

These assumptions fail in the presence of latent variables. Furthermore, merely guaranteeing **myopic safety** is insufficient: while the impact of latent variables may not be apparent in the short term, it can lead to **irrecoverable states** (e.g., entering a low-speed limit zone at high speeds).

**Core Problem**: How to effectively guarantee the long-term safety of stochastic systems under distribution shifts and partially unidentifiable dynamics induced by latent variables?

## Method

### Problem Modeling: Confounded Markov Decision Process

The system is modeled as a confounded MDP $(\mathcal{X}, \mathcal{U}, \mathcal{W}, \mathcal{P}, H)$:

- $X_t \in \mathcal{X}$: observable state
- $U_t \in \mathcal{U}$: control action
- $W_t \in \mathcal{W}$: unobservable latent variable
- Assumption 2.1 ensures that observable states satisfy the Markov property: $P(X_{t+1}|X_t,U_t) = P(X_{t+1}|\{X_\tau\}_{\tau \le t}, \{U_\tau\}_{\tau \le t})$

The long-term safety objective requires:

$$\mathbb{P}^{\hat{\pi},\pi}(C(X_t) \cap C(X_{t+1}) \cap \cdots \cap C(X_H) | X_0) \ge 1 - \epsilon, \quad \forall t$$

### Core Innovation 1: Invariance Condition in Probability Space

Traditional methods establish forward invariance conditions in the **state space**, which requires complete dynamics and state observability. This paper establishes the invariance condition in the **probability space**.

Define the long-term safety probability function:

$$\Psi^\pi(x,t) := \mathbb{P}^\pi(C(X_t) \cap \cdots \cap C(X_H) | X_t = x)$$

**Proposition 3.1** proves that this function is equivalent to the marginalized value function on an auxiliary MDP:

$$V^\pi([x^T, k]^T) = \Psi^\pi(x, H-k)$$

Key design of the auxiliary MDP: When the safety event $C(\hat{X}_t)$ does not hold, the state is frozen $\hat{X}_{t+1} = \hat{X}_t$; the reward is set as the safety indicator function at the terminal step $r = \mathbf{1}\{k=0\}\mathbf{1}\{C(x)\}$.

### Core Innovation 2: Safety Certificate based on Q-function

**Theorem 3.2**: If $\Psi^\pi(X_0, 0) > 1 - \epsilon$ and is satisfied at all time steps:

$$\mathbb{E}[V^\pi(\hat{Y}_{t+1}) | \hat{Y}_t, U_t] - V^\pi(\hat{Y}_t) \ge 0$$

then the safety objective holds. The implication of this condition is that each action step does not decrease the expected safety probability (analogous to the supermartingale condition).

Since the online transition distribution is unknown, the above formula cannot be directly computed. **Lemma 3.3** provides an equivalent but tractable form — the **safety certificate**:

$$S(X_t, U_t, t) := Q^\pi(\hat{Y}_t, U_t) - \mathbb{E}_{U \sim \pi}[Q^\pi(\hat{Y}_t, U)] \ge 0$$

Intuition: The Q-value of the selected action must not be lower than the average Q-value under policy $\pi$.

### Core Innovation 3: Persistent Feasibility Guarantee

**Theorem 3.4** proves that at all time steps, there always exists an action $U_t \in \mathcal{U}$ satisfying the safety certificate. The proof is straightforward: taking $u^* = \arg\max_u Q^\pi(\hat{Y}_t, u)$ yields the proof directly by the properties of the maximum.

### Bridging Distribution Shift with Causal Reinforcement Learning

Utilizing a mediator $M_t$ and the **front-door adjustment** formula to estimate unbiased Q-functions from offline data:

1. **Algorithm 1**: Construct an auxiliary dataset $\tilde{\mathcal{D}}$ from the original offline dataset $\mathcal{D}$.
2. Learn $Q_M^\pi$ by iteratively solving the least-squares problem (Eq. 44).
3. Recover $Q^\pi$ from $Q_M^\pi$ using Eq. 54.

In the online control phase (Algorithm 2), solve the optimization problem:

$$\arg\min_{u} J(U^n, u) \quad \text{s.t.} \quad S(X_t, u, t) \ge 0$$

This searches for an action that stays as close as possible to the nominal policy $\pi^n$ while guaranteeing safety.

## Key Experimental Results

### Experimental Setup

A simplified driving scenario with a discrete state space:

| Setup Item | Content |
|---|---|
| State $X_t$ | 2D integer vector $[position, velocity]^T$ |
| Action $U_t$ | $\{-3,-2,-1,0,1\}$ (acceleration/deceleration) |
| Latent Variable $W_t$ | $\{0,1,2,3\}$ (road wetness, reducing braking performance) |
| Episode Length $H$ | 10 |
| Safety Threshold $\epsilon$ | 0.2 |
| Number of Simulations | 100 simulations $\times$ 100 trajectories |
| Safety Constraints | Speed-changing restrictions on different road segments |
| Baseline | Discrete-Time Control Barrier Functions (DTCBF, Cosner et al. 2023) |

### Main Results

- **Ours**: Under conditions where latent variables are unobservable and real dynamics are unavailable, the Q-function estimated based on causal RL successfully maintains the long-term safety probability above $1 - \epsilon = 0.8$, satisfying the safety objective (Figure 2).
- **DTCBF Baseline**: The safety conditions estimated using offline statistics fail under distribution shifts, failing to satisfy the long-term safety objective, with the safety probability dropping significantly in subsequent time steps.
- 95% confidence intervals validate the statistical significance of the results.

## Highlights & Insights

1. **Outstanding Theoretical Contributions**: For the first time, causal reinforcement learning is combined with safety certificate design, establishing a complete theoretical chain from offline biased data to online safety guarantees.
2. **Invariance Condition in Probability Space**: It bypasses dependencies on full dynamics and state observability, cleverly transforming the safety probability quantification problem into a standard RL problem via the auxiliary MDP value function.
3. **Persistent Feasibility Proof**: Theorem 3.4 guarantees that the safety constraint "never puts the agent in a corner," which is a key property missing in many safe RL methods.
4. **Generality of the Framework**: Although the illustration uses front-door adjustment (Shi et al. 2024), the framework is compatible with other causal RL techniques (instrumental variables, back-door adjustment, etc.).

## Limitations & Future Work

1. **Simplified Numerical Experiments Only**: Validated only in a low-dimensional discrete driving scenario, lacking experiments in continuous high-dimensional environments or real physical robots, which limits its empirical strength.
2. **Discrete Space Assumption**: The Q-function estimation and front-door adjustment formula in Algorithm 2 assume discrete state/action spaces; extending them to continuous spaces requires additional function approximation.
3. **Limitations of Assumption 2.1**: Requires that the latent variable $W_t$ is conditionally independent of historical trajectories given $X_t$, excluding scenarios where latent variables have long term memory.
4. **Assumption on Mediators**: Under Assumption 3.5, the front-door adjustment requires an observable mediator $M_t$, which is difficult to guarantee in many practical systems.
5. **Impact of Q-function Estimation Error**: Theoretical results assume the Q-function is precisely known. Dynamic error propagation from finite-sample estimations to safety guarantees remains unanalyzed.
6. **Computational Complexity**: The efficiency of solving the online optimization problem (55) in high-dimensional action spaces has not been evaluated.

## Related Work & Insights

- **Safe Control**: Control Barrier Functions (Ames et al. 2016/2019), Predictive Safety Filters (Wabersich et al. 2021), Stochastic CBFs (Clark 2021) — all requiring full dynamics.
- **Causal RL**: Value function estimation in confounded MDPs (Wang et al. 2021b, Shi et al. 2024, Bennett & Kallus 2024) — providing tools for handling distribution shifts.
- **The Bridging Role of This Work**: Employs the deconfounding capabilities of causal RL for safety certificate design for the first time, potentially inspiring further cross-disciplinary research at the intersection of "causal inference $\times$ safe control."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Pioneering causal RL + safety certificate framework, novel probability-space invariance condition)
- Experimental Thoroughness: ⭐⭐ (Only a single low-dimensional discrete simulation experiment)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous and clear theoretical derivations with a unified notation system)
- Value: ⭐⭐⭐⭐ (Opens up an important new direction, but insufficient empirical validation limits immediate impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](../../ICLR2026/reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ICML 2025\] Embedding Safety into RL: A New Take on Trust Region Methods](embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[ICLR 2026\] Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics](../../ICLR2026/reinforcement_learning/mixture-of-world_models_scaling_multi-task_reinforcement_learning_with_modular_l.md)

</div>

<!-- RELATED:END -->
