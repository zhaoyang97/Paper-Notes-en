---
title: >-
  [Paper Note] Scalable Policy-Based RL Algorithms for POMDPs
description: >-
  [NeurIPS 2025][Reinforcement Learning][POMDP] This paper proposes approximating POMDPs as finite-state Superstate MDPs (where states are truncated histories)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "POMDP"
  - "Policy Optimization"
  - "TD Learning"
  - "Superstate MDP"
  - "Linear Function Approximation"
date: 2026-05-08
content_hash: 78ad95da6f9c3797
---

# Scalable Policy-Based RL Algorithms for POMDPs

**Conference**: NeurIPS 2025
**arXiv**: [2510.06540](https://arxiv.org/abs/2510.06540)  
**Code**: None  
**Area**: Reinforcement Learning / Theory
**Keywords**: POMDP, Policy Optimization, TD Learning, Superstate MDP, Linear Function Approximation

## TL;DR

This paper proposes approximating POMDPs as finite-state Superstate MDPs (where states are truncated histories), derives a tighter upper bound on the optimal value function gap (decaying exponentially with history length), and provides the first finite-time convergence guarantees for standard TD learning combined with policy optimization under non-Markovian sampling.

## Background & Motivation

Partially Observable Markov Decision Processes (POMDPs) serve as a general framework for decision-making under uncertainty, with broad applications in autonomous driving, medical diagnosis, and game playing. However, solving POMDPs faces fundamental challenges:

**Key Challenge**: POMDPs can be converted into fully observable MDPs via belief states, but belief states are continuous distributions that grow with history length, leading to PSPACE-complete computational complexity.

**Limitations of Prior Work**:
- History-based methods (Loch & Singh, Littman) are effective in practice but lack rigorous theoretical performance guarantees
- Kara & Yüksel (2023) proposed the Superstate MDP to approximate POMDPs as finite-state MDPs, but the value function gap bound is $O(1/(1-\gamma)^2)$ (overly loose)
- Cayci et al. (2024) resort to computationally expensive $m$-step TD learning as a workaround
- Q-learning methods face convergence difficulties under linear function approximation

**Core Problem**: Can standard RL algorithms (TD learning + Policy Optimization) directly approximate POMDPs with theoretical guarantees?

## Method

### Overall Architecture

1. **POMDP → Superstate MDP Conversion**: Given truncation length $l$, the history $H_t$ is truncated to the most recent $l$ (action, observation) pairs, forming the Superstate $\mathcal{G}(H_t)$
2. **Approximation Quality Analysis**: The gap between the optimal value function of the Superstate MDP and that of the original POMDP is shown to decay exponentially with $l$
3. **Policy Optimization Algorithm**: Alternates between TD learning (policy evaluation) and exponential policy updates (based on POLITEX)

### Key Designs

1. **Improved Approximation Guarantee (Theorem 2)**:
   Under the uniform filter stability condition (Assumption 1):
   $$\|V^*({\pi}(H)) - \tilde{V}(\mathcal{G}(H))\|_\infty \leq \frac{2\bar{r}(1-\rho)^l}{1-\gamma} + \frac{2\bar{r}\gamma(1-\rho)^l}{(1-\gamma)((1-\gamma)+\gamma(1-\rho)^l)}$$

   **Key Improvement**: Prior bounds scale with $(1-\gamma)^{-2}$ or $(1-\gamma)^{-3}$; the proposed bound approximates $\frac{2\bar{r}(1-\rho)^l}{1-\gamma}$ when $(1-\rho)^l$ is sufficiently small, reducing the horizon dependence from quadratic to linear. Moreover, this is a worst-case bound rather than an expectation bound.

2. **Key Algebraic Lemma (Lemma 2)**:
   For positive vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}$ with $\sum a_i = \sum c_i = 1$:
   $$|\sum a_i b_i - \sum c_i d_i| \leq \frac{\|a-c\|_1}{2}\max(\|b\|_\infty, \|d\|_\infty) + \|b-d\|_\infty - \frac{\|a-c\|_1}{4}\|b-d\|_\infty$$

   Conventional decomposition-based triangle inequality arguments introduce looseness; this inequality exploits the normalization of probability mass functions to obtain a tighter bound. The lemma has independent value and can also improve results of Subramanian & Mahajan (2019).

3. **Convergence of Standard TD Learning under Non-Markovian Sampling (Lemma 4)**:
   The core challenge is that samples are drawn from the belief state $\pi(H_t)$ of the true POMDP rather than the Superstate $\mathcal{G}(H_t)$. The authors show that under the filter stability condition, the transition operator of the Superstate MDP possesses a contraction property (Lemma 3), enabling standard TD learning to converge to a value function approximation of the Superstate MDP despite operating in a non-Markovian environment.

### Loss & Training

**Algorithm 2**: Policy Optimization for Superstate MDP
- Outer loop ($M$ policy updates): Exponential update rule from POLITEX: $\mu_i(a|B) \propto \exp(\eta \sum_{j=1}^i \bar{Q}^{\mu_{j-1}}(B,a))$
- Inner loop ($\tau + l'$ steps of TD learning): Standard SARSA updates + projection onto a ball of radius $R$
- First $l'$ steps serve as a burn-in period for filter stabilization
- Supports linear function approximation: $Q(B,a) = \phi^T(B,a)\theta$

## Key Experimental Results

This paper is a purely theoretical contribution; core results are presented as theorems.

### Main Results (Theorem 3, Regret Bound)

| Component | Description | Order |
|-----------|-------------|-------|
| $\xi_{\text{FA}}$ | Function approximation error | Depends on feature quality |
| $\xi_{\text{HA}}$ | History approximation error | $\sim (1-\rho)^l$, decays exponentially with $l$ |
| Main term | Regret convergence rate | $O(T^{3/4}\log T)$ |

Total regret: $\mathcal{R}_T \leq T \cdot (\xi_{\text{FA}} + \xi_{\text{HA}}) + O(T^{3/4}\log T)$

### Comparison with Prior Work

| Method | Approximation Bound | Algorithm | Computational Cost | Assumptions |
|--------|--------------------|-----------|--------------------|-------------|
| Kara & Yüksel (2023) | $O(1/(1-\gamma)^2)$ | Q-learning | Standard | Ergodicity + asymptotic convergence |
| Cayci et al. (2024) | Polynomial | $m$-step TD | **High** | Similar |
| Abel et al. (2016) | $O(1/(1-\gamma)^3)$ | — | — | — |
| **Ours** | $O((1-\rho)^l/(1-\gamma))$ | **Standard TD + PO** | **Standard** | Filter stability |

### Key Findings

- Standard TD learning is for the first time proven to be directly applicable to settings where the true dynamics are a POMDP, with convergence guarantees
- The $m$-step TD variant of Cayci et al. is not required, avoiding additional computational overhead
- Linear function approximation enables scalability to large state spaces
- The algebraic lemma (Lemma 2) can independently improve other POMDP approximation analyses

## Highlights & Insights

- **Practicality**: Provides theoretical justification for the intuitively appealing practice of treating finite histories as MDP states and solving with standard RL algorithms
- **Generality of the Algebraic Lemma**: Lemma 2 offers a tighter control than the triangle inequality for bounds of the form $|\sum a_ib_i - \sum c_id_i|$, with broad applicability
- **Separation of Concerns**: The value function error is decomposed into a function approximation error $\xi_{\text{FA}}$ (improvable via better features) and a history approximation error $\xi_{\text{HA}}$ (exponentially reducible by increasing $l$), providing clear guidance for hyperparameter tuning

## Limitations & Future Work

- The filter stability condition (Assumption 1) requires sufficient mixing of transition and observation kernels, which may not hold in certain deterministic environments
- The $O(T^{3/4}\log T)$ regret rate may not be optimal
- The exponentially growing Superstate space ($|\mathcal{Y}|^l \cdot |\mathcal{A}|^l$) still poses scalability challenges for large $l$
- No numerical experiments are provided to validate the tightness of theoretical bounds
- Future work may explore replacing linear function approximation with LSTMs or Transformers for greater expressive power

## Related Work & Insights

- Significantly improves upon the approximation bounds and algorithmic guarantees of Kara & Yüksel (2023)
- Extends the POLITEX algorithm from the average-reward to the discounted-reward setting
- Filter stability theory (van Handel, 2008) provides the key tool for belief state contraction
- Offers improved performance bounds for the approximate information state framework of Subramanian & Mahajan (2019)

## Rating

- Novelty: ⭐⭐⭐⭐ The algebraic lemma is novel and broadly useful; the convergence analysis of standard TD in POMDPs represents an important theoretical advance
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work with no experiments, but theorems are clearly and completely stated
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is well articulated, with thorough comparison to prior work
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for scalable POMDP solving with practical implications for RL system design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[NeurIPS 2025\] Horizon Reduction Makes RL Scalable](horizon_reduction_makes_rl_scalable.md)
- [\[NeurIPS 2025\] Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)

</div>

<!-- RELATED:END -->
