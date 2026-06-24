---
title: >-
  [Paper Note] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability
description: >-
  [ICML2025][Reinforcement Learning][Information Asymmetry] In online reinforcement learning scenarios characterized by information asymmetry (where the agent has private types and private actions functioning as confounders) and requiring cross-distribution knowledge transportability, this paper proposes an algorithm named OPME based on nonparametric instrumental variables (NPIV). It proves that OPME achieves an $\tilde{O}(1/\epsilon^2)$ sample complexity to learn an $\epsilon$…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Information Asymmetry"
  - "Knowledge Transportability"
  - "Instrumental Variables"
  - "Causal Identification"
  - "Sample Complexity"
  - "Principal-Agent Problem"
  - "Confounders"
  - "Online Learning"
date: 2026-05-08
content_hash: 017dc235c95724b0
---

# The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability

**Conference**: ICML2025  
**arXiv**: [2506.09940](https://arxiv.org/abs/2506.09940)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Information Asymmetry, Knowledge Transportability, Instrumental Variables, Causal Identification, Sample Complexity, Principal-Agent Problem, Confounders, Online Learning

## TL;DR

In online reinforcement learning scenarios characterized by information asymmetry (where the agent has private types and private actions functioning as confounders) and requiring cross-distribution knowledge transportability, this paper proposes an algorithm named OPME based on nonparametric instrumental variables (NPIV). It proves that OPME achieves an $\tilde{O}(1/\epsilon^2)$ sample complexity to learn an $\epsilon$-optimal policy, matching the corresponding lower bound.

## Background & Motivation

### Information Asymmetry and Multi-Agent Systems
Multi-agent systems represent a widespread paradigm in economics, social sciences, and robotics. In these systems, agents possess a **private type** $t_h$ and a **private action** $b_h$, which are unobservable to the principal and function as confounders. Agents strategically select actions depending on their private information to maximize their own utility, thereby confounding the feedback $e_h$ observed by the principal.

### Challenges of Knowledge Transportability
In many practical scenarios, the principal needs to transport knowledge learned from a **source distribution** $\mathcal{P}^s$ (the population generating the online data) to a **target distribution** $\mathcal{P}^t$ (the actual target population for service), because running large-scale experiments directly in the target environment might be infeasible. Examples include transporting clinical trial findings from New York to a Los Angeles population, or utilizing experimental data from LLMs to guide human-facing mechanism designs.

### Core Problem
Traditional RL methods assume that data are i.i.d. and unconfounded, rendering them inapplicable. The paper poses the following questions:

**Can one learn with non-i.i.d. actions in environments with confounders?**

**When the source and target distributions differ, is it possible to design near-optimal algorithms, and how does distribution shift affect the sample complexity?**

## Method

### Online Strategic Interaction Model

This paper proposes the **Online Strategic Interaction Model**, generalizing the strategic MDP formulation from Yu et al. (2022) to the online setting. The interaction process at each step $h \in [H]$ proceeds as follows:

1. The principal takes action $a_h$ in state $s_h$.
2. The private type $t_h \sim \mathcal{P}_h^s$ of the agent is sampled, and the agent strategically selects $b_h = \arg\max_b R_h^a(s_h, a_h, t_h, b)$.
3. The principal receives the manipulated feedback $e_h \sim F_h(\cdot | s_h, a_h, t_h)$ ($t_h, b_h$ are unobservable).
4. The principal receives a reward $r_h = R_h^*(s_h, a_h, e_h) + \xi_h$, where $\xi_h$ is the **endogenous noise** correlated with $t_h$.
5. State transitions to $s_{h+1} \sim P_h^*(\cdot | s_h, a_h, e_h)$.

The key challenge: both $\xi_h$ and $e_h$ are correlated with the private type $t_h$, leading to $\mathbb{E}[\xi_h | s_h, a_h, e_h] \neq 0$, which causes traditional regression methods to fail.

### Aggregated Model and Planning

Under the target distribution $\mathcal{P}^t$, the aggregated model $\bar{\mathcal{M}}^*$ is defined as:

$$\bar{R}_h^*(s_h, a_h) = \mathbb{E}_{t \sim \mathcal{P}_h^t, e \sim F_h(\cdot|s_h,a_h,t)}[R_h^*(s_h, a_h, e)]$$

$$\bar{P}_h^*(\cdot|s_h, a_h) = \mathbb{E}_{t \sim \mathcal{P}_h^t, e \sim F_h(\cdot|s_h,a_h,t)}[P_h^*(\cdot|s_h, a_h, e)]$$

The goal is to learn the optimal policy $\bar{\pi}^*$ of the aggregated model.

### Nonparametric Instrumental Variable (NPIV) Causal Identification

Core Observation: $(s_h, a_h)$ can act as the **instrumental variable (IV)** for $(s_h, a_h, e_h)$, because:

$$\mathbb{E}_{\mathcal{M}^*(\mathcal{P}^s)}[r_h - R_h^*(s_h, a_h, e_h) | s_h, a_h] = 0$$

This conditional moment equation holds because although $\xi_h$ depends on $t_h$, its expectation becomes zero once marginalized over $t_h$ given $(s_h, a_h)$.

### OPME Algorithm (Optimistic Planning with Minimax Estimation)

The algorithm follows the **optimism-in-the-face-of-uncertainty principle**. The core steps are:

**Step 1 — Minimax Risk Estimation**: Direct optimization of conditional least squares is intractable (due to the conditional expectation being inside the square). By Fenchel-Rockafellar duality, a discriminator function class $\mathcal{F}_h$ is introduced to perform minimax estimation:

$$\hat{L}_h^k(R_h) = \max_{f_h \in \mathcal{F}_h} \hat{l}_h^k(R_h, f_h) - \frac{1}{2}\sum_{\tau=1}^k f_h^2(s_h^\tau, a_h^\tau)$$

where $\hat{l}_h^k(R, f) = \sum_{\tau=1}^k f(s_h^\tau, a_h^\tau)(R(s_h^\tau, a_h^\tau, e_h^\tau) - r_h^\tau)$.

For the transition function $P_h^*$, an additional discriminator class $\mathcal{G}$ is introduced to capture the optimal value function of all candidate models, and a similar risk function is constructed.

**Step 2 — Constructing Confidence Sets**:

$$\mathcal{R}_h^k = \{R_h \in \mathcal{R}_h : \hat{L}_h^k(R_h) \leq \beta_1\}, \quad \mathcal{P}_h^k = \{P_h \in \mathcal{P}_h : \hat{L}_h^k(P_h) \leq \beta_2\}$$

**Step 3 — Knowledge Transportability and Optimistic Planning**: Use the source distribution data to estimate $R^*, P^*$, and then construct the confidence set $\bar{\mathcal{C}}^k$ of the aggregated model via the known target distribution $\mathcal{P}^t$ and the feedback manipulation distribution $F$. The optimal policy of the model with the largest cumulative reward within this set is chosen for exploration.

> In short: **Use $\mathcal{P}^s$ for estimation, and use $\mathcal{P}^t$ for exploration!**

## Theoretical Results

### Main Theorem (Sample Complexity)

| Parameter | Meaning |
|---|---|
| $\epsilon$ | Optimality gap |
| $d_{V,h}$ | Distributional Eluder dimension (model complexity) |
| $\tau_h$ | Ill-posedness parameter, measuring the estimation difficulty caused by confounders |
| $C_h^f$ | Knowledge transportability multiplier, measuring the cost of source-target distribution shift |
| $B$ | Upper bound of the function class range |

**Theorem 5.4**: Under the realizability assumption, the sample complexity of the OPME algorithm to learn an $\epsilon$-optimal policy is:

$$\tilde{O}\left(\sum_{h=1}^H B^2 d_{V,h} \tau_h C_h^f \log(|\mathcal{R} \times \mathcal{P} \times \mathcal{G} \times \mathcal{F}|/\delta) \cdot \epsilon^{-2}\right)$$

### Key Findings

- **Optimal dependency on $\epsilon^{-2}$**: This matches the lower bound of Domingues et al. (2021) (which requires $\tilde{O}(\epsilon^{-2})$ even without confounding variables).
- **Linear MDP Special Case**: $d_{V,h} \lesssim \tilde{O}(d)$, yielding a complexity of $\tilde{O}(H d \tau C^f \epsilon^{-2})$.
- **Ill-posedness parameter $\tau_h$**: This quantifies the gap between the projected execution mean squared error and the true mean squared error, representing the unavoidable overhead of confounding.
- **Transportability multiplier $C_h^f$**: When $\mathcal{P}^s = \mathcal{P}^t$, $C_h^f = 1$, which degenerates to the setting without distribution transportability.

## Highlights & Insights

- **First unified framework for information asymmetry + knowledge transportability in online RL**: This extends offline strategic MDPs to the online non-i.i.d. setting and introduces distribution shift.
- **Causal perspective on RL algorithm design**: Treating $(s_h, a_h)$ as an instrumental variable to handle endogenous noise elegantly bypasses the confounder problem.
- **Tight sample complexity**: The rate of $1/\epsilon^2$ matches the lower bound up to logarithmic factors, and the decomposition clearly showcases the independent contributions from different factors (confounding $\tau$, transportability $C^f$, and complexity $d_V$).
- **Rich practical motivation**: Examples from contract design and experimental design (such as using LLMs to replace human experiments) vividly demonstrate the application value of this model.
- **Technical innovation**: Addressing non-i.i.d. data where traditional concentration inequalities fail, this work develops a new martingale-based fast concentration analysis.

## Limitations & Future Work

- **Requirement of known target distribution $\mathcal{P}^t$ and feedback manipulation distribution $F$**: In practice, these may not be easily accessible, though the paper provides rationales for this assumption (e.g., in economics, type distributions are often known).
- **Purely theoretical contribution without experimental validation**: The algorithm involves computationally intractable steps such as maximizing over confidence sets.
- **Parameterized dependency of the lower bound is not fully characterized**: Matching lower bounds for individual parameters $\tau_h$, $C_h^f$, and $d_{V,h}$ remains an open problem.
- **Partially observable scenarios (POMDPs) are not covered**: It remains unclear whether NPIV is still effective when the state is also partially unobservable.
- **Strong concentrability assumption**: It requires the data distribution to cover the target distribution uniformly, restricting its scope of applicability.

## Related Work & Insights

- **Yu et al. (2022)**: Offline strategic MDP with i.i.d. data, which this work directly generalizes.
- **Angrist & Imbens (1995); Newey & Powell (2003)**: Classical sources of nonparametric instrumental variable methods.
- **Jin et al. (2021); Ayoub et al. (2020)**: Exploratory methods in RL using value target regression and general function approximation.
- **Pearl & Bareinboim (2011)**: Formalization of knowledge transportability in causal inference.
- **Chen & Zhang (2021); Liao et al. (2021)**: Pioneering works using instrumental variables to tackle confounded data in RL.
- **Myerson (1982)**: Economic foundations of generalized principal-agent problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic handling of information asymmetry + knowledge transportability + non-i.i.d. settings in online RL.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Tight sample complexity, novel martingale concentration analysis, and clean factorization.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rich examples, though the mathematical notation is heavy.
- Practicality: ⭐⭐⭐ — Purely theoretical framework, computational feasibility is not discussed.
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](../../ICLR2026/reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICLR 2026\] Information-based Value Iteration Networks for Decision Making Under Uncertainty](../../ICLR2026/reinforcement_learning/information-based_value_iteration_networks_for_decision_making_under_uncertainty.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)

</div>

<!-- RELATED:END -->
