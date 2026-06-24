---
title: >-
  [Paper Note] Automatic Reward Shaping from Confounded Offline Data
description: >-
  [ICML 2025][Reinforcement Learning][reward shaping] Proposes the first theoretically guaranteed data-driven method to automatically learn potential-based reward shaping (PBRS) functions from offline data contains unobserved confounders. The method uses the causal Bellman optimality equation to upper bound the optimal state value as the potential function, and proves that the resulting Q-UCB Shaping algorithm enjoys a superior gap-dependent regret bound compared to vanilla Q-U…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "reward shaping"
  - "confounded MDP"
  - "causal inference"
  - "PBRS"
  - "offline RL"
date: 2026-05-08
content_hash: 560e4682e07cd4ed
---

# Automatic Reward Shaping from Confounded Offline Data

**Conference**: ICML 2025  
**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim (Columbia University)  
**arXiv**: [2505.11478](https://arxiv.org/abs/2505.11478)  
**Code**: None  
**Area**: Reinforcement Learning, Causal Inference  
**Keywords**: reward shaping, confounded MDP, causal inference, PBRS, offline RL  

## TL;DR

Proposes the first theoretically guaranteed data-driven method to automatically learn potential-based reward shaping (PBRS) functions from offline data contains unobserved confounders. The method uses the causal Bellman optimality equation to upper bound the optimal state value as the potential function, and proves that the resulting Q-UCB Shaping algorithm enjoys a superior gap-dependent regret bound compared to vanilla Q-UCB on pseudo-suboptimal state-action pairs.

## Background & Motivation

**Background**: Potential-Based Reward Shaping (PBRS) is an effective technique to accelerate RL learning—by superimposing state potential difference signals on the original reward to guide the agent to find high-reward states faster, while guaranteeing that the optimal policy remains unchanged. Ng et al. (1999) pointed out that the optimal state value function $V_h^*(s)$ is an ideal candidate for the potential function.

**Limitations of Prior Work**: In practice, the design of potential functions either relies on domain experts' manual construction (expensive and prone to misguidance) or is estimated from offline data using standard off-policy methods. However, when the offline data originates from uncontrollable behavior policies, unobserved confounding variables (e.g., environmental states observed by the demonstrator but invisible to the learner) cause standard off-policy estimation to produce systematic bias—potentially even yielding incorrect potential signals like "consistently punishing stable states".

**Key Challenge**: In a confounded MDP (CMDP), the transition function $\mathcal{T}$ and the reward function $\mathcal{R}$ are not point-identifiable from observational data (regardless of sample size). Directly estimating the value function from behavioral data can deviate significantly from the true optimal value, causing the shaping signal to mislead the agent.

**Goal**: (1) How to robustly construct potential functions from confounded offline data? (2) What theoretical guarantees can online learners using these potential functions achieve?

**Key Insight**: Draws inspiration from partial identification in causal inference—although $\mathcal{T}$ and $\mathcal{R}$ cannot be precisely identified, they can be upper-bounded using Manski bounds. As long as the potential function upper-bounds the optimal state values (conservative optimism condition), both policy invariance and accelerated exploration can be guaranteed simultaneously.

**Core Idea**: Uses causal partial ordering relations to derive an upper bound on the optimal state values from confounded offline data, which serves as the potential function for PBRS to achieve automated reward shaping.

## Method

### Overall Architecture

The entire method is formulated as a two-stage pipeline:
1. **Offline Phase**: From multiple offline datasets that may contain confounding bias, calculate the causal upper bound of the optimal state values $\bar{V}_h(s)$ using the Causal Bellman Optimal Equation, which serves as the potential function $\phi_h(s)$.
2. **Online Phase**: Incorporate the potential function into an improved Q-UCB algorithm (Q-UCB Shaping) to interactively learn the optimal policy online.

### Key Designs

1. **Confounded MDP Modeling**:

    - Function: Formalize MDPs in the presence of unobserved confounding.
    - Mechanism: Under the standard MDP baseline, CMDP introduces exogenous noise $U_h$ that simultaneously affects the behavior policy $\beta_h(S_h, U_h)$, the reward $r_h(S_h, X_h, U_h)$, and the next state $\tau_h(S_h, X_h, U_h)$. In the causal DAG, this is represented as bidirectional arrows between action-reward and action-next state.
    - Design Motivation: The standard MDP assumption requires the behavioral data to satisfy the "unconfoundedness" condition, but in practice, demonstrators may make decisions based on information invisible to the learner (e.g., body stability in robot walking), leading to spurious correlations between actions, rewards, and transitions in observational data.

2. **Causal Bellman Optimal Equation**:

    - Function: Upper bound the state value of the optimal interventional policy from confounded offline data.
    - Mechanism: Apply partial identification bounds to transitions and rewards respectively: $\mathcal{T}_h(s,x,s') \leq \tilde{T}_h(s,x,s') P_h(x|s) + P_h(\neg x|s)$, and $\mathcal{R}_h(s,x) \leq \tilde{R}_h(s,x) P_h(x|s) + b \cdot P_h(\neg x|s)$. Substituting these bounds into the standard Bellman optimality equation yields the recursive upper bound: $\bar{V}_h(s) = \max_x [P_h(x|s)(\tilde{R}_h + \mathbb{E}_{\tilde{T}}[\bar{V}_{h+1}]) + P_h(\neg x|s)(b + \max_{s'} \bar{V}_{h+1}(s'))]$. Here, the term $P_h(\neg x|s)$ represents that, for actions not selected in the behavioral data, it is optimistically assumed that they would receive the maximum possible reward $b$ and transition to the optimal next state.
    - Design Motivation: Standard value iteration yields incorrect estimates on confounded data (e.g., in the Walking Robot example, data from a competent demonstrator results in identical values for stable/unstable states, whereas a poor demonstrator even punishes stable states). The causal upper bound method bypasses the unidentifiability issue by requiring only the validity of the upper bounds. When multiple datasets are available, taking the minimum upper bound for each state (Corollary 3.2) yields the tightest estimation.

3. **Q-UCB Shaping Algorithm**:

    - Function: Online model-free learning using the causal upper bound potential function.
    - Mechanism: Introduces three key modifications to Q-UCB: (1) initialize Q-values to zero (instead of $H$); (2) the UCB bonus term depends on the maximum potential value $\phi_m = \max_s \phi(s)$: $b_t = c\sqrt{H\phi_m^2\iota/t}$; (3) updates the learning using the shaped reward $r'_h = y_h - \phi_h(s_h) + \phi_{h+1}(s_{h+1})$, and replaces the original $[0, H]$ clipping with $\min(\max(Q, 0), \phi_m)$.
    - Design Motivation: The conservative optimism condition ($V^*(s) \leq \phi(s) \leq H$) enables zero initialization—the potential function itself provides sufficient exploration signals, rendering traditional optimistic initialization unnecessary. This also shrinks the clipping range from $[0,H]$ to $[0, \phi_m]$, reducing over-exploration.

### Loss & Training

The offline phase uses an improved value iteration (Algo. 2), backwardly updating from $h=H$, skipping unvisited state-action pairs, and finally taking the minimum across datasets. The online phase uses Q-learning updates with an adaptive learning rate $\alpha_t = (H+1)/(H+t)$.

## Key Experimental Results

### Main Results

Comparison in four Windy MiniGrid environments, with wind direction as the unobserved confounder:

| Environment | Q-UCB (No Shaping) | Shaping+Min Value | Shaping+Avg Value | **Shaping+Causal Bound (Ours)** |
|------|--------|--------|--------|--------|
| Empty World | Diverges | Converges | Converges | **Converges** (on par with baseline) |
| LavaCross Easy | Diverges | Slow/Unstable | Slow/Unstable | **Fastest convergence + lowest regret** |
| LavaCross Hard | Diverges | Misguided | Misguided | **Only one to converge correctly** |
| LavaCross Maze | Diverges | Misguided/Traps | Misguided/Traps | **Only one to find the correct policy** |

### Ablation Study

| Experiment | Results |
|------|------|
| Optimal Policy Consistency Test | Ours finds the optimal interventional policy 100% of the time across all environments |
| vs BCQ (Deep Offline RL) | BCQ's optimality rate on confounded data is only 14.9%~40.1%, significantly lower than Ours' |
| Different Behavior Policy Qualities | After merging data from three types of demonstrators (good, poor, random), only the causal bound correctly integrates information |
| Walking Robot Case Study | The causal upper bound $\phi(L=0,F=0)=9.0, \phi(L=0,F=1)=9.5$ correctly upper-bounds the true optimal value $V^*(L=0,F=0)=5.0, V^*(L=0,F=1)=5.5$, and preserves the ordinal relationship of "stable is better than unstable" |

### Key Findings

- Utilizing a confounded value function for shaping not only fails to accelerate learning but also misleads the agent—in LavaCross Hard, confounded values even guide the agent into the lava area.
- The core advantage of the causal upper bound method lies in the fact that, even when value functions cannot be accurately estimated, the upper-bound relationship is sufficient to provide the correct ordinal relationship for the potential function.
- Regret analysis reveals that for pseudo-suboptimal state-action pairs $\text{Sub}_\Delta$, the regret is improved from $O(H^6/\Delta)$ to $O(H^5/\Delta)$, matching SOTA Q-learning variants (Xu et al. 2021).

## Highlights & Insights

- **Intelligent combination of causal inference and reward shaping**: Formulates the unidentifiability challenge into a weaker "upper bound only" requirement, which perfectly fits the requirements of PBRS for potential functions—the potential function does not need to be precisely equal to the optimal values, but simply needs to maintain the correct ordinal relationship.
- **Conservative Optimism acts as a key bridge** connecting causal upper bounds to regret improvements—the upper-bounding property guarantees that Q-values remain non-negative, thereby enabling zero initialization instead of traditional $H$ initialization.
- **The Walking Robot example is highly educational**: It intuitively demonstrates how confounding bias can subvert standard off-policy estimation.

## Limitations & Future Work

- Theoretical guarantees are proven only within the tabular setting, without extending to function approximation scenarios.
- The evaluation environments are limited in scale (MiniGrid) and have not been validated in high-dimensional or continuous action spaces.
- The tightness of the causal upper bound depends on the magnitude of $P(\neg x|s)$—the bound is tighter when the behavior policy almost always selects a specific action, but can be overly loose when the behavior policy is close to uniform.
- Assumed a known upper bound $b$ for rewards, which may be unrealistic in certain scenarios.

## Related Work & Insights

- **vs Gupta et al. (2022, NeurIPS)**: They proved acceleration guarantees of PBRS for model-based learners under the assumption of a known potential function. This study is the first to deliver a gap-dependent regret bound for model-free learners, with the potential function automatically learned from data.
- **vs Zhang & Bareinboim (2024)**: They investigated off-policy evaluation bounds in confounded MDPs, whereas this study applies similar causal bounds to completely different downstream tasks (reward shaping + online learning).
- **vs Ball et al. (2023) / Song et al. (2023)**: These hybrid RL works warm-start online learning using offline data, but assume the NUC (no unobserved confounding) condition holds. This study is the first to handle offline data affected by confounding bias.

## Rating

| Dimension | Score | Rationale |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | A novel overlap of causal inference and PBRS, with a highly insightful proposal of the conservative optimism condition. |
| Technical Depth | ⭐⭐⭐⭐⭐ | Rigorous derivation of the causal Bellman equation, complete regret analysis, and an elegant convergence proof (uniqueness of the fixed point in a stationary CMDP). |
| Experimental Thoroughness | ⭐⭐⭐ | Relatively small environment scale, lacks evaluation in continuous control and large-scale environments, but the analysis of the Walking Robot case is meticulous. |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation, illustrative examples run throughout the paper, and concise proof sketches. |
| Practicality | ⭐⭐⭐ | Currently limited to the tabular setting, restricting immediate practicality, yet it opens a new direction for utilizing confounded data. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](action-dependent_optimality-preserving_reward_shaping.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](../../ICLR2026/reinforcement_learning/occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)
- [\[ICML 2025\] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)
- [\[ICLR 2026\] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs](../../ICLR2026/reinforcement_learning/tips_turn-level_information-potential_reward_shaping_for_search-augmented_llms.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](../../ICLR2026/reinforcement_learning/learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)

</div>

<!-- RELATED:END -->
