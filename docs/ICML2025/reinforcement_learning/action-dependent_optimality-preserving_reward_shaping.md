---
title: >-
  [Paper Note] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)
description: >-
  [ICML 2025][Reinforcement Learning][reward shaping] The ADOPS method is proposed. By querying the extrinsic/intrinsic value function estimates from the critic network, it adjusts rewards only when the intrinsic reward would change the preference for the optimal action. This achieves action-dependent optimality-preserving reward shaping, overcoming the limitation of PBRS which can only handle action-independent forms, and outperforms all previous optimality-preserving methods…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "reward shaping"
  - "intrinsic motivation"
  - "optimality preservation"
  - "PBRS"
  - "exploration"
  - "Montezuma's Revenge"
date: 2026-05-08
content_hash: 13399e7d19ff660a
---

# Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)

**Conference**: ICML 2025  
**arXiv**: [2505.12611](https://arxiv.org/abs/2505.12611)  
**Code**: [https://github.com/alirezakazemipour/PPO-RND](https://github.com/alirezakazemipour/PPO-RND) (based on this implementation)  
**Area**: Reinforcement Learning  
**Keywords**: reward shaping, intrinsic motivation, optimality preservation, PBRS, exploration, Montezuma's Revenge

## TL;DR

The ADOPS method is proposed. By querying the extrinsic/intrinsic value function estimates from the critic network, it adjusts rewards only when the intrinsic reward would change the preference for the optimal action. This achieves action-dependent optimality-preserving reward shaping, overcoming the limitation of PBRS which can only handle action-independent forms, and outperforms all previous optimality-preserving methods and the RND baseline on Montezuma's Revenge.

## Background & Motivation

In sparse-reward environments, Intrinsic Motivation (IM) such as RND and ICM is widely used to encourage exploration. However, IM suffers from a severe **reward hacking** problem—agents may optimize the intrinsic reward at the expense of the extrinsic reward, leading to suboptimal policies. Classic examples include the "noisy TV problem" and the "dancing with skulls" phenomenon in Montezuma's Revenge.

Existing methods attempt to address this issue:

- **PBRS (Potential-Based Reward Shaping)**: Guarantees that the set of optimal policies remains unchanged using the difference of potential functions. Ng et al. (1999) proposed the classic form $F = \gamma\Phi(s') - \Phi(s)$
- **PBIM**: Extends PBRS to finite-horizon IM settings, but requires compensating for all accumulated intrinsic rewards at the end of the episode
- **GRM (Generalized Reward Matching)**: A generalization of PBIM that compensates for intrinsic rewards in future time steps using a matching function
- **PIES (Policy-Invariant Explicit Shaping)**: Guarantees the optimality of the late-stage policy by linearly decaying the weight coefficient of IM

**Core Problem**: All of these methods fail in **complex, long-episode, sparse-reward environments**. For PBIM, the episode length causes the terminal penalty term to explode exponentially ($1/\gamma^{N-1} \approx 10^{19}$); GRM faces similar numerical issues caused by delayed compensation; PIES no longer provides IM in the second half of training, leading to a sharp decline in performance.

## Method

### Overall Architecture

The core idea of ADOPS is: instead of requiring the cumulative value of intrinsic reward to be action-independent (the approach of PBRS), it **actively detects whether the intrinsic reward will change the optimality preference of actions**, and adjusts the intrinsic reward only when it would lead to a preference reversal.

Specifically, given any initial shaping reward $F$, ADOPS transforms it into $F' = F + F_2$, where $F_2$ is a correction term. ADOPS leverages the agent's existing critic networks to estimate the extrinsic and intrinsic value functions ($\hat{V}_E^\pi, \hat{V}_I^\pi, \hat{Q}_E^\pi, \hat{Q}_I^\pi$), and determines at each time step:

1. If the current action is suboptimal in the extrinsic sense ($Q_E^\pi < V_E^\pi$), ensure it remains suboptimal after adding IM.
2. If the current action is optimal in the extrinsic sense ($Q_E^\pi \geq V_E^\pi$), ensure it remains optimal after adding IM.

### Key Design 1: ADOPS Reward Correction Formula

The core formula of ADOPS (practical version, using estimations from the current policy rather than the optimal policy):

$$F_2 = \begin{cases} \min(0, V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F - \epsilon) & \text{if } Q_E^\pi < V_E^\pi \\\\ \max(0, V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F) & \text{if } Q_E^\pi \geq V_E^\pi \end{cases}$$

Where $\epsilon$ is a tiny positive constant ($10^{-7}$ in the experiments).

Define an auxiliary quantity $\Omega = V_E^\pi - Q_E^\pi + V_I^\pi - \gamma V_I^\pi(s') - F$ and three indicator functions:

- $C_1 = \mathbf{1}(Q_E^\pi < V_E^\pi \land \Omega > 0)$: Suboptimal action but IM makes it look better
- $C_2 = \mathbf{1}(Q_E^\pi \geq V_E^\pi \land \Omega < 0)$: Optimal action but IM makes it look worse
- $C_3 = \mathbf{1}(Q_E^\pi < V_E^\pi \land \Omega \leq 0)$: Suboptimal action and IM does not reverse preferences

Thus, $F_2 = \Omega - (C_1 + C_2)\Omega - C_3\epsilon$.

**Intuition**: When IM does not cause an action preference reversal ($C_1 = C_2 = 0$), $F_2 = \Omega - C_3\epsilon$, and the agent receives almost the entire original IM. Only when IM would cause a reversal ($C_1 = 1$ or $C_2 = 1$) does ADOPS "clip" the reward so that it just barely avoids reversing. This allows ADOPS to preserve as much of the IM signal as possible.

### Key Design 2: ADOPES—A Progressive Version Merged with PIES

Since the critic estimates are inaccurate in the early stages of training, using ADOPS directly may be suboptimal. ADOPES (Action-Dependent Optimality-Preserving Explicit Shaping) combines the idea of PIES:

- Introduces a coefficient $\zeta$, but the direction is **opposite** to PIES: linearly increasing from 0 to 1
- Early training stage: $\zeta \approx 0$, the correction of $F_2$ barely takes effect, and the agent freely uses the original IM
- Late training stage: $\zeta \to 1$, $F_2$ becomes fully effective, ensuring the optimal policy remains unchanged

This both leverages the more accurate critic estimates in the late stage of training and allows the agent to continuously receive IM signals throughout the training process (unlike PIES which drops to zero in the second half).

### Key Design 3: Theoretical Guarantees—A Broader Class of Optimality Preservation than PBRS

Traditional PBRS guarantees optimality by making the intrinsic Q-value action-independent, forcing $Q_I^*(a) = V_I^* \; \forall a$. This is a **sufficient but not necessary condition** for keeping the optimal set unchanged.

ADOPS proves a more general optimality-preserving condition (Theorem 5.2):

$$\text{argmax}_a Q_{IE}^\pi = \text{argmax}_a Q_E^\pi \quad \forall \pi$$

The key step of the proof is to introduce the concept of **policy stability** (Assumption 5.1): the converged policy will not be "unstable"—meaning there does not exist another policy that differs by only one state and action and is strictly better under any mixture. This assumption holds for any reasonable learning algorithm.

The paper also proves (Theorem B.1) that there exist optimality-preserving shaping functions that **cannot** be written in any GRM/PBRS form, indicating that ADOPS indeed covers a broader family of functions.

## Key Experimental Results

### Table 1: Comparison of Final Average Extrinsic Return on Montezuma's Revenge

| Method | Average Extrinsic Return | Comparison with RND (p-value) | Optimality Preserving |
|------|------------|----------------|--------------|
| RND (baseline) | ~7400 | — | No |
| PBIM (normalized) | 0 | — | Yes (Theoretically) |
| PBIM (non-normalized) | 0 | — | Yes (Theoretically) |
| GRM (normalized, D=1) | ~5600 | p=0.009 (significantly worse) | Yes |
| GRM (non-normalized, D=1) | ~6200 | p=0.031 (significantly worse) | Yes |
| PIES | ~5400 | p=0.059 (marginally worse) | Yes |
| **ADOPS** | **~7800** | — | Yes |
| **ADOPES** | **~8400** | **p=0.038 (significantly better)** | Yes |
| **ADOPES w/ F/2** | **~8200** | — | Yes |

### Table 2: Comparison of Key Characteristics of Different Methods

| Characteristic | PBRS/PBIM | GRM | PIES | ADOPS |
|------|-----------|-----|------|-------|
| Requires episodic environment | Yes | Yes | No | No |
| Requires future-agnostic assumption | Yes | Yes | No | No |
| Allows action-dependent intrinsic reward | No | No | N/A | Yes |
| IM remains exploitable in late training | Yes | Yes | No | Yes |
| Viable in long-episode environments | No (exponential explosion) | No (numerical issues) | Yes | Yes |
| Extra architectural/computational overhead | Low | Low | Low | Low (reuses critic) |

## Key Findings

1. **PBIM completely fails in long-episode environments**: The terminal compensation term $1/\gamma^{N-1}$ reaches approximately $10^{19}$ in Montezuma's Revenge ($N=4500, \gamma=0.99$), causing immediate saturation of action probabilities and preventing the agent from receiving any extrinsic reward.
2. **Smaller delay parameter D in GRM is better**: Larger D values behave closer to PBIM's terminal compensation; while D=1 performs the best, it still underperforms compared to the RND baseline.
3. **PIES possesses a fundamental trade-off**: It performs well during the first half when IM is present (partly due to implicitly finding a better scaling factor for IM), but the performance drops sharply once IM is deactivated in the second half and fails to recover.
4. **The early PIES-like behavior of ADOPES highlights the importance of the IM scaling coefficient**: ADOPES w/ F/2 learns faster, demonstrating that the early success of PIES is mainly due to an accidental reduction of the IM coefficient rather than its optimality-preserving mechanism.
5. **ADOPES significantly outperforms the RND baseline** (p=0.038), being the only method that surpasses the baseline while guaranteeing optimality.

## Highlights & Insights

- **Elegant Perspective**: Instead of forcing IM into a potential-based format, it performs clipping at the level of action preferences. This paradigm shift overcomes the fundamental limitations of the PBRS framework.
- **Simple Implementation**: It only requires querying the existing critic networks with no extra neural networks or auxiliary optimization, making it truly plug-and-play.
- **Clever Design of ADOPES**: In the early stage, the critic is allowed to learn accurately (without ADOPS constraints), and once the critic becomes accurate, constraints are gradually enabled. This "explore first, constrain later" strategy is more reasonable than PIES's "free first, ban later" approach.
- **Solid Theoretical Contributions**: It not only proves the correctness of ADOPS but also proves the existence of optimality-preserving functions that ADOPS can handle but PBRS/GRM never can, validating the necessity of action-dependent formulations.

## Limitations & Future Work

1. **Dependence on critic estimation quality**: The correctness of ADOPS relies on the accuracy of the extrinsic and intrinsic value function estimates. Although ADOPES mitigates this through progressive enabling, it might still suffer in environments where the critic fails to learn.
2. **Practical fulfillment of Assumption 5.1**: It requires that the learning algorithm does not execute "unstable policies" after convergence, which, although intuitive, is difficult to strictly guarantee within a finite training budget.
3. **Evaluations limited to a single environment (Montezuma's Revenge)**: Even though it is a widely recognized benchmark in this field, other sparse-reward environments (e.g., Pitfall) were not tested.
4. **Requires decoupled extrinsic/intrinsic critics**: The implementation needs two sets of critic networks to separately estimate extrinsic and intrinsic value functions. Although RND already does this, it restricts the compatibility of ADOPS with RL algorithms that use a single critic.
5. **Selection of the epsilon hyperparameter**: Although theoretically any positive number suffices, in practice, excessively small values may cause numerical instability, while excessively large values might over-suppress the IM.

## Related Work & Insights

- **Comparison with EIPO**: EIPO also automatically scales IM, but it relies on an auxiliary neural network and lacks theoretical guarantees. ADOPS is simpler and mathematically supported.
- **Insight on IM scaling coefficient**: Experiments accidentally revealed that the default IM coefficient of 1.0 might be too large, and reducing it to 0.5 accelerated training. This warrants further exploration in other IM methods.
- **Generalizability**: The concept of ADOPS is not limited to RND and can theoretically be applied to any IM method (ICM, count-based, etc.), which is worth further validation.
- **A new paradigm for reward shaping**: Moving from "enforcing the shaping reward to fit a specific mathematical form" to "dynamically adjusting the shaping reward to protect action preferences" provides a fresh perspective on reward shaping.

## Rating

- Novelty: 5/5 — Overcomes the action-independent limitation of PBRS with a unique perspective
- Theoretical Depth: 5/5 — Complete optimality-preservation proof and counterexamples that cannot be covered by PBRS
- Experimental Thoroughness: 4/5 — Reaches SOTA on a recognized difficult benchmark, but only evaluated on one environment
- Practicality: 4/5 — Truly plug-and-play, but requires decoupled critics
- Overall: 5/5 — High-quality work with both strong theory and solid experiments

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Automatic Reward Shaping from Confounded Offline Data](automatic_reward_shaping_from_confounded_offline_data.md)
- [\[ICLR 2026\] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs](../../ICLR2026/reinforcement_learning/tips_turn-level_information-potential_reward_shaping_for_search-augmented_llms.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](../../ICLR2026/reinforcement_learning/learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)
- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](../../ICLR2026/reinforcement_learning/occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)

</div>

<!-- RELATED:END -->
