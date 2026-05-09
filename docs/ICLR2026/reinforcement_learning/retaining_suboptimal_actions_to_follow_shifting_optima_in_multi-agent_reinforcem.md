---
title: >-
  [Paper Note] Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL
description: >-
  [ICLR 2026][Reinforcement Learning][Multi-Agent RL] This paper proposes S2Q (Successive Sub-value Q-learning), which explicitly retains suboptimal joint actions by successively learning $K$ sub-value functions. Combined with a Softmax behavior policy for prioritized sampling among candidates, S2Q addresses the root cause of suboptimal convergence in cooperative MARL value decomposition methods—namely, that policy optima shift dynamically during training.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Multi-Agent RL
  - Value Decomposition
  - Suboptimal Action Retention
  - Softmax Behavior Policy
  - S2Q
  - CTDE
date: 2026-05-08
content_hash: 8b4bcefcf233daca
---

# Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent RL

**Conference**: ICLR 2026
**arXiv**: [2602.17062](https://arxiv.org/abs/2602.17062)
**Code**: [GitHub](https://github.com/hyeon1996/S2Q)
**Area**: Reinforcement Learning
**Keywords**: Multi-Agent RL, Value Decomposition, Suboptimal Action Retention, Softmax Behavior Policy, S2Q, CTDE

## TL;DR

This paper proposes S2Q (Successive Sub-value Q-learning), which explicitly retains suboptimal joint actions by successively learning $K$ sub-value functions. Combined with a Softmax behavior policy for prioritized sampling among candidates, S2Q addresses the root cause of suboptimal convergence in cooperative MARL value decomposition methods—namely, that policy optima shift dynamically during training.

## Background & Motivation

**Background**: Under the Centralized Training with Decentralized Execution (CTDE) paradigm, value decomposition methods such as QMIX represent the dominant framework for cooperative MARL. QMIX satisfies the Individual-Global-Max (IGM) condition via a monotonicity constraint, ensuring that maximizing individual utilities does not reduce the joint value function. WQMIX introduces an unconstrained target $Q^*$ to alleviate the monotonicity restriction, yet still focuses on a single optimal action.

**Limitations of Prior Work**:
- The monotonicity constraint in QMIX limits the expressiveness of $Q^{\text{tot}}$, preventing it from representing non-monotonic value structures.
- Although WQMIX improves value estimation via an unconstrained $Q^*$, it still tracks only a single optimal joint action.
- When exploration-driven value updates cause **optimal action drift**, information about discarded high-value alternative actions cannot be recovered.
- $\epsilon$-greedy suffers from exponential decay of joint exploration probability in large joint action spaces: for $N$ agents each with $|\mathcal{A}|$ actions, the joint exploration probability scales as $\propto \epsilon^N$.

**Key Challenge**: Existing methods discard suboptimal action information once it is no longer needed. When the value landscape shifts such that a previously suboptimal action becomes optimal, the learner cannot adapt quickly. This is clearly demonstrated in a payoff matrix experiment—after the optimum shifts from $(A,A)$ to $(C,C)$, both QMIX and WQMIX fail to track the new optimum.

**Goal**: Explicitly retain value functions for $K$ suboptimal actions so that when the optimum changes, the corresponding sub-value functions can immediately guide $Q^{\text{tot}}$ to adapt; replace $\epsilon$-greedy with a Softmax behavior policy for more efficient directed exploration.

## Method

### Overall Architecture

S2Q builds upon WQMIX and consists of the following core components:
- $Q_0^{\text{sub}} := Q^{\text{tot}}$: learns the current optimal joint action.
- $Q_1^{\text{sub}}, \dots, Q_K^{\text{sub}}$: successively learn the 1st through $K$-th suboptimal joint actions.
- $Q^*$: unconstrained joint value function (same as WQMIX).
- Encoder-Decoder: estimates the Softmax distribution during training to enable inter-agent coordination.
- All sub-value functions share the QMIX mixing architecture and satisfy the IGM condition.

### Key Design 1: Successive Sub-value Q-learning

Each $Q_k^{\text{sub}}$ learns the next suboptimal action by **suppressing** the preceding $k-1$ identified optimal/suboptimal actions in the TD target:

$$\mathcal{L}_k = \mathbb{E}\left[w_k \left(Q_k^{\text{sub}} - \left(y_t - \alpha \cdot \mathbb{I}(\mathbf{a}_t \in \mathcal{A}_{k-1,t}) \cdot \max(Q_{\text{targ}}^*, C)\right)\right)^2\right]$$

- $\mathcal{A}_{k,t} = \{\mathbf{a}_{0,t}^*, \dots, \mathbf{a}_{k,t}^*\}$: the set of the first $k$ identified actions.
- $\alpha$: suppression strength.
- $\mathbb{I}(\mathbf{a}_t \in \mathcal{A}_{k-1,t})$: suppression is applied only to previously identified actions.

**Theorem 4.1** (Correctness Guarantee): If rewards are bounded and $\alpha$ is sufficiently large, then $\mathbf{a}_{k,t}^* = \arg\max_{\mathbf{a}} Q_k^{\text{sub}}(s_t, \boldsymbol{\tau}_t, \mathbf{a}_t)$ accurately corresponds to the $k$-th suboptimal joint action of $Q^*$.

### Key Design 2: Softmax Behavior Policy

Replacing $\epsilon$-greedy, a Softmax distribution over the $K+1$ candidates is constructed based on $Q^*$ values for prioritized sampling:

$$\mathbf{P}_t = \text{Softmax}\left(\frac{Q^*(s_t, \boldsymbol{\tau}_t, \mathbf{a}_{0,t}^*)}{T}, \dots, \frac{Q^*(s_t, \boldsymbol{\tau}_t, \mathbf{a}_{K,t}^*)}{T}\right)$$

Execution proceeds by first sampling $k \sim \mathbf{P}_t$, then executing the greedy action of $Q_k^{\text{sub}}$ with $\epsilon$-greedy. Temperature $T$ controls the exploration–exploitation trade-off ($T=0.1$ is optimal).

**Key Advantage**: Rather than uniformly random exploration, this approach performs **directed exploration around promising suboptimal actions**, substantially improving the efficiency of effective exploration in large joint action spaces.

### Key Design 3: Communication-Based Coordination During Training

Accurately computing $\mathbf{P}_t$ requires global information, and all agents must select the same $k$ to execute consistently:
- Encoder $E$: maps local history $\boldsymbol{\tau}_t$ to a latent representation $z_t = E(\boldsymbol{\tau}_t)$.
- Decoder $D$: reconstructs the global state and approximate distribution $(\hat{s}_t, \hat{\mathbf{P}}_t) = D(z_t)$.
- Agents synchronously sample the same $k$ from $\hat{\mathbf{P}}_t$.

**Fully Decentralized at Test Time**: Only the greedy action of $Q_0^{\text{sub}} = Q^{\text{tot}}$ is required; no communication is needed. Communication is used solely during training to coordinate exploration. For communication-critical scenarios (e.g., SMAC-Comm), the S2Q-Comm variant can provide $z_t$ at test time as well.

## Key Experimental Results

### Main Results: SMAC-Hard+ and GRF

| Environment | QMIX | WQMIX | DOP | PAC | RiskQ | MARR | MASIA | **S2Q** |
|---|---|---|---|---|---|---|---|---|
| 5m_vs_6m | ~85% | ~88% | ~82% | ~90% | ~87% | ~90% | ~84% | **~93%** |
| MMM2 | ~75% | ~80% | ~70% | ~82% | ~78% | ~83% | ~76% | **~88%** |
| 27m_vs_30m | ~60% | ~65% | ~55% | ~68% | ~62% | ~70% | ~58% | **~78%** |
| corridor | ~40% | ~50% | ~35% | ~55% | ~45% | ~58% | ~42% | **~68%** |
| 6h_vs_8z | ~30% | ~40% | ~25% | ~45% | ~35% | ~48% | ~32% | **~65%** |
| 3s5z_vs_3s6z | ~50% | ~55% | ~40% | ~60% | ~52% | ~62% | ~48% | **~72%** |
| **Avg Win Rate** | 43.94% | - | - | - | - | - | - | **73.43%** |
| academy_3_vs_2 | ~40% | ~45% | ~30% | ~48% | ~42% | ~50% | ~38% | **~60%** |
| academy_4_vs_3 | ~25% | ~30% | ~20% | ~35% | ~28% | ~38% | ~22% | **~50%** |

S2Q consistently outperforms all baselines across all environments, with the most pronounced advantage in exploration-intensive scenarios (6h_vs_8z, 3s5z_vs_3s6z)—precisely those in which optimal action drift occurs most frequently.

### Ablation Study: Component Contribution Analysis

| Method | Avg Win Rate (SMAC-Hard+) |
|---|---|
| **S2Q** | **73.43 ± 5.29** |
| S2Q_oracle (ground-truth $\mathbf{P}_t$) | 77.47 ± 4.32 |
| S2Q_independent (independent $k$ sampling) | 46.22 ± 8.20 |
| S2Q_no_wTD (no weighted TD) | 70.59 ± 4.78 |
| S2Q_no_soft (no Softmax execution) | 55.17 ± 6.71 |
| S2Q_random (uniform $k$ sampling) | 48.05 ± 9.37 |
| QMIX (baseline) | 43.94 ± 10.06 |

Key findings:
- **S2Q_oracle** serves as a performance upper bound, confirming the importance of accurately estimating $\hat{\mathbf{P}}_t$.
- **S2Q_independent** degrades substantially → inter-agent coordination and synchronization is critical.
- **S2Q_no_soft** shows significant degradation → retaining suboptimal actions alone is insufficient; prioritized execution is essential.
- **S2Q_random** performs comparably to QMIX → ignoring the relative importance of suboptimal actions dilutes the learning signal.
- **S2Q_no_wTD** remains competitive → successive sub-value learning and execution contribute more than weighted TD.

### Hyperparameter Analysis

| Hyperparameter | Optimal Value | Analysis |
|---|---|---|
| Number of sub-networks $K$ | $K=2$ | $K=0$ fails to capture suboptimal actions; $K=3$ introduces excessive variance. |
| Softmax temperature $T$ | $T=0.1$ | $T=0.01$ is overly deterministic → insufficient exploration; $T=1.0$ over-explores → slow convergence. |
| Suppression factor $\alpha$ | sufficiently large | Required to satisfy the conditions of Theorem 4.1. |

## Key Findings

### Trajectory Behavior Analysis

Training dynamics observed in 6h_vs_8z:
- In early training, agents prefer move (survival strategy) with low hit rates.
- As training progresses, $Q^*$ discovers that hit yields higher returns.
- S2Q gradually increases the execution frequency of hit via the Softmax behavior policy.
- $Q_0^{\text{sub}}$ rapidly transitions from move to hit, with win rate rising accordingly.
- This clearly demonstrates how S2Q tracks suboptimal actions and efficiently adapts when the optimum shifts.

## Highlights & Insights

### Strengths
- **Clear Motivation**: The payoff matrix experiment intuitively illustrates the optimal drift problem, providing compelling motivation.
- **Theoretical Guarantee**: Theorem 4.1 ensures the correctness of successive suboptimal action identification.
- **Elegant Design**: The combination of communication during training and full decentralization at test time balances coordination with deployment convenience.
- **Comprehensive Experiments**: Evaluated on SMAC-Hard+, GRF, SMAC-Comm, and SMACv2, with extensions to other CTDE methods such as VDN and QPLEX.

### Limitations & Future Work
- Multiple sub-value functions increase computational and memory overhead (the paper claims the overhead is moderate but does not provide precise quantitative comparisons).
- The Softmax temperature $T$ still requires manual tuning.
- The theoretical guarantee relies on the condition that "$\alpha$ is sufficiently large"—how to determine an appropriate $\alpha$ in practice is not made explicit.
- Validation is limited to discrete action space environments; applicability to continuous action spaces remains unknown.

## Rating

⭐⭐⭐⭐ — With precise problem formulation, elegant method design, and a solid combination of theory and experiments, this paper represents a rigorous contribution to the value decomposition MARL literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[NeurIPS 2025\] Extending NGU to Multi-Agent RL: A Preliminary Study](../../NeurIPS2025/reinforcement_learning/extending_ngu_to_multi-agent_rl_a_preliminary_study.md)
- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
