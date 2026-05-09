---
title: >-
  [Paper Note] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses
description: >-
  [AAAI 2026][Reinforcement Learning][multi-agent bandit] This paper proposes MACO, a multi-agent conversational bandit framework that achieves online evaluation and user preference alignment for LLM responses through a local-agent phase elimination mechanism and an adaptive preference query strategy on a cloud server, attaining a near-optimal regret bound of $\tilde{O}(\sqrt{dMT})$.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - multi-agent bandit
  - conversational bandit
  - LLM response selection
  - user preference alignment
  - regret bound
date: 2026-05-08
content_hash: 580eb907c02cc818
---

# A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses

**Conference**: AAAI 2026
**arXiv**: [2501.01849](https://arxiv.org/abs/2501.01849)
**Code**: [GitHub](https://github.com/TarferSoul/MACO)
**Area**: Reinforcement Learning
**Keywords**: multi-agent bandit, conversational bandit, LLM response selection, user preference alignment, regret bound

## TL;DR

This paper proposes MACO, a multi-agent conversational bandit framework that achieves online evaluation and user preference alignment for LLM responses through a local-agent phase elimination mechanism and an adaptive preference query strategy on a cloud server, attaining a near-optimal regret bound of $\tilde{O}(\sqrt{dMT})$.

## Background & Motivation

**Background**: LLM response optimization primarily relies on offline evaluation (e.g., prompt engineering), and scoring responses individually incurs prohibitive computational costs — for instance, evaluating 205 zero-shot prompts on 784 GSM8K problems requires 78 GPU hours.

**Limitations of Prior Work**: ① Existing bandit methods suffer from high computational complexity when handling high-dimensional LLM features; ② most assume infinitely many arms, which is unsuitable for finite response sets; ③ fixed conversation frequencies lack adaptivity; ④ only single-agent settings are supported, precluding multi-device access.

**Key Challenge**: How to efficiently select the optimal LLM response online under multi-device, heterogeneous arm sets, and dynamic user preferences.

**Goal**: Design a multi-agent conversational bandit framework that online evaluates and selects user-preference-aligned LLM responses under anonymous multi-device access scenarios.

**Key Insight**: Combine phase elimination with an adaptive conversation mechanism that queries keywords to explore under-explored directions in the feature space, avoiding the high computational cost of G-optimal design.

**Core Idea**: Use eigendecomposition of the information matrix to identify weak directions in preference estimation, then adaptively supplement information via targeted keyword-based conversations.

## Method

### Overall Architecture

$M$ local agents (corresponding to different devices) communicate with a cloud server. Each agent maintains its own finite response set $\mathcal{A}_m$ and selects one response (arm) per round, receiving user satisfaction feedback. The server aggregates data to estimate the user preference vector $\bm{\theta}^*$ and guides agents to accelerate learning through keyword queries.

### Key Designs

1. **MACO-A (Local Agent)** — Online elimination mechanism. In each phase $p$, the information matrix $\bm{M}_m^p = \sum_{a} \frac{1}{|\mathcal{A}_m^p|} \bm{x}_a\bm{x}_a^T$ is computed and diagonalized; eigenvectors corresponding to eigenvalues below threshold $h_p$ (representing under-explored directions) are uploaded to the server. After collecting rewards for each arm, sub-optimal responses are eliminated based on the server-returned $\hat{\bm{\theta}}_p$.

2. **MACO-S (Cloud Server)** — For each agent's weak directions, the server selects the best-matching keyword $k = \arg\max_{i \in \mathcal{K}} \tilde{\bm{x}}_i^T \bm{v}_j$ and computes the query count $n_{m,k}^p$. It then aggregates data from all agents and estimates $\hat{\bm{\theta}}_p = \bm{G}^{-1}\bm{W}$.

3. **Adaptive Preference Mechanism** — Conversations are triggered only when eigenvalues fall below threshold $h_p$, avoiding wasteful queries at fixed intervals. The upper bound on conversation frequency is $\beta^{-2}(\frac{3}{4(1-2^{-2p})} - d\gamma)$.

### Loss & Training

- Linear reward model: $r_{m,t} = \langle \bm{x}_{a_{m,t}}, \bm{\theta}^*_t \rangle + \eta_{m,t}$
- Objective: minimize cumulative regret $R_M(T) = \sum_{m=1}^M \sum_{t=1}^T (\bm{x}_{a^*_{m,t}}^T\bm{\theta}^*_t - \bm{x}_{a_{m,t}}^T\bm{\theta}^*_t)$
- Phase elimination framework with doubling phase lengths; sub-optimal responses are eliminated at the end of each phase

## Key Experimental Results

### Main Results

Cumulative regret on the StyleEval dataset across different embedding models and numbers of agents:

| Algorithm | Google M=4 | Google M=16 | OpenAI M=4 | OpenAI M=16 |
|-----------|-----------|-------------|-----------|-------------|
| TRIPLE-SH | 5847.31 | 22673.76 | 7736.87 | 30138.45 |
| LinUCB | 495.67 | 2025.16 | 401.16 | 1625.90 |
| ConUCB | 237.62 | 960.33 | 190.36 | 779.50 |
| ConLinUCB-BS | 991.73 | 4011.74 | 781.52 | 3177.74 |
| **MACO** | **39.04** | **153.83** | **32.06** | **127.08** |

### Ablation Study

| Component | Function | Effect on Regret |
|-----------|----------|-----------------|
| Adaptive preference mechanism | Dynamically triggers keyword queries | Removal leads to significant regret increase |
| Phase Elimination | Eliminates sub-optimal responses per phase | Replacing with LinUCB raises regret |
| Multi-agent aggregation | Server aggregates data across devices | Regret scales as $\sqrt{M}$ in single-agent setting |

### Key Findings

- MACO outperforms all baselines by at least **8.29%** across all settings, with substantially larger margins in most cases
- Communication overhead is $O(d^2 M \log T)$, independent of the response pool size $A$
- The regret upper bound $\tilde{O}(\sqrt{dMT})$ matches the lower bound $\Omega(\sqrt{dMT})$, establishing minimax optimality

## Highlights & Insights

- **Avoiding G-optimal Design**: Adaptive keyword queries replace the computationally expensive G-optimal design while preserving theoretical guarantees.
- **Heterogeneous Multi-Agent Handling**: Agents are not required to share the same response set, which better reflects real-world deployment.
- **Strong Theory and Empirics**: The regret bound is tight, and empirical results substantially surpass all baselines.

## Limitations & Future Work

- The linear reward assumption may not hold for complex, nonlinear user preference scenarios.
- A predefined keyword set $\mathcal{K}$ satisfying the feature space coverage condition (Condition 1) is required, which may be difficult to guarantee in practice.
- The synchronous communication assumption needs to be relaxed for asynchronous or high-latency settings.
- Non-stationary preference scenarios (i.e., preference drift over time) remain to be explored.

## Related Work & Insights

- Extends the line of conversational bandit methods such as ConUCB and ConLinUCB to the multi-agent setting while avoiding G-optimal design.
- Provides a novel online evaluation paradigm for LLM response selection.
- May inspire response optimization in other multi-user LLM service platforms.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel combination of multi-agent conversational bandit and adaptive preference mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple embedding models, datasets, and parameter configurations
- Writing Quality: ⭐⭐⭐ Clear structure, though notation is dense
- Value: ⭐⭐⭐⭐ High theoretical and practical value; directly applicable to LLM service platforms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization](provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](perturbing_best_responses_in_zero-sum_games.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](bamas_structuring_budget-aware_multi-agent_systems.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](../../NeurIPS2025/reinforcement_learning/bandit_and_delayed_feedback_in_online_structured_prediction.md)

</div>

<!-- RELATED:END -->
