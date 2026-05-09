---
title: >-
  [Paper Note] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Hierarchical RL] This paper proposes SSE (Strict Subgoal Execution), a framework that strictly distinguishes between successful and failed subgoal reaching via **Frontier Experience Replay (FER)**, combined with a decoupled exploration policy and failure-aware path optimization. By enforcing subgoal completion within each high-level step, SSE substantially reduces the number of high-level decisions and improves success rates on long-horizon tasks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Hierarchical RL
  - Subgoal Execution
  - Graph Planning
  - Frontier Experience Replay
  - Long-Horizon Tasks
date: 2026-05-08
content_hash: 7c0f069e26f949b6
---

# Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2506.21039](https://arxiv.org/abs/2506.21039)
**Code**: [https://github.com/Jaebak1996/SSE](https://github.com/Jaebak1996/SSE)
**Area**: Hierarchical Reinforcement Learning / Goal-Conditioned RL
**Keywords**: Hierarchical RL, Subgoal Execution, Graph Planning, Frontier Experience Replay, Long-Horizon Tasks

## TL;DR

This paper proposes SSE (Strict Subgoal Execution), a framework that strictly distinguishes between successful and failed subgoal reaching via **Frontier Experience Replay (FER)**, combined with a decoupled exploration policy and failure-aware path optimization. By enforcing subgoal completion within each high-level step, SSE substantially reduces the number of high-level decisions and improves success rates on long-horizon tasks.

## Background & Motivation

- **Challenges of long-horizon goal-conditioned tasks**: Distant goals, sparse rewards, and difficult exploration.
- **Issues with HER at the high level**: Conventional graph-based hierarchical RL applies HER (Hindsight Experience Replay) to the high-level policy, treating intermediate states in failed trajectories as virtual subgoals. This leads to:
    - The high-level policy repeatedly selecting unreachable subgoals
    - Excessively long high-level trajectories, making credit assignment difficult
    - Highly inconsistent transitions for the same subgoal
- **Core Idea**: Rather than allowing the high-level policy to repeatedly attempt unreachable subgoals, SSE enforces strict execution — success continues the episode, while failure triggers immediate termination.

## Method

### Overall Architecture

SSE comprises three core components: Frontier Experience Replay (FER), a decoupled exploration policy, and failure-aware path optimization.

### 1. Frontier Experience Replay (FER)

FER categorizes high-level experiences into three types:

$$\mathcal{B}_F^h = \begin{cases} (s_t, g, \tilde{g}_t, \sum_{j=t}^{t'-1} r_j, s_{t'}) & \text{(success)} \\ (s_t, g, \tilde{g}_t, 0, s_T) & \text{(failure termination)} \\ (s_t, g, \text{wp}_{\text{final}}, \sum_{j=t}^{t_{\text{wp}}-1} r_j, s_{t_{\text{wp}}}) & \text{(partial success)} \end{cases}$$

- **Success**: The low-level policy successfully reaches the subgoal; the full return is recorded.
- **Failure termination**: The subgoal is unreachable ($\|\phi(s_{t'}) - \tilde{g}_t\| \geq \lambda$); the return is set to 0, the next state is set to the terminal state $s_T$, and the episode is immediately truncated.
- **Partial success**: Upon failure, the last successfully reached waypoint $\text{wp}_{\text{final}}$ is recorded.

### 2. Decoupled Exploration Policy

Two policies are maintained: an exploitation policy $\pi^h$ and an exploration policy $\pi^{\text{exp}}$.

Exploitation policy ($\epsilon$-greedy):
$$\pi^h(\tilde{g}_t | s_t, g) = \begin{cases} \arg\max_{\tilde{g}} Q^h(s_t, \tilde{g}, g) & \text{with probability } 1-\epsilon \\ \text{Uniform}(\mathcal{G}) & \text{with probability } \epsilon \end{cases}$$

Exploration policy:
$$\pi^{\text{exp}}(\tilde{g}_t | s_t, g) = \begin{cases} g & \text{with probability } 1/3 \\ \tilde{g}_{\max,t} & \text{with probability } 1/3 \\ \tilde{g}_{\text{novel}} \sim \text{Uniform}(V_{\text{novel}}) & \text{with probability } 1/3 \end{cases}$$

The two policies are mixed at ratio $\eta : (1-\eta)$.

### 3. Failure-Aware Path Optimization

Edge costs on graph $G = (V, E)$ are adjusted so that Dijkstra's algorithm avoids regions with frequent failures:

$$\tilde{d}(v_1 \to v_2) = d(v_1 \to v_2) \times \max(1, c_{\text{dist}} \cdot \text{ratio}_{\text{fail}}(v_2))$$

where $\text{ratio}_{\text{fail}}$ denotes the failure ratio of the target node.

### Two Implementation Variants

- **SSE (Grid)**: A grid-based discretization method suited for 2D/3D goal spaces.
- **SSE (Model)**: A neural network-based method scalable to high-dimensional goal spaces.

## Key Experimental Results

### Main Results: 9 Long-Horizon Tasks (5 seeds)

| Environment | HIRO | HRAC | HIGL | DHRL | BEAG | NGTE | SSE |
|---|---|---|---|---|---|---|---|
| U-maze | ✗ | ✗ | △ | △ | ✓ | ✓ | **✓✓** |
| π-maze | ✗ | ✗ | △ | △ | ✓ | ✓ | **✓✓** |
| AntMazeComplex | ✗ | ✗ | ✗ | △ | ✓ | △ | **✓✓** |
| AntMazeBottleneck | ✗ | ✗ | ✗ | ✗ | △ | ✗ | **✓✓** |
| AntKeyChest | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓✓** |
| AntDoubleKeyChest | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓✓** |

(✓✓ = high success rate with fast convergence, ✓ = success, △ = partial success, ✗ = failure)

### Ablation Study (AntDoubleKeyChest)

| Variant | Performance |
|---|---|
| SSE (full) | ✓ (solved at ~3M steps) |
| SSE + FPS (grid replaced by FPS) | Succeeds but converges slowly |
| SSE + HER (FER replaced by HER) | **Complete failure** |
| SSE w/o $\mathcal{B}_F^h$ (no FER) | **Complete failure** |
| SSE w/o $\pi^{\text{exp}}$ (no exploration policy) | Significant degradation |
| SSE w/o path optimization | Moderate degradation |

### Key Findings

- FER is the critical component — removing FER or substituting HER both lead to complete failure.
- SSE enables the agent to reach any reachable position on the map in **a single high-level step**.
- AntDoubleKeyChest is solved in only **3 high-level steps** (collecting two keys and reaching the goal).

## Highlights & Insights

1. **Mandatory subgoal completion**: Enforcing that each subgoal must be reached before proceeding fundamentally reduces the number of high-level decisions.
2. **Elegant design of FER**: The fine-grained partitioning of experiences into success, failure, and partial success precisely identifies the reachability boundary.
3. **No curriculum learning required**: SSE automatically discovers the correct execution sequence of subgoals.
4. **Computational efficiency**: Immediate episode termination upon failure avoids long, uninformative trajectories, accelerating practical iteration.

## Limitations & Future Work

- Additional hyperparameters are introduced ($\eta$, $c_{\text{dist}}$, $d_\mathcal{G}$), though ablation studies demonstrate stable effective ranges.
- The method assumes that the goal space $\mathcal{G}$ is known — a standard assumption in many environments.
- The grid-based variant is limited to low-dimensional goal spaces.
- Evaluation is conducted primarily in fixed-goal settings.

## Related Work & Insights

- **Goal-conditioned RL**: HER, UVFA, prioritized goal sampling.
- **Graph-planning HRL**: HIGL, DHRL, BEAG, NGTE.
- **Frontier exploration**: Unlike conventional frontier exploration (boundary states), the "frontier" in SSE refers to the success/failure boundary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of strict subgoal execution is simple yet highly effective.
- **Technical Depth**: ⭐⭐⭐⭐ — FER is elegantly designed, with well-motivated tripartite experience categorization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Nine environments, comprehensive comparison against 7 baselines, and detailed ablations.
- **Value**: ⭐⭐⭐⭐ — Substantially outperforms existing methods on complex long-horizon tasks.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](../../AAAI2026/reinforcement_learning/manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](../../AAAI2026/reinforcement_learning/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)

<!-- RELATED:END -->
