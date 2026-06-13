---
title: >-
  [Paper Note] Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning
description: >-
  [ICML 2026][LLM Evaluation][Credit Assignment] GraphGPO is proposed to aggregate all rollout trajectories into a unified state transition graph…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Credit Assignment"
  - "Graph-Based Policy Optimization"
  - "Multi-turn Agent Tasks"
  - "State Transition Graph"
  - "Critic-free RL"
date: 2026-05-08
content_hash: 50edc2553fb32d6a
---

# Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.26684](https://arxiv.org/abs/2605.26684)  
**Code**: [https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO](https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO)  
**Area**: Reinforcement Learning  
**Keywords**: Credit Assignment, Graph-Based Policy Optimization, Multi-turn Agent Tasks, State Transition Graph, Critic-free RL  

## TL;DR

GraphGPO is proposed to aggregate all rollout trajectories into a unified state transition graph, utilizing global shortest-path information to calculate distance-based advantages for each step. This achieves finer-grained credit assignment than trajectory-level attribution and significantly outperforms GRPO and GiGPO on ALFWorld, WebShop, and Sokoban.

## Background & Motivation

**Background**: Group-based reinforcement learning methods (such as GRPO) have achieved significant success in LLM post-training. Their core advantage is discarding resource-intensive critic models and relying solely on verifiable rewards and group statistics to estimate advantages. Several recent works have extended GRPO to multi-turn agent tasks.

**Limitations of Prior Work**: Credit assignment in GRPO and its variants essentially relies on trajectory-level attribution—all steps in successful trajectories receive positive credit, while all steps in failed trajectories are penalized. However, in multi-turn tasks, this attribution suffers from severe misalignment: approximately 22% of steps in failed trajectories actually advance task goals, whereas about 65% of steps in successful trajectories do not effectively advance the task. Redundant steps are wrongly rewarded, and valuable failed steps are wrongly penalized.

**Key Challenge**: Trajectory-level success/failure signals are too coarse-grained to reflect the true contribution of intermediate steps to task goals. Even though GiGPO introduced step-level advantages, its step-level reward $R^S = \lambda^{T-i} R(\boldsymbol{\tau})$ still depends on the final trajectory result $R(\boldsymbol{\tau})$, failing to truly decouple from trajectory-level attribution.

**Goal**: To design a step-level credit assignment method base entirely on global state structures without requiring additional critic models or introducing significant computational overhead.

**Key Insight**: If states from all rollout trajectories are merged into a directed state transition graph, graph connectivity can be used to determine how far each state is from the target. This allows for assigning "distance-reduction" based rewards to each step, which is completely independent of the final outcome of the trajectory containing that step.

**Core Idea**: Aggregate all rollout trajectories into a unified state transition graph, define step-level rewards using shortest-path distances, and compute advantages using group statistics of homologous edges on the graph.

## Method

### Overall Architecture

The pipeline of GraphGPO consists of three steps: (1) Aggregating $M$ rollout trajectories of the same task into a directed state transition graph $\mathcal{G} = (\mathcal{S}, \mathcal{E})$; (2) Using the Dijkstra algorithm on the graph to compute the shortest distance $d(s)$ from each state to the target state $s_{\text{succ}}$; (3) Calculating graph-level step rewards and advantages based on distances for each edge, finally combining them with trajectory-level advantages for PPO-style policy optimization.

### Key Designs

1.  **Aggregating State Transition Graph**:

    All states from $M$ trajectories are treated as nodes, and all transitions as directed edges, with identical states merged into a single node. The node set $\mathcal{S} = \bigcup_{m,t} \{s_t^m\}$, and an edge $(s, \boldsymbol{a}, s', c(s,\boldsymbol{a})) \in \mathcal{E}$ represents executing action $\boldsymbol{a}$ at state $s$ to transition to $s'$ with cost $c$. This allows the explicit representation of state sharing and path crossing across different trajectories—for instance, the first half of a failed trajectory might connect to the second half of a successful trajectory through a shared state.

2.  **Shortest-Path Based Step Rewards**:

    For each state $s$, the shortest distance to the target is calculated recursively: $d(s) = \min_{(s,a,s',c) \in \mathcal{E}} (c(s,\boldsymbol{a}) + d(s'))$, where $d(s_{\text{succ}})=0$, and $d(s)=d_{\max}+1$ for unreachable states. Then, the graph-level step reward is defined as $R^G(s, \boldsymbol{a}, s') = r_{\text{succ}} \cdot \omega^{d(s') + c(s,\boldsymbol{a})}$, where $\omega \in (0,1)$ is a distance discount factor. This implies that transitions closer to the target receive higher rewards, independent of the final success or failure of the trajectory.

3.  **Graph-level Advantage Estimation and Joint Optimization**:

    All outgoing edges from the same starting state $s$ are grouped together as $G^G(s)$, and the normalized advantage $A^G = (R^G - \mu) / \sigma$ is calculated within the group. When there is only one edge in a group, $A^G = 0$, so it must be combined with the trajectory-level advantage: $A(s,\boldsymbol{a},s') = \beta^G A^G + \beta^E A^E(\boldsymbol{\tau})$. Finally, policy updates are performed using the PPO clipped objective with KL penalty. The authors prove that the graph-level advantage possesses monotonicity (greater distance reduction leads to higher advantage) and variance reduction properties (conditional variance does not exceed trajectory-level feedback).

## Key Experimental Results

| Benchmark | Model | GRPO | GiGPO | GraphGPO | Gain (vs GRPO) |
|-----------|-------|------|-------|----------|----------------|
| ALFWorld | Qwen2.5-1.5B | 77.86% | 90.88% | **92.71%** | +14.85% |
| ALFWorld | Qwen2.5-7B | 83.33% | 94.27% | **95.31%** | +11.98% |
| WebShop (Succ.) | Qwen2.5-1.5B | 71.35% | 73.83% | **78.65%** | +7.30% |
| WebShop (Succ.) | Qwen2.5-7B | 75.00% | 78.38% | **80.31%** | +5.31% |
| Sokoban 6×6 | Qwen2.5-VL-3B | 67.1% | 76.92% | **86.98%** | +19.88% |

| Ablation Study/Feature | Conclusion |
|-------------------------|------------|
| Removing $A^E$ | Performance drops for both methods, but GraphGPO still outperforms GiGPO by 20.57% on Sokoban |
| Dynamic Sampling (+DS) | GraphGPO + DS reaches **98.43%** on ALFWorld and **85.68%** on WebShop |
| Computational Overhead | Graph construction 0.108s + advantage calculation 0.025s, accounting for only 0.04% of total time per round |
| Training Dynamics | Convergence speed is significantly faster in the early stages, especially when signals are more effective at low success rates |

## Highlights & Insights

-   **Value Extraction from Failed Trajectories**: Through the graph structure, effective steps in failed trajectories can obtain positive advantages (because they actually shorten the distance to the target), which traditional trajectory-level attribution cannot achieve.
-   **Natural Punishment of Redundant/Cyclic Behaviors**: Steps that form cycles in the graph inevitably increase distance ($d(s_{41}) > d(s_2)$) and thus naturally receive lower advantages without additional penalty mechanisms.
-   **Nearly Zero Additional Overhead**: It only requires executing a Dijkstra shortest path search once per training iteration. The complexity is $O((|\mathcal{V}|+|\mathcal{E}|) \log |\mathcal{V}|)$, taking 0.133s compared to a total time of 291s.
-   **Theoretical Guarantees**: Proofs for advantage monotonicity (Proposition 4.1) and conditional variance reduction (Proposition 4.2) provide analytical support for the effectiveness of the method.

## Limitations & Future Work

-   **Deterministic Environment Assumption**: Merging states on a graph assumes the environment is deterministic (executing the same action in the same state leads to the same successor). The effectiveness of state merging may decrease significantly in stochastic environments.
-   **State Definition Reliance on Manual Design**: It is necessary to define what constitutes the "same state" (the paper uses deterministic parts of environmental observations). Determining state equivalence in open-domain tasks (such as free-text dialogue) may be difficult.
-   **Cost Function $c(s,\boldsymbol{a})$ Simplified to 1**: In experiments, all transition costs were set to 1. The effects of non-uniform costs (such as actual time or monetary costs of tool calls) have not been explored.
-   **Graph Construction within a Single Training Iteration**: The graph for each iteration is based only on current rollout data and does not accumulate historical experience across iterations.

## Related Work & Insights

-   **GRPO** (Shao et al., 2024): The foundation of group-level RL; GraphGPO retains its core advantage of being critic-free.
-   **GiGPO** (Feng et al., 2025b): Introduces step-level grouping but still relies on trajectory outcomes; GraphGPO achieves complete decoupling through the graph structure.
-   **PPO** (Schulman et al., 2017): The policy optimization objective of GraphGPO follows the PPO clipped objective framework.
-   Insight: The graph structure perspective offers a new path for RL credit assignment. Similar ideas could be applied to Chain-of-Thought scenarios such as code generation and mathematical reasoning.

## Rating

-   Novelty: ⭐⭐⭐⭐ — The idea of aggregating trajectories into a state transition graph for credit assignment is novel and intuitively clear.
-   Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both text and vision agent tasks, includes complete ablations, and provides a thorough overhead analysis.
-   Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive illustrations, and close integration of theory and experiments.
-   Value: ⭐⭐⭐⭐ — Provides a practical and low-cost improvement for credit assignment in LLM agent RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/llm_evaluation/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
