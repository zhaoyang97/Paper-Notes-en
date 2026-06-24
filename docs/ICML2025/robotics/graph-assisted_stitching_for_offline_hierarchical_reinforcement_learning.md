---
title: >-
  [Paper Note] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning
description: >-
  [ICML2025][Robotics][Offline reinforcement learning] This paper proposes the Graph-Assisted Stitching (GAS) framework, which replaces explicit high-level policy learning with graph-search-based subgoal selection. By constructing a graph through clustering in the Temporal Distance Representation (TDR) space and performing shortest-path planning, GAS enables highly efficient cross-trajectory stitching in offline HRL. It achieves a breakthrough…
tags:
  - "ICML2025"
  - "Robotics"
  - "Offline reinforcement learning"
  - "hierarchical RL"
  - "trajectory stitching"
  - "graph search"
  - "temporal distance representation"
  - "subgoal selection"
date: 2026-05-08
content_hash: 3bd201844b52f7ee
---

# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2506.07744](https://arxiv.org/abs/2506.07744)  
**Code**: [GitHub](https://github.com/qortmdgh4141/GAS)  
**Area**: Offline Hierarchical RL  
**Keywords**: Offline reinforcement learning, hierarchical RL, trajectory stitching, graph search, temporal distance representation, subgoal selection

## TL;DR

This paper proposes the Graph-Assisted Stitching (GAS) framework, which replaces explicit high-level policy learning with graph-search-based subgoal selection. By constructing a graph through clustering in the Temporal Distance Representation (TDR) space and performing shortest-path planning, GAS enables highly efficient cross-trajectory stitching in offline HRL. It achieves a breakthrough, boosting the performance on the most challenging antmaze-giant-stitch task from the Prev. SOTA of 1.0 to 88.3.

## Background & Motivation

Offline hierarchical reinforcement learning (offline HRL) tackles long-horizon, sparse-reward tasks by utilizing a high-level policy to generate subgoals and a low-level policy to execute actions. However, existing methods suffer from three major bottlenecks:

**Temporal-unaware subgoal sampling**: Most methods sample subgoal candidates at fixed step intervals along a trajectory (e.g., every $c$ steps), neglecting the actual temporal distance relationships between states, which leads to redundant or inefficient subgoals.

**Insufficient long-horizon reasoning capability**: Methods like HIQL perform well on antmaze-medium but experience a sharp decline in ultra-long horizon environments like antmaze-giant, because sparse rewards provide extremely weak learning signals for the high-level policy.

**Lack of cross-trajectory stitching capability**: Offline datasets contain multiple trajectories aiming at different goals. Ideally, a policy should stitch useful segments from various trajectories to form new paths. However, prior methods lack an effective cross-goal stitching mechanism.

Core Idea: **Since training a high-level policy is difficult, it is better to replace it with graph search**—specifically, construct a state graph in the TDR space and directly select the sequence of subgoals using the Dijkstra shortest-path algorithm.

## Method

GAS comprises four core components:

### 1. Temporal Distance Representation (TDR) Pre-training

Embed states into a latent space $\mathcal{H}$ such that the Euclidean distance between two states equals the minimum number of time steps required for the optimal policy to transition between them:

$$d^*(s, g) = \|\psi(s) - \psi(g)\|_2$$

Train the representation on offline data via an IQL-style expectile regression loss:

$$\mathcal{L} = \mathbb{E}_{(s,s',g)\sim\mathcal{D}}\left[\ell_\tau^2\left(-\mathbb{1}\{s \neq g\} + \gamma \bar{V}(s',g) - V(s,g)\right)\right]$$

where $V(s,g) =  -\|\psi(s) - \psi(g)\|_2$, and $\ell_\tau^2(x) = |\tau - \mathbb{1}(x<0)| \cdot x^2$.

### 2. TD-Aware Graph Construction

Cluster all states in the TDR space with a fixed temporal distance interval $H_{\text{TD}}$:

- The first state becomes the center of the first cluster.
- Subsequent states are assigned to the nearest cluster if their distance to its center is $\leq H_{\text{TD}}$; otherwise, a new cluster is created.
- Cluster center = mean of the states in the cluster $\rightarrow$ used as graph nodes.
- Construct edges between nodes with distance $\leq H_{\text{TD}}$.

Semantically similar states from different trajectories are aggregated into the same node, **naturally enabling cross-trajectory stitching**.

### 3. Temporal Efficiency (TE) Metric Filtering

To prevent low-quality states (from suboptimal trajectories) from contaminating the graph structure, a TE metric is defined to measure the execution efficiency of transitions at the current state:

$$\theta_{\text{TE}} = \cos\left(\psi(s_{\text{opt}}) - \psi(s_{\text{cur}}),\ \psi(s_{\text{reached}}) - \psi(s_{\text{cur}})\right)$$

where $s_{\text{opt}}$ is the optimal future state along the trajectory at a temporal distance of $H_{\text{TD}}$, and $s_{\text{reached}}$ is the state actually reached after $H_{\text{TD}}$ steps. Only highly efficient states with $\theta_{\text{TE}} \geq 0.9$ are kept for graph construction. The selected states account for only 2%–8% of the dataset, and the number of nodes is less than 1% of the dataset overall.

### 4. Task Planning and Execution

No high-level policy is required during inference:

1. Precompute the shortest distances from all graph nodes to the final goal using Dijkstra's algorithm.
2. Find reachable nodes within the $H_{\text{TD}}$ range of the current state.
3. Select $v_{\text{subgoal}} = \arg\min_{v \in \mathcal{V}_{\text{near}}} (\text{Dists}[v] + \|h_{\text{cur}} - v\|_2)$.
4. The low-level policy executes actions along the direction vector $\vec{h}_{\text{dir}} = \text{dir}(\psi(s_t), \psi(s_{\text{sub}}))$.

The low-level policy is trained using DDPG+BC loss and IQL value functions, with subgoal sampling adjusted to TDR distance rather than fixed steps.

## Key Experimental Results

### State-based Benchmarks (OGBench)

| Task | GCBC | HIQL | HHILP | **GAS** |
|------|------|------|-------|---------|
| antmaze-medium-navigate | 33.1 | 95.3 | 96.3 | **96.3** |
| antmaze-large-navigate | 23.4 | 89.9 | 86.8 | **93.2** |
| antmaze-giant-navigate | 0.0 | 67.3 | 53.1 | **77.6** |
| antmaze-medium-stitch | 43.2 | 92.0 | 96.0 | **98.1** |
| antmaze-large-stitch | 2.3 | 71.7 | 34.1 | **96.3** |
| antmaze-giant-stitch | 0.0 | 1.0 | 0.0 | **88.3** |
| antmaze-large-explore | 0.0 | 2.9 | 2.4 | **94.2** |
| scene-play | 5.4 | 40.0 | 43.4 | **73.6** |
| kitchen-partial | 69.5 | 73.1 | 66.7 | **87.3** |

### Key Findings

- **Breakthrough improvement on stitching tasks**: Performance on antmaze-giant-stitch surges from HIQL's 1.0 $\rightarrow$ **88.3** (+8730%).
- **Enormous advantage in exploratory datasets**: Performance on antmaze-large-explore improves from HIQL's 2.9 $\rightarrow$ **94.2**.
- **Equally effective in visual environments**: Reaches 55.8 on visual-antmaze-giant-stitch (compared to only 3.6 for Prev. SOTA).

### TE Filtering Ablation

| Task | Graphing with All States | With TE Filtering | Gain |
|------|-------------|----------|------|
| antmaze-giant-navigate | 63.4 | **77.6** | +14.2 |
| antmaze-giant-stitch | 75.3 | **88.3** | +13.0 |
| antmaze-large-explore | 75.4 | **94.2** | +18.8 |

## Highlights & Insights

1. **Paradigm shift**: Replacing high-level policy learning with graph search bypasses the fundamental difficulty of training high-level policies under sparse rewards.
2. **Exquisite design of the TE metric**: Simply using Cosine Similarity effectively filters out suboptimal states. Utilizing only 2%–8% of the data yields better results than using the entire dataset.
3. **Unified $H_{\text{TD}}$ parameter**: A single hyperparameter simultaneously controls the clustering interval, edge construction threshold, subgoal sampling distance, and reachable range during execution, ensuring consistency between training and testing.
4. **Validation in extreme stitching scenarios**: Achieves a quantum leap from 1.0 to 88.3 on antmaze-giant-stitch, the task that demands the strongest stitching capability.

## Limitations & Future Work

1. **Performance gap in visual environments**: Compared to state-based environments, overall performance in visual environments remains suboptimal. The authors acknowledge a lack of specialized representation learning for high-dimensional visual inputs.
2. **$H_{\text{TD}}$ requires environment-specific tuning**: The optimal value of $H_{\text{TD}}$ varies across tasks, and there lacks an adaptive selection mechanism.
3. **Static graph structure**: The graph is constructed once during inference and is not updated online based on execution feedback.
4. **Dijkstra algorithm is re-run per episode**: Although the number of nodes is <1%, latency might still become an issue in large-scale environments.

## Related Work & Insights

- **HIQL** (Park et al., 2023): Utilizes a latent space representation learned via value functions for the high-level policy. GAS inherits the TDR and low-level training implementation from HIQL.
- **HHILP** (Park et al., 2024c): Introduces TDR representation learning. Building on this, GAS replaces the high-level policy with graph search.
- **OGBench** (Park et al., 2025a): Provides a standardized evaluation benchmark containing three categories of datasets: navigate, stitch, and explore.

## Rating

- Novelty: ⭐⭐⭐⭐ — Replacing high-level policies with graph search and the TE filtering metric are core innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers locomotion/navigation/manipulation, state/visual modalities, and 5 ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with intuitive diagrams and complete mathematical formulations.
- Value: ⭐⭐⭐⭐⭐ — Achieves order-of-magnitude improvements in offline HRL stitching tasks, showing significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Value-Decomposed Offline Reinforcement Learning for Whole-Body Control](../../ICLR2026/robotics/hierarchical_value-decomposed_offline_reinforcement_learning_for_whole-body_cont.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](../../NeurIPS2025/robotics/sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[ICML 2025\] Maximum Total Correlation Reinforcement Learning](maximum_total_correlation_reinforcement_learning.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](../../ICLR2026/robotics/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICML 2025\] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning](gradual_transition_from_bellman_optimality_operator_to_bellman_operator_in_onlin.md)

</div>

<!-- RELATED:END -->
