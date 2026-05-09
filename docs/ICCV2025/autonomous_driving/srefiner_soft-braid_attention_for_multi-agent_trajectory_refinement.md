---
title: >-
  [Paper Note] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement
description: >-
  [ICCV 2025][Autonomous Driving][trajectory prediction] This paper proposes Soft-Braid Attention, which explicitly models spatiotemporal topological relationships between trajectories and between trajectories and lanes via "soft crossing points" to guide multi-agent trajectory refinement. The method achieves significant improvements over four baseline methods on both Argoverse v2 and INTERACTION datasets, establishing a new state of the art for trajectory refinement.
tags:
  - ICCV 2025
  - Autonomous Driving
  - trajectory prediction
  - trajectory refinement
  - topological structure
  - braid theory
  - multi-agent interaction
date: 2026-05-08
content_hash: 4969f5e9f61440e0
---

# SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement

**Conference**: ICCV 2025
**arXiv**: [2507.04263](https://arxiv.org/abs/2507.04263)
**Code**: [github.com/Liwen-Xiao/SRefiner](https://github.com/Liwen-Xiao/SRefiner)
**Area**: Autonomous Driving
**Keywords**: trajectory prediction, trajectory refinement, topological structure, braid theory, multi-agent interaction

## TL;DR

This paper proposes Soft-Braid Attention, which explicitly models spatiotemporal topological relationships between trajectories and between trajectories and lanes via "soft crossing points" to guide multi-agent trajectory refinement. The method achieves significant improvements over four baseline methods on both Argoverse v2 and INTERACTION datasets, establishing a new state of the art for trajectory refinement.

## Background & Motivation

### Problem Definition

Multi-agent future trajectory prediction is a core task for safe decision-making in autonomous driving. Trajectory refinement, as a post-processing strategy, takes coarse trajectory predictions from a base model as input and outputs more accurate and plausible trajectories by modeling fine-grained interactions between trajectories and the scene.

### Limitations of Prior Work

The central limitation of existing trajectory refinement methods lies in **implicit interaction modeling**, which fails to exploit the topological structure of trajectories:

**QCNet**: Encodes coarse predicted trajectories as anchor queries and fuses them with scene context to predict trajectory offsets, but interaction remains implicit.

**R-Pred**: Applies local attention mechanisms to refine predictions among neighboring agents, but lacks explicit topological guidance.

**MTR++**: Uses intention points to guide inter-agent information exchange, yet still relies on implicit relational learning.

**SmartRefine**: Employs a scene-adaptive refinement strategy that dynamically adjusts the number of iterations and attention range, but does not account for topological relationships.

Braid theory has been proven effective in robotics (e.g., tethered drone path planning and grasping). Its direct application to trajectory prediction (BeTop) is effective but suffers from three key limitations:

- **Non-crossing interactions are ignored**: e.g., a red vehicle decelerating to yield to a blue vehicle — the trajectories do not cross, yet their behaviors are logically coupled.
- **Temporal dynamics are neglected**: braid topology only captures spatial crossing relationships and does not encode dynamic interaction over time.
- **Limited expressiveness**: it can only answer whether trajectories interact, not how they interact.

### Core Problem

There is a need for an **explicit and expressive** topological representation to guide trajectory refinement. The key insight is that even when two trajectories do not cross in space, the motion states and spatial relationships at the point of closest approach (the "soft crossing point") still encode rich interaction information.

## Method

### Overall Architecture

SRefiner is a multi-iteration, multi-agent trajectory refinement framework consisting of three core modules:
1. **Trajectory-Trajectory Soft-Braid Attention**: models soft-braid topology between trajectories.
2. **Trajectory-Lane Soft-Braid Attention**: models soft-braid topology between trajectories and lanes.
3. **Iterative Refinement with Topology Update**: at each iteration, topological information is recomputed from the trajectories refined in the previous round.

### Key Designs

#### 1. **Soft-Braid Topology**

- **Function**: Defines a "soft crossing point" for any two trajectories $y_i$ and $y_j$, encodes motion states and spatial relationships at that point, and constructs an expressive topological descriptor.
- **Mechanism**:

  The soft crossing point is defined as the time step at which the two trajectories are closest:
  $$t_{ij} = \arg\min_t \|y_i(t) - y_j(t)\|$$
  $$P_{i_j} = y_i(t_{ij}), \quad P_{j_i} = y_j(t_{ij})$$

  The soft-braid topology encoding contains six dimensions:
  $$\tilde{\sigma}_{i \leftarrow j} = [\dot{y}_i^{(i)}(t_{ij}), \dot{y}_j^{(i)}(t_{ij}), \ddot{y}_i^{(i)}(t_{ij}), \ddot{y}_j^{(i)}(t_{ij}), d_{ij}, \theta_{ij}^{(i)}]$$

  where $\dot{y}$ and $\ddot{y}$ denote velocity and acceleration in agent $i$'s local coordinate frame, respectively, $d_{ij}$ is the distance between soft crossing points, and $\theta_{ij}$ is the directional angle of the connecting vector.

- **Design Motivation**:
    - Unlike braid topology (which yields only a binary crossing/non-crossing relation), soft-braid topology establishes topological connections for **all** trajectory pairs.
    - Representation in the local coordinate frame ensures rotational invariance.
    - Encoding velocity and acceleration captures temporal dynamics (e.g., acceleration, deceleration, and yielding behaviors).
    - Distance and angle characterize the fine-grained spatial relationship.

#### 2. **Trajectory-Trajectory Soft-Braid Attention**

- **Function**: Uses soft-braid topological information to guide inter-trajectory feature fusion.
- **Mechanism**:

  Information exchange is realized via multi-head cross-attention, with topological information injected into the Keys and Values:
  $$F_i = \text{MHCA}(Q: F_i, \; K: \{F_j + \varphi(\tilde{\sigma}_{i \leftarrow j})\}_{j \in \Omega(i)}, \; V: \{F_j + \varphi(\tilde{\sigma}_{i \leftarrow j})\}_{j \in \Omega(i)})$$

  where $\varphi(\cdot)$ is a 3-layer MLP and $\Omega(i) = \{j | d_{ij} \leq \tau_a\}$ is the neighborhood set with $\tau_a = 50$m.

- **Design Motivation**: By adding topological features to the Keys and Values, attention weight computation naturally incorporates topological relationships — trajectory pairs that are spatially close and kinematically correlated receive higher attention weights.

#### 3. **Trajectory-Lane Soft-Braid Attention**

- **Function**: Extends soft-braid topology to model interactions between trajectories and HD map lane centerlines.
- **Mechanism**:

  The soft crossing point for trajectory $y_i$ and lane $L_k$ is defined as the point on $y_i$ closest to $L_k$:
  $$t_{ik} = \arg\min_t \|y_i(t) - L_k\|$$

  The trajectory-lane topological feature is:
  $$\tilde{\lambda}_{i \leftarrow k} = [\dot{y}_i^{(i)}(t_{ik}), \ddot{y}_i^{(i)}(t_{ik}), d_{ik}, \theta_{ik}^{(i)}]$$

  Feature fusion is likewise performed via MHCA, with neighborhood threshold $\tau_l = 10$m.

- **Design Motivation**: The relationship between a trajectory and surrounding lanes is critical for ensuring physical plausibility (e.g., staying within the drivable area, not intruding into opposing lanes). Encoding the agent's motion state at the closest point captures dynamic information such as whether the agent is approaching or departing from a lane.

#### 4. **Iterative Refinement with Progressive Topology Update**

- **Function**: Refines trajectories over multiple iterations, recomputing topology from the refined trajectories of the previous round at each step.
- **Mechanism**:
  $$\tilde{B}_{l-1}, \tilde{B}'_{l-1} = \mathcal{S}(Y_{l-1}, L)$$
  $$F_l = \text{SoftBraidAttn}(F_{l-1}, L, \tilde{B}_{l-1}, \tilde{B}'_{l-1})$$
  $$Y_l = \varphi(F_l) + Y_{l-1}$$

  The default number of iterations is $I=3$.

- **Design Motivation**: The topology derived from the initial coarse trajectories may be inaccurate. Recomputing topology from refined trajectories progressively improves topological quality, forming a positive feedback loop.

### Loss & Training

- **Joint Winner-Takes-All (WTA) Loss**: selects the mode with the minimum joint displacement error relative to ground truth among $K$ modes for supervision.
- Optimal mode selection: $k_l = \arg\min_{k \in [1,K]} \frac{1}{N}\sum_{i=1}^N \|Y_{l,i,k} - Y_{\text{gt},i}\|$
- **Huber Loss** supervises the output of each iteration.
- Total loss is the average over all iterations: $\mathcal{L} = \frac{1}{I}\sum_{l=1}^I \mathcal{L}_l$

## Key Experimental Results

### Main Results

Improvements over four baseline methods on Argoverse v2 and INTERACTION:

| Dataset | Baseline | Metric | Original | +SRefiner | Gain |
|--------|------|------|--------|-----------|------|
| AV2 Val | Forecast-MAE | avgMinFDE↓ | 1.642 | 1.477 | −10.1% |
| AV2 Val | FJMP | avgMinFDE↓ | 1.920 | 1.736 | −9.6% |
| AV2 Test | Forecast-MAE | avgMinFDE↓ | 1.679 | 1.521 | −9.4% |
| INTER Test | AutoBots | minJointFDE↓ | 1.015 | 0.906 | −10.7% |
| INTER Test | FJMP | minJointFDE↓ | 0.945 | 0.867 | −8.3% |
| INTER Test | HPNet | minJointFDE↓ | 0.823 | 0.797 | −3.2% |

Comparison with other refinement methods (Argoverse v2, baseline: Forecast-MAE):

| Refinement Method | avgMinFDE↓ | avgMinADE↓ | actorMR↓ | Latency |
|----------|-----------|-----------|----------|------|
| Baseline | 1.642 | 0.717 | 0.194 | – |
| DCMS | 1.601 | 0.702 | 0.190 | 5ms |
| R-Pred | 1.554 | 0.683 | 0.187 | 12ms |
| QCNet | 1.520 | 0.674 | 0.185 | 58ms |
| MTR++ | 1.495 | 0.670 | 0.183 | 54ms |
| **SRefiner** | **1.477** | **0.658** | **0.183** | 28ms |

### Ablation Study

| Configuration | avgMinFDE↓ | avgMinADE↓ | actorMR↓ |
|------|-----------|-----------|----------|
| Baseline (no refinement) | 1.642 | 0.717 | 0.194 |
| +Traj-Traj + Traj-Lane (no topology update) | 1.522 | 0.673 | 0.186 |
| +Traj-Traj + topology update (no Traj-Lane) | 1.497 | 0.670 | 0.183 |
| +Traj-Lane + topology update (no Traj-Traj) | 1.514 | 0.678 | 0.184 |
| **Full model** | **1.477** | **0.658** | **0.183** |

Comparison of topology types:

| Topology Type | avgMinFDE↓ |
|----------|-----------|
| No topology | 1.530 |
| Braid Topology (BeTop) | 1.512 |
| Soft-Braid (Traj-Traj only) | 1.497 |
| **Soft-Braid (full)** | **1.477** |

### Key Findings

1. SRefiner achieves consistent and significant improvements across all four baselines and both datasets, demonstrating strong generalization.
2. Soft-braid topology outperforms braid topology (1.477 vs. 1.512), validating the effectiveness of the soft crossing point design.
3. SRefiner achieves state-of-the-art accuracy with only 28ms latency, outperforming QCNet (58ms) and MTR++ (54ms).
4. The progressive topology update strategy contributes substantially — removing it degrades avgMinFDE from 1.477 to 1.522.
5. Qualitative visualizations show that SRefiner effectively reduces colliding trajectories and trajectories that deviate from the drivable area.

## Highlights & Insights

1. **From braid theory to deep learning**: Incorporating algebraic topology concepts into trajectory prediction is an exemplary case of cross-disciplinary integration.
2. **Elegance of the soft crossing point**: Relaxing the discrete crossing/non-crossing binary relation into a continuous distance-and-motion-state representation preserves the structured prior of topology while substantially enhancing expressiveness.
3. **Plug-and-play design**: SRefiner can be seamlessly integrated into existing multi-agent trajectory prediction pipelines.
4. **Efficiency advantage**: Compared to SmartRefine, which performs separate inference per agent ($24 \times N$ ms), SRefiner refines all agents simultaneously in 28ms.

## Limitations & Future Work

1. **Vehicles only**: Heterogeneous traffic participants such as pedestrians and cyclists are not addressed.
2. **Topology computation overhead**: Soft crossing points for all trajectory pairs must be recomputed at each iteration, resulting in $O(N^2 T)$ complexity.
3. **Fixed neighborhood thresholds**: $\tau_a = 50$m and $\tau_l = 10$m are fixed hyperparameters with no scene-adaptive adjustment.
4. **HD map dependency**: The Trajectory-Lane module requires high-definition map information and is not applicable in map-free settings.
5. **Multimodal uncertainty not considered**: The soft-braid topological representation is fixed across different prediction modes.

## Related Work & Insights

- **vs. BeTop**: BeTop directly applies the binary crossing relation from braid topology; SRefiner proposes a more expressive continuous soft-braid topology.
- **vs. MTR++**: MTR++ implicitly models interactions via intention points; SRefiner guides interaction through explicit topological features.
- The transfer of braid theory from robotics (tethered drone path planning) to autonomous driving trajectory prediction demonstrates the potential of cross-domain technology transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Extending braid theory into "soft-braid" form for trajectory refinement is conceptually novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, four baselines, five refinement method comparisons, and detailed ablations; however, downstream task evaluation in real-world scenarios is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation chain from braid theory to soft-braid is clear and well-illustrated.
- **Value**: ⭐⭐⭐⭐ — Provides a new topological perspective for trajectory refinement; the plug-and-play design offers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems](../../NeurIPS2025/autonomous_driving/regret_lower_bounds_for_decentralized_multi-agent_stochastic_shortest_path_probl.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)

<!-- RELATED:END -->
