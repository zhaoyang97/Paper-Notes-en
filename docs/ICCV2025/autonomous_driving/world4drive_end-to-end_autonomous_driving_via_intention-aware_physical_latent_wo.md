---
title: >-
  [Paper Note] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model
description: >-
  [Autonomous Driving] World4Drive constructs an intention-aware latent world model that leverages spatial-semantic priors from visual foundation models to achieve annotation-free end-to-end planning…
tags:
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 44e5986a2cc2f304
---

# World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2507.00603](https://arxiv.org/abs/2507.00603)
- **Code**: [GitHub](https://github.com/ucaszyp/World4Drive)
- **Area**: Autonomous Driving
- **Keywords**: End-to-end autonomous driving, world model, multimodal intention, visual foundation model, self-supervised learning

## TL;DR

World4Drive constructs an intention-aware latent world model that leverages spatial-semantic priors from visual foundation models to achieve annotation-free end-to-end planning, reducing L2 error by 18.1% and collision rate by 46.7%.

## Background & Motivation

End-to-end autonomous driving generates planning trajectories directly from raw sensor data, but faces the following challenges:

**High annotation cost for perception**: Methods such as UniAD, VAD, and SparseDrive rely on expensive perception annotations including 3D bounding boxes and HD maps, limiting scalability.

**Insufficient unimodal latent features**: Prior methods such as LAW extract unimodal latent features from raw images for self-supervised learning, but struggle to capture the spatial-semantic information of the physical world and multimodal driving intentions, resulting in slow training convergence and suboptimal performance. World4Drive achieves 3.75× faster training convergence compared to LAW.

**Uncertainty in multimodal intentions**: In complex scenarios such as intersections, multiple driving behaviors (left turn, straight, right turn) are all plausible choices, necessitating the modeling of intention uncertainty and the evaluation of future states under different intentions.

## Method

### Overall Architecture

World4Drive consists of two core modules: (1) a driving world encoding module that extracts driving intentions and physical world latent representations; and (2) an intention-aware world model that predicts future latent representations under multimodal intentions and scores multimodal planning trajectories.

### Intention Encoder

Given a trajectory vocabulary $\mathcal{V} \in \mathbb{R}^{N \times S \times 2}$ (N=8192 trajectories, S waypoints), intention points $P_I \in \mathbb{R}^{3 \times K \times 2}$ (3 command types × K=6 intentions) are obtained via K-means clustering. After sinusoidal positional encoding, intention-aware planning queries are produced through self-attention:

$$Q_{plan} = \text{SelfAttention}(Q_{ego} + Q_I)$$

### Physical World Latent Encoding

The **context encoder** comprises two key components:

- **Semantic understanding**: Grounded-SAM generates pseudo semantic labels $S_t = \text{GroundedSAM}(F_t)$, and semantic understanding is enhanced via cross-entropy loss $\mathcal{L}_{sem}$.
- **3D spatial encoding**: The metric depth model Metric3D v2 estimates multi-view depth maps $D_t$; forward projection yields 3D position maps $P_t$, from which positional embeddings are obtained via sinusoidal encoding and MLP: $E_t = \text{MLP}(\text{SPE}(P_t))$.

**Temporal aggregation**: Historical information is integrated via cross-attention: $L_t = \text{CrossAttention}(\hat{F_t}, \hat{F}_{t-1})$.

### Intention-aware World Model

**Action encoding**: The planning queries aggregate scene context to generate multimodal trajectories $T = \{T^1, \ldots, T^K\} \in \mathbb{R}^{K \times S \times 2}$, from which intention-aware action tokens $A \in \mathbb{R}^{K \times D}$ are obtained via MLP.

**World model prediction**: Future latent representations under different intentions are predicted using learnable queries and multi-layer cross-attention:

$$L_{t+n} = \text{CrossAttention}(Q_{future}, \text{Concat}(A, L))$$

### World Model Selector

The feature distance between each predicted future latent representation and the actual future latent representation is computed; the intention $j$ with the minimum distance is selected. The corresponding distance serves as the reconstruction loss $\mathcal{L}_{recon}$, and trajectory $T^j$ is adopted as the final plan. A ScoreNet is also trained to predict scores via Focal Loss: $\mathbb{S} = \text{Softmax}(\mathcal{C}(L_{t+n}))$.

### Total Loss

$$\mathcal{L} = 0.2\mathcal{L}_{sem} + 0.2\mathcal{L}_{recon} + 0.5\mathcal{L}_{score} + 1.0\mathcal{L}_{traj}$$

## Key Experimental Results

### nuScenes Open-loop Planning

| Method | Avg L2(m)↓ | Avg Collision(%)↓ | Annotations Required |
|--------|-----------|-------------------|----------------------|
| UniAD | 1.03 | 0.31 | ✓ |
| VAD | 0.72 | 0.23 | ✓ |
| GenAD | 0.52 | 0.19 | ✓ |
| LAW (Perception-based) | 0.49 | 0.19 | ✓ |
| BEV-Planner | 0.55 | 0.59 | ✗ |
| LAW (Perception-free) | 0.61 | 0.30 | ✗ |
| **World4Drive** | **0.50** | **0.16** | **✗** |

### NavSim Closed-loop Planning

| Method | PDMS↑ | NC↑ | DAC↑ | EP↑ |
|--------|-------|-----|------|-----|
| UniAD | 83.4 | 97.8 | 91.9 | 78.8 |
| LAW (Perception-free) | 83.8 | 97.2 | 93.3 | 78.8 |
| DiffusionDrive | 88.1 | 98.2 | 96.2 | 82.2 |
| **World4Drive** | **85.1** | 97.4 | **94.3** | **79.9** |

### Ablation Study

| Depth | Semantics | World Model | Intention | Avg L2 | Collision |
|-------|-----------|-------------|-----------|--------|-----------|
| ✗ | ✗ | ✓ | ✗ | 0.61 | 0.30 |
| ✗ | ✗ | ✓ | ✓ | 0.55 | 0.25 |
| ✓ | ✗ | ✓ | ✓ | 0.51 | 0.29 |
| ✓ | ✓ | ✗ | ✗ | 0.49 | 0.26 |
| ✓ | ✓ | ✗ | ✓ | 0.61 | 0.36 |
| ✓ | ✓ | ✓ | ✓ | **0.50** | **0.16** |

### Key Findings

1. **Annotation-free SOTA**: Achieves state-of-the-art performance without perception annotations, with a collision rate lower than annotation-dependent methods.
2. **Intention and world model are mutually indispensable**: Intention modeling alone without the world model leads to performance degradation (ablation row 5); the world model provides the capacity to evaluate planning plausibility under different intentions.
3. **Semantic priors reduce collisions**: Semantic priors substantially lower the collision rate, indicating enhanced understanding of obstacles.
4. **Robustness under adverse conditions**: Nighttime collision rate decreases by 63.7% and rainy-weather collision rate by 68.8%, attributed to the robustness of high-dimensional semantic information from visual foundation models against photometric inconsistency.

## Highlights & Insights

1. **Simulating human decision-making**: The model selects the optimal action by "imagining" future world states under different driving intentions, analogous to human driver decision processes.
2. **Effective exploitation of visual foundation models**: Grounded-SAM provides semantic priors and Metric3D provides spatial priors, eliminating the need for manual annotations.
3. **Coupled intention–world model design**: Intentions provide multimodal planning candidates, while the world model evaluates the plausibility of each candidate.
4. **3.75× convergence acceleration**: Attributed to the introduction of spatial-semantic priors.

## Limitations & Future Work

1. Reliance on pretrained visual foundation models (Grounded-SAM, Metric3D) introduces additional preprocessing overhead.
2. Closed-loop performance still lags behind LiDAR-based DiffusionDrive.
3. Future world prediction is performed solely in latent space, limiting interpretability.

## Related Work & Insights

- **End-to-end driving**: UniAD, VAD, SparseDrive, GenAD, DiffusionDrive
- **Driving world models**: DriveDreamer, Drive-WM, LAW, VaVAM
- **Intention modeling**: VADv2, Hydra-MDP probabilistic planning

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The intention-aware world model selector design is novel
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Physical latent encoding, intention encoding, and world model selector are elaborately designed
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Open-loop and closed-loop evaluation, multi-dimensional ablations, weather/lighting/maneuver analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-motivated ablation design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[ICCV 2025\] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
