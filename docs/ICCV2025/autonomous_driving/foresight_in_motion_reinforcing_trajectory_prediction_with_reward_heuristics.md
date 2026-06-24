---
title: >-
  [Paper Note] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics
description: >-
  [ICCV 2025][Autonomous Driving][Trajectory Prediction] This paper proposes a "First Reasoning, Then Forecasting" strategy that infers reward distributions over driving intentions via Query-centric Inverse Reinforcement Learning (QIRL), and couples this with a Bi-Mamba-enhanced DETR-style trajectory decoder to significantly improve prediction confidence and accuracy.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Trajectory Prediction"
  - "Inverse Reinforcement Learning"
  - "Intent Reasoning"
  - "Mamba"
date: 2026-05-08
content_hash: 7869f7bccc70f918
---

# Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics

**Conference**: ICCV 2025
**arXiv**: [2507.12083](https://arxiv.org/abs/2507.12083)  
**Code**: None  
**Area**: Autonomous Driving
**Keywords**: Trajectory Prediction, Inverse Reinforcement Learning, Intent Reasoning, Autonomous Driving, Mamba

## TL;DR

This paper proposes a "First Reasoning, Then Forecasting" strategy that infers reward distributions over driving intentions via Query-centric Inverse Reinforcement Learning (QIRL), and couples this with a Bi-Mamba-enhanced DETR-style trajectory decoder to significantly improve prediction confidence and accuracy.

## Background & Motivation

Trajectory prediction is a critical module in autonomous driving systems that bridges perception and planning. Existing data-driven methods primarily predict future motion through direct trajectory regression or endpoint classification, but suffer from the following issues:

**Lack of intent reasoning**: Directly predicting trajectories without explicitly reasoning about behavioral intentions leads to insufficient interpretability and reliability.

**Low prediction confidence**: Without intent priors as guidance, probability assignments in multimodal predictions are insufficiently accurate.

**Neglect of the planning perspective**: Human driving is hierarchical (first making high-level decisions such as lane changes, then executing specific maneuvers), yet prediction models rarely draw on this insight.

Core idea: Treat trajectory prediction as "planning on behalf of other agents," and leverage a reinforcement learning paradigm to reason about agent behavioral intentions, providing prior guidance for trajectory generation.

## Method

### Overall Architecture

FiM (Foresight in Motion) adopts an encoder–decoder structure comprising: (1) a query-centric scene context encoder; (2) a reward-based intent reasoner (QIRL); and (3) a Bi-Mamba-enhanced hierarchical DETR-style trajectory decoder.

### Key Designs

1. **Query-Centric Context Encoding**:

    - A 1D CNN encodes agent features $F_a \in \mathbb{R}^{N_a \times C}$, and a PointNet-like network encodes map features $F_m \in \mathbb{R}^{N_m \times C}$.
    - Learnable grid queries $Q_G \in \mathbb{R}^{H \times W \times C}$ are introduced; scene features are aggregated into spatial grid tokens via cross-attention.
    - Each grid query $Q_G^{s_i}$ corresponds to a region of resolution $d$ in the BEV plane.

2. **QIRL (Query-centric Inverse Reinforcement Learning)**:

    - **Core Idea**: Each grid cell $s_i$ serves as a state, with its corresponding query $Q_G^{s_i}$ as the contextual feature.
    - **Reward Inference**: A $1{\times}1$ CNN layer maps grid tokens to a reward distribution $R \in \mathbb{R}^{H \times W \times 1}$.
    - **Expert Demonstrations**: Future trajectories are quantized onto grids of resolution $d$ to serve as demonstrated states.
    - **MaxEnt IRL**: The log-likelihood of demonstration data is maximized under the maximum entropy principle; iterative convergence yields the reward distribution and optimal policy.
    - **Policy Rollout**: The learned policy is rolled out $L$ times ($L \gg K$), producing $L$ feasible Grid-based Reasoning Traversals (GRTs) $\Upsilon \in \mathbb{R}^{L \times \mathcal{H} \times 2}$.
    - Reasoning tokens $Q_G^\Upsilon \in \mathbb{R}^{L \times \mathcal{H} \times C}$ are extracted from grid tokens according to the GRTs.

3. **Auxiliary Spatial-Temporal Occupancy Grid Map (OGM) Prediction Head**:

    - Grid tokens $Q_G$ and rewards $R$ are fed into a U-Net architecture to predict occupancy grid maps over future $T_f$ timesteps.
    - This models future inter-agent interactions and enhances feature fusion.

4. **Bi-Mamba-Enhanced Hierarchical Trajectory Decoder**:

    - **Trajectory Proposal Generation**: Anchor-free queries $Q_P$ encode GRT reasoning features via cross-attention, generating $L$ trajectory proposals.
    - **Clustering**: K-means clustering reduces proposals to $K$ multimodal candidates $\bar{Y}$.
    - **Trajectory Refinement**: Each proposal serves as an anchor, re-encoded as $Q_T$, and context features are retrieved via a DETR-style architecture.
    - **Bi-Mamba Decoding**: Dual-mode queries $Q_M \in \mathbb{R}^{K \times 2 \times C}$ are designed with two CLS tokens appended at both ends of the trajectory query; bidirectional scanning captures forward and backward features, followed by a mode self-attention module to enhance multimodality.
    - Final trajectory: $Y = \bar{Y} + \Delta Y$

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \mathcal{L}_{IRL} + \alpha \mathcal{L}_{OGM} + \beta \mathcal{L}_{REG} + \gamma \mathcal{L}_{CLS}$$

- $\mathcal{L}_{IRL}$: MaxEnt IRL convergence loss
- $\mathcal{L}_{OGM}$: Focal BCE loss (occupancy prediction)
- $\mathcal{L}_{REG}$: Huber loss (trajectory regression) with winner-takes-all strategy
- $\mathcal{L}_{CLS}$: Max-margin loss (mode classification)

## Key Experimental Results

### Main Results

**Argoverse 1 Test Set (Single Model)**:

| Method | MR₆↓ | minADE₆↓ | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|--------|------|----------|----------|----------------|-------------|
| DenseTNT | 0.1258 | 0.8817 | 1.2815 | 1.9759 | 0.6944 |
| HiVT | 0.1267 | 0.7735 | 1.1693 | 1.8422 | 0.6729 |
| DSP | 0.1303 | 0.8194 | 1.2186 | 1.8584 | 0.6398 |
| **FiM** | **0.1250** | 0.8296 | **1.2048** | **1.8266** | **0.6218** |

**nuScenes Prediction Leaderboard**:

| Method | minADE₅↓ | MR₅↓ | minADE₁₀↓ | MR₁₀↓ |
|--------|----------|------|----------|-------|
| DeMo | 1.22 | 0.43 | 0.89 | 0.34 |
| Goal-LBP | 1.02 | 0.32 | 0.93 | 0.27 |
| UniTraj | 0.96 | 0.43 | 0.84 | 0.41 |
| **FiM** | **0.88** | **0.31** | **0.78** | **0.23** |

### Ablation Study

**Ablation of Reward-Driven Reasoning Strategy (Argoverse Validation Set)**:

| Strategy | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|----------|----------|----------------|-------------|
| Vanilla (no reasoning) | 2.185 | 2.879 | 0.694 |
| Cross-Attention replacing QIRL | 1.490 | 2.132 | 0.642 |
| **Ours (QIRL)** | **1.008** | **1.602** | **0.594** |

**Ablation of Bi-Mamba Decoder Components**:

| MLP | Bi-Mamba | Self-Attn | brier-minFDE₆↓ | Brier score↓ |
|-----|---------|-----------|----------------|-------------|
| ✓ | | | 1.720 | 0.622 |
| ✓ | ✓ | | 1.649 | 0.605 |
| ✓ | | ✓ | 1.682 | 0.617 |
| ✓ | ✓ | ✓ | **1.602** | **0.594** |

**Uni-Mamba vs. Bi-Mamba**:

| Mamba Type | minFDE₆↓ | brier-minFDE₆↓ | Brier score↓ |
|-----------|----------|----------------|-------------|
| Unidirectional | 1.034 | 1.636 | 0.603 |
| **Bidirectional** | **1.008** | **1.602** | **0.594** |

**Effect of Long-Horizon Intent Supervision (Argoverse 2 Validation Set)**:

| Variant | GRT Supervision Timesteps | minFDE₆↓ | brier-minFDE₆↓ | Brier↓ |
|---------|--------------------------|----------|----------------|--------|
| GRT-S | 30 | 0.529 | 1.147 | 0.617 |
| GRT-M | 45 | 0.530 | 1.134 | 0.604 |
| GRT-L | 60 | **0.528** | **1.131** | **0.603** |

### Key Findings

- **QIRL outperforms the cross-attention alternative by 0.53 in brier-minFDE** (2.132 vs. 1.602), validating the critical role of intent reasoning.
- **Longer-horizon intent supervision significantly improves prediction confidence**: GRT-S→GRT-L reduces Brier score from 0.617 to 0.603.
- **Bi-Mamba is more effective than Uni-Mamba**: The bidirectional scanning mechanism better integrates trajectory features.
- **FiM comprehensively surpasses existing methods on nuScenes**: minADE₅ of 0.88, substantially ahead of UniTraj (0.96).
- The OGM auxiliary head and trajectory refinement module each contribute independently, with their combination yielding the best performance.

## Highlights & Insights

- **"First Reasoning, Then Forecasting" paradigm**: This work is among the first to systematically introduce planning principles into trajectory prediction by using reward-driven intent reasoning as a prior.
- **Innovation of QIRL**: Extends traditional MaxEnt IRL, which requires structured grid environments, to vectorized representations via a query-centric paradigm, improving flexibility.
- **Application of Bi-Mamba in trajectory decoding**: Bidirectional state space models capture spatiotemporal sequential dependencies in trajectories.
- **Pronounced advantage in Brier score**: The substantial improvement in prediction confidence (beyond mere distance error) is more valuable for downstream planning modules.

## Limitations & Future Work

- The inner-loop iterations of MaxEnt IRL introduce additional computational overhead.
- The choice of grid resolution $d$ affects reasoning precision; coarse resolution leads to information loss.
- GRT discretization may not perfectly represent continuous driving intentions.
- No direct comparison with LRMs (e.g., OpenAI-o1) on intent reasoning capability is conducted.
- The transferability of the learned reward function warrants further exploration.

## Related Work & Insights

- **QCNet, DeMo**: Top-performing methods on the current Argoverse 2 leaderboard.
- **MaxEnt IRL**: A classic inverse reinforcement learning method that learns rewards from expert demonstrations.
- **Mamba**: Selective state space model demonstrating strong performance in sequence modeling.
- **Insight**: Trajectory prediction should not focus solely on distance error; prediction confidence (Brier score) is equally critical.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combining IRL with query-centric encoding for intent reasoning; the "First Reasoning, Then Forecasting" paradigm is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three large-scale datasets, extensive ablation studies, and qualitative visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Method is described in detail with complete derivations.
- **Value**: ⭐⭐⭐⭐⭐ Ranks first on the nuScenes leaderboard; the method is generalizable to other motion prediction scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[CVPR 2025\] Certified Human Trajectory Prediction](../../CVPR2025/autonomous_driving/certified_human_trajectory_prediction.md)
- [\[NeurIPS 2025\] DINO-Foresight: Looking into the Future with DINO](../../NeurIPS2025/autonomous_driving/dino-foresight_looking_into_the_future_with_dino.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

</div>

<!-- RELATED:END -->
