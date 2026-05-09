---
title: >-
  [Paper Note] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation
description: >-
  [ICCV 2025][Autonomous Driving][Traffic Simulation] This paper proposes InfGen, a unified autoregressive next-token prediction model that interleaves closed-loop motion simulation with scenario generation (dynamic agent insertion and removal), achieving for the first time stable long-term (30-second) traffic simulation. InfGen reaches state-of-the-art performance on short-term benchmarks and significantly outperforms all existing methods on long-term tasks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Traffic Simulation
  - Long-term Simulation
  - Autoregressive Generation
  - Scenario Generation
  - Next-Token Prediction
date: 2026-05-08
content_hash: e45616e4199e2f06
---

# Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation

**Conference**: ICCV 2025
**arXiv**: [2506.17213](https://arxiv.org/abs/2506.17213)
**Code**: [orangesodahub.github.io/InfGen](https://orangesodahub.github.io/InfGen)
**Area**: Autonomous Driving
**Keywords**: Traffic Simulation, Long-term Simulation, Autoregressive Generation, Scenario Generation, Next-Token Prediction

## TL;DR

This paper proposes InfGen, a unified autoregressive next-token prediction model that interleaves closed-loop motion simulation with scenario generation (dynamic agent insertion and removal), achieving for the first time stable long-term (30-second) traffic simulation. InfGen reaches state-of-the-art performance on short-term benchmarks and significantly outperforms all existing methods on long-term tasks.

## Background & Motivation

### Problem Definition

Traffic simulation aims to provide realistic driving experiences for autonomous driving systems. An ideal simulator should deliver complete trip-level realistic traffic flow, encompassing realistic environments, ego-vehicle dynamics, and all non-ego traffic participants.

### Limitations of Prior Work

The fundamental assumption of existing methods is that **the agent set remains fixed throughout the simulation horizon**, which entirely breaks down in long-term simulation:

**Agent disappearance**: As the ego vehicle moves into new regions, agents from the initial log progressively leave the field of view.

**Scene depopulation**: When the ego vehicle enters map regions not covered by the log, those regions contain no agents.

**Unrealistic empty scenes**: State-of-the-art models such as SMART produce scenes devoid of agents around the ego vehicle after 30 seconds of simulation (Figure 1).

Three categories of prior work exhibit the following limitations:
- **Closed-loop motion simulation** (SMART, CatK): simulates only the motion of pre-existing agents without generating new ones.
- **Scenario generation** (SceneGen, TrafficGen): generates only static initial scenes or short open-loop scenarios.
- **Adversarial scenario generation**: focuses on near-collision scenarios and is not suited for general long-term simulation.

### Root Cause

Long-term traffic simulation must simultaneously address two problems: (1) closed-loop motion simulation of existing agents; and (2) dynamic generation of new agents and removal of departing ones. InfGen unifies both tasks within an interleaved next-token prediction framework.

## Method

### Overall Architecture

InfGen formulates long-term traffic simulation as interleaved expansion over a "dynamic agent matrix":
- **Temporal axis expansion** (motion simulation): predicts the next-step motion token for each active agent.
- **Spatial axis expansion** (scenario generation): inserts new agent rows (pose tokens) or removes departing agent rows.

Formally:
$$p(\mathcal{A}'_{t+1:T'} | \mathcal{M}, \mathcal{A}_{0:t_0}) = \prod_{t=t_0}^{T'-1} p_{\text{scene}}(\mathcal{A}'_{t+1} | \mathcal{M}, \mathcal{A}_{t+1}) \times p_{\text{motion}}(\mathcal{A}_{t+1} | \mathcal{M}, \mathcal{A}'_{0:t})$$

### Key Designs

#### 1. **Unified Tokenization Scheme**

- **Function**: Converts map, motion, pose, and mode-control information into discrete token sequences.
- **Mechanism**:

  Four tokenizers are employed:
  - **Map Tokenizer**: Segments road elements into fixed-length vectors encoding start/end points, direction, and road type.
  - **Motion Tokenizer**: Encodes 0.5-second continuous trajectory segments into a discrete motion vocabulary $\mathcal{V}_{\text{motion}}$ via k-disks clustering, using nearest-neighbor indices.
  - **Pose Tokenizer**: Encodes the initial pose of new agents as a position token (grid index centered on the ego vehicle) and a heading token (360° uniform quantization).
  - **Mode Control Tokenizer**: Four special tokens govern task switching:
    - `<BEGIN MOTION>`: the next token is a motion token.
    - `<ADD AGENT>`: the next token is a pose token, inserting a new agent.
    - `<KEEP AGENT>`: the current agent is retained.
    - `<REMOVE AGENT>`: the current agent will be removed.

- **Design Motivation**: Reduces the complex mixed-task simulation problem to a simple sequence prediction problem. The four control tokens enable the model to learn **when** to switch tasks and **how** to decide agent insertion and removal.

#### 2. **Interleaved Next-Token Prediction**

- **Function**: Alternately executes motion simulation and scenario generation at each timestep.
- **Mechanism**:

  **Temporal motion simulation** (blue stream):
  For each active agent $i$, its motion token $m_i^t$ serves as query $q_{m_i^t}$ and passes through three attention layers:
  1. **Temporal Attention**: self-attention over the agent's past $t_w$ motion tokens.
  2. **Agent-Agent Attention**: cross-attention over other active agents within range $r^{a \leftrightarrow a}$ at the same timestep.
  3. **Map-Agent Attention**: cross-attention over map tokens within range $r^{m \leftrightarrow a}$.

  A motion head and a control head each sample from their respective token distributions. The control token is restricted to `<KEEP AGENT>` or `<REMOVE AGENT>`.

  **Spatial scenario generation** (green stream):
  Uses a learnable agent query $a_0$ passed through three attention layers, where Grid Attention replaces Temporal Attention. Grid Attention attends to occupancy grid tokens (binary occupancy indicators constructed from position tokens):
  $$q'_{a_0} = \text{MHCA}^g(q_{a_0}, \Gamma(\{k_{g_j}\}), \Gamma(\{v_{g_j}\}))$$

  The control token is restricted to `<ADD AGENT>` or `<BEGIN MOTION>`. Upon `<ADD AGENT>`, a new row is inserted and assigned a pose token; upon `<BEGIN MOTION>`, scenario generation concludes and the next timestep's motion simulation begins.

- **Design Motivation**:
  - Interleaved execution naturally couples the two tasks, enabling the model to autonomously determine whether agents should be added or removed based on the current scene state.
  - Occupancy grid encoding provides the scenario generation module with spatial awareness of the current scene, preventing redundant insertion at already-occupied locations.
  - The autoregressive nature of next-token prediction allows the model to **generalize from short logs during training to long-term simulation at inference** (6× extension).

#### 3. **Occupancy Grid Encoder**

- **Function**: Encodes the spatial distribution of agents in the current scene as occupancy grid features for use in scenario generation.
- **Mechanism**: Each position in the position token vocabulary $\mathcal{V}_\text{pos}$ is labeled 0 (empty) or 1 (occupied), converted to features via MLP, and fed into Grid Attention.
- **Design Motivation**: Enables the scenario generation module to efficiently reason about the spatial distribution of agents and determine where to insert new ones.

### Loss & Training

The total training loss is a weighted sum of standard NTP losses over multiple token types:
$$\mathcal{L} = \lambda_1 \mathcal{L}_\text{motion} + \lambda_2 \mathcal{L}_\text{pos} + \lambda_3 \mathcal{L}_\text{head} + \lambda_4 \mathcal{L}_\text{control} + \lambda_5 \mathcal{L}_\text{shape} + \lambda_6 \mathcal{L}_\text{type}$$

where $\lambda_1 = \lambda_3 = 1$, $\lambda_2 = \lambda_4 = 10$, $\lambda_5 = 0.2$, $\lambda_6 = 5$.

Training token sequences are constructed by arranging tokens in a fixed order at each timestep: motion token → control token (REMOVE/KEEP) → pose token (ADD) → BEGIN MOTION. Tokens of the same type are ordered by agent distance to the ego vehicle from nearest to farthest.

Training configuration: batch size 8, 8× A5000 GPUs, AdamW with cosine annealing, initial learning rate 0.0005.

## Key Experimental Results

### Main Results

Short-term simulation (WOSAC benchmark, 9s):

| Method | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ |
|--------|-----------|-----------|-------------|------|
| TrafficBots | 0.6976 | 0.3994 | 0.7103 | 0.8342 |
| GUMP | 0.7404 | 0.4773 | 0.7872 | 0.8339 |
| SMART-7M | 0.7521 | 0.4799 | 0.8048 | 0.8573 |
| CatK | 0.7603 | 0.4611 | 0.8103 | 0.8732 |
| **InfGen** | **0.7514** | **0.4754** | 0.7936 | 0.8502 |

Long-term simulation (30s, extended WOSAC metrics):

| Method | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ | Placement $N_+$↑ | $N_-$↑ | $D_+$↑ | $D_-$↑ |
|--------|-----------|-----------|-------------|------|-----|-----|-----|-----|
| SMART-7M | 0.6519 | 0.5839 | 0.7542 | 0.8102 | 0.4324 | 0.5713 | 0.4964 | 0.3371 |
| CatK | 0.6584 | 0.5850 | 0.7584 | 0.8186 | 0.4424 | 0.5842 | 0.5233 | 0.3371 |
| **InfGen** | **0.6606** | **0.5966** | **0.7619** | 0.8087 | **0.4542** | **0.6273** | **0.5635** | 0.3169 |

### Ablation Study

Agent Count Error (ACE) metric:

| Method | Mean ACE↓ | ACE Slope↓ | Note |
|--------|----------|-----------|------|
| SMART-7M | 12.0 | 0.31 | Scene progressively empties |
| CatK | 12.2 | 0.32 | Same issue |
| **InfGen** | **8.1** | **0.15** | Error growth rate half that of baselines |

Motion-only simulation (agent insertion/removal disabled, 30s):

| Method | Composite↑ | Kinematic↑ | Interactive↑ | Map↑ |
|--------|-----------|-----------|-------------|------|
| SMART-7M | 0.7428 | 0.5413 | 0.7626 | 0.8349 |
| CatK | 0.7316 | 0.5216 | 0.7347 | 0.8495 |
| InfGen | 0.7432 | 0.5495 | 0.7685 | 0.8213 |

### Key Findings

1. **InfGen significantly outperforms baselines in long-term simulation**: The advantage is most pronounced on Placement metrics, validating the core value of dynamic scenario generation.
2. **ACE Slope of 0.15 vs. baselines' 0.31–0.32**: InfGen's scene density error growth rate is half that of baselines.
3. **Competitive short-term performance**: Without any short-term-specific tuning, short-term performance approaches that of CatK.
4. **Similar performance across methods in motion-only setting**: Confirms that long-term performance gaps stem primarily from scenario generation capability rather than motion prediction quality.
5. **Train short, infer long**: Trained on ~9s logs, the model stably simulates 30s (6× extension), demonstrating the generalization potential of the autoregressive framework.
6. **Qualitative evidence**: SMART scenes become empty after 18s; InfGen consistently maintains realistic traffic density.

## Highlights & Insights

1. **Contribution to problem formulation**: The paper explicitly identifies the "fixed agent set" assumption as the fundamental bottleneck in long-term simulation—a critical yet previously overlooked issue.
2. **Inspiration from Chameleon and interleaved multimodal generation**: The interleaved generation paradigm from vision-language modeling is successfully transferred to "temporal motion–spatial layout" alternation.
3. **Unified NTP framework**: Motion simulation and scenario generation share a single transformer; four control tokens enable elegant task switching.
4. **Evaluation framework contribution**: The proposed ACE metric and extended WOSAC metrics establish an evaluation standard for long-term traffic simulation research.
5. **Elegant design of the Occupancy Grid Encoder**: Provides the scenario generation module with visibility into the current spatial occupancy state, preventing unreasonable overlapping placements.

## Limitations & Future Work

1. **Trip-level simulation not yet achieved**: 30 seconds remains far shorter than real trips (>5 minutes), primarily constrained by WOMD map coverage.
2. **Limitations of pure supervised learning**: The model may overfit causal relationships in training data; future work plans to incorporate interactive reinforcement learning.
3. **Slightly lower Map metric**: Newly inserted agents may appear in non-drivable areas or near road boundaries, slightly degrading map compliance.
4. **Limited agent types**: The work primarily focuses on vehicles; modeling of pedestrians and cyclists may be insufficient.
5. **Evaluated only on WOMD**: Generalization to nuPlan or other simulation environments remains untested.

## Related Work & Insights

- **Distinction from SMART**: SMART is a pure motion simulation NTP model; InfGen extends it with scenario generation capability.
- **Distinction from SceneGen/TrafficGen**: The latter generate static initial scenes; InfGen dynamically generates agents during closed-loop simulation.
- **Analogy to Chameleon**: The interleaved generation idea is transferred from "text + image" to "motion + scene layout."
- **Distinction from SLEDGE**: SLEDGE combines a generative model with a rule-based simulator; InfGen is fully end-to-end.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First work to introduce interleaved generation into traffic simulation, unifying motion and scenario generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on both short-term and long-term benchmarks, though genuine trip-level simulation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem motivation is clearly articulated; qualitative comparisons are compelling.
- **Value**: ⭐⭐⭐⭐⭐ — Represents an important step toward realistic trip-level traffic simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](../../NeurIPS2025/autonomous_driving/unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)

</div>

<!-- RELATED:END -->
