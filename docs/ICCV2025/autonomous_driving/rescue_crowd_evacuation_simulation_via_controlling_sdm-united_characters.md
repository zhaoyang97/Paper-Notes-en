---
title: >-
  [Paper Note] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters
description: >-
  [Autonomous Driving] This paper proposes RESCUE, the first online SDM (Sensing–Decision–Motion) unified 3D evacuation simulation framework, integrating a 3D adaptive social force model and a personalized gait controller to achieve real-time personalized evacuation simulation for hundreds of agents.
tags:
  - Autonomous Driving
date: 2026-05-08
content_hash: 9115a10abfe3aeaa
---

# RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2507.20117](https://arxiv.org/abs/2507.20117)
- **Code**: [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/RESCUE)
- **Area**: Autonomous Driving
- **Keywords**: Crowd evacuation simulation, social force model, physics engine, personalized gait, 3D character control

## TL;DR

This paper proposes RESCUE, the first online SDM (Sensing–Decision–Motion) unified 3D evacuation simulation framework, integrating a 3D adaptive social force model and a personalized gait controller to achieve real-time personalized evacuation simulation for hundreds of agents.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Crowd evacuation simulation is critical for public safety, yet existing methods fail to simultaneously satisfy the following requirements:

**Simplified representations in traditional simulation**: Social Force Models (SFM) rely on simplified 2D representations, where force feedback can only be approximated by unrealistic substitutes, failing to capture real physical behaviors such as 3D collisions and falls.

**Insufficient scalability of character control**: Physics-engine-based character control methods lack personalized motion in dense scenes and are prone to falls and collisions. Motion generation via diffusion models faces challenges in controllability and physical plausibility.

**Two critical requirements**: 3D proximal sensing (online dynamic adjustment in crowded conditions) and personalized gait (individuals with different attributes exhibiting distinct behaviors).

### Paper Goals

**Goal**: ### SDM Unified Framework

Inspired by the human brain's perception–decision–action loop:

1. **Sensing Module**: Each agent perceives its own state (position, velocity, humanoid state), the states of others (relative root positions), and environmental state.
2. **Decision Module**: 3D adaptive SFM decision mechanism — A* search for path planning + SFM driving + collision avoidance.
3. **Motion Module**: Pacer path-following controller + personalized gait transformer.

### 3D Adaptive SFM Decision Mechanism

**Basic Forces**: Driving force + repulsive force.

## Method

### SDM Unified Framework

Inspired by the human brain's perception–decision–action loop:

1. **Sensing Module**: Each agent perceives its own state (position, velocity, humanoid state), the states of others (relative root positions), and environmental state.
2. **Decision Module**: 3D adaptive SFM decision mechanism — A* search for path planning + SFM driving + collision avoidance.
3. **Motion Module**: Pacer path-following controller + personalized gait transformer.

### 3D Adaptive SFM Decision Mechanism

**Basic Forces**: Driving force + repulsive force.

**Evasive Force** (Core Innovation): Computes an evasion direction perpendicular to the desired direction:

$$F_{\text{evasive}} = A \, \text{sgn}(o_i \cdot p_i) \, p_i$$

where $A$ is a binary mask (set to 1 when an obstacle is ahead and lateral space is available), and $p_i$ is the perpendicular vector.

**Personalized SFM Coefficient Calibration**: Five agent groups (youth, middle-aged, elderly, patients, disabled individuals) with speeds calibrated so that simulated velocities match real-world values reported in the literature.

**Final Decision**: $\tilde{v}_{i,t+1} = v_i + \Delta t (F_{\text{drive}} + F_{\text{repulsive}} + F_{\text{evasive}})$

### Personalized Gait Controller

A CAMDM diffusion model is used to convert non-personalized motion frames into personalized motion frames: $a^0 = G(a^t, t; c)$. Gait frame matching (4 key events) is applied to align personalized and non-personalized frame pairs. During simulation, only upper-body motions are replaced.

### Part-Level Force Visualization

Force sensors are integrated into 24 body parts, and contact force magnitudes are visualized via a color gradient.

## Experiments

### Quantitative Comparison

### Main Results

| Method | Avg. Success Rate | Avg. Fall Rate |
|--------|------------------|---------------|
| OmniControl | 0.48 | — |
| MaskedMimic | 0.60 | 18.55% |
| **RESCUE** | **0.84** | **12.26%** |

### Velocity Diversity Analysis

### Ablation Study

| Group | Velocity Characteristics |
|-------|------------------------|
| Youth | Highest median speed, high variability |
| Middle-aged | Moderate speed distribution |
| Elderly | Slower speeds, narrow range |
| Patients | Limited speed |
| Disabled | Lowest speed |

### Key Findings

1. **Highest success rate**: 0.84 vs. MaskedMimic's 0.60, attributed to online obstacle avoidance by the 3D adaptive SFM.
2. **Fewest falls**: 12.26 vs. 18.55; evasive forces effectively prevent congestion-induced falls.
3. **Density–width trampling analysis**: At the same corridor width, greater crowd density leads to more severe trampling; at the same density, narrower corridors worsen trampling.
4. **Terrain effect validation**: Uneven and slippery surfaces are more likely to cause falls.

## Highlights & Insights

1. **Neuroscience inspiration**: The SDM loop endows the framework with biological plausibility.
2. **Practical innovation of evasive force**: In 3D environments, evasion is more realistic than waiting.
3. **Personalized coefficient calibration**: Compensates for friction in the physics engine to ensure simulated speeds match literature values.
4. **Part-level force visualization**: Provides unprecedented insight for evacuation analysis.

## Limitations & Future Work

1. Motion styles in the 100STYLE dataset are limited.
2. The five-group classification is relatively coarse.
3. Applicability to other multi-agent environments requires further validation.

## Related Work & Insights

- **Crowd simulation**: Social force models, agent-based models.
- **Character control**: Pacer, MaskedMimic, DRL-based methods.
- **Motion generation**: Diffusion models, OmniControl, EMAGE.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First SDM unified framework with 3D adaptive SFM.
- **Technical Depth**: ⭐⭐⭐⭐ — Deep integration of multiple modules.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-scenario analysis with user studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Rich visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)
- [\[NeurIPS 2025\] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning](../../NeurIPS2025/autonomous_driving/unimotion_a_unified_motion_framework_for_simulation_prediction_and_planning.md)

</div>

<!-- RELATED:END -->
