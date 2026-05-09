---
title: >-
  [Paper Note] SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration
description: >-
  [NeurIPS 2025][Autonomous Driving][simulation] This paper presents SimWorld-Robotics (SWR), a large-scale urban simulation platform built on Unreal Engine 5 that supports procedural generation of unlimited photorealistic city environments. Built upon this platform, two new benchmarks are introduced — SimWorld-MMNav for multimodal navigation and SimWorld-MRS for multi-robot search — which collectively reveal critical capability gaps in current VLMs on outdoor urban tasks.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - simulation
  - embodied-ai
  - urban-navigation
  - multi-robot
  - benchmark
date: 2026-05-08
content_hash: 3894534319f80305
---

# SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration

**Conference**: NeurIPS 2025
**arXiv**: [2512.10046](https://arxiv.org/abs/2512.10046)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: simulation, embodied-ai, urban-navigation, multi-robot, benchmark

## TL;DR

This paper presents SimWorld-Robotics (SWR), a large-scale urban simulation platform built on Unreal Engine 5 that supports procedural generation of unlimited photorealistic city environments. Built upon this platform, two new benchmarks are introduced — SimWorld-MMNav for multimodal navigation and SimWorld-MRS for multi-robot search — which collectively reveal critical capability gaps in current VLMs on outdoor urban tasks.

## Background & Motivation

1. **Indoor Bias**: Existing embodied AI simulators (Habitat 3, RoboTHOR, BEHAVIOR, etc.) focus predominantly on tabletop and household scenarios, lacking support for outdoor urban environments.
2. **Insufficient Urban Simulation**: CARLA and AirSim target autonomous driving and do not support procedural city generation; MetaDrive and MetaUrban are scalable but suffer from poor rendering quality and limited realism.
3. **Absence of Dynamic Interaction**: Pedestrian behaviors in existing outdoor simulators are limited to random walking, failing to capture complex urban dynamics such as traffic rules and social navigation.
4. **Multi-Agent Limitations**: Most simulators support only a single agent type (vehicles or robots) and do not enable unified control and asynchronous collaboration among humans, vehicles, and robots.
5. **Narrow Navigation Benchmarks**: Vision-and-Language Navigation datasets are predominantly static indoor environments or street-view panoramas, lacking joint evaluation of multimodal instructions (language + visual cues) in dynamic urban settings.
6. **Gap in Multi-Robot Collaboration**: Existing benchmarks (RoCo, RobotSlang, etc.) are either confined to small environments or do not address the joint evaluation of language communication and physical navigation in collaborative settings.

## Method

### 3.1 Procedural City Generation

SWR generates cities through four stages:
- **Road Network**: A priority-queue-based growth strategy that balances branching and depth, supporting road-end connections and intersection validation.
- **Building Placement**: Collision-aware sampling along roads combined with greedy gap-filling to maximize coverage.
- **Street Elements**: Context-aware placement of trees, traffic cones, benches, and parked vehicles.
- **Dynamic Traffic Elements**: Integration of vehicles and pedestrians with probabilistic intersection routing to introduce natural variability.

### 3.2 Embodied Agents

SWR provides unified support for three agent types: humans (26 action types), vehicles, and robots (scooters/quadruped robots). An asynchronous multi-agent control framework is adopted, enabling each agent to act independently without blocking others. The observation space includes RGB, depth, semantic segmentation, and 3D bounding boxes.

### 3.3 Pedestrian and Traffic Simulation

- Vehicle motion is driven by PID controllers with empirically tuned parameters.
- Pedestrians use a lightweight model that progressively adjusts heading based on angular difference.
- Intersections employ a probabilistic routing strategy to enhance scene diversity.

### 3.4 SimWorld-MMNav Multimodal Navigation Benchmark

Robots must navigate to targets in large-scale dynamic cities by following multimodal instructions (textual descriptions + visual cue images). Instructions are categorized into four types: direction alignment, road following, intersection turning, and destination arrival. Two difficulty levels are defined: Easy (no obstacles) and Hard (with pedestrians and traffic signals). Evaluation metrics include success rate (SR), sub-task SR, distance progress, and safety metrics (collision count, red-light violations).

### 3.5 SimWorld-MRS Multi-Robot Search Benchmark

Two robots must locate and rendezvous with each other in an unfamiliar city through natural language communication combined with physical navigation. The host robot has memorized 20+ city landmarks, while the follower robot has no prior knowledge. Metrics include collaborative success rate (CSR) and task progress.

### 3.6 SimWorld-20K Dataset

100 training maps with an average area of $2\ \text{km}^2$, 200 A*-oracle trajectories (average length $> 2.5\ \text{km}$), totaling 20K training steps. The environment area is $100\times$ larger than MetaUrban, and trajectory lengths exceed MetaUrban by more than $1.2\times$.

## Key Experimental Results

### Table 1: SimWorld-MMNav Easy Setting Results

| Model | SR(%) ↑ | Sub-task SR(%) ↑ | Distance Progress(%) ↑ |
|-------|---------|-----------------|----------------------|
| GPT-4o | 0 | 33.07 | 15.60 |
| Gemini 2.5 Flash | 0 | 37.06 | 31.29 |
| GPT-o3 | 5.0 | 42.50 | 38.43 |
| GPT-o3-pro | 8.3 | 46.35 | 39.46 |
| QwenVL 2.5 7B | 0 | 16.86 | 7.82 |
| QwenVL 2.5 72B | 0 | 23.80 | 17.50 |
| QwenVL 2.5 7B (ft) | **4.0** | **52.45** | **53.63** |

**Key Findings**: All zero-shot non-reasoning models achieve SR of 0; reasoning models (o3/o3-pro) are the first to achieve non-zero SR; the fine-tuned 7B model comprehensively outperforms closed-source large models.

### Table 2: SimWorld-MRS Multi-Robot Search Results

| Model | Method | CSR(%) ↑ | Task Progress(%) ↑ |
|-------|--------|---------|-------------------|
| GPT-4o | Oracle Planner | 65.00 | 76.90 |
| Gemini 2.5 Flash | Oracle Planner | 54.55 | 75.84 |
| GPT-4o | RoCo | 33.33 | 22.93 |
| QwenVL 72B | RoCo | 11.11 | 35.94 |

**Key Findings**: Even with an Oracle Planner, GPT-4o achieves only 65% CSR. The RoCo strategy suffers significant performance degradation due to linguistic ambiguity leading to execution deviations.

### Failure Mode Analysis

- **Intersection Distance Misjudgment** (53.33%): VLMs fail to accurately estimate distance to intersections.
- **Destination Viewpoint Matching Failure** (60.00%): Recognizing the same landmark from different viewpoints is challenging.
- **Turn Pattern Misinterpretation** (42.86%): High-level instructions such as "turn right at the intersection" lack sufficient action-level specificity.

## Highlights & Insights

- The first urban simulation platform to simultaneously support procedural generation, photorealistic rendering, and asynchronous control of humans, vehicles, and robots.
- Supports 26 human action types, far exceeding existing simulators ($\leq 10$).
- The two benchmarks are carefully designed to systematically expose VLM deficiencies in spatial reasoning, dynamic obstacle avoidance, and multi-agent communication.
- SimWorld-20K provides the first large-scale dataset for urban navigation training, with demonstrated fine-tuning efficacy.

## Limitations & Future Work

- Focuses exclusively on outdoor environments; indoor scenes are not supported.
- The human agent action space, while superior to prior work, remains limited and does not leverage motion generation models.
- SimWorld-20K training data is based on oracle trajectories, lacking error-correction or reinforcement learning data.
- In the Hard setting, reasoning models cannot perform real-time obstacle avoidance due to excessive inference latency.
- Rendering is based on UE5, but sim-to-real transfer effectiveness has not been quantitatively evaluated.

## Related Work & Insights

### vs. CARLA
As a classical autonomous driving simulator, CARLA does not support procedural city generation, exhibits low building diversity with high repetition, and only supports vehicle control and simple pedestrian behavior (2 action types). SWR supports unlimited city generation, unified asynchronous control of three agent types, and 26 human action types, comprehensively surpassing CARLA in scalability and realism.

### vs. MetaUrban
MetaUrban supports procedural generation and micro-mobility tasks, but its environment area is only $0.02\ \text{km}^2$ with low rendering quality (non-photorealistic). SWR achieves an environment area of $2\ \text{km}^2$ ($100\times$ larger), photorealistic rendering via UE5, and supports vehicular and pedestrian traffic systems, far exceeding MetaUrban in task complexity and evaluation dimensions.

### vs. RoCo/RobotSlang
RoCo is limited to tabletop manipulation tasks and does not involve environment exploration; RobotSlang evaluates robot communication but is confined to small environments. SimWorld-MRS is the first benchmark to evaluate joint multi-robot language communication and physical navigation in large-scale urban environments.

## Rating

- ⭐⭐⭐⭐ **Novelty**: First fully-featured urban embodied AI simulation platform.
- ⭐⭐⭐⭐ **Writing Quality**: Complete engineering implementation with systematic and comprehensive benchmark design.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Multi-model, multi-setting evaluation with in-depth failure mode analysis.
- ⭐⭐⭐ **Value**: UE5-based platform poses relatively high deployment barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](../../ICCV2025/autonomous_driving/leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[NeurIPS 2025\] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[ICCV 2025\] Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)

</div>

<!-- RELATED:END -->
