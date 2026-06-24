---
title: >-
  [Paper Note] Towards Autonomous Micromobility through Scalable Urban Simulation
description: >-
  [CVPR 2025 (Highlight)][Autonomous Driving][Micromobility] This paper proposes URBAN-SIM (a high-performance urban robot learning simulation platform) and URBAN-BENCH (a benchmark with 8 micromobility tasks). By incorporating three core modules—hierarchical urban scene generation, interactive dynamics generation, and asynchronous scene sampling—the framework enables training and evaluation of embodied agents in large-scale, diverse urban environments…
tags:
  - "CVPR 2025 (Highlight)"
  - "Autonomous Driving"
  - "Micromobility"
  - "Urban Simulation"
  - "Robot Learning"
  - "Embodied Agents"
  - "Benchmark"
date: 2026-05-08
content_hash: 3a5933757ae120eb
---

# Towards Autonomous Micromobility through Scalable Urban Simulation

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2505.00690](https://arxiv.org/abs/2505.00690)  
**Code**: None (project page is offline)  
**Area**: Autonomous Driving / Embodied AI  
**Keywords**: Micromobility, Urban Simulation, Robot Learning, Embodied Agents, Benchmark

## TL;DR
This paper proposes URBAN-SIM (a high-performance urban robot learning simulation platform) and URBAN-BENCH (a benchmark with 8 micromobility tasks). By incorporating three core modules—hierarchical urban scene generation, interactive dynamics generation, and asynchronous scene sampling—the framework enables training and evaluation of embodied agents in large-scale, diverse urban environments, serving as a systematic simulation solution for driving autonomous micromobility forward.

## Background & Motivation

**Background**: Micromobility refers to the transportation mode using lightweight mobile devices (such as delivery robots, electric wheelchairs, and electric scooters) in urban public spaces. Currently, most micromobility devices rely on manual operation (either on-site or teleoperation). As deployment scales up across cities, manual control faces safety hazards and efficiency bottlenecks.

**Limitations of Prior Work**: (1) Lack of simulation platforms designed specifically for micromobility—existing autonomous driving simulators (such as CARLA, nuPlan) cater to vehicular scenarios where road structures, traffic rules, and sensor configurations do not match micromobility devices; (2) Insufficient coverage of scene diversity in urban pedestrian zones—simulating unstructured environments like sidewalks, plazas, and parks is far more difficult than structured roads; (3) Lack of unified task definitions and evaluation standards for micromobility.

**Key Challenge**: Micromobility devices must safely navigate complex, unstructured urban pedestrian spaces, but training such skills requires a large volume of diverse, interactive urban environments. Real-world data collection is prohibitively expensive and encounters safety risks, while existing simulation platforms cannot provide sufficiently diverse and realistic pedestrian-level urban environments.

**Goal**: (1) Build a high-performance, scalable urban simulation platform to support large-scale robot learning; (2) Define a task benchmark covering the core capabilities of micromobility; (3) Evaluate the performance of various robot morphologies across different tasks.

**Key Insight**: Starting from three bottlenecks of simulation—scene diversity, interactive fidelity, and training efficiency—the authors propose targeted modules for each. Concurrently, they select 8 tasks covering three core skills (locomotion, navigation, and traversal) and evaluate four robot morphologies across wheeled and legged designs.

**Core Idea**: To construct a "simulation infrastructure"-level platform through procedural generation of large-scale diverse urban scenes, parallel asynchronous sampling on GPU, and dynamic generation of physical interactions, and further define and evaluate micromobility tasks based on this platform.

## Method

### Overall Architecture
The system consists of two main parts: **URBAN-SIM** (the simulation platform) responsible for environment generation and physical simulation, and **URBAN-BENCH** (the task benchmark) defining specific evaluation tasks and metrics. Built on Isaac Gym/Lab as the underlying physics engine, URBAN-SIM achieves high-performance and large-scale training via three core modules: Hierarchical Urban Generation for scene diversity, Interactive Dynamics Generation for realistic interactions, and Asynchronous Scene Sampling for high training efficiency.

### Key Designs

1. **Hierarchical Urban Scene Generation (Hierarchical Urban Generation)**:

    - **Function**: Procedurally generate a large and diverse set of urban scenes, including various terrains, building layouts, street furniture, and pedestrians.
    - **Mechanism**: A three-tier hierarchical structure is used for scene generation. **Macro-level**: extracts road network topologies and regional functional zones from real urban maps (e.g., OpenStreetMap) to outline the overall scene layout. **Meso-level**: procedurally places urban elements such as buildings, sidewalks, intersections, and parks on top of the road network, while randomizing their geometric parameters (width, height, materials). **Micro-level**: injects street furniture (street lamps, benches, trash bins), terrain variations (ramps, stairs, rough surfaces), and dynamic pedestrians into the scene. The randomization parameters for each layer are controlled independently, enabling combinatorial explosion for vast scene diversity.
    - **Design Motivation**: The environments micromobility devices face are far more complex than vehicular roads, involving irregular sidewalks, various curbs, street obstacles, etc. Hierarchical generation ensures structural plausibility of the scenes (relying on real maps) while providing abundant diversity through multi-level randomization.

2. **Interactive Dynamics Generation (Interactive Dynamics Generation)**:

    - **Function**: Inject realistic dynamic interactive elements into the simulation scene, particularly pedestrian behavior.
    - **Mechanism**: Pedestrians' basic motions are driven by the Social Force Model combined with reactive behaviors—pedestrians will avoid, pause, or change direction when robots approach. In addition, dynamic obstacles (such as opening vehicle doors, pedestrians pushing carts) and environmental changes (such as construction site blockages) are simulated. All dynamic elements are calculated in parallel on the GPU to prevent simulation bottlenecks.
    - **Design Motivation**: Policies trained in static environments often fail when deployed in the real world, as the reactive behavior of pedestrians is the core challenge to micromobility safety. Interactive dynamics ensure that the learned policies can handle dynamic disturbances.

3. **Asynchronous Scene Sampling (Asynchronous Scene Sampling)**:

    - **Function**: Enhance GPU utilization during large-scale parallel training.
    - **Mechanism**: In traditional RL training, all parallel environments share a single scene or switch scenes synchronously, making scene loading a bottleneck. URBAN-SIM adopts an asynchronous mechanism: each GPU worker maintains a scene buffer pool. When an environment finishes an episode, it immediately draws a new scene from the buffer pool to continue training without waiting for other environments. The scene buffer pool is continuously generated and filled in a background thread, decoupling training efficiency from scene diversity.
    - **Design Motivation**: Micromobility training requires a massive variety of scenes to prevent overfitting, but frequent scene transitions significantly degrade GPU utilization. Asynchronous sampling uses pipelining to run scene preparation and policy training in parallel.

### Loss & Training
All tasks are trained using the PPO reinforcement learning algorithm. Reward functions are designed individually for each task: locomotion tasks are based on target velocity tracking, navigation tasks depend on the distance to target destination and collision penalties, and traversal tasks comprehensively consider success rate, time efficiency, and safety.

## Key Experimental Results

### Main Results: Performance of Four Robots on URBAN-BENCH

| Task Category | Specific Task | Unitree Go2 (Legged) | Unitree H1 (Bipedal) | Clearpath Jackal (Wheeled) | LoCoBot (Wheeled) |
|--------------|--------------|--------------------|--------------------|------------------------|----------------|
| Urban Locomotion | Flat Ground Walking | **95.2%** | 88.7% | 93.1% | 91.5% |
| Urban Locomotion | Rough Terrain | **82.3%** | 71.5% | 45.6% | 38.2% |
| Urban Locomotion | Climbing Stairs | **76.8%** | 68.2% | 12.3% | 8.7% |
| Urban Navigation | Open Area Navigation | 87.5% | 82.1% | **91.3%** | 88.6% |
| Urban Navigation | Crowded Pedestrian Navigation | **78.4%** | 72.3% | 74.1% | 70.5% |
| Urban Navigation | Narrow Passage Navigation | 71.2% | 65.8% | **76.5%** | 73.9% |
| Urban Traversal | Composite Route Traversal | **68.5%** | 58.3% | 52.7% | 45.1% |
| Urban Traversal | Long-distance Traversal | 62.1% | 51.6% | **64.3%** | 56.8% |

### Ablation Study: Impact of URBAN-SIM Modules

| Configuration | Rough Terrain Success Rate | Crowded Navigation Success Rate | Composite Traversal Success Rate | Description |
|------|--------------|--------------|--------------|------|
| Full URBAN-SIM | **82.3%** | **78.4%** | **68.5%** | Full system |
| w/o Hierarchical Scene Generation | 72.1% | 69.2% | 56.3% | Insufficient scene diversity leads to poor generalization |
| w/o Interactive Dynamics | 80.5% | 61.7% | 55.8% | No pedestrian interaction, degradation in navigation tasks is significant |
| w/o Asynchronous Sampling | 81.8% | 77.1% | 67.2% | Performance is comparable but training time increases by 2.3x |
| Single Scene Training (Fixed) | 58.4% | 52.3% | 38.6% | Severe overfitting to a single environment |

### Key Findings
- **Morphology determines capability boundaries**: Legged robots (Go2) significantly outperform wheeled robots on rough terrain and stairs tasks (success rate 82.3% vs. 45.6%), but wheeled robots achieve higher navigation efficiency in flat scenarios.
- **Scene diversity is key to generalization**: Removing hierarchical generation drops the success rate across all tasks by 8-12 percentage points, and training on a fixed single scene leads to catastrophic degradation.
- **Interactive dynamics are crucial for navigation tasks**: Without pedestrian interaction, the success rate for crowded navigation drops from 78.4% to 61.7% (-17pp), although it has minor impact on pure locomotion tasks.
- **Asynchronous sampling primarily boosts efficiency rather than final performance**: The success rate remains almost unchanged, but training time is reduced by 2.3x.
- Composite traversal tasks (requiring a combination of locomotion, navigation, and obstacle avoidance) present the greatest challenge for all robots.

## Highlights & Insights
- **Clear system design formulation**: Systematically addresses simulation platform bottlenecks across three dimensions—scene generation, interactive dynamics, and training efficiency—rather than focusing on standalone issues. This positioning as a "simulation infrastructure" holds greater long-term impact than individual algorithmic innovations.
- **Hierarchical procedural generation** is a highly reusable paradigm: constraints from real-world data at the macro level combined with procedural randomization at Meso and Micro levels achieves a dual-balance between realism and diversity. This can be transferred to indoor robot simulation, UAV simulation, and other domains.
- **Heterogeneous robot comparison** provides valuable engineering insights: helping practitioners select appropriate robot morphologies based on actual deployment scenarios.

## Limitations & Future Work
- Simulation-to-real (sim-to-real) transfer remains unverified—all experiments were completed in simulation, leaving real-world transferability as an open question.
- Weather and lighting variations are not modeled; real-world rainy days or nighttime scenarios might cause policy failure.
- Only RL policies were evaluated, lacking comparison with imitation learning or vision-based end-to-end approaches.
- Pedestrian behavior models are relatively simple (Social Force Model), failing to cover more complex interactive scenarios (e.g., pedestrians suddenly rushing out, pets on leashes, etc.).

## Related Work & Insights
- **vs. CARLA**: CARLA targets vehicle autonomous driving, where road structures and traffic rules differ significantly from micromobility scenarios. URBAN-SIM focuses on pedestrian-level urban spaces, filling this gap.
- **vs. Habitat/iGibson**: Indoor navigation simulators focus on enclosed spaces, whereas URBAN-SIM handles large-scale scenes in open-world urban environments.
- **vs. MetaUrban**: The concurrent work, MetaUrban, also focuses on urban micromobility simulation, but URBAN-SIM offers superior scalability due to hierarchical generation and asynchronous sampling.

## Rating
- Novelty: ⭐⭐⭐⭐ Simultaneously defines a micromobility simulation platform and benchmark for the first time, filling a crucial gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 4 robots × 8 tasks, with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The system architecture is structurally clear and well-described, well-deserving of a CVPR Highlight.
- Value: ⭐⭐⭐⭐⭐ A simulation platform-level contribution, holding infrastructure-level value for micromobility and urban embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)
- [\[NeurIPS 2025\] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](../../NeurIPS2025/autonomous_driving/urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)

</div>

<!-- RELATED:END -->
