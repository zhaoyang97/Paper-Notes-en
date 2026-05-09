---
title: >-
  [Paper Note] RoboTron-Sim: Improving Real-World Driving via Simulated Hard-Case
description: >-
  [ICCV2025][Autonomous Driving][End-to-end autonomous driving] This paper proposes RoboTron-Sim, a framework that constructs a hard-case simulation dataset (HASS), introduces Scenario-aware Prompt Engineering (SPE) and an Image-to-Ego encoder (I2E), enabling MLLMs to effectively leverage simulated hard cases to improve real-world driving performance. On nuScenes hard scenarios, it achieves ~48% reduction in L2 distance and ~46% reduction in collision rate, establishing state-of-the-art open-loop planning performance.
tags:
  - ICCV2025
  - Autonomous Driving
  - End-to-end autonomous driving
  - Sim2Real transfer
  - multimodal large language models
  - simulation data augmentation
  - hard-case scenarios
date: 2026-05-08
content_hash: 1d8be7e4e2cc73ed
---

# RoboTron-Sim: Improving Real-World Driving via Simulated Hard-Case

**Conference**: ICCV2025
**arXiv**: [2508.04642](https://arxiv.org/abs/2508.04642)
**Code**: [Project Page](https://stars79689.github.io/RoboTron-Sim)
**Area**: Autonomous Driving / Sim2Real
**Keywords**: End-to-end autonomous driving, Sim2Real transfer, multimodal large language models, simulation data augmentation, hard-case scenarios

## TL;DR
This paper proposes RoboTron-Sim, a framework that constructs a hard-case simulation dataset (HASS), introduces Scenario-aware Prompt Engineering (SPE) and an Image-to-Ego encoder (I2E), enabling MLLMs to effectively leverage simulated hard cases to improve real-world driving performance. On nuScenes hard scenarios, it achieves ~48% reduction in L2 distance and ~46% reduction in collision rate, establishing state-of-the-art open-loop planning performance.

## Background & Motivation

**State of the Field — Data Scarcity Bottleneck**: End-to-end autonomous driving systems are highly data-driven, yet real-world data for high-risk, long-tail scenarios (e.g., nighttime driving, heavy rain, pedestrian jaywalking) is extremely scarce. In nuScenes, the day-to-night ratio is approximately 7:1 and the straight-to-turn ratio approximately 8:1.

**Limitations of Prior Work — Sim2Real Gap**: Conventional approaches (e.g., VAD) that directly mix simulated and real data yield marginal gains — only ~1% improvement in L2 distance. The root cause is the inherent domain gap between simulated and real inputs (visual style, sensor parameters, coordinate systems, etc.), which impedes cross-domain knowledge transfer.

**New Opportunities and Challenges with MLLMs**: Multimodal large language models possess strong reasoning and generalization capabilities, showing early promise for cross-domain fusion (LLaVA-OneVision outperforms VAD in Sim2Real settings). However, geometric misalignment between simulated and real data continues to constrain performance.

**Root Cause — Core Research Question**: How can MLLMs effectively leverage simulation data to improve real-world autonomous driving? This work represents the first in-depth investigation of Sim2Real transfer limitations for MLLMs in autonomous driving.

## Method

### Overall Architecture
RoboTron-Sim comprises two main components: (1) a data layer — construction of the hard-case simulation dataset HASS; and (2) a model layer — an MLLM-based driving framework equipped with SPE and an I2E Encoder to bridge the Sim2Real gap.

### 1. HASS Dataset Construction

#### Scenario Categorization Strategy
- **Common scenarios** are divided into Easy-to-Drive (E2D, e.g., daytime straight driving) and Hard-to-Drive (H2D, e.g., nighttime, fog, heavy rain)
- **Long-tail scenarios**: extremely rare but high-risk events, covering 13 categories of edge cases (pedestrian jaywalking, sudden lane changes, wrong-way intrusion, road construction, etc.)
- H2D and long-tail scenarios are the primary targets for supplementary data generation

#### Data Generation
- Built on the CARLA simulator, using Think2Drive (a world-model-driven RL architecture) as the core data generation engine
- Sensor configuration aligned with nuScenes: six 900×1600 cameras providing 360° coverage
- Total of 47,553 simulated samples

#### Data Balancing
- Day/Night: 58.65% / 41.35% (real data: 87.97% / 12.03%)
- Clear/Rain: 48.38% / 51.61% (real data: 80.16% / 19.84%)
- Straight/Turn: 46.42% / 53.58% (real data: 88.86% / 11.14%)

#### Coordinate Alignment
- CARLA's left-handed coordinate system is converted to nuScenes' right-handed coordinate system
- The coordinate origin is unified to the vehicle roof center

### 2. Scenario-aware Prompt Engineering (SPE)

A structured environmental description is prepended to the input sequence: `"You are driving in [City Name] under [Simulation/Real-World] scenario."`

- **Domain awareness**: explicitly informs the model of the data source (simulation/real-world), making it aware of differences such as sensor noise
- **Geographic conditioning**: embeds city-name priors (e.g., traffic rules, left/right-hand driving conventions), activating commonsense knowledge embedded in the LLM to adaptively adjust driving strategies

### 3. Image-to-Ego Encoder (I2E Encoder)

- **Design Motivation**: Camera intrinsic and extrinsic parameters differ between simulated and real-world settings, creating a critical cross-domain geometric gap
- **Mechanism**: Camera intrinsics and extrinsics are used to compute the image-to-ego transformation matrix, which is mapped to an embedding space via a two-layer MLP to capture the spatial context of each viewpoint
- **Integration**: The encoded output is concatenated with text tokens, enabling the model to directly incorporate spatial reasoning into the decision-making process

### 4. MLLM Baseline Architecture

- Visual feature extractor → two-layer MLP projector → LLM decoder (based on LLaVA-OneVision)
- Input: 6 cameras × 5 consecutive frames + high-level instructions (e.g., "turn left at the next intersection")
- Output: future trajectory waypoints + predicted vehicle speed
- Velocity supervision is introduced to enhance ego-state awareness

## Key Experimental Results

### Main Results — Open-Loop Planning on nuScenes (Tab. 3)

| Setting | Method | L2 (m) ↓ | Collision (%) ↓ | Out-of-bound (%) ↓ |
|---------|--------|-----------|-----------------|---------------------|
| w/o ego pose | OmniDrive | 0.84 | 0.94 | 4.29 |
| w/o ego pose | **RoboTron-Sim** | **0.56** | **0.58** | **3.02** |
| w/ ego pose | EMMA | 0.32 | — | — |
| w/ ego pose | OmniDrive | 0.33 | 0.30 | 3.00 |
| w/ ego pose | **RoboTron-Sim** | **0.23** | **0.26** | **2.62** |

### Scenario-Specific Improvements (Tab. 4, L2 Distance)

| Scenario | nuScenes Only | +HASS | Improvement |
|----------|--------------|-------|-------------|
| Night (H2D) | 1.40 | 0.81 | ↓42.1% |
| Turn (H2D) | 1.32 | 0.64 | ↓51.5% |
| Rain (H2D) | 1.15 | 0.56 | ↓51.3% |
| Day (E2D) | 0.59 | 0.54 | ↓8.5% |

### Ablation Study (Tab. 6)

| SPE | I2E | L2 (m) ↓ | Collision (%) ↓ | Out-of-bound (%) ↓ |
|-----|-----|-----------|-----------------|---------------------|
| ✗ | ✗ | 0.91 | 0.94 | 3.22 |
| ✓ | ✗ | 0.86 | 0.79 | 2.68 |
| ✓ | ✓ | 0.56 | 0.58 | 3.02 |

### Data Efficiency
- Using only 20% real data + HASS matches the performance of 100% real data
- Training on purely simulated data (0% real) still yields a reasonable L2 of 1.24 m

### HASS vs. GASS Comparison (Tab. 10)
- GASS (synthesized following nuScenes distribution): H2D L2 = 1.07 m
- HASS (hard-case targeted synthesis): H2D L2 = 0.67 m (↓37.4%), collision rate reduced from 1.74% to 0.96%

### Cross-Benchmark Generalization (NAVSIM)
- RoboTron-Sim + HASS achieves PDMS = 85.6, establishing state of the art on NAVSIM

### Deployment Efficiency
- RoboTron-Sim-7B latency: 612.8 ms
- RoboTron-Sim-0.5B latency: 141.4 ms, comparable to VAD (115.3 ms) with competitive performance

## Highlights & Insights

1. **First systematic study of Sim2Real for MLLMs**: This work is the first to thoroughly investigate the limitations of and solutions for MLLM-based utilization of simulation data in autonomous driving, addressing an important gap in the literature.

2. **Targeted hard-case synthesis**: HASS does not synthesize data uniformly but specifically supplements H2D and long-tail scenarios. The GASS comparison experiment clearly demonstrates the value of the targeted strategy — H2D improvement jumps from ~22% to ~48%.

3. **Elegance of SPE**: Rather than employing complex domain adaptation networks, domain awareness is achieved with a single line of text prompt. This leverages commonsense knowledge already embedded in the LLM (e.g., traffic rules in different cities), representing a lightweight Sim2Real solution uniquely suited to the MLLM era.

4. **I2E Encoder decouples sensor configuration**: By explicitly injecting geometric transformation matrices, the model is freed from dependence on a specific sensor configuration, contributing an additional 34.9% reduction in L2 distance — the largest single performance gain.

5. **Remarkable data efficiency**: 20% real data + HASS ≈ 100% real data, which has significant practical implications for reducing costly real-world data collection.

## Limitations & Future Work

1. **Open-loop evaluation only**: All experiments are conducted in the nuScenes open-loop setting; no closed-loop testing (e.g., CARLA Leaderboard) is performed. Open-loop metrics are known to have a significant gap with real driving performance.

2. **Inherent limitations of CARLA**: HASS relies on CARLA, whose visual fidelity remains limited. Performance may improve further with next-generation simulation engines (e.g., Unreal Engine 5).

3. **Limited long-tail coverage**: Only 13 categories of edge cases are included, whereas real-world long-tail distributions are far more complex. Automatically discovering and generating new hard scenarios remains an open problem.

4. **Hard-coded SPE format**: The prompt template is manually designed with a fixed format; learnable prompts or more flexible domain description strategies are not explored.

5. **High inference latency**: The 7B model incurs 612.8 ms latency, falling short of the real-time autonomous driving requirement (≤100 ms). Although the 0.5B variant approaches VAD's latency, it comes with a performance trade-off.

6. **Single simulator source**: Only CARLA is used; combinations of multiple simulators or neural rendering approaches for training data generation are not explored.

## Related Work & Insights

- **UniAD/VAD**: End-to-end autonomous driving baselines with limited capacity for leveraging simulation data
- **EMMA (Google)**: A multimodal end-to-end autonomous driving model achieving L2 = 0.32 m with ego pose
- **OmniDrive**: A full-stack framework leveraging LLMs for 3D perception, reasoning, and planning
- **LLaVA-OneVision**: The base model underlying the MLLM backbone in this work
- **Think2Drive**: A world-model-based RL driving agent used for HASS data collection
- **Senna/DriveVLM**: Autonomous driving systems combining MLLMs with end-to-end models
- **Key Insight**: Prompt engineering for MLLMs can serve as a lightweight domain adaptation strategy; explicit injection of geometric information is highly effective in cross-domain settings

## Rating
- Novelty: ⭐⭐⭐⭐ (Sim2Real from the MLLM perspective is a novel angle; SPE and I2E designs are conceptually clear)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive ablations, multi-benchmark validation, data efficiency analysis, deployment cost, VQA generalization)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, well-motivated, though the abundance of tables somewhat affects readability)
- Value: ⭐⭐⭐⭐ (significant reference value for the Sim2Real + MLLM direction; practical utility pending closed-loop validation)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LookOut: Real-World Humanoid Egocentric Navigation](lookout_real-world_humanoid_egocentric_navigation.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [\[CVPR 2026\] SimScale: Learning to Drive via Real-World Simulation at Scale](../../CVPR2026/autonomous_driving/simscale_learning_to_drive_via_real-world_simulation_at_scale.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)

<!-- RELATED:END -->
