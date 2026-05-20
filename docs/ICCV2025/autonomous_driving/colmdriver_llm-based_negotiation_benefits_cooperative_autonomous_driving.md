---
title: >-
  [Paper Note] CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][cooperative driving] The first end-to-end LLM-driven cooperative driving system. Through an Actor-Critic language negotiation module and an intention-guided trajectory generator…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "cooperative driving"
  - "V2V"
  - "LLM negotiation"
  - "actor-critic"
  - "waypoint planning"
  - "CARLA"
date: 2026-05-08
content_hash: 160b0ae6f1d64db3
---

# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2503.08683](https://arxiv.org/abs/2503.08683)  
**Code**: [cxliu0314/CoLMDriver](https://github.com/cxliu0314/CoLMDriver)  
**Area**: Autonomous Driving
**Keywords**: cooperative driving, V2V, LLM negotiation, actor-critic, waypoint planning, CARLA

## TL;DR

The first end-to-end LLM-driven cooperative driving system. Through an Actor-Critic language negotiation module and an intention-guided trajectory generator, it achieves an 11% higher success rate than existing methods across diverse V2V interaction scenarios.

## Background & Motivation

**Core Problem**: Vehicle-to-vehicle (V2V) cooperative autonomous driving can compensate for the perception and prediction uncertainties of single-vehicle systems by sharing sensory information, yet faces two key challenges:

**Limitations of Prior Work**: Existing V2V cooperative methods (e.g., CoDriving) rely on rigid cooperation protocols whose predefined interaction rules generalize poorly to unseen complex scenarios. In non-standard situations such as multi-vehicle intersections or emergency avoidance, fixed rules lack flexibility.

**Difficulties of Applying LLMs Directly to Driving**: Although LLMs possess general reasoning capabilities, two obstacles remain: (a) insufficient spatial planning ability — LLMs cannot directly output precise trajectory coordinates; (b) unstable inference latency — real-time driving demands extremely tight latency bounds, and variable LLM inference times cause control instability.

**Key Insight**: Rather than forcing LLMs to perform spatial planning, the paper leverages LLMs for what they excel at — language-based negotiation. The driving system is decomposed into two parallel pipelines: an LLM negotiates cooperative intentions via natural language (e.g., "I will pass first; you slow down and yield"), and a dedicated trajectory generator then translates the negotiation outcome into executable waypoints.

## Method

### Overall Architecture

CoLMDriver adopts a **parallel driving pipeline** architecture consisting of three modules:

1. **Cooperative Perception Module**: BEV feature fusion based on V2Xverse/CoDriving; multiple vehicles share LiDAR point clouds and extract voxel features via SpConv to produce a unified bird's-eye-view (BEV) representation.
2. **LLM-based Negotiation Module**: Employs an Actor-Critic paradigm in which vehicles conduct multi-round natural language negotiation.
3. **Intention-Guided Waypoint Generator**: Converts the negotiated driving intention into 20 future waypoints.

Collaborative workflow: the perception module provides environmental understanding → a VLM analyzes the scene and generates driving intentions → the LLM negotiation module coordinates intentions across vehicles → the waypoint generator outputs executable trajectories → a PID controller executes the plan.

### Key Designs 1: LLM Negotiation Module (Actor-Critic Paradigm)

This is the paper's central contribution, modeling multi-vehicle negotiation as an Actor-Critic iterative process:

**Actor (Negotiation Generation)**: Each vehicle hosts an LLM agent (Comm_Client) whose inputs include:
- Ego-vehicle information: ID, intention (e.g., "turn left"), speed
- Surrounding vehicle information: heading, course, distance, speed, intention
- Traffic rules (right-of-way, yielding to emergency vehicles, etc.)
- Historical negotiation records and Critic feedback

The LLM generates natural-language negotiation proposals, e.g., "I will slow down to let Vehicle 2 pass first, then proceed with my left turn."

**Critic (Negotiation Evaluation)**: Implemented by Nego_Client, evaluating negotiation quality along three dimensions:

| Evaluation Dimension | Computation | Weight |
|----------------------|-------------|--------|
| Consensus Score | LLM analyzes dialogue content, outputs a 0–100 consensus score | 3 |
| Safety Score | Based on minimum distance between predicted trajectory pairs using a sigmoid: $\frac{1}{1+e^{-2(d_{min}-3)}}$ | 5 |
| Efficiency Score | Evaluates whether each vehicle travels efficiently based on trajectory length | 2 |

Total score: $S_{total} = 3 \cdot S_{cons} + 5 \cdot \min(S_{safety}) + 2 \cdot S_{eff}$

**Iterative Feedback**: When the safety score < 0.7, safety suggestions are generated (e.g., "Vehicle X and Vehicle Y may conflict; one should yield"); when the efficiency score < 0.4, efficiency suggestions are generated (e.g., "Some vehicles can accelerate"). These suggestions are injected as Critic Feedback into the next Actor prompt. A maximum of 3 rounds (`local_max_round=3`) are performed, with early termination upon consensus.

### Key Designs 2: VLM Scene Understanding

A vision-language model (VLM) processes front-camera images together with a system prompt (containing vehicle measurements, communication information, and perception results) to output structured driving commands:
- **Directional commands** (6 classes): `turn left / turn right / straight / lane follow / lane change left / lane change right`
- **Speed commands** (4 classes): `Stop / Slow down / Hold / Accelerate`

The VLM is fine-tuned via LoRA (using the ms-swift framework) on data from the V2Xverse simulation dataset.

### Key Designs 3: Intention-Guided Waypoint Generator

The core model is `WaypointPlanner_e2e_cmd_attn_fix_20points`, with the following architecture:

**Input Fusion**:
- Occupancy map: 6-channel representation including obstacle and road information, shape `(B, T=5, C=6, H=192, W=96)`
- BEV perception features: 128-dimensional feature map from the cooperative perception module
- Navigation commands: directional embedding (Embedding, 6→256) + speed embedding (Embedding, 4→256) + target point encoding (MLP, 2→256)

**Encoder**: Spatio-temporal convolutional (STC) blocks alternating 2D convolution and Conv3D for spatial and temporal modeling:
- STC Block 1: Conv2D(32→64) → Conv3D(64→64) (temporal fusion 5→3 frames)
- STC Block 2: Conv2D(64→128) → Conv3D(128→128) (temporal fusion 3→1 frame)
- Spatial features further convolved to 256 dimensions

**Decoder**: Cross-attention mechanism that interacts navigation command features with BEV spatial features:
- Constructs query embeddings
- Alternates between command-to-query attention and spatial-to-query attention
- Final regression branch outputs 20 waypoints `(B, 20, 2)`, with cumulative summation enforcing trajectory continuity

### Loss & Training

A weighted L1 loss (WaypointL1Loss20) supervises trajectory prediction, applying decreasing weights across time steps (higher weight for near-term, lower for far-term), ranging from 0.14 to 0.04. ADE/FDE serve as auxiliary evaluation metrics.

### InterDrive Benchmark

The authors introduce **InterDrive**, a V2V interactive driving evaluation benchmark based on CARLA 0.9.10.1:
- **10 challenging interaction scenario types**: covering multi-vehicle intersection crossing, lane merging conflicts, emergency avoidance, etc.
- **92 test cases**: 46 cooperative-vehicle-only scenarios (no NPC) + 46 scenarios with background traffic participants (with NPC)
- **Evaluation metrics**: Success Rate, collision rate, timeout rate, etc.
- Supports two evaluation modes: ideal (ignoring inference latency) and realtime (accounting for inference latency)

## Key Experimental Results

### Main Results

Comparison against multiple baselines on the InterDrive benchmark:

| Method | Type | Success Rate (↑) |
|--------|------|-----------------|
| TCP | Single-vehicle end-to-end | baseline |
| CoDriving | V2V cooperative (rule-driven) | baseline |
| LMDrive | LLM-based driving (single vehicle) | baseline |
| UniAD | End-to-end planning | baseline |
| VAD | End-to-end planning | baseline |
| **CoLMDriver** | **V2V + LLM negotiation** | **+11% vs. best baseline** |

CoLMDriver substantially outperforms all methods in high-interaction V2V scenarios, demonstrating the effectiveness of language negotiation for cooperative driving.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o negotiation | Significant performance drop; validates the necessity of the negotiation module |
| w/o Critic feedback | Single-round negotiation only; performance drops, validating the value of iterative feedback |
| w/o VLM intention | Removing scene understanding and using rule-based commands instead; performance drops |
| Varying negotiation rounds | 3 rounds achieves a favorable trade-off |

### Key Findings

1. **Negotiation quality correlates positively with driving safety**: The Critic feedback mechanism elevates safety scores from the unstable levels of single-round negotiation to consistently high values.
2. **NPC scenarios are more challenging**: Success rates in scenarios with background traffic are generally lower than in cooperative-only scenarios.
3. **Impact of inference latency**: Performance degrades in realtime mode, but CoLMDriver's parallel pipeline design effectively mitigates this — negotiation and trajectory planning proceed asynchronously.

## Highlights & Insights

1. **"Let LLMs do what they are good at"**: Rather than forcing LLMs to output precise coordinates, the system assigns them natural-language negotiation while a dedicated model converts the outcome into trajectories. This division of labor is elegant.

2. **Actor-Critic negotiation paradigm**: The Actor-Critic concept from reinforcement learning is transferred to multi-vehicle negotiation. The Critic evaluates along three dimensions — safety, efficiency, and consensus — and automatically generates improvement suggestions, realizing interpretable closed-loop optimization.

3. **Clever Critic design**: The safety score is based on a sigmoid function of minimum inter-trajectory distance, $\frac{1}{1+e^{-2(d-3)}}$, where the 3-meter threshold corresponds to the safe following distance, making the physical interpretation explicit. The consensus score leverages the LLM's own comprehension ability to assess dialogue quality.

4. **Fully open-sourced**: The complete codebase — from perception to planning to negotiation — along with model weights, training scripts, and the InterDrive benchmark, is fully released, representing a substantial contribution to the community.

5. **Parallel pipeline**: The negotiation process runs in parallel with perception and planning, effectively alleviating the impact of LLM inference latency on real-time performance.

## Limitations & Future Work

1. **Simulation-to-real gap**: Experiments are conducted entirely in the CARLA simulator; real-world issues such as V2V communication delay, bandwidth constraints, and signal loss are not addressed.

2. **High LLM inference cost**: Each negotiation step requires multiple LLM calls (one per vehicle per round plus Critic evaluation); three rounds of negotiation involve substantial API calls, making deployment cost and latency ongoing challenges.

3. **Strong negotiation assumptions**: The framework assumes all cooperative vehicles honestly participate in negotiation and accurately report their intentions. Adversarial problems such as malicious vehicles and communication spoofing in real-world settings are not addressed.

4. **Limited scenario scale**: The InterDrive benchmark only covers interactions among 2–3 cooperative vehicles; the negotiation efficiency and scalability for larger platoons remain unvalidated.

5. **VLM relies on a single-view image**: Scene understanding is based solely on the front-facing camera, potentially missing lateral and rear information.

## Related Work & Insights

- **V2Xverse / CoDriving**: Foundation for the perception module, providing the V2V cooperative perception framework and simulation dataset.
- **TCP (Trajectory-guided Control Prediction)**: Serves as the single-vehicle end-to-end driving baseline.
- **LMDrive**: A pioneering work applying LLMs to single-vehicle driving, but lacking V2V cooperative capability.
- **Bench2Drive**: A CARLA driving evaluation benchmark that partially informs the paper's evaluation framework.

**Insights**:
- Natural language negotiation in multi-agent cooperation is a promising direction, extensible to robot formation control, drone swarms, and related domains.
- The Actor-Critic negotiation paradigm can be generalized to any scenario requiring multi-party consensus (e.g., multi-robot task allocation).
- Decoupling LLM language reasoning from domain-specific model precise control constitutes a broadly applicable system design principle.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | ⭐⭐⭐⭐ | First end-to-end LLM cooperative driving system; Actor-Critic negotiation paradigm is original |
| Technical Depth | ⭐⭐⭐⭐ | Full-stack perception–negotiation–planning design with well-reasoned inter-module coupling |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Introduces a new benchmark, compares against multiple baselines, and provides complete ablation studies |
| Writing Quality | ⭐⭐⭐⭐ | Problem formulation is clear; method description is detailed |
| Practical Value | ⭐⭐⭐ | Simulation results are convincing, but substantial work remains for real-world deployment |
| **Overall** | **⭐⭐⭐⭐** | **An important contribution to cooperative driving and a valuable exploration of LLM integration with autonomous driving** |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
