---
title: >-
  [Paper Note] C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination
description: >-
  [CVPR 2026][Autonomous Driving][Traffic Signal Control] This paper proposes the C2T framework, which converts traffic states into structured captions, leverages LLMs for offline preference judgments, and distills these judgments into an intrinsic reward function. This approach replaces hand-crafted rewards for traffic signal control (TSC) and achieves improvements in efficiency, safety, and energy consumption across multiple real-world urban networks on the CityFlow benchmark.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Traffic Signal Control
  - Multi-Agent Reinforcement Learning
  - LLM Preference Learning
  - Intrinsic Reward
  - Commonsense Reasoning
date: 2026-05-08
content_hash: 557b8466f4b33834
---

# C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination

**Conference**: CVPR 2026
**arXiv**: [2604.13098](https://arxiv.org/abs/2604.13098)
**Code**: N/A
**Area**: Autonomous Driving / Traffic Control
**Keywords**: Traffic Signal Control, Multi-Agent Reinforcement Learning, LLM Preference Learning, Intrinsic Reward, Commonsense Reasoning

## TL;DR

This paper proposes the C2T framework, which converts traffic states into structured captions, leverages LLMs for offline preference judgments, and distills these judgments into an intrinsic reward function. This approach replaces hand-crafted rewards for traffic signal control (TSC) and achieves improvements in efficiency, safety, and energy consumption across multiple real-world urban networks on the CityFlow benchmark.

## Background & Motivation

**Background**: MARL-based TSC optimizes local efficiency using hand-crafted rewards such as queue length, intersection pressure, and average delay.

**Limitations of Prior Work**: Hand-crafted rewards are myopic, local proxy metrics that fail to capture higher-level human-centric objectives such as safety, flow stability, and riding comfort. Aggressive intersection clearing can induce oscillating signals, hard braking, and unsafe headways—yielding systems that are efficient on paper but fragile in deployment.

**Key Challenge**: There is no notion of "traffic quality" that reflects human judgment and anticipates long-term effects such as platoon formation, and this must be achieved without modifying the simulator or invoking LLMs online.

**Goal**: To use traffic quality itself as a supervisory signal by learning an intrinsic reward offline from LLM preferences, thereby augmenting the standard MARL training pipeline.

**Key Insight**: LLMs can produce consistent pairwise judgments when comparing well-structured state descriptions, making them a viable source of commonsense knowledge.

**Core Idea**: Render traffic states as deterministic, unit-aware captions → obtain pairwise LLM preference labels → train a lightweight preference scorer → inject intrinsic rewards into standard PPO.

## Method

### Overall Architecture

Three stages: (1) **Stage 1**: Render traffic observations into structured captions using a deterministic schema; (2) **Stage 2**: Sample caption pairs, query an LLM for preference labels, and train a Bradley-Terry preference scorer as the intrinsic reward; (3) **Stage 3**: Asymmetrically blend the intrinsic reward into PPO training for the traffic light controller (TLC)—only the TLC receives the blended reward, while vehicles use only environment rewards—combined with a risk mask and a scheduling strategy.

### Key Designs

1. **Deterministic Structured Traffic Captions**:

    - **Function**: Convert traffic states into representations on which LLMs can render consistent judgments.
    - **Mechanism**: A deterministic, unit-aware schema enumerates key variables (queue, delay, TTC, violations, etc.) with explicit semantics and numerical values, eliminating the ambiguity and stylistic variation inherent in free-text descriptions.
    - **Design Motivation**: LLM judgments over free text are unstable, whereas structured quantitative descriptions elicit consistent preferences.

2. **Offline Preference Learning and Intrinsic Reward**:

    - **Function**: Distill LLM commonsense judgments into a reusable reward function.
    - **Mechanism**: Decisive LLM labels are collected from caption pairs (ambiguous judgments are discarded), and a lightweight scorer is trained via Bradley-Terry likelihood: $r_\phi(o) = f_\phi(\text{tok}(c), x(o))$. Optional safety/efficiency/energy heads are supported. All supervision is generated and cached offline; switching prompts or enabling different heads requires only swapping the cache.
    - **Design Motivation**: Avoid the latency, reliability, and scalability issues of online LLM invocation.

3. **Safety Risk Mask and Asymmetric Integration**:

    - **Function**: Enforce safety constraints and stabilize training.
    - **Mechanism**: The risk mask suppresses the intrinsic signal when low TTC percentiles, hard-braking clusters, or red-light violations are detected. The intrinsic reward is blended only into the TLC objective (not into vehicle rewards), since phase selection is the primary lever for shaping platoons and network rhythm. A scheduling strategy is applied: environmental constraints are satisfied first, after which commonsense preferences are gradually absorbed.
    - **Design Motivation**: Providing intrinsic signals to both parties introduces additional non-stationarity; safety constraints must take precedence over efficiency optimization.

### Loss & Training

Preference learning: weighted negative log-likelihood + L2 regularization + score centering. RL training: standard PPO with independent normalization and soft clipping per reward stream before weighted blending.

## Key Experimental Results

### Main Results

| Method | Jinan Travel Time↓ | Hangzhou Travel Time↓ | New York Travel Time↓ | TTC p10↑ |
|---|---|---|---|---|
| PressLight | 285.3s | 312.7s | 298.5s | 3.2s |
| CoLight | 278.1s | 305.2s | 291.3s | 3.5s |
| Advanced-CoLight | 272.5s | 298.8s | 285.7s | 3.8s |
| **C2T** | **265.2s** | **289.5s** | **278.1s** | **4.5s** |

### Ablation Study

| Configuration | Travel Time↓ | TTC p10↑ | Note |
|---|---|---|---|
| Full C2T | 265.2s | 4.5s | Intrinsic reward + safety mask |
| w/o intrinsic reward | 278.1s | 3.5s | Environment reward only |
| w/o safety mask | 268.5s | 3.8s | Intrinsic reward without safety constraint |
| Caption only (no numerics) | 270.3s | 4.2s | Structured caption alone is beneficial |

### Key Findings

- The intrinsic reward signal contributes most substantially (removing it increases travel time by 13 seconds); the safety mask is critical for TTC improvement.
- Structured captions alone improve performance; incorporating matched numerical values yields further, albeit smaller, gains.
- C2T's flexibility is demonstrated by its ability to generate "efficiency-first" vs. "safety-first" policies simply by switching prompts.

## Highlights & Insights

- Repositioning LLMs from online decision-makers to offline reward designers is a principled role assignment: it avoids the latency and reliability issues associated with LLMs in the control loop.
- The deterministic schema caption design makes LLM judgments reproducible and cacheable, representing a critical engineering decision.
- Asymmetric integration—providing intrinsic rewards only to the TLC—reduces multi-agent non-stationarity.

## Limitations & Future Work

- Validation is conducted on the CityFlow simulator; real-world deployment performance remains unknown.
- LLM preferences may encode implicit biases.
- The work focuses solely on traffic signal control and does not extend to explicitly modeled autonomous vehicles.
- The framework could be extended to more urban networks and adverse weather conditions.

## Related Work & Insights

- **vs. LLMLight**: LLMLight places LLMs directly in the control loop, incurring latency and reliability penalties; C2T's offline distillation avoids these issues.
- **vs. CoTV**: CoTV employs hand-crafted composite rewards (travel time + fuel + emissions); C2T learns a more comprehensive reward from LLM preferences.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Offline LLM preference learning for traffic reward design is a novel direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three cities + stress tests + detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly presented.
- **Value**: ⭐⭐⭐⭐ Broadly instructive for RL reward design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[AAAI 2026\] Generalising Traffic Forecasting to Regions without Traffic Observations](../../AAAI2026/autonomous_driving/generalising_traffic_forecasting_to_regions_without_traffic_observations.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](ttsg_text_to_traffic_scene_generation_from_natural_language.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](../../ICCV2025/autonomous_driving/foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[AAAI 2026\] Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](../../AAAI2026/autonomous_driving/unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)

<!-- RELATED:END -->
