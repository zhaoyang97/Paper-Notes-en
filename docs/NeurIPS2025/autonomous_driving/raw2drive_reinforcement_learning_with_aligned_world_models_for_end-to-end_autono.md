---
title: >-
  [Paper Note] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][Autonomous Driving][model-based reinforcement learning] This paper proposes RAW2Drive, the first model-based reinforcement learning (MBRL) end-to-end autonomous driving framework operating directly from raw sensor inputs to planning. Through a dual-stream world model design — first training a privileged world model, then guiding a raw-sensor world model via an alignment mechanism — RAW2Drive achieves state-of-the-art performance on CARLA v2 and Bench2Drive, substantially outperforming imitation learning (IL) methods.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - model-based reinforcement learning
  - world model
  - end-to-end driving
  - CARLA
  - dual-stream
date: 2026-05-08
content_hash: 4bc3258681ae017c
---

# RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving

**Conference**: NeurIPS 2025
**arXiv**: [2505.16394](https://arxiv.org/abs/2505.16394)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: model-based reinforcement learning, world model, end-to-end driving, CARLA, dual-stream

## TL;DR

This paper proposes RAW2Drive, the first model-based reinforcement learning (MBRL) end-to-end autonomous driving framework operating directly from raw sensor inputs to planning. Through a dual-stream world model design — first training a privileged world model, then guiding a raw-sensor world model via an alignment mechanism — RAW2Drive achieves state-of-the-art performance on CARLA v2 and Bench2Drive, substantially outperforming imitation learning (IL) methods.

## Background & Motivation

The dominant paradigm in end-to-end autonomous driving (E2E-AD) is imitation learning (IL), which faces two fundamental limitations: **causal confusion** (the model associates actions with spurious causes) and **distribution shift** (poor generalization to unseen scenarios).

Reinforcement learning (RL) is a promising alternative, yet applying it to E2E-AD remains highly challenging:
- **Low sample efficiency of model-free RL**: MaRLn requires approximately 50 million steps (~57 days of training) and still falls well short of IL performance.
- **Input gap for MBRL**: Think2Drive demonstrates that MBRL can solve CARLA v2, but relies on **privileged information** (GT bounding boxes, HD maps). Raw sensor data is high-dimensional, redundant, and noisy, making direct world model training extremely difficult.

At the time of this work, **no RL-based end-to-end method** existed for CARLA v2 — the gap that this paper addresses.

## Method

### Overall Architecture

RAW2Drive is a dual-stream MBRL framework:
- **Stream I (Privileged Stream)**: Trains a world model and associated policy using privileged inputs such as BEV semantic masks (following a DreamerV3-like design).
- **Stream II (Raw Sensor Stream)**: Uses multi-view images and IMU as inputs, training both a world model and an end-to-end policy under the guidance of the privileged stream.
- At inference time, only raw sensor inputs are used.

### Key Designs

1. **Privileged Stream**:

    - Input: Temporally stacked BEV semantic masks (standard MBRL input).
    - World model: Encoder + RSSM + three heads (Reward / Decoder / Continue), identical in architecture to DreamerV3.
    - Policy: Actor-Critic networks trained via world model rollouts.
    - Purpose: (I) Train a privileged policy; (II) Provide guidance to the raw sensor stream.

2. **Raw Sensor Stream**:

    - Encoder: BEVFormer encodes multi-view images into BEV features.
    - World model: Architecture mirrors the privileged stream, but uses **only the Decoder Head** — reward and continue signals are obtained from the privileged stream.
    - Key finding: Directly training reward/continue heads causes convergence failure, as adjacent frames are highly similar while reward values can fluctuate sharply, preventing the network from learning stable patterns.
    - RSSM parameters are initialized from the privileged world model.

3. **Guidance Mechanism** — the core of the dual-stream design:

   **Rollout Guidance**: Ensures consistent predictions between the two world models during rollouts.
    - **Spatial-Temporal Alignment**: Applies MSE loss on encoder states to enforce spatial consistency at each timestep.
    - **Abstract-State Alignment**:
      - Deterministic state $h$: L2 loss to maintain consistent predictions (modeling ego-vehicle state).
      - Stochastic state $s$: KL divergence to constrain distributional consistency (modeling behaviors of other traffic participants).
    - **Eliminating Cumulative Sampling Error**: The stochastic state is sampled once from the raw sensor stream's distribution and fed into both streams simultaneously, preventing divergence caused by independent sampling.

   **Head Guidance**: During raw sensor policy training, reward and continue flags are taken directly from the privileged world model rather than the raw sensor stream's own heads, ensuring accurate and stable supervision signals.

   Total Rollout Loss:
    $\mathcal{L}_{Rollout} = \beta_e \sum_t \sum_{i} \text{MSE}(e_t^i, \hat{e}_t^i) + \sum_t (\beta_s \text{KL}(s_t, \hat{s}_t) + \beta_h \text{MSE}(h_t, \hat{h}_t))$

### Loss & Training

Two-phase training:
- **Phase I**: Train the privileged world model and policy (results from Think2Drive can be reused, saving ~24 GPU days).
- **Phase II**: Train the raw sensor world model and policy under privileged stream guidance.

Total training cost: 64 H800 GPU days (only 40 when reusing Think2Drive), compared to UniAD's ~30 GPU days, which handles only 3–4 corner case types.

## Key Experimental Results

### Main Results

Bench2Drive closed-loop evaluation:

| Method | Paradigm | DS ↑ | SR (%) ↑ | Efficiency ↑ | Comfort ↑ |
|--------|----------|------|----------|-------------|-----------|
| UniAD-Base | IL | 45.81 | 16.36 | 129.21 | 43.58 |
| DriveAdapter* | IL | 64.22 | 33.08 | 70.22 | 16.01 |
| DriveTrans | IL | 63.46 | 35.01 | 100.64 | 20.78 |
| **RAW2Drive** | **RL** | **71.36** | **50.24** | 214.17 | 22.42 |
| Think2Drive | RL (Privileged) | 91.85 | 85.41 | 269.14 | 25.97 |

Multi-ability evaluation:

| Method | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|--------|---------|-----------|----------------|----------|-------------|------|
| UniAD | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| DriveTrans | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| **RAW2Drive** | **43.35** | **51.11** | **60.00** | **50.00** | **62.26** | **53.34** |

### Ablation Study

| Configuration | DS ↑ | SR | Notes |
|---------------|------|----|-------|
| No Decoder Head | 17.4 | 1.2/10 | No effective supervision |
| Decoder Head only | **83.5** | **7.5/10** | Best configuration |
| Decoder + Reward | 46.6 | 3.4/10 | Reward head introduces noise |
| Decoder + Reward + Continue | 34.5 | 2.2/10 | Further degradation |
| No spatial alignment | 9.24 | 0.8/10 | Equivalent to "blind driving" |
| No temporal alignment | 13.6 | 1.2/10 | Inconsistent future predictions |
| Full alignment | **83.5** | **7.5/10** | All three components are indispensable |

### Key Findings

- **RL substantially outperforms IL**: RAW2Drive achieves markedly higher DS and SR than all end-to-end IL methods (DS: 71.36 vs. DriveTrans 63.46).
- **Reward/Continue heads are detrimental**: Directly training these heads in the raw sensor stream degrades performance (DS drops from 83.5 to 34.5).
- **All three alignment components are indispensable**: Removing encoder, deterministic, or stochastic state alignment individually reduces the model to handling only simple straight-line driving.
- **RSSM and Decoder parameter sharing is beneficial**: It promotes more efficient representation learning.

## Highlights & Insights

- Core contribution: Demonstrates that MBRL can solve CARLA v2 using only raw sensor inputs, filling a critical gap in the field.
- Philosophy of the dual-stream design: Privileged information provides a low-dimensional, structured shortcut — by first building a well-trained world model from it, the framework can then guide the high-dimensional sensor stream to learn decision-relevant representations.
- The technique for eliminating cumulative sampling error is simple yet critical: sampling the stochastic state only once from the raw sensor stream prevents the two streams from diverging.
- Training cost of only 64 H800 GPU days demonstrates practical engineering viability.

## Limitations & Future Work

- A significant gap remains relative to the privileged method Think2Drive (DS 91.85), indicating that the expressive capacity of the raw-sensor world model is still limited.
- Evaluation is conducted solely in the CARLA simulator; performance on real-world data remains unvalidated.
- The BEVFormer encoder is computationally heavy, and inference latency may not satisfy real-time requirements.
- Although the privileged stream is unused at inference time, privileged information is required during training, which precludes direct training on real-world data.

## Related Work & Insights

- Relationship to Think2Drive: Think2Drive applies MBRL with privileged inputs; RAW2Drive extends it to raw sensor inputs, representing a natural progression.
- Relationship to DreamerV3: The world model architecture builds upon DreamerV3, augmented with the dual-stream design and guidance mechanism.
- Insight: The paradigm of transferring from privileged to end-to-end (learning first with simple inputs, then transferring to complex inputs) may generalize to other domains such as robotic manipulation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First MBRL E2E-AD system from raw sensor inputs; the dual-stream guidance mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two benchmarks (CARLA v2 + Bench2Drive) with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and method motivation is well articulated.
- Value: ⭐⭐⭐⭐⭐ Establishes the feasibility and superiority of RL for E2E-AD, with important implications for both academia and industry.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)

<!-- RELATED:END -->
