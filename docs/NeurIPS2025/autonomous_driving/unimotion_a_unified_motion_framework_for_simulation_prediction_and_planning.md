---
title: >-
  [Paper Note] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning
description: >-
  [NeurIPS 2025][Autonomous Driving][Motion Prediction] UniMotion proposes a unified motion framework built on a decoder-only Transformer, supporting motion simulation, trajectory prediction…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Motion Prediction"
  - "Trajectory Simulation"
  - "Autonomous Driving Planning"
  - "Unified Framework"
  - "GPT-style Model"
date: 2026-05-08
content_hash: 2e8966c34aaae231
---

# UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning

**Conference**: NeurIPS 2025  
**arXiv**: [2602.00566](https://arxiv.org/abs/2602.00566)  
**Code**: [https://github.com/LogosRoboticsGroup/UniMotion](https://github.com/LogosRoboticsGroup/UniMotion)  
**Area**: Autonomous Driving  
**Keywords**: Motion Prediction, Trajectory Simulation, Autonomous Driving Planning, Unified Framework, GPT-style Model

## TL;DR

UniMotion proposes a unified motion framework built on a decoder-only Transformer, supporting motion simulation, trajectory prediction, and ego-vehicle planning simultaneously through task-aware interaction patterns and training strategies. Joint training facilitates cross-task knowledge sharing, and after task-specific fine-tuning, the model achieves state-of-the-art performance across multiple tasks on the Waymo dataset.

## Background & Motivation

Motion understanding in autonomous driving encompasses three core tasks: motion simulation (generating diverse agent behaviors), trajectory prediction (forecasting future trajectories of traffic participants), and ego-vehicle planning (generating feasible driving trajectories for the ego vehicle). Although these tasks fundamentally share the same underlying capabilities—understanding multi-agent interactions, modeling motion behaviors, and reasoning about spatiotemporal dynamics—existing methods typically design task-specific models for each, which impedes cross-task generalization and system scalability.

More critically, this task isolation neglects the mutual benefits across tasks. For instance, the diversity-oriented objective of simulation can provide broader reference directions for prediction, while the long-range accuracy requirement of prediction can enhance the motion plausibility of simulation.

This paper abstracts motion tasks into two fundamental categories: **diverse motion generation** (corresponding to simulation) and **long-range trajectory prediction** (corresponding to prediction and planning), and constructs the model upon this unified conceptual framework. A decoder-only Transformer is chosen for its simplicity, scalability, and general generative capability across tasks.

## Method

### Overall Architecture

UniMotion adopts a decoder-only Transformer as its backbone, consisting of:
1. **Input Representation**: Agent trajectories and HD maps are discretized into token sequences via tokenization (2,048 agent tokens + 1,024 map tokens); trajectories are segmented at fixed time intervals and clustered to construct the token vocabulary.
2. **Scene Context Encoding**: Stacked self-attention modules encode map embeddings.
3. **Motion Decoding**: A factorized attention mechanism (temporal self-attention → agent-map cross-attention → agent-agent self-attention) is employed, with relative positional encoding to ensure invariance to coordinate transformations.
4. **Two-Stage Training**: Joint training followed by task-specific fine-tuning.

### Key Designs

1. **Dual-Objective Strategy for Joint Training**:

    - **Next Token Prediction (NTP)** for simulation: A causal attention mask is applied to model stepwise motion generation from the conditional distribution $P(A_t | A_{<t})$.
    - **Long-range Future Regression (LFR)** for prediction: Token classification is replaced with normalized trajectory regression, where each token predicts the complete long-range future trajectory rather than short segments. Dense supervision is applied to all tokens during training, and only a single forward pass is required at inference.
    - The planning task naturally benefits from both simulation and prediction training objectives, adopting a two-stage inference procedure: jointly predicting future trajectories of all agents → tokenizing the predictions → progressively generating the ego-vehicle trajectory.

2. **Task-Specific Fine-Tuning Strategies**:

    - **Simulation Fine-Tuning**: Reinforcement learning fine-tuning analogous to RLHF is introduced. GRPO is used to generate $n$ scene rollouts, computing kinematic rewards (log-likelihood of ground truth under the generated trajectory distribution) and collision rewards (indicator function); gradient updates are applied to only one rollout to reduce computational cost.
    - **Prediction Fine-Tuning**: A lightweight Transformer decoder is introduced to generate multimodal trajectories conditioned on intention points, with Gaussian NLL loss supervising multimodal outputs.
    - **Planning Fine-Tuning (Pred2Gen)**: Predicted tokens replace ground-truth tokens for ego-vehicle generation; incorrectly predicted tokens are filtered out and replaced with ground truth, and end-to-end fine-tuning is performed to eliminate the train-inference distribution mismatch.

3. **Attention Mask Design**: Different tasks employ distinct attention masks—causal masks for simulation, and sequence-to-sequence masks for prediction (historical tokens attend to each other, while future tokens attend to history and preceding future tokens)—enabling flexible multi-task processing within a single model.

### Loss & Training

The joint training loss is:

$$\mathcal{L} = \mathcal{L}_{ntp}(\mathcal{A}_s, \hat{\mathcal{A}}_s) + \mathcal{L}_{lfr}(\mathcal{A}_f, \hat{\mathcal{A}}_f)$$

where NTP uses cross-entropy loss and LFR uses smooth-L1 loss. During fine-tuning, each task has its dedicated loss: simulation fine-tuning augments cross-entropy with a GRPO policy loss (weight 0.1); prediction fine-tuning uses NLL + cross-entropy + auxiliary LFR regression; planning fine-tuning supervises ego-vehicle generation (NTP) and surrounding agent prediction (LFR, weight 0.5).

Training uses the AdamW optimizer with batch size 48, 8 A6000 GPUs, 30 epochs, and a learning rate of $5 \times 10^{-4}$ with cosine decay. Fine-tuning uses the same number of epochs but with a lower learning rate of $5 \times 10^{-5}$ for simulation and planning.

## Key Experimental Results

### Main Results — Sim Agents Challenge

| Method | Realism Meta | Kinematic | Interactive | Map-based | minADE |
|--------|-------------|-----------|-------------|-----------|--------|
| LLM2AD | 0.7779 | 0.4846 | 0.8048 | 0.9109 | 1.2827 |
| UniMM | 0.7829 | 0.4914 | 0.8089 | 0.9161 | 1.2949 |
| CATK | 0.7846 | 0.4931 | 0.8106 | 0.9177 | 1.3065 |
| **UniMotion** | **0.7851** | **0.4943** | 0.8105 | **0.9187** | 1.3036 |

### Main Results — WOMD Prediction Leaderboard

| Method | minADE↓ | minFDE↓ | MR↓ | mAP↑ | Soft mAP↑ |
|--------|---------|---------|-----|------|-----------|
| MTR++ | 0.5906 | 1.1939 | 0.1298 | 0.4329 | 0.4414 |
| EDA | 0.5718 | 1.1702 | 0.1169 | 0.4487 | 0.4596 |
| RMP-YOLO | 0.5737 | 1.1697 | 0.1160 | 0.4523 | 0.4673 |
| **UniMotion** | **0.5718** | **1.1643** | 0.1162 | **0.4534** | 0.4642 |

### Main Results — Waymo Planning

| Method | Collision↓ | Red Light↓ | Off Route↓ | Err@1s↓ | Err@3s↓ | Err@5s↓ |
|--------|-----------|------------|-----------|---------|---------|---------|
| DIPP | 1.802 | 1.235 | 0.506 | 0.227 | 1.187 | 3.335 |
| **UniMotion** | **1.565** | 1.309 | **0.477** | **0.083** | **0.591** | **2.246** |

### Ablation Study

| NTP | LFR | Sim Kin. | Sim Inter. | Pred minADE | Pred mAP |
|-----|-----|----------|------------|-------------|----------|
| ✓ | - | 0.4884 | 0.7961 | 0.7697 | 0.2629 |
| - | ✓ | 0.4401 | 0.7742 | 0.6668 | 0.2935 |
| ✓ | ✓ | **0.4892** | **0.7968** | **0.6508** | **0.3147** |

### Key Findings

- Joint training not only preserves per-task performance but also improves prediction accuracy (compared to LFR-only, mAP increases from 0.2935 to 0.3147).
- The combination of NTP and LFR yields a synergistic effect: diverse generation directions can serve as guidance for prediction targets.
- Task-specific fine-tuning consistently and significantly improves performance across all tasks.
- RL fine-tuning alone (without closed-loop supervision) is insufficient and must be combined with consistency constraints.
- Retaining intention anchors during prediction fine-tuning is critical; removing them leads to a decrease in mAP.

## Highlights & Insights

- The conceptual framework that abstracts motion tasks into "diverse generation" and "long-range prediction" is both clear and insightful.
- The design of realizing multi-task processing within a single decoder-only model via attention masks and training objectives is highly elegant.
- The RL fine-tuning strategy draws on the success of RLHF/GRPO in LLMs; the combination of kinematic and collision rewards is natural and effective.
- The Pred2Gen fine-tuning strategy directly addresses the distribution mismatch between predicted tokens and ground-truth tokens during planning inference.
- Achieving or approaching state-of-the-art performance simultaneously on simulation, prediction, and planning leaderboards validates the feasibility of the unified framework.

## Limitations & Future Work

- Factorized attention and relative positional encoding rely on detailed positional information, limiting the potential for adopting additional LLM-related techniques.
- Cross-dataset learning (e.g., joint training on Waymo and nuPlan) has not been explored.
- Due to computational constraints, the scaling behavior of larger models has not been thoroughly investigated.
- Red light violation rate is suboptimal (1.309 vs. 1.235 for DIPP); supervision from logged trajectories alone cannot enforce traffic rule compliance.
- The tokenization design emphasizes generation diversity, potentially at the cost of prediction accuracy in simulation tasks (relatively poor minADE).

## Related Work & Insights

- The paper inherits the autoregressive generation paradigm from GPT-style motion models such as BehaviorGPT and SMART, while innovatively extending it to prediction and planning.
- The approach is consistent with the multi-task training philosophy in LLMs (e.g., using different prompts to distinguish tasks).
- The RL fine-tuning strategy directly draws from the simplified designs of GRPO and DAPO.
- Inspiration: Future work could connect this unified framework with perception modules in an end-to-end manner.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SAML: A Differentiable Semantic Meta-Learning Framework for Long-Tail Motion Prediction](../../AAAI2026/autonomous_driving/differentiable_semantic_meta-learning_framework_for_long-tail_motion_forecasting.md)
- [\[NeurIPS 2025\] Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](flow_matching-based_autonomous_driving_planning_with_advanced_interactive_behavi.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](../../ICCV2025/autonomous_driving/long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)
- [\[NeurIPS 2025\] Availability-aware Sensor Fusion via Unified Canonical Space](availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
