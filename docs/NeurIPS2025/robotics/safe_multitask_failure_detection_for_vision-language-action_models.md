---
title: >-
  [Paper Note] SAFE: Multitask Failure Detection for Vision-Language-Action Models
description: >-
  [NeurIPS 2025][Robotics][Failure Detection] SAFE identifies consistent "failure regions" in the internal feature space of VLA models that generalize across tasks. Leveraging this observation, it trains lightweight MLP/LSTM failure detectors and applies Functional Conformal Prediction (FCP) for threshold calibration. The approach achieves 78% ROC-AUC on unseen tasks with less than 1% computational overhead, substantially outperforming token-uncertainty and action-consistency baselines.
tags:
  - NeurIPS 2025
  - Robotics
  - Failure Detection
  - VLA Models
  - Multitask Generalization
  - Functional Conformal Prediction
  - MLP/LSTM Detector
date: 2026-05-08
content_hash: c21f2b83358440cc
---

# SAFE: Multitask Failure Detection for Vision-Language-Action Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.09937](https://arxiv.org/abs/2506.09937)
**Code**: [https://vla-safe.github.io/](https://vla-safe.github.io/)
**Area**: Robot Learning / VLA Safety
**Keywords**: Failure Detection, VLA Models, Multitask Generalization, Functional Conformal Prediction, MLP/LSTM Detector

## TL;DR
SAFE identifies consistent "failure regions" in the internal feature space of VLA models that generalize across tasks. Leveraging this observation, it trains lightweight MLP/LSTM failure detectors and applies Functional Conformal Prediction (FCP) for threshold calibration. The approach achieves 78% ROC-AUC on unseen tasks with less than 1% computational overhead, substantially outperforming token-uncertainty and action-consistency baselines.

## Background & Motivation

**State of the Field**: VLA models (e.g., OpenVLA, π₀) achieve only 30–60% zero-shot success rates on unseen tasks. Real-world deployment requires automated in-execution failure detection to trigger human intervention or retry mechanisms.

**Limitations of Prior Work**: Existing failure detectors are task-specific—requiring failure rollout data for each new task. Token-uncertainty methods (logit-based) perform poorly on VLAs (ROC-AUC 45–60%). Action-consistency methods (STAC) demand 10× inference time.

**Root Cause**: VLA models are designed for open-world tasks, making it infeasible to pre-collect failure data for every possible task. What is needed is a failure detector trained on seen tasks that generalizes to new ones.

**Paper Goals**: Train efficient failure detectors that generalize across tasks without requiring any data collection for new tasks.

**Starting Point**: In the final-layer hidden states of VLAs, the feature distributions of successful and failed trajectories exhibit consistent separation patterns across different tasks. A simple MLP/LSTM can learn this "failure region" and generalize accordingly.

**Core Idea**: Train lightweight failure detectors (MLP/LSTM) on the last-layer hidden states of the VLA, combined with FCP-based threshold calibration, to achieve real-time, multitask-generalizable failure detection.

## Method

### Overall Architecture
During VLA execution → extract the last-layer hidden state $\mathbf{e}_t$ at each step → **MLP Detector** (per-step scoring with cumulative sum $s = \sum_t \sigma(g(\mathbf{e}_t))$) or **LSTM Detector** (sequential processing $s_t = \sigma(\text{LSTM}(\mathbf{e}_{0:t}))$) → **FCP Threshold Calibration** (derive time-varying thresholds corresponding to confidence level $\alpha$ using a validation set) → flag failure when threshold is exceeded.

### Key Designs

1. **"Failure Regions" in the VLA Feature Space**:

    - Function: Discover and exploit failure patterns in VLA internal representations.
    - Mechanism: The last-layer hidden states $\mathbf{e}_t$ (prior to action decoding) encode task execution status. Features from successful trajectories cluster in one region, while those from failed trajectories deviate into a distinct "failure region." This pattern is consistent across tasks.
    - Design Motivation: Compared to token probabilities/logits (which only reflect prediction confidence), hidden states encode richer information about execution status. Token uncertainty is nearly ineffective on VLAs (ROC-AUC 45–60%).

2. **Dual-Architecture Failure Detectors (MLP + LSTM)**:

    - Function: Predict failure probability from feature sequences.
    - Mechanism: MLP—independent per-step scoring followed by cumulative aggregation $f_{MLP} = \sum_\tau \sigma(g(\mathbf{e}_\tau))$, with loss $L = \sum_i [y_i \sum_t(t-s_t) + (1-y_i)\sum_t s_t]$, encouraging low scores for successful and high scores for failed trajectories. LSTM—sequential processing to capture temporal dependencies $f_{LSTM} = \sigma(\text{LSTM}(\mathbf{e}_{0:t}))$, trained with BCE loss.
    - Design Motivation: MLP is simpler and more efficient; LSTM captures temporal patterns (e.g., oscillatory behavior as a precursor to failure).

3. **Functional Conformal Prediction (FCP) Threshold Calibration**:

    - Function: Calibrate time-varying thresholds on a validation set to guarantee false positive rate (FPR) ≤ $\alpha$.
    - Mechanism: Compute upper confidence bands $C_\alpha = \{[lower_t, upper_t]\}$ from scores on successful validation trajectories. At test time, failure is declared when $s_t > upper_t$. Conformal prediction provides a theoretical guarantee: FPR ≤ $\alpha$.
    - Design Motivation: Fixed thresholds do not transfer across different VLAs or tasks; FCP provides adaptive, theoretically grounded thresholds with false alarm rate guarantees.

### Loss & Training
- MLP: Cumulative score loss (low for success / high for failure).
- LSTM: Stepwise BCE loss.
- Training data: Mixed successful and failed rollouts across multiple tasks.
- Inference overhead: <1% (only last-layer feature extraction + MLP forward pass).

## Key Experimental Results

### Main Results (ROC-AUC %)

| VLA Model | Benchmark | Seen Tasks | Unseen Tasks |
|-----------|-----------|------------|--------------|
| OpenVLA | LIBERO | 72.68 | 73.47 |
| π₀ | LIBERO | **92.98** | **84.48** |
| π₀-FAST | LIBERO | 90.06 | 80.44 |
| π₀* | SimplerEnv | 89.50 | 84.82 |
| **Average** | **All** | **81.43** | **78.00** |

### Method Comparison

| Method | ROC-AUC | Inference Overhead |
|--------|---------|--------------------|
| Token Uncertainty | 48–54% | ~0% |
| Sample Consistency (STAC) | ~71% | **10×** |
| Embedding Distance | 57–82% | ~1% |
| **SAFE (MLP/LSTM)** | **78–85%** | **<1%** |

### Key Findings
- Token uncertainty is nearly ineffective for VLAs—token probabilities do not reflect execution quality.
- SAFE degrades by only 3.4% on unseen tasks (81.43→78.00), indicating acceptable generalization.
- On real robots: π₀-FAST + Franka achieves 64.16% on unseen tasks; OpenVLA + WidowX achieves 88.42%.
- Detected failure modes include: imprecise insertion, oscillatory motion, grasp failure, and object slippage.
- FCP converges to near-optimal thresholds with approximately 100 calibration samples.

## Highlights & Insights
- **The discovery of "failure regions" is an insightful finding**: Failed trajectories form consistent clusters in the VLA hidden state space, suggesting that VLAs internally "know" they are failing—yet lack an explicit monitoring mechanism.
- **Extremely low computational overhead**: Less than 1% additional cost (vs. 10× for STAC), making real-time deployment feasible.
- **Conformal prediction provides theoretical guarantees**: The FPR ≤ $\alpha$ guarantee is particularly valuable for safety-critical applications.

## Limitations & Future Work
- Validation is limited to manipulation tasks; navigation and mobile manipulation remain untested.
- Only last-layer features are used; aggregating across multiple layers may yield stronger representations.
- Training still requires collecting successful and failed rollouts; purely zero-shot operation is not supported.
- Performance drops 8–13% on unseen tasks, which may be insufficient for safety-critical deployments.

## Related Work & Insights
- **vs. Token Uncertainty**: Token probabilities are ineffective for VLAs, whereas hidden-state features are substantially more informative—an important finding for the VLA safety community.
- **vs. STAC (Action Consistency)**: STAC requires multiple inference passes to check consistency; SAFE operates within a single inference pass.
- **vs. OOD Detection (LogpZO)**: OOD methods detect anomalous inputs, while SAFE detects execution failures—a more direct and practically relevant objective.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of "failure regions" and the FCP calibration design are genuinely novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 VLAs × 2 benchmarks + real robots + multiple baselines + FCP analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Addresses a core safety challenge in VLA deployment with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [\[AAAI 2026\] 10 Open Challenges Steering the Future of Vision-Language-Action Models](../../AAAI2026/robotics/10_open_challenges_steering_the_future_of_vision-language-ac.md)

<!-- RELATED:END -->
