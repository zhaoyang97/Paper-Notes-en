---
title: >-
  [Paper Note] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][Autonomous Driving][Causal Confusion] This work addresses causal confusion in end-to-end autonomous driving by leveraging perception outputs (lane centerlines, agent trajectories) and self-supervised learning to establish causal relationships, achieving state-of-the-art performance on the Bench2Drive closed-loop benchmark (Driving Score 78.08).
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Causal Confusion
  - Self-Supervised Learning
  - End-to-End Driving
  - Perception-Guided
  - Closed-Loop Evaluation
date: 2026-05-08
content_hash: 91ca5ac5c9c123fb
---

# Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving

**Conference**: NeurIPS 2025
**arXiv**: [2511.08214](https://arxiv.org/abs/2511.08214)
**Code**: Available
**Area**: Autonomous Driving / End-to-End Decision Making
**Keywords**: Causal Confusion, Self-Supervised Learning, End-to-End Driving, Perception-Guided, Closed-Loop Evaluation

## TL;DR

This work addresses causal confusion in end-to-end autonomous driving by leveraging perception outputs (lane centerlines, agent trajectories) and self-supervised learning to establish causal relationships, achieving state-of-the-art performance on the Bench2Drive closed-loop benchmark (Driving Score 78.08).

## Background & Motivation

**Background**: End-to-end autonomous driving systems perform well in open-loop evaluations but suffer significant performance degradation in closed-loop settings.

**Limitations of Prior Work**: Causal confusion is the root cause—models fail to associate driving behavior with primary environmental factors and instead learn spurious correlations from noisy signals. Existing approaches focus primarily on input noise (e.g., sensor noise) while neglecting noise inherent in the supervision signal itself.

**Key Challenge**: The imitation learning paradigm over-relies on expert trajectories, which themselves contain substantial noise arising from driving style, temporal delays, and control errors.

**Key Insight**: Rather than designing complex network architectures, this work proposes changing the source of the supervision signal—shifting from dependence on expert trajectories to reliance on perception outputs (lane centerlines and agent trajectories) to guide planning.

**Core Idea**: Positive constraints (MTPS/STPS) ensure correct fundamental driving behavior + negative constraints (NTPS) reinforce safe interaction = a complete causal reasoning framework.

## Method

### Overall Architecture

PGS (Perception-Guided Self-Supervision) is built upon a standard end-to-end architecture, comprising a perception module (outputting lane centerlines and future trajectories of dynamic objects), a unified motion prediction and planning module, and a three-level self-supervision mechanism (MTPS, STPS, NTPS).

### Key Designs

1. **Multi-modal Trajectory Planning Self-supervision MTPS (Target Lane Selection)**

    - **Function**: Reformulates multi-modal driving decisions as a **lane selection problem**.
    - **Mechanism**: From all lane centerlines produced by the perception module, a geometric filter selects three ego-relevant lanes (current, left, right); an MLP predicts a selection score for each lane (softmax-normalized).
    - **Design Motivation**: Lane information inherently encodes all feasible lateral choices; the supervision signal is derived from the distance between the expert trajectory endpoint and each candidate lane, thereby eliminating interference from expert driving style.

2. **Spatial Trajectory Planning Self-supervision STPS (Lane Centerline-Based)**

    - **Function**: Uses lane centerlines as a purely spatial reference to replace expert trajectories that carry temporal noise.
    - **Mechanism**: For each point of the expert trajectory, the nearest point on the target lane centerline is found; points within distance $\leq w$ are replaced by the centerline point, while others are retained.
    - **Design Motivation**: Lane centerlines naturally connect lane entry and exit, avoiding lane deviation caused by accumulated errors.

3. **Negative Trajectory Planning Self-supervision NTPS (Dynamic Object Interaction)**

    - **Function**: Uses predicted future bounding boxes as negative signals to enforce collision-free ego trajectories.
    - **Mechanism**: Collisions are detected via the Separating Axis Theorem (SAT); a distance margin is computed at collision timestamps and maximized via $\max(0, \beta - \|Traj_{ego}^t - Traj_{obj}^t\|_2)$.
    - **Design Motivation**: Positive supervision specifies what the model should do; negative supervision specifies what it should not do.

### Loss & Training

$$L'_{total} = L_{total} + w_{MTPS} L_{MTPS} + w_{STPS} L_{STPS} + w_{NTPS} L_{NTPS}$$

Two-stage training: Stage 1 (6 epochs) for perception learning; Stage 2 (6 epochs) for joint perception and planning optimization. Hyperparameters: $w_{MTPS}=1.0,\ w_{STPS}=0.3,\ w_{NTPS}=1.0$.

## Key Experimental Results

### Main Results (Bench2Drive Benchmark)

| Method | Driving Score↑ | Success Rate↑ | Efficiency↑ |
|--------|---------------|---------------|-------------|
| VAD-Base | 42.35 | 15.00% | 157.94 |
| UniAD-Base | 45.81 | 16.36% | 129.21 |
| DriveTransformer | 63.46 | 35.01% | 100.64 |
| DiffAD | 67.92 | 38.64% | - |
| **PGS (Ours)** | **78.08** | **48.64%** | **181.31** |

### Ablation Study (Multi-Scenario Capability)

| Scenario | VAD | DriveTransformer | DiffAD | **PGS** |
|----------|-----|-----------------|--------|---------|
| Merging | 8.11% | 17.57% | 30% | **35%** |
| Overtaking | 24.44% | 35% | 35.55% | **73.33%** |
| Emergency Brake | 18.64% | 48.36% | 46.66% | **55%** |
| Give Way | 20% | 40% | 40% | **60%** |
| Average | 18.07% | 38.60% | 38.79% | **53.40%** |

### Key Findings

- Compared to the VAD-Base baseline: Driving Score improves by 35.73 points (+84%), and Success Rate increases from 15% to 48.64% (+223%).
- A 73.33% success rate on the Overtaking scenario substantially outperforms all competing methods, demonstrating the benefit of causal reasoning for complex interaction scenarios.
- Surpassing more sophisticated methods with a simple VAD backbone validates the value of the paradigm shift.

## Highlights & Insights

- **Causal Perspective Shift**: Reframing the problem from "input noise" to "supervision noise" represents a precise and insightful diagnosis. The same principle is transferable to other imitation learning tasks.
- **Minimal Architectural Modification**: Substantial performance gains are achieved solely through changes to the training procedure without requiring complex network redesign, demonstrating the elegance of the approach.
- **Closed-Loop vs. Open-Loop Discrepancy**: Although open-loop L2 error is marginally higher than that of some baselines, the closed-loop Driving Score leads by a large margin—an important observation for the choice of evaluation methodology in autonomous driving research.

## Limitations & Future Work

- The STPS component assumes the availability of HD maps and accurate perception outputs, limiting applicability in map-free or perception-degraded settings.
- Closed-loop evaluation remains simulator-based; real-world performance requires further validation.
- The loss weight ratios may need scenario-specific tuning for different road conditions.
- The trade-off between inference speed and real-time requirements is not thoroughly discussed.

## Related Work & Insights

- **vs. ChauffeurNet**: ChauffeurNet mitigates causal confusion by randomly dropping ego-motion; PGS directly alters the source of the supervision signal, addressing the issue more fundamentally.
- **vs. DriveAdapter**: DriveAdapter relies on privileged information distillation, whereas PGS requires no additional information and operates solely on existing perception outputs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — A novel reinterpretation of causal confusion from the supervision perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Closed-loop testing on Bench2Drive with multi-scenario ablation.
- Writing Quality: ⭐⭐⭐⭐ — Clear, fluent, and logically rigorous.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core problem with a simple yet effective approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
