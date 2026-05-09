---
title: >-
  [Paper Note] WPT: World-to-Policy Transfer via Online World Model Distillation
description: >-
  [CVPR 2026][Model Compression][World Model] WPT proposes a world-to-policy transfer training paradigm that injects future-predictive knowledge from a world model into a teacher policy via a trainable reward model, then transfers this knowledge to a lightweight student policy through policy distillation and world reward distillation, achieving a closed-loop driving score of 79.23 with a 4.9× inference speedup.
tags:
  - CVPR 2026
  - Model Compression
  - World Model
  - Policy Distillation
  - Reward Model
  - Autonomous Driving
  - Online Distillation
date: 2026-05-08
content_hash: c39aad851e186c37
---

# WPT: World-to-Policy Transfer via Online World Model Distillation

**Conference**: CVPR 2026
**arXiv**: [2511.20095](https://arxiv.org/abs/2511.20095)
**Code**: None
**Area**: Model Compression
**Keywords**: World Model, Policy Distillation, Reward Model, Autonomous Driving, Online Distillation

## TL;DR
WPT proposes a world-to-policy transfer training paradigm that injects future-predictive knowledge from a world model into a teacher policy via a trainable reward model, then transfers this knowledge to a lightweight student policy through policy distillation and world reward distillation, achieving a closed-loop driving score of 79.23 with a 4.9× inference speedup.

## Background & Motivation
1. **State of the Field**: World models are widely used in autonomous driving to capture spatiotemporal dynamics and predict future scenes, but existing methods either suffer from tight runtime coupling or rely on offline reward signals.
2. **Limitations of Prior Work**: Direct integration of world models incurs severe inference latency; using world models as simulators introduces dependency on simulator fidelity.
3. **Root Cause**: World models provide valuable future-predictive knowledge, yet their computational cost is prohibitive at deployment time.
4. **Paper Goals**: Leverage world model knowledge during training while deploying only a lightweight policy network, realizing the paradigm of "use the world model at training time, discard it at deployment."
5. **Starting Point**: Use a reward model as a bridge to connect world model predictions with the policy's trajectory selection, then distill this capability into a student.
6. **Core Idea**: A trainable interactive reward model evaluates the consistency between candidate trajectories and world model predictions → the teacher policy learns future-aware planning → policy distillation + world reward distillation transfer this to the student.

## Method

### Overall Architecture
**Training phase**: The world model predicts future states → the reward model evaluates multimodal candidate trajectories → the teacher policy selects the optimal trajectory. **Distillation phase**: The student learns from the teacher via policy distillation (aligning planning representations) and world reward distillation (matching the teacher's highest-reward trajectory). Only the student policy is used at deployment.

### Key Designs

1. **Trainable Interactive Reward Model**:
    - **Function**: Evaluates the consistency between candidate trajectories and predicted future world states.
    - **Mechanism**: Each candidate trajectory $\tau_i$ is combined with the world model's predicted future state $F_{t+1}^w$ and passed through a trajectory encoder with two reward heads: (1) an imitation reward assessing alignment with human driving preferences; and (2) a simulation reward scoring driving quality based on metrics such as PDM scores. The final reward is a weighted combination of the two.
    - **Design Motivation**: Converts the world model's predictive capability into an optimizable reward signal, enabling the policy to learn end-to-end from future predictions.

2. **Policy Distillation**:
    - **Function**: Transfers the teacher's planning representation capability to the lightweight student.
    - **Mechanism**: Aligns the planning representations (features of planning queries after the decoder) of the teacher and student, enabling the student to produce planning outputs close to the teacher in a single forward pass.
    - **Design Motivation**: The student network is simple; learning a direct end-to-end mapping avoids the overhead of multimodal trajectory generation and world model interaction.

3. **World Reward Distillation**:
    - **Function**: Encourages the student to match the teacher's optimal trajectory in the predicted future world.
    - **Mechanism**: Encourages the student's output trajectory to receive a reward—as evaluated under the world model's predicted future—close to that of the teacher's highest-reward trajectory, effectively matching the teacher's decision logic.
    - **Design Motivation**: Aligning representations alone is insufficient; it is equally important to align the decision logic of "which trajectory is optimal in the future world."

### Loss & Training
Teacher training: imitation reward loss + simulation reward loss. Distillation: policy distillation loss + world reward distillation loss.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Ours | Prev. SOTA | Gain |
|-----------|--------|------|------------|------|
| Open-loop | L2 Error | 0.61 m | — | Competitive |
| Open-loop | Collision Rate | 0.11% | — | SOTA |
| Closed-loop | Driving Score | 79.23 | — | SOTA |
| Inference Speed | Speedup | 4.9× | 1× | Significant |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full WPT (Teacher) | Best | World-model-enhanced teacher |
| Full WPT (Student) | Close to teacher | Distillation retains most gains |
| w/o World Reward Distillation | Degraded | Reward distillation is important |
| w/o Reward Model | Significantly degraded | Reward model is the core component |

### Key Findings
- The student policy retains most of the teacher's performance gains while achieving a 4.9× inference speedup.
- World reward distillation provides additional improvement over pure policy distillation alone, demonstrating the importance of transferring decision logic.
- WPT generalizes effectively across different lightweight policy architectures, indicating strong framework generalizability.

## Highlights & Insights
- The **"use the world model at training time, discard it at deployment"** paradigm is highly appealing: it captures the benefits of world models without incurring deployment costs.
- The **trainable reward model** serving as a bridge for knowledge transfer is an elegant design that converts non-differentiable world knowledge into differentiable learning signals.
- Dual distillation (representation + reward) provides a more complete transfer than single-objective distillation.

## Limitations & Future Work
- Performance depends on the quality of the pretrained world model; low-fidelity predictions can lead to misleading reward signals.
- The variety of scenarios in closed-loop evaluation remains limited.
- Future work may explore extending this paradigm to general-purpose robot decision-making.

## Related Work & Insights
- **vs. WoTE/DriveDPO**: These methods directly integrate the world model into the policy, requiring autoregressive rollouts at inference time. WPT shifts this overhead entirely to the training phase.
- **vs. DriveWorld-style simulator methods**: These methods depend on simulator fidelity and are primarily evaluated in synthetic environments. WPT trains directly on real-world data.

## Rating
- Novelty: ⭐⭐⭐⭐ The training–deployment decoupling paradigm for world model usage is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on open-loop, closed-loop, and multiple policy architectures.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear and the pipeline is described completely.
- Value: ⭐⭐⭐⭐⭐ Achieves an excellent balance between efficiency and performance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](../../ICLR2026/model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)
- [\[CVPR 2026\] PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)

<!-- RELATED:END -->
