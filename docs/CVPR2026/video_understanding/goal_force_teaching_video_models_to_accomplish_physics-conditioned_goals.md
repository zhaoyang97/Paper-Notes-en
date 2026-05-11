---
title: >-
  [Paper Note] GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals
description: >-
  [CVPR 2026][Video Understanding][Video Generation] This paper proposes the GoalForce framework, which trains video generation models on simple synthetic data using multi-channel physical control signals (goal force…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Generation"
  - "Physics-Conditioned Goals"
  - "world model"
  - "Force Prompting"
  - "Causal Reasoning"
date: 2026-05-08
content_hash: 24175452cfc15e75
---

# GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals

**Conference**: CVPR 2026
**arXiv**: [2601.05848](https://arxiv.org/abs/2601.05848)
**Code**: [goal-force.github.io](https://goal-force.github.io/)
**Area**: Video Understanding
**Keywords**: Video Generation, Physics-Conditioned Goals, world model, Force Prompting, Causal Reasoning

## TL;DR

This paper proposes the GoalForce framework, which trains video generation models on simple synthetic data using multi-channel physical control signals (goal force, direct force, and mass), enabling the model to learn backward causal planning from desired effects. The approach achieves zero-shot generalization to complex real-world scenarios such as tool use and human–object interaction.

## Background & Motivation

Video generation models can serve as world models for planning and simulation; however, existing goal specification methods suffer from notable limitations:
- **Text instructions** are too abstract to capture precise physical details (a soccer player does not merely "shoot"—they kick with a specific force and angle).
- **Goal images** are often difficult to obtain or physically implausible (e.g., rendering the exact lighting of a ball entering a net).
- Existing force-conditioning methods (PhysGen, Force Prompting) support only *direct force* (apply force → observe outcome) and cannot handle *goal force* (specify desired outcome → plan antecedent actions).

Human reasoning about physical tasks more closely resembles **goal force**: when taking a penalty kick, one thinks about imparting a specific trajectory and speed to the ball rather than specifying precise pixel-level trajectories. Motivated by this observation, the paper proposes a paradigm shift from "specifying effects" to "generating causes."

## Method

### Overall Architecture

The framework is built on Wan2.2 (a MoE diffusion model) with a ControlNet architecture. The core contributions are a multi-channel physical control signal and a causal data training strategy.

### Key Designs

1. **Three-channel physical control tensor $\tilde{\pi} \in \mathbb{R}^{f \times 3 \times h \times w}$**:

   - **Channel 0 (Direct Force)**: Encodes the directly applied force (the "cause") using a moving Gaussian blob whose trajectory and duration are proportional to the force vector.
   - **Channel 1 (Goal Force)**: Encodes the desired goal force (the "effect") using a moving Gaussian blob to represent the intended motion of the target object.
   - **Channel 2 (Mass)**: Encodes physical properties such as object mass using a static Gaussian blob whose radius is proportional to mass; this channel is optional.

2. **Goal achievement via implicit planning**: The key training strategy is **random masking of causal information**:

   - For videos containing collisions, either the direct force (Ch0) or the goal force (Ch1) is provided at random, with the other channel set to zero.
   - The model is thereby forced to learn bidirectional reasoning:
     - Goal → Plan: given a goal force, infer and generate the antecedent direct force event.
     - Action → Outcome: given a direct force, simulate the collision result.
   - The mass channel is also randomly masked, compelling the model to operate with or without mass information.

3. **Synthetic training data** (approximately 12k videos total):

   - **Dominoes** (3k): generated in Blender; direct force → chain reaction → goal force.
   - **Rolling balls** (6k): Blender scenes comprising 4.5k collision and 1.5k non-collision clips.
   - **PhysDreamer carnations** (3k): non-rigid body dynamics.

4. **Architecture details**: ControlNet fine-tunes only the High-Noise Expert (responsible for global structure and low-frequency dynamics), cloning the first 10 DiT layers and connecting to the frozen base model via zero-convolution. Training requires only 3,000 steps on 4× A100 GPUs in under 48 hours.

### Loss & Training

- ControlNet is trained with the standard diffusion loss.
- Training videos: 81 frames at 16 FPS.
- Key design: text prompts provide semantic context (e.g., "a pool table") but do not specify low-level causal plans.
- Force and mass values are relatively normalized, requiring no absolute physical scale.

## Key Experimental Results

### Main Results

A human study (N=40, 2AFC) comparing GoalForce against a text-only baseline:

| Benchmark Category | Force Adh. | Realism | Visual Qual. |
|--------------------|-----------|---------|-------------|
| Two-object collision | **73.4%** | 67.2% | 66.0% |
| Multi-object collision | **72.0%** | 69.0% | 66.8% |
| Human–object interaction | **70.5%** | 47.5% | 48.9% |
| Tool–object interaction | **74.5%** | 61.6% | 58.7% |

(Percentages indicate the proportion of trials in which GoalForce was preferred.)

Physical planning accuracy (50 generations per scene):

| Scene | Valid | Success | Accuracy |
|-------|-------|---------|----------|
| Pool | 49 | 48 | 97.96% |
| Paper Balls | 50 | 49 | 98.00% |
| Kitchen Lemon | 50 | 50 | 100.00% |
| Coffee Cups | 44 | 41 | 93.18% |
| Duckie | 40 | 34 | 85.00% |
| Rubik's Cube | 49 | 46 | 93.88% |

The random baseline achieves at most 33.3%; the proposed model substantially exceeds this level.

### Ablation Study

Planning diversity experiment (6 dominoes, 5 candidate initiators):

| Distribution | Diversity Score $\delta(p)$ |
|---|---|
| **GoalForce model** | **0.6577** |
| Unif{0..4} (maximum diversity) | 1.0000 |
| Unif{0..1} | 0.6042 |
| Deterministic baseline | 0.3900 |

The model samples from multiple valid plans, avoiding mode collapse.

In the mass-awareness experiment, varying the masses of the projectile and target ball, the model correctly adjusts projectile velocity (satisfying 4/4 relationships in-distribution and 3/4 out-of-distribution).

### Key Findings

- **Remarkable zero-shot generalization**: trained exclusively on synthetic balls, dominoes, and a single flower, the model generalizes to complex scenes such as a golf club striking a ball and a hand picking up a rose.
- The model learns to grasp a rose by the stem rather than the petals and selects initiators unobstructed by obstacles.
- Text prompts alone are insufficient to specify goal forces—a fine-tuned text-only baseline still lags substantially on Force Adherence.
- Prior force-conditioning methods (PhysGen, PhysDreamer, Force Prompting) misinterpret goal forces as direct forces and cannot plan causal chains.

## Highlights & Insights

1. **The paradigm shift from "specifying causes" to "specifying effects"** is highly inspiring: Goal Force = effect-oriented planning, enabling the model to autonomously reason about causal chains rather than passively executing instructions.
2. **Minimal training data (~12k synthetic videos) and brief training time (3,000 steps / <48 h)** suffice to elicit complex planning capabilities, suggesting that physical causal understanding can be learned with high sample efficiency given the right inductive biases.
3. **The multi-channel control signal design is elegant**: physical control is decomposed into three orthogonal dimensions—cause, effect, and attribute—with bidirectional reasoning enabled through random masking.
4. **Implicit neural physics simulator concept**: the model acts as an approximate physical planner at inference time without relying on an external physics engine.

## Limitations & Future Work

- Force and mass values use relative normalization, precluding unified physical scales across domains.
- Training data covers only simple collisions and non-rigid body dynamics; generalization to more complex physical phenomena (fluids, deformations) remains unverified.
- The 81-frame video length limits exploration of long-horizon causal chain planning.
- Motion Realism (47.5%) and Visual Quality (48.9%) are relatively low for human–object interaction scenes, indicating that generalization to human body motion remains challenging.
- Integration with real robotic control has not been explored.

## Related Work & Insights

- GoalForce is complementary to video planning methods such as UniPi and Adapt2Act, providing a physics-based goal specification modality beyond text.
- The paradigm of training on simple synthetic data and generalizing to complex scenarios is broadly applicable to other video generation tasks requiring physical understanding.
- Concurrent works (Learning 3D Trajectories, Freefall fine-tuning) focus on local physical properties, whereas GoalForce targets causal interaction chains.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The paradigm shift from "applying force" to "goal force" is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation through human studies, accuracy metrics, diversity analysis, and mass-awareness experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The penalty kick analogy is precise and illuminating; Fig. 3 clearly contrasts direct force and goal force.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction for physics-conditioned video planning with broad potential applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation](hear_what_matters_text-conditioned_selective_video-to-audio_generation.md)
- [\[CVPR 2026\] MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters](maskadapt_learning_flexible_motion_adaptation_via_mask-invariant_prior_for_physi.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)

</div>

<!-- RELATED:END -->
