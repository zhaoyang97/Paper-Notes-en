---
title: >-
  [Paper Note] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds
description: >-
  [ICCV 2025][Human Understanding][egocentric vision] This paper proposes EgoAgent, a unified predictive agent model that simultaneously learns to represent egocentric visual observations, predict future world states, and generate 3D human motions within a single Transformer.
tags:
  - ICCV 2025
  - Human Understanding
  - egocentric vision
  - agent model
  - world model
  - 3D human motion prediction
  - joint embedding predictive architecture
date: 2026-05-08
content_hash: 79e01a1d50dff1fe
---

# EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds

**Conference**: ICCV 2025
**arXiv**: [2502.05857](https://arxiv.org/abs/2502.05857)
**Code**: [https://github.com/zju3dv/EgoAgent](https://github.com/zju3dv/EgoAgent)
**Area**: Human Understanding
**Keywords**: egocentric vision, agent model, world model, 3D human motion prediction, joint embedding predictive architecture

## TL;DR

This paper proposes EgoAgent, a unified predictive agent model that simultaneously learns to represent egocentric visual observations, predict future world states, and generate 3D human motions within a single Transformer.

## Background & Motivation

Humans continuously interact with their environment through a perception–action loop, simultaneously acquiring three capabilities: visual perception, world dynamics prediction, and action decision-making. The Common Coding Theory from cognitive science posits that perception and action are deeply intertwined and share a common representational space. However, existing methods decompose these three capabilities into independent tasks:

1. Visual representation learning (e.g., DINO, DoRA) — learning high-level representations of world observations
2. World models (e.g., JEPA) — learning predictive representations of state transitions
3. Action prediction (e.g., siMLPe) — predicting future human body motions

This fragmented paradigm fails to capture the intrinsic relationships among the three capabilities. The core challenge lies in the fact that human interaction with the world constitutes a continuous perception→action→observation loop, where observations and actions are tightly coupled in both time and causality. How to design a learning framework and supervision signals that capture such complex dependencies remains an open problem.

## Method

### Overall Architecture

EgoAgent adopts a Joint Embedding–Action–Prediction (JEAP) architecture, encoding egocentric video frames and 3D human poses into an interleaved "state–action–state–action" token sequence processed via causal attention. The framework comprises two asymmetric branches: a predictor branch that forecasts future states and actions, and an observer branch that extracts target states from raw observations. InternLM serves as the base architecture (without loading pretrained weights), supporting model scales of 300M and 1B parameters.

### Key Designs

1. **Interleaved Joint Prediction**: At each timestep $t$, a structured token sequence is constructed: image token $i_t$, action query token $q_a$, action token $a_t$, and state query token $q_s$. Via causal attention, $q_a$ aggregates $i_{[0:t]}$ and $a_{[0:t-1]}$ to predict the current action $A'_t$; $q_s$ aggregates $i_{[0:t]}$ and $a_{[0:t]}$ to predict the next world state $S'_{t+1}$. **Design Motivation**: This explicitly models the causal and temporal dependency chain "observation → triggers action → influences next state."

2. **Temporally Asymmetric Predictor–Observer**: The observer branch processes only image inputs, extracting current-frame features for self-supervised representation learning and next-frame features as supervision signals for state prediction. Observer parameters are updated via EMA from the predictor. **Key Advantage**: The query-based design decouples the shared state/representation components from the predictor's action components, avoiding gradient conflicts. Formally: $\mathcal{L}_{pred}(t) = \mathcal{L}_{dino}(S'_{t+1}, sg[S_{t+1}])$, $\mathcal{L}_{act}(t) = \mathcal{L}_1(A'_t, A_t)$.

3. **Learning in Semantic Feature Space**: Images are projected into continuous semantic embeddings via learnable convolutional layers, rather than discrete tokens produced by reconstruction-based tokenizers such as VQGAN. **Design Motivation**: Humans predict based on abstract concepts rather than pixels; the semantic feature space better aligns with cognitive processes. Experiments confirm that the pixel-level latent space of VQGAN leads to significant degradation in both world-state prediction and visual representation quality.

### Loss & Training

The overall objective is:

$$\mathcal{L} = \frac{1}{t}\sum_{k=0}^{t}(\lambda_{rep}\mathcal{L}_{rep} + \lambda_{pred}\mathcal{L}_{pred} + \lambda_{act}\mathcal{L}_{act})$$

- $\mathcal{L}_{rep}$: Self-supervised representation loss via DINO-style contrastive learning on different crops of the same frame ($\lambda_{rep}=2$)
- $\mathcal{L}_{pred}$: State prediction loss, DINO loss between predictor and observer outputs ($\lambda_{pred}=1$)
- $\mathcal{L}_{act}$: Action prediction loss, L1 loss between predicted and ground-truth 3D poses ($\lambda_{act}=3$)

Training is conducted on WalkingTours and Ego-Exo4D. One frame is sampled every 5 frames; all 3D poses are retained. The 300M model is trained on 32 A100 GPUs for 25 hours; the 1B model on 48 GPUs for 60 hours. Batch size is 1920, base learning rate is 6e-4, with FP16 acceleration.

## Key Experimental Results

### Main Results

**Comprehensive Three-Task Performance Comparison**:

| Method | World State Prediction Top1/mAP | Action Prediction MPJPE↓ (30fps) | Visual Representation ImgNet-1K Top1 |
|--------|--------------------------------|----------------------------------|--------------------------------------|
| DoRA | 30.15/45.01 | - | 34.52 |
| DINO | 28.24/43.42 | - | 22.18 |
| siMLPe | - | 13.33 | - |
| Diffusion Policy-T | - | 25.92 | - |
| **EgoAgent-300M** | **43.01/58.06** | **12.92** | **34.65** |
| **EgoAgent-1B** | **46.43/61.96** | **12.51** | **35.84** |

EgoAgent-1B surpasses DoRA by +16.28% Top1 on world-state prediction, improves upon siMLPe by −0.82 MPJPE on action prediction, and exceeds DoRA by +1.32% on ImageNet-1K.

### Ablation Study

**Joint Learning Ablation (14,400 iterations)**:

| Setting | State Prediction Top1 | Action MPJPE↓ | Representation ImgNet-100 Top1 |
|---------|-----------------------|---------------|-------------------------------|
| Full model | 37.77 | 14.49 | 41.64 |
| w/o $\mathcal{L}_{pred}$ | - | 14.70 | 39.12 |
| w/o $\mathcal{L}_{act}$ | 34.86 | - | 39.92 |
| w/o $\mathcal{L}_{rep}$ | 25.90 | 14.49 | - |
| $\mathcal{L}_{pred}$ only | 33.23 | - | - |
| $\mathcal{L}_{act}$ only | - | 14.32 | - |
| $\mathcal{L}_{rep}$ only | - | - | 40.80 |
| Pixel-level latent (w/o rep) | 20.62 | 13.57 | 1.00 |
| Pixel-level latent (w/ rep) | 15.63 | 16.25 | 31.20 |

**TriFinger Robotic Manipulation**:

| Method | Reach Cube | Move Cube |
|--------|-----------|-----------|
| DINO | 78.03% | 47.42% |
| DoRA | 82.40% | 48.13% |
| **EgoAgent-1B** | **85.72%** | **57.66%** |

### Key Findings

- **Mutual benefit across three tasks**: Removing any single task degrades the performance of the remaining two, confirming the complementarity of joint learning.
- **Representation as foundation**: Removing $\mathcal{L}_{rep}$ causes the largest drop in state prediction (−11.87% Top1), demonstrating that representation learning underpins both prediction and action.
- **Semantic vs. pixel space**: The VQGAN pixel-level latent space nearly eliminates visual representation capability (Top1 only 1.00%), validating the superiority of the semantic feature space.
- **Action diversity contributes to prediction**: Given the same observation, different pose conditions retrieve future images that correctly reflect motion dynamics.

## Highlights & Insights

1. **Cognitively-inspired architecture**: The Common Coding Theory is operationalized into the JEAP architecture; the interleaved placement of action and state queries elegantly mirrors the causal temporal sequence.
2. **Training LLM architecture from scratch**: Without language pretraining weights, the model demonstrates that visual perception and prediction can be learned solely from visual–action data.
3. **In-depth analysis of inter-task dependencies**: Representation → (prediction, action) → representation forms a positive feedback loop, yet neither prediction nor action alone is sufficient to improve representation.
4. **Elegant decoupling via EMA observer**: The query-based design allows the observer to receive stable EMA updates without processing the action modality.

## Limitations & Future Work

1. Only coarse-grained 3D body poses are used; fine-grained hand representations are excluded, limiting performance on dexterous manipulation tasks.
2. The absence of long-term memory mechanisms makes the 20-frame sliding window potentially insufficient for tasks requiring long-horizon dependencies.
3. Training data relies on automatically annotated poses from Ego-Exo4D, which introduce non-trivial noise.
4. World-state prediction is evaluated via feature retrieval rather than direct future-frame generation, limiting applicability.
5. Incorporating language instructions to enable goal-directed action planning is a promising future direction.

## Related Work & Insights

- **JEPA (LeCun)**: The theoretical foundation of the Joint Embedding Predictive Architecture; this work extends it into a reactive agent model that incorporates action prediction.
- **DoRA**: Object-level representation learning on egocentric video; serves as the primary baseline for visual representation.
- **MC-JEPA**: Combines self-supervised learning with optical flow estimation to learn content and motion dynamics; this work extends that paradigm by incorporating the action modality.
- **Common Coding Theory**: The cognitive science theory positing that perception and action share a common representational space, providing the theoretical grounding for the unified framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first egocentric agent model to unify representation, prediction, and action; the JEAP architecture is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive three-task evaluation with ablation studies that deeply illuminate inter-task dependencies.
- **Writing Quality**: ⭐⭐⭐⭐ The cognitive science framing is compelling, and the method description is clear.
- **Value**: ⭐⭐⭐⭐ Provides an important reference for unified modeling in embodied intelligence; code and models are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](../../CVPR2026/human_understanding/fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
