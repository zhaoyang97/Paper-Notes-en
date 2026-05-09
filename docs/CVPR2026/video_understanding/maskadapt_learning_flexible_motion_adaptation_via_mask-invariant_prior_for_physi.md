---
title: >-
  [Paper Note] MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters
description: >-
  [CVPR 2026][Video Understanding][Physics simulation] This paper proposes MaskAdapt, a two-stage residual learning framework that first trains a mask-invariant robust base policy and then trains a residual policy on top of the frozen base controller to modify target body parts, enabling flexible and precise motion adaptation for physics-based humanoid characters.
tags:
  - CVPR 2026
  - Video Understanding
  - Physics simulation
  - motion adaptation
  - residual learning
  - body-part masking
  - humanoid control
date: 2026-05-08
content_hash: fee8d7ffdd40cbcf
---

# MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters

**Conference**: CVPR 2026
**arXiv**: [2603.29272](https://arxiv.org/abs/2603.29272)
**Code**: Unavailable
**Area**: Video Understanding / Physics-Based Character Control
**Keywords**: Physics simulation, motion adaptation, residual learning, body-part masking, humanoid control

## TL;DR

This paper proposes MaskAdapt, a two-stage residual learning framework that first trains a mask-invariant robust base policy and then trains a residual policy on top of the frozen base controller to modify target body parts, enabling flexible and precise motion adaptation for physics-based humanoid characters.

## Background & Motivation

**State of the Field**: Physics-based humanoid character control is a central problem in computer graphics and robotics. Deep reinforcement learning (DRL)-driven physical character control has achieved remarkable progress in recent years, enabling reference motion imitation and physically plausible motion synthesis. However, flexibly modifying specific body-part behaviors while preserving the overall motion quality (e.g., having a walking character perform a different upper-body action) remains a significant challenge.

**Limitations of Prior Work**: Existing motion control methods typically learn a unified whole-body policy, making independent adaptation of individual body parts difficult. (1) Direct fine-tuning of an existing policy leads to catastrophic forgetting—modifying upper-body motion disrupts lower-body behavior; (2) training composite motions from scratch requires substantial engineering effort and computational cost; (3) existing methods lack flexibility in selecting target body parts—different applications require modifying different parts, necessitating separate training for each combination.

**Root Cause**: Motion adaptation must simultaneously satisfy two conflicting objectives: (1) precisely modifying the behavior of target body parts to match new motion goals, and (2) preserving the stability and naturalness of non-target parts. The coupled nature of whole-body policies makes it difficult to satisfy both simultaneously.

**Paper Goals**: Design a framework that (1) trains a base motion prior robust to missing observations of body parts, and (2) enables flexible adaptation of target parts via lightweight residual learning on top of this prior without disturbing other parts.

**Starting Point**: Drawing inspiration from masked pretraining—if a base policy is trained with observations of certain body parts masked, it can naturally transition when those parts are "taken over" during adaptation without causing system instability.

**Core Idea**: A two-stage paradigm—Stage 1 trains a base policy with random masking to make it robust to missing observations (anticipating future adaptation regions); Stage 2 trains a residual policy on top of the frozen base policy to modify only the target body parts.

## Method

### Overall Architecture

MaskAdapt adopts a two-stage residual learning approach. **Stage 1** trains a mask-invariant base policy by randomly masking observations of different body parts during standard motion imitation training, with a consistency regularization term enforcing similar action distributions regardless of masking. This produces a robust motion prior that remains stable even when observations of certain parts are absent. **Stage 2** freezes the base policy and trains a lightweight residual policy whose outputs are added to the base policy's actions to form the final control signal. The residual policy only needs to modify the target body parts; residuals for non-target parts are kept close to zero.

### Key Designs

1. **Stochastic Body-Part Masking**:

    - **Function**: Enables the base policy to maintain stable motion even when observations of some body parts are missing.
    - **Mechanism**: Observations of the humanoid character (joint angles, angular velocities, positions, etc.) are grouped by body part. At each training step, one or more body-part groups are randomly selected and their observations are zeroed out (or replaced with default values). Masking probabilities and combinations vary dynamically during training, exposing the policy to diverse missing patterns. A key consistency regularization term $\mathcal{L}_{consist}$ enforces that the policy's action distributions remain consistent before and after masking (via KL divergence constraints).
    - **Design Motivation**: If the base policy is trained only on complete observations, it will produce unstable behavior when the residual policy "takes over" certain parts in Stage 2 due to unexpected input patterns. Masking-based pretraining primes the policy to anticipate such body-part takeover scenarios.

2. **Residual Policy Adaptation**:

    - **Function**: Precisely modifies the motion of target body parts on top of the frozen base policy.
    - **Mechanism**: The residual policy network receives the same state inputs as the base policy along with target motion information, and outputs a residual vector of the same dimension as the base policy's actions. The final action is $a = a_{base} + a_{residual}$. To ensure the residual affects only target parts, L2 regularization is applied to residuals of non-target parts, driving them toward zero. Observations of target body parts are masked when fed to the base policy, so the base policy provides no "opinion" on those parts, leaving full control to the residual policy.
    - **Design Motivation**: Residual learning offers three advantages: (1) the frozen base policy guarantees unchanged behavior for non-target parts; (2) the residual policy only needs to learn incremental changes, improving training efficiency; (3) different masking configurations flexibly select which body-part combinations to adapt.

3. **Multi-Application Adaptation (Motion Composition & Text-Driven Tracking)**:

    - **Function**: Demonstrates the generality of the framework.
    - **Mechanism**: (1) *Motion Composition*: By varying the masking mask, different body parts within the same sequence are independently adapted, enabling flexible composition of multi-source motions (e.g., the upper body follows one motion while the lower body follows another). (2) *Text-Driven Partial Goal Tracking*: Target body parts track kinematic goals from a pretrained text-conditioned autoregressive motion generator (e.g., MotionGPT), while other parts maintain the original motion, realizing "specifying new motions for designated body parts via natural language."
    - **Design Motivation**: Demonstrates the synergy between the two preceding designs—mask-invariance ensures base policy stability across different masking configurations, while residual learning ensures precise and flexible adaptation.

### Loss & Training

**Stage 1 (Base Policy Training)**:
- Motion imitation reward: tracks joint angles, positions, and velocities of the reference motion.
- Consistency regularization: $\mathcal{L}_{consist} = D_{KL}(\pi_{mask} \| \pi_{full})$, enforcing alignment between the masked and full-observation policy distributions.
- Trained with PPO.

**Stage 2 (Residual Policy Training)**:
- Target tracking reward: measures alignment of target body-part joint angles/positions with the new motion goal.
- Non-target regularization: $\mathcal{L}_{reg} = \|a_{residual}^{non-target}\|^2$.
- The base policy is fully frozen; only the residual policy is optimized.
- Trained with PPO.

## Key Experimental Results

### Main Results (Motion Adaptation Quality)

| Method | Target Tracking Error↓ | Non-Target Preservation↑ | Physical Stability↑ | Overall |
|--------|------------------------|--------------------------|---------------------|---------|
| Direct Fine-tuning | Medium | Low (catastrophic forgetting) | Low | Poor |
| Train from Scratch | High (hard to compose) | N/A | Medium | Medium |
| Residual w/o Mask-inv | Low–Medium | Medium | Medium–Low | Medium |
| **MaskAdapt (Ours)** | **Lowest** | **Highest** | **Highest** | **Best** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| Full MaskAdapt | Best | Complete two-stage framework |
| w/o Masking Training | Unstable during adaptation | Base policy cannot handle missing observations |
| w/o Consistency Regularization | Large action deviation after masking | Inconsistent action distributions before/after masking |
| w/o Residual Regularization | Non-target parts modified | Residual influence spreads beyond target regions |
| Single-stage (no residual) | Inflexible adaptation | Requires retraining for each body-part combination |

### Key Findings

- Mask-invariant training is a prerequisite for successful adaptation—without it, the residual policy causes the base policy's behavior to collapse during adaptation.
- Consistency regularization plays a significant role—it minimizes action discrepancy induced by masking, providing a stable "foundation" for the residual policy to operate on.
- In motion composition tasks, MaskAdapt achieves independent multi-part adaptation within the same sequence, producing diverse composite behaviors.
- Text-driven adaptation demonstrates natural collaboration with motion generation models—kinematic goals are provided by a text-conditioned generator, while physical plausibility is ensured by MaskAdapt.

## Highlights & Insights

- **Elegant Transfer of Masked Pretraining**: Transferring the masked training paradigm widely used in NLP/CV to the domain of physical control—training the policy to anticipate partial observation absence—is a highly creative cross-domain analogy.
- **Residual Learning + Selective Freezing Combination**: Freezing the base policy guarantees stability; the residual policy handles adaptation; L2 regularization constrains the adaptation scope—three mechanisms work in concert to ensure precise and controllable partial adaptation.
- **Strong Practical Value**: A single base policy can be paired with multiple residual policies for different adaptation targets, eliminating the need to retrain the full system for each new motion.

## Limitations & Future Work

- Experiments are primarily conducted on standard humanoid morphologies; applicability to non-standard morphologies (e.g., non-humanoid robots) remains unverified.
- Body-part groupings are predefined, which may not cover all fine-grained adaptation needs (e.g., control of individual fingers).
- The robustness of the base policy is strongly tied to the masking training design—masking probability and grouping strategy selection requires empirical tuning.
- The current combination of residual and base policy outputs is simple addition; more complex combination schemes (e.g., gating mechanisms) may yield further improvements.
- Future work could explore integrating MaskAdapt with large-scale motion priors (e.g., MoFlow, MotionDiffuse) to enable richer adaptation capabilities.

## Related Work & Insights

- **vs. PHC/UHC and similar whole-body imitation controllers**: These methods pursue precise whole-body motion imitation but do not support flexible partial adaptation. MaskAdapt adds part-level control capability on top of such systems.
- **vs. CompositeMotion and multi-policy composition**: Traditional multi-policy composition methods require specialized fusion mechanisms for each combination; MaskAdapt provides a more unified and flexible framework through residual learning.
- **vs. MotionGPT-based physical control**: MaskAdapt decouples kinematic generation (non-physical) from physical control, bridging the two via residual policy—this decoupled design allows each component to be upgraded independently.
- The core idea of this work—"first make the model robust to missing information, then leverage this robustness for selective adaptation"—has potential applications in model editing, skill composition, and related areas.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of mask-invariant prior and residual adaptation is a novel design for the physical control domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Demonstrates two applications (motion composition and text-driven adaptation) with ablation studies, though limited detail access constrains full assessment.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; the two-stage structure is easy to follow.
- Value: ⭐⭐⭐⭐ Proposes an elegant solution to flexible partial adaptation for physical humanoid control with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[CVPR 2026\] GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)
- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](../../ICCV2025/video_understanding/flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[NeurIPS 2025\] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills](../../NeurIPS2025/video_understanding/kungfubot_physics-based_humanoid_whole-body_control_for_learning_highly-dynamic_.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)

<!-- RELATED:END -->
