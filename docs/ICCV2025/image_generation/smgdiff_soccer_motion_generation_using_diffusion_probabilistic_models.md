---
title: >-
  [Paper Note] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models
description: >-
  [ICCV 2025][Image Generation][motion generation] This paper proposes SMGDiff, a two-stage diffusion model framework that generates high-quality, diverse soccer motion animations in real time from user control signals, while refining ball-foot interaction details via a contact guidance module.
tags:
  - ICCV 2025
  - Image Generation
  - motion generation
  - diffusion model
  - soccer animation
  - human-object interaction
  - character control
date: 2026-05-08
content_hash: 563f0d7d3ba3b73c
---

# SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models

**Conference**: ICCV 2025
**arXiv**: [2411.16216](https://arxiv.org/abs/2411.16216)
**Code**: [https://github.com/SMGDiff/SMGDiff](https://github.com/SMGDiff/SMGDiff)
**Area**: Image Generation
**Keywords**: motion generation, diffusion model, soccer animation, human-object interaction, character control

## TL;DR

This paper proposes SMGDiff, a two-stage diffusion model framework that generates high-quality, diverse soccer motion animations in real time from user control signals, while refining ball-foot interaction details via a contact guidance module.

## Background & Motivation

Soccer is the world's most popular sport, with broad application demand in gaming and VR/AR. However, generating realistic soccer motion animation faces the following core challenges:

**Complex human-ball interaction**: Soccer involves precise physical contact between players and the ball, with particularly stringent accuracy requirements for ball-foot contact.

**Real-time requirements**: Games and interactive applications demand real-time inference, whereas existing diffusion models are typically computationally expensive.

**Motion diversity**: Soccer encompasses a wide range of skill categories—dribbling, tricks, shooting, etc.—requiring coverage of a broad motion spectrum.

**Lack of datasets**: No large-scale public mocap dataset for soccer motion exists.

Limitations of prior work:

- **Commercial games** (e.g., the EA SPORTS FC series) rely on massive pre-recorded motion libraries for motion matching, which is prone to visual artifacts.
- **Reinforcement learning / physics simulation methods** are limited to specific skills (e.g., dribbling, shooting, juggling) and cannot cover the full soccer motion spectrum.
- **General motion diffusion models** (e.g., CAMDM) focus solely on human motion style transitions and do not handle human-object interaction.
- **Human-object interaction generation methods** require time-consuming post-optimization, making them unsuitable for real-time interactive scenarios.

## Method

SMGDiff adopts a **two-stage framework**: the first stage generates trajectories, and the second stage generates soccer motions conditioned on those trajectories.

### Motion Representation

A soccer motion state $x^i = \{h, b, c\}$ consists of three components:

- **Body state** $h \in \mathbb{R}^{3+24 \times 6}$: root position and 6-DOF rotations of 24 SMPL joints.
- **Ball state** $b \in \mathbb{R}^{7}$: relative ball position, global ball velocity, and ball control weight.
- **Binary contact labels** $c = \{c_g, c_b\}$: foot-ground contact and foot-ball contact.

The ball control weight $w_b = 1 - \|b_p^{xy} - h_p^{xy}\| / r$ converts the global ball position to a relative position; when the ball is more than radius $r=2\text{m}$ from the character root, the weight approaches 0, effectively decoupling the ball representation between controlled and uncontrolled states.

### Stage 1: Trajectory Generation Model (TGM)

**Objective**: Transform coarse-grained user control signals (direction, speed, skill category) into fine-grained global character trajectories.

**Architecture**: A lightweight single-step diffusion model based on a Transformer Encoder. Inputs include:

- Soccer skill label $\mathbf{S}$ (6 categories: dribble, trick, shoot, stand, celebrate, off-the-ball move)
- Target waypoints $\mathbf{G}$ (computed from keyboard directional input and press intensity)
- Past trajectory $\mathbf{T}^{\mathcal{P}}$
- Gaussian noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ (for diversity)

At inference, the process mirrors a single-step DDPM reverse pass, completing trajectory generation in a single forward pass to ensure real-time performance.

**Trajectory blending**: The HFTE (Heuristic Future Trajectory Extension) strategy from CAMDM is adopted; when the user issues new control signals, the newly generated trajectory is blended with the previous frame's result to prevent abrupt changes in character direction and speed.

### Stage 2: Soccer Motion Diffusion Model

**Architecture**: An autoregressive diffusion model based on Transformers that generates soccer motion sequences conditioned on trajectories.

**Conditioning input** $\mathbf{C} = \{\mathbf{S}, \mathbf{X}^{\mathcal{P}}, \mathbf{T}^{\mathcal{F}}\}$:

- Skill label $\mathbf{S}$
- Past motion $\mathbf{X}^{\mathcal{P}}$ (10 history frames)
- Future trajectory $\mathbf{T}^{\mathcal{F}}$ (45 frames)

**Training loss** consists of four terms:

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}} + \lambda_{\text{foot}} \mathcal{L}_{\text{foot}}$$

- $\mathcal{L}_{\text{simple}}$: reconstruction loss for directly predicting $\mathbf{X}_0^{\mathcal{F}}$
- $\mathcal{L}_{\text{pos}}$: joint position loss obtained via forward kinematics
- $\mathcal{L}_{\text{vel}}$: velocity consistency loss
- $\mathcal{L}_{\text{foot}}$: foot-ground contact constraint loss (penalizing foot sliding during contact)

### Contact Guidance Module (CGM)

Contact guidance is introduced during the diffusion inference process to refine ball-foot contact via a dedicated loss function.

**Contact detection**: A foot-ball contact event is detected when ball acceleration exceeds threshold $\tau_a = 2\text{ m/s}^2$ (under ground friction alone, acceleration is small and constant):

$$\hat{c}_b = \mathbb{I}(\|b_a\| > \tau_a)$$

**Contact joint selection**: Distances from each foot joint to the ball are computed, with preference given to the airborne foot (the grounded foot's distance is penalized by weight $w_d=2$):

$$d = \min_{j \in \text{foot joints}} ((f_p^j - b_p) \cdot (1 + (w_d - 1) \cdot c_g^j))$$

**Contact loss**:

$$L = \sum_{i=1}^{F} d^i \cdot \frac{\mathbb{I}(d^i > \tau_d) \cdot \hat{c}_b^i}{\mathbb{I}(d^i > \tau_d) + \delta}$$

This activates only when the ball-foot distance exceeds threshold $\tau_d = 0.1\text{m}$ and a contact event is detected, guiding the foot toward the ball.

**Gradient guidance**: An adaptive step-size strategy from DSG is employed, blending the gradient direction with the unconditional sampling direction (guidance rate $w_r = 0.5$) to improve contact accuracy while preserving motion diversity.

**Deployment strategy**: Contact guidance is applied only during the final 2 of 8 denoising steps, since guidance applied under high noise levels in early steps tends to be ineffective and may produce unnatural results.

### Implementation Details

- Frame rate: 30 Hz; past frames $P=10$; future frames $F=45$
- Diffusion denoising steps: 8 (consistent between training and inference)
- Runtime environment: Unity (user interaction and visualization) + Python (model inference), communicating via TCP
- Hardware: Intel i7-10700K + NVIDIA RTX 3080 Ti

## Dataset: Soccer-X

The authors construct Soccer-X, the first large-scale dataset targeting data-driven soccer motion generation:

| Attribute | Value |
|-----------|-------|
| Capture system | 16 OptiTrack Prime x13 cameras |
| Capture volume | 6m × 7.5m × 2.5m |
| Raw frame rate | 240 fps (downsampled to 30 fps) |
| Number of players | 30 |
| Total frames | ~1.08M |
| Total duration | >10 hours |
| Body format | SMPL |
| Motion categories | 6 |

Six motion categories: Dribble (varying speed/foot/direction), Stand, Off-the-ball Move, Trick (5 types of skill moves), Shoot (captured indoors; full ball trajectories simulated via physics in Unity), and Celebrate.

## Key Experimental Results

### Quantitative Comparison

Comparison against three real-time controller baselines (LMP, MANN-DP, CM), using test-set trajectories as conditioning input:

| Method | FID↓ | Foot Slide↓ | Accel.↓ | Diversity↑ | Traj. Error↓ | Orient. Error↓ | Skill Acc.↑ |
|--------|------|-------------|---------|------------|--------------|----------------|-------------|
| LMP | 0.354 | 1.068 | 1.607 | 0.398 | 4.116 | 6.493 | 73.3% |
| MANN-DP | 0.359 | 1.351 | 1.565 | 0.475 | 4.069 | 5.299 | 69.1% |
| CM | 0.249 | 1.650 | 1.175 | 0.352 | 3.103 | 5.066 | 52.9% |
| **SMGDiff** | **0.181** | **0.854** | 1.200 | **0.618** | **2.413** | **4.939** | **93.3%** |

SMGDiff significantly outperforms all baselines on nearly every metric; skill accuracy reaches 93.3% (20 percentage points above the strongest baseline), and the lowest FID indicates that the generated motion distribution most closely matches real data.

### Ablation Study

| Variant | FID↓ | Foot Slide↓ | Accel.↓ | Diversity↑ |
|---------|------|-------------|---------|------------|
| w/o TGM | 0.365 | 1.003 | 1.197 | 2.433 |
| w/o CGM | 0.370 | 1.005 | 1.196 | 2.691 |
| Full | **0.358** | 1.005 | 1.201 | **2.693** |

- **Role of TGM**: Replacing straight-line trajectories with diverse generated ones substantially reduces FID and improves diversity.
- **Role of CGM**: Although contact guidance slightly increases foot slide and acceleration, it noticeably lowers FID, indicating more realistic human-ball interaction.

### Runtime Analysis

| Denoising Steps | 2 | 4 | **8** | 16 | 32 |
|-----------------|---|---|-------|----|----|
| Inference Time | 3ms | 6ms | **12ms** | 25ms | 52ms |
| FID | 0.395 | 0.390 | **0.370** | 0.366 | 0.338 |

Eight denoising steps achieve the best trade-off between inference speed and generation quality, requiring only 12ms to satisfy real-time constraints.

## Limitations & Future Work

1. **Absence of physical constraints**: No real physics simulation is incorporated during training or inference, which may yield physically implausible motions.
2. **Limited interaction types**: Only ball-foot interaction is considered; other body-part contacts such as headers and chest control are ignored.
3. **Single-player limitation**: Multi-player interaction scenarios (e.g., passing, tackling) are not addressed; future work could combine physics simulation to generate multi-player soccer motions.

## Personal Reflections

- The **two-stage decoupled design** is the central contribution: separating "coarse control → fine trajectory" from "trajectory → full-body motion" reduces the learning difficulty of each individual model and allows the trajectory to serve as a flexible intermediate representation.
- The **lightweight contact guidance strategy** is noteworthy: applying guidance only in the final 2 steps (rather than throughout) effectively improves contact quality while maintaining real-time performance—a "selective guidance" idea generalizable to other diffusion generation tasks with physical constraints.
- The **ball control weight** design is concise and effective; distance-based decay unifies global/relative ball position representations and prevents distant balls from interfering with motion generation.
- The construction and open-sourcing of Soccer-X is a valuable contribution to the community, though 30 fps indoor capture still differs from real-field soccer conditions.
- The most promising future directions include post-optimization with physics engines such as Isaac Gym and extension to multi-player interaction scenarios.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](../../CVPR2026/image_generation/causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

</div>

<!-- RELATED:END -->
