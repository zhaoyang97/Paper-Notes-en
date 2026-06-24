---
title: >-
  [Paper Note] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model
description: >-
  [ECCV 2024][Image Generation] Proposes MotionLCM, which first introduces consistency distillation into human motion generation, achieving real-time motion generation (~30ms/sequence) via single-step/few-step inference in the motion latent space, and realizes real-time controllable motion generation in the latent space through Motion ControlNet.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: deda74c4e279c5e7
---

# MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model

**Conference**: ECCV 2024  
**arXiv**: [2404.19759](https://arxiv.org/abs/2404.19759)  
**Area**: Image Generation

## TL;DR

Proposes MotionLCM, which first introduces consistency distillation into human motion generation, achieving real-time motion generation (~30ms/sequence) via single-step/few-step inference in the motion latent space, and realizes real-time controllable motion generation in the latent space through Motion ControlNet.

## Background & Motivation

- **Efficiency Bottlenecks of Diffusion Models**: Existing motion diffusion models (MDM ~24s, MLD ~0.2s) suffer from slow inference speed, failing to meet the requirements of real-time applications.
- **High Latency in Spatiotemporal Control**: Controllable motion generation methods such as OmniControl have an inference time of approximately 81s/sequence, leaving a huge gap for real-time applications.
- **Difficulty in Latent Space Control**: In latent diffusion models, the latent representations lack explicit motion semantics, making it impossible to directly manipulate control signals.
- **Opportunities of Consistency Models**: Consistency Models (CM) achieve highly efficient single-step/few-step generation by learning the consistency function on the PF-ODE trajectory, which perfectly aligns with the goal of accelerating motion generation.
- **Core Problem**: How to accelerate motion generation to real-time levels while preserving generation quality and controllability?

## Method

### Overall Architecture

Trained in two stages:
1. **Motion Latent Consistency Distillation**: Distills consistency models from a pre-trained MLD (Motion Latent Diffusion model), enabling 1-4 step inference.
2. **Latent Space Motion Control**: Introduces Motion ControlNet into the latent space of MotionLCM, and utilizes the VAE decoder to provide explicit control supervision in the motion space.

### Key Designs

**Motion Latent Consistency Distillation**:
- Using MLD as the teacher model, the consistency function $\textbf{f}_\Theta : (\mathbf{z}_t, t, w, \mathbf{c}) \mapsto \mathbf{z}_0$ is learned in the motion latent space.
- Adopts $k$-step skip consistency distillation (LCM scheme) instead of step-by-step consistency, significantly reducing convergence time.
- Integrates classifier-free guidance (CFG) into the distillation, where $w$ is uniformly sampled from $[5, 15]$ during training.
- Uses a DDIM solver (skip interval $k=20$) + Huber loss as the distance metric.

**Motion ControlNet**:
- Initialized with a trainable copy of MotionLCM, with zero-initialized linear layers appended to each layer.
- Control task definition: Given the initial pose of the first $\tau$ frames (global 3D positions of 6 key joints) and a text description, generate the subsequent motion.
- Trajectory Encoder: Stacks Transformer layers to encode trajectory signals, where the output feature of the [CLS] token is added to the noisy latent.

**Explicit Supervision in Motion Space** (Core Innovation):
- Reconstruction loss solely in the latent space is insufficient to provide detailed control constraints.
- Uses a frozen VAE decoder to decode the predicted latent $\hat{\mathbf{z}}_0$ into the motion space, calculating the control joint position error.
- Benefiting from the single-step inference capability of MotionLCM, this decoding process is more efficient compared to MLD.

### Loss & Training

**First Stage — Consistency Distillation Loss**:

$$\mathcal{L}_{LCD} = \mathbb{E}[d(\textbf{f}_\Theta(\mathbf{z}_{n+k}, t_{n+k}, w, \mathbf{c}), \textbf{f}_{\Theta^-}(\hat{\mathbf{z}}_n, t_n, w, \mathbf{c}))]$$

**Second Stage — Total Loss for Control Training**:

$$\Theta^a, \Theta^b = \arg\min_{\Theta^a, \Theta^b} (\mathcal{L}_{recon} + \lambda \mathcal{L}_{control})$$

where $\mathcal{L}_{recon}$ is the latent space reconstruction loss, $\mathcal{L}_{control}$ is the control joint position error in the motion space, and $\lambda=1.0$.

## Key Experimental Results

### Main Results

**Comparison of Text-to-Motion Generation (HumanML3D)**:

| Method | AITS(s)↓ | R-Precision Top3↑ | FID↓ | MM Dist↓ | Diversity→ | MModality↑ |
|------|----------|-------------------|------|----------|------------|------------|
| MDM | 24.74 | 0.611 | 0.544 | 5.566 | 9.559 | 2.799 |
| MotionDiffuse | 14.74 | 0.782 | 0.630 | 3.113 | 9.410 | 1.553 |
| MLD | 0.217 | 0.772 | 0.473 | 3.196 | 9.724 | 2.413 |
| MLD* (reprod.) | 0.225 | 0.796 | 0.450 | 3.052 | 9.634 | 2.267 |
| **MotionLCM (1-step)** | **0.030** | 0.803 | 0.467 | 3.022 | 9.631 | 2.172 |
| **MotionLCM (2-step)** | **0.035** | **0.805** | 0.368 | **2.986** | 9.640 | 2.187 |
| **MotionLCM (4-step)** | **0.043** | 0.798 | **0.304** | 3.012 | 9.607 | 2.259 |

**Comparison of Controllable Motion Generation**:

| Method | AITS(s)↓ | FID↓ | R-Precision Top3↑ | Traj. err.↓ | Loc. err.↓ | Avg. err.↓ |
|------|----------|------|-------------------|-------------|------------|------------|
| OmniControl | 81.00 | 2.328 | 0.557 | 0.3362 | 0.0322 | 0.0977 |
| MLD (LC&MC) | 0.552 | 0.555 | 0.754 | 0.2722 | 0.0215 | 0.1265 |
| **MotionLCM 1-step (LC&MC)** | **0.042** | **0.419** | **0.756** | **0.1988** | **0.0147** | **0.1127** |
| **MotionLCM 2-step (LC&MC)** | **0.047** | **0.397** | **0.759** | **0.1960** | **0.0143** | **0.1092** |

### Ablation Study

**Influence of Training Guidance Scale Range and EMA Rate**:

| Setting | R-Precision Top1↑ | FID↓ | MM Dist↓ | Diversity→ |
|------|-------------------|------|----------|------------|
| $w \in [5,15]$ (Default) | **0.502** | 0.467 | **3.022** | 9.631 |
| $w \in [2,18]$ | 0.497 | — | — | — |
| Huber Loss (Default) | **0.502** | **0.467** | — | — |
| L2 Loss | — | 0.592 | — | — |

### Key Findings

1. **Speed**: MotionLCM single-step inference takes only ~30ms, which is **1929 times faster** than OmniControl and **13 times faster** than MLD.
2. **Quality Improvement Instead of Decline**: Single-step inference outperforms the R-Precision performance of MLD with 50-step DDIM, and 4-step inference achieves the best FID (0.304).
3. **Controllability**: The latent representations generated by MotionLCM are more suitable for training Motion ControlNet than MLD, achieving significantly higher control accuracy under the same settings.
4. **Crucial Role of Motion Space Supervision**: By adding the explicit control loss in the motion space, the Traj. err. decreases from 0.2986 to 0.1988 (a 33% reduction).

## Highlights & Insights

- **First to Introduce Consistency Distillation into Motion Generation**: Demonstrates the feasibility and effectiveness of the LCM scheme within the motion latent space.
- **Real-time Controllable Generation**: MotionLCM + ControlNet enables autoregressive real-time motion generation (utilizing the final frame of the previous motion segment as the initial control signal for the subsequent one).
- **Latent-Motion Dual-Space Supervision**: Leverages the VAE decoder to decode latent space predictions into the motion space for explicit control supervision, cleverly addressing the lack of motion semantics in the latent space.
- **Pareto Optimality of Efficiency and Quality**: On the inference time vs. quality scatter plot, MotionLCM is closest to the origin, achieving the optimal balance between the two.

## Limitations & Future Work

- Relies on the quality of the pre-trained MLD model, where the upper bound of distillation is limited by the teacher model.
- The control task is currently only defined as initial pose control (the first 25% of frames), with more flexible spatiotemporal control patterns yet to be explored.
- The FID for single-step inference is slightly inferior to 4-step inference, indicating a minor quality loss under extreme acceleration.

## Rating

⭐⭐⭐⭐⭐ (5/5) — Pushes motion generation to the real-time level. The two-stage design of consistency distillation + ControlNet is simple yet efficient. The explicit supervision in the motion space solves the core challenge of latent space control, bearing great significance for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

</div>

<!-- RELATED:END -->
