---
title: >-
  [Paper Note] Learning Few-Step Diffusion Models by Trajectory Distribution Matching
description: >-
  [ICCV 2025][Image Generation][Diffusion Distillation] This paper proposes Trajectory Distribution Matching (TDM), a novel paradigm that unifies trajectory distillation and distribution matching by aligning the marginal distributions of student and teacher ODE trajectories at the distributional level. TDM enables efficient few-step diffusion model distillation, requiring only 2 A800 GPU-hours to distill PixArt-α into a 4-step generator that surpasses the teacher model.
tags:
  - ICCV 2025
  - Image Generation
  - Diffusion Distillation
  - Few-Step Generation
  - Trajectory Distribution Matching
  - Score Distillation
  - Text-to-Image Acceleration
date: 2026-05-08
content_hash: 5639579ddea6bb6d
---

# Learning Few-Step Diffusion Models by Trajectory Distribution Matching

**Conference**: ICCV 2025
**arXiv**: [2503.06674](https://arxiv.org/abs/2503.06674)
**Code**: [Project Page](https://tdm-t2x.github.io/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Diffusion Distillation, Few-Step Generation, Trajectory Distribution Matching, Score Distillation, Text-to-Image Acceleration

## TL;DR

This paper proposes Trajectory Distribution Matching (TDM), a novel paradigm that unifies trajectory distillation and distribution matching by aligning the marginal distributions of student and teacher ODE trajectories at the distributional level. TDM enables efficient few-step diffusion model distillation, requiring only 2 A800 GPU-hours to distill PixArt-α into a 4-step generator that surpasses the teacher model.

## Background & Motivation

Accelerating diffusion model sampling is critical for efficient deployment of AIGC systems. Existing distillation methods fall into two major categories, each with notable limitations:

**Distribution Matching methods** (e.g., DMD, SiD): achieve distribution-level alignment via score distillation and perform strongly in one-step generation, but are primarily optimized for single-step inference and **lack flexibility for multi-step sampling**—additional steps cannot be effectively exploited.

**Trajectory Distillation methods** (e.g., Progressive Distillation, Consistency Models): simulate teacher ODE trajectories at the instance level and support multi-step sampling, but **instance-level trajectory matching imposes high demands on model capacity**, and numerical errors from solving the teacher ODE propagate to the student.

**Root Cause**: Distribution matching discards intermediate trajectory information, while trajectory distillation suffers from the difficulty of instance-level matching.

**Core Idea of TDM**: Match trajectories at the **distributional level** rather than at the instance level. This non-trivially unifies the advantages of both paradigms—leveraging trajectory information for fine-grained knowledge transfer while reducing learning difficulty through distribution-level alignment.

## Method

### Overall Architecture

TDM parameterizes a $K$-step student model as a discrete ODE sampler that generates intermediate samples $x_{t_i}$ along the trajectory. Distribution-level score distillation is then applied to align the marginal distribution $p_{\theta, t_i}$ at each timestep on the student trajectory with the corresponding teacher distribution $p_{\phi, t_i}$. The entire process is data-free and does not require solving the teacher ODE.

### Key Designs

1. **TDM Objective**:

    - **Function**: Minimize the reverse KL divergence from the teacher distribution at each timestep along the student trajectory.
    - **Mechanism**: The objective is formulated as:
    $L(\theta) = \sum_{i=0}^{K-1} \sum_{\tau=t_i}^{t_{i+1}} \lambda_\tau \text{KL}(p_{\theta, \tau|t_i}(x_\tau) \| p_{\phi, \tau}(x_\tau))$
      where $p_{\theta, \tau|t_i}$ is the diffusion distribution of student trajectory samples. The gradient is computed as:
    $\nabla_\theta L(\theta) \approx \sum_{i,\tau} \lambda_\tau [s_\psi(x_\tau, \tau) - s_\phi(x_\tau, \tau)] \frac{\partial x_{t_i}}{\partial \theta}$
      A fake score $s_\psi$ is required to approximate the score of the student distribution.
    - **Design Motivation**: (1) Only student samples are needed—no teacher ODE sampling required (data-free and efficient); (2) numerical errors from teacher ODE solving are avoided; (3) distribution-level matching imposes lower model capacity requirements than instance-level matching. A key design ensures that diffusion intervals at different timesteps are **non-overlapping**, allowing a single fake score model to naturally distinguish between different distributions.

2. **Sampling-Steps-Aware Objective**:

    - **Function**: Enable a single model to support flexible deterministic multi-step sampling.
    - **Mechanism**: The objective is extended to an expectation over the number of sampling steps $K$:
    $\mathbb{E}_K \sum_{i=0}^{K-1} \sum_{\tau=t_i^K}^{t_{i+1}^K} \lambda_\tau \text{KL}(p_{\theta, \tau|t_i^K}(x_\tau|K) \| p_{\phi,\tau}(x_\tau))$
      where $K$ is injected as a conditioning signal into both the student and fake score models. The resulting model is referred to as TDM-unify.
    - **Design Motivation**: Existing deterministic sampling distillation methods bind their learning objectives to a fixed number of steps, making it impossible to flexibly switch to $M < K$ steps after training. Sharing a non-step-aware fake score introduces theoretical bias (formally derived in the paper), necessitating a steps-aware fake score.

3. **Pseudo-Huber Surrogate Training Objective**:

    - **Function**: Replace the L2 loss with a Pseudo-Huber metric to stabilize training.
    - **Mechanism**: The learning objective becomes:
    $L(\theta) = \sum_{i,\tau} \sqrt{\|x_{t_i} - \text{sg}(\tilde{x}_{t_i})\|_2^2 + c^2} - c$
      where $c = 0.00054\sqrt{d}$. This normalizes gradients and yields more stable training.
    - **Design Motivation**: TDM and Consistency Models share a structural similarity in their learning formulations (both minimize the distance between generated and corrected samples), motivating the transfer of the Pseudo-Huber metric from iCT to the score distillation framework.

### Loss & Training

- **Fake Score Training**: Denoising score matching with importance sampling is applied to efficiently learn the score in the neighborhood of student trajectory samples.
- **Better Teacher, Better Student**: For SD-v1.5, fine-tuning the teacher on high-quality data prior to distillation is recommended.
- Backpropagation considers only a single ODE step to reduce GPU memory consumption.
- Training efficiency is exceptional: SDXL requires only 2 A800 GPU-days; PixArt-α converges in 500 iterations / 2 A800 GPU-hours.

## Key Experimental Results

### Main Results

Text-to-image generation quality comparison (SDXL backbone, 4 steps):

| Method | HPS↑ | AeS↑ | CLIP↑ | Training Cost |
|--------|------|------|-------|---------------|
| SDXL-Lightning | 32.71 | 6.23 | 34.62 | — |
| Hyper-SD | 34.14 | 6.18 | 34.27 | — |
| DMD2 | 31.46 | 5.88 | 35.51 | 160 A100-days |
| LCM | 29.41 | 5.84 | 34.84 | 32 A100-days |
| **TDM (Ours)** | **34.88** | **6.28** | **36.08** | **2 A800-days** |

PixArt-α backbone (4 steps, 1024 resolution):

| Method | HPS↑ | AeS↑ | CLIP↑ | Data-Free? |
|--------|------|------|-------|------------|
| PixArt-α Teacher (25 steps) | 32.21 | 6.23 | 34.11 | — |
| LCM (4 steps) | 30.55 | 6.17 | 33.49 | ✗ |
| **TDM (4 steps)** | **33.21** | **6.42** | **33.66** | **✓** |

SD-v1.5 backbone (TDM-unify, 1 step & 4 steps):

| Method | Steps | HPS Avg↑ | AeS↑ | CLIP↑ |
|--------|-------|---------|------|-------|
| Hyper-SD | 1 | 28.01 | 5.64 | 30.87 |
| TDM-unify-SFT | 1 | **28.90** | **6.02** | **32.12** |
| DMD2 | 4 | 29.49 | 5.91 | 31.53 |
| TDM-unify-SFT | 4 | **31.31** | **6.08** | **32.77** |

### Ablation Study

| Configuration | HPS Avg↑ | AeS↑ | Note |
|---------------|---------|------|------|
| TDM w/o trajectory (K=1) | 28.54 | 5.97 | Degenerates to pure distribution matching |
| TDM w/ trajectory (K=4) | 30.83 | 6.07 | Trajectory information yields significant gains |
| TDM + Pseudo-Huber | 31.31 | 6.08 | Huber metric further improves performance |
| Shared fake score (non-step-aware) | ↓ | ↓ | Validates necessity of steps-aware objective |

LoRA adaptation to unseen custom models (Realistic, SD-v1.5, 4 steps):

| Method | HPS Avg↑ | FID↓ |
|--------|---------|------|
| LCM | 27.72 | 26.89 |
| Hyper-SD | 30.36 | 37.83 |
| **TDM** | **31.22** | **20.23** |

### Key Findings

- TDM outperforms SDXL-Lightning by +2.17 HPS and DMD2 by +3.42 HPS on SDXL, **while requiring less than 1/80 of DMD2's training cost**.
- The 4-step TDM-distilled PixArt-α **surpasses the 25-step teacher model** on both HPS and AeS.
- TDM-unify enables **flexible 1-step and 4-step sampling within a single model**.
- The method generalizes to video diffusion distillation: distilling CogVideoX-2B into a 4-step generator improves VBench total score from 80.91 to 81.65.

## Highlights & Insights

- **Unified Paradigm**: The first work to non-trivially unify trajectory distillation and distribution matching, with rigorous theoretical derivation of their connection.
- **Extreme Training Efficiency**: PixArt-α converges in 500 iterations (0.01% of teacher training cost); SDXL requires only 2 A800 GPU-days.
- **Fully Data-Free**: Requires no real image data, alleviating data acquisition and copyright concerns.
- **Deterministic + Flexible Sampling**: TDM-unify is the first distillation method that simultaneously supports deterministic sampling and flexible step count adjustment.
- **Theoretical Connection to Consistency Models**: Reveals a deep structural similarity between TDM and CM learning objectives.

## Limitations & Future Work

- The quality of fake score learning directly affects distillation performance and requires careful tuning of the training strategy.
- Performance is upper-bounded by teacher model quality (a trade-off inherent to data-free approaches); the "Better Teacher, Better Student" strategy partially mitigates this at the cost of additional preprocessing.
- Backpropagating through only a single ODE step is a memory-driven compromise; multi-step backpropagation may yield further gains.
- The steps-aware fake score increases training complexity.

## Related Work & Insights

- Closely related to but fundamentally different from DMD2: DMD2 predicts clean samples at each step and ignores intermediate trajectories, while TDM explicitly simulates the deterministic trajectory.
- The Pseudo-Huber metric from Consistency Models is effectively transferred to the score distillation framework.
- Key insight: Trajectory information is a unique asset of diffusion models that should not be discarded during distillation but instead leveraged at the distributional level.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Outstanding theoretical contribution in unifying two distillation paradigms with rigorous derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple backbones (SD-v1.5/SDXL/PixArt-α/CogVideoX), step counts, and metrics.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretically clear; the logical flow from trajectory distillation to distribution matching is coherent and well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — State-of-the-art performance combined with extreme training efficiency and a novel theoretical paradigm; high practical impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](inference-time_diffusion_model_distillation.md)

<!-- RELATED:END -->
