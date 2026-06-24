---
title: >-
  [Paper Note] OSV: One Step is Enough for High-Quality Image to Video Generation
description: >-
  [CVPR 2025][Video Generation][Single-step video generation] Proposes a two-stage training framework, OSV, which combines GAN adversarial training and consistency distillation to achieve high-quality single-step image-to-video generation, alongside a novel video discriminator that bypasses decoding.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Single-step video generation"
  - "Consistency Distillation"
  - "Adversarial Training"
  - "Video Diffusion Acceleration"
  - "GAN"
date: 2026-05-08
content_hash: d76e7cd491110d70
---

# OSV: One Step is Enough for High-Quality Image to Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2409.11367](https://arxiv.org/abs/2409.11367)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Single-step video generation, Consistency Distillation, Adversarial Training, Video Diffusion Acceleration, GAN

## TL;DR

Proposes a two-stage training framework, OSV, which combines GAN adversarial training and consistency distillation to achieve high-quality single-step image-to-video generation, alongside a novel video discriminator that bypasses decoding.

## Background & Motivation

Although video diffusion models can generate high-quality videos, iterative denoising incurs massive computational and time costs—generating a 2-second video with SVD takes over 30 seconds on an A100 GPU. Existing acceleration methods have their respective drawbacks: Consistency Distillation (LCM) performs poorly at low steps (1-2 steps), suffers from slow training convergence, and high CFG values cause exposure issues. GAN adversarial training converges quickly but is unstable during training and prone to mode collapse in later stages, while pixel-based discriminators require VAE decoding, which increases memory and computational overhead. More fundamentally, most existing video acceleration methods are simply adapted from image diffusion models without fully considering video-specific characteristics. This work addresses these issues through a two-stage decoupled strategy, leveraging GANs in the early stage to rapidly improve quality, and introducing consistency distillation in the later stage to stabilize training and lift the performance ceiling.

## Method

### Overall Architecture

OSV consists of three network components: a student model $\theta$, an EMA target model $\theta^-$, a frozen teacher model $\phi$, and a discriminator $\psi$. The first stage employs LoRA + GAN pre-training (with real data as the positive samples for the GAN) to rapidly improve the quality of low-step generation. The second stage introduces a consistency distillation loss (with teacher-generated data as the positive samples for the GAN) to stabilize training and elevate the performance upper bound. During inference, single-step generation is supported alongside multi-step refinement.

### Key Designs

1. **Latent-Space GAN Discriminator**: Innovatively replaces the VAE decoder with a simple upsampling operator, directly feeding the upsampled video latents into a pre-trained DINOv2 discriminator (frozen backbone + trainable spatiotemporal discriminator head). Compared to ADD which requires VAE decoding into the pixel space and SF-V which needs a UNet encoder as a backbone, OSV reduces the per-iteration time from 4.29s to 2.61s and GPU memory from 73.5GB to 35.8GB on an H800, while also avoiding floating-point overflow during half-precision training.

2. **Two-Stage Decoupled Training**: The first stage (LGP) uses only GAN adversarial loss + Huber loss, with real data $\mathbf{x}_0$ as the positive sample; LoRA training ensures retention of teacher knowledge while converging rapidly. The second stage (ACD) incorporates a consistency distillation loss, where the EMA model's output serves as the positive sample, jointly refined with the adversarial loss. Key mathematical insight: the adversarial loss in the LGP stage remains non-zero (even if the model achieves consistency), which would interfere with subsequent consistency learning; in the ACD stage, the adversarial loss converges to zero as training progresses, avoiding disruption to consistency.

3. **Multi-step ODE Solver + Time Travel Sampler**: Eliminates CFG (observing that CFG negatively impacts distilled models) and replaces single-step solver with a multi-step ODE solver during training to achieve higher distillation accuracy within the same training time. During inference, a Time Travel Sampler (TTS) is designed, which leverages predictions from lower timesteps to step back to higher timesteps and solve again, achieving the effect of high-order solving.

### Loss & Training

- First stage: $\mathcal{L}_{\text{OSV}}^{g_1} = \lambda^{LGP} \cdot \text{ReLU}(1 - D_\psi(f_\theta(\mathbf{x}_{t_n}))) + d(\mathbf{x}_0, f_\theta(\mathbf{x}_{t_n}))$
- Second stage: $\mathcal{L}_{\text{OSV}}^{g_2} = \lambda^{ACD} \cdot \text{ReLU}(1 - D_\psi(f_\theta(\mathbf{x}_{t_{n+m}}))) + \lambda(t_n) d(f_{\theta^-}(\mathbf{x}_{t_n}^\phi), f_\theta(\mathbf{x}_{t_{n+m}}))$
- Huber distance: $d(x,y) = \sqrt{\|x-y\|_2^2 + c^2} - c$
- First stage uses LoRA, second stage unfreezes specific layers for fine-tuning

## Key Experimental Results

### Main Results

| Method | Steps | FVD↓ | Remarks |
|------|------|------|------|
| AnimateLCM | 8 | 184.79 | Consistency Distillation |
| SVD | 25 | 156.94 | Original Diffusion |
| SF-V | 1 | High | GAN Acceleration |
| **OSV** | **1** | **171.15** | **Single-step outperforms AnimateLCM at 8 steps** |
| **OSV** | **2+** | **Lower** | **Further improved via multi-step refinement** |

### Ablation Study

| Design Choice | FVD↓ |
|---------|------|
| LCM only (no GAN) | Poor |
| GAN only (no LCM) | Unstable in late training |
| Pixel-space discrimination (ADD-style) | Higher + Large Memory |
| Latent-space discrimination (Ours) | Lower + Small Memory |
| Single-step ODE solving | Slow convergence |
| Multi-step ODE solving (m=5) | Fast convergence & Better results |

### Key Findings

- Single-step FVD of 171.15 outperforms AnimateLCM at 8 steps (184.79) and approaches SVD at 25 steps (156.94).
- The latent-space discriminator is not only computationally efficient but also yields better performance—DINOv2 features are sufficient to discriminate video quality.
- GAN is highly effective in early-stage training but becomes unstable later, whereas consistency distillation provides a stable performance ceiling.
- CFG is detrimental to distilled models—removing CFG yields better results.

## Highlights & Insights

- Theoretical analysis of the core difference between LGP and ACD adversarial losses (non-zero vs. converging to zero) guides the two-stage design.
- Directly performing GAN discrimination in the latent space is a simple yet effective insight, substantially reducing training costs.
- Decomposing the video acceleration problem into "fast initialization" and "fine-grained tuning" provides a clear and structured roadmap.
- Utilizing LoRA in the first stage to prevent mode collapse (avoiding static image generation) is a practical and effective trick.

## Limitations & Future Work

- Being based on the SVD architecture, the video length and resolution are bounded by SVD.
- Single-step generation may still exhibit weaker motion dynamics compared to full 25-step diffusion.
- The Time Travel Sampler increases complexity when additional inference steps are used.
- The methodology can be extended to other video generation tasks such as text-to-video.

## Related Work & Insights

- **vs AnimateLCM**: Pure consistency distillation suffers from slow convergence and poor quality at low step counts; OSV resolves this with a quick GAN pre-training initialization.
- **vs SF-V**: Pure GAN methods are unstable during training and incur high costs due to using UNet encoders as discriminators; OSV's two-stage decoupled design paired with latent-space discrimination is more stable and efficient.
- **vs ADD/LADD**: Pixel-space adversarial distillation requires VAE decoding, incurring massive memory overhead; OSV proves that latent-space discrimination is equally effective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two-stage decoupled strategy is theoretically grounded, and the latent-space discriminator is simple yet highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations, with quantitative evaluation on the OpenVid-1M benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ — Deep analysis of motivations and clear summarization of the core issues.
- **Value**: ⭐⭐⭐⭐⭐ — Single-step video generation holds immense practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)

</div>

<!-- RELATED:END -->
