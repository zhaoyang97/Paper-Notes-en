---
title: >-
  [Paper Note] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][Video Generation][distillation] DOLLAR combines variational score distillation (VSD) and consistency distillation to achieve few-step video generation, and introduces a latent-space reward model fine-tuning method to further optimize generation quality. The 4-step model generates 10-second videos (128 frames @ 12 FPS) achieving a VBench score of 82.57, surpassing both the teacher model and commercial baselines such as Gen-3 and Kling…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "distillation"
  - "consistency distillation"
  - "latent reward"
  - "few-step"
  - "VBench"
date: 2026-05-08
content_hash: 65c2d028667fa453
---

# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**Conference**: ICCV 2025
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)  
**Code**: N/A  
**Area**: Video Generation / Diffusion Model Acceleration
**Keywords**: video generation, distillation, consistency distillation, latent reward, few-step, VBench

## TL;DR
DOLLAR combines variational score distillation (VSD) and consistency distillation to achieve few-step video generation, and introduces a latent-space reward model fine-tuning method to further optimize generation quality. The 4-step model generates 10-second videos (128 frames @ 12 FPS) achieving a VBench score of 82.57, surpassing both the teacher model and commercial baselines such as Gen-3 and Kling; 1-step distillation achieves a 278.6× speedup.

## Background & Motivation
Video diffusion models require a large number of sampling steps (typically 50+) to generate high-quality videos, incurring prohibitive computational costs. Works such as SANA-Sprint have already achieved 1–4-step generation in the image domain, but step distillation for video is more challenging—videos must maintain temporal consistency, and naïvely reducing steps readily causes temporal flickering and quality degradation. Additionally, existing video distillation methods are generally tied to a fixed step count (e.g., exactly 4 steps), lacking flexibility.

## Core Problem
How can sampling steps be drastically reduced while preserving video quality and diversity? How can reward signals be used to further optimize specific quality metrics of a distilled model?

## Method

### Overall Architecture
DOLLAR proceeds in two stages: (1) Hybrid Distillation—combining VSD and consistency distillation to compress the teacher's 50-step capability into 1–4 steps; (2) Latent Reward Optimization—further fine-tuning the distilled student with a latent reward model to improve targeted quality dimensions.

### Key Designs
1. **Hybrid VSD + Consistency Distillation**: VSD aligns the teacher's multi-step distribution to the student's few-step distribution via distribution matching, preserving generative diversity; consistency distillation ensures the student produces consistent outputs across step counts, avoiding step-specific training. The two objectives are complementary—VSD preserves diversity, consistency distillation preserves quality.

2. **Latent Reward Model Fine-tuning**: Does not require a differentiable reward model; operates entirely in latent space, substantially reducing GPU memory requirements. Can optimize arbitrary designated reward metrics (aesthetic quality, text alignment, temporal consistency, etc.). This enables the distilled model to not only run faster but also surpass the teacher in specific dimensions.

3. **Few-Step Generation for 10-Second Long Videos**: Validated on 128-frame @ 12 FPS (~10 seconds) videos—considerably more challenging than the short videos (2–4 seconds) used by most prior methods.

### Loss & Training
Stage 1: Hybrid VSD loss + consistency loss. Stage 2: Latent reward fine-tuning.

## Key Experimental Results

| Method | Steps | VBench Score |
|--------|-------|-------------|
| Teacher (50-step DDIM) | 50 | < 82.57 |
| Gen-3 | — | < 82.57 |
| T2V-Turbo | — | < 82.57 |
| Kling | — | < 82.57 |
| **DOLLAR (4-step)** | **4** | **82.57** |

- The 4-step student **surpasses the teacher model** and commercial models including Gen-3 and Kling on VBench.
- 1-step distillation achieves a **278.6× speedup**, approaching real-time generation.
- Human evaluation further confirms the 4-step model outperforms the 50-step teacher.
- Validated on 10-second long videos (128 frames)—a more demanding setting.

### Ablation Study Highlights
- Hybrid distillation > VSD alone > consistency distillation alone.
- Latent reward fine-tuning yields further VBench score improvements.
- Advantages of latent reward: memory-efficient and does not require reward differentiability.

## Highlights & Insights
- **Distilled Student Surpasses Teacher**: 4-step generation exceeds 50-step sampling quality—an achievement analogous to what SANA-Sprint demonstrated in the image domain, now extended to video.
- **278.6× Speedup** with 1-step generation makes real-time video generation a tangible prospect.
- **Latent Reward Fine-tuning** is a practically motivated innovation—requires no reward differentiability, is memory-friendly, and can target arbitrary quality dimensions.
- **10-Second Video Validation** is substantially longer than most comparable works, closer to real-world application requirements.
- From Adobe Research, with a strong application-oriented focus.

## Limitations & Future Work
- Code and model weights are not released.
- VBench as the sole evaluation metric has known limitations and may not fully reflect human preference.
- Training costs of the distillation procedure are not reported in detail.
- No comparison with recent open-source models such as CogVideoX or Wan.

## Related Work & Insights
- **vs. SANA-Sprint**: SANA-Sprint uses sCM+LADD for 1-step image generation; DOLLAR applies VSD+consistency distillation to video—both adopt hybrid distillation paradigms across different domains.
- **vs. T2V-Turbo**: T2V-Turbo also performs video step distillation; DOLLAR reports superior VBench performance.
- **vs. AnimateLCM**: AnimateLCM accelerates video generation via LCM; DOLLAR's hybrid scheme is more advanced.
- **Insight**: The latent reward fine-tuning paradigm can be combined with VACE for targeted editing quality improvement in unified video editing frameworks; DOLLAR's distillation approach may also be applicable to Dita's action diffusion denoising to accelerate robot control response.

## Rating
- Novelty: ⭐⭐⭐⭐ — VSD+consistency hybrid distillation is a novel application in the video domain; latent reward fine-tuning is practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive VBench evaluation, human study, and 10-second long-video testing.
- Writing Quality: ⭐⭐⭐⭐ — Method presentation is clear.
- Value: ⭐⭐⭐⭐ — A significant advance in video diffusion model acceleration, approaching real-time video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](../../ICML2026/video_generation/sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2025/video_generation/flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[ICLR 2026\] Towards One-Step Causal Video Generation via Adversarial Self-Distillation](../../ICLR2026/video_generation/towards_one-step_causal_video_generation_via_adversarial_self-distillation.md)
- [\[CVPR 2026\] Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation](../../CVPR2026/video_generation/reward_forcing_efficient_streaming_video_generation_with_rewarded_distribution_m.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)

</div>

<!-- RELATED:END -->
