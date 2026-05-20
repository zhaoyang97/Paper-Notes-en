---
title: >-
  [Paper Note] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][Video Generation][distillation] DOLLAR combines variational score distillation (VSD) and consistency distillation to achieve few-step video generation…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "distillation"
  - "consistency distillation"
  - "latent reward"
  - "few-step"
  - "VBench"
date: 2026-05-08
content_hash: f8c2b03812b799cc
---

# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**Conference**: ICCV 2025
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)  
**Code**: N/A  
**Area**: Video Generation / Diffusion Model Acceleration
**Keywords**: video generation, distillation, consistency distillation, latent reward, few-step generation

## TL;DR

DOLLAR combines variational score distillation (VSD) and consistency distillation to achieve few-step video generation, and introduces a latent-space reward model fine-tuning method to further optimize specific quality dimensions. The 4-step student model achieves a VBench score of 82.57, surpassing the teacher model and commercial baselines such as Gen-3 and Kling, while 1-step distillation yields a 278.6× sampling speedup.

## Background & Motivation

**Background**: Diffusion probabilistic models have achieved remarkable progress in video generation, enabling high-quality text-to-video synthesis. However, these models typically require more than 50 iterative sampling steps to produce satisfactory results, incurring substantial computational overhead.

**Limitations of Prior Work**: Directly reducing the number of sampling steps tends to degrade video quality or diversity. Distillation methods developed for the image domain (e.g., SANA-Sprint, which supports 1–4-step image generation) face additional challenges in the video domain—videos must maintain temporal consistency, and naïvely reducing steps readily introduces temporal flickering and quality degradation. Furthermore, existing video distillation methods typically support only a fixed number of steps (e.g., exactly 4 steps), lacking step-count flexibility.

**Key Challenge**: How can sampling steps be drastically reduced while preserving video quality and diversity? How can reward signals be used to further optimize specific quality metrics of a distilled model?

**Goal**: DOLLAR proposes a hybrid distillation strategy for flexible-step, high-quality video generation, and introduces latent-space reward fine-tuning to further improve specific quality dimensions.

**Key Insight**: VSD (which preserves diversity) and consistency distillation (which preserves quality and step-count flexibility) are complementarily combined, while reward optimization is performed in latent space rather than pixel space to reduce GPU memory overhead.

**Core Idea**: Hybrid distillation + latent-space reward fine-tuning = few-step, high-quality video generation.

## Method

### Overall Architecture

DOLLAR employs a two-stage training strategy: (1) **Hybrid Distillation Stage**—combining variational score distillation (VSD) and consistency distillation (CD) to compress the teacher model's 50-step sampling capability into a 1–4-step student model; (2) **Latent Reward Optimization Stage**—fine-tuning the distilled student model with a latent reward model to selectively improve designated quality metrics.

### Key Designs

1. **Variational Score Distillation (VSD)**:
    - *Function*: Aligns the multi-step sampling distribution of the teacher model to the few-step distribution of the student model via distribution matching.
    - *Mechanism*: VSD minimizes the KL divergence between teacher and student output distributions. Denoting the teacher as $\epsilon_\phi$ and the student as $\epsilon_\theta$, the VSD objective is $\mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon_\phi(x_t, t)\|^2\right]$, where $x_t$ is the noise-perturbed video latent.
    - *Design Motivation*: Pure consistency distillation may cause mode collapse; VSD preserves generative diversity through distribution matching.

2. **Consistency Distillation (CD)**:
    - *Function*: Ensures the student model produces consistent outputs across different step counts.
    - *Mechanism*: For the same noise input, the student is required to produce consistent predictions at any intermediate timestep, i.e., $f_\theta(x_t, t) \approx f_\theta(x_{t'}, t')$, where $x_t$ and $x_{t'}$ lie on the same PF-ODE trajectory.
    - *Design Motivation*: Allows the student to operate flexibly between 1 and 4 steps without step-specific training.

3. **Hybrid Distillation Strategy**:
    - *Function*: Jointly trains VSD and CD with learned weighting.
    - *Mechanism*: The total loss is $\mathcal{L} = \lambda_{\text{VSD}} \mathcal{L}_{\text{VSD}} + \lambda_{\text{CD}} \mathcal{L}_{\text{CD}}$.
    - *Design Motivation*: VSD preserves diversity while CD preserves quality and step flexibility; the two objectives are complementary.

4. **Latent Reward Fine-tuning**:
    - *Function*: Further fine-tunes the distilled student using a latent reward model to improve specified quality dimensions.
    - *Mechanism*: The approach does not require a differentiable reward model; instead, reward signals are computed directly in latent space, bypassing video decoding to pixel space and substantially reducing GPU memory consumption. The method can target arbitrary reward metrics (aesthetic quality, text alignment, temporal consistency, etc.).
    - *Design Motivation*: Conventional reward fine-tuning requires gradient computation in pixel space, which is prohibitive for 128-frame videos; operating in latent space simultaneously addresses memory and differentiability constraints.

### Loss & Training

- **Stage 1**: Joint training with VSD loss and consistency loss, with dynamically adjusted weights throughout training.
- **Stage 2**: Latent reward fine-tuning using policy gradient methods to optimize reward metrics.
- **Training Data**: 50-step DDIM samples from the teacher model serve as the target distribution.
- **Video Specification**: 128 frames @ 12 FPS (~10 seconds), substantially longer than most prior methods (2–4 seconds).

## Key Experimental Results

### Main Results

| Method | Steps | VBench Score | Speedup |
|--------|-------|-------------|---------|
| Teacher (50-step DDIM) | 50 | ~81 | 1× |
| DOLLAR (4-step) | 4 | **82.57** | 12.5× |
| DOLLAR (1-step) | 1 | ~79 | **278.6×** |
| Gen-3 | — | <82.57 | — |
| T2V-Turbo | 4 | <82.57 | — |
| Kling | — | <82.57 | — |

### Ablation Study

| Configuration | VBench Score | Notes |
|--------------|-------------|-------|
| VSD only | ~80 | Good diversity, insufficient quality |
| CD only | ~79 | Good quality, poor diversity |
| VSD + CD | ~81.5 | Complementary improvement |
| VSD + CD + Latent Reward | **82.57** | Reward fine-tuning yields further gains |

### Key Findings

- The 4-step student model surpasses the 50-step teacher on VBench; distillation combined with reward fine-tuning can exceed the original model's quality.
- 1-step distillation achieves a 278.6× speedup, approaching real-time generation.
- Human evaluation further confirms that the 4-step student outperforms the 50-step teacher.
- Latent reward fine-tuning yields significant improvements in targeted dimensions (e.g., temporal consistency) for the distilled model.

## Highlights & Insights

- **Distillation + Reward = Surpassing the Teacher**: The distilled student is not only faster but can also exceed the teacher in quality through reward fine-tuning, challenging the common assumption that distillation inevitably incurs quality loss.
- **Latent Space vs. Pixel Space**: Performing reward computation in latent space substantially reduces memory requirements, making reward fine-tuning feasible for long videos (10 seconds, 128 frames).
- **Step-Count Flexibility**: The step flexibility conferred by consistency distillation (1–4 steps) provides a practical quality–speed trade-off for diverse application scenarios.
- **Validation on 10-Second Long Videos**: Experiments are conducted on 128-frame @ 12 FPS videos (~10 seconds), far exceeding the short-video settings of most prior methods.

## Limitations & Future Work

- Code is not publicly released, limiting reproducibility.
- Implementation details and training procedures for the latent reward model are insufficiently described.
- Evaluation is restricted to VBench; more diverse video quality benchmarks are lacking.
- Applicability to longer videos (e.g., 30+ seconds) or higher-resolution settings is not discussed.
- Reward fine-tuning is susceptible to over-optimization relative to reward model quality.

## Related Work & Insights

- **Consistency Models**: Song et al.'s consistency models provide the theoretical foundation for the CD component of this work.
- **ProlificDreamer/VSD**: Wang et al.'s VSD is extended by this paper to the video distillation setting.
- **SANA-Sprint**: A prior work achieving 1–4-step generation in the image domain.
- **T2V-Turbo**: A prior video distillation work; DOLLAR reports superior performance.
- **Insight**: The latent reward fine-tuning paradigm is generalizable to other generative models (3D generation, audio synthesis, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of VSD+CD hybrid distillation and latent reward fine-tuning is relatively novel in the video generation domain.
- Experimental Thoroughness: ⭐⭐⭐ — VBench results are convincing, but ablation details and multi-benchmark comparisons are insufficient.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated problem formulation.
- Value: ⭐⭐⭐⭐ — Near real-time video generation has significant practical value.

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

- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_fewstep_controllable_video_generation.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](../../CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)

</div>

<!-- RELATED:END -->
