---
title: >-
  [Paper Note] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization
description: >-
  [ICCV 2025][Video Generation][Video generation acceleration] This paper proposes DOLLAR, which combines Variational Score Distillation (VSD) and Consistency Distillation (CD) to achieve few-step video generation, and introduces a latent reward model fine-tuning strategy to further improve quality. The 4-step student model achieves a VBench score of 82.57, surpassing the teacher model and baselines such as Gen-3 and Kling, while the single-step distillation achieves a 278.6× speedup.
tags:
  - ICCV 2025
  - Video Generation
  - Video generation acceleration
  - distillation
  - consistency distillation
  - variational score distillation
  - latent reward optimization
date: 2026-05-08
content_hash: 73c5dfa227aabe39
---

# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**Conference**: ICCV 2025
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)
**Code**: None
**Area**: Video Generation / Diffusion Models
**Keywords**: Video generation acceleration, distillation, consistency distillation, variational score distillation, latent reward optimization

## TL;DR

This paper proposes DOLLAR, which combines Variational Score Distillation (VSD) and Consistency Distillation (CD) to achieve few-step video generation, and introduces a latent reward model fine-tuning strategy to further improve quality. The 4-step student model achieves a VBench score of 82.57, surpassing the teacher model and baselines such as Gen-3 and Kling, while the single-step distillation achieves a 278.6× speedup.

## Background & Motivation

**Background**: Diffusion probabilistic models have achieved remarkable progress in video generation, producing high-quality long videos (e.g., 10 seconds, 128 frames, 12 FPS). However, inference requires a large number of sampling steps (typically 50 DDIM steps), each involving a forward pass through a large-scale 3D UNet or DiT, making video generation a process that can take several minutes.

**Limitations of Prior Work**: Directly reducing the number of sampling steps severely degrades video quality and diversity. Existing acceleration methods follow two main lines: Consistency Distillation (CD) enables few-step sampling but tends to lose detail and diversity, while Score Distillation (SD) preserves quality but requires more steps. Each approach has its own drawbacks, and no unified solution simultaneously addresses quality, diversity, and speed.

**Key Challenge**: CD forces the model to directly predict the final output within one or few steps, which is prone to mode collapse and blurry details. Variational Score Distillation (VSD) preserves distribution matching but is insufficiently stable when compressed to few steps. Video, compared to images, is higher-dimensional and demands stricter temporal consistency, making these issues more pronounced.

**Goal**: To design a two-stage distillation scheme enabling a student model to generate high-quality, diverse videos in 1–4 steps, along with a general fine-tuning mechanism to further improve performance according to arbitrary reward metrics.

**Key Insight**: The authors observe that VSD and CD are complementary — VSD excels at preserving distributional diversity but performs poorly in single-step generation, while CD excels at few-step generation but is prone to overfitting. A sequential strategy of VSD warm-up followed by CD refinement can capture the advantages of both. Furthermore, introducing a reward model in latent space bypasses the decoding bottleneck, enabling efficient fine-tuning with any quality metric.

**Core Idea**: A two-stage distillation pipeline (VSD→CD) yields a high-quality few-step base model, which is then refined via Latent Reward Optimization (LRO) according to specified quality metrics, achieving an optimal balance among quality, speed, and diversity.

## Method

### Overall Architecture

DOLLAR's training pipeline consists of three stages: (1) **VSD stage**, where the student model learns the teacher's score function to establish initial few-step generation capability; (2) **CD stage**, where further training on top of VSD enables high-quality video output in 1–4 steps; (3) **LRO stage**, where a lightweight reward proxy model is trained in latent space and the student model is fine-tuned via the REINFORCE algorithm to maximize arbitrary reward metrics. At inference time, only 1–4 denoising steps are required to generate 10-second, 128-frame videos.

### Key Designs

1. **Two-Stage Distillation: VSD + CD (Variational Score Distillation + Consistency Distillation)**:

    - **Function**: Progressively compress the teacher model's sampling steps while preserving quality and diversity.
    - **Mechanism**: In the first stage (VSD), the student is trained to match the teacher's score function $\nabla_{x_t} \log p_\text{teacher}(x_t)$, ensuring that the student's predictions at each noise level are consistent with the teacher. The VSD loss is $\mathcal{L}_\text{VSD} = \mathbb{E}_{t, x_t}[\|\epsilon_\theta(x_t, t) - \epsilon_\text{teacher}(x_t, t)\|^2]$, augmented with an auxiliary model that estimates the student's own score to prevent mode collapse. In the second stage (CD), building on the VSD-pretrained model, the student is forced to produce consistent outputs at adjacent time steps: $\mathcal{L}_\text{CD} = \|f_\theta(x_t, t) - f_{\theta^-}(x_{t'}, t')\|$, where $\theta^-$ denotes EMA parameters.
    - **Design Motivation**: VSD alone is insufficient when distilled to 1–4 steps, while CD alone from random initialization is susceptible to mode collapse. The progressive VSD→CD strategy provides CD with a favorable initialization, facilitating convergence to high-quality solutions.

2. **Latent Reward Optimization (LRO)**:

    - **Function**: Fine-tune the distilled model using arbitrary quality metrics (e.g., VBench scores, aesthetic scores).
    - **Mechanism**: Conventional reward optimization requires decoding latent variables into pixel space before computing rewards, which is extremely memory-intensive for video. The core innovation of LRO is to train a lightweight reward proxy model $R_\phi(z)$ in latent space to approximate pixel-space rewards. Training data pairs of (latent representation $z$, corresponding reward $r$) are collected, and $R_\phi$ is trained via regression. The student model is then fine-tuned using the REINFORCE policy gradient: $\nabla_\theta \mathbb{E}[R_\phi(z)] \approx \mathbb{E}[\nabla_\theta \log p_\theta(z) \cdot R_\phi(z)]$.
    - **Design Motivation**: Operating in latent space avoids the cost of decoding (which is especially expensive for video), and REINFORCE does not require the reward function to be differentiable, making any black-box quality metric a valid optimization objective.

3. **Temporal Consistency Preservation**:

    - **Function**: Ensure that few-step generated videos remain smooth and coherent along the temporal dimension.
    - **Mechanism**: Temporal constraints are incorporated into the distillation losses during both the VSD and CD stages. Specifically, in addition to comparing per-frame quality, the distillation loss also compares optical flow consistency and feature-space similarity between adjacent frames. The CD stage employs 3D consistency constraints to ensure that the model's predictions at time steps $t$ and $t'$ are also consistent across the temporal dimension.
    - **Design Motivation**: Temporal consistency is the key distinction between video and image distillation. Without such constraints, few-step sampling may produce flickering or temporal incoherence even when per-frame quality is acceptable.

### Loss & Training

Three-stage training with the following losses: (1) **VSD stage**: $\mathcal{L} = \mathcal{L}_\text{VSD} + \lambda_\text{temp} \mathcal{L}_\text{temporal}$; (2) **CD stage**: $\mathcal{L} = \mathcal{L}_\text{CD} + \lambda_\text{temp} \mathcal{L}_\text{temporal}$; (3) **LRO stage**: REINFORCE policy gradient with KL regularization to prevent deviation from the distilled model. The three stages are trained sequentially, with total training cost approximately 10% of the teacher model's training budget.

## Key Experimental Results

### Main Results

| Method | Steps | VBench↑ | Quality↑ | Diversity↑ | Speedup |
|--------|-------|---------|----------|------------|---------|
| DOLLAR (4-step) | 4 | **82.57** | **84.1** | **78.3** | 12.5× |
| DOLLAR (1-step) | 1 | 80.12 | 81.5 | 76.1 | **278.6×** |
| Teacher (50-step) | 50 | 81.23 | 83.2 | 77.5 | 1× |
| Gen-3 | — | 80.45 | 82.1 | 76.8 | — |
| T2V-Turbo | 4 | 78.92 | 80.3 | 74.5 | 12.5× |
| Kling | — | 79.88 | 81.0 | 76.2 | — |

### Ablation Study

| Configuration | VBench↑ | Note |
|---------------|---------|------|
| Full DOLLAR (VSD+CD+LRO) | **82.57** | Complete three-stage pipeline |
| VSD+CD (w/o LRO) | 81.45 | Without reward optimization; already surpasses teacher |
| CD only (w/o VSD warm-up) | 79.23 | Mode collapse, poor diversity |
| VSD only (w/o CD refinement) | 80.18 | Acceptable at 4 steps but below CD-refined |
| VSD+CD + pixel-space reward | OOM | Out of memory after video decoding |
| LRO w/ differentiable reward | 82.31 | Replacing REINFORCE with differentiable reward; marginal difference |

### Key Findings

- The VSD→CD two-stage distillation is the critical design: CD alone underperforms the full method by 3.34 points (mode collapse), and VSD alone underperforms by 2.39 points (insufficient precision at few steps).
- LRO contributes an additional 1.12-point gain on top of VSD+CD, validating the effectiveness of latent reward optimization.
- The 4-step DOLLAR model achieves 82.57, surpassing the 50-step teacher model (81.23), further confirming the possibility that distilled students can outperform their teachers.
- The 1-step model achieves a 278.6× speedup, approaching real-time generation of 10-second videos.
- Human evaluations further confirm that the 4-step student model outperforms the 50-step teacher model.

## Highlights & Insights

- **Complementary VSD+CD Distillation Strategy**: Using VSD to establish a good initial distribution and then applying CD to compress the number of steps is an elegant "coarse-to-fine" distillation paradigm that can be generalized to accelerating diffusion models for other modalities (e.g., audio, 3D).
- **Elegant Design of the Latent Reward Model**: Computing rewards in latent space rather than pixel space elegantly resolves the memory bottleneck of video decoding. The use of REINFORCE also removes the requirement for differentiable rewards, making any black-box evaluation metric a valid optimization target — an idea of broad value for fine-tuning generative models with high-dimensional outputs.
- **Surpassing the Teacher Model**: The distilled student model is not only faster but also better, benefiting from the additional regularization introduced by consistency constraints and reward optimization. This phenomenon suggests that the distillation process can "inject" new inductive biases to correct deficiencies in the teacher.

## Limitations & Future Work

- The three-stage training pipeline is complex and requires careful tuning of hyperparameters and training durations for each stage.
- The quality of the latent reward model depends on the representativeness of the training data and may fail on out-of-distribution prompts.
- The current work focuses on 10-second videos; effectiveness on longer videos (>1 minute) remains unvalidated.
- Although VBench is a widely adopted metric, its correlation with human perception remains debated.
- Future directions include: extending LRO to multi-reward joint optimization; exploring zero-step generation (e.g., in the style of consistency models); and applying the method to image-to-video and video editing tasks.

## Related Work & Insights

- **vs. T2V-Turbo**: T2V-Turbo applies consistency distillation for video acceleration but lacks the VSD warm-up stage, making it susceptible to mode collapse. DOLLAR's VSD+CD two-stage strategy outperforms it in both quality and diversity.
- **vs. AnimateLCM**: AnimateLCM performs video distillation based on the Latent Consistency Model (LCM), focusing primarily on image quality while neglecting temporal consistency. DOLLAR explicitly models temporal constraints.
- **vs. Progressive Distillation (image domain)**: Progressive distillation schemes for images (e.g., SDXL-Turbo) need not address temporal consistency. DOLLAR extends the progressive distillation paradigm to video and addresses the novel challenges posed by the temporal dimension.
- **vs. RLHF for LLMs**: LRO shares the same spirit as RLHF for LLMs — both use a reward model to fine-tune a generative model — but DOLLAR operates in latent space to overcome the memory constraints inherent to video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The VSD+CD two-stage combination and LRO represent meaningful innovations, though each individual component is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive VBench evaluation, multi-baseline comparisons, human evaluation, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear; the three-stage pipeline is easy to follow.
- **Value**: ⭐⭐⭐⭐⭐ A 278.6× speedup with quality surpassing the teacher model is of significant practical importance for the deployment of video generation systems.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)

<!-- RELATED:END -->
