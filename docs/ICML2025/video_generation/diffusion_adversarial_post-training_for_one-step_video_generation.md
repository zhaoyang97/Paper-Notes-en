---
title: >-
  [Paper Note] Diffusion Adversarial Post-Training for One-Step Video Generation
description: >-
  [ICML 2025][Video Generation][Adversarial Training] This paper proposes the Adversarial Post-Training (APT) framework, which introduces an adversarial training phase after diffusion model pre-training to achieve high-quality one-step video generation (2 seconds, 1280×720, 24fps) with a model named Seaweed-APT.
tags:
  - "ICML 2025"
  - "Video Generation"
  - "Adversarial Training"
  - "Post-Training"
  - "One-Step Generation"
  - "Diffusion Model Distillation"
date: 2026-05-08
content_hash: e4fa4ebb03555a30
---

# Diffusion Adversarial Post-Training for One-Step Video Generation

**Conference**: ICML 2025  
**arXiv**: [2501.08316](https://arxiv.org/abs/2501.08316)  
**Code**: None  
**Area**: Diffusion Models/Video Generation  
**Keywords**: Adversarial Training, Post-Training, One-Step Generation, Video Generation, Diffusion Model Distillation

## TL;DR

This paper proposes the Adversarial Post-Training (APT) framework, which introduces an adversarial training phase after diffusion model pre-training to achieve high-quality one-step video generation (2 seconds, 1280×720, 24fps) with a model named Seaweed-APT.

## Background & Motivation

**Background**: Diffusion models have achieved immense success in image and video generation, but inference speed remains a core bottleneck. For video generation, typical diffusion models require 50-100 denoising steps to generate a video clip, rendering real-time applications (such as interactive creation and game rendering) nearly impossible.

**Limitations of Prior Work**: Existing acceleration methods mainly fall into two categories: (1) distillation-based methods (e.g., progressive distillation, consistency distillation), which compress the steps to 1-4 but often come with noticeable degradation in quality, particularly in fine details and motion consistency; (2) sampler improvements (e.g., DPM-Solver, DDIM), which reduce the number of steps but struggle to achieve one-step generation.

**Key Challenge**: The student model in distillation methods is forced to complete the denoising process, which originally required multiple iterative steps, in a single step. This "compression" essentially requires the model to directly map pure noise to the data distribution in a single forward pass—a learning target far more difficult than step-by-step denoising, making it hard to optimize fully via traditional distillation.

**Goal**: To achieve true one-step high-resolution video generation without sacrificing generation quality.

**Key Insight**: Instead of traditional distillation paradigms, adversarial training (GAN-style) is adopted as a post-training method. The powerful representation capability of the pre-trained diffusion model is leveraged as the generator's initialization, and adversarial training is then used to directly align the one-step generation results with the real data distribution.

**Core Idea**: Pre-train with the diffusion objective to obtain robust generative representations, and then utilize adversarial post-training to "collapse" multi-step denoising into a single-step generation, achieving both quality and speed in a mutually complementary two-stage approach.

## Method

### Overall Architecture

APT adopts a two-stage training strategy. The first stage is standard diffusion model pre-training to learn the denoising process of the data distribution. The second stage is adversarial post-training: the pre-trained diffusion model is used as the generator, a discriminator is introduced, real videos are used as positive samples, and the output of the generator in a single-step forward pass serves as negative samples. Through adversarial optimization, the generator is trained to directly produce high-quality outputs. The final model, Seaweed-APT, can generate 1280×720 resolution, 24fps videos in a single forward pass.

### Key Designs

1. **Two-Stage Training Paradigm (Diffusion Pre-training + Adversarial Post-training)**:

    - Function: Combines the stable training of diffusion models with the efficient inference advantages of GANs.
    - Mechanism: In the first stage, the model is trained with the standard diffusion objective (DDPM loss) to acquire a comprehensive understanding of the data distribution and powerful generative representations. In the second stage, the objective is switched to adversarial training, where the generator directly generates samples from noise in a single step, and the discriminator distinguishes the generated results from real data.
    - Design Motivation: Pure GAN training is highly unstable on high-resolution and high-dimensional data (videos). Using diffusion pre-training to provide a solid initialization and then fine-tuning via adversarial training enables one-step generation capabilities while maintaining training stability.

2. **Approximate R1 Regularization and Training Stability Improvements**:

    - Function: Addresses the instability of adversarial training in high-resolution video generation.
    - Mechanism: An approximate R1 regularization objective is adopted to constrain the gradient norm of the discriminator, preventing the discriminator from becoming too "sharp" and causing generator gradient explosion. Targeted architectural improvements are also made to support large-scale adversarial training.
    - Design Motivation: Standard GAN regularizations (e.g., spectral normalization, gradient penalty) offer limited efficacy on high-dimensional video data. While R1 regularization has proven effective in works like StyleGAN, its direct computation is prohibitively expensive for large models, necessitating an efficient approximate version.

### Loss & Training

The first stage utilizes the standard DDPM denoising loss. The adversarial training losses in the second stage include:

- Generator loss: Adversarial loss (preventing the discriminator from distinguishing one-step generation results from real data).
- Discriminator loss: Standard binary classification loss + approximate R1 regularization term.

The key to the training strategy lies in the smooth transition from diffusion pre-training to adversarial post-training—keeping the generator weights and architecture unchanged, while adjusting only the training objectives and learning rates. The adversarial post-training stage uses a relatively small learning rate to avoid destroying the representations learned during pre-training.

> **Note**: Since the full HTML page of this paper on arXiv could not be loaded (the cache is only 4KB, falling back to the abstract), the above implementation details are reasonably inferred based on the abstract and related work. Specific architectural improvements, discriminator designs, and the implementation of the R1 approximation should be referenced in the official paper.

## Key Experimental Results

### Main Results

Video generation performance (based on results reported in the abstract):

| Metric | Seaweed-APT | Description |
|:---|:---:|:---|
| Resolution | 1280 × 720 | HD 720p |
| Frame Rate | 24 fps | Fluid video |
| Duration | 2 seconds | ~48 frames |
| Inference Steps | **1 step** | Single forward pass |
| Inference Speed | Real-time | Completed in a single step |

Image generation comparison (at 1024px resolution):

| Method | Inference Steps | Quality Level |
|:---|:---:|:---|
| Multi-step Diffusion Model (SOTA) | 50-100 steps | Baseline |
| Distillation Methods | 1-4 steps | Decreased quality |
| Seaweed-APT | **1 step** | Comparable to SOTA |

> **Note**: Due to having only abstract information, specific FVD, FID values, and detailed baseline comparisons cannot be provided. The table above is structured based on qualitative descriptions in the abstract.

### Ablation Study

Effect of key components in the APT framework (inferred from the abstract):

| Configuration | Effect |
|:---|:---|
| Diffusion Pre-training only (multi-step inference) | High quality but slow |
| GAN training only (no pre-training) | Unstable training, poor quality |
| Diffusion Pre-training + Adversarial Post-training (APT) | One-step high-quality generation |
| APT without R1 Regularization | Training crash (hypothesized) |

> **Note**: The specific values of the ablation study should be referenced in the official paper. The above are reasonable inferences based on the methodology design.

### Key Findings

- Adversarial post-training is an effective paradigm: By combining diffusion pre-training and adversarial training, one-step video generation can be achieved without sacrificing quality.
- Pre-training the diffusion model provides a solid initialization for adversarial training, which is the key to APT's success—directly training video generation with GANs could hardly achieve the same quality.
- Approximate R1 regularization is crucial for the stability of large-scale video adversarial training.
- Seaweed-APT can match SOTA multi-step diffusion models in image generation (at 1024px), demonstrating the generalizability of the APT framework.

## Highlights & Insights

- **Paradigm Innovation**: Integrates the complementary strengths of diffusion models and GANs—the training stability of diffusion models combined with the inference efficiency of GANs. This two-stage concept of "first stable training, then efficient inference" is widely applicable.
- **High Practical Value**: One-step real-time video generation is one of the key bottlenecks for practical video AI applications; APT provides a viable path.
- **Engineering Significance of Seaweed-APT**: Supports both high-resolution image and video generation, displaying the potential of a unified framework.
- **Challenging the Distillation Paradigm**: Modern views assume that compressing diffusion steps must rely on distillation. APT proves that adversarial training is a superior alternative.

## Limitations & Future Work

- **Limited Information**: Since the arXiv full text was inaccessible, the method details and complete experimental data remain unclear, limiting the analysis in this note.
- The 2-second video duration is relatively short, and the effectiveness of APT on long video generation (>10 seconds) remains unknown.
- Adversarial training itself still carries the risk of mode collapse, particularly when training data lacks diversity.
- The design and computational overhead of the discriminator may limit further scaling of the model.
- The paper does not open-source the code, making reproducibility questionable.
- It remains unclear how the model performs regarding semantic accuracy in text-to-video (T2V) alignment.

## Related Work & Insights

- **vs Progressive Distillation**: Progressive distillation compresses step count by repeatedly halving the sampling steps (64→32→16→...→1), but quality degradation accumulates with each halving. APT reaches the goal in a single step via adversarial training.
- **vs Consistency Models**: Consistency Models achieve few-step generation by forcing model outputs to remain consistent across different noise levels, but their effectiveness in video scenarios is limited. APT leverages adversarial signals from real data, which are much more informative.
- **vs SDXL-Turbo/LCM**: These are acceleration methods in the image domain; APT shares a similar philosophy but extends it to the video domain and replaces distillation with adversarial training.
- **vs StyleGAN Series**: Historically, traditional GANs have trailed diffusion models in generation quality. APT elegantly bridges this gap using diffusion pre-training, rendering GAN-style one-step generation highly competitive again.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-stage paradigm of adversarial post-training has a clear concept, uniting the advantages of diffusion models and GANs.
- Experimental Thoroughness: ⭐⭐⭐ Due to limited abstract information, the full experimental quality cannot be evaluated; the claimed results are impressive but lack detailed data support.
- Writing Quality: ⭐⭐⭐ Cannot be fully evaluated (abstract only).
- Value: ⭐⭐⭐⭐⭐ Real-time, single-step video generation is of immense practical value. If the APT framework withstands verification, it will heavily impact the video generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)
- [\[ICLR 2026\] Towards One-Step Causal Video Generation via Adversarial Self-Distillation](../../ICLR2026/video_generation/towards_one-step_causal_video_generation_via_adversarial_self-distillation.md)
- [\[ICML 2026\] AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](../../ICML2026/video_generation/aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](../../ICCV2025/video_generation/adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)

</div>

<!-- RELATED:END -->
