---
title: >-
  [Paper Note] You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation
description: >-
  [ECCV 2024][Image Generation][One-step super-resolution] This work proposes YONOS-SR, which trains a Stable Diffusion-based super-resolution model via a Scale Distillation strategy. It achieves state-of-the-art results with only a single DDIM step, accelerating inference by 200 times compared to conventional methods.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "One-step super-resolution"
  - "diffusion model distillation"
  - "scale distillation"
  - "Stable Diffusion"
  - "fast inference"
date: 2026-05-08
content_hash: e206da04380964d7
---

# You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation

**Conference**: ECCV 2024  
**arXiv**: [2401.17258](https://arxiv.org/abs/2401.17258)  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: One-step super-resolution, diffusion model distillation, scale distillation, Stable Diffusion, fast inference

## TL;DR

This work proposes YONOS-SR, which trains a Stable Diffusion-based super-resolution model via a Scale Distillation strategy. It achieves state-of-the-art results with only a single DDIM step, accelerating inference by 200 times compared to conventional methods.

## Background & Motivation

Diffusion-based image super-resolution (SR) methods have achieved outstanding performance in perceptual quality. However, their critical bottleneck lies in **inference speed**, as they typically require tens to hundreds of denoising iterations, which severely limits real-world applications. For instance, StableSR requires 200 DDIM sampling steps to achieve optimal results, taking tens of seconds or even longer to process a single image.

A natural question arises: **Can the SR quality of diffusion models be compressed into a minimal number of inference steps?** Step distillation methods attempt to use a student model with fewer steps to mimic a teacher model with many steps, but directly distilling multiple steps into a single step often yields poor results due to the excessive knowledge gap.

This paper proposes a more elegant solution—**Scale Distillation**. The core observation is that low-magnification SR (e.g., $2\times$) is significantly simpler than high-magnification SR (e.g., $4\times$ or $8\times$). Therefore, a strong teacher model can be trained on low-magnification SR first (requiring only one step), and the teacher's output can then be utilized as the training target for the high-magnification student model. This progressive "easy-to-difficult" distillation strategy makes one-step inference feasible.

## Method

### Overall Architecture

The training pipeline of YONOS-SR is as follows: (1) training a teacher model on a low-magnification (e.g., $2\times$) SR task until its one-step inference yields satisfactory quality; (2) using the output of the teacher model as the target to train a student model for high-magnification (e.g., $4\times$) SR; (3) repeating this process until the target magnification factor (e.g., $8\times$) is reached. The final student model requires only a single DDIM step during inference to produce high-quality SR results.

### Key Designs

1. **Scale Distillation**:
    - **Function**: Achieving knowledge transfer from multi-step to one-step diffusion SR.
    - **Mechanism**: Traditional distillation operates along the step dimension (few steps mimicking many steps), whereas YONOS-SR operates along the scale dimension. Since low-magnification SR is simpler, the diffusion model performs well as a teacher even with few-step inference. The teacher's prediction at low noise levels is better suited as a student training target than directly using ground truth (GT)—this is because the teacher's prediction is "adapted to the current noise level", whereas GT remains identical across all noise levels.
    - **Design Motivation**: (a) The teacher provides a target matched to the current noise level instead of a fixed GT, which is more suitable for diffusion model training; (b) the low-magnification teacher task is simple, enabling accurate predictions with one-step inference and thus providing high-quality training signals.

2. **Iterative Scale Progression**:
    - **Function**: Progressively scaling up SR capabilities to high magnifications.
    - **Mechanism**: Training is conducted step-by-step in the order of $2\times \rightarrow 4\times \rightarrow 8\times$. The $2\times$ model acts as the teacher for the $4\times$ model, and the $4\times$ model acts as the teacher for the $8\times$ model. Distillation at each level controls the knowledge gap, preventing an excessive discrepancy when jumping directly from $2\times$ to $8\times$.
    - **Design Motivation**: Directly training a high-magnification (e.g., $8\times$) one-step model performs poorly because the task is too difficult; progressive training incrementally decomposes the difficulty.

3. **Decoder Fine-tuning**:
    - **Function**: Further enhancing output quality while keeping the U-Net frozen.
    - **Mechanism**: Once the U-Net is successfully trained via scale distillation (producing solid latent representations in a single step), its parameters are frozen, and only the VAE decoder is fine-tuned. The objective of decoder fine-tuning is to reconstruct higher-quality pixel-space images from the one-step latent outputs of the U-Net.
    - **Design Motivation**: Although the latent outputs of the one-step U-Net are rich in information, the standard VAE decoder is not optimized for such single-step outputs. Fine-tuning the decoder bridges this gap.

### Loss & Training

- **Scale distillation loss**: L2 and perceptual losses between the student's one-step prediction and the teacher's one-step prediction.
- **Decoder fine-tuning loss**: L1, perceptual, and GAN losses between the fine-tuned decoder output and the ground truth high-resolution image.
- **Training strategy**: Distilling the U-Net first, followed by fine-tuning the decoder, executed sequentially in two stages.
- **Noise levels**: Exploring the quality of teacher targets at different noise levels during training to select the optimal configuration.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (1 step) | StableSR (200 steps) | Gain |
|--------|------|------|----------|------|
| DIV2K | CLIPIQA ↑ | Superior | Baseline | 1 step surpasses 200 steps |
| RealSR | MUSIQ ↑ | SOTA | StableSR | Outperform |
| ImageNet | FID ↓ | Competitive | 200-step methods | Comparable |
| Inference Speed | Steps | 1 step | 200 steps | 200x speedup |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Direct training of $8\times$ (one-step) | Poor performance | The task is too difficult |
| Direct GT supervision (one-step) | Moderate | GT does not adapt to noise levels |
| Scale distillation (one-step) | Good | Teacher target is more appropriate |
| Scale distillation + Decoder fine-tuning | Optimal | Complete method |
| Step distillation vs. Scale distillation | Scale is superior | Knowledge transfer along the scale dimension is more natural |

### Key Findings

- Scale distillation is better suited for one-step SR than step distillation: low-magnification teachers provide higher-quality targets.
- The crucial reason teacher targets outperform fixed GT: the former adapts to the current noise level.
- The combination of U-Net distillation and decoder fine-tuning maximizes perceptual quality.
- YONOS-SR with one-step inference surpasses StableSR with 200 steps in perceptual quality metrics.

## Highlights & Insights

- "Scale distillation" is a highly elegant and intuitive concept—distilling knowledge from simpler tasks to more difficult ones.
- Achieving superior performance in 1 step compared to 200 steps is an impressive result; the 200x acceleration makes real-time SR viable.
- The analysis of "why the teacher target outperforms GT" provides profound insights into diffusion model training.
- The method can be generalized to accelerate other conditional generation tasks.

## Limitations & Future Work

- Scale distillation requires multi-stage training ($2\times \rightarrow 4\times \rightarrow 8\times$), which still incurs high overall training costs.
- It may underperform compared to traditional non-diffusion SR methods on fidelity metrics such as PSNR.
- Currently validated primarily on Stable Diffusion; its applicability to other diffusion architectures remains to be confirmed.
- Future work could explore whether intermediate stages can be bypassed (e.g., direct $2\times \rightarrow 8\times$).
- Integration with recent flow matching architectures might further enhance overall performance.

## Related Work & Insights

- **StableSR**: A Stable Diffusion-based SR method that delivers excellent quality but runs slowly (200 steps).
- **Progressive Distillation**: Distillation in the step dimension, which is complementary to the scale distillation proposed in this work.
- **Consistency Models**: Another approach to achieving few-step inference.
- **Insight**: Distillation can be performed not only along the step dimension but also along the scale/difficulty dimension, which may prove effective for other tasks as well.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The scale distillation concept is novel, providing deep insights with the "easy-to-difficult" distillation strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid, fair comparisons with multi-step methods and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The clear structure of the paper nicely integrates theoretical analysis with experiments.
- Value: ⭐⭐⭐⭐⭐ Achieving 200x speedup while maintaining quality holds immense value for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)

</div>

<!-- RELATED:END -->
