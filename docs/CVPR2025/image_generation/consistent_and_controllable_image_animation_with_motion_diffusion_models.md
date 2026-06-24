---
title: >-
  [Paper Note] Consistent and Controllable Image Animation with Motion Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Image Animation] This paper proposes Cinemo, an image animation method based on diffusion models. By learning the distribution of motion residuals (rather than directly predicting frames), it significantly improves temporal consistency with the input image. Combined with SSIM motion intensity control and DCT noise initialization, it achieves finely controllable I2V generation, comprehensively outperforming existing methods on UCF-101 and MSR-VTT.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Animation"
  - "Motion Residual Diffusion"
  - "Motion Intensity Control"
  - "DCT Initialization"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: 0fd2e308d7df8a92
---

# Consistent and Controllable Image Animation with Motion Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2407.15642](https://arxiv.org/abs/2407.15642)  
**Code**: [https://maxin-cn.github.io/cinemo_project](https://maxin-cn.github.io/cinemo_project)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Image Animation, Motion Residual Diffusion, Motion Intensity Control, DCT Initialization, Temporal Consistency

## TL;DR
This paper proposes Cinemo, an image animation method based on diffusion models. By learning the distribution of motion residuals (rather than directly predicting frames), it significantly improves temporal consistency with the input image. Combined with SSIM motion intensity control and DCT noise initialization, it achieves finely controllable I2V generation, comprehensively outperforming existing methods on UCF-101 and MSR-VTT.

## Background & Motivation

### Background

**Background**: Image-to-Video (I2V) generation has made rapid progress through diffusion models. Existing methods directly predict the latent representations of video frames but struggle to guarantee high consistency between the generated frames and the input image.

**Limitations of Prior Work**: (1) Directly predicting frames leads to color and structural drift relative to the input image; (2) Motion intensity is difficult to control finely—too weak results in static frames, while too strong leads to structural collapse; (3) FFT-based noise initialization, though providing global consistency, introduces high-frequency artifacts and color inconsistencies.

**Key Challenge**: The need to generate natural motion while maintaining consistency with the input image, while also allowing users to finely control the motion magnitude.

**Goal**: To achieve high consistency, fine-grained motion control, and artifact-free initialization in I2V generation.

**Key Insight**: Learning motion residuals (frame differences) instead of full frames—the residual space has a small magnitude and simple structure, making it easier for diffusion models to learn, while naturally preserving consistency with the input frame.

**Core Idea**: Learning the motion residual distribution in the latent space, utilizing SSIM motion buckets to control intensity, and replacing FFT with DCT initialization to eliminate high-frequency artifacts.

## Method

### Overall Architecture


### Key Designs

1. **Motion Residual Diffusion**: Evaluates the distribution of differences between subsequent frames and the first frame in the latent space, rather than directly predicting frames. During generation, the residual is added back to the first frame's latent to obtain the video frame. The residual magnitude is much smaller than that of a full frame, reducing learning difficulty and naturally preserving consistency with the input.

2. **SSIM Motion Intensity Control**: The training videos are divided into 20 motion buckets (0-19) based on SSIM values. During inference, users select a bucket number to control the motion magnitude finely. SSIM is more robust than optical flow and directly measures the amount of visual change.

3. **DCTInit Noise Refinement**: Refines initial noise using low-frequency DCT (instead of FFT) coefficients. Replacing the real and imaginary parts of FFT separately introduces high-frequency anomalies and color drift, whereas DCT, having only real coefficients, avoids this problem.

### Loss & Training
Fine-tuned based on the LaVie video diffusion model. Standard diffusion denoising loss. $320 \times 512$ resolution.

## Key Experimental Results

### Main Results

| Method | UCF-101 FVD↓ | IS↑ | FID↓ | MSR-VTT FVD↓ | CLIPSIM↑ |
|------|-------------|-----|------|-------------|----------|
| ConsistI2V | 177.66 | 56.22% | 15.74% | 104.58 | 0.2674 |
| SEINE | 306.49 | 54.02% | 26.00% | 152.63 | 0.2774 |
| **Cinemo** | **168.16** | **58.71%** | **13.17%** | **93.51%** | **0.2858** |

Achieves the best performance across all five metrics, also outperforming commercial tools (Gen-2, Pika Labs).

### Key Findings
- Motion residual learning substantially improves input consistency (FID 13.17 vs ConsistI2V 15.74).
- DCTInit is significantly better in visual quality than FFTInit, with no color drift.
- Motion buckets provide an intuitive interface for intensity control.

## Limitations & Future Work
- Fixed resolution of $320 \times 512$ (constrained by LaVie).
- UNet architecture, while a Transformer architecture might offer better scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ The motion residual approach is simple yet effective, and the discovery of DCTInit outperforming FFTInit is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark + commercial tool comparison + ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Provides a direct solution to the I2V consistency issue.


## Highlights & Insights
- The method design is simple and effective, with a clear core mechanism.
- Comprehensive experimental validation and thorough ablation analysis.
- Provides new solution pathways for key challenges in the field.


## Related Work & Insights
- **vs Representative methods in the same field**: Ours makes unique contributions in methodological design and is complementary to existing methods.
- **vs Traditional methods**: Compared to traditional approaches, ours achieves significant improvements in key metrics.
- **Insights**: Our technical pipeline provides valuable reference for subsequent related research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[CVPR 2025\] MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [\[ICLR 2026\] WithAnyone: Toward Controllable and ID Consistent Image Generation](../../ICLR2026/image_generation/withanyone_toward_controllable_and_id_consistent_image_generation.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)

</div>

<!-- RELATED:END -->
