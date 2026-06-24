---
title: >-
  [Paper Note] RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation
description: >-
  [NeurIPS 2025 (Spotlight)][Image Generation][Latent diffusion models] This paper proposes RepLDM, a reprogramming framework that enables pretrained latent diffusion models to generate high-quality, high-resolution images without retraining, via two stages: an attention guidance stage and a progressive upsampling stage, while substantially improving efficiency.
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Image Generation"
  - "Latent diffusion models"
  - "high-resolution generation"
  - "model reprogramming"
  - "attention guidance"
  - "progressive upsampling"
date: 2026-05-08
content_hash: a5a26b455354839f
---

# RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation

**Conference**: NeurIPS 2025 (Spotlight)

**arXiv**: [2410.06055](https://arxiv.org/abs/2410.06055)

**Code**: [GitHub](https://github.com/kmittle/RepLDM)

**Area**: Image Generation

**Keywords**: Latent diffusion models, high-resolution generation, model reprogramming, attention guidance, progressive upsampling

## TL;DR

This paper proposes RepLDM, a reprogramming framework that enables pretrained latent diffusion models to generate high-quality, high-resolution images without retraining, via two stages: an attention guidance stage and a progressive upsampling stage, while substantially improving efficiency.

## Background & Motivation

Latent diffusion models (e.g., Stable Diffusion) are designed for high-resolution image generation, yet frequently exhibit severe structural distortions when generating images beyond their training resolution.

Limitations of existing approaches:

**High retraining cost**: Training models from scratch at high resolutions requires substantial computational resources.

**Long inference time**: Existing reprogramming methods save on training but require a large number of denoising steps at inference.

**Poor quality**: Direct upsampling in latent space leads to severe artifacts.

**Structural inconsistency**: Self-attention behavior at high resolutions deviates from that at the training resolution.

## Method

### Overall Architecture

RepLDM consists of two stages: (1) an attention guidance stage that generates a high-quality initialization at the training resolution, and (2) a progressive upsampling stage that incrementally increases resolution in pixel space.

### Key Designs

**1. Attention Guidance Stage**

- Denoising is performed at the training resolution (e.g., 512×512).
- A **training-free self-attention guidance mechanism** is introduced to enhance structural consistency.
- Core idea: the Key/Value matrices of self-attention are modified to guide the generation of more structurally coherent latent representations.
- The resulting latent representations are better suited for subsequent upsampling compared to standard methods.

**2. Progressive Upsampling Stage**

- Key insight: upsampling is performed in **pixel space** rather than latent space.
- Upsampling in latent space causes encoder–decoder mismatch, producing severe artifacts.
- Pixel-space upsampling preserves VAE decode–encode consistency.
- Progressive schedule: 512 → 768 → 1024 → ..., with only a small number of denoising steps required at each stage.

**3. Efficient Denoising**

- The high-quality initialization provided by Stage 1 allows Stage 2 to require very few denoising steps (e.g., 5–10 steps).
- Total inference time is substantially lower than existing methods.

### Loss & Training

- **Training-free**: RepLDM involves no parameter training or fine-tuning.
- Only the attention computation and upsampling strategy within the inference pipeline are modified.
- The standard denoising objective of the pretrained model is used.

## Key Experimental Results

### Main Results

Image generation quality at 1024×1024 resolution (based on SD1.5, 512→1024):

| Method | FID ↓ | CLIP Score ↑ | Inference Time | Structural Consistency |
|--------|-------|-------------|----------------|----------------------|
| Direct Generation (SD) | 85.2 | 0.265 | 8.5s | Poor |
| MultiDiffusion | 42.3 | 0.285 | 45.2s | Moderate |
| ScaleCrafter | 38.5 | 0.292 | 52.8s | Moderate |
| DemoFusion | 32.1 | 0.301 | 68.5s | Good |
| RepLDM (Ours) | **25.8** | **0.315** | **18.2s** | **Excellent** |

At 2048×2048 resolution:

| Method | FID ↓ | CLIP Score ↑ | Inference Time |
|--------|-------|-------------|----------------|
| ScaleCrafter | 52.3 | 0.275 | 185s |
| DemoFusion | 45.8 | 0.288 | 235s |
| RepLDM (Ours) | **35.2** | **0.302** | **52s** |

### Ablation Study

Contribution of each component (512→1024):

| Configuration | FID | CLIP Score | Inference Time |
|---------------|-----|-----------|----------------|
| Standard upsampling (latent space) | 65.3 | 0.272 | 22s |
| + Attention guidance | 42.5 | 0.295 | 25s |
| + Pixel-space upsampling | 35.2 | 0.305 | 20s |
| + Progressive schedule (RepLDM) | **25.8** | **0.315** | **18.2s** |

### Key Findings

1. RepLDM simultaneously outperforms the state of the art in both quality (FID 25.8 vs. 32.1) and speed (18.2s vs. 68.5s).
2. Pixel-space upsampling is the critical factor for quality improvement, eliminating latent-space artifacts.
3. Attention guidance provides a better initialization for upsampling, reducing the number of subsequent denoising steps required.
4. The speed advantage becomes more pronounced at 4× upsampling (512→2048): 52s vs. 235s.

## Highlights & Insights

- **Spotlight paper**: Recognized as a Spotlight at NeurIPS 2025, attesting to its impact and quality.
- **Triple advantage**: High quality, high efficiency, and high resolution are achieved simultaneously rather than traded off against one another.
- **Key insight**: The choice to perform upsampling in pixel space rather than latent space circumvents a fundamental source of degradation.

## Limitations & Future Work

1. The method currently targets UNet-based architectures (SD1.5/SDXL); adaptation to DiT-based architectures remains to be validated.
2. Text–image alignment for long prompts in text-to-image generation may degrade at high resolutions.
3. The selection of intermediate resolutions in the progressive schedule lacks an adaptive mechanism.
4. Evaluation relies primarily on FID and CLIP Score; human perceptual evaluation is absent.

## Related Work & Insights

- **DemoFusion** (Du et al.): A denoising fusion method for progressive upsampling.
- **ScaleCrafter** (He et al.): Adapts convolutional operations to accommodate high resolutions.
- **MultiDiffusion**: Patch-based diffusion for panoramic image generation.

## Rating

- ⭐ Novelty: 8/10 — The pixel-space upsampling insight is simple yet critical.
- ⭐ Value: 9/10 — Fast, high-quality, and openly available.
- ⭐ Writing Quality: 8/10 — Overall quality consistent with Spotlight recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](../../CVPR2025/image_generation/diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICLR 2026\] Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis](../../ICLR2026/image_generation/latent_wavelet_diffusion_for_ultra_high-resolution_image_synthesis.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](../../ICCV2025/image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](../../CVPR2025/image_generation/stableanimator_high-quality_identity-preserving_human_image_animation.md)

</div>

<!-- RELATED:END -->
