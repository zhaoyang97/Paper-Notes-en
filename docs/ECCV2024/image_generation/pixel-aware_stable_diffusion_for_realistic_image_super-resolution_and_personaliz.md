---
title: >-
  [Paper Note] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization
description: >-
  [ECCV 2024][Image Generation] This paper proposes PASD (Pixel-Aware Stable Diffusion), which enables the diffusion model to perceive local image structures at the pixel level through a Pixel-Aware Cross Attention (PACA) module. Combined with a degradation removal module and an adjustable noise schedule, it achieves a unified framework for realistic image super-resolution and personalized stylization, where the style can be switched simply by replacing the base model.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 4546f38463c17b2d
---

# Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization

**Conference**: ECCV 2024  
**arXiv**: [2308.14469](https://arxiv.org/abs/2308.14469)  
**Area**: Image Restoration

## TL;DR

This paper proposes PASD (Pixel-Aware Stable Diffusion), which enables the diffusion model to perceive local image structures at the pixel level through a Pixel-Aware Cross Attention (PACA) module. Combined with a degradation removal module and an adjustable noise schedule, it achieves a unified framework for realistic image super-resolution and personalized stylization, where the style can be switched simply by replacing the base model.

## Background & Motivation

Utilizing pretrained text-to-image diffusion models (e.g., Stable Diffusion) for realistic image super-resolution (Real-ISR) is a promising direction, but faces core challenges:

**Difficulty in pixel-level structure preservation**: Methods like ControlNet simply add U-Net and conditional features through "zero convolutions," which fails to transmit precise pixel-level information, resulting in structural inconsistencies between the output and input.

**Limitations of skip connections**: Methods like StableSR pass pixel details through skip connections between the VAE encoder and decoder, but require additional training in image space, which limits their application in latent-space tasks (e.g., stylization).

**Train-test inconsistency**: The noise schedule of SD leaves residual signals at the terminal training timestep, whereas testing samples from pure Gaussian noise, creating an inconsistency.

The goal of PASD is to design a flexible model that simultaneously addresses Real-ISR and personalized stylization without requiring skip connections.

## Method

### Overall Architecture

PASD introduces four modules on top of the pretrained SD:

1. **Degradation Removal Module**: A pyramid network to extract multi-scale degradation-insensitive features.
2. **ControlNet**: Extracts low-level control features.
3. **PACA (Pixel-Aware Cross Attention)**: Achieves pixel-level guidance in the latent space.
4. **ANS (Adjustable Noise Schedule)**: An adjustable noise schedule to flexibly balance perception and fidelity.
5. **High-level Nets**: Utilizes ResNet/YOLO/BLIP to extract classification/detection/caption semantic information.

### Key Designs

**Pixel-Aware Cross Attention (PACA)**:

Instead of the zero-convolution connections in ControlNet, the U-Net features $\mathbf{x} \in \mathbb{R}^{h \times w \times c}$ and ControlNet features $\mathbf{y}$ are flattened and a cross-attention operation is performed using $\mathbf{y}$ as the context input:

$$PACA(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = Softmax(\frac{\mathbf{QK}^T}{\sqrt{d}}) \cdot \mathbf{V}$$

where $\mathbf{Q} = to\_q(\mathbf{x}')$, $\mathbf{K} = to\_k(\mathbf{y}')$, and $\mathbf{V} = to\_v(\mathbf{y}')$. Since the length of $\mathbf{y}'$ is $h*w$ (equal to the total number of pixels in the latent features) and it is not transformed by the VAE encoder, it preserves the original image structure, thereby achieving pixel-level perception.

**Degradation Removal Module**:

The pyramid network extracts feature maps at three scales (1/2, 1/4, 1/8). Each scale reconstructs an HQ image via a "toRGB" convolutional layer and is supervised with an L1 loss: $\mathcal{L}_{DR} = \sum_s \|\mathbf{I}_{hq}^s - \mathbf{I}_{sr}^s\|_1$. This allows the subsequent diffusion module to focus on restoring realistic details without having to process degradations.

**Adjustable Noise Schedule (ANS)**:

An additional Gaussian noise $\mathbf{z}'$ is introduced to control the residual signal strength:

$$\mathbf{z}_N = \sqrt{\bar{\alpha}_a \bar{\alpha}_N} \mathbf{z}_{LR} + \sqrt{1 - \bar{\alpha}_a \bar{\alpha}_N} \mathbf{z}''$$

$\bar{\alpha}_a \in [0, 1]$ controls the perception-fidelity trade-off: a larger $\bar{\alpha}_a$ yields higher fidelity but fewer details. Experiments choose $\bar{\alpha}_{900} = 0.1189$ as the optimal trade-off point.

**Personalized Stylization**:

Since PASD freezes the parameters of the base SD model and only trains the newly added modules, the base model can be directly replaced with personalized models (e.g., ToonYou for cartoonization, majicMIX realistic for old photo restoration) during testing without additional training.

### Loss & Training

Total training loss: $\mathcal{L} = \mathcal{L}_{DF-\epsilon} + \gamma \mathcal{L}_{DR}$

Diffusion loss: $\mathcal{L}_{DF-\epsilon} = \mathbb{E}_{z_0, t, c, I_{lq}, \epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon_\theta(z_t, t, c, I_{lq})\|_2^2]$

During training, 50% of the text prompts are randomly replaced with empty text to encourage the model to perceive semantics from the LQ image itself. The model is trained for 500K iterations with a batch size of 4 and a learning rate of $5 \times 10^{-5}$.

## Key Experimental Results

### Main Results

**Quantitative comparison of Real-ISR (DIV2K / RealSR / DRealSR)**:

| Method | FID↓ (RealSR) | LPIPS↓ | MUSIQ↑ | QAlign↑ |
|------|---------------|--------|--------|---------|
| RealESRGAN | 67.02 | 0.2729 | 59.69 | 3.92 |
| StableSR | 109.11 | 0.2565 | 60.71 | 3.87 |
| DiffBIR | 55.17 | 0.3633 | 65.52 | 4.10 |
| SeeSR | 58.32 | 0.2796 | 64.27 | 3.89 |
| **PASD** | **47.34** | **0.2806** | **65.60** | **4.13** |

**Quantitative comparison of cartoonization (FFHQ + Flicker2K)**:

| Method | FID↓ (FFHQ) | MUSIQ↑ | QAlign↑ |
|------|-------------|--------|---------|
| CartoonGAN | 53.75 | 71.98 | 3.61 |
| InstructPix2Pix | 39.33 | 72.95 | 3.90 |
| ControlNet | 37.96 | 74.90 | 4.01 |
| **PASD** | **37.67** | **75.02** | **4.00** |

**Quantitative comparison of old photo restoration**:

| Method | FID↓ | MUSIQ↑ | QAlign↑ |
|------|------|--------|---------|
| RealESRGAN | 265.54 | 59.47 | 3.67 |
| Wan et al. | 268.35 | 32.15 | 3.01 |
| DiffBIR | 262.18 | 60.39 | 3.92 |
| **PASD** | **240.26** | **64.40** | **3.98** |

### Ablation Study

**Ablation of each PASD component (RealSR dataset)**:

| Component Configuration | PSNR↑ | FID↓ | LPIPS↓ | Inference Time (s)↓ |
|---------|-------|------|--------|------------|
| W/o PACA (zero convolution only) | 26.11 | 56.79 | 0.3822 | 14.32 |
| + Degradation Removal | 27.87 | 53.90 | 0.3080 | 8.04 |
| + Degradation Removal + High-level Info | 27.09 | 52.34 | 0.2851 | 8.74 |
| + Degradation Removal + Negative Prompt | 27.38 | 50.25 | 0.2809 | 13.32 |
| **Full PASD** | 25.93 | **47.34** | **0.2806** | 14.59 |

### Key Findings

- PACA is key to structural preservation: without PACA, the output is inconsistent with the input in terms of color and structure.
- The degradation removal module makes the output cleaner, with a 1.76dB increase in PSNR.
- Negative prompts ("noisy", "blurry", "low resolution") contribute significantly to visual quality, reducing FID by 3.65.
- PASD achieves the highest QAlign scores across all three Real-ISR datasets and receives the most rank-1 votes in the user study.
- Stylization tasks require no additional training and can be achieved by simply replacing the base model.

## Highlights & Insights

1. **PACA decouples pixel control from latent space operations**: It achieves pixel-level structure preservation without requiring skip connections, making the model simultaneously applicable to Real-ISR and latent-space stylization tasks.
2. **ANS provides flexible adjustments**: A single hyperparameter $\bar{\alpha}_a$ can control the perception-fidelity trade-off during inference without requiring retraining.
3. **One model, multiple purposes**: The same PASD model can perform Real-ISR, cartoonization, and old photo restoration by simply replacing the base model, demonstrating extreme flexibility.
4. **Effective high-level semantics**: Leveraging ResNet classification + YOLO detection + BLIP captioning provides semantic guidance, which significantly improves results compared to using only empty text.

## Limitations & Future Work

- There is still a trade-off between fidelity and perceptual quality; unfaithful details may be generated when degradation is severe or semantic information is inaccurate.
- Inference time nearly doubles (approx. 14.6s vs 8.0s) when using classifier-free guidance.
- Extracting high-level information relies on multiple pretrained models (ResNet, YOLO, BLIP), which increases system complexity.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty: ★★★★ — The PACA module achieves pixel-level perception without introducing skip connections, demonstrating an ingenious design.
- Technical: ★★★★★ — A unified framework covering super-resolution, stylization, and restoration, with each module supported by theoretical backing.
- Experimental Thoroughness: ★★★★★ — Comprehensive evaluation across three tasks with complete ablations and user studies that enhance persuasiveness.
- Practicality: ★★★★ — The code is open-source and can directly utilize personalized models from the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] MagicEraser: Erasing Any Objects via Semantics-Aware Control](magiceraser_erasing_any_objects_via_semantics-aware_control.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)

</div>

<!-- RELATED:END -->
