---
title: >-
  [Paper Note] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)
description: >-
  [Image Generation] Proposes DPIR, an image restoration model based on SD3 (Diffusion Transformer). By incorporating a lightweight low-quality image condition branch and a global-local visual dual prompting branch, it introduces degradation image information from multiple perspectives, systematically applying DiT to image restoration for the first time and achieving SOTA performance.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 44d2e4585908e640
---

# Dual Prompting Image Restoration with Diffusion Transformers (DPIR)

## TL;DR

Proposes DPIR, an image restoration model based on SD3 (Diffusion Transformer). By incorporating a lightweight low-quality image condition branch and a global-local visual dual prompting branch, it introduces degradation image information from multiple perspectives, systematically applying DiT to image restoration for the first time and achieving SOTA performance.

## Background & Motivation

In the field of image restoration (IR), diffusion-model-based methods (e.g., StableSR, SUPIR) have demonstrated powerful generative capabilities, but they are all based on the U-Net architecture LDM. The new generation of Diffusion Transformers (DiTs) such as SD3 exhibits superior generation quality due to better scalability and long-range dependency modeling capabilities.

**Key Challenge**: How to effectively inject low-quality (LQ) image information into DiT?
- **ControlNet Scheme**: Replicates encoding layers, but designed for U-Net, which is unsuitable for DiTs composed of ViT blocks.
- **Lightweight Adapters (T2I-Adapter/StableSR)**: Perform moderately.
- **DiT Lacks Skip Connections**: Unlike U-Net, which maintains low-quality image information across layers through skip connections, LQ information in DiT gradually gets lost in deep layers.
- **Insufficient Text Descriptions**: Pure text prompts cannot completely describe the visual characteristics (texture, structure) of an image.

The core insight of this paper: In image restoration, **visual prompts should replace/supplement text prompts** to guide the DiT, as the detailed information of an image is far richer than textual descriptions.

## Method

### Overall Architecture

DPIR is based on SD3 and consists of two control branches: (1) a lightweight LQ image condition branch that injects the LQ prior into the first layer of DiT via a few convolutional layers and adaptive feature alignment; (2) a dual prompting control branch that extracts global and local visual features to replace the CLIP text embeddings, which are then concatenated with the T5 text prompt as the cross-attention condition input to each DiT block.

### Key Designs

#### 1. Degradation-Robust VAE Encoder

- **Function**: Encodes degraded images into SD3's 16-channel latent space, making the latent representation robust to degradation.
- **Mechanism**: Fine-tunes SD3's VAE encoder $\mathcal{E}_{dr}$ and supervises it with L1 + LPIPS + GAN losses. The GAN loss prevents the VAE from generating overly smoothed results.
- **Design Motivation**: The original SD3 VAE is trained only on high-quality images, so directly encoding degraded images yields inaccurate latents. The 16-channel VAE outperforms SDXL's 4-channel VAE, providing richer initial conditions.

#### 2. Lightweight LQ Image Condition Branch

- **Function**: Efficiently injects LQ image priors into the DiT backbone.
- **Mechanism**: Inspired by ControlNeXt, it extracts LQ features $\mathcal{F}_c(z_{\text{LQ}})$ using a few convolutional layers, which are adjusted via adaptive feature alignment $\eta(\cdot; \mu, \sigma)$ and then added to the output of the first DiT layer. The alignment function normalizes using the mean and variance of the first DiT layer's output.
- **Design Motivation**: Avoids a heavy replication scheme like ControlNet; injecting conditions only into the first layer is sufficient. Adaptive alignment resolves the distribution mismatch between conditional features and backbone features. The trainable parameters satisfy $\phi_c \ll \theta_d$, maintaining high efficiency.

#### 3. Global-Local Dual Prompting Control Branch

- **Function**: Provides rich visual conditional control to replace the CLIP text embeddings in DiT.
- **Mechanism**: Feeds the LQ image into two CLIP image encoders to extract local visual tokens $c_{\text{local}}^{\text{vis}}$ and the pooled embedding $c_{\text{pool}}$ to replace the original CLIP text embeddings. Simultaneously, it crops a global patch from the surrounding area to extract global visual tokens $c_{\text{global}}^{\text{vis}}$ to capture contextual semantics. The global and local tokens are concatenated with T5 text prompts to form dual prompts $c_{\text{dual}}$, which are injected into each DiT block via cross-attention.
- **Design Motivation**: Since DiT lacks U-Net's skip connections, relying solely on a lightweight conditional branch is insufficient to preserve LQ details. Replacing text prompts with visual prompts transmits structural and textual details that words cannot convey. The global patch compensates for the local patch's lack of global semantics, bringing the visual tokens closer to the semantic level of the original text tokens.

### Loss & Training

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, p_t(z|\epsilon), p(\epsilon)} \left[\|v_\theta(z_t, t) - (\epsilon - x_0)\|^2\right]$$

VAE fine-tuning loss: $\|\mathcal{D}(\mathcal{E}_{dr}(x_{\text{LQ}})) - x_{\text{HQ}}\|_1 + \alpha \mathcal{L}_{lpips} + \beta \mathcal{L}_{GAN}$

## Key Experimental Results

### Main Results (4× Super-Resolution)

**DIV2K-Val Dataset**:

| Method | PSNR↑ | LPIPS↓ | DISTS↓ | CLIPIQA↑ | MUSIQ↑ |
|------|-------|--------|--------|----------|--------|
| Real-ESRGAN | 22.62 | 0.3982 | 0.2240 | 0.5661 | 63.90 |
| StableSR | 22.87 | 0.3925 | 0.2085 | 0.4974 | 57.28 |
| SinSR | 22.10 | 0.4416 | 0.2160 | 0.6919 | 65.13 |
| SUPIR | 21.23 | 0.4152 | 0.1873 | 0.5239 | 66.49 |
| **DPIR (Ours)** | 21.61 | **0.3622** | **0.1677** | **0.7416** | **71.94** |

**RealSR Dataset**:

| Method | LPIPS↓ | DISTS↓ | CLIPIQA↑ | MUSIQ↑ |
|------|--------|--------|----------|--------|
| Real-ESRGAN | 0.2827 | 0.1936 | 0.5157 | 64.46 |
| SUPIR | 0.3996 | 0.2268 | 0.5223 | 58.68 |
| **DPIR (Ours)** | **0.2641** | **0.1642** | **0.6625** | **69.28** |

### Ablation Study

- Text Prompt vs. Dual Prompt: The dual prompting strategy consistently outperforms pure text prompts across all perceptual metrics (see Figure 1 for qualitative comparison).
- Joint Global-Local Training Outperforms Local-Only: Global patches provide contextual semantics, improving the global consistency of restoration.

### Key Findings

- **Significant Lead in Perceptual Quality Metrics**: DPIR comprehensively outperforms all baselines on perceptual metrics such as CLIPIQA, MUSIQ, and DISTS.
- **Non-Optimal PSNR**: The PSNR of DPIR is slightly lower than that of StableSR/Real-ESRGAN, illustrating the trade-off between generative quality and pixel fidelity.
- **DiT > U-Net**: SD3's DiT backbone with appropriate conditional injection yields restoration quality surpassing the SDXL-based SUPIR.
- **Training Data Scale**: Trained on over 20 million high-quality images, fully leveraging the scalability of DiT.

## Highlights & Insights

1. **Visual Prompts Replacing Text Prompts**: Core innovation—in restoration tasks, replacing CLIP text features with CLIP image features as conditional control for DiT communicates visual information more precisely than textual descriptions.
2. **Global-Local Hierarchical Design**: Solves the issue where local patches lack global semantics in high-resolution restoration, successfully aligning with the functional role of text embeddings in the pretrained DiT.
3. **DiT Adaptation Methodology**: Provides a systematic approach on how to adapt DiT to tasks other than generation—using a lightweight conditional branch + prompt replacement.

## Limitations & Future Work

1. **Lower PSNR Metrics**: An inherent drawback of generative restoration; pursuit of exceptional perceptual quality can sacrifice pixel-level fidelity.
2. **Inference Speed**: The inference overhead of DiT + iterative denoising is higher than single-step methods like Real-ESRGAN.
3. **Limitations of the CLIP Visual Encoder**: The quality of feature extraction by CLIP image encoders might degrade under severe image degradation.
4. **Only Validated on 4× Super-Resolution**: Other restoration tasks (e.g., denoising, deblurring) have not been systematically verified.
5. **High Training Cost**: Jointly training on 20M+ images and the DiT backbone entails substantial computational costs.

## Related Work & Insights

- **SUPIR**: Also performs restoration based on diffusion models, using SDXL + LLaVA captions, but is limited to U-Net and pure text prompts.
- **ControlNeXt**: A lightweight conditional injection method, serving as the design foundation for the conditional branch of this paper.
- **SD3/DiT**: Demonstrates the advantages of Transformer architectures in generative tasks, which this paper systematically applies to restoration for the first time.
- **Insights**: The text-image alignment mechanism in pretrained T2I models can be flexibly swapped; directly substituting text features with visual features is a more natural approach to conditional injection in image restoration.

## Rating

⭐⭐⭐⭐

This paper systematically introduces DiT to image restoration for the first time, featuring a precise design intuition of replacing text prompts with visual dual prompts. The perceptual quality metrics reach SOTA, although the PSNR performance is average. High practical utility and engineering completeness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)

</div>

<!-- RELATED:END -->
