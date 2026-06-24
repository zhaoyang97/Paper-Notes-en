---
title: >-
  [Paper Note] DPIR: Dual Prompting Image Restoration with Diffusion Transformers
description: >-
  [CVPR 2025][Image Restoration][Diffusion Transformer] This paper proposes DPIR, the first image restoration method based on the Diffusion Transformer (SD3). By utilizing a lightweight low-quality image conditioning branch and a dual visual-textual prompting control branch, DPIR enhances restoration quality and fidelity across both global context and local appearance visual dimensions.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Diffusion Transformer"
  - "Dual Prompting"
  - "Visual Prompting"
  - "SD3"
date: 2026-05-08
content_hash: 9dcf470ab24518c3
---

# DPIR: Dual Prompting Image Restoration with Diffusion Transformers

**Conference**: CVPR 2025  
**arXiv**: [2504.17825](https://arxiv.org/abs/2504.17825)  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Image Restoration, Diffusion Transformer, Dual Prompting, Visual Prompting, SD3

## TL;DR

This paper proposes DPIR, the first image restoration method based on the Diffusion Transformer (SD3). By utilizing a lightweight low-quality image conditioning branch and a dual visual-textual prompting control branch, DPIR enhances restoration quality and fidelity across both global context and local appearance visual dimensions.

## Background & Motivation

**Background**: Existing image restoration methods are mainly based on Latent Diffusion Models with U-Net architectures (e.g., StableSR, SUPIR). However, DiTs exhibit better generative potential due to their long-range dependency modeling and scalability.

**Limitations of Prior Work**: Conditioning methods such as ControlNet are designed for U-Net and cannot be directly applied to the ViT architecture of DiT. Pure text descriptions fail to adequately capture the rich visual features of low-quality images. Furthermore, DiT lacks the skip connections of U-Net, making it difficult to preserve information from the input image.

**Core Idea**: A CLIP image encoder is leveraged to extract local and global visual features as visual prompts to replace the CLIP text embedding in SD3, forming a dual prompt alongside the T5 text prompt.

## Method

### Key Designs

1. **Lightweight Low-Quality (LQ) Image Conditioning Branch**: A few convolutional layers extract LQ features, which are then injected into the first layer of the DiT via an adaptive feature alignment module (normalized to match the mean and variance of the first DiT layer's output).

2. **Dual Prompting Control Branch**: A CLIP image encoder extracts visual token embeddings (local) and cls embeddings (global) from the LQ image. After MLP adaptation, these replace the original CLIP text embeddings and are concatenated with the T5 text prompts to form a dual prompt.

3. **Degradation-Robust VAE Encoder**: The 16-channel SD3 VAE encoder is fine-tuned, incorporating LPIPS and GAN losses to preserve fine details.

### Loss & Training

The conditional flow matching objective of SD3 is employed. The training dataset contains over 20 million high-quality images. A global-local visual prompting training strategy is used: during training, patches are cropped to extract local information, while surrounding regions are utilized to extract the global context.

## Key Experimental Results

### Main Results

The method comprehensively outperforms existing approaches such as Real-ESRGAN, StableSR, SinSR, and SUPIR on datasets like DIV2K, achieving optimal performance in both visual quality and fidelity.

### Key Findings
- Visual prompting significantly improves restoration fidelity compared to pure text prompting (improving PSNR by approximately $1.2\text{ dB}$).
- Combining global and local visual information outperforms a single dimension ($+0.5\text{ dB}$ vs. local only).
- The DiT architecture offers clear advantages over U-Net in terms of restored visual quality, reducing LPIPS by $15\%$.
- The degradation-robust VAE encoder improves LPIPS by $20\%$ on severely degraded inputs.

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Real-ESRGAN | 24.3 | 0.72 | 0.35 |
| StableSR | 25.1 | 0.74 | 0.31 |
| SUPIR | 25.8 | 0.76 | 0.28 |
| **DPIR** | **26.5** | **0.78** | **0.24** |


- Visual prompts significantly improve restoration fidelity compared to pure text prompts.
- The combination of global and local visual information outperforms a single dimension.
- The DiT architecture shows a clear advantage in restoration quality over U-Net.

## Highlights & Insights

- Introduces SD3/DiT to image restoration for the first time.
- Replacing text embeddings with visual embeddings is simple yet effective.
- The degradation-robust VAE encoder preserves details of high-quality inputs.

## Limitations & Future Work

- The inference speed of SD3 is relatively slow, limiting its use in real-time application scenarios.
- The effective receptive field of global visual prompting is limited by CLIP; thus, extremely large images may lose global context.
- DiT architectures lack the skip connections of U-Net, which may result in a lower upper bound for fidelity compared to U-Net-based methods.
- The training dataset exceeds 20 million images, and its effectiveness in small-scale scenarios remains unverified.
- The degradation-robust VAE encoder is specifically designed for the 16-channel architecture of SD3, requiring redesign for transfer to other DiT models.
- Unified restoration for multiple degradation types has not been explored (currently only super-resolution is validated).
- The fusion strategy of visual and textual prompts is simple (direct concatenation), where more complex fusion mechanisms might further improve results.
- Generalization on in-the-wild images has not been fully evaluated, and discrepancies between training data and test distributions might affect real-world outcomes.

## Related Work & Insights
- **vs StableSR/SUPIR**: Based on U-Net architectures; DPIR introduces DiT to image restoration for the first time, demonstrating advantages in visual quality.
- **vs ControlNet**: ControlNet is designed for U-Net and is incompatible with the ViT architecture of DiT; DPIR's lightweight conditioning branch is an alternative designed specifically for DiT.
- **vs SinSR**: SinSR performs one-step distillation for acceleration without altering the base architecture, whereas DPIR introduces DiT at the architectural level.
- Technical Depth: 7/10 — Clever method design
- Experimental Thoroughness: 8/10 — Extensive comparisons
- Writing Quality: 7/10

### Methodological Insights
- The core contribution of this work lies in introducing a new architecture to the field, revealing new technical possibilities.
- The experimental design covers multiple baselines and scenarios, leading to statistically significant conclusions.
- The individual components of the method are replaceable, facilitating subsequent improvements and optimization.
- It exhibits good compatibility with the existing technical ecosystem, lowering the barrier to adoption.
- It provides a tunable balance between computational efficiency and generation quality.
- The open-sourced code and model weights are of significant value for community reproduction.
- It drives technical innovation based on practical application needs, with a clear problem definition.
- Comparison and analysis with concurrent related works are thorough, showing a clear positioning.
- Lightweight variants can be explored in the future to adapt to edge device deployment.
- Cross-modal and cross-task transferability represents an important direction for future validation.
- Integration with self-supervised learning and contrastive learning is worth exploring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](../../ICML2026/image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2026\] DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers](../../CVPR2026/image_restoration/diffdecompose_layer-wise_decomposition_of_alpha-composited_images_via_diffusion_.md)
- [\[ICLR 2026\] Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization](../../ICLR2026/image_restoration/analyzing_the_training_dynamics_of_image_restoration_transformers_a_revisit_to_l.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)

</div>

<!-- RELATED:END -->
