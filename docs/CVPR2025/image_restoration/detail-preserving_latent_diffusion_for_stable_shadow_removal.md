---
title: >-
  [Paper Note] Detail-Preserving Latent Diffusion for Stable Shadow Removal
description: >-
  [CVPR 2025][Image Restoration][Shadow Removal] This paper proposes a two-stage Stable Diffusion fine-tuning scheme for shadow removal: In the first stage, the denoiser is fine-tuned in the latent space to perform primary shadow removal. In the second stage, a shadow-aware Detail Injection module extracts features from the VAE encoder to modulate the decoder, recovering the high-frequency details lost in the first stage and achieving high-quality and highly generalizable shado…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Shadow Removal"
  - "Latent Diffusion Models"
  - "Stable Diffusion Fine-Tuning"
  - "Detail Injection"
  - "Cross-Dataset Generalization"
date: 2026-05-08
content_hash: 6bde4609b292ac00
---

# Detail-Preserving Latent Diffusion for Stable Shadow Removal

**Conference**: CVPR 2025  
**arXiv**: [2412.17630](https://arxiv.org/abs/2412.17630)  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Shadow Removal, Latent Diffusion Models, Stable Diffusion Fine-Tuning, Detail Injection, Cross-Dataset Generalization

## TL;DR
This paper proposes a two-stage Stable Diffusion fine-tuning scheme for shadow removal: In the first stage, the denoiser is fine-tuned in the latent space to perform primary shadow removal. In the second stage, a shadow-aware Detail Injection module extracts features from the VAE encoder to modulate the decoder, recovering the high-frequency details lost in the first stage and achieving high-quality and highly generalizable shadow removal.

## Background & Motivation

**Background**: Deep learning-based shadow removal methods have made significant progress by learning pixel-level mappings (e.g., ShadowFormer, HomoFormer). However, they are prone to overfitting due to limited training data, leading to degraded performance when generalizing to unseen data.

**Limitations of Prior Work**: (1) Existing methods suffer a sharp performance drop under cross-dataset evaluation; (2) Directly applying Stable Diffusion (SD) to shadow removal leads to lost high-frequency details—the VAE of SD performs lossy compression (W×H×3 → W/8×H/8×4) during mapping from pixel space to latent space; (3) Existing diffusion-based shadow removal methods (e.g., Refusion) are trained end-to-end on shadow datasets without utilizing the rich visual priors of pretrained SD.

**Key Challenge**: SD possesses strong generalizable priors but loses details in the latent space; directly performing diffusion in the pixel space is computationally expensive and loses global context. Furthermore, the dual objectives of "thorough shadow removal" and "preserving details in non-shadowed areas" can be contradictory.

**Goal**: To leverage pretrained SD priors to achieve highly generalizable shadow removal while preserving the fine-grained texture details of the input image.

**Key Insight**: Two-stage decoupling—performing "coarse but comprehensive" shadow removal in the latent space first, followed by "fine but localized" detail restoration in the pixel space.

**Core Idea**: Stage 1 fine-tunes LDM for shadow removal (with VAE fixed), and Stage 2 utilizes a Detail Injection (DI) module to extract features from the VAE encoder and inject them into the decoder to recover high-frequency details. The DI module concatenates shadowed and shadow-free features to achieve implicit shadow awareness.

## Method

### Overall Architecture
Stage 1: Fix the pretrained VAE, use the latent of the shadow image $z^x$ as the condition, and fine-tune the U-Net to generate the shadow-free latent $\hat{z}^y$, which is then decoded to produce a coarse shadow-free image. Stage 2: Fix the VAE encoder and decoder, and train the Detail Injection (DI) module. In each layer of the decoder, the encoder features of the shadow image $e_i$ are concatenated with the decoder features of the coarse shadow-free latent $d_i$. These features are processed by an RRDB network and added back to the decoder to recover high-frequency textures.

### Key Designs

1. **Latent Space Shadow Removal (Stage 1)**:

    - **Function**: Leverages SD generalization priors to complete primary shadow removal.
    - **Mechanism**: Concatenates the latent of the shadow image $z^x$ along the channel dimension with the noisy latent as a conditional input to the U-Net. Instead of $\epsilon$-prediction, it uses $z_0$-prediction (directly predicting the clean latent), which demonstrates lower variance and more stable results in experiments. DDIM is used for fast sampling.
    - **Design Motivation**: Although the latent space of the pretrained VAE is lossy, it effectively represents shadow-free images (experiments demonstrate that the VAE reconstruction quality for shadow-free images is acceptable). Performing global self-attention in the low-resolution latent space allows capturing long-range dependencies between shadow and non-shadow regions.

2. **Shadow-Aware Detail Injection (Stage 2, DI Module)**:

    - **Function**: Extracts and injects "shadow-free" high-frequency details from the original shadow image.
    - **Mechanism**: The DI module receives encoder features (which contain original image details but also shadow information) and decoder features (which are shadow-free but lack details), using an RRDB network to learn selective injection. DINOv2 features are additionally fused to enhance generalization. Crucially, concatenating both features allows the network to implicitly distinguish shadow regions (the encoder and decoder features differ significantly in shadowed regions, but are highly similar in non-shadowed regions), thereby injecting only shadow-free details.
    - **Design Motivation**: PCA visualization shows that the intermediate features of RRDB respond differently to shadowed regions (marked in green), demonstrating the shadow-aware capability of the module. Keeping the VAE decoder weights fixed preserves the pretrained priors.

3. **$z_0$-prediction + Low-Variance Sampling**:

    - **Function**: Provides a highly stable and consistent shadow removal output.
    - **Mechanism**: Uses $z_0$-prediction instead of standard $\epsilon$-prediction, reducing the output variance from 1.075 (DeS3) to 0.146.
    - **Design Motivation**: Shadow removal requires deterministic results (unlike image generation which benefits from diversity); low-variance sampling ensures consistent results across different runs.

### Loss & Training
Stage 1: $L_2$ loss in the latent space (predicting $z_0$). Stage 2: $L_1$ loss + LPIPS perceptual loss.

## Key Experimental Results

### Main Results (ISTD+ Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| ShadowFormer | 32.90 | 0.979 | - |
| ShadowRefiner | 34.67 | 0.983 | - |
| **Ours (Stage 2)** | **35.02** | **0.985** | **Best** |
| DeS3 (Diffusion Method) | 31.33 | - | - |

### Ablation Study

| Configuration | PSNR↑ | Variance↓ | Description |
|------|-------|------|------|
| Stage 1 ($\epsilon$-pred) | 29.66 | 0.239 | Standard prediction |
| Stage 1 ($z_0$-pred) | 29.95 | **0.146** | More stable |
| Stage 2 (Full) | **35.02** | 0.160 | Detail injection brings significant improvement |

### Key Findings
- Stage 2 detail injection improves PSNR by approximately 5dB (from 29.95 to 35.02), showing a significant effect.
- $z_0$-prediction reduces the variance by 39% compared to $\epsilon$-prediction, resulting in a more stable output.
- **Cross-dataset generalization**: When trained on one dataset and tested on another, the proposed method experiences the smallest performance drop, significantly outperforming other approaches.
- Supports high-resolution inputs (1920×1440) without requiring patch-based processing.

## Highlights & Insights
- The two-stage decoupled design is elegant and precise: Stage 1 utilizes SD priors for the "rough work" (global shadow elimination) while Stage 2 employs a lightweight CNN for the "fine work" (local detail restoration), perfectly complementing each other.
- The shadow-aware mechanism of the DI module is ingenious: it utilizes the feature discrepancy between shadowed and shadow-free representations to implicitly locate shadow regions, eliminating the need for an explicit shadow mask.
- The advantages of $z_0$-prediction in deterministic restoration tasks deserve wider attention in the community.

## Limitations & Future Work
- The two-stage inference process is slower than single-stage methods.
- Stage 1 still requires multi-step diffusion sampling (DDIM), which is slower than pure feed-forward methods.
- Training requires paired shadowed and shadow-free image data.
- Future work could consider distilling the two stages into a single feed-forward network.

## Related Work & Insights
- **vs Refusion**: Refusion performs diffusion in the latent space but is trained end-to-end, failing to utilize the pretrained SD priors and resulting in poor generalization. In contrast, the proposed method leverages pretrained SD combined with two-stage fine-tuning to significantly boost generalization.
- **vs ShadowDiffusion/DeS3**: These methods perform diffusion in the pixel space, which is computationally expensive and exhibits high variance. The proposed method operates in the latent space, offering higher efficiency.
- **vs ShadowFormer**: Pure feed-forward methods are fast but suffer from limited generalization. This work leverages SD priors to achieve stronger generalization capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of two-stage SD fine-tuning and detail injection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ SOTA results, cross-dataset generalization, variance analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline and convincing PCA visualizations.
- Value: ⭐⭐⭐⭐ A practical shadow removal solution with outstanding generalization capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](../../CVPR2026/image_restoration/phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2025\] MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)
- [\[CVPR 2026\] Rectifying Latent Space for Generative Single-Image Reflection Removal](../../CVPR2026/image_restoration/rectifying_latent_space_for_generative_single-image_reflection_removal.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)

</div>

<!-- RELATED:END -->
