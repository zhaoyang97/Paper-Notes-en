---
title: >-
  [Paper Note] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution
description: >-
  [ICCV 2025][Image Generation][3D-consistent super-resolution] 3DSR is proposed — an alternating iterative framework coupling diffusion-based SR with 3DGS to achieve 3D-consistent super-resolution: after each denoising step, SR images are used to train a 3DGS, yielding 3D-consistent renderings that are re-encoded into the latent space to guide the next denoising step. Without fine-tuning any model, it explicitly enforces cross-view consistency…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "3D-consistent super-resolution"
  - "diffusion models"
  - "3DGS"
  - "multi-view consistency"
  - "denoising guidance"
date: 2026-05-08
content_hash: f2ba5d1d4f81473f
---

# 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2508.04090](https://arxiv.org/abs/2508.04090)  
**Code**: [https://consistent3dsr.github.io/](https://consistent3dsr.github.io/)  
**Area**: Image Generation
**Keywords**: 3D-consistent super-resolution, diffusion models, 3DGS, multi-view consistency, denoising guidance

## TL;DR
3DSR is proposed — an alternating iterative framework coupling diffusion-based SR with 3DGS to achieve 3D-consistent super-resolution: after each denoising step, SR images are used to train a 3DGS, yielding 3D-consistent renderings that are re-encoded into the latent space to guide the next denoising step. Without fine-tuning any model, it explicitly enforces cross-view consistency, achieving +1.16 dB PSNR and 50% FID reduction on LLFF (vs. StableSR).

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: 3D scene reconstruction is fundamentally limited by the resolution of input images. Applying image SR frame-by-frame introduces inconsistent hallucinated details across views, causing blur and geometric artifacts after 3DGS training. Video SR implicitly models temporal consistency but does not guarantee 3D consistency. A framework is needed that explicitly leverages 3D representations to constrain multi-view consistency in diffusion-based SR.

### Mechanism

**Goal**: How can a diffusion model be exploited to generate high-quality SR details while simultaneously guaranteeing 3D structural consistency across views?

## Method

### Overall Architecture
Low-resolution multi-view images → Pretrained 3DGS (LR) → Iterative denoising (4-step StableSR-Turbo): at each step, (1) for each view: diffusion denoising yields clean latent $\hat{x}_0$ → decoded to SR image $H_i$; (2) all SR images are used to train 3DGS (5K iterations) → rendered to 3D-consistent image $R_i$; (3) $R_i$ is re-encoded to 3D-consistent latent $\tilde{x}_0$; (4) $\tilde{x}_0$ is combined with current $x_t$ to guide the next denoising step.

### Key Designs
1. **Diffusion–3D Representation Alternation**: The core innovation — 3DGS training and rendering are inserted at every denoising step. The diffusion model supplies fine-grained details, while 3DGS enforces 3D consistency. $\hat{x}_0$ (diffusion prediction: rich details but inconsistent) → train 3DGS → $R_i$ (3D-consistent but potentially losing details) → $\tilde{x}_0$ (3D-consistent latent) guides the next denoising step.
2. **Subsampling Regularization**: SR images are downsampled to LR resolution and aligned with the original LR input ($\mathcal{L}_{lr}$), preventing SR details from deviating excessively from the original signal. Total loss $= \mathcal{L}_{hr}$(SR image vs. rendering) $+ \lambda \cdot \mathcal{L}_{lr}$(original LR vs. downsampled rendering).
3. **Plug-and-Play without Fine-tuning**: The diffusion model, VAE, and 3DGS are all frozen or trained only with the rendering loss. No fine-tuning of video models or training of additional networks is required. The framework is compatible with arbitrary SR diffusion models or 3D representations.

### Loss & Training
- $\mathcal{L}_{\text{all}} = \mathcal{L}_{hr}(H_i, R_i) + \lambda \cdot \mathcal{L}_{lr}(L_i, R_{lr}^i)$
- $\mathcal{L}_{\alpha} = (1-\delta)\cdot\mathcal{L}_1 + \delta \cdot \mathcal{L}_{\text{DSSIM}}$, $\lambda=1$, $\delta=0.2$
- StableSR-Turbo (4-step denoising), 5K 3DGS training iterations per denoising step
- Mip-Splatting as the 3D representation; A6000 GPU

## Key Experimental Results

### LLFF (×8 downsampling, ×4 upsampling)

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | MEt3R (3D Consistency)↓ | FID↓ |
|--------|-------|-------|--------|------------------------|------|
| SuperGaussian (VSR) | 23.05 | 0.725 | 0.296 | 0.541 | 51.2 |
| StableSR (ISR) | 22.75 | 0.717 | 0.219 | 0.531 | 41.1 |
| DiSR-NeRF | 22.50 | 0.697 | 0.310 | 0.518 | 54.1 |
| **3DSR (Ours)** | **24.21** | **0.754** | **0.181** | **0.516** | **20.7** |

### MipNeRF360 (×8 downsampling, ×4 upsampling)

### Ablation Study

| Method | PSNR↑ | LPIPS↓ | FID↓ |
|--------|-------|--------|------|
| SuperGaussian | 25.25 | 0.303 | 32.7 |
| StableSR | 24.31 | 0.326 | 44.2 |
| **3DSR** | **26.10** | **0.222** | **22.4** |

## Highlights & Insights
- **3D Representation as a Consistency Constraint**: Inserting 3DGS into the diffusion denoising loop is a simple yet effective means of enforcing 3D consistency.
- **FID Halved**: From 41.1 → 20.7 (LLFF), demonstrating that 3D consistency constraints substantially reduce the distributional shift introduced by hallucinated details.
- **Plug-and-Play without Fine-tuning**: No pretrained model is modified; the method purely exploits the geometric inductive bias of 3DGS.
- **Compatible with Arbitrary SR Models**: The framework is applicable to both ISR and VSR approaches.

## Limitations & Future Work
- Training 5K 3DGS steps at every denoising step introduces significant computational overhead.
- The NIQE metric (no-reference perceptual quality) is inferior to StableSR, suggesting that 3D consistency constraints may sacrifice some perceptual quality.
- Validation is limited to ×4 upsampling.

## Related Work & Insights
- **vs. StableSR (ISR)**: Per-frame SR without 3D consistency leads to blurry results after 3DGS training; 3DSR explicitly guarantees consistency.
- **vs. SuperGaussian (VSR)**: Relies on the temporal consistency of video models rather than true 3D consistency.
- **vs. DiSR-NeRF**: Optimizes NeRF via SDS, producing black artifacts; 3DSR directly uses 3DGS renderings as guidance.

## Relevance to My Research
- The alternating diffusion–3D iteration framework is transferable to tasks such as 3D editing and 3D completion.
- The plug-and-play role of 3DGS within diffusion pipelines warrants further exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of alternating diffusion and 3DGS is intuitive yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, multiple SR baselines, and the MEt3R 3D consistency metric.
- Writing Quality: ⭐⭐⭐⭐ The motivation figure (Fig. 2) is persuasive and the method is described clearly.
- Value: ⭐⭐⭐⭐ The framework combining 3D consistency with diffusion-based SR offers strong reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[ICCV 2025\] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](../../CVPR2025/image_generation/difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

</div>

<!-- RELATED:END -->
