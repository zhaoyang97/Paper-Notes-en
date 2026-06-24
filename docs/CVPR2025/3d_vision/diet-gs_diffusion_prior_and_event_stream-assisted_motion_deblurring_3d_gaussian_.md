---
title: >-
  [Paper Note] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][Motion Deblurring] A two-stage framework, DiET-GS, is proposed. It jointly constrains 3DGS optimization using event double-integral (EDI) priors and a pre-trained diffusion model to reconstruct clean 3D representations from blurry multi-view images and event streams, achieving high-quality novel view synthesis with accurate colors and fine details.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Motion Deblurring"
  - "3D Gaussian Splatting"
  - "Event Camera"
  - "Diffusion Prior"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 9b45aac658154ea5
---

# DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.24210](https://arxiv.org/abs/2503.24210)  
**Code**: [https://diet-gs.github.io](https://diet-gs.github.io)  
**Area**: 3D Vision  
**Keywords**: Motion Deblurring, 3D Gaussian Splatting, Event Camera, Diffusion Prior, Novel View Synthesis

## TL;DR

A two-stage framework, DiET-GS, is proposed. It jointly constrains 3DGS optimization using event double-integral (EDI) priors and a pre-trained diffusion model to reconstruct clean 3D representations from blurry multi-view images and event streams, achieving high-quality novel view synthesis with accurate colors and fine details.

## Background & Motivation

Reconstructing sharp 3D representations from blurry multi-view images is a long-standing challenge in computer vision. Event cameras provide unique advantages for motion deblurring due to their high dynamic range and microsecond-level temporal resolution. However, existing methods suffer from two key limitations: (1) relying solely on blurry images for color restoration leads to inaccurate color reconstruction; (2) event streams, while providing blur-free details, are prone to introducing artifacts (due to unknown threshold $\Theta$ and event noise accumulation). The core idea of this work is: **the "model-driven" EDI prior provides precise physical constraints, while the "data-driven" diffusion prior provides regularization of natural image distributions, and the two complement each other to achieve optimal visual quality.**

## Method

### Overall Architecture

DiET-GS consists of two stages: Stage 1 (DiET-GS) jointly optimizes deblurred 3DGS using event-stream EDI constraints and diffusion RSD loss; Stage 2 (DiET-GS++) freezes Stage 1 parameters and introduces additional learnable Gaussian features $\mathbf{f_g}$ to maximize the efficacy of the diffusion prior for enhancing edge details.

### Key Designs

1. **Multi-level EDI Constraint System**:
    - **Function**: Extracts precise color and detail supervision from event streams.
    - **Mechanism**: Three levels of EDI constraints work synergistically:
        - $\mathcal{L}_{\text{edi\_gray}}$: Restores fine details in the luminance domain via a learnable CRF function.
        - $\mathcal{L}_{\text{edi\_color}}$: Performs channel-wise deblurring in RGB space to restore accurate colors.
        - $\mathcal{L}_{\text{edi\_simul}}$: Replaces real blurry images with simulated blurry images to construct EDI constraints, ensuring cycle consistency among objective functions.
    - **Design Motivation**: $\mathcal{L}_{\text{edi\_gray}}$ introduces a learnable CRF to compensate for the discrepancy between RGB values and pixel intensities, which is closer to the real world than directly treating each RGB channel as luminance; the two constraints complement and compensate for each other (grayscale $\rightarrow$ details but color bias, RGB $\rightarrow$ accurate color but over-smoothing).

2. **Renoised Score Distillation (RSD) Diffusion Prior**:
    - **Function**: Uses the natural image prior of a pre-trained diffusion model to regularize rendering results.
    - **Mechanism**: Encodes the rendered blurry image into the latent space $\mathbf{z}_0 = \mathcal{E}(\hat{\mathbf{C}}^B)$, adds noise at timesteps $t$ and $t-1$ to obtain $\mathbf{z}_t, \mathbf{z}_{t-1}$, and uses a diffusion UNet to predict the denoised result $\hat{\mathbf{z}}_{t-1}$, optimizing $\|\mathbf{z}_{t-1} - \hat{\mathbf{z}}_{t-1}\|$.
    - **Design Motivation**: While event stream constraints are precise, they are prone to producing unnatural artifacts. The diffusion prior provides naturalness constraints. The blurry GT is used as a condition to substitute for the unavailable sharp images.

3. **Stage 2 Learnable Latent Space Residual (DiET-GS++)**:
    - **Function**: Maximizes the diffusion prior effect to further enhance edge details.
    - **Mechanism**: Attaches a zero-initialized feature $\mathbf{f_g} \in \mathbb{R}^D$ to each 3D Gaussian, renders a 2D feature map $\mathbf{f}_{2D}$ via 3DGS, and adds it to the encoded rendered image $\mathbf{z}_0$ to obtain the refined latent variable $\mathbf{z}'_0 = \mathbf{z}_0 + \mathbf{f}_{2D}$, optimizing only $\mathbf{f_g}$ using the RSD loss.
    - **Design Motivation**: In Stage 1, the balance between event constraints and RSD weakens the diffusion effect. Stage 2 freezes the original parameters and only trains the residual, preventing the destruction of the event prior learned in Stage 1, while utilizing the rendering capability of 3DGS to directly obtain the latent space residual in novel views (which is simpler than DiSR-NeRF).

### Loss & Training

- **Stage 1**: $\mathcal{L}_{s1} = \lambda_{\text{blur}} \mathcal{L}_{\text{blur}} + \lambda_{\text{ev}} \mathcal{L}_{\text{ev}} + \lambda_{\text{edi}} \mathcal{L}_{\text{edi}} + \lambda_{\text{rsd}} \mathcal{L}_{\text{rsd}}$
    - $\lambda_{\text{blur}} = \lambda_{\text{edi}} = \lambda_{\text{rsd}} = 1.0$, $\lambda_{\text{ev}} = 0.1$
    - Trained for 100K iterations
- **Stage 2**: Only RSD loss, trained for 2K iterations ($\le$ 20 minutes), with a linearly decaying time schedule
- Initialization uses sharp images restored by EDI for SfM.

## Key Experimental Results

### Main Results

| Dataset | Metric | DiET-GS | DiET-GS++ | Prev. SOTA (Ev-DeblurNeRF) | Gain |
|--------|------|---------|-----------|------------------------|------|
| EvDeblur-blender | PSNR↑ | **26.69** | 26.23 | 24.76 | +1.93dB |
| EvDeblur-blender | LPIPS↓ | 0.1064 | **0.1052** | 0.1788 | -41% |
| EvDeblur-blender | MUSIQ↑ | 57.67 | **59.91** | 42.38 | +41% |
| EvDeblur-CDAVIS | PSNR↑ | **34.22** | 33.16 | 32.30 | +1.92dB |
| EvDeblur-CDAVIS | LPIPS↓ | **0.0496** | 0.0502 | 0.0571 | -13% |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | Description |
|------|-------|-------|--------|--------|------|
| $\mathcal{L}_{\text{blur}}$ only | 29.73 | 0.7797 | 0.2160 | 24.77 | Blurry reconstruction only |
| + $\mathcal{L}_{\text{ev}}$ | 32.74 | 0.8460 | 0.1173 | 39.69 | Event constraints +3.01dB |
| + edi_gray + edi_color | 34.92 | 0.9033 | 0.0624 | 43.79 | Complementary EDI dual constraints |
| + edi_simul | 35.04 | 0.9068 | 0.0587 | 45.04 | Cycle consistency regularization |
| + RSD (S1) | 34.89 | 0.9049 | 0.0600 | 45.37 | S1 diffusion slightly decreases PSNR |
| + RSD (S2, DiET-GS++) | 33.86 | 0.8846 | 0.0634 | **51.71** | Significant boost in no-reference quality |

### Key Findings

- EDI gray and color constraints are complementary: gray restores texture details but introduces color shift, whereas color restores accurate colors but is over-smoothed.
- Stage 2 (DiET-GS++) slightly decreases PSNR/SSIM but significantly improves no-reference quality metrics (MUSIQ +6.34, CLIP-IQA +0.037).
- Cycle consistency regularization $\mathcal{L}_{\text{edi\_simul}}$ visually suppresses local artifacts effectively.

## Highlights & Insights

- **Exquisite multi-level EDI constraint system design**: Combining detail restoration in the luminance domain, color restoration in the RGB domain, and regularization in the simulated domain forms a comprehensive constraint system.
- **Necessity of a two-stage training strategy**: It Grouping event constraints and diffusion priors and optimizing them in separate stages cleverly resolves the mutual dampening effect when jointly optimizing them.
- Utilizing the explicit rendering capability of 3DGS to directly obtain learnable latent space residuals is simpler than the DiSR-NeRF paradigm and requires no extra synchronization steps.

## Limitations & Future Work

- DiET-GS++ is driven by a generative model, which may produce details that are not fully consistent with the GT (resulting in a decrease in PSNR).
- The setting of the event threshold $\Theta$ still requires manual tuning (0.2 for synthetic, 0.25 for real).
- It relies on SfM initialization, which, although mitigated by EDI preprocessing, remains a limitation.
- Stage 2 is only trained for 2K iterations; longer training or a larger model might yield further improvements.

## Related Work & Insights

- **Core difference from Ev-DeblurNeRF**: It introduces a learnable CRF for EDI constraints in the luminance domain, which is more reasonable than directly treating RGB as luminance.
- **Core difference from DiSR-NeRF**: It utilizes 3DGS to directly render latent space residuals, which is more concise and efficient.
- **Inspiration**: The combination paradigm of event camera + diffusion model can be generalized to other degradation tasks (such as low-light, HDR).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The EDI multi-constraint system and two-stage strategy are exquisitely designed, though individual components are not entirely novel when viewed in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough evaluation on both synthetic and real datasets with detailed ablation studies, though the dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear system description and detailed formulation derivations, though some notation is slightly redundant.
- **Value**: ⭐⭐⭐⭐ Advances the SOTA in the niche field of event-assisted deblurring 3D reconstruction, while practical utility is limited by the prevalence of event cameras.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](../../ICCV2025/3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
