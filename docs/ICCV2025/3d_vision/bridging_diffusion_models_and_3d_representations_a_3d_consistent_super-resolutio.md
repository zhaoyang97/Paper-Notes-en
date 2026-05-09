---
title: >-
  [Paper Note] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework
description: >-
  [ICCV 2025][3D Vision][3D super-resolution] This paper proposes 3DSR, a framework that integrates 2D diffusion-based super-resolution with 3D Gaussian Splatting (3DGS) representations. At each diffusion denoising step, multi-view 3D consistency is enforced via 3DGS rendering, enabling high-fidelity and spatially consistent 3D scene super-resolution.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D super-resolution
  - diffusion models
  - 3D Gaussian splatting
  - multi-view consistency
  - novel view synthesis
date: 2026-05-08
content_hash: d906e2fd0f43bf42
---

# Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework

**Conference**: ICCV 2025
**arXiv**: [2508.04090](https://arxiv.org/abs/2508.04090)
**Code**: [Project Page](https://consistent3dsr.github.io/)
**Area**: 3D Vision
**Keywords**: 3D super-resolution, diffusion models, 3D Gaussian splatting, multi-view consistency, novel view synthesis

## TL;DR

This paper proposes 3DSR, a framework that integrates 2D diffusion-based super-resolution with 3D Gaussian Splatting (3DGS) representations. At each diffusion denoising step, multi-view 3D consistency is enforced via 3DGS rendering, enabling high-fidelity and spatially consistent 3D scene super-resolution.

## Background & Motivation

3D representation learning (e.g., NeRF, 3DGS) has achieved remarkable progress in novel view synthesis, yet synthesized images often lack fine-grained details due to the limited spatial resolution of input camera views. Existing super-resolution approaches suffer from three major limitations:

**Image Super-Resolution (ISR)**: Processes each view independently, introducing cross-view inconsistencies that lead to artifacts in 3D reconstruction.

**Video Super-Resolution (VSR)**: Achieves a degree of consistency through implicit spatio-temporal aggregation, but cannot guarantee 3D view consistency and exhibits poor detail preservation.

**Diffusion-based Super-Resolution**: Capable of generating high-quality details but prone to hallucinations, resulting in texture inconsistencies across views.

Core insight: diffusion models excel at generating high-frequency details but lack 3D geometric awareness, while 3DGS excels at enforcing multi-view consistency. Combining both yields the advantages of each.

## Method

### Overall Architecture

At each denoising timestep, 3DSR executes the following loop: (a) the diffusion SR model predicts clean latents → (b) decodes them into high-resolution images → (c) trains 3DGS with the high-resolution images and renders consistent views → (d) encodes the 3D-consistent renderings back into latent space → (e) performs the denoising step using the 3D-consistent latents. The encoder, decoder, and diffusion model are all kept frozen; only the 3D representation is optimized.

### Key Designs

1. **Multiview Consistent Noise Sampling**: Given a set of low-resolution (LR) input images $\{L^i\}$ and camera poses $\{P^i\}$, a 3DGS model is first pre-trained on the LR images. At the initial diffusion sampling step, noise latents $x_t^i$ are randomly initialized, and the SR model estimates clean latents $\hat{x}_0^i$, which are decoded into intermediate high-resolution images $H^i$. The critical step is to update the 3DGS representation with $H^i$ and obtain 3D-consistent high-resolution images $R^i$ via rendering. $R^i$ is re-encoded into latent space to yield 3D-consistent latents $\ddot{x}_0^i$, which replace the original estimate $\hat{x}_0^i$ for denoising:

    $x_{t-1}^i = \sqrt{\alpha_{t-1}} \ddot{x}_0^i + \eta_t \epsilon_\theta(x_t^i, t) + \sigma_t \epsilon_t$

2. **Subsampling-based Regularization**: The total loss comprises a high-resolution supervision term and a low-resolution consistency term:

    $\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{hr}}(H^i, R^i) + \lambda \mathcal{L}_{\text{lr}}(L^i, R_{lr}^i)$

   where $R_{lr}^i$ is the rendered image downsampled to LR resolution. Both terms use a combination of $\ell_1$ loss and D-SSIM loss:

    $\mathcal{L}_\alpha = (1-\delta)\mathcal{L}_1^\alpha + \delta \mathcal{L}_{\text{D-SSIM}}^\alpha$

3. **Plug-and-Play SR Model Compatibility**: The diffusion SR model in the framework is not restricted to single-image ISR or video VSR and is compatible with future, more advanced SR methods. Unlike SuperGaussian, the proposed method requires no fine-tuning of video models and enforces consistency more directly through an explicit 3D representation.

### Loss & Training

- Mip-Splatting is used as the 3D representation, pre-trained for 30K iterations on LR images.
- The number of diffusion sampling steps is set to 4 (using StableSR-Turbo), with 3DGS trained for 5K iterations per step.
- Subsampling weight $\lambda=1$, D-SSIM weight $\delta=0.2$.
- All experiments are conducted on an Nvidia A6000 GPU (49 GB).

## Key Experimental Results

### Main Results

**LLFF dataset** (×8 downsampling, ×4 upsampling):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | MEt3R↓ | FID↓ |
|--------|-------|-------|--------|-------|--------|------|
| SuperGaussian | 23.054 | 0.725 | 0.296 | 4.553 | 0.541 | 51.199 |
| DiSR-NeRF | 22.504 | 0.697 | 0.310 | 6.293 | 0.518 | 54.138 |
| StableSR | 22.748 | 0.717 | 0.219 | 4.793 | 0.531 | 41.129 |
| **3DSR (Ours)** | **24.212** | **0.754** | **0.181** | 4.632 | **0.516** | **20.731** |

**MipNeRF360 dataset** (×8 downsampling, ×4 upsampling):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | NIQE↓ | MEt3R↓ | FID↓ |
|--------|-------|-------|--------|-------|--------|------|
| SuperGaussian | 25.252 | 0.725 | 0.303 | 4.694 | 0.675 | 32.652 |
| StableSR | 24.313 | 0.700 | 0.326 | 5.177 | 0.644 | 44.174 |
| **3DSR (Ours)** | **26.097** | **0.746** | **0.222** | 5.065 | **0.625** | **22.438** |

### Ablation Study

Qualitative analysis is used to demonstrate the contribution of each component:

| Comparison | Observation |
|------------|-------------|
| SuperGaussian vs. 3DSR | SuperGaussian produces blurrier outputs due to limited training views in LLFF |
| StableSR vs. 3DSR | StableSR introduces artifacts and geometric distortions; hallucinated details are misaligned across views |
| DiSR-NeRF vs. 3DSR | DiSR-NeRF frequently produces black and blurry artifacts |
| 3DSR on indoor/outdoor scenes | Structural integrity and fine texture are preserved in both scene types |

- On LLFF (×8→×4), 3DSR achieves +1.46 dB PSNR and −20.4 FID compared to StableSR.
- On MipNeRF360, 3DSR achieves +0.85 dB PSNR and −10.2 FID compared to SuperGaussian.
- MEt3R (3D consistency metric) improves from 0.531/0.644 (StableSR) to 0.516/0.625 (3DSR).

### Key Findings

- Hallucinated details from diffusion models are inconsistent across views when SR is applied independently, causing blurring after 3DGS reconstruction.
- "Fusing" SR results from multiple views through 3DGS effectively smooths cross-view inconsistencies.
- The method requires no fine-tuning of diffusion or video models and serves as a general plug-and-play framework.
- 3DSR ranks second on the NIQE perceptual quality metric, suggesting that the 3D consistency constraint slightly limits single-image perceptual quality.

## Highlights & Insights

- **Elegant combination of diffusion and 3D representations**: The diffusion model is left unmodified; the 3D representation serves solely as a cross-view consistency regularizer.
- **Simple and general design**: The 3D representation can be replaced with other forms (e.g., NeRF), and the SR model can be substituted with any diffusion-based SR method.
- **Comprehensive evaluation metrics**: MEt3R (3D consistency) and FID are introduced into 3D super-resolution evaluation for the first time.
- Qualitative comparisons against StableSR clearly illustrate the fundamental difference between "SR then 3D" and "SR interleaved with 3D optimization."

## Limitations & Future Work

- Training 3DGS at every diffusion denoising step incurs high computational cost.
- Only ×4 and ×2 upsampling are evaluated; larger scale factors (e.g., ×8) remain unverified.
- StableSR-Turbo uses only 4 denoising steps; more steps may yield better results at higher computational cost.
- The impact of different 3D representations (e.g., NeRF) on performance is not explored.
- The NIQE metric is not optimal, indicating a trade-off between 3D consistency and single-image perceptual quality.

## Related Work & Insights

- MultiDiffusion and Generative Power of Ten achieve consistency at the 2D level via noise averaging; this paper extends that idea to 3D.
- SuperGaussian relies on the implicit consistency of VSR, whereas this work enforces consistency more directly through an explicit 3D representation.
- DiSR-NeRF employs Score Distillation Sampling; the proposed $x_0$-guidance strategy is more stable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of embedding 3DGS optimization within the diffusion denoising loop is natural and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across two datasets, multiple metrics, and multiple baselines; however, a quantitative ablation table is absent.
- **Writing Quality**: ⭐⭐⭐⭐ The method is clearly described, and the algorithmic pseudocode aids understanding.
- **Value**: ⭐⭐⭐⭐ Provides a general framework for high-quality 3D scene rendering with strong practical application potential.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](../../AAAI2026/3d_vision/debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)

<!-- RELATED:END -->
