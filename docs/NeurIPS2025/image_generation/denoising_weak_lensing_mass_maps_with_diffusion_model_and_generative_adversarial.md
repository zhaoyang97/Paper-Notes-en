---
title: >-
  [Paper Note] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network
description: >-
  [Image Generation] This work applies diffusion models (DM) to the task of weak gravitational lensing mass map denoising and conducts a systematic comparison with GAN (pix2pix) under identical experimental settings, demonstrating that DM comprehensively outperforms GAN in terms of training stability, robustness under multi-sample averaging, and reconstruction accuracy across multiple statistical estimators.
tags:
  - Image Generation
date: 2026-05-08
content_hash: fcb3346494a044c6
---

# Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network

## Basic Information
- **arXiv**: 2511.16415
- **Conference**: NeurIPS 2025
- **Authors**: Shohei D. Aoyama, Ken Osato, Masato Shirasaki
- **Affiliations**: Chiba University, National Astronomical Observatory of Japan
- **Code**: Not released

## TL;DR
This work applies diffusion models (DM) to the task of weak gravitational lensing mass map denoising and conducts a systematic comparison with GAN (pix2pix) under identical experimental settings, demonstrating that DM comprehensively outperforms GAN in terms of training stability, robustness under multi-sample averaging, and reconstruction accuracy across multiple statistical estimators.

## Background & Motivation
Weak lensing (WL) reconstructs the projected matter density field of the universe — commonly referred to as a "mass map" — by measuring subtle shape distortions of distant galaxies. However, due to the finite number of source galaxies, intrinsic galaxy shapes dilute the WL signal, giving rise to **shape noise**. Denoising is therefore a critical preprocessing step for accurate WL measurements.

Existing approaches:
- **GAN (pix2pix)**: Already applied to WL denoising, but suffers from training instability and mode collapse
- **Diffusion Models (DM)**: Demonstrated strong performance in image processing, yet a fair and systematic comparison with GANs on the WL denoising task has been lacking

## Core Problem
In the task of weak lensing mass map denoising, do diffusion models genuinely outperform GANs, and if so, in what specific respects?

## Method

### 1. Problem Formulation
The objective is to learn an optimal mapping from noisy mass maps to noise-free mass maps, which is essentially a conditional image-to-image translation problem.

### 2. GAN Denoising Model (pix2pix)
The pix2pix framework is adopted:
- **Generator $G$**: U-Net architecture that takes a noisy map as input and outputs a denoised map
- **Discriminator $D$**: Four convolutional blocks that classify input–target image pairs as real or fake
- Alternative loss functions including LSGAN and WGAN-gp were tested, but the original pix2pix loss yielded the best results
- Limitations: Poor generation diversity, as the generator tends to ignore the latent noise term; training instability

### 3. Diffusion Model Denoising (Palette)
The Palette framework is adopted for conditional image translation:
- **Forward process**: Gaussian noise is progressively added to the data over $T$ steps
- **Reverse process**: A deep neural network approximates the denoising step, iteratively generating the target data from Gaussian noise
- **Network architecture**: U-Net with 3 downsampling and upsampling levels
- **Noise schedule**: Quadratic scheduling achieves the best performance, as diffusion magnitude remains small over many steps, facilitating the capture of small-scale features
- **Diffusion steps**: $T=4000$ for training, $T=2000$ for inference
- Optimization target is the reweighted variational lower bound proposed by Ho et al. (2020)

### 4. Simulation Dataset
The $\kappa$TNG weak lensing simulation dataset is used:
- 10,000 pseudo-independent simulated mass maps derived from the IllustrisTNG cosmological hydrodynamic simulation
- Original size: $5 \times 5 \, \text{deg}^2$ with $1024^2$ grids
- Cropped into 4 equal-area sub-images and average-pooled to $256^2$
- Gaussian smoothing applied with FWHM = $2.5'$
- Final split: 39,000 training + 1,000 test images

### 5. Diversity Strategy
- **DM**: Given the same noisy input map, 5 denoised outputs are generated using different initial Gaussian noise realizations, reflecting the learned probability distribution
- **GAN**: Five networks are trained with different weight initializations, each producing one output; variance across outputs reflects uncertainty due to suboptimal weight convergence

## Key Experimental Results

### Pixel-Level Metrics

| Method | RMSE ($\times 10^{-2}$) ↓ | Pearson $\rho$ ↑ |
|--------|--------------------------|-------------------|
| No denoising | 1.47 | 0.67 |
| GAN single sample | 1.12 | 0.644 |
| DM single sample | 1.11 | 0.638 |
| GAN 5-sample mean | 0.87 | 0.758 |
| DM 5-sample mean | 0.86 | 0.757 |
| GAN 5-sample median | 0.90 | 0.743 |
| DM 5-sample median | 0.89 | 0.742 |

- Denoising substantially reduces RMSE; GAN and DM show minimal differences at the pixel level
- Averaging or taking the median over multiple samples further improves both metrics

### Statistical Reconstruction

**Angular Power Spectrum $C(\ell)$**:
$$C(\ell) = \frac{\sum_{\ell - \Delta\ell/2 < \ell' < \ell + \Delta\ell/2} |\tilde{\kappa}(\boldsymbol{\ell}')|^2}{\sum_{\ell - \Delta\ell/2 < \ell' < \ell + \Delta\ell/2} 1}$$

- DM reconstructs the power spectrum with error < 0.1 (normalized standard deviation) up to $\ell \lesssim 6000$
- GAN achieves accurate reconstruction only at large scales $\ell \lesssim 1000$
- Variance across the 5 GAN networks is large, whereas the 5 DM samples are highly consistent

**One-Point Probability Density Function (PDF)**:
- DM achieves reconstruction accuracy < 0.1 over the full range
- GAN exhibits larger deviations in the tail regions

**Other Statistics** (angular bispectrum, scattering transform, etc.): DM comprehensively outperforms GAN across all estimators

### Computational Cost (Single A100 GPU)

| Method | Training Time | Inference (1,000 images) |
|--------|--------------|--------------------------|
| GAN | ~28 hours (200 epochs) | ~minutes |
| DM | ~45 hours (85 epochs) | ~6 hours (22s/image) |

## Highlights & Insights
1. **First systematic fair comparison**: GAN and DM are compared on the same dataset under identical settings, filling a gap in the WL denoising literature
2. **Robust sampling with DM**: Consistency across 5 DM samples is substantially higher than across 5 GAN networks, reflecting a learned probability distribution rather than weight uncertainty
3. **Small-scale feature recovery**: DM accurately reconstructs the power spectrum even in noise-dominated regimes ($\ell > 2000$), while GAN fails to do so
4. **Quadratic noise scheduling**: A key finding for WL tasks — slowly increasing noise facilitates learning of small-scale features

## Limitations & Future Work
1. **Inference speed**: DM inference is ~100× slower than GAN (~22s vs. <1s per image), limiting applicability to large-scale observational data processing
2. **Limited resolution**: Validation is conducted only on $256^2$ grids; scalability to higher resolutions has not been explored
3. **Single cosmological model**: Testing is performed under fixed cosmological parameters; generalization across varying cosmologies remains unverified
4. **Shape noise only**: Other systematic effects (e.g., PSF residuals, photometric redshift errors) are not addressed

## Related Work & Insights
- **vs. Shirasaki+2019, Whitney+2024**: These works apply GANs to WL denoising but do not compare with DMs
- **vs. Remy+2023, Boruah+2025**: These works apply DMs to WL denoising but do not compare with GANs under the same experimental settings
- **Contribution of this paper**: The first systematic comparison within a unified framework, yielding definitive conclusions

The work also highlights broader connections:
- **Astronomy + generative models**: WL denoising is a canonical application of generative models in science — mapping noisy observations to clean signals
- **Multi-sample aggregation**: The DM multi-sample averaging strategy is transferable to other scientific image denoising tasks
- **Inference acceleration**: Techniques such as DDIM and consistency models can address the slow inference of DMs

## Rating
- Novelty: ⭐⭐⭐☆☆ — The methods themselves are applications of existing models; the contribution lies in the systematic comparison
- Technical Depth: ⭐⭐⭐⭐☆ — Comprehensive evaluation across multiple statistical estimators demonstrates cosmological expertise
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Broad coverage of statistics, though validation across different cosmological parameters is lacking
- Writing Quality: ⭐⭐⭐⭐☆ — Problem statement is clear, comparison is fair, and conclusions are well-defined

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)

<!-- RELATED:END -->
