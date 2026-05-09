---
title: >-
  [Paper Note] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][super-resolution] This paper proposes a calibration-based RAW-domain degradation modeling framework that accurately calibrates SR blur kernels and sensor noise models for multiple smartphone cameras, enabling the "unprocessing" of public sRGB images into realistic LR RAW data for training. The approach significantly outperforms baselines based on generic degradation pools in both camera-specific and cross-camera blind super-resolution settings.
tags:
  - CVPR 2026
  - Image Restoration
  - super-resolution
  - RAW domain
  - degradation modeling
  - smartphone camera
  - blur kernel calibration
  - sensor noise
date: 2026-05-08
content_hash: 7fe765720d6e0274
---

# RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.12493](https://arxiv.org/abs/2603.12493)
**Authors**: Ali Mosleh, Faraz Ali, Fengjia Zhang, Stavros Tsogkas, Junyong Lee, Alex Levinshtein, Michael S. Brown (Samsung AI Center-Toronto)
**Area**: Image Restoration
**Keywords**: super-resolution, RAW domain, degradation modeling, smartphone camera, blur kernel calibration, sensor noise

## TL;DR

This paper proposes a calibration-based RAW-domain degradation modeling framework that accurately calibrates SR blur kernels and sensor noise models for multiple smartphone cameras, enabling the "unprocessing" of public sRGB images into realistic LR RAW data for training. The approach significantly outperforms baselines based on generic degradation pools in both camera-specific and cross-camera blind super-resolution settings.

## Background & Motivation

Smartphone digital zoom relies on learning-based super-resolution (SR) models that operate directly on RAW sensor images. Acquiring high-quality paired training data, however, presents fundamental challenges:

- **Difficulty of real data acquisition**: Capturing the same scene at different focal lengths or with cameras of different quality levels to obtain HR-LR pairs requires precise alignment and static scenes, making the process time-consuming and labor-intensive.
- **Domain gap of synthetic data**: Commonly used bicubic downsampling ignores the complex degradations of real optical systems, leading to poor generalization of trained models on real smartphone images.
- **Limitations of generic degradation pools**: Methods such as Real-ESRGAN randomly sample isotropic/anisotropic Gaussian kernels and noise, but these hand-crafted parameters cannot accurately represent the true PSF of a target smartphone camera.
- **Problems with sRGB-domain degradation**: Most existing methods apply degradations in the sRGB domain, whereas blur and noise actually occur at earlier stages of the camera pipeline (the RAW domain), where the relationship between scene radiance and sensor response remains linear. Nonlinear processing in the sRGB domain introduces an additional domain gap.

Core insight: **Principled, carefully designed degradation modeling** can substantially improve real-world SR performance. Rather than relying on generic priors, device-specific degradation parameters should be obtained through calibration.

## Method

### Overall Architecture

The paper proposes a complete pipeline from calibration to data synthesis to SR training:
1. Calibrate SR blur kernels and sensor noise models for multiple smartphone cameras.
2. Use the calibration results to "unprocess" public sRGB images into LR RAW images.
3. Train a RAW-to-RGB SR model on the synthesized HR-LR pairs.

The LR image generation formula is:

$$\mathbf{y} = \mathcal{M}((\mathbf{K}\mathbf{x})_{\downarrow s}) + \mathbf{n}$$

where $\mathbf{x}$ is the latent HR image, $\mathbf{K}$ is the SR blur kernel, $\mathcal{M}$ is the Bayer CFA mosaicking operator, $\mathbf{n}$ is sensor noise, and $s$ is the downsampling factor.

### SR Blur Kernel Calibration (Sec 3.1)

The SR kernel combines the lens PSF with the sensor discretization operator, and is influenced by lens characteristics, sensor size, and the microlens array. The paper directly models the SR kernel in the mosaicked RAW domain via a **display-camera calibration prototype**:

1. **Display random patterns**: Display $B=20$ randomly structured patterns on a calibrated monitor and capture RAW images with the target camera (100-frame burst average per pattern for denoising).
2. **Radiometric and geometric alignment**: Use Gray-code structured patterns to establish dense sensor-display correspondences and calibrate the perspective transform $\mathcal{W}$; use gray/color patches to calibrate the display nonlinear response $\mathcal{D}^{-1}$.
3. **Joint optimization**: Divide the FOV into 128×128 patches and jointly optimize alignment parameters $\mathbf{H}$ and RGB-channel SR kernels $\{\hat{\mathbf{K}}_r, \hat{\mathbf{K}}_g, \hat{\mathbf{K}}_b\}$ per patch using an $\ell_1$ loss with the ADAM optimizer.
4. **No heuristic priors required**: Sufficient LR-HR paired data makes the problem well-posed, eliminating the need to explicitly impose non-negativity, energy conservation, or sparsity constraints.

### Sensor Noise Calibration (Sec 3.2)

A heteroscedastic Gaussian (HG) noise model is adopted: $\mathbf{n}_i \sim \mathcal{N}(0, \beta_{\kappa,c}^1 \mathbf{y}_i + \beta_{\kappa,c}^2)$

- For each camera, calibration is performed separately at 7 ISO levels: $\{50, 100, 200, 400, 800, 1600, 3200\}$.
- The four Bayer CFA color channels ($r, g_1, g_2, b$) are modeled independently.
- A quadratic curve is fit to the calibrated ISO-level parameters to enable interpolation at uncalibrated ISO values.
- Burst sampling over uniform regions of a color chart improves calibration accuracy.

### Training Data Synthesis (Sec 3.3)

A four-step procedure:
1. Rescale linear RGB image pixel values to the target sensor range (accounting for black level/white level).
2. Randomly select RGB blur kernels from the kernel pool, apply them, and downsample to the target SR scale.
3. Apply mosaicking according to the sensor CFA pattern.
4. Invert white-balance gains and add synthetic noise sampled at a randomly drawn ISO level.

### Network Architecture

- RRDBNet architecture is used for RAW-to-RGB SR.
- Input RAW is converted to a 4-channel GBRG image (2×2 block stacking), with an upsampling layer appended at the end.
- Only $\ell_1$ loss is used; GAN and perceptual losses are excluded to avoid hallucination artifacts.

## Key Experimental Results

### Experimental Setup

- **9 smartphone cameras**: Pixel 9 Pro Tele/Main, Pixel 6 Main, S24U Tele 2, S23U Tele 1/Main, S23+ Tele/Main, Mi 11 Main.
- **Two evaluation settings**: camera-specific SR (non-blind) and cross-camera SR (blind, where test camera degradations are not seen during training).
- **Metrics**: PSNR, SSIM (reference-based); MTF50, MTF25 (no-reference, measuring detail recovery).
- **Baselines**: Bicubic, KernelGAN, MANet, Real-ESRGAN, Degradation-Transfer, RAWSR, BSRAW.

### Table 1: Camera-Specific SR Quantitative Results (4× SR)

| Method | S23U Tele 1 PSNR/SSIM | S24U Tele 2 PSNR/SSIM | Pixel 9 Pro Main PSNR/SSIM | Pixel 9 Pro Tele PSNR/SSIM |
|---|---|---|---|---|
| Bicubic | 32.01 / 0.952 | 33.42 / 0.935 | 30.88 / 0.874 | 30.96 / 0.868 |
| KernelGAN | 33.12 / 0.950 | 33.48 / 0.937 | 31.10 / 0.856 | 32.66 / 0.883 |
| MANet | 33.42 / 0.959 | 33.47 / 0.935 | 33.14 / 0.889 | 32.93 / 0.886 |
| Real-ESRGAN | 32.92 / 0.941 | 33.03 / 0.949 | 33.38 / 0.888 | 33.21 / 0.889 |
| Degradation-Transfer | 33.47 / 0.954 | 33.01 / 0.948 | 32.98 / 0.885 | 32.76 / 0.878 |
| **Ours (Camera-specific)** | **33.59 / 0.961** | **33.53 / 0.956** | **33.47 / 0.901** | **33.32 / 0.889** |

The gap is even more pronounced in MTF50: on S24U Tele 2, the proposed method achieves MTF50 = 2.00 (i.e., a 200% improvement in the spatial frequency at which 50% contrast is retained after SR), whereas MANet achieves only 0.29 (29%).

### Table 2: Cross-Camera SR Quantitative Results (test camera degradations not included in training)

| Camera | Method | MTF50 | MTF25 | PSNR | SSIM |
|---|---|---|---|---|---|
| Pixel 6 Main | Real-ESRGAN | 0.38 | 0.50 | 31.75 | 0.884 |
| | BSRAW | 0.97 | 0.82 | 28.79 | 0.713 |
| | RAWSR | 0.96 | 0.90 | 28.99 | 0.790 |
| | **Ours (Cross-camera)** | **1.19** | **0.86** | **32.41** | **0.905** |
| Mi 11 Main | Real-ESRGAN | 0.33 | 0.40 | 35.08 | 0.914 |
| | BSRAW | 0.34 | 0.43 | 33.65 | 0.817 |
| | RAWSR | 0.37 | 0.53 | 33.76 | 0.850 |
| | **Ours (Cross-camera)** | **0.89** | **0.74** | **35.72** | **0.938** |

Advantages are even more pronounced in the cross-camera blind SR setting: on Pixel 6 Main, PSNR improves by **0.66 dB** over Real-ESRGAN and SSIM by **0.021**; MTF50 improves from 0.38 to 1.19 (a 3× improvement).

## Highlights & Insights

- **Calibration-driven vs. generic degradation pools**: This work provides the first systematic demonstration that carefully calibrated device-specific degradation models outperform large randomly sampled generic degradations, with consistent improvements across all metrics.
- **Necessity of RAW-domain modeling**: Modeling degradations in the RAW domain—where scene radiance and sensor response are linearly related—is more accurate than sRGB-domain modeling, reducing the domain gap introduced by nonlinear ISP processing.
- **Cross-device transferability**: Degradation characteristics across different smartphones exhibit structural similarity (verified via t-SNE visualization), and a model trained on calibrated degradations from 7 cameras achieves state-of-the-art performance on unseen devices.
- **MTF evaluation metrics**: MTF50/MTF25, derived from Siemens star targets, are introduced as no-reference evaluation metrics that better reflect detail recovery quality than conventional PSNR/SSIM.
- **Engineering completeness**: A complete technical pipeline is provided—spanning Gray-code alignment, display-sensor radiometric calibration, and kernel optimization—and the calibration data are made publicly available.

## Limitations & Future Work

- **Calibration equipment dependency**: Kernel calibration requires a specialized display-camera capture rig, making the process cumbersome and difficult to scale rapidly to new devices.
- **Scene limitations**: Conclusions apply only to smartphone cameras under favorable lighting conditions; applicability to DSLR cameras or low-light scenarios has not been verified.
- **Spatially invariant assumption**: Although kernels are calibrated patch-by-patch across the FOV, spatial invariance within each patch is still assumed.
- **Network architecture**: Only the RRDBNet architecture is evaluated; it remains unexplored whether stronger architectures such as Transformers could further amplify the gains from improved degradation modeling.
- **Simplified noise model**: The HG noise model does not account for more complex sensor noise components such as row noise and fixed-pattern noise.

## Related Work & Insights

| Direction | Representative Methods | Distinction from This Work |
|---|---|---|
| RAW-domain SR | RAWSR, BSRAW, Zhou et al. | This work calibrates each device precisely rather than sampling from a generic degradation pool. |
| Implicit degradation modeling | Bulat et al., CinCGAN, DSGAN | GANs implicitly learn degradations; this work uses explicit calibration for greater controllability and interpretability. |
| Explicit degradation modeling | Real-ESRGAN, KernelGAN, MANet | Generic parameterized kernels with sRGB-domain operations; this work uses device-specific kernels in the RAW domain. |
| Camera PSF calibration | Degradation-Transfer, Diamond et al. | Only 1× PSF is calibrated, which is insufficient for high-scale SR; this work directly calibrates the 4× SR kernel. |
| Unprocessing | Brooks et al., Graphics2RAW | Unprocessing pipelines with imprecise degradation parameters; this work uses calibrated parameters to generate more realistic RAW images. |

## Rating

- Novelty: ⭐⭐⭐⭐ — Bringing the rigor of camera calibration into SR degradation modeling and systematically demonstrating that "calibration beats random sampling" constitutes a clear methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 9 cameras, 6 baselines, reference and no-reference dual metrics, camera-specific and cross-camera dual settings, and t-SNE degradation distribution visualization; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Technically rigorous with well-documented calibration procedures and detailed supplementary materials.
- Value: ⭐⭐⭐⭐ — Direct engineering value for computational photography SR on smartphones; publicly released calibration data can facilitate future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[CVPR 2026\] BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting](bhcast_unlocking_black_hole_plasma_dynamics_from_a_single_blurry_image_with_long.md)

</div>

<!-- RELATED:END -->
