---
title: >-
  [Paper Note] HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching
description: >-
  [CVPR 2026][Image Generation][Fluorescence microscopy dehazing] This paper proposes HazeMatching, a guided conditional flow matching (Guided CFM) framework for microscopy image dehazing. By incorporating degraded observations as conditioning signals in the velocity field, the method achieves high data fidelity and high perceptual quality simultaneously without requiring an explicit degradation operator, while also providing well-calibrated uncertainty estimates.
tags:
  - CVPR 2026
  - Image Generation
  - Fluorescence microscopy dehazing
  - conditional flow matching
  - perception-distortion trade-off
  - posterior sampling
  - calibration analysis
date: 2026-05-08
content_hash: 01bcc65a6a8be163
---

# HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching

**Conference**: CVPR 2026
**arXiv**: [2506.22397](https://arxiv.org/abs/2506.22397)
**Code**: [https://github.com/juglab/HazeMatching](https://github.com/juglab/HazeMatching)
**Area**: Image Generation / Medical Imaging
**Keywords**: Fluorescence microscopy dehazing, conditional flow matching, perception-distortion trade-off, posterior sampling, calibration analysis

## TL;DR
This paper proposes HazeMatching, a guided conditional flow matching (Guided CFM) framework for microscopy image dehazing. By incorporating degraded observations as conditioning signals in the velocity field, the method achieves high data fidelity and high perceptual quality simultaneously without requiring an explicit degradation operator, while also providing well-calibrated uncertainty estimates.

## Background & Motivation

**State of the Field**: In fluorescence microscopy, widefield microscopes are cost-effective and easy to use but collect substantial out-of-focus light, resulting in blurry (hazy) images. Confocal microscopes physically filter out-of-focus light via pinholes to obtain sharp images but are significantly more expensive. Computational dehazing aims to recover confocal-quality images from widefield acquisitions using algorithmic methods. Existing approaches fall into two categories: deterministic point predictors (U-Net/RCAN, trained with MSE loss) and generative posterior models (diffusion models/flow matching).

**Limitations of Prior Work**: Deterministic methods optimize for fidelity (high PSNR) but produce over-smoothed predictions that lose fine structural detail. GAN-based methods yield perceptually realistic results but are prone to hallucinating non-existent structures. Existing CFM methods (e.g., SIFM) require a known degradation operator $m(x_1)$ and noise level $\sigma$, which are unavailable in microscopy where the degradation process is unknown and noise properties (Poisson noise, signal-dependent) vary considerably.

**Root Cause**: The perception-distortion trade-off — optimizing for data fidelity (low MSE) leads to over-smoothed predictions, while optimizing for perceptual quality (low LPIPS/FID) may introduce hallucinated structures. In scientific imaging, fidelity is critical (biological structures must not be hallucinated), yet among methods of comparable fidelity, maximal perceptual quality is desirable.

**Paper Goals**: How to recover both faithful and perceptually realistic results from hazy microscopy images without requiring an explicit degradation operator, while providing reliable uncertainty estimates.

**Starting Point**: Extend the conditional flow matching framework into a guided variant conditioned on degraded observations (hazy images), allowing the velocity field to jointly depend on the current interpolated state and the degraded input, thereby enabling data-driven transport mapping.

**Core Idea**: Explicitly incorporate the degraded observation as an additional conditioning input (via channel concatenation) into the CFM velocity field, achieving guided generative dehazing without any knowledge of the degradation operator.

## Method

### Overall Architecture
HazeMatching is built on the conditional flow matching (CFM) framework. During training: paired data are collected (widefield hazy image $\mathbf{x}_{M_0}$ and confocal clean image $\mathbf{x}_{M_1}$ of the same sample); a linear interpolation path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_{M_1}$ is constructed between Gaussian noise $\mathbf{x}_0 \sim \mathcal{N}(0,I)$ and the clean image $\mathbf{x}_{M_1}$; a neural network is trained to learn the velocity field $v_\theta(t, \mathbf{x}_t, \mathbf{x}_{M_0})$, where the hazy image is concatenated as an additional conditioning channel. During inference: given a new hazy image, starting from random noise, the learned velocity field drives Euler ODE integration to progressively generate a dehazed result. Multiple posterior samples can be obtained by sampling different initial noise realizations.

### Key Designs

1. **Guided Conditional Velocity Field**:

    - Function: Introduces degraded observation guidance into standard CFM to align the generative process with the observation.
    - Mechanism: The standard CFM conditional velocity field $v(t, \mathbf{x}_t | \mathbf{x}_{M_1})$ depends only on the interpolated state and the target image. HazeMatching extends this to $v(t, \mathbf{x}_t, \mathbf{x}_{M_0} | \mathbf{x}_{M_1})$, additionally conditioning on the degraded observation. The corresponding marginal velocity field becomes $v(t, \mathbf{x}_t, \mathbf{x}_{M_0}) = \int v(t, \mathbf{x}_t | \mathbf{x}_{M_1}, \mathbf{x}_{M_0}) p_{M_1}(\mathbf{x}_{M_1} | \mathbf{x}_t, \mathbf{x}_{M_0}) d\mathbf{x}_{M_1}$. The training objective remains predicting $\mathbf{x}_{M_1} - \mathbf{x}_0$, but the velocity network receives $\mathbf{x}_{M_0}$ as an additional input channel.
    - Design Motivation: Unlike SIFM and related methods that require an explicit degradation operator $m(\cdot)$, HazeMatching conveys degradation information implicitly via channel concatenation, making no assumptions about the functional form of the degradation process. This enables direct application to real microscopy data where the degradation is unknown.

2. **Posterior Sampling and MMSE Estimation**:

    - Function: Generates diverse dehazing predictions and provides uncertainty estimates.
    - Mechanism: At inference time, different initial noise realizations $\mathbf{x}_0 \sim \mathcal{N}(0,I)$ are sampled and integrated via ODE to yield individual posterior samples. Repeating this procedure (50 samples in the paper) enables: (1) averaging multiple samples to obtain an MMSE estimate with improved fidelity; (2) computing pixel-wise variance maps as uncertainty estimates. An Euler integrator with $T=20$ steps is used.
    - Design Motivation: MMSE estimation reduces stochastic variability across samples via averaging, yielding higher fidelity than any individual sample. Variance maps allow biologists to identify regions of unreliable prediction, which is critical in scientific imaging.

3. **Calibration Analysis Framework**:

    - Function: Verifies whether predicted uncertainty aligns with true prediction error.
    - Mechanism: Pixels are grouped into bins; per-bin RMSE (true error) and RMV (square root of predicted variance) are computed. A well-calibrated model exhibits an RMSE vs. RMV curve close to $y=x$. A linear calibration factor (scaling + offset) is optionally learned to further align the two. Proximity to the diagonal indicates model reliability.
    - Design Motivation: Generative models are frequently questioned regarding whether generated structures are real. Calibration analysis provides quantitative evidence that the variability among HazeMatching's posterior samples genuinely reflects the magnitude of true prediction error.

### Loss & Training
The training loss follows the standard CFM regression objective: $\mathcal{L} = \|v_\theta(t, \mathbf{x}_t, \mathbf{x}_{M_0}) - (\mathbf{x}_{M_1} - \mathbf{x}_0)\|^2$. A U-Net serves as the backbone. Training patch sizes are 64×64 or 128×128, with $T=20$ integration steps. The torchCFM library is used for interpolation and ODE integration. Full-resolution images are processed at evaluation time using inner tiling with 50% overlap.

## Key Experimental Results

### Main Results
HazeMatching is evaluated against 12 baselines across 5 datasets. On the PSNR vs. LPIPS/FID trade-off plot, it consistently occupies the optimal region (lower-left corner), simultaneously achieving high fidelity and high perceptual quality.

| Method Type | Representative Methods | Characteristics |
|-------------|----------------------|-----------------|
| Point predictors | U-Net, MIMO-UNet, RCAN, Restormer | High PSNR but over-smoothed (high LPIPS/FID) |
| GAN-based | ESRGAN | Good perceptual quality but poor fidelity; hallucinated structures |
| Iterative methods | InDI20, SIFM | Require known degradation operator or noise level |
| **HazeMatching** | **Ours** | **Highest fidelity among all posterior models, approaching the best deterministic methods, while offering significantly superior perceptual quality** |

### Calibration Experiment

| Dataset | Calibration Factor (scaling) | Offset | Calibration Quality |
|---------|------------------------------|--------|---------------------|
| Zebrafish | ~1.0 | ~0.0 | Near-diagonal; well calibrated |
| Organoids1 | ~1.0 | ~0.0 | Well calibrated |
| Organoids2 | ~1.0 | ~0.0 | Well calibrated |

### Key Findings
- HazeMatching achieves the best fidelity-perception balance across all 5 datasets; no other method maintains strong performance on both metrics simultaneously across all datasets.
- Among posterior models, it achieves the highest fidelity (MMSE-PSNR approaching the best deterministic baseline, MIMO-UNet) while substantially outperforming deterministic methods in perceptual quality.
- The model is inherently well calibrated (no separate calibration module required), with calibration factors close to 1.0.
- The number of integration steps $T$ provides a controllable knob for the perception-fidelity trade-off: more steps tend toward higher perceptual quality.
- No explicit degradation operator is required, enabling direct application to real microscopy data.

## Highlights & Insights
- Introducing observational conditioning via simple channel concatenation into the CFM velocity field is an elegant design — it requires no modification to the CFM training framework, only an additional input channel, yet achieves guided generation. This paradigm is directly transferable to other image restoration tasks.
- Calibration analysis is an angle largely overlooked in image restoration literature but critical in scientific imaging. HazeMatching's inherently good calibration properties provide a principled basis of trust for biologists.
- The practical guidance on posterior sampling is valuable: MMSE estimation is suited for scenarios requiring high fidelity; variance maps identify uncertain regions; in high-uncertainty regions, the MMSE estimate should be preferred over individual samples.

## Limitations & Future Work
- Paired training data (widefield and confocal images of the same sample) are required, which limits applicability.
- Generating MMSE estimates requires multiple forward passes (50 samples × 20 integration steps), resulting in high inference cost.
- Training and evaluation image sizes are relatively small (1024×1024); adaptation may be needed for larger field-of-view images.
- Validation is limited to the microscopy dehazing task and has not been extended to other inverse problems (e.g., super-resolution).
- The training set is small (15 training images), which may limit generalization to complex samples.

## Related Work & Insights
- **vs. SIFM**: SIFM requires an explicit degradation operator $m(x_1)$ and noise level $\sigma$, which are difficult to obtain for real microscopy data. HazeMatching implicitly conveys degradation information via channel concatenation, requiring no prior knowledge.
- **vs. PMRF**: PMRF requires an additional MMSE estimator and carefully tuned noise parameter $\sigma_s$. HazeMatching achieves both high fidelity and posterior sampling within a single unified framework.
- **vs. InDI**: InDI's iterative denoising also generates diverse predictions, but its fidelity-perception trade-off is less consistent than HazeMatching's on microscopy data.
- The guided CFM framework proposed here is directly applicable to other microscopy inverse problems, including computational super-resolution and deconvolution.

## Rating
- Novelty: ⭐⭐⭐⭐ The guided CFM extension is novel and elegant, though the core technique (channel-concatenation conditioning) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, 12 baselines, calibration analysis, and multi-dimensional metrics constitute a very comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, motivation is clearly articulated, and the discussion of the perception-distortion trade-off is thorough.
- Value: ⭐⭐⭐⭐ The method offers direct practical value to the microscopy imaging community; the calibration analysis provides a methodological reference for applying generative models in scientific imaging.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](../../ICLR2026/image_generation/flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[NeurIPS 2025\] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](../../NeurIPS2025/image_generation/leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[NeurIPS 2025\] Improving Posterior Inference of Galaxy Properties with Image-Based Conditional Flow Matching](../../NeurIPS2025/image_generation/improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)

<!-- RELATED:END -->
