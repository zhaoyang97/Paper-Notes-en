---
title: >-
  [Paper Note] Elucidating the SNR-t Bias of Diffusion Probabilistic Models
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] This paper reveals the pervasive SNR-t bias in diffusion models (the mismatch between the Signal-to-Noise Ratio of samples in the reverse process and their timestamps) and proposes Differential Correction in Wavelet domain (DCW). DCW is a training-free, plug-and-play method that enhances the generation quality across various diffusion models.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Models
  - SNR-t Bias
  - Differential Correction
  - Wavelet Domain
  - Training-free
date: 2026-05-08
content_hash: 8e24d6ad659ae902
---

# Elucidating the SNR-t Bias of Diffusion Probabilistic Models

**Conference**: CVPR 2026  
**arXiv**: [2604.16044](https://arxiv.org/abs/2604.16044)  
**Code**: [https://github.com/AMAP-ML/DCW](https://github.com/AMAP-ML/DCW)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, SNR-t Bias, Differential Correction, Wavelet Domain, Training-free

## TL;DR

This paper reveals the pervasive SNR-t bias in diffusion models (the mismatch between the Signal-to-Noise Ratio of samples in the reverse process and their timestamps) and proposes Differential Correction in Wavelet domain (DCW). DCW is a training-free, plug-and-play method that enhances the generation quality across various diffusion models.

## Background & Motivation

**State of the Field**: Diffusion Probabilistic Models (DPMs) have achieved great success in generation tasks such as image, audio, and video. During training, models strictly bind noisy samples to timestamps: $X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t$, where the signal-to-noise ratio $\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$ is entirely determined by the timestep $t$.

**Limitations of Prior Work**: During inference, due to the accumulation of network prediction errors and numerical solver discretization errors, reverse denoising trajectories inevitably deviate from the ideal path. This leads to a mismatch between the actual SNR of the predicted sample $\hat{x}_t$ and the SNR corresponding to the preset timestep $t$—defined here as SNR-t bias.

**Root Cause**: While SNR and timestamps are strictly coupled during training, this correspondence is broken during inference. When the network receives samples with mismatched SNRs, it produces significant prediction bias: samples with lower SNR lead to over-prediction of noise, while samples with higher SNR lead to under-prediction. Experiments confirm that reverse process samples consistently exhibit lower SNRs than forward process samples.

**Paper Goals**: (1) Provide a systematic empirical and theoretical proof of SNR-t bias; (2) Design a training-free correction method to mitigate this bias.

**Starting Point**: Utilizing the reconstructed sample $x^0_\theta$ generated at each step of the reverse denoising process. The differential signal between $x^0_\theta$ and the predicted sample $\hat{x}_{t-1}$ contains gradient information to push shifted samples toward the ideal trajectory.

**Core Idea**: Introducing differential correction into the wavelet domain to correct different frequency components separately, using dynamic weights designed based on the "low-frequency first, high-frequency later" denoising characteristic of diffusion models.

## Method

### Overall Architecture

DCW is embedded into each denoising step as a plug-and-play inference module. After each denoising step is completed: (1) Reconstructed sample $x^0_\theta$ and predicted sample $x_{t-1}$ are mapped to the wavelet domain via DWT; (2) Differential signals are calculated for low-frequency (LL) and high-frequency (LH, HL, HH) components, applying dynamic weighted correction; (3) The result is mapped back to pixel space via iDWT.

### Key Designs

1.  **Theoretical Proof of SNR-t Bias**:
    - **Function**: Provide a rigorous mathematical foundation for the bias phenomenon.
    - **Mechanism**: Assuming the reconstruction model $x^0_\theta(\hat{x}_t, t) = \gamma_t x_0 + \phi_t \epsilon_t$ ($0 < \gamma_t \leq 1$), the actual SNR of reverse process samples is derived as $\text{SNR}(t) = \hat{\gamma}_t^2 \bar{\alpha}_t / (1-\bar{\alpha}_t + (\frac{\sqrt{\bar{\alpha}_t}\beta_{t+1}}{1-\bar{\alpha}_{t+1}}\phi_{t+1})^2)$. Since $\hat{\gamma}_t \leq 1$ and the denominator contains an extra positive term, the reverse SNR is always lower than the forward SNR—theoretically proving the inevitability of the bias.
    - **Design Motivation**: The prior assumption $x^0_\theta = x_0 + \phi_t \epsilon_t$ contradicts the Tweedie formula and variance identity (it would lead to $\mathbb{E}[\|x^0_\theta\|^2] > \mathbb{E}[\|x_0\|^2]$), hence the more accurate $\gamma_t < 1$ assumption.

2.  **Pixel-space Differential Correction**:
    - **Function**: Use intrinsic information from the denoising process to guide shifted samples back to the ideal trajectory.
    - **Mechanism**: The differential signal $\hat{x}_{t-1} - x^0_\theta(\hat{x}_t, t)$ contains directional information pointing toward the ideal $x_{t-1}$. The correction formula is $\hat{x}_{t-1} = \hat{x}_{t-1} + \lambda_t (\hat{x}_{t-1} - x^0_\theta(\hat{x}_t, t))$, where $\lambda_t$ is the guidance coefficient. Intuitively, the differential signal pushes the predicted sample toward the noise direction (increasing SNR), mitigating the low SNR issue in the reverse process.
    - **Design Motivation**: The differential signal is a byproduct of the denoising process, requiring no extra computation. Correcting $\hat{x}_{t-1}$ is more efficient and effective than correcting $\hat{x}_t$.

3.  **Wavelet Domain Differential Correction (DCW)**:
    - **Function**: Apply differentiated correction to different frequency components.
    - **Mechanism**: DWT decomposes samples into four frequency components: LL, LH, HL, and HH. Differential correction is applied independently to each component: $\hat{x}^f_{t-1} = \hat{x}^f_{t-1} + \lambda^f_t (\hat{x}^f_{t-1} - x^{0,f}_\theta)$, where $f \in \{ll, lh, hl, hh\}$. The dynamic weights $\lambda^f_t$ adapt based on the denoising stage—prioritizing low-frequency correction (global structure) early on, and high-frequency correction (textures) later.
    - **Design Motivation**: The characteristic where diffusion models reconstruct low frequencies before high frequencies means different stages require correction for different components. Uniform pixel-space correction cannot distinguish these frequency-specific needs.

### Loss & Training

Entirely training-free. DCW is embedded in the inference process as a plug-and-play module without modifying model weights. Computational overhead is negligible, consisting only of DWT/iDWT and differential operations.

## Key Experimental Results

### Main Results

| Base Model | Original FID | + DCW FID | Gain |
| :--- | :--- | :--- | :--- |
| IDDPM | 8.45 | 6.72 | -1.73 |
| ADM | 4.59 | 3.97 | -0.62 |
| DDIM (50 steps) | 8.72 | 7.31 | -1.41 |
| EDM | 1.97 | 1.79 | -0.18 |
| FLUX | Improved | Improved | Significant |

### Ablation Study

| Configuration | FID Improvement |
| :--- | :--- |
| Pixel-space Correction (DC) | Moderate |
| Wavelet Domain Correction (DCW) | Best |
| Low-frequency only | Partial |
| High-frequency only | Partial |
| Dynamic vs Fixed weights | Dynamic is better |

### Key Findings

- DCW is effective across 8 different diffusion models (IDDPM, ADM, DDIM, A-DPM, EA-DPM, EDM, PFGM++, FLUX), proving the universality of SNR-t bias.
- It can be stacked with exposure bias correction models for additional gains, suggesting SNR-t bias is more fundamental than exposure bias.
- Consistently effective across datasets of different resolutions (CIFAR-10, ImageNet, etc.).
- Wavelet domain correction outperforms pixel-space correction, validating the necessity of frequency-separated correction.
- Computational overhead is negligible.

## Highlights & Insights

- **Reveals a fundamental problem**: SNR-t bias is an inherent issue for all DPMs and is more fundamental than exposure bias. The theoretical derivation of $\gamma_t < 1$ elegantly explains its inevitability.
- **Clever utilization of differential signals**: The byproduct of the denoising process naturally contains correction direction information, requiring no additional networks or searching.
- **Plug-and-play practicality**: Zero training cost and near-zero inference overhead; it can be directly applied to any DPM, including cutting-edge models like FLUX.

## Limitations & Future Work

- Currently, $\lambda_t$ needs to be adjusted for different models; automated setting strategies remain to be researched.
- Theoretical analysis is based on Gaussian assumptions; there may be a gap regarding actual data distribution shifts.
- Effectiveness in consistency models with extremely few steps (1-2 steps) needs further verification.
- Combination strategies with other improvement methods could be further explored.

## Related Work & Insights

- **vs ADM-ES / TS-DPM**: These works study exposure bias (differences between samples). SNR-t bias focuses on the mismatch between samples and timestamps, which is a lower-level problem. DCW can be used alongside them.
- **vs ADM-IP**: ADM-IP mitigates bias by re-perturbing training data, which requires retraining. DCW is training-free and plug-and-play.
- **vs FreeU**: FreeU reweights frequency components within the U-Net. DCW dynamically corrects frequency components during the denoising process. They operate at different levels.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal and prove SNR-t bias with deep theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across 8 models, multiple resolutions, and combined with other methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete and elegant logical chain from phenomenon to theory to method.
- Value: ⭐⭐⭐⭐⭐ Wide impact by revealing a fundamental issue and providing a practical solution.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[AAAI 2026\] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](../../ICCV2025/image_generation/smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)

<!-- RELATED:END -->
