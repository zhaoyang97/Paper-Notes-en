---
title: >-
  [Paper Note] Learning to See in the Extremely Dark
description: >-
  [Image Generation] This paper proposes a paired-to-paired data synthesis pipeline to construct SIED, a RAW image enhancement dataset for extremely dark scenes (down to 0.0001 lux), and designs a diffusion-model-based framework that achieves high-quality restoration of ultra-low-SNR RAW images via an Adaptive Illumination Correction Module (AICM) and a color consistency loss.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 4b51432d17c72168
---

# Learning to See in the Extremely Dark

| Info | Content |
|------|------|
| Conference | ICCV2025 |
| arXiv | [2506.21132](https://arxiv.org/abs/2506.21132) |
| Code | [JianghaiSCU/SIED](https://github.com/JianghaiSCU/SIED) |
| Area | Image Enhancement / Low-Light RAW Image Restoration |
| Keywords | Extremely dark scenes, RAW image enhancement, diffusion models, data synthesis, illumination correction |

## TL;DR

This paper proposes a paired-to-paired data synthesis pipeline to construct SIED, a RAW image enhancement dataset for extremely dark scenes (down to 0.0001 lux), and designs a diffusion-model-based framework that achieves high-quality restoration of ultra-low-SNR RAW images via an Adaptive Illumination Correction Module (AICM) and a color consistency loss.

## Background & Motivation

### Core Challenges

Low-light RAW image enhancement must simultaneously address global/local contrast enhancement, noise suppression, detail preservation, and color mapping. Existing methods primarily operate under moderate low-light conditions (e.g., 0.03–5.0 lux as in the SID dataset), but when ambient illuminance drops to 0.0001 lux, two fundamental difficulties arise:

**Data scarcity**: Under extremely dark conditions, long-exposure reference images cannot be reliably captured (residual noise and motion blur are introduced).

**Performance bottleneck**: Extremely low SNR causes existing methods to produce severe color distortion, blurred details, and noise amplification.

### Limitations of Prior Work

- **Single-stage methods** (SID, DID, etc.): A single network is tasked with modeling both denoising and RAW-to-sRGB transformation, leading to domain ambiguity.
- **Multi-stage methods** (LDC, MCR, etc.): Although tasks are decoupled, color bias remains under extremely dark conditions.
- **Pre-amplification strategy**: Most methods rely on ground-truth (GT) exposure values for pre-amplification, which are unavailable in practice.
- **Dataset issues**: SID, SDSD, SMID, etc. provide only coarse illuminance ranges and lack precise calibration.

## Method

### I. SIED Dataset Construction: Paired-to-Paired Synthesis Pipeline

Unlike conventional approaches that synthesize low-light images from normal-light images, this paper proposes a "synthesize even darker from low-light" strategy:

**Step 1: Optical Laboratory Calibration**
- Sony α7RIII and Canon EOS R cameras are used in a professional optical laboratory (equipped with a PHOTO-2000μ illuminance meter) to capture standard RAW images at three illuminance levels:
    - 0.01–0.1 lux
    - 0.001–0.01 lux
    - 0.0001–0.001 lux

**Step 2: Real-Scene Paired Capture**
- Low-light RAW and normal-light sRGB pairs are collected in diverse real scenes using a tripod and remote shutter.
- Reference image exposure time is 20–200× that of the low-light image.
- Images are cropped to 3840×2160 resolution; each subset contains 1,680 pairs.

**Step 3: Illumination Alignment**

$$I_{syn} = I_{cap} \times \left(\frac{\text{Expo}(I_{st})}{\text{Expo}(I_{cap})} + \eta\right)$$

Images are converted to YUV space via an ISP pipeline, and $\eta$ is manually adjusted so that the Y-channel illuminance histogram matches the standard data. KL divergence is kept below 0.06.

**Step 4: Noise Addition**
- Gaussian + Poisson noise distributions for Canon and Sony cameras are fitted from optical laboratory measurements.
- A dark-frame database is incorporated to handle non-Poisson–Gaussian noise at extremely low light.
- ISO range: 100–20,000 for 0.01–0.1 and 0.001–0.01 lux; 100–40,000 for 0.0001–0.001 lux.

The final dataset contains 1,500 training pairs + 180 evaluation pairs per illuminance level per camera subset.

### II. Diffusion-Model-Based Enhancement Framework

#### Overall Architecture

1. A RAW encoder $\mathcal{E}_{raw}$ and an sRGB encoder $\mathcal{E}_{rgb}$ map inputs to latent space.
2. AICM applies adaptive illumination correction to the RAW features.
3. The diffusion model uses the corrected RAW features as guidance to reconstruct sRGB features.
4. The sRGB decoder produces the final output image.

#### Adaptive Illumination Correction Module (AICM)

Rather than relying on GT exposure values for pre-amplification, AICM estimates amplification factors directly from low-light RAW features:

- Convolutional embedding → cascaded convolution + adaptive average pooling → per-channel amplification factor $A_{raw} \in \mathbb{R}^{1 \times 1 \times C}$
- An illumination correction loss based on Retinex theory:

$$\mathcal{L}_{icl} = \|\mathbf{L}_{\hat{\mathcal{F}}_{raw}} - \mathbf{L}_{\tilde{\mathcal{F}}_{raw}}\|_1 + \|\mathbf{R}_{\hat{\mathcal{F}}_{raw}} - \mathbf{R}_{\mathcal{F}_{raw}}\|_1$$

This ensures illumination improvement while preserving reflectance consistency.

#### Diffusion-Based RAW-to-sRGB Reconstruction

**Forward diffusion**: Encoded sRGB features $\mathcal{F}_{rgb}$ are progressively noised to Gaussian noise:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t$$

**Reverse diffusion**: Conditioned on corrected RAW features $\hat{\mathcal{F}}_{raw}$, clean sRGB features are recovered from noise:

$$p_\theta(\hat{\mathbf{x}}_{t-1}|\hat{\mathbf{x}}_t, \tilde{\mathbf{x}}) = \mathcal{N}(\hat{\mathbf{x}}_{t-1}; \boldsymbol{\mu}_\theta(\hat{\mathbf{x}}_t, \tilde{\mathbf{x}}, t), \sigma_t^2\mathbf{I})$$

#### Color Consistency Loss

A KL divergence constraint based on color histograms promotes accurate color mapping:

$$\mathcal{L}_{ccl} = \sum_{c \in [0,C)} \mathcal{H}_{\hat{\mathcal{F}}_{rgb}^c} \log\left(\frac{\mathcal{H}_{\hat{\mathcal{F}}_{rgb}^c}}{\mathcal{H}_{\mathcal{F}_{rgb}^c} + \tau}\right)$$

### III. Two-Stage Training

- **Stage 1**: The encoder–decoder and AICM are optimized while the diffusion model is frozen. $\mathcal{L}_{stage1} = \mathcal{L}_{con} + \mathcal{L}_{icl}$
- **Stage 2**: The diffusion model is optimized while all other modules are frozen. $\mathcal{L}_{stage2} = \mathcal{L}_{cdl} + \lambda\mathcal{L}_{ccl}$ ($\lambda=0.1$)

## Key Experimental Results

### Main Results: SIED Dataset (Canon Subset)

| Type | Method | 0.01–0.1 lux PSNR/SSIM/LPIPS | 0.001–0.01 lux | 0.0001–0.001 lux |
|------|------|--------------------------|----------------|-----------------|
| Single-stage | SID | 20.69/0.811/0.428 | 20.34/0.799/0.450 | 19.28/0.764/0.497 |
| Single-stage | SGN | 21.79/0.813/0.421 | 21.07/0.800/0.447 | 19.42/0.762/0.514 |
| Multi-stage | DNF | 24.03/0.813/0.456 | 23.47/0.796/0.486 | 21.63/0.769/0.522 |
| Multi-stage | RAWMamba | 22.63/0.791/0.461 | 21.99/0.782/0.482 | 21.05/0.757/0.521 |
| — | **Ours** | **24.85/0.849/0.360** | **24.02/0.839/0.379** | **22.52/0.811/0.435** |

State-of-the-art performance is achieved across all three illuminance levels and all metrics. At the 0.0001–0.001 lux level, the proposed method outperforms the second-best DNF by +0.89 dB PSNR.

### Ablation Study

| Variant | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Single-stage training | 22.96 (−1.89) | 0.809 (−0.040) | 0.431 (+0.071) |
| w/o AICM | 23.23 (−1.62) | 0.839 (−0.010) | 0.378 (+0.018) |
| Fixed amplification = 100 | 23.74 (−1.11) | 0.841 (−0.008) | 0.373 (+0.013) |
| Fixed amplification = 200 | 23.48 (−1.37) | 0.844 (−0.005) | 0.371 (+0.011) |
| Fixed amplification = 300 | 23.18 (−1.67) | 0.838 (−0.011) | 0.382 (+0.022) |
| w/o color consistency loss | 24.53 (−0.32) | 0.836 (−0.013) | 0.389 (+0.029) |
| **Full method** | **24.85** | **0.849** | **0.360** |

Key findings:
- AICM contributes the most (−1.62 dB); adaptive amplification substantially outperforms any fixed amplification factor.
- Two-stage training is critical (−1.89 dB); unstable encoded features in early training impede diffusion model learning.
- The color consistency loss primarily improves SSIM and LPIPS, enhancing color accuracy.

### Comparison on the SID Dataset

On the public SID dataset (Sony subset), the proposed method also achieves state-of-the-art performance: PSNR 31.20, surpassing DNF and RAWMamba by +0.58 dB each. Furthermore, the method requires no GT exposure values for pre-amplification.

## Highlights & Insights

1. **Data synthesis innovation**: The paired-to-paired strategy is more realistic than conventional normal-to-low-light synthesis; optical laboratory calibration ensures precise illuminance control.
2. **Elegant AICM design**: By eliminating dependence on GT exposure information, the module is more practical for real-world deployment.
3. **Novel application of diffusion models**: The generative capacity and inherent denoising properties of diffusion models are well-suited to RAW enhancement.
4. **Color histogram loss**: Distribution-level constraints replace pixel-level supervision, making them more appropriate for color mapping tasks.
5. **Generalization to real scenes**: Models trained on synthetic data transfer directly to real extremely dark scenes.

## Limitations & Future Work

- Data synthesis relies on costly optical laboratory equipment.
- Only Sony and Canon cameras are supported; cross-camera generalization remains to be validated.
- Diffusion model inference is slow (20-step sampling), limiting real-time applicability.
- Absolute PSNR values under extremely dark conditions remain modest (22.52 dB), reflecting irreversible information loss.

## Related Work & Insights

- **SID** (CVPR 2018): Pioneering work on RAW low-light enhancement, but with a limited illuminance range.
- **DNF** (CVPR 2023): A representative multi-stage method and the primary competitor of this work.
- **Retinex theory**: Provides the design inspiration for the illumination correction loss.
- The proposed data synthesis paradigm can be generalized to dataset construction under other extreme conditions (e.g., underwater, hazy scenes).

## Rating

⭐⭐⭐⭐ — The dataset contribution is outstanding (filling the gap for extremely dark scenes), and the method design is principled and effective; however, experimental validation is conducted primarily on the authors' own dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)

</div>

<!-- RELATED:END -->
