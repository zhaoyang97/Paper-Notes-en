---
title: >-
  [Paper Note] BlurDM: A Blur Diffusion Model for Image Deblurring
description: >-
  [NeurIPS 2025][Image Generation][image deblurring] BlurDM integrates the physical formation process of motion blur (progressive blur accumulation due to continuous exposure) into a diffusion model via a dual forward process (simultaneous noise addition and blurring) and a dual denoising-deblurring reverse process. It serves as a latent-space prior generator that consistently enhances four deblurring methods across four datasets, achieving an average gain of +0.31 dB on GoPro…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "image deblurring"
  - "blur diffusion"
  - "dual diffusion"
  - "motion blur"
  - "prior generation"
date: 2026-05-08
content_hash: fa76a9863af93b32
---

# BlurDM: A Blur Diffusion Model for Image Deblurring

**Conference**: NeurIPS 2025
**arXiv**: [2512.03979](https://arxiv.org/abs/2512.03979)  
**Code**: [https://jin-ting-he.github.io/BlurDM/](https://jin-ting-he.github.io/BlurDM/)  
**Area**: Image Generation
**Keywords**: image deblurring, blur diffusion, dual diffusion, motion blur, prior generation

## TL;DR
BlurDM integrates the physical formation process of motion blur (progressive blur accumulation due to continuous exposure) into a diffusion model via a dual forward process (simultaneous noise addition and blurring) and a dual denoising-deblurring reverse process. It serves as a latent-space prior generator that consistently enhances four deblurring methods across four datasets, achieving an average gain of +0.31 dB on GoPro and +0.78 dB on RealBlur-J, while adding only ~4 GFLOPs and ~9 ms.

## Background & Motivation
**Background**: Deep learning-based deblurring methods (CNN/Transformer) are constrained by regression losses, producing overly smooth results. Diffusion models generate rich details, but their standard noise-based forward process is physically mismatched with the motion blur formation process.

**Limitations of Prior Work**: Motion blur arises from the **structured, directional** accumulation of continuous exposure — $B = \frac{1}{\alpha_T}\int_0^{\alpha_T} H(\tau)d\tau$ — rather than the isotropic Gaussian noise perturbation of standard diffusion. Directly applying DDPM as a deblurring prior yields only +0.13 dB, which is nearly ineffective.

**Key Challenge**: A fundamental mismatch exists between the standard diffusion process (adding Gaussian noise) and the blur formation process (adding directional blur) — noise is stochastic while blur is structured.

**Goal**: Design a physically grounded diffusion process whose forward pass mimics blur formation and whose reverse pass naturally performs deblurring.

**Key Insight**: Decompose the blur formulation into a cumulative form — $I_t = \frac{\alpha_{t-1}}{\alpha_t}I_{t-1} + \frac{1}{\alpha_t}e_t + \beta_t\epsilon_t$ — where the first two terms represent progressive blurring and the last term represents noise, naturally yielding a dual diffusion process.

**Core Idea**: Design a diffusion model whose forward process simultaneously adds noise and blur, and whose reverse process simultaneously performs denoising and deblurring, operating in latent space as a general-purpose prior to enhance arbitrary deblurring networks.

## Method

### Overall Architecture
The training pipeline consists of three stages: (1) pre-train the Sharp Encoder, Prior Fusion Module (PFM), and deblurring network using GT sharp images to provide an "upper-bound" prior; (2) train the Blur Encoder and BlurDM to recover sharp priors from blurry latent representations; (3) joint optimization — inject BlurDM-generated priors into the deblurring network via PFM for end-to-end fine-tuning.

### Key Designs

1. **Dual Forward Diffusion Process**:

    - Function: Simultaneously adds Gaussian noise and blur residuals to the image.
    - Core formulation: $I_t = \frac{\alpha_{t-1}}{\alpha_t}I_{t-1} + \frac{1}{\alpha_t}e_t + \beta_t\epsilon_t$
    - Here $e_t = \int_{\alpha_{t-1}}^{\alpha_t} H(\tau)d\tau$ denotes the accumulated blur residual over the time interval $[\alpha_{t-1}, \alpha_t]$.
    - Terminal state: $q(I_T|I_0, e_{1:T}) = \mathcal{N}(I_T; \frac{\alpha_0}{\alpha_T}I_0 + \frac{1}{\alpha_T}\sum e_t, \bar{\beta}_T^2\mathbf{I})$
    - Physical interpretation: Simulates the continuous exposure process from shutter open to shutter close.

2. **Dual Denoising-Deblurring Reverse Process**:

    - Function: Learns two estimators — a blur residual estimator $e^\theta$ and a noise estimator $\epsilon^\theta$ — to simultaneously perform deblurring and denoising.
    - Reverse step: $I_{t-1} = \frac{\alpha_t}{\alpha_{t-1}}I_t - \frac{1}{\alpha_{t-1}}e^\theta(I_t,t,B) - (\frac{\alpha_t\bar{\beta}_t}{\alpha_{t-1}} - \bar{\beta}_{t-1})\epsilon^\theta(I_t,t,B)$
    - Conditioning: Conditioned on the blurry image $B$ to guide the deblurring direction.

3. **Latent BlurDM Architecture**:

    - Function: Runs BlurDM in latent space as a flexible prior generator.
    - Stage 1: Pre-trains the Sharp Encoder to extract GT sharp priors $Z^S$; PFM modulates decoder features via affine parameters: $F_i' = Z^{S,\alpha_i} \times F_i + Z^{S,\beta_i}$
    - Stage 2: BlurDM recovers $Z^S$ (sharp prior) from $Z^B$ (blurry latent + noise) in latent space, with loss $\mathcal{L}_{prior} = \|Z_0^B - Z^S\|_1$
    - Stage 3: Joint optimization of BlurDM + PFM + deblurring network.
    - Design Motivation: The three-stage design ensures BlurDM learns a meaningful prior; training only in Stage 3 jointly yields substantially worse results.

### Computational Overhead
BlurDM adds only ~4.16 GFLOPs (<8% overhead), ~3.33M parameters, and ~9 ms inference time. $T=5$ steps is the optimal number of iterations.

## Key Experimental Results

### Main Results
BlurDM as a plug-and-play module improves PSNR across four deblurring methods on four datasets:

| Dataset | MIMO-UNet | Stripformer | FFTformer | LoFormer | Avg. |
|--------|-----------|-------------|-----------|---------|------|
| GoPro | +0.49 | +0.44 | +0.13 | +0.16 | +0.31 |
| HIDE | +0.73 | +0.33 | +0.14 | +0.09 | +0.32 |
| RealBlur-J | +0.54 | +1.05 | +0.30 | +1.24 | +0.78 |
| RealBlur-R | +0.60 | +1.16 | +0.44 | +0.56 | +0.69 |

### Ablation Study

| Configuration | GoPro PSNR↑ |
|------|-------------|
| Baseline (no prior) | 31.78 |
| + DDPM prior | 31.91 (+0.13) |
| + RDDM residual diffusion | 32.03 (+0.25) |
| + **BlurDM** | **32.28 (+0.50)** |

Necessity of three-stage training:

| Configuration | PSNR |
|------|------|
| Stage 3 only | 31.80 |
| Stage 1+2 | 32.01 |
| Stage 1+3 | 31.95 |
| **Stage 1+2+3** | **32.28** |
| Oracle (GT prior) | 32.69 (upper bound) |

### Key Findings
- **Standard DDPM prior is nearly ineffective** (+0.13 dB only), validating the mismatch between noise diffusion and blur physics.
- BlurDM outperforms DDPM by +0.37 dB and RDDM (residual diffusion) by +0.25 dB — explicit blur residual modeling is more effective than implicit residual modeling.
- **Real-world blur datasets benefit more**: RealBlur-J averages +0.78 dB vs. +0.31 dB on GoPro — real blur relies more on physical priors.
- $T=5$ is optimal; gains diminish for $T \geq 6$.
- All three training stages are indispensable — Stage 2 prior pre-training and Stage 3 joint fine-tuning are both necessary.

## Highlights & Insights
- **Natural mapping from blur physics to diffusion process**: Continuous exposure → progressive blurring ≈ diffusion forward process. This methodology of deriving mathematical formulations from physical processes is worth drawing upon.
- **Dual estimator design**: Simultaneously estimating blur residuals and noise separates structured degradation from stochastic degradation, achieving higher precision than a single estimator.
- **Architecture-agnostic prior generator**: Rather than replacing existing deblurring methods, BlurDM serves as a universal prior enhancer — all four architectures (CNN/Transformer) benefit consistently.
- **Necessity of three-stage training**: Direct joint training performs poorly (31.80 vs. 32.28); learning the prior separately before joint optimization is critical.

## Limitations & Future Work
- Designed specifically for motion blur — defocus blur is a depth-dependent, non-temporal accumulation process and is not applicable.
- The blur accumulation model is an approximation and may be inaccurate for non-standard motion blur (e.g., rotational motion, non-rigid body motion).
- The stochasticity of diffusion models may affect content fidelity.
- No comparison with recent flow-matching-based methods.

## Related Work & Insights
- **vs. Standard diffusion-based deblurring (DvSR, DiffIR)**: These methods use standard noise diffusion without exploiting blur physics. BlurDM achieves better priors through physical modeling.
- **vs. RDDM (residual diffusion)**: RDDM implicitly models residuals, while BlurDM explicitly models blur accumulation, yielding better performance (+0.25 dB).
- **Implications as a general framework**: Similar ideas can be applied to diffusion modeling of other physical degradation processes, such as compression artifacts and rain streak degradation.

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating blur physics into the diffusion process is an elegant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 methods × 4 datasets + detailed ablations + three-stage analysis.
- Writing Quality: ⭐⭐⭐⭐ Physical motivation is clear and derivations are complete.
- Value: ⭐⭐⭐⭐ A plug-and-play universal deblurring enhancement tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](../../ICCV2025/image_generation/learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[NeurIPS 2025\] ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](argenseg_image_segmentation_with_autoregressive_image_generation_model.md)
- [\[NeurIPS 2025\] BADiff: Bandwidth Adaptive Diffusion Model](badiff_bandwidth_adaptive_diffusion_model.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)

</div>

<!-- RELATED:END -->
