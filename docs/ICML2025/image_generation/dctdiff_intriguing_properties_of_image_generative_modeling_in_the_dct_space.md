---
title: >-
  [Paper Note] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space
description: >-
  [ICML2025][Image Generation][Diffusion Models] Proposes DCTdiff, which performs end-to-end diffusion image generation directly in the Discrete Cosine Transform (DCT) frequency domain for the first time, seamlessly scaling to $512 \times 512$ resolution without a VAE and outperforming pixel-space diffusion models in both generation quality and training efficiency.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "DCT"
  - "Frequency-domain Modeling"
  - "Spectral Autoregression"
date: 2026-05-08
content_hash: 2d56e1089789e942
---

# DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space

**Conference**: ICML2025  
**arXiv**: [2412.15032](https://arxiv.org/abs/2412.15032)  
**Code**: [forever208/DCTdiff](https://github.com/forever208/DCTdiff)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, DCT, Frequency-domain Modeling, Image Generation, Spectral Autoregression

## TL;DR
Proposes DCTdiff, which performs end-to-end diffusion image generation directly in the Discrete Cosine Transform (DCT) frequency domain for the first time, seamlessly scaling to $512 \times 512$ resolution without a VAE and outperforming pixel-space diffusion models in both generation quality and training efficiency.

## Background & Motivation
- **Redundancy of Pixel-Space Modeling**: Traditional diffusion models directly model the high-dimensional RGB pixel space, which is computationally expensive and difficult to scale to high resolutions (e.g., UViT trained in the $256 \times 256$ pixel space of FFHQ suffers from a high FID of 120).
- **Costs of Latent-Space Methods**: Although Latent Diffusion Models (LDMs) reduce dimensionality via SD-VAE, the VAE itself requires ~9 million images for training, introduces extra computational overhead, and incurs a rapid increase in GFLOPs at high resolutions.
- **Natural Advantages of Frequency-Domain Compression**: JPEG encoding demonstrates that DCT can concentrate image energy into a few low-frequency coefficients, achieving significant near-lossless compression completely training-free. This work extends this concept to diffusion-based generative modeling.

## Method

### Overall Architecture
RGB image → YCbCr color space conversion → $2\times$ chroma subsampling → Block-wise 2D-DCT → Zigzag flattening + high-frequency truncation → Frequency tokenization ($4\text{Y}+1\text{Cb}+1\text{Cr}$) → ViT diffusion modeling → Inverse DCT reconstruction.

### 1. Color Space Conversion and Chroma Subsampling
Converts RGB to YCbCr and performs $2\times$ downsampling on the Cb/Cr channels, reducing the signal dimension from $3hw$ to $1.5hw$ ($2\times$ compression) by exploiting the human visual system's higher sensitivity to luminance than chrominance.

### 2. Block-wise DCT and High-Frequency Truncation
Executes Type-II DCT on $B \times B$ blocks of Y/Cb/Cr channels:

$$D(u,v) = \alpha(u)\alpha(v) \sum_{x=0}^{B-1}\sum_{y=0}^{B-1} A(x,y) \cos\!\left[\frac{(2x+1)u\pi}{2B}\right] \cos\!\left[\frac{(2y+1)v\pi}{2B}\right]$$

Flattens the coefficients in Zigzag order and truncates $m^*$ high-frequency coefficients based on the following criterion:

$$m^* = \arg\max_m \{m : \text{rFID}(P_\text{data}, P_\text{dct\_data}(m)) < \gamma\}, \quad \gamma = 0.5$$

Achieves ~$4\times$ near-lossless compression at $256 \times 256$ resolution, and ~$7.1\times$ at $512 \times 512$ resolution.

### 3. Frequency Tokenization
Each token is formed by concatenating 4 Y-blocks + 1 Cb-block + 1 Cr-block, yielding a dimension of $6(B^2 - m^*)$, which relates to the ViT patch size as $P = 2B$.

### 4. Entropy-Consistent Scaling
The upper and lower bounds of DCT coefficients across frequencies can differ by up to two orders of magnitude. The authors propose using the $\tau = 98.25$-th percentile of the DC component ($D(0,0)$) as a unified scaling factor $\eta$:

$$\eta = \max(|P_\tau|, |P_{100-\tau}|), \quad \bar{\mathbf{x}}_0 = \bar{\mathbf{x}}_0 / \eta$$

This preserves the distribution shape of each frequency, outperforming "Naive Scaling" where each frequency is scaled independently.

### 5. SNR Scaling
DCT concentrates energy in low frequencies, making high-frequency signals succumb to noise faster during the forward diffusion process. To compensate for the SNR drop caused by increasing the block size $B$ ($\eta$ doubles as $B$ doubles), the default noise schedule is adjusted via SNR scaling.

### 6. Entropy-Based Frequency Reweighting (EBFR)
Introduces a frequency entropy weight vector $\mathbf{H}(B)$ into the training loss, assigning higher weights to low-frequency (high-entropy) signals:

$$\mathcal{L}_\text{EBFR}(\theta) = \mathbb{E}_t \lambda(t) \mathbb{E}_{\bar{\mathbf{x}}_0, \bar{\mathbf{x}}_t} \left[\mathbf{H}(B) \| \mathbf{s}_\theta(\bar{\mathbf{x}}_t, t) - \nabla_{\bar{\mathbf{x}}_t} \log P_{0t}(\bar{\mathbf{x}}_t | \bar{\mathbf{x}}_0) \|^2_2 \right]$$

## Key Experimental Results

### FID-50k Comparison under the UViT Framework (DPM-Solver)

| Dataset | NFE | UViT (pixel) | DCTdiff |
|--------|-----|-------------|---------|
| CIFAR-10 | 100 | 5.80 | 5.28 |
| CelebA 64 | 100 | 1.57 | 1.71 |
| ImageNet 64 | 100 | 10.07 | 9.73 |
| FFHQ 128 | 100 | 9.18 | **6.25** |
| FFHQ 128 | 50 | 9.20 | **6.28** |

### High-Resolution No-VAE vs With-VAE (DPM-Solver, NFE=100)

| Dataset | UViT (latent+SD-VAE) | DCTdiff (No-VAE) |
|--------|---------------------|-----------------|
| FFHQ 256 | **4.26** | 5.08 |
| FFHQ 512 | 10.89 | **7.07** |
| AFHQ 512 | 10.86 | **8.76** |

### Training Efficiency Comparison

| Dataset | Model | Parameters | GFLOPs | Convergence Steps |
|--------|------|--------|--------|----------|
| FFHQ 128 | UViT | 44M | 11 | 750k |
| FFHQ 128 | DCTdiff | 44M | 11 | **300k (2.5$\times$ Speedup)** |
| AFHQ 512 | UViT (latent) | 131M+84M | 575 | 225k |
| AFHQ 512 | DCTdiff | 131M | 133 | 225k (**Only 1/4 Training Cost**) |

### FID-50k under the DiT Framework (DDPM sampler, NFE=100)

| Dataset | DiT | DCTdiff |
|--------|-----|---------|
| CelebA 64 | 5.11 | **3.84** |
| FFHQ 128 | 12.81 | **11.16** |

## Highlights & Insights
1. **Theoretical Contribution to Frequency-Domain Diffusion**: Proves that "image diffusion is equivalent to spectral autoregression." Since the power spectral density of natural images follows a power law $|\hat{x}_0(\omega)|^2 = K|\omega|^{-\alpha}$, the forward diffusion process disrupts high frequencies before low frequencies, while the reverse generation process restores low frequencies before high frequencies. This shares the same spirit as the coarse-to-fine generation in VAR.
2. **High-Resolution Generation Without VAE**: As a training-free, deterministic transform, DCT outperforms SD-VAE latent diffusion on $512 \times 512$ resolution, completely avoiding the training overhead and domain adaptation issues of VAEs.
3. **DCT Upsampling Theorem**: Proves the approximate relationship between low-resolution and high-resolution DCT coefficients as $\bar{D}(k,l) \approx \frac{1}{2}\cos(\frac{k\pi}{4B})\cos(\frac{l\pi}{4B}) D(k,l)$, where DCT upsampling (FID 9.79) outperforms bicubic interpolation (FID 12.53).
4. **Plug-and-Play**: DCTdiff does not alter the Transformer architecture, allowing direct application to UViT/DiT, and is highly effective across multiple samplers (DDIM, DPM-Solver, DDPM).

## Limitations & Future Work
- Not yet evaluated on conditional generation tasks such as text-to-image, remaining limited to unconditional/class-conditional generation.
- Marginal improvement on low-resolution datasets like CelebA 64, even slightly underperforming UViT under DPM-Solver.
- The DCT block size $B$ needs to be manually selected based on the resolution, lacking an adaptive selection mechanism.
- The percentile threshold $\tau = 98.25$ in Entropy-Consistent Scaling is empirical, and its robustness to new domains warrants further validation.
- The high-frequency truncation criterion $\gamma = 0.5$ may be too aggressive for texture-rich scenarios (e.g., medical imaging).
- Comparison with state-of-the-art Flow Matching / Rectified Flow methods was not conducted.

## Related Work & Insights
- **DCTransformer** (Nash et al., 2021): Implements autoregressive generation in the DCT space but does not adopt the diffusion paradigm.
- **VAR** (Tian et al., 2024): Performs coarse-to-fine autoregressive generation, which this work explains theoretically from a spectral perspective.
- **Latent Diffusion** (Rombach et al., 2022): Uses the SD-VAE compression scheme, which this work demonstrates can be replaced by training-free DCT.
- **EDM** (Karras et al., 2022): Decouples the design space of diffusion models, which this work extends into the frequency-domain design space.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to systematically perform diffusion generation in the DCT space, with a strong integration of theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across multiple frameworks (UViT/DiT), multiple datasets, and multiple samplers, supported by thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous theoretical proofs, and intuitive flowcharts.
- Value: ⭐⭐⭐⭐ — Provides a viable third alternative beyond the pixel and latent spaces, making it especially valuable for high-resolution generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Hessian Geometry of Latent Space in Generative Models](hessian_geometry_of_latent_space_in_generative_models.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](../../ICCV2025/image_generation/summdiff_generative_modeling_of_video_summarization_with_diffusion.md)

</div>

<!-- RELATED:END -->
