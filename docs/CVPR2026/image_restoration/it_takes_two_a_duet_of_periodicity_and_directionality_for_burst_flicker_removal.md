---
title: >-
  [Paper Note] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal
description: >-
  [CVPR 2026][Image Restoration][Burst Flicker Removal] This paper identifies two intrinsic physical properties of flicker artifacts—periodicity and directionality—and proposes Flickerformer, comprising three dedicated modules (PFM/AFFN/WDAM) for inter-frame/intra-frame periodicity and directionality modeling respectively. With only 3.92M parameters, the method achieves 31.226 dB PSNR on the BurstDeflicker benchmark, surpassing the second-best method AST by +0.580 dB while using only 19.70% of its parameters.
tags:
  - CVPR 2026
  - Image Restoration
  - Burst Flicker Removal
  - Phase Correlation
  - Autocorrelation
  - Wavelet Attention
  - Transformer
date: 2026-05-08
content_hash: 80046ac1b3e2fe2a
---

# Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal

**Conference**: CVPR 2026
**arXiv**: [2603.22794](https://arxiv.org/abs/2603.22794)
**Code**: [GitHub](https://github.com/qulishen/Flickerformer)
**Area**: Image Restoration
**Keywords**: Burst Flicker Removal, Phase Correlation, Autocorrelation, Wavelet Attention, Transformer

## TL;DR

This paper identifies two intrinsic physical properties of flicker artifacts—periodicity and directionality—and proposes Flickerformer, comprising three dedicated modules (PFM/AFFN/WDAM) for inter-frame/intra-frame periodicity and directionality modeling respectively. With only 3.92M parameters, the method achieves 31.226 dB PSNR on the BurstDeflicker benchmark, surpassing the second-best method AST by +0.580 dB while using only 19.70% of its parameters.

## Background & Motivation

**State of the Field**: When capturing short-exposure images under AC-powered artificial lighting, the light source intensity oscillates periodically at the AC frequency. Combined with the row-by-row exposure mechanism of modern rolling shutter cameras, this produces stripe-like luminance fluctuations in the image—known as flicker artifacts. Such artifacts not only degrade perceptual quality but also impair downstream vision tasks (HDR imaging, slow-motion video, motion capture, etc.).

**Limitations of Prior Work**: Traditional approaches rely on hardware sensor adjustments to control exposure (at the cost of motion blur) or assume known lighting parameters (limiting applicability to controlled scenes). Recent deep learning methods (DeflickerCycleGAN, BurstDeflicker benchmark methods) **treat flicker as generic image degradation**, directly applying denoising/deblurring/HDR frameworks while entirely ignoring the physical structural characteristics of flicker.

**Root Cause**: Flicker is not random noise but a **structured degradation** with specific spatiotemporal patterns. Generic restoration frameworks fail to capture these patterns, resulting in insufficient flicker suppression and ghosting artifacts during multi-frame fusion.

**Paper Goals**: To embed physical priors of flicker into a deep network for the first time, designing a dedicated architecture targeting the structural properties of flicker that effectively removes it while avoiding ghosting.

**Starting Point**: The authors identify two exploitable intrinsic properties of flicker artifacts: (1) **Periodicity**—flicker stripes exhibit repetitive patterns in both space and time, and swapping the phase components of two frames exchanges the spatial distribution of flicker, indicating that phase encodes flicker location information; (2) **Directionality**—due to the row-by-row scanning of rolling shutters, flicker stripes align along the scan direction (horizontal/vertical), producing directional high-frequency luminance oscillations and low-frequency dark bands.

**Core Idea**: Three complementary modules are designed—PFM and AFFN exploit periodicity (inter-frame phase correlation + intra-frame autocorrelation), while WDAM exploits directionality (wavelet-domain high-frequency guidance for low-frequency restoration)—forming the first Transformer architecture that integrates physical priors of flicker.

## Method

### Overall Architecture

Flickerformer adopts an asymmetric U-shaped encoder-decoder architecture. The input consists of 3 burst short-exposure flicker images (reference frame $\mathbf{I}_1$ and two neighboring frames $\mathbf{I}_0, \mathbf{I}_2$). Low-level features $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ are first extracted independently via grouped convolutions, then fed into PFM for inter-frame feature fusion to obtain $\mathbf{F}_0$. The encoder comprises 3 stages, each with 2 Transformer blocks, with channel dimensions [32, 64, 96] and attention heads [1, 2, 4]. AFFN is used in the encoder to enhance periodic representations, while WDAM is used in the decoder for directional attention. The network predicts a residual map $\mathbf{R}$, yielding the final output $\hat{\mathbf{I}}_1 = \mathbf{I}_1 + \mathbf{R}$.

### Key Designs

1. **PFM (Phase-based Fusion Module)**:

    - **Function**: Adaptively aggregates burst multi-frame features via inter-frame phase correlation, suppressing flicker while avoiding ghosting.
    - **Mechanism**: FFT is applied to each frame's features to obtain frequency-domain representations $\tilde{\mathbf{X}}_t = A_t(\mathbf{k})e^{i\Phi_t(\mathbf{k})}$. The phase correlation score between each reference frame and the base frame is computed as $\mathbf{S}_t(\mathbf{k}) = |e^{i\Phi_t(\mathbf{k})} \odot e^{-i\Phi_1(\mathbf{k})}|$, serving as a frequency-domain reliability indicator. Weight maps $\mathbf{W}_t$ are generated via convolution + sigmoid, the reference frame spectra are filtered accordingly, transformed back via IFFT, and the three enhanced features are concatenated and fused by convolution.
    - **Design Motivation**: The spatial distribution of flicker is encoded in the phase (swapping phases between frames swaps the flicker patterns). Phase correlation precisely measures the periodic differences in flicker between frames, selectively retaining useful information from reference frames while suppressing flicker interference. Frequency-domain multiplication is equivalent to spatial convolution, making PFM essentially an adaptive frequency-domain filter.

2. **AFFN (Autocorrelation Feed-Forward Network)**:

    - **Function**: Captures the spatial periodicity of flicker stripes within a single frame via autocorrelation.
    - **Mechanism**: Based on the Wiener–Khinchin theorem, spatial autocorrelation is efficiently computed as $\mathbf{R}_l = \mathcal{F}^{-1}(|\mathcal{F}(\mathbf{F}_l)|^2)$, which amplifies repetitive structures while suppressing uncorrelated noise. A dual-domain enhancement is then applied: the power spectrum is added in the frequency domain as $\hat{\mathbf{F}}_k = \mathcal{F}(\mathbf{F}_l) + \alpha|\mathcal{F}(\mathbf{F}_l)|^2$, and the autocorrelation is superimposed in the spatial domain as $\hat{\mathbf{F}}_l = \mathcal{F}^{-1}(\hat{\mathbf{F}}_k) + \beta\mathbf{R}_l$, where $\alpha$ and $\beta$ are learnable parameters. The output is produced through a depthwise gated FFN.
    - **Design Motivation**: While PFM captures inter-frame periodicity, AFFN complements it by modeling intra-frame spatial periodicity—flicker stripes within a single frame appear as regular patterns, and autocorrelation is naturally suited to detecting such repetitive signals. The two modules are complementary, forming a complete periodicity modeling pipeline.

3. **WDAM (Wavelet-based Directional Attention Module)**:

    - **Function**: Uses wavelet-domain directional high-frequency information to guide precise localization and restoration of low-frequency flicker dark bands.
    - **Mechanism**: Haar wavelet decomposition is applied to the input features, yielding a low-frequency subband $\mathbf{F}_{LL}$ and three high-frequency subbands (horizontal $\mathbf{F}_{LH}$, vertical $\mathbf{F}_{HL}$, diagonal $\mathbf{F}_{HH}$). The low-frequency component is processed by window-based multi-head attention for restoration. The horizontal and vertical high-frequency components are passed through convolution + sigmoid to generate a directional weight map $\mathbf{M}$, which element-wise modulates the Value branch of attention: $\text{Att} = \text{Softmax}(\frac{\mathbf{QK}^\top}{\sqrt{d}} + \mathbf{B})(\mathbf{M} \odot \mathbf{V})$. All subbands are reconstructed via IDWT.
    - **Design Motivation**: Flicker dark bands correspond to corrupted low-frequency information, whereas directional high-frequency signals (edge variations in LH/HL subbands) remain relatively stable and can precisely mark the location and direction of flicker regions. Using high-frequency as a "directional compass" to guide low-frequency restoration is more precise than isotropic standard attention. Moreover, since attention is computed only on the half-resolution low-frequency subband, computational complexity is reduced to approximately 25% of standard window attention, with FLOPs dropping from 139.42G to 128.76G.

### Loss & Training

An equal-weight combination of L1 loss and VGG-19 perceptual loss is used. The Adam optimizer is employed with a learning rate of $1 \times 10^{-4}$. The input consists of 3 burst frames, and the output is the deflickered result for the reference frame. The channel expansion factor is $\gamma = 2.66$.

## Key Experimental Results

### Main Results

Comparison with 16 state-of-the-art methods on the BurstDeflicker benchmark:

| Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ | Params(M) | Flops(G) |
|------|------|:-----:|:-----:|:------:|:---------:|:--------:|
| **Flickerformer (Ours)** | **Dedicated** | **31.226** | **0.920** | **0.045** | **3.92** | **128.76** |
| AST | General Restoration | 30.646 | 0.918 | 0.050 | 19.90 | 156.43 |
| Restormer | General Restoration | 30.630 | 0.917 | 0.055 | 26.10 | 141.16 |
| FPro | General Restoration | 30.551 | 0.910 | 0.051 | 22.38 | 247.04 |
| Uformer | General Restoration | 30.544 | 0.910 | 0.056 | 18.12 | 145.24 |
| HINT | General Restoration | 30.336 | 0.916 | 0.046 | 24.85 | 142.30 |
| HDRTransformer | HDR | 30.031 | 0.918 | 0.054 | 1.04 | 272.12 |
| RT-XNet | Low-light Enhancement | 29.718 | 0.909 | 0.058 | 3.66 | 245.82 |
| Retinexformer | Low-light Enhancement | 29.598 | 0.899 | 0.055 | 3.74 | 184.14 |
| FFTformer | Deblurring | 29.478 | 0.895 | 0.050 | 14.88 | 131.71 |
| MambaIR | General Restoration | 29.478 | 0.904 | 0.060 | 3.59 | 186.76 |
| FBANet | Burst SR | 29.459 | 0.896 | 0.052 | 4.76 | 432.07 |
| Burstormer | Burst SR | 29.439 | 0.910 | 0.056 | 0.17 | 141.05 |
| Stripformer | Deblurring | 29.223 | 0.892 | 0.058 | 19.71 | 681.64 |
| SAFNet | HDR | 29.223 | 0.892 | 0.058 | 1.12 | 169.74 |
| AFUNet | HDR | 28.922 | 0.903 | 0.066 | 1.14 | 301.36 |

Flickerformer achieves the best performance on all three metrics. Its PSNR surpasses the second-best method AST by +0.580 dB, with only 19.70% of AST's parameters (3.92M vs. 19.90M) and lower computational cost (128.76G vs. 156.43G).

### Ablation Study

**FFN Replacement Study** (replacing only AFFN with other FFN variants):

| FFN Variant | Params(M) | Flops(G) | PSNR(dB) |
|---------|:---------:|:--------:|:--------:|
| **AFFN (Ours)** | **3.92** | **128.76** | **31.226** |
| FRFN [AST] | 4.03 | 128.76 | 30.961 |
| GDFN [Restormer] | 3.92 | 128.76 | 30.959 |
| LeFF [Uformer] | 4.60 | 146.73 | 30.954 |
| FFN [SwinIR] | 4.45 | 139.31 | 30.876 |

**Attention Mechanism Replacement Study** (replacing only WDAM with other attention mechanisms):

| Attention Module | Params(M) | Flops(G) | PSNR(dB) |
|-----------|:---------:|:--------:|:--------:|
| **WDAM (Ours)** | **3.92** | **128.76** | **31.226** |
| ASSA [AST] | 3.92 | 139.42 | 30.997 |
| Condensed SA | 3.46 | 132.29 | 30.981 |
| Swin SA | 3.92 | 139.36 | 30.896 |
| Top-k SA | 3.96 | 145.20 | 30.894 |

**Individual Module Contributions** (replacing each module one at a time from the AST baseline):

| Config | Fusion | FFN | Attention | PSNR↑ | SSIM↑ |
|------|:----:|:---:|:------:|:-----:|:-----:|
| (a) AST baseline | CNN | FRFN | ASSA | 30.449 | 0.912 |
| (b) +PFM | **PFM** | FRFN | ASSA | 30.728 | 0.914 |
| (c) +AFFN | CNN | **AFFN** | ASSA | 30.831 | 0.915 |
| (d) +WDAM | CNN | FRFN | **WDAM** | 30.822 | 0.915 |
| (e) **Full Model** | **PFM** | **AFFN** | **WDAM** | **31.226** | **0.920** |

### Key Findings

- **All three modules contribute independently and are highly complementary**: PFM contributes +0.279 dB, AFFN contributes +0.382 dB, and WDAM contributes +0.373 dB; their combined gain of +0.777 dB exceeds the sum of individual contributions, indicating synergistic effects among modules.
- **Core reason AFFN outperforms FRFN**: Visualization shows that FRFN cannot distinguish between motion changes and flicker changes, introducing ghosting artifacts during fusion. AFFN focuses on periodic structures via autocorrelation, effectively separating flicker from motion.
- **WDAM achieves more precise directional localization**: Compared to isotropic attention mechanisms such as ASSA, WDAM restores subtle flicker regions (e.g., faces) more thoroughly, as high-frequency subbands provide accurate localization of flicker position and direction.
- **Significant efficiency advantage**: WDAM performs attention computation only on the half-resolution low-frequency subband, reducing complexity to approximately 25% of standard window attention, with FLOPs decreasing from 139.42G to 128.76G.

## Highlights & Insights

- **Physics-prior-driven network design**: Starting from the AC electrical origin of flicker and the rolling shutter mechanism, two exploitable properties—periodicity and directionality—are derived, with each module having a clear physical counterpart. This "understand the degradation → design the method" paradigm is equally applicable to other structured degradations (moiré, banding, rolling shutter distortion).
- **Counter-intuitive "high-frequency guides low-frequency" design**: Conventionally, low frequencies carry primary content while high frequencies encode details. However, in flicker scenarios, the low frequency (dark bands) is most severely corrupted, while directional high-frequency information (edge variations in LH/HL subbands) remains relatively stable. WDAM leverages stable high-frequency signals as a "directional compass" to guide low-frequency restoration—an elegant cross-frequency complementary design.
- **Phase encodes the spatial distribution of flicker**: Experiments verify that swapping the phases of two frames swaps their flicker patterns. This finding directly motivates the design of PFM—using phase correlation to measure inter-frame flicker differences, which outperforms direct pixel-level alignment or attention-based fusion.
- **Exceptionally high parameter efficiency**: SOTA performance is achieved with only 3.92M parameters. By embedding physical priors, the network requires less capacity to discover patterns autonomously, reducing the overall model size requirement.

## Limitations & Future Work

- **Inability to recover large fully-darkened regions**: When no clean information exists for a region across all burst frames (e.g., a large area where the light is completely off), the model can only partially restore the content—an inherently information-absent problem that requires more burst frames or the incorporation of generative priors.
- **Fixed Haar wavelet basis**: The current implementation uses a fixed Haar wavelet basis; learnable adaptive wavelet bases may better accommodate different types of flicker patterns.
- **Fixed burst frame count of 3**: The number of input frames is manually specified; adaptive burst size selection could provide greater flexibility.
- **Limited to short-exposure scenarios**: Extension to other forms of flicker, such as LED PWM flickering in long-exposure video, remains to be explored.

## Related Work & Insights

- **vs. General Restoration (Restormer/AST/Uformer)**: These frameworks do not model the periodicity or directionality intrinsic to flicker, resulting in insufficient flicker suppression and large parameter counts (20–26M vs. 3.92M). This demonstrates that embedding task-specific priors can simultaneously improve performance and efficiency.
- **vs. Burst SR (Burstormer/FBANet)**: Burst SR assumes spatially uniform degradation, limiting its effectiveness on non-uniform structured degradations like flicker (PSNR only 29.4–29.5 dB); FBANet also incurs high computational cost at 432G FLOPs.
- **vs. HDR Methods (HDRTransformer/SAFNet)**: HDR methods focus on exposure fusion rather than stripe pattern removal, introducing color bias and ghosting under severe flicker.
- **Implications for moiré removal**: Moiré is also a structured periodic degradation; the periodicity modeling ideas behind PFM and AFFN may transfer to that domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic identification of the periodicity + directionality physical properties of flicker; each of the three modules (PFM/AFFN/WDAM) has a clear physical motivation; the finding that phase encodes flicker distribution is particularly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparison with 16 SOTA methods spanning 6 task types; three ablation studies with per-module validation and feature visualizations; however, evaluation is conducted on a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from physical properties to module design is clear; the phase-swapping experiment in Figure 1 is intuitive and convincing; mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — The first dedicated architecture for burst flicker removal; 3.92M parameters outperform general-purpose methods with 20M+; the methodology of embedding physical priors into networks offers broad inspiration for structured degradation restoration.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [\[CVPR 2026\] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](ntire_2026_raindrop_removal_challenge.md)
- [\[ICCV 2025\] Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis](../../ICCV2025/image_restoration/benchmarking_burst_super-resolution_for_polarization_images_noise_dataset_and_an.md)

<!-- RELATED:END -->
