---
title: >-
  [Paper Note] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution
description: >-
  [ICCV 2025][Video Generation][Video Super-Resolution] This work is the first to introduce Mamba into video super-resolution, proposing the Dual Aggregation Mamba Block (DAMB) for long-range spatiotemporal dependency mode…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Video Super-Resolution"
  - "Mamba"
  - "State Space Model"
  - "Deformable Alignment"
  - "Frequency Loss"
date: 2026-05-08
content_hash: 1e4fc3760f1d638b
---

# VSRM: A Robust Mamba-Based Framework for Video Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2506.22762](https://arxiv.org/abs/2506.22762)
**Code**: N/A
**Area**: Video Generation
**Keywords**: Video Super-Resolution, Mamba, State Space Model, Deformable Alignment, Frequency Loss

## TL;DR

This work is the first to introduce Mamba into video super-resolution (VSR), proposing the VSRM framework. It achieves efficient spatiotemporal modeling via the Dual Aggregation Mamba Block, combined with Deformable Cross-Mamba Alignment and a frequency-domain loss, achieving state-of-the-art performance on multiple benchmarks.

## Background & Motivation

Video super-resolution requires processing long sequences and capturing inter-frame information over large receptive fields. Existing methods exhibit clear limitations:

- **CNN-based methods** (e.g., BasicVSR): receptive fields are confined to local regions, limiting the capture of long-range inter-frame information.
- **Transformer-based methods** (e.g., IART, PSRT): quadratic complexity of full attention is impractical for long sequences; window attention reduces complexity but sacrifices receptive field coverage.
- **Alignment modules**: most existing methods rely on fixed-weight interpolation (e.g., bilinear) for alignment, causing feature distortion; attention-based implicit alignment is also constrained by fixed reference windows.
- **Loss functions**: pixel-level losses produce over-smoothed outputs; perceptual losses introduce additional distortion; models also suffer from spectral bias.

Mamba's linear complexity, global receptive field, and data-dependent parameterization make it naturally suited for VSR, yet it had not been explored in this context prior to this work.

## Method

### Overall Architecture

VSRM consists of two main components: a feature extractor (Conv2d + Feature Propagation Block) and an upsampler (Reconstruction module). The Feature Propagation Block includes Deformable Cross-Mamba Alignment (DCA) and the Dual Aggregation Mamba Block (DAMB).

### Key Designs

1. **Dual Aggregation Mamba Block (DAMB)**: The core module, composed of $N$ S2TMBs and one T2SMB.

    - **S2TMB (Spatial-to-Temporal Mamba)**: Flattens the 3D sequence into 1D following a spatial-first, temporal-second order, and applies bidirectional (forward and backward) SSM scanning. Bidirectional scanning preserves spatial awareness while enabling temporal modeling. Formula: $S2T\text{-}Mamba(x,z)=Linear(x_1 \odot z + x_2 \odot z)$
    - **T2SMB (Temporal-to-Spatial Mamba)**: Applies only a forward scan (experiments show unidirectional scanning is superior), prioritizing temporal information extraction to complement S2TMB's spatial emphasis.
    - **TGFN (Temporal-Gated Feed-forward Network)**: Incorporates 3D depthwise separable convolutions to model spatiotemporal neighborhood relationships, with a gating mechanism (channel splitting + GELU) to optimize information flow: $TGFN(X)=W_p^2(W_d^1 LN(\hat{X}_1) \odot \sigma(W_d^2 LN(\hat{X}_2)))$

2. **Deformable Cross-Mamba Alignment (DCA)**: Addresses inter-frame motion alignment.

    - Optical flow is estimated using a pretrained SpyNet.
    - A deformable window mechanism is introduced during the compensation stage: a window $w$ is extracted from the reference frame, a reference region $r$ is initialized, and a lightweight offset network learns offsets $\epsilon_r$ to produce a dynamic reference region $\bar{r}=\phi(w; r+\epsilon_r)$.
    - A cross-Mamba module fuses target and dynamic reference features: $\bar{X}(x,y) = cross\text{-}mamba(R,Q)$, where $H_t = \bar{A}_R H_{t-1} + \bar{B}_R \bar{R}_t$, $\bar{X}_t = C_Q H_t$.
    - Compared to fixed-window alignment, DCA adapts more flexibly to motion of varying magnitude.

3. **Frequency Charbonnier-like Loss (FCL)**: Computes the loss in the frequency domain to recover high-frequency details.

    - FFT is applied to the images; Charbonnier losses are computed separately on the real and imaginary parts.
    - $\mathcal{L}_{FCL}=\sum_{i\in\{Re,Im\}} \lambda_i \sqrt{\|i\mathcal{F}(\mathbf{I}_{SR})-i\mathcal{F}(\mathbf{I}_{HR})\|^2+\epsilon^2}$
    - Real and imaginary parts are used directly instead of amplitude/phase, avoiding discontinuities introduced by square roots and arctan operations.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \lambda \mathcal{L}_{CL} + \mathcal{L}_{FCL}$

Hyperparameters: $\lambda=1.0$, $\lambda_{Re}=\lambda_{Im}=0.02$, $\epsilon=10^{-3}$. Training datasets: REDS and Vimeo-90K; task: ×4 super-resolution.

## Key Experimental Results

### Main Results

| Method | Frames | Params (M) | REDS4 PSNR | REDS4 SSIM | Vimeo-90K-T PSNR | Vid4 PSNR |
|--------|--------|------------|------------|------------|-----------------|-----------|
| BasicVSR++ | 30/14 | 7.3 | 32.39 | 0.9069 | 37.79 | 27.79 |
| VRT | 16/7 | 35.6 | 32.19 | 0.9006 | 38.20 | 27.93 |
| RVRT | 30/14 | 10.8 | 32.75 | 0.9113 | 38.15 | 27.99 |
| PSRT-recurrent | 16/14 | 13.4 | 32.72 | 0.9106 | 38.27 | 28.07 |
| IART | 16/7 | 13.4 | 32.90 | 0.9138 | 38.14 | 28.26 |
| **VSRM** | **16/7** | **17.1** | **33.11** | **0.9162** | **38.33** | **28.44** |

Under the 6-frame setting, VSRM also outperforms IART by 0.28 dB (32.43 vs. 32.15).

### Ablation Study

| Ablation | PSNR (dB) | Notes |
|----------|-----------|-------|
| 3D DW-Conv (replaces Mamba) | 30.84 | Mamba shows clear advantage (+0.25 dB) |
| Window Attention (replaces Mamba) | 30.97 | Mamba outperforms (+0.12 dB) |
| Full Attention | 31.06 | Comparable performance but 6.7× higher FLOPs |
| **Mamba (ours)** | **31.09** | Best performance–efficiency trade-off |
| w/o alignment module | 30.87 | Alignment contributes +0.22 dB |
| FGDA alignment | 30.92 | DCA outperforms by +0.17 dB |
| IA alignment | 31.00 | DCA outperforms by +0.09 dB |
| w/o T2SMB | 30.95 | T2SMB contributes +0.14 dB |
| w/o FCL | 30.97 | FCL contributes +0.12 dB |
| FFN (replaces TGFN) | 30.90 | TGFN contributes +0.19 dB |

### Key Findings

- VSRM outperforms IART by 0.21 dB on REDS4 and 0.18 dB on Vid4, demonstrating effectiveness under both large- and small-motion scenarios.
- Mamba achieves performance comparable to full attention (1018.1 G FLOPs) with only 159.2 G FLOPs.
- T2SMB with unidirectional forward scanning outperforms bidirectional scanning (31.09 vs. 31.02), indicating that redundant scanning is detrimental.
- Effective receptive field (ERF) visualization confirms that VSRM achieves a global receptive field, substantially larger than CNN and Transformer counterparts.

## Highlights & Insights

- This work is the first to validate the effectiveness of Mamba in VSR, opening a new backbone option for low-level vision tasks.
- The DCA module's design of "deformable windows + cross-Mamba" is elegant: deformable windows handle motion of varying magnitude, while cross-Mamba performs implicit alignment.
- FCL directly computes Charbonnier loss on real and imaginary parts, which is simpler and more effective than methods such as FFL.
- The integration of 3D depthwise convolutions in TGFN enables the feed-forward network to model spatiotemporal information as well.

## Limitations & Future Work

- Parameter count (17.1 M) and inference time (223 ms) are slightly higher than PSRT/IART (13.4 M, ~175 ms).
- Mamba acceleration and optimization remain an active area; further speedup is achievable.
- Only ×4 super-resolution is evaluated; other scale factors are not explored.
- The framework is extensible to other low-level video tasks such as deblurring, denoising, and colorization.

## Related Work & Insights

- The selective SSM mechanism of Mamba (S6) makes parameters input-dependent, overcoming limitations of conventional SSMs.
- Unlike MambaIR, VSRM must process multi-frame 3D sequences; the S2T/T2S scanning strategy is worth adapting for other video tasks.
- The cross-Mamba alignment paradigm is applicable to other video tasks requiring inter-frame correspondence.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of Mamba to VSR; the S2T/T2S bidirectional scanning design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations covering backbone, alignment, FFN, loss, and scanning direction; validated across multiple datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ Provides a new efficient backbone choice for VSR; state-of-the-art results are convincing.

---

# VSRM: A Robust Mamba-Based Framework for Video Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2506.22762](https://arxiv.org/abs/2506.22762)
**Code**: N/A
**Area**: Image Restoration / Video Super-Resolution
**Keywords**: Video Super-Resolution, Mamba, State Space Model, Deformable Alignment, Frequency Loss

## TL;DR

This work is the first to introduce Mamba into video super-resolution, proposing the Dual Aggregation Mamba Block (DAMB) for long-range spatiotemporal dependency modeling, the Deformable Cross-Mamba Alignment module (DCA) for more flexible inter-frame alignment, and the Frequency Charbonnier-like Loss (FCL) for improved high-frequency detail recovery, achieving state-of-the-art results on REDS4, Vid4, and Vimeo-90K.

## Background & Motivation

Video super-resolution (VSR) aims to generate high-resolution frames from low-resolution video by exploiting complementary multi-frame information. Current approaches are primarily CNN- or Transformer-based:

- **CNN-based methods** (e.g., BasicVSR) are constrained by local receptive fields and cannot effectively capture long-range inter-frame information.
- **Transformer-based methods** (e.g., PSRT, IART) offer powerful attention mechanisms, but full attention's quadratic complexity is impractical for long sequences; window attention reduces complexity at the cost of limited receptive field.
- **Alignment modules**: existing methods commonly use bilinear/nearest-neighbor interpolation for spatial alignment, where fixed weights cause feature distortion; IART proposes attention-based implicit interpolation but computes within fixed reference windows, limiting flexibility.
- **Loss functions**: pixel-level losses produce over-smoothing; perceptual losses introduce greater distortion; reconstruction–GT discrepancies are particularly pronounced in the frequency domain.

Mamba's linear complexity, long-sequence modeling capability, and data-dependent parameterization make it well-suited for VSR. This paper is the first to explore Mamba in this setting.

## Method

### Overall Architecture

VSRM consists of two components: a feature extractor (Conv2d + Feature Propagation Block, FPB) and an upsampler (Reconstruction module). The FPB includes Deformable Cross-Mamba Alignment (DCA) and the Dual Aggregation Mamba Block (DAMB). It first aligns neighboring frame features, then extracts deep spatiotemporal features, and finally generates high-resolution output via the upsampler.

### Key Designs

1. **Dual Aggregation Mamba Block (DAMB)**: Composed of $N$ S2TMBs and one T2SMB, jointly modeling long-range dependencies in both spatial and temporal dimensions.

    - **S2T-Mamba (Spatial-to-Temporal)**: Flattens the 3D video sequence into a 1D sequence with a spatial-first, temporal-second scan order, processed by bidirectional (forward and backward) SSMs. Formula: $S2T\text{-}Mamba(x,z) = Linear(x_1 \odot z + x_2 \odot z)$
    - **T2S-Mamba (Temporal-to-Spatial)**: Uses a temporal-first, spatial-second scan order with a unidirectional forward scan only. Experiments show S2TMB is biased toward spatial information, while T2SMB explicitly prioritizes temporal information, making the two complementary.
    - **TGFN (Temporal-Gated Feed-forward Network)**: Replaces the standard FFN with 3D depthwise separable convolutions and a gating mechanism to better model spatiotemporal neighborhood relationships and optimize information flow.

2. **Deformable Cross-Mamba Alignment (DCA)**: Optical flow is estimated using SpyNet, and a deformable window scheme is introduced during the compensation stage. The core idea is:
    - For each target pixel, the corresponding sampling location is identified in the reference frame via optical flow.
    - A window $w$ is constructed around the sampling location, and a reference region $r$ is initialized.
    - A learnable offset network $\mathcal{S}(w)$ predicts offsets $\epsilon_r$ to obtain a dynamic reference region $\bar{r} = \phi(w; r + \epsilon_r)$.
    - A cross-Mamba module fuses reference and target features to complete alignment: $\bar{X}(x,y) = cross\text{-}mamba(R, Q)$, based on the SSM recurrence $H_t = \bar{A}_R H_{t-1} + \bar{B}_R \bar{R}_t$, $\bar{X}_t = C_Q H_t$.

3. **Frequency Charbonnier-like Loss (FCL)**: Charbonnier losses are computed separately on the real and imaginary parts of the FFT-transformed images, rather than on the amplitude/phase (avoiding numerical instabilities from square roots and arctan operations).

   $$\mathcal{L}_{FCL} = \sum_{i \in \{Re, Im\}} \lambda_i \sqrt{\|i\mathcal{F}(\mathbf{I}_{SR}) - i\mathcal{F}(\mathbf{I}_{HR})\|^2 + \epsilon^2}$$

### Loss & Training

The total loss is a weighted combination of the spatial-domain Charbonnier loss and the frequency-domain FCL:

$$\mathcal{L}_{total} = \lambda \mathcal{L}_{CL} + \mathcal{L}_{FCL}$$

where $\lambda = 1.0$, $\lambda_{Re} = \lambda_{Im} = 0.02$, and $\epsilon = 10^{-3}$. Training datasets: REDS and Vimeo-90K.

## Key Experimental Results

### Main Results

| Method | Input Frames | Params (M) | REDS4 PSNR | REDS4 SSIM | Vid4 PSNR | Vid4 SSIM |
|--------|-------------|------------|------------|------------|-----------|-----------|
| BasicVSR++ | 30/14 | 7.3 | 32.39 | 0.9069 | 27.79 | 0.8400 |
| VRT | 16/7 | 35.6 | 32.19 | 0.9006 | 27.93 | 0.8425 |
| RVRT | 30/14 | 10.8 | 32.75 | 0.9113 | 27.99 | 0.8462 |
| PSRT-rec | 16/14 | 13.4 | 32.72 | 0.9106 | 28.07 | 0.8485 |
| IART | 16/7 | 13.4 | 32.90 | 0.9138 | 28.26 | 0.8517 |
| **VSRM** | **16/7** | **17.1** | **33.11** | **0.9162** | **28.44** | **0.8552** |

VSRM outperforms IART by 0.21 dB on REDS4 (16-frame setting) and 0.18 dB on Vid4, and also achieves the best result of 38.33 dB on Vimeo-90K-T.

### Ablation Study

| Ablation | PSNR (dB) | Params (M) | FLOPs (G) |
|----------|-----------|------------|-----------|
| 3D DW-Conv (replaces Mamba) | 30.84 | 19.49 | 149.8 |
| Window Attention (replaces Mamba) | 30.97 | 7.68 | 152.4 |
| Full Attention (replaces Mamba) | 31.06 | 7.68 | 1018.1 |
| **Mamba (ours)** | **31.09** | **8.61** | **159.2** |
| w/o DCA alignment | 30.87 | 8.53 | 120.4 |
| FGDA alignment | 30.92 | 8.70 | 154.3 |
| IA alignment | 31.00 | 8.57 | 148.7 |
| **DCA alignment (ours)** | **31.09** | **8.61** | **159.2** |
| w/o T2SMB | 30.95 | 7.87 | 155.6 |
| T2SMB (bidirectional) | 31.02 | 8.65 | 162.2 |
| **T2SMB (unidirectional, ours)** | **31.09** | **8.61** | **159.2** |
| FFN | 30.90 | 8.68 | 136.2 |
| **TGFN (ours)** | **31.09** | **8.61** | **159.2** |
| w/o FCL ($\lambda$=0) | 30.97 | — | — |
| **FCL ($\lambda$=0.02)** | **31.09** | — | — |

### Key Findings

- Mamba achieves performance comparable to full attention with only 1/6 the FLOPs (159 G vs. 1018 G).
- DCA outperforms FGDA and IA alignment by 0.17 dB and 0.09 dB, respectively, validating the advantage of the deformable window mechanism.
- T2SMB complements S2TMB's limited temporal information extraction (+0.14 dB), and unidirectional scanning outperforms bidirectional.
- Removing FCL causes a 0.12 dB drop, confirming the importance of frequency-domain regularization for high-frequency detail recovery.
- VSRM's effective receptive field (ERF) substantially exceeds that of CNN and Transformer methods.

## Highlights & Insights

- **First Mamba + VSR work**: Successfully validates the feasibility of Mamba in video super-resolution, achieving both linear complexity and a global receptive field.
- **Complementary S2T and T2S scanning**: Combining spatial-first and temporal-first scanning strategies enables complete spatiotemporal feature extraction—a VSR-specific Mamba adaptation.
- **DCA's deformable reference regions**: Unlike fixed-window implicit alignment, DCA dynamically adjusts reference regions via learned offsets, better handling motion of varying magnitude.
- **Simple and effective FCL design**: Computing Charbonnier loss directly on real and imaginary parts avoids the numerical instability of amplitude/phase-based computation.

## Limitations & Future Work

- Parameter count (17.1 M) and inference time (223 ms) are slightly higher than PSRT/IART (13.4 M, 173–180 ms); Mamba acceleration remains an open research direction.
- Only ×4 super-resolution is explored; other scale factors and degradation models are not evaluated.
- Mamba's hardware acceleration libraries and tooling for vision are less mature than those for Transformers.
- The framework is extensible to other low-level video tasks such as deblurring, denoising, and colorization.

## Related Work & Insights

- The selective SSM mechanism of Mamba (S6) makes parameters input-dependent, overcoming limitations of classical SSMs.
- Unlike MambaIR, VSRM processes multi-frame 3D sequences; the S2T/T2S scanning strategy is transferable to other video tasks.
- Comparisons with frequency-domain losses (FFL, WHFL) confirm FCL's advantage in balancing low- and high-frequency components.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of Mamba to VSR; bidirectional scanning and DCA designs are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablations cover every module; multi-metric, multi-dataset comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a solid Mamba-based baseline for low-level video vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Compressed-Domain-Aware Online Video Super-Resolution](../../CVPR2026/video_generation/compressed-domain-aware_online_video_super-resolution.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[ICCV 2025\] VACE: All-in-One Video Creation and Editing](vace_all-in-one_video_creation_and_editing.md)

</div>

<!-- RELATED:END -->
