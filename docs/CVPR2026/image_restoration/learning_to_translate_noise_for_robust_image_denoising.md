---
title: >-
  [Paper Note] Learning to Translate Noise for Robust Image Denoising
description: >-
  [CVPR 2026][Image Restoration][image denoising] This paper proposes a noise translation framework that converts unknown real-world noise into Gaussian noise via a lightweight noise translation network (NTN)…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "image denoising"
  - "noise translation"
  - "Gaussian noise"
  - "out-of-distribution generalization"
  - "Wasserstein distance"
date: 2026-05-08
content_hash: 59ab6f9748b05c57
---

# Learning to Translate Noise for Robust Image Denoising

**Conference**: CVPR 2026
**arXiv**: [2412.04727](https://arxiv.org/abs/2412.04727)
**Code**: [https://hij1112.github.io/learning-to-translate-noise/](https://hij1112.github.io/learning-to-translate-noise/)
**Area**: Image Restoration
**Keywords**: image denoising, noise translation, Gaussian noise, out-of-distribution generalization, Wasserstein distance

## TL;DR

This paper proposes a noise translation framework that converts unknown real-world noise into Gaussian noise via a lightweight noise translation network (NTN), which is then processed by a pre-trained Gaussian denoising network. The approach achieves an average PSNR gain of over 1.5 dB on OOD real-noise benchmarks, while the translation network contains only 0.29M parameters and is transferable across different denoisers.

## Background & Motivation

Deep learning-based image denoising methods perform well in controlled settings but suffer from severe generalization degradation when confronted with out-of-distribution (OOD) real-world noise:

**Distribution gap between synthetic and real noise**: Early methods assume Gaussian noise, causing significant performance drops in real-world scenarios.

**Overfitting to real-world datasets**: Models trained on real noisy-clean image pairs tend to overfit to the noise-signal correlations specific to the training data, failing to generalize to unseen noise types.

**Infeasibility of covering all real-world noise distributions**: Noise varies enormously across different cameras, sensors, and environments.

**Limitations of existing generalization methods**: Fixed transformations (e.g., the Anscombe transform) lack adaptability; test-time optimization methods (e.g., LAN) impose high computational costs and do not scale to large images.

**Key observation**: Adding additional Gaussian noise to real-noisy images before applying a Gaussian denoiser yields substantially improved results (PSNR increases from 29.63 dB to 32.73 dB). This motivates a "translate-then-denoise" strategy.

## Method

### Overall Architecture

A two-stage framework:
1. **Stage 1**: Train a Gaussian denoising network $\mathcal{D}(\cdot; \boldsymbol{\theta})$ specialized for Gaussian noise.
2. **Stage 2**: Freeze the denoising network and train a noise translation network $\mathcal{T}(\cdot; \boldsymbol{\phi})$ to convert arbitrary real-world noise into Gaussian noise.

Inference pipeline: $\hat{I}_\mathcal{T} = \mathcal{D}(\mathcal{T}(I; \boldsymbol{\phi}); \boldsymbol{\theta}^*)$

### Key Designs

1. **Implicit Noise Translation Loss $\mathcal{L}_{\text{implicit}}$**:

    - **Function**: End-to-end optimization of the combined translation and denoising performance.
    - **Mechanism**: $\|\mathcal{D}(\mathcal{T}(I; \boldsymbol{\phi}); \boldsymbol{\theta}^*) - I_{\text{GT}}\|_1$
    - **Design Motivation**: Rather than directly constraining the form of the translated noise, this loss leverages the frozen denoiser's performance to implicitly require that the translation network produces inputs suited to the Gaussian denoiser.

2. **Explicit Noise Translation Loss $\mathcal{L}_{\text{explicit}}$ (two components)**:

    - **Spatial-domain matching $\mathcal{L}_{\text{spatial}}$**: Uses the 1-Wasserstein distance to match the marginal distribution of the translated noise $n_\mathcal{T}$ against a Gaussian reference $n_\mathcal{G}$.
        - Implementation: Both are flattened and sorted per channel; the L1 distance between sorted elements is computed.
        - Ensures the translated noise follows a Gaussian distribution at the element level.
    - **Frequency-domain matching $\mathcal{L}_{\text{freq}}$**: Uses the 1-Wasserstein distance to match the distributions of Fourier coefficient magnitudes of both noise signals.
        - Mathematical basis: Spatially uncorrelated Gaussian noise has Fourier coefficient magnitudes that follow a Rayleigh distribution.
        - Implementation: FFT is applied to both the translated and reference noise; their magnitude distributions are matched.
        - Ensures spatial decorrelation of the translated noise, eliminating structured noise patterns.
    - Combined: $\mathcal{L}_{\text{explicit}} = \mathcal{L}_{\text{spatial}} + \beta \cdot \mathcal{L}_{\text{freq}}$

3. **Gaussian Injection Block (GIBlock)**:

    - **Function**: Injects Gaussian noise internally at each level of the U-Net.
    - **Mechanism**: Rather than adding noise at the input (which would distort the signal), Gaussian priors are progressively applied within network sub-modules.
    - **Composition**: NAFBlock + Gaussian noise injection + residual connection.
    - **Ablation evidence**: GIBlock is identified as the key component enabling the translation network to reliably map unseen noise to a Gaussian distribution at inference time.

### Loss & Training

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{implicit}} + \alpha \cdot \mathcal{L}_{\text{explicit}}$

- $\alpha = 5 \times 10^{-2}$, $\beta = 2 \times 10^{-3}$
- Gaussian injection noise level: $\tilde{\sigma} = 100$
- Denoising network training data: BSD400 + WED (with Gaussian noise at σ=15) + SIDD
- Translation network training data: SIDD real noisy-clean pairs only, augmented with random Gaussian noise $[0, 15]$
- The translation network is built upon a lightweight U-Net architecture.

## Key Experimental Results

### Main Results (OOD Average PSNR, dB)

| Method | SIDD (ID) | OOD Avg↑ | Gain |
|--------|-----------|----------|------|
| NAFNet | 39.97 | 38.43 | baseline |
| **NAFNet + NTN** | 39.24 | **39.94** | **+1.51** |
| Xformer | 39.98 | 38.58 | baseline |
| **Xformer + NTN** | 39.10 | **40.04** | **+1.46** |
| AFM (Prev. SOTA) | 38.29 | 39.07 | — |
| Mask-Denoising | 38.91 | 38.56 | — |
| CLIP-Denoising | 38.03 | 38.53 | — |

### Ablation Study

| Configuration | SIDD | OOD Avg | Note |
|---------------|------|---------|------|
| Baseline translation (implicit only) | 39.35 | 39.27 | simplest variant |
| + GIBlock | 39.05 | 39.61 | +0.34 OOD |
| + Explicit loss | 39.33 | 39.61 | +0.34 OOD |
| + Both (full model) | 39.24 | **39.94** | +0.67 OOD |

### Comparison with Naive Gaussian Noise Addition

| Input | SIDD | OOD Avg |
|-------|------|---------|
| Original noisy image $I$ | 37.77 | 17.89 |
| $I$ + N(σ=5) | 38.15 | 22.93 |
| $I$ + N(σ=10) | 38.76 | 39.22 |
| $I$ + N(σ=15) | 39.16 | 38.95 |
| **Translated $I_\mathcal{T}$** | **39.24** | **39.94** |

### Key Findings

1. **Limitations of fixed noise addition are evident**: σ=10 performs well on some datasets but poorly on others, and similarly for σ=15; different images and datasets require different noise levels, whereas the translation network adapts automatically.
2. **The translation network transfers across denoisers**: A translation network trained with NAFNet, when directly paired with Xformer, achieves OOD performance (39.94 dB) nearly identical to a translation network trained specifically for Xformer (40.04 dB).
3. **ID performance decrease reflects de-overfitting, not degradation**: The slight PSNR drop on SIDD is attributed to other methods overfitting to training-set artifacts (e.g., zipper-like textures), which the proposed method avoids.
4. **Negligible computational overhead**: The translation network has only 0.29M parameters and 1.07G MACs, compared to NAFNet's 29.1M/16.23G and Xformer's 25.1M/142.68G.

## Highlights & Insights

1. **From a naive observation to an elegant framework**: The paper derives a complete noise translation theory from the empirical observation that adding Gaussian noise improves denoising performance on real-world inputs.
2. **Strong mathematical motivation for the loss functions**:
    - Spatial domain: 1-Wasserstein matching enforces element-level Gaussian distribution.
    - Frequency domain: The mathematical property that FFT magnitudes of Gaussian noise follow a Rayleigh distribution is exploited to enforce spatial decorrelation.
3. **Plug-and-play architecture**: The translation network is fully decoupled from the denoising network; once trained, it can be paired with any pre-trained Gaussian denoiser.
4. **No test-time optimization required**: Unlike methods such as LAN, inference requires no per-pixel optimization, scaling to arbitrary image resolutions.
5. **Compelling visualizations**: Noise distribution histograms before and after translation clearly demonstrate the transition from structured noise to Gaussian noise.

## Limitations & Future Work

1. The translation network is trained exclusively on SIDD and may be limited when encountering noise types that differ substantially from SIDD.
2. A slight ID performance drop (~0.7 dB) is observed; additional fine-tuning may be required in scenarios where peak in-distribution performance is critical.
3. The denoising network must be pre-trained at the same Gaussian noise level (σ=15); the effect of noise level mismatch remains to be validated.
4. Validation is limited to image denoising; applicability to video denoising and other restoration tasks (e.g., deblurring, super-resolution) remains unexplored.

## Related Work & Insights

- **DnCNN**: A pioneering CNN-based denoiser; frequency-domain training improves generalization.
- **NAFNet / Restormer / KBNet**: Strong denoising backbones with limited generalization capability.
- **Anscombe transform / Pixel-Shuffle Downsampling**: Fixed transforms that simplify noise but lack adaptability.
- **LAN**: Test-time optimization of pixel-level offsets; effective but not scalable (limited to 256×256).
- **AFM**: Adversarial training for robustness, but still constrained by the training distribution.
- **Core insight**: "Rather than attempting to denoise all noise types, first translate all noise into a type you already handle well." This divide-and-conquer philosophy is broadly applicable across domains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The noise translation paradigm is original and mathematically well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Nine OOD benchmarks, extensive ablations, visualization analysis, and transferability verification.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative progresses logically from intuition to theory to experiments.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, lightweight, and efficient; the work has tangible practical impact on the image denoising field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](../../NeurIPS2025/image_restoration/scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)

</div>

<!-- RELATED:END -->
