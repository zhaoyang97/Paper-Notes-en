---
title: >-
  [Paper Note] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging
description: >-
  [CVPR 2026][Image Restoration][HRTEM Denoising] This paper proposes SCGN (Statistical Characteristic-Guided denoising Network), which adaptively enhances signal and suppresses noise in both spatial and frequency domains via window standard deviation weighting and frequency band-guided channel attention, respectively. Combined with an HRTEM-specific noise calibration method that generates realistic noisy datasets containing disordered structures, SCGN achieves high-quality denoising of high-resolution transmission electron microscopy images at millisecond-level acquisition speeds.
tags:
  - CVPR 2026
  - Image Restoration
  - HRTEM Denoising
  - Statistical Characteristic Guidance
  - Frequency-Domain Denoising
  - Spatial Deviation Weighting
  - Noise Calibration
date: 2026-05-08
content_hash: 085a2f233afe20ef
---

# Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.18834](https://arxiv.org/abs/2603.18834)
**Authors**: Hesong Li, Ziqi Wu, Ruiwen Shao, Ying Fu
**Code**: [HeasonLee/SCGN](https://github.com/HeasonLee/SCGN)
**Area**: Image Restoration
**Keywords**: HRTEM Denoising, Statistical Characteristic Guidance, Frequency-Domain Denoising, Spatial Deviation Weighting, Noise Calibration

## TL;DR

This paper proposes SCGN (Statistical Characteristic-Guided denoising Network), which adaptively enhances signal and suppresses noise in both spatial and frequency domains via window standard deviation weighting and frequency band-guided channel attention, respectively. Combined with an HRTEM-specific noise calibration method that generates realistic noisy datasets containing disordered structures, SCGN achieves high-quality denoising of high-resolution transmission electron microscopy images at millisecond-level acquisition speeds.

## Background & Motivation

High-resolution transmission electron microscopy (HRTEM) enables atomic-scale observation of nucleation dynamics and serves as a core tool for studying advanced solid-state materials. However, nucleation processes evolve rapidly on the millisecond timescale, necessitating short-exposure fast imaging, which introduces severe noise that obscures atomic position information.

**Limitations of Prior Work**:

- **General image denoising methods** (e.g., DnCNN, Restormer): These methods do not account for the distinctive statistical properties of HRTEM images—atomic regions and background regions differ significantly in spatial deviation and frequency distribution. Applying a uniform denoising strategy across all regions makes it difficult to simultaneously preserve atomic details and suppress background noise.
- **HRTEM noise modeling**: HRTEM noise differs from natural image noise, being influenced by electron beam shot noise and detector readout noise, among other factors. Existing Gaussian/Poisson noise models are insufficiently accurate.
- **Scarcity of training data**: There is a lack of training datasets that include disordered structures (a critical feature during nucleation) with realistic HRTEM noise characteristics.

**Core Insight**: In HRTEM images, the local standard deviation of atomic regions (high-signal areas) is significantly higher than that of background regions, and the signal is concentrated in specific frequency bands. These statistical characteristics can be exploited to guide the denoising process, applying adaptive processing strategies to different spatial locations and frequency bands.

## Method

### Overall Architecture

SCGN adopts a residual network architecture based on FFC (Fast Fourier Convolution):

$$\hat{I}_{clean} = I_{noisy} + \mathcal{F}(I_{noisy})$$

The network consists of a head conv (1→64 channels) → 8 FFCResnetBlocks → tail conv (64→1 channel), with a global skip connection. Within each FFCResnetBlock, features are split into a spatial branch (32 channels) and a frequency branch (32 channels), processed separately and then concatenated for fusion.

### Spatial Deviation-Guided Weighting

The core module, WindowStd, computes the local window standard deviation at each spatial location:

$$\sigma(x, y) = \sqrt{\frac{1}{K^2} \sum_{(i,j) \in \mathcal{W}} [F(i,j) - \bar{F}(x,y)]^2}$$

where $\mathcal{W}$ is a $3 \times 3$ window and $\bar{F}$ is the window mean. In practice, computation leverages the identity $\text{Var}(X) = E[X^2] - (E[X])^2$, efficiently implemented via two depthwise separable convolutions with mirror padding to ensure edge accuracy.

The standard deviation map is passed through a $1 \times 1$ convolution and Sigmoid to produce spatially adaptive weights:

$$W_{spatial} = \sigma\left(\text{Conv}_{1 \times 1}(\sigma(F))\right)$$

These weights are multiplied with the spatial convolution output, enabling the network to preserve more detail in high-deviation regions (atomic sites) while more aggressively denoising low-deviation regions (background).

### Frequency Band-Guided Weighting

The frequency branch is built upon the SpectralTransform module, operating in the frequency domain:

1. **FFT**: 2D rFFT is applied to the input features to obtain a frequency-domain representation.
2. **Coordinate encoding**: Normalized frequency coordinates $(u, v)$ are concatenated to the frequency-domain features, enabling the network to perceive the position of different frequency bands.
3. **Frequency-domain convolution**: $1 \times 1$ convolution processes the real and imaginary parts.
4. **Channel attention weighting**: A ChannelAttention module (average pooling + max pooling → shared FC → Sigmoid) applies adaptive weights to different frequency bands, enhancing bands containing atomic signals and suppressing noise-dominated bands.
5. **IFFT**: Inverse FFT maps the features back to the spatial domain.

The channel attention design allows the frequency branch to learn the differences in frequency distribution between signal and noise in HRTEM images, performing adaptive frequency-domain filtering.

### HRTEM-Specific Noise Calibration and Dataset

1. **Noise calibration**: The noise statistics (mean–variance relationship) of real HRTEM images are analyzed to establish a calibration model for electron beam shot noise and detector noise.
2. **Disordered structure generation**: Molecular dynamics simulations or random perturbation methods are used to generate disordered atomic structures representative of nucleation processes.
3. **Dataset construction**: The calibrated noise model is applied to synthetic disordered structure images, producing a paired dataset (noisy/clean image pairs) of 1,000 training samples and 100 test samples.

## Key Experimental Results

### Table 1: Quantitative Comparison on Synthetic HRTEM Data

| Method | PSNR (dB) ↑ | SSIM ↑ | IoU (%) ↑ | Parameters |
|--------|-------------|--------|-----------|------------|
| BM3D | 28.34 | 0.812 | 71.2 | - |
| DnCNN | 30.15 | 0.856 | 76.8 | 0.56M |
| FFDNet | 30.42 | 0.861 | 77.3 | 0.49M |
| SwinIR | 31.28 | 0.883 | 80.5 | 11.8M |
| Restormer | 31.56 | 0.889 | 81.2 | 26.1M |
| NAFNet | 31.43 | 0.886 | 80.8 | 17.1M |
| **SCGN (Ours)** | **32.14** | **0.901** | **84.6** | ~2.5M |

SCGN achieves the best performance across all three metrics: PSNR surpasses Restormer by **0.58 dB**, IoU improves by **3.4%**, with only approximately 1/10 of Restormer's parameter count. The substantial IoU gain demonstrates that denoising quality directly improves downstream atomic localization tasks.

### Table 2: Ablation Study

| Configuration | PSNR (dB) | SSIM | IoU (%) |
|---------------|-----------|------|---------|
| Baseline (CNN only) | 30.87 | 0.872 | 78.1 |
| + Frequency branch (FFC) | 31.45 | 0.888 | 81.3 |
| + Spatial std. deviation weighting | 31.78 | 0.894 | 83.0 |
| + Frequency band channel attention | 32.01 | 0.898 | 84.1 |
| + HRTEM noise calibration | **32.14** | **0.901** | **84.6** |

Each component yields consistent improvements: spatial standard deviation weighting contributes the most (+0.33 dB), the frequency branch and channel attention each contribute approximately +0.5 dB, and noise calibration provides an additional +0.13 dB.

### Results on Real HRTEM Images

On real fast-acquisition HRTEM data, images denoised by SCGN allow clear identification of individual atomic positions, with atomic localization accuracy superior to all competing methods. In particular, in disordered regions at nucleation fronts—where other methods tend to produce spurious atoms or miss real ones—SCGN's statistical characteristic-guided mechanism effectively avoids these artifacts.

## Highlights & Insights

- **Statistics-driven adaptive denoising**: This work is the first to use the spatial deviation and frequency band distribution of HRTEM images as explicit guidance signals rather than relying on the network to learn these properties implicitly, significantly improving differentiated processing of atomic and background regions.
- **Lightweight and efficient**: With approximately 2.5M parameters, SCGN surpasses large models such as Restormer (26.1M); the window standard deviation computation requires no trainable parameters, and frequency-domain operations are inherently efficient.
- **End-to-end differentiable standard deviation computation**: The $E[X^2] - (E[X])^2$ identity enables window standard deviation computation via convolution with full backpropagation support, elegantly embedding statistical quantities into the network.
- **Domain-customized noise modeling**: The HRTEM noise calibration method addresses the deficiencies of generic noise models on electron microscopy images, ensuring robust transfer from synthetic to real data.
- **Direct benefit to downstream tasks**: Beyond PSNR/SSIM image quality metrics, the method is evaluated on atomic localization IoU, demonstrating the practical scientific value of the denoising quality.

## Limitations & Future Work

- **Domain specificity**: The method is highly tailored to HRTEM images; the assumption underlying spatial standard deviation guidance (high deviation in atomic regions) may not transfer directly to other microscopy modalities or medical imaging.
- **Limited dataset scale**: The dataset of 1,000 training and 100 test images is relatively small; model generalizability under a wider range of HRTEM conditions remains to be validated.
- **Single noise level**: The current design appears optimized for fixed fast-acquisition conditions; adaptive capability across different exposure times and electron doses is not thoroughly explored.
- **Fixed architecture**: The design of 8 FFCResnetBlocks with 64 channels has not been subjected to systematic architecture search or scaling experiments.

## Related Work & Insights

- **General image denoising**: DnCNN (residual learning), FFDNet (noise level map input), SwinIR/Restormer (Transformer architectures), NAFNet (simplified attention) → None account for the statistical characteristics of HRTEM images.
- **Frequency-domain denoising**: FFC (NeurIPS 2020, fast Fourier convolution), DFCAN (frequency-domain enhancement) → SCGN builds upon FFC by introducing frequency band-guided channel attention.
- **Electron microscopy image processing**: Traditional methods are largely based on BM3D or Wiener filtering; recent end-to-end U-Net-based approaches exist but none exploit statistical characteristics as explicit guidance.
- **Positioning of SCGN**: By integrating physical priors (statistical characteristics) with data-driven learning (deep networks), SCGN achieves state-of-the-art HRTEM denoising while remaining lightweight.

## Rating

- Novelty: ⭐⭐⭐⭐ — The spatial–frequency adaptive denoising framework guided by statistical characteristics is conceptually clear and original; domain physical priors are elegantly embedded into the network design.
- Experimental Thoroughness: ⭐⭐⭐ — Both synthetic and real data are evaluated, but the dataset scale is limited and the set of comparison methods could be more comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is rigorously described, and code is publicly available.
- Value: ⭐⭐⭐⭐ — The method has direct application value for atomic-scale dynamic observation in materials science and is generalizable to other scientific imaging denoising scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](../../ICLR2026/image_restoration/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[NeurIPS 2025\] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy](../../NeurIPS2025/image_restoration/scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)

<!-- RELATED:END -->
