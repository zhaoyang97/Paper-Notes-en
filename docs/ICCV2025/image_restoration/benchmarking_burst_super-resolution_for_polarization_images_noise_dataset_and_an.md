---
title: >-
  [Paper Note] Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis
description: >-
  [ICCV 2025][Image Restoration][polarization image super-resolution] This paper addresses the lack of datasets and noise models for polarization image burst super-resolution (SR) by constructing two dedicated datasets—PolarNS (noise statistics) and PolarBurstSR (SR benchmark)—proposing a polarization noise propagation analysis model, and systematically benchmarking existing burst SR methods on polarization scenes, thereby establishing a standardized evaluation framework for po…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "polarization image super-resolution"
  - "burst super-resolution"
  - "noise modeling"
  - "polarization dataset"
  - "noise propagation analysis"
date: 2026-05-08
content_hash: 38729a7f9192297a
---

# Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis

**Conference**: ICCV 2025
**arXiv**: [2503.18705](https://arxiv.org/abs/2503.18705)  
**Code**: None  
**Area**: Image Restoration / Polarization Imaging
**Keywords**: polarization image super-resolution, burst super-resolution, noise modeling, polarization dataset, noise propagation analysis

## TL;DR

This paper addresses the lack of datasets and noise models for polarization image burst super-resolution (SR) by constructing two dedicated datasets—PolarNS (noise statistics) and PolarBurstSR (SR benchmark)—proposing a polarization noise propagation analysis model, and systematically benchmarking existing burst SR methods on polarization scenes, thereby establishing a standardized evaluation framework for polarization image reconstruction.

## Background & Motivation

**Background**: Snapshot polarization imaging captures color and polarization information simultaneously via a double Bayer pattern sensor, enabling computation of polarization state parameters—such as degree of linear polarization (DoLP) and angle of linear polarization (AoLP)—from four directional sub-images. This technology has important applications in industrial inspection, autonomous driving obstacle detection, and water surface reflection removal.

**Limitations of Prior Work**: Polarization cameras suffer from two severe problems due to the need to encode both color and polarization direction per pixel: (1) low light efficiency—each pixel receives only light of a specific polarization direction, substantially reducing effective photon throughput and significantly increasing noise; and (2) low spatial resolution—the double Bayer pattern reduces effective spatial sampling density to 1/4 that of a conventional camera. Although existing burst SR methods can reduce noise and enhance resolution through multi-frame fusion, directly applying them to polarization images is problematic.

**Key Challenge**: Conventional RGB burst SR methods are unaware of the special statistical properties of polarization noise—polarization noise is not simple Gaussian noise but a complex noise distribution correlated with polarization angle. More fundamentally, the absence of dedicated polarization burst SR datasets and reliable polarization noise ground truth prevents effective training and evaluation of SR models in polarization scenes.

**Goal**: (1) Construct PolarNS, a polarization noise statistics dataset characterizing the noise properties of polarization cameras; (2) construct PolarBurstSR, a polarization burst SR benchmark dataset supporting fair evaluation; (3) propose a polarization noise propagation analysis model; and (4) systematically compare mainstream burst SR methods in polarization scenes.

**Key Insight**: The authors approach the problem from the physical properties of polarization noise, arguing that noise modeling and dataset construction are prerequisites for improving polarization burst SR performance. Only by thoroughly understanding the noise characteristics can targeted SR methods be designed or trained.

**Core Idea**: By constructing a physically calibrated polarization noise dataset and a real-world burst SR dataset, the paper establishes a standardized evaluation framework for polarization burst SR, while guiding polarization image processing through a noise propagation analysis model.

## Method

### Overall Architecture

Rather than proposing a new SR network, this paper establishes a comprehensive research infrastructure for polarization burst SR: (1) the PolarNS dataset—collected in a darkroom environment to calibrate polarization camera noise statistics; (2) the PolarBurstSR dataset—collected across diverse real-world scenes with high-quality reference images as ground truth; (3) a polarization noise propagation model—mathematical derivations from sensor noise to polarization parameters; and (4) benchmark experiments—evaluating multiple SOTA burst SR methods on PolarBurstSR.

### Key Designs

1. **PolarNS Polarization Noise Statistics Dataset**:

    - **Function**: Provides detailed statistical characterization of polarization camera sensor noise.
    - **Mechanism**: Under controlled darkroom conditions, a polarization camera captures large numbers of calibration images at varying exposure times and illumination levels. Statistical analysis over many frames yields the noise mean, variance, and inter-channel noise correlations for each of the four polarization channels (0°, 45°, 90°, 135°). Particular attention is given to the relationship between polarization noise and incident light intensity (shot noise characteristics) and the distribution of dark current noise. The dataset spans multiple illumination intensities and temperature conditions, providing noise model parameters more accurate than simple Gaussian/Poisson assumptions.
    - **Design Motivation**: Most existing polarization image processing methods assume Gaussian noise, but actual polarization sensor noise patterns are more complex—sub-pixels of different polarization angles share physical neighborhoods, introducing spatial noise correlations. Without accurate noise calibration, it is impossible to generate reasonable training data or evaluate denoising performance reliably.

2. **PolarBurstSR Polarization Burst SR Dataset**:

    - **Function**: Serves as the standard evaluation benchmark for polarization burst SR methods.
    - **Mechanism**: Burst sequences are captured with a polarization camera across diverse real-world scenes (indoor, outdoor, varying illumination conditions), with multiple consecutive frames per scene. High-quality reference ground truth is obtained simultaneously via a high-resolution reference camera or long-exposure multi-frame averaging. Paired data consist of low-resolution polarization burst sequences mapped to high-resolution polarization reconstruction targets. The dataset provides evaluation metrics separately for polarization parameters (Stokes parameters $S_0, S_1, S_2$, DoLP, and AoLP).
    - **Design Motivation**: Existing burst SR datasets (e.g., BurstSR, SyntheticBurst) are designed for RGB images and contain no polarization information. Polarization burst SR must simultaneously assess spatial resolution enhancement and preservation of polarimetric measurement accuracy, necessitating dedicated datasets and evaluation protocols.

3. **Polarization Noise Propagation Analysis Model**:

    - **Function**: Derives the noise characteristics of polarization parameters (Stokes, DoLP, AoLP) from raw sensor noise.
    - **Mechanism**: Starting from the physical noise model of the polarization camera sensor (including shot noise, read noise, and dark current), error propagation formulas are applied to derive the noise variances of the Stokes parameters $S_0 = I_0 + I_{90}$, $S_1 = I_0 - I_{90}$, $S_2 = I_{45} - I_{135}$. The noise variances of DoLP $= \sqrt{S_1^2 + S_2^2}/S_0$ and AoLP $= \frac{1}{2}\arctan(S_2/S_1)$ are further derived. The model is validated on the PolarNS dataset, with predicted noise values showing high agreement with measured values.
    - **Design Motivation**: Understanding how noise propagates from raw sensor signals to final polarization parameters is the theoretical foundation for designing effective denoising/SR algorithms. For instance, the model reveals that AoLP noise increases sharply under low-illumination conditions (where division amplifies noise as DoLP approaches zero), providing direct guidance for algorithm design.

### Loss & Training

In the benchmark experiments, two training strategies are compared: (1) training on RGB burst SR datasets and testing on polarization data; and (2) training specifically on polarization data using PolarBurstSR. The latter significantly outperforms the former in polarimetric parameter reconstruction accuracy, demonstrating the necessity of polarization-specific training.

## Key Experimental Results

### Main Results

Comparison of multiple SOTA burst SR methods on PolarBurstSR (×4 SR):

| Method | PSNR (S0) ↑ | SSIM (S0) ↑ | DoLP MAE ↓ | AoLP MAE (°) ↓ | Training Data |
|--------|-------------|-------------|------------|----------------|---------------|
| Bicubic | 28.12 | 0.812 | 0.089 | 12.4 | - |
| DBSR | 31.45 | 0.882 | 0.062 | 8.7 | RGB |
| BIPNet | 32.18 | 0.895 | 0.055 | 7.9 | RGB |
| BSRT | 32.56 | 0.901 | 0.051 | 7.3 | RGB |
| DBSR (polarization training) | 33.21 | 0.918 | 0.038 | 5.2 | Polarization |
| BIPNet (polarization training) | 33.89 | 0.926 | 0.033 | 4.6 | Polarization |
| **BSRT (polarization training)** | **34.32** | **0.932** | **0.029** | **4.1** | **Polarization** |

### Ablation Study

Effect of polarization training vs. RGB training:

| Configuration | PSNR ↑ | DoLP MAE ↓ | AoLP MAE ↓ | Notes |
|---------------|--------|------------|------------|-------|
| BSRT + RGB training | 32.56 | 0.051 | 7.3 | Standard RGB training |
| BSRT + polarization training (no noise model) | 33.78 | 0.035 | 4.8 | Polarization data with simple Gaussian noise |
| BSRT + polarization training (PolarNS noise model) | **34.32** | **0.029** | **4.1** | Physically calibrated noise model |
| Single-frame SR (EDSR) | 30.12 | 0.072 | 9.8 | Advantage of burst fusion is evident |

### Key Findings

- **Polarization-specific training substantially outperforms RGB training**: Consistent improvement is observed across all metrics, with reductions in DoLP/AoLP error exceeding 40%.
- **Accuracy of the noise model matters**: Training with the physically calibrated noise model from PolarNS further improves polarimetric accuracy over the simple Gaussian noise assumption.
- **The gap is larger in low-light scenes**: Due to the low light efficiency of polarization cameras, noise is more severe in low-light conditions, making the advantage of polarization-specific training more pronounced.
- **Burst fusion vs. single-frame**: Multi-frame fusion yields a large advantage in polarization scenes (+4 dB), as the signal-to-noise ratio of polarization channels is inherently lower.
- **AoLP is the most difficult polarization parameter to reconstruct**: In low-DoLP regions, AoLP noise increases sharply, consistent with theoretical analysis.

## Highlights & Insights

- **Fills the dataset gap in polarization burst SR**—this is the first burst SR benchmark specifically designed for polarization imaging, making a significant contribution to subsequent research in the field. The broader strategy of "establishing standardized benchmarks for emerging modalities" is equally applicable to other novel visual sensors (e.g., event cameras, ToF cameras).
- **The noise propagation analysis model has high practical value**—it is useful not only for training data generation but also for guiding camera parameter selection (e.g., determining what exposure time ensures sufficient polarimetric measurement accuracy under a given illumination level). This methodology of "physics-driven algorithm design" is transferable to other sensor types.
- **Systematic comparative experiments establish the necessity of polarization-specific training**—it is not simply a matter of retraining RGB methods on a different dataset; the distinctive nature of polarization noise must be perceived and handled by the algorithm.

## Limitations & Future Work

- The dataset scale is relatively limited, and scene diversity (e.g., dynamic scenes, extreme weather) warrants expansion.
- No SR network architecture specifically designed for polarization characteristics is proposed—existing RGB architectures are merely retrained on polarization data.
- The polarization noise model assumes static scenes and perfect inter-frame alignment; motion and alignment errors in real-world scenarios are not fully addressed.
- Future work could explore embedding polarization physical constraints (e.g., consistency among Stokes parameters) directly into network architectures or loss functions.
- Extension to polarization video SR, leveraging temporal redundancy for further improvement, is a promising direction.

## Related Work & Insights

- **vs. BurstSR/BSRT**: These represent state-of-the-art RGB burst SR methods and datasets, performing well in RGB settings but lacking awareness of polarization noise characteristics. The proposed dataset enables fair evaluation and adaptation of these methods to polarization scenes.
- **vs. Polarization Demosaicking**: Polarization demosaicking is a related but distinct task—focused on recovering four polarization channels from a single frame—whereas burst SR exploits multi-frame information for simultaneous denoising and super-resolution. The two can be used in combination.
- **vs. Noise2Noise/Self-supervised Denoising**: Self-supervised denoising methods can also be applied to polarization scenes, but require guidance from a polarization noise model to design appropriate training strategies. The PolarNS dataset provides the necessary foundation for this.

## Rating

- Novelty: ⭐⭐⭐ — The primary contributions lie in dataset and benchmark construction; methodological innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Data collection is rigorous, comparative experiments are systematic and comprehensive, and the noise model is thoroughly validated.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and physical analysis is rigorous.
- Value: ⭐⭐⭐⭐ — Establishes important research infrastructure for the polarization imaging community with lasting impact on future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](../../ECCV2024/image_restoration/a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](../../CVPR2025/image_restoration/qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICLR 2026\] Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content](../../ICLR2026/image_restoration/exploring_real-time_super-resolution_benchmarking_and_fine-tuning_for_streaming_.md)

</div>

<!-- RELATED:END -->
