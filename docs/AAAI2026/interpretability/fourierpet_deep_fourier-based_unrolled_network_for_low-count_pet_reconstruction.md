---
title: >-
  [Paper Note] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction
description: >-
  [AAAI 2026][PET reconstruction] This work identifies three categories of degradation in low-count PET that are separable in the frequency domain — Poisson noise and photon deficiency induce high-frequency phase perturbations, while attenuation correction errors suppress low-frequency amplitude — and proposes FourierPET: an ADMM-unrolled, frequency-aware reconstruction framework that achieves comprehensive state-of-the-art performance across three datasets with only 0.44M parameters.
tags:
  - AAAI 2026
  - PET reconstruction
  - frequency domain analysis
  - ADMM unrolling
  - amplitude-phase decoupling
  - low-dose imaging
date: 2026-05-08
content_hash: 1890fa6c55955c7e
---

# FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2601.11680](https://arxiv.org/abs/2601.11680)
**Code**: Unavailable
**Area**: Interpretability
**Keywords**: PET reconstruction, frequency domain analysis, ADMM unrolling, amplitude-phase decoupling, low-dose imaging

## TL;DR

This work identifies three categories of degradation in low-count PET that are separable in the frequency domain — Poisson noise and photon deficiency induce high-frequency phase perturbations, while attenuation correction errors suppress low-frequency amplitude — and proposes FourierPET: an ADMM-unrolled, frequency-aware reconstruction framework that achieves comprehensive state-of-the-art performance across three datasets with only 0.44M parameters.

## Background & Motivation

**Triple degradation in low-count PET**: (i) Poisson noise reduces signal-to-noise ratio; (ii) photon deficiency causes loss of structural detail; (iii) attenuation correction (AC) errors introduce systematic intensity bias. These three degradations are entangled in the spatial domain and difficult to disentangle.

**Common limitations of existing methods**: Whether iterative algorithms (OSEM), end-to-end networks (DeepPET), or post-processing methods (RED), all treat degradations indiscriminately in the spatial domain without exploiting their separability.

**Core finding (frequency domain analysis)**: Through amplitude-phase swap experiments and frequency bias profile analysis, the following is quantitatively verified:
   - Phase variance is concentrated in the high-frequency HH subband → corresponding to noise/photon deficiency
   - Amplitude bias dominates the low-frequency LL subband → corresponding to AC bias
   - Separately correcting amplitude and phase yields complementary improvements

## Method

### Overall Architecture

A frequency-aware regularization term is embedded into an ADMM optimization framework, unrolled over $K=3$ iterations to form a learnable network:

$$\min_x \underbrace{\mathcal{L}(\mathbf{A}x, y)}_{\text{data fidelity}} + \lambda_a \underbrace{\mathcal{R}_{\text{amp}}(|\mathcal{F}(x)|)}_{\text{low-freq. amplitude correction}} + \lambda_p \underbrace{\mathcal{R}_{\text{phase}}(\angle\mathcal{F}(x))}_{\text{high-freq. phase stabilization}}$$

Each ADMM iteration comprises three modules: x-update (SCM) → z-update (APCM) → u-update (DAM).

### Key Designs

1. **Spectral Consistency Module (SCM, x-update)**:

    - Spatial branch: parallel 3×3 and 5×5 depthwise separable convolutions for multi-scale local feature extraction
    - Frequency branch: State-Space Fourier Neural Operator (SSFNO), which applies SSD processing to the real and imaginary parts of the FFT output and propagates hidden state $h$ across iterations
    - Measurement consistency is enforced via the back-projection matrix $\mathbf{A}^\top$

2. **Amplitude-Phase Correction Module (APCM, z-update)**:

    - Haar DWT decomposes the input into four subbands: LL/HL/LH/HH
    - Amplitude branch: 1×1 DWConv + BN + GELU, with an additional FFN applied to the LL subband to recover low-frequency components suppressed by AC bias
    - Phase branch: $(cos\Phi, sin\Phi)$ encoding + high-frequency FFN for HH subband phase drift correction + cross-subband fusion
    - Corrected output is reconstructed via iFFT + iDWT

3. **Dual Adjustment Module (DAM, u-update)**: A learnable scalar $\mu$ replaces the fixed step size to adaptively control the dual ascent step.

### Loss & Training

$$\mathcal{L}_{total} = 0.5 \cdot \mathcal{L}_{Smooth\text{-}L1} + 0.3 \cdot \mathcal{L}_{SSIM} + 0.01 \cdot \mathcal{L}_{freq}$$

- Frequency loss: $\mathcal{L}_{freq} = |\mathcal{F}(x_{out}) - \mathcal{F}(x_{gt})|_1$
- Optimizer: AdamW, learning rate $10^{-3} \to 10^{-5}$ (cosine annealing)
- Unrolling depth $K=3$, inner iterations $\mathcal{N}=2$; training performed on a single RTX 4090

## Key Experimental Results

### Main Results (Three-Dataset Comparison)

| Method | Params | BrainWeb SSIM↑ | BrainWeb PSNR↑ | In-House SSIM↑ | In-House PSNR↑ | UDPET SSIM↑ | UDPET PSNR↑ |
|------|--------|---------------|---------------|---------------|---------------|------------|------------|
| OSEM | - | 0.9078 | 28.35 | 0.7456 | 23.59 | 0.7607 | 19.87 |
| FBPnet | 21.35M | 0.9327 | 33.62 | 0.9592 | 34.19 | 0.8907 | 27.36 |
| RED | 28.93M | 0.9664 | 34.45 | 0.9472 | 34.15 | 0.8890 | 26.51 |
| LCPR-Net | 75.93M | 0.9769 | 33.75 | 0.9222 | 34.95 | 0.8919 | 27.77 |
| **FourierPET** | **0.44M** | **0.9859** | **35.36** | **0.9740** | **35.19** | **0.9083** | **27.98** |

### Ablation Study (In-House Dataset)

| Configuration | SSIM↑ | PSNR↑ | RMSE↓ | Notes |
|------|-------|-------|-------|------|
| Baseline (conv blocks) | 0.940 | 33.15 | 0.0237 | No frequency modules |
| + SCM (replacing x-update) | 0.971 | 34.62 | 0.0200 | +1.47 PSNR |
| + APCM (replacing z-update) | 0.967 | 34.05 | 0.0210 | +0.90 PSNR |
| Full FourierPET | **0.974** | **35.19** | **0.0190** | SCM + APCM are complementary |

| SCM Sub-module Ablation | SSIM | PSNR | Notes |
|---------------|------|------|------|
| w/o $\mathbf{A}^\top$ | 0.8328 | 22.55 | **Catastrophic degradation**; measurement consistency is critical |
| w/o SSFNO | 0.9530 | 33.69 | Global spectral modeling is beneficial |
| w/o spatial module | 0.9681 | 34.43 | Local features are complementary |
| Full SCM | **0.9740** | **35.19** | All three components are indispensable |

### Key Findings

- **Remarkable parameter efficiency**: With only 0.44M parameters — 65× fewer than RED (28.93M) and 172× fewer than LCPR-Net (75.93M) — FourierPET achieves superior performance across all metrics.
- **$\mathbf{A}^\top$ is a critical constraint**: Removing it causes SSIM to drop sharply from 0.974 to 0.833, underscoring the necessity of physical measurement consistency.
- **Complementarity of amplitude and phase branches**: The phase branch alone improves SSIM (structural fidelity), while the amplitude branch alone reduces RMSE (global bias); their combination yields the best overall performance.
- **Zero-shot cross-domain generalization**: A model trained on human PET can be directly applied to mouse PET while maintaining high reconstruction quality.

## Highlights & Insights

1. The **frequency-domain degradation separability hypothesis** is highly insightful and is quantitatively validated through amplitude-phase swap experiments and DWT frequency bias profiling.
2. The combination of **ADMM unrolling and frequency-aware priors** preserves physical interpretability while retaining the flexibility of data-driven learning.
3. The extreme parameter efficiency of **0.44M parameters** is of substantial value for resource-constrained clinical deployment.
4. Zero-shot cross-species generalization suggests that frequency-domain degradation patterns are universal.

## Limitations & Future Work

- Validation is limited to brain and whole-body PET; other modalities such as cardiac PET remain unexplored.
- The unrolling depth $K=3$ is fixed; adaptive depth could potentially yield further improvements.
- The frequency-domain analysis assumes strictly separable degradations, whereas real-world scenarios may involve coupling effects.
- No comparison is made against recent generative reconstruction methods such as diffusion models.

## Related Work & Insights

- **Traditional iterative methods** (OSEM, MAP): Accurate physical modeling but slow computation and difficult prior design.
- **End-to-end methods** (DeepPET, CNNBPnet): Purely data-driven, lacking physical constraints.
- **Unrolled networks** (ADMM-Net, ISTA-Net): This work belongs to this category but is the first to introduce frequency-aware amplitude-phase decoupling.
- **Fourier Neural Operator**: This work integrates SSM and FNO into SSFNO for global spectral modeling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The frequency-domain degradation separability hypothesis is novel and thoroughly validated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + detailed ablations + cross-domain generalization + frequency-domain visualization.
- Writing Quality: ⭐⭐⭐⭐ Physical motivation is clearly articulated; module design logic is rigorous.
- Value: ⭐⭐⭐⭐ The frequency-domain decoupling paradigm is transferable to other inverse problems.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](../../NeurIPS2025/interpretability/shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](../../NeurIPS2025/interpretability/deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)

<!-- RELATED:END -->
