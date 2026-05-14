---
title: >-
  [Paper Note] Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors
description: >-
  [ICCV 2025][Signal & Communication][non-line-of-sight imaging] This paper proposes two modules — Learnable Path Compensation (LPC) and Adaptive Phasor Field (APF) — to address material-dependent radiance intensity fallof…
tags:
  - "ICCV 2025"
  - "Signal & Communication"
  - "non-line-of-sight imaging"
  - "learnable physical priors"
  - "path compensation"
  - "adaptive phasor field"
  - "SPAD"
date: 2026-05-08
content_hash: 602a4c42e604f9c2
---

# Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors

**Conference**: ICCV 2025
**arXiv**: [2409.14011](https://arxiv.org/abs/2409.14011)
**Code**: None
**Area**: Signal & Communications
**Keywords**: non-line-of-sight imaging, learnable physical priors, path compensation, adaptive phasor field, SPAD

## TL;DR

This paper proposes two modules — Learnable Path Compensation (LPC) and Adaptive Phasor Field (APF) — to address material-dependent radiance intensity falloff and frequency-domain denoising under varying SNR conditions in NLOS imaging, respectively. Trained solely on synthetic data, the method achieves state-of-the-art generalization across multiple real-world datasets.

## Background & Motivation

### State of the Field

Non-line-of-sight (NLOS) imaging captures indirect reflections from hidden objects using time-of-flight (ToF) systems comprising pulsed lasers and SPAD detectors, enabling perception beyond the direct line of sight. This technology holds significant promise for applications in autonomous driving, remote sensing, and medical diagnostics.

### Limitations of Prior Work

**Empirical priors for radiance intensity falloff (RIF) are inadequate**: Reflected photon intensity attenuates with distance, and the degree of attenuation depends on surface material. Existing methods apply a single fixed coefficient (e.g., $1/r^2$ for Lambertian surfaces, $1/r^4$ for retroreflectors) to compensate the entire scene, despite the fact that a single scene may contain multiple material types.

**Poor generalization**: Under low-SNR conditions (e.g., short acquisition times), Poisson noise induces high-frequency aliasing, causing traditional methods to produce heavy artifacts and learned methods to suffer catastrophic generalization failure.

**Diverse noise sources**: SPAD dark counts and ambient light are the two primary noise sources; SNR drops sharply as acquisition time decreases.

### Root Cause

Path compensation based on a single material assumption inevitably fails in mixed-material scenes — enhancing reconstruction for one material type simultaneously degrades the SNR for others. Furthermore, a fixed frequency-domain filter window cannot adapt to varying SNR conditions.

### Starting Point

Building upon the virtual-wave phasor field framework, the paper designs two learnable modules: LPC learns adaptive compensation coefficients for each scan point, while APF learns an adaptive Gaussian window bandwidth to select the effective spectrum of the transient measurement.

## Method

### Overall Architecture

Built on the LFE framework, the pipeline proceeds as follows given a transient measurement: (1) a feature extraction module downsamples the input and extracts features $F_E$; (2) the LPC module predicts adaptive compensation coefficients and outputs $F_C$; (3) the APF module learns the optimal frequency-domain window width and produces $F_A$; (4) a wave propagation and rendering module converts $F_A$ from the spatio-temporal domain to the spatial domain, yielding an intensity image and a depth map.

### Key Designs

#### 1. **Learnable Path Compensation (LPC)**

- **Function**: Predicts adaptive path compensation coefficients for each scan point in the transient measurement, replacing the globally fixed compensation.
- **Mechanism**:
  1. Three physically motivated compensation weights $\{(G_Z)^r, r=1,2,4\}$ are predefined, corresponding to attenuation magnitudes for different material types.
  2. Each compensation weight is element-wise multiplied with the enhanced feature to obtain initial compensated features: $F_C^{ini} = \{(G_Z)^1, (G_Z)^2, (G_Z)^4\} \otimes F_E'$
  3. A CNN predicts Softmax probabilities for each compensation weight; the final compensated feature is obtained via weighted summation.
- **Design Motivation**: Rather than directly regressing a continuous compensation coefficient (which is difficult to constrain), the module predicts a probability distribution over three physically grounded compensation modes, preserving physical interpretability while enabling adaptability. This is particularly effective for distant regions.

#### 2. **Adaptive Phasor Field (APF)**

- **Function**: Adaptively learns the Gaussian window standard deviation $\sigma_{pred}$ of the illumination function to dynamically select the effective spectral band of the transient measurement.
- **Mechanism**:
  1. The compensated feature $F_C$ is transformed to the frequency domain.
  2. Spatial and spectral convolutions are applied in the frequency domain to enhance features.
  3. A fully connected layer predicts the standard deviation $\sigma_{pred}$.
  4. An adaptive Gaussian kernel is generated: $K_G(\sigma) = \sigma\sqrt{2\pi} \exp(-\sigma^2\Omega^2/2)$
  5. Bandwidth relationship: $\Delta\Omega = \frac{1}{2\pi\sigma}$
- **Key Formula**: $F_A = \mathcal{F}^{-1}(\mathcal{F}(F_C) \cdot \mathcal{F}(\mathcal{P}(x_p, t)))$
- **Design Motivation**: High-SNR inputs require a wide bandwidth to preserve fine details, while low-SNR inputs require a narrow bandwidth to suppress noise. A fixed empirical standard deviation cannot satisfy both conditions; adaptive learning dynamically adjusts the bandwidth based on input signal quality.

#### 3. **Loss & Training**

The model is trained end-to-end; the total loss is a weighted sum of the intensity reconstruction loss and the depth estimation loss:

$$\mathcal{L} = \mathcal{L_I}(I, \hat{I}) + \lambda \mathcal{L_D}(D, \hat{D})$$

where both $\mathcal{L_I}$ and $\mathcal{L_D}$ are MSE losses and $\lambda=1$.

## Key Experimental Results

### Main Results (Synthetic Data, Seen Test Set)

| Method | Backbone | Memory | Time | PSNR↑ | SSIM↑ | RMSE↓ | MAD↓ |
|--------|----------|--------|------|-------|-------|-------|------|
| LCT | Physics | 18GB | 0.11s | 19.51 | 0.3615 | 0.4886 | 0.4639 |
| FK | Physics | 26GB | 0.16s | 21.69 | 0.6283 | 0.6072 | 0.5801 |
| RSD | Physics | 33GB | 0.23s | 21.74 | 0.1817 | 0.5677 | 0.5320 |
| LFE | CNN | 13GB | 0.05s | 23.27 | 0.8118 | 0.1037 | 0.0488 |
| I-K | CNN | 14GB | 0.08s | 23.44 | 0.8514 | 0.1041 | 0.0476 |
| NLOST | Trans. | 38GB | 0.38s | 23.74 | 0.8398 | 0.0902 | 0.0342 |
| **Ours** | CNN | 17GB | 0.24s | **23.99** | **0.8703** | **0.0874** | **0.0312** |

### Ablation Study (Generalization, Unseen Test Set, Varying SNR)

| Method | 10dB PSNR | 5dB PSNR | 3dB PSNR | 10dB RMSE | 3dB RMSE |
|--------|-----------|----------|----------|-----------|----------|
| LFE | 23.22 | 23.15 | 23.10 | 0.1036 | 0.1044 |
| I-K | 23.45 | 23.38 | 23.32 | 0.1045 | 0.1099 |
| NLOST | 23.63 | 23.74 | 23.71 | 0.0939 | 0.0918 |
| **Ours** | **23.91** | **23.83** | **23.80** | **0.0893** | **0.0902** |

**Contribution of each module** (qualitative results on real data):

| Configuration | Effect |
|---------------|--------|
| Baseline (w/o LPC, w/o APF) | Loss of fine details; significant background noise |
| + LPC only | Enhanced object details (e.g., deer legs), but background artifacts remain |
| + APF only | Background artifact suppression, but insufficient detail |
| + LPC + APF (full method) | Complete details with clean background |

### Key Findings

- **PSNR surpasses SOTA by 0.25 dB** (vs. NLOST) while requiring **only 45% of NLOST's memory** (17 GB vs. 38 GB).
- **Highest SSIM** (0.8703), indicating strong structural preservation.
- **Depth estimation RMSE reduced by 3.1% and MAD by 8.8%**, with particularly notable improvements in distant regions and mixed-material areas.
- **Stable cross-SNR generalization**: performance degradation is minimal even at extremely low SNR (3 dB).
- **Strong real-data generalization**: trained exclusively on synthetic data, the method achieves the best reconstruction quality on both the FK dataset (10-minute acquisition) and the NLOST dataset.
- **Self-collected data validation**: achieves best performance on three newly constructed complex real-world scenes containing mixed materials.

## Highlights & Insights

1. **Elegant integration of physical priors and data-driven learning**: LPC does not regress compensation coefficients as a black box; instead, it predicts a probability distribution over three physically grounded compensation modes, achieving both interpretability and adaptability.
2. **Frequency-domain adaptive denoising**: APF directly learns the filter window in the frequency domain, which is more efficient than spatial-domain denoising and carries clear physical meaning.
3. **Zero-shot synthetic-to-real transfer**: training solely on synthetic data generalizes to multiple real imaging systems, offering high practical utility.
4. **Efficiency advantage**: compared to the Transformer-based NLOST, the CNN backbone achieves superior performance while halving memory usage and inference time.
5. **Self-collected data contribution**: three new real-world scenes are constructed to increase diversity in NLOS data.

## Limitations & Future Work

1. **Restricted to confocal imaging systems**: extension to non-confocal systems is identified as a future direction.
2. **Gap between synthetic noise modeling and real sensors**: the noise model used for synthetic data may not fully match real-world acquisition conditions.
3. **Occlusion and multi-bounce reflections not handled**: as with all phasor field methods, the approach assumes no occlusion or inter-reflection within the scene.
4. **Quantitative evaluation limited to synthetic data**: the absence of ground truth for real data allows only qualitative comparisons.
5. **Sufficiency of three compensation exponents**: $r=1,2,4$ covers known material types, but may be inadequate for novel materials.

## Related Work & Insights

- **Phasor Field (RSD)** provides the theoretical foundation of the wave propagation framework; this work introduces learnable physical priors within that framework.
- **LFE** is the first physics-guided learning framework for NLOS; the proposed LPC and APF modules are embedded into it.
- **NLOST** leverages Transformers to capture global correlations at the cost of substantial memory and computational overhead.
- The key bottleneck in NLOS imaging is shifting from algorithmic design toward sensor quality and data availability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The designs of LPC and APF elegantly integrate physical priors with learning, though the overall framework builds upon the existing LFE pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation on synthetic, publicly available real, and self-collected real data across multiple SNR conditions.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical derivations are clearly presented; the motivation for each design choice is well articulated.
- **Value**: ⭐⭐⭐⭐ — Substantially improves the generalization capability and practical applicability of NLOS imaging, with particular value for low-SNR scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](rectifying_magnitude_neglect_in_linear_attention.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/signal_comm/estimation_of_stochastic_optimal_transport_maps.md)

</div>

<!-- RELATED:END -->
