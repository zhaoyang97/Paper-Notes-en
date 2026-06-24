---
title: >-
  [Paper Note] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes
description: >-
  [NeurIPS 2025][Image Restoration][MEMS gyroscope] This paper proposes MoE-Gyro, a self-supervised Mixture-of-Experts framework that simultaneously addresses the fundamental range–noise trade-off in MEMS gyroscopes via an Over-Range Reconstruction Expert (ORE, incorporating Gaussian-Decay Attention and physics-informed constraints) and a Denoising Expert (DE, incorporating dual-branch complementary masking and FFT-guided augmentation). The measurable range is extended from ±45…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "MEMS gyroscope"
  - "self-supervised learning"
  - "Mixture of Experts"
  - "over-range reconstruction"
  - "signal denoising"
date: 2026-05-08
content_hash: 129896638d3ee251
---

# MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes

**Conference**: NeurIPS 2025
**arXiv**: [2506.06318](https://arxiv.org/abs/2506.06318)  
**Code**: [GitHub](https://github.com/2002-Pan/Moe-Gyro)  
**Area**: Signal Processing / Inertial Navigation / Self-Supervised Learning
**Keywords**: MEMS gyroscope, self-supervised learning, Mixture of Experts, over-range reconstruction, signal denoising

## TL;DR

This paper proposes MoE-Gyro, a self-supervised Mixture-of-Experts framework that simultaneously addresses the fundamental range–noise trade-off in MEMS gyroscopes via an Over-Range Reconstruction Expert (ORE, incorporating Gaussian-Decay Attention and physics-informed constraints) and a Denoising Expert (DE, incorporating dual-branch complementary masking and FFT-guided augmentation). The measurable range is extended from ±450°/s to ±1500°/s, and bias instability is reduced by 98.4%.

## Background & Motivation

MEMS gyroscopes play a critical role in inertial navigation and motion control, yet face a **fundamental trade-off**:
- Increasing the measurement range → degraded noise characteristics (ARW and BI)
- Optimizing for low noise → limited angular velocity measurement capability

Limitations of prior approaches:

**Hardware solutions** (resonant frequency tuning, closed-loop force feedback, multi-range readout circuits): increased manufacturing complexity and cost

**Deep learning denoising methods** (CNN, LSTM-GRU): require precisely aligned ground-truth signals and address only noise without resolving the range limitation

**Self-supervised methods** (LIMU-BERT, IMUDB): lack a unified framework for jointly handling denoising and over-range reconstruction

**Existing evaluations lack a unified standard** for assessing multi-dimensional signal enhancement

## Method

### Overall Architecture

MoE-Gyro comprises three core components:
- **Lightweight gating module**: routes input signal segments to the appropriate expert based on simple heuristic rules
- **Over-Range Reconstruction Expert (ORE)**: processes saturated/clipped signal segments
- **Denoising Expert (DE)**: processes in-range noisy signal segments

Both experts share a MAE (Masked Autoencoder) backbone and are trained end-to-end in a purely self-supervised manner.

### Key Designs

1. **Gating and Routing Mechanism (Algorithm 1)**:

    - Detects peak segments by identifying three consecutive clipped samples
    - Detects noise segments by identifying consecutive samples below threshold $\tau$
    - Supports invoking a single expert only (saving ~50% GPU memory) or both simultaneously
    - Concatenates expert outputs into a complete enhanced signal

2. **Gaussian-Decay Attention (GD-Attn) in ORE**:

    - **Design Motivation**: Information required to recover clipped peaks is concentrated in a short temporal window around the peak; fixed windows ignore sensor-specific characteristics and are non-differentiable
    - **Core Idea**: Replaces binary window masks with learnable continuous Gaussian biases
    - **Formulation**: $B_{ij} = -\frac{d_{ij}^2}{2\sigma^2}$, where $\sigma$ is a trainable parameter
    - Output $= \text{softmax}(QK^\top/\sqrt{d_k} + B) \cdot V$
    - As $\sigma \to \infty$, the mechanism degrades to standard global attention; finite $\sigma$ smoothly down-weights distant token contributions
    - **Optimal placement**: Decoder only (versus Encoder-only, Enc.+Dec., and other configurations)

3. **Correlation Loss**:

    - Pure L2 reconstruction tends to smooth peaks and ignores local dynamics
    - First term: alignment of first-order differences (local slopes)
    - Second term: amplitude preservation at sign-change positions (peaks/valleys)
    - $\mathcal{L}_{\text{corr}} = \frac{1}{|\mathcal{M}|}\sum_{t\in\mathcal{M}}(\Delta x_t - \Delta \hat{x}_t)^2 + \lambda_{\text{sign}}\frac{1}{|\mathcal{E}|}\sum_{t\in\mathcal{E}}(x_t - \hat{x}_t)^2$

4. **Physics-Informed Energy Loss (PINN)**:

    - Derived from the displacement–power relationship of the IMU proof mass
    - Computes first- and second-order discrete derivatives to define instantaneous specific power
    - Applies sigmoid normalization followed by a barrier term to penalize excessively high or low energy
    - Ensures physical plausibility of reconstructed waveforms and improves cross-sensor generalization

5. **Dual-Branch Complementary Masking Strategy in DE**:

    - Both branches share all encoder and decoder weights
    - Constructs complementary 50% masks (cross pattern): even-indexed patches are visible to branch A and masked for branch B, and vice versa
    - Guarantees $\mathcal{M}_A \cup \mathcal{M}_B = \{1,\ldots,L\}$ and $\mathcal{M}_A \cap \mathcal{M}_B = \varnothing$
    - Fusion: $y_{\text{final}} = y_A \cdot \mathcal{M}_A + y_B \cdot \mathcal{M}_B$
    - Weight-sharing regularization extracts generalizable features and effectively suppresses high-frequency random noise

6. **FFT-Guided Training Augmentation**:

    - Noise floor estimation → weak signal injection (sampled and scaled from real IMU motion clips) → spectrum-matched noise synthesis
    - Physically grounded noise characteristics based on IEEE standards and Allan variance analysis
    - Synthesizes noise targeting QN-, ARW-, and BI-dominated frequency bands
    - Training objective: mixed signal as input; original noise plus scaled motion as clean reference

### Loss & Training

- Total loss = L2 reconstruction loss + Correlation loss + PINN energy loss (ORE)
- Denoising Expert uses L2 reconstruction loss + FFT-guided augmentation
- Purely self-supervised training; no temporally synchronized ground-truth labels required
- Trained on a single NVIDIA RTX-4060 GPU

## Key Experimental Results

### Main Results: ISEBench Full-Metric Comparison

| Model | PSNR↑ | P_MSE↓ | Corr↑ | SNR↑(dB) | QN↓ | ARW↓ | BI↓ | Avg. Rank |
|------|-------|--------|-------|---------|-----|------|-----|---------|
| RAW | 2.67 | 1 | - | 10.18 | 0 | 0 | 0 | - |
| Matlab 2023 | 6.03 | 0.515 | 0.86 | 10.02 | -25.8% | -3.1% | +5.4% | 7.0 |
| EMD | 5.44 | 0.655 | 0.77 | 13.85 | -91.1% | -85.9% | -96.8% | 5.3 |
| SG_filter | 4.35 | 0.767 | 0.79 | 12.03 | -85.0% | -86.3% | -90.0% | 6.6 |
| HEROS_GAN | 7.70 | 0.354 | 0.89 | 16.86 | -92.8% | -51.6% | -58.3% | 3.6 |
| IMUDB | 6.59 | 0.442 | 0.82 | 17.76 | -85.8% | -87.8% | -93.7% | 3.4 |
| **MoE-Gyro** | **8.29** | **0.325** | **0.92** | **24.19** | **-98.0%** | **-94.1%** | **-98.4%** | **1** |

### Ablation Study

**MoE Architecture Ablation:**

| Model | PSNR↑ | SNR↑ | GPU Mem↓ |
|------|-------|------|------|
| ORE+DE (no routing) | 8.19 | 24.58 | 129 MB |
| SingleNet (equivalent size) | 7.23 | 12.9 | 64.7 MB |
| **MoE-Gyro** | **8.29** | **24.19** | **71.3 MB** |

**GD-Attn Placement Ablation:**

| Configuration | PSNR↑ | P_MSE↓ | Corr↑ |
|------|-------|--------|-------|
| No GD-Attn | 8.08 | 0.345 | 0.87 |
| Encoder only | 8.03 | 0.345 | 0.88 |
| Enc.+Dec. | 8.27 | 0.335 | 0.90 |
| **Decoder only** | **8.29** | **0.324** | **0.92** |

**ORE Component Ablation:**

| Component Combination | PSNR↑ | P_MSE↓ | Corr↑ |
|---------|-------|--------|-------|
| None | 7.68 | 0.369 | 0.88 |
| GD-Attn only | 7.96 | 0.364 | 0.90 |
| Corr only | 7.83 | 0.354 | 0.91 |
| PINN only | 7.95 | 0.345 | 0.90 |
| All | **8.29** | **0.324** | **0.92** |

**DE Weight Sharing Ablation:**

| Weight Sharing | SNR↑ | GPU Mem↓ | Params↓ |
|------------|------|---------|---------|
| No sharing | 24.51 | 116.8 MB | 27.8M |
| Encoder only | 24.35 | 76.2 MB | 17.15M |
| **Enc.+Dec.** | **24.19** | **64.4 MB** | **13.9M** |

### Key Findings

1. **Range extension**: At an actual angular velocity of −1731.8°/s, MoE-Gyro reconstructs −1453.7°/s from a ±450°/s clipped signal.
2. **Cross-device generalization**: Trained on iPhone 14, zero-shot transfer to Huawei P70 and Xiaomi 14 yields favorable results (PSNR of 8.28 and 7.95, respectively).
3. **Cross-task generalization**: Stable performance across periodic swinging, jump impact, and high-frequency torsion motion categories.
4. **Real-time performance**: Full model processes a 2.56-second segment in 117 ms; compressed model achieves 41 ms with only 1.85M parameters.
5. **MoE architecture advantage**: Compared to a SingleNet of equivalent size, PSNR improves by 1.06 dB and SNR by 11.29 dB.

## Highlights & Insights

- **Novel problem formulation**: The first unified self-supervised framework to simultaneously address over-range reconstruction and denoising, breaking the long-standing range–noise trade-off.
- **Elegant GD-Attn design**: Learnable Gaussian decay replaces hard windows, enabling adaptive focus on peak regions.
- **PINN physical constraints**: Embeds MEMS spring-mass dynamics into the loss function to improve generalization.
- **Practical FFT-guided augmentation**: Physically grounded noise generation based on Allan variance standards.
- **ISEBench contribution**: The first open-source evaluation benchmark for IMU signal enhancement, unifying seven evaluation metrics.

## Limitations & Future Work

- The relatively large architecture poses deployment challenges for resource-constrained embedded devices (e.g., MCUs).
- Training data is sourced from a single device (iPhone 14), despite reasonable zero-shot generalization performance.
- Validation is limited to gyroscopes and has not been extended to other IMU sensors such as accelerometers.
- The 100 Hz sampling rate constrains applicability to certain high-frequency scenarios.
- The gating routing relies on simple heuristic rules, which may lack flexibility.

## Related Work & Insights

- Unlike HEROS-GAN (a fully supervised generative approach), MoE-Gyro is purely self-supervised.
- The MAE (Masked Autoencoder) backbone provides an effective framework for time-series signal processing.
- The MoE architecture successfully mitigates gradient conflicts inherent in multi-task learning.
- ISEBench can serve as a standard evaluation platform for future inertial signal enhancement research.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First unified self-supervised IMU signal enhancement framework with multiple novel designs)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive ablations, cross-device generalization, real-time performance, and real-scenario validation)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure with detailed algorithmic descriptions)
- **Value**: ⭐⭐⭐⭐ (Addresses a practical engineering problem; code and benchmark are open-sourced)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)
- [\[ICML 2025\] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](../../ICML2025/image_restoration/timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)

</div>

<!-- RELATED:END -->
