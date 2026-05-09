---
title: >-
  [Paper Note] CuMoLoS-MAE: A Masked Autoencoder for Remote Sensing Data Reconstruction
description: >-
  [NEURIPS2025][Autonomous Driving][Masked Autoencoder] This paper proposes CuMoLoS-MAE, a Masked Autoencoder combining a curriculum masking strategy with Monte Carlo stochastic ensemble inference for high-fidelity reconstruction and pixel-wise uncertainty quantification of remote sensing atmospheric profile data.
tags:
  - NEURIPS2025
  - Autonomous Driving
  - Masked Autoencoder
  - Remote Sensing Data Reconstruction
  - Uncertainty Quantification
  - Curriculum Learning
  - Monte Carlo Ensemble
date: 2026-05-08
content_hash: 9894df10d5da016f
---

# CuMoLoS-MAE: A Masked Autoencoder for Remote Sensing Data Reconstruction

**Conference**: NEURIPS2025  
**arXiv**: [2508.14957](https://arxiv.org/abs/2508.14957)  
**Code**: To be confirmed  
**Area**: Autonomous Driving  
**Keywords**: Masked Autoencoder, Remote Sensing Data Reconstruction, Uncertainty Quantification, Curriculum Learning, Monte Carlo Ensemble  

## TL;DR

This paper proposes CuMoLoS-MAE, a Masked Autoencoder combining a curriculum masking strategy with Monte Carlo stochastic ensemble inference for high-fidelity reconstruction and pixel-wise uncertainty quantification of remote sensing atmospheric profile data.

## Background & Motivation

- Atmospheric profile data acquired by remote sensing instruments (Doppler lidars, radars, radiometers, etc.) are frequently corrupted by low-SNR conditions, range folding, and spurious discontinuities, resulting in large numbers of missing or distorted return values.
- Traditional gap-filling methods (e.g., sliding-window mean filtering) blur critical fine-scale structures such as wind shear lines, updraft/downdraft cores, and small eddies.
- Existing deep learning methods (e.g., VAEs) can recover sharper structures but provide no reconstruction uncertainty estimates, limiting their reliable use in data assimilation and early-warning systems.
- Consequently, a reconstruction approach is needed that simultaneously recovers fine atmospheric structures and provides pixel-level confidence information.

## Core Problem

1. **Fine Structure Recovery**: How to preserve key atmospheric features such as updraft cores and wind shear lines during denoising/inpainting?
2. **Uncertainty Quantification**: How to provide reliable per-pixel confidence estimates to support downstream data assimilation and early-warning decisions?
3. **Training Efficiency**: How to enable stable learning of reconstruction capability from sparse contexts?

## Method

### Overall Architecture

CuMoLoS-MAE (Curriculum-Guided Monte Carlo Stochastic Ensemble Masked Autoencoder) consists of two core stages: a curriculum-masked MAE during training and a Monte Carlo ensemble during inference.

### Micro-Patch MAE with Curriculum Masking

- **Input Patching**: The Doppler lidar time–height array is divided into 64×64 image patches, and each patch is further tokenized into 2×2 micro-patches to capture fine-scale structures and mesoscale dynamics.
- **Encoder–Decoder Architecture**: The encoder is a 12-layer ViT that processes only the unmasked visible tokens; the decoder is a lightweight 4-layer structure that reconstructs the full field from the visible tokens.
- **Curriculum Masking Strategy**:
    - The masking ratio is fixed at 50% for the first 5 epochs.
    - It is then progressively increased to 70% via cosine annealing, reaching 70% at epoch 30.
    - The ratio remains at 70% thereafter.
    - This progressive strategy forces the model to incrementally learn reconstruction from increasingly sparse contexts.
- **Loss Function**: MSE loss is computed only on masked pixels.

### Monte Carlo Ensemble Inference

- At inference time, $N=50$ independent random masks are sampled for each input image.
- The full mask → encode → decode pipeline is executed separately for each mask.
- The 50 reconstruction results are aggregated as follows:
    - **Mean** $\bar{X} = \frac{1}{N}\sum_{i=1}^{N}\hat{X}^{(i)}$ serves as the high-fidelity denoised reconstruction.
    - **Standard deviation** $\sigma_X = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{X}^{(i)} - \bar{X})^2}$ serves as the pixel-wise uncertainty map.

### Training Details

- **Data Preprocessing**: SNR filtering is applied (intensity ≥ 0.005), and valid velocities are clipped to $[-5, 5]$ m/s.
- **Training Data**: ARM SGP site data from June 1–9, 2011; the test set is June 15, 2011 (unseen data).
- **Optimizer**: AdamW (learning rate $1.5 \times 10^{-4} \cdot \frac{32}{256}$, weight decay 0.05).
- **Training Configuration**: 500 epochs, batch size 32, single NVIDIA A100 GPU.
- **Learning Rate Schedule**: Cosine schedule with warmup aligned to the masking curriculum.

## Key Experimental Results

### Main Results (1,028 Test Images)

| Method | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | FID ↓ | Spectral Fidelity ↑ |
|--------|------------|--------|-------|-------|---------------------|
| 8×8 Mean Filter | 23.41 | 0.4950 | 0.5186 | 5.13 | 91.67% |
| CVAE | 26.70 | 0.4190 | 0.4036 | 3.28 | 80.21% |
| DnCNN (Noise2Void) | 23.09 | 0.6466 | 0.6232 | 0.12 | 36.46% |
| U-Net (Noise2Void) | 27.70 | 0.7016 | 0.2581 | 0.44 | 49.48% |
| **CuMoLoS-MAE** | **29.45** | **0.7857** | **0.1854** | 1.87 | **93.75%** |

- CuMoLoS-MAE achieves the best performance on PSNR, SSIM, MSE, and spectral fidelity; its PSNR is 1.75 dB higher than the second-best U-Net.
- Low-frequency spectral fidelity reaches 93.75%, far surpassing the Noise2Void variants (36%–49%), preserving storm-scale energy.

### Uncertainty Quantification Quality

- Pearson correlation between the Monte Carlo standard deviation $\sigma_X$ and the absolute reconstruction error: **r = 0.961 ± 0.037**
- Global Spearman rank correlation: **ρ = 0.926**
- Per-pixel MAE sorted by $\sigma$ quantile increases monotonically from 0.028 to 0.999 (a 35.1× range).
- The top 1%/5%/10%/20% of pixels by $\sigma$ capture 10.1%/30.6%/43.4%/59.4% of the total error, respectively.
- These results demonstrate that the uncertainty estimates are highly reliable and can be effectively used for error triage.

### Ablation Study: Temporal Window Size

| Window Size | PSNR ↑ | SSIM ↑ | MSE ↓ | Spectral Fidelity ↑ |
|-------------|--------|--------|-------|---------------------|
| 64×64 | **29.45** | **0.7857** | **0.1854** | **93.75%** |
| 128×64 | 30.11 | 0.7697 | 0.2253 | 87.50% |
| 256×64 | 28.55 | 0.6103 | 0.3205 | 38.02% |

- The 64×64 window already provides sufficient context, suggesting that denoising is primarily a local process.
- Larger windows introduce more tokens and masked regions without increasing model capacity, leading to performance degradation.

### Ablation Study: Curriculum Masking

- Curriculum masking allows the reconstruction loss to fall below 0.20 approximately 26 epochs earlier (epoch 198 vs. 224), improving training efficiency by roughly 10%.
- Final metrics are approximately equivalent; the primary benefit of curriculum masking lies in accelerating convergence.

## Highlights & Insights

1. **Elegant Monte Carlo Ensemble Design**: By sampling multiple random masks independently at inference time, the method approximates the posterior predictive distribution without any modification to model architecture or training procedure, yielding high-quality uncertainty estimates.
2. **Curriculum Masking Strategy**: Progressively increasing the masking ratio enables the model to smoothly learn reconstruction from extremely sparse contexts, accelerating convergence.
3. **Highly Reliable Uncertainty Estimates**: The correlation between $\sigma$ and true error reaches 0.961, which is of high practical value for remote sensing data assimilation and extreme weather early warning.
4. **Micro-Patch Design**: The 2×2 micro-patch tokenization captures fine atmospheric structures more effectively than standard 16×16 patching, making it more suitable for physical field data of this type.
5. **Spectral Fidelity Metric**: The paper introduces a PSD-based low-frequency fidelity evaluation metric that better reflects physical field reconstruction quality than conventional pixel-level metrics.

## Limitations & Future Work

1. **Very Small Data Scale**: Training uses only nine days of data from a single site (ARM SGP) with one-day testing, raising concerns about generalizability.
2. **High Inference Cost**: Each image requires 50 forward passes, imposing substantial computational overhead for real-time deployment.
3. **Single Variable Validation**: Reconstruction is evaluated only on vertical velocity fields; other atmospheric variables such as temperature and humidity are not tested.
4. **Questionable Area Classification**: This paper belongs to remote sensing/meteorological data reconstruction rather than autonomous driving.
5. **Missing Stronger Baselines**: No comparison is made with recent diffusion model or Flow Matching denoising approaches.
6. **Lack of Cross-Sensor Validation**: Noise characteristics vary significantly across lidar models, necessitating additional cross-sensor experiments.

## Related Work & Insights

| Compared Method | Advantages | Disadvantages |
|----------------|------------|---------------|
| Sliding-Window Mean Filter | Simple and fast | Blurs fine structures; PSNR only 23.41 |
| CVAE | Capable of generating sharp structures | No uncertainty estimation; lowest SSIM (0.419) |
| Noise2Void (DnCNN/U-Net) | No paired data required; best FID | Very poor spectral fidelity (36%–49%); severely loses low-frequency information |
| CuMoLoS-MAE | Best reconstruction quality + reliable uncertainty + high spectral fidelity | High inference cost (50 samples) |

The Monte Carlo ensemble concept is transferable to other MAE applications (e.g., medical imaging, remote sensing inpainting) for obtaining uncertainty estimates without modifying the model. The curriculum masking strategy offers a useful reference for other MAE variants requiring high masking ratios during training. The PSD-based spectral fidelity metric is worth adopting more broadly in physical field reconstruction tasks. The uncertainty-weighted strategy in data assimilation can be applied to point cloud completion and sensor fusion in autonomous driving perception.

## Rating

- Novelty: 3.5/5 — The combination of Monte Carlo ensemble and curriculum masking is moderately novel, though neither component is entirely new.
- Experimental Thoroughness: 2.5/5 — Data scale and baseline coverage are insufficient; ablation experiments are relatively simple.
- Writing Quality: 4/5 — Structure is clear, with good coordination between formulas and visualizations.
- Value: 3/5 — Practically valuable for the meteorological remote sensing domain, but scale and generalizability remain to be verified.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [\[NeurIPS 2025\] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](../../AAAI2026/autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)

<!-- RELATED:END -->
