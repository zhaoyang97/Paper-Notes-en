---
title: >-
  [Paper Note] Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution
description: >-
  [ICLR 2026][Image Generation][EEG super-resolution] This paper proposes SRGDiff, a step-aware residual-guided diffusion model that reformulates EEG spatial super-resolution as a dynamic conditional generation task…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "EEG super-resolution"
  - "residual-guided diffusion"
  - "step-aware modulation"
  - "brain-computer interface"
  - "conditional generation"
date: 2026-05-08
content_hash: d9f93e8a3b36b861
---

# Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution

**Conference**: ICLR 2026
**arXiv**: [2510.19166](https://arxiv.org/abs/2510.19166)  
**Code**: [GitHub](https://github.com/DhrLhj/ICLR2026SRGDiff)  
**Area**: Diffusion Models / EEG Signals / Super-Resolution
**Keywords**: EEG super-resolution, residual-guided diffusion, step-aware modulation, brain-computer interface, conditional generation

## TL;DR

This paper proposes SRGDiff, a step-aware residual-guided diffusion model that reformulates EEG spatial super-resolution as a dynamic conditional generation task, achieving high-fidelity reconstruction via per-step residual direction correction and timestep-dependent affine modulation.

## Background & Motivation

EEG (electroencephalography) is a non-invasive brain activity monitoring technique widely used in brain-computer interfaces, epilepsy diagnosis, and emotion recognition. However:

**Limited spatial resolution**: High-density (HD) systems are costly and inconvenient to wear; low-density (LD) systems (8–16 electrodes) are practical but suffer from severe spatial aliasing.

**Limitations of existing super-resolution methods**:
- Direct feature mapping methods (CNN/Transformer) oversimplify nonlinear dependencies, producing overly smooth results
- GAN-based methods require large amounts of data and computation
- Static conditioning strategies in diffusion models lead to a trade-off between distributional shift and distortion

**Core challenge**: The tension between fidelity (generating HD-like content) and consistency (agreement with LD observations).

## Method

### Problem Formulation

- Low-density EEG: $X^L \in \mathbb{R}^{C_L \times Length}$
- High-density EEG: $X^H \in \mathbb{R}^{C_H \times Length}$, $C_H > C_L$
- Objective: Recover $X^H$ from $X^L$

### 1. Latent Diffusion Backbone

- A pre-trained VAE encoder-decoder is trained on HD EEG data
- The loss comprises a reconstruction term, an STFT spectral fidelity term, and a KL regularization term
- VAE parameters are frozen after convergence

### 2. Residual Direction Module (RDM)

**Core Idea**: Learn to predict residual directions from LD inputs along the forward noising trajectory, serving as a per-step correction signal.

- Residual target: $\delta z_t = z_0 - z_t$ (difference between HD latent and noised latent)
- Lightweight convolutional predictor $R_\phi$ predicts the residual: $Res_t = R_\phi(\tau(t), c)$
- Residual loss: $\mathcal{L}_{res} = \sum_t \|Res_t - \delta z_t\|_2^2$
- Additive fusion: $\hat{z}_t^{RDM} = \text{LayerNorm}(\hat{z}_t) + Res_t$

### 3. Step-Aware Modulation Module (SMM)

Controls the influence of residual conditioning on denoising:

- Fuses LD features and timestep embeddings: $\widetilde{h}_t = \sigma_t h_t + (1-\sigma_t) e_t$
- Predicts channel-wise scale and bias: $\hat{z}_t^{SMM} = \gamma_t \odot \hat{z}_t^{RDM} + \beta_t^c$
- Weight $\sigma_t$ decays linearly with the timestep

### 4. Two-Stage Training

**Stage 1**: VAE pre-training (HD data only)
**Stage 2**: Residual-guided latent diffusion

$$\mathcal{L}_{\text{Stage 2}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2] + \lambda_{res}\sum_t\|R_\varphi(c,t) - (z_0 - z_t)\|_2^2 + \lambda_{SMM}(\|\gamma_t - 1\|_2^2 + \|\beta_t\|_2^2)$$

## Key Experimental Results

### Datasets
- **SEED**: 62 channels, 1000 Hz, emotion recognition (positive/neutral/negative)
- **SEED-IV**: 62 channels, 4 emotion classes
- **Localize-MI**: 256 channels, 8000 Hz, epileptic stimulation

### Main Results (Localize-MI)

| Method | 2× SNR | 4× SNR | 8× SNR | 16× SNR |
|--------|--------|--------|--------|---------|
| SaSDim | 5.74 | 4.38 | 3.55 | 2.77 |
| SADI | 5.75 | 4.37 | 3.55 | 2.89 |
| RDPI | 5.73 | — | — | — |
| ESTformer | baseline | baseline | baseline | baseline |
| STAD | baseline+ | baseline+ | baseline+ | baseline+ |
| **SRGDiff** | **best** | **best** | **best** | **best** |

### Key Findings
- Relative SNR improvement of approximately 75% on the most challenging 8× setting
- Significant improvements in both topographic map visualization and EEG-FID metrics
- Effectively mitigates spatial-spectral shift between low-density and high-density recordings

### Three-Level Evaluation Protocol
1. **Signal level**: SNR, NMSE, PCC (temporal consistency, spectral fidelity, spatial topology)
2. **Feature level**: EEG-FID (representation quality)
3. **Downstream level**: Classification accuracy

### Ablation Study

| Component | SNR Change |
|-----------|-----------|
| w/o RDM | Significant drop |
| w/o SMM | Moderate drop |
| Static conditioning (concatenation / cross-attention) | Below dynamic conditioning |
| Full SRGDiff | Best |

## Highlights & Insights

1. **Dynamic conditional generation paradigm**: Couples the LD forward noising trajectory with the HD reverse denoising trajectory
2. **Residual guidance direction**: Unlike static conditioning, provides directional correction at each step
3. **Comprehensive three-level evaluation**: Goes beyond pointwise error to cover signal, feature, and downstream task dimensions
4. **Robustness across datasets and upscaling factors**

## Limitations & Future Work

1. Requires VAE pre-training and two-stage training, resulting in a relatively complex pipeline
2. Relies on spatial correspondence between LD and HD channels
3. Diffusion model inference speed limits real-time BCI applications
4. Accuracy at extreme super-resolution factors (e.g., 16×) still has room for improvement

## Related Work & Insights

- **EEG super-resolution**: EEGSR-GAN, ESTformer, STAD, DDPM-EEG
- **Time-series diffusion**: Diffusion-TS, SaSDim, SADI
- **Residual diffusion**: PET-MRI residual synthesis, event-driven video residual reconstruction

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Residual guidance combined with step-aware modulation is novel in the EEG domain
- **Value**: ⭐⭐⭐⭐ — Significant practical value for low-cost BCI devices
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three-level evaluation protocol is comprehensively designed
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear with sufficient ablation analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](../../CVPR2026/image_generation/physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)

</div>

<!-- RELATED:END -->
