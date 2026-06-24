---
title: >-
  [Paper Note] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach
description: >-
  [CVPR 2025][Image Restoration][Image Super-Resolution] PiSA-SR is proposed, which decouples pixel-level regression and semantic-level enhancement into two independent weight spaces via a dual-LoRA module. This achieves single-step diffusion for high-quality super-resolution and supports flexible adjustment of fidelity and perceptual quality at inference time using two guidance scales.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Image Super-Resolution"
  - "Dual LoRA"
  - "Adjustable Super-Resolution"
  - "Diffusion Models"
  - "Fidelity-Perception Trade-off"
date: 2026-05-08
content_hash: 8d4c0a93ed915446
---

# Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach

**Conference**: CVPR 2025  
**arXiv**: [2412.03017](https://arxiv.org/abs/2412.03017)  
**Code**: [GitHub](https://github.com/csslc/PiSA-SR)  
**Area**: Image Restoration  
**Keywords**: Image Super-Resolution, Dual LoRA, Adjustable Super-Resolution, Diffusion Models, Fidelity-Perception Trade-off

## TL;DR

PiSA-SR is proposed, which decouples pixel-level regression and semantic-level enhancement into two independent weight spaces via a dual-LoRA module. This achieves single-step diffusion for high-quality super-resolution and supports flexible adjustment of fidelity and perceptual quality at inference time using two guidance scales.

## Background & Motivation

Super-resolution based on diffusion priors faces a Key Challenge:
- **Pixel Fidelity vs. Perceptual Quality**: $\ell_2$ loss ensures fidelity but causes over-smoothing, whereas GAN/perceptual loss footprint enhances details but introduces artifacts.
- **Objective Entanglement**: Existing SD-based methods mix both objectives within a single diffusion process, making optimization difficult.
- **Diverse User Preferences**: Some users require content fidelity, while others favor rich semantic details, but models can only output a fixed style.

Core contribution: The SR problem is decomposed into two independently adjustable LoRA modules, allowing control over the SR style much like how CFG controls generation intensity.

## Method

### Overall Architecture

SD-based SR is formulated as residual learning $z_H = z_L - \lambda \epsilon_\theta(z_L)$, training two LoRAs on SD 2.1:
1. Pixel-LoRA ($\Delta\theta_{pix}$): Removes degradation using $\ell_2$ loss.
2. Semantic-LoRA ($\Delta\theta_{sem}$): Enhances semantic details using LPIPS + CSD loss.
During inference, the output intensities of the two LoRAs are controlled independently via $\lambda_{pix}$ and $\lambda_{sem}$.

### Key Design 1: Residual Learning Formulation

- **Function**: Simplifies multi-step diffusion SR into single-step residual prediction, supporting output scaling during inference.
- **Mechanism**: Single-step diffusion directly generates the HQ latent $z_H = z_L - \epsilon_\theta(z_L)$ from the LQ latent $z_L$, where the "noise" predicted by the U-Net is actually the residual from LQ to HQ. During training, $\lambda=1$ is fixed, while during inference, $\lambda$ is adjusted to control the enhancement intensity.
- **Design Motivation**: Residual learning allows the model to focus on high-frequency information, avoiding the extraction of irrelevant information from the LQ latent and accelerating convergence. More importantly, introducing the scaling factor $\lambda$ enables adjustability during inference.

### Key Design 2: Decoupled Dual-LoRA Training

- **Function**: Completely separates pixel fidelity and semantic enhancement into two distinct parameter spaces.
- **Mechanism**: First train the pixel-LoRA ($\ell_2$ loss, 4K steps). Once fixed, train only the semantic-LoRA (LPIPS + CSD loss, 8.5K steps) within the combined PiSA-LoRA framework. During inference, the output is decomposed following the CFG concept: $\epsilon_\theta = \lambda_{pix}\epsilon_{\theta_{pix}} + \lambda_{sem}(\epsilon_{\theta_{PiSA}} - \epsilon_{\theta_{pix}})$.
- **Design Motivation**: The sequential order of first removing degradation and then adding details ensures that semantic enhancement is not disturbed by noise or blur. The difference between the two LoRAs, $\epsilon_{\theta_{PiSA}} - \epsilon_{\theta_{pix}}$, precisely isolates the pure semantic enhancement component, achieving orthogonal control.

### Key Design 3: Replacing VSD with CSD Loss

- **Function**: Efficiently utilizes semantic priors from pre-trained SD for semantic enhancement.
- **Mechanism**: CSD (Classifier Score Distillation) loss extracts semantic gradients via the difference between conditional and unconditional noise predictions of SD, avoiding the bi-level optimization required by VSD. The gradient is $\nabla\ell_{CSD} = \mathbb{E}[w_t(f(z_t, \epsilon_{real}) - f(z_t, \epsilon_{real}^{\lambda_{cfg}}))]$.
- **Design Motivation**: While VSD is effective, its $\lambda_{cfg}=0$ component actually weakens semantic details; the CFG component of CSD is the core contributor to semantic enhancement. CSD bypasses the bi-level optimization of VSD, significantly reducing GPU memory consumption and training instability.

### Loss & Training

- Pixel-LoRA: $\mathcal{L}_{pix} = \|z_H^{pix} - z_{GT}\|_2^2$
- Semantic-LoRA: $\mathcal{L}_{sem} = \mathcal{L}_{LPIPS} + \mathcal{L}_{CSD}$

## Key Experimental Results

### Main Results: Effect of RealSR Guidance Scales

| $\lambda_{pix}$ | $\lambda_{sem}$ | PSNR↑ | LPIPS↓ | CLIPIQA↑ | MUSIQ↑ |
|---------|---------|-------|--------|----------|--------|
| 0.0 | 1.0 | 25.96 | 0.3426 | 0.4129 | 46.45 |
| 0.5 | 1.0 | 26.75 | 0.2646 | 0.5705 | 63.82 |
| 1.0 | 1.0 | 25.50 | 0.2672 | 0.6702 | 70.15 |
| 1.0 | 0.0 | ~28+ | ~0.35 | ~0.35 | ~40 |

Increasing $\lambda_{pix}$ eliminates degradation and improves PSNR, but values that are too large cause over-smoothing. Increasing $\lambda_{sem}$ enriches details and improves perceptual metrics, but excessive values introduce artifacts.

### Comparison with SOTA Methods (Synthetic Data)

Under default settings, PiSA-SR comprehensively outperforms or is competitive with methods such as StableSR, DiffBIR, SeeSR, and OSEDiff across metrics like PSNR, LPIPS, CLIPIQA, and MUSIQ, while requiring only a single-step diffusion.

### Key Findings

- The difference between the dual LoRAs indeed purely isolates semantic details (as clearly demonstrated by visualization).
- CSD outperforms VSD: more stable, lower memory footprint, and stronger semantic enhancement.
- Single-step inference is more stable and faster than multi-step methods.
- User studies validate the practical value of adjustable SR.

## Highlights & Insights

1. **Migration of CFG Concept to SR**: Applying the conditional/unconditional separation idea of CFG in generative models to the pixel/semantic separation in SR is a natural and innovative approach.
2. **Adjustability $\neq$ Retraining**: The two guidance scales can be adjusted directly during inference, eliminating the need to train separate models for different preferences.
3. **Generality of LoRA Decoupling**: The strategy of using dual LoRAs to decouple distinct optimization objectives can be generalized to other restoration tasks.

## Limitations & Future Work

- Adjustability requires running the U-Net twice during inference (once for pixel-LoRA and once for PiSA-LoRA).
- The optimal values of $\lambda_{pix}$ and $\lambda_{sem}$ vary across different images, meaning no universal optimal setting exists.
- Only supports $\times 4$ upscaling; retraining is required for other scales.
- Based on SD 2.1; exploration of SDXL or newer foundation models was not conducted.

## Related Work & Insights

- **OSEDiff**: A pioneer in single-step diffusion SR; PiSA-SR builds on it by introducing dual-LoRA decoupling.
- **SeeSR**: Diffusion SR guided by semantic tags; PiSA-SR achieves more direct semantic enhancement using CSD loss.
- **LDL**: A method utilizing local statistics constraints in GAN-based SR, which is related to the pixel-level LoRA objective of PiSA-SR.

## Rating

⭐⭐⭐⭐ — The concept of pixel/semantic decoupling is simple yet effective, offering strong practical utility for adjustable SR and high efficiency with single-step inference. The extra overhead of dual U-Net inference and parameter sensitivity are minor drawbacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](../../ECCV2024/image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](dpir_dual_prompting_restoration_dit.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)
- [\[CVPR 2025\] EQ-Reg: A Regularization-Guided Equivariant Approach for Image Restoration](a_regularization-guided_equivariant_approach_for_image_restoration.md)

</div>

<!-- RELATED:END -->
