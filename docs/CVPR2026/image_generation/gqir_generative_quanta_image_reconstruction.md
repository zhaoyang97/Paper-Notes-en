---
title: >-
  [Paper Note] gQIR: Generative Quanta Image Reconstruction
description: >-
  [CVPR 2026][Image Generation][Single-photon imaging] This paper proposes gQIR, a modular three-stage framework that adapts large-scale text-to-image (T2I) diffusion models to the extreme photon-limited domain of SPAD sensors. It employs a quanta-aligned VAE (with a frozen encoder copy to prevent collapse), an adversarially fine-tuned LoRA U-Net for single-step generation, and a latent-space FusionViT for spatiotemporal fusion, enabling high-quality color image and video reconstruction from extremely sparse binary photon events.
tags:
  - CVPR 2026
  - Image Generation
  - Single-photon imaging
  - SPAD
  - diffusion models
  - "denoising & demosaicing"
  - low-light reconstruction
  - VAE alignment
  - burst fusion
date: 2026-05-08
content_hash: e16b0aeec02c0c7d
---

# gQIR: Generative Quanta Image Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2602.20417](https://arxiv.org/abs/2602.20417)
**Code**: [https://github.com/Aryan-Garg/gQIR](https://github.com/Aryan-Garg/gQIR)
**Area**: Image Generation / Computational Imaging
**Keywords**: Single-photon imaging, SPAD, diffusion models, denoising & demosaicing, low-light reconstruction, VAE alignment, burst fusion

## TL;DR
This paper proposes gQIR, a modular three-stage framework that adapts large-scale text-to-image (T2I) diffusion models to the extreme photon-limited domain of SPAD sensors. It employs a quanta-aligned VAE (with a frozen encoder copy to prevent collapse), an adversarially fine-tuned LoRA U-Net for single-step generation, and a latent-space FusionViT for spatiotemporal fusion, enabling high-quality color image and video reconstruction from extremely sparse binary photon events.

## Background & Motivation

**Background**: SPAD sensors can capture images under extremely low illumination and at ultra-high frame rates (10k–100k fps), recording only whether a photon was detected per pixel (Bernoulli distribution), resulting in highly sparse and noisy raw quanta frames. Existing reconstruction methods (QBP, QUIVER, QuDI) rely on alignment-and-merging strategies or task-specific networks, without leveraging large-scale generative priors.

**Limitations of Prior Work**: (a) Conventional denoising networks (NAFNet, Restormer) are designed for Poisson-Gaussian noise and suffer from severe over-smoothing under Bernoulli quantization noise; (b) existing generative restoration models (InstantIR) perform well on standard degradations but completely fail in the photon-limited domain (PSNR of only 7.9); (c) naively fine-tuning the VAE encoder of a diffusion model leads to **encoder collapse**—the trainable encoder simultaneously governs both prediction and supervision, causing rapid convergence to a constant output.

**Key Challenge**: A large domain gap exists between Bernoulli noise statistics and the continuous natural images used to train diffusion models, causing shortcut learning under direct fine-tuning.

**Key Insight**: The paper identifies the symmetric structure of the degradation-removal loss as the root cause of encoder collapse, and introduces a frozen encoder copy to break this symmetry.

**Core Idea**: A frozen encoder copy serves as an LSA anchor to prevent collapse, adversarial LoRA fine-tuning enables single-step generation, and a latent-space FusionViT performs spatiotemporal fusion.

## Method

### Overall Architecture
The input consists of SPAD binary frames (nano-burst: 7 frames averaged to yield 3-bit representations), and the output is a high-quality RGB image. Three stages are trained independently: Stage 1 fine-tunes the VAE encoder to achieve latent-space denoising and demosaicing alignment; Stage 2 adversarially fine-tunes a LoRA U-Net for single-step, high-perceptual-quality generation; Stage 3 trains a FusionViT for multi-frame spatiotemporal fusion in the latent space.

### Key Designs

1. **Imaging Model**:

    - Clean sRGB images are gamma-corrected as $x_{lin} = x_{gt}^{2.2}$ to map into linear irradiance space.
    - SPAD output follows a Bernoulli distribution: $x_{spad} = \text{Bern}(1 - e^{-\alpha \cdot x_{lin}})$, where $\alpha=1.0$ corresponds to an expected PPP of 3.5.
    - A random Bayer pattern is applied, and $N$ frames are averaged: $x_{lq} = \frac{1}{N}\sum_{i=1}^{N} M_\pi[\text{Bern}(1-e^{-\alpha \cdot x_{lin}})]$

2. **Quanta-Aligned VAE (Stage 1)**:

    - **Deterministic mean encoding**: $\mu_\phi(x_{lq})$ is used in place of stochastic sampling to avoid variance amplification under Bernoulli noise.
    - **Latent Space Alignment (LSA) loss**: $\mathcal{L}_{lsa} = \|\mu_{\phi^*}(x_{lq}) - \mu_\phi(x_{gt})\|_2^2$, where $\mu_\phi$ denotes a **frozen** copy of the original pretrained encoder that provides a stable anchor.
    - Supplemented by pixel-space MSE ($\lambda=10^3$) and LPIPS ($\lambda=2$) losses.
    - Total loss: $\mathcal{L} = 0.1\mathcal{L}_{lsa} + 10^3\mathcal{L}_{MSE} + 2\mathcal{L}_{perc}$
    - **Design Motivation**: Existing degradation-removal formulations place the trainable encoder on both the prediction and supervision sides, yielding a degenerate optimum. The frozen copy fundamentally breaks this symmetry.

3. **Adversarial LoRA Fine-Tuning (Stage 2)**:

    - A multi-scale ConvNeXt-Large discriminator $\mathcal{V}_\theta$ is used to perform standard min-max GAN training against a LoRA-initialized U-Net.
    - Generator total loss: $\mathcal{L}_{G} = \mathcal{L}_{adv} + \mathcal{L}_{perc} + \|\mathcal{D}(G_{lora}(\mu_{\phi^*}(x_{lq}))) - x_{gt}\|_2^2$
    - **Design Motivation**: The extremely high acquisition rate of SPAD sensors demands single-step inference; LoRA initialization inherits diffusion weights to provide a stable starting point for GAN training.

4. **Latent-Space Burst Fusion (Stage 3 — FusionViT)**:

    - All frames $Y$ are first reconstructed using S1+S2, and optical flow is then estimated in the reconstructed domain using RAFT (pretrained on FlyingThings3D), avoiding the domain gap introduced by low-quality inputs.
    - A pseudo-3D miniViT with sub-quadratic window attention adaptively fuses frames along temporal and spatial axes, preventing motion blur caused by naive averaging.
    - The output is added residually to the center-frame latent encoding via a learnable scalar $\delta=0.05$ before being passed to $\mathcal{G}_{lora}$.
    - Training loss: $\mathcal{L}_{fusion} = \|\mathcal{F}(\mu_{\phi^*}(X)) - \mu_\phi(x_{gt})\|_2^2 + \|\mathcal{D}(\mathcal{G}(\mathcal{F}(\mu_{\phi^*}(X)))) - x_{gt}\|_2^2 + \mathcal{L}_{perc}$

### Training Configuration
- Stage 1: 8×A100, 600k steps, batch=8, Adam (lr=1e-5), training data: 2.81M images + 44k videos
- Stage 2: 1×RTX4090, 100k steps, 256×256
- Stage 3: 20k steps, S1+S2 frozen, FusionViT only

## Key Experimental Results

### Main Results: Single-Frame 3-bit Color Reconstruction

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | ManIQA↑ | ClipIQA↑ | MUSIQ↑ |
|--------|-------|-------|--------|---------|----------|--------|
| InstantIR | 7.93 | 0.101 | 0.736 | 0.197 | 0.358 | 32.21 |
| ft-Restormer | 26.43 | 0.739 | 0.388 | 0.235 | 0.395 | 36.03 |
| ft-NAFNet | 26.88 | 0.757 | 0.338 | 0.251 | 0.431 | 36.73 |
| qVAE (S1) | 26.28 | 0.791 | 0.435 | 0.272 | 0.432 | 38.61 |
| **gQIR (S1+S2)** | 25.48 | 0.766 | **0.361** | **0.313** | **0.490** | **42.04** |

### Burst Reconstruction Comparison

| Test Set (fps) | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------------|--------|-------|-------|--------|
| I2-2000fps | QBP | 16.04 | 0.549 | 0.468 |
| I2-2000fps | QUIVER | 25.06 | 0.874 | 0.366 |
| I2-2000fps | **Burst-gQIR** | **31.21** | **0.878** | **0.296** |
| XD (2k-100k) | QBP | 12.78 | 0.409 | 0.458 |
| XD (2k-100k) | **Burst-gQIR** | **30.33** | **0.895** | **0.316** |

### Ablation Study: Stage 1 Design Choices

| Variant | PSNR↑ | SSIM↑ | ManIQA↑ |
|---------|-------|-------|---------|
| w/o det. encoding | 20.56 | 0.435 | 0.167 |
| w/o LSA loss | 10.39 | 0.222 | 0.139 |
| w/o both | 10.30 | 0.218 | 0.136 |
| **Full method** | **24.78** | **0.665** | **0.194** |

### Ablation Study: Incremental Three-Stage Contribution (Video Set)

| Stage | PSNR↑ | SSIM↑ | $E_{warp}$↓ |
|-------|-------|-------|-------------|
| S1 Alignment | 20.04 | 0.759 | 9.088 |
| S2 Perception | 24.11 | 0.846 | 8.508 |
| S3 Fusion | **27.63** | **0.869** | **8.005** |

### Key Findings
- The LSA loss is **indispensable**—removing it causes PSNR to drop sharply from 24.78 to 10.39, with 100% encoder collapse.
- gQIR surpasses SOTA QuDI by +2.17 dB on the I2-2000fps benchmark (30.81 vs. 28.64).
- Adversarial fine-tuning substantially improves perceptual metrics: ClipIQA 0.432→0.490 (+13.4%), MUSIQ 38.6→42.0.
- FusionViT contributes most significantly on the extreme-motion XD dataset, with burst reconstruction outperforming single-frame by approximately +5 dB.
- Reconstruction from real color SPAD data requires only gray-world white balancing, with no hot-pixel or dark-count correction needed.

## Highlights & Insights
- The **frozen encoder copy to prevent collapse** is the core contribution and can be directly transferred to any degradation-aware VAE fine-tuning scenario.
- **Deterministic encoding under Bernoulli noise** avoids variance amplification and offers general guidance for adapting diffusion models to non-Gaussian noise domains.
- The paper introduces the first **color SPAD burst dataset** and the XD video benchmark, filling a critical evaluation gap.
- The three-stage decoupled training allows independent optimization of each stage, making the framework highly practical for engineering applications.

## Limitations & Future Work
- Training assumes a fixed PPP of 3.5; generalization to extremely low light (PPP≤1) is limited, and PPP should be incorporated as a conditioning signal.
- The pretrained VAE decoder's 8-bit constraint limits the native HDR capability of SPAD sensors.
- Adversarial training is conducted only at 256×256; higher resolutions rely on VAE tiling.
- Stage 3 depends on a two-step "reconstruct-then-estimate-flow" strategy; an end-to-end approach may be superior.

## Rating
- Novelty: ⭐⭐⭐⭐ First successful adaptation of T2I diffusion priors to single-photon imaging
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Single-frame + burst + real data + new datasets + multiple metrics + complete ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous physical modeling
- Value: ⭐⭐⭐⭐ Opens a new direction for adapting generative priors to extreme imaging conditions

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](cycle-consistent_tuning_for_layered_image_decomposition.md)

<!-- RELATED:END -->
