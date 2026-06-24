---
title: >-
  [Paper Note] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model
description: >-
  [ECCV 2024][Image Generation] This work proposes OmniSSR, the first zero-shot omnidirectional image super-resolution method based on diffusion models. By utilizing Octadecahedral Tangent Image Interaction (OTII) and Gradient Decomposition (GD) correction techniques, OmniSSR leverages the image prior of Stable Diffusion to achieve a balance between fidelity and realism, requiring no training or fine-tuning.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 2fd5652565a72a47
---

# OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2404.10312](https://arxiv.org/abs/2404.10312)  
**Area**: Image Generation

## TL;DR

This work proposes OmniSSR, the first zero-shot omnidirectional image super-resolution method based on diffusion models. By utilizing Octadecahedral Tangent Image Interaction (OTII) and Gradient Decomposition (GD) correction techniques, OmniSSR leverages the image prior of Stable Diffusion to achieve a balance between fidelity and realism, requiring no training or fine-tuning.

## Background & Motivation

Omnidirectional images (ODIs) are widely used in scenarios such as VR and surveillance, where super-resolution (SR) can enhance visual details. Existing omnidirectional image super-resolution (ODISR) methods face two major challenges:

**Limitations of End-to-End Methods**: Most methods (e.g., SphereSR, OSRT) can only produce deterministic outputs, yielding good data fidelity but poor perceptual quality (over-smoothing), and requiring large amounts of high-resolution ODIs for training (which are costly to collect).

**Difficulties in Direct Application of Diffusion Priors**: If an ERP image is divided into tangent projection (TP) images for independent super-resolution, inconsistencies will occur in overlapping regions when re-projecting back to ERP, disrupting global continuity.

The core idea of this work is to leverage the strong image prior of Stable Diffusion, iteratively transforming between ERP and TP to fuse information, and employing gradient decomposition correction to enforce consistency constraints.

## Method

### Overall Architecture

The workflow of OmniSSR consists of three steps:

1. **Preprocessing**: The low-resolution ERP image is upsampled and then projected into 18 TP images.
2. **Iterative Denoising**: In each denoising step, the TP images are denoised using SD (with a time-aware adapter), followed by OTII information interaction and GD correction.
3. **Post-processing**: The final TP images are back-projected to ERP format, and a final GD correction is applied.

### Key Designs

**Octadecahedral Tangent Image Interaction (OTII)**:

- A single ERP image is represented by 18 TP images.
- In each denoising step, intermediate results are alternately converted between ERP and TP formats: TP $\rightarrow$ ERP (information fusion) $\rightarrow$ TP (returning to the planar domain compatible with SD).
- This solves the global discontinuity problem caused by processing TP images independently.
- **Pre-upsampling Strategy**: Source images are bicubically upsampled before projection transformations, significantly mitigating information loss during projections.

**Gradient Decomposition (GD) Correction**:

Modeling super-resolution as an inverse problem $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}$, an analytical approximate solution is obtained via gradient descent:

$$\tilde{\mathbf{E}}_{0|t} = \mathbf{E}_{0|t} + \gamma \mathbf{A}^\dagger(\mathbf{E}_{init} - \mathbf{A}\mathbf{E}_{0|t})$$

This is decomposed into two terms:
- $\gamma \mathbf{A}^\dagger \mathbf{E}_{init}$: Ensures consistency with the input (fidelity).
- $(\mathbf{I} - \gamma \mathbf{A}^\dagger \mathbf{A})\mathbf{E}_{0|t}$: Iteratively updates the SD-generated results (realism).

The hyperparameter $\gamma$ balances fidelity and visual quality, set as $\gamma_p=1.0$, $\gamma_e=1.0$, and $\gamma_l=0.5$.

Concurrently, denoising results are fused in the latent space: $\tilde{\mathbf{z}}_{0|t}^{(i)} = (1-\gamma_l)\mathbf{z}_{0|t}^{(i)} + \gamma_l \mathcal{E}(\tilde{\mathbf{x}}_{0|t}^{(i)})$

### Loss & Training

OmniSSR is a zero-shot method and does not require training loss. Its core optimization is achieved during inference through GD correction, which inherently approximates gradient descent for the convex optimization problem $\arg\min_{\mathbf{x}} \|\mathbf{y} - \mathbf{Ax}\|_2^2 + \lambda \mathcal{R}(\mathbf{x})$.

## Key Experimental Results

### Main Results

**Quantitative Comparison on ODI-SR and SUN 360 Datasets ($\times 4$ Super-Resolution)**:

| Method | WS-PSNR↑ | WS-SSIM↑ | FID↓ | LPIPS↓ |
|------|----------|----------|------|--------|
| Bicubic | 25.43 | 0.7059 | 50.84 | 0.3755 |
| DDRM | 25.43 | 0.7367 | 32.69 | 0.3206 |
| DPS | 24.75 | 0.6594 | 120.74 | 0.4911 |
| GDP | 23.16 | 0.6692 | 77.43 | 0.4260 |
| PSLD | 21.72 | 0.5498 | 107.99 | 0.5329 |
| StableSR | 23.33 | 0.6577 | 49.95 | 0.3135 |
| **Ours** | **25.77** | **0.7279** | **30.97** | **0.2977** |

**Comparison with End-to-End Supervised Methods ($\times 4$ SR, ODI-SR)**:

| Method | WS-PSNR↑ | FID↓ | LPIPS↓ | NIQE↓ | DISTS↓ |
|------|----------|------|--------|-------|--------|
| SwinIR | 26.76 | 27.94 | 0.3321 | 5.3961 | 0.1710 |
| OSRT | 26.89 | 27.39 | 0.3258 | 5.4364 | 0.1695 |
| **Ours** | 25.77 | **30.97** | **0.2977** | **5.2891** | **0.1541** |

### Ablation Study

**Step-by-step Ablation of OTII and GD Correction (ODI-SR $\times 2$)**:

| Input Type | OTII | GD | WS-PSNR↑ | FID↓ | LPIPS↓ |
|---------|------|-----|----------|------|--------|
| ERP | ✗ | ✗ | 22.69 | 44.87 | 0.3039 |
| TP | ✗ | ✗ | 23.53 | 43.91 | 0.3113 |
| TP | ✓ | ✗ | 23.74 | 65.35 | 0.3748 |
| TP | ✗ | ✓ (post-processing only) | 26.77 | 15.41 | 0.1691 |
| TP | ✓ | ✓ | **28.58** | **13.01** | **0.1575** |

### Key Findings

- Under the zero-shot setting, OmniSSR achieves fidelity metrics (WS-PSNR) close to supervised methods, while fully outperforming them in perceptual quality metrics (LPIPS, NIQE, DISTS).
- The combined effect of OTII + GD far exceeds using them individually, with PSNR increasing from 22.69 to 28.58 (+5.89 dB).
- The pre-upsampling strategy increases the PSNR of ERP $\leftrightarrow$ TP transformations from 28.98 to 38.18 (under the (4,4) setting), greatly mitigating the information loss from projection transformations.
- Applying GD correction at every denoising step yields better performance than applying it only during post-processing.

## Highlights & Insights

1. **Zero-shot Paradigm**: Completely eliminates the need for omnidirectional image training data, leveraging off-the-shelf planar image SD priors and resolving the scarcity of high-resolution ODI data.
2. **Ingenious OTII Design**: Fuses global information during the denoising process through iterative ERP $\leftrightarrow$ TP transformations, compensating for the lack of continuity when processing TP images independently.
3. **Clear GD Correction Theory**: Based on a gradient descent framework for convex optimization, providing an analytical solution to the fidelity-realism trade-off, where the hyperparameter $\gamma$ has a clear physical meaning.
4. **General Extensibility**: The framework can be extended to tasks such as ODI editing, ODI inpainting, and 3D Gaussian Splatting enhancement.

## Limitations & Future Work

- Slow inference speed: It takes approximately 14 minutes for a single $1024\times2048$ ERP image, making real-time super-resolution challenging.
- Multiple ERP $\leftrightarrow$ TP transformations consume extra inference time.
- The hyperparameter $\gamma$ in GD correction requires manual grid search tuning and is not yet adaptive.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty: ★★★★★ — The first zero-shot ODISR method using a diffusion model.
- Technical depth: ★★★★ — Rigorous theoretical derivation of OTII and GD, leading to significant combined performance.
- Experimental thoroughness: ★★★★ — Evaluated across multiple datasets and metrics with thorough ablation.
- Practicality: ★ stars — Inference speed limits practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)

</div>

<!-- RELATED:END -->
