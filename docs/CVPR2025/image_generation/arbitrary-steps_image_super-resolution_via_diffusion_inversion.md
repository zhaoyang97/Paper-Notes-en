---
title: >-
  [Paper Note] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion
description: >-
  [CVPR 2025][Image Generation][Diffusion Inversion] This paper proposes InvSR, which achieves diffusion inversion by training a noise prediction network. Utilizing the image prior of a pre-trained diffusion model for super-resolution, it supports arbitrary-step sampling from 1 to 5 steps, achieving or exceeding the performance of existing state-of-the-art (SOTA) methods even with single-step sampling.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Inversion"
  - "Image Super-Resolution"
  - "Noise Prediction"
  - "Partial Noise Prediction"
  - "Arbitrary-Step Sampling"
date: 2026-05-08
content_hash: f5fb9156eb64ad32
---

# Arbitrary-Steps Image Super-Resolution via Diffusion Inversion

**Conference**: CVPR 2025  
**arXiv**: [2412.09013](https://arxiv.org/abs/2412.09013)  
**Code**: [https://github.com/zsyOAOA/InvSR](https://github.com/zsyOAOA/InvSR)  
**Area**: Diffusion Models / Image Super-Resolution  
**Keywords**: Diffusion Inversion, Image Super-Resolution, Noise Prediction, Partial Noise Prediction, Arbitrary-Step Sampling

## TL;DR

This paper proposes InvSR, which achieves diffusion inversion by training a noise prediction network. Utilizing the image prior of a pre-trained diffusion model for super-resolution, it supports arbitrary-step sampling from 1 to 5 steps, achieving or exceeding the performance of existing state-of-the-art (SOTA) methods even with single-step sampling.

## Background & Motivation

Image super-resolution (SR) is a fundamental problem in computer vision, aiming to recover a high-resolution (HR) image from a low-resolution (LR) image. Due to the complexity and unknown nature of degradation models in real-world scenarios, SR is inherently an ill-posed problem.

In recent years, large-scale text-to-image (T2I) diffusion models have demonstrated powerful image generation capabilities, leading to various attempts to leverage them as priors to alleviate the ill-posedness of SR. Existing methods mainly fall into two categories: one ensures consistency with LR images by optimizing intermediate features (e.g., DDRM, DDNM), which is computationally complex and relies on known degradation models; the other directly fine-tunes T2I models (e.g., StableSR, DiffBIR, SeeSR), which offers good performance but requires modifying the diffusion network structure.

The **Key Challenge** is: existing methods either require modifying the intermediate features/parameters of the diffusion model, which fails to fully utilize the diffusion prior, or require a fixed number of sampling steps, lacking flexibility. While GAN inversion has been applied in SR, multi-step stochastic sampling in diffusion models makes inversion more difficult—directly optimizing the noise map of each step incurs high computational overhead, and the iterative process accumulates prediction errors.

The **Key Insight** of this paper is: LR and HR images only differ in high-frequency details, and they must become indistinguishable after adding appropriate noise. Therefore, the intermediate state of the diffusion process can be constructed using the LR image plus the predicted noise, thereby achieving SR without modifying the diffusion network. The core idea is "Partial Noise Prediction" (PnP)—predicting the noise of only the starting step, while using random noise for subsequent steps.

## Method

### Overall Architecture

The pipeline of InvSR is as follows: given an LR image, a noise prediction network is used to estimate a noise map, which is then combined with the LR image according to the forward diffusion process to construct the intermediate state of the diffusion model as the sampling starting point. Then, the pre-trained diffusion model (SD-Turbo) is utilized to perform reverse sampling starting from this intermediate state to generate the HR image. The entire process does not modify the parameters of the diffusion network.

### Key Designs

1. **Partial Noise Prediction Strategy (PnP)**:

    - Core Observation: The primary difference between LR and HR images lies in the high-frequency components, and they become nearly indistinguishable after adding an appropriate level of noise.
    - Instead of predicting noise maps for all diffusion steps, it only predicts the noise map of the starting step (time step $t \le 250$, corresponding to SNR > 1.44).
    - Random Gaussian noise is used for intermediate steps (since the pre-trained diffusion model is already robust at low noise levels).
    - This simplifies the set of noise maps from $T = 1000$ to just 1, significantly reducing the prediction difficulty.
    - Predicts noise at the starting step allowing a non-zero mean (since the LR image replaces the HR image), where visualization shows that this noise is correlated with the LR image.

2. **Arbitrary-Step Sampling Mechanism**:

    - The noise prediction network is trained on multiple pre-selected starting steps using time embeddings, with the training set being $\{250, 200, 150, 100\}$.
    - During inference, users can freely select the starting step and combine it with accelerated sampling algorithms (such as DDIM) to achieve 1-to-5-step sampling.
    - Larger starting steps (e.g., 250) involve more sampling steps, generating richer details, which is suitable for blur degradation.
    - Smaller starting steps (e.g., 100) can finish in a single step, avoiding noise amplification, which is suitable for heavy noise degradation.
    - Users can flexibly select the number of steps based on the degradation type to achieve a balance between fidelity and realism.

3. **Noise Prediction Network**:

    - Based on the VQGAN encoder architecture, consisting of two downsampling blocks, each equipped with self-attention layers.
    - Utilizes the VAE reparameterization trick to predict the mean and variance of a Gaussian distribution instead of directly predicting the noise map.
    - Takes the LR image and time step as inputs and outputs the corresponding noise map.
    - The parameter size is only 33.84M, which is much smaller than existing diffusion-based SR methods.

### Loss & Training

The training loss consists of three parts:
- **L2 Loss**: Based on the MSE between the single-step denoising result of the diffusion model on the intermediate state and the GT HR image.
- **LPIPS Loss**: Perceptual loss with weight $\lambda_l = 2.0$, computed in the latent space (with fine-tuning).
- **GAN Loss**: Adversarial loss with weight $\lambda_g = 0.1$, using hinge loss with a discriminator based on the diffusion UNet with a multi-input multi-output strategy.

Training details:
- The base model is SD-Turbo, and all losses are computed in the VQGAN latent space to save GPU memory.
- Dataset: LSDIR + 20k FFHQ face subset, with degradations synthesized using the RealESRGAN pipeline.
- Adam optimizer, fixed learning rate of $5 \times 10^{-5}$, batch size of 64, trained for 100k+ iterations.
- The starting step is randomly selected from $\{250, 200, 150, 100\}$ for training at each iteration.

## Key Experimental Results

### Main Results

| Dataset | Metric | InvSR-1 | OSEDiff-1 | SinSR-1 | ResShift-4 | SeeSR-50 | Description |
|--------|------|---------|-----------|---------|------------|----------|------|
| ImageNet-Test | LPIPS↓ | 0.2517 | 0.2624 | 0.2209 | 0.1998 | 0.2187 | Single-step sampling |
| ImageNet-Test | NIQE↓ | **4.38** | 4.72 | 5.26 | 5.87 | 4.38 | Best reference-free |
| ImageNet-Test | CLIPIQA↑ | **0.709** | 0.682 | 0.662 | 0.615 | 0.587 | Best perceptual quality |
| ImageNet-Test | MUSIQ↑ | **72.29** | 70.39 | 67.76 | 65.59 | 71.24 | Best quality score |
| RealSR | NIQE↓ | **4.22** | 5.33 | 6.25 | 6.91 | 4.54 | Best on real data |
| RealSR | CLIPIQA↑ | 0.692 | **0.701** | 0.663 | 0.599 | 0.682 | Near-optimal |

InvSR comprehensively leads in reference-free perceptual quality metrics with just a single step. Furthermore, its parameter count (33.84M) is much smaller than SeeSR (751.5M) and DiffBIR (385.4M).

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | NIQE↓ | CLIPIQA↑ | Description |
|------|-------|--------|-------|----------|------|
| 5 steps $\{250,200,150,100,50\}$ | 22.70 | 0.2844 | 4.88 | 0.673 | Max steps, max details |
| 3 steps $\{250,150,50\}$ | 22.92 | 0.2762 | 4.80 | 0.682 | Balanced configuration |
| 3 steps $\{150,100,50\}$ | 23.84 | 0.2575 | 4.22 | 0.702 | Better starting with high SNR |
| 1 step $\{250\}$ | 23.84 | 0.2575 | 4.53 | 0.713 | High fidelity |
| 1 step $\{100\}$ | 24.66 | 0.2450 | 4.06 | 0.691 | High PSNR |

### Key Findings

- More steps are not necessarily better: For heavily noisy inputs, single-step sampling actually performs better than multi-step sampling (which might amplify noise).
- Selection of the starting step affects the fidelity-realism balance: High starting steps (250) favor detail restoration, while low starting steps (100) favor fidelity preservation.
- Intermediate steps do not require noise prediction: Under high SNR inheritance, the pre-trained diffusion model is sufficient to handle Gaussian noise in the intermediate steps.
- Significant inference speed advantage: A single step takes only 168ms on an A100 ($128 \to 512$), while StableSR-50 requires 1730ms.

## Highlights & Insights

- **Elegant Design Philosophy**: The core insight "LR and HR are indistinguishable after adding noise" is simple yet profound, simplifying complex full-process inversion into single-step noise prediction.
- **Unique Flexibility**: It is the first diffusion-based SR method that supports arbitrary-step sampling, allowing users to adaptively choose parameters based on the degradation type.
- **Extremely High Efficiency**: Completely preserves the pre-trained diffusion model without modification and only adds a lightweight noise predictor (33.84M), enabling fast inference speed.
- **Methodological Contribution**: The PnP strategy is not only suitable for SR but can, in principle, be generalized to other image restoration tasks based on diffusion inversion.

## Limitations & Future Work

- Text prompts are fixed to generic descriptions, falling short of leveraging image-content-adaptive semantic information.
- The SNR threshold of 1.44 (upper bound of time step 250) is manually selected, and the sensitivity of the threshold selection is not fully discussed.
- Evaluated only on the $\times 4$ SR task without generalizing to other scaling factors or image restoration tasks.
- The architecture of the noise prediction network is relatively simple (based on VQGAN encoder); a stronger architecture might yield better results.
- The degradation model still relies on a synthetic pipeline (RealESRGAN). Its performance under more extreme, real-world degradations remains to be validated.

## Related Work & Insights

- **Diffusion Prior for SR**: Unlike fine-tuning methods such as StableSR, DiffBIR, and SeeSR, InvSR keeps the diffusion model completely untouched. This "no modification of foundation models" philosophy aligns with the design principles of Adapter/LoRA.
- **GAN Inversion $\to$ Diffusion Inversion**: This work successfully transfers the concept of "finding the optimal latent space representation" from GAN inversion to diffusion models, providing a valuable methodological leap.
- **Single-Step Distillation Methods**: Compared to specialized single-step methods like OSEDiff, InvSR supports both single-step and multi-step sampling, offering greater flexibility.
- **Inspiration for Image Editing and Inpainting**: The core idea of the PnP strategy (harnessing approximations of target images to construct intermediate diffusion states) has broad applicability.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing diffusion inversion for SR is not completely new, but the design of the PnP strategy and arbitrary-step sampling is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive with multiple datasets, multiple metrics, multi-step ablations, comparisons with 9 methods, and running time analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivations, rigorous derivations, well-structured charts and tables, and highly coherent logic throughout.
- Value: ⭐⭐⭐⭐ Simple, efficient, and effective, with highly practical application value and potential for methodological generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

</div>

<!-- RELATED:END -->
