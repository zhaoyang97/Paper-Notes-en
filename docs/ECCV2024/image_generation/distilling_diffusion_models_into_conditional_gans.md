---
title: >-
  [Paper Note] Distilling Diffusion Models into Conditional GANs
description: >-
  [ECCV 2024][Image Generation][Diffusion Distillation] Proposes the Diffusion2GAN framework, which distills multi-step diffusion models into single-step conditional GANs. The core innovations are the E-LatentLPIPS latent-space perceptual loss and a multi-scale conditional discriminator based on pretrained diffusion models, achieving performance that surpasses DMD, SDXL-Turbo, and SDXL-Lightning on the zero-shot COCO benchmark.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Distillation"
  - "Conditional GAN"
  - "Perceptual Loss"
  - "Latent Space"
  - "Single-Step Generation"
date: 2026-05-08
content_hash: 73ec813ac9faa45d
---

# Distilling Diffusion Models into Conditional GANs

**Conference**: ECCV 2024  
**arXiv**: [2405.05967](https://arxiv.org/abs/2405.05967)  
**Code**: Yes (provided on project page)  
**Area**: Diffusion Models / Image Generation / Knowledge Distillation  
**Keywords**: Diffusion Distillation, Conditional GAN, Perceptual Loss, Latent Space, Single-Step Generation

## TL;DR

Proposes the Diffusion2GAN framework, which distills multi-step diffusion models into single-step conditional GANs. The core innovations are the E-LatentLPIPS latent-space perceptual loss and a multi-scale conditional discriminator based on pretrained diffusion models, achieving performance that surpasses DMD, SDXL-Turbo, and SDXL-Lightning on the zero-shot COCO benchmark.

## Background & Motivation

Diffusion models (e.g., Stable Diffusion, DALL·E 2) have achieved unprecedented success in image generation quality. However, their sampling process requires dozens or even hundreds of denoising steps, resulting in high inference latency (typically over 10 seconds), which severely hinders real-time interaction and downstream 3D/video application deployment.

**Why not directly train a single-step model?** Training GANs directly for text-to-image generation faces two simultaneous challenges: (1) establishing a correspondence between noise and natural images, and (2) effectively optimizing the generator to map noise to images. This "unpaired" learning is significantly more ill-posed than paired learning.

**Key Insight**: Decouple the two tasks—first establish noise-image correspondences using the ODE trajectories of a pretrained diffusion model, then utilize a conditional GAN to learn the mapping under a paired image-to-image translation framework. This approach combines the strengths of diffusion models (finding high-quality correspondences) and GANs (achieving fast mapping).

**Key Finding**: The regression loss designs in existing distillation methods are insufficient. The authors show that with a carefully designed regression loss (utilizing perceptual loss instead of $L_2$), direct distillation can achieve results competitive with recent distillation methods (such as Consistency Distillation) at a much lower computational cost. However, standard LPIPS requires decoding the latent space back to pixel space, which contradicts the efficiency of latent diffusion models. This motivates the proposal of E-LatentLPIPS.

## Method

### Overall Architecture

Diffusion2GAN consists of two stages:
1. **Dataset Construction**: Generate a large number of noise-latent pairs $\{(\mathbf{z}, \mathbf{c}, \mathbf{x})\}$ using the DDIM sampler (50 steps) of a pretrained diffusion model (e.g., SD 1.5).
2. **Paired Distillation Training**: Treat the noise-to-image mapping as a paired image-to-image translation problem, training a single-step generator using the E-LatentLPIPS regression loss + conditional GAN adversarial loss.

The generator $G$ shares its architecture (UNet) with the teacher diffusion model and is initialized with the teacher's weights. The final loss is:

$$\mathcal{L}_G = \mathcal{L}_{\text{E-LatentLPIPS}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

### Key Designs

#### 1. E-LatentLPIPS (Core Contribution)

**Function**: Directly computes perceptual distance within the latent space, eliminating the inefficient pixel-space decoding required by traditional LPIPS.

**Mechanism**:
- First, train LatentLPIPS: train a VGG network for ImageNet classification directly on the SD latent space, removing max pooling layers (as the latent space is already $8\times$ downsampled), modifying the input to 4 channels, and linearly calibrating intermediate features using the BAPPS dataset.
- Directly using LatentLPIPS causes wave-like artifacts due to blind spots in the loss landscape. Thus, an **ensemble augmentation strategy** is introduced, applying random differentiable augmentations (geometric transformations + cutout) to both the generated and target latent codes at each iteration.

$$d_{\text{E-LatentLPIPS}}(\mathbf{x}_0, \mathbf{x}_1) = \mathbb{E}_{\mathcal{T}}\left[\ell\left(F(\mathcal{T}(\mathbf{x}_0)), F(\mathcal{T}(\mathbf{x}_1))\right)\right]$$

**Design Motivation**:
- Traditional LPIPS requires decoding to pixel space, consuming an extra 117ms and 15GB VRAM per iteration.
- E-LatentLPIPS requires only 12.1ms and 0.6GB, speeding up perceptual loss computation by **9.7 times**.
- Ensemble augmentation is critical: without augmentation, pure LatentLPIPS fails to converge (as validated by single-image reconstruction experiments), whereas adding augmentations enables accurate reconstruction of target images.

#### 2. Conditional Diffusion Discriminator

**Function**: A GAN discriminator conditioned on noise $\mathbf{z}$, text $\mathbf{c}$, and image $\mathbf{x}$, designed to improve generation quality.

**Mechanism**: Reuse the pretrained diffusion UNet weights to initialize the discriminator, incorporating the following modifications:
- **Noise conditioning**: Add a zero-initialized conv layer at the input to process $\mathbf{z}$, which is added to the discriminator input.
- **Text conditioning**: Directly leverage the UNet's built-in cross-attention layers.
- **Multi-scale input and output**: Modify the UNet encoder to receive scaled inputs for each downsampling level, and append three readout layers at each scale of the decoder to make independent real/fake predictions.
- **Single-sample R1 regularization**: Compute R1 regularization on only one sample per batch, combined with lazy regularization (every 16 steps) to reduce VRAM consumption.
- **Mix-and-match augmentation**: During discriminator training, randomly replace a subset of generated latents with unrelated latents while keeping other conditioning elements fixed, enhancing text alignment and noise-conditioning capabilities.

**Design Motivation**: The UNet of a pretrained diffusion model contains rich image prior knowledge, which is more effective than training a GigaGAN discriminator from scratch. The multi-scale design ensures all UNet layers (from shallow skip connections to the deep bottleneck) participate in the prediction, strengthening low-frequency structural consistency.

### Loss & Training

- **Regression Loss**: $\mathcal{L}_{\text{E-LatentLPIPS}}$ — latent-space perceptual loss with ensemble augmentation.
- **Adversarial Loss**: $\mathcal{L}_{\text{GAN}} = -\mathbb{E}_{\mathbf{c},\mathbf{z}}[\log(D(\mathbf{c},\mathbf{z},G(\mathbf{z},\mathbf{c})))]$ — non-saturating GAN loss.
- **Discriminator Regularization**: Single-sample R1 regularization (interval of 16).
- **Training Configuration**: 3 million pairs from SD-CFG-3 dataset, 12 million pairs from SD-CFG-8 dataset; 64x A100-80GB GPUs, batch size of 1024.
- **Important Note**: The entire training process takes place entirely in the latent space and **never** requires decoding to the pixel space.

## Key Experimental Results

### Main Results

**COCO2014 Zero-shot Benchmark (distilling SD 1.5)**:

| Method | Type | FID-30k↓ | Inference Time (s) |
|------|------|----------|------------|
| Stable Diffusion 1.5 (Teacher) | Diffusion | 8.74 | 2.59 |
| GigaGAN | GAN | 9.09 | 0.13 |
| **Diffusion2GAN** | **Distillation** | **9.29** | **0.09** |
| DMD | Distillation | 11.49 | 0.09 |
| UFOGen | Distillation | 12.78 | 0.09 |
| InstaFlow-0.9B | Distillation | 13.10 | 0.09 |

**COCO2017 Benchmark (distilling SDXL)**:

| Method | FID-5k↓ | CLIP-5k↑ | DreamDiv↑ |
|------|---------|----------|-----------|
| SDXL-Base-1.0 (Teacher, 50-step) | 25.56 | 0.346 | 0.338 |
| **SDXL-Diffusion2GAN (1-step)** | **25.49** | **0.347** | **0.268** |
| SDXL-Turbo (1-step) | 28.10 | 0.342 | 0.232 |
| SDXL-Lightning (1-step) | 30.14 | 0.324 | 0.315 |

### Ablation Study

**Regression Loss Comparison (SD-CFG-3, 20k iter, batch=256)**:

| Loss Function | Space | FID↓ | CLIP↑ |
|---------|------|------|-------|
| MSE | Latent Space | 110.55 | 0.222 |
| Pseudo Huber | Latent Space | 87.60 | 0.230 |
| LPIPS | Pixel Space | 25.94 | 0.288 |
| LatentLPIPS | Latent Space | 67.17 | 0.244 |
| **E-LatentLPIPS** | **Latent Space** | **22.95** | **0.299** |

**Discriminator Components Ablation**:

| Configuration | FID-30k↓ | CLIP-30k↑ |
|------|----------|-----------|
| E-LatentLPIPS (Regression-only) | 14.72 | 0.292 |
| + Diffusion D | 12.04 | 0.300 |
| + z-conditioning | 11.97 | 0.302 |
| + Single-sample R1 | 10.60 | 0.303 |
| + Multi-scale training | 9.58 | 0.308 |
| + Mix-and-match augmentation | 9.45 | 0.310 |

### Key Findings

1. **E-LatentLPIPS is key**: Directly computing perceptual loss in latent space + ensemble augmentation reduces FID from 67.17 to 22.95 while improving computational efficiency by 9.7x.
2. **Single-step generator matches the teacher**: The FID of SDXL-Diffusion2GAN (25.49) is nearly identical to the 50-step teacher SDXL (25.56).
3. **ODE trajectory fidelity**: Diffusion2GAN preserves the noise-to-image mapping of the teacher model better than SDXL-Turbo and SDXL-Lightning (scoring lowest on DreamSim-5k).
4. Human preference evaluations show that Diffusion2GAN outperforms InstaFlow in realism and text alignment, and is comparable to or better than SDXL-Turbo/Lightning.

## Highlights & Insights

- **Reframing distillation as paired translation**: This perspective allows mature pix2pix toolkits to be directly applied to diffusion distillation.
- **Feasibility of latent perceptual loss**: Demonstrates that while compressing to latent space loses some low-level classification information, it retains perceptually relevant details.
- **Necessity of ensemble augmentation**: An optional improvement in pixel space, but a life-or-death factor in latent space—without augmentation, LatentLPIPS fails to converge at all.
- **Multiple uses of pretrained weights**: The weights of the teacher diffusion model are leveraged to initialize both the generator and the discriminator.

## Limitations & Future Work

- Although single-step inference is extremely fast (0.09s/image), **training costs are high**: requiring pre-generating millions of ODE pairs + training on 64x A100s.
- Multi-step teacher models are still preferred in human evaluations, indicating room for growth in the upper bound of single-step distillation.
- The DreamDiv metric should be reported jointly with the CLIP-score; otherwise, it might be misled by spurious diversity caused by low text alignment.
- Exploration of distilling newer diffusion architectures (such as DiT) remains unaddressed.

## Related Work & Insights

- Consistent with the "unpaired vs. paired" insight of CycleGAN, this paper demonstrates that the paired translation paradigm is far superior to the unpaired paradigm in distillation as well.
- The ensemble augmentation strategy inspired by E-LPIPS plays a much more significant role in the latent space than anticipated.
- The two new metrics, DreamSim-5k and DreamDiv, are worthy of adoption in future work to measure ODE trajectory fidelity and generative diversity, respectively.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — E-LatentLPIPS is a simple and effective innovation, and reframing distillation as paired translation offers a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual validation on SD 1.5 and SDXL, thorough ablations, human preference evaluations, and the introduction of new metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, with each design decision supported by experiments.
- **Value**: ⭐⭐⭐⭐ — Highly practical; high-quality single-step text-to-image generation is of significant importance for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ICML 2026\] Scalable GANs with Transformers](../../ICML2026/image_generation/scalable_gans_with_transformers.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
