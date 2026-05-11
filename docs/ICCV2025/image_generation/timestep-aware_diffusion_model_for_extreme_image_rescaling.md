---
title: >-
  [Paper Note] Timestep-Aware Diffusion Model for Extreme Image Rescaling
description: >-
  [Image Generation] This paper proposes TADM, which performs extreme image rescaling (16×/32×) in the latent space of a pretrained Stable Diffusion model. By introducing a Decoupled Feature Rescaling Module (DFRM) and a t…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 29ac512ac10ced06
---

# Timestep-Aware Diffusion Model for Extreme Image Rescaling

> **Conference**: ICCV 2025
> **arXiv**: [2408.09151](https://arxiv.org/abs/2408.09151)
> **Area**: Image Generation
> **Keywords**: image rescaling, timestep alignment, diffusion model, decoupled feature rescaling, extreme downscaling

## TL;DR

This paper proposes TADM, which performs extreme image rescaling (16×/32×) in the latent space of a pretrained Stable Diffusion model. By introducing a Decoupled Feature Rescaling Module (DFRM) and a timestep-aware alignment strategy, TADM dynamically allocates the generative capacity of the diffusion model to handle spatially non-uniform degradation.

## Background & Motivation

The storage and transmission demands of ultra-high-resolution images have made extreme scaling factors (16×, 32×) increasingly important. Image rescaling differs from conventional super-resolution (SR) in that it jointly optimizes the downscaling and upscaling processes to retain critical information for high-quality reconstruction.

However, existing methods face severe challenges at extreme scaling factors:

**Information bottleneck in INN-based methods (IRN, HCFlow)**: Invertible neural networks perform well at 2×/4×, but at extreme compression ratios they cannot retain sufficient detail, yielding over-smoothed results.

**Semantic errors in GAN-prior methods**: GRAIN (StyleGAN prior) is restricted to the face domain, while VQIR (VQGAN prior), though applicable to natural images, still exhibits semantic errors on complex structures such as faces and text.

**Spatially non-uniform degradation**: The information loss induced by rescaling varies greatly across different images and different regions within the same image (e.g., texture-rich regions vs. flat regions), yet existing methods apply fixed restoration strategies.

**Core Idea of TADM**:
- Perform rescaling operations in the SD latent space (aligned with the SD prior)
- Analogize rescaling degradation to the forward noising process of diffusion
- Adaptively predict a "noise density" (timestep) to dynamically allocate the generative capacity of the diffusion model

## Method

### Overall Architecture (Fig. 1)

TADM consists of four stages:
1. **Latent Encoding**: The pretrained VAE encoder $\mathcal{E}$ encodes the HR image $x$ into $z$
2. **Latent-Space Feature Rescaling**: DFRM rescales $z$ to the target resolution, outputting the LR image $y$ and the rescaled latent feature $\hat{z}$
3. **Denoising-Guided Perceptual Enhancement**: The pretrained SD U-Net performs single-step denoising on $\hat{z}$ to obtain the enhanced feature $\hat{z}_0$
4. **Latent Decoding**: The pretrained VAE decoder $\mathcal{D}$ decodes $\hat{z}_0$ into the reconstructed image $\hat{x}$

### Key Design 1: Decoupled Feature Rescaling Module (DFRM)

Conventional methods directly generate the LR image $y$ from $z$ and reconstruct $\hat{z}$ from $y$, causing a conflict between the guidance loss $\mathcal{L}_{gui}$ and the reconstruction loss $\mathcal{L}_{rec}$ (the former drives the SR objective while the latter drives the compression objective).

DFRM decouples rescaling into two independent transformation chains (Fig. 2):
- **Feature rescaling chain**: $(x, z) \rightarrow z_{lr} \rightarrow \hat{z}$, implemented by a CNN encoder $G_e$ and decoder $G_d$
- **Pixel mapping chain**: $z_{lr} \leftrightarrow y$, implemented by an invertible neural network $F$ for bidirectional mapping between the feature domain and pixel domain

$$z_{lr} = G_e(x, z), \quad y = F(z_{lr}), \quad \hat{z} = G_d(F^{-1}(y))$$

The reconstruction loss accounts for both paths, with and without quantization:

$$\mathcal{L}_{rec} = \|G_d(G_e(x,z)) - z\|_1 + \|G_d(F^{-1}(F(G_e(x,z)))) - z\|_1$$

A **pixel guidance module** is introduced to improve the visual quality of the LR image.

### Key Design 2: Timestep Alignment Strategy

**Core Observation (Fig. 3)**: The MSE introduced by rescaling corresponds to the MSE of the diffusion forward noising process — different scaling factors and different image contents correspond to different diffusion timesteps.

- **Timestep Prediction Module (TPM)**: A lightweight network predicts the timestep $t = \text{TPM}(\hat{z})$ from $\hat{z}$
- **Hybrid Timestep Scheduler**: A fixed scheduler (non-differentiable) combined with a learnable scheduler (neural network approximation, stabilized via zero-initialized convolutions):

$$\hat{z}_0 = \mathcal{S}_{fixed}(\hat{z}, \epsilon, t_0) + \mathcal{S}_{learned}(\hat{z}, \epsilon, t)$$

The fixed scheduler follows the standard formula $\hat{z}_0 = (\hat{z} - \sqrt{1-\bar{\alpha}_t} \epsilon) / \sqrt{\bar{\alpha}_t}$, while the learnable scheduler progressively refines the output via zero-initialized convolutions.

**Patch-based Inference**: Ultra-high-resolution images are processed in patches, with each patch predicting an independent timestep, enabling spatially adaptive allocation of generative capacity (Fig. 14).

### Loss & Training

Three-stage training:
1. Train DFRM ($\mathcal{L}_{res} = \lambda_{rec} \mathcal{L}_{rec} + \lambda_{gui} \mathcal{L}_{gui}$)
2. Jointly train LoRA + TPM + timestep scheduler ($\mathcal{L}_{enh} = \|x - \hat{x}\|_1 + \lambda_{pec}(\mathcal{L}_{lpips} + \mathcal{L}_{dists})$)
3. Joint fine-tuning of all modules with a small learning rate

## Key Experimental Results

### Main Results: Quantitative Comparison on Extreme Rescaling (Tab. 1)

**16× rescaling, DIV2K dataset**:

| Method | PSNR ↑ | LPIPS ↓ | DISTS ↓ | MUSIQ ↑ | CLIPIQA ↑ |
|--------|--------|---------|---------|---------|-----------|
| ESRGAN | 23.15 | 0.4478 | 0.2378 | 59.80 | 0.6161 |
| HCFlow | 26.66 | 0.4885 | 0.2866 | 46.43 | 0.2735 |
| VQIR | 23.91 | 0.3174 | 0.1024 | 64.04 | 0.6350 |
| S3Diff | 20.22 | 0.4033 | 0.1309 | 64.37 | 0.6228 |
| **TADM** | 23.98 | **0.2979** | **0.0886** | **66.56** | **0.7189** |

**32× rescaling, DIV2K dataset**:

| Method | PSNR ↑ | LPIPS ↓ | DISTS ↓ | MUSIQ ↑ | CLIPIQA ↑ |
|--------|--------|---------|---------|---------|-----------|
| HCFlow | 23.89 | 0.5816 | 0.3852 | 37.25 | 0.2792 |
| VQIR | 22.02 | 0.4568 | 0.2663 | 58.21 | 0.6293 |
| S3Diff | 17.81 | 0.4895 | 0.1810 | 67.92 | 0.6991 |
| **TADM** | **22.18** | **0.4221** | **0.1684** | **69.12** | **0.7204** |

TADM achieves state-of-the-art results on all perceptual metrics across all datasets. Notably, at DIV2K 32×, DISTS improves by **36.76%** over the second-best method VQIR.

### Ablation Study: Rescaling Space and SD Prior (Tab. 2)

| Latent Rescaling | Pixel Rescaling | SD Prior | LPIPS ↓ | DISTS ↓ |
|-----------------|----------------|---------|---------|---------|
| ✗ | ✓ | ✓ | 0.3630 | 0.1154 |
| ✓ | ✗ | ✗ | 0.4675 | 0.3109 |
| ✓ | ✗ | ✓ | **0.2979** | **0.0886** |

Key findings:
- Latent-space rescaling is better aligned with the SD prior than pixel-space rescaling (DISTS: 0.0886 vs. 0.1154)
- The SD prior is critical: removing SD enhancement causes DISTS to surge from 0.0886 to 0.3109

### Effectiveness of Timestep Alignment (Fig. 13)

- Fixed timestep = 1: high fidelity but low perceptual quality
- Fixed timestep = 999: low fidelity but high perceptual quality
- **Adaptive timestep**: achieves the best of both, optimizing fidelity and perceptual quality simultaneously

Fig. 14 visualizes the predicted timestep maps: regions with complex textures are assigned larger timesteps (stronger generative capacity), while flat regions are assigned smaller timesteps (higher fidelity).

## Highlights & Insights

1. The analogy of **rescaling as noising** is remarkably elegant, establishing a natural connection between image rescaling and diffusion models
2. The **decoupled design** resolves the fundamental conflict between reconstruction loss and guidance loss, with the INN module dedicated to feature-to-pixel mapping
3. **Timestep adaptivity** enables fine-grained handling of spatially non-uniform degradation, with each patch independently predicting its timestep during patch-based inference
4. **Single-step denoising** efficiently leverages the SD prior, avoiding the high latency of multi-step sampling

## Limitations & Future Work

- LR images exhibit some ringing artifacts and noise
- The method is built on SD 2.1-base and requires re-adaptation for newer SD versions
- Patch boundaries during patch-based inference may introduce spatial inconsistencies

## Related Work & Insights

- Image rescaling: IRN, HCFlow, CAR, VQIR, GRAIN
- Diffusion-based SR: SR3, StableSR, S3Diff, SinSR, InvSR
- Single-step diffusion: OSEDiff, ResShift

## Rating

- **Novelty**: ★★★★★ — The timestep-aware alignment strategy is original
- **Technical Depth**: ★★★★★ — DFRM decoupled design + hybrid scheduler + patch-based inference, with solid engineering details
- **Experimental Thoroughness**: ★★★★★ — 4 datasets × 2 scaling factors × 6 metrics + multi-dimensional ablation
- **Writing Quality**: ★★★★☆ — Clear structure with highly informative figures

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)

</div>

<!-- RELATED:END -->
