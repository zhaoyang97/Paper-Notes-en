---
title: >-
  [Paper Note] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis
description: >-
  [ICCV 2025][Image Generation][Arbitrary-resolution generation] This paper proposes InfGen, a "second-generation" paradigm that replaces the VAE decoder with a Transformer-based generator, decoding fixed-size latents into images at arbitrary resolution in a single forward pass—without modifying or retraining the diffusion model. It reduces 4K image generation to under 10 seconds, achieving over 10× speedup compared to the fastest existing method, UltraPixel.
tags:
  - ICCV 2025
  - Image Generation
  - Arbitrary-resolution generation
  - latent diffusion models
  - VAE decoder replacement
  - implicit neural positional encoding
  - efficient 4K generation
date: 2026-05-08
content_hash: 30ec11b221d38dc5
---

# InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis

**Conference**: ICCV 2025
**arXiv**: [2509.10441](https://arxiv.org/abs/2509.10441)
**Code**: [GitHub](https://github.com/taohan10200/InfGen)
**Area**: Image Generation
**Keywords**: Arbitrary-resolution generation, latent diffusion models, VAE decoder replacement, implicit neural positional encoding, efficient 4K generation

## TL;DR

This paper proposes InfGen, a "second-generation" paradigm that replaces the VAE decoder with a Transformer-based generator, decoding fixed-size latents into images at arbitrary resolution in a single forward pass—without modifying or retraining the diffusion model. It reduces 4K image generation to under 10 seconds, achieving over 10× speedup compared to the fastest existing method, UltraPixel.

## Background & Motivation

- **Demand for arbitrary-resolution image generation**: Diverse devices (smartphones, 4K displays, etc.) require consistent visual experience across resolutions.
- **Bottlenecks of existing methods**:
    - Computational cost of diffusion models scales quadratically with resolution; generating 4K images incurs latency exceeding 100 seconds.
    - Training-free methods (e.g., ScaleCrafter, FouriScale) achieve super-resolution by modifying the inference process (dilated convolutions, etc.), but are tightly coupled to specific network architectures and generalize poorly.
    - Methods such as Inf-DiT redesign attention mechanisms but remain slow (255 seconds for 2K image generation).
    - UltraPixel requires fine-tuning the diffusion model itself.
- **Key insights**:
    - The generative model has already completed content generation (Stage 1); the Stage 2 decoder requires only a single forward pass.
    - Enhancing the decoder's capacity to achieve high-resolution image generation is a more efficient pathway.
    - The mapping from latent to high-resolution image is intrinsically a super-resolution task, but requires generative capability to compensate for information loss.

## Method

### Overall Architecture

InfGen operates as a "second-generation" model:
1. **Stage 1**: A diffusion model generates a fixed-size content latent $z$ (e.g., $4 \times 64 \times 64$).
2. **Stage 2**: InfGen decodes $z$ into an image at arbitrary resolution $(h, w)$.

$$f: \text{InfGen}(z, (h, w)) \rightarrow x_{(h,w)}$$

This paradigm yields two key advantages: (1) high inference speed—avoiding multi-step denoising on high-resolution latents; and (2) plug-and-play compatibility—applicable to any diffusion model sharing the same latent space.

### Arbitrary-Resolution Decoder Architecture

Built upon the conventional VAE structure, a Transformer-based latent generator is introduced:
- The latent variable $z$ serves as **keys and values**.
- **Mask tokens** corresponding to the target image size $(h, w)$, with shape $(\lceil h/8 \rceil, \lceil w/8 \rceil)$, serve as **queries**.
- Across multiple Transformer blocks, mask tokens acquire information from latent keys via **cross-attention**.
- The resulting mask tokens are passed to an upsampling decoder to produce the final image at arbitrary resolution.

### Implicit Neural Positional Encoding (INPE)

INPE addresses the spatial alignment between fixed-size latents and dynamically sized mask tokens:

1. **Coordinate normalization**: Mask token and latent token coordinates are mapped to a unified scale.
$$(\hat{x}^m, \hat{y}^m) = (x^m/W^m, y^m/H^m)$$

2. **Spherical projection**: Normalized 2D coordinates are transformed into 3D Cartesian coordinates on a unit sphere, leveraging spherical geometry to capture complex spatial relationships.

3. **Fourier features + neural network mapping**: High-frequency Fourier features enhance the model's ability to capture fine patterns.
$$\gamma(x,y,z) = [\cos(B[x,y,z]^T), \sin(B[x,y,z]^T)]$$
These features are then passed through an implicit neural network to produce dynamic positional encodings.

### Loss & Training

$$L_{AE} = \ell_1(x, \hat{x}) + \lambda_P \mathcal{L}_P(x, \hat{x}) + \lambda_G \mathcal{L}_G(\hat{x})$$

where $\ell_1$ denotes the L1 reconstruction loss, $\mathcal{L}_P$ is the LPIPS perceptual loss, and $\mathcal{L}_G$ is the adversarial loss from a PatchGAN discriminator. Both $\lambda_P$ and $\lambda_G$ are set to 0.1.

### Training-Free Resolution Extrapolation

Ultra-high resolutions (e.g., 4K) beyond the training resolution are achieved through an iterative procedure:
$$L_n = \text{Encoder}(I_{n-1}), \quad I_n = \text{InfGen}(L_n, k_n^s)$$

The final resolution is: $R_f = 512 \cdot \prod_{i=1}^n s_i^h \times 512 \cdot \prod_{i=1}^n s_i^w$

## Key Experimental Results

### Main Results: Image Tokenizer Reconstruction Quality

| Method | Input→Output Resolution | ImageNet rFID↓ | PSNR↑ | SSIM↑ |
|--------|------------------------|---------------|-------|-------|
| VQGAN | 256²→256² | 1.19 | 23.38 | 0.762 |
| SD-VAE | 256²→256² | 0.74 | 25.68 | 0.820 |
| SDXL-VAE | 256²→256² | 0.68 | 26.04 | 0.834 |
| InfGen | 256²→256² | 1.07 | 24.61 | 0.798 |
| InfGen | 512²→512² | 0.61 | 27.92 | 0.867 |
| SD-VAE | 256²→512² | 1.43 | 24.14 | 0.759 |
| **InfGen** | **256²→512²** | **1.15** | **22.86** | **0.728** |

InfGen substantially outperforms SD-VAE on the cross-resolution reconstruction task (256²→512²).

### High-Resolution Enhancement over Diffusion Models

| Method | 512² FIDp↓ | 1024² FIDp↓ | 2048² FIDp↓ | 3072² FIDp↓ |
|--------|-----------|------------|------------|------------|
| DiT-XL/2 | 44.17 | 61.52 | 64.87 | 77.84 |
| InfGen+DiT | 39.81 (↓9.9%) | 41.75 (↓32%) | 56.21 (↓13.4%) | 45.94 (↓41%) |
| SD1.5 | 21.58 | 55.30 | - | - |
| InfGen+SD1.5 | 16.92 (↓21%) | 41.12 (↓26%) | - | - |
| FiTv2 | 42.04 | 66.95 | - | 79.30 |
| InfGen+FiTv2 | 38.77 (↓7.8%) | 61.56 (↓8.1%) | - | 45.72 (↓42%) |

Key finding: InfGen yields consistent and significant improvements across all resolution and model combinations, with gains reaching up to 44% at 3072² resolution.

### Comparison with SOTA High-Resolution Generation Methods

| Method | 1024² FIDp↓ | 2048² FIDp↓ | 1024² Latency (s) | 2048² Latency (s) |
|--------|-----------|-----------|-----------------|-----------------|
| ScaleCrafter | 55.36 | 144.61 | 7 | 97 |
| Inf-DiT | 48.48 | 142.05 | 50 | 255 |
| UltraPixel | 48.37 | 127.26 | 11 | 20 |
| InfGen+SD1.5 | 44.85 | 139.14 | 2.9+0.4 | 2.9+1.9 |
| **InfGen+SDXL** | **35.14** | **96.41** | **5.4+0.4** | **5.4+1.9** |

InfGen+SDXL leads by a wide margin in both FID and speed; generating 2K images takes approximately 7 seconds—4× faster than UltraPixel and 35× faster than Inf-DiT.

### Training Setup

- **Training data**: 5 million images with resolution >1024² from LAION-Aesthetic, plus a subset with resolution >2048².
- **Two-stage training**: Stage 1: 512²→1024² (batch=32, 500k iterations); Stage 2: 512²→2048² (batch=8, 100k iterations).
- **Hardware**: 8×A100, trained for 15 days.
- **Optimizer**: AdamW with cosine learning rate decay from 2e-4 to 1e-5.

## Highlights & Insights

1. **Paradigm shift**: The problem of arbitrary-resolution generation is reframed from "modifying the diffusion model" to "enhancing the decoder," substantially reducing complexity and computational cost.
2. **Plug-and-play design**: InfGen leaves the VAE encoder unchanged, enabling it to serve as a direct upgrade for existing diffusion models including SD1.5, SDXL, DiT, and SiT.
3. **Implicit Neural Positional Encoding**: Spherical projection combined with Fourier features elegantly resolves the spatial alignment challenge between fixed latents and dynamic target resolutions.
4. **Iterative extrapolation strategy**: Generation capability is extended to arbitrarily high resolutions without retraining, with each iteration applying a 2× upscaling factor.

## Limitations & Future Work

- InfGen performs slightly below the original VAE on same-resolution reconstruction at 256² (rFID 1.07 vs. 0.74), reflecting the greater complexity of its task.
- The iterative extrapolation pipeline introduces additional encoding–decoding cycles, each accumulating information loss.
- Adversarial training may cause generated content to deviate from the original semantics.
- The current model is trained only on SDXL's VAE latent space; transferring to other latent spaces requires retraining.

## Related Work & Insights

- **Relation to super-resolution methods**: InfGen is fundamentally a conditional generative super-resolution approach, conditioned on latents rather than low-resolution images.
- **Relation to VAE improvements**: Rather than improving the VAE itself, InfGen adds a generative capability layer on top of it.
- **Insight**: The second-generation paradigm decouples "resolution" from the diffusion model's burden, eliminating the need to address high-resolution or arbitrary-resolution generation within the diffusion stage itself.

## Rating ⭐⭐⭐⭐

The approach is conceptually clear, methodologically concise, and practically efficient. The plug-and-play design philosophy and 10× speedup confer strong practical value. The Implicit Neural Positional Encoding is an elegant and well-motivated design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](../../NeurIPS2025/image_generation/scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[ICCV 2025\] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[ICCV 2025\] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis](maskcontrol_spatio-temporal_control_for_masked_motion_synthesis.md)

<!-- RELATED:END -->
