---
title: >-
  [Paper Note] BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion
description: >-
  [ECCV2024][Image Restoration][image inpainting] Proposes BrushNet, a plug-and-play dual-branch diffusion model architecture for image inpainting. By decoupling masked image feature extraction and image generation into separate branches, it achieves layer-wise pixel-level feature injection, thoroughly outperforming existing methods in image quality, masked area preservation, and text alignment.
tags:
  - "ECCV2024"
  - "Image Restoration"
  - "image inpainting"
  - "diffusion models"
  - "dual-branch"
  - "plug-and-play"
  - "masked image feature"
date: 2026-05-08
content_hash: 4e040819fb588989
---

# BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion

**Conference**: ECCV2024  
**arXiv**: [2403.06976](https://arxiv.org/abs/2403.06976)  
**Code**: [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet)  
**Area**: Image Generation  
**Keywords**: image inpainting, diffusion models, dual-branch, plug-and-play, masked image feature

## TL;DR

Proposes BrushNet, a plug-and-play dual-branch diffusion model architecture for image inpainting. By decoupling masked image feature extraction and image generation into separate branches, it achieves layer-wise pixel-level feature injection, thoroughly outperforming existing methods in image quality, masked area preservation, and text alignment.

## Background & Motivation

Image inpainting aims to restore missing regions in an image while maintaining global consistency. The rise of diffusion models has brought significant progress to this task, but existing methods suffer from inherent limitations in two major paradigms:

1. **Sampling strategy modification methods** (e.g., Blended Latent Diffusion): During each denoising step, the masked region is sampled from the pretrained model, while the unmasked region is directly copied and pasted. Although adaptable to any diffusion backbone, this paradigm lacks context awareness of masked boundaries and unmasked regions, leading to semantically incoherent results.
2. **Dedicated inpainting models** (e.g., SD Inpainting, PowerPaint): These methods merge the masked image and mask information by expanding the input channel dimensions of the UNet. Although offering better generation quality, three key issues persist:
    - Mixing noisy latent, masked image latent, mask, and text at the very first layer of the UNet prevents subsequent layers from accessing clean masked image features.
    - Handling both conditioning and generation within a single branch increases the learning burden on the UNet.
    - Requiring fine-tuning for different diffusion backbones limits transferability.

The authors find that while ControlNet introduces an additional branch, its design is tailored for sparse structural control (e.g., skeletons) and is unsuitable for inpainting tasks that require pixel-level dense constraints, leading to suboptimal performance when directly applied to inpainting.

## Core Problem

How to design a plug-and-play image inpainting architecture that can efficiently extract pixel-level features of masked images and inject them into any pretrained diffusion model, while preserving the consistency of unmasked regions and the semantic coherence of generated areas?

## Method

### Overall Architecture

BrushNet adopts a dual-branch strategy: a frozen, pretrained UNet is responsible for image generation, while a trainable BrushNet branch handles masked image feature extraction. Features from both branches are fused layer by layer.

### Key Designs

**1. VAE Encoder for Masked Image Processing**

Unlike ControlNet, which uses randomly initialized convolutional layers, BrushNet employs a pretrained VAE encoder to map the masked image to the latent space, ensuring input features align with the data distribution of the pretrained UNet. The input to BrushNet is the concatenation of three components: noisy latent $z_t$, masked image latent $z_0^{masked}$, and downsampled mask $m^{resized}$.

**2. Layer-wise Full Feature Insertion**

BrushNet adds features from the auxiliary branch **layer-by-layer** into **every single layer** of the pretrained UNet (unlike ControlNet, which only adds residuals to select layers), achieving dense pixel-level control. The feature insertion formula is:

$$\epsilon_\theta(z_t, t, C)_i = \epsilon_\theta(z_t, t, C)_i + w \cdot \mathcal{Z}(\epsilon_\theta^{BrushNet}([z_t, z_0^{masked}, m^{resized}], t)_i)$$

where $\mathcal{Z}$ denotes zero convolution, $w$ is the control scale, and $i$ represents the layer index.

**3. Removal of Cross-Attention Layers**

The BrushNet branch is cloned from the pretrained UNet but has all text cross-attention layers removed, ensuring that this branch only processes pure image information and preventing text embeddings from interfering with masked image features.

### Blurred Blending Strategy

Due to inherent errors in VAE encoding/decoding and mask downsampling, directly blending in the latent space leads to distortion in unmasked areas. BrushNet proposes to first apply Gaussian blur to the mask in the pixel space, and then perform copy-and-paste using the blurred mask. Although there is a minor loss of precision at the mask boundaries, it is virtually imperceptible to the human eye, yielding significantly better boundary coherence.

### Flexible Controlling Capabilities

- **Plug-and-play**: Does not modify the weights of the pretrained model, making it directly compatible with various community fine-tuned diffusion models.
- **Preservation scale adjustment**: Controls the influence of BrushNet on the pretrained model via the weight parameter $w$.
- **Optional blending operation**: Further customizes the preservation level by adjusting the blur scale and toggling blending.

## Key Experimental Results

### Evaluation Setup

- Proposes **BrushBench** (600 images, including humans, animals, indoor, and outdoor scenes, with a balanced mix of natural and artistic style images) for segmentation mask inpainting evaluation.
- Proposes **BrushData** (segmentation mask annotations based on LAION-Aesthetic) for training.
- Uses **EditBench** (240 images) to evaluate random mask inpainting.
- Evaluates 7 metrics covering three aspects: image quality (IR, HPS, AS), masked area preservation (PSNR, LPIPS, MSE), and text alignment (CLIP Sim).

### Main Results (BrushBench inside-inpainting)

| Method | IR↑ | HPS↑ | AS↑ | PSNR↑ | CLIP Sim↑ |
|------|-----|------|-----|-------|-----------|
| BLD | 9.78 | 25.87 | 6.17 | 21.33 | 26.15 |
| SD Inpainting | 11.72 | 27.06 | 6.50 | 21.52 | 26.17 |
| HD-Painter | 11.68 | 26.90 | 6.42 | 22.61 | 26.37 |
| PowerPaint | 11.46 | 27.35 | 6.24 | 21.43 | 26.48 |
| ControlNet-Inp | 11.21 | 26.92 | 6.39 | 22.73 | 26.22 |
| **BrushNet** | **12.36** | **27.40** | **6.53** | 21.65 | **26.48** |
| **BrushNet*** | **12.64** | **27.78** | 6.51 | **31.94** | 26.39 |

*With blending operation

### EditBench Results

BrushNet similarly leads across the board on EditBench: IR reaches 4.40 (the second best, SDI, is only 1.86), HPS is 25.10, and CLIP Sim is 28.67, all being the top scores. With blending, PSNR achieves 33.66, far exceeding other methods (~23).

### Ablation Study

- Dual-branch vs. Single-branch: The dual-branch design outperforms single-branch SD Inpainting across all metrics.
- VAE vs. Conv Encoder: VAE leads significantly in PSNR (14.89→17.96) and LPIPS.
- Full Feature Injection vs. ControlNet-style Injection (full > half > CN): full achieves a PSNR of 19.86 vs. CN's 18.49.
- Removing cross-attention yields better results: retaining cross-attention instead degrades image quality metrics.
- Blurred blending vs. Direct pasting: blur blending performs best in PSNR (29.88) and coherence.

## Highlights & Insights

1. **Clear and Powerful Architectural Innovation**: Decouples the masked image features and the generation process into two branches, backed by convincing motivation analysis and rational design.
2. **Plug-and-play**: Freezes the weights of the pretrained model, enabling seamless adaptation to various community fine-tuned models (e.g., DreamShaper, MeinaMix).
3. **Thorough Ablation Studies**: Validates each design choice step-by-step, including VAE encoding, full feature injection, cross-attention removal, and blurred blending.
4. **Comprehensive Evaluation Protocol**: Introduces BrushBench and BrushData, distinguishes between inside/outside inpainting, and covers three dimensions with 7 evaluation metrics.
5. **Context Awareness**: In qualitative results, BrushNet can identify existing objects in the masked image (e.g., a goldfish), avoiding duplicate generation.

## Limitations & Future Work

1. **Generation Quality Dependency on Base Models**: As a plugin, BrushNet's output quality is highly correlated with the selected pretrained model. If there is a domain mismatch with the base model (e.g., using an anime model to process natural images), the results will be incoherent.
2. **Suboptimal Handling of Irregular Masks**: May still produce poor generation results for highly irregular mask shapes.
3. **Degraded Performance on Text-Image Mismatch**: Generation quality is affected when text prompts conflict with the content of the masked image.
4. **Validated Only on SD 1.5**: Generalizability to SDXL or newer architectures has not been verified.
5. **Mask Boundary Precision in Blurred Blending**: Although imperceptible to the human eye, errors may accumulate in fine-grained editing scenarios.

## Related Work & Insights

| Feature | BLD | SD Inpainting | PowerPaint | ControlNet-Inp | BrushNet |
|------|-----|---------------|------------|----------------|----------|
| Plug-and-play | ✓ | ✗ | ✗ | ✓ | ✓ |
| Flexible Scale | ✗ | ✗ | ✗ | ✗ | ✓ |
| Content-aware | ✗ | ✓ | ✓ | ✓ | ✓ |
| Shape-aware | ✗ | ✓ | ✓ | ✓ | ✓ |

BrushNet is the only method that simultaneously possesses all four capabilities: plug-and-play, flexible scale, content awareness, and shape awareness. Compared to ControlNet-Inpainting, BrushNet utilizes VAE encoding (rather than randomly initialized Conv), full layer-wise feature injection (rather than only decoder residuals), and cross-attention removal. These three improvements make it significantly superior to ControlNet's adaptation scheme for inpainting tasks.

## Related Work & Insights

1. The **dual-branch decoupling concept** is highly generalizable and can be transferred to other conditional generation tasks (e.g., image-to-image translation, virtual try-on), serving as a reference for any scenario requiring pixel-level conditioning injection.
2. The design insight of **removing cross-attention to maintain pure image features**: In multimodal conditioning injection, attention mechanisms should be selectively retained or removed based on the target information type.
3. The strategy of **aligning distributions via VAE encoding** is more effective than using randomly initialized encoders, offering valuable insights for improving ControlNet-like architectures.
4. The comparison between full layer-wise injection and sparse injection indicates that dense control tasks (e.g., inpainting) require stronger feature coupling, which is fundamentally different from sparse control tasks (e.g., pose guidance).

## Rating
- Novelty: ⭐⭐⭐⭐ — Dual-branch decoupling + three targeted designs, providing a clear architectural innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — New benchmark, 7 evaluation metrics, comprehensive ablation studies, and qualitative results across multiple domains
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation analysis with a well-explained differentiation from ControlNet
- Value: ⭐⭐⭐⭐ — Highly practical plug-and-play design, already open-source with high community adoption

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling](../../ICLR2026/image_restoration/adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling.md)
- [\[ICLR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](../../ICLR2026/image_restoration/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ECCV 2024\] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)

</div>

<!-- RELATED:END -->
