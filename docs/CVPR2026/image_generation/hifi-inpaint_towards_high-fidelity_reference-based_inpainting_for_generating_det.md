---
title: >-
  [Paper Note] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images
description: >-
  [CVPR 2026][Image Generation][Reference-based inpainting] This paper proposes HiFi-Inpaint, a framework that leverages high-frequency information to enhance product detail features via Shared Enhancement Attention (SEA), combined with a Detail-Aware Loss (DAL) for pixel-level high-frequency supervision, achieving state-of-the-art detail fidelity in human-product image generation.
tags:
  - CVPR 2026
  - Image Generation
  - Reference-based inpainting
  - high-fidelity detail preservation
  - human-product image generation
  - high-frequency guidance
  - DiT
date: 2026-05-08
content_hash: 0db227777ad86899
---

# HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images

**Conference**: CVPR 2026
**arXiv**: [2603.02210](https://arxiv.org/abs/2603.02210)
**Code**: [Project Page](https://correr-zhou.github.io/HiFi-Inpaint)
**Area**: Diffusion Models / Image Generation
**Keywords**: Reference-based inpainting, high-fidelity detail preservation, human-product image generation, high-frequency guidance, DiT

## TL;DR

This paper proposes HiFi-Inpaint, a framework that leverages high-frequency information to enhance product detail features via Shared Enhancement Attention (SEA), combined with a Detail-Aware Loss (DAL) for pixel-level high-frequency supervision, achieving state-of-the-art detail fidelity in human-product image generation.

## Background & Motivation

Human-product images — depicting interactions between people and products — are essential in advertising, e-commerce, and digital marketing. The core challenge in generating such images is **preserving product details with high fidelity**: shape, color, patterns, and text must be accurately reproduced, as even minor deviations can erode consumer trust.

Existing methods suffer from three key limitations:

**Insufficient data**: Large-scale, diverse training data for human-product images is scarce.

**Weak detail preservation**: Existing models (e.g., image customization, text-based editing) focus on global or high-level semantics and struggle to robustly maintain fine-grained details; the denoising process of diffusion models tends to "average out" or "hallucinate" content.

**Coarse supervision**: Reliance solely on latent-space MSE loss fails to provide precise pixel-level detail guidance.

Reference-based inpainting guides the inpainting process using a product reference image, yet existing methods (Paint-by-Example, ACE++, Insert Anything) still fall short of achieving high fidelity in texture, shape, and brand element reproduction.

## Method

### Overall Architecture

HiFi-Inpaint is built upon FLUX.1-Dev (MMDiT architecture). Given a text prompt $T$, a masked human image $\mathbf{I}_h$, and a product reference image $\mathbf{I}_p$, the model outputs an image $\mathbf{I}_g$ in which the product is seamlessly composited into the masked region. The framework introduces three key contributions: the HP-Image-40K dataset, a high-frequency-guided DiT framework with SEA, and the Detail-Aware Loss (DAL).

### Key Designs

1. **HP-Image-40K Dataset Construction**: Diptych-format images (left: product / right: human-product) are generated via FLUX.1-Dev, then filtered through Sobel edge detection for segmentation, YOLOv8+CLIP semantic filtering (computing CLIP similarity between cropped product regions and reference images), and InternVL text-consistency filtering, yielding 40,000+ high-quality samples. Each sample contains a text description, masked human image, product image, and target image. This self-synthesis and automatic filtering pipeline acquires large-scale, diverse data with minimal human annotation.

2. **High-Frequency-Guided DiT + Shared Enhancement Attention (SEA)**:

   - **High-Frequency Extraction**: The image is transformed to the frequency domain via DFT; a circular-mask high-pass filter (radius $r$) suppresses low-frequency components, and inverse DFT maps the result back to the spatial domain, yielding a high-frequency map $H(\mathbf{I}_p)$ that highlights textures, text, logos, and other fine details (more focused on salient details than Canny edge detection).
   - **Token Concatenation**: VAE-encoded tokens from the masked human image, product image, and noised target image are concatenated into a joint visual token: $\mathbf{z}_0 = \text{Concat}(\mathcal{E}(\mathbf{I}_h), \mathcal{E}(\mathbf{I}_p), N(\mathcal{E}(\mathbf{I}_{gt}), t))$; a corresponding high-frequency visual token is constructed as $\mathbf{z}_0' = \text{Concat}(\mathcal{E}(\mathbf{I}_h), \mathcal{E}(H(\mathbf{I}_p)), N(\mathcal{E}(\mathbf{I}_{gt}), t))$.
   - **SEA Core Formulation**: In each double-stream DiT block, a parameter-shared high-frequency branch is added. High-frequency features are fused into the original features via a learnable scalar weight $\alpha_i$, applied exclusively within the masked region to enhance fine-grained product features:
     $\mathbf{z}_i = B_i(\mathbf{z}_{i-1}) + \alpha_i \cdot \text{Mask}(B_i(\mathbf{z}_{i-1}'), \mathbf{M}_{ds})$
     Through parameter sharing, SEA introduces only one additional scalar $\alpha_i$ per layer, keeping the model compact. The learnable $\alpha_i$ outperforms a fixed value of 1 by avoiding visual artifacts and feature conflicts.

3. **Detail-Aware Loss (DAL)**: To address the inability of latent-space MSE loss to supervise fine-grained details precisely, DAL applies L2 supervision on the high-frequency components of the masked region in pixel space:
   $\mathcal{L}_{\text{DA}} = \|H(\hat{\mathbf{I}}_{gt}) \odot \mathbf{M} - H(\mathbf{I}_{gt}) \odot \mathbf{M}\|_2^2$
   where $H(\cdot)$ denotes high-frequency extraction and $\mathbf{M}$ is the mask. DAL compels the model to attend to high-frequency detail reconstruction, complementing the limitations of latent-space loss.

### Loss & Training

The total loss is the sum of the latent-space MSE loss and the pixel-level DAL:

$$\mathcal{L}_{\text{Overall}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{DA}}$$

Training uses flow matching with a learning rate of $5 \times 10^{-5}$, batch size of 24, for 10,000 steps at a resolution of $1024 \times 576$. Training data consists of approximately 14,000 internal samples combined with HP-Image-40K.

## Key Experimental Results

### Main Results

Evaluated on 1,000 test samples from HP-Image-40K at $1024 \times 576$ resolution:

| Method | CLIP-T↑(%) | CLIP-I↑(%) | DINO↑(%) | SSIM↑(%) | SSIM-HF↑(%) | LAION-Aes↑ | Q-Align-IQ↑ |
|---|---|---|---|---|---|---|---|
| Paint-by-Example | 31.6 | 69.1 | 63.4 | 54.0 | 34.9 | 4.09 | 4.06 |
| ACE++ | 34.9 | 93.1 | 90.7 | 58.3 | 37.2 | 4.18 | 4.00 |
| Insert Anything | 35.3 | 94.1 | 89.8 | 62.1 | 40.0 | 4.20 | 3.89 |
| FLUX-Kontext | 36.6 | 82.5 | 63.1 | 51.6 | 32.0 | 4.54 | 3.74 |
| **HiFi-Inpaint** | **36.1** | **95.0** | **91.9** | **63.4** | **42.9** | **4.40** | **4.36** |

HiFi-Inpaint achieves the best performance on visual consistency (CLIP-I, DINO, SSIM, SSIM-HF) and image quality (Q-Align-IQ).

### Ablation Study

| Config | Syn.Data | DAL | SEA | CLIP-I↑(%) | DINO↑(%) | SSIM↑(%) | SSIM-HF↑(%) | Note |
|---|---|---|---|---|---|---|---|---|
| A | ✗ | ✗ | ✗ | 91.8 | 85.4 | 57.7 | 38.4 | Baseline |
| B | ✓ | ✗ | ✗ | 94.5 | 89.9 | 62.4 | 41.2 | +Dataset, large gains |
| C | ✓ | ✓ | ✗ | 94.6 | 90.7 | 62.3 | 41.8 | +DAL, detail metrics improve |
| E | ✓ | ✓ | ✓ | 95.0 | 91.9 | 63.4 | 42.9 | All components, best |

### Key Findings

- **Dataset contributes most**: HP-Image-40K yields the most significant performance gains (A→B: DINO +4.5, SSIM +4.7).
- **SEA is critical for detail**: C→E shows consistent improvement across all consistency metrics; qualitative results demonstrate that SEA enables more precise texture and pattern alignment.
- **DAL targets detail reconstruction**: B→C yields a SSIM-HF gain of 0.6, confirming that DAL effectively guides high-frequency detail reconstruction.
- **User study** (31 participants / 11 groups): HiFi-Inpaint achieves substantially higher preference rates over all competing methods in text alignment (36.4%), visual consistency (41.5%), and generation quality (39.5%).
- **FLUX-Kontext underperforms**: Its general instruction-editing paradigm struggles to establish effective correspondence between the reference image and the masked region, frequently generating standalone product images rather than composited outputs.

## Highlights & Insights

- **Systematic exploitation of high-frequency information**: High-frequency maps extracted in the frequency domain are integrated throughout the entire framework — as input to an auxiliary branch (SEA) and as the target for pixel-level supervision (DAL) — forming a coherent "high-frequency enhancement" system.
- **Parameter-efficient SEA design**: By sharing the double-stream DiT block parameters and introducing only one learnable scalar $\alpha_i$ per layer, SEA incurs negligible additional parameter overhead.
- **Practical self-synthesis data pipeline**: The approach exploits the consistency generation capability of FLUX.1-Dev combined with multi-stage automatic filtering to construct large-scale, high-quality data at low cost.
- **SSIM-HF as a new metric**: Applying a high-pass filter to generated images prior to computing SSIM provides a more precise assessment of detail preservation capability.

## Limitations & Future Work

- The framework targets only human-product scenarios; generalization to broader reference-based inpainting tasks (e.g., scene replacement, multi-object composition) remains unverified.
- HP-Image-40K is synthetically generated by FLUX.1-Dev, which may introduce generation bias; the gap relative to real-world data has not been thoroughly analyzed.
- High-frequency extraction relies on a fixed-radius $r$ circular high-pass filter; different product types may require adaptive strategies.
- Inference efficiency is not reported; the auxiliary SEA branch still requires a forward pass at inference time.
- Evaluation is conducted solely on the authors' own test set, without validation on standard public benchmarks.

## Related Work & Insights

- **FLUX-Kontext**, as a general-purpose editing model, performs poorly in this scenario, demonstrating that reference-based inpainting tasks require dedicated detail preservation mechanisms.
- The **high-frequency supervision paradigm** is transferable to other generation tasks requiring detail preservation (e.g., texture transfer, virtual try-on).
- The **self-synthesis + automatic filtering** pipeline is generalizable to other generation tasks that lack large-scale training data.
- The SEA design principle — shared parameters with learnable scalar weights — is broadly applicable to any DiT framework requiring auxiliary information enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐ The systematic integration of high-frequency information within a DiT framework (SEA + DAL) constitutes a novel and effective design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven metrics, four baselines, comprehensive ablations, and a user study combining quantitative and qualitative evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a complete logical chain from motivation to method to experiments.
- Value: ⭐⭐⭐⭐ Direct applicability to e-commerce and advertising scenarios, with strong transferability of the proposed design principles.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[ICCV 2025\] CompleteMe: Reference-based Human Image Completion](../../ICCV2025/image_generation/completeme_reference-based_human_image_completion.md)
- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_virtual_tryon_efficient.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)

<!-- RELATED:END -->
