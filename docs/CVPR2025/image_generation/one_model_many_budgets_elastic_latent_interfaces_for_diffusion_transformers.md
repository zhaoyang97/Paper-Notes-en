---
title: >-
  [Paper Note] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers
description: >-
  [CVPR 2025][Image Generation][DiT] Reveals that DiT computation is uniformly distributed across spatial tokens (failing to reallocate redundant computation to difficult regions), and proposes ELIT. ELIT inserts a variable-length latent interface (Read/Write cross-attention) in DiT, randomly drops tail latents during training to learn an importance ordering, and adjusts the number of latents during inference to achieve a smooth quality-FLOPs trade-off…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "DiT"
  - "elastic inference"
  - "latent token"
  - "computation allocation"
  - "variable-length interface"
date: 2026-05-08
content_hash: a0967f2b9c0fb435
---

# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**Conference**: CVPR 2025  
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)  
**Code**: [https://snap-research.github.io/elit](https://snap-research.github.io/elit)  
**Area**: Image Generation  
**Keywords**: DiT, elastic inference, latent token, computation allocation, variable-length interface

## TL;DR

Reveals that DiT computation is uniformly distributed across spatial tokens (failing to reallocate redundant computation to difficult regions), and proposes ELIT. ELIT inserts a variable-length latent interface (Read/Write cross-attention) in DiT, randomly drops tail latents during training to learn an importance ordering, and adjusts the number of latents during inference to achieve a smooth quality-FLOPs trade-off, reducing FID by 53% on ImageNet 512px.

## Background & Motivation

**Background**: DiT achieves SOTA quality in image/video generation through a simple Transformer design, but FLOPs are tied to resolution—fixed computation uniformly distributed across all spatial locations.

**Limitations of Prior Work**: Two issues—(a) the inference computation budget cannot be flexibly adjusted (lacks support for latency-quality trade-offs), and (b) simple and complex regions receive the same amount of computation, wasting resources.

**Key Challenge**: Confirmed by zero-padding experiments that when DiT is given extra tokens (zeroed patches), it does not leverage the redundant computation to improve image quality. Attention maps show that zeroed tokens only attend to other zeroed tokens, indicating that DiT cannot reallocate computation.

**Goal**: Enable DiT to flexibly allocate computation: assign more computation to difficult regions and adjust the total computational cost on demand during inference.

**Key Insight**: Introduce a latent domain as an "intermediate layer" for computation—the Read layer selectively pulls information from spatial tokens to the latents, the Core Transformer blocks operate on the latents, and the Write layer writes back to space.

**Core Idea**: Decouple image resolution from computational cost using a variable-length latent interface, achieving both non-uniform computation allocation and an elastic inference budget.

## Method

### Overall Architecture

Insert a three-stage structure into the standard DiT: Spatial Head (a few spatial transformer layers for initial processing) $\rightarrow$ Read (cross-attention from spatial to latent) $\rightarrow$ Core Blocks (standard transformer on $K$ latent tokens) $\rightarrow$ Write (cross-attention from latent back to spatial) $\rightarrow$ Spatial Tail. The training objective remains unchanged (rectified flow).

### Key Designs

1. **Latent Interface (Variable-length latent tokens)**:

    - $K$ learnable latent tokens pull information from spatial tokens via Read cross-attention.
    - The Read layer naturally learns to prioritize difficult regions (high-loss regions) while ignoring simple regions (such as zero-padded regions).
    - The Write layer broadcasts the updated latents back to spatial tokens.
    - The number of latents is a user-controlled "knob"—directly setting the compute budget per step.

2. **Tail Token Dropping Training**:

    - Tail latent tokens are randomly dropped during training (similar to tail dropping).
    - Effect: Latents automatically learn an importance ordering—the preceding latents capture the global structure, while the trailing latents refine details.
    - During inference, trimming the tail latents smoothly reduces computation without retraining.

3. **Grouping Mechanism**:

    - Spatial tokens and latent tokens are grouped, and cross-attention is only performed within groups.
    - This reduces cross-attention complexity while preserving spatial locality.

4. **Autoguidance**:

    - Replace the unconditional model in CFG with a "weak model" using fewer latent tokens $\rightarrow$ reduces inference cost by approximately 33%.
    - CCFG (Cheap CFG): Drop class conditioning only in the guidance term and use the version with fewer tokens for guidance.

### Loss & Training

- The Rectified Flow objective remains unchanged.
- Randomized latent counts (multi-budget training): 60 budgets @512px, 16 budgets @256px.
- Train for 500K steps using Adam, EMA $\beta=0.9999$.

## Key Experimental Results

### Main Results (ImageNet-1K 512px, CFG)

| Model | FID ↓ | FDD ↓ | IS ↑ |
|------|------|------|------|
| DiT-XL | 9.5 | 233.6 | 86.4 |
| **ELIT-DiT-XL (multi-budget)** | **4.5 (-53%)** | **98.2 (-58%)** | **147.0 (+70%)** |
| U-ViT-XL | 5.3 | 125.9 | 117.2 |
| **ELIT-U-ViT (multi-budget)** | **3.8 (-28%)** | **83.1 (-34%)** | **159.3 (+36%)** |
| HDiT-XL | 6.3 | 150.7 | 107.3 |

### Ablation Study

| Configuration | Description |
|------|------|
| ELIT single-budget vs multi-budget | Multi-budget training is consistently better (40% FID reduction vs DiT) |
| Adjusting latent number vs reducing sampling steps | ELIT's latent count refinement yields a superior quality-FLOPs trade-off curve |
| Zero-padding experiment | DiT fails to utilize extra computation; ELIT successfully leverages it (matching the FID of DiT-B/1) |

### Key Findings

- **ELIT yields consistent improvements across all tested architectures**: DiT (+53%), U-ViT (+28%), HDiT (+23%)—indicating it is a general improvement rather than an architecture-specific trick.
- **More pronounced advantages at higher resolutions**: Improvement at 512px is much larger than at 256px, showing that higher resolutions contain greater pixel redundancy and make dynamic computation reallocation more valuable.
- **Autoguidance for free**: Using the few-token version for guidance saves 33% of the inference cost without loss of quality.
- **Compatible with TeaCache**: Can be used on top of training-free acceleration methods.
- **Larger models benefit more**: Scaling experiments from DiT-S to DiT-XL show that ELIT's relative gain increases as the model size grows.

## Highlights & Insights

- **The zero-padding experiment serves as an extremely simple and powerful motivation**: A 3-line experiment exposes DiT's uniform compute allocation problem. This methodology of "designing experiments to expose design flaws" is highly learning-worthy.
- **"Drop-in" design philosophy**: ELIT only adds two cross-attention layers, keeping the RF objective and DiT structure unchanged. This minimal-modification design philosophy makes the method highly practical—it can be directly applied to DiT/U-ViT/HDiT/MMDiT.
- **Tail dropping $\rightarrow$ Importance ordering**: Random dropping during training forces the model to automatically learn that preceding latents are important while trailing ones are secondary. This shares common ground with token importance ranking in NLP.

## Limitations & Future Work

- **Read/Write layers introduce extra overhead**: Although they are lightweight cross-attentions, their overhead ratio becomes non-negligible for extremely short sequences (low resolutions).
- **Unverified on text-to-image**: All experiments are based on ImageNet conditional generation, without being validated on large-scale T2I models.
- **Fixed grouping strategy**: Currently uses fixed grouping; adaptive grouping might be superior.
- **Synergy with other efficiency methods**: Combined usage with distillation, quantization, etc., remains unexplored.

## Related Work & Insights

- **vs RINs/FITs**: RIN uses similar latent tokens + read/write, but deviates from the DiT architecture and requires special optimizers (LAMB). ELIT's core advantage is full compatibility with standard DiT training.
- **vs MaskDiT / TREAD**: These methods drop tokens during training for acceleration but must restore all tokens during inference. ELIT allows flexible adjustment even during inference.
- **vs TiTok / DCAE**: These utilize variable-length tokens in autoencoders, while ELIT introduces variable-length tokens inside the generative model itself.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea is clear and elegant (latent interface + tail dropping), though the concept of latent tokens is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 architectures, 2 resolutions, video generation, multiple FLOPs curves, zero-padding experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly persuasive motivational experiments, clear and rich figures/tables.
- Value: ⭐⭐⭐⭐⭐ Practical improvement for the DiT family, drop-in applicable directly to industrial models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](tinyfusion_diffusion_transformers_learned_shallow.md)

</div>

<!-- RELATED:END -->
