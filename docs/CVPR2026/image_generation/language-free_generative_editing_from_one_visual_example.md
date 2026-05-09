---
title: >-
  [Paper Note] Language-Free Generative Editing from One Visual Example
description: >-
  [CVPR 2026][Image Generation][Image Editing] This paper reveals a critical text-visual alignment failure in text-guided diffusion models on simple visual transformations such as rain, haze, and blur, and proposes the VDC framework — which learns a purely visual conditioning signal from a single visual example pair (before and after transformation) to guide diffusion-based editing, requiring neither text nor training. VDC surpasses text-based and fine-tuning-based methods on tasks including deraining, dehazing, and denoising.
tags:
  - CVPR 2026
  - Image Generation
  - Image Editing
  - Diffusion Models
  - Visual Conditioning
  - Language-Free
  - Training-Free
date: 2026-05-08
content_hash: c39417071119f154
---

# Language-Free Generative Editing from One Visual Example

**Conference**: CVPR 2026
**arXiv**: [2603.25441](https://arxiv.org/abs/2603.25441)
**Code**: [Project Page](https://omaralezaby.github.io/vdc/)
**Area**: Image Generation
**Keywords**: Image Editing, Diffusion Models, Visual Conditioning, Language-Free, Training-Free

## TL;DR

This paper reveals a critical text-visual alignment failure in text-guided diffusion models on simple visual transformations such as rain, haze, and blur, and proposes the VDC framework — which learns a purely visual conditioning signal from a single visual example pair (before and after transformation) to guide diffusion-based editing, requiring neither text nor training. VDC surpasses text-based and fine-tuning-based methods on tasks including deraining, dehazing, and denoising.

## Background & Motivation

Text-guided diffusion models have achieved remarkable progress in image editing; however, surprisingly, SOTA methods **fail severely on simple, everyday visual transformations such as rain, blur, and haze**.

**Key Challenge**: Diffusion models are trained on image-caption pairs and only learn concepts explicitly described in captions. **Visual phenomena that are rarely or ambiguously described** (e.g., raindrops, haze) are poorly aligned in text-visual space — attention maps under the text condition "rain" remain object-centric and unrelated to rain characteristics.

Existing solutions:
- Fine-tuning: computationally expensive and data-demanding
- Stronger text conditioning: still cannot overcome the fundamental limit of text-visual alignment

**Core Idea**: **The editing capability of diffusion models is not lost — it is merely hidden by text.** Diffusion models already encode rich visual representations; accessing these representations through vision rather than language is feasible.

## Method

### Overall Architecture

VDC (Visual Diffusion Conditioning):
1. Takes as input a visual example pair (before $R_B$ and after $R_A$ transformation)
2. Optimizes a lightweight MLP to generate a visual conditioning embedding $C^s$
3. Performs DDIM inversion on the real input image, then applies conditioning guidance for editing
4. An inversion correction step preserves detail fidelity

### Key Designs

1. **Condition Steering**:
    - Function: Guides unconditional diffusion sampling using posterior score steering
    - Mechanism: Derived from the posterior score function, the noise prediction is rewritten as a weighted combination of conditional and unconditional predictions: $\epsilon_\theta(z_t, -C^s) = (1-w) \cdot \epsilon_\theta(z_t, C^s) + w \cdot \epsilon_\theta(z_t, \phi)$. For removal-type tasks (deraining/dehazing), negative conditioning direction is used to steer away from high-density regions of the target feature.
    - Design Motivation: Editing the inverted image ($out = Z(\phi) + Z(C_\theta)$) rather than generating from scratch ($out = Z(C_\theta)$) resembles a global residual connection in image-to-image networks, avoiding generation artifacts.

2. **Condition Generator**:
    - Function: Generates editing condition embeddings from a single visual example pair
    - Mechanism: Inspired by implicit neural representations (INR), a lightweight 3-layer MLP maps token indices to conditioning embeddings, with Fourier feature transformation applied to inputs for enhanced expressiveness. An independent $\text{MLP}_t$ is trained per diffusion timestep $t$, with the optimization objective: $\mathcal{L} = \|Z_0^B - Z^A\|_2^2 + \|D(Z_0^B) - R_A\|_2^2$
    - Design Motivation: Fully decoupled from text space. The MLP+INR formulation is more stable than directly optimizing text embeddings and enables optimization over all 77 tokens (text embedding optimization typically handles only a small number of tokens); the continuous function ensures natural inter-token communication.

3. **DDIM Inversion Correction**:
    - Function: Reduces accumulated errors from DDIM inversion
    - Mechanism: Iteratively optimizes the inversion starting point: $Z_p \leftarrow Z_p - \text{AdamGrad}(\|\hat{Z}_0 - Z_0\|_2^2)$, i.e., repeatedly executing forward-backward cycles to adjust the noisy latent for more accurate reconstruction.
    - Design Motivation: Accumulated errors in DDIM inversion lead to distortion in edited images; the correction step preserves perceptual quality without added complexity.

### Loss & Training

Condition generator optimization: $\mathcal{L} = \|Z_0^B - Z^A\|_2^2 + \|D(Z_0^B) - R_A\|_2^2$ (joint loss in latent and pixel space), optimized with Adam, with an independent MLP per diffusion timestep.

## Key Experimental Results

### Main Results

| Method | SR FID↓ | DeBlur FID↓ | DeNoise FID↓ | DeRain FID↓ | DeHaze FID↓ | Colorization FID↓ |
|------|---------|------------|-------------|------------|------------|------------------|
| P2P | 126.47 | 45.62 | 142.95 | 139.19 | 44.09 | 121.87 |
| Null-Opt | 73.48 | 51.89 | 160.88 | 167.61 | — | — |
| **VDC (Ours)** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

VDC establishes a new state of the art across all 6 benchmarks, outperforming both training-free and fully fine-tuned text-based editing methods.

### Ablation Study

| Configuration | Effect |
|------|------|
| Without inversion correction | LPIPS degrades significantly |
| Without Fourier features | Unstable condition generation |
| Global unified condition (non-per-step) | Reduced accuracy |
| Text embedding optimization instead of MLP | Unstable; only a small number of tokens can be optimized |

### Key Findings

- Diffusion model attention maps under text conditions such as "rain" are completely misaligned — remaining object-centric rather than degradation-centric
- Attention maps recovered by VDC correctly focus on rain streaks and haze regions
- A single visual example pair is sufficient to learn generalizable editing conditions
- VDC even surpasses methods specifically trained for image restoration tasks

## Highlights & Insights

- Reveals a fundamental failure of text-visual alignment on appearance-level transformations — a finding of independent scientific value
- The insight that "diffusion editing capability is hidden rather than lost" is central: visual conditioning can unlock capabilities inaccessible to text
- The INR-inspired condition generator design is elegant: continuous functions combined with Fourier features enable stable full-token optimization
- A truly training-free method with substantially lower computational cost than fine-tuning approaches

## Limitations & Future Work

- Requires a visual example pair (before and after transformation); obtaining such pairs may not always be straightforward
- Each new editing task requires re-optimizing the condition generator (~100 iterations)
- Validated only on image restoration/degradation tasks; applicability to semantic-level editing (e.g., object replacement) remains unexplored
- Relies on the quality of DDIM inversion; highly complex images may yield poor inversion results

## Related Work & Insights

- **vs. Prompt-to-Prompt / Null-Opt**: These methods modify text prompts or attention maps but still depend on text-visual alignment; VDC bypasses text entirely.
- **vs. InstructPix2Pix and variants**: Require large-scale instruction fine-tuning datasets and training; VDC requires zero training.
- **vs. Diffusion-based inverse problem methods**: Assume a known degradation operator (e.g., blur kernel) and cannot handle complex spatially-varying degradations; VDC learns from visual examples.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A paradigm shift from text to purely visual conditioning; revealing the text-visual alignment failure carries significant scientific value
- Experimental Thoroughness: ⭐⭐⭐⭐ Six editing benchmarks with multiple baselines, though semantic-level editing experiments are absent
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is exceptionally clear; the attention map visualizations in Figure 2 are highly convincing
- Value: ⭐⭐⭐⭐⭐ Offers important insights for diffusion-based editing and image restoration; the framework is concise and practical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2026\] Group Editing: Edit Multiple Images in One Go](group_editing_edit_multiple_images_in_one_go.md)
- [\[NeurIPS 2025\] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](../../NeurIPS2025/image_generation/sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)

</div>

<!-- RELATED:END -->
