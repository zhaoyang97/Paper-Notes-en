---
title: >-
  [Paper Note] SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing
description: >-
  [CVPR 2025][Image Generation][Text-driven Motion Generation] This work proposes SALAD, a skeleton-aware latent diffusion model that explicitly models fine-grained interactions among joints, frames, and text using a skeleton-temporal structured VAE and denoiser, and achieves zero-shot text-driven motion editing via cross-attention maps.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-driven Motion Generation"
  - "Skeleton-aware Diffusion"
  - "Latent Space"
  - "Attention Modulation"
  - "Zero-shot Editing"
date: 2026-05-08
content_hash: 28c06f5a33d4df03
---

# SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing

**Conference**: CVPR 2025  
**arXiv**: [2503.13836](https://arxiv.org/abs/2503.13836)  
**Code**: [Project Page](https://kaist-visual-ai-group.github.io/SALAD/)  
**Area**: Image Generation/Motion Generation  
**Keywords**: Text-driven Motion Generation, Skeleton-aware Diffusion, Latent Space, Attention Modulation, Zero-shot Editing

## TL;DR

This work proposes SALAD, a skeleton-aware latent diffusion model that explicitly models fine-grained interactions among joints, frames, and text using a skeleton-temporal structured VAE and denoiser, and achieves zero-shot text-driven motion editing via cross-attention maps.

## Background & Motivation

Text-driven motion generation has important applications in games, movies, and interactive media. Although diffusion models have achieved remarkable progress in this field, existing methods possess two key limitations: (1) representing poses as a single vector ignores spatial interactions among joints, and compressing text into a single vector overlooks word-level nuances, leading to detailed losses in the generated results; (2) pre-trained models lack interpretable intermediate representations, requiring tedious extra efforts such as manual masking, optimization, or fine-tuning for downstream editing tasks.

In the image domain, methods such as Prompt-to-Prompt have demonstrated that cross-attention maps can establish correspondences between text and spatial layouts, thereby enabling zero-shot editing. However, the motion generation field lacks similar capabilities because oversimplified representations restrict the rich interactions between text and motion. SALAD aims to address both problems simultaneously through a skeleton-temporal structured latent space and decoupled attention mechanisms.

## Method

### Overall Architecture

SALAD consists of three components: (1) a skeleton-temporal VAE that constructs a structured latent space $\mathbf{z} \in \mathbb{R}^{N' \times J' \times D}$; (2) a skeleton-aware denoiser performing text-conditional diffusion generation in this space; (3) a zero-shot editing method based on cross-attention modulation.

### Key Design 1: Skeleton-temporal VAE

**Function**: Constructing a compact motion latent space that preserves skeletal and temporal structures.

**Mechanism**: Skeleton-Temporal Convolutions (STConv) are utilized to decouple the joint and frame dimensions. Information exchange is performed using graph convolutions on adjacent joints and 1D convolutions on adjacent frames, respectively: $\text{STConv}(\mathbf{h}) = \text{SkelConv}(\mathbf{h}) + \text{TempConv}(\mathbf{h})$. Dimension reduction is conducted via Skeleton-Temporal Pooling (STPool), which aggregates adjacent joints along the skeletal dimension to preserve topological homeomorphism, and performs 1D pooling in the temporal dimension. This ultimately retains 7 atomic joints (root, spine, head, arms, legs). The encoder compresses $N \times J \times D$ to $N' \times J' \times D$.

**Design Motivation**: Direct operation of diffusion models on the raw space ($N \text{ frames} \times J \text{ joints} \times D \text{ dimensions}$) faces the curse of dimensionality and computational bottlenecks. Skeleton-temporal pooling compresses the dimensions while maintaining the topological structure, rendering diffusion sampling more efficient.

### Key Design 2: Skeleton-aware Denoiser

**Function**: Modeling fine-grained interactions among joints, frames, and text within the structured latent space.

**Mechanism**: The denoiser is composed of $L$ Transformer layers, where each layer contains: (1) Temporal Attention (TempAttn) to model inter-frame relationships; (2) Skeleton Attention (SkelAttn) to model inter-joint relationships; (3) Cross-Attention (CrossAttn) to interact with CLIP-encoded word-level text features. Every module is followed by a FiLM layer to modulate features based on diffusion timesteps. The model adopts the $\mathbf{v}$-prediction parameterization: $\mathbf{v}_t = \alpha_t \epsilon - \sigma_t \mathbf{x}$, which is more stable than $\epsilon$-prediction under high noise levels.

**Design Motivation**: Decoupling skeletal and temporal attention enables the model to independently process spatial relationships (which joints coordinate in movement) and temporal relationships (the rhythm of frame sequences), while cross-attention provides a fine-grained association between each word token and each skeleton-temporal unit.

### Key Design 3: Attention Modulation Zero-shot Editing

**Function**: Leveraging cross-attention maps of the pre-trained SALAD to achieve fine-tuning-free text-driven motion editing.

**Mechanism**: Four modulation strategies are proposed: (1) Word Replacement: swapping the attention maps of the source and target prompts; (2) Prompt Refinement: appending new attention maps for added words to enrich semantics; (3) Attention Re-weighting: amplifying or reducing the attention values of specific words; (4) Attention Mirroring: swapping attention values of symmetric body parts (e.g., left and right arms) to generate mirrored motions.

**Design Motivation**: Similar to image diffusion models, the cross-attention maps of SALAD capture the correspondence between text words and motion skeleton-temporal units, making it possible to achieve editing by directly modulating these maps without extra optimization or training.

### Loss & Training

VAE training: $\mathcal{L}_{\text{VAE}} = \mathcal{L}_\mathbf{m} + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}$ (motion reconstruction + joint position + joint velocity + KL divergence). Denoiser training: $\mathcal{L}_{\text{denoiser}} = \|\hat{\mathbf{v}}_t - \mathbf{v}_t\|_2^2$ (velocity prediction MSE), assisted by classifier-free guidance.

## Key Experimental Results

### Main Results: HumanML3D Text-driven Motion Generation

| Method | R-Precision Top-1 ↑ | FID ↓ | MM-Dist ↓ | Diversity → |
|------|-----|-----|-----|-----|
| MLD | 0.481 | 0.473 | 3.196 | 9.724 |
| MoMask | 0.521 | 0.045 | 2.958 | - |
| MotionGPT | 0.492 | 0.232 | 3.096 | 9.528 |
| **SALAD** | **0.524** | **0.064** | **2.926** | 9.549 |
| Real motion | 0.511 | 0.002 | 2.974 | 9.503 |

### KIT-ML Dataset

| Method | R-Precision Top-1 ↑ | FID ↓ | MM-Dist ↓ |
|------|-----|-----|-----|
| MLD | 0.390 | 0.404 | 3.204 |
| **SALAD** | **0.424** | **0.321** | **3.054** |

### Key Findings

- SALAD significantly outperforms all prior methods in text-motion alignment (R-Precision, MM-Dist), proving the effectiveness of the skeleton-temporal structured representation and word-level cross-attention.
- Although its FID is secondary to MoMask (0.064 vs. 0.045), it is superior in text alignment, showing that the method focuses more on semantic accuracy.
- Attention mirroring experiments verify that the cross-attention maps indeed encode the correspondences between body parts and text words.
- Zero-shot editing completely eliminates the need for extra optimization/fine-tuning, enabling diverse editing solely through attention map manipulation.

## Highlights & Insights

1. **Structured Latent Space**: The skeleton-temporal compression into 7 atomic joints preserves topological information while dramatically reducing the computational cost of diffusion sampling.
2. **Paradigm Shift from Image to Motion**: Successfully transferred the attention modulation editing concept of Prompt-to-Prompt from 2D images to 3D motions.
3. **Selection of v-prediction**: Balances the advantages of $\epsilon$- and $\mathbf{x}$-prediction, demonstrating higher stability under high noise levels.

## Limitations & Future Work

- Skeleton pooling down to 7 atomic joints might lose fine-grained motion details, such as finger movements.
- Zero-shot editing relies on the quality of attention maps; complex semantic editing may be less precise.
- The current work only handles single-person motion, leaving multi-person interaction scenarios unaddressed.
- The FID is slightly inferior to token-based methods (such as MoMask), leaving room for improvement in generation quality.

## Related Work & Insights

- **Prompt-to-Prompt**: An attention modulation method in image editing, successfully transferred to the motion domain.
- **AnimatableGaussians / Skeleton-aware Networks**: Pioneers of skeleton-temporal convolution architectures; this work integrates them into the VAE component of the diffusion model.
- **MLD**: Pioneering work in motion latent diffusion; SALAD significantly outperforms it through the structured latent space.

## Rating

⭐⭐⭐⭐ — The design concept of skeleton-temporal structuring is clear, forming a complete technical stack from VAE to the denoiser and to editing. The text-motion alignment achieves SOTA, and the zero-shot editing capability serves as a practical highlight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[ICCV 2025\] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait](../../ICCV2025/image_generation/float_generative_motion_latent_flow_matching_for_audio-driven_talking_portrait.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)

</div>

<!-- RELATED:END -->
