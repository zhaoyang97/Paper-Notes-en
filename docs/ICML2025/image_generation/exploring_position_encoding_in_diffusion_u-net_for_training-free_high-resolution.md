---
title: >-
  [Paper Note] Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation
description: >-
  [ICML 2025][Image Generation][High-resolution image generation] Through an in-depth analysis of the insufficient propagation of positional information caused by zero-padding in the convolutional layers of the diffusion U-Net at high resolutions, this paper proposes the Progressive Boundary Complement (PBC) method. PBC constructs progressive virtual boundaries inside the feature maps to enhance positional information propagation, achieving high-quality training-free high-resol…
tags:
  - "ICML 2025"
  - "Image Generation"
  - "High-resolution image generation"
  - "position encoding"
  - "U-Net"
  - "zero-padding"
  - "training-free method"
  - "diffusion models"
date: 2026-05-08
content_hash: 5a2c4c2a5efbffce
---

# Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation

**Conference**: ICML 2025  
**arXiv**: [2503.09830](https://arxiv.org/abs/2503.09830)  
**Authors**: Feng Zhou, Pu Cao, Yiyang Ma, Lu Yang, Jianqin Yin  
**Code**: Undisclosed  
**Area**: Image Generation  
**Keywords**: High-resolution image generation, position encoding, U-Net, zero-padding, training-free method, diffusion models  

## TL;DR

Through an in-depth analysis of the insufficient propagation of positional information caused by zero-padding in the convolutional layers of the diffusion U-Net at high resolutions, this paper proposes the Progressive Boundary Complement (PBC) method. PBC constructs progressive virtual boundaries inside the feature maps to enhance positional information propagation, achieving high-quality training-free high-resolution image generation.

## Background & Motivation

### Background
Latent Diffusion Models (LDMs) like Stable Diffusion are trained with a fixed resolution (e.g., $64\times64$ latent space in SD-2.1). When directly using a pre-trained U-Net to denoise latents at higher resolutions, the generated images exhibit **repetitive patterns** and **layout confusion**—i.e., "image elements appearing in incorrect positions". This issue becomes particularly pronounced at scaling factors above 2x.

### In-depth Analysis of Position Encoding Mechanism
This paper presents a key finding—the source and propagation mechanism of positional information in U-Net:

**Sole Source of Positional Information**: U-Net's attention layers contain no position embeddings; all positional information originates solely from the zero-padding mechanism in the convolutional layers (Islam et al., 2020).

**Propagation Path**: Zero-padding provides positional information at the borders of the feature maps, which progressively propagates from the boundaries to the center through stacked convolutional layers.

**High-Resolution Failure Mechanism**: When the feature map size increases, the propagation path becomes longer, and the positional information cannot fully propagate to the center, leading to positional encoding inconsistency.

### Limitations of Prior Work
Re-examining existing methods from a unified perspective of positional information propagation:
- **Dilated Convolution Methods** (ScaleCrafter, FouriScale): Accelerate positional information propagation by expanding the receptive fields of convolutional kernels, which are engineering-driven strategies lacking theoretical foundation.
- **Multi-stage Resolution Scaling Methods** (DemoFusion, FouriScale): Sustain positional information via progressive upsampling from the original resolution, but the rigid alignment constraints limit content diversity.
- **Attention Adjustment Methods** (Attn-Entro): Adjust attention scaling factors, failing to fundamentally address the propagation of positional information.
- **Dynamic Feature Map Methods** (HiDiffusion): Modify feature map dimensions to ease repetition, but similarly fail to address the root cause.

### Design Motivation
Directly address the problem from the perspective of positional information complementation—instead of forcibly aligning the denoising processes of different resolutions, this method increases the emission sources of positional information within high-resolution feature maps, thereby shortening the propagation distance.

## Method

### Overall Architecture
The PBC method is entirely training-free and directly applied during the inference process of a pre-trained U-Net. Its core operation is creating virtual image boundaries inside the feature maps to serve as relays for positional information.

### Theoretical Foundation of Zero-Padding Position Encoding
- In standard convolution, zero-padding (padding=1) provides boundary pixels with a different computational environment from center pixels.
- Border features "perceive" their boundary locations; for a $3\times3$ convolutional kernel, each conv layer propagates positional information 1 pixel toward the center.
- When the feature map increases from $64\times64$ to $128\times128$, the number of layers required to propagate to the center doubles, yet the U-Net depth remains fixed.
- The paper validates this hypothesis through quantitative experiments (comparing positional information intensity between boundary and center regions).

### Progressive Boundary Complement (PBC)
**1. Virtual Boundary Construction**: Virtual boundaries are inserted at specific locations inside the feature maps. These boundaries are a special type of **unidirectional padding** that simulates the zero-padding effect, creating new sources of positional information inside the feature maps.

**2. Hierarchical Progressive Placement**: Virtual boundaries are placed hierarchically (progressively) rather than uniformly, step-by-step correcting the positional encoding inconsistency from the outside in, while effectively expanding the image boundary range perceived by the model.

**3. Dynamic Boundary Adjustment**: The location and quantity of virtual boundaries are dynamically adjusted based on the ratio between the target and training resolutions, ensuring effective operation across various scale factors.

### Fundamental Difference from Dilated Convolution Methods
ScaleCrafter expands the receptive field by increasing the dilation rate, which is equivalent to accelerating the propagation speed of positional information. PBC approaches from another direction—without altering the propagation speed, it increases the emission sources (virtual boundaries) of positional information, thereby shortening the actual propagation distance.

### Byproduct of Content Diversity Enhancement
The virtual boundaries of PBC expand the perceived image borders. At high resolutions, the model "believes" the image consists of multiple sub-regions, each with its independent boundary perception, generating richer details and more diverse content. This is a pleasant byproduct—the high-resolution image not only avoids repetition but also has richer content than its low-resolution counterpart.

## Key Experimental Results

### Main Results: Quantitative Comparison with SOTA Methods (SD 2.1)

| Method | Type | Training Requirement | Repetitive Pattern Elimination | Content Richness | Overall Quality |
|------|------|---------|-------------|-----------|---------|
| SD Direct Generation | Baseline | None | ✗ Severe repetition | Low | Poor |
| Attn-Entro | Attention Adj. | None | Partially mitigated | Low | Lower-medium |
| ScaleCrafter | Dilated Conv | None | Mostly eliminated | Medium | Medium |
| HiDiffusion | Feature Map Adj. | None | Mostly eliminated | Medium | Medium |
| DemoFusion | Multi-stage | None | Eliminated | Medium (limited by original resolution) | Fairly Good |
| **PBC (Ours)** | Virtual Boundary | **None** | **Completely eliminated** | **High (surpassing original resolution)** | **Best** |

### Ablation Study: Contribution of PBC Components

| Configuration | Virtual Boundary | Progressive Placement | Dynamic Adjustment | Generation Quality | Content Richness |
|------|---------|-----------|---------|---------|-----------|
| Baseline (Direct High-Res Generation) | ✗ | ✗ | ✗ | Poor (Repetitive/Confused) | Low |
| + Fixed Position Single-Layer Boundary | ✓ | ✗ | ✗ | Medium | Medium |
| + Hierarchical Progressive Placement | ✓ | ✓ | ✗ | Fairly Good | Fairly Good |
| + Dynamic Resolution Adaptation (Full PBC) | ✓ | ✓ | ✓ | **Best** | **Best** |

Ablation Conclusion: Virtual boundaries are the fundamental building blocks (necessary condition), progressive placement brings significant quality improvements, and dynamic adjustment further optimizes adaptation across resolutions.

### Verification of Positional Information Propagation

| Resolution (Relative to Training Resolution) | Boundary Positional Information Intensity | Center Positional Information Intensity | Decay Level | Center Intensity after PBC |
|------------------------|----------------|----------------|---------|-------------|
| 1× (Training Resolution) | Strong | Medium | Low | — |
| 1.5× | Strong | Weaker | Medium | Restored to upper-medium |
| 2× | Strong | Weak | High | Restored to medium |
| 3× | Strong | Very weak | Very high | Restored to lower-medium |

## Highlights & Insights

- **Deep Analysis of Root Cause**: Systematically attributes high-resolution generation degradation to the insufficient propagation of zero-padding positional information for the first time, rather than simple distribution shifts or attention failures, providing a unified explanatory framework.
- **Elegant Method Design**: PBC merely inserts virtual boundaries inside the feature maps without modifying the network architecture, requiring no training, and introducing minimal computational overhead.
- **Strong Alignment between Theory and Method**: Insufficient positional information propagation $\rightarrow$ add intermediate sources $\rightarrow$ virtual boundaries; the logical chain is complete and rigorous.
- **Gains Beyond Just "Fixing"**: Virtual boundaries not only eliminate repetitive patterns but also yield additional benefits in enhancing content diversity.
- **Unified Framework**: Unified the understanding of existing methods through the lens of positional propagation—dilated convolutions accelerate propagation, multi-stage methods sustain propagation, and attention adjustments indirectly affect propagation.
- **Universality**: Applicable to all diffusion models based on U-Net with convolutional zero-padding.

## Limitations & Future Work

- **Only Applicable to U-Net Architectures**: PBC relies on the convolutional zero-padding mechanism and is not directly applicable to Transformer-based architectures like DiT and U-ViT, which use explicit position encodings.
- **Extremely High Resolutions Unverified**: It remains unknown whether increasing the number of virtual boundaries under extremely high upscaling factors like 8x or 16x introduces new artifacts or breaks global consistency.
- **Compatibility with Conditional Generation**: Compatibility with conditional control methods like ControlNet and IP-Adapter is not discussed, where virtual boundaries might interfere with the propagation of conditional signals.
- **Parameter Selection for Virtual Boundaries**: How to optimally select specific parameters for progressive placement (interval, count, hierarchical levels), and whether an adaptive search strategy exists remains to be explored.
- **Completeness of Quantitative Evaluation**: Due to cache truncation, complete quantitative data such as FID/IS was not obtained.

## Related Work & Insights

### Taxonomy of Training-Free High-Resolution Methods
1. **Architecture Modification**: ScaleCrafter (dilated convolutions), HiDiffusion (dynamic feature maps), Attn-Entro (attention entropy adjustment).
2. **Multi-Stage Scaling**: DemoFusion, FouriScale, MultiDiffusion—progressive resolution scaling to avoid one-step jumps.
3. **Patch-Level Methods**: MultiDiffusion, SyncDiffusion—independent patch generation followed by fusion.

### Key Connections
- **Islam et al. (2020)**: First proposed that zero-padding provides implicit position encoding for CNNs; this paper extends this observation to high-resolution generation in diffusion models.
- **FreeU (Si et al., 2024)**: Explored U-Net internal mechanisms (spectral characteristics of skip connections); this paper complements this understanding from the perspective of position encoding.
- **ScaleCrafter (He et al., 2023)**: Dilated convolutions accelerate propagation vs PBC adding information sources—two complementary solution pathways.

### Insights
- The implicit role of position encoding in visual generation far exceeds expectations—seemingly trivial operations like zero-padding bear critical spatial information encoding functions.
- The concept of "adding relays to shorten propagation distance" can be generalized to other scenarios requiring enhanced spatial information propagation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Analyzes high-resolution degradation from the perspective of zero-padding position encoding propagation, offering a fresh and profound perspective, and proposes an elegant solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Includes quantitative comparison, ablations, and positional propagation validation (some data is missing due to cache truncation).
- Writing Quality: ⭐⭐⭐⭐ — The narrative chain of problem $\rightarrow$ analysis $\rightarrow$ method $\rightarrow$ validation is clear and complete.
- Value: ⭐⭐⭐⭐⭐ — Provides a unified theoretical explanation and a concise solution, making a fundamental contribution to the understanding of internal mechanisms in diffusion models.

## Supplementary Technical Analysis

### Quantitative Verification of Zero-Padding Propagation
Measuring the intensity of position information in the center region of feature maps under different resolutions shows that the central position information significantly decays as resolution increases, consistent with theoretical analysis.

### Fundamental Difference: PBC vs Dilated Convolution
Dilated convolution accelerates propagation but maintains fixed boundaries, while PBC creates new internal boundaries—the latter not only corrects positions but also expands perceived borders, allowing for richer content.

### Implications for DiT Architectures
DiTs use attention + position encodings instead of convolutions, thereby remaining unaffected by this issue. However, U-Net remains a widely used architecture, making PBC of direct value to it.

### Value Beyond "Fixing"
Existing methods focus on forcibly aligning positional information, which restricts content diversity. PBC's virtual boundaries allow the model to "perceive" larger image spaces, generating richer details.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](../../ECCV2024/image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[ICML 2025\] Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation](editable_noise_map_inversion_encoding_target-image_into_noise_for_high-fidelity_.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[CVPR 2026\] Training-free, Perceptually Consistent Low-Resolution Previews with High-Resolution Image for Efficient Workflows of Diffusion Models](../../CVPR2026/image_generation/training-free_perceptually_consistent_low-resolution_previews.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](../../CVPR2025/image_generation/diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)

</div>

<!-- RELATED:END -->
