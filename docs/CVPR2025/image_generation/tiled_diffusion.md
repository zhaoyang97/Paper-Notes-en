---
title: >-
  [Paper Note] Tiled Diffusion
description: >-
  [CVPR 2025][Image Generation][Image tiling] Tiled Diffusion is proposed to enable seamless and coherent tileable image generation across various tiling scenarios ranging from self-tiling to complex many-to-many connections, by introducing tiling and similarity constraints directly in the latent space of diffusion models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image tiling"
  - "diffusion models"
  - "seamless textures"
  - "panoramic synthesis"
  - "latent space constraints"
date: 2026-05-08
content_hash: 31fbc6d123f9b9e8
---

# Tiled Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2412.15185](https://arxiv.org/abs/2412.15185)  
**Code**: [Project Page](https://arxiv.org/abs/2412.15185)  
**Area**: Image Generation  
**Keywords**: Image tiling, diffusion models, seamless textures, panoramic synthesis, latent space constraints

## TL;DR

Tiled Diffusion is proposed to enable seamless and coherent tileable image generation across various tiling scenarios ranging from self-tiling to complex many-to-many connections, by introducing tiling and similarity constraints directly in the latent space of diffusion models.

## Background & Motivation

Image tiling—joining different images seamlessly to create coherent visual scenes—is crucial in texture creation, game asset development, and digital arts. Traditional manual tiling construction is time-consuming and lacks flexibility. Existing automated methods (e.g., STI using inpainting models, AT using asymmetric tiling/circular padding) are primarily limited to simple self-tiling scenarios or texture synthesis.

The core limitations are: (1) STI generates each image independently and then inpaints the boundaries, failing to share tiling information during the generation process; (2) AT achieves circular tiling by modifying the network padding, but only supports self-tiling; (3) no existing method supports complex many-to-many tiling (where multiple edges of multiple images can connect to one another).

Tiled Diffusion operates directly on latent representations, creating all tiled images simultaneously during the generation process and naturally sharing the necessary tiling information.

## Method

### Overall Architecture

The tiling problem is formally defined with an image set $\mathcal{I} = \{I_1, ..., I_N\}$, where each image has four edges (right, left, top, bottom). A constraint set $\mathcal{C} = \{C_1, ..., C_M\}$ is defined, where each constraint $C_j = \{A_j, B_j\}$ specifies the tiling relationship between two groups of edges. Tiling and similarity constraints are applied at each step of the diffusion process to ensure global consistency and local seamless transitions.

### Key Design 1: Latent Tiling Constraint

**Function**: Ensure global structural consistency among tiled images.

**Mechanism**: In each diffusion step, a sub-region of a latent representation from constraint set $B_j$ is cropped and copied as padding into the latent space of $A_j$, and vice versa. These padded latents are processed together during the diffusion step, and the final decoded images are cropped back to their original dimensions. The context window size $w$ ($0 \leq w \leq H_{\text{latent}}/2$) controls the smoothness of the transition: a larger $w$ yields smoother transitions but fewer variations.

**Design Motivation**: Adjacent regions in the latent space of diffusion models naturally interact. By copying the context of tiling edges into the adjacent image's latent space, consistency in style and content is guaranteed during the generation process without requiring any post-processing.

### Key Design 2: Similarity Constraint

**Function**: Eliminate artifacts in many-to-many tiling scenarios.

**Mechanism**: When a constraint set involves multiple edges ($|A_j| > 1$ or $|B_j| > 1$), it is necessary to ensure that the content near these edges is similar. The latent representations of all edges in the same constraint set are copied to be identical within a small window (e.g., 5 pixels wide) near the borders.

**Design Motivation**: The tiling constraint only ensures global structural consistency, but in many-to-many scenarios, any two edges from the same group can be adjacent, requiring local contents near the borders to be highly similar. The key difference is that the tiling constraint affects the cropped region (indirectly), whereas the similarity constraint directly affects the retained region (directly).

### Key Design 3: Round-Robin Context Selection

**Function**: Balance the impact of multiple constraints on generation.

**Mechanism**: When a constraint involves multiple edges, different edges are selected in a round-robin manner at each diffusion step to provide the padding context. Constraints of different orientations are handled by rotating the latent space. This ensures each edge is exposed to all potential matches throughout the diffusion process.

**Design Motivation**: Uniformly cycling through all possible tiling combinations within a certain number of steps prevents any specific connections from being neglected.

### Loss & Training

No additional training loss is involved—the method is applied as an inference-time constraint during the standard diffusion sampling process. It supports both text-to-image and image-to-image modes, with the latter encoding and adding noise to the input image before applying the same constraints during the diffusion process.

## Key Experimental Results

### Main Results: Comparison of Tiling Quality (1000 LAION Prompts)

| Method | Scenario | FID↓ | TS↓ | CLIP-Score↑ | LPIPS↓ |
|------|------|------|-----|-------------|--------|
| AT | Self-tiling | 49.2 | 0.03 | 0.29 | 0.79 |
| STI | Self-tiling | 59.2 | 0.03 | 0.31 | 0.77 |
| **Tiled Diffusion** | **Self-tiling** | **47.9** | **0.03** | **0.30** | - |
| STI | One-to-one | 77.1 | 0.14 | 0.23 | 0.77 |
| **Tiled Diffusion** | **One-to-one** | - | **0.03** | **0.30** | - |
| **Tiled Diffusion** | **Many-to-many** | - | **0.03** | **0.29** | - |

### Ablation Study

| Configuration | TS↓ | CLIP-Score↑ |
|------|-----|-------------|
| Full Method | 0.03 | 0.30 |
| w/o Tiling Constraint (TC) | 0.29 | 0.31 |
| w/o Similarity Constraint (SC, self-tiling) | 0.03 | 0.30 |
| w/o Similarity Constraint (SC, many-to-many) | 0.12 | 0.28 |

### Key Findings

- The tiling constraint is a crucial component—removing it causes the TS to surge from 0.03 to 0.29 (similar to standard generation).
- The similarity constraint is critical in many-to-many scenarios: without SC, the TS increases from 0.03 to 0.12 (degrading tiling quality).
- STI achieves a TS of 0.14 in the one-to-one scenario, which is significantly worse than Tiled Diffusion's 0.03, due to STI's approach of independent generation followed by inpainting.
- The method scales well with tiling complexity ($n = |A_j| = |B_j|$ from 1 to 5), with the TS remaining largely constant.

## Highlights & Insights

1. **Flexible Formal Definition**: Unifies self-tiling, one-to-one, and many-to-many scenarios under a single framework based on edge-set constraint definitions.
2. **Inference-Time Constraints**: Requires no training or fine-tuning, directly integrating into the standard diffusion inference process, and is compatible with SD 1.5/2.0/XL/3.0 and ControlNet.
3. **Latent Context Sharing**: Generates all tiled images simultaneously with shared context, which is fundamentally superior to post-hoc inpainting paradigms.

## Limitations & Future Work

- The context window $w$ in the tiling constraint requires manual tuning to balance transition smoothness and content variations.
- Large-scale multi-image tiling may be constrained by GPU memory.
- While achievements in horizontal tiling for 360° synthesis are excellent, vertical boundaries remain unprocessed.
- Future research can explore extending this tiling mechanism to 3D textures.

## Related Work & Insights

- **STI (Seamless Tile Inpainting)**: Creates self-tiled images by shifting image quadrants and inpainting, but lacks context sharing during generation.
- **AT (Asymmetric Tiling)**: Modifies SD's padding to circular padding, yet is limited to self-tiling and lacks rotation support.
- **Wang Tiles**: A classic tiling theory; Tiled Diffusion can be viewed as its modern extension in the deep learning era.

## Rating

⭐⭐⭐⭐ — Clear problem definition, elegant and straightforward methodology; the unique capability for many-to-many tiling opens up new application domains. Serving as an inference-time constraint without training yields high practicality. The applications in texture synthesis and 360° panoramic generation are highly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Color Alignment in Diffusion](color_alignment_in_diffusion.md)
- [\[CVPR 2025\] Decentralized Diffusion Models](decentralized_diffusion_models.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[CVPR 2025\] Erasing Undesirable Influence in Diffusion Models (EraseDiff)](erasing_undesirable_influence_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
