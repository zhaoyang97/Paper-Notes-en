---
title: >-
  [Paper Note] SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation
description: >-
  [ECCV 2024][3D Vision][3D Human Generation] SemanticHuman-HD is proposed as the first 3D human image synthesis method that achieves semantic disentangling. By leveraging $K$ independent local generators and a 3D-aware super-resolution module, it enables semantically controllable human generation at `$1024^2$` resolution.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Human Generation"
  - "Semantic Disentangling"
  - "Neural Radiance Fields"
  - "Super-Resolution"
  - "GAN"
date: 2026-05-08
content_hash: a719f2afb2b0e8f2
---

# SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.10166](https://arxiv.org/abs/2403.10166)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Human Generation, Semantic Disentangling, Neural Radiance Fields, Super-Resolution, GAN

## TL;DR

SemanticHuman-HD is proposed as the first 3D human image synthesis method that achieves semantic disentangling. By leveraging $K$ independent local generators and a 3D-aware super-resolution module, it enables semantically controllable human generation at `$1024^2$` resolution.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Existing 3D human generation methods face two major challenges: (1) Inability to achieve semantically disentangled generation, meaning they cannot independently control different semantic components such as the body, tops, and bottoms; (2) Constrained by the high computational cost of NeRF, they can only synthesize images at a maximum resolution of `$512^2$`. Although methods like CNeRF attempt to use $K$ local generators, the disentanglement at the geometric level remains incomplete; AttriHuman-3D employs a single generator, which leaves different semantic parts still coupled.

## Method

### Overall Architecture

SemanticHuman-HD adopts a two-stage training scheme: the first stage synthesizes images, depth maps, semantic masks, and normal maps at `$256^2$` resolution; the second stage leverages a 3D-aware super-resolution module to upscale the resolution to `$1024^2$`.

### Key Designs

**Semantic Mapper**: Maps the random noise $z$ to $K=6$ semantic latent codes (corresponding to body, tops, outerwear, bottoms, shoes, and accessories). Consistency is enforced during training by constraining these latent codes to be equal, while they can be independently modified during inference to enable semantic editing.

**K Independent Local Generators**: Each generator independently produces a tri-plane representation. The key distinction lies in first converting local SDFs into local densities and then summing them up (rather than summing the global SDF first and then converting to density), which enables the disentanglement of both geometry and texture.

**3D-aware Super-Resolution Module**: Utilizes the depth maps and semantic masks generated in the first stage to guide the sampling, dramatically reducing the volume rendering sampling points from 432 ($72 \times 6$) to 11. This includes depth-guided sampling (aggregating depth from neighboring pixels) and semantic-guided sampling (rendering only the semantic part with the highest weight).

### Loss & Training

- **Stage 1**: $\mathcal{L}_1 = \mathcal{L}_{256} + \mathcal{L}_{AG3D}$, containing image, semantic, normal, and face discriminators.
- **Stage 2**: $\mathcal{L}_2 = \mathcal{L}_{1024} + \mathcal{L}_{upsample} + \mathcal{L}_{AG3D}$, freezing the generator while introducing a new upsampling consistency loss.

## Key Experimental Results

### Main Results

Quantitative comparison on the DeepFashion dataset (50K synthesized images):

| Method | Resolution | FID↓ | 1000×KID↓ | Local Editing | Semantic Disentanglement | 3D Garment Generation |
|------|--------|------|-----------|----------|----------|------------|
| AG3D | 512* | 11.33 | 5.75 | ✗ | ✗ | ✗ |
| EVA3D | 512 | 15.89 | 9.25 | ✗ | ✗ | ✗ |
| GSM | 512 | 15.78 | - | ✔ | ✗ | ✗ |
| AttriHuman-3D | 512* | 16.85 | - | ✔ | ✗ | ✗ |
| **Ours** | **512** | **10.04** | **5.02** | **✔** | **✔** | **✔** |
| **Ours** | **1024** | **8.70** | **4.04** | **✔** | **✔** | **✔** |

### Ablation Study

| Method | Resolution | FID↓ | 1000×KID↓ |
|------|--------|------|-----------|
| w/o SR (No Super-Resolution) | 256 | 13.47 | 9.13 |
| w/o DA (No Depth Aggregation) | 1024 | 9.38 | 4.56 |
| w/o UL (No Upsampling Loss) | 1024 | 13.52 | 8.18 |
| **Full Model** | **1024** | **8.70** | **4.04** |

Computational efficiency comparison: The proposed method requires only 10G VRAM at 512 resolution, which is significantly lower than EVA3D (34G) and AG3D (21G).

### Key Findings

- The `$1024^2$` resolution further reduces the FID compared to `$512^2$` (8.70 vs 10.04), demonstrating that the super-resolution module indeed enhances synthesis quality.
- The depth aggregation strategy effectively resolves the issue of depth discontinuity at edges.
- The upsampling loss is crucial for maintaining consistency between the low-resolution and high-resolution outputs.

## Highlights & Insights

1. **Fully independent semantic generation** is key to achieving dual disentanglement of both geometry and texture, which is fundamentally different from previous strategies utilizing a single generator with shared features.
2. The clever use of depth and semantic guidance reduces key sampling points from 432 to 11, rendering `$1024^2$` resolution generation feasible.
3. The disentangled representation opens up new applications, such as 3D garment generation, semantic virtual try-on, and cross-domain synthesis.

## Limitations & Future Work

- Dataset constraints: Performance is limited on rare poses and viewpoints.
- Achieving accurate 3D geometry under 2D supervision remains challenging.
- Hand generation quality needs improvement.

## Related Work & Insights

This work extends the tri-plane representation of EG3D to compositional human generation, with a conceptual approach similar to CNeRF but showing stronger independence. The design of the 3D-aware super-resolution module can be generalized to other NeRF-based generative models.

## Rating

- Novelty: ⭐⭐⭐⭐
- Utility: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] nuCraft: Crafting High Resolution 3D Semantic Occupancy for Unified 3D Scene Understanding](nucraft_crafting_high_resolution_3d_semantic_occupancy_for_unified_3d_scene_unde.md)
- [\[ECCV 2024\] High-Resolution and Few-shot View Synthesis from Asymmetric Dual-Lens Inputs](high-resolution_and_few-shot_view_synthesis_from_asymmetric_dual-lens_inputs.md)
- [\[ECCV 2024\] DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[CVPR 2025\] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](../../CVPR2025/3d_vision/disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)

</div>

<!-- RELATED:END -->
