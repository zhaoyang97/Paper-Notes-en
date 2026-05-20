---
title: >-
  [Paper Note] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment
description: >-
  [ICCV 2025][3D Vision][Compositional 3D Generation] This paper proposes REPARO, which generates compositional 3D assets from a single image by first reconstructing individual object meshes separately and then performing…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Compositional 3D Generation"
  - "Differentiable Rendering"
  - "Optimal Transport"
  - "Layout Alignment"
  - "Multi-Object Scene"
date: 2026-05-08
content_hash: 3780d14f4cb71ecc
---

# REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment

**Conference**: ICCV 2025
**arXiv**: [2405.18525](https://arxiv.org/abs/2405.18525)  
**Code**: [Project Page](https://reparo-3d.github.io/)  
**Area**: 3D Vision
**Keywords**: Compositional 3D Generation, Differentiable Rendering, Optimal Transport, Layout Alignment, Multi-Object Scene

## TL;DR

This paper proposes REPARO, which generates compositional 3D assets from a single image by first reconstructing individual object meshes separately and then performing layout alignment via optimal transport-based differentiable rendering.

## Background & Motivation

Existing image-to-3D generation models face fundamental challenges in multi-object scenes:

**Dataset Bias**: 3D training data predominantly consists of center-aligned single objects; preprocessing re-centers inputs, introducing inherent positional bias.

**Occlusion Handling**: Occluded objects are incorrectly represented as merged entities, causing the generated assets to be erroneously fused.

**Monolithic Mesh Representation**: Outputs are single meshes, requiring users to segment individual objects through error-prone post-processing.

The core mechanism of REPARO follows a divide-and-conquer strategy: leveraging the strengths of existing single-object generation models to reconstruct objects individually, then optimizing the layout via differentiable rendering.

## Method

### Overall Architecture

A two-stage pipeline:
1. **Single-Object Reconstruction**: Extract individual objects from the input image → complete occluded regions → generate 3D assets using off-the-shelf models.
2. **Layout Alignment**: Place all objects in a unified coordinate system → optimize spatial arrangement via differentiable rendering.

### Single-Object Extraction and Reconstruction

- SAM is used to segment each object and obtain binary masks.
- Occluded objects are completed via Stable Diffusion-based inpainting.
- Images are cropped to center each object, adapting to the center bias of reconstruction models.
- DreamGaussian or TripoSR is used to generate 3D assets.

### Long-Range Appearance Loss via Optimal Transport

The conventional pixel-wise $L_2$ loss yields zero gradients when rendered and reference images have no overlapping regions, causing optimization to become trapped in local minima. Optimal transport is introduced to establish global correspondences.

The cost function integrates RGB color, depth, and position:

$$c_{ij} = \alpha \cdot \|I_i - I_j^{ref}\|_2 + \beta \cdot \|D_i - D_j^{ref}\|_2 + \gamma \cdot \|p_i - p_j\|_2$$

The transport matrix $T$ is solved via Sinkhorn divergence to establish a one-to-one mapping $\sigma(\cdot)$, and the loss is defined as:

$$L_a(I, I^{ref}) = \frac{1}{N} \sum_i^N c_{i\sigma(i)}$$

Gradient propagation:

$$\frac{\partial L_a}{\partial \theta} = \frac{\partial L_a}{\partial I} \cdot \frac{\partial I}{\partial \theta} + \frac{\partial L_a}{\partial D} \cdot \frac{\partial F_D}{\partial I} \cdot \frac{\partial I}{\partial \theta} + \frac{\partial L_a}{\partial p} \cdot \frac{\partial p}{\partial \theta}$$

### High-Level Semantic Loss

A frozen DINOv2 is used to extract features and align the semantic relationship between rendered and reference images:

$$L_s(I, I^{ref}) = \frac{1}{K} \sum_i^K \|f_i - f_i^{ref}\|_2$$

### Total Loss

$$L(I, I^{ref}) = \lambda L_a(I, I^{ref}) + (1-\lambda) L_s(I, I^{ref})$$

The optimized parameters are per-object translation $t$ and scale $s$; rotation is excluded under the assumption that image-to-3D models preserve consistent orientation with the input.

## Key Experimental Results

### Main Results — Compositional 3D Asset Generation

| Method | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| DreamGaussian | 0.807 | 13.28 | 0.802 | 0.240 |
| TripoSR | 0.795 | 17.25 | 0.863 | 0.218 |
| Wonder3D | 0.801 | 13.69 | 0.807 | 0.238 |
| REPARO♣ | **0.833** | 17.28 | 0.826 | 0.234 |
| REPARO♠ | 0.822 | **17.75** | **0.865** | **0.216** |

REPARO achieves a significant improvement in CLIP score, validating the compositional approach's effectiveness in enhancing semantic consistency.

### Resource Consumption

| Stage | VRAM | Time |
|-------|------|------|
| SAM Segmentation | 6 GB | <1 s |
| Inpainting | 8 GB | 20 s |
| Single-Object Generation (TripoSR) | 6 GB | <1 s |
| Layout Alignment | 6 GB | 90 s |
| **Total (TripoSR)** | **≤8 GB** | **120 s** |

REPARO completes the full pipeline under a VRAM constraint of ≤8 GB, demonstrating practical applicability.

## Highlights & Insights

1. **Elegant Problem Decomposition**: Fully exploits the strengths of existing single-object models, circumventing the inherent difficulties of joint multi-object generation.
2. **Optimal Transport Resolves Vanishing Gradients**: The OT loss provides long-range correspondences, addressing the zero-gradient issue of standard $L_2$ loss in non-overlapping regions.
3. **Multi-Modal Cost Function**: Jointly considering RGB, depth, and positional signals enhances alignment robustness.
4. **Plug-and-Play Design**: Compatible with any image-to-3D model.

## Limitations & Future Work

- Performance depends on the quality of SAM and inpainting models; inpainting quality directly affects single-object reconstruction.
- Completion capability is limited for severely occluded objects with large missing regions.
- Rotation parameters are not optimized, relying on the assumption that image-to-3D models produce orientation-consistent outputs.
- The Sinkhorn algorithm incurs considerable computational overhead on large images.

## Related Work & Insights

- DreamFusion, DreamGaussian: Single-object 3D generation.
- TripoSR, Zero-1-to-3: Image-to-3D methods.
- DROT: Optimal transport for differentiable rendering.

## Rating

- Novelty: ⭐⭐⭐⭐ (OT-based layout alignment via differentiable rendering is novel)
- Technical Depth: ⭐⭐⭐⭐ (Elegantly designed loss functions)
- Experimental Thoroughness: ⭐⭐⭐ (Quantitative experiments could be more comprehensive)
- Practical Value: ⭐⭐⭐⭐ (Strong demand for real-world multi-object scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LACONIC: A 3D Layout Adapter for Controllable Image Creation](laconic_a_3d_layout_adapter_for_controllable_image_creation.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)

</div>

<!-- RELATED:END -->
