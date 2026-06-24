---
title: >-
  [Paper Note] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians
description: >-
  [ECCV 2024][3D Vision] Proposes WaSt-3D, which reformulates style transfer as an optimal transport problem between two Gaussian distributions using 3D Gaussian Splatting representations. By matching the 3D distributions of the content and style scenes via Sinkhorn divergence, it achieves the first 3D scene-to-scene geometric style transfer.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: b9b684d397e26382
---

# WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians

**Conference**: ECCV 2024  
**arXiv**: [2409.17917](https://arxiv.org/abs/2409.17917)  
**Code**: [Project Page](https://compvis.github.io/wast3d/)  
**Area**: 3D Vision

## TL;DR

Proposes WaSt-3D, which reformulates style transfer as an optimal transport problem between two Gaussian distributions using 3D Gaussian Splatting representations. By matching the 3D distributions of the content and style scenes via Sinkhorn divergence, it achieves the first 3D scene-to-scene geometric style transfer.

## Background & Motivation

### Background

**Background**: Existing 3D style transfer methods primarily modify textures/colors, making them almost incapable of transferring geometric styles.

### Limitations of Prior Work

**Limitations of Prior Work**: The "assemblage" technique in paintings (e.g., Arcimboldo assembling portraits with vegetables, Picasso assembling instruments with wooden blocks) reflects the artistic idea of assembling overall content from local stylistic elements, which prior methods cannot capture.

### Key Challenge

**Key Challenge**: NeRF-based methods (ARF, StyleRF, SNeRF) optimize within the RGB feature space, failing to alter the underlying geometry.

### Key Insight

**Core Idea**: Shift style transfer from "feature space generation" to "explicit matching of two 3D particle distributions."

## Method

### Overall Architecture

1. Train the content and style scenes separately as regularized Gaussian Splattings.
2. Cluster the content scene into $N$ sub-regions.
3. Find the best-matching style region for each content cluster (constrained optimization).
4. Minimize the Sinkhorn divergence between each content-style cluster pair.

### Key Designs

**Isotropic Gaussian Regularization**:
- In standard 3DGS training, Gaussians can be stretched into needle-like shapes, causing artifacts during segmentation.
- Anisotropy regularization is added to minimize the ratio between the maximum and minimum scales.
- Uniform size regularization is added to constrain all Gaussians to approach a similar size.

**Wasserstein-2 Distance & Sinkhorn Divergence**:
- Using entropy-regularized $W_2$ distance makes optimal transport solvable and smooth.
- $\text{Sinkhorn Divergence} = \mathcal{W}_{2,\gamma}^2(p_s, p_c) - \frac{1}{2}\mathcal{W}_{2,\gamma}^2(p_s, p_s) - \frac{1}{2}\mathcal{W}_{2,\gamma}^2(p_c, p_c)$, eliminating the bias and guaranteeing the distance is zero when two distributions are identical.
- $\gamma$ controls the smoothness of the transport plan: a large $\gamma$ yields a globally averaged effect, while a small $\gamma$ achieves precise one-to-one matching.

**Scene Segmentation (Resolving Large-Scale OT Intractability)**:
- K-Means clusters the content scene into $N=400$ clusters.
- For each content cluster, the best match is found by fitting it to the style scene via translation, rotation, and scaling.
- A k-nearest neighbor search selects the corresponding style Gaussian subset.
- The problem is decomposed into $N$ small-scale OT sub-problems.

**Optimization Objective**:
$$\mathcal{L}_{opt} = \sum_{i=1}^{N} \mathcal{SD}_{2,\gamma}^2(C_i, D_i(C_i))$$

The Sinkhorn divergence is computed simultaneously on coordinates and luminance/color channels. Optimizing luminance helps maintain the volumetric sense of light and shadow of the original scene.

### Loss & Training

Two-stage optimization: The first stage optimizes $\{t_i, R_i, S_i\}$ to find matches (Eq. 8); the second stage optimizes the color and coordinates of the style clusters to minimize the Sinkhorn divergence (Eq. 10). An optional ARAP (As-Rigid-As-Possible) elastic deformation loss can be added to prevent style clusters from stretching excessively.

## Key Experimental Results

### Main Results

| Method | CLIP High-level Similarity ↑ | Human Preference ↑ | Time ↓ | VRAM ↓ |
|------|----------------|----------|-------|-------|
| ARF | 74.79% | 12.5% | 11 min | 9GB |
| StyleRF | 74.94% | 1.5% | 18 min | 6GB |
| SNeRF | 76.99% | 10.5% | 30 min | 8GB |
| **WaSt-3D** | **84.40%** | **75.5%** | **8 min** | 16GB |

### Ablation Study

Effects of different parameter combinations for Sinkhorn divergence optimization:

| Optimization Parameters | Effect Description |
|----------|----------|
| Coordinates only | Shape matches but lacks light/shadow hierarchy |
| Coordinates + Luminance | Retains original scene light/shadow, recommended configuration |
| Coordinates + Luminance + Color | Most complete but potentially over-constrained |

Normal visualization comparison: WaSt-3D completely preserves the 3D geometric texture of the style scene, while other methods introduce noise due to optimizing in the RGB space.

### Key Findings

- Human preference of 75.5% far exceeds other methods (<12.5%), validating the visual appeal of geometric style transfer.
- CLIP high-level detail similarity of 84.40% significantly leads (vs SNeRF 76.99%), indicating that 3D stylistic details are faithfully preserved.
- The optimization time of 8 minutes is the shortest, as diffusion/NeRF-based methods require additional training while WaSt-3D only performs OT optimization.
- Reducing the number of content clusters ($400 \rightarrow 200$) decreases content fidelity.
- Replacing Sinkhorn divergence with ARAP elastic loss yields poorer results, demonstrating that distribution matching is more suitable than geometric constraints.

## Highlights & Insights

- Paradigm shift: Transforms style transfer from "optimizing in latent space" to "explicit matching of 3D distributions."
- Using optimal transport theory (Wasserstein distance) to address 3D style transfer is an entirely new technical approach.
- The idea of "assembling" the content scene using style elements directly aligns with the assemblage technique in art.
- Regularized Gaussian Splatting serves as an excellent base representation for 3D style transfer—explicit, manipulable, and efficient.

## Limitations & Future Work

- The style scene must also be represented by 3D Gaussian Splatting; 2D style images cannot be used directly.
- Requires 16GB of VRAM, leading to high memory overhead for large scenes.
- The cluster count $N$ requires manual adjustment.
- Certain extreme geometric deformations may cause unnatural effects.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Unique combination of 3D-to-3D style transfer and optimal transport
- Effectiveness: ⭐⭐⭐⭐ — Substantially leading user study results
- Practicality: ⭐⭐⭐ — Requires 3D style scenes as inputs
- Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
