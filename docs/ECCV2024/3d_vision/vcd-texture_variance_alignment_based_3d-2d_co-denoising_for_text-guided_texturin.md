---
title: >-
  [Paper Note] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing
description: >-
  [ECCV 2024][3D Vision] Proposes VCD-Texture, which unifies 2D and 3D self-attention learning (JNP) during the Stable Diffusion denoising process, addresses the variance decay issue caused by rasterization through Variance Alignment (VA), and handles inconsistent regions using inpainting refinement, achieving high-fidelity and highly consistent 3D texture synthesis.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 2adddd351a263d3c
---

# VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing

**Conference**: ECCV 2024  
**arXiv**: [2407.04461](https://arxiv.org/abs/2407.04461)  
**Code**: None  
**Area**: 3D Vision

## TL;DR

Proposes VCD-Texture, which unifies 2D and 3D self-attention learning (JNP) during the Stable Diffusion denoising process, addresses the variance decay issue caused by rasterization through Variance Alignment (VA), and handles inconsistent regions using inpainting refinement, achieving high-fidelity and highly consistent 3D texture synthesis.

## Background & Motivation

### Background

**Background**: Existing text-guided texture synthesis methods overlook the modality gap between 2D diffusion models and 3D objects.

### Limitations of Prior Work

**Limitations of Prior Work**: Progressive inpainting methods (e.g., TEXTure, Text2Tex) generate inconsistent textures from opposing viewpoints.

### Key Challenge

**Key Challenge**: Synchronous multi-view denoising methods (e.g., SyncMVD) ignore 3D spatial correspondences across different views.

### Mechanism

**Mechanism**: The process of feature aggregation $\rightarrow$ rasterization suffers from severe variance bias, leading to overly smooth textures.

### Core Problem

**Core Problem**: Rasterization, essentially as a convex combination operation, causes variance decay due to Jensen's inequality ($Var(\text{convex combination}) \le \text{convex combination of } Var$), degrading the capability of diffusion models to generate high-frequency details.

## Method

### Overall Architecture

Two-stage pipeline:
1. **3D-2D Co-Denoising**: Uses JNP (Joint Noise Prediction) and MV-AR (Multi-view Aggregation-Rasterization + Variance Alignment) during the denoising process of Stable Diffusion (SD).
2. **Inpainting Refinement**: Detects inconsistent regions and repairs them using Depth-SD.

### Key Designs

**JNP (Joint Noise Prediction)**:
- A 3D self-attention branch is incorporated into each Transformer block of the UNet.
- Multi-view 2D foreground features are lifted to the 3D space via rendering-projection relationships, partitioning the 3D attention receptive fields by voxel grids.
- 2D self-attention maintains global long-range consistency, while 3D self-attention captures cross-view local correspondences.
- Alternating between two different grid sizes eliminates boundary isolation effects.
- Entirely training-free (all parameters are frozen, adjusting only the attention receptive fields).

**MV-AR + VA (Multi-view Aggregation-Rasterization + Variance Alignment)**:
- Aggregates multi-view latent features onto 3D vertices using barycentric coordinates and view/distance scores, which are then rasterized back to 2D.
- **Variance Alignment (Core Theoretical Contribution)**: Since rasterization is fundamentally a convex combination, Jensen's inequality dictates that $Var(\text{convex combination}) \le \text{convex combination of } Var$, systematically reducing feature variance after rasterization.
- Solution: Precisely calculate the target variance using the variance and covariance of the aggregated 3D features, and then normalize and rescale the rasterized 2D features.

**Inpainting Refinement**:
- Computes the variance of multi-view pixels on 3D vertices, using a threshold of $\lambda=0.005$ to identify inconsistent vertices.
- Renders the 3D mask to 2D, performs dilation, and inpaints the region using Depth-SD.

### Loss & Training

No extra training loss is required; the entire process is executed during the inference stage of pre-trained SD. Variance alignment serves as a deterministic statistical correction operation.

## Key Experimental Results

### Main Results

Quantitative comparison across three sub-datasets:

| Dataset | Method | FID↓ | ClipFID↓ | ClipScore↑ | ClipVar↑ |
|--------|------|------|----------|------------|----------|
| SubTex | TEXTure | 150.21 | 26.92 | 26.90 | 82.37 |
| SubTex | Text2Tex | 112.41 | 16.26 | 30.08 | 81.45 |
| SubTex | SyncMVD | 65.30 | 16.76 | 28.78 | 81.93 |
| SubTex | Repaint3D | 78.65 | 10.65 | 30.88 | 78.96 |
| SubTex | **VCD-Texture** | **56.29** | **6.84** | **31.65** | **83.97** |
| SubObj | SyncMVD | 34.00 | 5.60 | 30.08 | 84.52 |
| SubObj | Repaint3D | 29.77 | 4.44 | 30.30 | 81.45 |
| SubObj | **VCD-Texture** | **21.19** | **2.33** | **30.42** | **83.64** |

### Ablation Study

| Component | FID↓ | ClipFID↓ | ClipScore↑ | ClipVar↑ |
|------|------|----------|------------|----------|
| MV-AR only | 58.87 | 7.39 | 31.32 | 82.87 |
| +DS (Distance Score) | 58.40 | 7.17 | 31.41 | 82.92 |
| +JNP | 57.30 | 6.98 | 31.57 | 83.45 |
| +VA | 56.70 | 6.90 | 31.60 | 83.80 |
| +IR (Inpainting Refinement) | **56.29** | **6.84** | **31.65** | **83.97** |

### Key Findings

- VCD-Texture achieves state-of-the-art results in FID and ClipFID, reaching an FID of 21.19 on SubObj (compared to 29.77 for Repaint3D).
- Variance Alignment (VA) effectively prevents SyncMVD-like methods from generating overly smoothed textures.
- The 3D attention in JNP significantly improves cross-view consistency (yielding a ClipVar Gain of 0.58).
- Inpainting refinement successfully bridges the inherent discrepancy between the latent domain and the pixel domain.
- Being training-free, the method generalizes well and demonstrates robust performance across diverse 3D objects and complex textual descriptions.

## Highlights & Insights

- Elegant and rigorous theoretical analysis of Variance Alignment: leveraging Jensen's inequality, it explains the root cause of blurry textures produced by aggregation-rasterization methods.
- Ingenious training-free design of 3D attention in JNP, which only modifies the receptive fields of self-attention without changing any parameters.
- Introduces the first 3D texture evaluation benchmark (featuring 3 subsets and 4 metrics), filling a significant gap in the field.
- The identification of the variance bias problem is universally applicable and potentially impacts all 3D generation pipelines relying on feature aggregation.

## Limitations & Future Work

- The 9-view layout may fail to fully cover highly complex geometries (e.g., deep cavities or thin, elongated structures).
- The inpainting refinement stage operates autoregressively, which might introduce new inconsistencies.
- As a training-free solution, it may not outperform training-based approaches (such as Paint3D) in extreme edge cases.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The theoretical insights on variance alignment are highly valuable.
- Effectiveness: ⭐⭐⭐⭐ — Leads comprehensively across quantitative benchmarks.
- Practicality: ⭐⭐⭐⭐⭐ — Training-free and directly applicable.
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)

</div>

<!-- RELATED:END -->
