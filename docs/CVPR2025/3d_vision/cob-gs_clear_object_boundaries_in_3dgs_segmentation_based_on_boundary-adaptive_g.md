---
title: >-
  [Paper Note] COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Proposes COB-GS, a boundary-adaptive Gaussian splitting technique driven by semantic gradient statistics. It jointly optimizes semantic information and visual texture to resolve the blurred object boundary issue in 3DGS segmentation, achieving clear object boundary segmentation while preserving visual quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "3D Segmentation"
  - "Boundary Optimization"
  - "Joint Semantic-Texture Optimization"
  - "Gaussian Splitting"
date: 2026-05-08
content_hash: 43883acda3fa7d89
---

# COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting

**Conference**: CVPR 2025  
**arXiv**: [2503.19443](https://arxiv.org/abs/2503.19443)  
**Code**: [https://github.com/ZestfulJX/COB-GS](https://github.com/ZestfulJX/COB-GS)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, 3D Segmentation, Boundary Optimization, Joint Semantic-Texture Optimization, Gaussian Splitting

## TL;DR

Proposes COB-GS, a boundary-adaptive Gaussian splitting technique driven by semantic gradient statistics. It jointly optimizes semantic information and visual texture to resolve the blurred object boundary issue in 3DGS segmentation, achieving clear object boundary segmentation while preserving visual quality.

## Background & Motivation

As a real-time radiance field rendering technology, 3D Gaussian Splatting (3DGS) provides new possibilities for 3D scene perception and interaction through its explicit representation. However, in 3D segmentation tasks, 3DGS faces a core issue of blurred boundaries: **Gaussian primitives possess volumetric properties and lack semantic guidance during training, causing them to span across object edges and "straddle" between two objects**.

Existing 3DGS segmentation methods fall into two categories: (1) Feature-based methods (learning semantic features for each Gaussian) suffer from high-dimensional feature ambiguity; (2) Mask-based post-processing methods assign labels to Gaussians using SAM-generated 2D masks, but the original scene reconstruction ignores semantic information, leading to ambiguous boundary Gaussian labels. Some methods (e.g., FlashSplat, SA3D) directly delete ambiguous boundary Gaussians, but this degrades visual quality.

**Key Challenge**: **Semantic segmentation requires precise object boundaries, but scene reconstruction in 3DGS only optimizes visual quality. These two objectives conflict in boundary regions**.

**Key Insight**: Proposes the first 3DGS segmentation method that jointly optimizes semantics and textures. It precisely locates and splits ambiguous boundary Gaussians through the gradient direction statistics of mask labels, while repairing textures on the optimized boundary structure, allowing the two aspects to promote each other mutually.

## Method

### Overall Architecture

The workflow is executed by alternating among three phases: (1) Mask optimization phase: optimizes mask labels and performs boundary-adaptive Gaussian splitting; (2) Texture optimization phase: restores scene textures on the correct boundary structures; (3) Robustness enhancement: extracts and optimizes tiny ambiguous Gaussians caused by inaccurate masks. Mask generation utilizes a two-stage method based on SAM2.

### Key Designs

1. **Boundary-Adaptive Gaussian Splitting**:
    - **Function**: Precisely locates and splits ambiguous Gaussians that span across object boundaries using semantic gradient statistics.
    - **Mechanism**: Introduces a continuous mask label $m_i \in (0,1)$ for each Gaussian and renders 2D masks via alpha-blending; analyzes the gradient direction of the mask loss with respect to $m_i$. If a Gaussian is supervised by both foreground pixels (negative gradient) and background pixels (positive gradient), it indicates that it spans boundaries. The supervision consistency strength is defined as $mask\_sig_{v,i} = |\frac{N_{v,i}^+ - N_{v,i}^-}{N_{v,i}^+ + N_{v,i}^- + \epsilon}|$. Gaussians whose multi-view average is below a threshold $\delta$ are identified as the set of ambiguous boundary Gaussians $\{G_i\}_B$, and large-scale Gaussians among them are split.
    - **Design Motivation**: Compared to existing methods that perform forward voting Gaussian-by-Gaussian (which is inefficient), leveraging gradient statistics from backpropagation is an efficient way to identify ambiguous Gaussians. The strong correlation between gradient directions and supervision categories provides theoretical guarantees.

2. **Boundary-Guided Scene Texture Restoration**:
    - **Function**: Restores texture degradation caused by splitting on the correct boundary structure after division.
    - **Mechanism**: Alternately optimizes mask labels (with geometry and texture frozen) and scene textures (with mask labels frozen) using the standard 3DGS texture loss $\mathcal{L}_{rgb} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$. Object-level semantic information effectively constrains the volume of boundary Gaussians, and optimizing texture on the accurate boundary structure improves novel view quality.
    - **Design Motivation**: Pure Gaussian splitting degrades visual quality, but precise boundaries in turn help to restore texture better—semantics and textures can mutually promote each other.

3. **Robustness Enhancement for Inaccurate Masks**:
    - **Function**: Handles tiny, non-convergent boundary Gaussians caused by inaccurate masks from pre-trained models (e.g., SAM2).
    - **Mechanism**: Distinguishes between two sources of boundary ambiguity—those caused by Gaussian volume (resolved via splitting) and those caused by inaccurate masks (requiring additional processing). After joint optimization, tiny ambiguous Gaussians with low $mask\_sig$ and small scale $s$ are identified and directly removed. These tiny Gaussians have minimal impact on overall visual quality but affect the clarity of segmentation boundaries.
    - **Design Motivation**: Masks predicted by 2D models are discrete and often inconsistent across different views, preventing the mask labels of some boundary Gaussians from converging.

### Loss & Training

- Mask Loss: $\mathcal{L}_{mask} = \sum M_{jk}^v \cdot M_{render}^v + \sum (1-M_{jk}^v) \cdot M_{render}^v$, constraining $m_i \in (0,1)$ to ensure convergence.
- Texture Loss: $\mathcal{L}_{rgb} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$.
- Alternating Optimization Strategy: Mask optimization (freeze geometry and texture) $\rightarrow$ Texture optimization (freeze mask labels) $\rightarrow$ Loop iteration.
- Multi-object Segmentation: Decomposed into sequential single-object 3DGS segmentation, updating the scene after optimizing the boundary of one object at a time.
- Two-stage Mask Generation: In the coarse stage, a low-confidence Grounding-DINO retrieves box prompts to drive SAM2's full-sequence prediction; in the fine stage, high-confidence queries supplement disconnected sub-sequences.

## Key Experimental Results

### Main Results (Segmentation on NVOS Dataset)

| Method | Type | mIoU (%) ↑ | mAcc (%) ↑ |
|------|------|-----------|-----------|
| NVOS | NeRF | 70.1 | 92.0 |
| ISRF | NeRF | 83.8 | 96.4 |
| SA3D | NeRF | 90.3 | 98.2 |
| SAGD | 3DGS | 90.4 | 98.2 |
| SA3D-GS | 3DGS | 90.7 | 98.3 |
| SAGA | 3DGS | 90.9 | 98.3 |
| FlashSplat | 3DGS | 91.8 | 98.6 |
| **COB-GS** | 3DGS | **92.1** | **98.6** |

### Ablation Study

| Component Configuration | mIoU (%) ↑ | mAcc (%) ↑ |
|---------|-----------|-----------|
| No Optimization (Baseline) | 91.2 | 98.3 |
| + Boundary-Adaptive Gaussian Splitting (BAGS) | 91.9 | 98.5 |
| + BAGS + Boundary-Guided Texture Restoration (BGTR) | 91.9 | 98.4 |
| + BAGS + BGTR + Robustness Enhancement (RAEM) | **92.1** | **98.6** |

### Texture Quality (PSNR)

| Method | Average PSNR ↑ |
|------|------------|
| Original Scene (Vanilla) | 23.06 |
| Mask Optimization Only (M.O) | 22.61 (Decrease) |
| Texture Optimization Only (T.O) | 23.07 |
| **Joint Optimization (M.O+T.O)** | **23.13** (Increase) |

### Key Findings

- COB-GS achieves an mIoU of 92.1%, outperforming all existing methods while maintaining the best visual quality (highest on all three CLIP-IQA metrics).
- Jointly optimizing semantics and texture does not degrade visual quality; instead, it slightly improves PSNR (from 23.06 to 23.13) due to the precise boundary structure.
- Performing only Gaussian splitting decreases texture quality (PSNR drops from 23.06 to 22.61), but incorporating texture restoration recovers and exceeds the original quality.
- $\delta=0.5$ is the optimal choice for the ambiguity threshold, as a larger value introduces too many unnecessary splits.

## Highlights & Insights

- **First 3DGS segmentation method to jointly optimize semantics and textures**: Proves that accurate boundaries can conversely improve visual quality, achieving a win-win scenario.
- **Gradient direction statistics as an ambiguous Gaussian detector**: The theoretical derivation is concise and powerful. The $mask\_sig$ variable is obtained directly from backpropagation, introducing almost zero extra computational overhead.
- Distinguishes between **ambiguity caused by Gaussian volume** and **ambiguity caused by inaccurate masks** as two different sources, handling them separately for higher precision.
- The strategy of decomposing multi-object segmentation into sequential single-object processing is simple and effective, resolving the fixed granularity issue of feature-based methods.
- The two-stage SAM2 mask generation resolves the object continuity issue in long sequences.

## Limitations & Future Work

- Floaters (floating artifacts) in 3DGS reconstruction are amplified after segmentation.
- The sequential single-object optimization strategy has low efficiency when dealing with a large number of objects.
- The splitting strategy may be less effective for highly overlapping objects.
- The threshold $\delta$ requires tuning and may require different values for different scenes.
- Relies heavily on the mask quality of SAM2 / Grounding-DINO.

## Related Work & Insights

- SA3D pioneered mask-based 3D segmentation but did not address the blurred boundary problem.
- FlashSplat reduces boundary blur using linear programming and background bias but sacrifices object structural integrity.
- SAGD uses Gaussian decomposition but does not jointly optimize texture.
- The "gradient statistics $\rightarrow$ splitting" paradigm of this method can be extended to other 3DGS tasks requiring precise boundaries (e.g., editing, generation).
- The joint optimization ideology could inspire segmentation tasks in other modalities.

## Rating

- Novelty: ⭐⭐⭐⭐ Splitting driven by semantic gradient statistics and joint optimization are unique contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative, qualitative, and ablation studies, validated on multiple scenes.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis and clear method derivation.
- Value: ⭐⭐⭐⭐ Holding practical significance for improving boundary quality in 3DGS-based segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [\[CVPR 2025\] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis](bfanet_revisiting_3d_semantic_segmentation_with_boundary_feature_analysis.md)
- [\[CVPR 2025\] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks](maskgaussian_adaptive_3d_gaussian_representation_from_probabilistic_masks.md)
- [\[CVPR 2026\] BEA-GS: BEyond RAdiance Supervision in 3DGS for Precise Object Extraction](../../CVPR2026/3d_vision/bea-gs_beyond_radiance_supervision_in_3dgs_for_precise_object_extraction.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)

</div>

<!-- RELATED:END -->
