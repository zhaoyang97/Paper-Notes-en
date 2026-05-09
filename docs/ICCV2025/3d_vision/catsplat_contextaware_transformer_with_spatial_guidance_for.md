---
title: >-
  [Paper Note] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image
description: >-
  [ICCV 2025][3D Vision][Single-view 3D reconstruction] This paper proposes CATSplat, a generalizable Transformer framework for feed-forward single-view 3DGS reconstruction. It enhances image features via dual cross-attention using VLM text embeddings (contextual prior) and 3D point cloud features (spatial prior), achieving comprehensive improvements over Flash3D in PSNR/SSIM/LPIPS on RE10K and other datasets, with strong cross-dataset generalization.
tags:
  - ICCV 2025
  - 3D Vision
  - Single-view 3D reconstruction
  - 3D Gaussian Splatting
  - vision-language model
  - text guidance
  - spatial guidance
  - point cloud features
date: 2026-05-08
content_hash: 1d13830f94da99f8
---

# CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image

**Conference**: ICCV 2025
**arXiv**: [2412.12906](https://arxiv.org/abs/2412.12906)
**Code**: [Project Page](https://kuai-lab.github.io/catsplat2025)
**Authors**: Wonseok Roh, Hwanhee Jung, Jong Wook Kim et al. (Korea University, Google, Purdue University)
**Area**: 3D Vision / Novel View Synthesis / 3DGS
**Keywords**: Single-view 3D reconstruction, 3D Gaussian Splatting, vision-language model, text guidance, spatial guidance, point cloud features

## TL;DR
This paper proposes CATSplat, a generalizable Transformer framework for feed-forward single-view 3DGS reconstruction. It enhances image features via dual cross-attention using VLM text embeddings (contextual prior) and 3D point cloud features (spatial prior), achieving comprehensive improvements over Flash3D in PSNR/SSIM/LPIPS on RE10K and other datasets, with strong cross-dataset generalization.

## Background & Motivation
- Generalizable feed-forward methods based on 3DGS (pixelSplat, MVSplat) have succeeded in multi-view settings by exploiting cross-view correspondence, but the **single-view** setting suffers from severe information scarcity.
- Flash3D pioneered single-view 3DGS feed-forward reconstruction (using a foundation depth estimation model), yet this setting remains underexplored.
- Multi-view methods can leverage physical techniques such as triangulation to obtain 3D cues, which are unavailable in the single-view case.
- **Core Insight**: Information from sources beyond visual cues must be incorporated—specifically, textual semantics and 3D geometric priors.

## Core Problem
How to compensate for the extreme information deficiency inherent to single-image input by introducing textual context and 3D spatial priors, so as to achieve high-quality, generalizable 3DGS reconstruction?

## Method

### Overall Architecture
1. Input single-view image $\mathcal{I} \in \mathbb{R}^{H \times W \times 3}$
2. A pretrained monocular depth estimation model (UniDepth) predicts depth map $D$
3. Image and depth map are concatenated → ResNet encoder → multi-scale image features $F_i^{\mathcal{I}}$
4. VLM (LLaVA) generates a single-sentence scene description → intermediate text embeddings $F^C$ are extracted
5. Depth map is back-projected into a 3D point cloud $P$ → PointNet encoder → 3D spatial features $F^S$
6. Multi-resolution Transformer (3 layers) sequentially applies:
   - Cross-attention: $F_i^{\mathcal{I}} \times F_i^C$ → contextual fusion
   - Cross-attention: result $\times F_i^S$ → spatial fusion
   - Self-attention: feature refinement
7. ResNet decoder → predicts per-pixel Gaussian parameters $\{\mu_j, \alpha_j, \Sigma_j, c_j\}$
8. Rasterization renders novel views

### Key Design 1: Contextual Prior via Text Guidance
- A pretrained VLM (LLaVA) generates a one-sentence scene description from the input image.
- The **intermediate text embeddings** $F^C \in \mathbb{R}^{N_c \times D^C}$ (rather than the final text output) are utilized, preserving rich multimodal semantic information.
- Text features are softly fused into image features via cross-attention: Q from image features, K/V from text features.
- The text embeddings encode scene type (e.g., kitchen), object identity (e.g., refrigerator, oven), spatial relationships, etc.—providing semantic bias for reasoning about occluded regions.
- **Prompt ablation**: a single-sentence description outperforms scene-type labels, object lists, and multi-sentence descriptions (which may introduce exaggerated information).

### Key Design 2: Spatial Prior via 3D Point Cloud
- The 2D depth map is back-projected into a 3D point cloud via camera parameters: $p = K^{-1} \cdot u \cdot d$
- A PointNet encoder extracts 3D features $F^S \in \mathbb{R}^{N_s \times D^S}$ from the point cloud.
- A second cross-attention fuses 3D features (Q from context-enhanced image features, K/V from 3D features).
- **Superiority over conventional 2D depth usage**: ablations demonstrate that cross-attention on 3D point features >> cross-attention on 2D depth features >> simple depth concatenation.

### Key Design 3: Ratio $\gamma$ for Fusion Strength Control
- A ratio $\gamma$ is introduced in the Add & Norm step to control the contribution of prior information:
  $$\tilde{F}_i = \text{Norm}(F_i^{\mathcal{I}} + \gamma \cdot \text{Dropout}(F_i^{\mathcal{I}CS}))$$
- This prevents the original visual features from being overwhelmed by the prior signals.

### Gaussian Parameter Prediction
- **Center $\mu$**: A depth offset $\delta$ is predicted to refine the estimated depth $\tilde{d} = d + \delta$; a 3D offset $\Delta_j$ is added after back-projection for fine alignment.
- **Opacity $\alpha$**: constrained to $[0,1]$ via sigmoid.
- **Covariance $\Sigma$**: rotation matrix $R$ and scaling matrix $S$ are predicted, $\Sigma = RSS^TR^T$.
- **Color $c$**: spherical harmonic coefficients.
- **Loss**: $\mathcal{L} = \lambda_{\ell1}\mathcal{L}_{\ell1} + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{lpips}\mathcal{L}_{lpips}$

## Key Experimental Results

### Main Results (RE10K, comparison with single-view methods)

| Method | n=5 PSNR | n=10 PSNR | Random PSNR |
|--------|----------|-----------|-------------|
| Flash3D | 28.46 | 25.94 | 24.93 |
| **CATSplat** | **29.09** | **26.44** | **25.45** |

### Interpolation / Extrapolation (RE10K)
- Interpolation: 25.23 dB (vs. Flash3D 23.87), narrowing the gap with two-view methods (pixelSplat 26.09).
- **Extrapolation: 25.35 dB, surpassing all two-view methods** (MVSplat 23.04); a single image outperforms two-image methods, validating the effectiveness of the priors.

### Cross-Dataset Generalization (trained on RE10K → zero-shot evaluation)

| Target Dataset | Flash3D PSNR | CATSplat PSNR |
|----------------|-------------|---------------|
| NYUv2 (indoor) | 25.09 | **25.57** |
| ACID (outdoor) | 24.28 | **24.73** |
| KITTI (driving) | 21.96 | **22.43** |

### Ablation Study

| Configuration | Random PSNR | Random LPIPS |
|---------------|------------|-------------|
| Baseline (no prior) | 25.02 | 0.159 |
| + Contextual prior | 25.40 | 0.153 |
| + Spatial prior | 25.42 | 0.153 |
| + Both | **25.45** | **0.151** |

### User Study
- RE10K: 88.42% of users prefer CATSplat (vs. Flash3D 11.58%)
- ACID: 91.41% prefer CATSplat

## Highlights & Insights
- **Intermediate VLM embeddings are more informative than final text outputs**: intermediate representations in the multimodal alignment space retain richer signals than raw textual descriptions.
- **3D point features >> 2D depth features**: back-projecting depth into a point cloud and encoding it via PointNet followed by cross-attention significantly outperforms simple 2D depth concatenation.
- **Extrapolation is a strength of single-view methods**: multi-view methods excel at interpolation but struggle with extrapolation (due to reliance on cross-view correspondence), whereas single-view methods augmented with priors actually surpass two-view methods on extrapolation.
- **Text prompt format matters**: a single-sentence description is optimal; descriptions that are too long or too short both degrade performance.
- **Iterative cross-attention yields cumulative gains**: applying cross-attention across all 3 layers outperforms applying it in only 1–2 layers.

## Limitations & Future Work
- Performance on occluded and truncated regions remains limited (as acknowledged by the authors).
- Training is conducted solely on RE10K; limited data diversity may restrict practical applicability, and extending to more datasets could improve generalization.
- VLM inference (LLaVA forward pass) introduces additional computational overhead, affecting real-time feasibility.
- Temporal consistency and video sequence settings are not explored.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The dual-prior combination of intermediate VLM embeddings and 3D point cloud features for single-view 3DGS is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-metric evaluation with detailed ablations and a user study; experiments are very thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with well-designed ablations.
- **Value**: ⭐⭐⭐⭐ — The idea of using VLM embeddings as 3D priors and the paradigm of fusing multimodal priors via cross-attention are worth adopting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)

</div>

<!-- RELATED:END -->
