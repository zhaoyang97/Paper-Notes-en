---
title: >-
  [Paper Note] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] AnchorSplat proposes an anchor-aligned feed-forward 3DGS framework that leverages 3D geometric priors (sparse point clouds) as anchors to predict Gaussians directly in 3D spa…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "feed-forward reconstruction"
  - "anchor alignment"
  - "geometric priors"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 3c3cf6f671c92fb8
---

# AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors

**Conference**: CVPR 2026
**arXiv**: [2604.07053](https://arxiv.org/abs/2604.07053)
**Code**: Coming soon
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, feed-forward reconstruction, anchor alignment, geometric priors, novel view synthesis

## TL;DR
AnchorSplat proposes an anchor-aligned feed-forward 3DGS framework that leverages 3D geometric priors (sparse point clouds) as anchors to predict Gaussians directly in 3D space. Using approximately 20× fewer Gaussians and half the reconstruction time, it achieves state-of-the-art performance on ScanNet++ v2 (PSNR 21.48) with superior depth estimation accuracy.

## Background & Motivation
**Background**: Scene-level 3D reconstruction is a core problem in computer vision. Optimization-based methods (3DGS, NeRF) achieve high quality but require per-scene iterative optimization, resulting in long processing times. Feed-forward 3DGS methods enable cross-scene generalization via a single forward pass.

**Limitations of Prior Work**:
   - Existing feed-forward methods adopt pixel-aligned strategies: each 2D pixel is mapped to one 3D Gaussian, causing the Gaussian count $N = H \times W \times V$ to grow linearly with the number of views;
   - Pixel-aligned representations are tied to 2D grids, leading to redundancy in flat regions and insufficient coverage in complex regions;
   - These methods are sensitive to occlusion, low-texture regions, and motion parallax, with inconsistent cross-view sampling patterns;
   - Feature interactions in 2D space are limited, lacking direct interaction among neighboring 3D points, which produces floaters and fragmented surfaces.

**Key Challenge**: How to achieve geometrically consistent, high-fidelity 3D reconstruction under feed-forward efficiency constraints?

**Key Insight**: Starting from 3D anchors rather than 2D pixels — MVS-predicted depths and poses are used to construct sparse 3D anchors, upon which Gaussians are predicted.

**Core Idea**: Anchor-aligned Gaussian representation + iterative refinement via Gaussian Refiner = fewer Gaussians, higher quality, independent of input resolution and number of views.

## Method

### Overall Architecture
Input: $V$ multi-view images → MapAnything MVS module predicts depths and poses → back-projection to 3D + FPS downsampling to $N$ anchors → 2D CNN feature extraction → projection onto anchors → Transformer Gaussian Decoder predicts anchor-aligned Gaussians → Gaussian Refiner refinement → rendering.

### Key Designs
1. **Anchor Predictor**:

    - **Function**: Generates a sparse set of 3D anchors from multi-view images.
    - **Mechanism**:
        - Pretrained MapAnything predicts depth $D_i$, intrinsics $K_i$, and extrinsics $P_i$
        - Back-projection to 3D: $P_w = R_i(D_i(u,v) K_i^{-1}[u,v,1]^\top) + T_i$
        - FPS downsampling to $N \ll H \times W \times V$ anchors
    - **Design Motivation**: Pixel-aligned methods produce $V \times H \times W$ Gaussians (e.g., 5.5M in AnySplat), whereas anchor alignment requires only ~247K, a reduction of ~20×. The anchor count is determined by scene complexity rather than image resolution.

2. **Gaussian Decoder**:

    - **Function**: Predicts Gaussian attributes from anchor features.
    - **Mechanism**:
        - 2D U-Net encodes image, depth, and camera rays: $F_i = E(I_i, D_i, \text{Ray}_i) \in \mathbb{R}^{h \times w \times C}$
        - Projects 2D features onto 3D anchors to obtain anchor features
        - 16-layer Transformer attention captures spatial interactions among anchors in 3D
        - MLP predicts 4 sets of Gaussian attributes $\{\delta\mu, \alpha, s, r, sh\}$ per anchor
        - Final Gaussian position: $\mu_j = A_j + \delta\mu_j$ (offset constrained within 10/128)
    - **Design Motivation**: Feature interactions in 3D space (rather than 2D) enable neighboring points to interact directly, reducing floaters and geometric inconsistencies.

3. **Gaussian Refiner**:

    - **Function**: Refines Gaussian attributes via rendering error feedback, analogous to a single-step differentiable rendering "optimization."
    - **Mechanism**:
        - Pretrained ResNet-18 extracts multi-scale features from rendered and ground-truth images
        - Per-view error is computed: $e_i = F_i - \hat{F}_i$
        - Differentiable back-projection maps 2D errors to 3D Gaussian positions
        - Transformer + Point Transformer combines current attributes, anchor features, and error features to predict attribute residuals
        - $\hat{\mathcal{G}}_j = \mathcal{G}_j + \delta\mathcal{G}_j$
    - **Design Motivation**: Feed-forward models with limited anchors may produce blurry or incomplete regions. The Refiner serves as a plug-and-play module that improves quality without retraining the full model.

### Loss & Training
- **Two-stage training**:
    - Stage 1: Train Gaussian Decoder (84M parameters), 5K steps
    - Stage 2: Freeze Decoder, train Gaussian Refiner (31M parameters), 5K steps
- Decoder loss: $L = \lambda_I \ell_I + \lambda_D \ell_D + \lambda_\alpha \ell_\alpha + \lambda_s \ell_s$
    - Rendering loss: $\ell_I = \ell_1 + 0.2(1- \text{SSIM}) + 0.2 \text{LPIPS}$
    - Depth loss, opacity regularization, volume regularization
- Refiner uses rendering loss $\ell_I$ only

## Key Experimental Results

### Main Results (ScanNet++ v2, 32 input views, 4 novel views)

| Method | Type | PSNR↑ | SSIM↑ | δ₁↑ | AbsRel↓ | # Gaussians | Recon. Time |
|--------|------|-------|-------|-----|---------|-------------|-------------|
| 3DGS | Optimization | 19.98 | 0.72 | 0.31 | 0.42 | 496K | 391s |
| AnySplat | Feed-forward | 20.20 | 0.73 | 0.71 | 0.16 | **5.55M** | 6.83s |
| AnchorSplat⋆ | Feed-forward | 20.96 | 0.78 | **0.94** | 0.068 | 247K | **3.11s** |
| **AnchorSplat** | Feed-forward | **21.48** | **0.79** | **0.94** | **0.066** | 247K | 5.52s |

### Ablation Study (varying number of input views)

| Setting | Method | PSNR↑ | # Gaussians | Recon. Time |
|---------|--------|-------|-------------|-------------|
| 3 views | AnySplat | 19.51 | 544K | 1.34s |
| 3 views | AnchorSplat⋆ | **19.99** | **247K** | 3.18s |
| 128 views | AnySplat | 20.47 | 21.6M | 14.2s |
| 128 views | AnchorSplat⋆ | **21.23** | **247K** | 3.94s |

### Key Findings
- The Gaussian count remains constant at 247K regardless of the number of input views (AnySplat scales from 544K to 21.6M)
- Depth accuracy substantially surpasses AnySplat (δ₁: 0.94 vs. 0.71), indicating stronger 3D geometric awareness from anchor alignment
- The Refiner significantly improves boundary sharpness and color consistency
- Performance remains stable under both extremely sparse (3 views) and extremely dense (256 views) settings

## Highlights & Insights
- **Anchor alignment is the key innovation**: it fundamentally decouples the Gaussian representation from 2D pixels, with count determined by scene complexity
- The plug-and-play design of the Gaussian Refiner is elegant and can independently enhance any feed-forward 3DGS method
- The substantial improvement in depth estimation (0.94 vs. 0.71) demonstrates that feature interaction in 3D space is critical for geometric understanding

## Limitations & Future Work
- Relies on the quality of MapAnything's depth and pose predictions; inaccurate MVS estimates degrade anchor quality
- Spherical harmonics are limited to degree 0 (no view-dependent color), restricting expressiveness for reflective materials
- SH degree 0 may be insufficient for scenes with complex illumination

## Related Work & Insights
- Compared to the voxel-aligned approach in AnySplat, anchor alignment more directly exploits 3D geometric priors
- The rendering error feedback in the Refiner is conceptually similar to G3R but is more lightweight

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The anchor-aligned paradigm represents a significant advance in feed-forward 3DGS
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on ScanNet++ with multi-view ablations
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and comparisons are fair
- Value: ⭐⭐⭐⭐⭐ 20× efficiency gain combined with quality improvement yields strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
