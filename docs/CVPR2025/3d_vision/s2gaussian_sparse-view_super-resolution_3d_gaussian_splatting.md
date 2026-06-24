---
title: >-
  [Paper Note] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] A two-stage framework, S2Gaussian, is proposed to address the joint sparse and low-resolution view scene reconstruction for the first time. The first stage optimizes low-resolution Gaussians with depth regularization and initializes high-resolution Gaussians via Gaussian Shuffle Split. The second stage refines the high-resolution scene using blur-free inconsistency modeling and a 3D robust optimization strategy.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Sparse-View"
  - "Super-Resolution"
  - "Novel View Synthesis"
  - "Depth Regularization"
date: 2026-05-08
content_hash: c213a2e487f84f99
---

# S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.04314](https://arxiv.org/abs/2503.04314)  
**Code**: [Project Page](https://jeasco.github.io/S2Gaussian/)  
**Area**: 3D Vision/3D Reconstruction  
**Keywords**: 3D Gaussian Splatting, Sparse-View, Super-Resolution, Novel View Synthesis, Depth Regularization

## TL;DR

A two-stage framework, S2Gaussian, is proposed to address the joint sparse and low-resolution view scene reconstruction for the first time. The first stage optimizes low-resolution Gaussians with depth regularization and initializes high-resolution Gaussians via Gaussian Shuffle Split. The second stage refines the high-resolution scene using blur-free inconsistency modeling and a 3D robust optimization strategy.

## Background & Motivation

3D Gaussian Splatting (3DGS) has achieved significant progress in novel view synthesis, but it heavily relies on dense, high-resolution input images. In practical applications (such as robotics and internet images), available views often suffer from **both** sparsity and low-resolution issues simultaneously.

Limitations of Prior Work:
- Sparse-view methods (such as FSGS, DNGaussian) assume high-resolution inputs.
- Super-resolution methods (such as SRGS, GaussianSR) assume dense view supervision.
- The two categories of methods exhibit an inherent incompatibility: super-resolution requires dense supervision to recover details, whereas sparse-view regularization tends to oversmooth the scene.
- Simply combining the two types of methods yields suboptimal results.

The key Challenge is how to reconstruct geometrically accurate and detail-rich 3D scenes under the dual constraints of sparse views and low resolution.

## Method

### Overall Architecture

A two-stage design: (1) HR GS Initialization Stage: optimizes low-resolution Gaussian representations with depth regularization first, and then initializes high-resolution Gaussians through Gaussian Shuffle Split densification; (2) HR GS Optimization Stage: enhances original and pseudo-views with a 2D super-resolution model, and refines them using blur-free inconsistency modeling combined with a 3D robust optimization strategy.

### Key Designs

### Key Design 1: Gaussian Shuffle Split

- **Function**: Densifies low-resolution coarse Gaussians into compact Gaussians suitable for representing high-resolution details.
- **Mechanism**: Replaces each original Gaussian with 6 sub-Gaussians offset along the positive and negative directions of the three principal axes. The offset distance is $\alpha \cdot s_i$ (default $\alpha=0.5$), the scale along the offset axis is reduced to 1/4 of the original, and the other axes are scaled by $1/\lambda$ ($\lambda=1.9$). Splitting is only executed for Gaussians with opacity >0.5, and the opacity is reset after splitting to allow automatic pruning.
- **Design Motivation**: Coarse Gaussians optimized at low resolution are sparse and rough, failing to simulate high-resolution details. Standard adaptive density control fails in the absence of dense or high-quality supervision. Shuffle Split provides a training-free local densification strategy.

### Key Design 2: Blur-Free Inconsistency Modeling

- **Function**: Mitigates multi-view inconsistency caused by 2D super-resolution models while preserving details.
- **Mechanism**: Connects a learnable Inconsistency Modeling (IM) module (two residual blocks) after the pre-trained SR model to simulate inter-view inconsistency: $I_{SR}^{IM} = I_{SR} + IM(I_{SR})$. Since IM tends to discard details to maintain consistency, a Blur Proposal (BP) module is additionally introduced to predict pixel-wise blur kernels $\mathcal{B}_k$. The loss is computed between the blurred rendered image $R_{HR}^{blur} = R_{HR} * \mathcal{B}_k$ and $I_{SR}^{IM}$, preventing the smoothing caused by directly constraining $R_{HR}$.
- **Design Motivation**: 2D SR models cannot guarantee multi-view consistency. Directly supervising with SR results forces the Gaussians to represent inconsistency, leading to blur. The combination of IM and BP allows inconsistency to be explicitly modeled rather than being absorbed by the Gaussians.

### Key Design 3: 3D Robust Optimization Strategy

- **Function**: Mitigates erroneous gradient updates caused by poorly optimized regions in pseudo-views.
- **Mechanism**: Observes that high-quality views yield stable and consistent gradients, while artifact-heavy views lead to chaotic and fluctuating gradients. Smoothness/clipping constraints are applied to the gradients of Gaussian parameters (e.g., scaling $s$) to suppress the influence of abnormal gradients.
- **Design Motivation**: Certain regions in pseudo-views (e.g., areas not covered by LR Gaussians) generate artifacts. Direct optimization with these views causes the 3D scene to converge to incorrect structures.

### Loss & Training

$\mathcal{L}_{SR} = (1-\beta)\mathcal{L}_1(R_{HR}^{blur}, I_{SR}^{IM}) + \beta \mathcal{L}_{D-SSIM}(R_{HR}^{blur}, I_{SR}^{IM})$, where $\beta=0.2$. The first stage utilizes a Pearson correlation depth loss for depth regularization.

## Key Experimental Results

### Main Results: Blender ×4 (8 views)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3DGS | Low | Low | High |
| FSGS + SR | Second-best | Second-best | Second-best |
| **S2Gaussian** | **Best** | **Best** | **Best** |

### LLFF ×4 (3 views) - Extremely Sparse Scenes

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Sparse-view methods | Better geometry | Better | Medium |
| SR methods | Medium | Medium | Medium |
| **S2Gaussian** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | PSNR | SSIM |
|------|------|------|
| W/o Shuffle Split | Baseline | Baseline |
| + Shuffle Split | +Significant Gain | +Gain |
| + IM (Inconsistency Modeling) | +Further Gain | +Gain |
| + BP (Blur Proposal) | +Detail Improvement | +Gain |
| + 3D Robust Optimization | **Best** | **Best** |

### Key Findings

- S2Gaussian achieves SOTA performance across Blender, LLFF, and Mip-NeRF360 datasets.
- The advantage is most pronounced in extremely sparse scenes (LLFF 3 views with ×4 super-resolution).
- Gaussian Shuffle Split effectively densifies the representation without damaging the original 3D geometry.
- The combination of IM and BP yields better results than using either module in isolation—IM addresses inconsistency while BP preserves details.

## Highlights & Insights

1. **First Joint Solution for Sparse and Low-Resolution Views**: Unifies two previously independently studied problems into a single framework, aligning better with practical application scenarios.
2. **Elegant Shuffle Split Design**: Training-free, local densification, without damaging the original representation.
3. **Complementary Design of IM+BP**: Inconsistency modeling absorbs view discrepancies, while blur proposal compensates for detail loss.

## Limitations & Future Work

- Depends on the quality of the pre-trained 2D SR model; different SR models will affect the final performance.
- The two-stage pipeline increases training time.
- Performance on extremely sparse scenes (e.g., 2 views) remains to be verified.
- 3D-aware super-resolution methods have not been explored.

## Related Work & Insights

- **FSGS, DNGaussian**: Sparse-view 3DGS methods utilizing depth priors.
- **SRGS, GaussianSR**: 3D super-resolution methods.
- **SuperGaussian**: 3D super-resolution using video upsampling models.
- The two-stage paradigm (coarse-to-fine) of S2Gaussian can be extended to 3D reconstruction under other degradation conditions.

## Rating

⭐⭐⭐⭐ — The problem definition is highly practical, and the two-stage design is logical. Shuffle Split and IM+BP each have unique strengths. SOTA performance on multiple benchmarks validates the effectiveness of the proposed method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting](../../CVPR2026/3d_vision/splatsure_selective_super-resolution_for_multi-view_consistent_3d_gaussian_splat.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)

</div>

<!-- RELATED:END -->
