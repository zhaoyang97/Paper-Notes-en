---
title: >-
  [Paper Note] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting
description: >-
  [CVPR2026][3D Vision][3D super-resolution] SR3R reformulates 3D super-resolution (3DSR) as a **feed-forward mapping** from sparse low-resolution views to high-resolution 3DGS, achieving high-fidelity HR 3DGS reconstruction via Gaussian offset learning and feature refinement, without per-scene optimization, while enabling strong zero-shot generalization.
tags:
  - CVPR2026
  - 3D Vision
  - 3D super-resolution
  - 3D Gaussian splatting
  - feed-forward reconstruction
  - Gaussian offset learning
  - sparse-view reconstruction
date: 2026-05-08
content_hash: d1b3575bfe0f81cc
---

# SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting

**Conference**: CVPR2026  
**arXiv**: [2602.24020](https://arxiv.org/abs/2602.24020)  
**Code**: [Project Page](https://xiangfeng66.github.io/SR3R/)  
**Area**: 3D Vision  
**Keywords**: 3D super-resolution, 3D Gaussian splatting, feed-forward reconstruction, Gaussian offset learning, sparse-view reconstruction

## TL;DR

SR3R reformulates 3D super-resolution (3DSR) as a **feed-forward mapping** from sparse low-resolution views to high-resolution 3DGS, achieving high-fidelity HR 3DGS reconstruction via Gaussian offset learning and feature refinement, without per-scene optimization, while enabling strong zero-shot generalization.

## Background & Motivation

- **Limitations of Prior Work**: Existing 3DSR methods rely on dense LR inputs and pretrained 2D super-resolution models to generate pseudo-HR images, which are then used as supervision for per-scene HR 3DGS optimization. This paradigm has three fundamental limitations:
  1. **Constrained high-frequency priors**: High-frequency knowledge is solely derived from 2DSR model priors, failing to capture 3D-specific high-frequency geometry and texture structures.
  2. **Reconstruction fidelity ceiling**: The quality of pseudo-HR labels inherently limits the upper bound of reconstruction quality.
  3. **High computational cost**: Dense multi-view synthesis combined with per-scene iterative optimization prevents cross-scene generalization.
- **Key Observation**: Feed-forward 3DGS reconstruction models can directly predict 3DGS from sparse views, but reconstruction quality is severely constrained by input resolution. This motivates the question: can 3DSR also be formulated as a feed-forward mapping that learns 3D-specific high-frequency priors from large-scale multi-scene data?
- **Paradigm Shift**: From "per-scene HR 3DGS self-optimization" to "generalizable HR 3DGS feed-forward prediction," fundamentally changing the way 3DSR acquires high-frequency knowledge.

## Method

### Overall Architecture

SR3R adopts a **plug-and-play** design with a four-stage pipeline:

1. **LR 3DGS Reconstruction**: Any feed-forward 3DGS backbone (e.g., NoPoSplat/DepthSplat) is used to obtain LR 3DGS $\mathcal{G}^{\text{LR}}$ from 2 LR views.
2. **Gaussian Densification**: $\mathcal{G}^{\text{LR}}$ is densified into $\mathcal{G}^{\text{Dense}}$ via Gaussian Shuffle Split, serving as a structural scaffold.
3. **Mapping Network**: Upsampled LR images are processed through a ViT encoder, feature refinement module, and ViT decoder to extract multi-view fused features.
4. **Gaussian Offset Learning**: Residual offsets from $\mathcal{G}^{\text{Dense}}$ to $\mathcal{G}^{\text{HR}}$ are predicted to obtain the final HR 3DGS.

The core formulation — feed-forward mapping definition:

$$f_{\boldsymbol{\theta}}: \{(\boldsymbol{I}^{v}_{lr}, \boldsymbol{K}^{v})\}_{v=1}^{V} \mapsto \mathcal{G}^{\text{HR}}$$

where each 3D Gaussian primitive is parameterized as $(\boldsymbol{\mu}, \alpha, \boldsymbol{r}, \boldsymbol{s}, \boldsymbol{c})$, corresponding to center position, opacity, quaternion rotation, scale, and spherical harmonic coefficients, respectively.

### Key Design 1: Gaussian Shuffle Split Densification

For each Gaussian primitive in $\mathcal{G}^{\text{LR}}$, six child Gaussians are generated along the positive and negative directions of its three principal axes, providing a finer structural scaffold:

$$\boldsymbol{\mu}_{j,k} = \boldsymbol{\mu}_j + \beta \, R_j \, \boldsymbol{e}_k \odot \boldsymbol{s}_j, \quad k=1,\dots,6$$

- $R_j$ is the rotation matrix corresponding to quaternion $\boldsymbol{r}_j$; $\boldsymbol{e}_k$ denotes unit vectors along positive/negative principal axes.
- $\beta = 0.5$ controls the offset magnitude; the child Gaussian scale along the offset axis is reduced to $\frac{1}{4}$ of the original.
- **Applied only to Gaussians with opacity > 0.5**, focusing on structurally significant regions.
- After densification, $\mathcal{G}^{\text{Dense}}$ contains $N = 6M$ primitives ($M$ being the number of LR Gaussians).

### Key Design 2: Feature Refinement Module

Upsampled LR images contain blurry and hallucinated high-frequency patterns introduced by interpolation, which can cause 3D geometry and texture artifacts if used directly. The feature refinement module aligns ViT-encoded features with geometry-aware features from the pretrained 3DGS backbone via **bidirectional cross-attention**:

$$\mathbf{U}_{o \leftarrow p} = \text{softmax}\!\left(\frac{(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_Q)(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_K)^\top}{\sqrt{d}}\right)(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_V)$$

$$\mathbf{U}_{p \leftarrow o} = \text{softmax}\!\left(\frac{(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_Q)(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_K)^\top}{\sqrt{d}}\right)(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_V)$$

The outputs of both attention directions are concatenated and fused through a fully connected layer to produce the refined feature $\boldsymbol{t}_{ca}$. The core mechanism is to **transfer reliable 3D geometric priors from the 3DGS backbone into the 2D feature space**, suppressing ambiguities introduced by upsampling.

### Key Design 3: Gaussian Offset Learning

This is the most critical module for SR3R's performance gains. The core idea is to **predict residual offsets from $\mathcal{G}^{\text{Dense}}$ to $\mathcal{G}^{\text{HR}}$ rather than directly regressing absolute Gaussian parameters**.

Procedure:
1. Each dense Gaussian center $\boldsymbol{\mu}_i$ is projected onto the image plane to obtain 2D coordinates $\boldsymbol{p}_i$.
2. Local features $\boldsymbol{F}_i$ are queried at position $\boldsymbol{p}_i$ from the ViT decoder feature map $\boldsymbol{t}_{de}$.
3. Gaussian centers, queried features, and camera intrinsics are aggregated and fed into PointTransformerV3 for spatial reasoning:

$$\boldsymbol{F} = \Phi_{\text{PTv3}}\!\left([\boldsymbol{\mu}_i;\, \{\boldsymbol{F}_i\}_{i=1}^{N};\, \boldsymbol{K}]\right)$$

4. A Gaussian Head (lightweight MLP) predicts residual offsets:

$$\Delta G = (\Delta\boldsymbol{\mu},\, \Delta\boldsymbol{\alpha},\, \Delta\boldsymbol{r},\, \Delta\boldsymbol{s},\, \Delta\boldsymbol{c}) = \Psi_{\text{GH}}(\boldsymbol{F})$$

5. Residual combination yields the final HR 3DGS:

$$\mathcal{G}^{\text{HR}} = \mathcal{G}^{\text{Dense}} + \Delta\mathcal{G}$$

**Design Motivation**: Since $\mathcal{G}^{\text{Dense}}$ already provides a reliable coarse structural scaffold, the remaining discrepancy primarily consists of local high-frequency signals. Learning offsets rather than absolute parameters constrains the search space to local neighborhoods, substantially improving training stability and reconstruction sharpness.

### Key Design 4: ViT Decoder Cross-View Fusion

The refined features $\boldsymbol{t}_{ca}$ are processed by the ViT decoder via:
- **Intra-view self-attention**: Aggregates global contextual information.
- **Inter-view cross-attention**: Fuses complementary information across views, mitigating inconsistencies caused by inaccurate poses or insufficient view overlap.

### Loss & Training

A combination of pixel-level MSE reconstruction loss and perceptual consistency LPIPS loss is adopted:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.05 \cdot \mathcal{L}_{\text{LPIPS}}$$

End-to-end training is performed via differentiable Gaussian rasterization.

## Key Experimental Results

### Experimental Setup

- **Datasets**: RealEstate10K (RE10K, indoor), ACID (outdoor aerial), DTU (object-centric), ScanNet++ (indoor)
- **Super-resolution scale**: 4× (64×64 → 256×256)
- **Backbones**: NoPoSplat, DepthSplat
- **Training**: 75K iterations, batch=8, lr=2.5e-5, 4×RTX 5090

### Main Results

| Method | Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | # Gaussians |
|--------|---------|--------|--------|---------|-------------|
| NoPoSplat | RE10K | 21.33 | 0.612 | 0.307 | 2.7M |
| Up-NoPoSplat | RE10K | 23.37 | 0.771 | 0.251 | 44.5M |
| **SR3R (NoPoSplat)** | **RE10K** | **24.79** | **0.827** | **0.188** | **16.5M** |
| DepthSplat | RE10K | 23.15 | 0.699 | 0.281 | 2.3M |
| Up-DepthSplat | RE10K | 24.71 | 0.793 | 0.244 | 38.3M |
| **SR3R (DepthSplat)** | **RE10K** | **26.25** | **0.856** | **0.165** | **14.2M** |
| NoPoSplat | ACID | 21.45 | 0.606 | 0.531 | 2.7M |
| Up-NoPoSplat | ACID | 23.91 | 0.692 | 0.384 | 44.5M |
| **SR3R (NoPoSplat)** | **ACID** | **25.54** | **0.746** | **0.283** | **16.5M** |
| DepthSplat | ACID | 23.80 | 0.624 | 0.437 | 2.3M |
| Up-DepthSplat | ACID | 25.32 | 0.721 | 0.322 | 38.3M |
| **SR3R (DepthSplat)** | **ACID** | **27.02** | **0.797** | **0.261** | **14.2M** |

**Key Findings**: SR3R achieves an average PSNR improvement of 1.4–3.5 dB while requiring only 37%–63% of the Gaussian parameters compared to direct upsampling (16.5M vs. 44.5M).

### Zero-Shot Generalization (RE10K → DTU)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Reconstruction Time |
|--------|--------|--------|---------|---------------------|
| SRGS (per-scene opt.) | 12.42 | 0.327 | 0.598 | 300s |
| FSGS+SRGS (per-scene opt.) | 13.72 | 0.444 | 0.481 | 420s |
| NoPoSplat | 12.63 | 0.343 | 0.581 | 0.01s |
| Up-NoPoSplat | 16.64 | 0.598 | 0.369 | 0.16s |
| **SR3R (NoPoSplat)** | **17.24** | **0.607** | **0.291** | **1.69s** |

SR3R not only outperforms all feed-forward baselines but also **surpasses per-scene optimization methods SRGS/FSGS+SRGS** (PSNR +3.5 dB), while being 177–248× faster.

### Ablation Study

| Component | PSNR ↑ | SSIM ↑ | LPIPS ↓ | # Gaussians |
|-----------|--------|--------|---------|-------------|
| NoPoSplat (baseline) | 21.33 | 0.612 | 0.307 | 2.7M |
| + Upsampling | 23.37 | 0.771 | 0.251 | 44.5M |
| + Cross-attention | 23.50 | 0.784 | 0.237 | 44.5M |
| + Gaussian offset (w/o PTv3) | 24.45 | 0.808 | 0.211 | 16.5M |
| + PTv3 (**Full SR3R**) | **24.79** | **0.827** | **0.188** | **16.5M** |

**Key Findings**:
1. **Gaussian offset learning contributes the most**: +0.95 PSNR, while reducing Gaussian count from 44.5M to 16.5M.
2. Cross-attention feature refinement improves structural consistency (+0.13 PSNR, LPIPS −0.014).
3. PTv3 multi-scale spatial reasoning further enhances sharpness (+0.35 PSNR, LPIPS −0.023).
4. All components are complementary, progressively improving reconstruction quality.

### Upsampling Strategy Robustness

| Upsampling Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------------------|--------|--------|---------|
| Bilinear | 24.59 | 0.795 | 0.204 |
| Bicubic | 24.66 | 0.817 | 0.193 |
| SwinIR | 24.79 | 0.827 | 0.188 |
| HAT | 24.78 | 0.819 | 0.183 |

Even with simple bilinear interpolation, SR3R surpasses all feed-forward baselines, demonstrating that the framework does not depend on any specific upsampling design.

## Highlights & Insights

- 🔄 **Paradigm Shift**: SR3R transitions 3DSR from "per-scene optimization with 2DSR pseudo-supervision" to "large-scale cross-scene feed-forward prediction," fundamentally changing how high-frequency knowledge is acquired.
- 🔌 **Plug-and-Play**: Compatible with any feed-forward 3DGS backbone, offering an elegant and practical design.
- 📐 **Offset Learning > Direct Regression**: Learning residual offsets rather than absolute parameters improves reconstruction quality while reducing Gaussian count to 37%.
- 🎯 **Zero-Shot Generalization**: Surpasses per-scene optimization methods on unseen scenes, with inference speeds two orders of magnitude faster.
- ⚡ **Efficient and Practical**: Complete HR 3D reconstruction from only 2 LR input views.

## Limitations & Future Work

- Inference time (1.69s), while far faster than optimization-based methods (300+s), is still approximately 100× slower than base feed-forward models (0.01s), limiting real-time applicability.
- Only 4× super-resolution is validated; performance at higher scales (8×/16×) remains unexplored.
- The densification strategy (fixed 6 child Gaussians) is heuristic; adaptive densification may be more effective.
- Training requires 4×RTX 5090, imposing a high computational resource threshold.
- Validation is limited to indoor, outdoor, and object-centric scenes; generalization to large-scale outdoor environments (e.g., autonomous driving) remains untested.

## Rating

- Novelty: ⭐⭐⭐⭐ — The feed-forward mapping paradigm for 3DSR is novel, and the Gaussian offset learning design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 3 datasets, zero-shot generalization, ablation studies, and upsampling strategy analysis.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with thorough motivation and rigorous formulations.
- Value: ⭐⭐⭐⭐ — Introduces a new paradigm for 3DSR with strong practical utility and plug-and-play applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](../../AAAI2026/3d_vision/arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)

</div>

<!-- RELATED:END -->
