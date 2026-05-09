---
title: >-
  [Paper Note] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes Self-Constrained Priors (SCP), which construct a TSDF distance field by fusing depth maps rendered from the current 3D Gaussians. This field serves as a prior to impose geometry-aware constraints on Gaussians (outlier removal, opacity constraint, and surface attraction), enabling high-fidelity surface reconstruction that achieves state-of-the-art performance on NeRF-Synthetic and DTU benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Surface Reconstruction
  - TSDF
  - Self-Constrained Prior
  - Geometric Constraint
date: 2026-05-08
content_hash: a6c4fc2e7141d2e3
---

# 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.19682](https://arxiv.org/abs/2603.19682)
**Code**: [https://github.com/takeshie/GSPrior](https://github.com/takeshie/GSPrior)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Surface Reconstruction, TSDF, Self-Constrained Prior, Geometric Constraint

## TL;DR
This paper proposes Self-Constrained Priors (SCP), which construct a TSDF distance field by fusing depth maps rendered from the current 3D Gaussians. This field serves as a prior to impose geometry-aware constraints on Gaussians (outlier removal, opacity constraint, and surface attraction), enabling high-fidelity surface reconstruction that achieves state-of-the-art performance on NeRF-Synthetic and DTU benchmarks.

## Background & Motivation

**Background**: 3DGS surpasses NeRF in both rendering speed and visual fidelity for novel view synthesis, yet its geometric reconstruction accuracy remains notably lacking. Existing methods either apply multi-view consistency constraints on depth, use implicit fields to regularize Gaussian motion, or rely on data-driven pretrained priors.

**Limitations of Prior Work**: These strategies either cannot directly impose constraints on 3D Gaussians or cannot operate in a geometry-aware, adaptive manner. Methods relying on external priors generalize poorly to complex or unseen scenes, leading to artifacts and degraded geometric quality.

**Key Challenge**: High rendering quality demands high-degree-of-freedom Gaussian representations, whereas geometric accuracy requires strict geometric constraints. The core challenge is how to balance these two objectives without relying on external priors.

**Key Insight**: The paper observes that depth maps rendered from the current Gaussians inherently encode surface estimation information, which can be fused into a TSDF distance field as a "self-constrained prior," eliminating the need for external data-driven priors.

**Core Idea**: A TSDF distance field is automatically generated from rendered depth maps to define a narrow band near the surface. Three geometric constraints are applied to Gaussians within this band. The prior is periodically updated and the band progressively narrowed to achieve coarse-to-fine optimization.

## Method

### Overall Architecture
Given multi-view images $\{v_i\}_{i=1}^I$, the method learns a set of 3D Gaussians $\{g_j\}_{j=1}^J$. The core loop proceeds as follows: render depth maps from current Gaussians → fuse into a TSDF grid $f^t$ → apply three geometric constraints using $f^t$ as prior → periodically update $f^t$ while progressively narrowing the constraint bandwidth.

### Key Designs

1. **Construction of Self-Constrained Prior (TSDF Distance Field)**:

    - Function: Fuses rendered depth maps into a distance field $f^t = \mathcal{F}(\{d_i'(t)\})$, with the zero-level set serving as a coarse surface estimate.
    - Mechanism: A narrow band is defined around the zero-level set (distance range $[-1, 1]$), with the truncation threshold $\sigma^t$ controlling band width. The prior is **periodically updated** (more recent depth maps are more accurate and multi-view consistent), and the bandwidth is progressively narrowed to strengthen constraints.
    - Design Motivation: Eliminates reliance on external priors by bootstrapping geometric accuracy from the model's own rendered outputs.

2. **Gaussian Outlier Removal**:

    - Function: Removes outlier Gaussians located outside the narrow band.
    - Mechanism: The signed distance $s_j = f^t(\mu_j)$ is interpolated at Gaussian center $\mu_j$; if $|s_j| = 1$ (at the band boundary), the Gaussian is removed.
    - Design Motivation: Reduces the negative impact of outlier Gaussians on depth rendering, yielding a more compact Gaussian distribution.

3. **Opacity Constraint (Core Innovation)**:

    - Function: Encourages Gaussians on the surface to have maximum opacity and those near but off the surface to have minimum opacity.
    - Mechanism: Gaussians within the narrow band are partitioned into an on-surface subset $\mathcal{N}_{on} = \{g_j : |s_j| \leq \delta^t\}$ and an off-surface subset $\mathcal{N}_{off} = \{g_j : \delta^t < |s_j| \leq 1\}$. The loss is defined as $L_{SCP} = \frac{1}{M}(\sum_{g_k \in \mathcal{N}_{on}} \varepsilon_k (o_k - 1)^2 + \sum_{g_{k'} \in \mathcal{N}_{off}} \varepsilon_{k'} o_{k'}^2)$, with distance-based weighting $\varepsilon_j = 1/(1+|s_j|)^2$.
    - Design Motivation: Produces a sharp surface boundary. Unlike GOF/GSDF, which directly learn SDF-to-opacity mappings, this constraint is geometry-aware and adaptive.

4. **Surface Attraction (Position Constraint)**:

    - Function: Projects Gaussians toward the surface using distance field gradients.
    - Mechanism: Gradient $\nabla f^t(\mu_j)$ is computed via finite differences, and positions are updated as $\mu_j \leftarrow \mu_j - s_j \cdot \nabla f^t(\mu_j)$.
    - Design Motivation: Operates as a positional update after densification rather than as part of iterative optimization, improving stability.

### Loss & Training

The full objective is: $L = L_{RGB} + \lambda_1 L_{Depth} + \lambda_2 L_{NS} + \lambda_3 L_{NM} + \lambda_4 L_{SCP}$

- $L_{RGB}$: Rendering loss combining MAE, SSIM, and NCC.
- $L_{Depth}$: Depth distribution consistency constraint along rays.
- $L_{NS}$: Consistency between rendered normals and depth-derived normals.
- $L_{NM}$: Cross-view geometric consistency via homography matrices.
- $L_{SCP}$: Opacity constraint from the self-constrained prior.

Hyperparameters: $\lambda_1=0.01, \lambda_2=0.1, \lambda_3=0.1, \lambda_4=0.01$.

Constraint schedule: $L_{SCP}$ is applied at every iteration; surface attraction is applied after densification, followed by outlier removal.

## Key Experimental Results

### Main Results (NeRF-Synthetic)

| Method | Category | CD$_{L1}$(×100)↓ | PSNR↑ |
|--------|----------|-------------------|-------|
| NeuS | Implicit | 2.33 | 30.20 |
| NeRO | Implicit | 1.92 | 27.48 |
| VolSDF | Implicit | 2.86 | 27.96 |
| 2DGS | Explicit | 2.26 | 33.07 |
| GS-UDF | Explicit | 2.25 | 33.37 |
| GS-Pull | Explicit | 2.31 | 33.29 |
| PGSR | Explicit | 2.18 | 34.05 |
| QGS | Explicit | 2.04 | 30.41 |
| **Ours** | Explicit | **1.87** | **34.21** |

Mean Chamfer Distance on DTU: **0.50** (vs. PGSR: 0.53, QGS: 0.54, 2DGS: 0.80). Training time: 42 minutes.

### Ablation Study (DTU Mean CD)

| Configuration | Mean CD |
|---------------|---------|
| Full model | **0.50** |
| w/o self-constrained prior $f^t$ | Degraded |
| w/o $L_{SCP}$ constraint | Accuracy degraded due to missing opacity constraint |
| w/o periodic update | Fixed prior fails to adapt to optimization |
| w/o outlier removal | Outlier Gaussians degrade rendering |
| w/o surface attraction | Gaussians fail to concentrate on the surface |

Visualizations confirm that: removing outlier Gaussians significantly reduces floaters; surface attraction further concentrates Gaussians on the surface; the full model achieves the best overall result.

### Key Findings
- The proposed method outperforms all implicit and explicit baselines on Chamfer distance while also achieving the highest PSNR (34.21), excelling in both geometric accuracy and rendering quality.
- On TNT, F1 = 0.51 (highest among explicit methods); competitive rendering metrics are also observed on Mip-NeRF360.
- Training time of 42 minutes offers a clear efficiency advantage over implicit methods (>12 hours).

## Highlights & Insights
- **Self-Constrained Paradigm**: This work is the first to use TSDF derived from rendered outputs as a self-supervisory constraint signal, eliminating dependence on external priors. This "self-supervised geometric constraint" paradigm is transferable to other explicit or implicit representations.
- **Distance-Weighted Asymmetric Constraint**: The weighting function $\varepsilon_j = 1/(1+|s_j|)^2$ automatically focuses constraints on Gaussians near the surface — encouraging high opacity on the surface and low opacity in its vicinity — which is more effective than symmetric constraints.
- **Progressive Refinement Strategy**: Periodic prior updates combined with progressively narrowing bandwidths resemble curriculum learning, preventing overly strict early constraints from disrupting optimization.

## Limitations & Future Work
- Validation is primarily conducted on opaque scenes; performance on transparent or translucent materials remains unknown.
- Performance on TNT is slightly lower than on NeRF-Synthetic, suggesting potential limitations in TSDF fusion accuracy for large-scale scenes.
- Poor depth map quality during early training stages may mislead optimization; although periodic updates partially mitigate this, initialization strategies warrant further investigation.
- Future directions include multi-scale TSDF pyramid structures, joint learning with implicit representations, and extension to dynamic scenes.

## Related Work & Insights
- **vs. GS-Pull/GS-UDF**: These methods require additionally learning an implicit field (UDF) to constrain Gaussians, whereas the proposed approach directly generates priors from rendered depth without extra learning modules.
- **vs. PGSR**: PGSR applies multi-view depth-normal consistency constraints, while this work fuses multi-view depth into a unified TSDF prior, enabling more geometry-aware constraints.
- **vs. SuGaR/2DGS**: These methods constrain Gaussians to 2D planar primitives (surfels), whereas this work preserves full 3D Gaussian degrees of freedom and employs TSDF priors rather than shape constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The self-constrained prior concept is novel, although TSDF fusion itself is a mature technique.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four standard benchmarks, comprehensive ablations, and complete quantitative and qualitative results.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure with complete derivations.
- Value: ⭐⭐⭐⭐⭐ Addresses a core limitation of 3DGS (geometric accuracy) with strong transferability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)

<!-- RELATED:END -->
