---
title: >-
  [Paper Note] Learnable Infinite Taylor Gaussian for Dynamic View Rendering
description: >-
  [CVPR 2025][3D Vision][dynamic scene] This paper proposes the Learnable Infinite Taylor Formula to model the temporal evolution of Gaussian primitives' position, rotation, and scale in dynamic scenes. It employs a third-order Taylor expansion to capture large-scale motion, and utilizes an MLP alongside Linear Blend Skinning (LBS) to construct the Peano remainder for compensating high-order terms. This achieves motion modeling with zero approximation error…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "dynamic scene"
  - "3D Gaussian Splatting"
  - "Taylor series"
  - "Peano remainder"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 37ba6a2992de1fa7
---

# Learnable Infinite Taylor Gaussian for Dynamic View Rendering

**Conference**: CVPR 2025  
**arXiv**: [2412.04282](https://arxiv.org/abs/2412.04282)  
**Code**: [Project Page](https://ellisonking.github.io/TaylorGaussian)  
**Area**: 3D Vision  
**Keywords**: dynamic scene, 3D Gaussian Splatting, Taylor series, Peano remainder, novel view synthesis

## TL;DR

This paper proposes the Learnable Infinite Taylor Formula to model the temporal evolution of Gaussian primitives' position, rotation, and scale in dynamic scenes. It employs a third-order Taylor expansion to capture large-scale motion, and utilizes an MLP alongside Linear Blend Skinning (LBS) to construct the Peano remainder for compensating high-order terms. This achieves motion modeling with zero approximation error, outperforming state-of-the-art methods on the N3DV and Technicolor datasets.

## Background & Motivation

**Background**: Dynamic 3D Gaussian Splatting (3DGS) has achieved significant progress in dynamic scene reconstruction. However, accurately modeling the continuous temporal changes of Gaussian primitive attributes (position, rotation, scale) remains a core challenge. Under the constraints of limited photometric data, the massive number of time-varying parameters easily leads to convergence to suboptimal solutions.

**Limitations of Prior Work**:
1. **End-to-end implicit methods** (e.g., D3DGS): Directly predict individual Gaussian deformations using MLPs. Lacking explicit supervision, they struggle to generate high-quality deformation fields, resulting in weak spatiotemporal consistency.
2. **Time-conditioned polynomial methods**: Explicitly interpretable but require extensive manual designs, making them difficult to generalize across different scenes.
3. **4DGS**: Models spatiotemporal features using hexplane decomposition, but faces difficulties in multi-view scenarios.
4. **StreamRF**: Online training strategies are not adaptable to large viewpoint changes.

**Design Motivation**: Can the flexibility of implicit methods be combined with the interpretability of explicit polynomials? The Taylor formula is naturally suited for approximating complex functions with polynomials, and its Peano remainder can be learned via neural networks to construct a complete series.

## Method

### Overall Architecture

The framework consists of four core stages: Gaussian initialization $\rightarrow$ sparse point sampling $\rightarrow$ Gaussian point interpolation $\rightarrow$ Gaussian deformation field modeling. Deformation field modeling is divided into two parts:
- **Taylor expansion part** $f_k(t)$: A third-order explicit polynomial to model large-scale motion.
- **Peano remainder part** $\mathcal{H}_k(t)$: MLP encoding + LBS interpolation to model high-order residuals.

The complete transformation is formulated as $\mathcal{T}_i(t) = f_k(t) + \mathcal{H}_k(t)$, which is free of approximation errors.

### Key Designs

#### 1. Third-Order Taylor Expansion Deformation Field

Separate temporal Taylor expansions are established for positional motion, scale consistency, and rotational motion:

- **Position**: $\bm{p}_i(t) = \sum_{k=0}^{n} \frac{1}{k!} f_p^{(k)}(t_\tau)(t - t_\tau)^k$, where $f_p^{(k)}(t_\tau)$ is the learnable parameter (Taylor coefficient), and $t_\tau$ is the temporal anchor.
- **Scale**: $\bm{s}_i(t) = \sum_{k=0}^{m} \frac{1}{k!} f_s^{(k)}(t_\tau)(t - t_\tau)^k$
- **Rotation**: $\bm{q}_i(t) = \sum_{k=0}^{l} \frac{1}{k!} f_q^{(k)}(t_\tau)(t - t_\tau)^k$ (represented as quaternions)

A third-order expansion is sufficient to capture large-scale motion patterns such as acceleration and deceleration while maintaining computational efficiency.

#### 2. Deformation Field Modeling with Peano Remainder (GP-LP Architecture)

Gaussian points are classified into two categories:
- **Global Primitives (GP)**: A small number of representative skeleton points selected via farthest point sampling, whose time-varying offsets are directly predicted by an MLP: $\Delta_{GP} = MLP(GP)$.
- **Local Primitives (LP)**: A vast number of detail points whose remainders are interpolated from neighboring GPs via **Linear Blend Skinning (LBS)**.

Positional offset of LP: $\Delta\mu_i^t = \sum_{j \in \mathcal{N}} w_{ij}(R_j^t(\mu_i - p_j) + p_j + \Delta d_j^t)$

The weights $w_{ij}$ are calculated based on a Gaussian RBF kernel (with a learnable radius parameter $r_j$), ensuring spatial consistency (maintaining the spatial relationship between LP and GP) and temporal consistency (rigid constraints on neighboring point motion).

#### 3. Time-Dependent Opacity Modeling

Opacity is modeled as an RBF decay around the temporal anchor $\mu_i^\tau$: $\sigma_i(t) = \sigma_i^s \cdot e^{-s_i^\tau |t - \mu_i^\tau|^2}$, where $\sigma_i^s$ represents the static opacity and $s_i^\tau$ represents the temporal scale factor. This allows Gaussian primitives to "appear" or "disappear" within specific temporal windows.

### Loss & Training

The standard 3DGS rendering loss (photometric consistency loss) is adopted without any additional specialized losses. The key lies in the mathematical expressiveness of the model itself, which facilitates easier convergence during optimization.

## Key Experimental Results

### Main Results

**N3DV Dataset (4 scene avg)**:

| Method | Cook Spinach PSNR | Sear Steak PSNR | Flame Steak PSNR | Cut Roast Beef PSNR |
|------|-----------|-----------|------------|-------------|
| 4DGS | 28.12 | 29.07 | 25.04 | 29.71 |
| SWinGS | 31.96 | 32.21 | 32.18 | 31.84 |
| D3DGS | 20.53 | 25.02 | 23.02 | 22.35 |
| **Ours** | **32.59** | **33.12** | **33.34** | **33.06** |

**Technicolor Dataset**:

| Method | Birthday PSNR | Painter PSNR | Train PSNR | Fatma PSNR |
|------|----------|---------|---------|---------|
| D3DGS | 33.81 | 37.38 | - | 38.40 |
| STG | 33.87 | 37.30 | 33.36 | 37.28 |
| **Ours** | **34.72** | **38.37** | **35.30** | **38.91** |

Compared to 4DGS, Ours achieves a **58.25%** Gain on Technicolor Birthday.

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| w/o Time-opacity | 31.17 | 0.952 | 0.096 |
| w/o Time-motion | 29.24 | 0.920 | 0.154 |
| w/o Time-rotation | 31.21 | 0.953 | 0.103 |
| w/o Time-scale | 31.40 | 0.953 | 0.097 |
| w/o Peano remainder | 31.51 | 0.935 | 0.103 |
| **Ours Full** | **33.03** | **0.970** | **0.052** |

### Key Findings

- Removing Time-motion has the most significant impact (resulting in a PSNR drop of nearly 4 dB), indicating that positional motion modeling is core.
- Removing the Peano remainder leads to a 1.5 dB decrease, validating the necessity of constructing a complete Taylor series (rather than a truncated approximation).
- Temporal opacity, rotation, and scale each contribute approximately 1.6-1.9 dB.

## Highlights & Insights

1. **Mathematical Elegance**: Bringing the complete Taylor formula to dynamic Gaussian modeling (third-order explicit expansion + Peano remainder neural network = complete series with zero approximation error) is a highly ingenious idea.
2. **GP-LP Hierarchical Design**: Using an MLP to predict only a small number of skeletal points (GPs) while obtaining a large number of detail points (LPs) via LBS interpolation. This simultaneously ensures efficiency and maintains spatiotemporal consistency.
3. **Both Interpretable and Flexible**: Taylor coefficients directly correspond to physical meanings (velocity, acceleration, etc.), while the Peano remainder provides compensation capabilities for arbitrary complexity.
4. **Significant Performance Gain**: Comprehensively outperforms baselines on two public datasets, showing clear advantages especially when handling complex motion scenes.

## Limitations & Future Work

1. Currently verified only under multi-view settings; the reconstruction capability for **monocular dynamic scenes** remains unexplored.
2. The number of GP points requires manual tuning (farthest point sampling count); an **adaptive GP selection** mechanism is worth exploring.
3. The order of the Taylor expansion is fixed to 3. Higher orders might further improve the accuracy of complex motions, but at the cost of increased computational overhead.
4. Lacks specialized handling for **topological changes** (such as objects appearing/disappearing or experiencing large deformations).

## Related Work & Insights

- **D3DGS** (Yang et al.): A representative implicit deformation field scheme, with which the Peano remainder part in this paper is highly complementary.
- **4DGS** (Wu et al.): Hexplane decomposition + lightweight MLP to predict deformation, which is inferior to ours in multi-view processing.
- **SC-GS** (Huang et al.): Sparse control point-guided editing, sharing a similar design philosophy with the GP-LP architecture in this work.
- **Insights**: The paradigm of Taylor expansion + network-learned remainder can be generalized to any scenario requiring the modeling of continuous time-varying functions (e.g., 4D human body reconstruction, dynamic SLAM).

## Rating

⭐⭐⭐⭐ — Elegant mathematical modeling, comprehensive experiments, and significant SOTA Gain; the originality lies in introducing the complete Taylor series to 4D Gaussian modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
