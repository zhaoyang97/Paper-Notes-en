---
title: >-
  [Paper Note] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning
description: >-
  [ICLR 2026][3D Vision][3D Gaussian Splatting] This paper proposes MEGS2, a method that compresses 3DGS from the perspective of rendering VRAM: it replaces spherical harmonics (SH) entirely with prunable, arbitrarily oriented spherical Gaussians (SG) to reduce per-primitive parameter count, and formulates the joint pruning of primitive count and lobe count as a single memory-constrained optimization problem via a unified soft pruning framework. The result is an 8× reduction in static VRAM and a 6× reduction in rendering VRAM with preserved rendering quality, enabling real-time 3DGS on mobile devices for the first time.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - memory compression
  - spherical harmonic replacement
  - Spherical Gaussians
  - unified pruning
date: 2026-05-08
content_hash: 4d4495f451b75362
---

# MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning

**Conference**: ICLR 2026
**arXiv**: [2509.07021](https://arxiv.org/abs/2509.07021)
**Code**: To be released
**Area**: 3D Vision / Rendering Compression
**Keywords**: 3D Gaussian Splatting, memory compression, spherical harmonic replacement, Spherical Gaussians, unified pruning

## TL;DR
This paper proposes MEGS2, a method that compresses 3DGS from the perspective of rendering VRAM: it replaces spherical harmonics (SH) entirely with prunable, arbitrarily oriented spherical Gaussians (SG) to reduce per-primitive parameter count, and formulates the joint pruning of primitive count and lobe count as a single memory-constrained optimization problem via a unified soft pruning framework. The result is an 8× reduction in static VRAM and a 6× reduction in rendering VRAM with preserved rendering quality, enabling real-time 3DGS on mobile devices for the first time.

## Background & Motivation

**Background**: 3DGS has become the dominant approach for novel view synthesis, but its high memory consumption severely limits deployment on edge devices. While many compression methods have been proposed, the vast majority focus on storage compression (file size) rather than rendering memory compression (VRAM).

**Limitations of Prior Work**: (1) Neural compression / VQ / hash-grid methods (CompactGaussian / EAGLES / HAC++) achieve high storage compression ratios, but require full parameter decompression before rendering, often consuming more VRAM than vanilla 3DGS. (2) Primitive pruning methods (GaussianSpa / Mini-Splatting) can reduce VRAM but offer limited compression ratios — aggressive pruning severely degrades quality. (3) SH as a color representation is parameter-inefficient: high-order coefficients are numerous but poorly utilized.

**Key Challenge**: Rendering VRAM = number of primitives × per-primitive parameter count. Existing methods optimize only one of these two factors. Breaking the VRAM bottleneck requires simultaneously reducing both.

**Key Insight**: SH are global basis functions that require many high-order coefficients to represent localized high-frequency details (sharp specular highlights). SG are local basis functions that efficiently model view-dependent appearance with a small number of lobes, where the lobe count is flexible — making them a natural fit for pruning.

**Core Idea**: Replace SH with prunable SG to reduce per-primitive parameter cost, then simultaneously prune primitive count and lobe count via unified constrained optimization, achieving VRAM-optimal resource allocation.

## Method

### Overall Architecture
Input: a 3DGS scene. Output: a memory-efficient 3DGS representation with substantially reduced static and rendering VRAM.

Three core components: (A) SG replacing SH as the color representation; (B) unified soft pruning framework; (C) post-processing (removal + color compensation + fine-tuning).

### Key Designs

1. **Arbitrarily Oriented Prunable Spherical Gaussians (SG) Replacing SH**:

    - Function: Completely replace SH with SG as the view-dependent color representation.
    - Mechanism: The color of each primitive is $c(\mathbf{v}) = c_0 + \sum_{i=1}^n G(\mathbf{v}; \mu_i, s_i, a_i)$, where $c_0$ is the diffuse component and each SG lobe is defined by an axis direction $\mu_i$, sharpness $s_i$, and RGB amplitude $a_i$. Crucially, lobe directions are unconstrained and need not be orthogonal — arbitrary orientations provide greater flexibility.
    - Design Motivation: Third-order SH requires 48 parameters (16 coefficients × 3 channels), whereas a 3-lobe SG requires approximately half as many parameters while better capturing localized high-frequency details. Fixing lobe axes to orthogonal directions (as in SG-Splatting) causes a 0.6 dB PSNR drop; arbitrarily oriented SG avoids this issue. The variable lobe count makes SG a natural fit for pruning.

2. **Unified Soft Pruning Framework (ADMM-inspired)**:

    - Function: Jointly formulate primitive-count pruning and per-primitive lobe-count pruning as a single memory-constrained optimization problem.
    - Mechanism: The optimization objective is $\min \mathcal{L}(\mathbf{o}, \mathbf{s}, \Theta)$ subject to $\rho_o \|\mathbf{o}\|_0 + \rho_s \|\mathbf{s}\|_0 \leq \kappa$, where $\rho_o = 11$ (base parameters per primitive), $\rho_s = 7$ (parameters per SG lobe), and $\kappa$ is the total parameter budget. Since the $\ell_0$ norm is non-differentiable, ADMM auxiliary variables are introduced to decompose the problem into tractable subproblems: a gradient step, a proximal projection step, and a dual variable update.
    - Design Motivation: Sequential pruning (first reducing primitives, then lobes) is suboptimal because the two quantities are coupled in their effect on quality. The unified framework automatically finds the optimal trade-off between primitive count and lobe count under a shared budget constraint, which experiments confirm is superior to sequential pruning.

3. **Post-processing: Color Compensation**:

    - Function: Compensate for the contribution of removed low-sharpness lobes to the diffuse color.
    - Mechanism: Upon removing lobe $i$, a compensation term $\Delta c_0 = a_i \cdot \frac{1 - e^{-2s_i}}{2s_i}$ is computed, and the diffuse color is updated as $c_0' = c_0 + \Delta c_0$. This closed-form solution is derived by minimizing the integral of color difference over the sphere.
    - Design Motivation: Directly removing low-sharpness lobes discards their contribution to the average color, causing a global color shift. The analytic compensation recovers this energy at negligible additional cost.

### Loss & Training
- Training follows the standard 3DGS pipeline (Kerbl et al., 2023).
- ADMM optimization: alternating gradient steps (minimizing rendering loss), proximal projection steps (enforcing sparsity), and dual variable updates.
- Post-processing: removal of near-zero-opacity primitives and near-zero-sharpness lobes → color compensation → brief fine-tuning to recover quality.

## Key Experimental Results

### Main Results (Mip-NeRF360)

| Method | PSNR | SSIM | LPIPS | Static VRAM (MB) | Rendering VRAM (MB) |
|--------|------|------|-------|-----------------|---------------------|
| 3DGS | 27.48 | 0.813 | 0.217 | 648 | 1717 |
| GaussianSpa | 27.56 | 0.824 | 0.215 | 115 | 448 |
| **MEGS2 (HQ)** | **27.54** | **0.824** | **0.209** | **55** | **265** |
| MEGS2 (LM) | 27.21 | 0.814 | 0.227 | 40 | 224 |

### Ablation Study (Mip-NeRF360)

| Configuration | PSNR | LPIPS | VRAM (MB) | Notes |
|---------------|------|-------|-----------|-------|
| GaussianSpa + Reduced3DGS | 26.05 | 0.280 | 402 | Naïve combination severely degrades quality |
| GaussianSpa (SH→SG) | 27.01 | 0.230 | 339 | Simple substitution is insufficient |
| soft→hard pruning | 27.23 | 0.228 | 288 | Hard pruning inferior to soft pruning |
| unified→sequential | 27.33 | 0.222 | 328 | Sequential inferior to unified |
| w/o color comp. | 27.46 | 0.213 | 265 | Color compensation is beneficial |
| **Full model** | **27.54** | **0.209** | **265** | All components work optimally together |

### Key Findings
- **VRAM compression**: 8× static VRAM reduction over 3DGS (648→55 MB) and 6× rendering VRAM reduction (1717→265 MB). Compared to SOTA GaussianSpa, MEGS2 achieves an additional 2× static and ~40% rendering VRAM reduction.
- **Quality preservation**: PSNR is nearly lossless (27.54 vs. 27.56), and LPIPS is even improved (0.209 vs. 0.215).
- **SG outperforms SH**: SG better fits localized high-frequency signals (sharp reflections and specular highlights), with notable advantages over SH in specular-heavy scenes such as Bicycle and Truck.
- **Lobe distribution**: Most primitives require only 0–1 lobes (predominantly diffuse surfaces); a small subset requires 2–3 lobes (specular highlights), yielding an average of 1.3–1.7 lobes per primitive.

## Highlights & Insights
- **Precision in problem formulation**: The key insight is distinguishing storage compression from memory compression. Prior work has overwhelmingly focused on the former while neglecting the latter, which is the true bottleneck for edge deployment.
- **Justification for replacing SH with SG**: The vast majority of surfaces in a scene are diffuse (requiring no lobes); only a small fraction exhibits specular or highlight behavior requiring lobes. The variable lobe count of SG perfectly matches this long-tail distribution.
- **ADMM formulation for unified pruning**: Casting two discrete optimization problems as a single continuous optimization and decomposing it via ADMM is elegant and theoretically grounded. This framework generalizes to any setting requiring simultaneous optimization of "entity count" and "per-entity complexity."
- **Closed-form color compensation**: The analytic solution derived via spherical integration introduces no additional computational overhead and is both simple and effective.

## Limitations & Future Work
- The work focuses on static VRAM compression; optimization of dynamic VRAM (which depends on renderer implementation) is left for future work.
- Performance in highly complex specular scenes (e.g., fully mirror-like objects) warrants further investigation.
- Combination with neural compression methods (e.g., HAC++) could jointly optimize both storage and VRAM.
- Optimal initialization strategies for SG lobes remain an open question.

## Related Work & Insights
- **vs. GaussianSpa**: Performs only primitive pruning while retaining full SH per primitive, resulting in a higher VRAM lower bound. MEGS2 breaks this bound by additionally compressing per-primitive parameters.
- **vs. Reduced3DGS**: Also attempts to prune SH coefficients, but the global nature of SH makes sparse pruning unsafe — removing high-order terms degrades detail globally. The locality of SG makes lobe pruning far safer.
- **vs. CompactGaussian / EAGLES**: Achieve high storage compression ratios but may actually increase VRAM (due to decompression requirements), fundamentally failing to address the rendering memory problem.

## Rating
- Novelty: ⭐⭐⭐⭐ Complete replacement of SH with SG combined with a unified pruning framework; conceptually clear and innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across three datasets, detailed ablations, and real-device validation via WebGL.
- Writing Quality: ⭐⭐⭐⭐ VRAM analysis is clear and problem decomposition is well-articulated.
- Value: ⭐⭐⭐⭐⭐ First systematic solution to the rendering memory bottleneck of 3DGS, directly enabling edge deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](../../CVPR2026/3d_vision/prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)

<!-- RELATED:END -->
