---
title: >-
  [Paper Note] Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment
description: >-
  [ECCV 2024][3D Vision][Bundle Adjustment] This paper proposes the Power Variable Projection (PoVar) algorithm, which extends the power series expansion method to the Variable Projection (VarPro) framework and further generalizes it to Riemannian manifold optimization. This achieves efficient optimization of initialization-free large-scale Bundle Adjustment (BA) for the first time.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Bundle Adjustment"
  - "Initialization-Free Optimization"
  - "Variable Projection"
  - "Power Series Expansion"
  - "Riemannian Manifold Optimization"
date: 2026-05-08
content_hash: ae6df10b1598a784
---

# Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment

**Conference**: ECCV 2024  
**arXiv**: [2405.05079](https://arxiv.org/abs/2405.05079)  
**Code**: [GitHub](https://github.com/tum-vision/povar)  
**Area**: Others  
**Keywords**: Bundle Adjustment, Initialization-Free Optimization, Variable Projection, Power Series Expansion, Riemannian Manifold Optimization

## TL;DR

This paper proposes the Power Variable Projection (PoVar) algorithm, which extends the power series expansion method to the Variable Projection (VarPro) framework and further generalizes it to Riemannian manifold optimization. This achieves efficient optimization of initialization-free large-scale Bundle Adjustment (BA) for the first time.

## Background & Motivation

**Background**: Bundle Adjustment (BA) is a core component of SfM and 3D reconstruction. Traditional BA relies on the Levenberg-Marquardt (LM) algorithm paired with the Schur complement trick, which requires high-quality initialization. Recently, PoBA has significantly accelerated traditional BA and improved its accuracy by using power series expansion to invert the Schur complement.

**Limitations of Prior Work**: Initialization-free BA—recovering camera poses and landmarks solely from image observations—remains a largely unexplored area. Existing methods (e.g., pOSE) employ a stratified BA strategy but rely on direct factorization (Cholesky/QR), which only scales to small problems involving dozens of cameras. While the VarPro algorithm benefits from a wide basin of convergence, its scalability has long been a major bottleneck.

**Key Challenge**: VarPro is highly suitable for initialization-free BA (due to its broad convergence basin) but lacks an efficient solver; direct factorization does not scale, and the convergence of Preconditioned Conjugate Gradient (PCG) is unfavorable to the undamped structure of VarPro.

**Goal**: Enable large-scale initialization-free BA to scale to scenes with thousands of cameras.

**Key Insight**: By extending the power series inversion method to VarPro (Phase 1) and Riemannian manifold optimization (Phase 2), the authors provide efficient solvers for both phases of stratified BA.

**Core Idea**: Although the Schur complement of VarPro shares a similar structure with traditional BA, its convergence behavior differs because of the undamped landmark variables. This paper proves that the power series expansion still holds in this setting, and further generalizes it to Riemannian manifold optimization under homogeneous coordinates.

## Method

### Overall Architecture

A three-phase strategy for stratified BA is adopted:
1. **Phase 1**: Optimization using the pOSE objective function (interpolant between affine and projective). The landmark variables are eliminated via VarPro, and PoVar is used to efficiently solve the reduced camera system.
2. **Phase 2**: Refinement using the standard projective objective function under homogeneous coordinates, solved via RiPoBA with power series expansion on Riemannian manifolds.
3. **Phase 3**: Metric upgrade, where the projective camera matrices are constrained to $SE(3)$.

### Key Designs

1. **Power Variable Projection (PoVar)**: A power series solver for VarPro.

    - Core idea of VarPro: In the least squares problem $\min_{x_p, \tilde{x}_l} \|G(x_p)\tilde{x}_l - z(x_p)\|_2^2$, the landmark variable $\tilde{x}_l$ is substituted with its analytical solution $\tilde{x}_l^*(x_p) = G(x_p)^\dagger z(x_p)$, leaving only the camera parameters $x_p$ for optimization.
    - The Schur complement of VarPro is $S^V = U_\lambda - WV_0^{-1}W^\top$. The key difference from traditional BA is that the landmark Hessian $V_0$ is undamped (only guaranteed to be semi-positive definite).
    - **Core Theorem**: Proves that the eigenvalues of $U_\lambda^{-1}WV_0^\dagger W^\top$ satisfy $0 \leq \mu < 1$, which ensures the convergence of the power series expansion:
    $x(m) = -\sum_{i=0}^{m}(U_\lambda^{-1}WV_0^{-1}W^\top)^i U_\lambda^{-1}(b_p - WV_0^{-1}b_l)$
    - **Design Motivation**: Although PoVar shares a similar algorithmic structure with PoBA, because VarPro only damps the camera parameters, its convergence behavior is completely different. PoVar achieves smoother convergence, particularly under high-precision requirements.

2. **Riemannian Power BA (RiPoBA)**: Extending the power series to Riemannian manifold optimization.

    - Phase 2 involves optimization under homogeneous coordinates (cameras $\text{vec}(\tilde{x}_p^i) \in S^{12}$ and landmarks $\tilde{x}_l^j \in S^4$). This introduces local scale ambiguities, demanding Riemannian manifold optimization.
    - Jacobians and damping parameters are projected onto the tangent space: $\tilde{J}_p = J_p \tilde{x}_p^\perp$ and $\tilde{J}_l = J_l \tilde{x}_l^\perp$.
    - With unified notation, the structure of the normal equation is consistent with standard BA. It is proved that the power series expansion of the Riemannian Schur complement also holds:
    $\tilde{S}^{-1} \approx \sum_{i=0}^{m}(\tilde{U}_{\tilde{\lambda}}^{-1}\tilde{W}\tilde{V}_{\tilde{\lambda}}^{-1}\tilde{W}^\top)^i \tilde{U}_{\tilde{\lambda}}^{-1}$
    - **Design Motivation**: Direct factorization is infeasible for large-scale problems, and PCG suffers from unstable convergence under the Riemannian framework. The block-diagonal structure of the matrices is exploited to implement memory-efficient tangent space projection and storage.

3. **Efficient Storage Strategy**: Exploits the sparse structure of BA by organizing landmarks into dense blocks. Tangent space projection matrices are applied to the pose Jacobian and landmark Jacobian within each block, maintaining memory efficiency (e.g., projecting the pose Jacobian from $\mathbb{R}^{2 \times 12}$ to $\mathbb{R}^{2 \times 11}$ and the landmark Jacobian from $\mathbb{R}^{2 \times 4}$ to $\mathbb{R}^{2 \times 3}$).

### Loss & Training

- Maximum iterations per phase: 50, with early stopping when the relative function tolerance is below $10^{-6}$.
- Damping factor $\lambda$ is initialized to $10^{-4}$ and updated following the LM strategy.
- Maximum power series order is set to 20 with a threshold of 0.01.
- Maximum inner iterations for the iterative method: 500.
- pOSE parameter $\eta = 0.1$.
- The implementation is based on C++ and extended from the PoBA codebase.

## Key Experimental Results

### Main Results

Evaluated on 97 real-world BA problems from the BAL dataset (ranging from 16 to 13,682 cameras). Performance profiles are used to assess speed and accuracy simultaneously:

| Experimental Stage | Metric (τ=0.001) | PoVar | PoBA | Direct Factorization | Iterative Method |
|----------|------|------|------|------|------|
| Phase 1 | Solve Percentage (High Precision) | **Best** | Second | Worst | Third |
| Dual-Phase Combination | PoVar+RiPoBA | **Best** | — | — | — |

PoVar+RiPoBA outperforms all competing combinations across all tolerance levels and relative runtime scales.

### Ablation Study

| Configuration | Effect | Explanation |
|---|---|---|
| PoVar vs PoBA (Phase 1) | PoVar is significantly better at τ=0.001 | The undamped structure of VarPro leads to different convergence behaviors. |
| RiPoBA vs RiPCG (Phase 2) | RiPoBA outperforms under all tolerances | The power series approach is superior to preconditioned conjugate gradient. |
| PoVar Convergence Curve | Smoother without stalling | In contrast, PoBA may stall in early iterations. |
| Venice-1672 (Large-scale) | PoVar+RiPoBA converges to the lowest error | Achieves initialization-free BA on a thousand-camera scale for the first time. |

### Key Findings

- The advantage of PoVar is especially striking under high-precision requirements ($\tau=0.001$), demonstrating the wide convergence basin of VarPro.
- PoVar yields a much smoother convergence curve than PoBA, which often experiences "stalling" in early iterations.
- Given the same Phase 1 solver, RiPoBA consistently outperforms RiPCG, validating the efficacy of the Riemannian power series expansion.
- On large-scale problems such as Venice-1672, PoVar+RiPoBA is the only combination capable of converging to highly accurate solutions.

## Highlights & Insights

- **First to address the scalability of large-scale initialization-free BA**: Unlike prior work limited to dozens of cameras, this work scales to thousands of cameras.
- **Rigorous mathematical proof**: Thorough proofs are provided for the power-series convergence of both the semi-positive definite $V_0$ in VarPro's Schur complement and the Riemannian framework.
- **Subtle differences between PoVar and PoBA**: Despite their similar algorithmic structures, damping only the camera variables in VarPro yields completely different convergence properties. This insight provides significant theoretical depth.
- **Bridge between Riemannian optimization and power series**: Unifying tangent space projection with the power series framework offers a novel direction for large-scale optimization on manifolds.

## Limitations & Future Work

- Assumes outlier tracks are pre-filtered; robust BA handling is not addressed.
- The metric upgrade phase uses known approximate focal lengths (from the BAL dataset). Real-world deployment would require intrinsic estimation.
- Verification is limited to the BAL dataset and can be extended to newer datasets such as MegaDepth.
- The method has not yet been integrated into a complete SfM pipeline (e.g., COLMAP) to evaluate end-to-end performance.

## Related Work & Insights

- **VarPro Lineages**: From the original Golub-Pereyra algorithm to its applications in computer vision by Hong et al., this work resolves its scalability bottleneck.
- **pOSE**: The proposed objective function, interpolating between affine and projective models, offers a broad basin of convergence and forms the foundation of Phase 1.
- **PoBA**: A breakthrough that substitutes PCG/direct factorization with a power series in traditional BA, which this work generalizes to VarPro and Riemannian frameworks.
- **Insight**: Generalizing power series expansions to new types of Schur complements (with different damping structures) is a promising and general methodology.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Solid theoretical contribution; solves the scalability of large-scale initialization-free BA for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering all 97 problems in BAL with performance profiles, though lacking end-to-end SfM evaluation.
- Writing Quality: ⭐⭐⭐⭐ Rigorously derived, but mathematical notation is dense, presenting a higher barrier to entry.
- Value: ⭐⭐⭐⭐ Highly valuable for engineering by paving the way for scalable, initialization-free SfM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Event-based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)
- [\[CVPR 2026\] Parallel Rigidity Matters for Bundle Adjustment](../../CVPR2026/3d_vision/parallel_rigidity_matters_for_bundle_adjustment.md)
- [\[ECCV 2024\] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](../../ICCV2025/3d_vision/back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
