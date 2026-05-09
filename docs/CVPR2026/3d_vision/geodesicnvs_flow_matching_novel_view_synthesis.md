---
title: >-
  [Paper Note] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis
description: >-
  [CVPR 2026][3D Vision][Flow Matching] This paper proposes Data-to-Data Flow Matching (D2D-FM) to directly learn deterministic transformations between view pairs, and regularizes flow paths via probability density geodesics so that trajectories propagate along high-density data manifolds, achieving improved view consistency and geometric fidelity in novel view synthesis.
tags:
  - CVPR 2026
  - 3D Vision
  - Flow Matching
  - Geodesic
  - Probability Density
  - Data-to-Data
  - Novel View Synthesis
date: 2026-05-08
content_hash: 9d555a80d8d80de7
---

# GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.01010](https://arxiv.org/abs/2603.01010)
**Code**: To be confirmed
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: Flow Matching, Geodesic, Probability Density, Data-to-Data, Novel View Synthesis

## TL;DR

This paper proposes Data-to-Data Flow Matching (D2D-FM) to directly learn deterministic transformations between view pairs, and regularizes flow paths via probability density geodesics so that trajectories propagate along high-density data manifolds, achieving improved view consistency and geometric fidelity in novel view synthesis.

## Background & Motivation

**Diffusion-based NVS relies on stochastic noise-to-data transformations**, whose inherent randomness obscures the deterministic geometric structure between views, leading to inconsistent predictions across viewpoints. While standard Conditional Flow Matching (CFM) offers a deterministic alternative, its linear interpolation path $x_t = (1-t)x_0 + tx_1$ may traverse low-density regions in latent space, producing unrealistic intermediate states.

**View transformation in NVS is intrinsically deterministic**—the projection of the same scene under different camera poses follows an exact geometric relationship. This demands that the generative model directly learn inter-view transformation mappings rather than sampling novel views from a noise distribution, motivating a Data-to-Data framework.

**Even within a D2D framework, linear interpolation remains suboptimal**: the straight-line path between the latents of two views may pass through regions outside the data manifold, yielding unnatural intermediate states. Ideal interpolation should follow geodesics on the manifold—shortest paths through high-probability-density regions.

## Method

### Overall Architecture

The method proceeds in two stages: (1) **Data-to-Data Flow Matching** learns a deterministic flow between view pairs—source and target images are VAE-encoded, and a U-Net velocity network conditioned on Plücker ray encodings and CLIP features integrates the ODE to generate the target view; (2) **Probability Density Geodesic Regularization**—the score function of a pretrained diffusion model serves as a density proxy, and a GeodesicNet is trained via variational distillation to produce manifold-aligned interpolation paths.

### Key Designs

1. **Data-to-Data Flow Matching (D2D-FM)**:

   - **Function**: Directly learns a deterministic flow between paired views, replacing the conventional noise-to-data paradigm.
   - **Mechanism**: Given latent representations of a source view $x_0$ and a target view $x_1$, the method learns a velocity field $v_\theta(x_t, t, c)$ such that $x_t$ evolves along the path from $x_0$ to $x_1$. The conditioning signal $c$ comprises Plücker ray camera pose encodings and CLIP source-view features. The linear variant is: $x_t = (1-t)x_0 + tx_1 + \sigma\epsilon$.
   - **Design Motivation**: Since view transformation is deterministic, D2D directly models data-to-data mappings, preserving structural correspondences without requiring a noise prior.

2. **Probability Density Geodesic Flow Matching (PDG-FM)**:

   - **Function**: Constrains flow paths to high-density regions of the data manifold.
   - **Mechanism**: A Riemannian metric $G(x) = p(x)^{-2}I$ is defined such that low-density regions incur high path cost and high-density regions incur low cost. Geodesics satisfying the Euler–Lagrange equation $$\ddot{\gamma} + \|\dot{\gamma}\|^2(I - \hat{\dot{\gamma}}\hat{\dot{\gamma}}^T)\nabla\log p(\gamma) = 0$$ are the desired paths.
   - **Design Motivation**: Linear interpolation may traverse low-density off-manifold regions, producing unnatural intermediate states; geodesics ensure the path remains within perceptually realistic regions throughout.

3. **Variational Distillation Training (GeodesicNet)**:

   - **Function**: Efficiently trains a geodesic interpolation network, decoupled from the FM training stage.
   - **Mechanism**: A teacher performs geodesic optimization (minimizing path energy) in DDIM-F space, and a student GeodesicNet is distilled into VAE space. The score function of a pretrained diffusion model, $\nabla\log p(x) \approx -\epsilon_\phi(x, t)/\sigma_t$, serves as a density proxy, eliminating the need for explicit density estimation.
   - **Design Motivation**: Performing geodesic optimization directly during FM training is computationally expensive; distillation decouples the two training stages and substantially reduces computational cost.

### Loss & Training

D2D flow matching loss: $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$. Geodesic training: minimization of the Euler–Lagrange residual. Optimizer: AdamW; batch size: 256.

## Key Experimental Results

### Main Results

| Setting | FID↓ | CLIP-S↑ | SSIM↑ |
|---------|------|---------|-------|
| D2D-FM (100NFE) | 5.43 | 89.0 | 0.863 |
| Naive FM (N2D) | 5.51 | 88.9 | 0.862 |
| Geodesic FM (LVIS) | 10.40 | 92.3 | 0.877 |
| Linear FM (LVIS) | 11.81 | 94.3 | 0.874 |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Geodesic vs. Linear AOFM | 13.70 vs. 1.04 | Geodesic paths exhibit genuine viewpoint rotation; linear paths are nearly static |
| D2D-FM 10NFE vs. 100NFE | Small gap | D2D advantage is more pronounced under fewer-step inference |
| With / without score regularization | Geodesic superior | Lower Euler–Lagrange residual; smoother paths |

### Key Findings

- The Average Optical Flow Magnitude (AOFM) of geodesic interpolation is substantially higher than that of linear interpolation, indicating that intermediate states capture meaningful viewpoint transitions rather than silent blending.
- D2D-FM demonstrates a more pronounced advantage at low-step inference (10NFE), as deterministic paths are more stable than stochastic sampling.
- Using diffusion scores as density proxies is effective—explicit estimation of complex high-dimensional densities is avoided entirely.

## Highlights & Insights

- The D2D-FM paradigm is fundamentally more principled for NVS: novel view synthesis is a deterministic mapping, not a noise-sampling problem. The mathematical framework of probability density geodesics is elegant, and using diffusion scores—a byproduct of pretrained models already available—as density proxies is an ingenious engineering choice that circumvents the difficulty of explicit density estimation.

## Limitations & Future Work

- The multi-stage training pipeline (D2D-FM + GeodesicNet distillation) is complex and may limit scalability.
- Geodesic optimization depends on the quality of the score function from the pretrained diffusion model.
- Evaluation is confined to synthetic datasets (Objaverse/GSO); large-scale validation on real-world scenes is absent.
- FID and CLIP-S exhibit inconsistent ranking directions across settings, warranting careful metric selection.

## Related Work & Insights

- **vs. Zero-1-to-3**: Zero-1-to-3 adopts the noise-to-data (N2D) conditional diffusion paradigm; GeodesicNVS eliminates stochasticity via D2D, achieving substantially better FID.
- **vs. Riemannian FM**: Riemannian FM assumes a fixed geometry, whereas this work employs a data-dependent density metric for adaptive manifold-aware flow.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ D2D-FM combined with probability density geodesics is pioneering within the NVS field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Objaverse/GSO evaluation is thorough, but real-scene validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematically rigorous with complete Euler–Lagrange derivations.
- **Value**: ⭐⭐⭐⭐ The D2D paradigm and geodesic regularization have cross-domain transfer potential.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[CVPR 2026\] Scaling View Synthesis Transformers (SVSM)](scaling_view_synthesis_transformers.md)
- [\[CVPR 2026\] Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow](easy3e_feed-forward_3d_asset_editing_via_rectified_voxel_flow.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)

<!-- RELATED:END -->
