---
title: >-
  [Paper Note] Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction
description: >-
  [ICML 2026][3D Vision][Photometric Ambiguity] AmbiSuR explicitly models two types of intrinsic photometric ambiguities in Gaussian Splatting (primitive boundary spillover and under-constrained pixel blending) and resolve…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Photometric Ambiguity"
  - "Gaussian Truncation"
  - "Ray-Color Consistency"
  - "SH Self-Indicator"
  - "Sparse Regularization"
date: 2026-05-08
content_hash: 86f60e33f48016bd
---

# Revisiting Photometric Ambiguity for Accurate Gaussian-Splatting Surface Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2605.12494](https://arxiv.org/abs/2605.12494)  
**Code**: Project page https://fictionarry.github.io/AmbiSuR-Proj/  
**Area**: 3D Vision / Surface Reconstruction / Gaussian Splatting  
**Keywords**: Photometric Ambiguity, Gaussian Truncation, Ray-Color Consistency, SH Self-Indicator, Sparse Regularization

## TL;DR
AmbiSuR explicitly models two types of intrinsic photometric ambiguities in Gaussian Splatting (primitive boundary spillover and under-constrained pixel blending) and resolves them using truncation and ray-color consistency. It further employs higher-order spherical harmonic coefficients as "self-indicators" to identify high-risk ambiguous primitives and applies amorphous local prior regularization. This reduces the average Chamfer distance on DTU to 0.46, surpassing the previous best GeoSVR (0.47).

## Background & Motivation
**Background**: Converting multi-view images into 3D surfaces has shifted from implicit SDF methods (e.g., NeuS, Neuralangelo) to explicit 3DGS families (e.g., 2DGS, GOF, PGSR, MILo, GeoSVR), which demonstrate clear advantages in efficiency and detail. These methods commonly rely on "multi-view photometric consistency," where the color of the same spatial point should remain consistent across different views.

**Limitations of Prior Work**: In real-world scenarios, photometric consistency is rarely perfectly satisfied due to textureless regions, reflective surfaces, shadows, and insufficient coverage, leading to severely ill-posed multi-view triangulation. Existing solutions either involve complex ray modeling for reflections (limited to specific scenes) or apply coarse-grained regularization using MonoDepth/Normal models (which tend to "smooth out" entire images), without addressing the root cause within the 3DGS representation itself.

**Key Challenge**: The 3DGS rendering equation $\mathbf{C} = \sum_i c_i \tilde\alpha_i \prod_{j<i}(1-\tilde\alpha_j)$ aims for "weighted sum equals ground truth," which suffices for pixel color synthesis but is severely under-constrained for "unique surface reconstruction." Any set $\{c_i, w_i\}$ that sums to $\mathbf{C}_{gt}$ is considered correct, allowing reconstruction to be misled by "pseudo-geometry and view-dependent effects."

**Goal**: To resolve ambiguities at both the representation and supervision levels—eliminating geometric ambiguities in Gaussian forward computation and actively identifying ambiguities in unreliable supervision, providing targeted prior compensation.

**Key Insight**: By systematically analyzing the 3DGS pipeline, the authors identified two types of intrinsic representation ambiguities (long-tail low-opacity primitives and excessive freedom in primitive color blending). They also observed that SH coefficients can naturally serve as "ambiguity detectors"—abnormally large higher-order SH coefficients indicate fitting view-dependent residuals, while abnormally small coefficients suggest insufficient supervision.

**Core Idea**: Address representation ambiguities with "primitive truncation + ray-color variance" constraints, then use higher-order SH coefficients as dual-ended indicators (upper/lower quantiles) to identify problematic primitives. Apply minimal-intrusion amorphous local normal regularization to these primitives.

## Method

### Overall Architecture
AmbiSuR is integrated into 3DGS reconstruction pipelines like PGSR, with two new components: (a) Gaussian Splatting photometric disambiguation—truncating Gaussian boundaries during forward computation and introducing ray-color consistency constraints; (b) SH ambiguity self-indication—selecting high-ambiguity primitives based on each primitive's $I_{SH} = \|f_{\text{rest}}\|_2^2$ upper/lower quantiles, applying depth-normal consistency regularization in their projection regions, and freezing parameters of other primitives. The total loss is $\mathcal{L} = \mathcal{L}_{photo} + \tau\mathcal{L}_{geo} + \mu_1\mathcal{N} + \mu_2\mathcal{R}$, with both a metric-depth-enhanced version (AmbiSuR) and a mono-depth-compatible version (AmbiSuR-Mono).

### Key Designs

1. **Gaussian Primitive Truncation + Ray-Color Consistency**:

    - **Function**: Resolves two types of 3DGS representation ambiguities—low-opacity primitive boundary spillover and under-constrained pixel color blending.
    - **Mechanism**: Each Gaussian projection is divided into a core region $\mathcal{G}_{core}$ and an edge region $\mathcal{G}_{edge}$. Only the core is retained during rendering: $\tilde\alpha_{\mathcal{T}}(\mathbf{x}) = \alpha\,\mathcal{G}_{core}(\mathbf{x})\cdot\mathbb{1}(\|\mathbf{x}-\mu_i\|\le \gamma\sigma_i)$ ($\gamma=2$), physically enforcing that each primitive "speaks" only within its $2\sigma$ range. Alpha-blending is treated as a probability distribution along the ray, defining ray-color consistency $\mathcal{R}(\mathbf{r}) = \sum_i w_i\|c_i - \mathbf{C}\|_2^2$ (only $c_i$ participates in gradients), ensuring that primitives representing the same surface have similar view-dependent colors.
    - **Design Motivation**: Truncation physically eliminates the disadvantageous behavior of "low-opacity long tails repeatedly mismatched across views." Ray-color variance upgrades the weak constraint of "weighted sum equals ground truth" to "color terms should be mutually consistent," forcing individual primitives to fit the true optical properties of the surface rather than cheating through blending.

2. **Higher-Order SH Coefficients as Photometric Ambiguity Self-Indicators**:

    - **Function**: Automatically identifies primitives most likely to suffer from supervision ambiguity or underfitting without external oracles.
    - **Mechanism**: Decompose each primitive's color function as $C(\mathbf{d}) = \bar C + \sum_i \beta_i Y_i(\mathbf{d})$. By orthogonality, view-dependent bias energy $\propto \sum_i \beta_i^2$, defining $I_{SH} = \|f_{\text{rest}}\|_2^2$. Use "dual-ended indication"—abnormally high $I_{SH}$ (upper $\eta_U$ quantile) suggests overfitting view-dependent residuals due to inconsistent supervision, while abnormally low $I_{SH}$ (lower $\eta_L$ quantile) suggests under-supervision or erroneous appearance baking. The union $\mathcal{S} = \mathcal{S}_U\cup\mathcal{S}_L$ forms the "high-risk ambiguity" primitive set.
    - **Design Motivation**: 3DGS inherently learns SH coefficients, making this a true free-lunch approach. Dual-ended indication covers both "overfitting view-dependent effects" and "underfitting textureless regions," outperforming reliance on external segmentation/uncertainty networks.

3. **Amorphous Local Regularization + Parameter Separation**:

    - **Function**: Applies geometric priors only within the projection regions of ambiguous primitives, avoiding contamination of well-converged regions.
    - **Mechanism**: Freeze parameters of non-ambiguous primitives and exclude opacity $\alpha$ and scale $s$ from regularization (only direction-related attributes are adjusted). Use the indicator set $\mathcal{S}$ to compute a soft mask $\mathbf{M} = \sum_i \mathbb{1}(i\in\mathcal{S})\,\tilde\alpha_i\prod_{j<i}(1-\tilde\alpha_j)$ for each pixel, then apply $\mathcal{N} = \mathrm{Mean}(\mathbf{M}\cdot(1 - \mathbf{N}_{\mathbf{D}}\cdot\mathbf{N}_P))$, where the angular difference between rendered depth-derived normals and prior normals is weighted by the soft mask, effective only in ambiguous projection regions.
    - **Design Motivation**: Traditional methods apply priors uniformly across the entire image, smoothing out good regions. This work uses "indicator-based primitive selection → soft mask-based pixel selection" for precise localization, ensuring priors are applied only to the weakest areas while maintaining compatibility with various prior sources (monocular, multi-view, stereo matching).

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{photo} + \tau\mathcal{L}_{geo} + \mu_1\mathcal{N} + \mu_2\mathcal{R}$, with fixed $\tau=0.1, \mu_1=0.1, \mu_2=10^{-5}$. Training involves 30k iterations. The metric version uses Depth Anything 3 for multi-view depth, while the Mono version uses Depth Anything V2 for monocular depth. TSDF mesh extraction is applied.

## Key Experimental Results

### Main Results

| Dataset | Metric | Prev. SOTA | AmbiSuR | Notes |
|---------|--------|------------|---------|-------|
| DTU Avg | Chamfer ↓ | 0.47 (GeoSVR) | 0.46 | 0.6h training |
| DTU 24/37/40 | Chamfer ↓ | 0.32/0.51/0.30 (GeoSVR) | 0.32/0.48/0.31 | Comparable or better across scenes |
| Tanks&Temples | F1 ↑ | Best per competitor | Highest F1 overall | Large scenes with significant photometric ambiguity |
| Training Time | — | NeuS >12h, Neuralangelo >128h | 0.6h | Explicit + efficient |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| PGSR baseline only | 0.52 (DTU Chamfer) | Starting point |
| + Primitive Truncation | Improved but limited | Addresses representation ambiguity only |
| + Ray-Color Consistency | Further improvement | Suppresses blending ambiguity |
| + SH Dual-Ended Indication + Amorphous Regularization | Converges to 0.46 | Resolves supervision ambiguity |
| Single-Ended vs Dual-Ended Indication | Dual-ended more stable | Low SH also provides value |
| Full-Image Regularization vs Amorphous | Amorphous preserves details | Validates selective regularization benefits |

### Key Findings
- The long-tail opacity of "representative" primitives is an overlooked source of ambiguity—simple truncation yields significant gains, revealing long-tolerated flaws in 3DGS representation.
- Higher-order SH coefficients serve as "free ambiguity detectors," highly beneficial for budget-sensitive industrial 3D reconstruction, as they eliminate the need for additional classifiers or large models.
- AmbiSuR-Mono achieves near-metric performance with monocular depth, demonstrating robustness to prior quality due to its design of applying priors only in ambiguous regions, rather than relying on specific depth sources.

## Highlights & Insights
- Using SH coefficients as indicators is a rare true free-lunch approach—repurposing existing learned parameters as new functional indicators with almost zero additional computation. This idea is transferable to other explicit representations (e.g., uncertainty in 3DGS-SLAM).
- Ray-color variance upgrades "weighted sum correctness" to "distribution consistency," providing an elegant tightening of under-constrained supervision. This concept is also applicable to other "weighted aggregation" tasks (e.g., NeRF density fields, light field reconstruction).
- The three-layer control of "primitive freezing + soft masking + direction-only parameters" is a key engineering technique for amorphous regularization, emphasizing that "where to add priors" is more critical than "what priors to add."

## Limitations & Future Work
- The truncation threshold $\gamma$ and upper/lower quantiles $\eta_U, \eta_L$ still require minor tuning per dataset (different values for DTU and outdoor scenes), lacking adaptive mechanisms.
- Transparent objects and strong specular surfaces (conflicting with BRDF and SH assumptions) remain untested, potentially requiring stronger view-dependent modeling.
- Indicators are based on single-frame SH, without leveraging temporal/viewpoint consistency, which needs extension for dynamic scenes.

## Related Work & Insights
- **vs 2DGS / GOF / PGSR / MILo / GeoSVR**: These works focus on "geometric alignment + high-quality mesh extraction," while this paper traces back to "photometric ambiguity" and provides dual-level disambiguation for representation and supervision, making it orthogonal to them.
- **vs MonoSDF / Neuralangelo**: SDF methods rely on implicit high-capacity MLPs, with training times exceeding tens of hours. AmbiSuR achieves superior results in 0.6h using explicit 3DGS, highlighting the engineering advantages of explicit representations.
- **vs VCR-GauS / Geometric Prior Embedding**: Previous methods applied strong priors uniformly across the entire image, while this work uses SH self-indicators to focus priors on "truly information-deficient" regions, avoiding the common issue of "smoothing out details with priors."

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine "SH higher-order coefficients as ambiguity indicators" and "ray-color variance."
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons on DTU and T&T, testing various prior sources, but lacks dedicated tests for reflective/transparent objects.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation structured around representation and supervision, with well-integrated formulas and visualizations.
- Value: ⭐⭐⭐⭐ Achieves SOTA in 0.6h training, robust to priors, and industrially deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[NeurIPS 2025\] GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction](../../NeurIPS2025/3d_vision/geosvr_taming_sparse_voxels_for_geometrically_accurate_surface_reconstruction.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
