---
title: >-
  [Paper Note] SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors
description: >-
  [ICLR 2026][3D Vision][2D Gaussian Splatting] SurfSplat proposes a feedforward 3D reconstruction framework based on 2DGS that binds Gaussian rotation and scale to local neighborhood positions via surface continuity priors, resolves color bias through a forced alpha blending strategy, and introduces the HRRC metric to reveal reconstruction quality discrepancies at high resolutions.
tags:
  - ICLR 2026
  - 3D Vision
  - 2D Gaussian Splatting
  - feedforward 3D reconstruction
  - surface continuity
  - high-resolution rendering consistency
  - sparse-view
date: 2026-05-08
content_hash: ea3df5512a5f3077
---

# SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors

**Conference**: ICLR 2026
**arXiv**: [2602.02000](https://arxiv.org/abs/2602.02000)
**Code**: [https://hebing-sjtu.github.io/SurfSplat-website/](https://hebing-sjtu.github.io/SurfSplat-website/)
**Area**: 3D Vision
**Keywords**: 2D Gaussian Splatting, feedforward 3D reconstruction, surface continuity, high-resolution rendering consistency, sparse-view

## TL;DR

SurfSplat proposes a feedforward 3D reconstruction framework based on 2DGS that binds Gaussian rotation and scale to local neighborhood positions via surface continuity priors, resolves color bias through a forced alpha blending strategy, and introduces the HRRC metric to reveal reconstruction quality discrepancies at high resolutions.

## Background & Motivation

Existing feedforward 3DGS methods appear to achieve strong NVS metrics at standard resolution, yet suffer from a critical overlooked problem:

**Degenerate 3D scenes**: Reconstructions are effectively discrete, color-biased point clouds rather than continuous surfaces, exposing severe artifacts (holes, color bias, surface discontinuities) under close-up or off-axis viewpoints.

**Underutilized anisotropy**: Learnable Gaussians struggle to disentangle geometry and texture through gradient supervision alone, causing Gaussians to degenerate toward near-spherical shapes.

**Metric failure**: Standard PSNR/SSIM/LPIPS at native resolution cannot capture geometric inaccuracies, masking true reconstruction quality.

The authors observe that **directly training 2DGS is more challenging than 3DGS** — the planar nature of 2D Gaussians causes small geometric perturbations to produce large deviations in rendered output, a problem exacerbated under limited supervision.

## Method

### Overall Architecture

SurfSplat employs a dual-path encoder (monocular Depth Anything V2 + multi-view cross-attention), fuses the resulting features through a U-Net to regress intermediate attributes (depth, scale multipliers, appearance components), and finally converts them to standard 2DGS attributes via surface continuity priors and forced alpha blending.

### Key Designs

1. **Surface Continuity Prior**

   Core assumption: visible geometry in real scenes consists primarily of smooth, continuous surfaces, and spatially adjacent surface elements correspond to adjacent pixels in the image. This constrains Gaussian rotation and scale:

   **Rotation**: For the 3D position $\mathbf{p}_0$ of pixel $(h,w)$ and its neighborhood, two tangent vectors $\mathbf{t}_1, \mathbf{t}_2$ are computed via Sobel filtering; the local normal $\mathbf{n}$ is obtained by their cross product, and the rotation matrix is derived via the Rodrigues formula:

   $\mathbf{R} = \mathbf{I} + [\mathbf{v}]_\times + \frac{1-c}{\|\mathbf{v}\|^2}[\mathbf{v}]_\times^2$

   where $\mathbf{v} = \mathbf{n}_0 \times \mathbf{n}$ and $c = \mathbf{n}_0^\top \mathbf{n}$.

   **Scale**: Coarse estimates $\bar{\sigma}_u, \bar{\sigma}_v$ are derived from image-space neighborhood distances; the network predicts scale multipliers $\hat{\sigma}_u, \hat{\sigma}_v$ in the range $[1/3, 3]$, yielding final scales $\sigma_u = \bar{\sigma}_u \hat{\sigma}_u$. The depth-axis scale $\sigma_w$ of 2DGS is fixed to zero.

   Gaussian attributes are thus derived from predicted 3D positions rather than regressed independently, ensuring spatial consistency.

2. **Forced Alpha Blending**

   Under the surface continuity prior, the model tends to learn high-opacity Gaussians, causing occluded Gaussians to contribute negligibly to alpha blending and preventing learning of 3D structure. The solution:

   - Clip opacity with an upper bound $\tau_{\text{opa}} < 1$ (set to 0.6), ensuring all Gaussians participate in rendering.
   - Initialize RGB colors to the DC component of the spherical harmonic basis.
   - Apply opacity-normalized compensation to the rendered output: $C = C/\alpha$ when $\alpha \geq \tau_\alpha$.

3. **HRRC: High-Resolution Rendering Consistency Metric**

   The reconstructed scene is rendered at $2\times$ and $4\times$ resolution and compared against bicubic-upsampled ground truth:

   $\text{HRRC}_{\text{metric}} = \text{metric}(\hat{I}^{HR}, \hat{I}^{GT\uparrow})$

   This effectively exposes sparsity holes, degenerate Gaussian shapes, and surface discontinuities, distinguishing models that truly recover 3D geometry from those that merely memorize sparse viewpoints.

### Loss & Training

$$L_{\text{gs}} = \sum_{m=1}^M \left(\text{MSE}(I_{\text{render}}^m, I_{\text{gt}}^m) + \lambda \cdot \text{LPIPS}(I_{\text{render}}^m, I_{\text{gt}}^m)\right)$$

$\lambda = 0.05$; training is conducted at $256\times256$ resolution. The Depth Anything V2 backbone uses a learning rate of $2 \times 10^{-6}$; all other layers use $2 \times 10^{-4}$.

## Key Experimental Results

### Main Results

| Method | RE10K 256 PSNR↑ | RE10K 512 PSNR↑ | RE10K 1024 PSNR↑ | RE10K Avg PSNR↑ |
|--------|-----------------|-----------------|------------------|-----------------|
| DepthSplat | **27.504** | 20.031 | 16.385 | 21.307 |
| MVSplat | 26.359 | 20.408 | 17.966 | 21.578 |
| Ours-L | 27.537 | **26.331** | **24.897** | **26.255** |

### Ablation Study

| Configuration | Standard PSNR | HRRC 2× PSNR | HRRC 4× PSNR | Notes |
|---------------|--------------|--------------|--------------|-------|
| 3DGS baseline (DepthSplat) | 27.504 | 20.031 | 16.385 | Severe HRRC degradation |
| 2DGS (SurfSplat-L) | 27.537 | 26.331 | 24.897 | HRRC remains stable |
| w/o surface prior | Degraded | Largely degraded | Largely degraded | Surface discontinuity |
| w/o forced blending | Degraded | Degraded | Degraded | Color bias |

### Key Findings

- **HRRC reveals the truth**: DepthSplat achieves the best standard-resolution score (27.5) but collapses to 16.4 at 1024 resolution; SurfSplat drops only from 27.5 to 24.9, demonstrating genuine 3D structure recovery.
- MVSplat and TranSplat similarly degrade substantially under HRRC (around 18 at 1024), indicating that surface degeneration is pervasive among feedforward 3DGS methods.
- Cross-dataset evaluation on DL3DV and ScanNet confirms generalization.
- pixelSplat (multiple Gaussians per pixel) performs comparatively better on HRRC (24.9), as redundant Gaussians partially compensate for surface holes.

## Highlights & Insights

1. **High value in problem identification**: The work exposes a widely overlooked surface degeneration issue in feedforward 3DGS; the HRRC metric deserves broader adoption.
2. **Geometry-driven attribute prediction**: Deriving rotation from positions via Sobel filtering and the Rodrigues formula is an elegant and physically intuitive design.
3. **First successful application of 2DGS in feedforward settings**: Demonstrates that 2DGS (planar primitives) is more suitable than 3DGS (ellipsoids) for feedforward reconstruction, offering stronger anisotropy and geometric precision.
4. **Clever design of forced blending**: Constraining the opacity upper bound resolves local optima and preserves multi-layer expressiveness.

## Limitations & Future Work

- At standard resolution, the method only marginally outperforms DepthSplat; advantages are primarily reflected in HRRC.
- The single-Gaussian-per-pixel design may provide insufficient coverage for complex scenes.
- The HRRC metric relies on bicubic-upsampled ground truth rather than true high-resolution references.
- Evaluation is limited to static scenes; extension to dynamic scenes remains future work.

## Related Work & Insights

- Huang et al. (2024) introduced 2DGS for per-scene optimization; SurfSplat is the first to bring it into a feedforward framework.
- The depth interaction design of DepthSplat (Xu et al., 2024b) is partially inherited, but SurfSplat replaces independent attribute regression with surface priors.
- The design philosophy of the HRRC metric generalizes to evaluation of other 3D generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The surface continuity prior and HRRC metric are original contributions, though individual components are combinations of established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multi-resolution HRRC, multiple backbone variants, and cross-dataset evaluation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, visual comparisons are compelling, and mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ The HRRC metric and the exposure of surface degeneration constitute important contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] GaussianImage++: Boosted Image Representation and Compression with 2D Gaussian Splatting](../../AAAI2026/3d_vision/gaussianimage_boosted_image_representation_and_compression_with_2d_gaussian_spla.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
