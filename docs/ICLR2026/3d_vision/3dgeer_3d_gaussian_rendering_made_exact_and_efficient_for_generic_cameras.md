---
title: >-
  [Paper Note] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras
description: >-
  [ICLR 2026][3D Vision][3D Gaussian Splatting] This paper proposes 3DGEER, a framework that derives a closed-form solution for integrating Gaussian density along rays, designs a Particle Bounding Frustum (PBF) for accurate and efficient ray–particle association, and introduces Bipolar Equal-Angle Projection (BEAP) to unify wide-FoV camera representations. 3DGEER achieves geometrically exact and real-time efficient 3D Gaussian rendering under arbitrary camera models, outperforming existing methods comprehensively on both fisheye and pinhole datasets.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Ray Tracing
  - Fisheye Camera
  - Wide-FoV Rendering
  - Real-Time Rendering
date: 2026-05-08
content_hash: 4ccc053e015626eb
---

# 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras

**Conference**: ICLR 2026
**arXiv**: [2505.24053](https://arxiv.org/abs/2505.24053)
**Code**: [https://zixunh.github.io/3d-geer](https://zixunh.github.io/3d-geer)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Ray Tracing, Fisheye Camera, Wide-FoV Rendering, Real-Time Rendering

## TL;DR
This paper proposes 3DGEER, a framework that derives a closed-form solution for integrating Gaussian density along rays, designs a Particle Bounding Frustum (PBF) for accurate and efficient ray–particle association, and introduces Bipolar Equal-Angle Projection (BEAP) to unify wide-FoV camera representations. 3DGEER achieves geometrically exact and real-time efficient 3D Gaussian rendering under arbitrary camera models, outperforming existing methods comprehensively on both fisheye and pinhole datasets.

## Background & Motivation

**State of the Field**: 3D Gaussian Splatting (3DGS) achieves efficient rendering by projecting 3D Gaussians onto 2D Gaussians via EWA splatting, striking a favorable balance between quality and efficiency in narrow-FoV scenarios.

**Limitations of Prior Work**: EWA splatting relies on a first-order Taylor expansion for linear approximation; under wide-FoV settings (e.g., fisheye cameras with 180° FoV), severe nonlinear distortion causes significant degradation in reconstruction quality. Existing fisheye extensions (FisheyeGS, GS++) remain constrained by projection approximations. Ray tracing methods (EVER, 3DGRT), while free of projection error, rely on BVH traversal, resulting in low frame rates. The hybrid method 3DGUT still introduces errors and grid-line artifacts through Unscented Transform approximations in the association stage.

**Root Cause**: The true projection of a 3D Gaussian under a nonlinear camera model is not a symmetric 2D Gaussian; any method relying on linear projection geometry inevitably introduces approximation errors. Meanwhile, accurate ray tracing methods cannot achieve real-time efficiency due to the algorithmic complexity and poor parallelizability of BVH.

**Paper Goals**: (a) How to obtain an exact closed-form solution for integrating Gaussian density along a ray? (b) How to perform accurate and efficient ray–particle association without BVH? (c) How to unify the representation of arbitrary-FoV cameras and improve reconstruction quality?

**Starting Point**: Starting from first principles, the paper maps each anisotropic Gaussian to a canonical coordinate system where it becomes isotropic, derives a closed-form integral, solves the association problem at the frustum level rather than in screen space, and designs equal-angle sampling to replace conventional projection.

**Core Idea**: By leveraging canonical coordinate transformation for exact closed-form transmittance, frustum-level PBF association, and BEAP unified projection, 3DGEER is the first method to simultaneously achieve geometric exactness and real-time efficiency for Gaussian rendering under arbitrary camera models.

## Method

### Overall Architecture
The input consists of multi-view images captured with arbitrary camera models (pinhole, fisheye, etc.), and the output is rendered images from novel viewpoints. The overall pipeline comprises three core components: (1) exact ray-integration rendering based on canonical coordinate transformation; (2) efficient ray–particle association via Particle Bounding Frustum (PBF); and (3) unified image representation via Bipolar Equal-Angle Projection (BEAP). During rendering, PBF–CSF association first identifies the Gaussians corresponding to each ray, and then the Gaussian density is exactly integrated along the ray to obtain the color.

### Key Designs

1. **Canonical Coordinate Transformation + Closed-Form Transmittance**:

    - **Function**: Maps anisotropic 3D Gaussians to a canonical isotropic space and derives a closed-form transmittance.
    - **Mechanism**: For each Gaussian, the transformation $\mathbf{x} = RS\mathbf{u} + \boldsymbol{\mu}$ is defined to reduce it to the standard isotropic form $\mathcal{G}_{\mathbf{I},\mathbf{0}}(\mathbf{u}) = \frac{1}{\rho}\exp(-\frac{1}{2}\|\mathbf{u}\|^2)$. The transmittance of a ray in canonical space is $T = \sigma \exp(-\frac{1}{2}D^2_{\mu,\Sigma})$, where $D^2 = \frac{\|\mathbf{o}_u \times \mathbf{d}_u\|^2}{\|\mathbf{d}_u\|^2}$ is the squared perpendicular Mahalanobis distance from the ray to the Gaussian center.
    - **Design Motivation**: This result shows that transmittance depends only on the distance between the ray and the Gaussian in canonical space, eliminating any projection approximation. The expression is equivalent to the previously used "maximum response" heuristic, but the paper reveals its mathematical foundation for projection exactness from first principles.

2. **Particle Bounding Frustum (PBF)**:

    - **Function**: Reformulates ray–particle association as intersection detection between camera sub-frustums (CSF) and particle bounding frustums (PBF).
    - **Mechanism**: Two spherical angles $\theta = \arctan(d_{c,x}/d_{c,z})$ and $\phi = \arctan(d_{c,y}/d_{c,z})$ are defined in camera space, and PBF is bounded by four planes. After transforming the constraints to canonical space, the angular bounds are obtained in closed form by solving the quadratic equation $\mathcal{T}_{22}c^2 - 2\mathcal{T}_{02}c + \mathcal{T}_{00} = 0$.
    - **Design Motivation**: This replaces BVH traversal and screen-space approximations such as EWA/UT, solving ray–particle association directly and exactly in 3D frustum space. PBF produces very tight bounds, with each tile associating on average only ~475 Gaussians — 3–5× fewer than EWA/UT — achieving a 2.5–5× speedup in the association stage.

3. **Bipolar Equal-Angle Projection (BEAP)**:

    - **Function**: Proposes a unified image representation space that uniformly samples rays for color supervision.
    - **Mechanism**: Rays are uniformly sampled in the $(\theta, \phi)$ spherical angle space and mapped to discrete image space via a linear transformation.
    - **Design Motivation**: (a) Effectively represents arbitrary-FoV cameras without loss of field of view; (b) image tiles and CSFs share the same parameterization, improving PBF association efficiency and GPU parallelism; (c) compared to the non-uniform sampling of pinhole projection and the center oversampling of equidistant projection, BEAP achieves a more uniform ray distribution in 3D space.

### Loss & Training
Standard photometric loss is applied to supervise training in BEAP space. After 30k iterations, the results are projected back to the original image space for full-FoV evaluation. The entire forward and backward passes are ensured to be numerically stable through closed-form derivations, requiring no post-processing such as degenerate Gaussian filtering.

## Key Experimental Results

### Main Results

**ScanNet++ Dataset (180° Fisheye):**

| Method | Train FoV | Full PSNR↑ | Full SSIM↑ | Full LPIPS↓ | Center PSNR | Edge PSNR |
|--------|-----------|-----------|-----------|------------|-------------|-----------|
| FisheyeGS | Full | 27.81 | 0.946 | 0.139 | 32.44 | 23.28 |
| EVER | Full | 29.47 | 0.924 | 0.167 | 29.93 | 28.72 |
| 3DGUT | Full | 30.64 | 0.944 | 0.150 | 31.87 | 28.84 |
| **3DGEER** | Full | **31.50** | **0.953** | **0.126** | **32.64** | **28.94** |

**MipNeRF360 Dataset (Narrow-FoV Pinhole):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|--------|-------|-------|--------|------|
| 3DGS | 27.21 | 0.815 | 0.214 | 343 |
| EVER | 27.51 | 0.825 | 0.233 | 36 |
| 3DGRT | 27.20 | 0.818 | 0.248 | 52 |
| 3DGUT | 27.26 | 0.810 | 0.218 | 265 |
| **3DGEER** | **27.76** | **0.821** | **0.210** | **327** |

### Ablation Study

**BEAP Comparison (ScanNet++):**

| Training Space | Full PSNR↑ | Full SSIM↑ | Full LPIPS↓ | Center PSNR | Edge PSNR |
|----------------|-----------|-----------|------------|-------------|-----------|
| Perspective (Central) | 29.84 | 0.943 | 0.131 | 32.21 | 26.23 |
| Perspective (Full) | 21.11 | 0.853 | 0.300 | 21.32 | 20.46 |
| Equidistant (Full) | 31.05 | 0.948 | 0.135 | 32.21 | 28.56 |
| **BEAP (Full)** | **31.50** | **0.953** | **0.126** | **32.64** | **28.94** |

**Transmittance Function Ablation (ScanNet++):**

| Transmittance Method | Full PSNR↑ | LPIPS↓ | # Gaussians (k) |
|----------------------|-----------|--------|-----------------|
| 3DGS Splats | 22.86 | 0.177 | 1396.4 |
| FisheyeGS Splats | 27.90 | 0.141 | 920.6 |
| **Exact Integration (Ours)** | **31.50** | **0.126** | **591.8** |

### Key Findings
- PBF produces extremely tight bounds, associating on average only ~475 Gaussians per tile — 3–5× fewer than EWA/UT — with a 2.5–5× speedup in the association stage.
- Brute-force scaling up the number of Gaussians (FisheyeGS extended to 3.6–5M) cannot compensate for projection approximation errors: PSNR saturates at 29.3, whereas 3DGEER achieves 32.1 with only 550–700K Gaussians.
- Experiments replacing only the transmittance function clearly demonstrate that inconsistency between exact association and approximate transmittance leads to clipping errors and artifacts.
- In cross-camera generalization experiments (training on pinhole, testing on fisheye), 3DGEER performs best, indicating that geometric exactness yields stronger generalization.

## Highlights & Insights
- **Closed-Form Solution in Canonical Space**: Revealing the mathematical essence of the "maximum response" heuristic as projection exactness is an elegant unification of theory and practice. Derivation from first principles ensures numerical stability, eliminating the need for techniques such as degenerate Gaussian filtering.
- **Frustum-Level Association**: Moving beyond screen-space thinking and directly solving the association problem in 3D frustum space represents a methodological breakthrough. By binding the quadratic equation to the true 3D covariance, the approach guarantees both exactness and efficiency.
- **Design Philosophy of BEAP**: At a fixed resolution, uniformity of information distribution matters more than a larger FoV — full-FoV perspective projection can even underperform central-region-only rendering. This finding has broad implications for any rendering task involving wide-FoV cameras.

## Limitations & Future Work
- Validation is limited to static scenes; extension to dynamic scenes and temporal data remains unexplored.
- Whether the uniform sampling strategy of BEAP can be further improved with content-aware weighted sampling has not been investigated.
- Experiments are conducted primarily on indoor (ScanNet++) and medium-scale scenes; performance on large-scale outdoor scenes is not validated.
- The numerical precision of canonical coordinate transformation for Gaussians with extreme shapes (e.g., highly elongated) may require further analysis.

## Related Work & Insights
- **vs. 3DGS**: 3DGS uses EWA splatting for 2D projection approximation — fast but error-prone at wide FoV; 3DGEER uses ray integration for exact computation, achieving comparable speed with comprehensively superior accuracy.
- **vs. EVER/3DGRT**: These ray tracing methods are geometrically exact but slow due to BVH dependency (36–68 FPS); 3DGEER replaces BVH with PBF to achieve 327 FPS, a more than 5× speedup.
- **vs. 3DGUT**: 3DGUT uses UT approximation for association — fast but prone to grid-line artifacts; 3DGEER performs exact association in frustum space, avoiding this issue entirely.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First method to simultaneously achieve geometric exactness and real-time efficiency under arbitrary camera models; all three components rest on strong theoretical foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 datasets, multiple camera models, cross-camera generalization experiments, and detailed ablation and runtime analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, narrative logic is clear, and figures and tables are highly informative.
- Value: ⭐⭐⭐⭐⭐ Removes a critical obstacle for deploying 3DGS in wide-FoV applications such as autonomous driving and robotics.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](radiogs_radiometric_gaussian_surfels.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[CVPR 2026\] RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](../../CVPR2026/3d_vision/rap_fast_feedforward_rendering-free_attribute-guided_primitive_importance_score_.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](../../ICCV2025/3d_vision/aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)
- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](../../NeurIPS2025/3d_vision/lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)

<!-- RELATED:END -->
