---
title: >-
  [Paper Note] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes AAA-Gaussians, which systematically addresses aliasing, projection distortion, and pop-in artifacts in 3DGS within a unified framework via three techniques: an adaptive 3D smoothing filter, a view-space perspective-correct bounding scheme, and frustum-based 3D culling. The method achieves state-of-the-art artifact-free real-time rendering under both in-distribution and out-of-distribution viewpoints.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - anti-aliasing
  - artifact elimination
  - 3D evaluation
  - real-time rendering
date: 2026-05-08
content_hash: 3af012ddef1e6c72
---

# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering

**Conference**: ICCV 2025
**arXiv**: N/A (CVF Open Access)
**Code**: [https://github.com/DerThomy/AAA-Gaussians](https://github.com/DerThomy/AAA-Gaussians)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, anti-aliasing, artifact elimination, 3D evaluation, real-time rendering

## TL;DR

This paper proposes AAA-Gaussians, which systematically addresses aliasing, projection distortion, and pop-in artifacts in 3DGS within a unified framework via three techniques: an adaptive 3D smoothing filter, a view-space perspective-correct bounding scheme, and frustum-based 3D culling. The method achieves state-of-the-art artifact-free real-time rendering under both in-distribution and out-of-distribution viewpoints.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) has attracted broad interest due to its real-time rendering capability and high-quality image synthesis. However, vanilla 3DGS approximates 3D Gaussians as 2D splats for rendering, a simplification that introduces various artifacts under non-standard camera configurations such as large FOV and VR scenarios.

2. **Limitations of Prior Work**: 3DGS exhibits four categories of artifacts: (1) deformation artifacts caused by the affine projection approximation, particularly at image borders and under large FOV; (2) aliasing artifacts arising from changes in viewing distance; (3) pop-in artifacts when Gaussians extend beyond the view frustum; and (4) popping artifacts due to the simplification of global depth sorting. Although prior works address these issues individually, they either incur significant performance penalties or resolve only a subset of the problems.

3. **Key Challenge**: 2D splat evaluation is fast but inaccurate, while 3D ray tracing is accurate but slow and requires additional acceleration structures. Recent hybrid 2D/3D methods evaluate Gaussians in 3D but still rely on screen-space computations (e.g., 2D bounding boxes, 2D anti-aliasing filters), making them susceptible to artifacts under out-of-distribution viewpoints.

4. **Goal**: To enforce the 3D nature of 3D Gaussians at every stage of the rendering pipeline, fundamentally eliminating all categories of artifacts while retaining the efficiency advantages of rasterization.

5. **Key Insight**: The authors observe that the fundamental issue with existing methods is that certain pipeline stages still operate in 2D screen space. Processing entirely in 3D space—covering bounding, culling, anti-aliasing, and evaluation—can unify the resolution of all artifacts.

6. **Core Idea**: An adaptive 3D smoothing filter is proposed to replace the 2D Mip filter; view-space bounding replaces screen-space bounding; frustum-based culling replaces 2D tile culling—together constituting the first fully artifact-free 3DGS rendering framework.

## Method

### Overall Architecture

AAA-Gaussians builds upon a hierarchical rasterizer, retaining the per-ray sorting of StopThePop and the 3D Gaussian evaluation of Hahlbohm et al., while replacing the bounding, culling, depth evaluation, contribution estimation, and anti-aliasing components with fully 3D implementations. Densification is performed using an MCMC scheme. The input is a set of training images; the output is a high-quality, view-consistent 3DGS scene representation.

### Key Designs

1. **Adaptive 3D Anti-Aliasing Filter**:

    - **Function**: Eliminates aliasing artifacts caused by under- and over-sampling due to varying camera distance.
    - **Mechanism**: The existing 3D smoothing filter of Yu et al. causes excessive transparency of Gaussians under the 3D evaluation mode, because its amplitude normalization factor $\sqrt{|\Sigma|/|\hat{\Sigma}|}$ scales with volumetric change, whereas 3D evaluation takes the maximum contribution point along the ray rather than integrating. This paper proposes adjusting the amplitude solely based on the area change in the plane perpendicular to the viewing direction $d$: $\hat{G}_\perp(x) = \sqrt{|\Sigma_\perp|/|\hat{\Sigma}_\perp|} \cdot \exp(-\frac{1}{2}(x-\mu)^\top \hat{\Sigma}^{-1}(x-\mu))$, where $\Sigma_\perp$ is the 2×2 covariance projected onto the subspace orthogonal to $d$. The maximum sampling frequency at training time $\hat{v}_{train}$ is also incorporated to prevent excessive shrinkage of Gaussians when the camera approaches.
    - **Design Motivation**: Naive amplitude adjustment by volumetric change over-suppresses the amplitude of highly anisotropic Gaussians, rendering them transparent. Adjusting by perpendicular area change better reflects the physical semantics of rasterization-based evaluation.

2. **View-Space Perspective-Correct Bounding**:

    - **Function**: Eliminates pop-in artifacts when Gaussians extend behind the image plane.
    - **Mechanism**: Hahlbohm et al. fit planes in screen space to bound Gaussians, discarding any Gaussian whose z-range exceeds the near or far plane, which causes pop-in artifacts. This paper instead fits bounding planes in view space using angles $\theta$ and $\phi$, obtaining the view-space angular extent by solving $\theta_{1,2} = \tan^{-1}(\frac{s_{1,3} \pm \sqrt{s_{1,3}^2 - s_{1,1}s_{3,3}}}{s_{3,3}})$, clamping to $[-\pi/2+\epsilon, \pi/2-\epsilon]$, and mapping back to screen space. Gaussians are discarded only when the camera is located inside them.
    - **Design Motivation**: Screen-space bounding fails entirely when a Gaussian straddles the image plane. View-space bounding via angular representation naturally circumvents this issue.

3. **Frustum-Based 3D Culling**:

    - **Function**: Accelerates rendering and reduces sorting overhead while eliminating artifacts caused by incorrect culling.
    - **Mechanism**: The 2D tile culling is elevated to 3D frustum culling. For each tile, a 3D frustum $\mathcal{F}$ defined by four screen-space planes is constructed; the point within the frustum closest to the origin in the Gaussian's normalized space (i.e., the maximum contribution point) is then found. If $\rho(x)^2 > \tau_\rho$, the Gaussian is culled. Computation is optimized by projecting only onto the x/y planes closest to the Gaussian center and their corresponding edges, reducing the naive evaluation from 4 faces + 4 edges to 2 faces + 3 edges.
    - **Design Motivation**: Axis-aligned bounding boxes (AABBs) provide loose bounds for non-axis-aligned ellipsoids, leading to substantial redundant computation. 3D frustum culling precisely determines whether a Gaussian contributes to a specific tile, significantly reducing sorting and rendering overhead.

### Loss & Training

- MCMC densification scheme is used, maintaining the same number of Gaussians as competing methods.
- 3D filter kernel size $k = 0.3$, consistent with Mip-Splatting.
- Hierarchical k-buffer sorting is employed for correct per-pixel blending order.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Artifacts |
|--------|------|-------|-------|--------|------|
| Mip-NeRF 360 | 3DGS | 27.44 | 0.814 | 0.215 | Distortion+aliasing+pop-in |
| Mip-NeRF 360 | MCMC | 28.03 | 0.836 | 0.187 | Distortion+aliasing+pop-in |
| Mip-NeRF 360 | **Ours** | **27.84** | **0.836** | **0.188** | **None** |
| Deep Blending | 3DGS | 29.51 | 0.902 | 0.237 | Distortion+aliasing+pop-in |
| Deep Blending | **Ours** | **30.49** | **0.913** | **0.222** | **None** |

**Large-FOV Evaluation (3× FOV enlargement)**:

| Dataset | MCMC | Taming-3DGS | **Ours** |
|--------|------|-------------|----------|
| Mip-NeRF 360 PSNR | 23.35 | 23.30 | **27.84** |
| T&T PSNR | 14.37 | 11.55 | **23.58** |
| Deep Blending PSNR | 18.32 | 20.33 | **30.49** |

### Ablation Study

| Configuration | PSNR↑ (MipNeRF360) | Artifacts |
|------|-------------------|------|
| Full model | 27.84 | None |
| w/o hier. sort | 27.90 (+0.06) | Popping present |
| w/o AA | 27.81 (−0.03) | Aliasing present |
| w/o 3D eval | 27.87 (+0.03) | Distortion present |

### Key Findings

- Under in-distribution viewpoints, removing individual components can paradoxically improve metrics (by exploiting per-view inconsistencies), yet the gap is dramatic under out-of-distribution conditions.
- Under large-FOV settings, PSNR of methods such as MCMC and Taming-3DGS drops by more than 10 dB, whereas the proposed method is nearly unaffected.
- Frustum-based culling halves the rendering time (14.40 ms → 7.72 ms), constituting a critical performance optimization.
- The additional overhead of 3D evaluation is negligible and may even be faster (7.64 ms vs. 7.72 ms).

## Highlights & Insights

- **Unified framework for all artifacts**: For the first time, aliasing, projection distortion, pop-in, and sorting artifacts are simultaneously eliminated within a single method, rather than addressing each problem in isolation as in prior work. This has significant implications for applications requiring large FOV, such as VR/AR.
- **Theoretical insight on the adaptive 3D filter**: The paper identifies that scaling amplitude by volume change is incorrect under 3D evaluation mode, and proposes instead scaling by perpendicular area change—a concise and physically grounded insight.
- **Deceptiveness of in-distribution metrics**: The paper demonstrates that removing certain components can inflate standard benchmark scores through per-view inconsistency exploitation, underscoring the importance of out-of-distribution evaluation.

## Limitations & Future Work

- Hierarchical sorting introduces approximately 2× performance overhead (7.72 ms vs. 4.11 ms without sorting).
- Blending artifacts arising from the non-overlapping assumption cannot be resolved without expensive volumetric integration.
- Current experiments are conducted primarily on static scenes; generalization to dynamic scenes requires further investigation.

## Related Work & Insights

- **vs. Mip-Splatting**: Mip-Splatting's 2D screen-space Mip filter cannot be combined with 3D evaluation; the proposed 3D adaptive filter operates entirely in 3D space.
- **vs. StopThePop**: StopThePop addresses popping caused by sorting but still suffers from projection and aliasing artifacts; the present work completes the full solution on top of it.
- **vs. Hybrid Transparency**: Hybrid Transparency still exhibits pop-in at image borders and has insufficient anti-aliasing capability.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified framework to resolve all 3DGS artifacts
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ In/out-of-distribution + multi-resolution + detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem analysis and rigorous mathematical derivation
- Value: ⭐⭐⭐⭐⭐ Significant practical impact for 3DGS deployment in VR/AR and related applications

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)

<!-- RELATED:END -->
