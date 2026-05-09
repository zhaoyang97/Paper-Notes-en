---
title: >-
  [Paper Note] GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][surface reconstruction] This paper proposes GeoSVR, an explicit surface reconstruction framework based on sparse voxels. By introducing voxel-uncertainty depth constraints and sparse voxel surface regularization, GeoSVR comprehensively outperforms existing 3DGS- and SDF-based methods in geometric accuracy, detail preservation, and reconstruction completeness.
tags:
  - NeurIPS 2025
  - 3D Vision
  - surface reconstruction
  - sparse voxels
  - depth constraint
  - voxel uncertainty
  - multi-view geometry
date: 2026-05-08
content_hash: 7c9ce4e3e127712f
---

# GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2509.18090](https://arxiv.org/abs/2509.18090)
**Code**: [Fictionarry/GeoSVR](https://github.com/Fictionarry/GeoSVR)
**Area**: 3D Vision
**Keywords**: surface reconstruction, sparse voxels, depth constraint, voxel uncertainty, multi-view geometry

## TL;DR

This paper proposes GeoSVR, an explicit surface reconstruction framework based on sparse voxels. By introducing voxel-uncertainty depth constraints and sparse voxel surface regularization, GeoSVR comprehensively outperforms existing 3DGS- and SDF-based methods in geometric accuracy, detail preservation, and reconstruction completeness.

## Background & Motivation

**Initialization bottleneck of 3DGS**: Existing 3D Gaussian Splatting-based surface reconstruction methods rely heavily on sparse point clouds from SfM for initialization. Inaccuracies and uncovered regions in these point clouds inherently hinder geometric optimization — an intrinsic limitation.

**Geometric ambiguity of Gaussian primitives**: Gaussian primitives lack well-defined boundaries, presenting trade-offs between representational clarity and computational precision, leading to high geometric ambiguity.

**Underutilized potential of geometric foundation models**: Monocular depth estimation models such as DepthAnything have advanced rapidly, yet their full potential cannot be realized due to the spatial incompleteness of 3DGS representations.

**Overlooked potential of sparse voxels**: Sparse voxel methods such as SVRaster have demonstrated efficient scene representation, but their potential for accurate surface reconstruction remains largely unexplored.

**Challenges in using monocular depth priors**: How to maximally leverage "good but imperfect" external depth constraints in high-precision surface reconstruction — while preventing estimation errors from degrading already-reliable geometry — remains unsolved. Existing methods can only adopt overly conservative strategies.

**Locality problem of small voxels**: The extreme locality of sparse voxels (sharing gradients only with nearest neighbors) is unfavorable for forming globally consistent surfaces.

## Method

### Overall Architecture

GeoSVR is built upon SVRaster, representing scenes with Octree-organized sparse voxels. Each voxel stores spherical harmonic color coefficients and $2\times2\times2$ corner densities for trilinear interpolation. Constant initialization (without SfM point clouds) ensures complete scene coverage. DepthAnythingV2 provides monocular depth priors, which are incorporated via voxel-uncertainty depth constraints and sparse voxel surface regularization. Meshes are finally extracted via TSDF.

### Key Design 1: Voxel-Uncertainty Depth Constraint

- **Function**: Adaptively determines the degree of reliance on external monocular depth constraints per pixel — strengthening constraints in uncertain regions while reducing dependence where geometry is already reliable.
- **Mechanism**: Octree level is used as a proxy for geometric uncertainty. Low-level voxels indicate sparse texture or insufficient view coverage (high uncertainty); high-level voxels, having undergone subdivision, indicate more reliable geometry. A level map $\mathbf{L}$ is rendered, and uncertainty weights $\mathbf{W}_{\text{unc}}$ are computed adaptively via per-view statistics to perform pixel-wise weighting of the patch-wise depth loss.
- **Design Motivation**: Direct inverse depth loss or sparse point constraints yield negligible improvement (validated by ablation), while full reliance on monocular depth degrades existing geometric quality due to estimation errors. A mechanism that adaptively regulates constraint strength based on reconstruction confidence is needed; voxel hierarchy naturally encodes this information.

### Key Design 2: Voxel Dropout

- **Function**: During multi-view geometric regularization (homography patch warping + NCC loss), voxels are randomly dropped at a ratio sampled from $[\gamma, 1]$, and only the remaining subset represents the scene.
- **Mechanism**: After dropping a portion of voxels, the remaining voxels must account for geometric consistency over a larger region, compelling each small voxel to satisfy global constraints rather than focusing solely on its immediate vicinity, thereby breaking erroneous local geometric organization.
- **Design Motivation**: The extreme locality of sparse voxels limits the effectiveness of planar multi-view geometric constraints — each voxel connects only a small number of neighboring corner points, restricting the propagation range of planar constraints and resulting in redundant erroneous structures. Dropout forces an expansion of each voxel's geometric influence range.

### Key Design 3: Surface Rectification + Scaling Penalty

- **Function**: Surface Rectification corrects the discrepancy between trilinear voxel density fields and rendering weights; Scaling Penalty suppresses large, low-precision voxels that occupy excessive sampling distances.
- **Mechanism**: Surface Rectification detects density transitions as rays enter and exit voxels to identify "surface voxels" (low entry-point density, high exit-point density, crossing threshold $T_\alpha=0.5$), encouraging low inlet density and high outlet density to form sharp surface boundaries. Scaling Penalty applies a density penalty to large voxels via $\log_2(\Delta t / \min(\mathbf{v}_s))$.
- **Design Motivation**: Trilinear interpolation causes a density increase in one voxel to affect its neighbors, shifting the peak rendering weight to lateral regions rather than the true highest-density location, producing depth bias. Large voxels offer low geometric modeling precision and should be suppressed from participating in surface formation.

## Loss & Training

The total loss is: $\mathcal{L} = \mathcal{L}_{\text{photo}} + 0.1\mathcal{L}_{\text{D-unc}} + 0.01\mathcal{L}_{\text{NCC}} + 10^{-5}\mathcal{R}_{\text{rec}} + 10^{-6}\mathcal{R}_{\text{sp}}$. Training runs for 20k iterations with the Adam optimizer; learning rates for density/SH0/other parameters are 0.05/0.01/0.00025, respectively. Voxel dropout ratio $\gamma$ is 0.5 for DTU and 0.3 for TnT. Octree pruning is performed every 2000 steps. All experiments are conducted on an RTX 3090 Ti.

## Key Experimental Results

### DTU Dataset (Chamfer Distance ↓)

| Method | Type | Mean CD ↓ | Training Time |
|--------|------|-----------|---------------|
| NeuS | Implicit | 0.84 | >12h |
| Neuralangelo | Implicit | 0.61 | >128h |
| GeoNeuS | Implicit | 0.51 | >12h |
| 2DGS | Explicit | 0.80 | 0.2h |
| GOF | Explicit | 0.74 | 1h |
| PGSR | Explicit | 0.52 | 0.5h |
| MonoGSDF | Explicit | 0.65 | hrs |
| GS2Mesh | Explicit | 0.68 | 0.3h |
| **GeoSVR** | **Explicit** | **0.47** | **0.8h** |

GeoSVR achieves a Chamfer Distance of 0.47, outperforming all methods, including the implicit SOTA GeoNeuS (0.51) and the explicit SOTA PGSR (0.52).

### Tanks and Temples Dataset (F1 Score ↑)

| Method | Barn | Caterpillar | Courthouse | Truck | Mean F1 ↑ | Time |
|--------|------|-------------|------------|-------|-----------|------|
| Neuralangelo | 0.70 | 0.36 | 0.28 | 0.48 | 0.50 | >128h |
| PGSR | 0.66 | 0.44 | 0.20 | 0.66 | 0.52 | 45m |
| MonoGSDF | 0.56 | 0.38 | 0.29 | 0.62 | 0.47 | 3h |
| **GeoSVR** | **0.68** | **0.49** | **0.34** | **0.66** | **0.56** | **68m** |

On real-world TnT scenes, GeoSVR achieves an F1 score of 0.56, surpassing Neuralangelo (0.50) and PGSR (0.52), with a particularly notable advantage on challenging scenes such as Courthouse (0.34 vs. second-best 0.29).

### Ablation Study (TnT, F1 ↑)

| Configuration | F1 Score |
|---------------|----------|
| SVRaster (baseline) | 0.397 |
| + Patch-wise Depth | 0.449 |
| + Multi-view Reg. | 0.538 |
| + Voxel Dropout | 0.546 |
| + Surface Rectif. + Scaling Penalty | 0.552 |
| + Voxel-Uncertainty Depth (full) | **0.560** |

Each module contributes positively; patch-wise depth yields the largest gain (+0.052), and uncertainty weighting provides further improvement (+0.008) on top of an already high-quality baseline.

## Highlights & Insights

- **No SfM point cloud initialization required**: Constant initialization of sparse voxels eliminates the strong dependence of 3DGS on sparse point clouds, fundamentally resolving incomplete coverage.
- **Voxel-level uncertainty-adaptive constraints**: Octree level is elegantly leveraged as a proxy for geometric confidence, enabling selective utilization of monocular depth priors.
- **Novel Voxel Dropout design**: Analogous to neural network Dropout, randomly dropping voxels expands the effective range of geometric constraints — a simple yet effective mechanism.
- **Favorable efficiency–quality trade-off**: SOTA accuracy is achieved with only 0.8h of training on DTU, far faster than implicit methods.

## Limitations & Future Work

- Performance on textureless regions and scenes with varying illumination remains insufficient; the authors acknowledge this as a direction for future work.
- Rendering quality (SSIM/LPIPS) on Mip-NeRF 360 falls short of GOF and PGSR, reflecting a trade-off between appearance quality and geometric accuracy.
- The method depends on DepthAnythingV2 as an external depth prior; the quality of the depth estimation model affects final reconstruction quality.
- Training time (0.8h), while faster than implicit methods, is slower than 2DGS (0.2h) and SVRaster (0.1h).

## Related Work & Insights

- **Implicit surface reconstruction**: NeuS, VolSDF, Neuralangelo, and others combine SDF with volume rendering, achieving high quality at the cost of extremely long training times.
- **3DGS-based surface reconstruction**: 2DGS flattens Gaussians into 2D surfels; PGSR introduces multi-view geometric constraints; GOF constructs an opacity field for mesh extraction — all are constrained by SfM initialization.
- **External depth priors**: MonoSDF, VCR-GauS, GS2Mesh, and others leverage depth/normal foundation models, but lack confidence estimation, leading to conservative usage strategies.
- **Sparse voxel representations**: SVRaster combines non-uniform sparse voxels with rasterization and serves as the foundation for this work. GeoSVR is the first to extend this representation to accurate surface reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ — Sparse voxels for surface reconstruction is a new direction; voxel uncertainty and Voxel Dropout are cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on three datasets (DTU/TnT/Mip-360) with complete ablations and both quantitative and qualitative results.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear, method derivation is natural, and figures and tables are professional.
- Value: ⭐⭐⭐⭐ — Offers a new solution for surface reconstruction beyond 3DGS, with convincing SOTA results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](../../ICLR2026/3d_vision/urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[NeurIPS 2025\] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
