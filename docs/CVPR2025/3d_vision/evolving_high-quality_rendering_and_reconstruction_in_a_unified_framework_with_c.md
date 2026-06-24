---
title: >-
  [Paper Note] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes CarGS. By identifying that the source of contribution conflicts between rendering and reconstruction tasks in Gaussian primitives lies in covariance, the authors design Lite-Geo, a lightweight residual structure to adaptively decouple the geometric contribution of the two tasks. Additionally, they introduce a normal + SDF double-guided densification strategy, achieving both SOTA rendering quality and reconstruc…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Surface Reconstruction"
  - "Novel View Synthesis"
  - "Contribution-Adaptive Regularization"
  - "Unified Framework"
date: 2026-05-08
content_hash: 31c7de620fbf4c07
---

# Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization

**Conference**: CVPR 2025  
**arXiv**: [2503.00881](https://arxiv.org/abs/2503.00881)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Surface Reconstruction, Novel View Synthesis, Contribution-Adaptive Regularization, Unified Framework

## TL;DR

This paper proposes CarGS. By identifying that the source of contribution conflicts between rendering and reconstruction tasks in Gaussian primitives lies in covariance, the authors design Lite-Geo, a lightweight residual structure to adaptively decouple the geometric contribution of the two tasks. Additionally, they introduce a normal + SDF double-guided densification strategy, achieving both SOTA rendering quality and reconstruction accuracy in a unified model, with a storage cost of only 9% of dual-model approaches.

## Background & Motivation

**Background**: Achieving high-quality rendering and accurate 3D surface reconstruction simultaneously from multi-view images is a core challenge in computer vision and computer graphics. 3D Gaussian Splatting (3DGS) has received widespread attention for its high-quality rendering and fast inference, but ensuring accurate geometric reconstruction remains difficult due to the unstructured nature of Gaussian point clouds.

**Limitations of Prior Work**: Existing methods primarily fall into two categories: (1) Primitive-based methods (such as SuGaR, 2DGS, PGSR) apply geometric regularization to Gaussian primitives, but there is an inherent conflict between rendering and reconstruction goals—adding geometric regularization degrades rendering quality; (2) Dual-model methods (such as GSDF) utilize 3DGS for rendering and SDF for reconstruction separately, which incurs enormous computational and storage overheads (1.7GB storage, 3 hours of training, non-real-time inference).

**Key Challenge**: Rendering requires some "floater" Gaussians far from the surface to capture rich texture, while reconstruction requires Gaussians to strictly adhere to the surface. The two tasks have opposing requirements for the contribution weights of Gaussians—rendering tends to increase the weights of Gaussians far from the surface, whereas reconstruction tends to suppress them. Through experiments, the authors found that after discarding 50% of Gaussians with high CD values (far from the surface), the reconstruction quality F1 score remained almost unchanged, but the rendering PSNR dropped sharply, intuitively proving the existence of the contribution conflict.

**Goal**: Adaptively adjust the contribution of each Gaussian primitive to both rendering and reconstruction tasks within a unified model, maintaining rendering quality while achieving high-quality reconstruction.

**Key Insight**: Through experimental analysis, the authors discovered that covariance (rather than opacity) is the key attribute causing the contribution conflict. Therefore, an additional set of covariance parameters can be predicted for geometric reconstruction, decoupled from the rendering covariance, to achieve adaptive contribution.

**Core Idea**: Learn a geometry-specific covariance offset for each Gaussian primitive via a lightweight residual MLP, predicting the residual on top of the rendering covariance, thereby decoupling the contribution requirements of the two tasks within a unified model.

## Method

### Overall Architecture

CarGS is built upon the anchor-based framework of Scaffold-GS. Given multi-view images, the initial point cloud and anchors are obtained via SfM. Each anchor predicts the attributes (color, position offset, covariance, opacity) of $k$ neural Gaussians through an MLP. The core innovations are the introduction of the Lite-Geo residual module to predict additional covariance for geometry, and a normal + SDF guided geometric densification strategy. Finally, the mesh is extracted from depth maps via TSDF fusion.

### Key Designs

1. **Lite-Geo Residual Structure (Contribution-Adaptive Regularization)**:

    - **Function**: Predicts geometry-reconstruction covariance parameters decoupled from rendering, while inheriting implicit depth information from the rendering task.
    - **Mechanism**: On top of the covariance (scale and rotation) output by the rendering MLP, a Geo-MLP is added to predict the residual offset. Geometry covariance $y = M_\Sigma^{rgb}(\mathbf{f}, \theta; \phi_1) + \Delta y$, where $\Delta y = M_\Sigma^{geo}(\mathbf{f}, \theta; \phi_2)$, and $\phi_2 = \lambda \phi_1$ is initialized by scaling the parameters of the rendering MLP. This residual design allows the geometric covariance to inherit the implicit depth information learned during the rendering optimization process, avoiding overfitting issues of a separate geometric MLP.
    - **Design Motivation**: Directly using an independent MLP to predict geometric covariance leads to severe overfitting, because geometric regularization (planar constraints + cross-view constraints) only provides continuity and consistency constraints, lacking accurate depth supervision. The rendering task implicitly contains depth information through photometric loss, and the residual structure can effectively exploit this information.

2. **Geometry-Guided Densification Strategy**:

    - **Function**: Uses dual clues of normals and SDF to guide the growth of Gaussian primitives in high-frequency detail regions.
    - **Mechanism**: In addition to traditional gradient-based densification, a geometry-guided term is introduced. The densification criterion is computed as $\epsilon_g = \nabla_g + \omega_g(\zeta(s) \cdot (\omega_n n \text{ if } \mu(s) < \theta \text{ else } 1))$ using the SDF value $s = D(x) - Z(x)$ and the normal difference $n = N_d(x) - N(x)$, where $\zeta(s)$ is a Gaussian function, making points closer to the zero-level set more inclined to generate new Gaussians. Regions with large normal differences (high-frequency detail regions) are also prioritized for densification.
    - **Design Motivation**: Existing geometry-guided densification approaches (such as GSDF) only use SDF values, failing to adequately capture high-frequency detail regions. Introducing normal cues increases Gaussian density in visually complex regions near the surface, improving rendering sharpness and reconstruction accuracy.

3. **In-depth Analysis of Attribute Contribution**:

    - **Function**: Identify the core attributes causing the rendering-reconstruction conflict.
    - **Mechanism**: Through gradient detachment experiments, gradients of opacity and covariance are respectively blocked in geometric losses. Experiments revealed that blocking opacity gradients has minimal impact on reconstruction quality (F1 remains at 0.64 vs 0.65), whereas blocking covariance gradients leads to a dramatic drop in reconstruction quality (F1 drops from 0.65 to 0.46). This proves that covariance is the primary driver of the geometric regularization's influence on the rendering-reconstruction contribution.
    - **Design Motivation**: Once the root cause of the conflict is precisely located, a minimally invasive solution can be designed—just adding a geometric branch for covariance, while keeping other attributes shared.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_c + \alpha \mathcal{L}_{plane} + \beta \mathcal{L}_{cross}$, where $\mathcal{L}_c$ is the RGB reconstruction loss (L1 + D-SSIM), $\mathcal{L}_{plane}$ is the planar constraint (aligning depth-deduced normals with rendered normals), and $\mathcal{L}_{cross}$ is the cross-view consistency constraint (forward-backward projection error). $\alpha=0.01$, $\beta=0.2$. In training, the model is pre-trained with rendering-only objectives for several epochs, and then jointly optimized with the Lite-Geo module and geometric losses.

## Key Experimental Results

### Main Results

| Method | TnT F1↑ | TnT PSNR↑ | Mip360 PSNR↑ | Storage | Training Time | FPS |
|------|---------|-----------|-------------|------|---------|-----|
| NeuS | 0.38 | 23.71 | - | 1.4GB | >24h | <1 |
| Neuralangelo | 0.50 | 25.43 | - | 4.2GB | >24h | <1 |
| GSDF | 0.46 | 26.05 | - | 1.7GB | 3h | <1 |
| PGSR | 0.60 | 26.12 | 27.43 | 0.42GB | 1.2h | 103 |
| **CarGS** | **0.65** | **26.41** | **27.68** | **0.16GB** | 1.2h | 90 |

CarGS achieves an F1 score of 0.65 on the TnT dataset (outperforming PGSR's 0.60), while also reaching the highest PSNR of 26.41. Its storage is only 38% of PGSR and 9% of GSDF.

### Ablation Study

| Configuration | F1↑ | PSNR↑ | Description |
|------|-----|-------|------|
| (a) Direct addition of geometric losses | 0.38 | 25.72 | Base method has poor reconstruction quality |
| (b) + Geo-MLP | 0.44 | 25.75 | Reconstruction improved but suffers from overfitting |
| (c) + Lite-Geo residual | 0.62 | 26.13 | Residual structure significantly boosts both tasks |
| (d) + Geo-Densify | **0.65** | **26.41** | Geometric densification yields further gains |

### Key Findings

- The Lite-Geo residual structure is the most critical design, with F1 increasing from 0.38 to 0.62 (+63%) and PSNR from 25.72 to 26.13 from configuration (a) to (c).
- The geometry-guided densification strategy brings an additional boost on top of Lite-Geo, from F1 0.62 to 0.65 and PSNR 26.13 to 26.41.
- Covariance, rather than opacity, is the root cause of the contribution conflict: detaching covariance gradients causes F1 to plunge from 0.65 to 0.46.
- On the Mip-NeRF 360 dataset, CarGS achieves the highest rendering quality (PSNR 27.68) among all GS methods without sacrificing reconstruction accuracy.

## Highlights & Insights

- **The analysis identifying covariance as the root conflict is highly refined**: Locating the problem precisely with a simple gradient detachment experiment avoids over-engineering the entire framework. This analytical approach of identifying key factors through controlled variable experiments is highly worth studying.
- **Double ingenuity of the residual structure**: (1) It inherits implicit depth information from rendering covariance to avoid overfitting in the geometry branch; (2) initialization via scaled parameter $\lambda$ ensures that initial predictions are close to rendering covariance, yielding more stable convergence.
- **Impressive storage efficiency**: 0.16GB vs GSDF's 1.7GB, achieving a 10x compression while outperforming it. This benefits from the unified model design and the compactness of the anchor-based framework.

## Limitations & Future Work

- Normal-guided densification relies on the quality of normals derived from depth maps, which can be unreliable at depth discontinuities or occlusion boundaries.
- Reconstruction quality in outdoor large-scale scenes (such as the complete TnT Courthouse scene) still has room for improvement (F1 = 0.17).
- The residual weight $\lambda$ of Lite-Geo and the geometry-guided densification hyperparameters $\omega_g, \omega_n$ require manual tuning.
- Currently only static scenes are supported; extending contribution adaptation to dynamic scenes is a more challenging task.

## Related Work & Insights

- **vs PGSR**: PGSR is a typical primitive-regularization method that directly restrains shared Gaussian attributes with geometric losses, leading to rendering-reconstruction conflicts. CarGS resolves this issue by decoupling covariance with Lite-Geo, boosting F1 from 0.60 to 0.65, alongside rendering performance.
- **vs GSDF**: GSDF employs a dual-model (3DGS+SDF) design to handle both tasks separately, which performs well but at a high computational and storage cost. CarGS achieves superior results in a unified model, reducing training time by 60% and storage by 91%, while supporting real-time rendering.
- **vs 2DGS**: 2DGS improves geometric alignment via 2D Gaussian primitives, but also suffers from degraded rendering quality. The adaptive contribution strategy of CarGS provides a more general solution.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of contribution adaptation is novel, and focusing on covariance decoupling is precise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons on TnT and Mip-NeRF360, with well-designed ablations.
- Writing Quality: ⭐⭐⭐⭐ The analysis logic is clear, and figures/tables are well designed.
- Value: ⭐⭐⭐⭐ Establishes a strong baseline for unified rendering + reconstruction; the contribution-decoupling concept is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction](../../CVPR2026/3d_vision/plug-and-play_pde_optimization_for_3d_gaussian_splatting_toward_high-quality_ren.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](../../ECCV2024/3d_vision/citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)

</div>

<!-- RELATED:END -->
