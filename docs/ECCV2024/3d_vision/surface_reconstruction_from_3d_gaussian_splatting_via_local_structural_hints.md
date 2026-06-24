---
title: >-
  [Paper Note] Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] To address the issue of poor surface reconstruction quality in 3DGS, this paper proposes utilizing monocular normal/depth priors to enhance the geometric organization of Gaussian primitives, constructing local signed distance fields via Moving Least Squares (MLS), and jointly learning a neural implicit network for regularization, significantly improving the surface reconstruction precision of 3DGS.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Surface Reconstruction"
  - "Moving Least Squares"
  - "Neural Implicit Field"
  - "Monocular Geometric Prior"
date: 2026-05-08
content_hash: 03de16d090280c0d
---

# Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints

**Conference**: ECCV 2024  
**Paper Link**: [OpenReview](https://openreview.net/forum?id=cwrF0p1Oqh) | [DOI](https://doi.org/10.1007/978-3-031-72627-9_25)  
**Code**: [GitHub](https://github.com/QianyiWu/gsrec)  
**Area**: 3D Vision / Surface Reconstruction  
**Keywords**: 3D Gaussian Splatting, Surface Reconstruction, Moving Least Squares, Neural Implicit Field, Monocular Geometric Prior  

## TL;DR
To address the issue of poor surface reconstruction quality in 3DGS, this paper proposes utilizing monocular normal/depth priors to enhance the geometric organization of Gaussian primitives, constructing local signed distance fields via Moving Least Squares (MLS), and jointly learning a neural implicit network for regularization, significantly improving the surface reconstruction precision of 3DGS.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has gained significant attention due to its high efficiency and quality in novel view synthesis, becoming one of the most popular 3D representations after NeRF. However, 3DGS represents scenes using millions of unorganized Gaussian primitives optimized primarily for rendering quality, lacking the modeling of underlying geometric structures.

**Limitations of Prior Work**: The core challenge for 3DGS is to extract high-quality surface meshes from a large number of unordered Gaussian primitives. Due to the lack of geometric constraints during optimization, their spatial distribution is often disorganized—many Gaussians deviate from the true surface, overlap with each other, and exhibit inconsistent normals. Existing methods like SuGaR attempt to align Gaussians with the surface through regularization, but they yield limited effectiveness, especially in geometrically complex regions.

**Key Challenge**: The original design of 3DGS aims to maximize rendering quality rather than geometric accuracy. Rendering only requires presenting correct colors from the camera perspective, whereas surface reconstruction demands that Gaussian primitives accurately align with the true surface. There is an inherent contradiction between these two objectives—multiple discrete Gaussian configurations can produce the same rendering result, but only one reflects the true geometry.

**Goal**: (1) How to introduce effective geometric constraints during 3DGS training to align Gaussian primitives to the actual surface; (2) How to accurately extract continuous surface meshes from discrete Gaussian primitives.

**Key Insight**: Taking the perspective of "local structural hints," the authors propose a two-level geometric guidance scheme: the first level utilizes existing monocular depth/normal estimation to provide global geometric priors for Gaussian primitives, while the second level constructs local MLS signed distance fields to offer fine-grained surface alignment signals. These two levels of guidance establish coarse-to-fine geometric constraints.

**Core Idea**: Enhance the organization of Gaussian primitives through monocular geometric priors, and then construct local signed distance fields via MLS with joint learning of a neural implicit network, enabling precise alignment of 3DGS Gaussian primitives to the true surface.

## Method

### Overall Architecture
The overall pipeline of GSrec consists of three stages. First, basic 3DGS training is augmented with geometric supervision signals from monocular normal and depth estimation models (such as Omnidata) to guide the means and covariance matrices (position and shape) of the Gaussian primitives to roughly align with the surface. Second, a signed distance field is constructed in each local region from surrounding Gaussian primitives using the MLS method. Finally, a lightweight neural implicit network is introduced to fit the MLS field, where the network output and the MLS field mutually regularize each other to drive the Gaussian primitives into precise alignment. The final surface is extracted from the optimized field using Marching Cubes.

### Key Designs

1. **Monocular Geometry Guidance**:

    - **Function**: Utilizes pre-trained monocular depth and normal estimation models to provide initial geometric alignment signals for Gaussian primitives.
    - **Mechanism**: For each training view, a model like Omnidata is used to estimate depth and normal maps. Depth and normals are rendered from the current 3DGS and used to compute losses against the monocular estimates: depth loss $L_{depth} = \|D_{render} - D_{mono}\|$ and normal loss $L_{normal} = 1 - \cos(N_{render}, N_{mono})$. These losses backpropagate to the parameters of the Gaussian primitives, adjusting their positions (means) and shapes (covariance matrices). Specifically, the normal loss guides the shortest axis of the Gaussian ellipsoids to align with the surface normals, encouraging the "pancake-like" Gaussians to be flush with the surface.
    - **Design Motivation**: Pure rendering loss cannot constrain the geometric configurations of Gaussian primitives (where a single rendering outcome corresponds to infinite Gaussian distributions). Although monocular estimates may lack absolute precision, they provide sufficient directional guidance. This prior converts an unconstrained optimization problem into a biased search.

2. **MLS-based Signed Distance Field**:

    - **Function**: Constructs a continuous signed distance field in local regions using surrounding Gaussian primitives to compensate for their discrete nature.
    - **Mechanism**: For a query point $x$, a set of neighboring Gaussian primitives $\{g_i\}$ is collected. Each Gaussian primitive contributes a signed distance value (based on the dot product of the displacement from $x$ to the Gaussian center and the Gaussian's normal) and a weight (based on a spatial distance decay function). MLS computes a weighted average of these contributions to obtain the signed distance value at $x$: $f_{MLS}(x) = \frac{\sum_i w_i(x) \cdot d_i(x)}{\sum_i w_i(x)}$, where $d_i(x)$ is the signed distance of $x$ relative to the $i$-th Gaussian, and $w_i(x)$ is the weight. The zero isosurface of the MLS field represents the reconstructed surface. Crucially, the normal information of the Gaussian primitives (derived from monocular prior guidance) provides the signed distance with directionality.
    - **Design Motivation**: A single Gaussian primitive is merely a discrete "point" and cannot represent a continuous surface. MLS glues discrete Gaussians into a continuous field through local weighted averaging, while preserving detailed features due to its local nature. Compared to global fitting (such as SDF networks), the locality of MLS avoids interference of distant Gaussians on local geometry.

3. **Joint Neural Implicit Learning**:

    - **Function**: Trains a lightweight MLP to fit the MLS field, allowing mutual regularization between the two.
    - **Mechanism**: A small MLP network $f_{NN}(x)$ is introduced to predict the SDF value of a query point $x$. The training objective is to map the outputs of $f_{NN}$ close to $f_{MLS}$, while backpropagating learning signals of $f_{NN}$ to the parameters of the Gaussian primitives. Specifically, the joint loss is $L_{joint} = \|f_{NN}(x) - f_{MLS}(x)\|^2$. Thanks to the inherent smoothness and generalization capabilities of neural networks, they can correct local noise in the MLS field caused by non-uniform Gaussian distributions while passing this smoothing signal back to the Gaussian primitives, driving them towards more plausible geometric configurations. This "student-teacher" style of bidirectional regularization outperforms using either MLS or the neural network alone.
    - **Design Motivation**: The MLS field directly depends on the positions and normals of Gaussian primitives, making it susceptible to individual outlier Gaussians. Neural networks filter out this noise through implicit regularization (such as weight decay and frequency biases inherent in the network architecture). In turn, the local accuracy of the MLS field guides the neural network to learn more precise geometry. Joint optimization achieves a synergistic effect that leverages the strengths of both.

### Loss & Training
The total loss is defined as $L = L_{render} + \lambda_d L_{depth} + \lambda_n L_{normal} + \lambda_j L_{joint} + \lambda_e L_{eikonal}$, where $L_{render}$ is the standard 3DGS rendering loss, and $L_{eikonal}$ is the Eikonal regularization (constraining the SDF gradient magnitude to 1). Training adopts a two-stage strategy: first, optimization is conducted using rendering and monocular geometric losses for some time to achieve rough alignment of Gaussian primitives, after which joint refinement with MLS and the neural implicit network is incorporated.

## Key Experimental Results

### Main Results

| Dataset | Metric | GSrec | SuGaR | 2DGS | NeuS |
|--------|------|-------|-------|------|------|
| DTU | Chamfer Dist↓ | 0.83 | 1.47 | 0.95 | 0.91 |
| Replica | F-Score↑ | 88.2 | 79.5 | 84.1 | 86.3 |
| ScanNet | F-Score↑ | 52.6 | 41.3 | 47.8 | 49.2 |
| Tanks&Temples | F-Score↑ | 45.8 | 37.2 | 42.1 | 43.5 |

### Ablation Study

| Configuration | DTU Chamfer | Description |
|------|------------|------|
| Full GSrec | 0.83 | Full model |
| w/o monocular priors | 1.21 | Gaussians lack guidance on normals and positions |
| w/o MLS field | 1.05 | Uses only monocular priors, lacking local geometric refinement |
| w/o neural implicit | 0.95 | Uses only MLS, lacking smoothing regularization |
| w/o joint learning | 0.98 | MLS and neural network are trained separately |

### Key Findings
- Monocular priors contribute the most; removing them increases the Chamfer distance by 46%, showing that geometric priors are the cornerstone of high-quality reconstruction.
- MLS and the neural implicit network both contribute significantly, but joint learning performs better than their simple combination, validating the value of bidirectional regularization.
- On indoor scenes such as Replica and ScanNet, the advantage of GSrec is particularly pronounced because these scenes contain numerous planar structures, which are highly suitable for local MLS fitting.
- Compared to pure implicit methods (such as NeuS), GSrec maintains comparable geometric accuracy while offering rendering speeds that are faster by an order of magnitude.

## Highlights & Insights
- **Well-designed coarse-to-fine geometric guidance strategy**: From monocular priors, to MLS, and finally to the neural implicit representation, the three-level guidance progressively refines precision. This hierarchical design provides valuable context for 3D reconstruction.
- **Natural yet effective application of MLS on Gaussian primitives**: Since each Gaussian primitive already carries country position and normal information, using MLS to merge them into a continuous field is a seamless design choice.
- **Positioning of the neural network as a "regularizer" rather than the "main driver"**: Instead of directly predicting the SDF with a neural network, the network assists and regularizes the MLS field. This design avoids the training instability common in pure implicit methods.

## Limitations & Future Work
- Performance depends on the accuracy of monocular depth/normal estimation models; thus, performance may degrade in scenes where these models fail (e.g., highly transparent or reflective surfaces).
- Constructing the MLS field requires querying neighboring Gaussian primitives, which could lead to holes in areas where Gaussians are sparse.
- The two-stage training strategy increases both training complexity and time costs.
- Introducing the neural implicit network adds extra parameters and computational overhead; although the network is small, its impact remains in large-scale scenes.

## Related Work & Insights
- **vs SuGaR**: SuGaR aligns Gaussians with surfaces through regularization, but the regularization strength is globally uniform, which is inflexible for geometrically complex regions. GSrec's MLS offers adaptive local geometric constraints.
- **vs 2D Gaussian Splatting**: 2DGS degenerates 3D Gaussians into 2D disks to force surface alignment, limiting the representation degrees of freedom of the Gaussians. GSrec maintains the flexibility of 3D Gaussians while guiding the alignment via an external field.
- **vs NeuS/NeuralAngelo**: Pure implicit methods yield high geometric accuracy but slow rendering. GSrec combines the rendering efficiency of 3DGS with the geometric accuracy of implicit methods.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of joint MLS and neural implicit regularization for 3DGS is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluations on multiple datasets covering indoor and outdoor scenes with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-described method motivation and technical details.
- Value: ⭐⭐⭐⭐⭐ Significantly advances practical 3DGS surface reconstruction, Open-Source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)

</div>

<!-- RELATED:END -->
