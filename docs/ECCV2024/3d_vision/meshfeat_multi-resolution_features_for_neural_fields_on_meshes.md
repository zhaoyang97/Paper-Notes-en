---
title: >-
  [Paper Note] MeshFeat: Multi-Resolution Features for Neural Fields on Meshes
description: >-
  [ECCV 2024][3D Vision][Parametric Feature Encoding] This paper proposes MeshFeat, a parametric multi-resolution feature encoding method for neural fields on meshes. It constructs multi-resolution feature representations using mesh simplification algorithms, achieving a 13x inference speedup while maintaining reconstruction quality.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Parametric Feature Encoding"
  - "Multi-Resolution Mesh"
  - "Neural Fields"
  - "Mesh Simplification"
  - "Texture Reconstruction"
date: 2026-05-08
content_hash: cd8e099b1eccdf93
---

# MeshFeat: Multi-Resolution Features for Neural Fields on Meshes

**Conference**: ECCV 2024  
**arXiv**: [2407.13592](https://arxiv.org/abs/2407.13592)  
**Code**: [https://maharajamihir.github.io/MeshFeat/](https://maharajamihir.github.io/MeshFeat/)  
**Area**: 3D Vision  
**Keywords**: Parametric Feature Encoding, Multi-Resolution Mesh, Neural Fields, Mesh Simplification, Texture Reconstruction

## TL;DR

This paper proposes MeshFeat, a parametric multi-resolution feature encoding method for neural fields on meshes. It constructs multi-resolution feature representations using mesh simplification algorithms, achieving a 13x inference speedup while maintaining reconstruction quality.

## Background & Motivation

**Background**: Neural Fields are widely applied in 3D scene representation. Parametric feature grids in Euclidean space (such as the hash grids of Instant NGP) significantly accelerate inference speed by decoupling spatial information from MLP decoding.

**Limitations of Prior Work**: 
   - Feature encoding methods on meshes (e.g., Texture Fields, Intrinsic Neural Fields) rely on frequency encoding (positional encoding/Fourier features/LBO eigenfunctions).
   - Frequency encoding stores all information within the MLP weights, requiring a large MLP (6 layers $\times$ 128 width), leading to slow inference.
   - Intrinsic Neural Fields (INF) require computing Laplace-Beltrami operator (LBO) eigenfunctions, which can take hours for preprocessing, and store a massive number of eigenfunctions per vertex, resulting in large model sizes.
   - Directly leveraging Euclidean embeddings for encoding in Euclidean space causes bleeding artifacts when there is a significant discrepancy between Euclidean distance and geodesic distance.

**Key Challenge**: While parametric feature grids in Euclidean space have achieved real-time inference, mesh-based neural fields still remain in the paradigm of frequency encoding combined with large MLPs, leading to low inference efficiency.

**Goal**: Translate the concept of parametric multi-resolution feature grids from Euclidean space to meshes, thereby achieving small MLPs and fast inference.

**Key Insight**: Utilize the mesh's vertex topology as the storage location for features (instead of regular voxel grids) and construct multi-resolution representations via mesh simplification algorithms (rather than octrees or hash tables).

**Core Idea**: Mesh simplification acts as a multi-resolution feature grid on the mesh, enabling small MLPs and fast inference.

## Method

### Overall Architecture

MeshFeat pipeline: Input mesh $\to$ Generate a multi-resolution mesh sequence using a mesh simplification algorithm $\to$ Store learnable feature matrices at each resolution $\to$ Map features of all resolutions back to the original resolution using vertex mappings $\to$ Perform barycentric coordinate interpolation to obtain the feature at any point $\to$ Decode into target values using a small MLP (2 layers $\times$ 32 width).

### Key Designs

1. **Multi-resolution Construction via Mesh Simplification**:

    - Employs Garland-Heckbert's Quadric Error Metrics (QEM) simplification algorithm.
    - Simplification ratio sequence $r^{(i)} \in \{1, 0.1, 0.05, 0.01\}$, corresponding to keeping 100%/10%/5%/1% of the vertices.
    - Generates a mesh sequence $((V^{(i)}, F^{(i)}))_i$.
    - Records the mapping $m^{(i)}: V \to V^{(i)}$ of which new vertex each original vertex is collapsed to during the simplification process.
    - When $r^{(1)}=1$, $m^{(1)}$ is an identity mapping (the original mesh serves as the finest resolution).
    - **Key Advantage**: Circumvents the computation of geometric mappings between resolutions, reducing computational overhead.

2. **Multi-resolution Feature Aggregation**:

    - Each resolution $i$ maintains a learnable feature matrix $Z^{(i)} \in \mathbb{R}^{|V^{(i)}| \times d}$, where $d$ is the feature dimension.
    - **Key Difference from Euclidean Feature Grids**: Instead of performing independent interpolation at each resolution and then concatenating, features from all resolutions are first mapped ("pulled back") to the finest resolution via mapping $m^{(i)}$ and then summed.
    - Reason: A point at a certain resolution on the mesh might not exist on meshes of other resolutions (and its Euclidean embedding might not even lie on those meshes).
    - Aggregated feature of vertex $v$ on the original mesh:

    $\phi_v = \sum_i Z^{(i)}(m^{(i)}(v))$

3. **Barycentric Coordinate Feature Interpolation**:

    - For any point $x$ on the mesh, compute its barycentric coordinates $p(x) = [\lambda_1, \lambda_2, \lambda_3]^\top$ within its containing triangle.
    - Final feature encoding:

    $\phi(x) = \sum_i \lambda_i \phi_{v_i} = [\phi_{v_1}, \phi_{v_2}, \phi_{v_3}] \, p(x)$

    - Requires only a single barycentric coordinate interpolation (instead of interpolating at each resolution), making it computationally efficient.

4. **Mesh Laplacian Feature Regularization**:

    - Problem: Under sparse training data, some vertices may never be intersected by any ray-mesh intersection, leaving their features without supervision signals.
    - Solution: Smoothness regularization based on the mesh Laplacian.

    $\mathcal{L}_{reg} = \sum_{i,j} |(\hat{L}\Phi)_{i,j}|$

      where $\hat{L} = L / \|L\|_2$ is the spectral-norm normalized Laplacian, and $\Phi$ is the aggregated feature matrix of all vertices.
    - L1 norm is used (instead of L2) to allow sparse large variations (preserving texture edges) while penalizing large fluctuations between neighboring features.
    - Although regularization is only computed at the highest resolution, it affects all resolutions because it acts on the aggregated features.
    - Implemented using the robust Laplacian by Sharp & Crane.

### Loss & Training

- Texture Reconstruction: L1 color loss + Laplacian regularization ($\lambda_{reg} = 1.5 \times 10^{-6}$).
- BRDF Estimation: L1 + gamma correction (to handle the imbalance between bright and dark regions in HDR images).
- Dual Learning Rate Strategy: MLP parameters $lr_\theta = 2 \times 10^{-4}$, feature encoding $lr_Z = 5 \times 10^{-3}$ (higher learning rate for features).
- Feature dimension $d=4$, feature initialization $\sigma = 5 \times 10^{-4}$.
- MLP: 2 hidden layers $\times$ 32 width + ReLU + sigmoid output + L2 regularization ($10^{-5}$).
- batch size = 8000, texture training for 1000 epochs, BRDF training for 500 epochs.

## Key Experimental Results

### Texture Reconstruction — human mesh ($|V|$=129k)

| Method | PSNR↑ | DSSIM↓×100 | LPIPS↓×100 | Params | Speedup |
|------|-------|-----------|-----------|--------|----------|
| NeuTex | 27.32 | 0.549 | 0.954 | 793k | 1.0× |
| TF+RFF | 32.10 | 0.232 | 0.423 | **331k** | 1.96× |
| INF | 32.46 | 0.215 | 0.390 | 133k (eigenfunctions 130k) | 3.06× |
| **Ours (d=4)** | **32.51** | **0.202** | **0.400** | 604k | **13.49×** |
| Non-neural | 32.01 | 0.225 | 0.432 | 391k | 28.32× |

### BRDF Reconstruction — DiLiGenT-MV (5-object average)

| Method | PSNR↑ | DSSIM↓×100 | LPIPS↓×100 | Speedup |
|------|-------|-----------|-----------|----------|
| TF+RFF | 42.13 | 0.672 | 1.50 | 1.00× |
| INF | 42.21 | 0.666 | 1.53 | 1.08× |
| **Ours** | **42.17** | **0.670** | 1.60 | **7.58×** |

### Key Findings

- **Speed**: MeshFeat achieves a 13.49× speedup (relative to NeuTex) because the MLP is scaled down from 6 layers $\times$ 128 width to 2 layers $\times$ 32 width.
- **Quality**: PSNR/DSSIM/LPIPS are on par with or slightly better than the best frequency encoding method (INF).
- **Parameter Count Trade-off**: The parameter count is slightly higher than TF+RFF (due to storing feature matrices) but far smaller than INF (which needs to store a large number of LBO eigenfunctions).
- **Multi-resolution vs. Single-resolution**: The multi-resolution scheme achieves higher quality with fewer parameters; single-resolution cannot compensate even if the feature dimension is increased.
- **Critical Role of Regularization**: Without Laplacian regularization, PSNR drops by 1.26dB (human dataset), leading to obvious visual artifacts (e.g., in ear and shoe regions).
- **Native Support for Deforming Meshes**: Features are bound to vertices, so when the mesh topology remains unchanged, deformation does not require any additional computation.

## Highlights & Insights

- **Elegant Conceptual Transfer**: Mapping the concept of "regular grid $\to$ multi-resolution" in Euclidean space to "vertex topology $\to$ mesh simplification" on meshes is conceptually clear and elegant.
- **"Pull Back" instead of "Resolution-by-Resolution Interpolation"**: Since there is no natural point correspondence between different mesh resolutions, pulling features back to the finest resolution before interpolation solves the core challenge of multi-resolution feature interpolation on meshes.
- **L1 Laplacian Regularization**: Better balances smoothness and edge preservation compared to L2 regularization, making it highly suitable for textured scenes.
- **Native Support for Deforming Meshes**: This is a beneficial byproduct of binding the parametric encoding to vertices, which holds great value for animation applications.

## Limitations & Future Work

1. The multi-resolution ratios $r^{(i)}$ are fixed and not adaptively adjusted based on local mesh complexity.
2. The feature dimension $d$ is small (4), which may be insufficient for highly detailed textures (though increasing $d$ can cause overfitting on highly detailed meshes).
3. Only two tasks, texture reconstruction and BRDF estimation, were verified; other neural field applications like SDF or occupancy functions were not explored.
4. The mesh simplification algorithm (QEM) might lose critical topology during extreme simplification, which could fail on small objects or thin structures.
5. Although the preprocessing time (mesh simplification + mapping computation) is much faster than computing LBO eigenfunctions (seconds vs. hours), it still introduces overhead.

## Related Work & Insights

- **Instant NGP** (Euclidean Space): Multi-resolution features on hash grids $\to$ MeshFeat transfers this concept to the mesh surface.
- **Intrinsic Neural Fields**: LBO eigenfunctions = frequency encoding on meshes $\to$ MeshFeat = parametric encoding on meshes. The two are complementary encoding paradigms.
- **Texture Fields**: The earliest neural field method on meshes, but extrinsic $\to$ MeshFeat's vertex-bound features are intrinsic.
- **Insight**: When data possesses a natural hierarchical structure (like mesh topology), utilizing domain-specific simplification/hierarchical algorithms is more efficient than generic hash/grid-based methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The idea of utilizing mesh simplification for multi-resolution features is simple and elegant)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive experiments covering texture, BRDF, deforming meshes, and ablation studies)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Concise formulas, clear tables/figures, and well-justified design motivations for each module)
- **Value**: ⭐⭐⭐⭐ (Provides a practical solution for efficient neural field representations on meshes, with great potential for the animation field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)
- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[ECCV 2024\] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)

</div>

<!-- RELATED:END -->
