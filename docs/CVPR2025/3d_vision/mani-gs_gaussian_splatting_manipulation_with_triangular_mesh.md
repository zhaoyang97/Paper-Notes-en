---
title: >-
  [Paper Note] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Mani-GS proposes a method for manipulating 3D Gaussian Splatting based on triangular meshes. By defining a local coordinate system on each triangle to bind Gaussians, the position, rotation, and scale of Gaussians adaptively adjust when the mesh deforms. This enables various manipulation types such as large deformation, local editing, and soft-body simulation, while maintaining high-quality rendering and showing high tolerance for…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Mesh Manipulation"
  - "Local Triangle Coordinate System"
  - "Editable Rendering"
  - "Soft-body Simulation"
date: 2026-05-08
content_hash: 4bde63256c3406ce
---

# Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh

**Conference**: CVPR 2025  
**arXiv**: [2405.17811](https://arxiv.org/abs/2405.17811)  
**Code**: Not yet released (see project page)  
**Area**: 3D Vision / 3DGS Editing  
**Keywords**: 3D Gaussian Splatting, Mesh Manipulation, Local Triangle Coordinate System, Editable Rendering, Soft-body Simulation

## TL;DR
Mani-GS proposes a method for manipulating 3D Gaussian Splatting based on triangular meshes. By defining a local coordinate system on each triangle to bind Gaussians, the position, rotation, and scale of Gaussians adaptively adjust when the mesh deforms. This enables various manipulation types such as large deformation, local editing, and soft-body simulation, while maintaining high-quality rendering and showing high tolerance for mesh inaccuracy.

## Background & Motivation

1. **Background**: 3DGS has rapidly become a star in the 3D representation field due to its high-fidelity real-time rendering capabilities. However, as an explicit point cloud representation, it currently lacks effective manipulation methods—users lack intuitive tools to stretch, bend, or locally edit 3DGS scenes.
2. **Limitations of Prior Work**: (1) NeRF-based editing methods (NeRF-Editing, NeuMesh) suffer from slow training/inference and limited rendering quality; (2) SuGaR strictly binds Gaussians to triangle faces (flat binding), heavily relying on mesh precision—missing or redundant geometry is produced where the mesh is inaccurate; (3) Simply adding position offsets to Gaussians improves static rendering, but offsets configured for specific views cause noise and distortion after deformation; (4) Different types of manipulation (large deformation/local/simulation) require different algorithms, making employment cumbersome.
3. **Key Challenge**: How to balance free-floating Gaussians (for high rendering quality) and strict binding with the mesh (for easy manipulation)?
4. **Goal**: Design a general Mesh-Gaussian binding strategy that allows Gaussians to deviate from triangle faces (compensating for mesh inaccuracy) while adaptively adjusting after mesh deformation (preserving relative spatial relations).
5. **Key Insight**: Establish a Local Triangle Space (local coordinate system) on each triangle, optimize Gaussian attributes within this local space, and update only the global transformation of the triangle coordinate system during deformation, keeping local attributes intact.
6. **Core Idea**: Bind Gaussians to the local coordinate system of triangles for optimization; adaptively adjust the global position, rotation, and scale of Gaussians via coordinate system transformations during deformation to achieve high-quality manipulable rendering.

## Method

### Overall Architecture
Three-stage pipeline: (1) Mesh extraction—extract triangular mesh from 3DGS (Marching Cubes / Screened Poisson) or NeuS; (2) Gaussian binding and optimization—bind $N=3$ Gaussians to each triangle and optimize their attributes in the local triangle space; (3) Manipulation—edit the mesh using tools like Blender, with the 3DGS automatically adapting.

### Key Designs

1. **Local Triangle Space**:
    - **Function**: Establish a local coordinate system for each triangle that transforms with its deformation, defining Gaussian attributes within this space.
    - **Mechanism**: Given three vertices of a triangle $(v_1, v_2, v_3)$, define three axes—the first axis along the first edge $r_1^t = (v_2 - v_1)/\|v_2 - v_1\|$, the second axis as the normal direction $r_2^t = n^t$, and the third axis as their cross product $r_3^t = r_1^t \times n^t$. These construct the rotation matrix $R^t$. The local position $\mu^l$ and rotation $R^l$ of Gaussians are optimized, and global attributes are obtained via transformation: $R = R^t R^l$, $\mu = R^t \mu^l + \mu^t$.
    - **Design Motivation**: Standard offset-based schemes define offsets in global space, which fails to adapt post-deformation; attributes in the local coordinate system remain constant during deformation, requiring only global transformation updates to adapt.

2. **Triangle Shape Aware Adaption**:
    - **Function**: Automatically adjust the scale and position of Gaussians when triangle shapes change due to deformation (stretching/compression).
    - **Mechanism**: Define an adaptive vector $e = [e_1, e_2, e_3]$ where $e_1$ is the first edge length $l_1$, $e_3$ is the average length of the second and third edges $0.5(l_2+l_3)$, and $e_2 = 0.5(e_1+e_3)$. The post-deformation global attributes are $s = \beta \cdot e \cdot s^l$ and $\mu = e \cdot R^t \mu^l + \mu^t$ (where $\beta$ is a hyperparameter). When a triangle scales up, the scale and distance of Gaussians scale proportionally.
    - **Design Motivation**: Without shape-aware adaptation, Gaussians that originally covered the surface tightly would leave gaps when a triangle scales up, or overly overlap when compressed. Proportional scaling ensures Gaussians always match the triangle surface area.

3. **Compatibility with Multiple Mesh Extraction Methods (High Mesh Tolerance)**:
    - **Function**: Achieve high-quality rendering even with low-precision meshes.
    - **Mechanism**: Support three mesh sources—(1) 3DGS Marching Cubes (lowest quality, bloated boundaries); (2) Screened Poisson (moderate quality, potentially containing disconnected regions); (3) NeuS (highest quality but slow). Since Gaussians can deviate from triangle faces in local space, free-floating Gaussians can compensate for geometric defects even if the mesh is inaccurate.
    - **Design Motivation**: SuGaR's flat binding entirely inherits mesh defects—missing mesh implies missing rendering, redundant mesh implies redundant rendering. Mani-GS allows Gaussians to "float out" of triangle faces, achieving high-fidelity rendering even on imprecise meshes.

### Loss & Training
Two-stage training: first stage takes 30K iterations to extract the mesh (or use an existing mesh), second stage takes 20K iterations to bind Gaussians to the mesh and optimize using multi-view rendering loss (L1 + SSIM). $N=3$ Gaussians are initialized on each triangle and uniformly distributed on the face. Meshes are reduced to approximately 300K triangles via decimation. Trained on a single A100 GPU.

## Key Experimental Results

### Main Results

**NeRF Synthetic Static Rendering**:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NeRF-Editing | — | — | — |
| SuGaR | 32.06 | 0.958 | 0.039 |
| Ours (SuGaR mesh) | 33.52 | 0.966 | 0.030 |
| Ours (NeuS mesh) | 33.65 | 0.966 | 0.031 |

**DTU Real Data**:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NeuMesh | 28.29 | 0.921 | 0.117 |
| SuGaR | 30.06 | 0.921 | 0.136 |
| Ours | 31.50 | 0.943 | 0.088 |

Ours surpasses SuGaR by ~1.5 PSNR even when using the identical SuGaR mesh.

### Ablation Study

| Config | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3DGS On Mesh (flat binding) | 30.87 | 0.9521 | 0.0447 |
| Mesh + Offset | 32.48 | 0.9625 | 0.0341 |
| Ours + Marching Cube Mesh | 32.11 | 0.9602 | 0.035 |
| Ours + Poisson Mesh | 33.42 | 0.9638 | 0.0324 |
| Ours + NeuS Mesh | 33.45 | 0.9646 | 0.0309 |

### Key Findings
- **Local Triangle Space is core**: There is a 2.6 PSNR gap between flat binding (30.87 PSNR) and the proposed method (33.45 PSNR), proving that allowing Gaussians to deviate from the triangle surface for free optimization is crucial.
- **Offset schemes perform well statically but fail during deformation**: Mesh+Offset yields a reasonable PSNR of 32.48 in static rendering, but produces severe noise and distortion after deformation because offsets lack deformation invariance.
- **High mesh tolerance**: Even using the worst Marching Cube mesh (32.11 PSNR), the proposed method outperforms flat binding using the best mesh (30.87 PSNR) by 1.2 PSNR.
- **Minor difference between NeuS mesh and Poisson mesh** (33.45 vs 33.42), showing the method is insensitive to the mesh source.
- Support for large deformation (stretching/bending), local editing (sauce mixing, cymbal relocation), and soft-body simulation are all achieved through mesh transmission.

## Highlights & Insights
- **Elegant and versatile local coordinate system design**: Defining a local space on each triangle, with concise and clear core equations $R = R^t R^l$ and $\mu = e \cdot R^t \mu^l + \mu^t$, achieves the optimal balance between binding and free floating. This concept can be transferred to any scenario requiring the binding of point clouds/particles to deforming meshes (e.g., cloth simulation, character animation).
- **Unified manipulation framework**: No need to design separate algorithms for large deformation, local editing, or physical simulation—as long as the triangular mesh can be manipulated, the 3DGS automatically follows. Zero extra computation is required after deformation (only matrix multiplications to update global attributes).
- **High tolerance to mesh inaccuracy**: In reality, obtaining high-precision meshes remains difficult. Allowing the use of coarse meshes greatly lowers the barrier to application.

## Limitations & Future Work
- The authors acknowledge that rendering artifacts still occur when local regions undergo highly non-rigid deformation, as the local rigidity assumption is violated.
- Physical simulation on meshes with >35K triangles may take several hours.
- Currently does not support topological changes (e.g., cutting, merging).
- Only 3 Gaussians are bound to each triangle—this may be insufficient for highly detailed regions, making adaptive adjustment of Gaussian density potentially better.
- Training requires mesh extraction before optimizing Gaussians, making the two-stage pipeline not fully end-to-end.
- No quantitative comparison was made with PhysGaussian (physics-based 3DGS simulation).

## Related Work & Insights
- **vs SuGaR**: SuGaR's flat binding restricts Gaussians entirely to the triangle faces, inheriting all mesh defects; Mani-GS allows free offsets in local space, achieving 1.5+ higher PSNR on the identical mesh.
- **vs GaMeS**: GaMeS also constrains Gaussians as flat ellipses inside triangle faces; Mani-GS allows Gaussians to deviate from the surface with adaptive rotation.
- **vs NeRF-Editing**: NeRF-Editing requires constructing a tetrahedral mesh for inverse mapping, making both training and inference slow; Mani-GS completes instantly based on explicit 3DGS manipulation with superior rendering.
- **vs NeuMesh**: NeuMesh defines radiance fields on mesh vertices with explicit deformation rendering, yielding blurry details; Mani-GS utilizes 3DGS splatting to preserve rich details.

## Rating
- Novelty: ⭐⭐⭐⭐ The local triangle coordinate system and shape-aware adaptation designs are novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablations on synthetic and real data, with a sufficient demonstration of various manipulation types, though lacking quantitative post-manipulation rendering comparisons.
- Writing Quality: ⭐⭐⭐⭐ Mathematical formulations are clear, and diagrams are intuitive, although some details require referring to the appendix.
- Value: ⭐⭐⭐⭐ Provides a practical and generic solution for 3DGS editing, with direct application value for 3D content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
