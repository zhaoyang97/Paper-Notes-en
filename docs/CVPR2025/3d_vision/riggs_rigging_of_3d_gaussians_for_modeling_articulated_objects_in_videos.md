---
title: >-
  [Paper Note] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos
description: >-
  [CVPR 2025][3D Vision][3D Gaussians] This paper proposes RigGS, an automated skeleton-driven modeling method without template priors, which extracts 3D skeletons from monocular videos and rigs 3D Gaussian representations to support novel view synthesis, pose editing, motion interpolation, and motion transfer.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussians"
  - "Skeleton Extraction"
  - "Articulated Object Modeling"
  - "Dynamic Scenes"
  - "Template-free Prior"
date: 2026-05-08
content_hash: 48edf0ddf0acb5ce
---

# RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos

**Conference**: CVPR 2025  
**arXiv**: [2503.16822](https://arxiv.org/abs/2503.16822)  
**Code**: [Project Page](https://yaoyx689.github.io/RigGS.html)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussians, Skeleton Extraction, Articulated Object Modeling, Dynamic Scenes, Template-free Prior

## TL;DR

This paper proposes RigGS, an automated skeleton-driven modeling method without template priors, which extracts 3D skeletons from monocular videos and rigs 3D Gaussian representations to support novel view synthesis, pose editing, motion interpolation, and motion transfer.

## Background & Motivation

Rigging of articulated objects is a core technology in film, games, and AR/VR. Traditional methods rely on manual design or parametric template models (such as SMPL, MANO, SMAL), facing the following challenges:

- **Template reliance limits generalization**: SMPL can only handle human bodies and MANO can only handle hands, failing to process non-standard objects such as humans with accessories, diverse animals, or robots.
- **Reliance on high-quality 3D reconstruction**: Existing template-free methods typically reconstruct 3D meshes before extracting skeletons, heavily relying on reconstruction quality. Methods based on Medial Axis Transform require precise surface reconstruction.
- **Dense skeletons lacking semantics**: Although optimization methods can extract skeletons, the results are often too dense and lack a concise semantic structure.
- **Lack of editability in Neural Bones**: Methods like BANMo use learnable bones to represent motion, but these bones do not carry semantic information, making it difficult to use them to create plausible novel actions.
- **Scarcity of 3D data**: Methods that extract skeletons from point cloud sequences are limited by the scarce availability of 3D training data.

## Method

### Overall Architecture

RigGS adopts a three-stage pipeline: (1) Initialization stage: a deformation field controlled by skeleton-aware nodes is used to drive the canonical 3D Gaussian representation, completing 4D reconstruction and obtaining candidate skeleton nodes; (2) Skeleton construction stage: a sparse skeleton tree is extracted from the candidate nodes using a coarse-to-fine heuristic algorithm; (3) Skeleton-driven modeling stage: 3D Gaussians are driven using learnable skinning weights and pose-dependent detail deformations.

### Key Design 1: Skeleton-Aware Node-Controlled Deformation — Simultaneous Reconstruction and Candidate Skeleton Extraction

**Function**: Simultaneously obtain the canonical 3D Gaussian representation and candidate control nodes with skeleton semantics during the initial 4D reconstruction process.

**Mechanism**: Define a set of learnable skeleton-aware nodes $\mathbf{C} = \{\mathbf{c}\}$, where each node at time $t$ has a rotation $\tilde{\mathbf{R}}_\mathbf{c}^t$ and translation $\tilde{\mathbf{t}}_\mathbf{c}^t$ (predicted by an MLP). The deformation of each Gaussian is determined by a weighted blend of its $k$-nearest neighbor nodes:

$$\mu_i^t = \sum_{\mathbf{c} \in \mathcal{N}(\mu_i)} w_{\mu_i,\mathbf{c}} (\tilde{\mathbf{R}}_\mathbf{c}^t (\mu_i - \mathbf{c}) + \mathbf{c} + \tilde{\mathbf{t}}_\mathbf{c}^t)$$

The weights are calculated based on Gaussian kernel distance. Concurrently, a 2D skeleton projection constraint $L_\text{proj}^t$ is introduced to guarantee that the nodes are aligned with the medial axis of the object.

**Design Motivation**: Unlike the two-step approach that reconstructs first and then extracts the skeleton, this design integrates skeleton information into the reconstruction process, reducing dependence on 3D reconstruction quality. The 2D skeleton projection constraint is simpler and more robust than directly extracting the medial axis from 3D models.

### Key Design 2: Coarse-to-Fine Skeleton Construction — Fusing Geometric, Motion, and Semantic Information

**Function**: Extract a sparse, semantically meaningful skeleton tree structure from dense skeleton-aware nodes.

**Mechanism**: First, the time step closest to the average pose is selected as the new canonical shape. Then, a uniformly distributed subset of nodes is obtained through FPS sampling to construct a minimum spanning tree based on the average inter-frame distance $\beta_{ij} = \sum_t \|\mathbf{c}_i^t - \mathbf{c}_j^t\| / |\mathcal{I}|$. Next, redundant branches are removed and adjacent intersections are merged to obtain a dense skeleton tree. Finally, semantic labels from DINOv2 are utilized to ensure symmetry, and BFS is used to determine the root node and parent-child relationships, yielding the final sparse skeleton tree $\mathcal{T} = \{\mathcal{J}, \mathcal{A}\}$.

**Design Motivation**: Comprehensively utilizing geometric (spatial distance), motion (inter-frame distance variations), and semantic (DINOv2 features) information generates more plausible skeleton structures than purely geometric methods.

### Key Design 3: Skeleton-Driven Dynamic Modeling — Learnable Skinning + Pose Detail Deformation

**Function**: Achieve high-quality rendering and flexible pose editing via a skeleton-driven deformation field.

**Mechanism**: LBS (Linear Blend Skinning) is used for coarse deformation: $\hat{\mu}_i^t = \mathbf{T}_1^t (\sum_{j} \hat{\omega}_{i,j} \mathbf{T}_j^t \bar{\mu}_i)$, where the skinning weights $\hat{\omega}_{i,j}$ are jointly determined by a learnable scaling factor $\eta_{i,j}$ (learned via an MLP) and the distance to the bones. Building on this, pose-dependent detail deformations $\delta_{i,t} = F_\Pi(\mu_i, \{\hat{\mathbf{r}}_j^t\})$ are added, resulting in the final position $\mu_i^t = \hat{\mu}_i^t + \delta_{i,t}$.

**Design Motivation**: LBS provides global rigid deformation but falls short on fine details like cloth wrinkles. A pose-dependent (rather than time-dependent) detail deformation module ensures that plausible details are generated even when creating novel motions. The regularization term $L_\text{detail}$ controls the magnitude of the detail deformation.

### Loss & Training

$$L^t = L_\text{render}^t + w_{\tilde{\text{proj}}}^t L_{\tilde{\text{proj}}}^t + w_\text{detail}^t L_\text{detail}^t + w_\text{id} L_\text{id}^t$$

where $L_\text{render}$ is the $\ell_1$ + D-SSIM rendering loss, $L_{\tilde{\text{proj}}}$ represents the skeleton projection constraint with adaptive weights, $L_\text{detail}$ is the detail deformation regularization, and $L_\text{id}$ constrains the skeleton pose at the canonical frame to be close to the identity translation.

## Key Experimental Results

### Main Results: Novel View Synthesis (PSNR/SSIM/LPIPS)

| Method | D-NeRF | DG-Mesh |
|------|--------|---------|
| D-NeRF | 30.48 / 0.973 / 0.049 | 28.17 / 0.957 / 0.078 |
| TiNeuVox | 32.60 / 0.983 / 0.044 | 31.95 / 0.967 / 0.048 |
| 4D-GS | 33.25 / 0.989 / 0.023 | 33.96 / 0.979 / 0.027 |
| SC-GS | **43.04** / 0.998 / 0.007 | **38.96** / 0.993 / 0.014 |
| AP-NeRF | 30.94 / 0.970 / 0.035 | 31.83 / 0.967 / 0.046 |
| **RigGS** | 40.82 / 0.996 / 0.011 | 37.65 / 0.991 / 0.017 |

### ZJU-MoCap Real Dataset Comparison

| Method | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| AP-NeRF | 25.62 | 0.919 | 0.093 |
| **RigGS** | **33.54** | **0.975** | **0.033** |

### Ablation Study

| Setup | Effect |
|------|------|
| Anisotropic vs. Isotropic Gaussians | Anisotropic yields higher quality but generalizes poorly to novel poses; isotropic is chosen |
| Without 2D projection loss | Skeleton cannot be embedded inside the Gaussian shape |
| Fixed weight projection loss | Skeleton protrudes outside the shape in some frames |

### Key Findings

- RigGS is close to SC-GS in rendering quality (about ~2dB lower), but SC-GS does not support skeleton editing.
- RigGS significantly outperforms AP-NeRF on ZJU-MoCap (+7.92dB), demonstrating the correctness of reducing reliance on 3D reconstruction quality.
- Adaptive projection loss weights filter out inaccurate 2D skeletons via the 3-sigma rule, effectively avoiding misguidance.

## Highlights & Insights

1. **Integrating skeleton extraction into the reconstruction process** instead of doing it post-reconstruction fundamentally reduces the dependency on reconstruction quality.
2. **Heuristic skeleton simplification via multi-information fusion** combines geometric, motion, and semantic (DINOv2) information, which is more robust than purely geometric methods.
3. **Pose-dependent rather than time-dependent detail deformation** ensures generalization ability to novel poses.
4. **High engineering completeness**: An interactive GUI is provided to support real-time editing and rendering.

## Limitations & Future Work

- The performance is limited in scenarios with sparse views, inaccurate camera poses, or rapid, dramatic motions.
- Pose-dependent appearance changes (such as lighting and material variations) are not modeled.
- Semantic editing of skeletons (e.g., guided by text or images) is an interesting direction for future work.
- The rendering quality is slightly lower compared to SC-GS, because the skeleton-driven deformation field possesses fewer degrees of freedom.

## Related Work & Insights

- **SC-GS**: Uses a deformation field with 512 control points, which has high degrees of freedom but does not support semantic editing.
- **AP-NeRF**: A similar template-free skeleton extraction method, but relies on TiNeuVox reconstruction and MAT skeleton extraction.
- **BANMo / BAGS**: Express deformation using neural bones, but lack semantic information of tree-like skeleton structures.
- **DINOv2**: Provides effective visual semantic features for skeleton simplification.

## Rating

⭐⭐⭐⭐ — The complete template-free articulated object rigging pipeline is elegantly designed, with a clear three-stage logic. Although the rendering quality is slightly lower than that of free-form deformation methods, the editability provided by the rigging offers unique value. It significantly outperforms the comparable method AP-NeRF on real-world datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] Category-Agnostic Neural Object Rigging](category-agnostic_neural_object_rigging.md)
- [\[CVPR 2025\] SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction](sgcr_spherical_gaussians_for_efficient_3d_curve_reconstruction.md)
- [\[CVPR 2026\] Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting](../../CVPR2026/3d_vision/part2gs_part-aware_modeling_of_articulated_objects_using_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
