---
title: >-
  [Paper Note] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians
description: >-
  [ICCV 2025][3D Vision][3D head reconstruction] SHeaP replaces traditional differentiable mesh rendering with 2D Gaussian Splatting for self-supervised 3DMM prediction training. By binding Gaussians to the 3DMM mesh for re-animation, and introducing a graph-convolution-based Gaussian regressor together with geometry consistency regularization, SHeaP surpasses all self-supervised methods on the NoW and Nersemble benchmarks.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D head reconstruction
  - 2D Gaussian Splatting
  - 3DMM
  - self-supervised
  - face geometry
  - rigged avatar
date: 2026-05-08
content_hash: 9bb33e0f1db8717f
---

# SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians

**Conference**: ICCV 2025
**arXiv**: [2504.12292](https://arxiv.org/abs/2504.12292)
**Institution**: Woven by Toyota, Toyota Motor Europe, TU Munich, Kyoto University
**Area**: 3D Vision / Face Reconstruction / Self-Supervised Learning
**Keywords**: 3D head reconstruction, 2D Gaussian Splatting, 3DMM, self-supervised, face geometry, rigged avatar

## TL;DR
SHeaP replaces traditional differentiable mesh rendering with 2D Gaussian Splatting for self-supervised 3DMM prediction training. By binding Gaussians to the 3DMM mesh for re-animation, and introducing a graph-convolution-based Gaussian regressor together with geometry consistency regularization, SHeaP surpasses all self-supervised methods on the NoW and Nersemble benchmarks.

## Background & Motivation
Real-time 3D head reconstruction from a single 2D image is a core task in computer vision, with applications in AR/VR and digital avatars. Due to the scarcity of large-scale 3D ground-truth data, self-supervised methods that learn from 2D video have become the mainstream approach. Traditional self-supervised methods rely on differentiable mesh rendering (e.g., DECA/EMOCA), but face two major bottlenecks: (1) discontinuities in mesh rasterization lead to inaccurate gradients; (2) mesh rendering lacks photorealism, limiting the quality of supervision signals from photometric losses. Furthermore, mesh rendering requires precise facial masks to exclude hair and shoulder regions, increasing preprocessing overhead.

## Core Problem
How can the training effectiveness of self-supervised 3DMM predictors be improved? The core challenges are: (1) **Rendering quality** — the limitations of differentiable mesh rendering reduce the effectiveness of photometric losses; (2) **Geometry–appearance coupling** — how to ensure that appearance supervision on Gaussians is effectively propagated to the underlying 3DMM geometry parameters.

## Method

### Overall Architecture
SHeaP adopts a source-target re-animation paradigm for self-supervised training. Given a source image, a ViT predicts 3DMM parameters (shape $\beta$, pose $\theta$, expression $\psi$) and identity features $f$; a Gaussian Regressor predicts a set of 2D Gaussians bound to the 3DMM mesh from $f$ and DINOv2 features. The rigged head avatar is then re-animated to the target frame's pose and expression, rendered, and compared with the target ground-truth image via a photometric loss for backpropagation.

### Key Designs

1. **3DMM Parameter Estimator**: A ViT architecture similar to TokenFace is adopted. The face image is divided into patches and fed into the ViT, with five learnable tokens appended (shape / expression / pose / lighting / features). The outputs pass through LayerNorm and MLP heads to produce the respective 3DMM parameters. ViT weights are initialized with FaRL.

2. **Gaussian Regressor (Core Contribution)**: Consists of two sub-networks —

    - **UV Map Generator**: The identity features $f$ output by the ViT are reshaped into a feature map, then cross-attended with DINOv2 features via a lightweight GAN architecture to produce a feature tensor $M$ in UV space.
    - **Graph Convolutional Network**: Each Gaussian has a parent face and a learnable embedding $e_i$. The embedding is concatenated with region features sampled from the UV map and passed through a ResNet-style graph convolutional network to produce the final Gaussian attributes (offset $x$, scale $s$, rotation $q$, albedo $c$, opacity $\sigma$, totaling 14 dimensions). The adjacency matrix is defined by the 2-ring neighborhood of mesh faces.

3. **Dynamic Gaussian Densification/Pruning**: The opacity mean and positional gradient of each Gaussian prototype are tracked. Every $t_{\text{densify}}$ steps, the $n_{\text{prune}}$ prototypes with the lowest opacity are removed, and the $n_{\text{densify}}$ prototypes with the largest positional gradients are duplicated with added noise. The total count is kept constant, with each face hosting a minimum of 1 and a maximum of 6 Gaussians.

4. **2DGS Binding Mechanism**: The binding formulation of GaussianAvatars is extended by replacing isotropic scaling with an anisotropic scaling matrix $S_p = \text{diag}(s_u, s_v, s_n)$, where $s_u / s_v$ are the lengths of the triangular face along the UV directions and $s_n = \min(s_u, s_v)$. The scaling along the normal direction does not affect the final scale of a 2D Gaussian, but does affect the center position $\mu$, allowing Gaussians to offset from the mesh surface along the normal to capture details not covered by the mesh.

5. **Illumination Model**: The Gaussian Regressor outputs albedo, which is combined with a Lambertian shading model based on spherical harmonics (SH). The ViT predicts illumination PCA weights from the source image, which are transformed into SH coefficients via the Basel Illumination Prior. The final color is computed as albedo $\times$ SH illumination.

### Loss & Training
- **Landmarks Loss**: L1 loss between projected mesh landmarks and 2D detected landmarks (with a very small weight to mitigate inaccuracies in landmark detection).
- **Photometric Loss**: Four loss terms between the target image and the rendered image (L1 + perceptual + SSIM, etc.).
- **Geometry Consistency Regularization (Key)**: Constrains Gaussian normals to align with their parent face normals, and Gaussian depth maps to align with mesh-rendered depth maps. This ensures that gradients from appearance optimization are effectively propagated to 3DMM geometry parameters.

## Key Experimental Results

### NoW Benchmark (Neutral Face Geometry Evaluation)

| Method | Training Data | Median↓ | Mean↓ | Std↓ |
|--------|--------------|---------|-------|------|
| DECA (3D sup.) | 2D+3D | 1.09 | 1.38 | 1.18 |
| MICA (3D sup.) | 2D+3D | 0.91 | 1.14 | 0.95 |
| TokenFace (3D sup.) | 2D+3D | **0.87** | **1.07** | **0.88** |
| DECA (self-sup.) | 2D only | 1.09 | 1.38 | 1.18 |
| SMIRK (self-sup.) | 2D only | 1.20 | 1.47 | 1.16 |
| **SHeaP** | 2D only | **0.97** | **1.22** | **1.04** |

- Training with purely 2D self-supervision surpasses all self-supervised methods and approaches the 3D-supervised method MICA.

### Nersemble Benchmark (Expressive Face Geometry Evaluation, Newly Proposed)
- SHeaP substantially outperforms all publicly available methods on non-neutral expression reconstruction.
- Emotion classification accuracy on AffectNet also reaches state of the art.

### Ablation Study
- 2DGS vs. 3DGS: 2DGS is overall superior (more accurate normals and depth, enhancing geometry coupling).
- Geometry consistency regularization contributes most (removing it degrades NoW median from 0.97 to 1.15+).
- The combination of UV Map Generator + Graph Conv outperforms pure MLP or pure CNN alternatives.
- The densification/pruning mechanism yields a further improvement of 0.02 in median.

## Highlights & Insights
- **2DGS rendering overcomes mesh rendering bottlenecks**: The advantages of 2D surfels (accurate depth/normals, closed-form normal computation) inherently improve the quality of self-supervised signals.
- **Geometry consistency regularization is critical**: Depth and normal consistency constraints effectively propagate gradients from appearance learning to 3DMM parameters — this is the core of the method's success.
- **Graph-convolutional Gaussian regressor is elegantly designed**: The UV map provides global identity information while graph convolutions coordinate local Gaussians, yielding greater stability than direct prediction.
- **No facial mask required**: The flexibility of Gaussians allows the model to naturally cover hair and shoulders, eliminating the dependency on precise face masks required by traditional methods.
- **New benchmark**: The expressive geometry evaluation established on Nersemble fills the gap of benchmarks for non-neutral expressions.

## Limitations & Future Work
- Validated only on the FLAME 3DMM; generalization to other morphable models remains unexplored.
- At inference, only the 3DMM mesh is output; Gaussians are used solely during training — the possibility of directly outputting a rigged Gaussian avatar has not been explored.
- Training requires paired same-identity video frames (source-target pairs), imposing relatively high data requirements.
- The illumination model is limited to Lambertian + SH and cannot model complex lighting effects such as specular reflections.

## Related Work & Insights
- **vs. DECA/EMOCA**: Traditional self-supervised methods based on differentiable mesh rendering; SHeaP significantly improves supervision signal quality via 2DGS rendering.
- **vs. SMIRK**: SMIRK similarly pursues expression accuracy but uses a neural renderer conditioned on mesh rendering for photometric loss; SHeaP's 2DGS approach is more direct and achieves better results.
- **vs. GaussianAvatars**: GaussianAvatars assumes 3DMM tracking is already given and optimizes Gaussians only; SHeaP simultaneously predicts 3DMM parameters and Gaussians, posing a more challenging problem.
- **vs. TokenFace**: TokenFace relies on 3D supervised data; SHeaP achieves comparable performance using only 2D training data.

The application of 2DGS to self-supervised 3D morphable model learning provides a new paradigm for other deformable 3D reconstruction tasks (hands, body). The geometry consistency regularization concept is transferable to any self-supervised pipeline that learns geometry from appearance. The design of using Gaussians as a "high-quality rendering proxy" during training rather than as the final representation deserves attention. Performing densification/pruning in latent space (prototype space) rather than in Gaussian space offers a new approach to dynamic Gaussian count management. The closed-form normal computation of 2DGS makes geometry consistency regularization feasible — this design choice would be problematic with 3DGS, where normal definitions are ambiguous.

## Technical Details
- Training data: VoxCeleb2 video dataset (large-scale in-the-wild face videos).
- The FLAME model is used as the 3DMM (shape: 300 dimensions, expression: 100 dimensions, pose: 6 dimensions).
- Each face is initially assigned 2 Gaussians; after densification/pruning, the average converges to approximately 4.
- Inference speed: approximately 15 ms per image (ViT + 3DMM parameter prediction), meeting real-time application requirements.
- DINOv2 features provide semantic information that helps distinguish different facial regions (eyes, nose, mouth, hairline), resulting in a more reasonable Gaussian distribution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce 2DGS into self-supervised 3DMM learning; the graph conv + UV map Gaussian regressor design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ NoW + new Nersemble benchmark + AffectNet + detailed ablations, though comparisons across more 3DMMs are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear pipeline, well-motivated components, good coordination of figures and equations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new state of the art in self-supervised face reconstruction; the 2DGS + 3DMM paradigm has broad potential impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] S3E: Self-Supervised State Estimation for Radar-Inertial System](s3e_self-supervised_state_estimation_for_radar-inertial_system.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](../../NeurIPS2025/3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)

<!-- RELATED:END -->
