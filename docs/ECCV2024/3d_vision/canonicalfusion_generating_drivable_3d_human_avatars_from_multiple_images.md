---
title: >-
  [Paper Note] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images
description: >-
  [ECCV 2024][3D Vision][Drivable 3D Human] This paper proposes the CanonicalFusion framework, which achieves direct canonicalization by jointly predicting depth maps and compressed LBS weight maps, and fuses information from multiple input images using forward skinning differentiable rendering to generate drivable 3D human avatars from multiple input images.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Drivable 3D Human"
  - "Canonical Space Fusion"
  - "Forward Skinning"
  - "Differentiable Rendering"
  - "LBS Weight Compression"
date: 2026-05-08
content_hash: 37d03c1e0dbc33b0
---

# CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images

**Conference**: ECCV 2024  
**arXiv**: [2407.04345](https://arxiv.org/abs/2407.04345)  
**Code**: [Available](https://github.com/jsshin98/CanonicalFusion)  
**Area**: 3D Vision  
**Keywords**: Drivable 3D Human, Canonical Space Fusion, Forward Skinning, Differentiable Rendering, LBS Weight Compression

## TL;DR

This paper proposes the CanonicalFusion framework, which achieves direct canonicalization by jointly predicting depth maps and compressed LBS weight maps, and fuses information from multiple input images using forward skinning differentiable rendering to generate drivable 3D human avatars from multiple input images.

## Background & Motivation

Generating 3D human avatars from images is a key technology for the metaverse and AR/VR. Limitations of existing methods:

**Implicit methods** (PIFu family): Resolution is limited by voxel space.

**Explicit methods** (sandwich-like depth maps): Difficulty in multi-view fusion.

**Template-driven methods** (ARCH family): Inaccurate initialization for points far from the template surface.

The core idea of CanonicalFusion: **fusing multi-view reconstruction results in the canonical space**.

## Method

### Overall Architecture

A two-step pipeline:
1. **Initial Mesh Prediction**: A shared encoder-dual decoder network predicts double-sided depth maps and compressed LBS weight maps, achieving direct canonicalization to obtain the initial mesh.
2. **Forward Skinning Differentiable Rendering Optimization**: The canonical mesh is deformed via forward skinning -> rendered through differentiable rasterization -> to minimize geometric and photometric errors.

### Key Designs

#### 1. Compact Representation of LBS Weights

A stacked autoencoder MLP is used to compress the 55-dimensional skinning weights into a 3D latent space. The training data consists of approximately 800K samples (interpolated from SMPL-X UV coordinates). Loss functions: L1 + non-zero element loss (approximated by Gaussian radial basis functions) + KL divergence. A softmax layer at the end ensures the weights sum to 1. After pre-training, only the decoder is used during inference.

#### 2. Joint Depth and LBS Prediction

The ATUNet architecture features a shared encoder, a depth decoder, and an LBS decoder. It takes RGB and SMPL-X depth maps as input and outputs front-and-back depth maps and 3D compressed LBS weight maps. An additional UNet texture prediction network outputs a shading-free color map.

#### 3. Canonical Mesh Reconstruction

Direct inverse skinning is performed from LBS weights to the canonical space. Invisible areas (e.g., underarms, inner thighs) are filled via SDF integration: combining the signed distance functions of the reconstructed mesh and the SMPL-X template mesh, switching based on whether a point is near the reconstructed mesh. After Marching Cubes extraction, Flexicubes are used to convert it into a differentiable, compact mesh.

#### 4. Forward Skinning Differentiable Rendering Optimization

The canonical mesh is deformed to various input poses through forward skinning and rendered using NDS rasterizers. The canonical mesh vertex positions, colors, and 3D pose parameters are optimized simultaneously. A progressive scheme is used: first optimize the poses, then fix the poses to optimize shape and color. The resolution is upsampled by a factor of 4 every 500 iterations, for a total of 2000 iterations.

### Loss & Training

Optimization objectives: Laplacian smoothing + normal consistency regularization + normal map L1 + mask MSE + Chamfer distance + color L2. Training takes 2 days on 4x RTX 3090 GPUs, and inference takes about 11 minutes.

## Key Experimental Results

### Main Results: Comparison of Monocular Human Reconstruction

| Method | Training Data | RP P2S↓ | RP CF↓ | TH3.0 P2S↓ | TH3.0 CF↓ |
|------|----------|---------|--------|------------|-----------|
| PIFuHD | RP | 1.420 | 1.434 | 1.534 | 1.527 |
| ICON* | RP | 1.296 | 1.364 | 1.371 | 1.437 |
| 2K2K* | TH2.0+RP | 1.097 | 1.195 | 1.416 | 1.542 |
| TeCH* | N/A | 1.489 | 1.523 | 1.721 | 1.795 |
| **Ours** | **TH2.0+RP** | **0.886** | **0.943** | **1.072** | **1.165** |

(P2S = point-to-surface distance in cm, CF = Chamfer distance in cm, * = using GT SMPL-X)

### Comparison with SCANimate (Canonical Space Accuracy)

| Method | No. of Views | SET1 P2S↓ | SET2 P2S↓ |
|------|--------|-----------|-----------|
| SCANimate | 5 | 1.362 | 1.076 |
| SCANimate | 15 | 1.103 | 0.997 |
| **Ours** | **5** | **0.244** | **0.180** |
| **Ours** | **15** | **0.199** | **0.149** |

The P2S error is only about 1/5 of SCANimate's.

### Ablation Study

- **Pose Error Correction**: Jointly optimizing pose parameters effectively corrects errors such as bent arms in the initial mesh.
- **Loose Clothing**: Starting from the initial canonical mesh (rather than a template), the topology is closer to the target, allowing for the recovery of loose clothing.
- **Multi-frame Fusion**: Multiple frames significantly improve the completeness of the model.
- **Real-world Scenes**: Realistic avatars can be generated for both Actors-HQ and in-the-wild captures.

### Key Findings

1. Explicit depth prediction guided by SMPL-X remains an effective approach without the need for complex implicit techniques.
2. Diverse datasets consistently improve performance.
3. Forward skinning outperforms inverse skinning (SCANimate degrades severely under pose errors).
4. Compressing LBS weights to 3D results in almost no accuracy loss.

## Highlights & Insights

1. **3D LBS Compression**: Compressing 55-dimensional sparse skinning weights into 3D reduces prediction difficulty while allowing for visualization.
2. **Canonical Space Fusion**: Avoids the geometric alignment difficulties of multi-view fusion in the observation space.
3. **Joint Pose-Shape Optimization**: Mitigates the cascading effects of pose estimation errors.
4. **SDF Integration for Void Filling**: Cleverly combines the reconstructed mesh and the template mesh.
5. **Arbitrary Number of Inputs**: Supports anywhere from 1 to dozens of input images.

## Limitations & Future Work

1. **Non-rigid Clothing Deformation**: Excessive deformation between frames may cause blurriness.
2. **Hand Details**: Requires assistance from an external hand replacement module.
3. **Reliance on SMPL-X Estimation**: The initial depth map depends on the quality of the SMPL-X parameters.
4. **Future Work**: Handling non-rigid deformation of hair and clothing, and incorporating generative technologies.

## Related Work & Insights

- Compared to the PIFu family, explicit depth prediction allows for handling higher resolutions.
- SCANimate's cycle consistency is inherited and improved by canonical space fusion.
- SNARF's forward skinning field inspired the forward skinning differentiable rendering in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ — The LBS compression and canonical space fusion strategies are novel and practical.
- Practicality: ⭐⭐⭐⭐ — Directly drivable, open-source, and supports an arbitrary number of inputs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset and multi-method comparisons, combined with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear workflow and rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images](divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [\[CVPR 2025\] CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](../../CVPR2025/3d_vision/cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)
- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
