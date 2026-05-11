---
title: >-
  [Paper Note] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images
description: >-
  [3D Vision] This paper proposes UFO-4D, a unified feedforward framework that directly predicts dynamic 3D Gaussian representations from two unposed images, enabling jointly consistent estimation of 3D geometry, 3D motion…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 601a6ec663a00cd1
---

# UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images

- **Conference**: ICLR 2026
- **arXiv**: [2602.24290](https://arxiv.org/abs/2602.24290)
- **Code**: [Project Page](https://ufo-4d.github.io/)
- **Area**: 3D Vision / 4D Reconstruction
- **Keywords**: 4D Reconstruction, Dynamic 3D Gaussians, Feedforward, Scene Flow, Unposed, Self-Supervised

## TL;DR

This paper proposes UFO-4D, a unified feedforward framework that directly predicts dynamic 3D Gaussian representations from two unposed images, enabling jointly consistent estimation of 3D geometry, 3D motion, and camera pose, achieving up to 3× improvement over existing methods on geometry and motion benchmarks.

## Background & Motivation

Joint estimation of camera pose, 3D geometry, and 3D motion from casually captured images (4D scene reconstruction) is a fundamental challenge in computer vision. Existing methods suffer from the following limitations:

**Test-time optimization methods** are slow (hours) and depend on precomputed depth and optical flow.

**Feedforward models** (DUST3R, MonST3R, DynaDUSt3R) perform well on individual tasks but lack a unified architecture.

**Scarcity of 4D training data**: synthetic data suffers from domain gaps, while real-world annotations are sparse and noisy.

**Core Insight**: Differentiable rendering of multiple signals (appearance, depth, motion) from a single dynamic 3D Gaussian representation provides self-supervised training signals, and geometric coupling allows different supervision signals to mutually regularize each other.

## Method

### Overall Architecture

Given two unposed images $\mathbf{I}_t, \mathbf{I}_{t+1}$ and camera intrinsics:

$$f_\theta(\mathbf{I}_t, \mathbf{I}_{t+1}) \mapsto (\mathcal{G}, \mathbf{P})$$

The model outputs a set of dynamic 3D Gaussians $\mathcal{G}$ and relative camera pose $\mathbf{P}$. Each dynamic Gaussian contains:
- 3D center $\boldsymbol{\mu} \in \mathbb{R}^3$
- 3D motion $\mathbf{v} \in \mathbb{R}^3$
- Covariance parameters (quaternion rotation $\mathbf{r}$, scale $\mathbf{s}$)
- Spherical harmonic color $\mathbf{h}$, opacity $o$

### Network Architecture

- **Encoder**: Weight-shared ViT processing each image independently
- **Decoder**: ViT with cross-attention layers for fusing information from both images
- **Heads**:
    - Center head (DPT) → 3D position
    - Attribute head (DPT) → rotation, scale, color, opacity
    - Velocity head (DPT) → 3D motion vectors
    - Pose head (3-layer MLP) → relative pose (translation + quaternion)
- **Initialization**: NoPoSplat (Gaussian heads) + MASt3R (remaining components)

### Differentiable 4D Rasterization

**Key Innovation**: The standard 3DGS rasterizer is extended to unify rendering of images, point maps, and scene flow.

Temporal interpolation (linear motion assumption):

$$\mathcal{G}(t') = \{(\boldsymbol{\mu} + \Delta t \cdot \mathbf{v}, \mathbf{v}, \mathbf{r}, \mathbf{s}, \mathbf{h}, \mathbf{c}, o)_\mathbf{p}\}$$

Unified $\alpha$-blending for rendering point maps and motion maps:

$$\mathbf{X}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \boldsymbol{\mu}_i o_i \prod_{j=1}^{i-1}(1-o_j)$$

$$\mathbf{V}_{t'}(\mathbf{p}) = \sum_{i \in \mathcal{N}_\mathbf{p}^{t'}} \mathbf{v}_i o_i \prod_{j=1}^{i-1}(1-o_j)$$

### Loss & Training

**Total Loss** = Supervised Loss + Self-Supervised Loss:

$$L_{total} = L_{sup} + L_{self}$$

**Supervised Loss** (scene flow + point map + pose):

$$L_{sup} = L_{motion} + w_{point} L_{point} + w_{pose} L_{pose}$$

- $L_{motion}$: jointly constrains Gaussian center motion $\mathbf{v}$ and rendered motion $\mathbf{V}$
- $L_{point}$: jointly constrains Gaussian position $\boldsymbol{\mu}$ and rendered point map $\mathbf{X}$
- $L_{pose}$: separately constrains translation and quaternion

**Self-Supervised Loss** (photometric + smoothness):

$$L_{self} = L_{photo} + w_{smooth} L_{smooth}$$

- $L_{photo} = \text{MSE} + w_{lpips} \text{LPIPS}$
- $L_{smooth}$: edge-aware smoothness regularization

### Downstream Tasks

- Depth = last channel of point map
- Optical flow = 2D projection of 3D scene flow
- Motion segmentation = scene flow thresholding
- 4D interpolation = rendering at arbitrary time and viewpoint

## Key Experimental Results

### Training Data

Mixed training: Stereo4D (60%) + PointOdyssey (20%) + Virtual KITTI 2 (20%)

### Main Results

**Geometry Estimation** (point map EPE, depth metrics):

| Method | Stereo4D EPE | KITTI EPE | Sintel EPE |
|------|-------------|-----------|-----------|
| DynaDUSt3R | ~0.15 | ~0.80 | - |
| ZeroMSF | ~0.12 | ~0.65 | - |
| **UFO-4D** | **~0.05** | **~0.25** | **Best** |

UFO-4D surpasses competing methods by **3× or more** on Stereo4D and KITTI.

**Motion Estimation** (scene flow EPE): Similarly achieves substantial improvements.

### Key Findings

1. **Self-supervised loss** substantially improves both geometry and motion estimation quality.
2. Direct pose estimation outperforms post-hoc regression (as in DUSt3R).
3. Mixed synthetic and real training effectively mitigates domain gaps.
4. 4D interpolation generalizes well across novel viewpoints and time steps.

### 4D Interpolation Application

For the first time, spatiotemporal interpolation is achieved from feedforward outputs: images, depth, and motion can be rendered at arbitrary intermediate time steps and viewpoints.

## Highlights & Insights

1. **Unified Representation**: A single dynamic 3D Gaussian representation jointly addresses geometry, motion, and pose estimation.
2. **Self-Supervised Training**: Photometric reconstruction loss requires no annotations, effectively overcoming data scarcity.
3. **Coupled Regularization**: Geometry and motion share Gaussian primitives, enabling mutual regularization across supervision signals.
4. **New Application**: Feedforward 4D interpolation enabling spatiotemporal interpolation of images, geometry, and motion.
5. **Performance Leap**: Over 3× improvement in EPE metrics.

## Limitations & Future Work

1. The linear motion assumption limits modeling of complex non-rigid motions.
2. Only two frames are processed as input, precluding modeling of long-term temporal dependencies.
3. Camera intrinsics are required as input (though typically available).
4. The optimal mixing ratio of training data requires manual tuning.
5. Reconstruction in heavily occluded regions may be inaccurate.

## Related Work & Insights

- **Static 3D Reconstruction**: DUSt3R (Wang et al., 2024b) and MASt3R (Leroy et al., 2024) learn strong priors for end-to-end reconstruction.
- **Dynamic 3D Reconstruction**: MonST3R (Zhang et al., 2025a) fine-tunes static models for dynamic scenes but lacks temporal correspondence.
- **Dense 4D Reconstruction**: Test-time optimization methods achieve high quality but are slow; existing feedforward methods require pose inputs or use separate task heads.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unified 4D representation with a self-supervised framework is a significant contribution.
- **Practicality**: ⭐⭐⭐⭐ — Feedforward inference offers strong potential for real-time applications.
- **Clarity**: ⭐⭐⭐⭐⭐ — Method is described systematically with complete mathematical derivations.
- **Significance**: ⭐⭐⭐⭐⭐ — Advances dense 4D reconstruction from optimization-based to feedforward paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/3d_vision/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] Universal Beta Splatting](universal_beta_splatting.md)

</div>

<!-- RELATED:END -->
