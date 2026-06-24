---
title: >-
  [Paper Note] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering
description: >-
  [ICCV 2025][Autonomous Driving][dynamic scene rendering] This paper proposes UGSDF, a method that jointly learns an SDF network and 3D Gaussian Splatting to model dynamic objects in urban scenes. Using only 2D priors (a depth network and a point tracker), UGSDF achieves state-of-the-art rendering quality without requiring LiDAR data, 3D motion annotations, or human body templates.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "dynamic scene rendering"
  - "3D Gaussian Splatting"
  - "SDF"
  - "2D priors"
  - "novel view synthesis"
date: 2026-05-08
content_hash: d3d07e64f384af8c
---

# Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering

**Conference**: ICCV 2025
**arXiv**: [2510.13381](https://arxiv.org/abs/2510.13381)  
**Code**: [GitHub](https://dynamic-ugsdf.github.io/)  
**Area**: Autonomous Driving
**Keywords**: dynamic scene rendering, 3D Gaussian Splatting, SDF, 2D priors, novel view synthesis

## TL;DR

This paper proposes UGSDF, a method that jointly learns an SDF network and 3D Gaussian Splatting to model dynamic objects in urban scenes. Using only 2D priors (a depth network and a point tracker), UGSDF achieves state-of-the-art rendering quality without requiring LiDAR data, 3D motion annotations, or human body templates.

## Background & Motivation

Reconstruction and rendering of dynamic urban scenes is a core requirement for autonomous driving simulation, with direct implications for 3D detection, motion planning, and safety-critical scenario simulation. Existing methods suffer from the following limitations:

**Over-reliance on 3D annotations**: Scene graph methods (e.g., OmniRe) require 3D bounding box tracklets, LiDAR data, and even SMPL human body templates, all of which are costly to obtain.

**Trade-offs among NeRF, 3DGS, and SDF**: NeRF/3DGS achieves high visual quality but limited geometric accuracy; SDF is geometrically precise but requires dense representations to reach comparable visual fidelity.

**Limitations of dynamic decomposition methods**: Methods such as S3Gaussians model all dynamic objects with a single dynamic field, lacking fine-grained control.

The authors' core insight is that class-agnostic 2D priors (depth estimation + point tracking) combined with a dual SDF+3DGS representation can match or surpass methods that rely on 3D annotations, without requiring any such annotations.

## Method

### Overall Architecture

UGSDF takes as input an RGB image sequence together with camera parameters and dynamic object masks, with optional LiDAR input. The core pipeline consists of four steps:
1. **Canonical space construction**: UniDepth and CoTracker are used to align multi-frame point clouds of dynamic objects into a canonical coordinate system.
2. **Motion modeling**: Gaussian motion is represented as a learned linear combination of basis trajectories.
3. **Joint SDF+3DGS representation**: SDF governs geometric accuracy while 3DGS handles high-fidelity rendering.
4. **Bidirectional guided optimization**: SDF guides Gaussian densification/pruning, while Gaussians guide SDF ray sampling.

### Key Designs

1. **2D-prior-based dynamic object modeling**: SAM2 generates per-frame segmentation masks; CoTracker tracks pixel trajectories; UniDepth estimates metric depth. A canonical-frame point cloud is constructed via back-projection and cross-frame warping, entirely avoiding dependence on 3D tracklets or SMPL templates. The core back-projection formula is:

$$\mathbf{x}_i = \mathbf{D}_t^c(\boldsymbol{p}) \times (\mathbf{K}^c)^{-1} \tilde{\boldsymbol{p}}_i$$

2. **SDF deformation network**: Models the geometry of dynamic objects. The network comprises three sub-modules: a deformation network $\varphi_{def}$ that maps observed points to canonical space, a topology-aware network $\varphi_{hyp}$ that handles topological changes (e.g., pedestrian motion), and a multi-resolution feature grid $\mathcal{V}$ that encodes geometric detail. SDF values are obtained via $S_i = \varphi_{sdf}(\mathbf{v}_i, \mathbf{w}_{i,t})$. A learnable deformation code $\mathbf{z}_t$ is introduced to adapt the network to deformations at different timesteps.

3. **Bidirectional SDF–Gaussian guidance**: This is the central contribution of the method.

    - **SDF → Gaussian densification**: The space around each object is divided into an $N^3$ voxel grid, and the SDF value at each grid center is queried. A low SDF value indicates proximity to the surface; if the number of Gaussians in that voxel is below threshold $\tau_n$, new Gaussians are sampled from the back-projected depth map for densification.
    - **SDF → Gaussian pruning**: Whether a Gaussian has drifted away from the surface is determined by accumulating SDF values across multiple timesteps:

$$\sum_{t} \exp\left(\frac{-S_i(t) + \sum_{j \in \text{NN}(i)} S_j(t)}{\gamma}\right) < \tau_{pr}$$

    - **Gaussian → SDF ray sampling**: The depth map $\hat{\mathbf{D}}_t$ rendered by Gaussians is used to narrow the SDF ray sampling range, improving surface reconstruction accuracy.

4. **Gaussian motion representation**: Motion is modeled as a learned linear combination of basis trajectories, with position and rotation weighted by shared motion coefficients:

$$\boldsymbol{\mu}(t) = \boldsymbol{\mu}_o + \sum_{j=1}^B \mathbf{c}_j(t) \mathbf{b}_j^\mu(t)$$

A sparsity penalty is imposed to encourage generalization using only a small number of basis trajectories.

### Loss & Training

The SDF loss comprises an RGB rendering loss $\mathcal{L}_{rgb}$, a depth loss $\mathcal{L}_d$, a free-space loss $\mathcal{L}_{fs}$, an Eikonal regularization $\mathcal{L}_{eik}$, and a smoothness loss $\mathcal{L}_{sm}$. The 3DGS loss includes per-frame color, depth, and mask L2 losses, along with motion constraints based on 2D tracking and depth. The two representations are trained in alternation through joint iterative optimization.

## Key Experimental Results

### Main Results

**Waymo Open Dataset (NOTR split)**

| Method | Input | Recon. PSNR↑ | Recon. SSIM↑ | NVS PSNR↑ | NVS SSIM↑ | NVS LPIPS↓ |
|--------|-------|-------------|-------------|----------|----------|-----------|
| StreetGS | M,T | 29.11 | 0.921 | 25.71 | 0.764 | 0.218 |
| OmniRe | T,M,S | 33.79 | 0.942 | 29.35 | 0.780 | 0.186 |
| **UGSDF** | M,PT | **33.98** | **0.944** | **30.63** | **0.871** | **0.129** |
| UGSDF w/o LiDAR | M,PT | 33.88 | 0.942 | 30.32 | 0.871 | 0.145 |

**Waymo: Vehicles vs. Pedestrians**

| Method | Ped. PSNR (Recon.) | Veh. PSNR (Recon.) | Ped. PSNR (NVS) | Veh. PSNR (NVS) |
|--------|-------------------|-------------------|----------------|----------------|
| OmniRe | 28.15 | 28.91 | 24.36 | 27.57 |
| **UGSDF** | 27.89 | **30.34** | **25.48** | **28.68** |

### Ablation Study

| Configuration | Ped. PSNR↑ | Veh. PSNR↑ | Notes |
|---------------|-----------|-----------|-------|
| Full | 27.89 | 30.34 | Complete model |
| w/o SG4GP | 22.47 | 22.27 | Removing SDF-guided Gaussian distribution causes severe degradation |
| w/ Sparse | 24.82 | 25.14 | Sparser representation; pedestrian quality degrades more |
| w/o GPS4S | 25.82 | 27.83 | Removing Gaussian-guided SDF sampling |

### Key Findings

- Even without LiDAR, UGSDF outperforms OmniRe, which relies on 3D tracklets and SMPL templates.
- SDF-guided Gaussian distribution (SG4GP) is the most critical component; its removal causes a PSNR drop of over 5 dB.
- Maintaining a dense representation is especially important for thin objects such as pedestrians and cyclists.
- UGSDF substantially outperforms OmniRe on the vehicle category (+1.4 PSNR) and is only marginally lower on pedestrians.
- The method also supports scene editing tasks including object removal, scene decomposition, and scene composition.

## Highlights & Insights

- **Replacing 3D annotations with 2D priors**: This paradigm is highly practical, significantly reducing data acquisition costs and offering important scalability benefits for autonomous driving simulation.
- **Complementary dual representation of SDF+3DGS**: The method creatively leverages SDF geometric precision to improve the Gaussian distribution quality of 3DGS, while using 3DGS to accelerate SDF training, forming a positive feedback loop.
- **Robust modeling of non-rigid objects**: The topology-aware network and motion basis trajectories enable handling of complex motions exhibited by cyclists and pedestrians.
- This is the first work to combine SDF and 3DGS for modeling individual dynamic objects in dynamic urban scenes.

## Limitations & Future Work

- The method is sensitive to the quality of 2D tracking by CoTracker; tracking failures degrade motion estimation.
- The expressive capacity of the SDF network is inferior to that of SMPL templates, resulting in slightly lower performance on the pedestrian category compared to OmniRe.
- Novel view synthesis quality degrades for viewpoints far from the training trajectory, a limitation shared by all methods.
- Future work may explore incorporating priors from video generative models to enhance reconstruction quality.

## Related Work & Insights

- The method builds upon scene graph approaches (OmniRe) and deformation-based methods (DeformGS) while simplifying input requirements.
- The joint SDF+3DGS learning paradigm is potentially transferable to indoor scene reconstruction.
- The paradigm of replacing 3D annotations with 2D priors may reshape annotation workflows in autonomous driving data pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to jointly employ SDF and 3DGS for dynamic urban scenes; the bidirectional guidance mechanism is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on both Waymo and KITTI datasets with thorough ablations and rich qualitative results.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear and figures are intuitive, though some equations involve heavy notation.
- **Value**: ⭐⭐⭐⭐ — Substantially reduces annotation requirements for dynamic scene reconstruction, offering high practical value for autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[ICCV 2025\] Extrapolated Urban View Synthesis Benchmark](extrapolated_urban_view_synthesis_benchmark.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)
- [\[ICCV 2025\] CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving](coda-4dgs_dynamic_gaussian_splatting_with_context_and_deformation_awareness_for_.md)

</div>

<!-- RELATED:END -->
