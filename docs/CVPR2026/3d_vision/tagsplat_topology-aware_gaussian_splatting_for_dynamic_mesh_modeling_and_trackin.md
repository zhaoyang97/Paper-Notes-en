---
title: >-
  [Paper Note] TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking
description: >-
  [CVPR2026][3D Vision][Gaussian Splatting] TagSplat is a topology-aware Gaussian splatting framework that explicitly encodes spatial connectivity among Gaussian primitives…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "Gaussian Splatting"
  - "Topology Consistency"
  - "Dynamic Mesh Reconstruction"
  - "3D Keypoint Tracking"
  - "Manifold Preservation"
date: 2026-05-08
content_hash: 3143593480570fb8
---

# TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking

**Conference**: CVPR2026
**arXiv**: [2512.01329](https://arxiv.org/abs/2512.01329)
**Code**: [Project Page](https://haza628.github.io/tagSplat/)
**Area**: 3D Vision
**Keywords**: Gaussian Splatting, Topology Consistency, Dynamic Mesh Reconstruction, 3D Keypoint Tracking, Manifold Preservation

## TL;DR

TagSplat is a topology-aware Gaussian splatting framework that explicitly encodes spatial connectivity among Gaussian primitives, enabling the generation of topologically consistent mesh sequences in dynamic scene reconstruction while supporting accurate 3D keypoint tracking.

## Background & Motivation

The core workflow of the animation industry is mesh-based: rendering, skinning, and editing all require topologically consistent triangular meshes. However, existing 4D reconstruction methods face critical challenges:

- **NeRF-based methods** (HyperNeRF, D-NeRF, etc.) adopt implicit representations and cannot impose explicit topological constraints on dynamic objects. Even when Marching Cubes is applied per-frame to extract meshes, inter-frame topological continuity cannot be guaranteed.
- **3DGS-based methods** (Dynamic 3DGS, DG-Mesh, Dynamic 2DGS, etc.) achieve high-quality rendering but reconstruct meshes independently per frame, resulting in topological inconsistency—vertex counts and connectivity vary across frames, making skeleton binding and keypoint tracking infeasible.
- **Topo4D** binds Gaussians to vertices of a static template mesh but relies on high-quality Metahuman templates and is restricted to head models.
- **GauSTAR** uses optical flow to guide reconstruction and tracking, but its preprocessing pipeline is complex and difficult to deploy.

The core bottleneck: existing methods disrupt manifold structure during Gaussian densification and pruning, causing topology to change across training iterations.

## Core Problem

How to preserve manifold topology invariant throughout 3D Gaussian splatting training, so as to produce dynamic mesh sequences with consistent vertex counts and connectivity?

## Method

### Overall Architecture

TagSplat proceeds in four stages:

1. **Initialization**: Reconstruct a high-quality mesh from the first frame using multi-view images, then convert mesh vertices into a topologically structured Gaussian point cloud.
2. **Topology-Aware Optimization**: Perform densification and pruning of Gaussians in canonical space while maintaining manifold topology.
3. **Temporal Consistency Training**: Apply 1-ring-neighborhood-based temporal regularization to subsequent frames to ensure inter-frame topological consistency.
4. **Modeling & Tracking**: Output topologically consistent mesh sequences and support 3D keypoint tracking via barycentric coordinates.

### 3.1 Gaussian Primitive Initialization

Initialization is critical to downstream optimization quality. NeuS2 is applied to the first frame for multi-view geometric reconstruction, yielding a mesh $\mathcal{M}=(\mathcal{V},\mathcal{F})$. Each vertex $\boldsymbol{v}_i$ is then initialized as a Gaussian primitive:

| Parameter | Initialization Strategy |
|-----------|------------------------|
| Position $\boldsymbol{p}_i$ | Directly set to vertex coordinate $\boldsymbol{v}_i$ |
| Color $\boldsymbol{c}_i$ | Vertex RGB color |
| Opacity $\alpha_i$ | Uniformly set to 1 |
| Rotation $\boldsymbol{R}_i$ | Derived from vertex normal $\boldsymbol{n}_i$; normal direction serves as local z-axis |
| Scale $s_{i,x}, s_{i,y}$ | Based on distances to neighboring vertices to ensure coverage |
| Scale $s_{i,z}$ | Set to a small value $\epsilon$ to keep Gaussians surface-aligned |

The resulting Gaussian set is $\mathcal{G}=(\{(\boldsymbol{p}_i,\boldsymbol{R}_i,\boldsymbol{s}_i,\boldsymbol{c}_i,\alpha_i)\}_{i=1}^{|\mathcal{V}|},\mathcal{F})$, where the face index $\mathcal{F}$ is directly inherited from the mesh and defines topological connectivity among Gaussian primitives.

### 3.2 Topology-Aware Densification and Pruning (Core Contribution)

This is the most critical technical innovation of the paper, addressing the problem that standard 3DGS destroys manifold topology during densification and pruning.

**Problem with vanilla 3DGS**: Decisions to add or remove Gaussians are made solely based on local attributes (projected gradients, opacity, scale), with no explicit connectivity between added or removed points. This causes the manifold structure of the geometric surface to deteriorate over iterations.

#### Topology-Preserving Densification

For triangular faces whose projected gradient exceeds a threshold, a new Gaussian point is inserted in parameter space:

- **New point attributes**: Averaged from the Gaussian parameters of the triangle's three vertices.
- **Topology update**: The original triangle is subdivided into three new triangles: $(\mu_0,\mu_1,\mu_{\text{new}})$, $(\mu_1,\mu_2,\mu_{\text{new}})$, $(\mu_2,\mu_0,\mu_{\text{new}})$.
- **Scale handling**: If the average scale $\bar{s} < \tau_s$, the new point is used directly (under-reconstruction region); otherwise, the scales of both the new point and the original vertices are halved (over-reconstruction region), analogous to the clone and split logic in vanilla 3DGS.

#### Topology-Preserving Pruning

Inspired by the edge collapse strategy of Hoppe et al., a cost function tailored for Gaussian topology is designed:

$$C = \omega_g \cdot \|\mathbf{p}_i - \mathbf{p}_j\| + \omega_a \cdot \|\mathbf{c}_i - \mathbf{c}_j\|$$

- $E_g = \|\mathbf{p}_i - \mathbf{p}_j\|$: Geometric error measuring Euclidean distance between centers.
- $E_a = \|\mathbf{c}_i - \mathbf{c}_j\|$: Attribute error measuring color discrepancy.
- The edge with the minimum cost is selected for collapse, minimizing geometric and visual loss during simplification.

For each Gaussian primitive to be pruned, the edge collapse cost with all 1-ring neighbors is computed, and the neighbor with the minimum cost is selected for merging.

**Key constraint**: Densification and pruning are performed exclusively in canonical space. Topology is fixed for all subsequent frames, ensuring topological consistency throughout dynamic reconstruction.

### 3.3 Loss Functions for First-Frame Gaussian Optimization

The first frame is optimized jointly with a multi-term loss:

| Loss Term | Formula | Role | Weight |
|-----------|---------|------|--------|
| Gaussian color loss $\mathcal{L}_c^{gs}$ | $0.8\cdot\|I_{gs}-I_{gt}\|+0.2\cdot\mathcal{L}_{ssim}$ | Align rendered image with ground truth | 1.0 |
| Mesh color loss $\mathcal{L}_c^{mesh}$ | Same, rendered via Nvdiffrast | Improve mesh rendering quality | 1.0 |
| Gaussian mask loss $\mathcal{L}_m^{gs}$ | $0.8\cdot\|I_m-I_{mask}\|+0.2\cdot\mathcal{L}_{ssim}$ | Ensure coverage of the target region | 3.0 |
| Mesh mask loss $\mathcal{L}_m^{mesh}$ | Same | Same | 3.0 |
| 2D scale constraint $\mathcal{L}_{2d}$ | $\sum_{i=1}^k s_z^i$ | Constrain z-axis scale to keep Gaussians surface-aligned | 1.0 |
| Laplacian smoothing $\mathcal{L}_{lap}$ | $\frac{1}{k}\sum\|\boldsymbol{\delta}_i\|^2$ | Suppress surface spike artifacts | 5.0 |
| Normal consistency $\mathcal{L}_n$ | $\sum\|\boldsymbol{n}_i^{gs}-\boldsymbol{n}_i^{mesh}\|$ | Align Gaussian orientations with mesh normals | 1.0 |

The Laplacian smoothing term $\boldsymbol{\delta}_i = \boldsymbol{\mu}_i - \frac{1}{|m|}\sum_{j=1}^m \boldsymbol{\mu}_j$ measures the deviation of each Gaussian from the centroid of its 1-ring neighborhood.

### 3.4 Temporal Consistency Constraints (Subsequent Frame Training)

Three additional topology-based temporal regularization terms are introduced for subsequent frames on top of the first-frame losses:

**Edge length consistency loss** (weight 4.0): Constrains the distance between adjacent Gaussians to remain consistent across frames.

$$\mathcal{L}_{len} = \sum_{i=1}^{k_l} \|l_{t,i} - l_{t-1,i}\|$$

**Rigidity constraint loss** (weight 4.0): Exploits the 1-ring neighborhood to suppress local deformation, ensuring that relative position changes among neighboring Gaussians conform to a local rigidity assumption.

$$\mathcal{L}_{rigid} = \sum_{i=1}^k \sum_{j=1}^m \omega_{i,j} \|(\boldsymbol{\mu}_{j,t-1}-\boldsymbol{\mu}_{i,t-1}) - \Delta\boldsymbol{R}_t(\boldsymbol{\mu}_{j,t}-\boldsymbol{\mu}_{i,t})\|$$

where $\Delta\boldsymbol{R}_t = \boldsymbol{R}_{i,t-1}\boldsymbol{R}_{i,t}^{-1}$, and the weight $\omega_{i,j}=\exp(-\lambda_w \cdot l_{i,j})$ assigns stronger constraints to closer neighbors.

**Rotation consistency loss** (weight 20.0): Ensures that rotation changes of neighboring Gaussians are smooth across frames.

$$\mathcal{L}_{rot} = \sum_{i=1}^k \sum_{j=1}^m \omega_{i,j} \|\boldsymbol{q}_{j,t}\otimes\boldsymbol{q}_{j,t-1}^{-1} - \boldsymbol{q}_{i,t}\otimes\boldsymbol{q}_{i,t-1}^{-1}\|$$

where $\boldsymbol{q}$ denotes normalized quaternions and $\otimes$ denotes quaternion multiplication. The highest weight (20.0) assigned to the rotation consistency loss reflects the importance of rotational smoothness for temporal coherence.

### 3.5 Mesh Modeling and 3D Keypoint Tracking

After training, the positional parameters of Gaussians in each frame directly correspond to the mesh vertex coordinates, with topological connectivity derived from the learned manifold structure. Since all frames share the same topology, any 3D point on the mesh surface can be represented via barycentric coordinates and a triangle face index. This parameterization is universally applicable across all frames, enabling full-sequence tracking of arbitrary target points.

## Key Experimental Results

### Datasets

- **MIX-TAG** (synthetic): 3 dynamic human subjects (Worker/Dancer/Boxer), 42 virtual cameras, 1080×1080 resolution, 90–130 frames, with GT meshes provided.
- **TalkBody4D** (real): 59 calibrated RGB cameras at 20 FPS, 3000×4000 resolution; 30 views are used.

### Main Results (MIX-TAG Dataset, Boxer Scene)

| Method | PSNR↑ | SSIM↑ | CD↓ | Tracking MSE↓ | Topo. Consistent? |
|--------|-------|-------|-----|---------------|-------------------|
| Dynamic 3DGS | 30.56 | 0.97 | ✗ | 0.000676 | ✗ |
| DG-Mesh | 22.46 | 0.95 | 1.48 | 0.013502 | ✗ |
| Deformable-GS | 34.38 | 0.98 | ✗ | 0.006631 | ✗ |
| Dynamic 2DGS | 34.27 | 0.97 | 0.47 | 0.009148 | ✗ |
| **TagSplat** | **34.76** | **0.98** | **0.32** | **0.000569** | **✔** |

In the Dancer scene, TagSplat achieves a tracking MSE of 0.000101, approximately 70% lower than Dynamic 3DGS (0.000329) and approximately 99.8% lower than Dynamic 2DGS (0.042436).

### TalkBody4D Real Dataset (Mesh Rendering Quality)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| DG-Mesh | 24.90 | 0.91 | 0.097 |
| Dynamic-2DGS | 23.58 | 0.90 | 0.081 |
| **TagSplat** | **26.82** | **0.93** | **0.077** |

### Ablation Study

- Removing mesh losses: PSNR drops from 32.90 to 27.70; CD increases from 0.360 to 0.396.
- Removing the topology-preserving D&P module: PSNR drops to 30.22.
- Removing rotation consistency loss: tracking MSE increases from 0.000368 to 0.000372 (smallest impact, but still contributes).
- Removing rigidity loss: tracking MSE increases to 0.000395 (significant impact).

## Highlights & Insights

1. **First method to maintain manifold topology during dynamic Gaussian updates**, enabling truly topologically consistent mesh sequence reconstruction.
2. **Elegantly designed topology-aware densification and pruning**: densification via triangle face subdivision and pruning via minimum-cost edge collapse, simultaneously adding geometric detail while preserving manifold structure.
3. **Significantly superior tracking accuracy**: leveraging topological consistency for 3D keypoint tracking, with MSE one to two orders of magnitude below the second-best method.
4. **End-to-end framework**: from multi-view video to topologically consistent mesh sequences and 3D tracking, without complex preprocessing pipelines.
5. Dual supervision via both Gaussian splatting rendering and differentiable mesh rasterization (Nvdiffrast) enhances geometric accuracy.

## Limitations & Future Work

1. **Cannot handle topological changes**: scenarios such as cloth tearing or object splitting are beyond the current framework, which assumes stable topological relations.
2. **Relies on high-quality first-frame mesh reconstruction**: initialization via NeuS2 means poor first-frame reconstruction quality propagates to all subsequent frames.
3. **Restricted to foreground human bodies**: experiments are conducted on single-person dynamic reconstruction; multi-person or general dynamic scenes are not addressed.
4. **Densification and pruning are performed only in canonical space**: the framework cannot adaptively adjust topological resolution for subsequent frames.
5. Experiments use a single RTX 3090 GPU; training time and inference speed are not reported.

## Related Work & Insights

- **vs. DG-Mesh**: DG-Mesh attaches Gaussians to meshes and automatically reconstructs dynamic meshes, but topology is independent per frame; TagSplat maintains consistent topology across frames.
- **vs. Dynamic 2DGS**: Dynamic 2DGS improves mesh accuracy with 2D Gaussians but still applies per-frame Poisson reconstruction; TagSplat generates consistent meshes directly from Gaussian topology.
- **vs. Topo4D**: Topo4D depends on high-quality Metahuman templates and is limited to head models; TagSplat is more general, starting from a data-driven initial mesh.
- **vs. Dynamic 3DGS**: Dynamic 3DGS achieves acceptable rendering quality but does not output meshes; TagSplat achieves both high-quality rendering and mesh reconstruction.
- **vs. GauSTAR**: GauSTAR requires complex preprocessing (optical flow guidance); TagSplat offers a simpler pipeline.

The topology-preserving densification and pruning strategy is generalizable to other tasks requiring structured Gaussian representations (e.g., editable 3D content generation). The 1-ring-neighborhood temporal regularization approach can inform temporal consistency constraint design in other dynamic reconstruction frameworks. The edge collapse cost function, which jointly considers geometric and color errors, offers a novel perspective on mesh simplification. The tracking strategy based on barycentric coordinates and fixed topology is concise and effective, with direct applicability to downstream tasks such as animation production and motion capture post-processing.

## Rating
- Novelty: ⭐⭐⭐⭐ (Topology-aware Gaussian densification/pruning is a clear and well-motivated innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Synthetic + real datasets with thorough ablations, though speed comparisons are absent)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured with detailed algorithmic descriptions)
- Value: ⭐⭐⭐⭐ (An important step toward bridging 3DGS and the mesh-based workflows of the animation industry)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[CVPR 2026\] EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy](emgauss_continuous_slice-to-3d_reconstruction_via_dynamic_gaussian_modeling_in_v.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)

</div>

<!-- RELATED:END -->
