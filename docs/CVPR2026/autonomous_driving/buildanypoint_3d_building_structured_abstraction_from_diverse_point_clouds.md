---
title: >-
  [Paper Note] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds
description: >-
  [CVPR 2026][Autonomous Driving][Building abstraction reconstruction] This paper proposes BuildAnyPoint, which employs a **loosely-coupled cascaded diffusion Transformer (Loca-DiT)** to achieve unified reconstruction from…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Building abstraction reconstruction"
  - "point cloud completion"
  - "latent diffusion"
  - "autoregressive mesh generation"
  - "cascaded generation framework"
date: 2026-05-08
content_hash: 8e075f000e9a1819
---

# BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds

**Conference**: CVPR 2026
**arXiv**: [2602.23645](https://arxiv.org/abs/2602.23645)  
**Code**: [Project Page](https://ai4city-hkust.github.io/BuildAnyPoint/)  
**Area**: Autonomous Driving / 3D Vision / Urban Reconstruction
**Keywords**: Building abstraction reconstruction, point cloud completion, latent diffusion, autoregressive mesh generation, cascaded generation framework

## TL;DR

This paper proposes BuildAnyPoint, which employs a **loosely-coupled cascaded diffusion Transformer (Loca-DiT)** to achieve unified reconstruction from diverse point cloud distributions (airborne LiDAR, SfM, sparse noisy point clouds) to structured 3D building meshes — first recovering the underlying point cloud distribution via hierarchical latent diffusion, then generating compact polygonal meshes via an autoregressive Transformer.

## Background & Motivation

**Background**: Recovering lightweight 3D building models from urban point clouds is a critical requirement for applications such as digital twins, navigation, and disaster simulation. Existing approaches include optimization-based methods (plane detection and assembly) and learning-based solutions, but they typically **handle only specific point cloud distributions**.

**Limitations of Prior Work**:
   - Point2Building: pioneered direct autoregressive mesh generation from point clouds, but single-step autoregression frequently produces **geometric ambiguities** and mesh–point cloud misalignment.
   - ArcPro: introduces a building grammar intermediate representation to reduce ambiguity, but is constrained by **predefined geometric primitives** (e.g., cylindrical extrusions), cannot handle complex structures such as sloped roofs, and assumes relatively complete local point clouds for each module.

**Key Challenge**: How can one simultaneously maintain **generalizability** to arbitrary point cloud distributions and ensure **structural consistency and geometric accuracy** of the generated meshes? Directly feeding heterogeneous point clouds into autoregressive mesh generators performs poorly, as such generators require high-quality, clean, and complete point clouds.

**Goal**: To construct the first **unified framework** that recovers structured building abstraction meshes from point clouds of arbitrary distributions (LiDAR, SfM, extremely sparse and noisy).

**Key Insight**: Exploit **explicit 3D generative priors** to constrain the solution space — rather than generating meshes directly from heterogeneous point clouds, first recover the underlying uniform dense point cloud distribution, then pass it to an existing high-quality mesh generator.

**Core Idea**: Loosely-coupled cascade = hierarchical latent diffusion (distribution recovery) + autoregressive Transformer (mesh generation), progressively bridging the modality gap from unstructured point clouds to structured meshes through a series of latent space transformations.

## Method

### Overall Architecture

Loca-DiT (Fig. 3) learns the conditional distribution $p_\text{BAP}(\mathcal{M} | \mathcal{P}_{in})$, decomposed into two stages:

1. **Geometry Completion Stage** (latent diffusion): $p(\mathcal{P}_{out} | \mathcal{P}_{in})$ — recovering a uniform, dense, complete point cloud from sparse/noisy input.
2. **Structured Mesh Generation Stage** (autoregressive Transformer): $p(\mathcal{M} | \mathcal{P}_{out})$ — autoregressively generating mesh token sequences from the recovered point cloud.

### Key Designs

#### 1. Three-Level Latent Space Transformation

- **Function**: Three latent spaces are designed to progressively bridge the representational gap from point clouds to meshes.
- **Mechanism**:
    - **Dense latent grid $\mathcal{G}_d$**: The ground-truth point cloud is low-resolution voxelized and encoded by a sparse VAE; at the bottleneck layer, the sparse grid is **densified** into a dense grid — providing the decoder with complete spatial context to "carve out" unoccupied regions.
    - **Sparse latent grid $\mathcal{G}_s$**: High-resolution voxelization followed by sparse VAE encoding — for refining geometric details.
    - **Serialized tokens $\mathcal{T}_P$**: A pretrained point cloud encoder encodes the recovered point cloud into a fixed-length token sequence — aligned with the target mesh token sequence $\mathcal{T}_M$.
- **Design Motivation**: Point clouds are well-suited for encoding geometric details in continuous dense latent spaces, whereas meshes require discrete serialized representations to generate structural topology — using distinct latent spaces at different stages allows each stage to specialize.

#### 2. Hierarchical Latent Diffusion

- **Function**: Recovers the complete geometric prior of buildings in two levels.
- **Mechanism**:
    - Coarse-level diffusion model $p_{\theta_d}(\mathcal{G}_d | \mathcal{P}_{in})$: denoises on the dense grid to recover the coarse shape.
    - Fine-level diffusion model $p_{\theta_s}(\mathcal{G}_s | \mathcal{G}_d)$: conditioned on the coarse-level output, denoises on the sparse grid to refine high-resolution geometry.
    - Training objective: standard denoising loss $\min_\theta \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t)\|_2^2]$
    - Conditioning: a point cloud encoder quantizes $\mathcal{P}_{in}$ into a voxel grid, which is concatenated to the latent features.
- **Design Motivation**: The hierarchical approach first recovers coarse structure and then refines details, yielding greater training stability than single-level diffusion.

#### 3. Autoregressive Mesh Generation

- **Function**: Generates low-polygon, topologically consistent building meshes conditioned on the recovered point cloud.
- **Mechanism**:
    - Based on a decoder-only Transformer from MeshAnything V2.
    - Input sequence $\mathcal{T} = [\mathcal{T}_P; \mathcal{T}_M^{<t}]$; the model autoregressively predicts the next mesh token.
    - Training objective: maximize the conditional log-likelihood $\max_\phi \sum_{t=1}^N \log P(t_m^N | \mathcal{T}_P, \mathcal{T}_M^{<t}; \phi)$
- **Design Motivation**: The recovered high-quality point cloud with normals simulates the "artist-quality" input required by autoregressive mesh generators.

### Loss & Training

- Sparse VAE: BCE (generated vs. target) + KL divergence + normal learning.
- Diffusion models: denoising MSE loss.
- Transformer: cross-entropy next-token prediction loss.

## Key Experimental Results

### Main Results: Building Structured Abstraction

| Method | #V↓ | #F↓ | #P↓ | FR↓ | CD↓ |
|------|-----|-----|-----|-----|-----|
| City3D (optimization-based) | 173 | 72 | 14 | 6% | 0.167 |
| Point2Building (learning-based) | 20 | 34 | 18 | 1% | 0.043 |
| **BuildAnyPoint** | **10** | **16** | **8** | **0%** | **0.036** |

- **Only 10 vertices** (vs. 20 for P2B), 16 faces (vs. 34), yielding more compact low-polygon meshes.
- Failure rate of 0% and lowest CD.

### Point Cloud Completion Benchmark

| Method | F-score↑ | CD↓ | Uniformity↓ | EMD↓ |
|------|----------|-----|-------------|------|
| PoinTr | 0.85 | 0.41 | 0.25 | 0.12 |
| AnchorFormer | 0.82 | 0.39 | 1.27 | 0.13 |
| **BuildAnyPoint** | **0.91** | **0.35** | **0.04** | **0.10** |

- Uniformity score of 0.04, **nearly an order of magnitude lower** than all competing methods.

### Ablation Study

| Setting | #V↓ | #F↓ | CD↓ |
|------|-----|-----|-----|
| Without 3D generative prior (direct generation from $\mathcal{P}_{in}$) | 78 | 127 | 0.107 |
| **Full model** (generation from $\mathcal{P}_{out}$) | **38** | **70** | **0.034** |

- Removing coarse level $\mathcal{G}_d$: recovered point cloud becomes disordered.
- Removing fine level $\mathcal{G}_s$: a "double surface" artifact appears and misleads subsequent mesh generation.
- Replacing the Transformer with a conventional solver: fails to generate valid surfaces even given a well-recovered point cloud.

### Key Findings

1. **3D generative prior is critical**: Without the prior, CD degrades from 0.034 to 0.107 and face count explodes from 70 to 127.
2. **Both levels of hierarchical diffusion are indispensable**: the coarse level provides basic shape; the fine level provides surface accuracy.
3. **Effective across all three point cloud distributions**: LiDAR, SfM, and sparse noisy — genuinely realizing the "BuildAnyPoint" objective.

## Highlights & Insights

1. **Elegant decoupling**: Rather than attempting end-to-end mesh generation from heterogeneous point clouds in a single step, the problem is decomposed into "distribution recovery" and "mesh generation" — each addressed by the most suitable generative paradigm (diffusion vs. autoregression). This cascaded design philosophy is broadly applicable to other cross-modal generation tasks.
2. **Clever densification bottleneck**: Densifying the sparse grid at the sparse VAE bottleneck enables the decoder to simultaneously observe occupied and unoccupied regions for shape reasoning.
3. **Intermediate output is already state-of-the-art**: The recovered point clouds, serving merely as an intermediate representation within the framework, already achieve SOTA performance on the building point cloud completion benchmark.

## Limitations & Future Work

1. **Limited geometric diversity in datasets**: Publicly available building datasets are skewed toward simple geometries; the framework's capacity to model complex structures (e.g., Gothic architecture, irregular morphologies) remains limited.
2. **No exploitation of height/geographic priors**: Physical constraints (e.g., gravity, symmetry) and geospatial information are not utilized.
3. **Inference speed not reported**: The cascaded pipeline of diffusion sampling and autoregressive decoding may be slow.
4. Evaluation is confined to The Hague/Rotterdam datasets; geographic generalizability is unknown.

## Related Work & Insights

- **MeshAnything series**: The paradigm of autoregressive mesh generation from point clouds is rapidly evolving; this work demonstrates the **decisive influence of input point cloud quality** on mesh quality.
- **XCube**: A hierarchical sparse VAE + diffusion framework for 3D generation; this paper extends it with conditioning and densification operations.
- **Loosely-coupled cascades**: Decomposing the generation process into probabilistically independent sub-stages, each handled by the most suitable generative model — this "divide and conquer" strategy merits broader adoption in 3D generation tasks.

## Rating

⭐⭐⭐⭐ — Elegant framework design with impressive generalization across three point cloud distributions; the intermediate output alone achieves SOTA, though the application domain is relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](../../AAAI2026/autonomous_driving/understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[CVPR 2026\] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](../../ICCV2025/autonomous_driving/robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)
- [\[ICCV 2025\] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration](../../ICCV2025/autonomous_driving/mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](../../ICCV2025/autonomous_driving/a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

</div>

<!-- RELATED:END -->
