---
title: >-
  [Paper Note] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion
description: >-
  [ICCV 2025][3D Vision][City-scale 3D generation] This paper presents Sat2City, the first 3D generation framework capable of simultaneously producing city-scale geometry and appearance from a single satellite image. By in…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "City-scale 3D generation"
  - "satellite imagery"
  - "sparse voxel grids"
  - "cascaded latent diffusion"
  - "appearance modeling"
date: 2026-05-08
content_hash: da427a9a0f7e51e0
---

# Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion

**Conference**: ICCV 2025
**arXiv**: [2507.04403](https://arxiv.org/abs/2507.04403)  
**Code**: [ai4city-hkust/Sat2City](https://ai4city-hkust.github.io/Sat2City/)  
**Area**: 3D Vision
**Keywords**: City-scale 3D generation, satellite imagery, sparse voxel grids, cascaded latent diffusion, appearance modeling

## TL;DR

This paper presents Sat2City, the first 3D generation framework capable of simultaneously producing city-scale geometry and appearance from a single satellite image. By integrating sparse voxel grids with a cascaded latent diffusion model, it introduces a Re-Hash multi-scale feature grid and an inverse sampling strategy, achieving high-fidelity generation superior to existing methods on a self-constructed 3D city dataset.

## Background & Motivation

**City-scale 3D scene generation** has broad application demands in gaming, urban planning, and digital twins. Existing methods suffer from three core bottlenecks:

**2D rendering methods lack genuine 3D structure**: Neural rendering approaches such as CityDreamer generate street-view images via GANs or diffusion models, but can only render from limited viewpoints and cannot explicitly reconstruct 3D structures. The absence of direct supervision over 3D texture coordinates leads to severe 3D geometric ambiguity.

**3D methods fail to scale to city level**: Although Sat2Scene first applied diffusion models to 3D point cloud color generation, it relies on predefined dense point clouds (~400 points/m²), incurring prohibitive computational costs and lacking geometric refinement, limiting applicability to street-level scenes.

**Scarcity of high-quality city-scale 3D data**: The lack of city-scale 3D training data with both high-quality geometry and appearance constrains the training and evaluation of methods.

**Core insight**: Works such as XCube demonstrate great potential in combining sparse voxel grids with latent diffusion models for large-scale outdoor scene generation, yet existing methods handle geometry only while neglecting appearance. Sat2City aims to encode appearance as voxel color attributes and achieves joint generation of geometry and appearance through three key technical contributions.

## Method

### Overall Architecture

Sat2City employs a **two-stage training paradigm** (VAE first, then diffusion model) and **three-level cascaded inference** (dense geometry → sparse geometry → appearance):

- **Input**: Satellite height map, lifted to point cloud $P_h$
- **3D training data**: Colored point cloud $P_C \in \mathbb{R}^{N \times 6}$, voxelized into sparse voxel grid $G$
- **Output**: 3D city model with geometric structure and textured appearance

### Key Design 1: Triplet Bottleneck VAE

Conventional VAEs use a single sparse latent variable, which is insufficient for jointly encoding geometry and appearance. Sat2City introduces three distinct bottleneck structures:

1. **Dense Neck**: Expands the sparse encoding into a dense volume, enabling the diffusion model to explicitly distinguish occupied from unoccupied regions. This is critical for correcting spurious voxels introduced by noisy height maps. Densification is performed at the bottleneck layer rather than the input layer, keeping computational costs manageable.

2. **Sparse Neck**: A standard sparse encode–decode structure that directly decodes geometry and normal attributes. The sparse latent $X_S$ serves dual purposes: (a) providing structured pruning guidance for the appearance grid during training; and (b) eliminating redundant voxels not suppressed by dense decoding.

3. **Re-Hash Neck** — the paper's most central contribution. A hierarchical coarsening mechanism iteratively resamples the sparse feature grid to construct a multi-level representation:

$$v_n = 2^n v_0, \quad o_n = \frac{v_n}{2}$$

At each level, features are sampled from the previous level via trilinear interpolation:

$$X_{Cn} = \text{Tri}(G_{Cn}, X_{Cn-1})$$

The multi-level structure preserves fine appearance details while providing global contextual information, which is essential for smooth appearance transitions.

**Two-stage appearance training**: Geometry is trained first (sparse VAE encode–decode); after epoch $E$, the geometry encoder is frozen, the finest-level appearance latent $X_{C0}$ is initialized with $X_S$, and the appearance decoder $\mathcal{D}_c$ is then trained.

### Key Design 2: Inverse Sampling

Directly assigning point cloud colors to voxel grid vertices presents a dilemma:

- **Nearest-neighbor assignment**: Lacks smoothness, causing color discontinuities.
- **Trilinear splatting**: Conflicting color contributions from multiple overlapping points.

Sat2City adopts an **inverse sampling** strategy: rather than directly learning per-vertex color attributes $A_C$, implicit supervision is applied at input point cloud positions. During training, for each level $X_{Cn}$ of the appearance latent hierarchy, the decoded per-vertex color features are trilinearly sampled at the colored point cloud positions $P_C$; features from all levels are concatenated and fed to an MLP to produce estimated colors:

$$\tilde{P}_C = \text{MLP}\{\oplus_{k=0}^n \text{Tri}(P_C, \mathcal{D}_c(X_{Cn}))\}$$

During inference, colors are similarly sampled at the predicted grid vertices $\tilde{G}$:

$$\tilde{A}_C = \text{MLP}\{\oplus_{k=0}^n \text{Tri}(\tilde{G}, \mathcal{D}_c(X_{Cn}))\}$$

### Key Design 3: Conditional Cascaded 3D Latent Diffusion

Single-stage latent diffusion cannot handle large-scale 3D scenes. Sat2City employs a three-level sequential conditional diffusion pipeline in which each stage is conditioned on the output of the previous one:

1. **Dense geometry latent diffusion**: Conditioned on the height map $P_h$, generates dense latent $X_D$ encoding the overall spatial layout:
$$p(X_D, G, A_N) = p_{\mathcal{D}_d}(G, A_N | X_D) \cdot p_{\Psi_D}(X_D | c(P_h))$$

2. **Sparse geometry latent diffusion**: Conditioned on $\{G, A_N\}$ decoded from stage one, fits fine surface structures and records voxel pruning decisions $\textit{struct}$:
$$p(X_S, \textit{struct}) = p_{\mathcal{D}_s}(G, A_N | X_S) \cdot p_{\Psi_S}(X_S | G, A_N)$$

3. **Appearance latent diffusion**: Uses $\textit{struct}$ as structural guidance for multi-level appearance generation:
$$p(G, A_N, A_C) = p_{\mathcal{D}_c}(G, A_N, A_C | X_C) \cdot \prod_{n=0}^N p_{\Psi_{Cn}}(X_{Cn} | \textit{struct})$$

### Dataset

City mesh models are created by artists in Blender; 100 million points are sampled to form colored point clouds. Height maps (2268×3423 pixels) covering an area of 2090×3449.4 m² are rendered via orthographic cameras. Training samples are cropped to 300×300 pixels, yielding 3,110 instances (90% training, 10% test/validation).

## Key Experimental Results

### Main Results: Geometry Quality Comparison

| Method | MMD↓ (CD) | MMD↓ (EMD) | COV↑ (CD) | COV↑ (EMD) |
|--------|-----------|------------|-----------|------------|
| NFD (unconditional) | 0.0445 | 0.2363 | 22.66% | 29.66% |
| BlockFusion (unconditional) | 0.0326 | 0.1865 | 50.49% | 55.66% |
| **Sat2City (conditional)** | **0.0165** | **0.1157** | **100.00%** | **60.00%** |

Sat2City outperforms existing methods across all geometry metrics: COV(CD) reaches 100% while MMD(CD) is reduced by 49.4%, indicating stable generation quality with minimal mode collapse.

### Perceptual Quality Evaluation

| Method | TPQ↑ | TSC↑ | GPQ↑ | GSC↑ |
|--------|------|------|------|------|
| Sat2Scene (2D) | 6.17 | 5.90 | - | - |
| CityDreamer (2D) | 6.40 | 6.63 | - | - |
| CityDreamer (3D)* | 4.48 | 4.48 | 3.60 | 3.38 |
| Sat2Scene* (retrained) | 3.18 | 3.30 | 3.03 | 3.02 |
| **Sat2City** | **7.35** | **8.03** | **6.27** | **7.02** |

In a subjective evaluation with 60 participants, Sat2City achieves the highest scores on both texture perceptual quality (TPQ/TSC) and geometry perceptual quality (GPQ/GSC). Notably, the appearance quality of direct 3D generation (TPQ=7.35) even surpasses that of 2D rendering-based methods.

### Ablation Study

| Ablation | Key Findings |
|----------|-------------|
| Bottleneck design | Single-dense and dual-sparse variants exhibit consistently zero color loss throughout training (gradient conflict); Re-Hash converges faster and more stably than dual-dense, with no artifacts |
| Inverse sampling | Without inverse sampling, color splatting produces severe rendering artifacts; inverse sampling ensures vertex colors are constrained by points within each voxel |
| Cascaded diffusion levels | Sparse structure alone fails to capture unoccupied regions, leading to chaotic generation; removing sparse while retaining Re-Hash results in appearance anomalies due to the absence of structured pruning |

## Highlights & Insights

1. **First 3D generation framework to jointly model city-scale geometry and appearance** (the only method unifying all three capabilities).
2. **Elegant Re-Hash design**: Hierarchical coarsening resolves the challenge of encoding smooth appearance on sparse voxels while avoiding the computational cost of dense volumes.
3. **Clever inverse sampling**: Circumvents the dilemma of color assignment during point cloud voxelization by applying implicit supervision at the point cloud level to achieve smooth transitions.
4. Rapid generation of 3D cities from a single satellite image (~1 minute) without auxiliary inputs such as segmentation maps.

## Limitations & Future Work

1. Training data consists of synthetic cities created in Blender; a domain gap remains relative to paired real satellite imagery and 3D Tiles data.
2. The dataset is relatively limited in scale (3,110 instances), potentially constraining generation diversity.
3. Validation is conducted solely on the self-constructed dataset, without comparison against real-world city data.

## Related Work & Insights

- **Object-level 3D generation**: XCube/SCube (sparse voxels + diffusion), 3D latent set diffusion
- **City-level neural rendering**: InfiniCity (GAN + NeRF), CityDreamer (semantic voxel partitioning), Sat2Scene (3D point cloud diffusion)
- **Asset retrieval methods**: Assembling cities from Blender asset libraries, constrained by the diversity of available assets

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 5 |
| Experimental Thoroughness | 3 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)
- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)

</div>

<!-- RELATED:END -->
