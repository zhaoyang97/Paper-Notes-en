---
title: >-
  [Paper Note] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction
description: >-
  [AAAI 2026][3D Vision][sparse-view] SparseSurf is proposed to achieve simultaneous high-accuracy surface reconstruction and high-quality novel view synthesis under sparse-view settings via Stereo Geometry-Texture Alignment and Pseudo-Feature Enhanced Geometry Consistency, attaining state-of-the-art performance on the DTU, BlendedMVS, and Mip-NeRF360 datasets.
tags:
  - AAAI 2026
  - 3D Vision
  - sparse-view
  - surface reconstruction
  - Gaussian splatting
  - stereo matching
  - multi-view consistency
date: 2026-05-08
content_hash: f0384705df27d2ce
---

# SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2511.14633](https://arxiv.org/abs/2511.14633)
**Code**: [Project Page](https://miya-oi.github.io/SparseSurf-project)
**Area**: 3D Vision
**Keywords**: sparse-view, surface reconstruction, 3D Gaussian splatting, stereo matching prior, pseudo-view consistency

## TL;DR

SparseSurf is proposed to achieve simultaneous high-accuracy surface reconstruction and high-quality novel view synthesis under sparse-view settings, via Stereo Geometry-Texture Alignment (SGTA) and Pseudo-Feature Enhanced Geometry Consistency (PFEGC).

## Background & Motivation

Reconstructing accurate 3D surface geometry from sparse-view images is a long-standing challenge. While 3D Gaussian Splatting has achieved strong results in dense-view surface reconstruction, **the optimization process tends to overfit when input views are sparse**, leading to severe quality degradation.

### Core Motivation

Existing sparse-view methods face the following key contradictions:

**Contradiction between flattened Gaussians and overfitting**: To better conform to surface geometry, methods such as FatesGS flatten Gaussians into 2D planar primitives. However, the resulting **high anisotropy** under sparse views **exacerbates overfitting** risk — the reconstruction may appear normal from training views but degrades severely from novel viewpoints.

**Scale ambiguity in monocular depth priors**: Existing methods commonly employ monocular depth estimators for geometric constraints, but monocular depth suffers from **scale ambiguity** and lacks confidence estimation, which introduces noise under sparse views and leads to multi-view inconsistency.

**Disconnect between rendering quality and geometric accuracy**: Existing NVS methods prioritize rendering quality with loose geometric constraints, while surface reconstruction-focused methods often sacrifice rendering quality.

### Core Insights

- **Stereo geometry priors** can provide metric-level supervision signals that are more reliable than monocular depth priors.
- As rendering quality improves during training, stereo rendering quality improves as well, yielding more accurate priors and forming a **virtuous cycle**.
- Combining feature consistency across training views and pseudo-views can effectively mitigate the overfitting caused by flattened Gaussians.

## Method

### Overall Architecture

SparseSurf consists of two core modules:
1. **Stereo Geometry-Texture Alignment (SGTA)**: Generates metric-level depth and normal priors via stereo matching.
2. **Pseudo-Feature Enhanced Geometry Consistency (PFEGC)**: Mitigates overfitting through multi-view feature consistency across training views and pseudo-views.

### Key Designs

#### 1. Stereo Geometry-Texture Alignment (SGTA)

**Function**: Obtains metric-level depth and normal priors from rendered stereo image pairs to supervise the geometric structure of Gaussians.

**Mechanism**:
- For each training viewpoint $\mathbf{P}_i$, a stereo viewpoint with horizontal baseline $b$ is generated.
- The stereo viewpoint image is rendered and paired with the original view image as input to a pretrained stereo matching network (Foundation Stereo).
- Depth $\mathcal{D}^*$ is converted from the disparity map, and normals $\mathcal{N}^*$ are computed accordingly.

**Key Formulas**:

The stereo geometry supervision loss consists of four components:

$$\mathcal{L}_{stereo} = (\lambda_d \mathcal{L}_{depth} + \lambda_n \mathcal{L}_{normal} + \lambda_{nd} \mathcal{L}_{nd}) \mathcal{M}^* + \lambda_s \mathcal{L}_{smooth}$$

- $\mathcal{L}_{depth} = \mathcal{L}_1(D, \mathcal{D}^*)$: depth L1 loss
- $\mathcal{L}_{normal} = 1 - \text{Cosine}(N, \mathcal{N}^*)$: cosine alignment between rendered and stereo normals
- $\mathcal{L}_{nd} = 1 - \text{Cosine}(N_d, \mathcal{N}^*)$: alignment between depth-derived normals and stereo normals
- $\mathcal{L}_{smooth}$: edge-aware Laplacian smoothness loss
- $\mathcal{M}^*$: reliability mask generated via stereo view consistency checking

**Design Motivation**:
- Stereo matching produces **metric-level depth**, avoiding the scale ambiguity of monocular depth.
- The consistency mask filters unreliable pixels to prevent erroneous supervision from rendering noise.
- Stereo priors are updated periodically (every 300 iterations) during training, establishing a rendering→prior virtuous cycle.

#### 2. Pseudo-Feature Enhanced Geometry Consistency (PFEGC)

**Function**: Alleviates overfitting of flattened Gaussians under sparse views through multi-view feature consistency constraints from pseudo-views.

**Mechanism**: Comprises two sub-modules — pseudo-view feature consistency and training-view feature alignment.

**Pseudo-view Feature Consistency**:
1. A frozen feature extractor (Vis-MVSNet) extracts features $\mathcal{F}^*$ from GT images.
2. Each Gaussian is augmented with 8-dimensional feature attributes, learned via feature distillation loss: $\mathcal{L}_f = 1 - \text{Cosine}(F, \mathcal{F}^*)$
3. Feature maps are rendered at random pseudo-viewpoints $\mathcal{V}_p$, and feature consistency is verified via bidirectional warping.
4. **Patch-level** cosine similarity (rather than pixel-level) is adopted to prevent low-fidelity pseudo-view regions from contaminating training-view features:

$$\mathcal{L}_{pseudo} = \sum_{i,j} \mathcal{M}_{feat}^{(i,j)} [1 - \text{Cosine}(\bar{\mathcal{F}}_{p2t}^{(i,j)}, \bar{\mathcal{F}}_r^{(i,j)})]$$

**Training-view Feature Alignment**:
- Pixel-level feature consistency is enforced between ground-truth training views: $\mathcal{L}_{train} = 1 - \text{Cosine}(\mathcal{F}_{s2t}, \mathcal{F}_s)$

**Design Motivation**:
- Pseudo-views supplement the insufficient coverage of sparse training views, though their rendering quality may be lower.
- Patch-level rather than pixel-level consistency provides robustness against rendering noise in pseudo-views.
- Binary confidence mask $\mathcal{M}_{feat}$ further filters unreliable regions.
- Pixel-level constraints between training views provide stronger geometric consistency guarantees.

### Loss & Training

Overall loss function (7,000 iterations, single RTX 3090):

$$\mathcal{L} = \mathcal{L}_c + \mathcal{L}_{stereo} + \lambda_1 \mathcal{L}_f + \lambda_2 \mathcal{L}_{pseudo} + \lambda_3 \mathcal{L}_{train} + \lambda_4 \mathcal{L}_s + \lambda_5 \mathcal{L}_{dn}$$

Different losses are activated at different stages:
- $\mathcal{L}_c, \mathcal{L}_s, \mathcal{L}_f$: activated from iteration 0
- $\mathcal{L}_{stereo}$: activated from iteration 500 (after rendering quality improves)
- $\mathcal{L}_{pseudo}, \mathcal{L}_{dn}$: activated from iteration 3,000

## Key Experimental Results

### Main Results

**DTU Surface Reconstruction (Chamfer Distance↓, little-overlap setting, 3 views)**:

| Method | Type | Mean CD↓ |
|---|---|---|
| COLMAP | Traditional MVS | 2.61 |
| NeuSurf | Neural Implicit | 1.35 |
| FatesGS | GS Surface Recon. | 1.37 |
| UFORecon | Generalizable Implicit | 1.40 |
| **SparseSurf** | **Ours** | **1.05** |

**DTU Surface Reconstruction (large-overlap setting, 3 views)**:

| Method | Mean CD↓ |
|---|---|
| FatesGS | 0.92 |
| NeuSurf | 0.99 |
| UFORecon | 0.99 |
| **SparseSurf** | **0.89** |

**DTU Sparse-View Novel View Synthesis (NVS)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | AVGE↓ |
|---|---|---|---|---|
| CoR-GS | 19.21 | 0.853 | 0.119 | 0.082 |
| Binocular3DGS | 20.71 | 0.862 | 0.111 | — |
| NexusGS | 20.21 | 0.869 | 0.102 | 0.071 |
| **SparseSurf** | **21.31** | **0.886** | **0.089** | **0.067** |

### Ablation Study

Ablation on DTU (large-overlap setting):

| Configuration | Accuracy↓ | Completion↓ | Average↓ | Note |
|---|---|---|---|---|
| Baseline (no extra losses) | 1.318 | 2.302 | 1.810 | RGB supervision only |
| + $\mathcal{L}_{stereo}$ | 0.822 | 1.612 | 1.217 | Stereo depth yields large improvement |
| + $\mathcal{L}_{pseudo}$ | 0.610 | 1.327 | 0.969 | Pseudo-views further reduce overfitting |
| + $\mathcal{L}_{train}$ (full) | **0.533** | **1.239** | **0.886** | Training-view alignment enhances robustness |

### Key Findings

1. **Stereo depth contributes the most**: Mean CD decreases from 1.810 to 1.217 (33% improvement).
2. **Progressive module stacking yields consistent gains**: Each module demonstrates clear independent contribution.
3. **Robustness to different stereo matching networks**: Both Foundation Stereo and Stereo Anywhere are effective.
4. **Insensitivity to baseline selection**: Baselines of 3%, 7%, and 10% of scene radius all perform well.
5. **Simultaneous SOTA on reconstruction and rendering**: Unlike prior methods that trade off between the two objectives.

## Highlights & Insights

1. The idea of **replacing monocular depth with stereo depth** is insightful — leveraging the rendering capability of 3DGS to generate stereo pairs for metric-level priors.
2. **Virtuous cycle design**: improved rendering quality → more accurate stereo priors → better geometric optimization → further improved rendering quality.
3. The choice of **patch-level vs. pixel-level feature consistency** reflects a deep understanding of noise in pseudo-view rendering.
4. **Both reconstruction and rendering objectives are addressed simultaneously**, unlike prior methods that focus on only one.

## Limitations & Future Work

1. Reliance on pretrained stereo matching networks and feature extractors introduces additional computational overhead at inference time.
2. Performance under extreme sparsity (e.g., 2 views) remains to be validated.
3. The pseudo-view generation strategy (interpolated from training camera positions) may lack flexibility.
4. TSDF fusion for mesh extraction introduces an additional post-processing step.

## Related Work & Insights

- **FatesGS** (AAAI25): Sparse-view surface reconstruction using flattened Gaussians and monocular depth; the most direct baseline for comparison.
- **GS2Mesh**: The most closely related prior work, which also uses stereo matching for mesh extraction but fails under sparse views due to rendering quality degradation.
- **CoR-GS** (ECCV24): A representative method that alleviates sparse-view issues through training process optimization.
- **2DGS**: Foundational method for flat Gaussian surface reconstruction.
- **NeuSurf**: State-of-the-art neural implicit surface reconstruction under sparse views.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The bootstrapped stereo prior update mechanism and hierarchical feature consistency design represent clear innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, two sparsity settings, comprehensive ablation and supplementary experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation analysis is thorough and method exposition is clear.
- **Value**: ⭐⭐⭐⭐⭐ — Achieving simultaneous SOTA on both reconstruction and rendering under sparse views demonstrates strong practical applicability.

---

# SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2511.14633](https://arxiv.org/abs/2511.14633)
**Code**: [Project Page](https://miya-oi.github.io/SparseSurf-project)
**Area**: 3D Vision
**Keywords**: sparse-view, surface reconstruction, Gaussian splatting, stereo matching, multi-view consistency

## TL;DR

SparseSurf is proposed to achieve simultaneous high-accuracy surface reconstruction and high-quality novel view synthesis under sparse-view settings via Stereo Geometry-Texture Alignment and Pseudo-Feature Enhanced Geometry Consistency, attaining state-of-the-art performance on the DTU, BlendedMVS, and Mip-NeRF360 datasets.

## Background & Motivation

3D Gaussian Splatting enables efficient high-quality surface reconstruction under dense views, but is prone to overfitting under sparse views, leading to severe quality degradation. Existing methods face two key challenges:

**Challenge 1: Flattened Gaussians exacerbate overfitting**
- Recent methods (FatesGS, Sparse2DGS) adopt flattened 2D Gaussian primitives to better conform to surface geometry.
- However, flattening increases anisotropy, which in turn exacerbates the overfitting risk under sparse views.
- The reconstruction appears acceptable from training views but degrades severely from novel viewpoints.

**Challenge 2: Limitations of monocular depth priors**
- Existing methods use monocular depth estimation as geometric constraints.
- However, monocular depth suffers from scale ambiguity and lacks confidence estimation.
- Under sparse views, noise-induced multi-view inconsistency becomes more severe.

The core insight of the authors is that stereo matching should be leveraged to provide metric-level supervision, while multi-view feature consistency should be used to alleviate overfitting, enabling surface reconstruction and novel view synthesis to improve synergistically.

## Method

### Overall Architecture

SparseSurf is built upon flattened 3DGS (similar to PGSR/GaussianSurfels) and consists of two core modules:
1. **Stereo Geometry-Texture Alignment**: Stereo view pairs are rendered and fed into a pretrained stereo matching network to obtain metric-level depth priors.
2. **Pseudo-Feature Enhanced Geometry Consistency**: Multi-view feature consistency is enforced across training views and pseudo-unseen views.

### Key Designs

#### 1. **Stereo Geometry-Texture Alignment**: Bridging Rendering Quality and Geometric Estimation

The core idea is to exploit the interpolation rendering capability of 3DGS to render stereo view pairs, from which a pretrained stereo matching network yields accurate metric-level geometric priors.

**Stereo Prior Estimation**:
- For each training camera pose $\mathbf{P}_i$, a stereo viewpoint is generated at a horizontal baseline $b$.
- The rendered stereo view image is paired with the original view image and fed into a pretrained stereo matching network to obtain a disparity map.
- Depth $\mathcal{D}^*$ is recovered from the disparity using the known baseline and focal length.
- Normals $\mathcal{N}^*$ are computed from the depth map.
- A reliability mask $\mathcal{M}^*$ is generated via stereo view consistency checking to filter unreliable pixels.
- Priors are re-rendered and updated periodically during training (every 300 iterations).

**Stereo Geometry Supervision**:
$$\mathcal{L}_{depth} = \mathcal{L}_1(D, \mathcal{D}^*)$$
$$\mathcal{L}_{normal} = 1 - \mathcal{C}osine(N, \mathcal{N}^*)$$
$$\mathcal{L}_{nd} = 1 - \mathcal{C}osine(N_d, \mathcal{N}^*)$$

An edge-aware Laplacian smoothness loss is further introduced:
$$\mathcal{L}_{smooth} = \mathcal{S}mooth(N, \mathcal{N}^*) + \mathcal{S}mooth(N_d, \mathcal{N}^*)$$

Total stereo loss:
$$\mathcal{L}_{stereo} = (\lambda_d \mathcal{L}_{depth} + \lambda_n \mathcal{L}_{normal} + \lambda_{nd} \mathcal{L}_{nd})\mathcal{M}^* + \lambda_s \mathcal{L}_{smooth}$$

**Design Motivation**: As training progresses, improved rendering quality leads to more accurate stereo depth priors, which in turn provide better geometric supervision and further improve rendering quality, forming a positive feedback loop.

#### 2. **Pseudo-Feature Enhanced Geometry Consistency**: Alleviating Overfitting

This module consists of two sub-modules:

**Pseudo-view Feature Consistency**:
- Feature attributes are augmented into each Gaussian primitive, learning multi-view feature representations from a frozen feature extraction model via feature distillation.
- Feature distillation loss: $\mathcal{L}_f = 1 - \mathcal{C}osine(F, \mathcal{F}^*)$
- Feature maps are rendered at random pseudo-viewpoints, and a confidence mask is generated via bidirectional warping to measure feature discrepancy.
- Patch-level cosine similarity is adopted to avoid pixel-level noise contamination:

$$\mathcal{L}_{pseudo} = \sum_{i,j} \mathcal{M}_{feat}^{(i,j)} [1 - \mathcal{C}osine(\bar{\mathcal{F}}_{p2t}^{(i,j)}, \bar{\mathcal{F}}_r^{(i,j)})]$$

**Training-view Feature Alignment**:
- High-confidence features from training views are used to reinforce multi-view consistency at the pixel level.
- $\mathcal{L}_{train} = 1 - \mathcal{C}osine(\mathcal{F}_{s2t}, \mathcal{F}_s)$

This joint constraint combining sparse training views and pseudo-unseen views effectively mitigates the overfitting of flattened Gaussians under sparse-view settings.

#### 3. **Multi-view Feature Representation**: Efficient Feature Distillation

8-dimensional multi-view features are extracted using Vis-MVSNet. A key design choice is encoding features into Gaussian attributes, avoiding the computational overhead of re-extracting pseudo-view features at each iteration and keeping the overall pipeline efficient.

### Loss & Training

The total training loss includes rendering loss, stereo loss, and feature consistency loss. Stereo priors are introduced from iteration 500 and updated every 300 iterations, enabling progressive geometric guidance throughout training.

## Key Experimental Results

### Main Results (DTU Surface Reconstruction — Chamfer Distance↓)

| Method | little-overlap | large-overlap | Type |
|------|----------------|---------------|------|
| COLMAP | 2.61 | 1.52 | MVS |
| NeuSurf | 1.35 | 0.99 | Neural Implicit |
| FatesGS | 1.37 | 0.92 | GS Surface Recon. |
| 2DGS | 2.52 | 1.69 | GS Surface Recon. |
| Sparse2DGS | — | 1.13 | GS Surface Recon. |
| **SparseSurf** | **1.05** | **0.89** | GS Surface Recon. |

SparseSurf achieves the best Chamfer Distance under both sparse-view settings on DTU.

### DTU Novel View Synthesis

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | AVGE↓ |
|------|-------|-------|--------|-------|
| CoR-GS | 19.21 | 0.853 | 0.119 | 0.082 |
| Binocular3DGS | 20.71 | 0.862 | 0.111 | — |
| NexusGS | 20.21 | 0.869 | 0.102 | 0.071 |
| **SparseSurf** | **21.31** | **0.886** | **0.089** | **0.067** |

SparseSurf also achieves comprehensive best results in novel view synthesis, demonstrating that surface reconstruction and rendering quality can improve synergistically.

### Ablation Study

| Configuration | Accuracy↓ | Completion↓ | Average CD↓ | Note |
|------|-----------|-------------|-------------|------|
| Baseline (no modules) | 1.318 | 2.302 | 1.810 | Baseline |
| + $L_{stereo}$ | 0.822 | 1.612 | 1.217 | Stereo constraint yields significant improvement |
| + $L_{stereo}$ + $L_{pseudo}$ | 0.610 | 1.327 | 0.969 | Pseudo-views further improve |
| + Full ($L_{train}$) | **0.533** | **1.239** | **0.886** | Training-view alignment provides additional gain |

### Key Findings

1. Stereo priors are the largest performance contributor (CD decreases from 1.810 to 1.217, a 33% reduction).
2. Pseudo-view feature consistency effectively alleviates overfitting (CD from 1.217 to 0.969).
3. Training-view feature alignment provides additional robustness gains (0.969→0.886).
4. Patch-level feature consistency is more robust than pixel-level, preventing noise propagation.

## Highlights & Insights

1. **Synergy between surface reconstruction and rendering**: The traditional trade-off of "better surface fitting → worse rendering" is overcome.
2. **Virtuous cycle design of stereo priors**: improved rendering quality → better stereo priors → better geometry → further improved rendering, achieving self-reinforcement.
3. **Feature-level supervision of pseudo-views**: Multi-view feature consistency constraints are more effective than prior approaches relying solely on RGB or monocular depth supervision for pseudo-views.
4. **Computational efficiency consideration**: Encoding features into Gaussian attributes avoids the overhead of re-extracting features for pseudo-views at each iteration.
5. **Judicious use of flattened Gaussians**: The overfitting risk introduced by flattening is recognized and mitigated through consistency constraints.

## Limitations & Future Work

1. Reliance on pretrained stereo matching network quality may introduce noisy priors early in training when rendering quality is poor.
2. The pseudo-view generation strategy is relatively simple (near training cameras); more intelligent viewpoint selection strategies could be explored.
3. Computational overhead: additional stereo matching inference and feature extraction are required.
4. No specific optimization has been made for large-scale scenes (e.g., Mip-NeRF360 outdoor scenes).
5. The 3-view sparse setting is fixed; performance across varying degrees of sparsity has not been explored.

## Related Work & Insights

- **GS2Mesh**: The most closely related work, using stereo matching to extract meshes from 3DGS; ineffective under sparse views.
- **FatesGS/Sparse2DGS**: Surface reconstruction methods with flattened Gaussians; SparseSurf identifies their overfitting issues.
- **DNGaussian**: A depth regularization method whose geometric constraints are too loose for accurate surface reconstruction.
- **Inspiration**: Stereo matching as geometric supervision for 3DGS is a promising direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The stereo self-reinforcing prior and feature-level pseudo-view consistency designs are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, two sparsity settings, comprehensive ablation and comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation analysis is thorough and method derivation is clear.
- **Value**: ⭐⭐⭐⭐ — Sparse-view surface reconstruction has broad application demands.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[AAAI 2026\] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](fantasystyle_controllable_stylized_distillation_for_3d_gaussian_splatting.md)

<!-- RELATED:END -->
