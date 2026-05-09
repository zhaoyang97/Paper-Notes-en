---
title: >-
  [Paper Note] Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences
description: >-
  [CVPR 2026][3D Vision][dynamic surface reconstruction] Neu-PiG proposes a fast optimization framework based on preconditioned multi-resolution latent grids, encoding the position and normal directions of a keyframe reference mesh into a unified latent space. A lightweight MLP decodes these features into per-frame 6-DoF deformations, achieving high-fidelity dynamic surface reconstruction more than 60× faster than existing training-free methods, without requiring category-specific priors or explicit correspondences.
tags:
  - CVPR 2026
  - 3D Vision
  - dynamic surface reconstruction
  - preconditioned latent grids
  - Sobolev preconditioning
  - multi-resolution voxels
  - deformation estimation
date: 2026-05-08
content_hash: 49107204198b68d0
---

# Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences

**Conference**: CVPR 2026
**arXiv**: [2602.22212](https://arxiv.org/abs/2602.22212)
**Area**: 3D Vision
**Keywords**: dynamic surface reconstruction, preconditioned latent grids, Sobolev preconditioning, multi-resolution voxels, deformation estimation

## TL;DR

Neu-PiG proposes a fast optimization framework based on preconditioned multi-resolution latent grids, encoding the position and normal directions of a keyframe reference mesh into a unified latent space. A lightweight MLP decodes these features into per-frame 6-DoF deformations, achieving high-fidelity dynamic surface reconstruction more than 60× faster than existing training-free methods, without requiring category-specific priors or explicit correspondences.

## Background & Motivation

Temporally consistent surface reconstruction of dynamic 3D objects is a core yet highly challenging problem in computer vision. Existing methods fall into two broad categories:

**Optimization-based methods** (e.g., DynoSurf, PDG): directly optimize deformations per sequence, achieving high accuracy but requiring long runtimes (typically over 30 minutes), and prone to drift and degradation on long sequences.

**Learning-based methods** (e.g., CaDeX, M2V): achieve fast inference via category-specific priors, but generalization is limited to the training domain and requires predefined correspondences.

Both categories have complementary strengths and weaknesses. Nevertheless, a **general, efficient, and category-agnostic** approach to dynamic shape modeling remains an open challenge. The core difficulties are:

- Error accumulation in long sequences leads to correspondence drift
- Hash encodings are compact but their collision and non-smooth properties are ill-suited for coherent non-rigid motion
- Per-frame deformation field optimization cannot maintain global temporal consistency over long sequences

Neu-PiG's key insight is that encoding **complete deformations across all timesteps** into a **unified latent space parameterized by the reference surface**, combined with Sobolev preconditioning to ensure optimization stability, can simultaneously achieve both speed and quality.

## Method

### Overall Architecture

The Neu-PiG pipeline consists of the following steps:

1. **Keyframe selection and initialization**: A keyframe $t_{\text{key}}$ is selected from the sequence, and a reference mesh $\mathcal{X}_{t_{\text{key}}}$ is generated via Poisson surface reconstruction.
2. **Multi-resolution latent grid encoding**: Learnable feature vectors are stored in a position grid $\mathcal{G}_p$ and a normal direction grid $\mathcal{G}_n$.
3. **Temporal modulation and MLP decoding**: Latent features are concatenated with Fourier time encodings and decoded by an MLP into per-frame 6-DoF transformations.
4. **Sobolev preconditioned optimization**: Sobolev filtering is applied to gradient updates in the latent space.

### Multi-Resolution Latent Grid Design

The **position grid** $\mathcal{G}_p$ adopts an 8-level multi-resolution hierarchy:

- Coarsest level: $2^3$ cells
- Finest level: $32^3$ cells
- Resolution increases by 3 elements per level
- Each cell stores a **30-dimensional** latent feature

Key design: outputs across all levels are combined via **average aggregation** (rather than independent processing) to form a unified representation:

$$\boldsymbol{z}_p(\boldsymbol{x}_{i,t_{\text{key}}}) = \frac{1}{L} \sum_{l=1}^{L} \boldsymbol{z}_p^l(\boldsymbol{x}_{i,t_{\text{key}}})$$

This ensures the network learns **absolute deformations** rather than decomposed ones, substantially improving optimization stability.

The **normal direction grid** $\mathcal{G}_n$ uses a single $4^3$ resolution, storing **2-dimensional** features per cell. The normal grid allows spatially adjacent regions with differing normals to deform independently.

### Temporal Encoding and MLP Decoding

The Fourier time encoding maps the normalized timestep $\tilde{t} = (t-1)/(T-1)$ as:

$$\boldsymbol{\gamma}(t) = [\sin(\pi \nu_j \tilde{t}), \cos(\pi \nu_j \tilde{t})]_{j=1}^{M}$$

where $\nu_j = 2^{j-1}$ and $M=4$ yields an 8-dimensional time embedding.

The MLP input vector is $\boldsymbol{y}_i = (\boldsymbol{z}_n, \boldsymbol{z}_p, \boldsymbol{\gamma}(t))^T \in \mathbb{R}^{40}$. The network consists of 3 fully connected layers (512 hidden units + LeakyReLU) and outputs a 7-dimensional transformation vector (4-dimensional quaternion + 3-dimensional translation).

### Transformation Mapping

- **Rotation**: An offset quaternion scalar component $\hat{\boldsymbol{q}} = \frac{(1+q_w, q_x, q_y, q_z)^T}{\|(1+q_w, q_x, q_y, q_z)^T\|}$ ensures that zero output corresponds to the identity rotation.
- **Translation**: $\hat{\boldsymbol{d}} = \tanh(0.1 \cdot \boldsymbol{d})$ constrains the displacement range.
- **Final transformation**: $\hat{\boldsymbol{x}}_{i,t} = \boldsymbol{R}(\hat{\boldsymbol{q}}_i) \boldsymbol{x}_{i,t_{\text{key}}} + \hat{\boldsymbol{d}}_i$

### Loss & Training

The total loss comprises a deformation loss and an isometry loss: $\mathcal{L} = \mathcal{L}_{\text{def}} + w_{\text{iso}} \mathcal{L}_{\text{iso}}$

The **deformation loss** uses a robust Chamfer distance with temporally adaptive confidence weights:

$$\mathcal{L}_{\text{def}} = \frac{1}{T} \sum_{t=1}^{T} w_{\text{conf}}(t) \cdot L_{\text{CD}}(\hat{\mathcal{X}}_t, \mathcal{P}_t)$$

The **isometry loss** penalizes edge length changes to preserve local structure:

$$\mathcal{L}_{\text{iso}} = \frac{1}{T|\mathcal{E}|} \sum_{t=1}^{T} \sum_{(i,j) \in \mathcal{E}} |\|\hat{\boldsymbol{e}}_{ij,t}\| - \text{sg}(\|\hat{\boldsymbol{e}}_{ij,t_{\text{key}}}\|)|$$

### Sobolev Preconditioning

Grid parameter updates employ Sobolev preconditioned gradients:

$$\boldsymbol{z}^l \leftarrow \boldsymbol{z}^l - \eta (\mathbf{I} + \lambda^l \boldsymbol{L}^l)^{-2} \frac{\partial \mathcal{L}}{\partial \boldsymbol{z}^l}$$

where $\boldsymbol{L}^l$ is the Laplacian matrix, and $(\mathbf{I}+\lambda^l\boldsymbol{L}^l)^{-2}$ acts as a low-pass filter that couples adjacent cells, ensuring spatial coherence in latent updates.

## Key Experimental Results

### Main Results (Tab. 1: DFAUST / DT4D / AMA)

| Dataset | Method | CD (×10⁻⁵) ↓ | NC ↑ | F-0.5% ↑ | Corr. ↓ | Time ↓ |
|--------|------|-------------|------|----------|---------|--------|
| DFAUST | DynoSurf | 2.13 | 0.953 | 0.980 | 0.010 | 30 min |
| DFAUST | PDG | 0.52 | 0.957 | 0.988 | 0.018 | 7 min |
| DFAUST | **Ours†** | **0.40** | **0.967** | **0.989** | **0.008** | **8 s** |
| DT4D | DynoSurf | 15.18 | 0.919 | 0.773 | 0.032 | 30 min |
| DT4D | PDG | 1.53 | 0.960 | 0.961 | 0.058 | 7 min |
| DT4D | **Ours†** | **0.96** | **0.969** | **0.962** | **0.034** | **8 s** |
| AMA | DynoSurf | 1.01 | 0.918 | 0.921 | 0.044 | 30 min |
| AMA | PDG | 0.47 | 0.939 | 0.985 | 0.030 | 7 min |
| AMA | **Ours** | **0.44** | **0.951** | **0.988** | **0.018** | **32 s** |

### Long-Sequence Scalability (AMA, Tab. 2)

| Frames T | Method | CD (×10⁻⁵) ↓ | NC ↑ | Corr. ↓ | Time ↓ |
|--------|------|-------------|------|---------|--------|
| 40 | PDG | 0.66 | 0.923 | 0.042 | 28 min |
| 40 | **Ours** | **0.53** | **0.947** | **0.019** | **47 s** |
| 80 | PDG | 1.35 | 0.906 | 0.089 | 93 min |
| 80 | **Ours** | **1.04** | **0.940** | **0.023** | **84 s** |
| 120 | PDG | 30.20 | 0.788 | 0.118 | 158 min |
| 120 | **Ours** | **1.31** | **0.926** | **0.028** | **110 s** |

PDG's CD explodes to 30.20 at 120 frames, whereas Neu-PiG's remains at 1.31, demonstrating strong long-sequence stability.

### Ablation Study (Tab. 3: Component Ablation)

| Configuration | CD (×10⁻⁵) ↓ | NC ↑ | Corr. ↓ |
|------|-------------|------|---------|
| Hash encoding replacement | 1.23 | 0.903 | 0.045 |
| Without preconditioning | 0.98 | 0.955 | 0.036 |
| Without normal encoding $\boldsymbol{z}_n$ | 0.91 | 0.968 | 0.036 |
| Single resolution L=1 | 0.98 | 0.965 | 0.036 |
| **Full model** | **0.87** | **0.969** | **0.034** |

Key finding: replacing with hash encoding causes the largest degradation (CD: 0.87→1.23), validating the superiority of the multi-resolution average aggregation design.

## Highlights & Insights

1. **Unified latent space design**: All timesteps share a single latent grid (as opposed to PDG's per-frame independent deformation fields), which is the key to long-sequence stability — different timesteps share information through the unified space, naturally preventing drift.
2. **Average aggregation vs. concatenation**: Multi-resolution features are averaged rather than concatenated, so each level learns absolute deformations rather than decomposed ones, resulting in more stable optimization.
3. **Extension of Sobolev preconditioning**: Sobolev preconditioning is extended from the raw deformation fields in PDG to high-dimensional latent vectors, enabling richer local variation while maintaining global smoothness.
4. **Root of the 60× speedup**: Joint optimization over the full sequence (rather than per-frame), combined with a lightweight MLP and preconditioned convergence acceleration, together enable reconstruction in seconds.
5. **No priors or correspondences required**: Entirely category-agnostic, applicable to diverse object types including humans and animals.

## Limitations & Future Work

1. **Fixed topology assumption**: The method relies on the keyframe mesh for correct topology and cannot handle topological changes such as object splitting or merging.
2. **Limited latent grid capacity**: For extremely long sequences or highly complex motions, a fixed-size grid may be insufficient to encode all deformation information.
3. **Sensitivity to large motions and occlusions**: Correspondences are inferred implicitly via Chamfer distance; extreme motions or severe occlusions may degrade reconstruction accuracy.
4. **Point cloud input only**: End-to-end reconstruction from RGB images or depth maps is not addressed.

## Rating

⭐⭐⭐⭐ (4/5)

The method is elegantly designed, and the idea of introducing Sobolev preconditioning into the latent space is both novel and effective. The experimental results — over 60× speedup with improved accuracy — are highly impressive, and the scalability to long sequences is particularly notable. However, the method still assumes fixed topology and operates solely on point cloud inputs, which limits its applicable scope.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[CVPR 2026\] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy](neural_field-based_3d_surface_reconstruction_of_microstructures_from_multi-detec.md)
- [\[AAAI 2026\] Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction](../../AAAI2026/3d_vision/surface-based_visibility-guided_uncertainty_for_continuous_active_3d_neural_reco.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)

<!-- RELATED:END -->
