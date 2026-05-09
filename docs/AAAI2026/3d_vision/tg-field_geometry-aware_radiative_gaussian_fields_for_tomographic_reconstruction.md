---
title: >-
  [Paper Note] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction
description: >-
  [AAAI 2026][3D Vision][CT Reconstruction] This paper proposes TG-Field, a geometry-aware Gaussian deformation framework for extremely sparse-view CT reconstruction. It employs a multi-resolution hash encoder to model spatial geometric priors, a spatiotemporal attention module and a motion flow network to handle dynamic CT, achieving state-of-the-art performance on both static and dynamic CT reconstruction.
tags:
  - AAAI 2026
  - 3D Vision
  - CT Reconstruction
  - 3D Gaussian Splatting
  - Sparse-View
  - Dynamic CT
  - Deformation Field
date: 2026-05-08
content_hash: 501a813f717637cd
---

# TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2602.11705](https://arxiv.org/abs/2602.11705)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: CT Reconstruction, 3D Gaussian Splatting, Sparse-View, Dynamic CT, Deformation Field

## TL;DR

This paper proposes TG-Field, a geometry-aware Gaussian deformation framework for extremely sparse-view CT reconstruction. It employs a multi-resolution hash encoder to model spatial geometric priors, a spatiotemporal attention module and a motion flow network to handle dynamic CT, achieving state-of-the-art performance on both static and dynamic CT reconstruction.

## Background & Motivation

### Problem Definition

Cone-beam computed tomography (CBCT) is widely used in medicine, biology, and industry. High-quality 3D reconstruction typically requires hundreds of X-ray projections, yet dense angular sampling entails radiation exposure risks. Sparse-view CBCT reconstruction aims to reduce the number of projections while preserving reconstruction fidelity.

### Limitations of Prior Work

**Traditional methods**:
- FDK (Feldkamp filtered back-projection): quality degrades sharply under sparse-view settings.
- SART (iterative reconstruction): computationally expensive and sensitive to hyperparameters.

**NeRF-based methods** (NAF, SAX-NeRF, etc.):
- Self-supervised and require no paired data, but mostly target static reconstruction.
- Require dense ray sampling, incurring high computational cost.
- STNF4D attempts dynamic CT but suffers from slow convergence and poor reconstruction quality.

**Two key challenges for 3DGS-based methods**:

**Insufficient robustness under extremely sparse views**: the absence of explicit geometric regularization prevents single-Gaussian optimization from maintaining geometric consistency, causing severe artifacts.

**Difficulty in dynamic CT reconstruction**: non-rigid deformations such as respiratory motion cannot be modeled, and temporal consistency is hard to guarantee.

### Root Cause

Existing 3DGS-based CT methods (e.g., R²-Gaussian, X-Gaussian) perform reasonably well under moderate sparsity but degrade sharply in extremely sparse settings. The key reason is that each Gaussian primitive is optimized independently without constraints from spatial geometric context. The authors propose introducing a **geometry-aware deformation field** that captures local spatial priors via a hash encoder to constrain spatial correlations among Gaussian primitives, thereby preserving structural coherence even under extremely sparse conditions.

## Method

### Overall Architecture

The TG-Field pipeline proceeds as follows:
1. A high-quality initial point cloud is generated via **iterative initialization**.
2. A **multi-resolution hash encoder** captures spatial geometric features.
3. A **multi-head deformation decoder** predicts attribute offsets for each Gaussian primitive.
4. For dynamic CT, a **spatiotemporal attention module** and a **motion flow network** are incorporated.
5. **Semantic consistency regularization** enhances cross-view consistency.

The deformed Gaussian primitives are ultimately rendered into X-ray projections and voxelized into CT volumes.

### Key Designs

#### 1. **Iterative Initialization Strategy**: Point Cloud Initialization with High-Quality Geometric Priors

**Function**: A two-stage iterative initialization — CGLS (Conjugate Gradient Least Squares) first yields a coarse volumetric reconstruction, followed by ASD-POCS (Adaptive Steepest Descent–Projection Onto Convex Sets) with TV constraints for refinement.

**Mechanism**: Unlike existing methods that rely on uniform cube sampling (lacking geometric information) or FDK initialization (poor quality under sparse conditions), the iterative approach extracts richer geometric information from sparse projections:
- Stage 1: CGLS iteratively approximates the volumetric solution under sparse projection constraints.
- Stage 2: ASD-POCS enforces TV constraints to reduce noise while preserving structural edges.

**Design Motivation**: High-quality initialization is critical for 3DGS convergence. Under extremely sparse conditions such as 5 views, uniformly sampled point clouds contain almost no meaningful structural information, rendering optimization highly challenging.

#### 2. **Geometry-Aware Splatting Field**: Modeling Spatial Correlations via Hash Encoder

**Function**: A multi-resolution hash grid encoder captures the spatial context of each Gaussian primitive, and a multi-head decoder predicts attribute offsets accordingly.

**Mechanism**: For a Gaussian primitive at position $\boldsymbol{\mu}_i$, multi-scale features are obtained via hash encoding:

$$h_\phi(\boldsymbol{\mu}_i) = \text{concat}_{s \in S}[f_s(\boldsymbol{\mu}_i)] \in \mathbb{R}^{|S| \cdot C}$$

A multi-head decoder separately predicts offsets for position, rotation, scale, and density:

$$G'_i = (\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i, R_i + \Delta R_i, S_i + \Delta S_i, \rho_i + \Delta\rho_i)$$

**Design Motivation**: The hash encoder naturally maps spatially neighboring Gaussian primitives to similar feature spaces, thereby enforcing geometric consistency among them. This is particularly important under extremely sparse views — when observational information is severely limited, spatial prior constraints can compensate for missing geometric information.

#### 3. **Spatiotemporal Attention Block (STAB)**: Addressing Hash Collisions and Temporal Drift in 4D CT

**Function**: An attention mechanism is applied over jointly encoded spatiotemporal hash features to resolve spatiotemporal ambiguities.

**Mechanism**: For each Gaussian primitive $i$, embeddings within a temporal window are stacked:

$$\mathbf{H}_i = [h_\phi(\boldsymbol{\mu}_i, t_1), \ldots, h_\phi(\boldsymbol{\mu}_i, t_T)]^\top$$

Scaled dot-product attention is then applied:

$$\text{Attn}(\mathbf{H}_i) = \text{softmax}\left(\frac{QK^\top}{\sqrt{C}}\right)V$$

**Design Motivation**: Jointly hashing spatial and temporal coordinates causes hash collisions — when the same or similar spatial positions recur at different time steps, hash buckets produce ambiguous embeddings. STAB aggregates temporal context to resolve ambiguities in colliding buckets, yielding more stable dynamic deformations.

#### 4. **Motion Flow Network**: Modeling Fine-Grained Respiratory Motion

**Function**: A ResFields MLP predicts a fine displacement field that further corrects Gaussian center positions on top of the deformation field output.

$$\hat{\boldsymbol{\mu}}_i(t) = \boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t) + \text{Flow}(\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t), t)$$

**Design Motivation**: The initial deformation field may miss subtle local anatomical deformations (e.g., local tissue sliding during pulmonary respiration). The motion flow network serves as a residual correction module to capture these fine-grained motions.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{TV}\mathcal{L}_{TV} + \lambda_{sem}\mathcal{L}_{sem}$

- L1 loss + D-SSIM: supervises rendered X-ray projections.
- 3D TV regularization: enforces a homogeneity prior.
- Semantic consistency regularization $\mathcal{L}_{sem}$: extracts visual features using a pretrained DINO-ViT to enforce cross-view semantic consistency.

Training proceeds in two stages: R²-Gaussian is first pre-trained for 5,000 iterations (warm-up), followed by deformation field refinement.

## Key Experimental Results

### Main Results

**Static CT Reconstruction (Synthetic + Real Datasets)**:

| Method | Syn. 5-view PSNR/SSIM | Syn. 10-view PSNR/SSIM | Syn. 20-view PSNR/SSIM | Real 10-view PSNR/SSIM |
|---|---|---|---|---|
| FDK | 11.83/0.112 | 15.21/0.186 | 18.48/0.293 | 17.57/0.225 |
| SART | 22.10/0.683 | 24.32/0.768 | 27.24/0.845 | 28.72/0.846 |
| SAX-NeRF | 24.05/0.740 | 27.55/0.801 | 31.93/0.875 | 32.26/0.835 |
| R²-Gaussian | 23.81/0.735 | 28.15/0.833 | 32.25/0.923 | 32.73/0.859 |
| **Ours** | **24.54/0.779** | **28.95/0.849** | **32.92/0.936** | **33.59/0.872** |

**Dynamic CT Reconstruction**:

| Method | XCAT PSNR/SSIM | TCIA PSNR/SSIM | SPARE PSNR/SSIM | Avg. PSNR/SSIM |
|---|---|---|---|---|
| Hex-plane | 21.79/0.866 | 23.91/0.835 | 26.43/0.856 | 24.04/0.852 |
| K-plane | 20.57/0.847 | 24.59/0.855 | 26.59/0.876 | 23.92/0.859 |
| STNF4D | 25.73/0.928 | 29.37/0.919 | 28.75/0.887 | 27.95/0.911 |
| 4DGS | 33.95/0.955 | 34.44/0.948 | 30.01/0.898 | 32.80/0.933 |
| **Ours** | **35.51/0.969** | **35.41/0.955** | **30.41/0.905** | **33.78/0.943** |

### Ablation Study

| Setting | Components | PSNR↑ | SSIM↑ | Note |
|---|---|---|---|---|
| Static | HE only | 28.71 | 0.841 | Hash encoder only |
| Static | HE + SR | 28.95 | 0.849 | +Semantic regularization, +0.24 dB |
| Dynamic | HE + STAB | 34.89 | 0.945 | +Spatiotemporal attention |
| Dynamic | HE + STAB + MF | 35.23 | 0.952 | +Motion flow network, +0.34 dB |
| Dynamic | All (HE+STAB+MF+SR) | **35.41** | **0.955** | Full model |

### Key Findings

1. **Significant advantage under extremely sparse views**: At 5 views, the method surpasses R²-Gaussian by 0.73 dB (synthetic) and 0.65 dB (real), indicating that geometric prior constraints are especially critical when observational information is extremely limited.
2. **Comprehensive superiority on dynamic CT**: Average PSNR exceeds 4DGS by 0.98 dB, and by 1.56 dB on XCAT.
3. **Initialization strategy has a notable impact**: Iterative initialization outperforms FDK and uniform sampling across 2–8 view settings.
4. **Incremental contributions from each component**: HE → +STAB → +MF → +SR yields progressive performance gains.
5. **Motion flow network primarily improves motion-sensitive regions**: e.g., local deformations caused by pulmonary respiratory motion.

## Highlights & Insights

1. **Critical role of geometric priors**: Injecting spatial correlations into Gaussian optimization via the hash encoder is the paper's most central contribution, addressing the lack of global consistency in per-Gaussian independent optimization.
2. **Novel iterative initialization**: Cleverly combines classical iterative reconstruction methods (CGLS + ASD-POCS) to provide a high-quality starting point for 3DGS.
3. **VFMs for CT regularization**: Leveraging semantic features from a pretrained visual foundation model (DINO-ViT) for cross-view consistency constraints represents a worthwhile attempt to transfer natural-image foundation models to medical imaging.
4. **Unified static/dynamic framework**: The same framework extends to 4D CT by incorporating the temporal dimension.

## Limitations & Future Work

1. **Computational overhead not thoroughly reported**: The additional training/inference time introduced by the hash encoder and attention modules is not quantitatively compared.
2. **Questionable benefit of semantic regularization**: The effectiveness of DINO-ViT pretrained on natural images for X-ray images may be limited, and the domain gap could attenuate its impact.
3. **CBCT-only validation**: The method is not evaluated on parallel-beam CT or other imaging modalities.
4. **Marginal advantage on the SPARE dataset**: PSNR is only 0.4 dB higher than 4DGS, suggesting limited improvement on clinically realistic data.

## Related Work & Insights

- The differentiable voxelization proposed in R²-Gaussian laid the foundation for direct CT volume reconstruction via 3DGS.
- The deformation field paradigm from 4DGaussians is inherited and made more robust through the addition of a geometry-aware encoder.
- The semantic consistency regularization idea is generalizable to other sparse reconstruction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of introducing geometric priors into 3DGS-based CT reconstruction is valuable, though individual components (hash encoder, attention, motion flow) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across static/dynamic × synthetic/real × multi-view settings with complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, though notation is dense.
- **Value**: ⭐⭐⭐⭐ — High potential clinical application value in medical imaging.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)

<!-- RELATED:END -->
