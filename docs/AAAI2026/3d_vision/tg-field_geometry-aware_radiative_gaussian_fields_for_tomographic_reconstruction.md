---
title: >-
  [Paper Note] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction
description: >-
  [AAAI 2026][3D Vision][CT Reconstruction] This paper proposes TG-Field, a geometry-aware Gaussian deformation framework for extremely sparse-view CT reconstruction. By incorporating a multi-resolution hash encoder to model spatial geometric priors, alongside a spatiotemporal attention module and a motion flow network to handle dynamic CT, it achieves state-of-the-art (SOTA) performance in both static and dynamic CT reconstruction.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "CT Reconstruction"
  - "3D Gaussian Splatting"
  - "Sparse-view"
  - "Dynamic CT"
  - "Deformation Field"
date: 2026-05-08
content_hash: ab673c8d285e3b29
---

# TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2602.11705](https://arxiv.org/abs/2602.11705)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: CT Reconstruction, 3D Gaussian Splatting, Sparse-view, Dynamic CT, Deformation Field

## TL;DR

This paper proposes TG-Field, a geometry-aware Gaussian deformation framework for extremely sparse-view CT reconstruction. By incorporating a multi-resolution hash encoder to model spatial geometric priors, alongside a spatiotemporal attention module and a motion flow network to handle dynamic CT, it achieves state-of-the-art (SOTA) performance in both static and dynamic CT reconstruction.

## Background & Motivation

### Problem Definition

Cone-Beam Computed Tomography (CBCT) is widely used in medical, biological, and industrial fields. High-quality 3D reconstruction typically requires hundreds of X-ray projections, but dense angular sampling poses radiation exposure risks. Sparse-view CBCT reconstruction aims to reduce the number of projections while maintaining reconstruction fidelity.

### Limitations of Prior Work

**Traditional Methods**:
- FDK (Feldkamp-Davis-Kress filtered back-projection): Quality degrades drastically under sparse views
- SART (Simultaneous Algebraic Reconstruction Technique): High computational cost and sensitive to hyperparameters

**NeRF-based Methods** (NAF, SAX-NeRF, etc.):
- Self-supervised without paired data, but mostly limited to static reconstruction
- Requires dense ray sampling, causing high computational overhead
- STNF4D attempts dynamic CT but suffers from slow convergence and sub-optimal reconstruction quality

The two key challenges for **3DGS-based Methods**:

**Insufficient robustness under extremely sparse views**: Lack of explicit geometric regularization prevents individual Gaussian optimization from maintaining geometric consistency, leading to severe artifacts.

**Difficulty in dynamic CT reconstruction**: Inability to model non-rigid deformations such as respiratory motion makes temporal consistency hard to guarantee.

### Core Motivation

Existing 3DGS CT methods (e.g., R²-Gaussian, X-Gaussian) perform reasonably well under moderate sparsity but experience a sharp drop in performance under extreme sparsity. The key reason is that each Gaussian-primitive is optimized independently, lacking constraints from spatial geometric context. To address this, the authors propose introducing a **geometry-aware deformation field** that leverages a hash encoder to capture local spatial priors and constrain the spatial correlation between Gaussian primitives, thereby maintaining structural coherence even under extremely sparse conditions.

## Method

### Overall Architecture

The workflow of TG-Field is as follows:
1. Generate a high-quality initial point cloud through **iterative initialization**.
2. Use a **multi-resolution hash encoder** to capture spatial geometric features.
3. Predict the attribute offsets of Gaussian primitives via a **multi-head deformation decoder**.
4. For dynamic CT, incorporate a **spatiotemporal attention module** and a **motion flow network**.
5. Use **semantic consistency regularization** to enhance cross-view consistency.

Finally, the deformed Gaussian primitives are rendered into X-ray projections and voxelized into the CT volume.

### Key Designs

#### 1. **Iterative Initialization Strategy**: Point Cloud Initialization with High-Quality Geometric Priors

**Function**: Two-stage iterative initialization—first uses CGLS (Conjugate Gradient Least Squares) to obtain a coarse volume reconstruction, and then refines it using ASD-POCS (Adaptive Steepest Descent-Projection Onto Convex Sets) with TV constraints.

**Mechanism**: Unlike existing methods that utilize uniform cube sampling (which lacks geometric information) or FDK initialization (which suffers from poor quality under sparse views), the iterative method extracts richer geometric information from sparse projections:
- First stage: CGLS iteratively approximates the volume solution under sparse projection constraints.
- Second stage: ASD-POCS enforces TV constraints to reduce noise and preserve structural boundaries.

**Design Motivation**: High-quality initialization is crucial for 3DGS convergence. Under extremely sparse views (such as 5 views), uniformly sampled point clouds contain baseline structural information that is too scarce to guide effective optimization.

#### 2. **Geometry-Aware Splatting Field**: Modeling Spatial Correlation with Hash Encoder

**Function**: Captures the spatial context of each Gaussian primitive using a multi-resolution hash grid encoder, and then predicts attribute offsets using a multi-head decoder.

**Mechanism**: For a Gaussian primitive at position $\boldsymbol{\mu}_i$, multi-scale features are obtained via hash encoding:

$$h_\phi(\boldsymbol{\mu}_i) = \text{concat}_{s \in S}[f_s(\boldsymbol{\mu}_i)] \in \mathbb{R}^{|S| \cdot C}$$

The multi-head decoder predicts offsets for position, rotation, scaling, and density separately:

$$G'_i = (\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i, R_i + \Delta R_i, S_i + \Delta S_i, \rho_i + \Delta\rho_i)$$

**Design Motivation**: The hash encoder naturally maps spatially proximal Gaussian primitives to a similar feature space, enforcing geometric consistency among them. This is particularly vital under extremely sparse-view scenarios: when observational information is severely deficient, spatial prior constraints can compensate for the missing geometric information.

#### 3. **Spatiotemporal Attention Module (STAB)**: Resolving Hash Collision and Temporal Drift in 4D CT

**Function**: Applies an attention mechanism to the jointly encoded spatiotemporal hash features to eliminate spatiotemporal ambiguity.

**Mechanism**: For each Gaussian primitive $i$, the embeddings within the time window are stacked:

$$\mathbf{H}_i = [h_\phi(\boldsymbol{\mu}_i, t_1), \ldots, h_\phi(\boldsymbol{\mu}_i, t_T)]^\top$$

Then, scaled dot-product attention is applied:

$$\text{Attn}(\mathbf{H}_i) = \text{softmax}\left(\frac{QK^\top}{\sqrt{C}}\right)V$$

**Design Motivation**: Joint hashing of spatial and temporal coordinates can lead to hash collisions. When identical or similar spatial positions recur at different times, the hash buckets produce ambiguous embeddings. STAB aggregates temporal context to resolve ambiguities in these collided buckets, yielding more stable dynamic deformations.

#### 4. **Motion Flow Network**: Modeling Fine-grained Respiratory Motion

**Function**: Uses a ResFields MLP to predict a fine displacement field, further refining the Gaussian center locations on top of the deformation field's output.

$$\hat{\boldsymbol{\mu}}_i(t) = \boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t) + \text{Flow}(\boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t), t)$$

**Design Motivation**: The initial deformation field might miss subtle local anatomical deformations (such as local tissue sliding during lung respiratory motion). The motion flow network acts as a residual correction module to capture these fine-grained dynamics.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{TV}\mathcal{L}_{TV} + \lambda_{sem}\mathcal{L}_{sem}$

- L1 loss + D-SSIM: Supervises the rendered X-ray projections
- 3D TV regularization: Homogeneity prior
- Semantic consistency regularization $\mathcal{L}_{sem}$: Extracts visual features using a pre-trained DINO-ViT to enforce cross-view semantic consistency

Training is conducted in two stages: first pre-training R²-Gaussian for 5000 iterations (warm-up), followed by refining with the deformation field.

## Key Experimental Results

### Main Results

**Static CT Reconstruction (Synthetic + Real Datasets)**:

| Method | Synthetic 5-View PSNR/SSIM | Synthetic 10-View PSNR/SSIM | Synthetic 20-View PSNR/SSIM | Real 10-View PSNR/SSIM |
|------|-------------------|-------------------|-------------------|-------------------|
| FDK | 11.83/0.112 | 15.21/0.186 | 18.48/0.293 | 17.57/0.225 |
| SART | 22.10/0.683 | 24.32/0.768 | 27.24/0.845 | 28.72/0.846 |
| SAX-NeRF | 24.05/0.740 | 27.55/0.801 | 31.93/0.875 | 32.26/0.835 |
| R²-Gaussian | 23.81/0.735 | 28.15/0.833 | 32.25/0.923 | 32.73/0.859 |
| **Ours** | **24.54/0.779** | **28.95/0.849** | **32.92/0.936** | **33.59/0.872** |

**Dynamic CT Reconstruction**:

| Method | XCAT PSNR/SSIM | TCIA PSNR/SSIM | SPARE PSNR/SSIM | Average PSNR/SSIM |
|------|---------------|---------------|----------------|---------------|
| Hex-plane | 21.79/0.866 | 23.91/0.835 | 26.43/0.856 | 24.04/0.852 |
| K-plane | 20.57/0.847 | 24.59/0.855 | 26.59/0.876 | 23.92/0.859 |
| STNF4D | 25.73/0.928 | 29.37/0.919 | 28.75/0.887 | 27.95/0.911 |
| 4DGS | 33.95/0.955 | 34.44/0.948 | 30.01/0.898 | 32.80/0.933 |
| **Ours** | **35.51/0.969** | **35.41/0.955** | **30.41/0.905** | **33.78/0.943** |

### Ablation Study

| Setting | Component | PSNR↑ | SSIM↑ | Description |
|------|------|-------|-------|------|
| Static | HE only | 28.71 | 0.841 | Hash encoder only |
| Static | HE + SR | 28.95 | 0.849 | + Semantic regularization, +0.24dB improvement |
| Dynamic | HE + STAB | 34.89 | 0.945 | + Spatiotemporal attention |
| Dynamic | HE + STAB + MF | 35.23 | 0.952 | + Motion flow network, +0.34dB improvement |
| Dynamic | All (HE+STAB+MF+SR) | **35.41** | **0.955** | Full components |

### Key Findings

1. **Significant advantage in extremely sparse views**: Outperforms R²-Gaussian by 0.73dB (synthetic) and 0.65dB (real) under 5 views, demonstrating that geometric prior constraints are particularly crucial when observational information is highly lacking.
2. **Comprehensive sweep in dynamic CT**: Achieves an average PSNR higher than 4DGS by 0.98dB, and 1.56dB higher on XCAT.
3. **Significant impact of initialization strategy**: Iterative initialization consistently outperforms FDK and uniform sampling across 2 to 8 view settings.
4. **Incremental contributions of each component**: Stepwise performance improvements are observed from HE $\rightarrow$ +STAB $\rightarrow$ +MF $\rightarrow$ +SR.
5. **Motion flow network mainly improves motion-sensitive regions**: Such as local deformations caused by lung respiratory motion.

## Highlights & Insights

1. **Crucial Role of Geometric Priors**: Infusing spatial correlation into Gaussian optimization via a hash encoder stands as the most core contribution. This effectively alleviates the lack of global consistency when optimizing each Gaussian primitive independently.
2. **Innovative Iterative Initialization**: Elegantly combines classical iterative reconstruction methods (CGLS + ASD-POCS) to provide high-quality starting points for 3DGS.
3. **VFM for CT Regularization**: Using semantic features of pre-trained visual foundation models (DINO-ViT) for cross-view consistency constraints represents a promising attempt to transfer foundation models from the natural image domain to medical imaging.
4. **Unified Static/Dynamic Framework**: The identical framework can be extended to 4D CT simply by introducing the temporal dimension.

## Limitations & Future Work

1. **Computational Overhead Not Detailed**: The additional training/inference time introduced by the hash encoder and attention module lacks quantitative comparison.
2. **Rationality of Semantic Regularization**: Using DINO-ViT pre-trained on natural images may exert limited efficacy on X-ray projection images, as domain gaps might attenuate its performance.
3. **Restricted to CBCT**: Has not been validated on parallel-beam CT or other imaging modalities.
4. **Marginal Gain on the SPARE Dataset**: The PSNR is only 0.4dB higher than 4DGS, indicating limited space for improvement on real clinical data.

## Related Work & Insights

- The differentiable voxelization proposed by R²-Gaussian laid the foundation for 3DGS to directly reconstruct CT volumes.
- The deformation field approach in 4DGaussians is inherited by this work, but updated with a geometry-aware encoder to improve robustness.
- The concept of semantic consistency regularization is generalizable to other sparse reconstruction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of introducing geometric priors to 3DGS CT reconstruction holds value, although individual components (hash encoder, attention, motion flow) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering static/dynamic $\times$ synthetic/real $\times$ multi-view settings, with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured but mathematically dense.
- **Value**: ⭐⭐⭐⭐ — Demonstrates high potential value for clinical applications in medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)

</div>

<!-- RELATED:END -->
