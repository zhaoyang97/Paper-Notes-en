---
title: >-
  [Paper Note] Discretized Gaussian Representation for Tomographic Reconstruction
description: >-
  [3D Vision] This paper proposes Discretized Gaussian Representation (DGR) for CT reconstruction, which directly reconstructs 3D voxels end-to-end via discretized Gaussian functions and introduces a highly parallelized fa…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 5370ce5937f210c7
---

# Discretized Gaussian Representation for Tomographic Reconstruction

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2411.04844](https://arxiv.org/abs/2411.04844)
- **Code**: [wskingdom/DGR](https://github.com/wskingdom/DGR)
- **Area**: 3D Vision / CT Reconstruction
- **Keywords**: CT Reconstruction, 3D Gaussian, Discretized Representation, Fast Volume Reconstruction, Sparse-View CT, Limited-Angle CT

## TL;DR

This paper proposes Discretized Gaussian Representation (DGR) for CT reconstruction, which directly reconstructs 3D voxels end-to-end via discretized Gaussian functions and introduces a highly parallelized fast volume reconstruction technique. DGR surpasses both deep learning-based and instance-based reconstruction methods in sparse-view and limited-angle CT settings without any training data.

## Background & Motivation

- **CT reconstruction challenges**: Ionizing radiation limits the number of projections; emergency scenarios demand fast reconstruction; methods must adapt to diverse CT configurations (cone-beam/fan-beam, sparse-view/limited-angle).
- **Limitations of Prior Work — Deep Learning Reconstruction (DLR)**: Requires large-scale training data, generalizes poorly (e.g., cone-beam-trained models degrade on fan-beam data), and diffusion-based models incur high computational overhead.
- **Limitations of Prior Work — Instance Reconstruction Methods**:
    - NeRF-based methods (NAF, SAX-NeRF): Require hours to reconstruct a single instance, making clinical deployment impractical.
    - 3DGS-based methods (R2-Gaussian, X-Gaussian): Suffer from **integration bias** (density inconsistency), and there is a fundamental mismatch between view-dependent rendering in 3DGS and voxel-based reconstruction.
- **Core Motivation**: Rethink the CT reconstruction framework around three design principles — discretized representation, efficient reconstruction, and unified global optimization.

## Method

### Overall Architecture

DGR represents a 3D volume as a set of learnable discretized Gaussian functions. Gaussian contributions are aggregated onto a voxel grid via fast volume reconstruction and then globally optimized in the projection domain.

### 1. Discretized Gaussian Representation

**Continuous Gaussian**:

$$G(p, \mu, \Sigma) = e^{-\frac{1}{2}(p-\mu)^\top \Sigma^{-1}(p-\mu)}$$

**Isotropic Gaussians** are adopted (tissue attenuation in CT exhibits isotropic characteristics). The voxel intensity is the sum of all Gaussian contributions:

$$V(p) = \sum_{i=1}^{n} G(p, \mu_i, \Sigma_i) \cdot I_i$$

**Bounded Gaussian influence**: Each Gaussian's influence is restricted to a local cuboid $B_0$ of size $w_0 \times h_0 \times c_0$, substantially reducing computation.

**Discretization and alignment**: The continuous mean $\mu$ cannot be directly used as a discrete voxel index (non-differentiable). A residual $\Delta\mu = \mu - \lfloor\mu\rfloor$ is defined, and the local box coordinates are adjusted as:

$$\lfloor B \rfloor_{n,w_0,h_0,c_0,d} = B_{w_0,h_0,c_0,d} - \Delta\mu_{n,1,1,1,d}$$

This ensures gradients can propagate through the aggregation operation, maintaining differentiability throughout the pipeline.

### 2. Fast Volume Reconstruction (FVR)

**Parallel computation**: The Mahalanobis distance is computed in parallel using Einstein summation:

$$D^2_{n,w_0,h_0,c_0} = \sum_d \lfloor B \rfloor_{n,w_0,h_0,c_0,d} C^{-1}_{n,d,d} \lfloor B \rfloor_{n,w_0,h_0,c_0,d}$$

**Decomposition acceleration**: $D^2$ is decomposed into four smaller Einstein summations:

$$D^2 = (B^\top C^{-1} B) - (B^\top C^{-1} \Delta\mu) - (\Delta\mu^\top C^{-1} B) + (\Delta\mu^\top C^{-1} \Delta\mu)$$

where $B$ is only $\frac{1}{n}$ the size of $\lfloor B \rfloor$, yielding a significant improvement in computational efficiency. The final contribution is:

$$\Gamma = e^{-\frac{1}{2}D^2} \cdot I$$

which is then aggregated onto the voxel grid: $V_{x,y,z} \leftarrow V_{x,y,z} + \Gamma_{i,x,y,z}$

**Plug-and-play**: The FVR module can be seamlessly integrated into other 3DGS-based methods (e.g., X-Gaussian).

### 3. Global Optimization

Unlike 3DGS's per-view/per-tile optimization, DGR jointly optimizes all Gaussians, converging within <1K iterations (vs. 30K for 3DGS).

Projection transform: $\hat{P} = \mathcal{T}(V)$. Loss function:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1(\hat{P}, P) + \lambda_2 \mathcal{L}_{SSIM}(\hat{P}, P) + \lambda_3 \mathcal{L}_{TV}(V)$$

with $\lambda_1=0.6, \lambda_2=0.2, \lambda_3=1.0$.

Adaptive density control (cloning/splitting/pruning Gaussians) further improves reconstruction quality.

## Key Experimental Results

### Main Results: Cone-Beam Sparse-View CT (FIPS Dataset, Real-World)

| Method | 75-view PSNR↑ | 75-view SSIM↑ | Time↓ | 50-view PSNR↑ | 25-view PSNR↑ |
|--------|---------------|---------------|-------|---------------|---------------|
| NAF | 38.58 | 0.848 | 51m | 36.44 | 32.92 |
| SAX-NeRF | 34.93 | 0.854 | 13h | 34.89 | 33.49 |
| X-Gaussian* | 38.27 | 0.894 | 10m | 37.80 | 35.12 |
| R2-Gaussian (30k) | 39.40 | 0.875 | 14m | 38.24 | 34.83 |
| **DGR (300 iter)** | 39.91 | 0.937 | **3m36s** | 38.66 | 35.16 |
| **DGR (1000 iter)** | **41.28** | **0.952** | 13m | **39.27** | 34.58 |

- DGR at 300 iterations (3 minutes) already surpasses R2-Gaussian at 30K iterations (14 minutes).
- 75-view PSNR exceeds R2-Gaussian by 1.88 dB.

### Ablation Study

**Local box size ablation (60-view AAPM-Mayo)**:

| Box-Size | Time (min) | V-RAM (GiB) | PSNR / SSIM |
|----------|-----------|-------------|-------------|
| 13×13×13 | 7.92 | 10.70 | 35.99 / 0.960 |
| 15×15×15 | 11.07 | 13.29 | 38.90 / 0.973 |
| **17×17×17** | **16.58** | **16.87** | **40.25 / 0.985** |
| 19×19×19 | 26.99 | 21.32 | 40.98 / 0.987 |

17×17×17 represents the optimal trade-off point.

**Fast Volume Reconstruction efficiency**:

| Method | VRAM (GiB) | Time/iter (s) |
|--------|------------|---------------|
| Direct Reconstruction | 16662.50 (est.) | / |
| FVR w/o Decomposition | 16.87 | 1.05 |
| **FVR w/ Decomposition** | **16.87** | **0.09** |

The decomposition technique reduces per-iteration reconstruction time from 1.05s to 0.09s (11.7× speedup) with no additional memory cost.

### Fan-Beam Sparse-View CT (AAPM-Mayo LDCT)

| Method | Extra Data | 180-view PSNR | 120-view PSNR | 60-view PSNR |
|--------|----------|---------------|---------------|--------------|
| FBPConvNet | 4839 | 42.23 | 39.45 | 35.63 |
| SWORD | 4839 | 45.08 | 42.49 | 38.49 |
| **DGR** | **0** | **46.13** | **44.64** | **40.25** |

DGR surpasses all DLR methods requiring large-scale training data under a zero-training-data condition.

### Limited-Angle CT (90° Reconstruction)

| Method | Axial PSNR↑ | Coronal PSNR↑ | Sagittal PSNR↑ |
|--------|-------------|---------------|----------------|
| DiffusionMBIR | 34.92 | 32.48 | 28.82 |
| **DGR** | **38.22** | **39.32** | **38.35** |

DGR achieves substantial gains across all three orientations (+3.3 / +6.8 / +9.5 dB) without requiring any prior knowledge.

## Highlights & Insights

1. **Elegant representation design**: Modeling the voxel grid directly with discretized Gaussians fundamentally avoids the integration bias and view-direction bias inherent in 3DGS.
2. **Zero-data superiority over DLR**: DGR requires no training data yet outperforms deep learning methods that rely on thousands of training images across multiple settings.
3. **Extreme efficiency**: With over 150K Gaussians on a 256³ volume, each iteration takes only 0.09 seconds; high-quality reconstruction is achieved in approximately 3 minutes at 300 iterations.
4. **Strong generalizability**: The unified framework adapts to cone-beam/fan-beam and sparse-view/limited-angle CT configurations without any modification.
5. **Plug-and-play FVR**: Can be directly integrated into existing methods such as X-Gaussian, endowing them with volumetric reconstruction capability.

## Limitations & Future Work

- **Isotropic assumption**: Only isotropic Gaussians are used, which may be insufficient for tissues with pronounced directional structures.
- **Local box size trade-off**: Larger box sizes improve quality but memory and computation grow exponentially.
- **Insufficient clinical validation**: Diagnostic accuracy has not been validated in real clinical scenarios.
- **Limited 2D slice-level analysis**: The work primarily demonstrates 3D volumetric reconstruction; analysis of fine-grained details at the 2D slice level is limited.

## Related Work & Insights

- R2-Gaussian identified the integration bias in 3DGS and proposed a correction, but at additional computational cost; DGR avoids this issue by design.
- NeRF-based methods such as NAF and SAX-NeRF provide continuous representations but suffer from slow inference; DGR's discretized representation is a key factor in its acceleration.
- Diffusion-based methods such as SWORD are powerful but depend on large-scale data and high computational budgets.
- **Insight**: The discretized Gaussian paradigm could be extended to other medical imaging modalities such as MRI reconstruction and PET imaging.

## Rating ⭐⭐⭐⭐⭐

The method innovates comprehensively across representation, reconstruction, and optimization. The theoretical derivations are rigorous (Einstein summation decomposition), experiments are thorough (three datasets, four CT configurations), and the method consistently outperforms data-hungry approaches under a zero-data condition. The plug-and-play nature of FVR and the practical 3-minute reconstruction time endow DGR with high clinical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
