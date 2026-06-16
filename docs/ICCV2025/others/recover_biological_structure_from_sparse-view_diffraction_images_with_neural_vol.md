---
title: >-
  [Paper Note] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior
description: >-
  [ICCV 2025][sparse-view reconstruction] This paper proposes Neural Volumetric Prior (NVP), a hybrid neural representation combining an explicit 3D feature grid with an implicit MLP…
tags:
  - "ICCV 2025"
  - "sparse-view reconstruction"
  - "neural volumetric prior"
  - "diffractive optics"
  - "refractive index reconstruction"
  - "fluorescence diffraction tomography"
date: 2026-05-08
content_hash: 376e0f9e71f5e577
---

# Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior

**Conference**: ICCV 2025
**arXiv**: [2510.16391](https://arxiv.org/abs/2510.16391)  
**Code**: None  
**Area**: Other / Computational Imaging
**Keywords**: sparse-view reconstruction, neural volumetric prior, diffractive optics, refractive index reconstruction, fluorescence diffraction tomography

## TL;DR
This paper proposes Neural Volumetric Prior (NVP), a hybrid neural representation combining an explicit 3D feature grid with an implicit MLP, integrated with a physically accurate diffraction-based rendering equation. NVP enables, for the first time, high-fidelity volumetric reconstruction of the 3D refractive index of semi-transparent biological specimens from sparse-view inputs (as few as 6–7 fluorescence images), reducing the required number of images by approximately 50× and processing time by 3×.

## Background & Motivation

**Background**: Optical tomography reconstructs 3D biological structures from multi-angle 2D images and is an important tool for label-free imaging of living cells. Fluorescence diffraction tomography (FDT) uses internal fluorescent sources within the specimen, enabling imaging without requiring transmission from the opposite side, making it suitable for in vivo imaging.

**Limitations of Prior Work**:
- FDT requires **hundreds of 2D images** to reconstruct a single 3D volume, necessitating that the specimen remain stationary during acquisition (seconds to minutes), which precludes capturing rapid dynamic processes such as cardiomyocyte contraction or embryonic development.
- Unlike natural scene 3D reconstruction, biological specimens are **semi-transparent**, requiring reconstruction of the entire volume rather than just the surface, with a far greater number of unknown voxels.
- The numerical aperture of the microscope limits the available angular range, and the spatial distribution of fluorescent sources further restricts usable angles, resulting in an extremely sparse-view setting.
- Existing neural field methods are based on **geometric optics** (e.g., NeRF), assuming rectilinear light propagation; however, diffraction effects are significant at the microscale, rendering geometric optics models inadequate.

**Key Challenge**: How can a 3D volume with a large number of unknown voxels be reconstructed from as few as ~6 images? The physical model must be upgraded from geometric to wave optics, while the neural representation must provide sufficient regularization under extremely sparse data.

**Key Insight**: Design a hybrid neural representation (explicit grid + implicit MLP) that retains the sparse prior of explicit representations while capturing spatial correlations through the MLP to compensate for missing information, combined with a diffractive optics physical prior for physically accurate rendering.

## Method

### Overall Architecture
The NVP pipeline proceeds as follows: (1) a randomly initialized 3D feature grid $W_{xyz}$ is defined and mapped to a refractive index volume $\hat{n}$ via a 3-layer MLP $F_{\text{nvp}}$; (2) predicted images $\hat{I}$ are generated from the predicted refractive index and self-calibrated fluorescent source positions using a multi-layer Born approximation diffraction rendering equation; (3) the grid features and MLP parameters are optimized via backpropagation through the loss between predicted and ground-truth images.

### Key Designs

1. **Neural Volumetric Prior**:

    - **Explicit grid** $W_{xyz}(x,y,z) \in \mathbb{R}^F$: stores an $F$-dimensional learnable feature vector at each voxel, providing direct spatial structural priors and sparsity.
    - **Implicit MLP** $F_{\text{nvp}}$: a 3-layer fully connected network that maps feature vectors to scalar refractive index values, capturing implicit spatial correlations encoded in the explicit grid.
    - **Hybrid representation**: $\hat{n}(x,y,z) = F_{\text{nvp}}(W_{xyz}(x,y,z))$
    - **Adaptive resolution**: grid resolution is dynamically adjusted according to the spatial variation distribution of the refractive index.
    - Comparison with other representations: purely explicit representations (Plenoxels) lack spatial correlation encoding and require more views; purely implicit representations (NeRF-style MLPs) are computationally inefficient; triplane low-rank decomposition introduces grid artifacts.

2. **Diffraction-Based Physical Rendering Equation**:

    - The imaging volume is modeled as $N_z$ thin slabs, with the optical field $\hat{E}_{k,i}(\mathbf{r})$ propagating between successive slabs:
     $$\hat{E}_{k,i}(\mathbf{r}) = \mathcal{P}_{\Delta z}\{t_k(\mathbf{r}) \cdot \hat{E}_{k-1,i}(\mathbf{r})\}$$
     where $\mathcal{P}_{\Delta z}$ is the propagation operator and $t_k(\mathbf{r})$ is the transmission function of the $k$-th slab (related to the refractive index).
    - The intensity image captured by the camera is: $\hat{I}_i(\mathbf{r}) = |\hat{E}_{N_z,i}(\mathbf{r})|^2$
    - Efficient rendering is achieved via GPU-parallel computation with precomputed propagation kernels.

3. **Coherence Alignment and Self-Calibration**:

    - **Coherence mask**: addresses the mismatch between partially coherent/incoherent fluorescence in experiments and the coherent light assumed by the model.
    - **Viewpoint self-calibration**: fluorescent source positions are estimated from fluorescence images via Gaussian fitting and jointly optimized with MLP parameters.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{\text{img}} + \tau \mathcal{R}_{\text{ri}}$$
- $\mathcal{L}_{\text{img}}$: image-level loss combining L1, L2, and SSIM.
- $\mathcal{R}_{\text{ri}}$: total variation regularization on the refractive index (promoting smoothness while preserving detail).

## Key Experimental Results

### Main Results: Quantitative Comparison on Synthetic Data (Varying Number of Illuminations)

| Method | 6 illum. PSNR/SSIM/LPIPS | 7 illum. PSNR/SSIM/LPIPS | 20 illum. PSNR/SSIM/LPIPS |
|:---|:---|:---|:---|
| Explicit | 28.62 / 0.854 / 0.182 | 28.73 / 0.847 / 0.178 | 28.88 / 0.865 / 0.119 |
| Triplane | 28.73 / 0.713 / 0.215 | 30.43 / 0.762 / 0.139 | 30.61 / 0.962 / 0.062 |
| **NVP** | **30.73 / 0.891 / 0.103** | **30.96 / 0.897 / 0.090** | **31.38 / 0.896 / 0.054** |

**Key Findings**: NVP with 6 images matches or exceeds the performance of Explicit and Triplane with 20 images. Reducing from 20 to 6 images results in only a 0.65 dB drop in PSNR for NVP.

### Real Biological Specimen Experiment (MDCK Live Cells)

| Method | SSIM↑ | LPIPS↓ | PSNR↑ |
|:---|:---|:---|:---|
| Explicit | 0.9944 | 0.0051 | 36.65 |
| Triplane | 0.9762 | 0.0285 | 32.69 |
| **NVP** | **0.9977** | **0.0015** | **40.70** |

NVP reconstructs MDCK live cells from 19 fluorescence images, achieving a PSNR 4.05 dB higher than Explicit and 8.01 dB higher than Triplane. The Explicit method exhibits discontinuities and noise, while Triplane produces severe grid artifacts. NVP converges to SSIM > 0.99 within 20 minutes, whereas the Explicit baseline requires 60 minutes.

### Key Findings
1. NVP achieves approximately **50× reduction in measurements** (from 100+ images to only 6–7) and a **3× improvement in processing speed**.
2. On the synthetic tissue dataset, NVP achieves an SSIM of 0.4775, substantially outperforming Explicit (0.2954) and Triplane (0.1323).
3. NVP simultaneously reconstructs both continuous structures (blood vessels) and sparse structures (neurons), demonstrating robustness across different morphologies.
4. The self-calibration module contributes significantly to reconstruction quality (ablation results provided in the appendix).

## Highlights & Insights

1. **First sparse-view biological volume reconstruction from diffraction fluorescence images**: Reconstruction of the 3D refractive index from as few as ~6 images opens new possibilities for real-time dynamic biological imaging.
2. **Physics-driven neural representation design**: Integrating diffractive optics physical priors into a neural field framework resolves the failure of geometric optics models at the microscale.
3. **Demonstrated advantages of hybrid representation**: The paper clearly illustrates how the explicit–implicit hybrid representation overcomes the limitations of each component individually—the explicit grid provides sparse spatial priors to prevent overfitting, while the implicit MLP provides spatial correlation encoding to compensate for missing information.

## Limitations & Future Work

1. Validation is currently limited to the FDT imaging modality; generalization to other optical imaging systems has not been demonstrated.
2. A domain gap persists between synthetic and real data; coherence alignment is only an approximate solution.
3. Volume size is constrained by GPU memory; patch-based processing may be required for larger 3D volumes.
4. Refractive index reconstruction bias remains substantial for complex scattering specimens such as tissue.

## Related Work & Insights

- **Sparse-view 3D reconstruction**: NeRF-based methods (RegNeRF, DietNeRF), depth regularization, diffusion priors, etc., all based on geometric optics.
- **Diffractive optics 3D reconstruction**: Born approximation, multi-layer Born models, CNN/implicit neural field-based phase retrieval, all requiring large numbers of images.
- **Hybrid neural representations**: K-Planes, TensoRF, Instant-NGP, etc., lacking wave optics physical priors.

## Rating
- Novelty: ★★★★☆ (combinatorial innovation of physical priors and hybrid representation; first sparse-view reconstruction in FDT)
- Experimental Thoroughness: ★★★★☆ (comprehensive synthetic and real experiments, though baselines are limited and self-implemented)
- Value: ★★★★★ (~50× reduction in measurements is highly significant for in vivo dynamic imaging)
- Writing Quality: ★★★★☆ (physical model clearly articulated with rich illustrations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICCV 2025\] Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)

</div>

<!-- RELATED:END -->
