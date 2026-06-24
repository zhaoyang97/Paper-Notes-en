---
title: >-
  [Paper Note] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis
description: >-
  [ECCV 2024][Medical Imaging][3D Gaussian Splatting] This paper proposes X-Gaussian, the first framework to apply 3D Gaussian Splatting (3DGS) to X-ray novel view synthesis. By designing a radiative Gaussian point cloud model (replacing spherical harmonics) and an angle-pose cuboid uniform initialization strategy (replacing SfM), it outperforms SOTA NeRF methods by 6.5 dB while achieving 73× inference acceleration with only 15% of the training time.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "3D Gaussian Splatting"
  - "X-ray imaging"
  - "novel view synthesis"
  - "CT reconstruction"
  - "differentiable rendering"
date: 2026-05-08
content_hash: b62be2249ce983c0
---

# Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2403.04116](https://arxiv.org/abs/2403.04116)  
**Code**: [caiyuanhao1998/X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian)  
**Area**: Medical Imaging  
**Keywords**: 3D Gaussian Splatting, X-ray imaging, novel view synthesis, CT reconstruction, differentiable rendering

## TL;DR

This paper proposes X-Gaussian, the first framework to apply 3D Gaussian Splatting (3DGS) to X-ray novel view synthesis. By designing a radiative Gaussian point cloud model (replacing spherical harmonics) and an angle-pose cuboid uniform initialization strategy (replacing SfM), it outperforms SOTA NeRF methods by 6.5 dB while achieving 73× inference acceleration with only 15% of the training time.

## Background & Motivation

**Background**: X-ray novel view synthesis (NVS) aims to generate X-ray images from unscanned angles using existing projections, which is of great clinical value for reducing patient radiation exposure and assisting CT reconstruction. Current methods are mostly based on NeRF, which samples a large number of 3D points along rays and computes them one by one, resulting in extremely slow training and inference speeds.

**Limitations of Prior Work**: Even NAF, the most efficient NeRF-based method, requires more than 1 hour of training and achieves an inference speed of only 2 fps, which falls far short of clinical real-time requirements. While 3DGS has demonstrated efficiency vastly superior to NeRF in natural light imaging, it has not yet been applied to the X-ray domain.

**Key Challenge**: The two core components of 3DGS—Spherical Harmonics (SH) and Structure-from-Motion (SfM) initialization—are both inapplicable to X-ray imaging. SH is designed to model view-dependent anisotropic colors, whereas X-ray radiative intensity is isotropic. SfM relies on feature detection and matching, but X-ray images are grayscale, low-contrast, and exhibit transmission overlap, which severely degrades SfM accuracy.

**Goal**: How to transfer the high-efficiency rendering advantages of 3DGS to the X-ray imaging domain while resolving physical imaging model mismatches and inapplicable initialization strategies.

**Key Insight**: Starting from the physical nature of X-ray imaging—penetration attenuation rather than surface reflection—the radiative intensity representation function and rasterization process of Gaussian point clouds are redesigned, and the X-ray scanner parameters are directly utilized to compute camera parameters and initialize the point cloud.

**Core Idea**: Based on the physical characteristics of isotropic transmission imaging in X-rays, this work designs a view-independent radiative Gaussian point cloud model and an SfM-free cuboid uniform initialization strategy, fundamentally achieving efficient adaptation of 3DGS to X-ray NVS.

## Method

### Overall Architecture

The pipeline of X-Gaussian consists of three steps: (1) using the ACUI strategy to compute intrinsic and extrinsic matrices from X-ray scanner parameters, and uniformly sampling the initial point cloud within a bounding cuboid of the scanned object; (2) using the radiative Gaussian point cloud model to learn the radiative features of each 3D point; (3) rendering X-ray projection images through Differentiable Radiative Rasterization (DRR) and optimizing them by computing the loss against the ground truth.

### Key Designs

1. **Radiative Gaussian Point Cloud Model**: The core innovation is replacing Spherical Harmonics (SH) in the original 3DGS with a Radiative Intensity Response Function (RIRF). Each Gaussian point cloud learns a feature vector $\mathbf{f} \in \mathbb{R}^{N_f}$, and the radiative intensity is calculated by the following function:

    $\mathbf{i}(\mathbf{f}) = \text{RIRF}(\mathbf{f}) = \text{Sigmoid}(\boldsymbol{\lambda} \odot \mathbf{f})$

   where $\boldsymbol{\lambda} \in \mathbb{R}^{N_f}$ is a constant weight vector. The key is that this function **does not include the viewing direction $\mathbf{d}$**, which aligns with the isotropic nature of X-rays. In comparison, the original SH models view-dependent anisotropic color $\mathbf{c}(\mathbf{d}, \mathbf{k}) = \sum_{l,m} k_l^m Y_l^m(\theta, \phi)$, whereas RIRF completely eliminates view dependency.

   Design Motivation: Natural light imaging relies on surface reflection, where the color of the same point varies from different angles (anisotropic). X-ray imaging relies on penetration attenuation, where radiative density is an intrinsic property of the material, independent of the observation direction (isotropic).

2. **Differentiable Radiative Rasterization (DRR)**: Given a perspective, 3D Gaussians are projected onto a 2D detector plane. The projection center coordinates are transformed via extrinsic and intrinsic matrices:

    $\tilde{\mathbf{t}}_i = \mathbf{M}_{ext} \tilde{\boldsymbol{\mu}}_i, \quad \tilde{\mathbf{u}}_i = \mathbf{M}_{int} \tilde{\mathbf{t}}_i$

   The 3D covariance matrix is projected into camera coordinates using a Jacobian matrix: $\boldsymbol{\Sigma}_i' = \mathbf{J}_i \mathbf{W}_i \boldsymbol{\Sigma}_i \mathbf{W}_i^\top \mathbf{J}_i^\top$. The intensity of pixel $p$ is calculated via alpha blending:

    $\mathbf{I}(p) = \sum_{j \in \mathcal{N}} \mathbf{i}_j \sigma_j \prod_{k=1}^{j-1}(1-\sigma_k), \quad \sigma_j = \alpha_j P(\mathbf{x}_j | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$

   Since computations related to viewing direction are bypassed, both the forward and backward passes of DRR are faster than standard RGB rasterization. The entire DRR is implemented in CUDA for highly efficient GPU parallelization.

3. **Angle-pose Cuboid Uniform Initialization (ACUI)**: This completely bypasses SfM by directly computing camera matrices using the known parameters of a cone-beam X-ray scanner:

    $\mathbf{M}_{ext} = \begin{bmatrix} -\sin\phi & \cos\phi & 0 & 0 \\ 0 & 0 & -1 & 0 \\ -\cos\phi & -\sin\phi & 0 & L_{SO} \\ 0 & 0 & 0 & 1 \end{bmatrix}$

   where $L_{SO}$ is the source-to-object distance, and $\phi$ is the angular pose. Point cloud initialization is obtained by uniformly sampling at intervals $d$ inside the bounding cuboid of the object:

    $\mathcal{P} = \left\{ \left(\frac{n_1 S_1 d}{M_1}, \frac{n_2 S_2 d}{M_2}, \frac{n_3 S_3 d}{M_3}\right) \right\}$

   Design Motivation: X-ray images are grayscale, low-contrast, and involve transmission overlapping, causing a severe degradation in the feature detection and matching accuracy of SfM. Since the geometric parameters of cone-beam scanning are completely known, camera intrinsics and extrinsics can be directly calculated, avoiding the time-consuming SfM process.

### Loss & Training

The training objective is a weighted sum of the $\mathcal{L}_1$ loss and the SSIM loss:

$$\mathcal{L} = (1-\gamma)\mathcal{L}_1 + \gamma\mathcal{L}_{\text{SSIM}}$$

where $\gamma = 0.2$. An Adam optimizer ($\beta_1=0.9$, $\beta_2=0.999$) is used for $2 \times 10^4$ iterations. The learning rate for point cloud positions is exponentially decayed from $1.9 \times 10^{-4}$ to $1.9 \times 10^{-6}$. The adaptive density control strategy from original 3DGS is employed to dynamically adjust the number of point clouds $N_p$.

## Key Experimental Results

### Main Results

| Method | Inference Speed | Training Time | Average PSNR (dB) | Average SSIM |
|------|---------|---------|-------------|---------|
| InTomo | 0.62 fps | 125 min | 30.187 | 0.9611 |
| NeRF | 0.14 fps | 313 min | 30.289 | 0.9617 |
| TensoRF | 0.77 fps | 178 min | 30.477 | 0.9194 |
| NeAT | 1.78 fps | 69 min | 34.201 | 0.9366 |
| NAF | 2.01 fps | 63 min | 36.942 | 0.9627 |
| **X-Gaussian** | **148 fps** | **9 min** | **43.404** | **0.9993** |

X-Gaussian out-performs the best NeRF method, NAF, by **6.5 dB**, with an inference speed **73× faster** and a training time that is only **15%** of NAF's.

### Ablation Study

| Configuration | PSNR (dB) | SSIM | Training Time (s) | Inference Speed (fps) |
|------|---------|------|-----------|-------------|
| Original 3DGS (baseline) | 37.21 | 0.9813 | 1898 | 64 |
| + ACUI (replace SfM) | 38.87 (+1.66) | 0.9871 | 1172 (↓34%) | 72 |
| + DRR (replace RGB rasterization) | **43.40 (+4.53)** | **0.9993** | **538 (↓54%)** | **148 (2.1×)** |

Initialization strategy comparison:

| Strategy | PSNR (dB) | Training Time (s) | Inference Speed (fps) |
|------|---------|-----------|-------------|
| Random | 41.33 | 601 | 112 |
| Spherical | 42.84 | 575 | 136 |
| FDK | **43.47** | 1394 | 93 |
| **Cubic (ACUI)** | 43.40 | **538** | **148** |

### Key Findings

1. **The radiative Gaussian point cloud model contributes the most**: Replacing RGB rasterization with DRR yields a 4.53 dB PSNR improvement (from 38.87 to 43.40), while halving training time and doubling inference speed. This demonstrates that modeling the isotropic characteristics of X-rays is crucial.
2. **ACUI is highly efficient and effective**: Although FDK initialization yields a slightly higher PSNR (+0.07 dB), its training time is 2.59× longer than ACUI's, and its inference speed is slower. ACUI achieves the best balance between performance and efficiency.
3. **Sparse-view CT reconstruction shows great practical value**: Using X-Gaussian to generate novel-view projections to assist ASD-POCS reconstruction yields a PSNR improvement of 13.53 dB (from 17.03 to 30.56), far surpassing NAF's improvement of 10.88 dB.
4. **Convergence speed is significantly faster than original 3DGS and NAF**: At 1000 training iterations, X-Gaussian's point cloud already begins to outline the object shape; within 60 seconds of training, the rendering quality already surpasses NAF's after 180 seconds of training.
5. **Feature dimension $N_f=16$ offers the best trade-off**: The performance gap between $N_f=16$ and $N_f=32$ is only 0.013 dB, yet the former yields faster training and inference.

## Highlights & Insights

- **Physical-driven model design**: Designing each component of 3DGS starting from the physical nature of X-ray imaging (penetration attenuation and isotropy). Rather than simply "subtracting SH," the authors rethink how radiative intensity is represented; the designed RIRF is elegant and highly efficient.
- **Extreme efficiency boost**: An inference speed of 148 fps enables real-time X-ray view synthesis, while the 9-minute training time significantly reduces the barrier to clinical deployment.
- **End-to-end system design**: From initialization to the point cloud model and then to rasterization, every stage is custom-tailored for X-ray characteristics, forming a complete technology stack.
- **Practical application in sparse-view CT reconstruction** demonstrates clinical value beyond pure academic research, enabling high-quality CT reconstruction with fewer X-ray scans and directly reducing patient radiation exposure.

## Limitations & Future Work

1. **High implementation complexity**: The core CUDA implementation is difficult to debug and has low interpretability, making it less straightforward to replicate and modify than a PyTorch implementation.
2. **Limited to cone-beam scanning scenarios**: The ACUI strategy relies on standard scanning geometries and requires adaptation for non-standard scanning configurations (e.g., helical CT or non-uniform angular intervals).
3. **Polychromatic X-rays not considered**: X-rays in actual scans are polychromatic with differing attenuation characteristics. The current model's assumption of a single attenuation coefficient may be inaccurate in complex scenarios.
4. **Lack of verification on real clinical data**: The experiments use projections simulated with the TIGRE toolbox, without verification on data collected from real X-ray scanner hardware.
5. **Extensibility to other penetrative imaging modalities**: For example, ultrasound and near-infrared spectroscopy, which would require redesigning the radiative models based on the physical properties of each modality.

## Related Work & Insights

- **3DGS** [Kerbl et al.]: Gaussian splatting representation and rasterization for RGB scenes forms the baseline framework of this work. This paper demonstrates that the 3DGS design concept can be successfully transferred to non-natural light imaging domains.
- **NAF** [Zha et al.]: An Instant-NGP-based X-ray NeRF method, which was the previous SOTA and serves as the main baseline for comparison in this paper.
- **NeAT** [Rückert et al.]: Neural Adaptive Tomography, which outperforms NAF in some scenarios but is relatively slow.
- **TensoRF** [Chen et al.]: Tensor Radiance Fields reconstruction, an efficiency optimization method in RGB NeRF, which underperforms when directly applied to X-rays.
- **Key Insight of this work**: When transferring a general method (like 3DGS) to a specific domain (such as X-rays), it is vital to re-examine the applicability of each component based on **first physical principles** rather than applying things blindly.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply 3DGS to X-ray NVS. RIRF and ACUI design philosophies are clear, presenting a physical-driven methodology that is worth learning from.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 scenarios, rich ablation analyses (component ablation, initialization comparisons, hyperparameter sensitivity, convergence speed), and validation of downstream CT reconstruction applications.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous mathematical derivations in the method section and clear figures, though CUDA implementation details are somewhat sparse.
- **Value**: ⭐⭐⭐⭐⭐ The 148 fps inference speed and 9-minute training time make it highly practical for clinical deployment. The boost in sparse-view CT reconstruction holds direct significance for reducing patient radiation dose.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](../../CVPR2026/medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](../../AAAI2026/medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[NeurIPS 2025\] FOXES: A Framework For Operational X-ray Emission Synthesis](../../NeurIPS2025/medical_imaging/foxes_a_framework_for_operational_x-ray_emission_synthesis.md)
- [\[CVPR 2026\] Adaptive Anisotropic Gaussian Splatting for Multi-contrast MRI Arbitrary-Scale Super-Resolution with Anatomy Guidance](../../CVPR2026/medical_imaging/adaptive_anisotropic_gaussian_splatting_for_multi-contrast_mri_arbitrary-scale_s.md)
- [\[ECCV 2024\] A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image](a_cephalometric_landmark_regression_method_based_on_dual-encoder_for_high-resolu.md)

</div>

<!-- RELATED:END -->
