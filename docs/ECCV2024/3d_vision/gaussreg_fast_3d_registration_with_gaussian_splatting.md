---
title: >-
  [Paper Note] GaussReg: Fast 3D Registration with Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][Gaussian Splatting] This work presents the first exploration of registration between 3D Gaussian Splatting scenes, proposing a coarse-to-fine GaussReg framework. The coarse stage utilizes point cloud registration to estimate the initial transformation, while the fine stage extracts volumetric features from rendered images for fine-grained alignment, achieving comparable accuracy to HLoc while being 44x faster.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Gaussian Splatting"
  - "3D Registration"
  - "Coarse-to-Fine"
  - "Image-Guided"
  - "Scene Reconstruction"
date: 2026-05-08
content_hash: 9bacc061ad9aca84
---

# GaussReg: Fast 3D Registration with Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2407.05254](https://arxiv.org/abs/2407.05254)  
**Code**: [https://jiahao620.github.io/gaussreg](https://jiahao620.github.io/gaussreg)  
**Area**: 3D Vision  
**Keywords**: Gaussian Splatting, 3D Registration, Coarse-to-Fine, Image-Guided, Scene Reconstruction

## TL;DR

This work presents the first exploration of registration between 3D Gaussian Splatting scenes, proposing a coarse-to-fine GaussReg framework. The coarse stage utilizes point cloud registration to estimate the initial transformation, while the fine stage extracts volumetric features from rendered images for fine-grained alignment, achieving comparable accuracy to HLoc while being 44x faster.

## Background & Motivation

**Background**: Point cloud registration is relatively mature (e.g., ICP, GeoTransformer). NeRF and 3D Gaussian Splatting (3DGS) have rapidly emerged as novel 3D scene representations, creating an increasingly urgent demand for registration between sub-scenes in large-scale scene reconstruction.

**Limitations of Prior Work**: NeRF registration methods mainly fall into two categories: (a) NeRFuser utilizes SfM methods to re-estimate poses from rendered images, which is highly time-consuming; (b) DReg-NeRF converts implicit radiation fields into explicit voxels before registration, but struggles with unbounded scenes and is limited by voxel resolution.

**Key Challenge**: 3DGS provides an explicit representation similar to point clouds, which in theory can be directly processed by point cloud registration methods. However, GS point clouds themselves contain significant noise and geometric distortions, leading to low accuracy in direct coarse registration. Meanwhile, the rich image information contained in GS remains unexploited.

**Goal**: How to rapidly and accurately register two 3D Gaussian Splatting scene models?

**Key Insight**: Leveraging the dual properties of GS—both the geometric structure of point clouds (for coarse registration) and the highly efficient rendering of images (for fine registration).

**Core Idea**: A two-stage framework combining point cloud registration for coarse alignment and volume feature extraction from GS-rendered images for fine alignment.

## Method

### Overall Architecture

GaussReg consists of two stages: **Coarse Registration** and **Image-Guided Fine Registration**.

The input consists of two GS models of overlapping scenes A and B, $Gaussian_A$ and $Gaussian_B$, along with their respective camera poses of training images $\{C_i^A\}$ and $\{C_j^B\}$. The objective is to estimate a rigid transformation $\{s, R, T\}$ (including a scale factor) to align scene B with A.

### Key Designs

1. **Coarse Registration Module**: Extracts point clouds $Points_A$, $Points_B$ from the GS models, filtering reliable points with opacity $\alpha > 0.7$, where each point uses $(x, y, z, \alpha, r, g, b)$ as a 7-channel input. The workflow of GeoTransformer is adopted—extracting multi-scale features via a shared KPConv-FPN, where low-level features $F^{low}$ are used for superpoint matching and high-level features $F^{high}$ are used for ICP point matching. During training, the input GS point clouds are augmented not only with rotation/translation but also with **scale augmentation** to address scale uncertainty in monocular video reconstruction. The loss functions follow GeoTransformer's overlap-aware circle loss and point matching loss.

2. **Overlap Image Selection**: A prefix step for fine registration—efficiently selecting a subset of cameras with high overlap from both scenes based on the coarse registration results. The process consists of 3 steps:

    - Uniformly sample 30 poses from each training pose set and align the cameras of B to the coordinate system of A using the coarse transformation $\{s_c, R_c, T_c\}$;
    - Among the 30x30 camera pairs, first filter out the top-k pairs ($k=10$) using **camera orientation cosine similarity**, and then utilize rapid rendering of low-resolution depth maps in GS to calculate the **view-shared area**, finding the camera pair with the maximum overlap $(C_{i0}^a, \hat{C}_{j0}^b)$;
    - Select a group of training cameras around the best camera pair and render the image sets $I_A$, $I_B$.

3. **I3D Feature Extraction**: Inspired by Multi-View Stereo (MVS) principles. Rendered images are fed into a 2D CNN to extract reference features $Ref_A$ and source features $\{Src_A^k\}$, and the cost volume $Cost_A$ is constructed via differentiable homography warp. After 3D CNN regularization, the probability volume $P_A \in \mathbb{R}^{D \times H \times W}$ and feature volume $F_A \in \mathbb{R}^{C \times D \times H \times W}$ are obtained. For each pixel $p$, the depth hypothesis layer with the maximum probability is selected:

$$l_0 = \arg\max_{l=0,...,D-1} \{P_A^l(p) + P_A^{l+1}(p)\}$$

and cost volume features $F_A^{Cost}(p)$, depth $Depth_A(p)$, and confidence $Conf_A(p)$ are obtained via probability-weighted interpolation. Finally, $Ref_A$, $F_A^{Cost}$, and $Depth_A$ are concatenated and passed through convolutions, then filtered by confidence to obtain high- and low-resolution features:

$$f_A^{high} = Conv(Concat(Ref_A, F_A^{Cost}, Depth_A))[Conf_A > Mean(Conf_A)]$$

which are then projected into 3D space for registration.

### Loss & Training

The total loss of fine registration consists of two parts:

$$L_{total} = \lambda L_{depth} + L_{regis}$$

where $\lambda = 10$, and $L_{depth}$ is the cross-entropy loss supervising the probability volume:

$$L_{depth} = \sum_{p \in \Omega_A} -P_A^{gt}(p) \log P_A(p) + \sum_{p \in \Omega_B} -P_B^{gt}(p) \log P_B(p)$$

$L_{regis}$ is consistent with the registration loss in coarse registration. The two networks are trained separately for 40 epochs, with the learning rate starting at $1e\text{-}4$ and exponentially decaying by 0.05 each epoch. In fine registration, $n=5$ images are rendered, with $D=64$ depth hypotheses.

### GS Fusion and Filtering

Both GS models are merged after registration: applying the transformation $\{s_f, R_f, T_f\}$ to the position, rotation, and scale of $Gaussian_B$, where the rotation of Spherical Harmonics (SH) is computed via pseudo-inverse matrices leveraging the linear transformation properties of SH. During fusion, Gaussians near their respective centers are preserved, while redundant boundary Gaussians are filtered out.

## Key Experimental Results

### Main Results

**ScanNet-GSReg Dataset (82 test scenes)**:

| Method | RRE↓ | RTE↓ | RSE↓ | Success Rate↑ | Time (s)↓ |
|------|------|------|------|---------|----------|
| HLoc (SP+SG) | 2.725 | 0.099 | 0.098 | 75.6% | 212.3 |
| FGR | 157.126 | 3.328 | 0.268 | 100% | 3.4 |
| REGTR | 80.095 | 2.768 | 0.408 | 100% | 3.5 |
| **GaussReg** | **2.827** | **0.042** | **0.032** | **100%** | **4.8** |

**Objaverse Dataset (44 objects)**:

| Method | RRE↓ | ATE↓ |
|------|------|------|
| FGR | 61.59 | 13.50 |
| REGTR | 113.78 | 43.31 |
| DReg-NeRF | 9.67 | 3.85 |
| **GaussReg (w/o fine)** | **2.47** | **3.46** |

### Ablation Study

| Configuration | RRE↓ | RTE↓ | RSE↓ | Success Rate↑ | Time (s)↓ |
|------|------|------|------|---------|----------|
| HLoc | 2.725 | 0.099 | 0.098 | 75.6% | 212.3 |
| Coarse registration only | 3.403 | 0.061 | 0.034 | 100% | 3.7 |
| Coarse registration + HLoc refinement | 1.104 | 0.186 | 0.278 | 51.2% | 206.8 |
| **Full GaussReg** | **2.827** | **0.042** | **0.032** | **100%** | **4.8** |

Ablation of Top-k overlap image selection: $k=10$ is the optimal balance point, where $k<10$ leads to a drop in accuracy, and $k>10$ yields marginal returns.

### Key Findings

- GS point clouds contain high noise, making direct registration difficult for traditional point cloud methods (FGR, REGTR). GaussReg is **44x faster** than HLoc (4.8s vs. 212.3s) with a 100% success rate.
- Image-guided fine registration performs better than pure HLoc refinement, achieving a 100% success rate vs. 51.2%.
- On Objaverse, using coarse registration alone outperforms DReg-NeRF, demonstrating the effectiveness of utilizing GS features (e.g., the opacity channel).

## Highlights & Insights

- **Novelty**: The first method targeting 3DGS scene registration, demonstrating that the "dual properties" of GS point clouds (geometry + rendering) can be fully leveraged.
- **Practical Value**: A 44x speedup with the success rate increased from 75.6% to 100%, carrying significant engineering relevance.
- **Clear Design Logic**: The coarse-to-fine framework is simple yet effective, and the image guidance capitalizes on the fast rendering advantage of GS.
- **Comprehensive Benchmark**: Constructs two evaluation datasets: ScanNet-GSReg (1379 scenes) and GSReg (10 outdoor/wild scenes).

## Limitations & Future Work

- The GS fusion strategy is simple, which may lead to inconsistencies at fusion boundaries when the illumination conditions of the two scenes differ.
- Fine registration requires the camera pose information of training images, limiting its usage on pure GS models (which lack pose information).
- The possibility of training both stages in an end-to-end manner has not been explored.
- Ablation studies show that removing I3D feature extraction results in better depth accuracy but worse registration performance, indicating that the fusion mechanism of image and geometric features still has room for optimization.

## Related Work & Insights

- **GeoTransformer**: The backbone for coarse registration, providing a mature point cloud registration pipeline.
- **DReg-NeRF**: A representative method for NeRF registration, but limited by voxelized representations.
- **MVSNet**: The multi-view stereo concept is referenced for I3D feature extraction.
- Insight: The rapid rendering capability of GS can provide additional information for more downstream tasks (such as depth completion and semantic segmentation).

## Rating

- Novelty: ⭐⭐⭐⭐ The first GS registration method; the coarse-to-fine approach is reasonable but not a disruptive innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on multiple datasets + exhaustive ablation studies + self-constructed benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ 44x speedup + 100% success rate, highly practical, laying the foundation for large-scale GS scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)

</div>

<!-- RELATED:END -->
