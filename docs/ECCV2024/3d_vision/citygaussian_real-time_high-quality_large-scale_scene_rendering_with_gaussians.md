---
title: >-
  [Paper Note] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians
description: >-
  [ECCV2024][3D Vision][3D Gaussian Splatting] CityGaussian (CityGS) is proposed to enable high-quality 3D Gaussian Splatting training and cross-scale real-time rendering for city-scale scenes (> 1.5 km²) for the first time, leveraging a divide-and-conquer training strategy and a block-wise Level-of-Detail (LoD) mechanism.
tags:
  - "ECCV2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Large-Scale Scene Reconstruction"
  - "Level-of-Detail"
  - "Divide-and-Conquer"
  - "novel view synthesis"
date: 2026-05-08
content_hash: fdcc7298108f4738
---

# CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians

**Conference**: ECCV2024  
**arXiv**: [2404.01133](https://arxiv.org/abs/2404.01133)  
**Code**: [dekuliutesla/citygs](https://github.com/DekuLiuTesworworla/citygs)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Large-Scale Scene Reconstruction, Level-of-Detail, Divide-and-Conquer, novel view synthesis

## TL;DR

CityGaussian (CityGS) is proposed to enable high-quality 3D Gaussian Splatting training and cross-scale real-time rendering for city-scale scenes (> 1.5 km²) for the first time, leveraging a divide-and-conquer training strategy and a block-wise Level-of-Detail (LoD) mechanism.

## Background & Motivation

3D Gaussian Splatting (3DGS) has achieved real-time, high-quality results in novel view synthesis of small scenes thanks to its explicit Gaussian primitives and efficient tile-based rasterization. However, directly applying it to large-scale scenes (e.g., city-scale areas) faces two core bottlenecks:

1. **Training Out-Of-Memory (OOM)**: Covering a 1.5 km² city area requires over 20 million Gaussian primitives, which leads to OOM even on a 40GB A100 GPU. A single 24GB RTX 3090 crashes when the number of Gaussians exceeds 11 million.
2. **Rendering Speed Degradation**: As the number of Gaussians grows from millions to tens of millions, depth sorting becomes a significant bottleneck. On the MatrixCity scene (23 million Gaussians), the frame rate drops to only 21 FPS, which falls short of real-time requirements.

Existing NeRF-based large-scale methods (such as Block-NeRF, Mega-NeRF, and Switch-NeRF) adopt divide-and-conquer strategies but are built on implicit representations, resulting in insufficient detail fidelity and slow rendering speeds. Although VastGaussian extends 3DGS to large-scale environments, it still fails to achieve real-time rendering.

## Core Problem

How to efficiently train city-scale 3DGS under limited GPU memory, while maintaining real-time rendering across massive scale transitions (from close-up views to high-altitude bird's-eye views)?

## Method

### 1. Global Gaussian Prior Generation

First, standard 3DGS training is performed on the COLMAP point clouds using all training images for 30,000 iterations to generate a coarse global Gaussian prior $\mathbf{G}_K$. This prior provides:

- Global geometric distribution awareness, preventing geometrically inaccurate floaters in subsequent block-wise training.
- Cleaner rendered images to facilitate subsequent data partitioning.

### 2. Divide-and-Conquer Training Strategy

**Spatial Contraction and Partitioning**: Large-scale scenes are typically unbounded. Directly partitioning with uniform grids leads to many empty blocks and imbalanced workloads. The proposed method:

- Defines the foreground region $[\mathbf{p}_{min}, \mathbf{p}_{max}]$ and normalizes the Gaussian positions to $[-1, 1]$.
- Applies a non-linear contraction mapping ($L_\infty$-norm) to the background regions outside the foreground to compress the unbounded space into a $[-2, 2]$ cube.
- Performs uniform grid partitioning in the contracted space to achieve a more balanced distribution of Gaussians.

**Adaptive Data Partitioning**: Instead of selecting training views simply based on distance, the views' contributions to the block are determined based on SSIM loss:

- **Principle 1 (Eq. 3)**: Render images with and without the block's Gaussians. If the SSIM difference is $> \epsilon$, this view has a significant contribution to the block and is kept.
- **Principle 2 (Eq. 4)**: If the camera position lies within the block boundary, the view is directly kept (to prevent artifacts near the block boundaries).
- The union of both principles is taken as the final data assignment.

**Parallel Fine-Tuning and Fusion**:

- Each block is initialized with the global prior and independently trained for 30,000 iterations.
- A combined L1 + SSIM loss is used.
- After training, Gaussians are cropped at the spatial boundaries of each block and directly concatenated to achieve seamless fusion (since the global prior resolves inter-block interference).

### 3. Block-wise Level-of-Detail (LoD)

**Multi-level Detail Generation**: Generate three levels of detail for the fused Gaussians using the LightGaussian compression strategy:

- LoD 2 (50% compression): Most detailed
- LoD 1 (34% compression): Medium detail
- LoD 0 (25% compression): Coarsest detail

**Block-wise Visibility Logic and Level Selection**:

- Evaluate block visibility by calculating the Intersection-over-Union (IoU) between the view frustum and the 8 corners of the block configured during training.
- Apply the MAD (Median Absolute Deviation) algorithm to eliminate the impact of floaters on the bounding boxes, yielding tighter boundaries.
- Select the detail level based on the minimum distance from the block's 8 corners to the camera center: LoD 2 for 0-200m, LoD 1 for 200-400m, and LoD 0 for >400m.
- All Gaussians inside the same block share the same LoD level to avoid point-wise distance calculation overhead.

**Fused Rendering**: Gaussians from different LoD levels are directly concatenated and fed into the rasterizer, showing virtually no visible discontinuities.

## Key Experimental Results

### Rendering Quality (No-LoD Version vs. SOTA)

| Method | MatrixCity PSNR↑ | MatrixCity SSIM↑ | Rubble PSNR↑ | Building PSNR↑ |
|------|:-:|:-:|:-:|:-:|
| Mega-NeRF | - | - | 24.06 | 20.93 |
| Switch-NeRF | - | - | 24.31 | 21.54 |
| 3DGS† | 23.67 | 0.735 | 25.47 | 20.46 |
| **CityGS** | **27.46** | **0.865** | **25.77** | **21.55** |

- PSNR improves by **+3.79 dB** and SSIM by +0.13 on MatrixCity (2.7 km² synthetic city).
- Successfully reconstructs the entire MatrixCity (camera altitude 150m-500m) for the first time, whereas prior methods failed to train stably.

### LoD Results

| Mode | SSIM | PSNR | FPS |
|------|:-:|:-:|:-:|
| Without LoD | 0.865 | 27.46 | 21.6 |
| LoD 2 Only | 0.863 | 27.54 | 45.6 |
| LoD 0 Only | 0.825 | 26.57 | 69.4 |
| **LoD (Hybrid)** | **0.855** | **27.32** | **53.7** |

- The LoD strategy accelerates FPS from 21.6 to 53.7 (**2.5× speedup**) with only a 0.14 dB drop in PSNR.
- Under extreme high-altitude viewpoints, only the hybrid LoD can maintain a minimum real-time performance of FPS > 25 across all altitudes.

### Ablation Study

| Configuration | PSNR | SSIM | Number of Gaussians |
|------|:-:|:-:|:-:|
| Baseline (Nearest Camera Selection) | 23.98 | 0.779 | 12.2M |
| + Global Prior | 25.01 | 0.801 | 15.4M |
| CityGS (Full Strategy) | **25.77** | **0.813** | 9.7M |

- The global prior significantly improves quality (+1.03 dB).
- Adaptive data partitioning (Eq. 3 + 4) yields a further boost of +0.76 dB while reducing Gaussian consumption by 37%.

### LoD Strategy Ablation

- Block-wise selection vs. point-wise selection: FPS 53.7 vs. 30.3, maintaining comparable quality while rendering 77% faster.
- Impact of distance intervals: [0,200], [200,400], [400,∞] m achieves the best balance between quality and speed.

## Highlights & Insights

1. **Global Prior + Divide-and-Conquer Fine-Tuning**: A two-stage coarse-to-fine strategy elegantly resolves inter-block interference and floaters, ensuring seamless fusion.
2. **SSIM-Based Adaptive Data Partitioning**: Compared to simple spatial distance selection, this precisely filters out training views that contribute substantially to each block, reducing irrelevant data overhead and Gaussian point count.
3. **Block-wise LoD**: Selecting detail levels based on blocks instead of individual points eliminates the overhead of point-wise distance computing, achieving consistent real-time rendering across scales.
4. **MAD for Floater Removal**: Estimating block bounding boxes with Median Absolute Deviation effectively filters out the interference of floaters during frustum culling.

## Limitations & Future Work

1. **Static Scene Assumption**: Inability to handle dynamic objects (e.g., pedestrians, vehicles) limits practical urban applications.
2. **Performance Degradation with Mixed Perspectives**: The paper admits that training with combined aerial and street-view perspectives degrades performance, which remains an open challenge.
3. **Dependence on External Compression**: The LoD detail generation directly adopts LightGaussian without presenting a specialized compression design tailored to large-scale scenes.
4. **Block Boundary Handling**: Although the global prior mitigates inter-block transitions, visible seams might still occur under extreme conditions.
5. **Training Overhead**: The total training time is relatively long because it requires training a global prior followed by block-wise fine-tuning.

## Related Work & Insights

| Method | Representation | Divide-and-Conquer | LoD | Real-time | Large-Scale Quality |
|------|:----:|:----:|:---:|:----:|:----------:|
| Mega-NeRF | Implicit MLP | ✓ | ✗ | ✗ | Fair |
| Switch-NeRF | Implicit MLP | ✓ (Learnable) | ✗ | ✗ | Fair |
| BungeeNeRF | Implicit MLP | ✗ | ✓ (Progressive) | ✗ | Fair |
| VastGaussian | 3DGS | ✓ | ✗ | ✗ | Good |
| **CityGS** | **3DGS** | **✓** | **✓** | **✓** | **SOTA** |

- Compared to VastGaussian: CityGS additionally introduces LoD to achieve real-time rendering, and avoids handling appearance variations thanks to the global prior.
- Compared to NeRF-based methods: Rendering quality is significantly superior (PSNR +1.5~3.8 dB) with added real-time capabilities.

## Related Work & Insights

1. **Divide-and-Conquer + Global Prior** represents a general paradigm for explicit large-scale representations, which can be extended to tasks like large-scale mesh reconstruction and point cloud completion.
2. The concept of **Block-wise LoD** can be combined with hierarchical Gaussians (e.g., Octree-GS) to achieve finer-grained multi-scale representations.
3. **Dynamic Scene Extension** is a clear future direction, possibly integrating 4D Gaussians/Deformable Gaussians to manage time-varying contents.
4. The **Adaptive Data Partitioning Method** (based on SSIM contribution) holds reference value for other 3DGS variants needing efficient data utilization.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combined framework of divide-and-conquer training and block-wise LoD is highly systematic with well-designed modules)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 scenes of different scales, comprehensive ablation studies, multi-perspective FPS analysis, and street-view generalization validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables, and well-motivated discussions)
- Value: ⭐⭐⭐⭐ (The first work to achieve real-time rendering for city-scale 3DGS, holding both engineering and academic value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](../../ICCV2025/3d_vision/momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](../../CVPR2025/3d_vision/sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](../../CVPR2025/3d_vision/evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[ECCV 2024\] Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction](deceptive-nerf3dgs_diffusion-generated_pseudo-observations_for_high-quality_spar.md)

</div>

<!-- RELATED:END -->
