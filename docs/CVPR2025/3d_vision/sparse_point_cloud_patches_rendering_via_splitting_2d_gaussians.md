---
title: >-
  [Paper Note] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians
description: >-
  [CVPR 2025][3D Vision][Point Cloud Rendering] A novel method is proposed to directly predict 2D Gaussians from point clouds for photorealistic rendering. By employing an entire-patch architecture, it achieves cross-category generalization. A splitting decoder upsamples sparse point clouds into denser Gaussian primitives, achieving state-of-the-art rendering quality and a real-time rendering speed of 142 FPS with only 2K-100K points.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Rendering"
  - "2D Gaussian Splatting"
  - "Sparse Point Cloud"
  - "Cross-Category Generalization"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: e4a595136d4bca12
---

# Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians

**Conference**: CVPR 2025  
**arXiv**: [2505.09413](https://arxiv.org/abs/2505.09413)  
**Code**: [https://github.com/murcherful/GauPCRender](https://github.com/murcherful/GauPCRender)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Rendering, 2D Gaussian Splatting, Sparse Point Cloud, Cross-Category Generalization, Novel View Synthesis

## TL;DR

A novel method is proposed to directly predict 2D Gaussians from point clouds for photorealistic rendering. By employing an entire-patch architecture, it achieves cross-category generalization. A splitting decoder upsamples sparse point clouds into denser Gaussian primitives, achieving state-of-the-art rendering quality and a real-time rendering speed of 142 FPS with only 2K-100K points.

## Background & Motivation

**Background**: Point cloud rendering is a fundamental task in 3D vision. Traditional methods directly project points onto planes or spheres, which yields results full of holes and lacking realism. Recently, deep learning methods (such as NPBG, NPBG++) require multi-view images as additional inputs. Modern approaches (like TriVol, Point2Pix) predict NeRFs from point clouds, while PFGS predicts 3D Gaussians but requires subsequent refinement on the rendered images.

**Limitations of Prior Work**: (1) TriVol and Point2Pix rely on NeRF representations, resulting in slow rendering speeds (1.62 FPS); (2) although PFGS utilizes 3D Gaussians, it requires an additional recurrent decoder to refine rendering results, which is not truly end-to-end; (3) all of these methods rely on dense point clouds (80K-100K) and only generalize within a single category; (4) the normal ambiguity of 3D Gaussians leads to inaccurate geometry.

**Key Challenge**: High-quality point cloud rendering requires dense representations, whereas point clouds in real-world scenes are often sparse and unevenly distributed. Existing methods process the entire point cloud, leading to a heavy reliance on category priors and poor generalization.

**Goal**: To design a method capable of (1) directly predicting high-quality Gaussian primitives from sparse point clouds for rendering, (2) producing photorealistic quality without refinement, and (3) generalizing across categories and datasets.

**Key Insight**: Compared to 3DGS, 2D Gaussian Splatting (2DGS) features explicit normals, and point cloud normals can be easily estimated. This means point cloud normals can be utilized to initialize 2D Gaussians, providing a solid starting point for the network. Processing point cloud patches instead of the entire point cloud removes the constraints of category priors.

**Core Idea**: An entire-patch architecture is used to predict 2D Gaussians at the patch level (utilizing the global prediction as the "background" to achieve full-image supervision), and a splitting decoder is employed to split each point into K Gaussian primitives to handle sparse point clouds.

## Method

### Overall Architecture

The input is a point cloud (coordinates + colors), and the output is a rendered image. The network contains two identical 2D Gaussian Prediction Modules—one processing the entire point cloud ($\mathcal{N}_e$) and the other processing a point cloud patch ($\mathcal{N}_p$). During training, $\mathcal{N}_e$ is first pre-trained, and then frozen while training $\mathcal{N}_p$. The Gaussians predicted by $\mathcal{N}_e$ for the non-patch parts serve as the "background," which are merged with the patch Gaussians predicted by $\mathcal{N}_p$ to render the complete image for loss calculation. During inference, one can directly use $\mathcal{N}_p$ to process arbitrary point clouds in patches.

### Key Designs

1. **Normal-based 2D Gaussian Initialization**:

    - **Function**: To utilize point cloud geometry to provide a solid initialization for the network.
    - **Mechanism**: For each input point, the normal $\hat{n}$ is estimated. This normal is used to initialize the orientation of the 2D Gaussian (normal $n$ + rotation angle around normal $\alpha=0$), the point coordinates initialize the position, the point color initialises the spherical harmonic coefficients, the nearest neighbor distance initializes the scale, and the opacity is set to 1. Unlike the random quaternion initialization in 3DGS/2DGS, normal-based initialization ensures each Gaussian faces the correct direction and remains visible from any viewpoint, avoiding the difficulty of predicting quaternion rotations for the network.
    - **Design Motivation**: Ablation studies show that without normal-based initialization, the PSNR drops drastically from 27.88 to 9.62, as randomly initialized Gaussians are invisible from many viewpoints, causing the network to fail to converge.

2. **Splitting Decoder**:

    - **Function**: To "split" each initialized Gaussian into K finer Gaussians.
    - **Mechanism**: First, local features $F_l$ are extracted using a PointMLP encoder. Then, six splitting decoders predict K offsets for six parameters: position, scale, color, normal, rotation angle, and opacity. Taking position as an example, $\Delta_1^x, ..., \Delta_K^x = D_x(F_l, X^i, C^i, N^i, S^i)$, and the final position is $X^p = \bigcup_{j=1}^{K}(\Delta_j^x + X^i)$. Each decoder consists of weight-shared MLPs. When K=4, each point splits into 4 Gaussians, effectively increasing the representation density.
    - **Design Motivation**: Sparse point clouds cannot cover all details of the object's surface. Through splitting and upsampling, denser Gaussians can be generated using fewer input points to capture fine texture and geometric details. The splitting offsets are predicted based on learned local features, making it smarter than simple upsampling.

3. **Entire-Patch Architecture**:

    - **Function**: To enable the network to operate at the patch level for cross-category generalization.
    - **Mechanism**: The entire point cloud and the patch are processed by two structurally identical prediction modules. During training, a center point is randomly selected, and the nearest $N_p=2048$ points are taken as the patch. The Gaussians predicted by $\mathcal{N}_e$ for the entire point cloud (with the patch portion removed) serve as the "background," which is merged with the patch Gaussians predicted by $\mathcal{N}_p$ to render the complete image. Key Insight: Images rendered directly from patches are incomplete, making it impossible to compute effective loss functions. However, filling in the background with the entire point cloud prediction guarantees the completeness of the supervision signal.
    - **Design Motivation**: Processing local patches decouples the network from global point distribution patterns, allowing it to generalize to different categories and densities of point clouds. Previous methods required separate training for each category, whereas patch-based methods inherently possess cross-category capabilities.

### Loss & Training

A combination of MSE and SSIM loss is employed: $\mathcal{L} = \frac{1}{N_c}\sum_{i=1}^{N_c}(\beta \mathcal{L}_{MSE} + (1-\beta)\mathcal{L}_{SSIM})$, where $\beta=0.8$. Each sample is rendered from $N_c=8$ viewpoints for supervision. Image resolution: 256×256 for objects, 640×512 for scenes, 512×512 for human bodies. Splitting factor K=4, patch points $N_p=2048$. Adam optimizer with lr=1e-4, batch size of 8, up to 480 epochs. $\mathcal{N}_e$ is pre-trained first, followed by $\mathcal{N}_p$ training. Training takes approximately 30 hours on an RTX 3090.

## Key Experimental Results

### Main Results

| Dataset | Points | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|------|-------|-------|--------|
| Car (ShapeNet) | 100K | TriVol | 27.22 | 0.927 | 0.084 |
| Car (ShapeNet) | 100K | PFGS | 27.34 | 0.942 | 0.077 |
| Car (ShapeNet) | 100K | **Ours** | **28.73** | **0.960** | **0.060** |
| Car (ShapeNet) | 20K | **Ours** | **27.88** | **0.949** | **0.068** |
| THuman2.0 | 80K | PFGS | 34.74 | 0.983 | 0.009 |
| THuman2.0 | 80K | **Ours** | **35.43** | **0.987** | **0.009** |
| ScanNet | 100K | PFGS | 19.86 | 0.758 | 0.452 |
| ScanNet | 100K | **Ours** | **20.24** | **0.759** | **0.490** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| No Initialization | 9.62 | 0.685 | 0.655 | No normal initialization → training collapse |
| No $\mathcal{N}_e$ | 12.94 | 0.676 | 0.441 | No background completion → patch cannot be effectively trained |
| $\mathcal{N}_p$ only | 23.23 | 0.829 | 0.126 | Patch only processing the whole → high resource consumption |
| **Full model** | **27.88** | **0.949** | **0.068** | Complete model |

| Rendering Speed | Method | FPS |
|----------|------|-----|
| | NPBG++ | 37.45 |
| | TriVol | 1.62 |
| | PFGS | 3.80 |
| | **Ours** | **142.86** |

### Key Findings

- **Normal initialization is a prerequisite for the method**: Removing it causes the PSNR to plunge from 27.88 to 9.62, and training diverges immediately. Randomly initialized Gaussians are invisible in many views, blocking gradient propagation.
- **Outperforming the 100K baseline using only 20% of the points**: On the Car category, the proposed method with 20K points (27.88 PSNR) already outperforms TriVol-100K (27.22) and PFGS-100K (27.34), demonstrating extremely robust sparse point cloud processing capabilities.
- **Outstanding generalization ability**: The model trained on Car-20K directly evaluated on the DTU scene dataset achieves a PSNR of 16.71 with 20K points, compared to only 8.07 for PFGS under the same settings.
- **Overwhelming rendering speed**: 142.86 FPS vs. PFGS's 3.80 FPS (a 37x speedup), thanks to direct rasterization using 2DGS without post-processing refinement.
- Increasing the splitting factor K continuously improves sparse point cloud performance, with K=4 striking the optimal balance between performance and computational overhead.

## Highlights & Insights

- **Clever combination of 2D Gaussians + normal initialization**: The advantage of 2DGS over 3DGS is its explicit normals, which can be easily estimated from point clouds. This combination transforms initialization from "random quaternions" into "geometrically aligned normal directions," which is key to the success of the method. This philosophy of "finding a natural match between representation and data characteristics" is highly instructive.
- **Splitting Decoder as an implicit upsampler**: Unlike traditional point cloud upsampling methods, the splitting decoder operates in the Gaussian parameter space, performing upsampling and attribute prediction in a single step, which is highly efficient. This design can be generalized to other scenarios requiring dense representations generated from sparse ones.
- **Entire-Patch architecture solves the supervision issue in patch training**: Using the full prediction as a "background" to enable patch training is a highly practical engineering trick.

## Limitations & Future Work

- When a part of the point cloud is completely missing (rather than just sparse), the method cannot "hallucinate" the missing areas due to the lack of prior support.
- The improvement on the ScanNet indoor scenes dataset is less pronounced compared to object categories (20.24 vs. 19.86 PSNR), possibly because global contexts are more crucial in indoor scenes.
- The entire-patch architecture requires two-stage training (first $\mathcal{N}_e$, then $\mathcal{N}_p$), which increases training complexity.
- Future work could integrate point cloud completion techniques to handle large missing areas.
- Exploring robust handling of noisy, real-world point clouds obtained from physical scanners is a promising direction.

## Related Work & Insights

- **vs PFGS**: PFGS also predicts Gaussians + renders them, but requires a two-stage process (prediction + refinement), yielding a speed of only 3.8 FPS and poor generalization. The proposed method directly predicts Gaussians for rendering, running 37 times faster and supporting cross-category generalization.
- **vs TriVol**: TriVol uses tri-plane representations to predict NeRFs, resulting in extremely slow speeds (1.62 FPS) and only generalizing within a single category. The proposed method excels past its 100K results using only 20K sparse points.
- **vs 2DGS/3DGS**: Original 2DGS/3DGS requires thousands of optimization iterations individually per scene. The proposed method performs feed-forward prediction, allowing it to generalize to new point clouds once trained, without requiring separate optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of normal initialization, splitting decoder, and entire-patch architecture is highly ingenious, though individually the components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 5 datasets (scenes, objects, human bodies), multiple point cloud densities, generalization experiments, speed comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear and readable, intuitive figures and charts, with comprehensive comparative dimensions in Table 1.
- Value: ⭐⭐⭐⭐⭐ Combining real-time rendering, sparse point clouds, and cross-category generalization makes it extremely practical, and the code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] Toward Robust Neural Reconstruction from Sparse Point Sets](toward_robust_neural_reconstruction_from_sparse_point_sets.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

</div>

<!-- RELATED:END -->
