---
title: >-
  [Paper Note] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes Horizon-GS, which achieves the first unified 3D Gaussian Splatting reconstruction and real-time rendering of both aerial and street-view perspectives through a coarse-to-fine two-stage training strategy, a camera distribution balance mechanism, and a multi-resolution LOD structure, achieving SOTA rendering quality on multiple urban scene datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Unified Aerial-to-Ground Reconstruction"
  - "Coarse-to-Fine Training"
  - "Multi-Scale LOD"
  - "Large-Scale Scene Rendering"
date: 2026-05-08
content_hash: d3c1b327bce80a7a
---

# Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.01745](https://arxiv.org/abs/2412.01745)  
**Code**: [https://city-super.github.io/horizon-gs/](https://city-super.github.io/horizon-gs/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Unified Aerial-to-Ground Reconstruction, Coarse-to-Fine Training, Multi-Scale LOD, Large-Scale Scene Rendering

## TL;DR

This paper proposes Horizon-GS, which achieves the first unified 3D Gaussian Splatting reconstruction and real-time rendering of both aerial and street-view perspectives through a coarse-to-fine two-stage training strategy, a camera distribution balance mechanism, and a multi-resolution LOD structure, achieving SOTA rendering quality on multiple urban scene datasets.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3D-GS) has become the mainstream method for neural scene reconstruction due to its outstanding visual quality and real-time rendering speed. In large-scale urban scenes, excellent results have been achieved separately for aerial perspectives (VastGaussian, CityGaussian, etc.) and street-view perspectives (Hierarchical-3DGS, etc.).

2. **Limitations of Prior Work**: Existing methods can only handle a single perspective type—either aerial or street-view—failing to unify both in joint training and rendering. Directly training 3DGS jointly on both perspectives yields poorer results than training them individually.

3. **Key Challenge**: Joint training of aerial and street-view perspectives suffers from two main conflicts: (1) **Gradient accumulation conflict**—the street-view perspective causes the densification strategy to remove occluded Gaussians, while the aerial perspective demands regrowth in the same regions, causing mutual interference and destabilizing the densification process; (2) **Imbalanced camera distribution**—there are many street-view cameras capturing nearby details, whereas there are fewer aerial cameras focusing on large-scale far-range areas, causing the training to bias toward street-view details while neglecting aerial structures.

4. **Goal**: To achieve high-quality reconstruction and rendering of both aerial and street-view perspectives in a unified Gaussian model.

5. **Key Insight**: The authors observe that the aerial perspective provides the global geometric framework, while the street-view perspective provides local details. Therefore, a staged strategy is adopted: first establishing a rough geometric skeleton using the aerial perspective, and then filling in details using the street-view perspective.

6. **Core Idea**: Coarse-to-fine two-stage training (aerial skeleton construction, street-view detail filling) + balanced camera sampling + multi-level LOD = unified aerial-to-ground reconstruction.

## Method

### Overall Architecture

Horizon-GS is based on the anchor + neural Gaussian design of Scaffold-GS. For large-scale scenes, the scene is first divided into chunks, each trained independently and then merged. The training of each chunk is split into two stages: the first stage primarily uses the aerial perspective to establish rough geometry and initialize the multi-level LOD anchor structure; the second stage freezes the parameters of the first stage and primarily uses the street-view perspective to add fine details through an enhanced densification strategy. The framework simultaneously supports 3D Gaussians (for high-quality rendering) and 2D Gaussians (for accurate geometric reconstruction).

### Key Designs

1. **Coarse-to-Fine Two-Stage Training Strategy**:

    - **Function**: Resolving the aerial-to-ground gradient conflict.
    - **Mechanism**: The first stage (60k iterations) is dominated by the aerial perspective (sampling probability $R/(R+1)$, where $R=2$ corresponds to 67% aerial), accumulating gradients only from aerial images for densification to let Gaussians fully develop and cover the global feature space. The street-view perspective guides the placement of initial fine Gaussians but does not participate in densification decisions. The second stage (40k iterations) freezes the MLP weights and Gaussian attributes from the first stage to maintain the global skeleton, reduces the aerial sampling ratio ($R=1$, 50%), and refines details using the street-view perspective. Crucially, the densification strategy in the second stage is enhanced—instead of using the average voxel gradient from Scaffold-GS, it considers the maximum gradient $\nabla_g$, average opacity $\sigma$, and maximum projection radius $r$ of a single neural Gaussian. Gaussians satisfying $\nabla_g \cdot r \cdot \sigma^{\tau_\sigma} > \tau_g$ are promoted to new anchors.
    - **Design Motivation**: Direct joint training causes the gradients of both perspectives to cancel each other out, leading to failed Gaussian densification. The conflicts are avoided by staging the training—allowing the aerial perspective to establish the global structure without interference first, and then allowing the street-view perspective to add details without interference. Freezing the first-stage parameters in the second stage ensures the global skeleton remains intact.

2. **Multi-Resolution LOD Construction**:

    - **Function**: Adapting to the vast detail differences between aerial and street-view perspectives, supporting real-time rendering.
    - **Mechanism**: Drawing inspiration from the LOD strategy of Octree-GS, the required LOD levels are automatically calculated based on the distances of the aerial and street-view cameras. The number of aerial levels is $K_{aerial} = \lfloor \log_2(D_{aerial}/d_{aerial}) \rfloor + 1$, and the total number of levels is $K = K_{aerial} + \lfloor \log_2(d_{aerial}/d_{street}) \rfloor$. The first stage only activates the $K_{aerial}$ levels, and new anchors are added to the corresponding LOD level. The second stage opens all levels, and new anchors are added to the next LOD level to capture higher-frequency street-view information.
    - **Design Motivation**: The aerial perspective corresponds to a coarse scale, while the street-view perspective corresponds to a fine scale. Using a single static resolution of Gaussians cannot satisfy the requirements of both simultaneously. The LOD structure allows details of different scales to be represented on different levels, supporting real-time rendering (51.5 FPS for large-scale scenes).

3. **Large-Scale Scene Chunk Training and Merging Strategy**:

    - **Function**: Scaling large-scale scenes to a size manageable by GPUs.
    - **Mechanism**: The scene is divided into $m \times n$ chunks along the ground projection, extending boundaries for each chunk to ensure sufficient overlap. For the aerial perspective, visible cameras and auxiliary point clouds are increased; for the street-view perspective (which suffers from severe occlusion), point clouds are generated from depth maps to ensure coverage during training. After independent training of each chunk, Gaussians outside the boundaries are discarded, and the remaining ones are concatenated. To accelerate rendering, the hybrid representation is converted to a fully explicit representation (by discarding the view-dependent MLP and replacing the color MLP with spherical harmonics prediction).
    - **Design Motivation**: VastGaussian's projection-based partitioning method suffers from projection errors when handling street-view perspectives. This issue is resolved by customizing data augmentation strategies for aerial and street-view perspectives respectively.

### Loss & Training

The rendering loss is defined as $\mathcal{L}_R = \mathcal{L}_1 + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{vol}\mathcal{L}_{vol} + \lambda_d\mathcal{L}_d + \lambda_o\mathcal{L}_o$, where the depth supervision weight decays exponentially from 1 to 0.01. For geometric reconstruction, an additional normal consistency loss is added: $\mathcal{L}_S = \mathcal{L}_R + \lambda_n\mathcal{L}_n$. Mask regularization $\mathcal{L}_o$ is used to eliminate the impact of pedestrians, vehicles, and sky. Training is conducted on an A100 80G GPU, taking approximately 4 hours in parallel of chunks for large-scale scenes.

## Key Experimental Results

### Main Results

**Small-Scale Scene Rendering Quality (Table 1, Block_Small Scene)**:

| Method | Aerial PSNR↑ | Aerial LPIPS↓ | Street PSNR↑ | Street LPIPS↓ |
|------|-------------|--------------|-------------|--------------|
| 3D-GS | 25.44 | 0.325 | 21.81 | 0.371 |
| Scaffold-GS | 28.44 | 0.191 | 23.84 | 0.271 |
| Hier-GS | 28.31 | 0.189 | 23.75 | 0.220 |
| **Ours** | **30.59** | **0.094** | **23.80** | **0.209** |

**Large-Scale Scene Rendering Quality (Table 2, Block_A Scene)**:

| Method | Aerial PSNR↑ | Aerial LPIPS↓ | Street PSNR↑ | Street LPIPS↓|
|------|-------------|--------------|-------------|--------------|
| 2D-GS | 20.63 | 0.595 | 19.57 | 0.477 |
| Scaffold-GS* | 27.62 | 0.206 | 23.10 | 0.277 |
| **Ours*** | **28.89** | **0.151** | **23.66** | **0.255** |

### Ablation Study

**Perspective Extrapolation Comparison on UC-GS Dataset (Table 3)**:

| Method | Held-out PSNR↑ | +1m PSNR↑ | +1m 5°down PSNR↑ |
|------|---------------|-----------|-------------------|
| 3D-GS | 23.47 | 20.83 | 21.25 |
| UC-GS | 25.95 | 23.52 | 24.15 |
| **Ours** | 25.35 | **25.46** | **25.37** |

### Key Findings

- Horizon-GS achieves highly significant PSNR gains on the aerial perspective (more than 2 dB higher than Scaffold-GS), demonstrating that the two-stage strategy effectively resolves the aerial-ground conflict.
- On the street-view perspective, the rendering quality is basically on par with Hierarchical-3DGS (specially designed for street view), proving that the unified model does not sacrifice single-perspective quality.
- In the perspective extrapolation experiment (Table 3 on UC-GS), when the testing perspective shifts from the training distribution (+1m shift), the performance drop of Horizon-GS is far smaller than other methods, demonstrating stronger generalization ability.
- Chunk-based training (marked with *) significantly helps large-scale scenes: Scaffold-GS* vs Scaffold-GS shows an aerial PSNR improvement of about 5 dB on Block_A.
- The rendering speed of large-scale scenes reaches 51.5 FPS, supporting real-time applications.

## Highlights & Insights

- **The "skeleton-first, details-later" two-stage training philosophy**: Using the aerial perspective to build global geometry and the street-view perspective to supplement local details, this coarse-to-fine staged strategy is applicable to all multi-scale fusion scenes. Similar ideas can be transferred to satellite-UAV-ground multi-level reconstruction.
- **The "protection mechanism" of freezing first-stage parameters**: Preventing the refinement stage from destroying the established global structure is a simple and effective strategy to fight forgetting.
- **Cross-perspective dataset contribution**: Constructing 5 synthetic + 2 real-world aerial-street-view aligned datasets fills a gap in this field, offering major value for future research.

## Limitations & Future Work

- The PSNR on real-world scenes (Real) is significantly lower than on synthetic scenes (Synthetic), indicating room for improvement in handling real-world data.
- Pre-processing of depth maps (Depth-Anything-V2) and semantic segmentation masks (Grounded-SAM) is required, making the pipeline heavily dependent on external models.
- The PSNR improvement on the street-view perspective is not as significant as on the aerial perspective, potentially requiring more specialized street-view optimization strategies.
- Inconsistencies might exist at the partition boundaries of chunk-based training; although the overlapping design mitigates this problem, it is not fully resolved.
- Dynamic object modeling is not considered (pedestrians and vehicles are simply filtered out with masks), limiting its application in dynamic scenes like autonomous driving.

## Related Work & Insights

- **vs VastGaussian/CityGaussian**: These methods only handle large-scale scene reconstruction from aerial perspectives. Horizon-GS not only unifies both perspectives but also outperforms them in aerial rendering quality.
- **vs Hierarchical-3DGS**: The LOD hierarchy of Hier-GS, specifically designed for street views, shows advantages in single-perspective scenarios but cannot naturally support aerial perspectives. Horizon-GS's two-stage LOD construction accommodates both perspectives.
- **vs UC-GS**: UC-GS introduces cross-view uncertainty but lacks scalability for large-scale scenes. Horizon-GS solves the scaling issue through the chunk-based strategy and is more stable in perspective extrapolation.
- **vs BungeeNeRF**: BungeeNeRF's multi-scale NeRF inspired the coarse-to-fine strategy of this paper, but Horizon-GS achieves real-time rendering based on Gaussian Splatting.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem definition of unified aerial-street-view reconstruction is highly valuable, and the two-stage strategy design is reasonable, although some components are inherited from existing works
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 scenes covering synthetic/real and small/large scales, with comprehensive comparisons against various baselines; the perspective extrapolation experiments are highly convincing
- Writing Quality: ⭐⭐⭐⭐⭐ Deep analysis of problems (the conflict analysis in Figure 3), fluent overall narrative, and high-quality figures and tables
- Value: ⭐⭐⭐⭐ Directly valuable to applications requiring cross-scale scenes, such as digital twins, VR/AR, and autonomous driving, with a prominent dataset contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2026\] Urban-GS: A Unified 3D Gaussian Splatting Framework for Compact and High-Fidelity Aerial-to-Street Reconstruction](../../CVPR2026/3d_vision/urban-gs_a_unified_3d_gaussian_splatting_framework_for_compact_and_high-fidelity.md)

</div>

<!-- RELATED:END -->
