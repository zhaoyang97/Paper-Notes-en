---
title: >-
  [Paper Note] SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer
description: >-
  [ECCV 2024][3D Vision][Video-to-4D generation] SC4D proposes a sparse-controlled video-to-4D generation framework. By decoupling the motion and appearance of dynamic 3D objects into sparse control points (~512) and dense Gaussian volumes (~50k), combined with Adaptive Gaussian Initialization (AG) and Gaussian Alignment Loss (GA) to address shape degradation, it achieves high-quality generation and cross-entity motion transfer based on control point trajectories.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Video-to-4D generation"
  - "motion decoupling"
  - "sparse control points"
  - "3D Gaussian Splatting"
  - "motion transfer"
date: 2026-05-08
content_hash: f532799810e5ff6a
---

# SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer

**Conference**: ECCV 2024  
**arXiv**: [2404.03736](https://arxiv.org/abs/2404.03736)  
**Code**: [https://github.com/JarrentWu1031/SC4D](https://github.com/JarrentWu1031/SC4D)  
**Area**: 3D Vision  
**Keywords**: Video-to-4D generation, motion decoupling, sparse control points, 3D Gaussian Splatting, motion transfer

## TL;DR

SC4D proposes a sparse-controlled video-to-4D generation framework. By decoupling the motion and appearance of dynamic 3D objects into sparse control points (~512) and dense Gaussian volumes (~50k), combined with Adaptive Gaussian Initialization (AG) and Gaussian Alignment Loss (GA) to address shape degradation, it achieves high-quality generation and cross-entity motion transfer based on control point trajectories.

## Background & Motivation

**Background**: 4D content generation (dynamic 3D object generation) is an emerging research direction with broad application prospects in fields such as AR/VR, film, and animation. Recent works like Consistent4D and 4DGen utilize Score Distillation Sampling (SDS) to distill prior knowledge from 2D diffusion models for video-to-4D generation. These methods typically represent dynamic scenes as dynamic NeRFs or dense 3D Gaussian volumes.

**Limitations of Prior Work**: (1) Due to their implicit nature, dynamic NeRF methods struggle to maintain spatiotemporal consistency under the randomness of SDS, often resulting in flickering and acfifacts. (2) Dense 3D Gaussian methods require predicting independent motion trajectories and rotations for tens of thousands of Gaussians, which is extremely difficult under single-view conditions and prone to motion distortion. (3) Both types of methods struggle to balance reference-view alignment, spatiotemporal consistency, and motion fidelity. (4) The coupled modeling of motion and appearance prevents flexible transfer of the learned motion to other entities.

**Key Challenge**: Video-to-4D generation requires simultaneously reconstructing both appearance and motion characteristics. However, single-view conditions are severely under-constrained, leading to an excessively large search space for dense motion prediction, while implicit representation lacks explicit control over motion.

**Goal**: (1) How to reduce the difficulty of motion prediction and improve motion fidelity? (2) How to prevent shape and motion degradation during the texture refinement stage? (3) How to enable the transfer of motion extracted from video to other entities?

**Key Insight**: Inspired by the SC-GS dynamic scene reconstruction method, motion is modeled as transformations of a small number of sparse control points, which then drive the dense Gaussians via Linear Blend Skinning (LBS). This significantly reduces the degrees of freedom in motion prediction, and the trajectories of control points can be extracted and reused.

**Core Idea**: Replace 50k dense Gaussians with 512 sparse control points to model motion, decoupling motion and appearance to achieve high-quality 4D generation and motion transfer.

## Method

### Overall Architecture

SC4D adopts a two-stage framework. **Coarse Stage**: Initializes 512 spherical control Gaussians, predicts their motion via a time-conditional MLP, and optimizes control Gaussian locations, appearance, and motion MLP parameters under the joint guidance of a reference-view reconstruction loss and an SDS novel-view loss. **Fine Stage**: Control Gaussians transition to implicit control points. Dense Gaussians are randomly generated around each control Gaussian using Adaptive Gaussian Initialization (AG) and driven by LBS. Control point locations, the motion MLP, and dense Gaussian parameters are jointly optimized to refine the texture.

### Key Designs

1. **Sparse Control Point Initialization in the Coarse Stage**:

    - **Function**: Provide a coarse but reasonable initial estimate of the shape and motion of the dynamic object.
    - **Mechanism**: Initialize $M=512$ spherical Gaussians with identical scaling parameters $s$. An MLP $\Psi$ is used to predict the translation $T_i^t$ and rotation $R_i^t$ of each control point given the time $t$ and positions. The training objective includes a reference-view reconstruction loss $L_\text{ref} = \|\hat{I}^t - I_r^f\|_2^2$, a foreground mask loss $L_\text{mask}$, and an SDS novel-view loss. Densification and pruning are performed during the first 1000 iterations, followed by Farthest Point Sampling (FPS) to re-select $M$ control Gaussians for another 500 training iterations.
    - **Design Motivation**: Spherical constraints ensure uniform distribution of control Gaussians, preventing degeneration into disordered distributions. Densification followed by FPS balances coverage and uniformity.

2. **Adaptive Gaussian Initialization (AG)**:

    - **Function**: Generate dense Gaussian initialization for the fine stage from the coarse-stage control Gaussians.
    - **Mechanism**: For the $M$ control Gaussians learned in the coarse stage, each is treated as a sphere with radius $s$. Within each sphere, $K$ dense Gaussians are randomly initialized, resulting in a total of $N = M \times K$ dense Gaussians. The dense Gaussians initialized this way are naturally distributed near the object surface and inherit the shape and motion learned in the coarse stage.
    - **Design Motivation**: Uniformly initializing dense Gaussians directly within the sphere without constraints would cause mismatch in shape and motion compared to the coarse stage, leading to thickening, spatial drift, and blurry textures. AG initialization guarantees a smooth transition from coarse to fine.

3. **Gaussian Alignment Loss (GA)**:

    - **Function**: Prevent shape and motion degradation caused by SDS optimization in the fine stage.
    - **Mechanism**: At the start of the fine stage, the control point parameters and deformation MLP from the coarse stage are saved as a reference. The L2 distance between current control point positions and reference positions is calculated as regularization: $L_\text{GA} = \|p^t - \bar{p}^t\|_2^2$. This constrains control points from deviating too far from the trajectories learned in the coarse stage during optimization.
    - **Design Motivation**: The SDS loss biases toward global shape optimization at large noise timesteps and texture details at small noise timesteps. Without shape constraints in the texture refinement stage, SDS gradually destroys the learned shape and motion. GA loss outperforms the Chamfer loss, which might cause control points to cluster around specific target points and destroy uniform distribution.

### Loss & Training

Coarse Stage: $L = \lambda_\text{ref} L_\text{ref} + \lambda_\text{mask} L_\text{mask} + \lambda_\text{SDS} L_\text{SDS}$, with weights set to 5000, 500, and 1, respectively. SDS uses Zero123 as the prior model. Fine Stage adds the GA loss: $L = \lambda_\text{ref} L_\text{ref} + \lambda_\text{mask} L_\text{mask} + \lambda_\text{SDS} L_\text{SDS} + \lambda_\text{GA} L_\text{GA}$, where the GA weight is 10000. Noise timesteps are linearly decayed from 800 to 200 to progressively transition from shape optimization to texture refinement. The overall training takes approximately 1 hour on a single V100 GPU.

## Key Experimental Results

### Main Results

Quantitative comparison on the Consistent4D dataset (average over 10 videos):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Temp↓ | Time |
|------|-------|-------|--------|-------|-------|------|
| Consistent4D | 23.97 | 0.91 | 0.09 | 0.89 | 0.0089 | 1.9h |
| 4DGen | 21.80 | 0.90 | 0.10 | 0.87 | 0.0089 | 3.0h |
| **Ours** | **29.50** | **0.95** | **0.08** | **0.90** | **0.0081** | **1.0h** |

Ours (SC4D) leads significantly across all metrics, showing an improvement of 5.5dB in PSNR while maintaining the fastest training speed.

### Ablation Study

Ablation of AG initialization and GA loss:

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Description |
|------|-------|-------|--------|-------|------|
| Baseline (w/o AG, w/o GA) | 29.81 | 0.95 | 0.10 | 0.82 | Severe shape degradation |
| +GA loss | 30.19 | 0.96 | 0.09 | 0.83 | Control point shape retention |
| +AG initialization (Full) | 31.35 | 0.96 | 0.08 | 0.89 | Substantial CLIP improvement |

Control point quantity ablation:

| Control Points M | PSNR↑ | SSIM↑ | CLIP↑ | Temp↓ |
|-----------|-------|-------|-------|-------|
| 128 | 26.46 | 0.934 | 0.890 | 0.0175 |
| 256 | 26.75 | 0.936 | 0.890 | 0.0176 |
| 512 | 27.00 | 0.936 | 0.893 | 0.0178 |
| 1024 | 27.30 | 0.938 | 0.893 | 0.0180 |

### Key Findings

- Motion-appearance decoupling is central to the success of SC4D, simplifying motion learning while improving appearance optimization quality.
- AG initialization has the greatest impact on novel-view quality (CLIP improved from 0.82 to 0.89), demonstrating that a good initialization is crucial for mitigating shape degradation.
- Although the GA loss is simple (only L2 distance), it outperforms the Chamfer loss, which tends to disrupt the uniform distribution of control points.
- Increasing the number of control points enhances rendering quality but degrades temporal consistency; $M=512$ provides the optimal trade-off between performance and rigidity.
- User studies indicate an overwhelming preference for SC4D in both reference-view and novel-view evaluations.

## Highlights & Insights

- **Elegance of Sparse Control**: Driving 50k Gaussians with just 512 points naturally incorporates local rigidity constraints.
- **Motion Transfer Application**: Decoupled modeling enables extracting motion from a 4D object and transferring it to a new text-described entity.
- **Coarse-to-Fine Strategy**: The two-stage protocol of learning motion skeleton first and refining texturing later is robust and effective.
- **Intuitive AG Initialization**: Randomly initializing dense Gaussians inside the control Gaussian spheres inherits shape and motion seamlessly.

## Limitations & Future Work

- Reliance on Zero123 for novel-view priors limits performance on complex objects.
- 4D generation in moving-camera scenarios is not yet considered.
- Reference video quality directly affects the generation results.
- Motion estimation in occluded regions remains challenging under single-view conditions.
- Integrating more powerful multi-view diffusion models (e.g., SV3D) instead of Zero123 is a promising direction.

## Related Work & Insights

- **SC-GS**: Sparse-Controlled Gaussian Splatting, used for dynamic scene reconstruction, which is a direct inspiration for SC4D.
- **Consistent4D**: A dynamic NeRF-based video-to-4D method.
- **4DGen**: A dense Gaussian-based 4D generation method.
- **DreamGaussian**: An initialization strategy for 3D Gaussian Splatting.
- **Insight**: Porting representation paradigms from scene reconstruction (like SC-GS) to generative tasks is an excellent strategy to bridge the reconstruction and generation communities.

## Rating

- Novelty: ⭐⭐⭐⭐ Sparse-controlled 4D generation and motion transfer applications are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Diverse evaluations with quantitative, qualitative, user studies, and ablation analyses.
- Writing Quality: ⭐⭐⭐ Method description is adequate but somewhat verbose.
- Value: ⭐⭐⭐⭐ Significantly outperforms prior methods with innovative applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2025/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
