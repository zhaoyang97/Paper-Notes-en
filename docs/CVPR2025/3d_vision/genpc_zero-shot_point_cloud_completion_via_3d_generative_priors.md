---
title: >-
  [Paper Note] GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
description: >-
  [CVPR 2025][3D Vision][Point Cloud Completion] This paper proposes GenPC, a zero-shot point cloud completion framework. It uses a Depth Prompting module to convert partial point clouds into depth maps to generate RGB images as input for Image-to-3D models. Then, a Geometric Preserving Fusion module aligns and fuses the generated 3D shape with the original point cloud, achieving faster and better real-world scan completion compared to SDS-based optimization methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Zero-shot"
  - "3D Generative Priors"
  - "Depth Prompting"
  - "Geometric Preserving Fusion"
date: 2026-05-08
content_hash: 6b08170c93cce39f
---

# GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors

**Conference**: CVPR 2025  
**arXiv**: [2502.19896](https://arxiv.org/abs/2502.19896)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Zero-shot, 3D Generative Priors, Depth Prompting, Geometric Preserving Fusion

## TL;DR

This paper proposes GenPC, a zero-shot point cloud completion framework. It uses a Depth Prompting module to convert partial point clouds into depth maps to generate RGB images as input for Image-to-3D models. Then, a Geometric Preserving Fusion module aligns and fuses the generated 3D shape with the original point cloud, achieving faster and better real-world scan completion compared to SDS-based optimization methods.

## Background & Motivation

**Background**: Learning-based point cloud completion methods (such as PoinTr and SnowflakeNet) perform well on synthetic datasets, but heavily rely on the distribution of training data, showing limited generalization to real-world scans. SDS-Complete made the first attempt to use 2D diffusion priors for zero-shot completion, but it requires optimizing SDF/Gaussian representations from scratch, which is time-consuming and yields coarse geometric details.

**Limitations of Prior Work**: (1) Learning-based methods suffer from severe performance degradation on out-of-distribution real scans, and even trained categories perform poorly due to domain gaps; (2) SDS optimization takes minutes to hours, and SDS loss often yields coarse geometric details; (3) Existing methods are sensitive to scale variations, leading to unstable outputs when the input scale changes; (4) 2D diffusion priors are implicit and cannot provide accurate 3D geometric information.

**Key Challenge**: High-quality zero-shot completion requires strong 3D priors, but 2D diffusion models only provide implicit 2D priors, while feed-forward 3D generative models require image inputs instead of point clouds.

**Goal**: How to leverage explicit 3D priors from feed-forward 3D generative models (e.g., LGM, InstantMesh) for zero-shot point cloud completion. Key sub-problems include: (1) How to generate image inputs suitable for Image-to-3D models from point clouds, and (2) How to align the generated 3D shape with the original point cloud while preserving the original geometry.

**Key Insight**: Feed-forward 3D generative models are already capable of generating high-quality 3D objects from a single image in seconds with strong generalization. The authors observe that depth maps can serve as a "springboard" to convert point clouds into images, bridging the modality gap between point clouds and Image-to-3D models.

**Core Idea**: Use depth maps as a springboard to convert partial point clouds into RGB inputs acceptable by Image-to-3D models, leveraging explicit geometric priors from feed-forward 3D generative models for fast, high-quality zero-shot completion.

## Method

### Overall Architecture

The workflow of GenPC consists of two main modules: (1) The **Depth Prompting** module converts partial point clouds into depth maps, and then generates RGB images via ControlNet to be fed into the Image-to-3D model to produce complete 3D shapes; (2) The **Geometric Preserving Fusion** module aligns the generated shape with the original point cloud through dynamic scale adaptation, with optional further refinement via SDS optimization. The input consists of a partial point cloud $P_{in}$ and a text prompt $T_{in}$, and the output is the completed point cloud $P_{out}$.

### Key Designs

1. **Depth Prompting Module**:

    - **Function**: Generates RGB image inputs suitable for Image-to-3D models from partial point clouds.
    - **Mechanism**: Completed in three steps — (a) **Viewpoint Selection**: Place $M$ cameras uniformly around the point cloud. Spherical flipping is performed for each camera to obtain the mirrored point cloud $\hat{P_{in}}$. The convex hull of $\hat{P_{in}} \cup V_i$ is constructed, and the camera containing the most visible points on the convex hull is identified as the scanning viewpoint $V_{scan}$. (b) **Depth Completion**: Projection from $V_{scan}$ yields a sparse depth map $D_{raw}$. A pre-trained 2D diffusion model is used to fill the missing regions to obtain a complete depth map $D_c$ (first using large-pixel projection to obtain a full-coverage mask $M_{FULL}$, then performing XOR on $M_{FULL}$ and $\neg D_{raw}$ to obtain the completion mask). (c) **Image Generation**: Input $D_c$ and the text prompt $T_{in}$ into ControlNet to generate the corresponding RGB image $I_{gen}$.
    - **Design Motivation**: Distance-based viewpoint estimation methods can select reversed viewpoints leading to depth inversion; the convex hull method avoids this issue through geometric reasoning. Depth completion resolves the issue where sparse point clouds (such as KITTI LiDAR scans) yield overly sparse depth maps after projection.

2. **Dynamic Scale Adaptation**:

    - **Function**: Aligns the generated 3D shape with the original partial point cloud in pose and scale.
    - **Mechanism**: First, $P_{in}$ is colored using $I_{gen}$ to obtain a colored partial point cloud $P_{partial}$, and both are normalized to $[-0.5, 0.5]$. $P_{gen}$ is scaled with an interval of 0.1 within the range of $[0.8, 1.2]$. For each scale, ICP registration is executed to comprehensively evaluate the geometric Chamfer Distance and RGB Chamfer Distance: $\arg\min_{s} (\alpha \cdot CD_{XYZ} + \beta \cdot CD_{RGB})$. The optimal registration is selected, and points in $P_{gen}$ close to $P_{partial}$ are removed to avoid overlap, yielding the initial completed point cloud $P_{all}$.
    - **Design Motivation**: The 3D shapes output by generative models are usually inconsistent in scale and pose with the input point clouds. Directly aligning them wastes rich geometric priors. Leveraging color as semantic info (different parts have different colors) assists alignment, improving fusion accuracy.

3. **SDS Refining (Optional)**:

    - **Function**: Further mitigates accumulated errors from the multi-stage pipeline and optimizes the geometry of missing regions.
    - **Mechanism**: $P_{all}$ is initialized as 3D Gaussians. For the **partial region** $G_{partial}$, coordinates, colors, scales, and opacities are all frozen to preserve the original geometry. For the **missing region** $G_{miss}$, scales and opacities are fixed, colors are fine-tuned with a low learning rate, and coordinates are the primary optimization targets. The reference image $I_{optim}$ is rendered from $V_{scan}$, and $\tilde{I}^i_{optim}$ is rendered from random viewpoints. Zero123's SDS loss is used to optimize $G_{miss}$. Concurrently, a preservation loss $L_{Presv} = w_1 \cdot MSE(I_{optim}, I^i_{optim}) + w_2 \cdot MSE(D_{optim}, D^i_{optim})$ is employed to prevent the optimization process from affecting the partial region.
    - **Design Motivation**: Errors from prior steps propagate (viewpoint estimation $\rightarrow$ depth completion $\rightarrow$ image generation $\rightarrow$ 3D generation $\rightarrow$ alignment). SDS refining corrects these accumulated errors. Region-specific Gaussian parameter settings are key — freezing the original parts guarantees that the input geometry is unchanged, only optimizing the missing parts.

### Loss & Training

GenPC is a zero-shot framework that requires no training on specific datasets. The core losses include: the comprehensive CD (geometry + RGB) for ICP alignment, the SDS loss (optional), and the preservation loss. The SDS refining step is optional, and competitive results can be achieved even without SDS.

## Key Experimental Results

### Main Results (Redwood Real Scan Dataset)

| Method | Mean CD↓ | Mean EMD↓ | Type |
|------|---------|---------|------|
| **GenPC** | **1.74** | **2.88** | Zero-shot |
| GenPC (w/o Refining) | 1.98 | 3.16 | Zero-shot |
| SDS-Complete | 2.72 | 4.06 | Zero-shot |
| PoinTr | 2.89 | 5.24 | Learning-based |
| SnowflakeNet | 2.96 | 5.64 | Learning-based |
| AdaPoinTr | 4.45 | 6.19 | Learning-based |

### Ablation Study

| Configuration | CD↓ | EMD↓ | Description |
|------|-----|------|------|
| Full model | 1.74 | 2.88 | Full GenPC |
| A: w/o Viewpoint Selection (Distance-based) | 2.44 | 3.79 | May select incorrect viewpoints, leading to depth inversion |
| B: w/o ControlNet | 4.31 | 6.80 | Lack of color information severely affects subsequent operations |
| C: w/o Depth Completion | 2.23 | 3.60 | Acceptable for dense point clouds, but severely degrades on sparse ones |
| D: w/o 3D Generative Model | 4.65 | 6.13 | Replaced with Gaussian noise, proving the crucial role of the prior |
| E: w/o Dynamic Scale Adaptation | 4.38 | 4.52 | Direct alignment wastes geometric priors |
| F: w/o SDS Optimization | 1.98 | 3.16 | Still competitive; SDS mainly refines details |

### Key Findings

- **GenPC significantly outperforms all other methods**: On the Redwood dataset, GenPC reduces CD by 36% and EMD by 29% compared to SDS-Complete, demonstrating that explicit 3D priors are vastly superior to implicit 2D priors.
- **Learning-based methods fail dramatically**: Even for trained categories (e.g., chairs, sofas), PoinTr/SnowflakeNet/AdaPoinTr perform worse on real scans than GenPC, illustrating a massive synthetic-to-real domain gap.
- **The 3D generative model and dynamic scale adaptation are the two most critical modules**: Removing them increases the CD to 4.65 and 4.38, respectively, which is a much larger impact than other modules.
- **Depth completion is of paramount importance for sparse point clouds**: The impact is moderate on dense Redwood scans (2.23 vs 1.74) but massive on sparse ScanNet scans (3.57 vs 1.62).
- **SDS refining is optional**: Even without SDS, a CD of 1.98 is highly competitive; SDS mainly serves to enhance fine-grained details.

## Highlights & Insights

- **Clever design of using depth maps as a modality bridge**: The conversion pipeline (point cloud $\rightarrow$ depth map $\rightarrow$ RGB image $\rightarrow$ 3D shape) is neat and effective, bridging the powerful capabilities of feed-forward 3D generative models to the point cloud completion task.
- **Convex hull viewpoint selection**: Formulates the scanning viewpoint estimation problem as a hidden point removal problem. Utilizing spherical flipping and convex hull construction avoids depth inversion, making it much more robust than simple distance-based methods.
- **Color as a semantic alignment cue**: Incorporating RGB Chamfer Distance as an extra supervision signal in ICP registration leverages semantic constraints provided by color variations in different parts. This technique is transferable to other point cloud registration scenarios.
- **Region-specific Gaussian parameter configuration**: Freezing Gaussian parameters for the original parts while optimizing only the missing parts serves as a simple and effective strategy to preserve the input geometry.

## Limitations & Future Work

- Dependence on text prompts to generate ControlNet images requires users to provide text descriptions for unseen categories.
- Error propagation exists across the multi-stage pipeline; a failure in viewpoint estimation will lead to complete failure downstream.
- The shapes generated by Image-to-3D models may match the class of the input point cloud but differ in details, leading to semantically plausible but geometrically inaccurate results after fusion.
- Though optional, SDS refining still requires minutes of optimization time. A fully feed-forward pipeline would be more efficient.
- For extremely sparse or heavily occluded inputs (very small surface areas), both depth projection and viewpoint selection can become unreliable.

## Related Work & Insights

- **vs SDS-Complete**: SDS-Complete uses implicit priors from 2D diffusion models to optimize SDF from scratch, which is slow and produces coarse geometry. GenPC leverages explicit priors from 3D generative models for feed-forward inference, reducing CD by 36% with a dramatic speedup.
- **vs PoinTr / SnowflakeNet**: Learning-based methods rely on synthetic data. GenPC's zero-shot performance significantly outperforms them, underscoring the generalization advantage brought by large-scale pre-training.
- **vs Huang et al.**: Also a zero-shot method but uses SDS optimization with 3DGS + Zero123, which is slow and limited by SDS quality. GenPC utilizes feed-forward 3D models to circumvent the inherent issues of SDS.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce feed-forward 3D generative models to zero-shot point cloud completion. The modality-bridging idea of Depth Prompting is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three real-world datasets (Redwood/ScanNet/KITTI) with extensive ablation studies validating all modules.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with comprehensive explanations of design motivations and comparison of alternative methods for each module.
- Value: ⭐⭐⭐⭐ Provides a practical zero-shot solution for real-world point cloud completion, with concepts generalizable to other modality-bridging tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[CVPR 2025\] Parametric Point Cloud Completion for Polygonal Surface Reconstruction](parametric_point_cloud_completion_for_polygonal_surface_reconstruction.md)
- [\[CVPR 2025\] Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](../../CVPR2026/3d_vision/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
