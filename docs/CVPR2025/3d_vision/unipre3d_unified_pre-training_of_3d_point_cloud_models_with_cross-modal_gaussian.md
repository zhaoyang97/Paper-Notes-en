---
title: >-
  [Paper Note] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Pre-training] UniPre3D proposes the first unified 3D pre-training framework that predicts Gaussian primitives and renders images using differentiable Gaussian Splatting to provide pixel-level supervision. Meanwhile, it introduces a scale-adaptive cross-modal fusion strategy, making the pre-training method applicable to point clouds of arbitrary scales (both object-level and scene-level) and 3D models of arbitrary architectures.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Pre-training"
  - "Point Clouds"
  - "Gaussian Splatting"
  - "Cross-Modal Fusion"
  - "Unified Learning"
date: 2026-05-08
content_hash: 27e434bcbb8d17a2
---

# UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2506.09952](https://arxiv.org/abs/2506.09952)  
**Code**: [https://github.com/wangzy22/UniPre3D](https://github.com/wangzy22/UniPre3D)  
**Area**: 3D Vision  
**Keywords**: 3D Pre-training, Point Clouds, Gaussian Splatting, Cross-Modal Fusion, Unified Learning  

## TL;DR

UniPre3D proposes the first unified 3D pre-training framework that predicts Gaussian primitives and renders images using differentiable Gaussian Splatting to provide pixel-level supervision. Meanwhile, it introduces a scale-adaptive cross-modal fusion strategy, making the pre-training method applicable to point clouds of arbitrary scales (both object-level and scene-level) and 3D models of arbitrary architectures.

## Background & Motivation

**Background**: Representation learning for 3D point clouds is divided into point-based methods (such as PointNet++, which excel in object-level fine-grained local structures) and voxel-based methods (such as SparseUNet, which excel in modeling scene-level long-range relationships). Consequently, current 3D pre-training approaches are split into two paradigms: the Masked Autoencoder (MAE) paradigm for object level, and the contrastive learning paradigm for scene level.

**Limitations of Prior Work**: MAE methods perform poorly on scene-level data because Chamfer Distance loss is computationally expensive and provides imprecise supervision for large-scale data; contrastive learning tends to saturate too quickly on object-level data, leading to limited pre-training effectiveness. Each paradigm only suits specific scenarios, making unification difficult.

**Key Challenge**: The scale diversity of point clouds is the fundamental challenge—scene point clouds can contain hundreds of times more points than object point clouds, making it impossible for a single pre-training paradigm to suit both scales simultaneously. However, in the 2D image domain, the discrepancy in information density between object and scene images is much smaller.

**Goal**: To design a unified 3D pre-training method that can: (1) apply to point clouds of arbitrary scales (object-level and scene-level); (2) apply to 3D models of arbitrary architectures (point-based and voxel-based).

**Key Insight**: The authors observe that the scale differences in the 2D image domain are much smaller than those in the 3D point cloud domain. Hence, they propose using the image domain as an intermediate medium to bridge the scale gap in point cloud data. Rendering projected images from 3D data as a pre-training task allows the difficulty of the task to scale adaptively with the data.

**Core Idea**: Use 3D Gaussian Splatting (3DGS) to predict Gaussian primitives from point clouds and render images, training the 3D backbone end-to-end via pixel-level image reconstruction loss, while fusing 2D features from a pre-trained image model to regulate task complexity.

## Method

### Overall Architecture

The pipeline of UniPre3D consists of two modal branches. The **3D branch** comprises a point cloud backbone, a lightweight Gaussian predictor, and a differentiable image renderer. The **2D branch** includes a pre-trained image feature extractor, a 2D-to-3D geometric projector, and a scale-adaptive fusion module. The overall forward propagation is divided into three stages: Extract, Fuse, and Render.

The inputs are the point cloud $P \in \mathbb{R}^{N \times 3}$ and a reference view image $I_{\text{ref}}$, and the output is the image $I_r$ rendered from the predicted Gaussian primitives, which is optimized end-to-end using the MSE loss against the ground-truth image.

### Key Designs

1. **Gaussian Primitive Prediction and Rendering**:

    - **Function**: Converts the 3D point cloud pre-training task into an image rendering task, achieving pixel-level precise supervision.
    - **Mechanism**: After the 3D backbone extracts point cloud features, a lightweight MLP head predicts the 3D Gaussian primitive parameters for each point (position offsets, opacity, scale, rotation quaternion, spherical harmonics, totaling 23 dimensions). Then, multi-view images are rendered using differentiable Gaussian Splatting. The covariance attribute of the Gaussian primitives determines their effective boundary; small-scale point clouds learn larger covariances (slight blur is acceptable), while large-scale point clouds have denser Gaussians to bring out more details.
    - **Design Motivation**: Compared with NeRF (used in the Ponder series), 3DGS has three major advantages: (1) scale adaptation—the covariance self-adjusts; (2) lightweight design—it only requires an MLP head, preventing pre-training knowledge from being absorbed by auxiliary modules; (3) efficiency—it supports full-image supervision, running about twice as fast as PonderV2.

2. **Object-level Feature Fusion**:

    - **Function**: Fuses 2D image features with 3D features during object-level pre-training to provide color and texture information.
    - **Mechanism**: Since object-level datasets lack depth maps, the 2D-3D correlation is established by projecting 3D points onto the 2D plane. For each point, its pixel coordinates $(u, v)$ on the reference image are computed, and points with the minimum depth are selected as surface points to align with the pixel grid. Then, the aligned 2D features are concatenated with the 3D features from the last decoder layer of the backbone, and fused through an MLP: $F_{\text{fuse}} = \text{MLP}(\text{cat}(F_{3D}, \hat{F}_{2D}))$.
    - **Design Motivation**: Object-level point clouds lack color but the rendering target has color; directly learning this forces the backbone to extract downstream-irrelevant color features. Fusing features from the reference view image provides color cues, allowing the backbone to focus on learning geometric structure.

3. **Scene-level Point Fusion**:

    - **Function**: Enhances visual guidance by adding pseudo-point clouds in scene-level pre-training, reducing the difficulty of the pre-training task.
    - **Mechanism**: Utilizing the depth maps of the scene dataset, 2D pixels are back-projected into the 3D space using the camera's intrinsic and extrinsic parameters to obtain a pseudo-point cloud $P_{2D}$. This is merged with $P_{3D}$ output from the first encoder layer of the backbone to form a cross-modal meta-point cloud. After voxelized downsampling, the fused features are extracted through the remaining network. This operation increases the number of Gaussian primitives by about 70%.
    - **Design Motivation**: Scene point clouds are highly sparse and geometrically complex; using only the point cloud as input makes the pre-training task excessively difficult. While the feature fusion strategy performs poorly at the scene level (as confirmed by ablation studies), point fusion lowers the optimization difficulty by supplementing dense 2D information.

### Loss & Training

- **Loss Function**: Uses pixel-level MSE loss. Object-level pre-training introduces foreground/background weighting: $\mathcal{L}^{\text{obj}} = \omega_{\text{fg}} \mathcal{L}(I_r^{\text{fg}}, I_{\text{gt}}^{\text{fg}}) + \omega_{\text{bg}} \mathcal{L}(I_r^{\text{bg}}, I_{\text{gt}}^{\text{bg}})$, where $\omega_{\text{fg}}=4, \omega_{\text{bg}}=1$.
- **Rendering Strategy**: The rendering views and the reference views do not overlap to prevent information leakage; scene-level pre-training limits the reference/rendering view interval (<5 frames) to improve the utilization of image knowledge.
- **Training Details**: Object-level pre-training is performed on ShapeNet for 50 epochs using 1x 3090Ti; scene-level is performed on ScanNetV2 for 100 epochs using 8x 3090Ti. The 2D branch utilizes the Stable Diffusion autoencoder.

## Key Experimental Results

### Main Results

| Task | Backbone | Dataset | Metric | No Pre-training | UniPre3D |
|------|----------|--------|------|----------|----------|
| Classification | Std. Transformer | ScanObjectNN PB_T50_RS | OA(%) | 77.24 | **87.93** |
| Classification | Mamba3D | ScanObjectNN PB_T50_RS | OA(%) | 92.6 | **93.4** |
| Part Segmentation | PointMLP | ShapeNetPart | mIoU_C | 84.6 | **85.5** |
| Semantic Segmentation | SparseUNet | ScanNet200 | mIoU | 25.0 | **28.3** |
| Semantic Segmentation | PTv3 | ScanNet200 | mIoU | 35.2 | **36.4** |

### Ablation Study

| Ablation Configuration | ScanNet200 mIoU |
|----------|----------------|
| No 2D Fusion | 26.8 |
| Feature Fusion (Scene-level) | 27.0 |
| Point Fusion (Scene-level) | **28.3** |
| No Fore/Background Weighting (Object-level) | OA drops by ~0.5% |

### Key Findings

- UniPre3D achieves consistent improvements in both object-level and scene-level tasks, validating the effectiveness of unified pre-training.
- Even for existing high-performance backbones (such as Mamba3D at 92.6% and PTv3 at 77.45%), UniPre3D still brings performance gains.
- The scene-level point fusion strategy significantly outperforms the feature fusion strategy because scene data is sparser and more complex.
- Although the images rendered by 3DGS at the scene level are relatively blurry, they are sufficient for learning basic geometric relationships.

## Highlights & Insights

- Achieves unified 3D pre-training for the first time, breaking down the barrier between object-level and scene-level pre-training paradigms.
- The idea of using the image domain as an "intermediate medium" to bridge the scale gaps of point clouds is highly elegant.
- Different fusion strategies (feature fusion vs. point fusion) are designed specifically for different scales of data rather than taking a one-size-fits-all approach.
- The overall design is lightweight and efficient, ensuring that auxiliary pre-training components do not overshadow the representations learned by the backbone.

## Limitations & Future Work

- Low-quality (blurry) rendered images at the scene level may limit the ceiling of pre-training performance.
- The 2D branch uses a fixed Stable Diffusion autoencoder; more powerful vision foundation models are yet to be explored.
- The impacts of larger-scale datasets and longer pre-training schedules remain to be explored.
- Future work could consider incorporating depth, normals, and other geometric signals to enhance rendering supervision.

## Related Work & Insights

- **Ponder/PonderV2**: Also use generative pre-training but based on NeRF; UniPre3D replaces NeRF with 3DGS to achieve a 2x speedup and full-image supervision.
- **TAP**: Another generative pre-training method that uses an attention-based predictor which is bulky; UniPre3D's MLP head is much more lightweight.
- **Point-MAE Series**: The MAE paradigm is effective at the object level but fails at the scene level, which is the core challenge addressed in this work.

## Rating

- **Novelty**: 8/10 — The first unified 3D pre-training method, featuring an innovative idea of bridging 2D and 3D via 3DGS.
- **Experimental Thoroughness**: 9/10 — Covers multiple backbones, tasks, and datasets with comprehensive ablation studies.
- **Writing Quality**: 8/10 — Clear logic with well-explained design motivations.
- **Value**: 8/10 — Promotes advancements in the 3D pre-training field, though there is still room for improvement in scene-level rendering quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)
- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](crossover_3d_scene_cross-modal_alignment.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](../../ICCV2025/3d_vision/towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[CVPR 2026\] Low-Rank Test-Time Training for Pre-Trained Point Cloud Models](../../CVPR2026/3d_vision/low-rank_test-time_training_for_pre-trained_point_cloud_models.md)

</div>

<!-- RELATED:END -->
