---
title: >-
  [Paper Note] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation
description: >-
  [ICCV 2025][3D Vision][Object Pose Estimation] This paper proposes BoxDreamer, which adopts 3D bounding box corners as an intermediate representation. A reference-view-based corner synthesizer predicts 2D corner projecti…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Object Pose Estimation"
  - "Sparse-View"
  - "3D Bounding Box"
  - "Occlusion Handling"
  - "Generalizable"
date: 2026-05-08
content_hash: 5d65d74d996ade02
---

# BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation

**Conference**: ICCV 2025
**arXiv**: [2504.07955](https://arxiv.org/abs/2504.07955)  
**Code**: [https://zju3dv.github.io/boxdreamer](https://zju3dv.github.io/boxdreamer)  
**Area**: 3D Vision
**Keywords**: Object Pose Estimation, Sparse-View, 3D Bounding Box, Occlusion Handling, Generalizable

## TL;DR

This paper proposes BoxDreamer, which adopts 3D bounding box corners as an intermediate representation. A reference-view-based corner synthesizer predicts 2D corner projections in query images, and 6DoF poses are recovered via PnP using the resulting 2D–3D correspondences. The method significantly outperforms existing approaches under sparse-view and heavy-occlusion conditions.

## Background & Motivation

Generalizable object pose estimation (without CAD models) faces two major challenges:

**Retrieval-based methods** (e.g., Gen6D): Pose is initialized by retrieving the most similar reference image and then refined. However, under sparse views it is difficult to find a close reference viewpoint, and occlusion causes retrieval failures.

**Matching-based methods** (e.g., OnePose++): The target object's point cloud is first reconstructed, then 2D–3D matches are established. These methods rely on complete dense point cloud reconstruction, which degrades under sparse views, and occlusion reduces matching effectiveness.

Core observation: **3D bounding boxes** are compact geometric primitives that can be reliably recovered even from extremely sparse reference views. Their corner points, as semantic object landmarks, naturally establish 2D–3D correspondences and are inherently robust to occlusion.

## Method

### Overall Architecture

BoxDreamer operates in two stages:
1. **3D Bounding Box Recovery**: Sparse-view reconstruction tools (e.g., DUSt3R) are used to estimate camera poses and recover approximate object geometry, from which the 3D bounding box is computed.
2. **2D Corner Projection Prediction**: 3D corners are projected onto reference images to generate heatmaps, which are then fed into a Transformer decoder to predict the 2D projections of 8 corners in the query image.
3. PnP is applied to the 2D–3D correspondences to recover the 6DoF pose.

### Key Designs

1. **3D Bounding Box as Object Representation**: Compared to dense point clouds, 3D bounding boxes can be reliably recovered from very sparse inputs (as few as 5 reference images) without requiring precise reconstruction. Pointmaps are obtained via feed-forward reconstruction methods such as DUSt3R; irrelevant points are filtered using object detection results, and the bounding box is then computed.

2. **2D Heatmaps as Corner Representation**: Raw 8-corner coordinate signals are too sparse for ViT learning. Inspired by the Gaussian smoothing idea in CornerNet, a revised heatmap function is defined as $\mathbf{H}(x,y,i) = \exp\left(-\frac{\sqrt{(x-x_i)^2+(y-y_i)^2}}{2\sigma^2}\right)$, where $2\sigma^2$ is set to the square of one-tenth of the distance from the corner to the 2D object center, yielding smoother signals.

3. **Transformer-based Corner Synthesizer**: DINOv2 is used to extract features $\mathbf{F} \in \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times d}$ from both reference and query images. Reference heatmaps are patchified and linearly projected, then element-wise added to the image features as $\mathbf{F}_i' = \mathbf{F}_i + \mathbf{H}_i^p$. The query image uses learnable query tokens. The concatenated tokens are passed through a 12-layer full self-attention Transformer decoder to produce the bounding box heatmap for the query viewpoint.

### Loss & Training

A coarse-to-fine two-level supervision strategy with Smooth L1 Loss is adopted:
- **Coarse loss**: Smooth L1 reconstruction loss over the full heatmap.
- **Fine loss**: Smooth L1 loss over the 8 corner coordinates.
- Final loss: $L = L_{\text{coarse}} + \lambda L_{\text{fine}}$, with $\lambda=2.0$.

Training data includes Objaverse (45K+ synthetic objects) and OnePose (50 real objects), totaling 2.9M+ images. Data augmentation includes random 3D bounding box rotation (to break semantic associations), RGB augmentation (motion blur/noise), random background composition, and random occlusion. AdamW optimizer is used; the model is trained for 100 epochs on 8 A100 GPUs.

## Key Experimental Results

### Main Results (Tables)

Comparison on Occluded LINEMOD (ADD(s)-0.1d / Proj-2d@5px, with 5 and 25 reference images):

| Method | 5-ref ADD(s) | 25-ref ADD(s) | 5-ref Proj2D | 25-ref Proj2D |
|--------|-------------|---------------|-------------|---------------|
| OnePose++ | - | 3.8 | - | 3.1 |
| Gen6D | 4.7 | 16.0 | 6.3 | 23.4 |
| Gen6D† | 10.2 | 25.5 | 11.6 | 36.1 |
| **BoxDreamer** | **26.5** | **43.6** | **21.9** | **47.9** |

YCB-Video Sparse Database (16 reference images, ADD-S / ADD):

| Method | # Refs | ADD-S Avg | ADD Avg |
|--------|--------|-----------|---------|
| Gen6D‡ | 16 | 42.0 | 18.8 |
| Gen6D† | 16 | 58.3 | 36.2 |
| OnePose++ | 16 | 20.1 | 9.7 |
| **BoxDreamer** | 5 | 60.5 | 36.8 |
| **BoxDreamer** | 16 | **69.2** | **47.6** |

On LINEMOD with only 5 reference images, BoxDreamer (53.1) achieves more than twice the performance of the second-best method GS-Pose (25.6).

### Ablation Study (Tables)

Based on information described in the ablation section:

| Setting | Description |
|---------|-------------|
| 5 refs → full set | Performance with only 5 reference images already surpasses all competitors; continues to improve as reference count increases |
| Bounding box accuracy | Pose estimation remains effective even when the 3D bounding box is imprecise |
| Inference speed | More than 40× faster than GS-Pose's refinement stage |

Transformer component ablations (in supplementary material) cover decoder depth, number of reference views, and data augmentation strategies.

### Key Findings

- On heavy-occlusion benchmarks (Occluded LINEMOD), BoxDreamer substantially outperforms existing methods; on some objects it even surpasses the instance-level method PVNet, which requires CAD models.
- The global semantic nature of corner points enables the model to infer occluded corner locations from visible object parts and reference demonstrations.
- Practical performance is achievable with as few as 5 reference images, addressing the bottleneck of collecting dense reference databases.

## Highlights & Insights

- **Elegant representation choice**: Using 3D bounding boxes as an intermediate representation for object pose cleverly balances information richness and robustness — easier to recover from sparse views than dense point clouds, and more physically meaningful than direct pose regression.
- **End-to-end Transformer prediction**: DINOv2 pretrained features combined with cross-attention mechanisms efficiently predict query-view corners from reference corner demonstrations.
- **Engineering simplicity**: No CAD models, no dense reference views, and no depth information are required; the method operates on RGB images alone.
- Fast inference (approximately 0.02s/query vs. 0.96s for GS-Pose).

## Limitations & Future Work

- Extreme occlusion cases (e.g., only a tiny portion of the object is visible) remain challenging.
- The bounding box representation cannot capture fine geometry of non-convex objects.
- Corner matching is ambiguous for symmetric objects (e.g., bowls).
- The method depends on external reconstruction tools such as DUSt3R to obtain the initial 3D bounding box; reconstruction quality constrains the performance ceiling.
- No direct comparison with depth-input methods such as FoundationPose is provided.

## Related Work & Insights

- The Gen6D series and OnePose series are the most direct baselines.
- GS-Pose represents objects with 3D Gaussians but requires lengthy construction time.
- The corner heatmap idea from CornerNet is adapted for 3D object representation.
- The method has practical value for applications requiring fast pose estimation, such as robotic grasping and AR.

## Rating

- Novelty: ⭐⭐⭐⭐ Using bounding box corners as an intermediate representation for pose estimation is a novel and elegant idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison with multiple methods across 4 datasets under various reference-count settings.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with thorough comparative analysis against existing paradigms.
- Value: ⭐⭐⭐⭐⭐ Highly practical; achieves significant breakthroughs on the core pain points of sparse views and occlusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [\[CVPR 2026\] Exploring 6D Object Pose Estimation with Deformation](../../CVPR2026/3d_vision/exploring_6d_object_pose_estimation_with_deformation.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
