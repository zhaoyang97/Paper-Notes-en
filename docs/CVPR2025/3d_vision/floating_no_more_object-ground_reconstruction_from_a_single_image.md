---
title: >-
  [Paper Note] Floating No More: Object-Ground Reconstruction from a Single Image
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] The ORG framework is proposed to jointly model object 3D geometry, camera parameters, and the object-ground relationship from a single image for the first time. By predicting two compact dense representations—pixel height maps and perspective fields—it solves the problem of "floating/tilting" in reconstructed objects, significantly improving the realism of shadow generation and pose manipulation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "Object-Ground Relationship"
  - "Pixel Height"
  - "Perspective Field"
  - "Shadow Generation"
date: 2026-05-08
content_hash: 9c539b254d55fb53
---

# Floating No More: Object-Ground Reconstruction from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2407.18914](https://arxiv.org/abs/2407.18914)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Single-image 3D Reconstruction, Object-Ground Relationship, Pixel Height, Perspective Field, Shadow Generation

## TL;DR
The ORG framework is proposed to jointly model object 3D geometry, camera parameters, and the object-ground relationship from a single image for the first time. By predicting two compact dense representations—pixel height maps and perspective fields—it solves the problem of "floating/tilting" in reconstructed objects, significantly improving the realism of shadow generation and pose manipulation.

## Background & Motivation

1. **Background**: Single-image 3D reconstruction has made significant progress in recent years. Mainstream methods include monocular depth estimation (MiDaS, LeReS), category-specific reconstruction based on implicit representations (PIFu), and novel-view synthesis based on diffusion (Zero-123).
2. **Limitations of Prior Work**: These methods focus on the accuracy of object shapes but neglect the relationship among the object, the ground, and the camera. Depth estimation methods require extra camera parameters to project into 3D point clouds and suffer from unknown scale/shift factors, leading to reconstruction distortion. Category-specific methods and novel-view synthesis methods often assume simple orthographic cameras or prior knowledge of camera parameters, limiting their application in unconstrained scenes.
3. **Key Challenge**: The lack of explicit modeling of the object-ground relationship often causes the reconstructed objects to appear "floating" or "tilted" when placed on a flat ground, which severely degrades the realism of 3D editing applications such as shadow rendering and reflection generation.
4. **Goal**: (1) How to simultaneously estimate object 3D shape, camera parameters, and the ground relationship from a single image? (2) How to encode these three relationships in compact pixel-level representations? (3) How to efficiently convert these representations into depth maps and point clouds?
5. **Key Insight**: It is observed that the representation of pixel height is naturally decoupled from camera models, intuitively measuring the distance from the object to the ground; meanwhile, the perspective field encodes camera intrinsic and extrinsic parameters in a dense manner. Combining both simultaneously captures the object-ground-camera trinity relationship.
6. **Core Idea**: Jointly model the object-ground-camera relationship using two compact dense representations—pixel height maps and perspective fields—achieving the first "grounded" reconstruction framework from a single image.

## Method

### Overall Architecture
The input is a single object-centric image, and the outputs are the pixel height maps of the front and back surfaces of the object, the latitude field, and the up-direction field. The model uses PVTv2-b3 as the encoder and SegFormer as the decoder to predict these dense fields via regression. After prediction, the "Perspective Field-Guided Pixel Height Reprojection" module transforms the two representations into depth maps and 3D point clouds.

### Key Designs

1. **Dual-Surface Pixel Height Representation**:

    - **Function**: Encodes the pixel distance from the front and back surfaces of the object to the ground.
    - **Mechanism**: For each pixel, a camera ray passing through the object has an entry point (front surface $\mathbf{p}_f$) and an exit point (back surface $\mathbf{p}_b$). The pixel distances from these points to their ground projections are predicted and normalized by the image height. Unlike depth, pixel height is decoupled from the camera model and can be directly inferred from the image context without requiring additional camera information.
    - **Design Motivation**: The original pixel height formulation only considers the front surface with strict camera viewpoint constraints. This work extends it to dual surfaces and jointly models camera parameters to relax these constraints, making it applicable not only for shadow generation but also for complete 3D reconstruction.

2. **Perspective Field Representation**:

    - **Function**: Encodes camera intrinsic and extrinsic parameters in a pixel-level dense manner.
    - **Mechanism**: The perspective field contains a latitude field (encoding the elevation angle of each pixel relative to the horizon) and an up-direction field (encoding the roll angle direction of each pixel). The latitude field is normalized to $[0,1]$, and the angle $\theta$ of the up-direction field is represented by $(\sin\theta, \cos\theta)$ to avoid $0/2\pi$ ambiguity. By performing a grid search optimization over the predicted perspective fields, the camera field-of-view $\alpha$ and rotation matrix $\mathbf{R}$ can be recovered.
    - **Design Motivation**: Both pixel height and perspective fields exhibit invariance or equivariance under image cropping, rotation, and translation. They are naturally suited for neural network modeling in dense prediction tasks, and their joint prediction enables a self-contained 3D reconstruction pipeline.

3. **Perspective Field-Guided Pixel Height Reprojection Module**:

    - **Function**: Efficiently converts the predicted pixel height and perspective fields into depth maps and 3D point clouds.
    - **Mechanism**: It first recovers the camera focal length $f = H/(2\tan(\alpha/2))$ and camera intrinsic/extrinsic matrices from the perspective field. Then, leveraging two constraints—(1) all points on the ground share the same z-coordinate, and (2) an object point and its ground projection share the same XY physical coordinates—the unknown depth $d$ is eliminated, yielding the normalized 3D coordinates $\mathbf{P}_n^{world} = (X_n Y_n)/(XY) \cdot (X, Y, Z)$.
    - **Design Motivation**: It proves that the pixel height and the perspective field encode sufficient information for full 3D reconstruction, allowing the model's output to be fairly compared with existing depth-based methods.

### Loss & Training
All regression tasks use the $\ell_2$ loss. Training is conducted using the AdamW optimizer with a learning rate of 0.0005, weight decay of 1e-2, for a total of 60K steps, with a batch size of 8 on 4×A100 GPUs. The learning rate is decayed by a factor of 10 at 30K, 40K, and 50K steps. Data augmentation includes horizontal flipping, random cropping, and color jittering.

## Key Experimental Results

### Main Results

| Method | Camera Parameters | AbsRel↓ | δ₁↑ | LSIV↓ | CD↓ |
|------|---------|---------|-----|-------|-----|
| MiDaS + Ctrl-C | Off-the-shelf | 22.7 | 77.9 | 1.22 | 1.39 |
| LeReS + Ctrl-C | Off-the-shelf | 30.0 | 63.1 | 1.05 | 1.31 |
| ORG (Ours) | Ours | **19.1** | **81.2** | **0.93** | **1.26** |

### Ablation Study

| Object Geometry | Camera Parameters | LSIV↓ | Gain |
|----------|---------|-------|------|
| depth | OFS estimator | 1.25 | Baseline |
| depth | perspective field | 1.01 | -0.24 |
| pixel height | OFS estimator | 0.98 | -0.27 |
| pixel height | perspective field | **0.81** | **-0.44** |

### Key Findings
- **Pixel height representation outperforms depth representation**: Under the same data and training configurations, point cloud reconstruction via pixel height is superior to depth estimation. This is because it focuses more on the object-ground geometry rather than the object-camera geometry, making it easier to infer from the image.
- **Joint training yields the largest contribution**: The joint estimation of pixel height and perspective fields achieves the most significant improvement compared to any single-replacement baseline (LSIV drops from 1.25 to 0.81).
- **The advantage is more pronounced with increased viewpoint diversity**: ORG achieves the greatest gain over the baseline during large viewpoint changes (LSIV gain of -0.27 vs. -0.02 for small viewpoints), as conventional camera estimation performs poorly under extreme pitch angles.

## Highlights & Insights
- **Insight on replacing depth with pixel height**: Pixel height is naturally decoupled from camera models and encodes the object-ground relationship, which is more "natural" and easier to infer than depth. The paradigm of changing representations to simplify the problem can be transferred to any task requiring object-scene relationship modeling.
- **Dense representations encoding sparse parameters**: Utilizing pixel-level dense fields to encode global camera parameters (perspective fields) both preserves precise spatially-varying information and exploits the inductive bias of dense prediction networks, which is highly elegant.
- **Lightweight pipeline**: The entire method requires only a single forward inference pass to obtain depth maps, point clouds, and shadows, avoiding the need for multi-view inputs or expensive diffusion-based generation.

## Limitations & Future Work
- **Assumption of a flat ground**: It is inapplicable to non-"grounded" scenarios such as hanging objects or objects in water.
- **Modeling only front and back surfaces**: It may be insufficient for topologically complex objects (e.g., toroidal or hollowed-out structures).
- **Reliance on Objaverse synthetic data for training**: Although it exhibits good generalization, fine-tuning on real-world data could further boost performance.
- **Extensible areas**: Extending pixel height to a multi-layer representation to handle more complex shapes; introducing semantic priors to enhance the robustness of ground detection.

## Related Work & Insights
- **vs. LeReS/MiDaS**: These methods perform general monocular depth estimation but do not model ground relationships, which causes the reconstructed objects to float. ORG addresses this fundamental issue through the joint modeling of pixel height and perspective fields.
- **vs. Zero-123**: Zero-123 performs novel-view synthesis but assumes simple camera models, making it unable to accurately place objects on the ground. ORG's lightweight output can be directly used for shadow generation, which is much more efficient than diffusion-based methods.
- **vs. original pixel height methods**: The original methods suffer from strict viewpoint constraints and only consider the front surface. ORG significantly broadens the scope of application through dual-surface modeling and joint camera estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ It models the object-ground-camera trinity relationship jointly for the first time, and the insight of replacing depth with pixel height is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluations (depth, point cloud, shadow, reflection) and detailed ablation studies, though it lacks large-scale real-world quantitative evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rigorous derivations, and high-quality figures.
- Value: ⭐⭐⭐⭐ Direct practical value for 3D editing applications (shadows/reflections/pose manipulation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)
- [\[CVPR 2025\] PhysGen3D: Crafting a Miniature Interactive World from a Single Image](physgen3d_crafting_a_miniature_interactive_world_from_a_single_image.md)
- [\[CVPR 2025\] GenVDM: Generating Vector Displacement Maps From a Single Image](genvdm_generating_vector_displacement_maps_from_a_single_image.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)

</div>

<!-- RELATED:END -->
