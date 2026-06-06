---
title: >-
  [Paper Note] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision
description: >-
  [ICCV 2025][3D Vision][Egocentric vision] This paper proposes Fish2Mesh, a fisheye-aware Transformer model that embeds the spherical geometry of fisheye images into a Swin Transformer via an Egocentric Positional Encodin…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Egocentric vision"
  - "fisheye undistortion"
  - "human mesh recovery"
  - "SMPL"
  - "positional encoding"
date: 2026-05-08
content_hash: acde3f58724a0877
---

# Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision

**Conference**: ICCV 2025
**arXiv**: [2503.06089](https://arxiv.org/abs/2503.06089)  
**Code**: Available (project website)  
**Area**: 3D Vision
**Keywords**: Egocentric vision, fisheye undistortion, human mesh recovery, SMPL, positional encoding

## TL;DR
This paper proposes Fish2Mesh, a fisheye-aware Transformer model that embeds the spherical geometry of fisheye images into a Swin Transformer via an Egocentric Positional Encoding (EPE) based on equirectangular projection, enabling accurate 3D human mesh recovery from a head-mounted fisheye camera in egocentric perspective.

## Background & Motivation

**Background**: Egocentric human estimation has progressed from binocular stereo systems to monocular setups, yet mainstream research remains focused on joint pose estimation. Human Mesh Recovery (HMR) captures richer body shape and volume information compared to joint estimation, but poses greater challenges under egocentric fisheye viewpoints.

**Limitations of Prior Work**: (1) Egocentric datasets are scarce and difficult to annotate; (2) fisheye lenses introduce severe spatial distortion, especially near image boundaries; (3) self-occlusion is severe — arms occlude the torso, and the head limits visibility of the lower body; (4) the current SOTA method EgoHMR employs a diffusion model, producing high output uncertainty that is unsuitable for real-time XR and robotic interaction, and does not address fisheye distortion.

**Key Challenge**: Standard Transformer positional encodings assume a regular grid and cannot express the nonlinear spatial warping of fisheye projections. Applying standard models directly to fisheye images loses 3D spatial context.

**Goal**: Design a Transformer architecture that natively understands fisheye geometry to accurately regress SMPL parameters and 3D human meshes from egocentric fisheye views.

**Key Insight**: Treating fisheye images as spherical projections, the method generates learnable 3D positional encodings via equirectangular projection–spherical coordinate conversion to encode the spherical spatial information of each pixel.

**Core Idea**: The 2D pixel coordinates of a fisheye image are converted to 3D spherical coordinates $(x_{3D}, y_{3D}, z_{3D})$ via equirectangular projection, discretized to construct a learnable positional embedding table that replaces the standard relative position bias in Swin Transformer, enabling the model to natively perceive fisheye geometric distortion.

## Method

### Overall Architecture
Given an input fisheye RGB image, EPE positional encodings are first computed from equirectangular projection and fed together with image patches into the patch merging layer of a Swin Transformer. Hierarchical features are extracted through four Swin blocks and then passed to three task heads that respectively predict SMPL shape/pose parameters, camera transformation, and global orientation. Auxiliary 3D/2D joint losses supervise training.

### Key Designs

1. **Egocentric Positional Encoding (EPE)**:

    - Function: Encodes 3D spherical information of fisheye geometry into the Transformer.
    - Mechanism: The 2D fisheye pixel coordinates $(x_{2D}, y_{2D})$ are first converted to spherical longitude and latitude $(\lambda, \varphi)$ via the equirectangular projection formula, then to 3D spherical coordinates such as $x_{3D} = R \cdot \sin(\varphi) \cdot \cos(\lambda)$. The continuous coordinates are discretized to query a learnable embedding table $POS[x_{3D}, y_{3D}, z_{3D}]$, which is added to the image tokens. The original relative position bias in Swin Transformer is simultaneously removed, as EPE already encodes positional information.
    - Design Motivation: Standard positional encodings cannot express spherical distortion; EPE directly injects 3D spatial constraints into the model, allowing self-attention to operate in a geometrically correct feature space.

2. **Multi-task Head Joint Optimization**:

    - Function: Jointly regresses SMPL parameters and auxiliary joint coordinates.
    - Mechanism: Three task heads output SMPL shape parameters $\Theta_s$, pose parameters $\Theta_p$, camera transformation $\Pi$, and global orientation $O$, while simultaneously predicting 3D joint coordinates and their 2D projections as auxiliary supervision to ensure 3D–2D consistency.
    - Design Motivation: The SMPL parameter space is high-dimensional and abstract; auxiliary joint losses provide more direct geometric constraints and accelerate convergence.

3. **Weakly Supervised Data Augmentation**:

    - Function: Mitigates the scarcity of egocentric training data.
    - Mechanism: A pretrained 4D-Human model generates SMPL pseudo-labels from third-person camera images, combined with a prompt-based natural motion capture system to produce training data covering daily activities with realistic head motion and self-occlusion.
    - Design Motivation: Genuine egocentric data is extremely scarce and costly to annotate; the weakly supervised strategy substantially enlarges the training set.

### Loss & Training
$\mathcal{L} = a(\mathcal{L}_{SMPL} + \mathcal{L}_{orient}) + b \cdot \mathcal{L}_{3D} + c \cdot \mathcal{L}_{2D}$, where the SMPL loss is an L2 loss on shape and pose parameters, and orientation and 3D/2D joint losses are L1. The model is trained end-to-end from scratch.

## Key Experimental Results

### Main Results

| Model | MPJPE↓(mm) | MPVPE↓(mm) | PA-MPJPE↓(mm) | PA-MPVPE↓(mm) |
|---|---|---|---|---|
| 4DHuman | 390.0 | 521.3 | 90.0 | 129.8 |
| EgoHMR (diffusion) | Worse | Worse | Competitive | Competitive |
| **Fish2Mesh** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Effect | Note |
|---|---|---|
| w/o EPE (standard positional encoding) | Degraded | Fisheye distortion unaddressed |
| w/o auxiliary 3D/2D losses | Degraded | Lack of geometric constraints |
| w/o data augmentation | Significantly degraded | Insufficient training data |
| Full Fish2Mesh | Best | All components are complementary |

### Key Findings
- EPE positional encoding is the largest single contributor to performance gains, confirming that geometry awareness is central to fisheye HMR.
- Deterministic regression (this work) is better suited to real-time applications than diffusion-based generation (EgoHMR).
- Weakly supervised data augmentation is critical for compensating the scarcity of egocentric data.
- The model consistently outperforms SOTA across multiple egocentric datasets.

## Highlights & Insights
- **Geometric Intuition of EPE**: Fisheye distortion is no longer treated as an error to be corrected; instead, its spherical nature is encoded into the model via 3D positional encodings. This is more elegant than pre-undistorting images and retains more information.
- **From Pose Estimation to Mesh Recovery**: The paper highlights inconsistencies in keypoint definitions across datasets (COCO 32-point vs. H36M 17-point); mesh recovery naturally circumvents this problem.
- **Practicality-Oriented Design**: The method is explicitly designed for real-time XR and robotic interaction scenarios, avoiding the uncertainty drawbacks of diffusion models.

## Limitations & Future Work
- Known fisheye lens parameters (focal length, field of view, etc.) are required; different devices require recalibration.
- Training data remain confined to laboratory environments; performance in real-world in-the-wild scenarios requires further validation.
- Only single frames are processed; temporal information is not exploited to improve estimation under occlusion.
- The approach could be extended to multi-person scenarios or combined with hand and face estimation.

## Related Work & Insights
- **vs. EgoHMR**: Uses a diffusion model but suffers from high output uncertainty and does not handle fisheye distortion; Fish2Mesh adopts deterministic regression with geometry-aware positional encoding.
- **vs. FisheyeViT**: Addresses fisheye via patch undistortion, but patch splitting incurs substantial preprocessing overhead; EPE is simpler and more efficient.
- The EPE concept is transferable to any vision Transformer task employing fisheye or panoramic cameras.

## Rating
- Novelty: ⭐⭐⭐⭐ EPE positional encoding, which natively incorporates fisheye geometry into a Transformer, is an innovative design.
- Experimental Thoroughness: ⭐⭐⭐ Limited datasets and baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated problem motivation.
- Value: ⭐⭐⭐⭐ Practical value for egocentric perception in XR and robotics scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/3d_vision/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](../../CVPR2026/3d_vision/fall_risk_gait_analysis_hmr.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)

</div>

<!-- RELATED:END -->
