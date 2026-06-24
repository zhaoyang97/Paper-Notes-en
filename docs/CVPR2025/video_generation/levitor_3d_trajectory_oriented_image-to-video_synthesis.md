---
title: >-
  [Paper Note] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis
description: >-
  [CVPR 2025][Video Generation][3D Trajectory Control] LeviTor introduces 3D object trajectory control into image-to-video synthesis for the first time. By clustering object masks into a small number of representative points via K-means and incorporating depth information as control signals injected into the SVD model, it achieves precise control over complex 3D motions such as occlusion, forward/backward movement, and orbiting, reaching FID/FVD of 25.41/190.44 on DAVIS.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "3D Trajectory Control"
  - "Image-to-Video"
  - "Video Diffusion Models"
  - "K-means Clustering Points"
  - "Depth Information"
date: 2026-05-08
content_hash: 996fd8a53d58e302
---

# LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2412.15214](https://arxiv.org/abs/2412.15214)  
**Code**: [https://github.com/ant-research/LeviTor](https://github.com/ant-research/LeviTor)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: 3D Trajectory Control, Image-to-Video, Video Diffusion Models, K-means Clustering Points, Depth Information

## TL;DR

LeviTor introduces 3D object trajectory control into image-to-video synthesis for the first time. By clustering object masks into a small number of representative points via K-means and incorporating depth information as control signals injected into the SVD model, it achieves precise control over complex 3D motions such as occlusion, forward/backward movement, and orbiting, reaching FID/FVD of 25.41/190.44 on DAVIS.

## Background & Motivation

**Background**: Trajectory-controlled video generation has made significant progress. DragNUWA converts sparse strokes into a dense optical flow space, DragAnything extracts entity representations to achieve entity-level control, and TrackGo defines motion via free-form masks and arrows. However, all these methods operate trajectories on a 2D plane.

**Limitations of Prior Work**: All existing methods only consider 2D trajectories, which leads to ambiguity in real 3D environments. For instance, when making a hot air balloon fly past a building, a 2D trajectory cannot distinguish whether the balloon passes in front of or behind the building; when an object moves forward or backward, a 2D trajectory fails to express the depth change, resulting in unrealistic object scaling; and complex 3D motions like orbiting cannot be expressed by 2D trajectories at all.

**Key Challenge**: The dimensionality of 2D trajectory information is insufficient to express motion in 3D space—an identical 2D trajectory can correspond to infinitely many 3D trajectories, leading to blurry generation results or violations of perspective projection laws. However, requiring users to input precise 3D trajectories is excessively difficult.

**Goal**: (1) Design an implicit control signal that represents 3D motion without requiring explicit 3D trajectory annotation; (2) Provide a simple way for users to input 3D control information; (3) Accurately control occlusion relations and depth changes between objects.

**Key Insight**: The authors discover that the spatial distribution change of K-means clustered points of an object mask can implicitly encode 3D motion information—points gathering indicates the object is moving away, points dispersing indicates the object is drawing near, and points disappearing behind another object indicates occlusion. Combined with relative depth values estimated by DepthAnythingV2, a simple 2D annotation with depth values can approximate a 3D trajectory.

**Core Idea**: Use the spatial distribution change of K-means clustered points together with relative depth values as a proxy representation of 3D motion, injecting it into a video diffusion model via ControlNet to achieve 3D trajectory control.

## Method

### Overall Architecture

Training Phase: High-quality videos and object mask annotations are obtained from the SA-V (VOS) dataset. For each frame, K-means clustering is applied to each mask to obtain representative points. DepthAnythingV2 is used to estimate the depth map for each frame, and depth values are assigned to the clustered points. The 2D Gaussian heatmap trajectory, instance information, and depth information are concatenated as control signals and injected into the SVD model via ControlNet for training. Inference Phase: The user selects an object mask to move on the initial image, draws a 2D trajectory, and specifies depth changes. The system projects the pixel points of the object mask into 3D space, moves them according to the user's specification, and renders them back to 2D to obtain frame-by-frame masks. Then, K-means is used to extract control points to generate the video.

### Key Designs

1. **K-means Clustered Points Control Signal**:

    - **Function**: Compress dense object masks into a small number of representative points to serve as motion control signals.
    - **Mechanism**: For each object mask in each frame, the area ratio is calculated and multiplied by a hyperparameter $\alpha$ to determine the cluster number $k$. If the object mask area changes by more than 10 times over the temporal dimension (indicating occlusion or depth change), $k \geq 3$ is enforced to better represent the variance, while limiting $k \leq 8$. The **gathering/dispersing** of clustered points implicitly reflects depth changes, while the **disappearance/appearance** of points reflects occlusion.
    - **Design Motivation**: Using dense masks as controls restricts the object to pure translation and prevents non-rigid deformation; a small number of clustered points provides the generative model with more freedom to add motion details. Additionally, the number of K-means points adaptively adjusts based on the mask size.

2. **3D Rendered Mask Inference Pipeline**:

    - **Function**: Convert user's sparse 3D trajectory input into physically consistent, frame-by-frame control signals.
    - **Mechanism**: The 2D pixels and depth values of the initial image are projected into the 3D camera coordinate system $[X_i, Y_i, Z_i]^T = \mathbf{K}^{-1} \cdot [x_i, y_i, 1]^T \cdot d_i$. The selected object's points are translated in 3D space according to the user-specified displacement vector $\mathbf{T}$. Then, the PyTorch3D renderer projects all 3D points back onto the 2D image plane, obtaining a sequence of masks that naturally incorporate occlusion relations and perspective scaling.
    - **Design Motivation**: Moving objects in 3D space and rendering them back to 2D inherently ensures that occlusion relations and size variations conform to perspective projection laws, sparing the user from manually inputting complex multi-point trajectories.

3. **Depth + Instance Information Fusion**:

    - **Function**: Add depth and object association dimensions to the control signals.
    - **Mechanism**: DepthAnythingV2 is employed to estimate relative depth, and depth values $d_t^i = D_t(x_t^i, y_t^i)$ are sampled at each clustered point's location. Concurrently, each control point is labeled with its corresponding instance ID. The Gaussian heatmap trajectories, depth maps, and instance maps are concatenated and fed into ControlNet.
    - **Design Motivation**: Ablation studies demonstrate that instance information is the most critical component (without it, FVD degrades from 190 to 228) because the model needs to distinguish which control points belong to the same object; depth information provides auxiliary 3D positional clues.

### Loss & Training

The standard diffusion training objective is used: $\mathcal{L} = \mathbb{E}_{z_t, z^0, t, \epsilon}[\|\epsilon - \epsilon_\theta^c(z_t; t, z^0, c_{\text{traj}})\|^2]$. Based on the SVD model, ControlNet is used to inject control signals. Trained for 200K iterations using AdamW with lr=1e-5 on 16 A100 GPUs, with a batch size of 16 and a resolution of 288×512.

## Key Experimental Results

### Main Results

| Setting | Method | FID↓ | FVD↓ | ObjMC↓ |
|------|------|------|------|--------|
| Multi-Points | DragAnything | 36.04 | 324.95 | 38.86 |
| Multi-Points | DragNUWA 1.5 | 42.34 | 299.96 | 23.12 |
| Multi-Points | **LeviTor** | **25.41** | **190.44** | 25.97 |
| Single-Point | DragAnything | 36.69 | 327.41 | 42.19 |
| Single-Point | DragNUWA 1.5 | 44.82 | 330.17 | 33.03 |
| Single-Point | **LeviTor** | **28.79** | **226.45** | 37.39 |

### Ablation Study

| Depth | Instance | FID↓ | FVD↓ | ObjMC↓ |
|:-----:|:--------:|------|------|--------|
| ✗ | ✗ | 27.83 | 227.58 | 29.82 |
| ✓ | ✗ | 28.04 | 221.29 | 29.13 |
| ✗ | ✓ | 25.45 | 199.44 | 25.40 |
| ✓ | ✓ | **25.41** | **190.44** | 25.97 |

### Key Findings

- Instance information is significantly more critical than depth information: adding Instance drops the FVD from 228 to 199, whereas adding Depth only drops it from 228 to 221. This demonstrates that the model must explicitly know which control points belong to the same object.
- Moving from Single-Point to Multi-Points brings substantial improvements to both LeviTor and DragNUWA, validating that multi-point control better expresses object size variations and occlusions.
- LeviTor performs slightly worse than DragNUWA on the ObjMC metric because LeviTor does not utilize tracking methods to acquire full trajectories, but its FID/FVD metrics vastly outperform all baselines.
- The number of control points requires a trade-off: too few points permit large motion range but can cause distortion, while too many points resemble mask control, limiting the object to translation only.

## Highlights & Insights

- **K-means clustered points as a proxy representation of 3D motion is highly ingenious**—it eliminates the need for explicit 3D annotations. By implicitly encoding depth variations through the gathering and dispersing of points, it conveys sufficient information while leaving generative freedom to the diffusion model. This "loose control" design paradigm can be extended to other generative tasks requiring 3D control.
- **The 3D rendered mask pipeline is the key to user friendliness**—users only need to draw a 2D trajectory and adjust depth values, and the system performs physically correct mask transformations in 3D space, significantly lowering the barrier to 3D input.
- **The model learns 3D control capability during training using only VOS data and depth estimation**, requiring no 3D trajectory annotations, which makes the data acquisition cost exceptionally low.

## Limitations & Future Work

- Limited by the quality of SAM segmentation; objects that are not correctly segmented cannot be controlled.
- Lacks understanding of physics laws and cannot automatically generate plausible motions, relying completely on user-provided trajectories.
- Cannot control internal pose changes of an object (e.g., leg movements during walking) because tracking data was not used during training.
- Based on the SVD base model, the resolution and frame count are constrained; the reconstruction quality of small object faces is suboptimal, and artifacts occur under large motion amplitudes.
- Can be extended to stronger base video models (such as CogVideoX) to improve generation quality and motion ranges.

## Related Work & Insights

- **vs DragAnything**: DragAnything extracts entity representations for entity-level control but is confined to 2D trajectories. It utilizes single-point trajectories with first-frame mask semantic information during training, showing limited improvement when increasing the number of trajectories. LeviTor achieves true 3D control via multi-points and depth.
- **vs DragNUWA**: DragNUWA converts sparse strokes into a dense optical flow space. It performs well on the ObjMC metric but does not support selecting operating regions, and often misinterprets object motion as camera motion. LeviTor circumvents this issue by incorporating masks for all objects.
- **vs 3D-TrajMaster**: 3D-TrajMaster controls multi-entity 3D motion using 6DoF pose sequences, but requires precise 6DoF input, creating a high barrier for users. LeviTor's approach of 2D line drawing combined with depth values is much more practical.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce 3D trajectory control in I2V, with a highly creative control signal design based on K-means clustered points and depth values.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient quantitative and qualitative comparison, with well-designed ablation studies, although quantitative evaluation is restricted to DAVIS.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure and highly intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for 3D control in I2V, offering a significant push to the video generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency](geometry-guided_online_3d_video_synthesis_with_multi-view_temporal_consistency.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] IDOL: Instant Photorealistic 3D Human Creation from a Single Image](idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)
- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
