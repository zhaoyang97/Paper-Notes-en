---
title: >-
  [Paper Note] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation
description: >-
  [ICCV 2025][Video Generation][6D pose control] This paper proposes SynFMC, a synthetic dataset (the first video dataset with complete 6D pose annotations for both camera and objects) and the FMC method, enabling independent or simultaneous 6D pose control of camera and objects in text-to-video generation. The approach produces high-fidelity videos across diverse scenarios and is compatible with multiple personalized T2I models.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "6D pose control"
  - "camera motion"
  - "object motion"
  - "synthetic dataset"
  - "text-to-video"
date: 2026-05-08
content_hash: 22b42446ab296c8a
---

# Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation

**Conference**: ICCV 2025
**arXiv**: [2501.01425](https://arxiv.org/abs/2501.01425)  
**Code**: [https://henghuiding.com/SynFMC/](https://henghuiding.com/SynFMC/)  
**Area**: Video Generation
**Keywords**: video generation, 6D pose control, camera motion, object motion, synthetic dataset, text-to-video

## TL;DR

This paper proposes SynFMC, a synthetic dataset (the first video dataset with complete 6D pose annotations for both camera and objects) and the FMC method, enabling independent or simultaneous 6D pose control of camera and objects in text-to-video generation. The approach produces high-fidelity videos across diverse scenarios and is compatible with multiple personalized T2I models.

## Background & Motivation

**What is the core challenge of motion control?** Precisely controlling dynamic objects and camera motion in video generation is a meaningful yet highly challenging task (analogous to a film director choreographing actor movements and camera trajectories), yet existing methods face two fundamental limitations:

**Lack of datasets with complete 6D pose annotations**:
   - Existing object motion datasets (VideoHD, DragNUWA) provide only 2D image-space trajectories, making it impossible to distinguish "object moving right" from "camera moving left"
   - Datasets for camera motion (RealEstate10K, MVImgNet) primarily cover static scenes and lack dynamic objects
   - A small number of synthetic datasets (360°-Motion, HumanVid-Syn) are either limited to static cameras or human body actions, with insufficient motion diversity

**Lack of methods capable of simultaneously controlling 6D poses of both objects and camera**:
   - CameraCtrl controls only camera motion; Motion-Zero controls only objects without 3D awareness (e.g., orientation)
   - MotionCtrl trains separate modules for camera and object control but cannot achieve synchronized realistic control due to the absence of complete 6D annotations
   - Image-space trajectory methods inherently couple object and camera motion

## Method

### Overall Architecture

The system consists of three core components: the SynFMC dataset construction pipeline, the Camera Motion Controller (CMC), and the Object Motion Controller (OMC), trained using a three-stage strategy.

### SynFMC Dataset

A synthetic dataset built with Unreal Engine, comprising 62K videos organized into four groups:

- **15K static single-object** / **15K static multi-object** (objects fixed, camera may move)
- **16K dynamic single-object** / **16K dynamic multi-object** (both objects and camera may move)

**Object motion design**: Trajectories are designed using Bézier curves, with rotations derived from tangent and normal vectors; control points are constrained according to object speed attributes. Motion types include stationary, horizontal translation, non-horizontal translation, and in-place motion.

**Camera motion design**: Decomposed into three independent dimensions — viewing angle (front/back/left/right/top), distance (zoom in/zoom out/static), and height (up/down/static) — enabling fine-grained combinatorial control.

**Asset annotation**: InternVL combined with human annotation is used to label objects with category, habitat, speed, size, and other attributes, ensuring the plausibility of generated motions.

### FMC Method

Built upon AnimateDiff V3 with a three-stage training procedure:

**Stage 1: Domain LoRA**
LoRA adapters are injected into spatial modules and trained on randomly sampled frames from the synthetic data. The goal is to bridge the domain gap between synthetic and real data. The LoRA weights are discarded at inference to preserve the quality of the base model.

**Stage 2: Camera Motion Controller (CMC)**
Composed of a Camera Encoder (processing Plücker embeddings) and a Camera Adapter (modulating temporal module features). A camera loss $L_{cam}$ is used to emphasize background regions:

$$L_{cam} = E[\mathcal{M}_{bg}\|\varepsilon_{\theta,\theta_c}(\mathbf{z}_t^{1:N}, t, \mathbf{C}_p, \mathcal{C}_{RT}) - \epsilon\|^2 + \lambda_c\|\varepsilon_{\theta,\theta_c}(\cdot) - \epsilon\|^2]$$

where $\mathcal{M}_{bg}$ is the background mask and $\lambda_c = 0.6$. Since background dynamics are solely determined by camera motion, focusing on the background enables more accurate learning of camera control.

**Stage 3: Object Motion Controller (OMC)**
Takes object 6D pose information as input and replicates pose features relative to the camera within the corresponding object regions. A Gaussian blur kernel centered at the object centroid is used instead of precise masks, relieving users from providing accurate segmentation. An object loss $L_{obj}$ emphasizes foreground regions:

$$L_{obj} = E[\mathcal{M}_{fg}\|\varepsilon_{\theta,\theta_c,\theta_o}(\cdot) - \epsilon\|^2 + \lambda_o\|\varepsilon_{\theta,\theta_c,\theta_o}(\cdot) - \epsilon\|^2]$$

where $\lambda_o = 0.3$. The OMC output is multiplied by a coarse mask before being added to spatial features, preventing interference with the background.

### User Interface

Two interaction modes are provided: users directly draw 3D curves to specify trajectories, or specify a motion type and let a rule-based algorithm automatically generate the corresponding trajectory.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | FID↓ | FVD↓ | CLIPSIM↑ | CamTransErr↓ | CamRotErr↓ | ObjTransErr↓ | ObjRotErr↓ |
|--------|------|------|----------|--------------|------------|--------------|------------|
| AnimateDiff | 149.61 | 868.97 | 29.33 | - | - | - | - |
| CameraCtrl | 137.96 | 805.25 | 29.21 | 18.16 | **0.94** | - | - |
| MotionCtrl | 125.52 | 952.31 | 26.83 | **17.84** | 1.11 | 80.66 | 1.77 |
| **FMC (ours)** | 133.42 | 846.51 | **31.01** | 18.12 | 1.03 | **42.25** | **0.96** |

FMC substantially outperforms MotionCtrl in object motion control (ObjTransErr: 42.25 vs. 80.66; ObjRotErr: 0.96 vs. 1.77) while achieving camera control on par with dedicated methods.

### Ablation Study

| Setting | CamTransErr | CamRotErr | ObjTransErr | ObjRotErr |
|---------|-------------|-----------|-------------|-----------|
| MotionCtrl (w/o $\mathcal{C}_{RT}$) | 18.24 | 1.08 | 78.82 | 1.65 |
| MotionCtrl (w/ $\mathcal{C}_{RT}$) | 18.24 | 1.08 | 55.33 | 1.26 |
| FMC (w/o $L_{cam}$) | 20.35 | 1.19 | - | - |
| FMC (w/o $L_{obj}$) | 18.12 | 1.03 | 46.62 | 1.15 |
| **FMC (full)** | **18.12** | **1.03** | **42.25** | **0.96** |

**Key Findings**:
- Training MotionCtrl on SynFMC with camera pose conditioning reduces ObjTransErr from 78.82 to 55.33, demonstrating the value of complete pose annotations
- Removing $L_{cam}$ leads to a significant increase in camera error, as the model tends to move foreground objects to simulate relative motion
- Removing $L_{obj}$ degrades object appearance quality

### User Study

| Method | Quality | Text Similarity | Camera Motion | Object Motion |
|--------|---------|----------------|---------------|---------------|
| CameraCtrl | 0.88 | 0.84 | 0.95 | - |
| MotionCtrl | 0.89 | 0.81 | 0.93 | 0.53 |
| **FMC** | **0.91** | **0.95** | **0.95** | **0.98** |

The margin in object motion scores is substantial (0.98 vs. 0.53), validating the advantage of 3D-aware control.

## Highlights & Insights

1. **Data is the key to disentangled control**: Complete 6D pose annotations enable the model to learn to distinguish global (camera) from local (object) dynamics — something image-space trajectories fundamentally cannot achieve.
2. **Elegant design of the rule-based generation algorithm**: Decomposing camera motion into three independent dimensions (angle/distance/height) supports controllable generation of complex cinematic shots.
3. **Practical Domain LoRA strategy**: Used during training and discarded at inference, this approach leverages synthetic data for motion control learning while preserving visual realism.
4. **Coarse masks as a substitute for precise segmentation**: The Gaussian blur kernel design lowers the barrier for user input, making the system highly engineering-friendly.

## Limitations & Future Work

- Control capability for complex multi-object motion remains limited
- Object motion evaluation metrics require improvement (current approach relies on depth estimation to infer global positions)
- The method is based on a U-Net architecture (AnimateDiff V3) and has not been extended to more recent DiT architectures
- The absence of reference image input prevents customization of motion videos for specific subjects

## Related Work & Insights

- **CameraCtrl**: A representative method for camera-only motion control
- **MotionCtrl**: The most closely related work, training separate camera and object modules but unable to synchronize them effectively
- **3DTrajMaster**: Provides object 6D poses but is limited to static cameras
- **Insight**: The synthetic data + domain adaptation paradigm is generalizable to other video generation tasks requiring precise physical annotations (e.g., hand manipulation, robot planning)

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — First dataset and method to provide complete 6D pose annotations for both camera and objects simultaneously
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Independent/joint control comparison, ablation, user study, multi-style adaptation
- **Value**: ⭐⭐⭐⭐ — Clear application scenarios in filmmaking, gaming, and AR
- **Writing Quality**: ⭐⭐⭐⭐ — Rich figures and tables, clear method description

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](../../CVPR2025/video_generation/motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] Dynamic Camera Poses and Where to Find Them](../../CVPR2025/video_generation/dynamic_camera_poses_and_where_to_find_them.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)

</div>

<!-- RELATED:END -->
