---
title: >-
  [Paper Note] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control
description: >-
  [ICCV 2025][Video Generation][Camera Control] RealCam-I2V integrates monocular metric depth estimation to construct 3D scenes for metric-scale-aligned training, provides an interactive 3D scene trajectory drawing interface, and introduces a scene-constrained noise shaping mechanism, addressing the scale inconsistency and real-world usability issues inherent in existing trajectory-guided I2V methods.
tags:
  - ICCV 2025
  - Video Generation
  - Camera Control
  - Metric Depth
  - Video Diffusion
  - Noise Shaping
  - I2V
date: 2026-05-08
content_hash: f10f45a2225fd029
---

# RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control

**Conference**: ICCV 2025
**arXiv**: [2502.10059](https://arxiv.org/abs/2502.10059)
**Code**: [https://zgctroy.github.io/RealCam-I2V](https://zgctroy.github.io/RealCam-I2V)
**Area**: 3D Vision / Controllable Video Generation / Camera Control
**Keywords**: Camera Control, Metric Depth, Video Diffusion, Noise Shaping, I2V

## TL;DR
RealCam-I2V integrates monocular metric depth estimation to construct 3D scenes for metric-scale-aligned training, provides an interactive 3D scene trajectory drawing interface, and introduces a scene-constrained noise shaping mechanism, addressing the scale inconsistency and real-world usability issues inherent in existing trajectory-guided I2V methods.

## Background & Motivation

Camera trajectory-guided I2V generation offers more precise camera control than text-based approaches. However, existing methods (MotionCtrl, CameraCtrl, CamI2V) suffer from two **core problems**:

**Scale inconsistency**: Camera parameters in training data are derived from COLMAP's relative-scale reconstruction, where the scale varies across different videos. This means that identical translation parameters correspond to entirely different camera motion magnitudes across scenes, preventing the model from learning physically consistent camera motion patterns.

**Poor real-world usability**: When faced with arbitrary real-world images, users have no knowledge of scene depth or scale and therefore cannot provide accurate camera extrinsics. Even experienced photographers find it difficult to supply reasonable 6DoF trajectory parameters from scratch.

**Core Insight**: Introducing metric depth estimation as a preprocessing step simultaneously resolves both problems — providing a unified metric scale for training and an intuitive 3D interactive interface for inference.

## Method

### Overall Architecture
- **Training**: Align camera parameters from relative scale to metric scale.
- **Inference**: Construct a point cloud using metric depth → user interactively draws trajectories in the 3D scene → scene-constrained noise shaping enhances control.

### Metric Scene-scale Alignment

For each training video:
1. Predict the metric depth map of the reference frame using Depth Anything V2 (metric version): $D(u,v) = f_{depth}(I)$
2. Back-project the depth map into 3D space to construct a metric point cloud.
3. Align the metric point cloud with the relative-scale point cloud from COLMAP reconstruction to obtain a scaling factor $\alpha$.
4. Convert relative translations to metric translations:

$$c_{\text{cam}}^{\text{metric}} = \begin{bmatrix} R & \alpha \cdot T \\ 0 & 1 \end{bmatrix}$$

This ensures that camera parameters across videos share a consistent physical meaning — identical translation values correspond to the same physical distance in all scenes.

### Interactive 3D Scene Inference

At inference time, the user provides a reference image, and the system automatically:
1. Estimates metric depth → back-projects into a 3D point cloud.
2. The user drags and draws a camera trajectory within the 3D point cloud scene.
3. A trajectory preview video is rendered (without running the diffusion model, providing real-time feedback).
4. Video generation is triggered once the user is satisfied.

### Scene-constrained Noise Shaping

During the **high-noise stage** ($t > 0.9$) of diffusion denoising, the preview video guides generation:

$$z_t = m \cdot (\alpha_t z_{\text{preview}} + \sigma_t \epsilon) + (1-m) \cdot z_t$$

where $m$ identifies selected reference pixels. Key details:
- Only pixels visible under the current viewpoint are selected.
- Edge pixels whose neighborhoods contain invisible pixels are excluded (to avoid depth estimation errors).
- $\epsilon$ is resampled at each timestep (to prevent fixed noise from overwriting useful information).
- Applied only during the high-noise stage ($t > 0.9$); the conditional model takes over in the low-noise stage to preserve dynamic content generation capability.

### Training Details
- Built upon DynamiCrafter as the I2V base model.
- Base model and depth predictor parameters are frozen; only newly added modules are trained.
- RealEstate10K dataset: 58K training / 6K test splits.
- Adam optimizer, learning rate $1\times10^{-4}$, mixed precision fp16 + ZeRO-1.

## Key Experimental Results

### Main Results: Comparison with SOTA Methods

| Method | RotErr ↓ | TransErr (Relative) ↓ | TransErr (Metric) ↓ | CamMC (Relative) ↓ | FVD (VideoGPT) ↓ |
|---|---|---|---|---|---|
| DynamiCrafter | 3.34 | 9.80 | 14.14 | 15.73 | 106.02 |
| MotionCtrl | 1.05 | 2.29 | 6.82 | 7.23 | 70.29 |
| CameraCtrl | 0.74 | 1.76 | 5.51 | 5.76 | 69.20 |
| CamI2V | 0.41 | 1.34 | 3.29 | 3.42 | 62.44 |
| **RealCam-I2V** | **0.39** | **1.29** | **2.23** | **2.36** | **53.72** |

Metric-scale error is reduced by over 32%; FVD improves by 14.8%.

### Ablation Study: Effect of MSA and SNS

| Method | MSA | SNS | TransErr (Metric) ↓ | FVD (VideoGPT) ↓ |
|---|---|---|---|---|
| CamI2V baseline | - | - | 3.29 | 62.44 |
| +MSA | ✓ | - | 2.65 | 60.52 |
| **+MSA+SNS (RealCam-I2V)** | ✓ | ✓ | **2.23** | **53.72** |

**Key Findings**:
- MSA alone reduces metric error by 20%, validating the importance of scale alignment.
- SNS further reduces error by 16% and substantially improves visual quality (FVD reduced by 11%).
- MSA generalizes effectively to all baseline models (improvements observed on both MotionCtrl and CameraCtrl).

## Highlights & Insights
1. **Addresses a fundamental pain point**: Scale inconsistency is a root-level problem in camera-controlled video generation; the proposed solution is both simple and effective.
2. **Single-pass generation vs. iterative refinement**: By decoupling camera adjustment and video generation via 3D preview, costly multi-round diffusion sampling is avoided.
3. **Plug-and-play**: MSA and SNS can be seamlessly integrated into existing I2V base models as drop-in modules.
4. **Extended applications**: Supports looping video generation, generative frame interpolation, and smooth scene transitions.

## Limitations & Future Work
- Metric depth estimation may be insufficiently accurate for outdoor or large-scale scenes.
- Scene-constrained noise shaping may inhibit dynamic content generation in highly dynamic scenes.
- Training data relative poses depend on COLMAP; videos where COLMAP fails are discarded.
- The 3D scene is constructed from single-frame depth estimation only, leaving occluded regions without geometric information.

## Related Work & Insights
- Trajectory control: MotionCtrl, CameraCtrl, and CamI2V operate at relative scale.
- Metric depth: 4DiM and AC3D also employ metric depth but via different approaches.
- Training-free methods: CamTrol renders static point clouds but lacks training-time alignment.

## Rating
- **Novelty**: ★★★★☆ — The combination of metric scale alignment and noise shaping is both effective and novel.
- **Practicality**: ★★★★★ — The interactive 3D interface substantially lowers the barrier for end users.
- **Experimental Thoroughness**: ★★★★☆ — Ablations are thorough; cross-baseline generalization validates universality.
- **Writing Quality**: ★★★★☆ — Problem analysis is in-depth and method motivation is clearly articulated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)

</div>

<!-- RELATED:END -->
