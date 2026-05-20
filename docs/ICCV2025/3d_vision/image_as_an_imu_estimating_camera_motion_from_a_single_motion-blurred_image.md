---
title: >-
  [Paper Note] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image
description: >-
  [ICCV 2025][3D Vision][motion blur] This paper reframes motion blur from an "unwanted artifact" into a "valuable motion cue." By predicting a dense optical flow field and a monocular depth map from a single blurred image…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "motion blur"
  - "camera motion estimation"
  - "6DoF velocity estimation"
  - "single-image motion estimation"
  - "IMU alternative"
date: 2026-05-08
content_hash: 35af3db7c115b480
---

# Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image

**Conference**: ICCV 2025
**arXiv**: [2503.17358](https://arxiv.org/abs/2503.17358)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: motion blur, camera motion estimation, 6DoF velocity estimation, single-image motion estimation, IMU alternative

## TL;DR

This paper reframes motion blur from an "unwanted artifact" into a "valuable motion cue." By predicting a dense optical flow field and a monocular depth map from a single blurred image, and subsequently recovering the camera's 6DoF instantaneous velocity via a differentiable least-squares solver, the method achieves motion estimation accuracy comparable to or surpassing that of an IMU, with real-time performance at 30 FPS.

## Background & Motivation

Camera motion estimation is fundamental to 3D reconstruction, SLAM, and VR/AR. Conventional VO/SfM methods assume the camera is approximately static during exposure, treating each frame as a scene snapshot and computing relative pose via inter-frame feature matching. Under rapid motion, however, this assumption completely breaks down—motion blur severely degrades feature matching, causing methods such as COLMAP and ORB-SLAM to fail entirely.

**Limitations of existing approaches**:
- **Discarding blurred frames**: Loses information; very few usable frames remain under sustained rapid motion.
- **Incorporating an IMU**: Increases hardware cost, complicates sensor synchronization, and suffers from integration drift—velocity estimates can deviate noticeably after as little as 20 seconds.
- **Learning-based methods** (DUSt3R, MASt3R, etc.): More robust under sparse viewpoints, but still require multi-frame input and exhibit degraded feature matching under severe motion blur.

**A paradigm-shifting perspective**: Motion blur encodes the camera's motion directly during the exposure interval—the direction and length of blur streaks reflect the camera's direction and speed of motion. Decoding these blur trajectories enables recovery of the camera's instantaneous motion from a single image. In essence, this allows the camera to serve as an IMU-like motion sensor—without additional hardware, without drift, and yielding velocity directly rather than acceleration that must be integrated.

**Key insight**: A motion-blurred image can be viewed as the superposition of multiple "virtual images" captured during the exposure interval. The blur trajectories provide "virtual correspondences" between the first and last virtual images, which are equivalent in nature to an optical flow field. Combined with depth estimation, these pixel displacements can be lifted to recover 3D camera motion.

## Method

### Overall Architecture

The pipeline consists of two stages: (1) a **network stage** that predicts a motion flow field $\mathcal{F}$ and a monocular depth map $\mathcal{D}$ from a single blurred image; and (2) a **solver stage** that substitutes the flow and depth into the motion-field equations and applies a differentiable least-squares solver to compute the camera's 6DoF instantaneous velocity $(v_x, v_y, v_z, \omega_x, \omega_y, \omega_z)$.

The method requires only a single motion-blurred image (plus known focal length and exposure time) and outputs the instantaneous linear and angular velocity of the camera during the exposure interval.

### Key Designs

1. **Optical Flow and Depth Prediction Network**

    - **Function**: Simultaneously predicts pixel-level motion flow and metric depth from a blurred image.
    - **Mechanism**: A shared SegNeXt backbone encoder feeds into two separate decoders, producing optical flow $\mathcal{F} \in \mathbb{R}^{2 \times H \times W}$ and depth $\mathcal{D} \in \mathbb{R}^{1 \times H \times W}$. The flow is defined as the pixel displacement from the first to the last virtual frame.
    - **Key design — redirection function** $h_f$: Because the direction of motion cannot be determined from a blurred image alone (leftward and rightward motion produce identical horizontal blur), a redirection function is used during training to select the ground-truth label consistent with the predicted direction:
    $h_f(\hat{\mathcal{F}}_{fw}, \hat{\mathcal{F}}_{bw}; \mathcal{F}) = \begin{cases} \hat{\mathcal{F}}_{fw} & \text{if } \langle\hat{\mathcal{F}}_{fw}, \mathcal{F}\rangle > \langle\hat{\mathcal{F}}_{bw}, \mathcal{F}\rangle \\ \hat{\mathcal{F}}_{bw} & \text{otherwise} \end{cases}$
    - **Training loss**: $\mathcal{L}_1 = \lambda_F\|\mathcal{F} - h_f(\hat{\mathcal{F}}_{fw}, \hat{\mathcal{F}}_{bw})\| + \lambda_D\|\mathcal{D} - \hat{\mathcal{D}}\|$

2. **Differentiable Velocity Solver**

    - **Function**: Recovers camera translation and rotation from the flow field and depth map.
    - **Mechanism**: Classical motion-field equations decompose the optical flow at each pixel into translational and rotational components:
    $F_x = \frac{t_z p_x - t_x f}{d} - \theta_y f + \theta_z p_y + \frac{\theta_x p_x p_y}{f} - \frac{\theta_y(p_x)^2}{f}$
    $F_y = \frac{t_z p_y - t_y f}{d} + \theta_x f - \theta_z p_x - \frac{\theta_y p_x p_y}{f} + \frac{\theta_x(p_y)^2}{f}$
    All pixels are assembled into an overdetermined linear system $\bm{A}\bm{x}=\bm{b}$, solved in the least-squares sense: $\bm{x} = (\bm{A}^\top\bm{A})^{-1}\bm{A}^\top\bm{b}$
    - **Design motivation**: The least-squares solver is fully differentiable, enabling end-to-end training of the entire network with pose supervision. The end-to-end loss is:
    $\mathcal{L}_2 = \lambda_R\|R - h_p(\hat{R})\|_2 + \lambda_t\|\bm{t} - h_p(\hat{\bm{t}})\|_2 + \mathcal{L}_1$

3. **Direction Disambiguation**

    - **Function**: Resolves the 180° directional ambiguity inherent to motion blur (leftward and rightward motion yield identical blur patterns).
    - **Mechanism**: Neighboring frames in the video are used by warping the current frame with both the forward and backward flow fields, then comparing the photometric error against the adjacent frames to determine the correct direction:
    $e_{fw} = \mathcal{P}(I_{i+1}, I'_{i,fw}) + \mathcal{P}(I_{i-1}, I'_{i,bw})$
    The direction with the smaller photometric error is selected as the final motion direction.
    - **Design motivation**: This is an inference-time post-processing heuristic that exploits video temporal information to resolve the inherent 180° ambiguity.

### Loss & Training

Three-stage training:
1. Flow and depth decoders are trained on synthetic data only (no pose supervision), with batch size 32.
2. End-to-end training with pose supervision is applied, with batch size 8 for 300K steps.
3. Fine-tuning for 10K steps on real-world motion-blurred data via the differentiable pipeline.

The synthetic dataset is generated from 150 ScanNet++v2 sequences, yielding approximately 120K training samples. The RIFE frame interpolation network is used to generate virtual frames between real frames, which are then averaged in linear color space to produce synthetic blurred images. Metric depth ground truth is obtained by upsampling low-resolution ARKit depth with PromptDA.

## Key Experimental Results

### Main Results (Angular velocity RMSE across 4 real-world scenes, rad/s)

| Method | Input | billiards | commonroom | dining | office | Mean |
|--------|-------|-----------|------------|--------|--------|------|
| COLMAP | Multi-frame | ×failed | ×failed | ×failed | ×failed | - |
| MASt3R | Two-frame | 5.30/2.85/4.45 | 3.70/3.75/3.26 | 2.36/0.84/1.67 | 4.78/3.03/6.21 | 4.04/2.62/3.90 |
| DROID-SLAM | Multi-frame | 5.39/3.33/5.31 | 3.01/5.89/3.57 | 2.92/1.20/1.98 | 6.33/4.90/5.56 | 4.41/3.83/4.10 |
| **Ours** | **Single frame** | **1.31/0.87/1.60** | **0.93/0.88/1.04** | **0.87/0.50/1.33** | **1.76/1.38/3.08** | **1.22/0.91/1.76** |

### Ablation Study (Translational velocity RMSE, m/s)

| Method | Input | Mean $v_x / v_y / v_z$ | Notes |
|--------|-------|------------------------|-------|
| COLMAP (SIFT) | Multi-frame | Complete failure | Motion blur causes feature matching collapse |
| COLMAP (D+LG) | Multi-frame | — (partial failure) | Learned features slightly more robust but still unstable |
| MASt3R | Two-frame | 1.60/1.54/2.17 | Learning-based; relatively robust under sparse viewpoints |
| DROID-SLAM | Multi-frame | 2.23/1.63/1.51 | Flow-based SLAM |
| **Ours** | **Single frame** | **1.11/1.03/0.92** | 24% reduction in translational velocity RMSE |
| Zero-velocity baseline | — | 2.01/1.61/1.24 | Lower bound assuming a stationary camera |

### Key Findings

- COLMAP fails completely on sequences with severe motion blur, unable to reconstruct even coarse poses, underscoring the significance of the problem.
- Using only a single frame as input, the proposed method outperforms all multi-frame baselines: angular velocity RMSE is reduced by 31% and translational velocity RMSE by 24%.
- Real-time inference at 30 FPS on an RTX 3090—more than 10× faster than MASt3R (2.83 FPS).
- Compared against a real IMU: IMU integration exhibits noticeable drift after 20 seconds, whereas the proposed method remains stable throughout with no drift.
- Successful estimation of end-effector velocity in a robotic arm experiment demonstrates practical applicability.

## Highlights & Insights

- **A fundamental shift in perspective**: Motion blur is transformed from an adversary into an ally—frames that would ordinarily be discarded become a motion sensor. This "turning a weakness into a strength" philosophy is highly inspiring.
- **Fully differentiable pipeline**: The entire process, from flow/depth prediction to least-squares solving, is differentiable, enabling end-to-end fine-tuning on real data to close the synthetic-to-real gap.
- **Drift-free by design**: Unlike IMU integration drift, each frame is estimated independently, making the method inherently drift-free.
- **Data synthesis pipeline**: The approach demonstrates how to generate blur training data with ground-truth labels from standard visual datasets (ScanNet++v2), offering a methodology worth adopting in related work.
- **Elegant direction disambiguation**: A simple heuristic leveraging video temporal context and photometric consistency resolves the 180° ambiguity effectively.

## Limitations & Future Work

- The method assumes a rigid scene and does not account for rolling shutter or non-uniform motion within the exposure interval (though experiments suggest some robustness to these conditions).
- Training data is limited to indoor scenes; generalization to outdoor environments has not been validated.
- Known exposure time and focal length are required.
- A 180° directional ambiguity remains; resolving it requires neighboring frames, making pure single-frame disambiguation infeasible.
- Translational estimation accuracy may be bounded by the precision of depth estimation.

## Related Work & Insights

- **Klein & Murray (2005)**: Pioneering work on estimating rotation from a single blurred image, but restricted to pure rotation.
- **MBA-VO (Liu et al., 2021)**: A blur-aware VO method, but requires two frames and an explicit deblurring step.
- **MASt3R/DUSt3R**: Current state-of-the-art in learning-based 3D vision, but not specifically designed for blurred scenarios.
- **Insight**: Motion-blur-based estimation can serve as an alternative or complementary source to IMU data in sensor fusion, particularly well-suited for settings where IMU installation is impractical.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The paradigm of extracting motion from blur represents a genuine conceptual shift)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Real-world evaluation, IMU comparison, and robotic application included, though dataset scale is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Motivation is clearly articulated; the pipeline is compact; figures and tables are intuitive)
- Value: ⭐⭐⭐⭐⭐ (Real-time, drift-free, single-frame—significant practical value for SLAM and VR)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [\[ICCV 2025\] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)

</div>

<!-- RELATED:END -->
