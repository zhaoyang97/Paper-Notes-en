---
title: >-
  [Paper Note] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation
description: >-
  [CVPR 2026][Video Generation][Video diffusion models] SymphoMotion is a unified motion control framework that simultaneously and precisely controls camera motion and object 3D trajectories in video generation via two mechanisms — Camera Trajectory Control (CTC) and Object Dynamics Control (ODC) — alongside a large-scale real-world jointly annotated dataset, RealCOD-25K, containing 25K samples.
tags:
  - CVPR 2026
  - Video Generation
  - Video diffusion models
  - camera control
  - object motion control
  - 3D awareness
  - motion decoupling
date: 2026-05-08
content_hash: d45c35fe49a6421f
---

# SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation

**Conference**: CVPR 2026
**arXiv**: [2604.03723](https://arxiv.org/abs/2604.03723)
**Code**: [Project Page](https://grenoble-zhang.github.io/SymphoMotion/)
**Area**: Video Generation / Motion Control
**Keywords**: Video diffusion models, camera control, object motion control, 3D awareness, motion decoupling

## TL;DR

SymphoMotion is a unified motion control framework that simultaneously and precisely controls camera motion and object 3D trajectories in video generation via two mechanisms — Camera Trajectory Control (CTC) and Object Dynamics Control (ODC) — alongside a large-scale real-world jointly annotated dataset, RealCOD-25K, containing 25K samples.

## Background & Motivation

**Background**: Precise control of motion dynamics in video generation is receiving increasing attention. Camera control methods (CameraCtrl, Uni3C, etc.) regulate viewpoint changes by injecting camera parameters, but are limited to static or quasi-static scenes. Object control methods (TrackGo, MagicMotion, etc.) rely on 2D motion cues (bounding boxes, optical flow, keypoints), yet these image-plane representations cannot distinguish true object motion from camera-induced parallax.

**Limitations of Prior Work**: (1) Single-modality motion control methods are mutually incompatible — camera control methods degrade under significant foreground dynamics, while object control methods become unreliable under camera motion. (2) Recent joint control methods (e.g., MotionPrompting, ATI) mix camera parallax and object motion within the same 2D motion field, leading to supervisory ambiguity — different 3D motions can produce similar image-plane projections, especially in scenes with large depth variation. (3) Although MotionCtrl separates the two motion branches, object control remains confined to 2D image space. (4) FMC uses a 6D pose representation but relies on synthetic data and requires full 6-DoF input, limiting its practicality.

**Key Challenge**: Camera motion produces global parallax and viewpoint changes, while objects follow independent 3D paths — both are highly coupled in the 2D image plane, making disentanglement difficult.

**Goal**: How can a single model unify and decouple the control of camera trajectories and object 3D dynamics such that the two are spatially consistent and non-interfering?

**Key Insight**: The authors argue that the key lies in introducing 3D awareness: augmenting camera control with structural awareness via Plücker embeddings and point-cloud geometry priors, and endowing object control with depth awareness via 2D visual guidance combined with 3D trajectory embeddings. Concurrently, the paper constructs the first large-scale real-world dataset with joint annotations of camera poses and object 3D trajectories.

**Core Idea**: 2D visual guidance provides image-plane anchors, and 3D trajectory embeddings provide depth-aware motion supervision — together enabling decoupled joint control of camera and object motion within a unified framework.

## Method

### Overall Architecture

SymphoMotion is built upon the pretrained Wan-I2V video diffusion model. At inference time it accepts four inputs: a reference image, a text prompt, a camera trajectory sequence, and object 3D motion trajectories. The framework comprises two complementary mechanisms: CTC injects 3D geometric priors and camera motion information via a Viewpoint Control Module; ODC manipulates objects via an Object Motion Module that combines 2D visual guidance with 3D trajectory conditioning. The two mechanisms are injected into the diffusion model via ControlNet and cross-attention, respectively.

### Key Designs

1. **Camera Trajectory Control (CTC) and 3D Geometric Priors**:

    - **Function**: Achieve precise and stable viewpoint transition control.
    - **Mechanism**: Given a reference image, Depth-Pro is first used to estimate a point cloud, camera intrinsics, and pose $C^f$. The point cloud is rendered from the target camera pose sequence $\{C^1,\ldots,C^N\}$ to obtain a geometry-aware frame sequence $\mathcal{V}$. Two encoders process the inputs separately: a camera encoder encodes Plücker embeddings into a motion representation $c_{cam}$, and a Wan encoder extracts geometry-aware features $c_{pcd}$ from the rendered frames. $c_{pcd}$ is concatenated with the noisy latent $z_t$, added to $c_{cam}$, and fed into the Viewpoint Control Module implemented as a ControlNet.
    - **Design Motivation**: Pure Plücker embeddings only encode camera pose and lack scene 3D structure information. Incorporating point-cloud rendered frames provides complementary spatial structural cues, helping the model maintain geometric consistency under viewpoint changes.

2. **Object Dynamics Control (ODC) — 2D Visual Guidance**:

    - **Function**: Provide explicit spatial anchors in the image plane for object motion.
    - **Mechanism**: Each object's 3D trajectory $P_i$ is projected onto the image plane using the target camera poses to obtain a 2D trajectory $P_{2D_i}$. Per-frame bounding boxes are fitted from the projected points and rendered directly onto the point-cloud frames $\mathcal{V}$. By overlaying motion boxes on the rendered inputs rather than encoding them solely in latent space, the model receives strong visual cues to track each object's image-plane motion path.
    - **Design Motivation**: 2D anchors establish expected position constraints for objects in the image plane, compensating for potential projection inaccuracies of pure 3D trajectories and enhancing training stability.

3. **Object Dynamics Control (ODC) — 3D Trajectory Conditioning**:

    - **Function**: Provide depth-aware object motion guidance.
    - **Mechanism**: Each object's 3D trajectory $P_i \in \mathbb{R}^{N \times N_p \times 3}$ is first transformed into the reference camera coordinate frame, then encoded into a latent embedding via linear projection and a temporal downsampler. Simultaneously, a frozen language encoder encodes entity prompts $y_i$ into semantic embeddings. The two embeddings are added element-wise to form a motion-aware representation $c_{obj}$. 3D motion information is injected into the diffusion model by adding a cross-attention layer over $c_{obj}$ in each transformer block: $Z_i' = Z_i + \text{CrossAttn}(Q=Z_i, K=c_{obj}, V=c_{obj})$.
    - **Design Motivation**: 3D trajectories encode object motion paths in real-world space, independent of camera viewpoint, enabling object motion control to remain spatially consistent even under camera motion.

### Loss & Training

The flow matching training objective is: $\min_\theta \mathbb{E}[\|v_\theta(z_t, t, c_y, c_f, \phi_\theta(c_{cam}, c_{pcd}), \psi_\theta(P, y)) - v_t\|^2]$. A two-stage training strategy is adopted: CTC is trained first to learn camera control, then ODC is trained with CTC frozen to learn object motion control. The base model Wan-I2V remains frozen throughout. Training is conducted on 32 H100 GPUs with batch size 32, on 81-frame video sequences at 832×480 resolution. The AdamW optimizer is used, with a linear warm-up of the learning rate to $1 \times 10^{-5}$ over the first 400 steps.

## Key Experimental Results

### Main Results

| Method | FID↓ | FVD↓ | CLIPSIM↑ | CamTransErr↓ | CamRotErr↓ | Box-IoU↑ |
|--------|------|------|----------|-------------|-----------|----------|
| CameraCtrl | 196.84 | 1019.49 | 0.29 | 0.68 | 0.12 | – |
| ViewCrafter | 303.83 | 1690.73 | 0.28 | 0.80 | 0.21 | – |
| Uni3C | 86.66 | 404.21 | 0.31 | 0.44 | 0.06 | – |
| MotionCtrl | 182.15 | 738.41 | 0.30 | 0.83 | 0.23 | 31.42 |
| **SymphoMotion** | **70.47** | **332.50** | **0.31** | **0.37** | **0.05** | **61.88** |

### Ablation Study

| Setting | FVD↓ | CamTransErr↓ | CamRotErr↓ | Box-IoU↑ |
|---------|------|-------------|-----------|----------|
| w/o point-cloud prior | 330.64 | 0.46 | 0.07 | 56.74 |
| w/o 2D bounding boxes | 337.14 | 0.36 | 0.06 | 54.32 |
| w/o 3D trajectories | 343.80 | 0.36 | 0.06 | 52.16 |
| **SymphoMotion** | **332.50** | **0.37** | **0.05** | **61.88** |

### Key Findings

- FID 70.47 vs. second-best Uni3C 86.66, indicating superior visual quality; FVD 332.50 vs. Uni3C 404.21, indicating better temporal consistency.
- Box-IoU 61.88 vs. MotionCtrl 31.42 — object trajectory accuracy nearly doubles.
- Camera control precision is comparable to dedicated camera control methods (Uni3C): CamTransErr 0.37 vs. 0.44, CamRotErr 0.05 vs. 0.06.
- Ablations show 3D trajectories contribute most to Box-IoU (+9.72), with 2D boxes and point-cloud priors also providing clear improvements.
- In user studies, SymphoMotion comprehensively outperforms baselines on visual quality (4.87/5), object motion (4.58/5), and other dimensions.

## Highlights & Insights

- **Balance between decoupling and unification**: CTC and ODC interact with the diffusion model as independent mechanisms via different injection pathways (ControlNet vs. cross-attention), achieving decoupled control while sharing the same diffusion backbone to ensure overall consistency.
- **Elegant 2D+3D complementary design**: In object control, 2D bounding boxes provide image-plane anchor constraints for localization, while 3D trajectories provide depth-aware spatial guidance — the two respectively address "where" and "how far."
- **Complete dataset construction pipeline**: Starting from millions of videos, a full pipeline of aesthetic filtering → motion filtering → VLM denoising → 120 person-hours of manual review → automatic annotation is employed, with high reproducibility.
- **User-friendly interaction system**: Users select objects via SAM2 → automatic 3D detection boxes → drag-and-edit 3D trajectories, lowering the barrier to 3D motion specification.

## Limitations & Future Work

- The two-stage training strategy may result in insufficient coordination between CTC and ODC; end-to-end joint training could potentially yield further improvements.
- RealCOD-25K annotations rely on multiple perception models (SegAnyMo, SpatialTrackerV2, Depth Anything V2, etc.), whose accuracy is an upper bound on annotation quality.
- Training and evaluation are conducted only at 832×480 resolution with 81 frames; capability at higher resolutions and longer durations remains unverified.
- Object 3D trajectories are represented as position sequences of sampled points, which cannot encode object rotation or deformation.
- Compared to MotionCtrl's plug-and-play design, reliance on Depth-Pro for point-cloud reconstruction introduces additional computational overhead at inference time.

## Related Work & Insights

- CameraCtrl [He et al.] pioneered the paradigm of injecting camera control into video diffusion models; the present work builds on this by introducing 3D geometric priors.
- MotionCtrl [Wang et al.] was the first to attempt decoupled control of camera and object motion, but object control was limited to 2D — SymphoMotion extends this to 3D.
- ViewCrafter [Yu et al.]'s approach of using point clouds as 3D priors is adopted and refined in SymphoMotion's CTC module.
- RealCOD-25K fills the gap in real-world datasets with joint camera and object motion annotations, providing lasting value for subsequent unified motion control research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified framework and complementary 2D+3D object control design are novel, though individual components (ControlNet injection, cross-attention conditioning) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive quantitative comparisons (6 metrics), ablation studies validating each component individually, and a complete user study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, high-quality figures, and a tight logical chain from motivation to method.
- **Value**: ⭐⭐⭐⭐⭐ — Full contributions of dataset, method, and interaction system set a new standard for controllable video generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](../../ICLR2026/video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](../../ICCV2025/video_generation/free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)

<!-- RELATED:END -->
