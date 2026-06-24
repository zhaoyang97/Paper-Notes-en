---
title: >-
  [Paper Note] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion
description: >-
  [CVPR 2025][Video Generation][Trajectory-guided video generation] This work proposes PoseTraj, a pose-aware trajectory-controlled video diffusion model. By leveraging a two-stage pose-aware pre-training (utilizing the synthetic dataset PoseTraj-10K and 3D bounding box intermediate supervision) and camera motion decoupling fine-tuning, PoseTraj achieves 3D-aligned rotational motion video generation from 2D trajectories.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Trajectory-guided video generation"
  - "6D pose awareness"
  - "synthetic data pre-training"
  - "camera motion decoupling"
  - "3D bounding box supervision"
date: 2026-05-08
content_hash: 6223c542a2ba26dc
---

# PoseTraj: Pose-Aware Trajectory Control in Video Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2503.16068](https://arxiv.org/abs/2503.16068)  
**Code**: [Project Homepage](https://robingg1.github.io/Pose-Traj/)  
**Area**: Object Detection/Video Generation  
**Keywords**: Trajectory-guided video generation, 6D pose awareness, synthetic data pre-training, camera motion decoupling, 3D bounding box supervision

## TL;DR

This work proposes PoseTraj, a pose-aware trajectory-controlled video diffusion model. By leveraging a two-stage pose-aware pre-training (utilizing the synthetic dataset PoseTraj-10K and 3D bounding box intermediate supervision) and camera motion decoupling fine-tuning, PoseTraj achieves 3D-aligned rotational motion video generation from 2D trajectories.

## Background & Motivation

- Trajectory-guided video generation has attracted widespread attention due to its interactive friendliness, but existing models struggle to handle object motion involving 6D pose changes, especially large-angle rotations.
- Methods like DragNUWA and DragAnything only constrain object-following trajectories in the 2D image space, ignoring changes in object pose.
- Rotational trajectories are scarce and difficult to automatically annotate in real video data, leading to a lack of 3D understanding in models.
- Inferring potential rotation from 2D trajectories is inherently an ill-posed problem.
- Camera motion and object motion are coupled in real videos, making them difficult to decouple accurately.
- Existing methods are prone to object entity collapse under large-angle rotation scenarios.
- Existing evaluation datasets lack precise 3D annotations containing rotational trajectories.
- Pre-trained models such as SVD possess insufficient understanding of rotational motion.

## Method

### Overall Architecture

PoseTraj is built upon Stable Video Diffusion (SVD) and comprises three training stages: **Stage 1** performs 3D bounding box-guided pre-training on the synthetic dataset PoseTraj-10K (simultaneously generating the object and its 3D bbox); **Stage 2** removes the bbox supervision to focus on object appearance details; **Stage 3** performs camera motion decoupling fine-tuning on real videos (VIPSeg). During inference, users freely draw trajectories, and the model generates pose-aware videos. The core module, Traj-ControlNet, is a trainable copy of the SVD encoder blocks that receives trajectory features and predicts residual features.

### Key Designs

**Design 1: Two-Stage Pose-Aware Pre-training**
- **Function**: To enable the model to perceive potential 3D rotation variations from 2D trajectories.
- **Mechanism**: In the first stage, the 3D bounding box is rendered into the pixel space and generated alongside the object, where the bbox provides position and pose information as intermediate supervision. In the second stage, the bbox is removed, and the model is fine-tuned to focus on appearance details. This employs "injection-by-reconstruction," meaning the bbox is directly used as a reconstruction target instead of an input condition.
- **Design Motivation**: Directly regressing 3D parameters is difficult, whereas generating the bbox into the image space leverages the pixel-level reconstruction capability of the diffusion model to enhance continuous 3D perception. Furthermore, the second stage can easily replace the reconstruction target to remove the bbox, preventing signal mismatch during inference.

**Design 2: PoseTraj-10K Synthetic Dataset**
- **Function**: To provide large-scale training data containing rotational trajectories and precise 3D bounding box annotations.
- **Mechanism**: 2,000 high-quality 3D models are sampled from Objaverse (filtered via GPT-4V + manual selection). In Blender, 5 random rotational trajectories are generated for each model, rendering 10,000 videos (14 frames, 5 fps, $320\times576$).
- **Design Motivation**: Rotational motion in real-world videos is rare and difficult to annotate with 6D poses; synthetic data provides precise trajectories and 3D bbox annotations, avoiding camera motion interference.

**Design 3: Camera Motion Decoupling Fine-Tuning + Spatial Promotion Loss**
- **Function**: To enhance the generalization capability from synthetic to real domains and decouple object motion from camera motion.
- **Mechanism**: During fine-tuning on VIPSeg, camera extrinsics are introduced as an auxiliary input (concatenated with trajectory features after being encoded by an MLP), with a 50% probability of dropping camera information during training. The spatial promotion loss $\mathcal{L}_{\text{SPA}}$ randomly samples single-frame trajectories for image reconstruction, updating only spatial layers.
- **Design Motivation**: The camera is static in synthetic data, whereas camera motion in real videos is unpredictable. The spatial promotion loss addresses the issue of object entity collapse under large-angle rotations.

### Loss & Training

The total loss is $\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SPA}}\mathcal{L}_{\text{SPA}}$. Therein, $\mathcal{L}_{\text{MSE}}$ is the standard video diffusion denoising loss (with varying conditions at different stages), and $\mathcal{L}_{\text{SPA}}$ is the single-frame spatial reconstruction loss, where backpropagation only updates spatial layers.

## Key Experimental Results

### Main Results: Trajectory-Guided Video Generation Comparison

| Method | VIPSeg 320×576 ObjMC↓ | FID↓ | FVD↓ | DAVIS ObjMC↓ | FVD↓ |
|------|----------------------|------|------|-------------|------|
| DragNUWA 1.5 | 133.05 | 41.88 | 289.15 | 74.07 | 952.87 |
| DragAnything | 91.12 | 39.29 | 275.93 | 47.01 | 771.78 |
| **PoseTraj** | **77.48** | **38.41** | **267.33** | **29.92** | **729.16** |

### Ablation Study: Impact of Pre-training Designs

| Variant | ObjMC↓ | FID↓ | FVD↓ |
|------|--------|------|------|
| Full method | 77.48 | 38.41 | 267.33 |
| No bbox stage | 81.36 | 41.90 | 275.40 |
| No pretrain | 145.72 | 42.62 | 486.84 |
| No Cam-disen | 83.22 | 39.71 | 279.15 |
| No SPA-loss | 137.26 | 39.79 | 436.56 |

### Key Findings
- Compared to DragAnything, PoseTraj improves trajectory accuracy (ObjMC) by 15% on VIPSeg and by 36% on DAVIS.
- Without the two-stage pre-training, ObjMC degrades to 145.72 (+88%), and the spatial promotion loss is similarly critical (+77% degradation).
- 3D bbox pre-training primarily affects the visual accuracy of object pose localization, having a relatively smaller impact on quantitative metrics.
- In user evaluations, PoseTraj received 43% and 39% more votes in trajectory-following accuracy and visual quality, respectively.

## Highlights & Insights

1. **Pre-training strategy using synthetic data + 3D bbox intermediate supervision**: Ingeniously utilizes synthetic data to address the scarcity of rotation annotations in real videos.
2. **Injection-by-reconstruction paradigm**: Treats 3D information as a reconstruction target instead of an input condition, enabling a seamless transition after training.
3. **Spatial promotion loss**: Significantly improves object preservation under large-angle rotations through a single-frame sub-task.
4. **Strong OOD generalization**: Evaluated on DAVIS without being trained on it, demonstrating outstanding performance.

## Limitations & Future Work

- The object categories (2,000) and rendering quality in the synthetic data are still limited.
- Rotational control for non-rigid deformation (such as human body movement) has not been verified yet.
- The camera motion decoupling module does not use camera pose during inference, which might limit its effectiveness.
- Future work can explore pose-aware extensions based on DiT architectures (such as Tora).

## Related Work & Insights

- Unlike DragAnything, which uses segmentation masks to extract entity features, PoseTraj internalizes 3D understanding through 3D pre-training.
- Similar to how PuppetMaster utilizes synthetic data for part-level animation, PoseTraj focuses on rotational trajectories.
- The concept of spatial promotion loss can be generalized to other video generation tasks requiring spatial consistency guarantees.

## Rating

⭐⭐⭐⭐ — The pre-training strategy is reasonably and effectively designed, with the synthetic data + 3D bbox intermediate supervision being a major highlight; however, the practical application demand for rotational scenarios remains relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation](tora_trajectory-oriented_diffusion_transformer_for_video_generation.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[ICLR 2026\] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](../../ICLR2026/video_generation/learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)

</div>

<!-- RELATED:END -->
