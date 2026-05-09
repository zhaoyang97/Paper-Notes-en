---
title: >-
  [Paper Note] Aether: Geometric-Aware Unified World Modeling
description: >-
  [ICCV 2025][Image Generation][World Model] Aether proposes a geometric-aware unified world modeling framework that jointly trains reconstruction, prediction, and planning capabilities on synthetic 4D data, built upon post-training of CogVideoX to achieve zero-shot generalization to real-world scenes.
tags:
  - ICCV 2025
  - Image Generation
  - World Model
  - 4D Reconstruction
  - Action-Conditioned Video Prediction
  - Visual Planning
  - Diffusion Model
date: 2026-05-08
content_hash: b8941b4392d233d2
---

# Aether: Geometric-Aware Unified World Modeling

**Conference**: ICCV 2025
**arXiv**: [2503.18945](https://arxiv.org/abs/2503.18945)
**Code**: [https://github.com/AetherWorld](https://github.com/AetherWorld)
**Area**: World Models / Video Generation
**Keywords**: World Model, 4D Reconstruction, Action-Conditioned Video Prediction, Visual Planning, Diffusion Model

## TL;DR

Aether proposes a geometric-aware unified world modeling framework that jointly trains reconstruction, prediction, and planning capabilities on synthetic 4D data, built upon post-training of CogVideoX to achieve zero-shot generalization to real-world scenes.

## Background & Motivation

World models serve as a foundational paradigm for building autonomous systems, requiring three core capabilities: **perception** (4D dynamic reconstruction), **prediction** (action-conditioned video generation), and **planning** (goal-directed path planning). However, existing methods typically focus on only one of these aspects:

**Reconstruction methods** (DUSt3R, MonST3R, etc.) recover geometric structure only, without prediction or planning capabilities.

**Video generation models** (CogVideoX, SVD) produce visually realistic videos but lack geometric consistency.

**Planning systems** generally rely on explicit state representations and struggle to plan directly from visual inputs.

The authors' core insight is that these three capabilities can be learned synergistically through shared geometric priors — reconstruction teaches the model to understand geometry, prediction enables learning of dynamic evolution, and planning combines both for decision-making. A further key motivation is the **scarcity of 4D data**; the authors opt for synthetic data training and leverage pretrained priors from video generation models to achieve zero-shot transfer to real-world scenes.

## Method

### Overall Architecture

Aether is built upon post-training of CogVideoX-5b-I2V. The target latent variable $z_0$ encompasses three modalities:

- **Color video latent $z_c$**: RGB video encoding
- **Depth video latent $z_d$**: normalized disparity encoding of depth video
- **Action latent $z_a$**: raymap encoding of camera trajectories

By randomly combining different input conditions $c = c_c \otimes c_a$, the same model supports distinct tasks:

- **4D Reconstruction**: input full video → output depth + camera parameters
- **Video Prediction**: input observed frame(s) (+ optional action conditioning) → generate future frames
- **Visual Planning**: input start + goal images → generate intermediate path video

### Key Designs

**1. Automated 4D Synthetic Data Annotation Pipeline**

The pipeline consists of four stages:

- **Dynamic Masking**: dynamic objects (pedestrians/vehicles) are segmented using Grounded SAM 2 to ensure accurate camera estimation.
- **Video Clipping**: frames unsuitable for reconstruction are filtered via SIFT keypoint detection and RAFT optical flow estimation.
- **Coarse Camera Estimation**: initial intrinsic and extrinsic parameters are estimated using DroidCalib.
- **Camera Refinement**: long-range correspondences are obtained via CoTracker3, followed by Bundle Adjustment + Ceres Solver for refinement.

**2. Depth Video Processing — Scale-Invariant Normalized Disparity**

Depth is converted to a disparity representation compatible with the VAE: depth → clip → sqrt → reciprocal → normalize to $[-1, 1]$ → replicate to 3 channels → VAE encoding.

**3. Camera Trajectory Raymap Representation**

Camera parameters are converted into a 6-channel raymap video (3 ray directions + 3 ray origins). The translation component undergoes scale normalization and signed-log transformation, with spatial downsampling by $8\times$ and temporal concatenation every 4 frames to align with the spatiotemporal dimensions of the DiT.

**4. Multi-Task Conditional Random Masking**

During training, conditioning inputs are randomly masked to accommodate different tasks: 30% visual planning (first and last frames), 40% video prediction (first frame only), 28% reconstruction (full video), 2% unconditional; action conditioning is retained or masked with 50% probability each.

### Loss & Training

**Two-stage training**:

- **Stage 1**: Standard latent-space MSE denoising loss.
- **Stage 2** (1/4 of total steps): decoding to image space with additional MS-SSIM loss (color video), scale-shift invariant loss (depth), and pointmap loss (depth + raymap projection consistency).

A hybrid FSDP + DDP parallelism strategy is employed across 80 A100-80GB GPUs, with an effective batch size of 320, trained for approximately two weeks. AdamW optimizer with OneCycle learning rate scheduling is used.

## Key Experimental Results

### Main Results

**Video Depth Estimation** (zero-shot, per-sequence scale alignment):

| Method | Sintel AbsRel↓ | Sintel δ<1.25↑ | KITTI AbsRel↓ | KITTI δ<1.25↑ |
|--------|:-:|:-:|:-:|:-:|
| MonST3R-GA | 0.378 | 55.8 | 0.168 | 74.4 |
| CUT3R | 0.421 | 47.9 | 0.118 | 88.1 |
| DepthCrafter | 0.590 | 55.5 | 0.124 | 86.5 |
| **Aether** | **0.324** | 50.2 | **0.056** | **97.8** |

**Action-Conditioned Video Prediction VBench Metrics (overall)**:

| Method | Subject Consistency | Background Consistency | Motion Smoothness | Dynamic Degree | Weighted Mean |
|--------|:-:|:-:|:-:|:-:|:-:|
| CogVideoX | 90.51 | 92.77 | 98.24 | 86.76 | 79.92 |
| **Aether** | **91.54** | **94.06** | **98.56** | **94.85** | **80.71** |

### Ablation Study

**Effect of Reconstruction Objective on Navigation Performance**:

| Method | PSNR↑ | SSIM↑ | MS-SSIM↑ | LPIPS↓ |
|--------|:-:|:-:|:-:|:-:|
| Aether-no-depth | 18.97 | 0.5353 | 0.5376 | 0.3074 |
| **Aether (full)** | **19.70** | **0.5545** | **0.5760** | **0.2659** |

Incorporating the depth reconstruction objective yields consistent improvements across all navigation metrics, validating the effectiveness of multi-task joint learning.

### Key Findings

1. **Zero-shot generalization**: trained entirely on synthetic data, the model achieves an AbsRel of only 0.056 on KITTI real-world depth estimation, surpassing all specialized models.
2. **Reconstruction-enhanced generation**: the depth reconstruction objective simultaneously improves video prediction and planning quality.
3. **Raymap action representation**: camera trajectory raymaps provide more precise action control compared to text descriptions.
4. **Efficient inference**: reconstruction tasks require only 4 denoising steps.

## Highlights & Insights

1. **The unified framework is remarkably elegant**: a single model supports reconstruction, prediction, and planning simultaneously via a conditional masking strategy, without any task-specific modules.
2. **Successful synthetic-to-real zero-shot transfer**: the pretrained priors of video generation models effectively bridge the domain gap.
3. **Raymap representation**: converting camera trajectories into 6-channel representations aligned with video frames elegantly resolves the heterogeneous modality fusion problem.
4. **The 4D annotation pipeline** carries independent engineering value.

## Limitations & Future Work

1. Only camera trajectories are supported as the action space; robot joint actions and similar modalities are not directly accommodated.
2. Generated camera trajectories are noisy and require Kalman filter post-processing.
3. Training cost is high (80 × A100, two weeks).
4. Depth estimation on the BONN dataset is moderate (AbsRel 0.273), with room for improvement in indoor close-range scenes.
5. Planning is limited to visual path planning, without physical engine interaction.

## Related Work & Insights

- **DUSt3R / MASt3R / MonST3R / CUT3R**: end-to-end 3D/4D reconstruction methods serving as the primary baselines for the reconstruction component.
- **CogVideoX**: the base video generation model, into which geometric awareness is injected via post-training.
- **Genie 2 / Cat3D / Cat4D**: alternative world modeling approaches lacking unified reconstruction capability.
- **DepthCrafter / DA-V**: diffusion-based video depth estimation methods.
- **Insights**: the unified paradigm (reconstruction + generation + planning) represents an important pathway toward embodied AI; synthetic data combined with pretrained priors can effectively alleviate data scarcity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](a0_an_affordance-aware_hierarchical_model_for_general_robotic_manipulation.md)
- [\[ICCV 2025\] Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)

<!-- RELATED:END -->
