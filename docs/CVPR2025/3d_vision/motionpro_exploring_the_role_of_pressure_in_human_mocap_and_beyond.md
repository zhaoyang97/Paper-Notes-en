---
title: >-
  [Paper Note] MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond
description: >-
  [CVPR 2025][3D Vision][human motion capture] This work constructs MotionPRO, a large-scale pressure-RGB-optical motion capture dataset (70 subjects / 400 action classes / 12.4M frames), and proposes the FRAPPE baseline to fuse pressure signals with monocular RGB. This significantly improves the physical plausibility and global trajectory accuracy of full-body pose estimation, further extending the pressure prior to humanoid robot control.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "human motion capture"
  - "pressure sensing"
  - "multi-modal fusion"
  - "global trajectory"
  - "humanoid robot"
date: 2026-05-08
content_hash: 76a5b089497d894a
---

# MotionPRO: Exploring the Role of Pressure in Human MoCap and Beyond

**Conference**: CVPR 2025  
**arXiv**: [2504.05046](https://arxiv.org/abs/2504.05046)  
**Code**: [Project Page](https://nju-cite-mocaphumanoid.github.io/MotionPRO/)  
**Area**: 3D Vision  
**Keywords**: human motion capture, pressure sensing, multi-modal fusion, global trajectory, humanoid robot

## TL;DR

This work constructs MotionPRO, a large-scale pressure-RGB-optical motion capture dataset (70 subjects / 400 action classes / 12.4M frames), and proposes the FRAPPE baseline to fuse pressure signals with monocular RGB. This significantly improves the physical plausibility and global trajectory accuracy of full-body pose estimation, further extending the pressure prior to humanoid robot control.

## Background & Motivation

**Background**: Existing human motion capture methods mainly focus on visual geometric similarity, relying on RGB images for pose estimation, and have achieved promising performance in terms of local pose accuracy.

**Limitations of Prior Work**: When used for driving virtual avatars in 3D scenes or real-world humanoid robots, existing methods suffer from severe dynamic inaccuracies: temporal drift and jitter, spatial foot sliding, floating, and penetration. These issues stem from neglecting the **physical interaction** between the human body and the scene.

**Key Challenge**: RGB lacks depth information, making it unable to capture the contact relations and dynamic physical interactions between the human body and the ground. Existing pressure datasets are either limited to prone poses or have single action categories (e.g., yoga/tai chi) and very small scales.

**Goal**: (1) Build a large-scale full-body pressure motion capture dataset; (2) Validate the necessity and effectiveness of pressure signals for pose estimation; (3) Explore the role of pressure in humanoid robot control.

**Key Insight**: Revisiting the motion capture problem from the perspective of human interaction with the physical world, pressure signals encode the ground support force acting on the human body, containing rich dynamic mechanisms and physical information.

**Core Idea**: Pressure signals provide physical interaction priors that cannot be obtained from RGB. Fusing both enables accurate and physically plausible full-body motion capture.

## Method

### Overall Architecture

The system consists of two main parts: (1) MotionPRO dataset construction—synchronously capturing 4-view RGB (30Hz), 120×160 full-body pressure mat data (100Hz -> 30Hz), and 50-marker optical motion capture (120Hz -> 30Hz), subsequently obtaining SMPL annotations via Mosh++; (2) two baseline tasks—pressure-only pose estimation and pressure-RGB fused pose estimation (FRAPPE).

### Key Designs

#### 1. Pressure-Only Pose and Trajectory Estimation

Pressure signals are highly sparse (only the feet contact the ground in daily activities), and a single frame of pressure can correspond to thousands of possible poses. The network consists of three components:
- **Small-kernel Pressure Encoder**: Shrinks the ResNet convolutional kernel size to extract finer sparse pressure features.
- **Long-Short term Attention Module (LSAM)**: Leverages GRU to extract short-term contextual motion and multi-head self-attention to capture long-term dependencies.
- **Human Pose Regressor**: Predicts SMPL pose parameters and global translation.

#### 2. FRAPPE: Pressure-RGB Fusion

This adds an RGB branch and a Fusion Cross-Attention Module (FCAM) on top of the pressure-only baseline:
- The RGB branch uses a frozen pre-trained HRNet as the image encoder.
- FCAM uses pressure features as Query and image features as Key/Value (as pressure contains more real-world physical information).
- It adopts **orthographic projection** instead of weak perspective projection to maintain scale invariance in the depth direction, avoiding the misguidance of 3D trajectories by 2D images.

#### 3. Full-Body Contact Annotation Strategy

For each joint in SMPL, it is vertically projected onto the ground to calculate the neighborhood pressure sum $P_j$ and the distance to the ground $D_j$. A joint is annotated as in a contact state when $P_j \geq \tau_1$ and $D_j \leq \tau_2$.

### Loss & Training

Pressure-only network: $\mathcal{L} = \lambda_{pose}\mathcal{L}_{pose} + \lambda_{3d}\mathcal{L}_{3d} + \lambda_{trans}\mathcal{L}_{trans} + \lambda_{contact}\mathcal{L}_{contact}$

Based on this, FRAPPE adds an orthographic 2D alignment loss $\mathcal{L}_{2d}$ to constrain the consistency of the orthographic projection along the camera axis.

## Key Experimental Results

### Main Results: Global Pose Estimation

| Method | MPJPE↓ | PMPJPE↓ | PVE↓ | Accel↓ |
|------|--------|---------|------|--------|
| VIBE | 59.7 | 40.9 | 82.9 | 19.6 |
| CLIFF | 54.7 | 39.7 | 68.6 | 24.3 |
| WHAM | 160.4 | 28.3 | 227.5 | 2.9 |
| **FRAPPE** | **41.8** | **30.2** | **58.6** | **3.0** |

### Global Trajectory Estimation

| Method | WMPJPE↓ | RTE↓ | Jitter↓ | WBCE↓ |
|------|---------|------|---------|-------|
| TRACE | 141.2 | 1193 | 68.6 | 10272 |
| WHAM | 75.6 | 1023 | 9.2 | 1218 |
| **FRAPPE** | **60.8** | **41.6** | **6.0** | **110.2** |

FRAPPE reduces RTE by **96%** and WBCE by **91%** compared to WHAM.

### Ablation Study

| Variant | GTraj↓ | GMPJPE↓ | MPJPE↓ |
|------|--------|---------|--------|
| w/o LSAM | 93.1 | 92.9 | 44.4 |
| w/o FCAM | 66.0 | 70.7 | 39.5 |
| w/o contact loss | 68.7 | 75.9 | 42.6 |
| w/o 2d loss | 82.1 | 95.5 | 52.2 |
| **Full** | **62.2** | **68.6** | **40.5** |

### Key Findings

1. Pressure alone can provide accurate global trajectories and plausible lower-body poses (lower-body PMPJPE of 32.4mm).
2. Removing the 2D loss actually leads to better Jitter and FS, suggesting a trade-off between 2D and 3D loss.
3. In humanoid robot control, the CoM/CoP error of FRAPPE (15.32/22.35mm) is far lower than that of pure RGB methods.

## Highlights & Insights

- **Leading Dataset Scale**: 70 subjects, 400 action classes, and 12.4M frames, significantly outperforming prior works like MoYo (1 subject, 82 actions).
- **Physical Interaction Perspective**: Redefines the motion capture problem from a pure vision problem to a human-scene physical interaction problem.
- **Insight on Orthographic Projection**: Recognizes that weak perspective projection leads to pose-trajectory coupling when focusing on global trajectories, whereas orthographic projection maintains scale in the depth direction.
- **From Virtual Avatars to Real Robots**: Fully validates the feasibility of pressure priors transferring from SMPL-driven avatars to NAO humanoid robots.
- **Contact Annotation Method**: Achieves full-body contact annotation through a dual-threshold strategy combining pressure and distance to the ground.

## Limitations & Future Work

1. The pressure mat restricts the range of motion, requiring all movements to be performed on the mat.
2. The dataset lacks outdoor scenes and non-flat ground conditions.
3. The assumption of known camera intrinsic parameters in FRAPPE is limited in in-the-wild scenarios.
4. The pressure mat resolution (120×160) may be insufficient for fine-grained foot movements.

## Related Work & Insights

- **MoYo** first introduced Center of Pressure (CoP) constraints, but only worked effectively for quasi-static actions; MotionPRO extends full-body contacts to dynamic actions.
- **WHAM** uses contact labels computed solely based on foot velocity, which has low accuracy and ignores contact with other body parts.
- **PIMesh** utilizes multi-frame pressure information for high-accuracy prone pose estimation but cannot generalize to daily movements.
- **Inspiration**: The paradigm of pressure signals as "physical sensors" can be extended to other interactive sensing modalities (e.g., torque, IMU), providing richer environmental interaction information for embodied AI.

## Rating

⭐⭐⭐⭐ — Significant engineering effort in dataset construction, comprehensive experiments (virtual avatars + real robots), and valuable insights. However, the method design leans toward a simple baseline, showing potential for further in-depth exploration of fusion mechanisms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[NeurIPS 2025\] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](../../NeurIPS2025/3d_vision/cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)
- [\[CVPR 2025\] Gaussian Eigen Models for Human Heads](gaussian_eigen_models_for_human_heads.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
