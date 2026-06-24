---
title: >-
  [Paper Note] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera
description: >-
  [3D Vision] Dyn-HaMR proposes the first optimization framework to recover 4D global motion trajectories of interacting hands from monocular videos captured by a dynamic camera. Utilizing a three-stage pipeline (hierarchical initialization $\rightarrow$ SLAM-guided global motion optimization $\rightarrow$ interacting motion prior optimization), it decouples camera motion from hand motion and significantly outperforms existing methods across multiple datasets.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 023f285643281c1c
---

# Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera

## TL;DR

Dyn-HaMR proposes the first optimization framework to recover 4D global motion trajectories of interacting hands from monocular videos captured by a dynamic camera. Utilizing a three-stage pipeline (hierarchical initialization $\rightarrow$ SLAM-guided global motion optimization $\rightarrow$ interacting motion prior optimization), it decouples camera motion from hand motion and significantly outperforms existing methods across multiple datasets.

## Background & Motivation

Reconstructing 3D hand meshes from monocular videos is a critical task for understanding human behavior, with important applications in AR/VR. However, existing methods face severe limitations:

1. **Weak perspective camera assumption**: Methods such as HaMeR, IntagHand, and ACR assume a weak perspective camera model, which only models motion within a limited camera frustum and fails to recover the **global 3D trajectory**.
2. **Camera-hand motion coupling**: In egocentric scenarios, the camera moves with the body, tangling hand motion and camera motion together, which existing methods cannot decouple.
3. **Depth estimation noise**: Relying solely on 2D cues leads to depth ambiguity, resulting in noisy or incorrect depth estimations.
4. **Interaction occlusion**: Frequent two-hand interactions cause serious occlusions, truncations, and missing detections.
5. **Lack of datasets**: There is a lack of sufficient temporal datasets for learning 4D global interactions.

Key Challenge: How to recover the global hand motion in the world coordinate system from the camera coordinate system under dynamic cameras and complex hand interactions?

## Method

### Overall Architecture

Dyn-HaMR is a three-stage multi-objective optimization pipeline: Stage I performs hierarchical initialization and motion completion; Stage II estimates camera motion using SLAM and optimizes global trajectories; Stage III refines interactions by introducing hand motion priors and biomechanical constraints. Using the MANO parameterized hand model, the global motion is represented as a sequence of hand states $\mathbf{Q}^h = \{q_t^h\}_{t=1}^T$.

### Key Designs

#### 1. Hierarchical Initialization and Generative Motion Completion (Stage I)

- **Function**: Initializing frame-by-frame hand states from raw video and filling in missing detections caused by occlusion.
- **Mechanism**: Merging four hand detection/reconstruction methods (ViTPose, ACR, HaMeR, MediaPipe) for hierarchical initialization—first using ViTPose to obtain hand bounding box sequences, and then utilizing ACR/HaMeR/MediaPipe to extract frame-by-frame MANO parameters. For missing frames, a generative completion is performed using latent space optimization under the HMP hand motion prior.
- **Design Motivation**: Monocular/single-frame methods lack temporal consistency and frequently fail in detection, making a single method insufficiently robust. Hierarchical fusion of multiple methods improves coverage, and generative completion aligns better with kinematic constraints compared to simple interpolation.

#### 2. SLAM-Guided 4D Global Motion Optimization (Stage II)

- **Function**: Decoupling camera motion and hand motion to recover global hand trajectories in the world coordinate system.
- **Mechanism**: DPVO (a SLAM system) is used to estimate the relative camera motion $\mathbf{C}_t = \{\mathbf{R}_t, \boldsymbol{\tau}^c_t\}$, and the global trajectory is obtained by combining camera motion and hand motion. Crucially, a **world scale factor** $\omega$ is introduced to model the relative scale between camera displacement and hand motion. The global trajectory, orientation, local pose, and camera extrinsics are jointly optimized.
- **Design Motivation**: The scale of camera motion provided by SLAM is inherently ambiguous, whereas hand motion is constrained within physically reasonable ranges. Optimizing $\omega$ leverages two-hand motion to further constrain the camera scale, resolving monocular depth ambiguity. The optimization is conducted in two steps: first, 20 steps of optimization for global orientation and translation; followed by 60 steps for local pose, shape, scale factor, and camera extrinsics.

#### 3. Interacting Motion Prior Optimization (Stage III)

- **Function**: Refining two-hand interactions using learned motion priors and biomechanical constraints to produce more realistic motions.
- **Mechanism**: Optimization is performed within the latent space of the HMP motion prior, introducing three types of extra constraints: (a) motion prior loss to ensure motion likelihood; (b) penetration loss to prevent interpenetration of the two hand meshes; (c) biomechanical constraints to ensure joint angles, bone lengths, and hand palmar shape remain within physiologically reasonable ranges.
- **Design Motivation**: The reprojection loss in Stage II lacks sufficient constraints, which can lead to physically implausible poses. The motion prior provides kinematic prior knowledge, and the penetration and biomechanical constraints resolve physical implausibilities during interactions. It is solved in two phases: first, 200 steps of optimization for global motion; followed by 200 steps incorporating latent codes, local poses, and camera parameters.

### Loss & Training

**Stage II** Global Optimization Objective:

$$E_I = \lambda_{2d}\mathcal{L}_{2d} + \lambda_s\mathcal{L}_{smooth} + \lambda_{cam}\mathcal{L}_{cam} + \lambda_J\mathcal{L}_J + \lambda_\beta\mathcal{L}_\beta$$

**Stage III** Interaction Optimization Objective:

$$E_{II} = \mathcal{L}_{prior} + \mathcal{L}_{pen} + \mathcal{L}_{bio} + \lambda_{2d}\mathcal{L}_{2d} + \lambda_s\mathcal{L}_{smooth} + \lambda_{cam}\mathcal{L}_{cam} + \lambda_J\mathcal{L}_J + \lambda_\beta\mathcal{L}_\beta$$

where $\mathcal{L}_{prior} = \lambda_z\mathcal{L}_z + \lambda_\phi\mathcal{L}_\phi + \lambda_\tau\mathcal{L}_\tau$ (motion prior likelihood + global orientation consistency + translation consistency), $\mathcal{L}_{bio}$ contains joint angles, bone lengths, and hand palmar constraints, and $\mathcal{L}_{pen}$ is the Chamfer distance based on interpenetrating vertices of both hands.

## Key Experimental Results

### Main Results

**H2O Dataset (Dynamic Camera, Tab. 2)**:

| Method | G-MPJPE↓ | GA-MPJPE↓ | MPJPE↓ | Acc Err↓ |
|------|----------|-----------|--------|----------|
| ACR | 113.6 | 88.5 | 46.8 | 14.3 |
| IntagHand | 105.5 | 81.5 | 45.6 | 13.5 |
| HaMeR | 96.9 | 75.7 | 32.9 | 9.21 |
| Ours (w/o III) | 51.9 | 41.2 | 24.9 | 9.5 |
| **Dyn-HaMR** | **45.6** | **34.2** | **22.5** | **4.2** |

**InterHand2.6M (Static Camera, Tab. 1)**:

| Method | MPJPE↓ | MPVPE↓ | Acc Err↓ |
|------|--------|--------|----------|
| ACR | 8.75 | 9.01 | 3.99 |
| HaMeR | 9.84 | 10.13 | 5.13 |
| **Dyn-HaMR** | **7.94** | **8.15** | **2.76** |

### Ablation Study (Tab. 2 & Tab. 6)

| Variant | G-MPJPE↓ | MPJPE↓ | Acc Err↓ |
|------|----------|--------|----------|
| w/o Stage III | 51.9 | 24.9 | 9.5 |
| w/o Biomechanical constraints | - | Improved but inferior to Full Model | - |
| w/o Penetration loss | - | Severe hand interpenetration | - |
| w/o Generative completion | - | Performance degradation | - |
| **Full Model** | **45.6** | **22.5** | **4.2** |

### Key Findings

1. **Substantial lead in global motion recovery**: On the H2O dataset, G-MPJPE is reduced from HaMeR's 96.9 to 45.6 (**53% reduction**), and GA-MPJPE from 75.7 to 34.2 (**55% reduction**).
2. **Effective even with static cameras**: Dyn-HaMR achieves new SOTA performance even on InterHand2.6M (static camera), with MPJPE reaching 7.94 compared to ACR's 8.75.
3. The motion prior and interaction constraints in Stage III contribute significantly, with acceleration error decreasing from 9.5 to 4.2.

## Highlights & Insights

1. **First to resolve 4D global interacting hand motion recovery under dynamic cameras**: This fills a key research gap and holds great significance for egocentric gesture interaction in AR/VR.
2. **Ingenious scale factor design**: By optimizing $\omega$, it decouples the relative scale between camera displacement and hand motion, leveraging the plausibility constraints of hand motions to conversely assist in estimating the camera scale.
3. **Generative motion completion outperforms interpolation**: It utilizes the HMP prior to perform motion completion within the latent space, simultaneously addressing temporal smoothing and missing detections.
4. **Reasonable three-stage progressive optimization design**: From coarse to fine, introducing progressively stronger constraints at each stage.

## Limitations & Future Work

1. **Optimization efficiency**: The three-stage optimization pipeline (totaling 480+ L-BFGS steps) is computationally intensive, making real-time application difficult.
2. **Dependence on SLAM quality**: DPVO may fail in scenarios with extreme motion blur or low texture, affecting downstream global motion recovery.
3. **Limitations of HMP prior**: The hand motion prior is only trained on the Arctic dataset, resulting in limited generalization ability to unseen interaction types.
4. **Unmodeled object interactions**: Although evaluated on datasets containing objects, hand-object interaction constraints are not explicitly modeled.
5. Lack of evaluation and scalability analysis on ultra-long videos (>1000 frames).

## Related Work & Insights

- **HaMeR / ACR / IntagHand**: Single-frame hand reconstruction methods; Dyn-HaMR uses them as the foundation for initialization.
- **HuMoR / WHAM**: Global human motion recovery; Dyn-HaMR introduces similar concepts into the hand domain.
- **HMP**: Hand motion prior model, providing kinematic constraints for Stage I/III.
- **DPVO**: Data-driven SLAM system, providing camera motion estimation.
- Insight: 4D hand reconstruction can draw inspiration from the methodological frameworks of human motion recovery (SLAM + motion prior + staged optimization).

## Rating: ⭐⭐⭐⭐

The problem definition is clear and practical (hand reconstruction under dynamic cameras is highly demanded in AR/VR), the three-stage optimization design is comprehensive, and the experiments are extensive (over 6 datasets). The substantial lead in global motion recovery is convincing. One star is deducted because optimization efficiency limits practical deployment, and the generalization of the HMP prior has not been fully verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](../../ICCV2025/3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[ICCV 2025\] RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](../../ICCV2025/3d_vision/rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

</div>

<!-- RELATED:END -->
