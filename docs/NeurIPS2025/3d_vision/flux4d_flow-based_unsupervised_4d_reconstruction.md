---
title: >-
  [Paper Note] Flux4D: Flow-based Unsupervised 4D Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][4D reconstruction] Flux4D is proposed as an unsupervised and generalizable 4D dynamic driving scene reconstruction framework. It employs a feed-forward network to directly predict 3D Gaussians a…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "unsupervised"
  - "3D Gaussian"
  - "scene flow"
  - "autonomous driving"
date: 2026-05-08
content_hash: d1f74e86b3da3d36
---

# Flux4D: Flow-based Unsupervised 4D Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2512.03210](https://arxiv.org/abs/2512.03210)  
**Code**: [https://waabi.ai/flux4d](https://waabi.ai/flux4d)  
**Area**: 3D Vision / Autonomous Driving / 4D Reconstruction
**Keywords**: 4D reconstruction, unsupervised, 3D Gaussian, scene flow, autonomous driving

## TL;DR
Flux4D is proposed as an unsupervised and generalizable 4D dynamic driving scene reconstruction framework. It employs a feed-forward network to directly predict 3D Gaussians and their motion velocities, achieving large-scale scene reconstruction using only photometric loss and a static-preference regularization. The method surpasses all unsupervised approaches on PandaSet and Waymo while approaching the performance of supervised methods.

## Background & Motivation

**Background**: Reconstructing 4D dynamic scenes from visual observations is a core problem in computer vision. Current methods primarily rely on differentiable rendering with NeRF or 3DGS, decomposing scenes into static backgrounds and dynamic objects, but require manually annotated 3D tracklets or dynamic masks to distinguish static from dynamic elements.

**Limitations of Prior Work**: (a) Manual annotation is costly and difficult to scale to large quantities of unlabeled data; (b) automatic labeling using pretrained perception models introduces noise and artifacts; (c) existing unsupervised methods rely on complex regularization schemes (geometric constraints, cycle consistency, multi-stage training) that are sensitive to hyperparameters and slow to train (requiring hours per scene); (d) existing generalizable methods can only handle a small number of low-resolution inputs (≤12 frames, ≤360px).

**Key Challenge**: Expensive annotations combined with per-scene optimization overhead make scaling to large-scale data infeasible.

**Goal**: To achieve fast (second-level), scalable, and generalizable 4D scene reconstruction without any annotations.

**Key Insight**: Training across a large number of scenes enables the network to automatically learn static/dynamic decomposition as a data-driven prior; LiDAR is incorporated to handle high-resolution (≥1080p) dense multi-view (≥60 frames) inputs.

**Core Idea**: A minimalist design — photometric loss plus static-preference regularization only — leverages cross-scene learning to enable a feed-forward network to automatically perform static/dynamic decomposition and 4D reconstruction.

## Method

### Overall Architecture
**Inputs**: Multi-timestep camera images $\mathcal{I} = \{\mathbf{I}_k\}$ and LiDAR point clouds $\mathcal{P} = \{\mathbf{P}_k\}$. **Outputs**: A scene representation encoding geometry, appearance, and 3D flow. The pipeline consists of three steps: (1) initialize 3D Gaussians from per-frame sensor data; (2) predict 3D flow and refined attributes via a network; (3) train with reconstruction loss and static-preference loss only.

### Key Designs

1. **LiDAR-Guided Scene Initialization**:

    - **Function**: Initialize 3D Gaussian positions, scales, and colors from LiDAR point clouds.
    - **Mechanism**: Gaussian positions are initialized from LiDAR points $\mathbf{P}_k$; scales are determined by the mean distance to neighboring points; colors are obtained by projecting points onto the corresponding camera image $\mathbf{I}_k$. Each Gaussian is assigned a timestamp $t_i$ and an initial velocity $\mathbf{v}_i = 0$. Additional random points are placed on a distant sphere to model the sky.
    - **Design Motivation**: This exploits LiDAR data commonly available in autonomous driving, avoids learning geometry from scratch, and enables the system to handle high-resolution (≥1920×1080) dense inputs.

2. **Feed-Forward Flow Prediction Network**:

    - **Function**: A 3D sparse convolutional U-Net predicts refined attributes and motion velocities from the initialized Gaussians.
    - **Mechanism**: $\mathcal{G}, \mathcal{V} = f_\theta(\mathcal{G}_{\mathrm{init}}, \mathcal{T})$, where $\mathcal{V} = \{\mathbf{v}_i\}$ denotes the 3D velocity of each Gaussian. A linear motion model propagates Gaussians from the source timestep to the target timestep: $\mathbf{p}_i^{t'} = \mathbf{p}_i^{t_i} + \mathbf{v}_i \cdot (t' - t_i)$.
    - **Design Motivation**: Feed-forward inference (second-level) replaces per-scene optimization (hour-level); operating in 3D space ensures cross-view geometric consistency and reduces appearance-motion ambiguity.

3. **"As Static as Possible" Regularization and Unsupervised Learning**:

    - **Function**: Training relies solely on reconstruction loss and velocity regularization, without complex regularization schemes.
    - **Mechanism**: The total loss is $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}$, where $\mathcal{L}_{\text{recon}} = \lambda_{\text{rgb}} \mathcal{L}_{\text{rgb}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$, and the velocity regularization $\mathcal{L}_{\text{vel}} = \frac{1}{M}\sum_i \|\mathbf{v}_i\|_2$ encourages Gaussians to remain stationary.
    - **Design Motivation**: Training across a large number of scenes allows the network to automatically learn static/dynamic decomposition; data-driven priors replace hand-crafted regularization. This is the central finding of the paper.

4. **Iterative Refinement and Motion Augmentation (Flux4D-full)**:

    - **Function**: Iteratively refines appearance via 3D gradient feedback and replaces the linear motion model with a polynomial model.
    - **Mechanism**: After the forward pass, 3D gradients are computed and fed back as input to a refinement network $f_\phi$; two iterations suffice to correct color inconsistencies and missing details. Rendered 2D flow is used to re-weight losses in motion regions, assigning larger loss weights to high-velocity areas.
    - **Design Motivation**: Single feed-forward capacity is limited; iterative feedback improves detail. Pixel-level flow re-weighting addresses the imbalanced loss contribution caused by the low proportion of dynamic objects.

### Loss & Training
- Loss weights: $\lambda_{\text{rgb}}=0.8, \lambda_{\text{SSIM}}=0.2, \lambda_{\text{depth}}=0.01, \lambda_{\text{vel}}=5\times10^{-3}$
- Training is conducted on 4×L40S GPUs for 30,000 iterations (~2 days) across 93 PandaSet training scenes.
- Full-resolution images (≥1920×1080) are processed; inference takes ~1.8 seconds per 1-second clip.

## Key Experimental Results

### Main Results (PandaSet Novel View Synthesis, 1s Clips)

| Method | Unsup.? | General.? | Dynamic PSNR↑ | Full PSNR↑ | D_RMSE↓ | V_RMSE↓ | Speed |
|--------|---------|-----------|---------------|------------|---------|---------|-------|
| NeuRAD (supervised) | ✗ | ✗ | 23.01 | 24.61 | 2.30 | — | ~60min |
| StreetGS (supervised) | ✗ | ✗ | 20.06 | 23.38 | 0.84 | — | ~28min |
| G3R (supervised) | ✗ | ✓ | 21.85 | 24.35 | 1.96 | — | 17s |
| EmerNeRF† | ✓ | ✗ | 17.79 | 22.80 | 4.24 | 0.432 | ~100min |
| DeSiRe-GS† | ✓ | ✗ | 19.08 | 22.25 | 24.89 | 0.322 | ~120min |
| STORM† | ✓ | ✓ | 17.65 | 20.79 | 4.80 | 0.238 | 0.07s |
| **Flux4D (Ours)** | **✓** | **✓** | **21.99** | **23.84** | **1.07** | **0.157** | **1.8s** |

On the Waymo dataset, Flux4D outperforms DrivingRecon by **+5.99 dB PSNR** and +0.21 SSIM.

### Ablation Study

| Configuration | Dynamic PSNR↑ | Full PSNR↑ | Note |
|---------------|---------------|------------|------|
| Flux4D-base (linear motion) | 21.43 | 23.52 | Base version |
| + Iterative refinement | 21.75 | 23.72 | Improved detail |
| + Motion augmentation | 21.99 | 23.84 | Full model |
| Trained on 1 scene | Lower | Lower | Per-scene overfitting |
| Trained on 93 scenes | 21.99 | 23.84 | More data is better |

### Key Findings
- **Cross-scene training is critical**: Training on more scenes significantly improves static/dynamic decomposition quality; data-driven priors are more effective than hand-crafted regularization.
- The unsupervised Flux4D achieves a dynamic-region PSNR of 21.99, approaching the supervised G3R (21.85).
- On the future frame prediction task, Flux4D even surpasses supervised methods (PSNR 19.07 vs. G3R 18.93).
- Although scene flow estimation is not the primary objective, Flux4D outperforms dedicated scene flow methods on most metrics.

## Highlights & Insights
- **Minimalist design philosophy**: Static/dynamic decomposition is achieved through photometric loss plus velocity regularization alone, relying on data-driven priors from cross-scene training. This offers a broadly applicable insight — simple methods with large data may outperform complex methods with small data.
- **3D spatial design vs. 2D pixel space**: Predicting Gaussians and flow in 3D space naturally ensures cross-view consistency and supports more high-resolution input frames.
- **Effective use of LiDAR initialization**: LiDAR point clouds serve as initialization rather than supervision signals, preserving the unsupervised nature while providing a strong geometric prior.

## Limitations & Future Work
- Linear/polynomial motion models have limited capacity to model non-rigid motion (e.g., pedestrian articulation).
- Dependence on LiDAR data restricts transferability to purely vision-based autonomous driving.
- The constant-velocity assumption within short time windows (~1s) may lead to accumulated errors in long-horizon prediction.
- Training requires 4×48GB GPUs for approximately 2 days, which remains computationally expensive.

## Related Work & Insights
- **vs. EmerNeRF/DeSiRe-GS**: Per-scene-optimized unsupervised methods that require pretrained visual models for assistance and take 1–2 hours per scene to train. Flux4D achieves inference in 1.8 seconds without external models.
- **vs. STORM/DrivingRecon**: Also feed-forward unsupervised methods, but limited to ≤12 frames at low resolution (≤360px) and dependent on pretrained visual models. Flux4D handles ≥60 frames at full HD resolution.
- **vs. G3R**: A supervised generalizable method requiring 3D tracklets. Flux4D achieves comparable performance without any annotations.
- The core paradigm is transferable: cross-scene training combined with minimal loss design is worth exploring in other tasks requiring dynamic scene understanding, such as video editing and robotic simulation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The central insight of minimalist design combined with cross-scene data priors is original, though individual technical components (3DGS, sparse convolutional U-Net) are established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets (PandaSet + Waymo), four tasks (NVS / flow estimation / future prediction / controllable simulation), and multiple supervised and unsupervised baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly structured with high-quality figures; the paradigm comparison diagram is particularly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ The first work to bring unsupervised generalizable 4D reconstruction close to supervised-level performance, with practical significance for autonomous driving simulation.

## Related Papers

- [\[CVPR 2025\] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](../../CVPR2025/3d_vision/floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[NeurIPS 2025\] UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss](ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[NeurIPS 2025\] Web-Scale Collection of Video Data for 4D Animal Reconstruction](web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)
- [\[NeurIPS 2025\] UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss](ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)

</div>

<!-- RELATED:END -->
