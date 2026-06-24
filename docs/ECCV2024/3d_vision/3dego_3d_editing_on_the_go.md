---
title: >-
  [Paper Note] 3DEgo: 3D Editing on the Go!
description: >-
  [ECCV 2024][3D Vision][3D Editing] 3DEgo compresses the traditional three-stage 3D editing pipeline (COLMAP pose estimation $\rightarrow$ unedited scene initialization $\rightarrow$ iterative editing and update) into a single-stage framework: first performing multi-view consistent 2D editing on video frames using an autoregressive noise blending module, and then directly reconstructing the 3D scene from the edited frames using COLMAP-free 3DGS…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Editing"
  - "3D Gaussian Splatting"
  - "Multi-view Consistency"
  - "Text-guided Editing"
  - "COLMAP-free"
date: 2026-05-08
content_hash: b590ebfc1b158d38
---

# 3DEgo: 3D Editing on the Go!

**Conference**: ECCV 2024  
**arXiv**: [2407.10102](https://arxiv.org/abs/2407.10102)  
**Code**: [https://3dego.github.io/](https://3dego.github.io/) (with project page)  
**Area**: 3D Vision  
**Keywords**: 3D Editing, 3D Gaussian Splatting, Multi-view Consistency, Text-guided Editing, COLMAP-free  

## TL;DR
3DEgo compresses the traditional three-stage 3D editing pipeline (COLMAP pose estimation $\rightarrow$ unedited scene initialization $\rightarrow$ iterative editing and update) into a single-stage framework: first performing multi-view consistent 2D editing on video frames using an autoregressive noise blending module, and then directly reconstructing the 3D scene from the edited frames using COLMAP-free 3DGS, boosting the speed by approximately 10x and supporting videos from arbitrary sources.

## Background & Motivation
Text-driven 3D scene editing is an active research direction. Methods like IN2N pioneered the paradigm of editing NeRF scenes using InstructPix2Pix (IP2P). However, existing methods suffer from three core *Limitations of Prior Work*: (1) they must rely on COLMAP for SfM pose estimation, which restricts the sources of input videos; (2) they require initializing the 3D model with the original unedited images first, which is time-consuming and redundant; (3) the iterative editing and updating process requires a large number of training iterations to merge inconsistent editing results, ultimately taking about 285 minutes. These limitations prevent 3D editing from scaling to casually captured daily videos.

## Core Problem
How to directly generate a text-guided edited 3D scene from a monocular video without COLMAP poses and without unedited scene initialization? The *Key Challenge* lies in: (1) how to ensure the multi-view consistency of 2D diffusion editing across frames? (2) how to reconstruct the 3D scene from edited frames without pre-computed poses? These two problems have never been resolved simultaneously in previous works.

## Method

### Overall Architecture
The input consists of a monocular video $V$ and an editing prompt $\mathcal{T}$. The pipeline is divided into two main stages:

**Stage 1: Multi-View Consistent 2D Editing** — After extracting frames from the video, an LLM (GPT-3.5) is used to parse the text and extract key editing attributes, after which SAM generates a Key Editing Area (KEA) mask for each frame. A zero-shot point tracker ensures mask consistency across frames. Then, an autoregressive noise blending module is applied to perform IP2P editing on all frames, ensuring consistency in editing adjacent frames.

**Stage 2: COLMAP-free 3D Reconstruction** — Using a monocular depth estimator to initialize the 3DGS point cloud of each frame, the relative camera poses between frames are estimated by learning SE-3 affine transformations, progressively expanding the global 3D scene. A KEA identity vector is added to each Gaussian point for localized fine-grained editing control.

### Key Designs

1. **Autoregressive Noise Blender**: When editing the $i$-th frame, not only is the original image of the current frame used as a condition, but the previous $w$ edited frames are also introduced as conditions. Specifically, for each edited frame $E_n$, the image-conditional noise estimate $\epsilon_\theta^n(e_t, E_n, \emptyset_\mathcal{T})$ is calculated, and then weighted-averaged using exponentially decaying weights $\beta_n = \lambda_d^{w-n} / \sum_{j=1}^{w}\lambda_d^{w-j}$ to obtain the blended noise $\bar{\epsilon}_\theta$. The final noise prediction is the weighted sum of the standard IP2P prediction for the current frame and the blended noise: $\epsilon_\theta(e_t, f, \mathcal{T}, W) = \gamma_f \tilde{\epsilon}_\theta(e_t, f, \mathcal{T}) + \gamma_E \bar{\epsilon}_\theta(e_t, \emptyset_\mathcal{T}, W)$. This enables a natural transition of edits between adjacent frames without requiring extra training or fine-tuning.

2. **KEA Identity Parameterization**: A learnable vector $m$ of length 2 (corresponding to foreground and background categories) is appended to each 3D Gaussian, and the KEA identity label is obtained via softmax. During training, $m$ is optimized simultaneously to precisely restrict the edits to the target area, avoiding global color drift (e.g., the issue in IN2N where editing the tire color alters the color of the entire car).

3. **Progressive 3D Scene Expansion**: Starting from a single frame, 3DGS is initialized with monocular depth. For each new frame, the existing Gaussian parameters are first frozen to learn the SE-3 transformation and estimate relative poses (Eq. 10), and then all parameters are unlocked for global optimization and densification. **Pyramidal Feature Scoring** is used to record the anchor status of the KEA Gaussians, and an intra-point-cloud loss is applied to constrain the consistency between newly added Gaussians and anchors, repairing residual 2D editing inconsistencies.

### Loss & Training
The total loss consists of four terms:

$$\mathcal{L}_T = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{KEA}\mathcal{L}_{KEA} + \lambda_{ipc}\mathcal{L}_{ipc} + \lambda_{pc}\mathcal{L}_{pc}$$

- $\mathcal{L}_{rgb} = (1-\gamma)\mathcal{L}_1 + \gamma\mathcal{L}_{\text{D-SSIM}}$: Standard photometric loss
- $\mathcal{L}_{KEA} = \lambda_{BCE}\mathcal{L}_{BCE} + \lambda_{JSD}\mathcal{L}_{JSD}$: KEA identity loss, including 2D binary cross-entropy and a 3D Jensen-Shannon divergence regularization (constraining the identity vectors of k-nearest neighbor Gaussians to be similar)
- $\mathcal{L}_{ipc}$: Intra-point-cloud loss in the pyramid, which is the weighted MSE between anchors and current Gaussian parameters
- $\mathcal{L}_{pc}$: Chamfer distance regularization for pose estimation

## Key Experimental Results

| Dataset | Metric | 3DEgo (Ours) | IN2N | IP2P+COLMAP |
|--------|------|-------------|------|-------------|
| Average of 6 Datasets | CTIS↑ | **Best** | Second Best | Third |
| Average of 6 Datasets | CDCR↑ | **Best** | Second Best | Third |
| Average of 6 Datasets | E-PSNR↑ | **Best** | Second Best | Third |
| GS25 Runtime | Minutes | **25min** | 285min | - |
| GS25 (Reconstruction Quality) | PSNR/SSIM/LPIPS | 27.86/0.90/0.18 | - | 23.87/0.79/0.23 |

Operational efficiency: Ours takes 25 minutes in total vs IN2N which requires COLMAP (13 min) + initialization (22 min) + editing (250 min) = 285 minutes, achieving an approximately **11x** speedup.

### Ablation Study
- **Removing $\mathcal{L}_{ipc}$ leads to the largest performance drop**: PSNR drops from 27.86 to 22.46, and SSIM drops from 0.90 to 0.78 — showing that pyramidal feature scoring is critical to suppressing unnecessary densification.
- Removing $\mathcal{L}_{KEA}$: PSNR drops to 26.73, showing a relatively minor impact — KEA is primarily used for local editing accuracy rather than reconstruction quality.
- Removing $\mathcal{L}_{pc}$: PSNR drops to 25.18 — pose regularization moderately contributes to global consistency.
- The IP2P+COLMAP baseline only achieves 23.87 PSNR — indicating that our method significantly outperforms direct COLMAP from the editing consistency perspective.

## Highlights & Insights
- **COLMAP-free Unified Framework**: For the first time, 3D editing is simplified from a three-stage "pose estimation $\rightarrow$ initialization $\rightarrow$ iterative editing" process to a single-stage "edited frames $\rightarrow$ direct reconstruction" pipeline, offering a 10x speedup.
- **Autoregressive Noise Blender** is a clever training-free multi-view consistency scheme: without training new models, it achieves editing consistency merely by weighted-averaging the noise predictions of adjacent frames. This concept can be transferred to any scenario requiring multi-view consistent diffusion genaration.
- **Elegant and Simple KEA Identity Vector Design**: Adding a 2D vector to each Gaussian point achieves foreground/background separation, which, combined with JSD regularization, ensures consistency in 3D space.
- Supports 360-degree videos and casually captured videos, offering high practicality.

## Limitations & Future Work
- **Total reliance on IP2P's editing capabilities**: The inherent limitations of IP2P propagate directly to the 3D editing results; for example, it struggles with fine-grained local edits (such as changing only the car window color) (see Fig. 7).
- **Risk of error accumulation in autoregressive editing**: Editing errors in earlier frames propagate to subsequent frames through noise blending, which may lead to gradual drift in long videos.
- **KEA only supports binary classification** (foreground/background), making it incapable of handling scenarios where multiple edited areas require different edits.
- Pose estimation accuracy might be inferior to COLMAP, particularly in scenes with large view changes or sparse textures.
- Future directions: (1) replacing IP2P with stronger editing models (such as InstructDiffusion); (2) extending KEA to multi-class support to enable multi-region editing; (3) introducing global attention consistency to replace autoregressive sequential dependency.

## Related Work & Insights
- **vs IN2N (Instruct-NeRF2NeRF)**: IN2N requires COLMAP poses + original scene initialization + frame-by-frame iterative editing (285 min), and suffers from poor editing locality (prone to global color changes). Ours requires neither COLMAP nor initialization (25 min) and achieves precise localized editing via KEA. However, IN2N might yield better geometric quality on standard scenes with existing COLMAP poses.
- **vs Gaussian Grouping**: Gaussian Grouping focuses on grouped editing and object removal, but it requires COLMAP poses and suffers from poor inpainting quality (many artifacts) after removal. Ours yields better results for object removal tasks when coupled with LAMA inpainting.
- **vs DATENeRF / GaussCtrl**: These methods also attempt to address multi-view editing consistency but all depend on COLMAP poses. Our noise blender is a more lightweight consistency scheme and eliminates COLMAP dependency.

### Insights & Connections
- The design of KEA identity (attaching semantic attributes to Gaussian points and regularizing with JSD) can be generalized to open-vocabulary 3D understanding scenarios.
- The unified framework of COLMAP-free reconstruction plus editing provides a paradigm reference for "end-to-end 3D content creation".

## Rating
- Novelty: ⭐⭐⭐⭐ For the first time, simplifying 3D editing into a single-stage, pose-free pipeline. Both the noise blender and KEA identity designs are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons across 200 edits on 6 datasets; however, it lacks user studies and perceptual evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, well-organized methodology descriptions, and intuitive figure/table presentations.
- Value: ⭐⭐⭐⭐ Significantly lowers the threshold of 3D editing and holds high value for practical applications, although restricted by the ceiling of IP2P editing quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[ECCV 2024\] DATENeRF: Depth-Aware Text-based Editing of NeRFs](datenerf_depth-aware_text-based_editing_of_nerfs.md)
- [\[ICCV 2025\] 3D Mesh Editing using Masked LRMs](../../ICCV2025/3d_vision/3d_mesh_editing_using_masked_lrms.md)

</div>

<!-- RELATED:END -->
