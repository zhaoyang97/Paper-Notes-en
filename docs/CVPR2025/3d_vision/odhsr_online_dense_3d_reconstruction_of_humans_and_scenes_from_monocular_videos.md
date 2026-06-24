---
title: >-
  [Paper Note] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos
description: >-
  [CVPR 2025][3D Vision][Human Reconstruction] ODHSR proposes the first unified framework to simultaneously perform online camera tracking, human pose estimation, and joint human-scene dense reconstruction from monocular RGB videos. Based on 3D Gaussian Splatting, it achieves a speedup of up to 75x compared to offline methods while matching or exceeding SOTA reconstruction quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Reconstruction"
  - "Scene Reconstruction"
  - "3D Gaussian Splatting"
  - "SLAM"
  - "Monocular Video"
date: 2026-05-08
content_hash: 8b30596b190937ac
---

# ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos

**Conference**: CVPR 2025  
**arXiv**: [2504.13167](https://arxiv.org/abs/2504.13167)  
**Code**: [https://eth-ait.github.io/ODHSR](https://eth-ait.github.io/ODHSR)  
**Area**: 3D Vision  
**Keywords**: Human Reconstruction, Scene Reconstruction, 3D Gaussian Splatting, SLAM, Monocular Video

## TL;DR

ODHSR proposes the first unified framework to simultaneously perform online camera tracking, human pose estimation, and joint human-scene dense reconstruction from monocular RGB videos. Based on 3D Gaussian Splatting, it achieves a speedup of up to 75x compared to offline methods while matching or exceeding SOTA reconstruction quality.

## Background & Motivation

**Background**: Reconstructing the 3D world from monocular videos remains a core problem in computer vision. Existing methods either focus solely on human reconstruction (e.g., 3DGS-Avatar, GauHuman) or exclusively on scene reconstruction (such as SLAM methods like MonoGS). The few approaches that simultaneously handle both humans and scenes (e.g., HSR, HUGS) either require pre-calibrated camera and human poses or demand days of training time.

**Limitations of Prior Work**: HSR is based on implicit neural fields (NeRF/SDF), leading to extremely slow training (on the order of days) and limited rendering quality. Although HUGS utilizes 3D Gaussian Splatting, its triplane scheme exhibits slow feature convergence under online settings, making it unsuitable for real-time processing. Vid2Avatar suffers from poor scene reconstruction and is prone to numerous artifacts. Most importantly, these methods are offline and cannot meet the real-time demands of applications like robotics.

**Key Challenge**: Under an online setting, how to decouple camera motion, human body movement, and reconstruct a high-fidelity joint human-scene representation simultaneously from merely a monocular RGB video? Factors such as sparse viewpoints, human occlusion, dynamic clothing, and lighting variations make this problem highly challenging.

**Goal**: To design a unified online framework that simultaneously outputs: (1) camera trajectory, (2) global human pose, and (3) dense photometric reconstruction of both the human and the scene, without relying on any pre-calibration information.

**Key Insight**: 3D Gaussian Splatting provides direct explicit gradient flow, allowing joint optimization of Gaussian parameters and pose parameters. Concurrently, monocular depth priors and SMPL human priors are utilized to establish spatial associations between the human and the scene.

**Core Idea**: To extend 3DGS-based SLAM into a joint human-scene framework, which performs tracking, pose estimation, and dense reconstruction simultaneously in an online pipeline through occlusion-aware human silhouette rendering and monocular geometric priors.

## Method

### Overall Architecture

The input is a monocular RGB video sequence. For each frame, the system first renders the predicted image of the current frame using the existing Gaussian representation and calculates the residual to jointly optimize camera and human poses (Tracking thread). After keyframe filtering, the Mapping thread jointly trains the 3D Gaussian representations of the human and the scene using a local keyframe window. Finally, global Bundle Adjustment is executed across all keyframes to fine-tune the overall representation. The system consists of two parallel threads, Tracking and Mapping, to ensure efficiency.

### Key Designs

1. **3D Avatar Representation (Rigid + Non-rigid Deformation)**:

    - **Function**: To represent the human body using 3D Gaussians in the canonical space and transform them into world coordinates via SMPL-driven skeletal deformation.
    - **Mechanism**: Each Gaussian has a position, offset, rotation, scale, opacity, color, and learnable LBS weights. The deformation is decomposed into a rigid part (LBS driven by SMPL joints) and a non-rigid part (time/pose-dependent local deformation). The non-rigid deformation is modeled by a multi-resolution hash encoding network $F_\phi$, which takes the Gaussian centers, timestamps, and pose parameters as input, and outputs the local displacement $\Delta\mu'_H$, rotation offset $\Delta R_H$, and ambient occlusion factor $\Delta c_H$. Then, the canonical space Gaussians are transformed into world coordinates via the LBS-weighted joint transformation $P = \sum_j W_{H,j} M_j$.
    - **Design Motivation**: Pure skeletal deformation can only model rigid joint movements, failing to handle dynamic deformations such as clothing wrinkles. The non-rigid module employs a hash encoding network instead of a large MLP to balance expressiveness and training speed.

2. **Occlusion-Aware Human Silhouette Rendering**:

    - **Function**: To correctly handle the occlusion relationship between the human and the scene in joint rendering, generating realistic human silhouettes.
    - **Mechanism**: Human and scene Gaussians are merged into a global set $G = G_S + G_H$ and fed into the rasterizer. Since Gaussians are sorted by depth, occlusions are handled naturally. The human silhouette is defined as $\hat{O}_H = \sum_j \alpha_j \prod_{k=1}^{N_j}(1-\alpha_k)$, where $N_j$ contains all Gaussians (including scene Gaussians) whose depth is smaller than that of the $j$-th human Gaussian. Consequently, when a scene object is in front of the human, the human silhouette is correctly occluded.
    - **Design Motivation**: Simply rendering a human mask ignores scene occlusions, leading to failed human-scene decoupling. Occlusion-aware rendering provides more accurate silhouette supervision, effectively guiding the separation of human and scene Gaussians.

3. **Joint Pose Optimization (Tracking)**:

    - **Function**: To simultaneously estimate the camera pose $T$ and human pose $\theta$ for each frame.
    - **Mechanism**: Holding the Gaussian representations fixed, joint optimization is performed using multiple complementary losses. These include RGB photometric loss, optical flow loss (used to avoid local minima of the RGB loss, computed only on static regions), monocular depth consistency loss (aligning rendered depth and predicted depth in the inverse depth space), human silhouette loss, and 2D keypoint loss. The final objection is the weighted sum of all terms: $L_{pose} = \lambda_{rgb}L_{rgb} + \lambda_{flow}L_{flow} + \lambda_{disp}L_{disp} + \lambda_{sil}L_{sil} + \lambda_{kp}L_{kp}$.
    - **Design Motivation**: A single loss is insufficient to constrain such a high-dimensional pose space. Optical flow provides inter-frame motion constraints, monocular depth provides geometric priors, and silhouettes combined with keypoints offer human-specific constraints. These complementary signals guarantee convergence to the correct solution.

### Loss & Training

During the Mapping phase, three additional regularizations are introduced: (1) LBS weight regularization $L_{LBS}$, which supervises skinning weights with SMPL weights to prevent overfitting; (2) Canonical center regularization $L_{center}$, which prevents excessive Gaussian displacement; and (3) Local deformation regularization $L_{deform}$, which penalizes the deformation magnitude to stabilize training. Mapping utilizes a local keyframe window supplemented with two random historical keyframes to balance new observations with global memory.

Initialization Strategy: The initial human pose for the first frame is obtained via WHAM and refined using the 2D keypoint loss from ViTPose. Depths from the monocular depth estimator are aligned with the SMPL mesh depths using RANSAC to solve for scale and shift, thereby initializing the scene Gaussians with scaled depth.

## Key Experimental Results

### Main Results

| Dataset / Metric | ODHSR (Full) | HUGS | HSR | Vid2Avatar | 3DGS-Avatar |
|------------|-------------|------|-----|------------|-------------|
| EMDB Full-image PSNR↑ | **23.790** | 21.605 | 18.675 | 16.656 | - |
| EMDB Full-image SSIM↑ | **0.767** | 0.659 | 0.463| 0.413 | - |
| EMDB Human PSNR↑ | **28.955** | 26.165 | 25.127 | 24.258 | 27.952 |
| NeuMan Full-image PSNR↑ | **26.470** | 26.667 | 21.676 | 15.640 | - |
| Training FPS↑ | **0.141** | 0.042 | 0.002 | <0.001 | 0.112 |
| Rendering FPS↑ | **85** | 40 | 0.05 | 0.02 | 60 |

### Ablation Study

| Configuration | PSNR↑ | ATE RMSE↓ | WA-MPJPE↓ |
|------|-------|-----------|-----------|
| Full model | **23.790** | **0.084** | **175.215** |
| w/o $L_{flow}$ | 22.593 | 0.214 | 301.621 |
| w/o $L_{keypoint}$ | 22.263 | 0.121 | 230.875 |
| w/o $L_{disp}$ | 22.769 | 0.165 | 252.547 |
| w/o $L_{sil}$ | 22.648 | 0.148 | 240.838 |

### Key Findings

- Removing the optical flow loss impacts camera tracking the most (ATE increases from 0.084 to 0.214), as the inter-frame motion constraint is the core of tracking.
- Removing the keypoint loss affects the global human pose estimation the most (WA-MPJPE increases from 175 to 231), as keypoints provide the most direct signal for human alignment.
- Synergistic effects exist among the four losses: keypoints and silhouettes accelerate human pose convergence, accurate poses facilitate depth alignment, and clear geometry in turn improves optical flow and depth consistency.
- The reconstruction quality of this online method comprehensively surpasses the offline HSR (based on NeRF/SDF) and HUGS (based on triplanes), demonstrating the clear superiority of the explicit Gaussian representation combined with joint optimization.
- Human pose estimation achieves a WA-MPJPE of 175mm, significantly outperforming the WHAM initialization (636mm), which demonstrates that the post-reconstruction pose refinement is highly effective.

## Highlights & Insights

- **Pioneering Unified Online Framework**: Unifying three separate tasks—tracking, pose estimation, and reconstruction—into a single 3DGS-SLAM framework, achieving 75x faster training than HSR with superior quality. This system-level design methodology is highly referenceable.
- **Humans as a Scale Reference**: Using the known dimensions of the SMPL human body to calibrate the scale of monocular depth, cleverly resolving the scale ambiguity problem inherent in monocular systems. This trick can be transferred to any monocular reconstruction scenario with known object sizes.
- **Elegant Implementation of Occlusion-Aware Silhouettes**: Without requiring additional networks, it directly utilizes the depth-sorting characteristics of the Gaussian Splatting rasterizer to achieve correct occlusions, which is both efficient and natural.

## Limitations & Future Work

- It relies on pre-trained models like WHAM and ViTPose for initialization, which may perform poorly in scenarios where these models fail (e.g., severe occlusion, unconventional poses).
- Currently, only single-person scenes are supported; multi-person scenes require additional human detection and segmentation.
- The keyframe selection strategy relies on heuristic rules, which may omit highly informative frames.
- The hash encoding network for non-rigid deformation is sensitive to temporal extrapolation, potentially causing drift in long sequences.

## Related Work & Insights

- **vs HSR**: HSR utilizes implicit NeRF/SDF representations for joint human-scene modeling, requiring days of training time. In contrast, this work employs explicit Gaussian representations, resulting in training and rendering speeds that are several orders of magnitude faster, along with better reconstruction quality.
- **vs HUGS**: HUGS also employs 3DGS but uses a triplane to represent the human body, requiring pre-calibrated poses in an offline setting. This work runs online and uses a simpler canonical space Gaussian representation combined with hash deformation.
- **vs 3DGS-Avatar**: Focuses solely on human reconstruction and requires known camera poses. This work simultaneously addresses scene reconstruction and pose estimation, rendering it much more generalized.

## Rating

- Novelty: ⭐⭐⭐⭐ The first unified online framework, though the individual components (3DGS SLAM, SMPL deformation, monocular priors) are not entirely novel on their own.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons are conducted on two datasets with clear ablation studies, but lacks evaluation on multi-person scenes and larger-scale assessments.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with fluent descriptions of motivation and methodology, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐ Significantly promotes the real-time capability of joint human-scene reconstruction, providing excellent engineering reference value for system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](slam3r_real-time_dense_scene_reconstruction_from_monocular_rgb_videos.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)
- [\[CVPR 2025\] GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)

</div>

<!-- RELATED:END -->
