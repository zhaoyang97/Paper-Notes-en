---
title: >-
  [Paper Note] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation
description: >-
  [CVPR 2025][3D Vision][Human Mesh Recovery] This paper proposes BLADE, which decouples perspective projection parameters by accurately estimating the pelvic depth $T_z$ of the human body, recovers the human mesh using a $T_z$-aware pose estimator, and finally solves for the focal length and XY translation via differentiable rasterization. It realizes, for the first time, accurate perspective projection parameter and 3D human mesh recovery from a single image without relying o…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Perspective Projection Parameter Estimation"
  - "Depth Estimation"
  - "Close-range Human Reconstruction"
  - "SMPL-X"
date: 2026-05-08
content_hash: b0a69e2d6e0bbfdc
---

# BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2412.08640](https://arxiv.org/abs/2412.08640)  
**Code**: [Project Page](https://research.nvidia.com/labs/amri/projects/blade/)  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Perspective Projection Parameter Estimation, Depth Estimation, Close-range Human Reconstruction, SMPL-X

## TL;DR

This paper proposes BLADE, which decouples perspective projection parameters by accurately estimating the pelvic depth $T_z$ of the human body, recovers the human mesh using a $T_z$-aware pose estimator, and finally solves for the focal length and XY translation via differentiable rasterization. It realizes, for the first time, accurate perspective projection parameter and 3D human mesh recovery from a single image without relying on orthogonal camera heuristic assumptions.

## Background & Motivation

**Background**: Single-image human mesh recovery (HMR) is a classic problem in 3D vision, involving the simultaneous estimation of human shape, pose, and camera parameters from 2D images. Existing methods (e.g., HMR, CLIFF, AiOS, TokenHMR) usually assume weak-perspective projection—assuming that the person is far enough from the camera, and the focal length can be heuristically estimated (e.g., derived from image resolution or fixed as a constant of 5000).

**Limitations of Prior Work**: (1) The weak-perspective assumption completely fails on close-range images—when a person is close to the camera, perspective distortion is significant, and the orthogonal model cannot represent it; (2) Existing methods cannot simultaneously achieve accurate 3D pose and 2D alignment—TokenHMR pointed out that improving 2D alignment deteriorates 3D accuracy, and vice versa; (3) Methods claiming to support perspective projection (such as ZOLLY) still rely on heuristic formulas to convert orthogonal parameters to perspective parameters, and these approximations are severely inaccurate at close ranges.

**Key Challenge**: Recovering all parameters (shape $\beta$, pose $\theta$, focal length $f$, 3D translation $T_x, T_y, T_z$) from a single image is highly under-determined. Existing methods reduce the number of unknowns through orthogonal assumptions, but sacrifice accuracy for close-range images. How can these variables be decoupled in stages without making orthogonal assumptions?

**Goal**: To establish a fully perspective projection-based HMR pipeline that can handle various images from far to close ranges, while simultaneously achieving accurate 3D pose, 2D alignment, and perspective parameter recovery.

**Key Insight**: The authors discover a key fact that has been widely misunderstood—**perspective distortion is determined by $T_z$ (the Z-distance of the human to the camera), not the focal length $f$**. The focal length only affects scaling, whereas $T_z$ nonlinearly affects projection distortion (especially varying drastically when $T_z < 1.2$m). Therefore, $T_z$ can be reliably estimated from the level of distortion in the image.

**Core Idea**: A three-stage decoupling—first estimating $T_z$ from the image (since distortion is observable), then conditioning pose estimation on $T_z$ (since distortion affects pose appearance), and finally solving for $f, T_x, T_y$ via differentiable rasterization (since they degenerate into alignment parameters when $T_z$ is known).

## Method

### Overall Architecture

BLADE is a three-stage pipeline: (1) Pelvis depth estimator $F_{T_z}$ estimates $T_z$ from the cropped image $I_{crop}$; (2) $T_z$-aware pose estimator $F_{pose}$ estimates SMPL-X parameters $(\beta, \theta)$ from the original full image $I$ and $T_z$; (3) Camera solver optimizes $(f, T_x, T_y)$ via differentiable rasterization to align the rendered mesh with the human segmentation mask. The input is a single image containing a human, and the outputs are the SMPL-X Mesh parameters and the full perspective projection parameters.

### Key Designs

1. **Pelvis Depth Estimator ($F_{T_z}$)**:

    - **Function**: Directly estimates the Z-distance from the pelvis to the camera from the cropped human image.
    - **Mechanism**: Utilizes pre-trained Depth Anything V2 (DAv2) as the backbone to extract image appearance features, followed by a learnable ConvNet + Transformer Head to regress the $T_z$ value. The key training strategy is to use a weighted $L_1$ loss $L_{depth} = \frac{1}{T_z^{GT}} \cdot \|T_z - T_z^{GT}\|_1$—errors on close-range samples are weighted more heavily, because perspective distortion nonlinearly amplifies at close ranges (the derivative of $1/T_z$ is very large at small $T_z$). To address the lack of close-range samples in existing datasets, the authors created a synthetic dataset Bedlam-cc (2 million images, 80% of samples having $T_z \in [0.3, 1.2]$m). Ablation studies show that DAv2 is the optimal backbone ($E_{T_z}=15.4$cm on SPEC-MTP), outperforming DINOv2 (30cm) and Sapiens (21cm).
    - **Design Motivation**: Perspective distortion is an observable signal of $T_z$—a person close up displays a distinct "large head, small feet" effect. Recent advances in monocular depth estimation (DAv2) provide a strong prior for accurately estimating $T_z$. The weighted loss ensures the model achieves the highest accuracy in the most critical close-up range.

2. **$T_z$-aware Pose Estimator ($F_{pose}$)**:

    - **Function**: Estimates more accurate SMPL-X shape and pose parameters under the condition of a known $T_z$.
    - **Mechanism**: Adopts a ControlNet-style architecture to inject $T_z$ information into the pre-trained AiOS pose estimator. The original AiOS backbone is frozen, and a trainable copy of it is created. The copy's output is added to the frozen backbone's output after passing through a zero-initialized MLP. $T_z$ is encoded into depth features via two MLPs and injected into the encoder features of the trainable backbone. The training loss includes shape loss $L_{shape} = L_1(\beta, \beta^{GT})$, pose angle error $L_{pose} = E_{ang}(\theta, \theta^{GT})$, joint position loss $L_{joint} = L_1(J, J^{GT})$, and vertex loss $L_{vert} = L_1(V, V^{GT})$, with weight allocations of $w_{shape}=1, w_{pose}=1, w_{joint}=5, w_{vert}=5$.
    - **Design Motivation**: The image appearance of the same person in the exact same pose varies significantly at different $T_z$ due to different perspective distortions. Without informing the pose estimator of "how far the person is from the camera," it confuses distortion with pose variation. The advantage of a ControlNet architecture is that the zero-initialized MLP ensures that the original performance of AiOS is not disrupted during the initial training phase, and it gradually learns to utilize the $T_z$ information. Ablations show that directly fine-tuning AiOS degrades performance (PVE increases from 110.9 to 120.6), whereas ControlNet-style $T_z$ conditioning significantly improves it (PVE drops to 99.6).

3. **Differentiable Rasterization Camera Solver**:

    - **Function**: Recovers the focal length $f$ and XY translation $T_x, T_y$ from $T_z$ and the mesh parameters.
    - **Mechanism**: Once $T_z$ is known and $(\beta, \theta)$ of the mesh is estimated, $(f, T_x, T_y)$ essentially act as alignment parameters—$T_x, T_y$ control the position of the mesh on the image plane, and $f$ controls projection scale. Initializing $T=[0,0,T_z]$ and $f_{init}=h$ (image height), the SMPL-X mesh is rendered into a binary mask through differentiable rasterization, and then $(f, T_x, T_y)$ are optimized to maximize the IoU between the rendered mask and the human mask provided by an off-the-shelf segmenter. Gaussian smoothing is applied to both masks to ensure global gradient flow. The optimization process also simultaneously fine-tunes $T_z$ and the global orientation to further improve quality.
    - **Design Motivation**: This step transforms camera parameter estimation into a mask alignment optimization problem—which is much more robust than directly regressing $(f, T_x, T_y)$ because it leverages clear geometric constraints instead of statistical correlation. Differentiable rasterization makes the entire optimization process end-to-end differentiable, which is a key enabling technology for joint optimization.

### Loss & Training

Two-stage training: Stage 1 trains $F_{T_z}$ (128 batch size, 8×A100, 4 epochs) using the weighted $L_1$ depth loss; Stage 2 freezes $F_{T_z}$ and trains $F_{pose}$ (336 batch size, 48×A100, 4 epochs) using a combination of the four losses. The camera solver does not require training. Training data includes H36M, PDHuman, HuMMan, and the self-built Bedlam-cc dataset.

## Key Experimental Results

### Main Results

| Method | SPEC-MTP $E_{T_z}$↓ | SPEC-MTP PVE↓ | SPEC-MTP mIoU↑ | Bedlam-cc $E_{T_z}$↓ | Bedlam-cc mIoU↑ |
|------|------------|---------|----------|------------|----------|
| ZOLLY | 0.899 | 126.7 | 62.3 | 0.539 | 51.8 |
| AiOS* | 1.035 | 110.9 | 48.7 | 2.340 | 54.6 |
| TokenHMR* | 0.909 | 124.3 | 49.7 | 2.378 | 54.2 |
| SMPLer-X* | 0.980 | 102.6 | 53.0 | 2.057 | 53.0 |
| **BLADE** | **0.129** | **111.9** | **68.7** | **0.326** | **74.6** |
| **BLADE (real-world)** | **0.127** | **99.6** | **69.5** | **0.325** | **75.0** |

### Ablation Study

| Configuration | SPEC-MTP PA-MPJPE↓ | SPEC-MTP PVE↓ | Description |
|------|---------------------|---------------|------|
| raw AiOS | 62.8 | 110.9 | Original AiOS pre-trained model |
| ft. AiOS | 64.9 | 120.6 | Direct fine-tuning of AiOS, performance degrades instead |
| **BLADE ($T_z$ cond.)** | **56.7** | **99.6** | ControlNet-style $T_z$ conditioning |

| Depth Backbone | SPEC-MTP $E_{T_z}$(m)↓ |
|------|----------|
| DINOv2 | 0.300 |
| Sapiens | 0.210 |
| DAv2 | 0.154 |
| **BLADE (DAv2+Bedlam-cc)** | **0.127** |

### Key Findings

- **$T_z$ estimation accuracy improved by ~7 times**: On SPEC-MTP, BLADE's $E_{T_z}=0.127$m is significantly better than ZOLLY's 0.899m (an 85.9% improvement), proving that directly estimating $T_z$ is far more accurate than heuristic conversion from orthogonal parameters.
- **Significant lead in 2D alignment**: BLADE achieves a 69.5% mIoU on SPEC-MTP, whereas the second-best, ZOLLY, only achieves 62.3%—a relative improvement of 11.6%. The gap is even larger on Bedlam-cc (75.0% vs. 54.6%), showing that accurate perspective parameters are crucial for 2D alignment.
- **$T_z$ conditioning is key to pose estimation**: Directly fine-tuning AiOS causes it to overfit to the small-scale close-range dataset and lose generalizability (PVE rises from 110.9 to 120.6), whereas ControlNet-style $T_z$ injection preserves AiOS's generalizability while learning distortion information (PVE drops to 99.6).
- **Value of the Bedlam-cc dataset**: Integrating synthetic close-range data reduced $E_{T_z}$ from 15.4cm to 12.7cm, demonstrating the importance of close-range training data.
- **Finding that focal length does not affect distortion**: While mathematically simple, this geometric fact has been neglected by the HMR community for a long time. Correcting this misunderstanding makes parameter decoupling possible.

## Highlights & Insights

- **Correcting a long-standing misconception**: Clearly demonstrates that **focal length is merely a scaling factor, and $T_z$ is the sole source of distortion**—this seemingly simple cognitive correction directly inspires the entire method design. In research, challenging "default-accepted assumptions" often leads to breakthrough progress.
- **Elegant three-stage decoupled design**: Decomposes a highly under-determined joint estimation problem into three well-defined sub-problems—first estimating $T_z$ via observable signals, then conditioning the pose estimation on $T_z$, and finally converting the remaining parameters into an alignment optimization. Each step has clear geometric/physical justification. This decoupling approach can be applied to other ill-posed joint estimation problems involving multiple variables.
- **ControlNet-style knowledge preservation strategy**: Zero-initialized MLPs ensure that the pre-trained knowledge is not disrupted, while allowing the injection of new conditional information. The superior experimental result compared to direct fine-tuning (99.6 vs. 120.6) is robust empirical evidence.

## Limitations & Future Work

- **Single-person scene limitation**: The current method only processes one person at a time, requiring an external detector to handle multi-person scenes.
- **Lens distortion ignored**: Assumes a standard pinhole camera model, making it inapplicable to non-standard cameras like fisheye lenses.
- **Reliance on segmentation masks**: The accuracy of the camera solver is capped by the quality of the off-the-shelf segmenter—optimization fails when the mask is severely inaccurate.
- **Heavy computational resource demands**: The training phase requires 48×A100 GPUs, which is not highly accessible to academia.
- **Future directions**: Extending to video sequences to utilize temporal information; learning a differentiable camera solver to replace rasterization optimization for improved robustness; handling multi-person scenes; and incorporating lens distortion models.

## Related Work & Insights

- **vs ZOLLY**: ZOLLY also estimates $T_z$, but still relies on the heuristic formula $f = s \cdot h \cdot T_z / 2$ to translate orthogonal parameters into perspective ones. BLADE completely discards heuristics, directly solving via differentiable rasterization—resulting in a 7x accuracy improvement in $T_z$ and an 11+ percentage point improvement in mIoU.
- **vs TokenHMR**: TokenHMR identifies the trade-off between 2D alignment and 3D accuracy and proposes the TALS loss function to balance them, yet still operates within the orthogonal framework. BLADE fundamentally resolves this trade-off through accurate perspective modeling—simultaneously achieving the best 2D and 3D accuracy.
- **vs AiOS**: BLADE uses AiOS as the backbone for pose estimation, injecting $T_z$ information through a ControlNet-style architecture. This greatly improves close-range accuracy without damaging the powerful generalizability of AiOS, serving as an excellent paradigm for "making incremental improvements on existing strong models."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core insight (focal length does not affect distortion, $T_z$ decoupling) is mathematically simple but has been long overlooked, and the three-stage design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 datasets, 9+ metrics, comparisons with multiple SOTA methods, comprehensive ablations, real-world qualitative evaluations, and self-built datasets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear chain of logic (Discovery $\rightarrow$ Validation $\rightarrow$ Design $\rightarrow$ Experiments), exquisite figures, and extremely detailed supplementary materials.
- **Value**: ⭐⭐⭐⭐⭐ Thoroughly addresses the long-standing issue of close-range HMR. Produced by NVIDIA with open-source code, it has direct value for applications such as video conferencing and AR/VR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](relative_pose_estimation_through_affine_corrections_of_monocular_depth_priors.md)
- [\[CVPR 2025\] TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features](tritex_learning_texture_from_a_single_mesh_via_triplane_semantic_features.md)
- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](../../ECCV2024/3d_vision/multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
