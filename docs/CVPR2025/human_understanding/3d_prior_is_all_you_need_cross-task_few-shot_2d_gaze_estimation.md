---
title: >-
  [Paper Note] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation
description: >-
  [CVPR 2025][Human Understanding][Gaze Estimation] This work proposes cross-task few-shot 2D gaze estimation, which leverages a pre-trained 3D gaze model as a prior. Through a **physics-based differentiable projection module** (with 6 learnable screen parameters), the 3D gaze direction is projected onto 2D screen coordinates. With only 10 annotated images, this approach adapts 2D gaze estimation to unseen devices, achieving over 25% improvement on MPIIGaze/EVE/GazeCapture comp…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Gaze Estimation"
  - "Cross-Task"
  - "Few-shot"
  - "Differentiable Projection"
  - "3D-to-2D"
date: 2026-05-08
content_hash: 9c6f198e1e243e1c
---

# 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation

**Conference**: CVPR 2025  
**arXiv**: [2502.04074](https://arxiv.org/abs/2502.04074)  
**Code**: www.yihua.zone/work/gaze322  
**Area**: LLM Pre-training  
**Keywords**: Gaze Estimation, Cross-Task, Few-shot, Differentiable Projection, 3D-to-2D  

## TL;DR
This work proposes cross-task few-shot 2D gaze estimation, which leverages a pre-trained 3D gaze model as a prior. Through a **physics-based differentiable projection module** (with 6 learnable screen parameters), the 3D gaze direction is projected onto 2D screen coordinates. With only 10 annotated images, this approach adapts 2D gaze estimation to unseen devices, achieving over 25% improvement on MPIIGaze/EVE/GazeCapture compared to EFE and IVGaze.

## Background & Motivation
3D gaze estimation and 2D gaze estimation are traditionally treated as two independent research directions. 3D methods estimate the gaze direction vector in the camera coordinate system, which is independent of specific devices. In contrast, 2D methods directly predict screen pixel coordinates, but since they are coupled with device-specific factors such as screen size and camera-to-screen poses, they struggle to generalize across different devices. Traditional eye-tracking systems achieve precise 2D gaze estimation using 3D eyeball models combined with calibration. However, within the deep learning paradigm, this "3D estimation followed by projection to 2D" pipeline remains unexplored.

## Core Problem
How can a pre-trained 3D gaze model be rapidly adapted to 2D gaze estimation on an unseen device using only a few (10) images with 2D annotations? The challenges lie in: (1) the domain gap between 3D and 2D gaze; (2) unknown screen poses (without screen calibration); and (3) extremely limited training data.

## Method

### Overall Architecture
2D gaze estimation is decomposed into two steps: **3D gaze estimation** + **gaze projection**. $\mathcal{H}_{2D}(\mathbf{I}) = \mathcal{P}(\mathcal{H}_{3D}(\mathbf{I}), \mathbf{o})$, where the projection module $\mathcal{P}$ models the screen pose with 6 learnable parameters (rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$).

### Key Designs
1. **Differentiable Projection Module**: Defines the screen plane normal vector $\mathbf{n} = \mathbf{R}[:,2]$ and a point $\mathbf{t}$ on the screen. It computes the intersection point $\mathbf{p}_{3D}$ of the 3D gaze direction $\mathbf{g}$ pointing from the face center $\mathbf{o}$ with the screen plane, which is then transformed into screen coordinates to obtain the 2D pixel coordinates. The entire process is fully differentiable, requiring the learning of only 6 parameters ($\mathbf{r}, \mathbf{t}$) without any screen calibration. The rotation vector $\mathbf{r}$ is converted to a rotation matrix $\mathbf{R}$ via Rodrigues' formula to guarantee valid rotation constraints.

2. **Dynamic Pseudo-labeling Strategy**: Flip augmentation is highly challenging for 2D gaze labels due to its dependency on the unknown screen pose. The core idea is to perform **inverse projection** (2D$\rightarrow$3D) using learnable screen parameters, flip the gaze in the 3D space, and project it back to 2D. However, fine-tuning deviates the 3D model from the camera coordinate system to an unknown coordinate system, rendering pseudo-labels unreliable. Solution: Learn a dynamic transformation $\mathcal{T}$ by aligning the outputs of the initial 3D model and the fine-tuned model via SVD, thereby mapping the unknown coordinate system back to the camera coordinate system.

3. **Uncertainty Minimization**: Minimizes the prediction variance on color-jittered augmented images to enhance robustness.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{proj} + 0.4\mathcal{L}_{flip} + 0.25\mathcal{L}_{unc}$$
- GazeTR (ResNet18 + 6-layer Transformer) pre-trained on Gaze360 is used as the 3D model.
- 10 training samples per subject, 80 epochs, RTX 3090, training takes only ~1.2 minutes.

## Key Experimental Results

### Cross-Task Few-shot 2D Gaze Estimation (10-shot, mm error $\downarrow$)

| Method | EVE (Desktop) | MPIIGaze (Laptop) | GazeCapture (Phone) |
|------|---------|------------|-------------|
| IVGaze | 228.2 | 106.2 | 101.2 |
| EFE | 60.3 | 73.9 | 48.9 |
| **Ours** | **43.4** | **56.7** | **35.7** |

Ours improves over EFE by 28%+, and outperforms IVGaze significantly (IVGaze suffers from underfitting in 10-shot scenarios due to its use of transformers). It is consistently effective across three different device types (desktop monitor, laptop, and phone), demonstrating device independence. Notably, the desktop monitor (EVE) scenario is the most challenging due to the large screen size and long camera distance, but the error still reaches 43.4 mm.

### Performance Variation with Different Shot Numbers

| No. of Shots | EVE(mm) | MPIIGaze(mm) | GazeCapture(mm) |
|--------|---------|-------------|----------------|
| 5 | ~55 | ~65 | ~42 |
| 10 | 43.4 | 56.7 | 35.7 |
| 50 | ~38 | ~48 | ~30 |

Performance scales approximately logarithmically with the number of shots, with 10-shot representing an excellent sweet spot.

### Reference: Fully Supervised 2D Methods

| Method | MPIIGaze | GazeCapture |
|------|----------|-------------|
| Full-Face (Fully supervised) | 42.3 | - |
| iTracker (Fully supervised) | - | 26.8 |
| **Ours (10-shot)** | **56.7** | **35.7** |

Using only 10 images, the performance is remarkably close to fully supervised 2D methods!

### Ablation Study
- **Projection Module Only**: EVE 46.6 (significantly outperforming Direct Learning at 180.6)
- **+ Pseudo-labels**: EVE 45.3 (-1.3); flip augmentation is performed in the 3D space via inverse projection and then projected back to 2D
- **+ Uncertainty**: EVE 43.4 (-3.2 total); minimizing the prediction variance of color-jittered augmented images enhances robustness
- **Without transformation $\mathcal{T}$**: EVE 88.1 (collapses due to unreliable pseudo-labels), verifying the necessity of dynamic coordinate system alignment

## Highlights & Insights
- **Pioneering Task Formulation**: Formulates cross-task adaptation of 3D-to-2D gaze estimation systematically without requiring screen calibration for the first time.
- **Physics-Driven Design**: The differentiable projection is based on geometric optics principles with only 6 parameters, making it highly interpretable.
- **Extremely Efficient**: Requires only 10 images and ~1.2 minutes of training to deploy to a new device, making it highly suitable for quick calibration scenarios.
- **Coordinate Alignment for Dynamic Pseudo-labeling**: Elegantly addresses the coordinate drift problem during fine-tuning by learning a dynamic transformation $\mathcal{T}$ via SVD.

## Limitations & Future Work
- When there is no valid face in the input image, the mathematical projection can yield extreme values (e.g., when the gaze is nearly parallel to the screen).
- A performance gap still exists between 10-shot and fully supervised methods (56.7 vs 42.3 on MPIIGaze), and the utilization of unlabelled data remains unexplored.
- Screen PPI needs to be known (though typically accessible).
- A performance gap remains compared to methods with known screen poses (43.4 vs 39.4 on EVE); pose estimation is still a bottleneck.
- The SVD alignment for dynamic pseudo-labeling assumes that the coordinate drift after fine-tuning can be approximated by an affine transformation, which may fail in cases of large drift or non-linear distortion.
- Only validated on GazeTR (ResNet18 + Transformer). Generalizability to other 3D gaze networks remains unknown, as accuracy differences across various 3D backbones might affect projection quality.

## Related Work & Insights
- **EFE**: Employs projection but requires screen calibration. The proposed learnable parameters + calibration-free setup is more practical.
- **IVGaze**: Refines projection points using a transformer, but severely underfits in 10-shot scenarios.
- **Traditional Calibration Methods**: Require professional equipment. The proposed method can substitute them using only 10 images.
- **Direct Learning Baseline**: Fine-tuning directly with 3D gaze labels for 2D prediction leads to a high error of 180.6 mm on EVE, demonstrating that cross-dimensional domain gaps cannot be bridged by simple fine-tuning.

## Related Work & Insights
- The "3D prior + differentiable projection to 2D" paradigm can be extended to other cross-dimensional tasks (e.g., 3D pose $\rightarrow$ 2D action, 3D object $\rightarrow$ 2D detection).
- The SVD-based dynamic coordinate alignment can be applied to other feature space drift issues during fine-tuning.
- This method essentially re-implements the traditional eye-tracking paradigm of "3D eyeball model + calibration" using deep learning, converting explicit calibration parameterization into learnable parameters, making it a classic example of combining traditional and deep learning approaches.
- The minimal design of the differentiable projection module with only 6 parameters inspires a "few-parameter adaptation" mindset: cross-domain transfer does not require modifying the entire network, but only a physics-driven, lightweight bridging module.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering task formulation, elegant design of differentiable projection and dynamic pseudo-labeling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets (laptop/desktop/phone) + rich ablation studies + robustness + visualization of pseudo-label trajectories.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation of the problem is exceptionally clear, with rigorous physical derivations and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ High practicality (can be deployed to new devices within 1 minute), though the gaze estimation field is somewhat niche.
- Reproducibility: ⭐⭐⭐⭐ Simple method with publicly available code, trainable on a single RTX 3090 GPU.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/human_understanding/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[NeurIPS 2025\] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](../../NeurIPS2025/human_understanding/a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)

</div>

<!-- RELATED:END -->
