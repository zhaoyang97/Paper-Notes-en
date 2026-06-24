---
title: >-
  [Paper Note] UniK3D: Universal Camera Monocular 3D Estimation
description: >-
  [CVPR 2025][3D Vision][Universal camera depth estimation] Presents UniK3D, the first universal monocular 3D estimation method supporting arbitrary camera models (from pinhole to panorama). By employing a spherical 3D output space (radial distance instead of perpendicular depth) and a model-free camera ray representation based on spherical harmonics, it achieves zero-shot SOTA performance across 13 datasets, outperforming existing methods by a large margin…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Universal camera depth estimation"
  - "Spherical harmonics representation"
  - "Panoramic depth"
  - "Fisheye camera"
  - "Zero-shot 3D reconstruction"
date: 2026-05-08
content_hash: 60d42d527fcd85d4
---

# UniK3D: Universal Camera Monocular 3D Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.16591](https://arxiv.org/abs/2503.16591)  
**Code**: [GitHub](https://github.com/lpiccinelli-eth/unik3d)  
**Area**: 3D Vision  
**Keywords**: Universal camera depth estimation, Spherical harmonics representation, Panoramic depth, Fisheye camera, Zero-shot 3D reconstruction

## TL;DR

Presents UniK3D, the first universal monocular 3D estimation method supporting arbitrary camera models (from pinhole to panorama). By employing a spherical 3D output space (radial distance instead of perpendicular depth) and a model-free camera ray representation based on spherical harmonics, it achieves zero-shot SOTA performance across 13 datasets, outperforming existing methods by a large margin, especially in large field-of-view (FoV) and panoramic settings.

## Background & Motivation

- Existing monocular depth/3D estimation methods rely on oversimplified assumptions: pinhole camera models or rectified images.
- These limitations lead to severe performance degradation in real-world scenarios (fisheye, panoramic lenses) and result in a significant loss of contextual information.
- The output spaces (disparity or log-depth) of existing methods (such as DepthAnything, UniDepth) are mathematically ill-posed when the field of view exceeds 180°.
- Even if the models are exposed to various camera types during training, they still fail to learn universal camera estimation effectively due to inflexible camera assumptions.
- Methods requiring rectification parameters or known intrinsics are difficult to apply in zero-shot scenarios.
- Network predictions suffer from "distribution shrinkage": tending to regress to the most common narrow field-of-view patterns in the training data.
- There is a lack of a unified framework capable of handling arbitrary back-projection problems.
- The decoupling between camera parameters and scene geometry is insufficient.

## Method

### Overall Architecture

UniK3D consists of three modules: an **Encoder** (ViT-based) to extract dense features and class tokens; an **Angular Module** to predict 19 parameters (4 domain parameters + 15 spherical harmonic coefficients) from class tokens, reconstructing the camera ray pencil $\mathbf{C} = \theta || \phi$ via inverse spherical harmonic transform; and a **Radial Module** that uses a Transformer Decoder to condition encoder features on the angular representation to predict the log-radial distance $\mathbf{R}_{\log}$. The final 3D output $\mathbf{O} = \mathbf{C} || \mathbf{R}$ obtains the point cloud via spherical-to-Cartesian coordinate transformation.

### Key Designs

**Design 1: Spherical Output Space and Spherical Harmonics Camera Representation**
- **Function**: Unifies the handling of the back-projection problem for arbitrary camera geometries.
- **Mechanism**: The output 3D space uses a global spherical representation, employing radial distance (rather than perpendicular depth) to represent the scene range. Camera rays are represented as a linear combination of spherical harmonics: $\mathbf{C} = \sum_{l=0}^{L}\sum_{m=-l}^{l}\mathbf{H}_{lm}\mathcal{B}_{lm}(\theta,\phi)$. Only a 3rd-degree spherical harmonics (15 coefficients) + 4 domain parameters are required to accurately represent most camera types.
- **Design Motivation**: Traditional disparity/log-depth is mathematically ill-posed under large fields of view. The spherical representation ensures that the projected size of an object is uniquely related to the radial distance (rather than depth), making it easier to learn. The spherical harmonics basis provides inductive biases such as continuity and differentiability.

**Design 2: Asymmetric Angular Loss to Prevent Distribution Shrinkage**
- **Function**: Addresses the issue where network predictions bias towards narrow fields of view (the most frequent pattern in the training data).
- **Mechanism**: Uses an asymmetric L1 loss based on quantile regression: $\mathcal{L}_{\text{AA}}^{\alpha}(\hat{\theta}, \theta^*) = \alpha\sum_{\hat{\theta}>\theta^*}|\hat{\theta}-\theta^*| + (1-\alpha)\sum_{\hat{\theta}\leq\theta^*}|\hat{\theta}-\theta^*|$, using $\alpha=0.7$ for the polar angle $\theta$ (to penalize underestimation) and $\alpha=0.5$ for the azimuthal angle $\phi$ (symmetric).
- **Design Motivation**: Simple data rebalancing alters the diversity of 3D scenes. Quantile regression only requires searching for $\alpha$ within the $[0,1]$ interval, which is concise and highly efficient.

**Design 3: Enhanced Camera Conditioning Strategy**
- **Function**: Ensures the model effectively utilizes camera information instead of ignoring or being misled by it.
- **Mechanism**: (1) Uses non-learnable sinusoidal encoding (static encoding) to encode camera rays; (2) Curriculum learning: feeds ground-truth camera parameters with a probability of $1 - \tanh(s/10^5)$ at the early stage of training, gradually transitioning to predicted values; (3) Applies stop-gradient to the gradients flowing from the Angular Module output to the Radial Module to mimic an external information stream; (4) Disables LayerScale in cross-attention to prevent bypassing the conditioning.
- **Design Motivation**: Weak conditioning causes the model to route local distortions back to the encoder feature space instead of integrating field-of-view information, which is particularly severe in large field-of-view configurations.

### Loss & Training

The total loss consists of three parts: the angular loss $\mathcal{L}_A = \beta\mathcal{L}_{AA}^{0.7}(\theta) + (1-\beta)\mathcal{L}_{AA}^{0.5}(\phi)$ (with $\beta=0.75$), the radial loss $\mathcal{L}_{rad} = \|\hat{\mathbf{R}}_{\log} - \mathbf{R}_{\log}^*\|_1$, and a confidence loss.

## Key Experimental Results

### Main Results: Cross-Camera Domain Zero-Shot Evaluation (ViT-L Backbone)

| Method | S.FoV $\delta_1^{SSI}$↑ | L.FoV $\delta_1^{SSI}$↑ | Pano $\delta_1^{SSI}$↑ | S.FoV $F_A$↑ | L.FoV $F_A$↑ | Pano $F_A$↑ |
|------|------------------------|------------------------|----------------------|-------------|-------------|-------------|
| DepthAnything | 92.2 | 47.5 | 10.4 | - | - | - |
| UniDepth | 94.9 | 68.6 | 33.0 | 59.0 | 16.9 | 2.0 |
| DepthPro | 87.4 | 64.5 | 31.8 | 56.0 | 26.1 | 1.9 |
| **UniK3D-Large** | **96.1** | **91.2** | **81.4** | **68.1** | **71.6** | **80.2** |

### Comparison with Panorama-Specific Methods (Stanford-2D3D, Zero-Shot)

| Method | Training Data | $\delta_1$↑ | A.Rel↓ |
|------|---------|------------|--------|
| BiFuse++ | Matterport3D | 91.4 | 10.7 |
| UniFuse | Matterport3D | 91.3 | 9.42 |
| **UniK3D** | Ours | **96.8** | **8.01** |

### Key Findings
- On large field-of-view (L.FoV), UniK3D improves $\delta_1^{SSI}$ from UniDepth's 68.6% to 91.2% (+22.6%).
- Under panoramic settings, it increases from 33.0% to 81.4% (+48.4%), significantly outperforming all existing methods.
- It still maintains SOTA performance (96.1%) under standard narrow-FoV pinhole setups, indicating no performance trade-off.
- It outperforms methods specifically trained on panoramic data (e.g., BiFuse++) in a zero-shot manner.
- Only 19 parameters (15 spherical harmonic coefficients + 4 domain parameters) are required to represent almost any camera geometry.

## Highlights & Insights

1. **Unified Omnidirectional Spherical Representation**: Solves the mathematical ill-posedness of traditional methods when the field of view exceeds 180° for the first time.
2. **Elegance of Spherical Harmonics Camera Representation**: Replaces explicit camera model parameters with 15 coefficients + 4 domain parameters, achieving true model-agnostic capability.
3. **Asymmetric Angular Loss**: Conceptually elegant and highly efficient in addressing distribution shift issues without requiring complex data rebalancing.
4. **Complete Decoupling of Camera and Scene**: The spherical framework ensures that the projected scale is solely related to the radial distance, greatly simplifying the learning task.

## Limitations & Future Work

- Training requires large-scale datasets with various camera types; data collection remains a bottleneck.
- The precision of spherical harmonic coefficients is limited by the order of the basis functions (currently 3rd degree); extreme distortions may require a higher order.
- Inference speed is not discussed in detail; the inverse spherical harmonic transform introduces some computational overhead.
- Future work can explore integration with video sequences to achieve temporally consistent 3D reconstruction.

## Related Work & Insights

- Unlike UniDepth, which decouples camera and depth but still assumes a pinhole model, UniK3D completely removes any camera assumptions.
- Spherical harmonics are widely used in graphics (e.g., environmental lighting); this work innovatively applies them to camera geometry representation.
- The concept of asymmetric loss can be generalized to other learning tasks that exhibit data distribution shifts.

## Rating

⭐⭐⭐⭐⭐ — Truly solves the core problem of universal camera depth estimation. The spherical + spherical harmonics design is elegant and effective, and the dramatic improvement under large-FOV setups holds significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2026\] UniDAC: Universal Metric Depth Estimation for Any Camera](../../CVPR2026/3d_vision/unidac_universal_metric_depth_estimation_for_any_camera.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)

</div>

<!-- RELATED:END -->
