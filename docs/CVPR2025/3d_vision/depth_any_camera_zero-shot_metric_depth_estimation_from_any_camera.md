---
title: >-
  [Paper Note] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera
description: >-
  [CVPR 2025][3D Vision][Metric Depth Estimation] The Depth Any Camera (DAC) framework is proposed, which enables zero-shot metric depth estimation generalizing to fisheye and 360° cameras with training restricted solely to perspective images. By utilizing ERP unified representation, pitch-aware transformation, and FoV alignment, DAC improves $\delta_1$ accuracy by up to 50% on wide-FoV datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Metric Depth Estimation"
  - "Zero-Shot Generalization"
  - "Fisheye Cameras"
  - "Omnidirectional Cameras"
  - "Equirectangular Projection"
date: 2026-05-08
content_hash: d7e343facf4670c1
---

# Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera

**Conference**: CVPR 2025  
**arXiv**: [2501.02464](https://arxiv.org/abs/2501.02464)  
**Code**: [https://yuliangguo.github.io/depth-any-camera](https://yuliangguo.github.io/depth-any-camera)  
**Area**: 3D Vision  
**Keywords**: Metric Depth Estimation, Zero-Shot Generalization, Fisheye Cameras, Omnidirectional Cameras, Equirectangular Projection  

## TL;DR

The Depth Any Camera (DAC) framework is proposed, which enables zero-shot metric depth estimation generalizing to fisheye and 360° cameras with training restricted solely to perspective images. By utilizing ERP unified representation, pitch-aware transformation, and FoV alignment, DAC improves $\delta_1$ accuracy by up to 50% on wide-FoV datasets.

## Background & Motivation

While existing metric depth foundation models (such as Metric3Dv2, UniDepth, and ZoeDepth) generalize well on perspective images, their performance severely degrades on wide field of view (FoV) cameras (e.g., fisheye, 360°). The core challenges include:

1. **How to choose a unified camera model** to represent different FoVs?
2. **How to leverage perspective training data** to generalize to the highly distorted regions characteristic of wide-FoV cameras?
3. **Huge variations in training sample sizes**: Different FoVs map to extremely mismatched regions in a unified space.
4. **Inconsistency between training and testing resolutions**.

Notably, while UniDepth attempts to handle different FoVs by learning spherical transformations, its performance on wide-FoV images remains poor, demonstrating that neural networks have limited capability when extrapolating to data spaces outside the training domain.

## Method

### Overall Architecture

DAC adopts Equirectangular Projection (ERP) as the unified image representation. During training, perspective images from various cameras are transformed into ERP patches; during testing, wide-FoV images are similarly mapped to the ERP space for inference. Core pipeline: Training images $\rightarrow$ Pitch-aware ERP transformation + Data augmentation $\rightarrow$ FoV alignment $\rightarrow$ Multi-resolution training $\rightarrow$ Model inference. The network architecture uses iDisc, and the loss function is the SIlog loss.

### Key Designs

1. **Pitch-Aware Image-to-ERP Transformation**:
    - **Function**: Efficiently converts perspective images into ERP patches, simulating the highly distorted regions typical of wide-FoV cameras.
    - **Mechanism**: Utilizes closed-form mapping of Gnomonic geometry to establish correspondence between uniform grid points of the ERP patch and input image coordinates, achieving efficient transformation via grid sampling. The key innovation lies in setting the tangent center latitude $\lambda_c$ as the camera pitch angle. When the camera orientation is known or can be estimated, perspective data can be projected to different latitudes of the ERP space, simulating high-distortion areas that only wide-FoV cameras can observe. Pitch augmentation with noise added to $\lambda_c$ is also supported.
    - **Design Motivation**: Neural networks exhibit limited generalization when extrapolating beyond the training data space, making it necessary to simulate the distribution of the target domain during training.

2. **FoV Alignment**:
    - **Function**: Normalizes training samples with different FoVs to a unified ERP patch size, resolving size discrepancy issues.
    - **Mechanism**: Applies a scale augmentation of $s_\sigma = \text{FoV}_c / \text{FoV}_e$ to each input image to match its FoV with a predefined ERP patch FoV. The ERP patch FoV is determined by $\text{FoV}_e = H_e \pi / H_E$. Thus, a single predefined-size ERP patch can maximize visual content representation and minimize computational waste from zero padding.
    - **Design Motivation**: Camera FoV in datasets like HM3D varies dramatically from 36° to 124°. Standardizing without alignment leads to severe information loss or computational waste for certain samples.

3. **Multi-Resolution Training**:
    - **Function**: Addresses the mismatch between training and testing resolutions.
    - **Mechanism**: Each ERP patch is additionally scaled to two lower resolutions ($0.7\times$ and $0.4\times$ of the original size). A batch inputs these three different-resolution images and sums their losses. This enforces the model to learn scale-equivariant features.
    - **Design Motivation**: Wide-FoV test images can differ significantly from training patches in aspect ratio and resolution, especially when using attention modules where varying numbers of image tokens degrade performance.

### Depth Representation

- Uses Euclidean distance (distance from the camera center) instead of Z-buffer format, as the latter yields inaccurate low depth values under spherical projection.
- Depth scaling operations follow the canonical model paradigm of Metric3D.

## Key Experimental Results

### Main Results

| Test Dataset | Method | $\delta_1$↑ | Abs Rel↓ | Remarks |
|------|------|------|----------|------|
| Matterport3D (360°) | Metric3Dv2 | 0.429 | 0.279 | Indoor 670K training |
| | **DAC (Ours)** | **0.773** | **0.156** | **Trained on same data, $\delta_1$ improved by 80%** |
| Pano3D-GV2 (360°) | Metric3Dv2 | 0.506 | 0.261 | Indoor 670K training |
| | **DAC (Ours)** | **0.812** | **0.139** | **$\delta_1$ improved by 60%** |
| ScanNet++ (Fisheye 150°) | Metric3Dv2 | 0.649 | 0.192 | Indoor 670K training |
| | **DAC (Ours)** | **0.852** | **0.132** | **$\delta_1$ improved by 31%** |
| KITTI360 (Fisheye 180°) | Metric3Dv2 | 0.768 | 0.152 | Outdoor 130K training |
| | **DAC (Ours)** | **0.786** | **0.156** | Slight improvement |

### Ablation Study

| Configuration | Pano3D $\delta_1$↑ | ScanNet++ $\delta_1$↑ | Remarks |
|------|---------|------|------|
| DAC (Full) | 0.725 | 0.654 | Full framework |
| w/o Pitch-Aware ERP | 0.491 | - | $\delta_1$ decreases by 32% |
| w/o Pitch Aug | 0.691 | - | Remove pitch augmentation |
| w/o FoV Align | 0.408 | - | $\delta_1$ decreases by 44% |
| w/o Multi-Reso | 0.513 | - | $\delta_1$ decreases by 29% |

### Key Findings

- On indoor 360° datasets, DAC achieves an improvement in $\delta_1$ of nearly **50%** compared to Metric3Dv2 (which is trained with a larger backbone and more data).
- FoV alignment is the most critical component: without it, $\delta_1$ drops by 44%.
- The improvement is smaller on the outdoor KITTI360 dataset, because LiDAR points are concentrated in low-distortion regions and the training data features limited pitch variation.
- Despite learning spherical transformations, UniDepth still performs poorly on wide-FoV datasets, validating the advantage of geometric priors over purely data-driven methods.
- DAC outperforms much larger models using DINOv2/ViT-L even when using a lighter ResNet101 backbone.

## Highlights & Insights

- **Unchanged training data, changed representation space**: The core contribution of DAC is the data transformation pipeline rather than a new network architecture, making it widely applicable to existing depth networks.
- **Domain expansion effect of Pitch-aware ERP**: By geometrically transforming perspective training data to "project" onto high-distortion regions typical of wide FoVs, the extrapolation challenge is effectively solved.
- **Euclidean distance vs. Z-buffer**: The necessity of using Euclidean distance under spherical projection is a valuable engineering insight.
- **Complementarity with Metric3D**: DAC does not replace the Metric3D pipeline; rather, it offers a superior, unified representation space.

## Limitations & Future Work

- Limited performance gain in outdoor scenes, constrained by narrow pitch distributions in training data.
- Integration with stronger backbones (e.g., DINOv2 ViT-G) was not attempted.
- Requires known or estimable camera parameters (pitch angle).
- Although efficient, Gnomonic mapping for each ERP patch still incurs data preprocessing overhead.
- Does not handle dynamic scenes or video depth estimation.

## Related Work & Insights

- Metric3D achieves cross-dataset training through canonical camera normalization of perspective images. DAC extends this concept to the ERP space.
- Compared to specialized methods for wide FoVs (which use deformable convolutions, etc.), DAC requires no specialized network architectures.
- **Insight**: The choice of representation space is more important than network architecture—data augmentation driven by geometric priors can be key to cross-domain generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of unified ERP representation coupled with pitch-aware transformation is simple yet effective, and the FoV alignment design is highly practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 360°, fisheye, indoor, and outdoor settings with clear ablations, though outdoor improvements are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Rich and clear diagrams, with comprehensive technical details.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a practical bottleneck in cross-camera generalization for foundation depth models, exhibiting high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniDAC: Universal Metric Depth Estimation for Any Camera](../../CVPR2026/3d_vision/unidac_universal_metric_depth_estimation_for_any_camera.md)
- [\[CVPR 2025\] SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation](sharpdepth_sharpening_metric_depth_predictions_using_diffusion_distillation.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)

</div>

<!-- RELATED:END -->
