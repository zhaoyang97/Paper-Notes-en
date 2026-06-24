---
title: >-
  [Paper Note] Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision
description: >-
  [CVPR 2025][3D Vision][RGB-NIR Fusion] This paper develops a prism-based pixel-aligned RGB-NIR stereo camera system mounted on a mobile robot to collect a large-scale dataset under diverse illumination conditions. It proposes two methods—image fusion and feature fusion—to enable existing RGB pre-trained vision models to leverage NIR information with little to no fine-tuning, achieving significant improvements in tasks such as depth estimation, object detection, and SfM.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "RGB-NIR Fusion"
  - "Pixel Alignment"
  - "Stereo Depth Estimation"
  - "Robot Vision"
  - "Multispectral Dataset"
date: 2026-05-08
content_hash: 48b93b9ad6318ecd
---

# Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision

**Conference**: CVPR 2025  
**arXiv**: [2411.18025](https://arxiv.org/abs/2411.18025)  
**Code**: None  
**Area**: Autonomous Driving / Robot Vision / Multispectral Imaging  
**Keywords**: RGB-NIR Fusion, Pixel Alignment, Stereo Depth Estimation, Robot Vision, Multispectral Dataset

## TL;DR

This paper develops a prism-based pixel-aligned RGB-NIR stereo camera system mounted on a mobile robot to collect a large-scale dataset under diverse illumination conditions. It proposes two methods—image fusion and feature fusion—to enable existing RGB pre-trained vision models to leverage NIR information with little to no fine-tuning, achieving significant improvements in tasks such as depth estimation, object detection, and SfM.

## Background & Motivation

**Background**: RGB cameras are the primary data source for visual computing, but their performance degrades severely in low-light conditions (nighttime, dark rooms). Near-infrared (NIR) imaging, aided by active illumination, provides effective information in wavelengths invisible to the human eye (750-1000 nm), complementing RGB. RGB-NIR multispectral fusion has been extensively studied in tasks such as object detection and 3D reconstruction.

**Limitations of Prior Work**: Existing RGB-NIR systems (such as Kinect, Intel RealSense D415) use separate RGB and NIR cameras with different viewpoints, resulting in pixel misalignment. To align the images, image registration and pose estimation are required, which introduces error accumulation and creates a "chicken-and-egg" dilemma—depth estimation requires fused images, while image fusion requires depth for registration.

**Key Challenge**: Sequential capture schemes are limited to static scenes. Custom CFA four-channel sensors enforce identical exposures for RGB and NIR, failing to handle challenging scenes with large dynamic range discrepancies (e.g., dark rooms with active NIR illumination).

**Goal**: (1) To construct a truly pixel-aligned RGB-NIR stereo imaging system; (2) to collect a large-scale dataset covering diverse illumination conditions; (3) to design fusion methods enabling existing RGB models to directly utilize NIR information.

**Key Insight**: Using a dichroic prism to separate RGB and NIR light along the same optical axis onto two independent CMOS sensors, achieving pixel-level alignment from the hardware level and fundamentally bypassing the registration issue.

**Core Idea**: Achieving pixel-aligned RGB-NIR imaging through prism beamsplitting to eliminate the registration and depth-dependency problems of traditional multispectral systems; and proposing a learned spatially adaptive weight fusion method to benefit downstream models without modifications.

## Method

### Overall Architecture

The system consists of a pair of pixel-aligned, dual-sensor RGB-NIR stereo cameras mounted on a mobile robot (AgileX Ranger Mini 2.0), active NIR illumination, and a LiDAR. Inside each camera, an incoming beam is split by a dichroic prism into an RGB path (transmitting) and an NIR path (reflecting), which are captured by independent CMOS sensors. The output consists of four-channel (R, G, B, NIR) pixel-aligned image pairs plus LiDAR point clouds. Based on this, two fusion methods are proposed: (1) Image-level fusion, which fuses RGBN into a three-channel image to be directly fed into pre-trained models; and (2) feature-level fusion, which fuses RGB and NIR in the feature space and fine-tunes the downstream networks.

### Key Designs

1. **Beamsplitting Pixel-Aligned Camera**:

    - **Function**: Simultaneously acquires pixel-level aligned RGB and NIR images along the same optical axis.
    - **Mechanism**: A JAI FS-1600D-10GE dual-sensor camera is used, where an internal dichroic prism separates visible and near-infrared light onto two independent CMOS sensors. The RGB and NIR sensors can independently set their exposure times and gains ($t_R = t_G = t_B \neq t_{NIR}$), adapting to dynamic range differences between the two spectrums under various lighting conditions.
    - **Design Motivation**: Resolves the RGB-NIR pixel misalignment issue completely at the hardware level without any image registration steps, eliminating the "depth-registration" chicken-and-egg problem.

2. **Learned Spatially Adaptive Weight Image Fusion**:

    - **Function**: Fuses four-channel RGBN into three-channel images, which can be directly fed into RGB pre-trained models.
    - **Mechanism**: In the HSV color space, the hue $H$ and saturation $S$ are kept constant, while the value channel $V$ and NIR are weighted-fused as $I_{fusion} = M^{-1}[I_H, I_S, \alpha I_V + \beta I_{NIR}]$. Unlike traditional predefined fixed weights, this method uses an attention-based MLP to learn spatially varying $\alpha, \beta$: a ResNet encoder first extracts 256-channel feature maps for RGB and NIR, respectively; an attention fusion module produces fused features; and a decoder regresses pixel-wise $\alpha, \beta$. Guided filtering is finally applied with the NIR image for refinement.
    - **Design Motivation**: The information content of RGB and NIR varies across different scene regions (e.g., active NIR illumination dominates nearby areas, whereas ambient light dominates far regions). Fixed weights fail to adapt, but learning adaptive weights dynamically selects the optimal fusion ratio based on the scene.

3. **Alternating Correlation Volume Feature Fusion for Stereo Depth Estimation**:

    - **Function**: Fuses RGB-NIR information at the feature level for stereo depth estimation.
    - **Mechanism**: Based on the RAFT-Stereo network, features are extracted from the RGB and NIR images of the left and right cameras, obtaining $F_{fusion}$ via attention fusion. The key innovation is constructing two correlation volumes—one for the fused features and one for the NIR features—and alternately using them during the GRU iterations for disparity estimation. $F_{fusion}^{left}$ is utilized as the context feature to initialize the GRU hidden state.
    - **Design Motivation**: Alternating between fused and NIR correlation volumes fully exploits cross-spectral information. NIR remains robust under ambient light variations, whereas the fused features preserve the discriminative color cues of RGB, complementing each other to improve matching accuracy.

### Loss & Training

The image fusion model is trained using only photometric loss and stereo depth reconstruction loss, requiring no training of downstream vision models. The feature fusion stereo depth network is fine-tuned using disparity reconstruction loss on synthetic datasets and LiDAR loss on real-world datasets.

## Key Experimental Results

### Main Results

**Comparison of Image Fusion Methods (Pre-trained RAFT-Stereo + YOLO)**:

| Method | Depth RMSE (m) ↓ | Detection mAP ↑ |
|------|----------------|-----------|
| RGB only | 8.943 | 0.756 |
| NIR only | 9.646 | 0.703 |
| HSV baseline | 7.692 | 0.744 |
| DarkVision | 8.313 | 0.762 |
| Adaptive | 7.830 | 0.773 |
| **Ours** | **7.567** | **0.809** |

**Comparison of Stereo Depth Estimation with Feature Fusion**:

| Method | Depth RMSE (m) ↓ |
|------|----------------|
| RAFT-Stereo (RGB) | 8.943 |
| RAFT-Stereo (NIR) | 9.646 |
| CS-Stereo | 8.941 |
| DPSNet | 7.633 |
| Image fusion (Ours) | 7.567 |
| **Feature fusion (Ours)** | **6.747** |

### Ablation Study

| Correlation Volume Configuration | Depth Error (m) ↓ | Description |
|------------|--------------|------|
| Fused Volume Only | 7.440 | Lacks robust NIR matching |
| Alternating RGB-NIR Volume | 8.571 | Difficulties in cross-spectral matching |
| Alternating Fused-RGB-NIR Volume | 7.426 | Information redundancy |
| **Alternating Fused-NIR Volume** | **6.747** | Optimal configuration |

### Key Findings

- The strategy of alternating between fused and NIR correlation volumes is optimal, demonstrating that the illumination robustness of NIR and the discriminative power of fused features show the best complementary effect.
- Image-level fusion (without retraining downstream models) already yields significant improvements, showing that pixel alignment itself is a substantial advantage.
- The advantages of RGB-NIR fusion are particularly prominent in low-light and nighttime scenes, where active NIR illumination effectively compensates for the deficiencies of RGB.

## Highlights & Insights

- **Hardware-Algorithm Co-Design Approach**: Resolving registration at the hardware level using prism beamsplitting avoids the error accumulation inherent in software-based registration. This philosophy of "solving the problem at the right system level" is highly instructive.
- **Fused Images Instantly Compatible with Pre-trained Models**: Enhancing performance without retraining significantly reduces deployment costs, making it highly attractive for practical robotic systems.
- **Independent Exposure Control**: Since the RGB and NIR sensors can be exposure-controlled independently, they adapt to diverse scenes ranging from bright sunlight to complete darkness, which is unachievable with single-sensor four-channel designs.

## Limitations & Future Work

- The system relies on a specific dual-sensor prism camera (JAI FS-1600D), which incurs higher costs compared to consumer-grade sensors.
- Active NIR illumination has a limited range, potentially resulting in insufficient NIR information in far-range scenes.
- The dataset scale (43 scenes, 80K frames) is relatively small compared to large-scale autonomous driving datasets.
- The potential of utilizing pixel-aligned RGBN in generative models remains unexplored, which is an interesting direction for future work.
- Incorporating LiDAR time-of-flight information could be considered to further improve 3D imaging accuracy.

## Related Work & Insights

- **vs. Kinect / RealSense**: These commercial devices locate RGB and NIR cameras at different viewpoints, necessitating depth-dependent registration. In contrast, this work fundamentally circumvents this issue.
- **vs. Custom CFA Four-Channel Sensors**: Single-sensor global exposure cannot handle scenes with large dynamic range differences between RGB and NIR, whereas the independent dual-sensor exposure implemented in this work offers superior flexibility.
- **vs. RGB-Thermal Beamsplitting Systems [Guo et al.]**: This work ports the beamsplitting paradigm from thermal infrared to near-infrared and further integrates stereo vision and LiDAR synchronization, demonstrating higher system completeness.

## Rating

- Novelty: ⭐⭐⭐⭐ Excellent hardware system design, but the algorithmic innovation is relatively incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple tasks (depth, detection, SfM) with ablation studies, though comparisons with a broader range of SOTA methods are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear system description, with a logical organization of hardware and algorithmic components.
- Value: ⭐⭐⭐⭐ The dataset and hardware system hold significant value for the robot vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_extended_dr_3d.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)

</div>

<!-- RELATED:END -->
