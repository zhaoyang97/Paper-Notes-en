---
title: >-
  [Paper Note] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging
description: >-
  [3D Vision] Proposes a Dual-Exposure Stereo method that utilizes Automatic Dual-Exposure Control (ADEC) to apply different exposures in alternating frames, combined with a motion-aware dual-exposure feature fusion network for disparity estimation. This extends the effective dynamic range of stereo cameras to 160% and achieves robust 3D imaging under extreme lighting conditions.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 03f28480d4eb2704
---

# Dual Exposure Stereo for Extended Dynamic Range 3D Imaging

## TL;DR

Proposes a Dual-Exposure Stereo method that utilizes Automatic Dual-Exposure Control (ADEC) to apply different exposures in alternating frames, combined with a motion-aware dual-exposure feature fusion network for disparity estimation. This extends the effective dynamic range of stereo cameras to 160% and achieves robust 3D imaging under extreme lighting conditions.

## Background & Motivation

Robust stereo 3D imaging is crucial in scenarios such as autonomous driving and robotic navigation. However, the dynamic range (DR) of real-world illumination far exceeds the capture capability of conventional cameras (only 42dB for 8-bit cameras), leading to severe degradation of depth estimation in over-exposed and under-exposed regions.

Limitations of Prior Work:
- **Single-Exposure AEC**: Adjusts only a single exposure value to adapt to the scene, failing to extend the camera's inherent dynamic range.
- **Exposure Bracketing**: Uses predefined multi-exposure settings, which cannot adapt to dynamic range variations in the scene and increases acquisition time.
- **Independent Exposure for Stereo Cameras**: Using different exposures for the left and right cameras disrupts stereo consistency, degrading matching quality.

The key insight of this work is: by alternating different exposures temporally (rather than spatially between the left and right cameras), the luminance consistency of stereo images is maintained while the dynamic range is effectively extended through two-frame complementarity.

## Method

### Overall Architecture

The system consists of two core components: (1) An Automatic Dual-Exposure Control (ADEC) module that adaptively adjusts the exposure values of alternating frames based on histogram statistics; (2) A motion-aware dual-exposure disparity estimation network that fuses features from dual-exposure frames and compensates for inter-frame motion for depth estimation.

### Key Designs

#### 1. Automatic Dual-Exposure Control (ADEC)

- **Function**: Adaptively adjusts the dual exposure values $(e_1, e_2)$ based on the dynamic range of the scene, diverging exposures when the scene DR exceeds the camera DR and converging exposures when the scene DR is manageable.
- **Mechanism**: Utilizes histogram skewness $S_i$ and the ratio of extreme pixels $L_i, H_i$ as statistical indicators. When $L_i > \tau_h$ and $H_i > \tau_h$ (indicating a simultaneous presence of significant over-exposure and under-exposure), the scene DR is determined to exceed the camera DR, and the dual exposures are diverged proportionally; otherwise, the exposure is adjusted to drive the skewness toward zero.
- **Design Motivation**: Combines the advantages of both AEC and exposure bracketing—adaptive adjustment and DR extension. A constraint on the exposure difference $\Delta e < \tau_{\Delta e}$ is introduced to prevent instability caused by excessive divergence. It runs at over 120 FPS, supporting real-time applications.

#### 2. Motion-Aware Dual-Exposure Feature Fusion

- **Function**: Spatially aligns and fuses features from alternating frames with different exposures to generate a unified feature map containing high-dynamic-range information.
- **Mechanism**: (1) Uses a pre-trained optical flow network to estimate inter-frame motion; (2) warps the features of the second frame to the viewpoint of the first frame; (3) performs weighted fusion using a brightness-based trapezoidal weight function, where well-exposed regions receive high weights and over-/under-exposed regions receive low weights.
- **Design Motivation**: The temporal gap between dual-exposure frames introduces motion displacement, which must be compensated for effective fusion. The trapezoidal weight function $W_i^c$ automatically assigns fusion weights based on pixel brightness (thresholds $\alpha=0.02, \beta=0.98$), fully utilizing the valid information in each frame.

#### 3. Stereo Disparity Estimation Based on Fused Features

- **Function**: Builds a correlation volume from the fused left and right feature maps and estimates the disparity map.
- **Mechanism**: The fused features $\hat{F}^{\text{left}}, \hat{F}^{\text{right}}$ encode high-dynamic-range information, so a standard correlation volume $C(x,y,d) = \hat{F}^{\text{left}}(x,y) \cdot \hat{F}^{\text{right}}(x+d, y)$ can successfully capture matching information in both bright and dark regions. Multi-scale feature fusion is adopted to enhance robustness.
- **Design Motivation**: The core idea is to perform HDR fusion at the feature level rather than the image level, avoiding information loss introduced by preprocessing steps such as tone mapping.

### Loss & Training

Supervised training is performed on the fused disparity map using the standard disparity estimation loss of RAFT-Stereo, followed by fine-tuning on synthetic datasets.

## Key Experimental Results

### Main Results

| Method | Synthetic Data Disp MAE (px) ↓ | Real Data Depth MAE (m) ↓ | FPS ↑|
|------|-------------------------|-------------------------|-------|
| AverageAE | 2.823 | 2.7679 | 616.27 |
| GradientAE | 2.948 | 2.5847 | 42.10 |
| NeuralAE | 2.778 | 1.9232 | **0.25** |
| **ADEC (Ours)** | **1.355** | **1.9142** | 124.58 |

### Ablation Study

| ADEC | Weighted Fusion | Motion Compensation | Disp MAE (px) ↓ |
|------|---------|---------|-----------------|
| ✗ | ✓ | ✓ | 6.2775 |
| ✓ | ✗ | ✓ | 3.3968 |
| ✓ | ✓ | ✗ | 8.3657 |
| ✓ | ✓ | ✓ | **2.9010** |

### Key Findings

- **DR Extension Rate**: High depth accuracy is maintained even at a 160% DR extension rate, whereas other AEC methods fail to extend the DR.
- **Motion Compensation is Most Critical**: Ablation studies show that removing motion compensation leads to the largest error (8.37 vs 2.90), indicating that inter-frame motion alignment is the bottleneck for dual-exposure methods.
- **ADEC Module is Second Most Critical**: Fixed dual exposure (removing ADEC) results in an error of 6.28, showing that adaptive adjustment brings significant improvements.
- **Real-time Performance**: ADEC runs at 124 FPS, which is far superior to NeuralAE's 0.25 FPS, while achieving comparable depth accuracy.

## Highlights & Insights

1. **Precise Problem Definition**: Decomposes the DR extension problem into three sub-problems—exposure control, feature fusion, and motion compensation—each having a concise and effective solution.
2. **Elegant Combination of AEC and Bracketing**: Alternating exposures temporally maintains left-right consistency while extending the DR, presenting a natural unification of the two classic approaches.
3. **Complete System**: Delivers a full-link contribution from hardware (robotic vision system) to algorithm (ADEC + network) and dataset (real + synthetic).
4. **High Practicality**: The ADEC running at 120+ FPS is highly suitable for real-time systems, and the method can be applied to cameras with any bit-depth.

## Limitations & Future Work

1. **Two-Frame Temporal Latency**: Alternating exposures introduce inter-frame motion, which may lead to alignment failure in fast-moving scenes (e.g., high-speed driving).
2. **Exposure Difference Constraint**: The constraint $\tau_{\Delta e}=2.5$ limits the DR extension capability under extreme scenarios.
3. **Reliance on Optical Flow Estimation**: The accuracy of the optical flow network directly affects the fusion quality, and optical flow estimation itself may fail under extreme exposure differences.
4. **Only Evaluated on Stereo Disparity**: Generalization to other 3D tasks such as monocular depth estimation or multi-view reconstruction has not yet been explored.
5. **Domain Gap Between Synthetic and Real Data**: The model is trained on CARLA synthetic data, and its generalization to real-world scenes requires further validation.

## Related Work & Insights

- **RAFT-Stereo**: Serves as the backbone network for disparity estimation, upon which a dual-exposure fusion module is integrated.
- **HDR 3D Imaging**: Solutions utilizing unconventional sensors such as event cameras or SPADs incur high hardware costs, whereas this work achieves DR extension using conventional cameras.
- **Insights**: Finding an optimal balance between physical sensor limitations and algorithmic compensation—instead of relying on expensive HDR sensors, the effective DR can be extended through a simple exposure strategy combined with network-based fusion.

## Rating

⭐⭐⭐⭐

The problem is of high practical importance and clearly defined. The system design is complete (hardware + algorithm + dataset), and the proposed method is elegant and effective. The ablation studies comprehensively validate the necessity of each component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](dune_distilling_a_universal_encoder_from_heterogeneous_2d_and_3d_teachers.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)

</div>

<!-- RELATED:END -->
