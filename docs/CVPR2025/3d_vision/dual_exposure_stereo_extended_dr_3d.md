---
title: >-
  [Paper Note] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging
description: >-
  [CVPR 2025][3D Vision][Dual-Exposure Stereo] This paper proposes a dual-exposure stereo method (Dual-Exposure Stereo) that extends the effective dynamic range by automatically controlling the dual-exposure parameters of stereo cameras, and designs a motion-aware dual-exposure depth estimation network to achieve robust 3D imaging in wide dynamic range scenes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dual-Exposure Stereo"
  - "High Dynamic Range"
  - "Depth Estimation"
  - "Auto-Exposure Control"
  - "Robot Vision"
date: 2026-05-08
content_hash: a8af49ea2e0a80f3
---

# Dual Exposure Stereo for Extended Dynamic Range 3D Imaging

**Conference**: CVPR 2025  
**arXiv**: [2412.02351](https://arxiv.org/abs/2412.02351)  
**Code**: None (To be released)  
**Area**: 3D Vision / Stereo Vision  
**Keywords**: Dual-Exposure Stereo, High Dynamic Range, Depth Estimation, Auto-Exposure Control, Robot Vision

## TL;DR

This paper proposes a dual-exposure stereo method (Dual-Exposure Stereo) that extends the effective dynamic range by automatically controlling the dual-exposure parameters of stereo cameras, and designs a motion-aware dual-exposure depth estimation network to achieve robust 3D imaging in wide dynamic range scenes.

## Background & Motivation

**Background**: Stereo imaging is a popular 3D imaging technique. However, traditional cameras have a limited dynamic range. Under extreme lighting conditions (e.g., tunnel exits, strong night lights), overexposed and underexposed areas severely impair disparity estimation.

**Limitations of Prior Work**: Existing auto-exposure control (AEC) methods adjust capture settings on a per-frame basis and cannot extend the camera's native dynamic range. Exposure bracketing techniques use predefined exposures and fail to adapt to scene changes.

**Key Challenge**: The dynamic range of a scene can far exceed the camera's native dynamic range, making a single exposure insufficient to cover both bright and dark regions simultaneously.

**Goal**: To design a dual-exposure strategy that utilizes different exposure settings in alternating frames, combining the advantages of AEC and exposure bracketing.

**Key Insight**: Capture stereo images with different exposures in alternating frames, and automatically diverge the dual-exposure settings when the scene dynamic range exceeds the camera's dynamic range.

**Core Idea**: Automatic Dual-Exposure Control (ADEC) dynamically adjusts dual-exposure parameters, combined with a motion-aware feature fusion network to leverage dual-exposure images for depth estimation.

## Method

### Overall Architecture

The system consists of two parts: (1) ADEC automatically controls dual-exposure parameters based on histogram statistics; (2) a dual-exposure depth estimation network estimates the disparity map from four images (two frames $\times$ two views), extending the effective dynamic range through optical flow alignment and weighted feature fusion.

### Key Designs

1. **Automatic Dual-Exposure Control (ADEC)**:

    - **Function**: Adaptively adjusts dual-exposure parameters to cover a wider scene dynamic range.
    - **Mechanism**: Computes the histogram skewness $S_i$ and the proportions of extreme pixels $L_i, H_i$. When $L_i > \tau_h$ and $H_i > \tau_h$, the scene dynamic range is determined to exceed the camera's dynamic range, prompting the dual-exposure to diverge; otherwise, the dual-exposure converges by driving the skewness toward zero.
    - **Design Motivation**: Combines the adaptiveness of AEC with the dynamic range extension capabilities of exposure bracketing.

2. **Dual-Exposure Feature Fusion**:

    - **Function**: Fuses features under different exposures to obtain a unified feature representation covering both bright and dark regions.
    - **Mechanism**: Uses an optical flow network to estimate inter-frame motion and align the features of the second frame to the first frame. Weighted fusion is performed using a trapezoidal weight function $W_i^c$ based on pixel intensity: $\hat{F}^c = (W_1^c \cdot F_1^c + W_{2\to1}^c \cdot F_{2\to1}^c) / (W_1^c + W_{2\to1}^c + \epsilon)$.
    - **Design Motivation**: Lowers the weights of overexposed/underexposed pixels and increases the weights of well-exposed pixels to ensure the quality of fused features.

3. **Motion-Aware Disparity Estimation**:

    - **Function**: Constructs cost volumes from fused features and estimates disparity maps.
    - **Mechanism**: Extracts dual-exposure features using a pre-trained feature extractor, performs weighted fusion after optical flow alignment, constructs the correlation volume $C(x,y,d)$ between Left and Right views, and feeds it into the disparity estimation network.
    - **Design Motivation**: Dual-exposure feature fusion encodes information from both bright and dark regions, extending the effective dynamic range for 3D imaging.

### Loss & Training

A synthetic dataset containing 1000 training videos was generated using the CARLA simulator, covering various lighting conditions such as daytime, twilight, and nighttime. The disparity estimation network was fine-tuned on this synthetic data.

## Key Experimental Results

### Main Results

The proposed method outperforms other exposure control methods on both synthetic and real-world datasets, significantly improving depth estimation accuracy in wide dynamic range scenes.

### Ablation Study

- Comparison of Fixed Exposure vs. ADEC: ADEC demonstrates significant advantages in wide dynamic range scenes.
- Dual-Exposure vs. Single-Exposure Depth Estimation: Dual-exposure fusion significantly improves depth accuracy in overexposed/underexposed regions.

### Key Findings

- The dual-exposure method can be applied to cameras of any bit depth and is not limited to specific hardware.
- Excessive exposure differences can hinder stereo matching, necessitating a constraint on the dual-exposure interval.

## Highlights & Insights

- Constructed a real robotic vision system (wheeled robot + stereo camera + LiDAR) to collect real-world data.
- The skewness-divergence mechanism of ADEC is designed to be simple and highly efficient.
- Provided both synthetic and real-world datasets.

## Limitations & Future Work

- Optical flow estimation may be inaccurate in scenes with rapid motion.
- The dual-exposure strategy halves the effective frame rate.
- The scale of the real-world dataset is limited.

## Related Work & Insights

- Complementary to alternative sensor approaches such as event cameras.
- Adaptable to multi-exposure ($>2$) schemes.
- The dynamic range extension concept can be applied to other vision-based 3D tasks.

## Rating

- Novelty: 7/10 — The dual-exposure concept is straightforward yet effective.
- Technical Depth: 7/10 — The system design is comprehensive.
- Experimental Thoroughness: 8/10 — Validated on both lyric and real-world data.
- Writing Quality: 7/10 — Clear and standardized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision](pixel-aligned_rgb-nir_stereo_imaging_and_dataset_for_robot_vision.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)

</div>

<!-- RELATED:END -->
