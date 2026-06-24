---
title: >-
  [Paper Note] Focal Split: Untethered Snapshot Depth from Differential Defocus
description: >-
  [CVPR 2025][Depth from Differential Defocus] Inspired by jumping spider vision, this work constructs Focal Split, the first untethered (battery-powered) snapshot depth-from-differential-defocus camera. By utilizing a beamsplitter to split the optical path across two sensors with different focal distances, it estimates depth in real-time on a Raspberry Pi using only 500 FLOPs/pixel and 4.9W of power.
tags:
  - "CVPR 2025"
  - "Depth from Differential Defocus"
  - "Jumping Spider Biomimetic"
  - "Beamsplitter"
  - "Passive Ranging"
  - "Low-Power Edge Devices"
date: 2026-05-08
content_hash: 69fad52912b1db55
---

# Focal Split: Untethered Snapshot Depth from Differential Defocus

**Conference**: CVPR 2025  
**arXiv**: [2504.11202](https://arxiv.org/abs/2504.11202)  
**Code**: [https://focal-split.qiguo.org](https://focal-split.qiguo.org) (Available, includes DIY guide)  
**Area**: Others / Computational Photography  
**Keywords**: Depth from Differential Defocus, Jumping Spider Biomimetic, Beamsplitter, Passive Ranging, Low-Power Edge Devices

## TL;DR

Inspired by jumping spider vision, this work constructs Focal Split, the first untethered (battery-powered) snapshot depth-from-differential-defocus camera. By utilizing a beamsplitter to split the optical path across two sensors with different focal distances, it estimates depth in real-time on a Raspberry Pi using only 500 FLOPs/pixel and 4.9W of power.

## Background & Motivation

### Background

**Background**: Depth sensing is a fundamental requirement for robotics and AR. Mainstream solutions include active methods (structured light/ToF/LiDAR, which consume high power) and passive methods (stereo/monocular DNNs, which demand high computational cost). Depth from Differential Defocus (DfDD) is an elegant passive alternative—inferring depth by comparing the degree of blur under different focal distances.

**Limitations of Prior Work**: Existing DfDD methods either require sequential capture of two frames (capturing twice by changing the focal distance, which fails in dynamic scenes) or require high-end workstations (e.g., Focal Track requires a GPU server). There is no untethered depth camera capable of running in real-time on edge devices.

**Key Challenge**: DfDD requires two images of different focal distances, but temporal acquisition introduces motion blur, while spatial acquisition (using two sensors) is generally considered challenging to calibrate and bulky.

**Key Insight**: Inspiration from jumping spider eyes—jumping spiders perceive depth by simultaneously acquiring images of different focal planes using multi-layered retinae (with different focal distances). This mechanism is simulated using a beamsplitter combined with two sensors placed at different distances.

**Core Idea**: Beamsplitter + dual sensors at different focal distances = real-time passive depth sensing at 500 FLOPs/pixel.

### Proposed Solution

**Goal**: ### Key Designs

1. **Optical System Design**:

    - Function: Simultaneously acquire two images of different focal distances
    - Mechanism: A beamsplitter is placed behind a 30mm lens to split the incoming light into two paths.


## Method

### Overall Architecture


### Key Designs

1. **Optical System Design**:

    - Function: Simultaneously acquire two images of different focal distances
    - Mechanism: A beamsplitter is placed behind a 30mm lens to split the incoming light into two paths. Two OV5647 sensors are placed at different distances, $s_1$ and $s_2$, from the lens. The closer sensor captures the "near-focus" image, while the farther one captures the "far-focus" image.
    - Design Motivation: Spatial separation eliminates motion artifacts from temporal acquisition, and the beamsplitter ensures perfect synchronization between the two image paths.

2. **Ultra-Low Computation Depth Estimation**:

    - Function: Infer scene depth from the focal difference
    - Mechanism: Depth $Z = a/(b + \tilde{I}_s / \nabla^2 \tilde{I})$, where $\tilde{I}_s$ is the gradient in the focal direction (obtained from the difference between the two sensor images) and $\nabla^2 \tilde{I}$ is the spatial Laplacian operator. It requires only simple image subtraction and convolution, amounting to a mere 500 FLOPs per pixel.
    - Design Motivation: DNN methods require millions of FLOPs per pixel; this method achieves three orders of magnitude reduction in computation by relying on an analytical formula.

3. **Confidence Measure and Magnification Correction**:

    - Function: Identify unreliable depth estimates and correct magnification differences caused by varying sensor distances
    - Mechanism: Confidence $C = \tilde{I}_s^2$—a larger focal gradient indicates a stronger signal and more accurate estimation. Magnification correction aligns the two sensor images using an affine transformation.
    - Design Motivation: In texture-sparse regions, $\tilde{I}_s \approx 0$ and the depth estimation degrades; the confidence metric automatically flags these regions.

### Loss & Training

No training process involved—entirely analytical. The parameters $(a, b)$ are determined via optical calibration. An $L=21$ box filter is used for noise smoothing. Hardware cost is approximately $500, with a physical volume of 4×5×6 cm³.

## Key Experimental Results

### Main Results

| Metric | Focal Split | Focal Flow | Focal Track |
|------|-------------|------------|-------------|
| Power Consumption | **4.9W** | ~100W | ~300W |
| Computation/Pixel | **500 FLOPs** | ~10⁶ | ~10⁷ |
| Dynamic Scene MAE | **~42mm** | 179.25mm | 107.69mm |
| Working Distance | 860mm | - | - |

### Key Findings
- **Significant Advantage in Dynamic Scenes**: Snapshot acquisition eliminates motion artifacts, reducing errors in dynamic scenes by 60-75% compared to sequential methods.
- **Extremely Low Power**: 4.9W total power consumption (including a Raspberry Pi), achieving the first battery-powered passive depth camera.
- **Sparse but Reliable**: The working range reaches 860mm after discarding the 40% lowest-confidence pixels.

## Highlights & Insights
- **Elegant Implementation of Biomimetic Design**—While jumping spiders perceive depth using multi-layered retinae, Focal Split implements the same principle using a beamsplitter and dual sensors, costing only $500.
- **Three Orders of Magnitude Computational Savings**—Scaling down from millions of FLOPs in DNNs to 500 FLOPs in an analytical formula, making it truly suitable for edge deployment.
- **Fully Open-Source DIY Solution**—Provides 3D printing files, code, and assembly instructions, allowing anyone to replicate the project.

## Limitations & Future Work
- Only sparse depth maps are generated (textureless regions cannot be estimated).
- The working range is limited by SNR decay (~1 meter).
- Precise optical calibration is required.
- Lower resolution (480×360).
- Inherent limitation of passive methods—failure in texture-sparse regions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Biomimetic design + first wireless snapshot depth camera, cross-disciplinary innovation
- Experimental Thoroughness: ⭐⭐⭐ Thorough real-world hardware demonstration, but quantitative evaluation scenarios are limited
- Writing Quality: ⭐⭐⭐⭐ Clear optical derivation, practical DIY guide
- Value: ⭐⭐⭐⭐ Opens up a new ultra-low-power route for depth sensing on edge devices


## Related Work & Insights
- **vs Representative Methods in the Same Field**: Ours makes unique contributions to methodological design and complements existing methods.
- **vs Traditional Methods**: Compared to traditional solutions, ours achieves significant improvements in key metrics.
- **Insights**: Our technical roadmap provides important references for future related work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](../../NeurIPS2025/others/depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)

</div>

<!-- RELATED:END -->
