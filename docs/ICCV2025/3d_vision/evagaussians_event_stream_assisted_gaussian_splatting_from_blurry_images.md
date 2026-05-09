---
title: >-
  [Paper Note] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes EvaGaussians, a framework that leverages the high temporal resolution event streams from event cameras to assist 3D Gaussian Splatting in learning from motion-blurred images. Through event-assisted initialization, joint blur/event reconstruction losses, and event-assisted geometric regularization, the method achieves high-fidelity novel view synthesis while maintaining real-time rendering efficiency.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Event Camera
  - Motion Deblurring
  - Novel View Synthesis
  - Bundle Adjustment
date: 2026-05-08
content_hash: 7bbf8690f427d8d6
---

# EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images

**Conference**: ICCV 2025
**arXiv**: [2405.20224](https://arxiv.org/abs/2405.20224)
**Code**: Available (project page released)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Event Camera, Motion Deblurring, Novel View Synthesis, Bundle Adjustment

## TL;DR

This paper proposes EvaGaussians, a framework that leverages the high temporal resolution event streams from event cameras to assist 3D Gaussian Splatting in learning from motion-blurred images. Through event-assisted initialization, joint blur/event reconstruction losses, and event-assisted geometric regularization, the method achieves high-fidelity novel view synthesis while maintaining real-time rendering efficiency.

## Background & Motivation

While 3D Gaussian Splatting (3D-GS) achieves remarkable performance in novel view synthesis, it critically relies on high-quality sharp images and accurate camera poses:

**Prevalence of Motion Blur**: In high-speed UAV and robotics scenarios or low-light environments, motion blur is nearly unavoidable. Blurry images cause COLMAP feature matching to fail, rendering pose estimation and point cloud initialization unreliable.

**Limitations of Prior Work**:
- Deblurring NeRF methods (e.g., BAD-NeRF) can model the blur process, but suffer from slow training and lack real-time rendering support.
- BAD-Gaussians extends 3DGS to handle blur, but still relies on COLMAP initialization, which fails under severe blur.
- Event-assisted NeRF methods (E2NeRF, EDNeRF) leverage event streams but are equally slow to train.

**Opportunity from Event Cameras**: Event cameras asynchronously record per-pixel brightness changes at microsecond-level temporal resolution with high dynamic range, making them naturally suited for addressing motion blur.

**Lack of Benchmarks**: No evaluation dataset simultaneously containing event streams and RGB frames exists.

## Method

### Overall Architecture

EvaGaussians seamlessly integrates event streams into both the initialization and optimization stages of 3D-GS:
1. **Event-Assisted Initialization**: Recovers latent sharp frames from blurry images using the EDI model for COLMAP pose estimation.
2. **Event-Assisted Bundle Adjustment**: Jointly optimizes 3D-GS parameters and camera trajectories within exposure time.
3. **Event-Assisted Geometric Regularization**: Stabilizes 3D-GS geometry using intensity images derived from event streams.

### Key Designs

**1. Event-Assisted Initialization**

A motion-blurred image is the temporal average of instantaneous images within the exposure interval:
$$\mathbf{B} = \frac{1}{\tau}\int_{s-\tau/2}^{s+\tau/2}\mathbf{I}(t)dt$$

Using the Event-based Double Integral (EDI) model:
$$\mathbf{B} = \mathbf{I}(s) \cdot \frac{1}{\tau}\int \exp(c\mathbf{E}(t))dt$$

Given the event stream and blurry image, $\mathbf{I}(s)$ can be solved, and then $\mathbf{I}(t) = \mathbf{I}(s) \cdot \exp(c\mathbf{E}(t))$ is used to uniformly sample $n$ latent images within the exposure time. These texture-rich images are fed into COLMAP to obtain initial poses and point clouds.

**2. Event-Assisted Bundle Adjustment**

A learnable offset $\tilde{\mathbf{P}}_i = \mathbf{P}_i + \mathbf{d}_i$ is added to each EDI-derived pose and jointly optimized during training:

- **Blur Reconstruction Loss**: $n$ images are rendered along the camera trajectory and averaged to simulate blur: $\tilde{\mathbf{B}} = \frac{1}{n}\sum_{i=1}^n \tilde{\mathbf{I}}_i$, then compared against the real blurry image:
$$\mathcal{L}_{blur} = (1-\lambda_1)\|\mathbf{B} - \tilde{\mathbf{B}}\|_1 + \lambda_1 \cdot \text{D-SSIM}(\mathbf{B}, \tilde{\mathbf{B}})$$

- **Event Reconstruction Loss**: The rendered image sequence is converted to an event map via a differentiable event simulator and compared against the ground-truth event map:
$$\mathcal{L}_{event} = \frac{1}{m}\sum_{i=1}^m \|\mathbf{E}_i - \tilde{\mathbf{E}}_i\|_1$$

**3. Event-Assisted Geometric Regularization**

The event stream enables derivation of continuous grayscale intensity images $\mathbf{G}(t)$ beyond the exposure interval:

- **Intensity Reconstruction Loss**: Randomly samples time points $t$ between adjacent blurry frames, constraining the rendered grayscale image to match the event-derived grayscale image:
$$\mathcal{L}_{int} = (1-\lambda_2)\|\mathbf{G}(t) - \tilde{\mathbf{G}}(t)\|_1 + \lambda_2 \cdot \text{D-SSIM}$$

- **Intensity-Aware Depth Regularization**: Based on the observation that depth variation should correlate with intensity variation, Sobel gradients are used:
$$\mathcal{L}_{depth} = \frac{1}{N}\sum_{x,y}(|\partial_x\tilde{\mathbf{D}}|e^{-\beta|\partial_x\mathbf{G}|} + |\partial_y\tilde{\mathbf{D}}|e^{-\beta|\partial_y\mathbf{G}|})$$

### Loss & Training

Total loss: $\mathcal{L}_{total} = \lambda_{blur}\mathcal{L}_{blur} + \lambda_{event}\mathcal{L}_{event} + \lambda_{int}\mathcal{L}_{int} + \lambda_{depth}\mathcal{L}_{depth}$

- Hyperparameters: $\lambda_{blur}=1.0$, $\lambda_{event}=5e{-3}$, $\lambda_{int}=1e{-3}$, $\lambda_{depth}=1e{-2}$
- Trained for 50k iterations; event loss is introduced after 3k iterations, bypassing densification to simplify subsequent optimization.
- Progressive training: starts at $0.3\times$ downsampled resolution for the first 30% of iterations, gradually increasing to full resolution.
- $n=9$ camera poses are optimized within each exposure interval.
- Single RTX 4090 GPU.

## Key Experimental Results

### Main Results

**EvaGaussians-Blender Synthetic Dataset (novel view synthesis, averaged across scenes per scale):**

| Scene Type | Metric | B-NeRF | BAD-NeRF | BAD-GS | EDNeRF | **Ours** |
|---------|------|--------|----------|--------|--------|----------|
| Large | PSNR↑ | 21.33 | 23.85 | 23.86 | 24.63 | **26.02** |
| Large | SSIM↑ | .6781 | .7323 | .7325 | .7525 | **.8064** |
| Large | LPIPS↓ | .4249 | .3480 | .3473 | .3279 | **.2680** |
| Medium | PSNR↑ | 24.08 | 28.46 | 28.46 | 28.91 | **30.47** |
| Object-level | PSNR↑ | 22.28 | 27.33 | 27.86 | 29.83 | **30.24** |

**EvaGaussians-DAVIS Real-World Dataset (no-reference quality metrics):**

| Metric | B-3DGS | BAD-GS | EDNeRF | **Ours** | Improvement |
|------|--------|--------|--------|----------|--------|
| BRISQUE↓ | 73.80 | 60.89 | 58.63 | **53.96** | 15.4% |
| NIQE↓ | 12.01 | 9.902 | 9.011 | **8.371** | 19.5% |
| PIQE↓ | 52.74 | 43.51 | 44.63 | **41.53** | 11.5% |
| RankIQA↓ | 7.542 | 6.223 | 5.320 | **4.895** | 22.8% |

### Ablation Study

**Loss function ablation (large + medium synthetic scenes):**

| $\mathcal{L}_{blur}$ | $\mathcal{L}_{event}$ | $\mathcal{L}_{int}$ | $\mathcal{L}_{depth}$ | Large PSNR | Medium PSNR |
|-------|---------|-------|---------|-----------|-----------|
| ✓ | | | | 24.98 | 29.10 |
| ✓ | ✓ | | | 25.71 | 29.94 |
| ✓ | ✓ | ✓ | | 25.54 | 30.05 |
| ✓ | ✓ | ✓ | ✓ | **26.02** | **30.47** |

**Robustness to blur severity:**

| Blur Level | PSNR | SSIM | LPIPS |
|---------|------|------|-------|
| Mild | 26.38 | .8163 | .2694 |
| Moderate | 25.71 | .7949 | .2745 |
| Severe | 25.04 | .7886 | .2802 |

### Key Findings

- Comprehensively outperforms SOTA across all scene scales: +1.39 dB PSNR on large scenes, +1.56 dB on medium scenes.
- The event reconstruction loss is the largest contributing factor (+0.73 PSNR), followed by depth regularization.
- Robust across varying blur severities, with PSNR fluctuation of only 1.34 dB.
- Achieves substantial gains over all baselines on NR-IQA metrics for real-world data.
- Pose optimization significantly reduces ATE (Absolute Trajectory Error).
- $n=9$ poses achieves the best balance between performance and efficiency.

## Highlights & Insights

1. **Full Exploitation of Event Streams**: Event information is utilized at every stage — initialization (EDI deblurring → COLMAP), optimization (event reconstruction loss), and regularization (event-derived intensity maps → depth constraints).
2. **Physically Consistent Blur Modeling**: The method explicitly models the camera trajectory and blur formation process within the exposure time, rather than learning a deblurring kernel.
3. **Progressive Training Strategy**: Starting from low resolution and gradually scaling to full resolution, combined with delayed introduction of the event loss, leads to more stable training.
4. **New Dataset Contribution**: The synthetic and real-world datasets fill the gap in benchmarks for joint event and RGB evaluation.

## Limitations & Future Work

1. Scenes with extremely complex textures under severe blur remain challenging.
2. The event threshold $c$ in the EDI model requires manual tuning for synthetic vs. real data.
3. The limited resolution of real event cameras (346×260) constrains practical applicability.
4. Training for 50k iterations is considerably more time-consuming compared to standard 3DGS at 7k iterations.
5. Integration with more advanced event representations (e.g., time surface, voxel grid) remains unexplored.

## Related Work & Insights

- The bundle adjustment strategy from BAD-NeRF/BAD-GS is inherited and enhanced with better event-assisted initialization and constraints.
- E2NeRF/EDNeRF pioneered joint event and RGB modeling; EvaGaussians substantially surpasses both in efficiency and reconstruction quality.
- The intensity-aware depth regularization is inspired by classical edge-aware smoothing techniques.

## Rating

- Novelty: ⭐⭐⭐⭐ (Event stream-assisted 3DGS is a novel combination, though individual sub-modules have precedents)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Synthetic + real data, multi-scale scenes, 9 baselines, detailed ablation)
- Writing Quality: ⭐⭐⭐⭐ (Clear method description with complete derivations)
- Value: ⭐⭐⭐⭐ (Addresses an important practical problem; dataset contribution adds extra value)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](../../AAAI2026/3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)

<!-- RELATED:END -->
