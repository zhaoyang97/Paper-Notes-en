---
title: >-
  [Paper Note] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration
description: >-
  [ICCV 2025][3D Vision][Monocular camera calibration] This paper proposes DM-Calib, a diffusion-based monocular camera intrinsic estimation method. It introduces a Camera Image representation that losslessly encodes intrinsics as a 3-channel image (azimuth + elevation + grayscale), fine-tunes Stable Diffusion to generate Camera Images, and extracts intrinsics via RANSAC. The method outperforms all baselines on 5 zero-shot datasets and extends camera calibration to metric depth estimation, pose estimation, and sparse-view 3D reconstruction.
tags:
  - ICCV 2025
  - 3D Vision
  - Monocular camera calibration
  - Camera Image
  - diffusion model prior
  - metric depth estimation
  - sparse-view reconstruction
date: 2026-05-08
content_hash: 4de1c5ca45817f8c
---

# Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration

**Conference**: ICCV 2025
**arXiv**: [2411.17240](https://arxiv.org/abs/2411.17240)
**Code**: [https://github.com/JunyuanDeng/DM-Calib](https://github.com/JunyuanDeng/DM-Calib)
**Area**: 3D Vision / Camera Calibration / Diffusion Models
**Keywords**: Monocular camera calibration, Camera Image, diffusion model prior, metric depth estimation, sparse-view reconstruction

## TL;DR
This paper proposes DM-Calib, a diffusion-based monocular camera intrinsic estimation method. It introduces a Camera Image representation that losslessly encodes intrinsics as a 3-channel image (azimuth + elevation + grayscale), fine-tunes Stable Diffusion to generate Camera Images, and extracts intrinsics via RANSAC. The method outperforms all baselines on 5 zero-shot datasets and extends camera calibration to metric depth estimation, pose estimation, and sparse-view 3D reconstruction.

## Background & Motivation

### State of the Field

**Background**: Monocular camera calibration is an ill-posed problem. Traditional methods rely on strong priors such as the Manhattan World assumption or calibration boards, resulting in poor generalization. Learning-based methods are constrained by training data scale. Diffusion models implicitly understand the relationship between focal length and image content (e.g., telephoto → shallow depth of field / compression effect; wide-angle → exaggerated perspective), and this prior knowledge can be leveraged for camera calibration.

### Mechanism

**Goal**: How can the implicit imaging priors encoded in diffusion models be effectively extracted for high-accuracy monocular camera intrinsic estimation? The key challenge is that numerical camera parameters $(f_x, f_y, c_x, c_y)$ are not directly compatible with image-based diffusion models.

## Method

### Overall Architecture
Input RGB image → VAE encodes RGB latent → GT intrinsics encoded as Camera Image representation → noise added to Camera Image latent → conditional UNet denoises to predict Camera Image → VAE decodes → RANSAC extracts intrinsics $(f_x, f_y, c_x, c_y)$ from Camera Image.

### Key Designs
1. **Camera Image Representation**: Encodes intrinsics as a 3-channel "image": Channel 1 = $\arctan(r_1/r_3)$ (azimuth), Channel 2 = $\arccos(r_2)$ (elevation), Channel 3 = grayscale image. Preserving high-frequency details from the input image reduces the domain gap with real images, making VAE encoding/decoding errors negligible. By contrast, incidence maps exhibit a large domain gap with real images.
2. **RANSAC Intrinsic Extraction**: Each pixel in the Camera Image encodes a ray direction. Any two pixels yield a linear equation $(\tan(\theta) \cdot f_x + c_x = u)$. RANSAC fits a line over all pixels — slope = focal length, intercept = principal point.
3. **Extension to Metric Depth**: The Camera Image is used as a conditional input to the UNet with single-step deterministic forward inference (no multi-step denoising), while the VAE decoder is jointly trained. This achieves the first diffusion-prior-based metric depth estimation.

### Loss & Training
- Intrinsic estimation: v-prediction loss + multi-resolution noise
- Metric depth: $\mathcal{L} = \|M \odot (D(U(z_x, z_c)) - d)\|$
- Intrinsics: AdamW, lr = 3e-5, 30K iterations, BS = 196, 8× A800
- Depth: same optimizer, BS = 96, ~5 days training

## Key Experimental Results

### Zero-Shot Camera Calibration (focal length error $e_f$ ↓)

| Method | Waymo | RGBD | ScanNet | MVS | Scenes11 | Average |
|--------|-------|------|---------|-----|----------|---------|
| WildCame | 0.210 | 0.097 | 0.128 | 0.170 | 0.170 | 0.155 |
| DiffCalib | 0.188 | 0.092 | 0.089 | 0.135 | 0.108 | 0.122 |
| **DM-Calib** | **best** | **best** | 0.089 | **best** | **best** | **best** |

### Sparse-View 3D Reconstruction (relative distance error)

| | Scene1 | Scene2 | Scene3 | Scene4 |
|---|--------|--------|--------|--------|
| w/o intrinsics | 1.67 | 0.87 | 1.03 | 1.43 |
| w/ intrinsics (DM-Calib) | **1.37** | **0.68** | **0.68** | **1.06** |

Reconstruction error reduced by ~20%.

### Ablation Study
- Camera Image $(\theta, \phi, g)$ > $(\theta, \phi, \theta)$: $e_f$ drops from 24.36° to ~4°
- Multi-resolution noise further reduces error
- Metric depth: removing Camera Image conditioning drops $\delta_1$ from 85.8 to 83.8
- Single-step vs. multi-step inference: single-step performs better (multi-step training with sparse GT is difficult)
- Fine-tuning the VAE decoder is critical for metric depth

## Highlights & Insights
- **Camera Image Design**: The key insight is placing a grayscale image in the third channel to reduce the domain gap, making VAE reconstruction error negligible — seemingly simple, but experimentally shown to make a dramatic difference.
- **RANSAC Intrinsic Extraction**: Transforms the mapping from dense Camera Image to 4 scalars into a straightforward line-fitting problem, achieving both robustness and efficiency.
- **Diffusion Models Understand Focal Length**: SD models genuinely encode the imaging characteristics of different focal lengths (as shown in Fig. 1 with telephoto vs. wide-angle generation), a finding that is valuable in its own right.
- **Intrinsics → Metric Depth**: Given accurate intrinsics, affine-invariant depth can be upgraded to metric depth — a critical yet often overlooked step in the pipeline.

## Limitations & Future Work
- Performance degrades on ultra-wide-angle (small focal length) images due to underrepresentation in training data
- Inference still requires multi-step diffusion sampling (can be accelerated with few-step methods)
- Metric depth training still requires sparse ground truth from LiDAR/RGBD sensors
- Non-pinhole camera models (e.g., radial distortion) are not addressed

## Related Work & Insights
- **vs. DiffCalib**: Also uses diffusion models but generates incidence maps, which suffer from a large domain gap and require joint training with depth; DM-Calib's Camera Image is more compatible with diffusion models and can be trained independently.
- **vs. WildCame/GeoCalib**: Non-diffusion methods relying on geometric features (e.g., vanishing points) with limited generalization.
- **vs. UniDepth**: Jointly trains intrinsics and depth, but mutual interference degrades intrinsic accuracy.

## Related Work & Insights
- The paradigm of applying diffusion model priors to 3D geometric tasks is worth studying.
- The Camera Image design philosophy — encoding non-image signals into image format to leverage pretrained diffusion models — has broad applicability.
- Intrinsic estimation is a critical component for any pipeline performing 3D reconstruction from in-the-wild images.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The Camera Image representation combined with diffusion priors for calibration is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers calibration, depth, pose, reconstruction, and metric evaluation across 5 zero-shot datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear; VAE reconstruction error analysis is convincing.
- **Value**: ⭐⭐⭐⭐⭐ The paradigm of encoding non-visual signals as images to exploit diffusion priors is highly inspiring.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector](charm3r_towards_unseen_camera_height_robust_monocular_3d_detector.md)
- [\[ICCV 2025\] ri3d few-shot gaussian splatting with repair and inpainting diffusion priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

<!-- RELATED:END -->
