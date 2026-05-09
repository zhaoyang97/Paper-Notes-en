---
title: >-
  [Paper Note] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] A Neural ODE is used to model continuous camera motion trajectories during exposure, combining rigid-body transformations with a learnable Continuous Motion Refinement (CMR) transform to reconstruct sharp 3D Gaussian scenes from motion-blurred images, achieving substantial improvements over the state of the art across all benchmarks.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - motion blur
  - neural ODE
  - camera trajectory
  - deblurring
date: 2026-05-08
content_hash: 2900151fd083ece8
---

# CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images

**Conference**: ICCV 2025
**arXiv**: [2503.05332](https://arxiv.org/abs/2503.05332)
**Code**: [https://github.com/Jho-Yonsei/CoMoGaussian](https://github.com/Jho-Yonsei/CoMoGaussian) (Project page: [https://Jho-Yonsei.github.io/CoMoGaussian/](https://Jho-Yonsei.github.io/CoMoGaussian/))
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, motion blur, neural ODE, camera trajectory, deblurring

## TL;DR
A Neural ODE is used to model continuous camera motion trajectories during exposure, combining rigid-body transformations with a learnable Continuous Motion Refinement (CMR) transform to reconstruct sharp 3D Gaussian scenes from motion-blurred images, achieving substantial improvements over the state of the art across all benchmarks.

## Background & Motivation
3DGS relies on sharp image inputs; however, in practice, small apertures lead to long exposures and near-inevitable camera motion blur. Existing deblurring methods (Deblur-NeRF, DP-NeRF, BAD-Gaussians, etc.) attempt to estimate camera motion trajectories to simulate the blur process, but none genuinely enforces **continuity** of motion — they either rely on simple linear interpolation or spline fitting, or directly predict discrete poses. This causes the estimated motion to exhibit abrupt changes or piecewise discontinuities that deviate from the true physical camera trajectory, particularly under complex nonlinear motion.

## Core Problem
How to accurately model **continuous** camera motion trajectories during exposure within the 3DGS framework, enabling high-quality sharp scene reconstruction from motion-blurred images. The key challenges are: (1) motion blur is inherently a continuous-time integral, and discrete sampling introduces errors; (2) rigid-body transformations preserve geometric consistency but lose accuracy under limited sampling.

## Method

### Overall Architecture
The inputs are a set of motion-blurred images together with COLMAP poses. For each image, the calibrated pose is treated as the center point, and a Neural ODE generates $N=9$ continuous camera poses over the exposure time interval. A sharp image is rendered from each pose via Mip-Splatting, and the blurred image is synthesized through pixel-wise weighted summation, supervised against the input blurred image. At inference, sharp images are obtained by rendering directly from the calibrated poses.

### Key Designs
1. **Continuous Rigid-Body Motion Modeling**: Image indices are embedded as features, passed through an encoder to obtain an initial latent state, and then a Neural ODE (4th-order Runge-Kutta solver) generates continuous screw axis parameters in the latent space, which are mapped to SE(3) transformation matrices via matrix exponentiation. A key improvement over DP-NeRF is the decoupling of the rotation axis $\hat{\omega}$ and rotation angle $\theta$ into independently modeled quantities, with the rotation axis normalized to a unit vector.
2. **Continuous Motion Refinement (CMR) Transform**: Because numerical integration is discrete (only $N$ samples), rigid-body transformations cannot perfectly approximate continuous motion. CMR employs a separate Neural ODE to generate an additional transformation matrix with higher degrees of freedom (not strictly SO(3)), initialized near the identity matrix, with an orthogonality regularization to keep it close to a valid rotation. It functions as a "residual correction" on top of the rigid-body motion.
3. **Pixel-wise Weights and Mask**: A shallow CNN generates pixel-wise softmax weights for the $N$ rendered images to synthesize the blurred output. A scalar mask blends the sharp and blurred renderings — pixels that are inherently sharp are rendered directly without blurring.
4. **Advantages of Neural ODE**: Forward and backward passes share the same neural derivative function, ensuring the entire trajectory lives in a consistent function space and yielding natural temporal continuity. Compared to MLP (no sequential structure) and GRU (requiring separate units for forward/backward passes, causing discontinuities), Neural ODE produces visually the smoothest trajectories.

### Loss & Training
- Total loss: $\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c\mathcal{L}_{\text{D-SSIM}} + \lambda_o\mathcal{L}_o + \lambda_\mathcal{M}\mathcal{L}_\mathcal{M}$
- $\lambda_c=0.3$, $\lambda_o=10^{-4}$, $\lambda_\mathcal{M}=10^{-3}$
- Staged training: only Gaussian primitives are trained for the first 1k iterations; motion transforms are introduced from 1k–3k without pixel weights/mask; all components are trained jointly from 3k onward.
- Total of 40k iterations on a single RTX 4090.

## Key Experimental Results

| Dataset | Metric | CoMoGaussian | Deblurring 3DGS | BAGS | Gain over Prev. SOTA |
|--------|------|-------------|-----------------|------|-----------|
| Deblur-NeRF Synthetic | PSNR/SSIM/LPIPS | **31.02/0.917/0.049** | 28.24/0.858/0.105 | 27.34/0.835/0.112 | +2.78 PSNR |
| Deblur-NeRF Real | PSNR/SSIM/LPIPS | **27.85/0.843/0.082** | 26.61/0.822/0.110 | 26.70/0.824/0.096 | +1.15 PSNR |
| ExbluRF Real | PSNR/SSIM/LPIPS | **30.15/0.756/0.311** | 27.36/0.680/0.399 | 24.70/0.584/0.528 | +2.79 PSNR |

### Ablation Study
- Rigid-body transformation alone improves over the baseline (Mip-Splatting) by 5+ PSNR.
- CMR yields a further gain of +0.55/+0.02/+0.013 (PSNR/SSIM/LPIPS) on top of the rigid-body transform.
- Orthogonality regularization has modest numerical impact (+0.11 PSNR) but qualitatively preserves finer detail by preventing shearing/scaling distortions.
- Neural ODE > GRU (+0.45 PSNR) > MLP (+0.42 PSNR), and all substantially outperform physical-space methods (linear interpolation: 21.0; B-spline: 21.7).
- Performance saturates near $N=9$; with CMR, $N=9$ surpasses pure rigid-body motion at $N=13$.
- Training time is 1.33 h (slower than other 3DGS methods, though rendering speed is identical).

## Highlights & Insights
- **Neural ODE is a natural choice for camera motion modeling**: continuous-time dynamics directly correspond to continuous camera motion, and the shared forward/backward parameterization ensures trajectory consistency.
- **The CMR design pattern is reusable**: when discrete approximations are insufficient, adding a residual correction module initialized near the identity with regularization constraints is both simple and effective.
- **Decoupling rotation axis and angle** demonstrates the importance of attending to the mathematical structure of the problem.
- Performance on sharp images is on par with Mip-Splatting (27.56 vs. 27.71), confirming that no degradation is introduced.

## Limitations & Future Work
- No distinction is made between moderate and extreme blur; all images use the same $N=9$ sampling — adaptive $N$ could improve efficiency.
- Training time is longer than other 3DGS methods (1.33 h vs. 0.2–0.83 h), primarily because $N$ renders are required per image.
- Only camera motion blur is addressed; defocus blur and rolling shutter are not handled.
- Dynamic objects in the scene are not considered (static scene assumption).

## Related Work & Insights
- vs. **BAD-Gaussians**: uses linear interpolation/B-splines to interpolate poses in physical space; the PSNR gap is large (21.69 vs. 27.85), demonstrating that simple interpolation is insufficient.
- vs. **Deblurring 3DGS**: synthesizes blurred training images by modifying Gaussian parameters without explicitly modeling the motion trajectory; PSNR is lower by approximately 1–3 dB.
- vs. **BAGS**: estimates a blur-agnostic degradation kernel via CNN; performs poorly on ExbluRF (24.7 vs. 30.15), showing that 2D kernels cannot substitute for 3D motion modeling.
- vs. **SMURF**: also employs Neural ODE but warps rays only in 2D pixel space; CoMoGaussian provides a more complete 3D spatial model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Introducing Neural ODE into 3DGS deblurring is a natural yet effective combination; the CMR design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, extensive ablations (components, ODE type, $N$, regularization, sharp images), complete per-scene results.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations (the appendix even derives the Rodrigues formula), though the prose is occasionally verbose.
- **Value**: ⭐⭐⭐ — A task-specific method, but the Neural ODE and residual correction design patterns have transfer value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)
- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)

<!-- RELATED:END -->
