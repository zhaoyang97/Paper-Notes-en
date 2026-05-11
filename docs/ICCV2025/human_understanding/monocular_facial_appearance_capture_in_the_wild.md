---
title: >-
  [Paper Note] Monocular Facial Appearance Capture in the Wild
description: >-
  [ICCV 2025][Human Understanding][facial appearance capture] This paper proposes a method for reconstructing facial appearance attributes (diffuse albedo, specular intensity…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "facial appearance capture"
  - "inverse rendering"
  - "occlusion-aware"
  - "monocular video"
  - "split-sum approximation"
date: 2026-05-08
content_hash: 4e746f3b10b1eec7
---

# Monocular Facial Appearance Capture in the Wild

**Conference**: ICCV 2025
**arXiv**: [2412.12765](https://arxiv.org/abs/2412.12765)
**Code**: N/A
**Area**: Human Understanding
**Keywords**: facial appearance capture, inverse rendering, occlusion-aware, monocular video, split-sum approximation

## TL;DR

This paper proposes a method for reconstructing facial appearance attributes (diffuse albedo, specular intensity, specular roughness) from monocular head-rotation videos. By introducing an occlusion-aware split-sum approximation shading model, the method achieves studio-grade facial appearance capture quality without imposing any simplifying assumptions on the illumination environment.

## Background & Motivation

High-quality 3D facial scans are critical for film, gaming, telepresence, and related applications. Traditional approaches rely on multi-camera, controlled-lighting studio setups, which yield accurate appearance maps (diffuse albedo, specular intensity, specular roughness) but at prohibitive cost. While lightweight facial reconstruction has advanced in recent years, existing in-the-wild methods are subject to the following limitations:

- **SunStage** assumes a single point light source (the sun) in the scene, restricting applicable scenarios.
- **CoRA** requires a smartphone flash in a dark room.
- **FLARE** employs the standard split-sum approximation, ignoring self-occlusion and causing illumination to be baked into the albedo.
- **NextFace** relies on statistical priors, limiting its expressive capacity.

The core problem is that existing methods either assume specific lighting conditions or disregard facial self-occlusion effects (e.g., the shadow cast by the nose onto the cheek), leading to inaccurate appearance decomposition.

## Method

### Overall Architecture

The input is a monocular head-rotation video. In the preprocessing stage, keypoint-based monocular tracking is used to obtain an initial 3DMM mesh, a fixed camera pose, and per-frame head poses. An inverse rendering optimization via differentiable rendering then jointly solves for geometry, appearance parameters (diffuse albedo $\rho$, specular intensity, specular roughness), and environmental illumination.

### Key Designs

1. **Geometry Optimization (Laplacian Preconditioning)**: Vertex positions are optimized directly rather than through 3DMM blend weights. Following a framework similar to Nicolet et al., gradient steps are biased toward smooth solutions via the matrix $(I + \lambda_{geo} L)^2$:
$$v \leftarrow v - \eta(I + \lambda_{geo} L)^2 \frac{\partial \mathcal{L}}{\partial v}$$
With $\lambda_{geo}=19$, large learning rates can be employed while maintaining a smooth, self-intersection-free mesh. **Design Motivation**: This enables geometry and texture to be optimized simultaneously, eliminating the need for conventional two-stage pipelines.

2. **Occlusion-Aware Shading Model (Visibility-Modulated Split-Sum)**: The standard split-sum approximation factorizes the rendering equation into a BRDF integral and a pre-filtered environment map term, but neglects self-occlusion. This work introduces a visibility term $V(\mathbf{x}, \omega_i)$, incorporating per-point ray visibility modulation into the second integral. For the specular (low-roughness) component, an approximation $\tilde{V}(\mathbf{x}, \omega_r) \approx \frac{1}{K}\sum_{k=1}^{K} \frac{V(\mathbf{x}, \omega_k)}{D(\mathbf{n}, \omega_k, \omega_r, r)}$ is used to soften visibility via Monte Carlo sampling. For the diffuse component, OptiX ray tracing with multiple importance sampling is employed. **Design Motivation**: Correctly modeling self-occlusion is essential to prevent shadow baking into the albedo.

3. **Diffuse Regularization**: A weak regularization term $\mathcal{L}_{diffuse} = \|I_{diffuse}\|_2^2$ is added to encourage the diffuse rendering to be as small as possible. **Design Motivation**: This prevents specular signals from being excessively baked into the diffuse component, enabling better diffuse/specular separation.

### Loss & Training

The total loss is:
$$\mathcal{L} = \mathcal{L}_{img} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{Lap}\mathcal{L}_{Lap} + \lambda_{light}\mathcal{L}_{light} + \lambda_{rough}\mathcal{L}_{rough} + \lambda_{diffuse}\mathcal{L}_{diffuse}$$

- $\mathcal{L}_{img}$: L1 image reconstruction loss
- $\mathcal{L}_{mask}$: L1 mask loss (using MODNet segmentation)
- $\mathcal{L}_{Lap}$: Laplacian regularization to keep the optimized mesh close to the initial 3DMM
- $\mathcal{L}_{light}$: white light regularization
- $\mathcal{L}_{rough}$: total variation regularization on the roughness texture
- Geometry and texture are optimized simultaneously (no two-stage pipeline)

## Key Experimental Results

### Main Results (Reconstruction Error, Computed on Skin Regions)

| Method | PSNR ↑ | MAE ↓ | SSIM ↑ | LPIPS ↓ |
|--------|--------|-------|--------|---------|
| NextFace | 25.30 | 10.63 | 0.78 | 0.31 |
| SunStage | 29.47 | 5.28 | 0.88 | 0.14 |
| FLARE | 30.40 | 2.01 | 0.94 | 0.15 |
| Ours (w/o vis) | 34.55 | 1.79 | 0.96 | 0.10 |
| **Ours** | **38.09** | **1.18** | **0.97** | **0.10** |

The proposed method outperforms all baselines across every metric, achieving a PSNR gain of 7.69 dB over FLARE and reducing MAE by 41%.

### Ablation Study (Ray Visibility & Component Analysis)

| Configuration | Effect |
|---------------|--------|
| w/o visibility (standard split-sum) | Shadows baked into albedo; incorrect nostril-region shadows under relighting; PSNR 34.55 vs. 38.09 |
| w/o $\mathcal{L}_{diffuse}$ regularization | Specular signals excessively baked into the diffuse component |
| Optimizing 3DMM blend weights vs. direct vertex optimization | Direct vertex optimization recovers correct nose geometry from shading with lower reconstruction error |
| Ray tracing vs. modified split-sum (specular) | Ray tracing causes flickering and high-frequency artifacts in inverse rendering; modified split-sum is more stable |

Evaluation on a synthetic dataset further confirms the substantial advantage of occlusion-aware modeling: diffuse albedo error is markedly reduced, and shape recovery in self-occluded regions (e.g., lips) is more accurate.

### Key Findings

- The method operates across diverse indoor and outdoor scenes without any assumptions about illumination (no sun or flash required).
- Correct visibility modeling is the key to high-quality diffuse/specular separation.
- Direct vertex position optimization with preconditioning is more flexible than 3DMM parameterization.
- The modified split-sum for the specular component is more stable than direct ray tracing.

## Highlights & Insights

- The technical contribution is solid: incorporating the visibility term into the split-sum approximation is an elegant and efficient solution, and employing distinct strategies for specular and diffuse components (approximate visibility vs. ray tracing) reflects sound engineering judgment.
- The end-to-end framework requires no two-stage training; the ability to simultaneously optimize geometry and texture stems from the Laplacian preconditioning technique.
- Results are convincing: qualitative comparisons demonstrate clearly superior diffuse/specular separation over FLARE, with relighting quality approaching studio-grade.
- The method has high practical value: a simple head-rotation video suffices to produce facial assets compatible with VFX pipelines.

## Limitations & Future Work

- The method depends on the accuracy of head pose estimation; imprecise poses severely degrade reconstruction quality.
- Appearance cannot be recovered when the face remains in extreme shadow across all frames.
- The current facial template does not include an eye model.
- Correct skin color recovery is not guaranteed due to the inherent illumination–appearance ambiguity.
- The assumption of static expression limits applicability to videos with speech or rich facial motion.

## Related Work & Insights

- The core distinction from FLARE lies in visibility modeling: FLARE's standard split-sum ignores self-occlusion and incurs substantial baking artifacts, whereas this work addresses the problem at its source through a modified formulation.
- The distinction from SunStage lies in the lighting assumption: SunStage assumes a known solar position as a single point light source.
- Munkberg et al.'s differentiable split-sum serves as the foundation for the shading model proposed here.
- This work offers important insights for the facial scanning field: the gap between lightweight capture and studio-grade quality is steadily narrowing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The occlusion-aware split-sum approximation is the central innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive validation on both synthetic and real data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are clear and figures are of high quality.
- **Value**: ⭐⭐⭐⭐ Substantially narrows the gap between lightweight capture and studio-grade quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](../../CVPR2026/human_understanding/wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)
- [\[ICCV 2025\] NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction](ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)
- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[ICCV 2025\] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)

</div>

<!-- RELATED:END -->
