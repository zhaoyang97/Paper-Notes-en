---
title: >-
  [Paper Note] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes DiffusionGS, which bakes 3D Gaussian point clouds into the denoiser of a diffusion model, enabling single-stage…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Diffusion Models"
  - "Single-view 3D Generation"
  - "Scene Reconstruction"
  - "Mixed Training"
date: 2026-05-08
content_hash: e4d86181f41e8db9
---

# Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2411.14384](https://arxiv.org/abs/2411.14384)  
**Code**: [https://caiyuanhao1998.github.io/project/DiffusionGS/](https://caiyuanhao1998.github.io/project/DiffusionGS/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Diffusion Models, Single-view 3D Generation, Scene Reconstruction, Mixed Training

## TL;DR

This paper proposes DiffusionGS, which bakes 3D Gaussian point clouds into the denoiser of a diffusion model, enabling single-stage, view-consistent single-view 3D object generation and scene reconstruction. Combined with a scene-object mixed training strategy and RPPC camera conditioning encoding, the method substantially outperforms existing approaches on PSNR/FID metrics while requiring only ~6 seconds for inference.

## Background & Motivation

Single-view image-to-3D is an important and challenging task. Existing methods follow three technical paradigms, each with its own limitations:

**Two-stage methods (mainstream)**: A 2D diffusion model first generates multi-view images, which are then fed into a 3D reconstruction model. The fundamental drawback is that 2D diffusion cannot guarantee 3D consistency and tends to collapse when the prompt view direction varies significantly.

**One-stage 3D diffusion methods**: 3D diffusion models based on triplane-NeRF representations. However, triplane resolution is limited, volume rendering is slow, and scaling to large scenes is difficult.

**Single-view scene reconstruction**: Relies on monocular depth estimators, which tend to fail under severe occlusion or large viewpoint changes.

Furthermore, 3D data is scarce (scene-level data amounts to only ~90K samples), and the distribution gap between object and scene data is large (objects: no background + surrounding cameras; scenes: dense background + trajectory cameras), making direct mixed training prone to divergence.

## Method

### Overall Architecture

At each diffusion timestep $t$, DiffusionGS directly predicts pixel-aligned 3D Gaussian point clouds $\mathcal{G}_\theta$ rather than predicting noise $\epsilon$. These Gaussian point clouds are rendered into multi-view images via differentiable rasterization for 2D supervision. Inference uses 30-step DDIM sampling.

- **Input**: 1 clean condition view $\mathbf{x}_{con}$ + $N$ noisy views $\mathcal{X}_t$ + corresponding viewpoint conditions
- **Output**: $(N+1) \times H \times W$ pixel-aligned Gaussian primitives

### Key Designs

1. **Pixel-Aligned 3D Gaussian Diffusion Denoiser**:

    - Each pixel predicts one Gaussian primitive $G_t^{(k)}(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, \boldsymbol{c})$, for a total of $N_g = (N+1)HW$ primitives.
    - Gaussian centers lie on pixel-aligned rays: $\boldsymbol{\mu}_t^{(k)} = \boldsymbol{o}^{(k)} + u_t^{(k)} \boldsymbol{d}^{(k)}$
    - Depth is linearly interpolated between near and far bounds: $u_t^{(k)} = w_t^{(k)} u_{near} + (1 - w_t^{(k)}) u_{far}$
    - The denoiser follows a Transformer architecture, with each block comprising MSA + MLP + LN; timestep conditioning is injected via adaLN.
    - Outputs are mapped to 14-channel per-pixel Gaussian parameter maps through a Gaussian decoder.
    - $x_0$-prediction (rather than $\epsilon$-prediction) is adopted to ensure clean texture and complete 3D structure at each step.

2. **Scene-Object Mixed Training Strategy**:

    - **Viewpoint selection constraints**: Limits are imposed on the polar angle $\theta_{cd}^{(i)} \leq \theta_1$ and azimuth angle $\cos(\varphi_1)$ between noisy and condition views to guarantee sufficient overlap.
    - **Dual Gaussian decoders**: Separate MLP decoders for object-level and scene-level data with different depth ranges (objects: [0.1, 4.2]; scenes: [0, 500]); one decoder is removed during fine-tuning.
    - **Distribution alignment**: Controls consistency in camera conditioning, Gaussian point cloud distribution, and imaging depth.

3. **Reference-Point Plücker Coordinate (RPPC)**:

    - The moment vector in conventional Plücker coordinates $\boldsymbol{r} = (\boldsymbol{o} \times \boldsymbol{d}, \boldsymbol{d})$ has limitations in perceiving depth and 3D geometry.
    - RPPC replaces the moment vector with the point on the ray closest to the world coordinate origin:
    $$\boldsymbol{r} = (\boldsymbol{o} - (\boldsymbol{o} \cdot \boldsymbol{d})\boldsymbol{d}, \boldsymbol{d})$$
    - This satisfies the translation invariance assumption of the 4D light field.
    - The reference point directly encodes ray position and relative depth, and flows through each Transformer block via skip connections.

### Loss & Training

The denoising loss is a weighted combination of L2 loss and VGG-19 perceptual loss:
$$\mathcal{L}_{de} = \mathcal{L}_2(\hat{\mathcal{X}}_{(0,t)}, \mathcal{X}_0) + \lambda \cdot \mathcal{L}_{VGG}(\hat{\mathcal{X}}_{(0,t)}, \mathcal{X}_0)$$

The novel-view loss $\mathcal{L}_{nv}$ shares the same structure. The point distribution loss $\mathcal{L}_{pd}$ is used during training warm-up to regularize the Gaussian point cloud distribution toward a target standard deviation $\sigma_0=0.5$.

The overall training objective is:
$$\mathcal{L} = (\mathcal{L}_{de} + \mathcal{L}_{nv}) \cdot \mathbf{1}_{iter>iter_0} + \mathcal{L}_{pd} \cdot \mathbf{1}_{iter \leq iter_0} \cdot \mathbf{1}_{object}$$

Training pipeline:
- Mixed training: 32×A100, Objaverse+MVImgNet+RealEstate10K+DL3DV10K, 40K iterations
- Separate fine-tuning: 64×A100, objects 80K / scenes 54K iterations
- High-resolution fine-tuning: 256→512 resolution, 20K iterations

## Key Experimental Results

### Main Results (Single-view Object Generation)

| Method | ABO PSNR↑ | ABO FID↓ | GSO PSNR↑ | GSO FID↓ | Inference Time |
|--------|-----------|----------|-----------|----------|----------------|
| LGM | 16.01 | 86.32 | 14.27 | 75.55 | 4.1s |
| GS-LRM | 18.78 | 123.55 | 17.70 | 112.96 | - |
| DMV3D | 23.69 | 32.28 | 20.82 | 33.48 | 31.4s |
| **DiffusionGS** | **25.89** | **9.03** | **22.07** | **11.52** | **5.8s** |

Single-view scene reconstruction (RealEstate10K):

| Method | PSNR↑ | FID↓ | LPIPS↓ |
|--------|-------|------|--------|
| PixelNeRF | 17.46 | 159.52 | 0.5525 |
| Splatter-Image | 18.21 | 120.35 | 0.4839 |
| Flash3D | 20.29 | 35.03 | 0.3610 |
| **DiffusionGS** | **21.63** | **15.87** | **0.2743** |

### Ablation Study (GSO Dataset)

| Configuration | PSNR↑ | SSIM↑ | FID↓ | Note |
|---------------|-------|-------|------|------|
| Baseline (no timestep control) | 17.63 | 0.7928 | 118.31 | Starting point |
| + DiffusionGS framework | 20.57 | 0.8120 | 47.86 | +2.94 dB |
| + Point distribution loss $\mathcal{L}_{pd}$ | 20.94 | 0.8423 | 28.41 | +0.37 dB |
| + Mixed training | 21.73 | 0.8515 | 17.79 | +0.79 dB |
| + RPPC | **22.07** | **0.8545** | **11.52** | +0.34 dB |

User study (25 participants, 6-point scale): DiffusionGS scores 4.88, substantially outperforming LGM (3.04), DMV3D (3.16), and 12345++ (3.81).

### Key Findings

- **3D consistency is the core advantage**: Predicting 3D Gaussians at each step eliminates the view misalignment problem inherent in 2D multi-view diffusion, maintaining geometric correctness even when the prompt view is not frontal.
- **Mixed training improves texture realism**: Scene data introduces real-world texture priors, reducing the unrealistic textures caused by training on synthetic data alone.
- **RPPC improves depth perception**: Compared to conventional Plücker coordinates, RPPC yields +0.28 dB PSNR and −7.09 FID improvement on scene reconstruction.
- **No depth estimator required**: By generating multi-view predictions along camera trajectories, the method predicts finer Gaussian point clouds and outperforms depth-estimation-dependent methods in occluded regions.
- Compared to the SOTA 2D method PhotoNVS + post-hoc GS, DiffusionGS completes in 6 seconds what PhotoNVS requires 2,478 seconds to achieve, while also surpassing it by 6.32 dB in PSNR.

## Highlights & Insights

- "Baking 3DGS into the diffusion denoiser" is an elegant design that elegantly resolves the compatibility issue between 3D representations and the diffusion framework using pixel-aligned Gaussians.
- The choice of $x_0$-prediction over $\epsilon$-prediction is well-motivated: noisy Gaussians lack texture information and would compromise view consistency.
- The viewpoint constraint design in the mixed training strategy is practical; the two angular constraints separately control position and orientation, ensuring training stability.
- The intuition behind RPPC is clear: the reference point encodes the ray's position in 3D space more directly than the moment vector.

## Limitations & Future Work

- Training demands substantial computational resources (32–64×A100), limiting reproducibility for academic groups.
- Scene-level data consists of only ~90K samples with limited viewpoint variation; larger and more diverse data could further improve performance.
- The number of pixel-aligned Gaussians scales linearly with resolution, creating significant memory and computational pressure at high resolutions.
- The current design supports only fixed near/far depth ranges, which may be limiting for scenes with highly variable depth.
- Text-to-3D relies on external 2D generators (Stable Diffusion / FLUX / Sora); an end-to-end approach may be preferable.

## Related Work & Insights

- Compared to DMV3D (triplane-NeRF diffusion), DiffusionGS replaces NeRF with 3DGS, achieving over 5× speedup while supporting scene-level tasks.
- The key distinction from Flash3D/VistaDream is that the method does not rely on monocular depth estimators, instead learning geometry through the diffusion process itself.
- The mixed training strategy is extensible to additional 3D data sources (indoor scans, autonomous driving data, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Integrating 3DGS into the diffusion denoiser, the RPPC design, and the mixed training strategy are all original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers object/scene generation and reconstruction across multiple datasets, with user studies and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline figures are clear and mathematical derivations are complete, though notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ A unified framework for object generation and scene reconstruction with 6-second inference time offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)

</div>

<!-- RELATED:END -->
