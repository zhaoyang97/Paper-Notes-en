---
title: >-
  [Paper Note] Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction
description: >-
  [ECCV2024][3D Vision][Sparse-view reconstruction] A fine-tuned Stable Diffusion + ControlNet is used to transform coarse NeRF/3DGS renderings into high-quality pseudo-observations. By densifying sparse input views by $5$-$10\times$ before retraining, this approach outperforms methods like FreeNeRF by $1$-$2\text{ dB}$ PSNR on datasets such as Hypersim, LLFF, and ScanNet, while training about $10\times$ faster than diffusion-regularization methods.
tags:
  - "ECCV2024"
  - "3D Vision"
  - "Sparse-view reconstruction"
  - "Diffusion models"
  - "NeRF"
  - "3D Gaussian Splatting"
  - "Pseudo-observation generation"
date: 2026-05-08
content_hash: 4c69148a071492d3
---

# Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-observations for High-Quality Sparse-View Reconstruction

**Conference**: ECCV2024  
**arXiv**: [2305.15171](https://arxiv.org/abs/2305.15171)  
**Code**: [https://xinhangliu.com/deceptive-nerf-3dgs](https://xinhangliu.com/deceptive-nerf-3dgs)  
**Area**: 3D Vision  
**Keywords**: Sparse-view reconstruction, Diffusion models, NeRF, 3D Gaussian Splatting, Pseudo-observation generation

## TL;DR
A fine-tuned Stable Diffusion + ControlNet is used to transform coarse NeRF/3DGS renderings into high-quality pseudo-observations. By densifying sparse input views by $5$-$10\times$ before retraining, this approach outperforms methods like FreeNeRF by $1$-$2\text{ dB}$ PSNR on datasets such as Hypersim, LLFF, and ScanNet, while training about $10\times$ faster than diffusion-regularization methods.

## Background & Motivation

**Background**: NeRF/3DGS performs exceptionally well with dense views, but suffers from severe artifacts and geometric collapse under sparse views (3-20 images) due to insufficient constraints.

**Limitations of Prior Work**: Existing sparse-view methods fall into two categories: (a) frequency regularization (e.g., FreeNeRF), which is limited to simple priors; (b) diffusion model regularization (e.g., DiffusioNeRF, ReconFusion), which invokes the diffusion model at every training step and is extremely slow.

**Key Challenge**: Sparse views lack observational information $\rightarrow$ additional prior/generative information is required to fill the gap, but using diffusion models as step-by-step regularizers leads to an explosion in training time.

**Goal**: Can a diffusion model be used to generate a sufficient number of high-quality pseudo-observations in a single pass, which can then be directly treated as real training views?

**Key Insight**: Train a coarse NeRF/3DGS using sparse inputs $\rightarrow$ render RGB + depth + uncertainty maps at novel viewpoints $\rightarrow$ use a diffusion model to repair/enhance them into realistic images $\rightarrow$ mix the generated images into the training set for retraining.

**Core Idea**: Shift the diffusion model from a "step-by-step regularizer" to a "one-off data augmenter"—the generated pseudo-observations are realistic enough ("deceptive") to be directly used as real views for training.

## Method

### Overall Architecture
Iterative pipeline: (1) Train an initial coarse NeRF/3DGS using sparse views; (2) Render RGB, depth, and uncertainty maps from novel viewpoints; (3) Pass these three as conditions to a fine-tuned diffusion model to generate pseudo-observations; (4) Discard the 20% lowest-quality pseudo-observations (filtered by LPIPS); (5) Retrain NeRF/3DGS using both real and pseudo-observational images. Densification of $5$-$10\times$ can be achieved.

### Key Designs

1. **Correspondence-based Uncertainty Map**:

    - **Function**: Computes the uncertainty of each pixel in the novel viewpoint to guide the diffusion model on which regions require more generation and which are trustworthy.
    - **Mechanism**: Uses the rendered depth to warp the nearest input view to the novel viewpoint; the squared pixel difference between the warped image and the rendered image represents the uncertainty. No modification to the NeRF/3DGS representation is required.
    - **Design Motivation**: Satisfies epipolar constraints—low-uncertainty regions (already covered by input views) should remain consistent, while high-uncertainty regions (occluded/unseen areas) are left to the diffusion model to generate freely.

2. **Deceptive Diffusion Model**:

    - **Function**: Fine-tuning Stable Diffusion + ControlNet to generate realistic pseudo-observations conditioned on 5 channels (3 RGB + 1 depth + 1 uncertainty).
    - **Mechanism**: ControlNet encodes conditioning information, while textual embedding combines text prompts with textual inversion to enhance robustness to artifacts.
    - **Training Data**: Approximately 20,000 images from 200 scenes in Hypersim + CO3D are used to construct "coarse-to-fine NeRF pairs" as training pairs. RGB-D pairs enhanced with Gaussian noise are additionally introduced for data augmentation.

3. **Quality Filtering and Progressive Densification**:

    - **Function**: After generating pseudo-observations, filters them based on their LPIPS similarity to the coarse rendering, discarding the worst 20%.
    - **Mechanism**: Quiet poor pseudo-observations would introduce noise; moderate filtering performs better than utilizing all generated samples.
    - **Densification $5\times$ vs. $10\times$**: $5\times$ reduces artifacts but lacks detail recovery, while $10\times$ yields the best results.

### Loss & Training
The diffusion model is trained using the standard denoising loss (DDPM objective). NeRF/3DGS retraining employs the original reconstruction loss ($L_2$ + LPIPS) without distinguishing between real and pseudo-observations—since the pseudo-observations are of high enough quality to be directly utilized as real images.

## Key Experimental Results

### Main Results

**Hypersim Dataset**:

| Method | 5-view PSNR | 10-view PSNR | 20-view PSNR | 5-view LPIPS |
|------|------------|-------------|-------------|-------------|
| FreeNeRF | 17.20 | 18.06 | 20.20 | 0.431 |
| Deceptive-NeRF | 18.91 | 19.88 | 21.23 | 0.322 |
| **Deceptive-3DGS** | **19.31** | **21.45** | **21.61** | **0.265** |

**LLFF Dataset**:

| Method | 3-view PSNR | 6-view PSNR | 9-view PSNR |
|------|------------|-------------|-------------|
| FreeNeRF | 19.63 | 23.73 | 25.13 |
| Deceptive-3DGS | **19.95** | **24.15** | **25.30** |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| Baseline (w/o densification) | 18.79 | 0.489 | 0.352 |
| + Depth conditioning | 20.49 | 0.619 | 0.290 |
| + Two-stage data augmentation | 21.59 | 0.758 | 0.236 |
| + Image prompt | 20.58 | 0.744 | 0.239 |
| **Full Model** | **22.41** | **0.812** | **0.202** |

### Key Findings
- **Depth conditioning contributes the most**: $+1.7\text{ dB}$ PSNR, ensuring the geometric consistency of the generated images.
- **Two-stage data augmentation is also crucial**: The coarse-to-fine NeRF pairs enable the diffusion model to learn to fix artifacts.
- **$10\times$ speedup over step-by-step methods**: The diffusion model is invoked only once during the densification phase rather than at every training step.
- **Super-resolution capability**: Fine texture details can be recovered even when the input consists of $4\times$ downsampled images.

## Highlights & Insights
- **Paradigm Shift**: Shifts the diffusion model from a "regularizer" to a "data augmenter," generating pseudo-training data in a single pass. This dramatically simplifies the pipeline and accelerates training by $10\times$. This methodology can be generalized to other few-shot 3D reconstruction tasks.
- **Simple and Effective Uncertainty Map Design**: No modification is needed for the internal representations of NeRF. Approximating reliability solely using warping differences enables the diffusion model to recognize where to remain conservative and where to be creative.
- **Compatible with both NeRF and 3DGS**: The approach is representation-agnostic, requiring only the capability to render RGB + depth, which demonstrates the versatility of the framework.

## Limitations & Future Work
- **Requires scene-specific diffusion model fine-tuning**: Currently trained on Hypersim + CO3D; its generalization to large outdoor scenes has not been fully verified.
- **Bottleneck in pseudo-observation quality**: The final reconstruction quality is bounded by the generation quality of the diffusion model. If the coarse rendering is too poor, leading to insufficient conditioning information, the diffusion model may generate inconsistent content.
- **No explicit multi-view consistency constraints**: There is no geometric consistency guarantee among the pseudo-observations (relying solely on filtering to discard bad samples), which may lead to contradictions between different pseudo-views.
- **Potential Improvements**: Incorporate consistency constraints from multi-view diffusion models (e.g., Zero-1-to-3++); use video diffusion to generate continuous trajectory pseudo-observations.

## Related Work & Insights
- **vs. DiffusioNeRF/ReconFusion**: They invoke the diffusion model at every training step for SDS/regularization, which is extremely slow. Deceptive directly generates training data in one go, which is $10\times$ faster.
- **vs. FreeNeRF**: FreeNeRF only performs frequency regularization without extra information injection. Deceptive injects scene priors using a diffusion model, which is fundamentally a more powerful prior source.
- **vs. ZeroNVS/Zero-1-to-3**: These methods directly generate novel views using diffusion models but do not guarantee geometric consistency. Deceptive utilizes rendered depth and uncertainty as conditions, offering better geometric consistency.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing the diffusion model as a data augmenter instead of a regularizer is a solid idea, though the broader framework of pseudo-observation augmentation has had prior prototypes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluations across multiple datasets and thorough ablation studies, though it lacks direct comparison with concurrent work ReconFusion.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly described, and the illustrations are intuitive.
- Value: ⭐⭐⭐⭐ High practical value—the method is simple, fast, and effective, making it directly integrable into existing NeRF/3DGS workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](../../CVPR2025/3d_vision/matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[ECCV 2024\] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)
- [\[ECCV 2024\] SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction](splatfields_neural_gaussian_splats_for_sparse_3d_and_4d_reconstruction.md)

</div>

<!-- RELATED:END -->
