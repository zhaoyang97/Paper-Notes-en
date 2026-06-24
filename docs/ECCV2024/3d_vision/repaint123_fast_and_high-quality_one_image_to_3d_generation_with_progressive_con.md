---
title: >-
  [Paper Note] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting
description: >-
  [ECCV 2024][3D Vision][Single Image to 3D Generation] Repaint123 proposes a progressive controllable repainting strategy that uses 2D diffusion models to generate multi-view consistent, high-quality images, and then rapidly optimizes 3D representations via a simple MSE loss. It generates 3D content with delicate textures and multi-view consistency from a single image in just 2 minutes, significantly outperforming SDS-based methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Single Image to 3D Generation"
  - "Progressive Repainting"
  - "Multi-view Consistency"
  - "Score Distillation Sampling"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: cc740611a673bf9d
---

# Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting

**Conference**: ECCV 2024  
**arXiv**: [2312.13271](https://arxiv.org/abs/2312.13271)  
**Code**: [https://pku-yuangroup.github.io/repaint123/](https://pku-yuangroup.github.io/repaint123/)  
**Area**: 3D Vision / Diffusion Models  
**Keywords**: Single Image to 3D Generation, Progressive Repainting, Multi-view Consistency, Score Distillation Sampling, 3D Gaussian Splatting

## TL;DR

Repaint123 proposes a progressive controllable repainting strategy that uses 2D diffusion models to generate multi-view consistent, high-quality images, and then rapidly optimizes 3D representations via a simple MSE loss. It generates 3D content with delicate textures and multi-view consistency from a single image in just 2 minutes, significantly outperforming SDS-based methods.

## Background & Motivation

**Background**: Single-image to 3D generation is an important intersection task of computer vision and computer graphics, which is widely applied in robotics, VR/AR, etc. Current mainstream methods leverage the prior knowledge of 2D diffusion models to distill and optimize learnable 3D representations (such as NeRF) by rendering them into multi-view images via Score Distillation Sampling (SDS).

**Limitations of Prior Work**: SDS methods suffer from three severe limitations: (1) multi-view inconsistency (the multi-face problem), where textures generated from different viewpoints may conflict; (2) texture quality degradation, manifesting as over-saturation and over-smoothing; (3) slow generation speed, which usually takes from 30 minutes to several hours. These issues stem from the intrinsic conflict between SDS loss and 3D representation optimization.

**Key Challenge**: SDS generates gradients independently for different views during each sampling step, which fails to guarantee texture consistency between adjacent views. Meanwhile, SDS itself is a high-variance optimization objective, which easily leads to over-saturated/over-smoothed textures. There is a hard-to-reconcile trade-off between speed and quality.

**Goal**: (1) How to generate multi-view consistent, high-quality novel view images; (2) how to avoid the texture degradation caused by SDS; (3) how to elevate the generation speed to the minute level.

**Key Insight**: The authors observe that 2D diffusion models themselves possess powerful image generation capabilities, and the issue lies in how to constrain the output of different viewpoints to remain consistent. If high-quality, multi-view consistent 2D images can be generated first, one can rapidly optimize the 3D representation using a simple MSE loss, completely bypassing SDS.

**Core Idea**: A progressive repainting strategy is utilized to gradually generate consistent textures for adjacent views starting from the reference view, combining depth guidance and reference attention injection to guarantee consistency, followed by using MSE loss to directly optimize the 3D mesh texture.

## Method

### Overall Architecture

Repaint123 adopts a two-stage framework. Coarse stage: utilizes 3D Gaussian Splatting + SDS to obtain a coarse 3D model in about 1 minute. Refinement stage: converts the coarse model into a Mesh representation, and rotates the camera bi-directionally from the reference view with a progressive increment of 40° each time. For each novel view, a 2D diffusion model is used to repaint the invisible regions (occluded areas) while keeping the visible regions (overlapping areas) pixel-aligned. The generated multi-view consistent, high-quality images are used to optimize the mesh texture via a simple MSE loss. The entire refinement stage takes about 1 minute.

### Key Designs

1. **Progressive Repainting**:

    - **Function**: Generates multi-view consistent novel-view images.
    - **Mechanism**: Performs DDIM Inversion on the coarsely rendered novel-view images to obtain deterministic intermediate noise latent variables, preserving the consistent color information from the coarse 3D model. During the denoising process, the latent variables of the overlapping regions are replaced with the inverted latents to maintain pixel alignment: $x_{t-1} = x_{t-1}^{inv} \odot (1-M) + x_{t-1}^{rev} \odot M$, where $M$ is the occlusion mask. Meanwhile, ControlNet is applied to impose depth map guidance to guarantee geometric consistency. A bi-directional rotation strategy ensures that consistency is also maintained at the junctions of the front and back views.
    - **Design Motivation**: Unlike leveraging SDS to independently optimize each view, progressive repainting utilizes the overlapping regions of adjacent views as "anchors" to gradually propagate texture information, which naturally guarantees short-range view consistency.

2. **Mutual Self-Attention**:

    - **Function**: Mitigates accumulated texture bias and guarantees long-range view consistency (especially for back views).
    - **Mechanism**: In each denoising step, the Key/Value features of the novel views are replaced with the attention features of the reference view: $\text{Attention}(Q_t, K_r, V_r) = \text{Softmax}(Q_t K_r^T / \sqrt{d}) V_r$. This enables the novel-view images to directly query the high-quality texture details from the reference image, preventing texturing quality from gradually degrading as the repainting angle increases.
    - **Design Motivation**: Although progressive repainting ensures adjacent view consistency, texture bias can scale up and accumulate as the angle accumulates. Injecting the attention features of the reference view provides a global consistency constraint.

3. **Visibility-aware Adaptive Repainting**:

    - **Function**: Adaptively adjusts the repainting strength in the overlapping regions to balance fidelity and image quality.
    - **Mechanism**: Computes a visibility map $V$ from the normal map, which reflects the best observing angle ($\cos\theta^*$) of each pixel in historical views. Based on the orthogonal projection theorem, the projection resolution of a fragment is proportional to $\cos\theta$; thus, the repainting strength is set to $1 - \cos\theta^*$. A timestep-aware binarization is used to convert the soft visibility map into a hard mask: $M_t^{i,j} = 1$ if $V^{i,j} > 1 - t/T$, else $0$. This allows low-visibility regions (previously observed only from oblique angles) to receive stronger repainting, while high-visibility regions preserve their original textures.
    - **Design Motivation**: Prior methods employ a fixed repainting strength for all regions, failing to handle low-resolution texture issues caused by oblique viewpoints. The adaptive strategy achieves an optimal trade-off between fidelity and realism.

### Loss & Training

The coarse stage uses SDS loss to optimize the 3D Gaussian Splatting; the refinement stage uses a simple pixel-level MSE loss $\mathcal{L}_{MSE} = \|I_{fine} - I\|_2^2$ to directly optimize the mesh texture. In addition, IP-Adapter is used to encode the reference image into 16 image prompt tokens for Classifier-Free Guidance (CFG), boosting the generation quality.

## Key Experimental Results

### Main Results

| Method | Type | CLIP↑ | Contextual↓ | PSNR↑ | LPIPS↓ | Time |
|------|------|-------|-------------|-------|--------|------|
| RealFusion | NeRF | 0.71 | 2.20 | 19.24 | 0.194 | 20min |
| Make-It-3D | NeRF | 0.81 | 1.82 | 16.56 | 0.177 | 1h |
| Zero123-XL | NeRF | 0.83 | 1.59 | 19.56 | 0.108 | 30min |
| Magic123 | NeRF | 0.82 | 1.64 | 19.68 | 0.107 | 1h(+2h) |
| DreamGaussian | GS | 0.77 | 1.61 | 18.94 | 0.111 | 2min |
| **Repaint123** | **GS** | **0.85** | **1.55** | 19.00 | **0.101** | **2min** |

### Ablation Study

| Configuration | CLIP↑ | Contextual↓ | PSNR↑ | LPIPS↓ |
|------|-------|-------------|-------|--------|
| Coarse only | 0.71 | 1.78 | 21.17 | 0.133 |
| + Repaint | 0.71 | 1.62 | 22.41 | 0.049 |
| + Mutual Attention | 0.78 | 1.56 | 22.42 | 0.048 |
| + Image Prompt | 0.84 | 1.52 | 22.40 | 0.048 |
| + Adaptive (Full) | **0.88** | **1.50** | 22.38 | **0.048** |

### Key Findings

- Progressive repainting itself is the most contributing component, reducing Contextual Distance from 1.78 to 1.62 and significantly decreasing LPIPS from 0.133 to 0.049.
- Mutual Attention and Image Prompt primarily improve CLIP similarity (multi-view semantic consistency), bringing gains of +0.07 and +0.06, respectively.
- An angular interval of 40° is the optimal choice; although 60° yields slightly higher metrics, it is prone to the multi-face problem.
- The NeRF version of Repaint123 also significantly outperforms Magic123, validating the universality of the method.

## Highlights & Insights

- **Completely bypassing the SDS pipeline**: The key insight of this paper is transforming the 3D generation problem into a two-step strategy: "first generate consistent multi-view 2D images, then reconstruct with MSE". This is much more controllable, stable, and faster than directly optimizing 3D representations using SDS.
- **Clever design of timestep-aware binarization**: Associating the continuous visibility map with the denoising timesteps allows repainting only high-requirement areas in the early denoising stage (large noise) and expanding the repainting scope in the later stage (small noise), achieving progressive refinement. This trick can be transferred to any scenario involving inpainting + denoising.
- **Speed advantage**: The 2-minute generation speed is 10-30 times faster than NeRF-based methods, which mainly benefits from using MSE loss to replace SDS and the highly efficient representation of 3DGS.

## Limitations & Future Work

- The technical maturity of current 3D Gaussian Splatting is limited, and the extracted mesh may exhibit geometric artifacts such as holes.
- The PSNR of reference view reconstruction is lower than that of NeRF-based methods (19.00 vs 24.69), indicating that the conversion from GS to Mesh loses accuracy.
- The 40° interval repainting strategy may require denser sampling for highly asymmetric objects.
- Multi-reference image input scenarios are not explored, which may limit the realism of the back texture.
- Combining with state-of-the-art multi-view diffusion models (e.g., Zero123++) to generate initial multi-view images before refining with the repainting strategy could be considered.

## Related Work & Insights

- **vs DreamGaussian**: Both utilize 3DGS + 2-minute generation, but DreamGaussian still uses SDS during the refinement stage, resulting in over-smoothed textures. Repaint123 replaces SDS with repainting + MSE, improving CLIP from 0.77 to 0.85.
- **vs Magic123**: Magic123 uses 2D SDS of Zero123 + DMTet. It takes more than 1 hour but achieves higher PSNR (19.68 vs 19.00), demonstrating that NeRF-level geometric quality still holds advantages.
- **vs HiFi-123**: Although both utilize inversion + attention injection, Repaint123 chooses ControlNet + depth guidance instead of a depth-based diffusion model, which is more flexible and generalizable.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of replacing SDS with progressive repainting is clean and effective, and the visibility-aware repainting strength design is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ The comparison on multiple datasets, detailed ablation studies, NeRF version validation, and angular analysis are fairly thorough.
- Writing Quality: ⭐⭐⭐⭐ The illustrations are clear, the methodology description is well-organized, and the pipeline is easy to understand.
- Value: ⭐⭐⭐⭐ Provides a practical solution for fast, high-quality single-image 3D generation, offering valuable insights for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ICLR 2026\] FlashWorld: High-quality 3D Scene Generation within Seconds](../../ICLR2026/3d_vision/flashworld_high-quality_3d_scene_generation_within_seconds.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)

</div>

<!-- RELATED:END -->
