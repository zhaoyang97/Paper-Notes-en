---
title: >-
  [Paper Note] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion
description: >-
  [ICCV 2025][3D Vision][3D human reconstruction] AdaHuman is proposed as a framework that generates high-fidelity, animatable 3D human avatars from a single image via a pose-conditioned 3D joint diffusion model and a comp…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D human reconstruction"
  - "3D Gaussian splatting"
  - "multiview diffusion model"
  - "pose-conditioned generation"
  - "animatable avatar"
date: 2026-05-08
content_hash: b4f47d215c6d23b1
---

# AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion

**Conference**: ICCV 2025
**arXiv**: [2505.24877](https://arxiv.org/abs/2505.24877)  
**Code**: [https://nvlabs.github.io/AdaHuman](https://nvlabs.github.io/AdaHuman) (code and models to be released)  
**Area**: 3D Vision
**Keywords**: 3D human reconstruction, 3D Gaussian splatting, multiview diffusion model, pose-conditioned generation, animatable avatar

## TL;DR

AdaHuman is proposed as a framework that generates high-fidelity, animatable 3D human avatars from a single image via a pose-conditioned 3D joint diffusion model and a compositional 3DGS refinement module.

## Background & Motivation

Generating high-quality animatable 3D human avatars from a single image has significant applications in gaming, animation, and VR. Existing methods face two primary challenges: (1) **self-occlusion**—avatars are typically generated in the same pose as the input image, making it difficult to complete occluded regions under complex poses, which impairs rigging and animation; and (2) **detail blurriness**—feed-forward 3D reconstruction models are constrained by fixed output resolutions (e.g., 256×256), preventing the capture of fine-grained details such as facial features and clothing textures.

SDS-based methods are flexible but suffer from over-saturation artifacts and slow generation. Multiview generation-and-reconstruction pipelines improve speed and realism but still fail to address these two core issues. A new framework capable of handling pose transformation while enhancing local detail is therefore required.

## Method

### Overall Architecture

AdaHuman consists of two core modules: (1) **pose-conditioned 3D joint diffusion**—simultaneously performing multiview image synthesis and 3DGS reconstruction during the diffusion process, supporting avatar generation under arbitrary pose conditions; and (2) **compositional 3DGS refinement**—seamlessly integrating refined local 3DGS representations into a complete avatar through image-to-image refinement of local body parts and crop-aware camera ray maps.

### Key Designs

1. **Pose-Conditioned 3D Joint Diffusion Model**:

    - Given a full-body image, local views of different body parts (head, upper body, lower body) are generated and combined with the input to form the input view set $\mathcal{I}_{i=1}^V$
    - Each view is represented by a triplet $\{x_i, p_i, c_i\}$—RGB image, 2D semantic pose map (rendered from SMPL), and camera ray map
    - 2D self-attention in a single-image LDM is replaced with 3D attention to enable multiview-consistent generation
    - At each denoising step $t$, a 3DGS generator $\mathbf{G}$ produces $\mathcal{G}^t$ from the predicted "clean" images $x_j^{t\to 0}$, which are then re-rendered as 3D-consistent images for the subsequent denoising step
    - **Key advantage**: Pose conditioning enables the model to generate avatars in A-pose without requiring standard-pose training data, minimizing self-occlusion and naturally supporting rigging and animation
    - Reconstruction mode selects 3 target views at 90° intervals from the same frame; re-posing mode selects 4 target views from different frames

2. **Compositional 3DGS Refinement Module**:

    - **Local body part refinement**: Four canonical views at 90° intervals are rendered from the coarse 3DGS $\mathcal{G}_\text{coarse}$, with magnified local views rendered for three body regions (head, upper body, lower body)
    - Local renderings are refined via image-to-image diffusion (analogous to SDEdit), substantially enhancing detail
    - **Crop-aware camera ray map**: Pixel coordinates $(u,v)$ in a local view are mapped back to global view coordinates $(i,j)$ through cropping box parameters, computing the corresponding camera ray embedding $\mathcal{R}(i,j) = (\mathbf{o}(i,j), \mathbf{o}(i,j) \times \mathbf{d}(i,j))$
    - **Design motivation**: Establishes precise 3D coordinate correspondences between local and global views, enabling the 3DGS generator to process inputs at different scales uniformly in global space without architectural modifications

3. **Visibility-Aware 3DGS Composition**:

    - Two criteria determine which 3D Gaussian primitives to retain:
        - **View coverage**: The number of input views covering each Gaussian primitive is counted; primitives with low coverage lack multiview consensus and may be unreliable
        - **Visibility saliency**: The gradient magnitude of the alpha channel across all rendered views is measured; primitives with low saliency contribute little to appearance and may constitute noise
    - When a Gaussian primitive from a coarser body region is well covered by views from a finer region (e.g., head), the redundant coarse primitive is discarded
    - **Design motivation**: Directly merging all local 3DGS representations produces floating artifacts; this strategy ensures only the most reliable and visually salient Gaussian primitives are retained through principled filtering

### Loss & Training

- Jointly trained on MVHumanNet (6,209 subjects) and CustomHuman (589 meshes)
- Both the LDM and 3DGS generator are initialized from pretrained weights
- Reconstruction is trained for 30K steps, followed by 10K steps of re-posing fine-tuning
- The LDM is supervised with MSE loss on image latents; the 3DGS is supervised with MSE + LPIPS rendering loss + surface regularization
- An additional 12 views are sampled beyond the target views to provide dense supervision for the 3DGS

## Key Experimental Results

### Main Results (Avatar Reconstruction)

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | CD(cm)↓ |
|-------|-------|-------|--------|------|---------|
| LGM | 18.99 | 0.8445 | 0.1664 | 122.3 | 2.175 |
| SiTH | 20.77 | 0.8727 | 0.1277 | 42.9 | 1.389 |
| SIFU | 20.59 | 0.8853 | 0.1359 | 92.6 | 2.009 |
| Human3Diffusion | 21.08 | 0.8728 | 0.1364 | 35.3 | 1.230 |
| **AdaHuman** | **21.46** | **0.8925** | **0.1087** | **27.3** | **0.962** |

In a user study, AdaHuman achieves preference rates of 88.3%, 99.2%, 79.7%, and 93.8% against SiTH, SIFU, Human3Diffusion, and the coarse 3DGS, respectively.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Note |
|---------------|-------|-------|--------|------|------|
| Coarse 3DGS (no refinement) | 20.84 | 0.8789 | 0.1296 | 31.9 | Lacks fine details such as facial features |
| Direct composition (no filtering) | 20.41 | 0.8700 | 0.1350 | 36.2 | Produces substantial floating artifacts |
| Learnable composition (network prediction) | 20.87 | 0.8788 | 0.1270 | 28.0 | Marginal improvement but artifacts remain |
| Without joint diffusion | 20.79 | 0.8762 | 0.1283 | 27.6 | View inconsistency |
| **Full method** | **21.46** | **0.8925** | **0.1087** | **27.3** | Best |
| + GT pose condition | 23.00 | 0.9028 | 0.1086 | 27.0 | Improved pose alignment |

### Key Findings

- On the re-posing task, PSNR reaches 24.64 (vs. 21.21 for SiTH) and LPIPS drops to 0.0863, representing a substantial margin
- Strong generalization to complex and loose clothing in in-the-wild images
- Despite not being trained on standard-pose data, the model successfully generalizes to A-pose by leveraging the diverse pose distribution in MVHumanNet
- Inference requires approximately 70 seconds on an A100 GPU

## Highlights & Insights

- **Pose-decoupled design**: Unifies avatar reconstruction and pose transformation within a single diffusion framework, enabling animation-ready avatar generation without standard-pose training data
- **Local-global compositional strategy**: Crop-aware ray maps elegantly introduce multi-scale inputs without modifying the 3DGS generator architecture
- **Two animation modes compared**: Direct re-posing (more realistic clothing deformation but slower) vs. LBS-based animation (real-time but limited clothing deformation), offering options for different application scenarios

## Limitations & Future Work

- The local refinement strategy may produce artifacts in occluded or low-coverage regions such as hands and arms
- Animation remains dependent on the SMPL model and skinning weights, limiting accuracy for facial expressions, hand gestures, and clothing deformation
- Future work may explore video diffusion models to improve animation quality and temporal consistency
- Simulation-based approaches have the potential to improve physically correct deformation of loose garments

## Related Work & Insights

- Builds upon the joint diffusion-and-reconstruction paradigm of Human3Diffusion; the core innovations lie in pose conditioning and compositional refinement
- Compared with contemporaneous methods such as IDOL and LHM, the diffusion-based approach leverages stronger generative priors
- The design of crop-aware ray maps is generalizable to other scenarios requiring multi-scale 3DGS reconstruction

## Rating
- Novelty: ⭐⭐⭐⭐ — The pose-conditioned joint diffusion and visibility-aware composition scheme are elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive quantitative and qualitative evaluation, user study, and thorough ablation
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear with intuitive illustrations
- Value: ⭐⭐⭐⭐ — Addresses a practical need for single-image animatable avatar generation with significantly superior results

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)

</div>

<!-- RELATED:END -->
