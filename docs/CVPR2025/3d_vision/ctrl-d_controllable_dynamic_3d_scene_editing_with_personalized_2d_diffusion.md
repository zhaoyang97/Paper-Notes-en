---
title: >-
  [Paper Note] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion
description: >-
  [CVPR 2025][3D Vision][Dynamic 3D Editing] Fine-tunes the InstructPix2Pix model using a single edited reference image to "learn" the editing capability, which, combined with a two-stage deformable 3D Gaussian optimization, achieves controllable and consistent dynamic 3D scene editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dynamic 3D Editing"
  - "Deformable Gaussian"
  - "InstructPix2Pix"
  - "Personalized Diffusion"
  - "Scene Editing"
date: 2026-05-08
content_hash: d69badb043ff78a1
---

# Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2412.01792](https://arxiv.org/abs/2412.01792)  
**Code**: [Project Page](https://IHe-KaiI.github.io/CTRL-D/)  
**Area**: 3D Vision / Dynamic Scene Editing  
**Keywords**: Dynamic 3D Editing, Deformable Gaussian, InstructPix2Pix, Personalized Diffusion, Scene Editing

## TL;DR

Fine-tunes the InstructPix2Pix model using a single edited reference image to "learn" the editing capability, which, combined with a two-stage deformable 3D Gaussian optimization, achieves controllable and consistent dynamic 3D scene editing.

## Background & Motivation

- Dynamic 3D scene editing is a key requirement for VR/AR, data augmentation, and content creation, but existing methods suffer from inconsistent editing and poor controllability.
- Prior works such as Instruct 4D-to-4D rely on pre-trained diffusion models (e.g., vanilla IP2P) and are limited by the capabilities of their editing backbones, preventing precise local editing.
- Tracking edited regions is far more difficult in dynamic scenes than in static ones, and traditional methods relying on noise discrepancies to determine editing regions are unstable.
- Key Insight: Simplifying the complex dynamic scene editing task into a simple 2D image editing problem—users only need to edit a single image, and the editing effects can then be propagated to the entire dynamic scene.

## Method

### Overall Architecture

A three-stage pipeline: (1) edit a single reference image using any 2D editing tool; (2) fine-tune the IP2P model using the pre- and post-edited image pairs to obtain a personalized editor; (3) conduct two-stage optimization of the deformable 3D Gaussian scene.

### Key Designs

1. **Personalized InstructPix2Pix Fine-tuning**:
    - **Function**: Enables IP2P to "learn" specific editing capabilities from a single edited reference image.
    - **Mechanism**: Uses GPT-4V to generate text instructions $C_T^{\star}$ and introduces a special token `<V>` to enhance specificity; incorporates Prior Preservation Loss (inspired by DreamBooth) to maintain the model's generalization capability.
    - **Design Motivation**: The editing capability of the vanilla IP2P is limited by the training data distribution. Personalized fine-tuning allows the model to directly learn the editing region and style from the reference image without explicit tracking of editing regions.
    - **Fine-tuning Loss**: $\mathcal{L}_{\text{finetune}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, I, C_T)\|_2^2] + \lambda \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t^{\star}, t, I_d, C_T^{\star})\|_2^2]$
    - **Data Augmentation**: Augments the source and edited images via affine transformations (rotation, translation, shearing) to prevent overfitting during single-image fine-tuning.

2. **Two-stage Dynamic Gaussian Optimization**:
    - **Function**: Progressively edits the pre-trained dynamic 3D Gaussian scene.
    - **Mechanism**: Stage 1 only optimizes the canonical space and performs Gaussian densification (freezing the deformation field); Stage 2 simultaneously optimizes the deformation field and 3D Gaussians, utilizing an edited image buffer to accelerate convergence.
    - **Design Motivation**: Multi-stage optimization allows first establishing a rough geometry of the edited region, followed by global optimization to achieve temporal consistency.

3. **Edited Image Buffer**:
    - **Function**: Accelerates the editing process and enhances temporal consistency.
    - **Mechanism**: At each iteration, unedited frames are randomly selected, edited via the personalized IP2P, and added to the buffer; only the images in the buffer are used to train the 3D Gaussians and the deformation field (warm-up phase).
    - **Design Motivation**: To avoid editing from the original frames every time, leveraging existing editing results to accelerate convergence.

### Loss & Training

- Total scene optimization loss: $\mathcal{L} = (1-\lambda_d)\mathcal{L}_1 + \lambda_d \mathcal{L}_{\text{D-SSIM}} + \lambda_t \mathcal{L}_{\text{temp}}$
- Parameter settings: $\lambda_d = 0.2$, $\lambda_t = 0.001$
- Monocular scenes are modeled using [Yang et al.], with Stage 1 set as the first 300 iterations; multi-camera scenes utilize [Wu et al.], with Stage 1 set as the first 100 iterations.
- An image is edited every 50 iterations.

## Key Experimental Results

### Main Results

| Scene | Method | CLIP Score↑ | Consistency↑ | Time↓ |
|------|------|-------------|-------------|-------|
| Portrait | Ctrl-D | **27.75** | **0.953** | **60 min** |
| Portrait | IN4D | 27.38 | 0.933 | 2 hours |
| Cat | Ctrl-D | **31.81** | **0.968** | **60 min** |
| Cat | IN4D | 31.72 | 0.964 | 2 hours |
| Steak | Ctrl-D | **28.52** | **0.988** | **40 min** |
| Steak | IN4D | 28.23 | 0.983 | 2 hours |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| w/o Data Augmentation | Blurry, temporally inconsistent | Overfitting during IP2P fine-tuning causes unstable editing |
| w/ Data Augmentation | High quality, good consistency | Affine transformations effectively prevent overfitting |
| w/o Edited Buffer | Still close to vanilla scene after 1000 steps | Inefficient training due to random frame selection |
| w/ Edited Buffer | Successfully edited after 1000 steps | Focuses on edited frames to accelerate convergence |

### Key Findings

- The total time (fine-tuning + optimization) is less than half of IN4D.
- Editing capabilities can generalize across domains: IP2P fine-tuned on cat images can be applied to portraits and full-body scenes.
- Supports multiple 2D editing modalities, including text-driven, image-driven, and style transfer.

## Highlights & Insights

- Simplifying complex 4D editing into a 2D editing task drastically lowers the barrier to dynamic scene editing.
- Personalized IP2P can directly learn the editing region from the reference image, bypassing the difficult target tracking in dynamic scenes.
- Cross-domain generalization of editing capabilities proves that personalized fine-tuning learns general "editing skills" rather than overfitting to a single scene.

## Limitations & Future Work

- When the reconstruction quality of the dynamic 3D Gaussians is poor (e.g., motion-blurred hands), the edited results will also be blurry.
- Adding complex content to empty regions (e.g., adding a bag to a dog) still suffers from multi-view consistency issues.
- Future work could employ stronger reconstruction backbones and more powerful foundation diffusion models.

## Related Work & Insights

- InstructPix2Pix → Fundamental editing capabilities; DreamBooth → Prior Preservation concept
- Deformable 3DGS → Dynamic scene representation
- Instruct-NeRF2NeRF → Inspiration source for the Iterative Dataset Update strategy

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm of simplifying dynamic editing into 2D editing is novel, and the personalized fine-tuning strategy is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive qualitative and quantitative comparisons, thorough ablation studies, and convincing cross-domain generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed pipeline descriptions.
- Value: ⭐⭐⭐⭐ Highly practical, lowering the threshold for dynamic scene editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)
- [\[CVPR 2025\] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture](towards_high-fidelity_3d_talking_avatar_with_personalized_dynamic_texture.md)
- [\[ICLR 2026\] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](../../ICLR2026/3d_vision/color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
