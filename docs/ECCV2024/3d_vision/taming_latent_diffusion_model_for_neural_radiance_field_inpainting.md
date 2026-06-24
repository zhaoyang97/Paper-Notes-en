---
title: >-
  [Paper Note] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting
description: >-
  [ECCV 2024][3D Vision][NeRF Inpainting] MALD-NeRF is proposed to achieve high-quality NeRF inpainting through masked adversarial training and a scene-customized latent diffusion model, effectively addressing the multi-view inconsistency and texture shift problems of diffusion models.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "NeRF Inpainting"
  - "Latent Diffusion Model"
  - "Adversarial Training"
  - "3D Consistency"
  - "Object Removal"
date: 2026-05-08
content_hash: ebc7ce81d91bf904
---

# MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting

**Conference**: ECCV 2024  
**arXiv**: [2404.09995](https://arxiv.org/abs/2404.09995)  
**Code**: [Project Page](https://hubert0527.github.io/MALD-NeRF)  
**Area**: 3D Vision  
**Keywords**: NeRF Inpainting, Latent Diffusion Model, Adversarial Training, 3D Consistency, Object Removal

## TL;DR

MALD-NeRF is proposed to achieve high-quality NeRF inpainting through masked adversarial training and a scene-customized latent diffusion model, effectively addressing the multi-view inconsistency and texture shift problems of diffusion models.

## Background & Motivation

Using 2D latent diffusion models for NeRF inpainting faces two core challenges: (1) The inpainting results of diffusion models are inconsistent across views, and using pixel-level losses ($L_1$/$L_2$) leads to a foggy, blurry inpainted region; (2) The autoencoder error of latent diffusion models causes a texture shift between the inpainted pixels and the original pixels, producing a conspicuous inpainting boundary seam. Although perceptual loss (LPIPS) can mitigate this to some extent, it does not fundamentally solve the problem.

## Method

### Overall Architecture

MALD-NeRF contains three core components: (1) Masked Adversarial Training—replacing pixel-level losses to supervise the inpainted region; (2) Scene-Customized Diffusion—reducing the generation diversity of the diffusion model through LoRA fine-tuning; (3) Iterative Dataset Update—gradually reducing diffusion noise to propagate 3D-consistent information.

### Key Designs

**Masked Adversarial Training**: adversarial training is conducted with patches of the inpainted images as "real" samples and patches rendered by NeRF as "generated" samples. The key innovation lies in the **mask design**: for both real and generated images, only the pixels within the inpainting mask are retained, while the area outside the mask is filled with black. This hides the inpainted/non-inpainted boundary from the discriminator, eliminating the impact of the texture shift. Meanwhile, a discriminator feature matching loss is introduced to provide fine-grained supervision.

**Scene-Customized Diffusion**: LoRA fine-tuning is conducted for each scene to learn scene-specific text tokens. It is trained with a self-supervised inpainting loss (random rectangular masks), setting the loss in the object removal mask area to zero. After fine-tuning, the cross-view consistency of the generation results of the diffusion model is substantially improved.

**Iterative Dataset Update and Noise Scheduling**: The inpainted images are updated once every $U$ iterations, using partial DDIM starting from the current NeRF rendering. The noise timestep decreases with training progress: $t = t_{max} - (t_{max} - t_{min}) \cdot \sqrt{k/K}$, achieving coarse-to-fine propagation of 3D-consistent information.

### Loss & Training

- **Reconstructed Region**: $L^r = \lambda_{pix}L_{pix} + \lambda_{inter}L_{inter} + \lambda_{distort}L_{distort} + \lambda_{decay}L_{decay}$
- **Inpainted Region**: $L^m = -\lambda_{adv}L_{adv} + \lambda_{fm}L_{fm}$ + regularization terms
- **Discriminator**: $L^D = L_{adv} + \lambda_{GP}L_{GP}$ (R1 regularization)
- The inpainted region does not use pixel/perceptual losses at all.

## Key Experimental Results

### Quantitative Comparison on SPIn-NeRF Dataset

| Method | LPIPS↓ | M-LPIPS↓ | FID↓ | KID↓ |
|------|--------|----------|------|------|
| SPIn-NeRF | 0.5356 | 0.4019 | 219.80 | 0.0616 |
| SPIn-NeRF (LDM) | 0.5568 | 0.4284 | 227.87 | 0.0558 |
| Inpaint3D | 0.5437 | 0.4374 | 271.66 | 0.0964 |
| InpaintNeRF360 | 0.4694 | 0.3672 | 222.12 | 0.0544 |
| **MALD-NeRF** | **0.4345** | **0.3344** | **183.25** | **0.0397** |

### Ablation Study

| Settings | LPIPS↓ | FID↓ | KID↓ |
|------|--------|------|------|
| w/o Adversarial + L1 Reconstruction | 0.6623 | 305.60 | 0.1177 |
| w/o Adversarial + LPIPS | 0.4231 | 192.86 | 0.0447 |
| Ours + L1 Reconstruction | 0.5106 | 256.82 | 0.0827 |
| Ours + LPIPS | **0.4130** | **185.79** | 0.0419 |
| Ours - Scene Customization | 0.4894 | 224.29 | 0.0596 |
| **MALD-NeRF (Full)** | 0.4345 | 183.25 | **0.0397** |

### Key Findings

- FID drops significantly from 219.80 to 183.25, demonstrating that adversarial training substantially improves generation quality.
- Pixel-level L1 loss is actually detrimental to the inpainting task (FID 305.60), and LPIPS is also sub-optimal; adversarial training is the correct choice.
- Scene customization reduces the generation of out-of-context objects, dramatically improving 3D consistency.
- Masked adversarial training is the key design to eliminate the texture shift.

## Highlights & Insights

1. **Deep Problem Analysis**: clearly points out the limitations of pixel-level and perceptual losses in NeRF inpainting tasks.
2. Clever masked adversarial training concept—hiding information so that the discriminator cannot exploit the texture shift at the inpainting boundaries.
3. The combination of scene customization, iterative updates, and noise scheduling forms a complete scheme for 3D consistency enhancement.

## Limitations & Future Work

- The use of an internally pre-trained diffusion model might require replacement with open-source models for public reproduction.
- Adversarial training can be unstable.
- Geometric reconstruction of large occluded regions remains challenging.

## Related Work & Insights

The proposed masked adversarial training design is conceptually similar to AmbientGAN, but with different application goals. The "taming" strategies for diffusion models in 3D tasks hold significant guiding significance for future work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Usability: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)
- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)

</div>

<!-- RELATED:END -->
