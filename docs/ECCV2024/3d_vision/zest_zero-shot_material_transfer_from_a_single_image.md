---
title: >-
  [Paper Note] ZeST: Zero-Shot Material Transfer from a Single Image
description: >-
  [ECCV 2024][3D Vision] ZeST is proposed, a zero-shot, training-free material transfer method. By combining three parallel branches—extracting material representations via IP-Adapter, providing geometric guidance through ControlNet, and utilizing a foreground grayscale image for lighting cues—it achieves 2D material transfer from a single material exemplar image to a target object.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 50156020950146c6
---

# ZeST: Zero-Shot Material Transfer from a Single Image

**Conference**: ECCV 2024  
**arXiv**: [2404.06425](https://arxiv.org/abs/2404.06425)  
**Code**: [Project Page](https://ttchengab.github.io/zest)  
**Area**: 3D Vision

## TL;DR

ZeST is proposed, a zero-shot, training-free material transfer method. By combining three parallel branches—extracting material representations via IP-Adapter, providing geometric guidance through ControlNet, and utilizing a foreground grayscale image for lighting cues—it achieves 2D material transfer from a single material exemplar image to a target object.

## Background & Motivation

### Background

**Background**: Editing the materials of objects in images (e.g., transforming marble to steel) is of great value in applications such as game design and e-commerce.

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional methods require explicit 3D geometry, illumination estimation, and material parameter specification, which is highly complex.

### Key Challenge

**Key Challenge**: Text-driven methods struggle to precisely describe the fine texture details of materials.

### Core Idea

**Core Idea**: Existing approaches like TextureDreamer require 3-5 material images for DreamBooth fine-tuning, which is time-consuming and non-scalable.

### Core Problem

**Core Problem**: How to transfer material to an object in a target image from a single material exemplar image under training-free conditions.

## Method

### Overall Architecture

Three parallel branches are input to Stable Diffusion XL Inpainting:
1. **Material Encoding Branch**: IP-Adapter encodes the material exemplar to extract the material latent representation $z_M$.
2. **Geometric Guidance Branch**: DPT estimates the depth map $\rightarrow$ ControlNet provides structural constraints.
3. **Lighting Guidance Branch**: A foreground grayscale image $I_{init}$ is created $\rightarrow$ Inpainting model initialization.

$$I_{gen} = \mathcal{S}(z_M, D_I, I_{init}, F)$$

### Key Designs

**Material Encoding (IP-Adapter)**:
- Utilizes a CLIP image encoder to extract the feature representation of the material exemplar.
- Injects it into the diffusion model via cross-attention.
- Requires no DreamBooth fine-tuning; a single image is sufficient.

**Geometric Guidance (ControlNet)**:
- Depth-based ControlNet obtains structural information from the depth map of the input image.
- Overrides the geometric information in the material encoding $z_M$, ensuring the generated object retains its original shape.
- IP-Adapter + Img2Img fails to preserve the original geometry (Key Finding).

**Lighting Guidance via Foreground Grayscale (Core Design Choice)**:
- Directly using the original image: The original object's color acts as a strong prior that interferes with the material color (e.g., an orange pumpkin).
- Random noise initialization: Loses lighting and shading direction information.
- **Foreground grayscale image (Optimal)**: Removes color priors while preserving lighting and shading information.
- $I_{init} = F \odot I_{gray} + (1-F) \odot I$

**Implementation Details**:
- For depth estimation, DPT is used, and for foreground extraction, Rembg is used.
- Based on SDXL Inpainting + the corresponding versions of ControlNet and IP-Adapter.
- Takes about 15 seconds to generate one image on a single A10 GPU.

### Loss & Training

No training is involved; the entire process is completed during the inference stage of the pre-trained models.

## Key Experimental Results

### Main Results

Quantitative comparison on the synthetic dataset (9 materials $\times$ 10 meshes = 90 pairs):

| Method | PSNR↑ | LPIPS↓ | CLIP↑ |
|------|-------|--------|-------|
| IP-Adapter + InstructPix2Pix | 16.92 | 0.096 | 0.745 |
| Dreambooth + Geo/Illum Guidance | 25.46 | 0.053 | 0.893 |
| **ZeST** | **25.82** | **0.046** | **0.899** |

User study (real images, 1-5 scale):

| Method | Material Fidelity↑ | Realism↑ |
|------|-----------|---------|
| IP-Adapter + InstructPix2Pix | 1.48 | 3.23 |
| Dreambooth + Geo/Illum Guidance | 3.25 | 3.41 |
| **ZeST** | **4.05** | **3.78** |

### Ablation Study

Comparative validation of lighting guidance methods: The original image preserves the object's base color, interfering with the material color (Setting 1); random noise leads to incorrect lighting direction (Setting 2); the foreground grayscale image is optimal (Setting 3).

Robustness testing:
- Changing the illumination direction and rotation angle of the material exemplar $\rightarrow$ the generation results are highly consistent.
- Scaling the material exemplar image $\rightarrow$ the model automatically adjusts the texture scale to fit the target object.

### Key Findings

- ZeST significantly leads in user ratings for material fidelity (4.05 vs. 3.25), demonstrating that the zero-shot method outperforms the fine-tuning method.
- Foreground grayscale is the optimal choice for lighting guidance (the value of core design choices).
- ZeST is robust to changes in lighting, rotation, and scaling of the material exemplar.
- The DreamBooth encoding process loses material information and causes color shifts (especially in real-world scenes).
- Can be extended to multi-object editing (iterating with SAM) and illumination-aware material transfer.

## Highlights & Insights

- A purely engineering yet extremely elegant pipeline design—three branches each perform their own functions, completely training-free.
- The insight of foreground grayscaling is simple yet key: removing color $\rightarrow$ removing color priors, while preserving grayscale $\rightarrow$ preserving lighting and shading.
- As a pioneer of a new problem (2D-to-2D material transfer), it proposes both synthetic and real-world evaluation datasets.
- Can be combined with 3D texturing methods like Text2Tex to bring material-exemplar-driven texturing into 3D.

## Limitations & Future Work

- Sometimes transfers material only to the most "plausible" regions of the object (partial transfer issue).
- Multiple materials contained within the material exemplar may get blended.
- IP-Adapter lacks region-level material extraction capabilities.
- The latent space control of diffusion models is sometimes unpredictable.

## Rating

- Novelty: ⭐⭐⭐⭐ — New problem definition + clever training-free scheme
- Effectiveness: ⭐⭐⭐⭐ — Significant lead in user studies
- Practicality: ⭐⭐⭐⭐⭐ — Zero-shot, training-free, 15-second generation
- Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)

</div>

<!-- RELATED:END -->
