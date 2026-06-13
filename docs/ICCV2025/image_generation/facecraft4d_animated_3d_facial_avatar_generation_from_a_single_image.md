---
title: >-
  [Paper Note] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image
description: >-
  [ICCV 2025][Image Generation][4D head generation] This paper proposes FaceCraft4D, a framework that generates animatable 360° 4D facial avatars from a single image by combining three complementary priors: a 3D shape prio…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "4D head generation"
  - "single-image-driven"
  - "3D Gaussian"
  - "FLAME"
  - "multi-view consistency"
date: 2026-05-08
content_hash: 5baca08a7a8181e1
---

# FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image

**Conference**: ICCV 2025
**arXiv**: [2504.15179](https://arxiv.org/abs/2504.15179)  
**Code**: N/A  
**Area**: Diffusion Models / 3D Vision / Face Generation
**Keywords**: 4D head generation, single-image-driven, 3D Gaussian, FLAME, multi-view consistency

## TL;DR

This paper proposes FaceCraft4D, a framework that generates animatable 360° 4D facial avatars from a single image by combining three complementary priors: a 3D shape prior (PanoHead GAN inversion), a 2D image prior (diffusion model texture enhancement), and a video prior (LivePortrait expression animation). A COIN training strategy is introduced to address multi-view data inconsistency, enabling high-quality real-time rendering at 156 FPS.

## Background & Motivation

**Background**: 4D head generation (drivable 3D head models) has broad applications in gaming, filmmaking, and education. High-quality methods such as HQ3DAvatar and GaussianAvatar typically require multi-view video input and accurate camera pose estimation.

**Limitations of Prior Work**:
   - Multi-view video capture is impractical for real-world deployment.
   - Single-image methods either fail to produce 360° coverage (e.g., Portrait-4D) or cannot model expression animation (e.g., PanoHead).
   - 2D-based methods (e.g., AniPortrait) inherently lack multi-view consistency.
   - Hybrid 2D+3D methods cannot handle extreme camera angles.

**Key Challenge**: A single image is extremely information-limited — it lacks depth, multi-view, and dynamic expression information, while collecting multi-view full-head animation datasets for end-to-end training is prohibitively difficult.

**Goal**: Starting from a single image, simultaneously achieve: (a) 360° full-view coverage; (b) controllable expression animation; (c) multi-view-consistent high-quality texture; (d) a purely 3D representation for real-time rendering.

**Key Insight**: Divide and conquer — leverage three complementary priors (a 3D GAN for shape, an image diffusion model for texture, and a video model for expression) to synthesize personalized multi-view data, then train an explicit 3D representation.

**Core Idea**: Combine 3D shape, 2D image, and video priors to synthesize multi-view expression data, and robustly reconstruct a 4D Gaussian avatar from inconsistent data via the COIN training strategy.

## Method

### Overall Architecture

FaceCraft4D consists of two major stages: **personalized multi-view generation** and **4D representation optimization**. The first stage sequentially applies three priors to generate personalized, high-quality multi-view expression images; the second stage uses the synthesized data to train an animatable 3D Gaussian representation.

### Key Designs

1. **Shape Prior — PanoHead GAN Inversion**

    - Function: Obtain a coarse 3D shape and approximate texture from the single input image.
    - Mechanism: A pretrained PanoHead (a 360°-capable 3D-GAN) is used for GAN inversion. The latent vector $z$ is first optimized by minimizing a combined $\mathcal{L}_2$ + LPIPS loss, after which $z$ is fixed and the GAN parameters are fine-tuned. The optimized generator renders coarse multi-view images $\{I_i\}$ and depth maps $\{D_i\}$.
    - Design Motivation: PanoHead, trained on large-scale head data, encodes a strong prior over complete head geometry, providing a full 3D shape initialization — including the back of the head — for subsequent stages.

2. **Image Prior — Diffusion Model Texture Enhancement**

    - Function: Enhance the coarse multi-view textures produced by GAN inversion using a 2D diffusion model (Cosmicman), while maintaining cross-view consistency.
    - Mechanism: Two key constraints are proposed:
        - **Cross-view Mutual Attention**: Inspired by MasaCtrl, during diffusion model denoising, the K/V of self-attention in novel views are replaced with those from the reference image. Multi-view images are processed as a batch, with the reference image serving as a unified information source.
        - **Warping-based Control**: Depth maps are used to project textures from anchor views (the reference image and the back view) onto neighboring views. A visibility mask filters occluded regions, and the result is blended with the intermediate diffusion latent.
    - Design Motivation: Direct image-to-image translation breaks cross-view consistency and alters semantic content; triplane-based methods are sensitive to focal length and degrade significantly with varying focal lengths. Geometry-guided warping combined with attention sharing ensures that enhanced textures remain cross-view consistent.

3. **Video Prior — LivePortrait Expression Animation**

    - Function: Use LivePortrait to generate synchronized expression animation data from multi-view static images.
    - Mechanism: The enhanced multi-view images $\{I_i^*\}$ and the reference image are fed as source images into LivePortrait, driven by the same driving video (from the NerSemble dataset) to ensure expression synchronization. Because the input views share a consistent identity, the output videos also maintain identity consistency.
    - Design Motivation: The static generation stage cannot provide expression-dependent texture information (e.g., interior mouth details); the video prior supplements this dynamic information.

4. **COIN Training Strategy (COnsistent-INconsistent Training)**

    - Function: Robustly reconstruct a high-quality 4D representation from multi-view data with minor inconsistencies.
    - Mechanism: Two representations are jointly trained — a **consistent GaussianAvatar** (FLAME-based 3D Gaussians supervised with an LPIPS loss to capture structural fidelity) and an **inconsistent MLP** (learning per-view color offsets $c_{offset} = \text{MLP}(e_{view}, c, e_g; \theta)$, supervised with an L1+SSIM loss to capture high-frequency detail). During inference, the view embedding of the reference view is fixed.
    - Design Motivation: Synthesized multi-view data inevitably contains minor color and feature misalignments. Training directly on inconsistent data produces blurry textures. COIN isolates inconsistencies into a separate MLP, preventing contamination of the base representation — analogous to robust regression.

### Loss & Training

- Pixel-level loss: $\mathcal{L}_{pixel} = \lambda_1 \mathcal{L}_1(I_i^{IC}, I_i^*) + \lambda_{SSIM} \text{SSIM}(I_i^{IC}, I_i^*)$
- Structural supervision loss: $\mathcal{L}_{struc} = \lambda_{LPIPS} \text{LPIPS}(I_i^C, I_i^*)$
- Regularization loss: $\mathcal{L}_{reg} = \lambda_{offset} \mathcal{L}_1(c_{offset}, 0)$
- Hyperparameters: $\lambda_1 = 0.8$, $\lambda_{SSIM} = 0.2$, $\lambda_{LPIPS} = 0.05$, $\lambda_{offset} = 1$
- Static optimization runs for 30K iterations, followed by COIN fine-tuning for 90K iterations; total generation time is approximately 2.5 hours; inference runs at 156 FPS @ 512×512.

## Key Experimental Results

### Main Results: Static 3D Head Generation (Quantitative Comparison)

| Method | CLIP-I ↑ | ID ↑ | FID ↓ |
|--------|---------|------|-------|
| GaussianCube | 0.6830 | 0.4300 | 258.81 |
| PanoHead | 0.8233 | 0.4246 | 195.28 |
| SV3D | 0.7656 | 0.4331 | 234.86 |
| Portrait3D | 0.7066 | 0.3719 | 302.74 |
| **FaceCraft4D** | **0.8053** | **0.5082** | **174.36** |

3D Head Animation Comparison:

| Method | CLIP-I ↑ | ID ↑ | FID ↓ |
|--------|---------|------|-------|
| AniPortrait | 0.4653 | 0.4171 | 364.99 |
| Portrait-4D | 0.5236 | 0.4592 | 248.36 |
| **FaceCraft4D** | **0.5737** | **0.4602** | **201.76** |

### Ablation Study

Multi-view image generation module ablation:

| Configuration | CLIP-I ↑ | ID ↑ | FID ↓ |
|---------------|---------|------|-------|
| w/o Warp w/o MA | 0.7328 | 0.4787 | 182.27 |
| w/o Warp | 0.7886 | 0.4915 | 171.08 |
| w/o MA (mutual attention) | 0.8151 | 0.4951 | 172.43 |
| Full | **0.8162** | **0.4984** | **166.96** |

Animation module ablation:

| Configuration | CLIP-I ↑ | ID ↑ | FID ↓ |
|---------------|---------|------|-------|
| w/o COIN | 0.7688 | 0.4952 | 144.95 |
| Full | **0.7729** | **0.5010** | **142.80** |

### Key Findings
- **Significant advantage in identity preservation**: FaceCraft4D's ID score (0.5082) substantially outperforms all baselines, demonstrating strong identity consistency.
- **Warping module preserves fine details**: Fine-grained details such as tattoos are faithfully transferred to novel views; mutual attention resolves semantic issues such as gender consistency.
- **COIN training is critical**: Without COIN, textures become blurry and high-frequency details such as teeth and hair strands are lost.
- **Strong robustness to diverse inputs**: Consistent 4D avatars can be generated from cartoons, line drawings, and images with extreme poses.
- Total generation time is approximately 2.5 hours, comparable to optimization-based methods such as GaussianAvatar (~2 hours), while inference speed is extremely fast (156 FPS).

## Highlights & Insights

- **Three-prior combination strategy**: Shape, texture, and dynamics priors each serve a distinct role, elegantly addressing the highly ill-posed problem of single-image 4D reconstruction. This modular prior-composition paradigm offers strong transferable insights for other ill-posed tasks.
- **Novel COIN training paradigm**: Explicitly isolating inconsistencies into a separate module is superior to conventional robust loss functions (e.g., L1), as it preserves the spatial localization of inconsistencies. This idea generalizes to any setting where learning from imperfect synthetic data is required.
- **High engineering completeness**: The full pipeline from generation to rendering supports FLAME parameter-driven animation and real-time rendering at 156 FPS, making it highly practical.

## Limitations & Future Work

- Generating a single avatar takes approximately 2.5 hours (dominated by the COIN training stage), which is far from real-time application requirements.
- The pipeline depends on the quality of PanoHead GAN inversion as initialization; inversion failures (e.g., under extreme occlusion) will propagate through the entire pipeline.
- Diffusion model texture enhancement may still introduce minor cross-view inconsistencies — while COIN mitigates this, addressing the root cause would be preferable.
- Robustness to extreme occlusion (sunglasses, masks) or illumination variation is not discussed.

## Related Work & Insights

- **vs. PanoHead/Portrait3D**: These methods support 360° view synthesis but yield limited texture quality and do not support animation. FaceCraft4D builds on them by adding texture enhancement and expression driving.
- **vs. Portrait-4D**: Based on a hybrid 2D+3D representation, it cannot handle extreme camera angles and lacks back-view modeling. FaceCraft4D uses a purely 3D representation (Gaussians), which structurally guarantees multi-view consistency.
- **vs. AniPortrait**: As a purely 2D method, it achieves high texture quality but suffers severe identity loss under large rotation angles. FaceCraft4D fundamentally avoids this issue through its 3D representation.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-prior combination and COIN training strategy are novel designs, though the individual modules largely build on existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both static and animation settings include comparisons and ablations, but a user study is absent.
- Writing Quality: ⭐⭐⭐⭐ Well-illustrated with clear pipeline descriptions; Tab. 1 is particularly informative.
- Value: ⭐⭐⭐⭐ A solid systems paper offering a practical staged solution; the COIN training strategy has independent value beyond this work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICCV 2025\] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling](music-aligned_holistic_3d_dance_generation_via_hierarchical_motion_modeling.md)
- [\[ICCV 2025\] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis](omegance_a_single_parameter_for_various_granularities_in_diffusion-based_synthes.md)

</div>

<!-- RELATED:END -->
