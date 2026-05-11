---
title: >-
  [Paper Note] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal
description: >-
  [ICCV 2025][Image Generation][Portrait Shadow Removal] This paper formulates portrait shadow removal as a diffusion inpainting problem. It trains an illumination-invariant structure extraction network to obtain structure…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Portrait Shadow Removal"
  - "Diffusion Inpainting"
  - "Structure Map Guidance"
  - "Detail Recovery"
  - "Relighting Data Synthesis"
date: 2026-05-08
content_hash: 4b9b0902fe9afe38
---

# Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal

**Conference**: ICCV 2025
**arXiv**: [2507.04692](https://arxiv.org/abs/2507.04692)
**Code**: [https://github.com/wanchang-yu/Structure-Guided-Diffusion-for-Portrait-Shadow-Removal](https://github.com/wanchang-yu/Structure-Guided-Diffusion-for-Portrait-Shadow-Removal)
**Area**: Image Generation
**Keywords**: Portrait Shadow Removal, Diffusion Inpainting, Structure Map Guidance, Detail Recovery, Relighting Data Synthesis

## TL;DR

This paper formulates portrait shadow removal as a diffusion inpainting problem. It trains an illumination-invariant structure extraction network to obtain structure maps free of shadow boundaries, uses these maps to guide an inpainting diffusion model for shadow region restoration, and applies a gradient-guided detail recovery diffusion model to reconstruct fine facial details. The proposed method substantially outperforms existing approaches on benchmark datasets.

## Background & Motivation

**Practical Need**: Selfie portraits are frequently degraded by shadows cast by external objects, which not only impairs visual aesthetics but also hinders downstream tasks such as face detection and recognition. Professional shadow removal remains inaccessible to general users.

**Limitations of Prior Work**:
- **General image shadow removal methods** (ShadowDiffusion, HomoFormer): Ill-suited for portraits, which have zero tolerance for distortion — even minor facial color or detail artifacts are unacceptable.
- **Synthetic data training methods** (PSM, BSR): Simple brightness/saturation adjustments ignore facial geometry and albedo, resulting in poor real-world generalization.
- **GAN inversion methods** (UPSR): Leverage StyleGAN2 priors but tend to alter facial identity.
- **Relighting methods** (IC-light): Estimating environmental lighting and facial geometry from a single image is inherently difficult and prone to unnatural results.

**Core Idea**: Shadow removal is reformulated as inpainting — the key challenge lies in obtaining a facial structure map free of shadow boundaries to guide the restoration.

## Method

### Overall Architecture (Three-Stage Pipeline)

1. **SE-Net**: Extracts an illumination-invariant facial structure map from the shadowed image.
2. **Structure-Guided Inpainting Diffusion Model**: Restores the shadow regions.
3. **Gradient-Guided Detail Recovery Diffusion Model**: Recovers fine facial details.

### Key Design 1: Illumination-Invariant Structure Extraction (SE-Net)

**Training Data Synthesis**:
- Shadow-free portraits $I$ are selected from CelebA.
- Physically-based relighting generates $I_{relit}$.
- A random facial mask blends the two: $I_{syn} = M \odot I_{relit} + (1-M) \odot I$

**Training Objective**: An existing edge extraction model PDG generates pseudo ground-truth from the original image. SE-Net is trained to produce the same structure map from the synthesized image containing illumination discontinuities:

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1\mathcal{L}_{perceptual} + \lambda_2\mathcal{L}_{GAN}$$

$$\mathcal{L}_{rec} = \|G_s(I_{syn}) - G_p(I)\|_1, \quad \mathcal{L}_{perceptual} = \mathcal{L}_{LPIPS}(G_s(I_{syn}), G_p(I))$$

### Key Design 2: Structure-Guided Inpainting Diffusion Model

The denoising process is conditioned on the structure map $S$ and the masked input $I_M$:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\cdot\mathbf{e}_t}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\mathbf{e}_t$$

$$\mathbf{e}_t = \epsilon_\theta(x_t, I_M, S, M, t)$$

- During training, random masks are applied to shadow-free portraits to learn conditional reconstruction.
- DDIM sampling is employed for acceleration; the structure map ensures structurally consistent restoration.
- **Mask Refinement**: The difference between the inpainted result and the original image is thresholded via Otsu's method to produce a refined mask, excluding non-shadow regions from modification.

### Key Design 3: Gradient-Guided Detail Recovery

The structure extraction model focuses on large-scale structure and may miss fine details such as eyelashes, moles, and freckles. The proposed solution:
- Extracts **image gradients** from the original shadow region as a guidance condition.
- Trains a second diffusion model conditioned on gradient maps to refine the inpainting result.
- The forward/reverse processes share the same formulation as the structure-guided diffusion model, differing only in conditioning.

## Experiments

### Quantitative Comparison (Real Portrait Shadow Dataset)

| Method | SSIM↑ | LPIPS↓ | RMSE↓ | Shadow SSIM↑ | Shadow RMSE↓ |
|---|---|---|---|---|---|
| ShadowDiffusion | 0.650 | 0.177 | 38.25 | 0.901 | 15.56 |
| Inpaint4Shadow | 0.766 | 0.094 | 22.71 | 0.910 | 17.25 |
| HomoFormer | 0.786 | 0.100 | 21.72 | 0.913 | 16.25 |
| IC-light | 0.514 | 0.278 | 62.54 | 0.916 | 33.06 |
| UPSR (GAN) | 0.731 | 0.109 | 26.92 | 0.900 | 20.66 |
| **Ours (full)** | **0.830** | **0.056** | **17.16** | **0.973** | **10.20** |

### Ablation Study

| Configuration | SSIM↑ | LPIPS↓ | RMSE↓ |
|---|---|---|---|
| w/o structure guidance | 0.757 | 0.101 | 23.60 |
| PDG structure (with shadow boundaries) | 0.779 | 0.092 | 19.96 |
| PSM data synthesis strategy | 0.785 | 0.086 | 19.80 |
| w/o detail recovery | 0.814 | 0.062 | 18.54 |
| **Full method** | **0.830** | **0.056** | **17.16** |

### Key Findings

- The full method achieves an LPIPS of 0.056, substantially surpassing the second-best HomoFormer at 0.100.
- Shadow-region SSIM reaches 0.973 (near-perfect), demonstrating that structure guidance ensures precise restoration.
- **Illumination invariance of SE-Net is critical**: using PDG structure maps containing shadow boundaries yields only 0.779 SSIM.
- The **detail recovery diffusion model** reduces LPIPS from 0.062 to 0.056, confirming the contribution of fine details such as eyelashes.
- The method is robust to imprecise shadow masks — coverage of the shadow region is sufficient.

## Highlights & Insights

1. **Elegant problem formulation**: Recasting shadow removal as inpainting fully exploits the generative capacity of diffusion models.
2. **Unsupervised training**: Relighting-based data synthesis eliminates the need for real paired shadowed/shadow-free portrait data.
3. **Three-stage cascade**: Each stage addresses a distinct core problem — structure extraction → region restoration → detail recovery.
4. **Mask refinement strategy** enhances robustness to imprecise input masks.

## Limitations & Future Work

- Inference requires three cascaded models, resulting in slower processing.
- The method depends on shadow mask input, necessitating an additional shadow detection step.
- A domain gap remains between synthesized training data and real-world shadows.

## Related Work & Insights

- **Shadow Removal**: ShadowFormer, HomoFormer, Inpaint4Shadow
- **Portrait Shadow**: PSM, BSR, UPSR
- **Diffusion Inpainting**: RePaint, MAT

## Rating

- Novelty: ⭐⭐⭐⭐ — Reformulating shadow removal as inpainting offers a fresh perspective.
- Technical Depth: ⭐⭐⭐⭐ — The three-stage design is tightly integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablation on real-world data.
- Practical Value: ⭐⭐⭐⭐ — Applicable to selfie enhancement and face recognition preprocessing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PersonalVideo: High ID-Fidelity Video Customization without Dynamic and Semantic Degradation](personalvideo_high_id-fidelity_video_customization_without_dynamic_and_semantic_.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](../../CVPR2026/image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/image_generation/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)

</div>

<!-- RELATED:END -->
