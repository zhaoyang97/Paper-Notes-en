---
title: >-
  [Paper Note] Bridging the Gap: Studio-Like Avatar Creation from a Monocular Phone Capture
description: >-
  [ECCV2024][Image Generation][avatar creation] This work proposes a method to generate studio-quality facial texture maps from monocular phone videos, combining the $W^+$ space parameterization of StyleGAN2 and diffusion-model-based super-resolution to bridge the gap from smartphone scans to high-quality 3D avatars.
tags:
  - "ECCV2024"
  - "Image Generation"
  - "avatar creation"
  - "texture map"
  - "StyleGAN2"
  - "diffusion model"
  - "phone capture"
  - "studio lighting"
date: 2026-05-08
content_hash: 93cbb9e72730711d
---

# Bridging the Gap: Studio-Like Avatar Creation from a Monocular Phone Capture

**Conference**: ECCV2024  
**arXiv**: [2407.19593](https://arxiv.org/abs/2407.19593)  
**Code**: No public code  
**Area**: Image Generation  
**Keywords**: avatar creation, texture map, StyleGAN2, diffusion model, phone capture, studio lighting

## TL;DR

This work proposes a method to generate studio-quality facial texture maps from monocular phone videos, combining the $W^+$ space parameterization of StyleGAN2 and diffusion-model-based super-resolution to bridge the gap from smartphone scans to high-quality 3D avatars.

## Background & Motivation

Traditional high-quality avatar creation relies on complex and expensive studio setups like LightStage for multi-view capture under uniform lighting. Although recent neural representation-based methods can quickly generate drivable 3D avatars from smartphone scans, they suffer from three core limitations:

1. **Baked-in Lighting**: The ambient lighting during capture is directly encoded into the textures, making it inseparable from the reflectance.
2. **Lack of Details**: The resolution of facial details (e.g., wrinkles, pores) is insufficient.
3. **Incomplete Regions**: Missing regions or holes exist in unobserved areas, such as behind the ears.

These issues result in phone-captured avatars having much lower quality than studio-grade ones, limiting the practical deployment of consumer-grade avatar creation.

## Core Problem

How to generate texture maps with studio-grade uniform illumination, complete coverage, and high-resolution facial details from a short monocular phone video? The key challenges lie in removing environmental lighting while preserving identity consistency, and filling in invisible areas.

## Method

The overall pipeline consists of two stages: StyleGAN2-based illumination transfer and region completion (GMug), and diffusion-model-based facial detail super-resolution.

### Stage 1: GMug — StyleGAN2 Illumination Transfer

**W+ Space Parameterization**: First, the texture map captured by the phone is mapped into the $W^+$ latent space of StyleGAN2 via GAN Inversion, achieving near-perfect reconstruction. Each layer in the $W^+$ space has an independent style vector, where low-resolution layers encode identity information and high-resolution layers encode lighting and details.

**Adversarial Fine-tuning**: StyleGAN2 is fine-tuned using a small number of studio-grade texture maps as real samples for the discriminator. The key design is to only optimize the network parameters after the $8 \times 8$ resolution (denoted as $\theta(8+)$), while freezing the low-resolution parameters to prevent the identity representation from being altered.

**Optimization Objective**:

$$\min_{\text{GMug}^{\theta(8+)}} \max_{D_{Studio}} \mathcal{L}_{Adv} + \mathcal{L}_{R1} + \mathcal{L}_{Percp\text{-}Recons} + \lambda_1 \mathcal{L}_{Percp} + \lambda_2 \mathcal{L}_{FaceID}$$

Functions of each loss term:

- **$\mathcal{L}_{Adv}$**: Adversarial loss, driving the generated results to approach the studio lighting distribution.
- **$\mathcal{L}_{R1}$**: Discriminator regularization to stabilize training.
- **$\mathcal{L}_{FaceID}$**: Identity preservation loss based on a face recognition network to prevent identity drift.
- **$\mathcal{L}_{Percp}$**: Perceptual loss, improving training stability and preserving facial structures.
- **$\mathcal{L}_{Percp\text{-}Recons}$**: Perceptual reconstruction loss, using a small amount of paired data to prevent global skin tone shift.

### Stage 2: Diffusion Model-Based Facial Detail Enhancement

While the output of GMug has uniform lighting and complete coverage, its resolution is limited by the generative capacity of StyleGAN2. To address this, a diffusion model is designed for texture map super-resolution. Its key characteristic is using the image gradient of the phone-captured texture map as a guidance signal to ensure that the enhanced details align with the original facial features.

## Key Experimental Results

### Optimization Resolution Ablation

| Setting | FaceID ↓ | KID ↓ |
|------|----------|-------|
| Full Network Optimization | 5.01e-4 | **1.36e-3** |
| Optimization after 8×8 (Ours) | **4.31e-4** | 1.42e-3 |
| Optimization after 16×16 | 4.30e-4 | 1.63e-3 |

Freezing parameters before the $8 \times 8$ resolution achieves the best balance between identity preservation and distribution realism.

### Loss Function Ablation

| Setting | FaceID ↓ |
|------|----------|
| Full Loss | **5.36e-4** |
| w/o $\mathcal{L}_{FaceID}$ | 1.33e-3 |
| w/o $\mathcal{L}_{FaceID}$ & $\mathcal{L}_{Percp}$ | 2.79e-3 |

Removing the identity loss degrades the FaceID metric by 2.5x; simultaneously removing the perceptual loss leads to training divergence.

### Qualitative Results

Comparisons on unpaired phone-captured data show that the proposed method comprehensively outperforms previous work in terms of identity preservation, realism of facial details, lighting uniformity, and missing region inpainting.

## Highlights & Insights

1. **Clever Use of StyleGAN2's Hierarchical Structure**: By freezing low-resolution layers to preserve identity and fine-tuning high-resolution layers to transfer lighting, the decoupling of identity and illumination is achieved.
2. **Works with Minimal Studio Data**: Requires only a small amount of studio-grade textures as adversarial training signals, avoiding the need for large-scale paired datasets.
3. **Highly Complementary Two-Stage Design**: The GAN is responsible for global illumination transfer and region completion, while the diffusion model handles local high-frequency detail enhancement.
4. **End-to-End Practicality**: Takes standard phone videos as input and outputs high-quality texture maps directly ready for rendering.

## Limitations & Future Work

1. **Head-Only Modeling**: Critical regions such as shoulders and torso are not covered, limiting applications to full-body avatars.
2. **Inability to Handle Head Accessories**: Accessories like hats and hairbands are processed incorrectly because the studio training dataset does not contain such items.
3. **Reliance on Pre-trained 3D Models**: It requires a Universal Prior Model (such as AVA) to render the final results.
4. **Uncertain Generalization**: Robustness under extreme lighting conditions (e.g., backlighting, colored lights) has not been fully verified.

## Related Work & Insights

- **AVA (Cao et al.)**: Provides a drivable avatar framework, but the texture quality is constrained by phone capture.
- **StyleGAN-ADA**: This work borrows its few-shot fine-tuning concepts but introduces designs specifically tailored for the texture map domain.
- **Traditional GAN Inversion**: Rather than simple image editing, this work addresses cross-domain transfer (phone lighting to studio lighting).
- **Diffusion-based Super-Resolution**: Uses image gradient guidance instead of simple upsampling, ensuring the fidelity of facial details.

### Insights & Connections

- **Hierarchical GAN freezing strategies** can be extended to other style transfer tasks that require preserving specific semantic attributes.
- The **few-shot adversarial fine-tuning** paradigm is suitable for scenarios where high-quality data is scarce but low-quality data is abundant.
- Complementary to the relighting field: This work performs lighting normalization at the texture map level, rather than pixel-level relighting.
- The pattern of using a diffusion model as a post-processing enhancer is generic and worthy of adoption in other 3D reconstruction pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combined design of StyleGAN2 $W^+$ space, hierarchical freezing, and diffusion enhancement is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The ablation studies are comprehensive, but it lacks user studies and references few quantitative baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ — The methodology is clear, and the motivation is well-articulated.
- Value: ⭐⭐⭐⭐ — Offers direct practical value for consumer-grade avatar generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICLR 2026\] Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution](../../ICLR2026/image_generation/bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[ICML 2026\] SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](../../ICML2026/image_generation/spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)

</div>

<!-- RELATED:END -->
