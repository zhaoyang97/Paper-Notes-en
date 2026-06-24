---
title: >-
  [Paper Note] Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior
description: >-
  [CVPR 2025][3D Vision][Monocular video human reconstruction] Proposes Vid2Avatar-Pro, which utilizes a Universal Prior Model (UPM) learned from multi-view dressed human motion capture data of thousands of individuals to create photorealistic and animatable 3D human avatars from monocular in-the-wild videos, significantly surpassing existing methods in novel view/pose synthesis.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Monocular video human reconstruction"
  - "Universal prior model"
  - "3D Gaussian Splatting"
  - "Animatable avatar"
  - "Front-back view parameterization"
date: 2026-05-08
content_hash: 3ad74a9218afcf8b
---

# Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior

**Conference**: CVPR 2025  
**arXiv**: [2503.01610](https://arxiv.org/abs/2503.01610)  
**Code**: None (Meta Internal Project)  
**Area**: 3D Vision  
**Keywords**: Monocular video human reconstruction, Universal prior model, 3D Gaussian Splatting, Animatable avatar, Front-back view parameterization

## TL;DR

Proposes Vid2Avatar-Pro, which utilizes a Universal Prior Model (UPM) learned from multi-view dressed human motion capture data of thousands of individuals to create photorealistic and animatable 3D human avatars from monocular in-the-wild videos, significantly surpassing existing methods in novel view/pose synthesis.

## Background & Motivation

Creating animatable, high-quality 3D human avatars from monocular in-the-wild videos faces two core challenges: (1) **Poor pose generalization** — the limited variety of poses in video leads to artifacts and deformations in poses outside the training distribution; (2) **View overfitting** — limited view coverage makes inverse rendering prone to overfitting, producing distortion when rendering from unseen views.

Existing methods mitigate these problems through statistical priors such as SMPL or heuristic regularization (like Laplacian smoothing). However, these priors have fundamental limitations: SMPL models minimally clothed naked bodies and lacks appearance information; Laplacian regularization penalizes deformation uniformly without distinguishing materials.

Core Argument: **There is a need to directly learn priors from large-scale dressed human data**, rather than relying on generic geometric priors.

## Method

### Overall Architecture

Two stages: (1) **UPM Training** — On multi-view dressed human capture data of thousands of individuals, using front-back view texture maps as identity conditions and pose position maps as inputs, a U-Net is trained to predict pose-dependent 3D Gaussian attribute maps; (2) **In-the-wild Personalization** — Reconstructs a canonical template from a monocular video, using a diffusion model to inpaint invisible textures, and fine-tunes the UPM to restore personal details.

### Key Design 1: Front-Back View Universal Parameterization

- **Function**: Provides a unified 2D parameterized representation for dressed humans of different identities and clothing topologies.
- **Mechanism**: Projects the canonical 3D template of each subject onto the front and back views via orthogonal projection, obtaining the position map $\mathcal{P}_c$ and texture map $\mathcal{T}_c$. Skeleton normalization ($\beta=0, \theta=\theta_{\text{cano}}$) ensures spatial alignment across identities. The texture map $\mathcal{T}_c$ serves as the identity condition input to the U-Net.
- **Design Motivation**: UV parameterization requires manual design and is limited to fixed topologies; SMPL UV only covers naked bodies. Front-back view parameterization is automatically acquired, supports diverse clothing topologies, and maximizes spatial alignment.

### Key Design 2: Universal Prior Model (UPM) Architecture and Training

- **Function**: Learns cross-identity pose-dependent deformation and appearance variations of dressed humans.
- **Mechanism**: The U-Net is conditioned on the identity texture $\mathcal{T}_c$ with the pose position map $\mathcal{P}_d(\Theta)$ as input to predict the pose-dependent Gaussian attribute map $\mathcal{G}(\Theta)$. Instead of absolute values, it predicts position and color offsets $\Delta\mathbf{x}(\Theta), \Delta\mathbf{c}(\Theta)$ to focus the model on learning fine-grained details. After LBS transformation to the pose space, rasterized rendering is performed with the loss $\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{lpips}}\mathcal{L}_{\text{lpips}} + \lambda_{\text{offset}}\mathcal{L}_{\text{offset}}$.
- **Design Motivation**: Predicting offsets is easier to learn than predicting absolute attributes (the prior template already provides coarse position and color). Trained on large-scale data of 1000 identities $\times$ ~5000 frames/person, the UPM learns realistic pose-dependent deformation patterns.

### Key Design 3: In-the-Wild Personalization Pipeline (with Diffusion Texture Inpainting)

- **Function**: Adapts the UPM to monocular in-the-wild videos to create personalized avatars.
- **Mechanism**: Three steps—(a) Preprocess the video using a SMPL-X estimator + Sapiens keypoints + SAM segmentation; (b) Train a diffusion model (DiT architecture) for 2D inpainting on the canonical texture map to complete invisible regions; (c) Fine-tune the UPM network weights and pose parameters using the complete texture as a condition.
- **Design Motivation**: Monocular videos inevitably lack texture in regions like the back. Diffusion inpainting leverages the context complementarity of the front and back views to compensate for missing parts. Fine-tuning for a small number of iterations both restores personal details and preserves the generalization ability of the prior.

### Loss & Training

$\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{lpips}}\mathcal{L}_{\text{lpips}} + \lambda_{\text{offset}}\mathcal{L}_{\text{offset}}$, including $L_1$ rendering loss, LPIPS perceptual loss, and offset regularization.

## Key Experimental Results

### Interpolation Synthesis: NeuMan Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓(×100) |
|------|-------|-------|-------------|
| HumanNeRF | 27.06 | 0.967 | 1.92 |
| GaussianAvatar | 29.94 | 0.980 | 1.24 |
| ExAvatar | 31.39 | 0.981 | 1.64 |
| **Ours** | **32.71** | **0.983** | **1.19** |

### Extrapolation Synthesis: MonoPerfCap Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓(×100) |
|------|-------|-------|-------------|
| Vid2Avatar | 28.49 | 0.976 | 2.46 |
| ExAvatar | 30.29 | 0.979 | 2.19 |
| **Ours** | **31.97** | **0.981** | **1.37** |

### Ablation Study: Number of Training Identities

| Number of Training Identities | PSNR↑ | LPIPS↓ |
|-----------|-------|--------|
| 4 | 31.28 | 1.53 |
| 128 | 31.34 | 1.45 |
| **1000** | **31.97** | **1.37** |

### Key Findings

- Increasing the number of training identities from 4 to 1000 leads to continuous quality improvement, validating the value of data scaling.
- Skeleton normalization is crucial for extrapolation performance (removing it drops LPIPS from 1.37 to 1.51).
- Diffusion texture inpainting effectively compensates for the missing areas in monocular videos.
- The fine-tuning phase is indispensable (without fine-tuning, PSNR is 29.24 vs. 31.97).

## Highlights & Insights

1. **Learning dressed human priors from data**: For the first time, a universal prior for dressed humans is trained from multi-view capture data of thousands of individuals, making it more suitable for real-world applications than statistical priors like SMPL.
2. **Universality of front-back view parameterization**: The simple and elegant design supports diverse clothing topologies without requiring manual UV mapping.
3. **Significant quality advantage over peers**: PSNR is improved by 1.3+ dB on NeuMan and 1.7+ dB on MonoPerfCap.

## Limitations & Future Work

- Relies on multi-view capture... data of thousands of individuals (Meta internal resource), making reproducibility difficult.
- Currently only supports SMPL-X skeleton control, lacking fine-grained facial and hand control.
- Diffusion texture inpainting may generate textures inconsistent with reality.

## Related Work & Insights

- The paradigm of "universal prior + personalized fine-tuning" can be extended to other 3D reconstruction tasks requiring few-shot personalization.
- The front-back view parameterization approach serves as a reference for any task requiring a unified representation across topologies.

## Rating

⭐⭐⭐⭐⭐ — Large-scale data-driven universal priors represent the correct direction for solving monocular video human reconstruction. The experimental results are convincing, and the front-back view parameterization design is elegant. However, the high barrier to data acquisition limits reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Synthetic Prior for Few-Shot Drivable Head Avatar Inversion](synthetic_prior_for_few-shot_drivable_head_avatar_inversion.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](../../ICCV2025/3d_vision/moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)

</div>

<!-- RELATED:END -->
