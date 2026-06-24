---
title: >-
  [Paper Note] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video
description: >-
  [CVPR 2025][Human Understanding][Gaussian Head Avatar] FATE is proposed to reconstruct animratable full-head Gaussian avatars from monocular videos. By employing a sampling-based densification strategy (replacing threshold splitting), neural baking (converting discrete Gaussians into continuous UV texture maps to support editing), and a general completion framework (synthesizing the appearance of the back of the head), FATE achieves highly efficient and high-quality reconstru…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Gaussian Head Avatar"
  - "Texture Editing"
  - "Sampling-based Densification"
  - "Neural Baking"
  - "FLAME"
date: 2026-05-08
content_hash: 28a2bf1c6221f86b
---

# FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video

**Conference**: CVPR 2025  
**arXiv**: [2411.15604](https://arxiv.org/abs/2411.15604)  
**Code**: Yes (Project Page)  
**Area**: Human Understanding / Head Reconstruction  
**Keywords**: Gaussian Head Avatar, Texture Editing, Sampling-based Densification, Neural Baking, FLAME

## TL;DR

FATE is proposed to reconstruct animratable full-head Gaussian avatars from monocular videos. By employing a sampling-based densification strategy (replacing threshold splitting), neural baking (converting discrete Gaussians into continuous UV texture maps to support editing), and a general completion framework (synthesizing the appearance of the back of the head), FATE achieves highly efficient and high-quality reconstruction with a PSNR of 28.37 dB using only 49K Gaussians.

## Background & Motivation

**Background**: 3D Gaussian Splatting-based head avatar reconstruction has developed rapidly (e.g., FlashAvatar, SplattingAvatar). These methods bind Gaussians to FLAME meshes for animation control, but suffer from sub-optimal densification strategies and the inability to edit textures.

**Limitations of Prior Work**: (1) The threshold-based densification of standard 3DGS is redundant—splitting Gaussians where the gradient exceeds a threshold, but different head regions (e.g., hair vs. skin) require vastly different densities, and a unified threshold leads to over-density in some regions and insufficiency in others; (2) Discrete Gaussian points cannot be directly edited—users cannot modify the head appearance (e.g., changing hair color, adding makeup) like they would edit texture maps; (3) Monocular videos lack back-of-the-head information.

**Key Challenge**: The discreteness of Gaussian splatting vs. the continuity requirement of texture editing—Gaussians are independent points scattered in 3D space and cannot form editable continuous surfaces.

**Key Insight**: (1) Replace threshold-based densification with position gradient-based importance sampling; (2) Use "neural baking" to map discrete Gaussian attributes to a continuous UV space.

**Core Idea**: Sampling-based densification + neural baking to UV space + back-of-the-head completion via generative models = editable full-head Gaussian avatar.

## Method

### Key Designs

1. **Sampling-based Densification Strategy**:

    - **Function**: Adaptively add new Gaussians on faces with large gradients to avoid redundancy.
    - **Mechanism**: Calculate the position gradient magnitude of Gaussians on each FLAME face to serve as importance weights, and use importance sampling to generate new Gaussians on faces with high gradients. The probability is proportional to the gradient magnitude, naturally adding more Gaussians in areas with rich details (e.g., eye and mouth regions).
    - **Design Motivation**: Standard threshold-based splitting requires tedious hyperparameter tuning and easily leads to over-density (e.g., SplattingAvatar requires 558K Gaussians), whereas the sampling-based strategy keeps FATE at 49K (a 10x compression).

2. **Neural Baking**:

    - **Function**: Convert discrete Gaussian attributes into continuous UV texture maps to support texture editing.
    - **Mechanism**: Train a U-Net to learn a smooth mapping of Gaussian attributes (color/spherical harmonics/opacity) to FLAME's UV space: $f(\mathbf{p}) = (\mathcal{F} * \mathcal{H} * \mathcal{B})(\mathbf{p})$, where $\mathcal{F}$ is the Gaussian feature map, $\mathcal{H}$ is the Gaussian diffusion kernel, and $\mathcal{B}$ represents U-Net smoothing. After baking, users can directly edit on the UV map and map back to Gaussian rendering.
    - **Design Motivation**: Directly rasterizing Gaussians to UV space yields sparse and discrete point maps that are not editable. The U-Net interpolates discrete samples into continuous textures.

3. **Universal Back-of-the-Head Completion Framework**:

    - **Function**: Synthesize the unseen back-of-the-head appearance from a monocular frontal video.
    - **Mechanism**: Use the pretrained SphereHead generative model to invert frontal information into the generative space via PTI (Pivotal Tuning Inversion), synthesizing the texture and geometry of the back of the head.
    - **Design Motivation**: Monocular videos typically only capture the front, and the back-of-the-head data is missing.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{L1} + 0.1\mathcal{L}_{VGG} + 100\mathcal{L}_{lap} + 100\mathcal{L}_{FLAME} + 0.1\mathcal{L}_{scale}$. Laplacian smoothing regularizes mesh deformation, FLAME constrains the learnable blendshape offsets, and scale penalizes Gaussians with excessive anisotropy.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussians |
|------|-------|-------|--------|--------|
| FlashAvatar | 27.41 | 0.9397 | 0.0603 | - |
| MonoGaussianAvatar | 28.07 | 0.9405 | 0.0618 | - |
| SplattingAvatar | 27.89 | 0.9324 | 0.0643 | 558K |
| **FATE** | **28.37** | **0.9439** | **0.0586** | **49K** |

### Ablation Study

| Configuration | PSNR | LPIPS | Description |
|------|------|-------|------|
| Without densification | - | 0.0740 | Uneven distribution |
| Without learnable blendshapes | PSNR-4.58 | 0.1112 | Expression fitting collapse |
| Two-stage baking | 27.78 | - | Outperforms single-stage (27.42) |

### Key Findings
- **49K vs 558K Gaussians**: FATE achieves better metrics with only 1/10 of the Gaussians, showing that sampling-based densification is much more efficient than threshold-based splitting.
- **Minimal baking loss**: PSNR drops by only ~0.6 dB after baking, while gaining texture editing capability.
- **Learnable blendshapes are crucial**: Removing them causes the PSNR to drop by 4.58 dB.

## Highlights & Insights
- **Bridge between discrete and continuous**: Neural baking is the first attempt to combine the discrete representation advantages of 3DGS with the continuous editing requirements of traditional textures.
- **Sampling vs. Thresholding**: Replacing hard thresholds with probability-based sampling is a general improvement paradigm that can be extended to other 3DGS scenarios.

## Limitations & Future Work
- Assumes uniform and consistent lighting, and cannot handle dynamic lighting changes.
- Back-of-the-head completion relies on training biases of SphereHead, which may lead to identity drift.
- Fixed UV texture resolution, where extreme geometry might require MipMapping.

## Rating
- Novelty: ⭐⭐⭐⭐ The sampling-based densification and neural baking are practical and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully ablated with 20 subjects across 4 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear and comprehensive.
- Value: ⭐⭐⭐⭐ Unlocks editing capabilities for 3DGS-based head avatars.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[CVPR 2026\] Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](../../CVPR2026/human_understanding/avatar_forcing_real-time_interactive_head_avatar_generation_for_natural_conversa.md)
- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](../../ICCV2025/human_understanding/controllable_and_expressive_one-shot_video_head_swapping.md)

</div>

<!-- RELATED:END -->
