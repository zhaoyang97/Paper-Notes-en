---
title: >-
  [Paper Note] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture
description: >-
  [CVPR 2025][3D Vision][Audio-driven facial animation] This work proposes the TexTalk4D dataset (100-minute scan-level 8K dynamic textures) and the TexTalker framework, achieving simultaneous generation of facial motion and corresponding dynamic textures (wrinkle changes) from audio for the first time, and enabling disentangled control of motion/texture styles via a style pivot-based injection strategy.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Audio-driven facial animation"
  - "dynamic texture"
  - "3D talking avatar"
  - "diffusion models"
  - "disentangled style control"
date: 2026-05-08
content_hash: 8242573be973d5f8
---

# Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture

**Conference**: CVPR 2025  
**arXiv**: [2503.00495](https://arxiv.org/abs/2503.00495)  
**Code**: [Project Page](https://xuanchenli.github.io/TexTalk/)  
**Area**: 3D Vision  
**Keywords**: Audio-driven facial animation, dynamic texture, 3D talking avatar, diffusion models, disentangled style control

## TL;DR

This work proposes the TexTalk4D dataset (100-minute scan-level 8K dynamic textures) and the TexTalker framework, achieving simultaneous generation of facial motion and corresponding dynamic textures (wrinkle changes) from audio for the first time, and enabling disentangled control of motion/texture styles via a style pivot-based injection strategy.

## Background & Motivation

Audio-driven 3D facial animation has been extensively studied, but most prior works focus solely on geometric motion (mesh/vertex displacement), neglecting the importance of dynamic textures. The compression and formation of facial wrinkles during speech reflect muscle strain; the absence of dynamic textures significantly reduces rendering realism and even induces the uncanny valley effect.

Two key challenges exist: (1) **Lack of high-quality dynamic texture datasets** — existing 4D datasets are either estimated from monocular videos (low accuracy and temporal inconsistency) or captured via professional systems but lacking textures (e.g., VOCASET, MEAD-3D). Multiface, the only dataset containing dynamic textures, has only 13 subjects with a resolution of 1024 and lacks diversity. (2) **Joint generation of geometry and texture remains unstudied** — motion consists of 3D vertex displacements whereas texture consists of 2D images. Their representation spaces differ significantly, making it difficult to learn cross-domain correlations.

## Method

### Overall Architecture

TexTalker consists of three stages: (1) Unifying facial motion and texture variations into motion maps and wrinkle maps in the UV space, learning compact facial animation primitives via quantized autoencoders; (2) Training a Transformer-based Latent Diffusion Model (LDM) guided by audio to jointly generate latent codes for motion and wrinkles; (3) Utilizing style pivots to achieve disentangled control of motion and wrinkle styles.

### Key Design 1: Learning Facial Animation Primitives (Unified Representation)

- **Function**: Unifies heterogeneous 3D geometric motion and 2D texture variations into a similar latent space.
- **Mechanism**: Map vertex offsets $\mathbf{m}_t$ to the UV space to obtain motion maps $\mathbf{f}_t$, and represent texture variations as ratios relative to the neutral expression to obtain wrinkle maps $\mathbf{w}_t$. Two VQGAN encoders $\mathcal{E}_f$ and $\mathcal{E}_w$ are trained separately to compress them into a $16 \times 16 \times 16$ discrete latent space with a codebook size of 1024.
- **Design Motivation**: Direct generation on UV maps is computationally expensive, and the large discrepancy between motion and texture representations makes correlation learning difficult. Unifying them into the UV space and applying quantized encoding not only compresses the dimensionality but also places both modalities in the same representation framework, facilitating subsequent joint modeling.

### Key Design 2: Joint Motion-Wrinkle Latent Diffusion Model

- **Function**: Jointly generates temporally consistent sequences of facial motion and texture variations guided by audio.
- **Mechanism**: Concatenate the latent codes of motion and wrinkles into $\mathbf{X}^0 = [\mathbf{z}_f, \mathbf{z}_w]$, and employ an 8-layer Transformer decoder as the denoising network conditioned on HuBERT-extracted audio features. A sliding window strategy ($T_w=90, T_p=10$) is adopted to learn long-term dependencies, and an alignment mask is used to ensure that motion-wrinkle features are correlated only with the audio of the same frame.
- **Design Motivation**: Geometry and texture are intrinsically highly correlated (wrinkles reflect muscle movement); joint generation leverages complementary information to enhance the quality of both. Experiments demonstrate that joint learning outperforms separate learning across both modalities.

### Key Design 3: Disentangled Style Injection Based on Style Pivots

- **Function**: Achieves independent control over speaking style and wrinkle style.
- **Mechanism**: Leverage the clustering property of the learned codebook space, where latent codes for the same subject naturally cluster, and the cluster center (style pivot $\mathbf{p} = \frac{1}{T}\sum_{t=1}^T \mathbf{z}_t$) captures style features. The LDM is modified to predict the offset from the pivot $\Delta\mathbf{z} = \mathbf{z} - \mathbf{p}$ (which is phonetic-related but style-independent). During inference, arbitrary combinations can be achieved by adding $\mathbf{p}_{f,i}$ and $\mathbf{p}_{w,j}$ from different subjects.
- **Design Motivation**: One-hot style embeddings fail to capture complex styles and generalize poorly, while exemplar-based methods require additional networks. The style pivot is directly derived from the learned latent space, which is simple yet highly expressive.

### Loss & Training

- Animation primitives: $\mathcal{L}_{\text{latent}} = \mathcal{L}_{\text{rec}} + \eta_{\text{per}}\mathcal{L}_{\text{per}} + \eta_{\text{adv}}\mathcal{L}_{\text{adv}} + \eta_{\text{code}}\mathcal{L}_{\text{code}}$
- LDM training: $\mathcal{L}_{\mathcal{F}} = \|\hat{\mathbf{X}}^0 - \mathbf{X}^0\|^2$ (simple MSE loss)

## Key Experimental Results

### Main Results: Comparison of Facial Motion Quality (TexTalk4D-Test-A)

| Method | LVE↓ ($10^{-2}$mm) | MVE↓ ($10^{-2}$mm) | FDD↓ ($10^{-3}$mm) |
|------|-----|-----|-----|
| FaceFormer | 1.80 | 2.94 | 1.68 |
| CodeTalker | 1.83 | 2.80 | 1.38 |
| FaceDiffuser | 1.53 | 2.38 | 1.64 |
| **TexTalker** | **1.49** | **2.34** | **1.20** |

### Texture Quality Comparison

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Realism↑ | Consistency↑ |
|------|-------|-------|--------|----------|-------------|
| Static Texture | 39.79 | 0.967 | 0.0146 | 3.10 | 2.65 |
| Li et al. | 42.34 | 0.981 | 0.0187 | 3.91 | 3.78 |
| **Ours** | **44.13** | **0.985** | **0.0101** | **4.13** | **3.97** |

### Ablation Study

| Variant | LVE↓ | MVE↓ | PSNR↑ | SSIM↑ |
|------|------|------|-------|-------|
| w/o Wrinkle (Motion Only) | 1.73 | 2.76 | - | - |
| w/o Motion (Wrinkle Only) | - | - | 43.87 | 0.985 |
| Joint Codebook | 1.71 | 2.68 | 43.45 | 0.981 |
| w/o Pivot (one-hot) | 1.70 | 2.60 | 43.61 | 0.984 |
| **Full** | **1.49** | **2.34** | **44.13** | **0.985** |

### Key Findings

- Joint learning outperforms separate learning in both motion and texture quality, validating the value of cross-modal complementary information.
- Separate codebooks perform better than a joint codebook, indicating that while motion and texture are correlated, their optimal representation spaces differ.
- Style pivot injection significantly outperforms one-hot style embedding (LVE: 1.49 vs 1.70).

## Highlights & Insights

1. **First audio-driven 3D talking head generation method with dynamic textures**: Fills a gap in the literature and highlights the crucial role of dynamic textures in rendering realism.
2. **TexTalk4D Dataset**: A benchmark dataset featuring 100 subjects, 8K resolution, and scan-level precision.
3. **Elegance of Style Pivot**: Directly extracts style representations from the learned latent space without requiring extra networks, and achieves disentangled control.

## Limitations & Future Work

- Currently, dynamic textures are generated at a resolution of 512, requiring a super-resolution network to upsample them to 8K, which may cause loss of fine details.
- The dataset only contains young Asian faces, offering limited diversity.
- Expression-driven texture generation remains unexplored (only audio-driven).

## Related Work & Insights

- The unified representation in UV space can be extended to other tasks requiring the joint generation of heterogeneous modalities.
- The approach of extracting styles based on the clustering property of the latent space is simple yet effective, and can be applied to other stylized generation tasks.

## Rating

⭐⭐⭐⭐ — Proposes a new task and a high-quality dataset, with an elegant methodology design (especially the style pivot) and comprehensive experiments. The open-sourcing of the dataset will drive forward the development of this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[ICCV 2025\] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity](../../ICCV2025/3d_vision/hineus_high-fidelity_neural_surface_mitigating_low-texture_and_reflective_ambigu.md)
- [\[ICLR 2026\] LumiTex: Towards High-Fidelity PBR Texture Generation with Illumination Context](../../ICLR2026/3d_vision/lumitex_towards_high-fidelity_pbr_texture_generation_with_illumination_context.md)
- [\[CVPR 2025\] PERSE: Personalized 3D Generative Avatars from A Single Portrait](perse_personalized_3d_generative_avatars_from_a_single_portrait.md)

</div>

<!-- RELATED:END -->
