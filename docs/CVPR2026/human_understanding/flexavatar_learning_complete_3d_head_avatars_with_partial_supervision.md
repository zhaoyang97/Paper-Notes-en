---
title: >-
  [Paper Note] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision
description: >-
  [CVPR 2026][Human Understanding][3D head avatars] FlexAvatar introduces learnable *bias sink* tokens to unify training across monocular and multi-view data…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "3D head avatars"
  - "single-image reconstruction"
  - "bias sinks"
  - "3D Gaussian Splatting"
  - "Transformer"
date: 2026-05-08
content_hash: de6ef039dc702a6c
---

# FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision

**Conference**: CVPR 2026
**arXiv**: [2512.15599](https://arxiv.org/abs/2512.15599)  
**Code**: [Available](https://tobias-kirschstein.github.io/flexavatar/)  
**Area**: Human Understanding / 3D Head Avatar Generation
**Keywords**: 3D head avatars, single-image reconstruction, bias sinks, 3D Gaussian Splatting, Transformer

## TL;DR

FlexAvatar introduces learnable *bias sink* tokens to unify training across monocular and multi-view data, resolving the entanglement between driving signals and target viewpoints, and enables the generation of complete, high-quality, animatable 3D head avatars from a single image.

## Background & Motivation

Creating high-quality, animatable 3D head avatars from a single image is a highly challenging problem. The difficulty stems from two aspects: (1) the large number of unobservable regions makes 3D reconstruction severely under-constrained; and (2) the model must infer plausible facial animations for expressions never seen during inference.

**Dilemma of existing methods**:

- **Multi-view data** provides complete 3D supervision but is limited in scale and difficult to acquire.
- **Monocular video data** (e.g., in-the-wild face videos) covers a wide range of identities but offers only a single viewpoint, introducing a strong frontal bias that causes trained models to reconstruct incomplete 3D heads.
- **3DMM priors** (e.g., FLAME) provide coarse geometry and animation capability but constrain expressiveness.

**Core finding**: The authors identify the root cause as the **entanglement between driving signals and target viewpoints** in monocular training data. Specifically, in a monocular self-reenactment setting, the expression control signal is extracted from the target image itself, allowing the model to infer the target viewpoint from the expression input—incentivizing the model to predict only a partial 3D head while still satisfying the loss. Naively mixing monocular and multi-view training data does not resolve this entanglement.

## Method

### Overall Architecture

FlexAvatar adopts an encoder–decoder architecture:

1. **Encoder $E$**: Extracts a compact avatar code $\mathcal{A} \in \mathbb{R}^{H_l \times W_l \times D}$ (a 2D latent code in UV space) from the input image $I$.
2. **Decoder $D$**: Fuses facial expression $z_{exp}$ into the avatar representation to produce animated 3D Gaussian attributes.
3. **Renderer $\mathcal{R}$**: Differentiable rasterization based on 3DGS for rendering from arbitrary viewpoints.

### Key Designs

#### 1. Encoder: Projection onto the Avatar Manifold

- Employs a pretrained DINOv2 with a shallow learnable ViT to extract image features $f_{img}$.
- Defines queries $Q$ in the UV space of a template head mesh (via sinusoidal positional encoding).
- Maps image features into UV space via cross-attention to obtain a viewpoint- and expression-agnostic avatar code $\mathcal{A}$.

Mechanism: Query points anchored in UV space retrieve information from image features via cross-attention.

#### 2. Bias Sinks — Core Contribution

Problem: In monocular data, $I_{drive} = I_{target}$, so the expression code $z_{target}$ leaks information about the target viewpoint $\pi_{target}$.

Solution: Two learnable tokens are introduced—$z_{2D}$ (for monocular data) and $z_{3D}$ (for multi-view data)—appended to the expression encoding sequence $s_{exp}$:

$$s_{exp} \leftarrow [s_{exp}, z_{bias}]$$

Design Motivation and Mechanism:
- **During training**: monocular samples use $z_{2D}$; multi-view samples use $z_{3D}$, allowing the decoder to explicitly distinguish data sources.
- The model learns to predict a partial 3D head via the $z_{2D}$ path and a complete avatar via the $z_{3D}$ path.
- Crucially, knowledge is still shared across dataset types: the $z_{3D}$ path benefits from the generalization capacity brought by monocular data.
- **During inference**: $z_{3D}$ is always used, yielding both strong **generalization** and **3D completeness**.

#### 3. Decoder + StyleGAN-PixelShuffle Upsampler

- Cross-attention between the avatar code and the serialized expression encoding enables model-free animation.
- A hybrid upsampling architecture combining PixelShuffle and StyleGAN2 CNN blocks achieves an overall upsampling rate of 8×.
- Per-Gaussian attributes are decoded via grid sampling and an MLP.
- Gaussian positions are initialized on the template mesh surface with learned residual offsets.

#### 4. Avatar Latent Space Fitting

Training naturally yields a smooth avatar latent space that supports additional capabilities:
- **Few-shot avatar creation**: An initial $\mathcal{A}^{init}$ is obtained by encoding a single image, followed by optimization over all available observations.
- **Monocular video avatar creation**: The same fitting procedure applies, optimizing only $\mathcal{A}$ with the decoder frozen.
- Unlike autodecoder methods, the encoder provides an initialization estimate, accelerating optimization.

### Loss & Training

The reconstruction loss combines four terms:

$$\mathcal{L}_{rec} = \mathcal{L}_1 + \mathcal{L}_{SSIM} + \mathcal{L}_{DINO} + \mathcal{L}_{SAM}$$

| Loss Term | Description |
|-----------|-------------|
| $\mathcal{L}_1$ | L1 pixel loss |
| $\mathcal{L}_{SSIM}$ | Structural similarity loss |
| $\mathcal{L}_{DINO}$ | Perceptual loss on DINOv2 intermediate feature maps |
| $\mathcal{L}_{SAM}$ | Perceptual loss on SAM intermediate feature maps |

Training details:
- Joint training on 5 datasets (2 monocular + 2 multi-view + 1 synthetic multi-view).
- Adam optimizer with learning rate 1e-4.
- Perceptual losses introduced after 400k steps to avoid early overfitting to high-frequency details.
- Total of 1M steps, batch size 20, approximately 3 weeks on a single A100.

## Key Experimental Results

### 3D Portrait Animation (VFHQ Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CSIM↑ |
|--------|-------|-------|--------|-------|
| GAGAvatar | 21.83 | 0.818 | 0.122 | 0.816 |
| LAM | 22.65 | 0.829 | 0.109 | 0.822 |
| **FlexAvatar** | **23.47** | **0.837** | **0.099** | **0.830** |

### Single-Image Avatar Creation (Ava256 Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | AKD↓ | CSIM↑ |
|--------|-------|-------|--------|------|-------|
| Portrait4Dv2 | 11.9 | 0.671 | 0.404 | 7.77 | 0.578 |
| GAGAvatar | 12.7 | 0.709 | 0.371 | 7.45 | 0.555 |
| LAM | 13.1 | 0.702 | 0.399 | 11.2 | 0.411 |
| **FlexAvatar** | **16.9** | **0.762** | **0.265** | **5.52** | **0.695** |

PSNR improves by 3.8+ dB with a substantial lead in LPIPS, demonstrating markedly superior 3D completeness and quality over prior methods.

### Ablation Study

| Configuration | 2D | 3D | Bias Sinks | StyleGAN | PSNR↑ | CSIM↑ |
|---------------|:---:|:---:|:---:|:---:|-------|-------|
| only 2D | ✓ | | | ✓ | 13.7 | 0.593 |
| only 3D | | ✓ | | ✓ | 13.2 | 0.119 |
| w/o bias sinks | ✓ | ✓ | | ✓ | 14.5 | 0.583 |
| w/o StyleGAN | ✓ | ✓ | ✓ | | 17.1 | 0.614 |
| Ours_ref | ✓ | ✓ | ✓ | ✓ | 17.2 | 0.621 |
| Ours + fitting | ✓ | ✓ | ✓ | ✓ | 16.9 | **0.682** |

### Key Findings

- **Monocular data only**: Good generalization but incomplete 3D reconstruction (due to entanglement).
- **Multi-view data only**: Complete 3D but severely degraded generalization (CSIM only 0.119).
- **Naive data mixing** (without bias sinks): Fails to resolve entanglement; performance is comparable to the monocular-only setting.
- **Bias sinks are effective**: They enable the model to adopt distinct strategies for different data sources.
- **Fitting further improves results**: Identity preservation (CSIM) and expression fidelity (AKD) improve noticeably with a fitting time of approximately 1 minute.

## Highlights & Insights

1. **Precise problem diagnosis**: The identification of the driving-signal–target-viewpoint entanglement as the core obstacle reflects deeper insight than simply accumulating more data.
2. **Minimalist yet effective bias sink design**: Two learnable tokens suffice to decouple dataset-specific biases without complex architectural modifications.
3. **Freedom from 3DMM constraints**: Facial animation is learned in a data-driven manner, no longer restricted to FLAME's predefined expression space.
4. **Unified framework across multiple scenarios**: A single model handles single-image, few-shot, and monocular video avatar creation.
5. **On the NeRSemble benchmark**: 10-minute fitting surpasses CAP4D's 4-hour fitting.

## Limitations & Future Work

- Lighting is baked in from the input image and cannot be explicitly controlled, which may produce unnatural results when the avatar is placed in different virtual environments.
- Although the architecture is model-free, all experiments use FLAME expression codes, limiting fine details such as the tongue.
- The approach could potentially generalize to full-body avatars or general dynamic novel-view synthesis, but has currently only been validated for the head.
- Training requires approximately 3 weeks on a single A100, representing a considerable computational cost.

## Related Work & Insights

- The encoder design of **LAM** (UV-space queries + cross-attention) inspired the architecture of FlexAvatar.
- The model-free animation paradigm of **Avat3r** (cross-attention to expression encoding sequences) is adopted in this work.
- The per-image embedding concept from **NeRF-in-the-wild** shares conceptual similarities with bias sinks, though bias sinks operate at the dataset level rather than the image level.
- The design principle of bias sinks (learnable tokens absorbing specific biases) may have broad applicability to other multi-source mixed-data training settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Precise diagnosis of the viewpoint–expression entanglement; bias sinks are concise and original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four tasks, three datasets, and detailed ablations validate each design choice.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical structure, intuitive figures, and thorough problem exposition.
- **Value**: ⭐⭐⭐⭐⭐ — A substantial advance in single-image 3D avatar creation; the general bias sink design principle merits broader adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[CVPR 2026\] MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](../../ICCV2025/human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)

</div>

<!-- RELATED:END -->
