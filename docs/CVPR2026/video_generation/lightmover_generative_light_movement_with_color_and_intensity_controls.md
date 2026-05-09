---
title: >-
  [Paper Note] LightMover: Generative Light Movement with Color and Intensity Controls
description: >-
  [CVPR 2026][Video Generation][Light source manipulation] LightMover leverages video diffusion priors to model light source editing as a sequence-to-sequence prediction problem. Through a unified control token representation, it achieves precise manipulation of light source position, color, and intensity. An adaptive token pruning mechanism reduces control sequence length by 41%, and the method outperforms existing approaches on both light movement and object movement tasks.
tags:
  - CVPR 2026
  - Video Generation
  - Light source manipulation
  - video diffusion models
  - illumination editing
  - adaptive token pruning
  - physically rendered data
date: 2026-05-08
content_hash: 9204fdfe1e5fa0ab
---

# LightMover: Generative Light Movement with Color and Intensity Controls

**Conference**: CVPR 2026
**arXiv**: [2603.27209](https://arxiv.org/abs/2603.27209)
**Code**: [Project Page](https://gengzezhou.github.io/LightMover/)
**Area**: Video Generation
**Keywords**: Light source manipulation, video diffusion models, illumination editing, adaptive token pruning, physically rendered data

## TL;DR

LightMover leverages video diffusion priors to model light source editing as a sequence-to-sequence prediction problem. Through a unified control token representation, it achieves precise manipulation of light source position, color, and intensity. An adaptive token pruning mechanism reduces control sequence length by 41%, and the method outperforms existing approaches on both light movement and object movement tasks.

## Background & Motivation

Precisely editing illumination from a single image is highly challenging due to the complex global interactions among lighting, geometry, materials, and occlusion. Existing methods fall into two categories: (1) inverse rendering approaches (which reconstruct geometry, materials, and lighting before re-rendering) are severely ill-posed from a single image and computationally expensive; (2) diffusion-based editing methods (e.g., LightLab) can adjust color tone and ambient light but cannot model the spatial movement of light sources. General-purpose image editing models (SDEdit, InstructPix2Pix, Gemini, etc.) lack explicit illumination parameterization and thus cannot achieve physically plausible lighting control.

The root cause lies in the fact that existing methods either lack the ability to model the spatial position of light sources or implicitly embed lighting into an object movement framework, failing to correctly propagate shadows, reflections, and light attenuation. The core idea of this paper is to extend the token sequence framework of ObjectMover to the domain of illumination editing, designing dedicated control tokens for color and intensity, and realizing physically consistent light source manipulation through a "2.5D" learning paradigm (approximating 3D light transport on 2D images using a video diffusion model).

## Method

### Overall Architecture

LightMover is built upon a 5B-parameter video diffusion Transformer and encodes all inputs as a pseudo-video frame sequence. The input sequence comprises: (1) a reference image $I_{\text{ref}}$; (2) a target object crop $I_{\text{obj}}$; (3) a movement map $I_{\text{move}}$ (encoding the source region in the R channel and the target region in the GB channels); (4) a color control frame $I_{\text{color}}$; (5) an intensity control frame $I_{\text{intensity}}$; and (6) a noise frame to be generated $X^t$. All frames are encoded into latent tokens via a VAE and jointly processed by the diffusion Transformer.

### Key Designs

1. **Multi-Signal Positional Encoding (MSPE)**:
    - Function: Enables the diffusion Transformer to correctly distinguish the semantic roles of different input frames.
    - Mechanism: Integrates four orthogonal positional subspaces — spatial encoding $(W, H)$ preserves intra-frame spatial structure, temporal encoding $(T)$ reflects sequence order, condition type encoding $(C)$ differentiates modalities (reference/object/movement/color/intensity/output), and frame role encoding $(R)$ distinguishes conditional inputs from predicted outputs. The four components are projected and additively combined, then modulated via a RoPE-like mechanism.
    - Design Motivation: Standard positional encoding cannot differentiate semantically distinct input frames. MSPE enables the model to jointly reason about spatial alignment and conditional dependencies.

2. **Adaptive Token Pruning**:
    - Function: Significantly reduces control sequence length while maintaining editing precision.
    - Mechanism: For spatial control signals (e.g., movement maps), the method retains full resolution or downsamples proportionally based on the bounding box area ratio. For non-spatial controls (color, intensity), a learnable downsampling rate compresses token counts. Overall, control token length is reduced by 41%.
    - Design Motivation: Naively encoding color and intensity as additional full-resolution frames causes sequence length to inflate, limiting model efficiency and maximum supported resolution.

3. **Physically Decoupled Rendering Data Pipeline**:
    - Function: Generates large-scale paired training data covering systematic variations in light source position, color, and intensity.
    - Mechanism: Using 25 indoor scenes and 100 light source assets in Blender, Monte Carlo path tracing decomposes each frame into an ambient base $I_{\text{amb}}$ and a direct light contribution $I_{\text{light}}$. Infinite lighting variants are then generated in post-processing via $I_{\text{relit}}(\alpha, G_{\text{illum}}, \mathbf{c}_t) = \alpha I_{\text{amb}} + G_{\text{illum}} I_{\text{light}} \odot \mathbf{c}_t$, where the intensity gain $G_{\text{illum}} = 2^{S_{\text{EV}}}$ is expressed in photographic exposure value (EV) units.
    - Design Motivation: Physical decoupling enables the model to learn the causal effects of lighting (how shadows shift, how reflections brighten, how indirect light propagates) rather than merely learning surface-level pixel changes.

### Loss & Training

The model is trained with a flow matching objective: $\mathcal{L} = \mathbb{E}_{t,X^0,X^1}[\|v(S^t, t; \theta)_{[6]} - V^t\|^2]$

Training employs a multi-task mixing strategy with a synthetic-to-real data ratio of 10:1. The seven task types (light movement : object movement : color change : intensity change : joint change : light removal : light insertion) are mixed at a ratio of 6:3:3:3:1:1. Training resolutions are mixed between 512×512 and 1024×1024 at a 1:1 ratio.

## Key Experimental Results

### Main Results

**Light Source Movement (LightMove-A)**:

| Method | PSNR ↑ | DINO ↑ | CLIP ↑ |
|---|---|---|---|
| Qwen-Image | 19.01 | 69.94 | 87.27 |
| Gemini-2.5-Flash | 19.59 | 72.46 | 89.72 |
| ObjectMover | 19.49 | 78.12 | 90.48 |
| **LightMover** | **20.38** | **81.27** | **91.85** |

**Light Source Color/Intensity Variation (LightMove-B)**:

| Method | Color PSNR | Intensity PSNR | Combined PSNR |
|---|---|---|---|
| Gemini-2.5-Flash | 22.14 | 25.09 | 18.42 |
| **LightMover** | **24.06** | **27.12** | **19.97** |

**General Object Movement (ObjMove-A)**:

| Method | PSNR ↑ | DINO ↑ | CLIP ↑ |
|---|---|---|---|
| ObjectMover | 25.27 | 85.07 | 93.16 |
| **LightMover** | **25.73** | **88.59** | **91.86** |

### Ablation Study

| Configuration | PSNR | DINO | CLIP | Notes |
|---|---|---|---|---|
| w/o Light Aug/Color/Intensity | 19.88 | 79.93 | 91.06 | Baseline, no auxiliary tasks |
| +Light Aug | 20.07 | 79.73 | 91.62 | Lighting augmentation improves quality |
| +All (full model) | 20.38 | 81.27 | 91.85 | Multi-task joint training is best |
| w/o frame-as-condition | 19.53 | 77.32 | 90.01 | Frame encoding outperforms embedding encoding |
| w/o adaptive downsample | 19.38 | 75.62 | 89.81 | Adaptive pruning is indispensable |

### Key Findings

- Multi-task joint training yields strong mutual reinforcement: position, color, and intensity signals mutually regularize each other, improving accuracy across all lighting tasks.
- Adaptive token pruning reduces control sequence length by 41% with negligible performance degradation (PSNR 20.39 with pruning vs. 20.38 without).
- LightMover not only leads on light source manipulation tasks but also surpasses ObjectMover on general object movement, demonstrating the generalization capability of the unified framework.

## Highlights & Insights

- The "2.5D" paradigm is particularly elegant — approximating 3D light transport in 2D space via a video diffusion model avoids the computational cost of full 3D reconstruction.
- Unifying illumination editing within an object movement framework is a natural design choice; the control token formulation enables a single model to simultaneously support object and light source editing.
- The physically decoupled rendering pipeline exemplifies best practices in data engineering: by separating ambient and direct light, it generates unlimited variants in post-processing.

## Limitations & Future Work

- The current method supports manipulation of visible light sources only; control over invisible light sources (e.g., natural light from outside windows) remains limited.
- Synthetic data covers only 25 indoor scenes; scene diversity can be further improved.
- Modeling of long-range light transport effects (e.g., caustics, subsurface scattering) during light source movement may be insufficiently accurate.
- The method supports single-image input only and has not yet been extended to light source manipulation in video sequences.

## Related Work & Insights

- **vs. ObjectMover**: LightMover extends ObjectMover's token framework by adding lighting-specific control signals, and also outperforms it on object movement tasks.
- **vs. LightLab**: LightLab supports light tone and on/off control but not spatial movement; LightMover is the first to achieve precise control over light source position.
- **vs. Gemini-2.5-Flash-Image**: General-purpose LLM-based editors lack illumination parameterization and are noticeably inferior to LightMover in terms of physical consistency in light propagation.
- **Insights**: The approach of encoding non-spatial attributes (color, intensity) as frame tokens is worth borrowing for other controllable generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First to unify spatial light source movement with color/intensity control in a video diffusion framework; adaptive token pruning is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers light movement/color/intensity/joint control/object movement/insertion/removal with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, physical modeling equations are rigorous, and pipeline visualizations are of high quality.
- Value: ⭐⭐⭐⭐ Has direct application value in photo post-processing and virtual scene production, advancing research in fine-grained lighting control.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[ICLR 2026\] Arbitrary Generative Video Interpolation](../../ICLR2026/video_generation/arbitrary_generative_video_interpolation.md)
- [\[ICLR 2026\] MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)

<!-- RELATED:END -->
