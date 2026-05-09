---
title: >-
  [Paper Note] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing
description: >-
  [ICCV 2025][3D Vision][Unified generative model] This paper proposes UniVG, a unified image generation model built on MM-DiT that supports T2I generation, editing, identity-preserving generation, layout-guided synthesis, depth estimation, and more within a single set of weights, achieved via channel-wise input concatenation, progressive multi-task training, and external condition injection.
tags:
  - ICCV 2025
  - 3D Vision
  - Unified generative model
  - MM-DiT
  - multi-task training
  - instruction-based editing
  - Flow Matching
date: 2026-05-08
content_hash: 123dea3dba4ccc29
---

# UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing

**Conference**: ICCV 2025
**arXiv**: [2503.12652](https://arxiv.org/abs/2503.12652)
**Code**: Unavailable
**Area**: 3D Vision
**Keywords**: Unified generative model, MM-DiT, multi-task training, instruction-based editing, Flow Matching

## TL;DR

This paper proposes UniVG, a unified image generation model built on MM-DiT that supports T2I generation, editing, identity-preserving generation, layout-guided synthesis, depth estimation, and more within a single set of weights, achieved via channel-wise input concatenation, progressive multi-task training, and external condition injection.

## Background & Motivation

Diffusion models have achieved remarkable progress across T2I generation, editing, personalization, and related tasks, yet this progress has led to **model fragmentation**: each task typically requires its own architecture, training pipeline, and parameter set.

Existing unified models (OmniGen, OneDiffusion) have demonstrated feasibility but exhibit notable shortcomings:

1. **OmniGen**: concatenates input images along the sequence dimension, resulting in low training and inference efficiency.
2. Both models **lack rigorous ablation studies** — optimal data mixing ratios and training strategies remain unclear.
3. The synergistic and conflicting relationships among tasks have not been thoroughly explored.

**Core Problem**: How can a model unify diverse image generation tasks without sacrificing core T2I capability?

## Method

### Minimal Architectural Modification

Input image latents, noisy latents, and masks are concatenated along the **channel dimension** (rather than the sequence dimension):

$$d = [z_t \oplus \text{VAE}_{Enc}(\mathcal{V}) \oplus \text{Resize}(\mathcal{M})]$$

This yields a fixed-length sequence, avoiding the variable-length inefficiencies and context-awareness disruptions caused by OmniGen's sequence-wise concatenation.

### Condition Injection

External conditions (e.g., facial features) are injected via **embedding replacement**: condition features $f = \mathcal{H}(\mathcal{C})$ are extracted and used to replace the embeddings of pre-designated placeholder tokens.

### Multi-Task Training

Flow Matching loss:
$$\mathcal{L} = \mathbb{E}[\|\mathcal{F}([\{p\}, t, d]) - u_t\|^2]$$

**Three-stage training recipe**:
1. **Base training**: T2I training from scratch (400K steps, bs=512).
2. **Multi-task training**: introduces editing (47%), inpainting/completion (20%), auxiliary tasks (3%), layout guidance (2%), etc. (400K steps).
3. **Further fine-tuning**: adds identity-preserving generation (1:1 ratio, 40K steps).

### Task-Specific Input Formats

- **T2I**: blank image + all-white mask + `<t2i>`
- **Instruction editing**: input image + blank mask + `<ie>` instruction
- **Depth estimation**: input image + all-white mask + `<depth>`
- **Layout guidance**: layout visualization + all-white mask + `<lg>` + description
- **Identity-preserving**: CLIP face embeddings replace the `<p>` placeholder

### Dual CFG Inference

$$\mathcal{F} \implies \mathcal{F}(\varnothing,t,\{z_t,\varnothing,\varnothing\}) + \alpha_\mathcal{V}(\cdot) + \alpha_\mathcal{X}(\cdot)$$

Separate guidance scales control image conditioning and text conditioning respectively.

## Key Experimental Results

### Text-to-Image Generation

| Method | Params | GenEval↑ | CompBench↑ | DSG↑ | HPSv2↑ |
|--------|--------|----------|------------|------|--------|
| SDXL | 2.6B | 0.55 | 0.42 | 0.72 | 27.7 |
| FLUX.1 | 12.0B | 0.66 | 0.47 | 0.73 | 29.2 |
| SD3 | 8.0B | 0.71 | 0.49 | 0.76 | 28.9 |
| OmniGen | 3.8B | 0.70 | 0.46 | 0.66 | 27.7 |
| **UniVG** | **3.7B** | **0.70** | **0.48** | **0.75** | **28.2** |

### Instruction Editing (CLIP-T / CLIP-I)

| Method | MagicBrush CLIP-T↑ | MagicBrush CLIP-I↑ | EmuEdit CLIP-T↑ |
|--------|-------------------|-------------------|----------------|
| InsP2P | 24.5 | 83.7 | 21.9 |
| EmuEdit | 26.1 | 89.7 | 23.1 |
| OmniGen | 25.8 | 86.3 | 23.1 |
| **UniVG** | **29.5** | 86.3 | **25.9** |

### Key Findings

1. **T2I and editing can coexist** — multi-task training does not degrade core T2I capability, achieving a GenEval score of 0.70.
2. **Auxiliary tasks enhance editing** — depth estimation and segmentation improve spatial understanding for image editing.
3. UniVG substantially outperforms all baselines (including task-specific models) on CLIP-T, demonstrating superior instruction-following ability in editing.
4. With fewer parameters (3.7B) than OmniGen (3.8B), UniVG achieves better performance, reflecting architectural efficiency.

## Highlights & Insights

1. **Channel-wise vs. sequence-wise concatenation** — concatenation along the channel dimension is key to efficiency, preserving fixed-length sequences.
2. **Progressive training** — establishing T2I foundations first, then introducing multi-task objectives, and finally identity-preserving generation mitigates catastrophic forgetting.
3. **Discovery of inter-task synergies** — auxiliary perception tasks (depth/segmentation) demonstrably benefit generative and editing tasks.
4. **Placeholder token-based condition injection** — maintains constant sequence length while flexibly accommodating diverse condition types.

## Limitations & Future Work

- Identity-preserving generation requires a separate Stage 3 of training, which can conflict with other tasks.
- Insufficient normal estimation quality limits its contribution as an auxiliary task.
- The model is not open-sourced, making independent verification difficult.

## Related Work & Insights

- **T2I models**: Stable Diffusion, FLUX.1, SD3
- **Unified models**: OmniGen, OneDiffusion, TransFusion
- **Editing models**: InstructPix2Pix, EmuEdit, MGIE

## Rating

- Novelty: ⭐⭐⭐⭐ (unified modeling is not entirely new, but the training strategy analysis is thorough)
- Technical depth: ⭐⭐⭐⭐ (comprehensive ablation studies and training recipe)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (extensive multi-task evaluation)
- Value: ⭐⭐⭐⭐ (single model for multiple tasks, deployment-friendly)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)

<!-- RELATED:END -->
