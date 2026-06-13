---
title: >-
  [Paper Note] SynCity: Training-Free Generation of 3D Worlds
description: >-
  [ICCV 2025][LLM Pretraining][3D world generation] SynCity proposes a training- and optimization-free method for 3D world generation. Through carefully designed prompt engineering strategies…
tags:
  - "ICCV 2025"
  - "LLM Pretraining"
  - "3D world generation"
  - "training-free"
  - "tile-based"
  - "2D/3D generators"
  - "prompt engineering"
date: 2026-05-08
content_hash: 601c0344c280cce5
---

# SynCity: Training-Free Generation of 3D Worlds

**Conference**: ICCV 2025
**arXiv**: [2503.16420](https://arxiv.org/abs/2503.16420)  
**Code**: None  
**Area**: LLM Pretraining
**Keywords**: 3D world generation, training-free, tile-based, 2D/3D generators, prompt engineering

## TL;DR

SynCity proposes a training- and optimization-free method for 3D world generation. Through carefully designed prompt engineering strategies, it combines a pretrained language model, a 2D image generator (Flux), and a 3D generator (TRELLIS) to autoregressively synthesize large-scale, high-quality, freely navigable 3D scenes in a tile-by-tile fashion.

## Background & Motivation

- **3D content generation** has broad applications (games, VR, visual effects, simulation), yet manually creating 3D scenes is time-consuming and labor-intensive; automated generation can substantially reduce this burden.
- **Existing 3D generation models** (e.g., TRELLIS) mostly focus on single-object generation and cannot directly produce large-scale scenes.
- **Image-based scene extension methods** (e.g., the DreamFusion family, Text2Room) can leverage the artistic quality of 2D image generators but struggle to maintain 3D geometric consistency across large regions, typically producing only "3D bubbles" that cannot be truly walked through.
- **Direct 3D scene generation methods** (e.g., BlockFusion, LT3SD) can generate spatially coherent large environments but are limited in diversity and quality due to the scarcity of 3D training data.
- **Core motivation**: Can the geometric precision of 3D generation models be combined with the artistic expressiveness of 2D image generators without any retraining?

## Method

### Overall Architecture

SynCity organizes the world as a grid of $W \times H$ square tiles and generates the world tile by tile. The pipeline consists of four steps:
1. **Language prompting**: expand a high-level text description into per-tile prompts.
2. **2D image generation**: generate a 2D image for each tile using an isometric prompting strategy.
3. **3D reconstruction**: feed the 2D image into an image-to-3D model to obtain a 3D representation.
4. **3D blending**: merge adjacent tile boundaries in the TRELLIS latent space.

### Key Designs

1. **LLM Prompting**:

    - The high-level world description $p_0$ is fed into ChatGPT o3-mini-high to produce a grid-structured world layout.
    - The output includes an individual text prompt $p_{xy}$ for each tile and a global style prompt $p_\star$.
    - Both manual control and automatic generation modes are supported.

2. **Isometric Tile Inpainting**:

    - Core idea: a gray isometric base plate $B$ and a cuboid mask $M$ are used as conditions to guide Flux ControlNet in generating regular isometric-view tile images.
    - For subsequent tiles ($x,y > 0$), the already-generated 3D world is rendered as a context image, and new tiles are generated via inpainting within the existing scene context.
    - Tall structures that might occlude the new tile are trimmed to avoid visual obstruction.
    - **This is purely prompt engineering — no model fine-tuning is required.**

3. **3D Generator Prompting (Rebasing + TRELLIS)**:

    - The new tile region is extracted from the 2D image using rembg and alpha matting for background removal.
    - **Rebasing**: a slightly larger gray base plate is placed beneath the extracted tile image to provide a "frame" for the 3D generator, ensuring the reconstructed geometry is regular, square, and complete at the bottom.
    - The processed image is fed into TRELLIS to obtain 3D Gaussian Splats.
    - Geometric validation is performed (checking whether the base is square and complete); tiles that fail are regenerated with a different random seed.
    - Post-processing: crop the base plate, rescale to unit size, and re-orient.

4. **3D Blending (Latent Space Blending)**:

    - The 3D latent representations $\gamma^1, \gamma^2$ of two adjacent tiles are concatenated into a joint volume $\gamma$.
    - In the second stage of TRELLIS ($R=64$), the boundary region $|x - R/2| \leq r$ is re-denoised while all other regions are held fixed.
    - At the 2D level, a front-view render of the adjacent tiles is inpainted to produce a blending reference image.
    - A sparse latent upsampling scheme is proposed: the occupancy volume is upsampled first, then a new latent is denoised conditioned on multi-view inputs, avoiding artifacts from naive interpolation.

### Loss & Training

This method is **training-free** and involves no training or loss functions. All components (LLM, Flux, TRELLIS) use pretrained weights; the contribution lies entirely in the design of the prompting and post-processing strategies.

## Key Experimental Results

### Main Results

| Evaluation Dimension | SynCity Win Rate (%) |
|---|---|
| Overall | **90.9** |
| Geometry | **81.8** |
| Exploration | **90.9** |
| Diversity | **90.9** |
| Realism | **86.4** |

> Human preference study against BlockFusion ($n=22$); SynCity leads comprehensively on all dimensions.

### Ablation Study

| Method | Base Area (voxel) | Squareness ↑ | Completeness ↑ |
|---|---|---|---|
| No Rebasing | 2271 | 0.92 | 0.73 |
| **Ours** | **4096** | **1.00** | **1.00** |

| Latent Upsampling Method | LPIPS ↓ | SSIM ↑ | FID ↓ | KID ↓ |
|---|---|---|---|---|
| Naive interpolation | 0.5914 | 0.3093 | 200.5 | 0.243 |
| Ours (single frame) | 0.3517 | 0.5149 | 111.6 | 0.069 |
| **Ours (multi-frame)** | **0.3212** | **0.5312** | **89.1** | **0.051** |

### Key Findings

- **Isometric prompting** is critical for guiding the 2D generator to produce regular tiles; without it, viewpoints are arbitrary and unsuitable for 3D reconstruction.
- **Context-aware generation** ensures consistent object scale across adjacent tiles; removing it leads to noticeable scale inconsistencies.
- **Rebasing** ensures square and complete tile geometry, which is a prerequisite for reliable stitching.
- **3D blending** effectively eliminates discontinuities at tile boundaries.
- **Multi-frame conditional upsampling** significantly outperforms naive interpolation on all perceptual metrics.

## Highlights & Insights

- **Entirely training-free** is the most distinctive aspect: through carefully crafted prompt engineering, the method chains off-the-shelf LLM, 2D, and 3D generators without any fine-tuning — an elegant engineering approach.
- The isometric perspective, common in games, is cleverly exploited as it falls within the distribution of pretrained models.
- Performing 3D blending in TRELLIS's latent space is more principled than pixel-space fusion.
- The work demonstrates that, even without training a scene-level 3D generation model, combining object-level 3D models with 2D generators suffices to produce navigable large-scale scenes.
- Generated scenes are not confined to a single "3D bubble" and support non-trivial free navigation.

## Limitations & Future Work

- Generation quality is bounded by the capabilities of the underlying models (TRELLIS, Flux).
- The regular grid structure of tiles limits layout flexibility; random tile offsets and scaling could be explored in future work.
- A coarse-to-fine modeling strategy could be adopted to improve global coherence.
- If scene-level 3D training data were available, fine-tuning select components could further improve quality.
- Global consistency currently relies on local context and may drift in very large grids.

## Related Work & Insights

- **BlockFusion**: learns to autoregressively diffuse 3D mesh blocks but requires domain-specific 3D training data and produces limited diversity.
- **LT3SD**: generates 3D environments patch-by-patch in a coarse-to-fine manner but is restricted to indoor scenes.
- **TRELLIS**: the core 3D generator in this work; although designed for object-level generation, it can handle local multi-object compositions.
- **Flux ControlNet Inpainting**: the primary 2D generation backbone, enabling regular tile generation via isometric prompting.
- **Insight**: pretrained models can be composed to solve complex problems without training new ones — "composition as innovation."

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Extending an object-level 3D generator to scene-level generation via prompt engineering is a distinctive and practical idea.
- **Experimental Thoroughness**: ⭐⭐⭐ — The human preference study is convincing but small-scale ($n=22$); quantitative comparisons against more baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear illustrations, well-structured method description.
- **Value**: ⭐⭐⭐⭐ — Demonstrates the viability of training-free approaches for 3D world generation with significant practical and inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](../../NeurIPS2025/llm_pretraining/through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)

</div>

<!-- RELATED:END -->
