---
title: >-
  [Paper Note] ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop
description: >-
  [CVPR 2026][Image Generation][Expression Editing] This paper presents ExpressEdit, a fully open-source Photoshop plugin that achieves noise-free editing of stylized facial expressions within 3 seconds on a single consume…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Expression Editing"
  - "Photoshop Plugin"
  - "Diffusion Models"
  - "Stylized Characters"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 51ce8a2de86ccf23
---

# ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop

**Conference**: CVPR 2026
**arXiv**: [2604.03448](https://arxiv.org/abs/2604.03448)  
**Code**: [https://github.com/kenantang/ExpressEdit](https://github.com/kenantang/ExpressEdit)  
**Area**: Diffusion Models
**Keywords**: Expression Editing, Photoshop Plugin, Diffusion Models, Stylized Characters, Retrieval-Augmented Generation

## TL;DR

This paper presents ExpressEdit, a fully open-source Photoshop plugin that achieves noise-free editing of stylized facial expressions within 3 seconds on a single consumer-grade GPU, leveraging a SPICE-based diffusion model backend combined with a Danbooru expression tag database and a RAG system, significantly outperforming commercial models such as GPT, Grok, and Nano Banana 2.

## Background & Motivation

1. **Background**: Facial expressions are a core element of visual storytelling. Current AI image editing models (e.g., FLUX.2, GPT, Grok, Nano Banana 2) can assist with expression generation and editing, but are primarily designed for photorealistic faces with insufficient support for stylized expressions in 2D/3D animated characters.
2. **Limitations of Prior Work**:
    - High demands on textual descriptions — users must provide detailed expression prompts; otherwise, generated results lack diversity, imposing significant cognitive burden.
    - Commercial models introduce global noise and watermarks during editing; noise accumulates over multiple editing steps, leading to severe image degradation.
    - Poor integration with professional software such as Photoshop, causing resolution changes and pixel drift.
3. **Key Challenge**: Existing models cannot simultaneously maintain image quality and precisely control the size and position of facial elements, and inference speed is slow (mostly 7–50 seconds).
4. **Goal**: To achieve fast, noise-free, and iterative stylized expression editing within professional editing software.
5. **Key Insight**: Leveraging the open-source diffusion model SPICE as the backend, combined with native Photoshop operations (Liquify, Selection, Scale) for precise spatial control, while constructing a database of 135 expression tags to lower the barrier to use.
6. **Core Idea**: Deeply integrating an open-source diffusion model backend with native Photoshop operations — using Canny edge control to eliminate pixel drift and a RAG-based expression tag system to enable fast, lossless stylized expression editing.

## Method

### Overall Architecture

ExpressEdit consists of two pipelines: (1) a Retrieval-Augmented Prompt Generator that converts user intent (story/instruction) into structured expression tag prompts; and (2) an Expression Editor that accepts prompts and user-provided images/transformations/selections, generates edited results via the SPICE diffusion model backend, and returns them as a new layer in Photoshop.

### Key Designs

1. **Retrieval-Augmented Prompt Generator (RAG)**:
    - **Function**: Converts free-text user intent into tag-format prompts compatible with diffusion models.
    - **Mechanism**: Constructs a multimodal database of 135 Danbooru expression tags, each accompanied by a definition, 3,375 example images, 332 alternative tags (in Chinese, Japanese, Korean, and English), and 2,700 short stories. Upon user input of a story, a VLM retrieves relevant tags from the database and inserts them into a prompt template composed of a prefix (image content) and suffix (style control).
    - **Design Motivation**: Tag-format prompts differ fundamentally from natural language; directly inputting text is challenging for new users. The RAG system bridges the gap between large and small models, enabling use even with limited computational resources.

2. **SPICE-Based Noise-Free Editing Backend**:
    - **Function**: Performs clean expression editing within the user-defined selection without affecting pixels outside it.
    - **Mechanism**: Uses WAI-illustrious-SDXL as the base model, paired with SDXL Canny ControlNet. The SPICE backend employs explicit Canny edge control to ensure precise alignment between edited region edges and the original image, eliminating pixel drift. Users draw selections at full hardness using the Selection Brush; the model modifies only the selected content.
    - **Design Motivation**: Even commercial models that offer selection functionality produce edge mismatches after editing (e.g., Nano Banana Pro generates artifacts at earlobes and chin). SPICE fundamentally resolves this issue via Canny edge constraints.

3. **Photoshop Native Operation Synergy**:
    - **Function**: Leverages native Photoshop operations such as Liquify, Scale, and Quick Selection to provide precise spatial control.
    - **Mechanism**: Users first roughly reposition facial elements via Liquify or adjust iris size via Scale; the diffusion model then corrects the resulting distortion artifacts into natural outputs. This eliminates the need to specify directional or numerical instructions in text, avoiding the insensitivity of multimodal models to spatial commands.
    - **Design Motivation**: All baseline models fail to respond to precise numerical descriptions (e.g., "reduce iris diameter by 50%"), whereas SPICE correctly interprets and executes such instructions when Photoshop transformations are provided as visual prompts.

### Loss & Training

ExpressEdit performs inference directly using the pretrained SPICE backend without additional training. A Speed-Up LoRA is supported to reduce sampling steps from 30 to 8, lowering API latency from 4.06 seconds to 2.18 seconds.

## Key Experimental Results

### Main Results

| Method | Latency (s) | Noise Introduction | Selection Support | Iterative Editing |
|---|---|---|---|---|
| FLUX.2 [max] | 49.94±13.39 | Severe | None | Degradation |
| GPT | 46.01±11.74 | Severe | None | Degradation |
| Grok | 7.11±0.50 | Moderate | None | Degradation |
| Nano Banana 2 Fast | 23.18±3.92 | Severe (diagonal pattern) | Yes (edge artifacts) | Degradation |
| Nano Banana 2 Pro | 41.92±22.08 | Severe (diagonal pattern) | Yes (edge artifacts) | Corrupted after 8 steps |
| **ExpressEdit (30 steps)** | **4.06±0.02** | **None** | **Native** | **Stable over 100 steps** |
| **ExpressEdit (8 steps + LoRA)** | **2.18±0.02** | **None** | **Native** | **Stable over 100 steps** |

### Functionality Comparison

| Feature | ExpressEdit | Baseline Models |
|---|---|---|
| Precise numerical control (iris scale 50%) | ✓ (via Scale transform) | ✗ (all fail) |
| Directional control (gaze shift) | ✓ (via Liquify) | Requires text description; unreliable |
| High-resolution support (1664×2432) | ✓ | Nano Banana 2 Pro exhibits degradation |
| Multi-tag combination | ✓ ("+_+" + ":O") | Requires complex text descriptions |
| Open-source and free | ✓ | ✗ |

### Key Findings

- Nano Banana 2 Pro's output is completely corrupted by noise after 8 consecutive editing steps, whereas ExpressEdit exhibits only minor noise at selection boundaries even under a stress test of 100 strictly overlapping selections — and this is recoverable in a single step.
- 35 of the 135 expression tags require Photoshop transform assistance for reliable editing; the remaining 100 can be edited directly without transformations.
- The Speed-Up LoRA reduces latency by 46%, with only minimal differences in fine details such as eyelashes.

## Highlights & Insights

- **Photoshop Native Operations as Visual Prompts**: The approach cleverly uses the "coarse results" of Liquify/Scale transformations as spatial prompts for the diffusion model, circumventing the fundamental insensitivity of text instructions to spatial direction. This idea is transferable to any image editing task requiring precise spatial control.
- **Selection as Control**: By restricting the diffusion model to edit only within the selection and using Canny edge control to eliminate pixel drift, the system achieves truly non-destructive editing — a far more reliable approach than attempting to describe unedited regions in text prompts.
- **Systematic Construction of the Expression Tag Database**: 135 tags, 3,375 example images, 2,700 stories, and multilingual alternative tags together form a comprehensive knowledge base for expression editing.

## Limitations & Future Work

- Currently limited to Photoshop; although the code is open-source, general users may lack a Photoshop license.
- No quantitative evaluation of RAG retrieval accuracy is provided; user study data on tag retrieval quality is absent.
- Generalization to extreme styles (e.g., ink wash painting, pixel art) has not been validated.
- Expression tags are sourced from Danbooru, which may introduce dataset bias; performance on non-East-Asian-style characters has not been thoroughly tested.

## Related Work & Insights

- **vs. FLUX.2/GPT/Grok**: These general-purpose editing models are capable but introduce global noise, making them unsuitable for iterative editing workflows. ExpressEdit addresses this through selection-based control and Canny constraints.
- **vs. Nano Banana 2**: The closest competitor — faster than most baselines but still subject to diagonal noise patterns and pixel drift. ExpressEdit is faster and entirely noise-free.
- **vs. SPICE**: The core backend of ExpressEdit. The contribution of this paper lies in engineering SPICE into a complete Photoshop plugin and building the surrounding expression tag ecosystem.

## Rating

- **Novelty**: ⭐⭐⭐ — No major technical innovation in methodology; the primary contribution is engineering integration and system design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparison against multiple commercial models, though quantitative user studies are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear argumentation with rich figures and tables, though the paper is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ — Open-source and practical, with direct applicability to the animation and gaming industries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](../../AAAI2026/image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2026\] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)

</div>

<!-- RELATED:END -->
