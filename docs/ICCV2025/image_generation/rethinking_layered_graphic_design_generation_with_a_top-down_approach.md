---
title: >-
  [Paper Note] Rethinking Layered Graphic Design Generation with a Top-Down Approach
description: >-
  [Image Generation] This paper proposes Accordion, a top-down framework that converts AI-generated rasterized design images into editable layered designs (comprising background, foreground object, and vectorized text layers), where a VLM plays distinct roles across three stages: reference creation, design planning, and layer generation.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 0e9ed3c8ead3f908
---

# Rethinking Layered Graphic Design Generation with a Top-Down Approach

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2507.05601](https://arxiv.org/abs/2507.05601)
- **Code**: Not released
- **Area**: Diffusion Models · Graphic Design Generation
- **Keywords**: Layered design, VLM, top-down, text rendering, design automation

## TL;DR
This paper proposes Accordion, a top-down framework that converts AI-generated rasterized design images into editable layered designs (comprising background, foreground object, and vectorized text layers), where a VLM plays distinct roles across three stages: reference creation, design planning, and layer generation.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Graphic designs (posters, advertisements, etc.) are inherently layered — background layers, foreground object layers, and vectorized text layers. Generative AI models can produce visually appealing rasterized designs, yet lack editability (e.g., text cannot be modified, elements cannot be separated).

Existing methods adopt a **bottom-up** strategy (e.g., COLE sequentially generates the background, adds objects, and finally places text), leading to:
1. Lack of global coordination among visual elements — elements added later may conflict with earlier decisions
2. Text may occupy excessive space or overlap with foreground objects
3. No global visual reference serves as an overall design blueprint

**Core Insight**: Human designers typically consult existing designs for inspiration (color, layout, typographic style) before creating layers — this is the **top-down** strategy. This paper represents the first attempt to convert AI-generated rasterized design images into editable layered designs.

## Method

### Overall Architecture: Three-Stage VLM Pipeline

Accordion is built around a VLM (LLaVA-1.5-7B), which assumes different roles across three stages:

### Stage 1: Reference Creation

VLM role: **Prompt Enhancer**
- Input: user's short intent $I$ or sketch draft $S$
- The VLM expands the brief description into a detailed prompt $P_{des}$ via in-context learning
- Fed into a T2I model (Flux) to generate a reference image $R$

### Stage 2: Design Planning

VLM role: **Design Planner**
- Input: reference image $R$ and composite prompt $P = P_{task} + P_{des} + P_{ocr}$
- The VLM outputs an ordered dictionary sequence $\{D_*\}$: each entry contains a bounding box and attributes
- Text attributes include content, color, font, alignment, line count, angle, etc.
- Key capability: correcting meaningless AI-generated text into semantically valid content

### Stage 3: Layer Generation

VLM role: **Quality Selector**
- Layers are extracted sequentially according to the plan:
  1. Text removal: text regions are removed conditioned on text bounding boxes
  2. Object extraction: SAM extracts foreground objects → inpainting model fills the background
  3. Result selection: the VLM evaluates multiple candidates and selects the best
- Final composition: background $B$ + object layers $\{O_n\}$ + vectorized text layer $T$

### Training Data Construction

Three types of references are used to fine-tune the VLM:
1. **Original designs**: parsed directly to train text de-rendering capability
2. **Designs with meaningless text**: text regions are corrupted using an SD1.5 inpainting model (strength 0.5–0.7) to train repair capability
3. **Text-free backgrounds**: all text removed to train the ability to add text from the background

A total of 156,932 training samples (39K × 3 reference types + questionnaire dataset).

## Key Experimental Results

### Main Results: Quantitative Comparison on the DesignIntention Benchmark

| Method | Design & Layout | Content Relevance | Typography & Color | Graphics & Imagery | Creativity | Average |
|--------|----------------|-------------------|--------------------|--------------------|------------|---------|
| COLE | 6.0 | 6.9 | 5.7 | 6.2 | 5.1 | 6.0 |
| Open-COLE | 6.3 | 7.0 | 5.6 | 7.1 | 5.3 | 6.3 |
| **Accordion** | **6.7** | **7.4** | **6.1** | **7.3** | 5.1 | **6.5** |

Accordion achieves an average score of 6.5, surpassing COLE (6.0) and Open-COLE (6.3), while using only 39K training samples — far fewer than COLE's 100K.

### Designer User Study (29 designers, 30 cases)

| Evaluation Dimension | Accordion Preferred over COLE (%) |
|----------------------|-----------------------------------|
| Text-to-template editability | 73.5% |
| Sketch-to-description reasonableness | 87.2% |

### Key Findings

1. Accordion produces an average text length of 61.7 characters vs. COLE's 42.3 (1.5×) — indicating more effective use of available space
2. Aesthetic score of 4.98 vs. COLE's 4.72 — global reference ensures visual harmony among elements
3. Layer count detection MAE: 0.494 for text layers, 0.274 for object layers — indicating high layering accuracy
4. VLM questionnaire-based selection improves text removal PSNR from 18.01 to 21.18

## Highlights & Insights

1. **Paradigm shift**: top-down vs. bottom-up — establishing a global reference before layered extraction avoids incremental visual conflicts
2. **Triple-role VLM**: a single VLM serves as prompt enhancer, design planner, and quality selector across different stages
3. **Model agnosticism**: any T2I model (Flux / SD3 / future models) can be plugged in as the reference source without retraining
4. **Design variants**: supports creative exploration via upstream model variants, inference-time variants, and downstream model variants

## Limitations & Future Work

- SAM achieves an IoU of only 68.4%, with difficulties in extracting transparent or hollow objects
- The framework assumes the text layer always resides above object layers, precluding more complex layer hierarchies
- The text layer supports only 2,000 predefined styles, with no support for freeform text or special-effect typography
- Inference time of 36.7 seconds per sample leaves room for improving interactive usability

## Related Work & Insights

- **Layered design generation**: COLE, Open-COLE, De-Render
- **Text rendering**: TextDiffuser, TextDiffuser-2
- **VLM applications**: LLaVA, GPT-4V applied to design evaluation

## Rating
- Novelty: ★★★★☆ — Top-down strategy and first attempt at converting AI-generated designs into editable layered formats
- Technical Depth: ★★★★☆ — Elegantly designed three-stage pipeline with thoughtful training data construction
- Practicality: ★★★★★ — Directly addresses real-world design needs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)

</div>

<!-- RELATED:END -->
