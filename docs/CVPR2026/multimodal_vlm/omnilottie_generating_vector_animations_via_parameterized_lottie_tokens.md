---
title: >-
  [Paper Note] OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens
description: >-
  [CVPR 2026][Multimodal VLM][Lottie] OmniLottie proposes a Lottie Tokenizer that converts Lottie JSON files into structured command-parameter sequences…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Lottie"
  - "vector animation"
  - "tokenization"
  - "multimodal instruction"
  - "VLM generation"
date: 2026-05-08
content_hash: b0fea873d3dafc85
---

# OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens

**Conference**: CVPR 2026
**arXiv**: [2603.02138](https://arxiv.org/abs/2603.02138)
**Code**: To be confirmed (paper references a Project Page)
**Area**: Multimodal VLM / Vector Animation Generation
**Keywords**: Lottie, vector animation, tokenization, multimodal instruction, VLM generation

## TL;DR
OmniLottie proposes a Lottie Tokenizer that converts Lottie JSON files into structured command-parameter sequences, enabling pretrained VLMs to generate high-quality vector animations from multimodal cross-modal instructions. The work also introduces the MMLottie-2M large-scale dataset to support training.

## Background & Motivation

### State of the Field
Vector animations (e.g., SVG animations, Lottie format) are widely used in UI design, mobile applications, and web development. They are compact, resolution-independent, and programmatically editable. However, automated vector animation generation remains largely unexplored—existing work primarily focuses on static vector graphics or pixel-level video generation.

### Limitations of Prior Work

**Redundancy in Lottie JSON**: Raw Lottie files contain large amounts of invariant structural metadata and format tokens (e.g., brackets, key names), which constitute significant noise for learning animation generation.

**Lack of training data**: No large-scale paired dataset of vector animations and text descriptions exists.

**VLMs cannot understand animation formats**: Existing VLMs generate only text or images and cannot directly output structured animation representations.

### Root Cause
Lottie is the most popular vector animation format, yet its JSON representation is unfriendly to machine learning—redundant format tokens cause sequence lengths to explode, making it difficult to learn effective generative models.

### Core Idea
Design a **Lottie Tokenizer** that converts Lottie JSON into compact command-plus-parameter sequences (eliminating all structural redundancy), enabling pretrained VLMs to autoregressively generate vector animations in the same manner as natural language generation.

## Method

### Overall Architecture
OmniLottie comprises three core components:
1. **Lottie Tokenizer**: JSON → command-parameter sequences
2. **OmniLottie Model**: Built upon a pretrained VLM, accepts multimodal instruction inputs, and autoregressively generates Lottie token sequences
3. **MMLottie-2M Dataset**: Large-scale vector animation corpus with text and visual annotations

### Key Designs

#### 1. Lottie Tokenizer
- **Function**: Converts Lottie JSON files into structured command-parameter sequences
- **Mechanism**: Traverses the JSON tree structure and extracts three categories of meaningful information:
    - **Shape Commands**: Geometric instructions such as `MOVE_TO(x, y)` and `BEZIER(cx1, cy1, cx2, cy2, x, y)`
    - **Animation Functions**: Keyframe interpolation descriptors such as `EASE_IN(start_frame, end_frame, start_val, end_val)`
    - **Control Parameters**: Color, opacity, transformation matrices, etc.
- **Design Motivation**: All redundant JSON formatting (indentation, brackets, key names) is eliminated, compressing sequence length to approximately 15–20% of the original
- **Novelty**: Methods that train directly on raw JSON must handle sequences of ~10k+ tokens; the Lottie Tokenizer reduces this to ~1–2k tokens

#### 2. OmniLottie Model Architecture
- **Function**: Extends a pretrained VLM (e.g., LLaVA) by augmenting its vocabulary with Lottie command tokens, enabling autoregressive generation of Lottie sequences from multimodal instructions
- **Mechanism**:
    - Approximately 200 Lottie-specific tokens (shape commands, animation function names, etc.) are added to the VLM vocabulary
    - Parameter values (coordinates, colors, etc.) are represented as quantized numeric tokens
    - Standard next-token prediction loss is used during training
- **Design Motivation**: Leverages the language and visual understanding capabilities of pretrained VLMs, casting vector animation generation as a sequence generation problem
- **Multimodal Support**: Accepts diverse inputs including text descriptions (e.g., "draw a bouncing ball"), reference images, and sketches

#### 3. MMLottie-2M Dataset
- **Function**: A large-scale vector animation dataset containing 2 million professionally designed vector animations
- **Mechanism**: Professionally designed Lottie animations are collected from platforms such as LottieFiles; VLMs are used to automatically generate text descriptions and visual annotations
- **Scale**: 2 million animations with text descriptions and visual annotations (keyframe screenshots)

## Key Experimental Results

### Main Results: Vector Animation Generation Quality

| Method | FID ↓ | CLIP Score ↑ | Human Preference (%) |
|--------|-------|-------------|----------------------|
| DeepSVG + Motion | 142.3 | 0.21 | 12.3 |
| SVGDreamer | 98.7 | 0.28 | 22.8 |
| AnimateDiff (pixel) | 45.2 | 0.35 | 28.4 |
| **OmniLottie** | **38.6** | **0.41** | **36.5** |

### Ablation Study

| Configuration | CLIP Score ↑ | Notes |
|---------------|-------------|-------|
| Full OmniLottie | 0.41 | Complete method |
| w/o Lottie Tokenizer (raw JSON) | 0.24 | Raw JSON text; overly long sequences degrade quality |
| w/o Animation Functions | 0.33 | Only static shapes generated; no animation |
| w/o MMLottie Pretrain | 0.31 | No large-scale dataset pretraining |

### Key Findings
- **The Lottie Tokenizer is the critical component**—removing it drops CLIP Score from 0.41 to 0.24, as raw JSON is too verbose for the model to learn effectively
- Generated vector animations play smoothly on mobile devices at roughly 1/100th the file size of pixel-based video
- The flexibility of multimodal instruction is validated—text, images, and sketches all yield semantically aligned animations
- The model can generate complex scenes involving multiple objects and multi-layer animations

## Highlights & Insights
- **Casting vector animation generation as sequence generation**—the Lottie Tokenizer design elegantly aligns this seemingly unconventional task with the LLM paradigm
- **MMLottie-2M fills a critical data gap**—a professionally designed vector animation dataset at the 2-million scale is a valuable community resource
- **High practical utility**—generated Lottie files can be directly integrated into app and web development without post-processing
- **Broader implications for structured format design**—the Lottie Tokenizer approach is generalizable to other structured format generation tasks (e.g., CAD, SVG, code ASTs)

## Limitations & Future Work
- Currently limited to the Lottie format; extension to SVG animation or CSS animation is not addressed
- Generation quality for complex animations (e.g., those involving masks, blend modes, or expressions) requires further improvement
- Parameter value quantization introduces precision loss—subtle animation curves may be coarsened
- Automatic evaluation metrics for animation temporal quality are lacking—FID and CLIP Score primarily assess static frames
- The model does not support interactive editing of generated animations

## Related Work & Insights
- **vs. DeepSVG**: DeepSVG targets static vector graphic generation via VAEs and does not support animation. OmniLottie is specifically designed for animation dynamics.
- **vs. AnimateDiff**: AnimateDiff generates pixel-level video. OmniLottie produces vector format output that is compact and editable.
- **vs. SVGDreamer**: SVGDreamer uses diffusion models to generate SVGs but does not support animation or multimodal input.
- **Insight**: Tokenization of structured formats serves as a key bridging mechanism for integrating traditional design tools with AI generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to model vector animation generation as sequence generation; the Lottie Tokenizer design is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Human evaluation + automatic metrics + ablation study; animation temporal quality evaluation is absent
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; tokenizer design visualizations are well executed
- Value: ⭐⭐⭐⭐⭐ Triple contributions of dataset, method, and application value; pioneering significance for the vector animation generation field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VecGlypher: Unified Vector Glyph Generation with Language Models](vecglypher_unified_vector_glyph_generation_with_language_models.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)
- [\[CVPR 2026\] Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)

</div>

<!-- RELATED:END -->
