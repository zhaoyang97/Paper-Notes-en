---
title: >-
  [Paper Note] POSTA: A Go-to Framework for Customized Artistic Poster Generation
description: >-
  [CVPR 2025][Segmentation][Poster Generation] This paper proposes POSTA, a modular artistic poster generation framework driven by diffusion models and multimodal large language models (MLLMs). It achieves highly customizable, professional-grade poster creation through three modules: background generation, layout design planning, and artistic text stylization.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Poster Generation"
  - "Diffusion Models"
  - "Multimodal Large Language Models"
  - "Artistic Text"
  - "Typography Design"
date: 2026-05-08
content_hash: 3a56c96814ed8fe5
---

# POSTA: A Go-to Framework for Customized Artistic Poster Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.14908](https://arxiv.org/abs/2503.14908)  
**Code**: [Project Page](https://haoyuchen.com/POSTA)  
**Area**: Image Segmentation / Image Generation  
**Keywords**: Poster Generation, Diffusion Models, Multimodal Large Language Models, Artistic Text, Typography Design

## TL;DR

This paper proposes POSTA, a modular artistic poster generation framework driven by diffusion models and multimodal large language models (MLLMs). It achieves highly customizable, professional-grade poster creation through three modules: background generation, layout design planning, and artistic text stylization.

## Background & Motivation

### Background

**Background**: Poster design serves as a key medium for visual communication, with widespread demand in advertising, education, and art spheres.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing automatic poster generation methods face four major challenges: inaccurate text generation (spelling errors, character distortions), inflexible control over design elements (the "one-click generation" paradigm lacks fine-grained adjustment), insufficient aesthetic quality, and the lack of a systematic workflow.

### Key Challenge

**Key Challenge**: Artistic posters (for movies, exhibitions, concerts, etc.) impose dual requirements of content accuracy and high visual impact.

### Core Idea

**Core Idea**: While existing end-to-end methods like COLE cover the entire pipeline, their processes are complex and require a large number of intermediate inspections.

### Supplementary Note

**Additional Note**: Existing text generation models are still prone to generating gibberish and unreadable characters when handling long text sequences.

### Supplementary Note

**Additional Note**: There is a critical need for a modular, controllable, and highly aesthetic systematic solution.

## Method

### Overall Architecture

POSTA consists of three modules: (1) Background Diffusion generates thematic backgrounds based on FLUX and multiple style LoRAs; (2) Design MLLM predicts layout and typographic attributes (position, size, font, color, alignment, rotation angle) based on the LLaVA architecture (CLIP ViT-L/14 + Llama3 7B), then directly renders 100% accurate text; (3) ArtText Diffusion stylizes title text using BrushNet (a ControlNet-like inpainting architecture). Each stage is fully controllable and customizable, aligning with the workflows of professional designers.

### Key Designs

**1. Stylized Background Generation (Background Diffusion)**
- **Function**: Generates high-quality background images in various artistic styles based on user inputs.
- **Mechanism**: Multiple professional LoRA blocks are trained on top of the FLUX model (approximately 50 images per style, embedding dimension 64, resolution 1024) to independently control styles like minimalism, retro, and modern art. An MLLM-driven Magic Prompter is integrated to automatically expand the details of user-input prompts.
- **Design Motivation**: Professional-grade backgrounds are the foundation of high-quality posters. LoRA fine-tuning enables the model to master specific artistic styles while retaining its general generation capabilities, and the modular design allows users to upload custom backgrounds.

**2. Layout and Typography Planning (Design MLLM)**
- **Function**: Intelligently plans layouts, text positions, and typographic attributes based on the background image and user requirements.
- **Mechanism**: LLaVA is fine-tuned via visual instruction tuning. After comprehending the semantics of the background image, the model predicts complete attributes (coordinates, font, size, color, alignment, rotation angle) for each text element. Text elements are categorized into titles, subtitles, and general information. **Key Innovation**: Instead of relying on generative models to "draw" text, the system directly renders text based on the predicted attributes, ensuring 100% text accuracy.
- **Design Motivation**: Direct rendering entirely eliminates the text error issues inherent in generative models. Vector-format text elements support post-generation editing, aligning seamlessly with professional design workflows.

**3. Artistic Text Stylization (ArtText Diffusion)**
- **Function**: Applies artistic effects (3D, metallic textures, gradients, outlines, etc.) to the title text, merging the text harmoniously with the background style.
- **Mechanism**: An inpainting model guided by masks is trained based on SDXL's BrushNet (resolution 1216), utilizing text masks to achieve localized generation. The blending formula is $I_{\text{blended}} = M \odot I_1 + (1-M) \odot I_2$, with a Gaussian kernel applied to the mask to ensure smooth boundaries.
- **Design Motivation**: The text style must coordinate with the background's semantics and style (e.g., text on retro posters should feature aged effects). Context-aware training data enables the model to learn to generate text effects that harmonize with the background.

### Loss & Training

Each module utilizes its respective standard training loss: Background Diffusion uses the standard denoising loss of diffusion models; Design MLLM employs the cross-entropy loss of language models; and ArtText Diffusion utilizes the inpainting diffusion loss.

## Key Experimental Results

### Main Results: Human and GPT-4V Evaluation

| Method | Visual Appeal | Text Readability | Prompt Relevance |
|------|-----------|-----------|-----------|
| SD3 | ~4.5 | ~3.0 | ~4.5 |
| FLUX-dev | ~5.5 | ~4.0 | ~5.5 |
| Recraft V3 | ~6.0 | ~5.0 | ~6.0 |
| Ideogram v2 | ~6.5 | ~5.5 | ~6.0 |
| **POSTA** | **~8.0** | **~9.0** | **~8.0** |

*\*Ratings from 60 users with AI tool experience and GPT-4V (on a scale of 1-10). POSTA significantly leads across all metrics.*

### Text Accuracy (OCR Evaluation)

| Method | Precision | Recall |
|------|--------|--------|
| AnyText | ~0.3 | ~0.25 |
| TextDiffuser-2 | ~0.35 | ~0.3 |
| FLUX | ~0.4 | ~0.35 |
| **POSTA** | **~1.0** | **~1.0** |

*\*POSTA achieves near-perfect text accuracy through direct rendering.*

### Key Findings

- All other models (including commercial products like Recraft and Ideogram) experience severe accuracy drops when handling longer text sequences.
- The modular design of POSTA allows for independent adjustments to every aspect, including background, layout, fonts, and styles.
- Artistic text stylization automatically adjusts text lighting and color effects according to different background areas.
- Tasks such as reference-based poster generation and poster editing demonstrate the versatility of the framework.

## Highlights & Insights

1. **The "Non-Generative Text" Strategy**: Bypassing the text generation errors of generative models completely by predicting typographic attributes + direct rendering offers utility far exceeding end-to-end generation methods.
2. **Alignment of Modularity and Professional Workflows**: The three-stage pipeline aligns with the actual practices of professional designers. Each stage is independently adjustable, which dramatically enhances practical usability.
3. **PosterArt Dataset**: A high-quality artistic poster dataset meticulously curated by professional designers, featuring detailed typographic annotations and pixel-level text segmentations.

## Limitations & Future Work

- Design MLLM is currently limited to generating relatively simple layout designs with constrained font options.
- The dataset size (approximately 2,000 backgrounds and 2,500 text segmentations) is relatively small, which limits design diversity.
- The stylization of titles by ArtText Diffusion may not always coordinate well with certain extreme backgrounds.
- Scalability can be enhanced in the future by expanding the dataset and implementing more advanced model architectures.

## Related Work & Insights

- Compared to the systematic design pipelines of COLE/OpenCOLE, POSTA is more concise, efficient, and delivers higher aesthetic quality.
- BrushNet's mask-guided inpainting paradigm shows excellent performance in localized generation tasks.
- The strategy of using MLLMs for design planning (rather than direct generation) can be extended to other creative tasks.

## Rating

⭐⭐⭐⭐ — An outstanding practical solution to the poster generation challenge. The modular design philosophy is advanced, and the approach to text accuracy is elegant and clever. The experimental evaluation is comprehensive (covering both open-source and commercial models), and the user evaluation is highly convincing. However, there is still substantial room for improvement regarding training data scale and design complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)
- [\[CVPR 2025\] EditAR: Unified Conditional Generation with Autoregressive Models](editar_unified_conditional_generation_with_autoregressive_models.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->
