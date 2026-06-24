---
title: >-
  [Paper Note] AutoPresent: Designing Structured Visuals from Scratch
description: >-
  [CVPR 2025][Image Generation][Slide Generation] This paper proposes the AutoPresent framework and the SlidesBench benchmark to systematically study the task of generating presentation slides from natural language instructions for the first time. By leveraging LLMs to generate Python code (instead of end-to-end image generation) to create PPTX slides, combined with the SlidesLib utility library and iterative refinement, an 8B open-source model achieves performance close to GPT…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Slide Generation"
  - "Code Generation"
  - "Large Language Models"
  - "Visual Design"
  - "Benchmark"
date: 2026-05-08
content_hash: 2a93b44d9e5aca1a
---

# AutoPresent: Designing Structured Visuals from Scratch

**Conference**: CVPR 2025  
**arXiv**: [2501.00912](https://arxiv.org/abs/2501.00912)  
**Code**: [https://github.com/para-lost/AutoPresent](https://github.com/para-lost/AutoPresent)  
**Area**: Image Generation  
**Keywords**: Slide Generation, Code Generation, Large Language Models, Visual Design, Benchmark

## TL;DR

This paper proposes the AutoPresent framework and the SlidesBench benchmark to systematically study the task of generating presentation slides from natural language instructions for the first time. By leveraging LLMs to generate Python code (instead of end-to-end image generation) to create PPTX slides, combined with the SlidesLib utility library and iterative refinement, an 8B open-source model achieves performance close to GPT-4o.

## Background & Motivation

Designing structured visual content (such as presentation slides) is a core skill for effective communication, requiring both **content creation** (text, images, charts) and **visual planning** (layout, color schemes, typography). Even human experts can spend hours iteratively refining slides.

Recently, AI agents have demonstrated excellent performance in tasks like software engineering and web navigation, but their capabilities in generating semi-structured creative media, such as slides, remain largely untested. Existing methods primarily focus on generating slides by extracting content from documents, or merely generating the content without handling the visual layout.

**Key Challenge**:
1. Slides require precise control over text content, image locations, color configurations, and element layouts.
2. End-to-end image generation (e.g., Stable Diffusion, DALL-E) struggles to generate legible text and is uneditable.
3. The success rate of small open-source models directly generating complex code is extremely low.
4. There is a lack of standardized evaluation benchmarks.

**Key Insight**: Modeling slide generation as a **program generation** task—the model receives natural language instructions and generates a Python program (using the `python-pptx` library), which yields editable PPTX files when executed. This approach inherently supports precise control and manual editing.

## Method

### Overall Architecture

The AutoPresent pipeline:
1. The user provides natural language instructions (of three difficulty levels).
2. The model generates Python code (optionally using the high-level function library SlidesLib).
3. The code is executed to generate PPTX slides.
4. Optional iterative refinement: the model views the rendered screenshot of the slide and self-improves the code.

### Key Designs

1. **SlidesBench Benchmark Construction**:

    - Collected 310 public slide decks across 10 domains (art, business, technology, etc.).
    - 7k training samples + 585 test samples.
    - Three difficulty levels of instruction design:
        - **Detailed Instruction + Image**: Provides full content, layout specifications, and image paths (easiest).
        - **Detailed Instruction Only**: Provides layout specifications but replaces images with natural language descriptions (medium).
        - **High-level Summary Instruction**: Provides only thematic descriptions, e.g., "create a title page for Airbnb" (hardest).
    - Instruction generation pipeline: For each slide deck, 3 examples were first manually written, then batch-generated using gpt-4o-mini, followed by manual review and correction for the test set.

2. **Evaluation Metrics System**:

    - **Reference-Based Metrics**:
        - Element Matching: The ratio of total matched element areas (text boxes, images, shapes).
        - Content Similarity: Content similarity of matched element pairs (sentence-transformers for text, CLIP for images).
        - Color Similarity: Calculated using the CIEDE2000 formula for differences in font and background colors.
        - Position Similarity: Calculated via Manhattan distance after normalizing coordinates.
    - **Reference-Free Metrics**: Based on slide design principles, using GPT-4o for scoring (0-5 scale).
        - Text: Concise titles, refined content, legible fonts.
        - Image: High-quality images, reasonable aspect ratios.
        - Layout: Element alignment, no overlapping, sufficient margins.
        - Color: High contrast, avoiding eye-straining colors.
    - ICC Validation: The ICC between model ratings and two human annotators ranges from 73.8% to 85.3%, representing "high agreement."

3. **SlidesLib Tool Library**:

    - Simplifies `python-pptx` code from an average of 170 lines to an average of 13 lines of high-level function calls.
    - 4 basic functions: `add_title`, `add_text`, `add_bullet_points`, `add_image`.
    - 3 image-related functions: `generate_image` (calls DALL-E 3), `search_image` (Bing Search), `search_screenshot`.
    - The model learns to use these by reading function documentation and 2 in-context examples.
    - Significantly increases the code execution success rate for small models.

4. **AutoPresent Model Training**:

    - LoRA fine-tuning (rank=128) based on Llama-3.1-8B-Instruct.
    - Training data: Extracted canonical programs for each slide (rule-based element extraction $\rightarrow$ generate `python-pptx` code).
    - Two program versions: Original `python-pptx` + SlidesLib version.
    - Four training set combinations (selected 4 out of 3 instructions $\times$ 2 program versions), 7k samples each.

5. **Iterative Refinement**:

    - Inputs the original instruction, the first-round code, and the rendered slide screenshot together into GPT-4o.
    - Requests the model to generate improved code by adjusting colors, spacing, etc.
    - The first round of refinement yields the largest improvement, with diminishing marginal returns in subsequent rounds.

### Loss & Training

As an LLM fine-tuning task, a standard autoregressive language modeling loss (next token prediction) is employed to perform parameter-efficient fine-tuning with LoRA on (instruction, code) pairs.

## Key Experimental Results

### Main Results

Detailed Instruction + Image Setting (Core Comparison):

| Method | Execution Rate % | Content ↑ | Layout ↑ | Color ↑ | Text ↑ | Image ↑ | Overall ↑ |
|------|---------|-------|-------|-------|-------|-------|----------|
| Stable Diffusion | 100 | 33.4 | 36.9 | 40.5 | 19.6 | 45.1 | 48.0 |
| DALL-E 3 | 100 | 39.9 | 56.7 | 53.4 | 32.7 | 87.3 | 50.2 |
| Llama 8B (w/o SlidesLib) | 2.1 | 94.6 | 50.0 | 50.0 | 50.0 | 8.3 | 1.3 |
| GPT-4o (w/o SlidesLib) | 89.2 | 91.6 | 53.7 | 54.7 | 51.9 | 72.8 | 55.1 |
| AutoPresent (w/o SlidesLib) | 79.0 | 79.7 | 54.2 | 60.9 | 45.3 | 62.7 | 45.2 |
| GPT-4o (+SlidesLib) | 86.7 | 92.5 | 70.5 | 59.4 | 54.6 | 83.7 | 58.0 |
| AutoPresent (+SlidesLib) | 84.1 | 92.2 | 58.6 | 64.7 | 47.8 | 73.2 | 55.0 |

Detailed Instruction / High-level Summary Instruction Setting:

| Method | Detailed Only Overall | High-level Overall |
|------|------------------|--------------|
| GPT-4o (+SlidesLib) | 56.3 | 58.5 |
| AutoPresent (+SlidesLib) | 55.2 | 47.8 |
| Llama 8B (+SlidesLib) | 37.4 | 43.7 |

### Ablation Study

Iterative refinement performance (GPT-4o + SlidesLib):

| Iterations | Detailed + Image | Detailed Only | High-level Summary |
|----------|-----------|--------|----------|
| 0 (Initial) | 58.0 | 56.3 | 58.5 |
| 1 | 59.5 | 59.5 | 59.8 |
| 2 | 59.3 | 60.1 | 61.3 |
| 3 | **60.1** | 59.4 | **61.4** |

| Configuration | Description |
|------|------|
| Code Generation vs. End-to-End Image | Code generation significantly outperforms image generation in content similarity (91.6 vs 39.9). |
| With SlidesLib vs. Without | SlidesLib improves the overall score of Llama from 1.3 to 33.5 (+32.2). |
| VLM (Llava) vs. LLM (Llama) | VLM performs better without tools, while LLM surpasses it once SlidesLib is integrated. |

### Key Findings

- **Code generation far outperforms end-to-end image generation**: End-to-end methods fail to produce legible text and are uneditable.
- **Small models are almost unusable for direct generation**: Llama 8B achieves an execution success rate of only 2.1% without SlidesLib.
- **SlidesLib is a key enabler**: It reduces code length from 170 lines to 13 lines, enabling small models to generate runnable programs.
- **AutoPresent (8B) is close to GPT-4o**: In the Detailed Instruction + Image setting, a paired t-test shows no statistically significant difference ($p=0.657$).
- **All models still lag behind humans**: The design quality metrics of reference slides are consistently higher than the best model outputs.
- **Iterative refinement is effective but shows diminishing returns**: The first round gives the most substantial improvement, with subsequent gains tapering off.
- **Image sourcing remains a bottleneck**: When images are not provided, GPT-4o's overall score drops from 55.1 to 28.7 (without SlidesLib).

## Highlights & Insights

- **Pioneering Task Definition**: First to model slide generation as a systematic NL-to-code task, accompanied by a comprehensive benchmark.
- **Deep Methodological Insight**: The paradigm of "transforming complex visual design tasks into code generation" is far more controllable and practical than end-to-end generation.
- **SlidesLib Design Philosophy**: Lowering code complexity allows small models to participate, demonstrating the value of "tool augmentation" in AI agents.
- **Comprehensive Evaluation System**: Formulates both reference-based and reference-free evaluation metrics, validating the reliability of model-based scoring with ICC.
- **Three-level Difficulty Design**: Detailed + Image $\rightarrow$ Detailed Only $\rightarrow$ High-level Summary, progressively scaling the challenge to match real-world user scenarios.
- **Open-Source Contribution**: Open-sourcing the 8B model, datasets, and tool library yields significant value for the community.

## Limitations & Future Work

- Only supports single-page slide generation; does not address consistent design representation (unified style/color palette) for multi-slide decks.
- Currently generates the entire code in one shot, without leveraging progressive/interactive design workflows.
- Does not support presentation-specific features like animations and transitions.
- SlidesLib has a small function set (only 7 functions) and lacks support for complex elements like charts and tables.
- Iterative refinement relies on GPT-4o's visual comprehension capability; the self-improvement capability of open-source models needs enhancement.
- The quality and relevance of image sourcing (search/generation) remain a bottleneck.
- Using GPT-4o as a reference-free metric evaluator introduces potential model bias.

## Related Work & Insights

- **Design2Code**: A pioneering work in webpage HTML generation. This paper draws inspiration from its reference-based evaluation approach, but slide generation presents greater challenges (requiring both content creation and visual design capabilities).
- **Visual Programming**: This work continues the paradigm of code generation driving visual tasks, extending it from TikZ/SVG to more complex slides.
- **Tool-augmented LLM**: The success of SlidesLib validates the effectiveness of tool augmentation in reducing task complexity.
- **Commercial Products like Gamma**: AI slide generation tools are actively commercializing. This paper provides academic benchmarks and open-source baselines.
- **Implications for other structured visual designs (posters, resumes, infographics)**: Demonstrates that the code generation paradigm is highly applicable to these formats.

## Rating

- Novelty: ⭐⭐⭐⭐ The first study to systematically investigate slide generation, making key contributions with its benchmark and tool library; however, code generation itself is not a novel methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, involving 10 domains, 585 test samples, comparisons across 8+ methods, 3 difficulty settings, user studies, and iterative refinement analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, systematically organized experiments, and rich, intuitive figures. A high-quality output jointly produced by UC Berkeley and CMU.
- Value: ⭐⭐⭐⭐ Opens up a new research direction for structured visual generation. The open-source toolchain is highly beneficial for pushing the community forward, though a gap still exists before practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)
- [\[ICLR 2026\] Factuality Matters: When Image Generation and Editing Meet Structured Visuals](../../ICLR2026/image_generation/factuality_matters_when_image_generation_and_editing_meet_structured_visuals.md)
- [\[CVPR 2025\] IDEA-Bench: How Far are Generative Models from Professional Designing?](idea-bench_how_far_are_generative_models_from_professional_designing.md)
- [\[CVPR 2025\] Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget](stretching_each_dollar_diffusion_training_from_scratch_on_a_micro-budget.md)
- [\[CVPR 2025\] Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing](beyond_convolution_a_taxonomy_of_structured_operators_for_learning-based_image_p.md)

</div>

<!-- RELATED:END -->
