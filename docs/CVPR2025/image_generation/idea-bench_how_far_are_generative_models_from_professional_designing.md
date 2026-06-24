---
title: >-
  [Paper Note] IDEA-Bench: How Far are Generative Models from Professional Designing?
description: >-
  [CVPR 2025][Image Generation][benchmark] Proposes IDEA-Bench, the first comprehensive benchmark for professional-grade image design, covering 100 real-world design tasks (posters, picture books, typography, visual effects, etc.) and 5 input/output modes. It reveals that the strongest current model scores only 22.48/100, showing a massive gap still remaining before achieving professional-level design.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "benchmark"
  - "professional design"
  - "evaluation"
  - "MLLM"
  - "visual effects"
  - "storyboard"
date: 2026-05-08
content_hash: a61193c8f0574bf5
---

# IDEA-Bench: How Far are Generative Models from Professional Designing?

**Conference**: CVPR 2025  
**arXiv**: [2412.11767](https://arxiv.org/abs/2412.11767)  
**Code**: [https://github.com/ali-vilab/IDEA-Bench](https://github.com/ali-vilab/IDEA-Bench)  
**Area**: Image Generation  
**Keywords**: benchmark, professional design, image generation, evaluation, MLLM, visual effects, storyboard

## TL;DR

Proposes IDEA-Bench, the first comprehensive benchmark for professional-grade image design, covering 100 real-world design tasks (posters, picture books, typography, visual effects, etc.) and 5 input/output modes. It reveals that the strongest current model scores only 22.48/100, showing a massive gap still remaining before achieving professional-level design.

## Background & Motivation

**Background**: T2I models such as DALL-E 3 and FLUX-1 perform exceptionally on academic benchmarks, attracting millions of daily active users. Benchmarks like GenEval and DreamBooth cover basic text-to-image and single-image editing evaluations.

**Limitations of Prior Work**: (1) Existing benchmarks focus only on isolated academic tasks (e.g., T2I alignment, simple editing), dissolving connection from real-world professional design requirements; (2) Prompts are too short (mean < 11 words), far below the long, detailed instructions used by professional designers; (3) Multi-image input/output evaluation dimensions are lacking; (4) Traditional metrics like FID/CLIPScore fail to capture the subtle nuances of aesthetics, context, and multi-modal integration.

**Key Challenge**: Professional designers still rely on traditional tools like Photoshop, indicating that generative models severely lack the capability to handle complex and diverse professional tasks, yet there lacks a systematic evaluation framework to quantify this gap.

**Key Insight**: Tasks are collected from real design platforms and professional designers, categorized by model capability levels, to establish a multi-tiered evaluation system.

## Method

### Overall Architecture

1. **Task Collection**: Acquired 100 representative tasks from online design platforms and professional designers.
2. **Taxonomy**: Categorized into 5 main groups based on input/output modalities — T2I, I2I, Is2I, T2Is, and I(s)2Is.
3. **Annotation Pipeline**: GPT-4o generates task definitions and prompts, followed by manual construction of 6 hierarchical evaluation questions (basic $\rightarrow$ quality $\rightarrow$ detail).
4. **Evaluation**: Human evaluation (full set) + MLLM automatic evaluation (subset of 18 tasks, IDEA-Bench-mini).

### Key Designs

**1. Five-Level Task Taxonomy**
- **T2I (Text-to-Image)**: 11 tasks, including long-prompt scenarios like posters, business cards, game UI, and Logos (averaging 138.68 words, vs. <11 words in existing benchmarks).
- **I2I (Image-to-Image)**: 13 tasks such as packaging rendering, image retouching, style transfer, and relighting.
- **Is2I (Images-to-Image)**: Multi-reference image input tasks such as brand merchandise generation and character fusion.
- **T2Is (Text-to-Images)**: Consistency-requiring multi-image outputs like multi-view generation and picture book creation.
- **I(s)2Is (Images-to-Images)**: The most complex tasks such as storyboard design and character set generation.
- **Design Motivation**: With the development of unified generative models, there is a need to cover a complete capability spectrum from simple to complex.

**2. Hierarchical Binary Evaluation System**
- **Function**: 6 binary determination questions (0/1) per case, divided into 3 levels: basic task understanding (Q1-2) $\rightarrow$ execution quality (Q3-4) $\rightarrow$ detailed aesthetics (Q5-6).
- **Core Rule**: Hierarchical dependency — if lower levels do not achieve full marks, higher levels are automatically scored as 0.
- **Design Motivation**: Prioritizes task completion correctness over aesthetics, aligning with professional design standards (do it right first, then make it beautiful).

**3. MLLM Automatic Evaluation (IDEA-Bench-mini)**
- **Function**: Uses Gemini 1.5 Pro to automatically evaluate on 18 representative tasks, averaging the score over 3 runs per case.
- **Mechanism**: Customizes evaluation questions for each case (rather than sharing them), and performs iterative manual calibration to align MLLM scores with human annotations.
- **Design Motivation**: Addresses issues where MLLMs are sensitive to image order and unreliable in multi-image comprehension.

### Prompting Strategy

For models that do not natively support multi-image generation (e.g., FLUX-1, SD3), GPT-4o is used to rephrase multi-modal inputs into individual prompts for each image. This allows base T2I models to participate in multi-image generation evaluations.

## Key Experimental Results

### Main Results — All Categories Scores

| Model | T2I | I2I | Is2I | T2Is | I(s)2Is | **Avg** |
|---|---|---|---|---|---|---|
| FLUX-1† | 46.06 | 12.13 | 4.89 | 20.15 | 29.17 | **22.48** |
| SD3† | 24.04 | 10.79 | 4.69 | 21.59 | 13.06 | 14.83 |
| DALL-E 3† | 24.34 | 6.95 | 5.27 | 14.36 | 14.44 | 13.07 |
| OmniGen† | 21.41 | 8.17 | 2.77 | 23.52 | 21.39 | 15.45 |
| Emu2† | 17.98 | 7.05 | 8.98 | 15.53 | 12.78 | 12.46 |
| Emu2 (Native) | 17.98 | 7.05 | 8.98 | – | – | 6.81 |
| Anole (7B) | 0.00 | 0.64 | 0.00 | 1.74 | 0.00 | 0.48 |

(† indicates the use of GPT-4o prompt rephrasing to adapt to all tasks)

### T2I Subtasks

| Model | Architecture | Business Card | Game UI | Infographic | Poster | **Avg** |
|---|---|---|---|---|---|---|
| FLUX-1 | 100 | 38.89 | 5.56 | 0 | 56.67 | 46.06 |
| DALL-E 3 | 22.22 | 0 | 0 | 0 | 23.33 | 24.34 |
| Anole | 0 | 0 | 0 | 0 | 0 | 0 |

### Key Findings

1. **Severe insufficiency in professional design capabilities**: The strongest model, FLUX-1, scores only 22.48, showing a massive gap from the passing mark (60).
2. **Reversal between generalist and specialist models**: The best generalist model Emu2 scores only 6.81, performing worse than T2I models with prompt rephrasing.
3. **Multi-image generation is the biggest bottleneck**: All models score < 9 in the Is2I category, demonstrating that multi-reference image understanding is practically non-functional.
4. **FLUX-1 dominates in T2I**: It performs excellently on text-only tasks (e.g., scoring 100 in architecture style), but degrades sharply on image-guided tasks.
5. **Anole scores nearly zero**: Multi-modal interleaved generation models are completely unusable for professional design tasks.
6. **Prompt length challenge**: The average prompt length in IDEA-Bench is 138.68 words, far exceeding the <11 words of existing benchmarks, exposing weaknesses in long-prompt following capabilities.

## Highlights & Insights

- Systematically introduces professional design tasks into generative model evaluation for the first time, bridging the gap between academic benchmarks and practical demands.
- The methodology of the five-level task taxonomy and hierarchical binary evaluation serves as an valuable reference for future benchmarks.
- Extends the range of evaluable models by using GPT-4o prompt rephrasing to allow base T2I models to participate in multi-image tasks.
- Reveals a key observation: generative models fail before even getting the task "correct", making aesthetic enhancement a secondary concern.
- Demonstrates the practicality of the MLLM automatic evaluation and human calibration closed-loop solution.

## Limitations & Future Work

- Many of the 100 tasks are overly difficult for existing models, resulting in numerous zero scores and restricted discriminative power.
- Subjectivity in human evaluation persists, especially regarding the quality and aesthetic judgments in Q3-6.
- MLLM automatic evaluation only covers 18 subtasks, requiring broader coverage.
- Lacks human baseline scores from professional designers, making it difficult to quantify the human-machine gap.
- The strict hierarchical dependency in evaluation may overly penalize minor mistakes.

## Related Work & Insights

- **GenEval**: Evaluates on 6 T2I tasks with short prompts; this work significantly upgrades the complexity.
- **ImagenHub**: Covers 7 task types but remains confined to academic definitions.
- **DEsignBench**: Focuses on design scenarios but its task scope and capability coverage are narrower than this work.
- **Insight**: Benchmarks for generative models need a paradigm shift from "what can they generate" to "what design tasks can they complete".

## Rating

⭐⭐⭐⭐ — The first systematic benchmark specialized for professional design. The tasks are meticulously designed, and the evaluation system is rational, offering highly significant guidance for field development. However, some tasks are excessively difficult for current models, leading to a lack of distinction in scoring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [\[ICLR 2026\] PICABench: How Far are We from Physically Realistic Image Editing?](../../ICLR2026/image_generation/picabench_how_far_are_we_from_physical_realistic_image_editing.md)
- [\[CVPR 2025\] Image Generation Diversity Issues and How to Tame Them](image_generation_diversity_issues_and_how_to_tame_them.md)
- [\[CVPR 2025\] PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?](physicsgen_can_generative_models_learn_from_images_to_predict_complex_physical_r.md)
- [\[CVPR 2025\] Goku: Flow Based Video Generative Foundation Models](goku_flow_based_video_generative_foundation_models.md)

</div>

<!-- RELATED:END -->
