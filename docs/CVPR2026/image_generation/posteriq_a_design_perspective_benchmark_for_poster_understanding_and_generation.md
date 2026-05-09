---
title: >-
  [Paper Note] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation
description: >-
  [CVPR 2026][Image Generation][Poster Design] This paper introduces PosterIQ, a comprehensive benchmark for poster design evaluation, comprising 7,765 understanding annotations and 822 generation prompts across 24 task categories — including OCR, font perception, layout reasoning, design intent understanding, and compositionally-aware generation — to systematically diagnose the gap between current MLLMs and diffusion models in design cognition.
tags:
  - CVPR 2026
  - Image Generation
  - Poster Design
  - Multimodal Benchmark
  - Visual Understanding
  - Typographic Design
date: 2026-05-08
content_hash: 2852bdf45b27276b
---

# PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation

**Conference**: CVPR 2026  
**arXiv**: [2603.24078](https://arxiv.org/abs/2603.24078)  
**Code**: [https://github.com/ArtmeScienceLab/PosterIQ-Benchmark](https://github.com/ArtmeScienceLab/PosterIQ-Benchmark)  
**Area**: Multimodal VLM / Image Generation  
**Keywords**: Poster Design, Multimodal Benchmark, Visual Understanding, Image Generation, Typographic Design

## TL;DR

This paper introduces PosterIQ, a comprehensive benchmark for poster design evaluation, comprising 7,765 understanding annotations and 822 generation prompts across 24 task categories — including OCR, font perception, layout reasoning, design intent understanding, and compositionally-aware generation — to systematically diagnose the gap between current MLLMs and diffusion models in design cognition.

## Background & Motivation

**State of the Field**: Multimodal large language models (MLLMs) have made substantial progress in visual understanding, and image generation models have matured in text-to-image synthesis and style control. Existing benchmarks (e.g., MMBench, Creation-MMBench) have begun evaluating multimodal creative capabilities, but remain largely text-centric or focused on aesthetic quality.

**Limitations of Prior Work**: Posters constitute a highly constrained visual communication medium requiring the integration of thematic interpretation, information hierarchy, typographic conventions, text-image coupling, stylistic consistency, and audience awareness within a confined space. Existing evaluation frameworks lack systematic coverage of these design dimensions — OCR benchmarks ignore typographic semantics in design contexts, and image generation evaluations overlook compositional constraints and design intent communication.

**Root Cause**: Effective poster design is not merely about visual appeal but about effective communication — key information must be perceived, understood, and retained. Current models can produce visually attractive images, yet exhibit systematic deficiencies in information hierarchy organization, typographic semantics, and creative expression under compositional constraints.

**Paper Goals**: To establish a poster design benchmark spanning both understanding and generation tasks, evaluating model creativity and design cognition from a design-theory perspective rather than a purely aesthetic one.

**Starting Point**: Poster design is decomposed into five capability dimensions — text legibility/readability, layout reasoning/hierarchical organization, semantic-style consistency, text-image coordination/salience control, and rhetorical modeling/metaphor generation — each addressed by dedicated evaluation tasks.

**Core Idea**: Replace holistic scoring with decoupled, task-specific evaluation dimensions to systematically diagnose model capabilities and deficiencies across all aspects of design cognition.

## Method

### Overall Architecture

PosterIQ is organized into two major modules: understanding and generation. The understanding module encompasses global quality assessment and four task families: OCR tasks (5 sub-tasks, 3,005 items), font perception tasks (4 sub-tasks, 2,788 items), spatial reasoning tasks (6 sub-tasks, 1,178 items), advanced visual design understanding (3 sub-tasks, 575 items), and scoring tasks (219 items). The generation module contains 5 task categories (822 prompts): dense generation, font generation, style generation, composition generation, and intent generation.

### Key Designs

1. **Multi-level OCR Evaluation**:

    - Function: Systematically evaluates models' visual text recognition capability across a simple-to-difficult spectrum.
    - Mechanism: Five sub-tasks form a difficulty gradient — Logo OCR (highly stylized/distorted logotype text), real poster OCR (multi-font/multi-scale/dense layouts), simple OCR (standard text on white background, measuring upper bound), hard OCR (disordered letters + textured background + rotation + random color), and multi-size OCR (14 font sizes testing scale robustness). Robustness is measured by the performance gap $\Delta$ between simple and hard conditions.
    - Design Motivation: Text in posters is highly stylized and intertwined with graphic elements; standard scene-text recognition fails to reflect the genuine OCR challenges of design contexts.

2. **Font Understanding and Spatial Reasoning**:

    - Function: Evaluates models' perception of typographic semantics and layout structure.
    - Mechanism: Font matching (visual-style-only judgment without font name priors), font attribute perception (consistency across 37 human-defined attributes), recognition of traditional/advanced typographic effects. Spatial reasoning covers text localization (normalized coordinate bounding boxes), alignment/rotation inference, whitespace perception (7×7 grid IoU), layout comparison (professional vs. design-principle-violating layouts), and layout generation (generating bounding boxes given text specifications).
    - Design Motivation: Typography and layout are central to poster design — fonts convey emotion and theme, while layout determines information hierarchy. These constitute foundational competencies in design literacy.

3. **Advanced Design Understanding and Intent Communication**:

    - Function: Evaluates models' high-level understanding of style, compositional techniques, and visual metaphor.
    - Mechanism: Style classification (17 design styles including minimalism, Memphis, diffused light, etc.), compositional structure understanding (description of operations such as displacement, nesting, cropping, repetition, and mirroring), and intent/metaphor interpretation (e.g., stacked phones → burger, toy soldiers → peace dove). An LLM scores generated descriptions based on coverage of key concepts from human annotations.
    - Design Motivation: Advanced design concerns not only visual presentation but meaning communication. Metaphor and rhetoric are the core expressions of design creativity.

### Loss & Training

PosterIQ is an evaluation benchmark rather than a training methodology and involves no training. In generation evaluation, MLLMs serve as automated judges to verify whether generated outputs contain target elements.

## Key Experimental Results

### Main Results on Understanding Tasks

| Model | Logo OCR | Poster OCR | Simple→Hard Δ↓ | Font Matching | Style Understanding | Composition Understanding | Intent Understanding |
|------|---------|------------|----------------|---------|---------|---------|---------|
| GPT-5 | 0.952 | 0.922 | 0.469 | 0.668 | 0.851 | 0.730 | 0.824 |
| Claude-4.5 | 0.902 | 0.884 | 0.372 | 0.699 | 0.813 | 0.608 | 0.761 |
| Gemini-2.5-Pro | 0.923 | 0.952 | 0.525 | 0.362 | 0.830 | **0.802** | 0.788 |
| Qwen3-VL-8B | 0.882 | 0.931 | **0.156** | 0.063 | 0.610 | 0.684 | 0.710 |
| MiniCPM-V-4.5 | 0.883 | 0.932 | 0.468 | -0.001 | 0.631 | 0.635 | 0.691 |

### Generation Task Comparison

| Model | Dense Generation | Font Diversity | Style Generation | Composition Generation | Intent Generation | Average |
|------|---------|-----------|---------|---------|---------|------|
| Seedream-4.0 | 0.618 | 0.342 | 0.591 | 0.848 | 0.645 | 0.609 |
| Gemini-2.5-Flash | **0.622** | **0.391** | 0.590 | **0.866** | 0.663 | **0.626** |
| GPT-Image-1 | 0.508 | 0.299 | **0.633** | 0.856 | **0.670** | 0.593 |
| Qwen-Image | 0.464 | 0.286 | 0.620 | 0.801 | 0.589 | 0.552 |

### Key Findings

- **Significant variation in OCR robustness**: Qwen3-VL-8B exhibits the smallest simple→hard OCR gap (Δ=0.156), indicating greater robustness to visual disturbances. However, its font matching performance is near chance (0.063), revealing extremely weak fine-grained typographic perception.
- **Systematic gap between closed-source and open-source models**: On advanced design understanding (style/composition/intent), GPT-5 and Gemini-2.5-Pro consistently outperform open-source models. Gemini leads in composition understanding (0.802); GPT-5 leads in intent interpretation (0.824).
- **Font perception is a universal weakness**: Aside from GPT-5 and Claude-4.5, most models perform near or below chance on font matching (including negative scores), revealing a severe deficiency in fine-grained typographic style recognition across current MLLMs.
- **Font diversity is the bottleneck in generation**: All models score below 0.4 on font generation richness (best: Gemini at 0.391), while composition generation reaches 0.866, identifying font control as the weakest link in generative models.
- **Global-local trade-off**: Models proficient in composition and intent (global planning) tend to underperform on style and font tasks (local precision), reflecting a trade-off between global planning and local fidelity in model training.

## Highlights & Insights

- The **design-theory-driven benchmark construction methodology** is the paper's most significant contribution. Unlike purely technically oriented benchmarks, PosterIQ defines capability dimensions from design principles, with each task grounded in explicit design theory. This "domain knowledge + AI" evaluation paradigm is generalizable to other creative fields (e.g., UI design, illustration).
- The strategy of **replacing holistic scoring with decoupled evaluation** is particularly valuable. Experiments show that models' overall scores correlate poorly with human judgment (highest sim=0.483), whereas specific sub-tasks yield diagnostically meaningful information — an important methodological insight for evaluating creative tasks.
- The **VLM→T2I iterative loop** experiment demonstrates how understanding capability can directly enhance generation quality: VLM analyzes design issues → prompt is revised → image is regenerated → quality improves. This represents a practical design-assistance paradigm.

## Limitations & Future Work

- MLLMs are used as automated judges for generation quality; however, experiments reveal that MLLMs are "insensitive" to design quality (unable to effectively distinguish good from poor designs), introducing a closed-loop evaluation problem.
- The dataset is relatively limited in scale (7,765 understanding annotations) and primarily covers English-language posters, with insufficient coverage of complex typographic systems such as Chinese and Japanese.
- Fine-grained user studies are absent — it remains to be validated whether the benchmark tasks genuinely reflect the capability dimensions that professional designers care about.
- Composition generation evaluation relies on coarse-grained key-concept matching, which struggles to capture subtle design distinctions such as spatial relationships.

## Related Work & Insights

- **vs. Creation-MMBench**: Creation-MMBench evaluates general creative capabilities, whereas PosterIQ focuses on the highly constrained domain of poster design, with deeper task design closely aligned to design theory.
- **vs. GenEval/DPG-Bench**: These benchmarks assess general text-image alignment and prompt following; PosterIQ adds design-specific dimensions: typographic hierarchy, compositional techniques, and visual rhetoric/metaphor.
- **vs. PosterLLaVA/COLE**: These are poster generation methods; PosterIQ provides a standardized framework for evaluating such methods.

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark to evaluate MLLMs from a design-theory perspective, with substantive task definitions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight understanding models and four generation models evaluated across 24 task categories with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Task definitions are clear and analysis is in-depth, though the paper structure is somewhat complex and information density is very high.
- Value: ⭐⭐⭐⭐ Provides a systematic framework for evaluating design cognition in MLLMs and exposes critical deficiencies such as typographic perception.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation](innoads-composer_efficient_condition_composition_for_e-commerce_poster_generatio.md)
- [\[CVPR 2026\] GIST: Towards Design Compositing](gist_towards_design_compositing.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

<!-- RELATED:END -->
