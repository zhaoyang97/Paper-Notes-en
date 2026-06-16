---
title: >-
  [Paper Note] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation
description: >-
  [CVPR 2026][Image Generation][Paper Note] This paper proposes PosterIQ, a comprehensive benchmark for poster design containing 7,765 understanding annotations and 822 generation prompts. Spanning 24 task categories such as OCR, font awareness, layout reasoning, design intent understanding, and composition-aware generation, it systematically evaluates the gap i
tags:
  - CVPR 2026
  - Image Generation
date: 2026-05-08
content_hash: 7c8265090b9552d4
---
# PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation

**Conference**: CVPR 2026  
**arXiv**: [2603.24078](https://arxiv.org/abs/2603.24078)  
**Code**: [https://github.com/ArtmeScienceLab/PosterIQ-Benchmark](https://github.com/ArtmeScienceLab/PosterIQ-Benchmark)  
**Area**: Multi-modal VLM / Image Generation  
**Keywords**: Poster Design, Multi-modal Benchmark, Visual Understanding, Image Generation, Typography Design

## TL;DR

This paper proposes PosterIQ, a comprehensive benchmark for poster design containing 7,765 understanding annotations and 822 generation prompts. Spanning 24 task categories such as OCR, font awareness, layout reasoning, design intent understanding, and composition-aware generation, it systematically evaluates the gap in design cognition between MLLMs and diffusion models.

## Background & Motivation

**Background**: Multi-modal Large Language Models (MLLMs) have made significant progress in visual understanding, while image generation models have matured in text-to-image synthesis and style control. Existing benchmarks (e.g., MMBench, Creation-MMBench) have begun evaluating multi-modal creative capabilities, but they are primarily text-centric or focused on aesthetic quality.

**Limitations of Prior Work**: Posters are a highly constrained visual communication medium that requires integrating theme interpretation, information hierarchy, typographic rules, text-image coupling, stylistic consistency, and audience preference within a limited space. Existing evaluation systems lack systematic coverage of these design dimensions—OCR benchmarks ignore typographic semantics in design contexts, and image generation evaluations do not focus on compositional constraints and design intent communication.

**Key Challenge**: Effective poster design is not just about being "visually appealing" but "effective communication"—key information must be perceived, understood, and remembered. Current models can generate visually attractive images but exhibit systematic shortcomings in information hierarchy organization, typographic semantics, and creative expression under compositional constraints.

**Goal**: Establish a poster design benchmark covering both understanding and generation to evaluate creative capabilities and design cognition from a design theory perspective (rather than a purely aesthetic one).

**Key Insight**: Decompose poster design into five capability dimensions—text understanding/readability, layout reasoning/hierarchical organization, semantic-style consistency, text-image coordination/saliency control, and rhetorical modeling/metaphor generation—designing specialized evaluation tasks for each.

**Core Idea**: Replace holistic scoring with decoupled, task-specific evaluation dimensions to systematically diagnose model capabilities and shortcomings across various aspects of design cognition.

## Method

### Overall Architecture

PosterIQ is divided into two major modules: Understanding and Generation. The understanding module includes global quality assessment and four task families: OCR tasks (5 sub-tasks, 3,005 items), font awareness tasks (4 sub-tasks, 2,788 items), spatial reasoning tasks (6 sub-tasks, 1,178 items), advanced visual design understanding (3 sub-tasks, 575 items), and scoring tasks (219 items). The generation module includes 5 task categories (822 prompts): dense generation, font generation, style generation, composition generation, and intent generation.

### Key Designs

**1. Multi-level OCR Evaluation: Using Difficulty Gradients to Probe "Real Recognition Ability in Design Scenarios"**

Text in posters consists of highly stylized design elements intertwined with graphics rather than simple black-and-white scans. Ordinary scene text recognition cannot detect where models fail in such scenarios. Thus, the benchmark splits OCR into five difficulty-increasing sub-tasks: Logo OCR for deformed and artistic logo text; Real Poster OCR for real layouts with multiple fonts, scales, and dense arrangements; Simple OCR (standard text on white background) to measure upper-bound performance; Hard OCR with intentionally overlaid out-of-order letters, textured backgrounds, rotations, and random coloring; and Multi-size OCR with 14 font sizes to examine scale stability. The key is not the absolute score of a single tier, but the performance drop $\Delta$ between simple and hard tiers—this value directly quantifies the model's robustness to visual interference.

**2. Font Understanding and Spatial Reasoning: Decoupling Typographic Semantics and Layout Structure**

Fonts convey emotion and themes, while layout determines information hierarchy; these are the foundations of poster design but are easily obscured by holistic scoring. For fonts, the name prior is removed, forcing the model to perform font matching based solely on visual style. 37 human-defined attributes are used to examine the consistency of font attribute perception, alongside traditional and advanced font effect recognition. For spatial reasoning, five aspects are covered: text localization (outputting normalized bbox coordinates), alignment and rotation inference, white space perception (calculating IoU on a $7 \times 7$ grid), layout contrast (choosing between professional and principle-violating layouts), and layout generation (generating bboxes given text specifications). Each sub-task targets a specific design literacy.

**3. Advanced Design Understanding and Intent Communication: Seeing "Meaning" Rather Than Just "Pixels"**

At higher design levels, the focus shifts from visual presentation to meaning delivery—style, compositional techniques, and visual metaphors are core creative expressions. This category includes three components: style classification distinguishes between 17 design styles like Minimalism, Memphis, and Diffuse Light; composition structure understanding describes operations like misalignment, nesting, cropping, repetition, and mirroring; intent and metaphor interpretation analyzes visual puns like "stacked phones forming a burger" or "toy soldiers arranged as a peace dove." Since answers are open-ended descriptions rather than options, LLMs are used to verify the coverage of key concepts from human annotations in the model-generated descriptions.

**4. Composite-aware Generation Evaluation: Integrating Generation into Design Cognition Assessment via "Element Coverage"**

A "understanding + generation" benchmark requires more than just evaluation of comprehension. The generation module uses 822 prompts to cover five tasks: dense generation (organizing high-density text and images), font generation (diverse fonts), style generation (controllable generation by target style), composition generation (reorganizing elements via specific operations), and intent generation (generating conceptual posters with visual metaphors). Instead of holistic aesthetic scores, the evaluation reuses human annotations from the understanding tasks: MLLMs check whether target design elements (compositional operations, metaphoric concepts, etc.) appear in the generated images to calculate coverage/match rates. This creates a closed-loop diagnosis where "understanding $\rightarrow$ generation" shares the same evaluation anchors.

### Loss & Training

PosterIQ is an evaluation benchmark, not a training method, and thus involves no training. MLLMs are used as automatic reviewers in generation evaluation to verify if the output contains the target elements.

## Key Experimental Results

### Main Results (Understanding Tasks)

| Model | Logo OCR | Poster OCR | Simple→Hard Δ↓ | Font Matching | Style Understanding | Composition Understanding | Intent Understanding |
|------|---------|------------|----------------|---------|---------|---------|---------|
| GPT-5 | 0.952 | 0.922 | 0.469 | 0.668 | 0.851 | 0.730 | 0.824 |
| Claude-4.5 | 0.902 | 0.884 | 0.372 | 0.699 | 0.813 | 0.608 | 0.761 |
| Gemini-2.5-Pro | 0.923 | 0.952 | 0.525 | 0.362 | 0.830 | **0.802** | 0.788 |
| Qwen3-VL-8B | 0.882 | 0.931 | **0.156** | 0.063 | 0.610 | 0.684 | 0.710 |
| MiniCPM-V-4.5 | 0.883 | 0.932 | 0.468 | -0.001 | 0.631 | 0.635 | 0.691 |

### Generation Comparison

| Model | Dense Gen | Font Diversity | Style Gen | Composition Gen | Intent Gen | Average |
|------|---------|-----------|---------|---------|---------|------|
| Seedream-4.0 | 0.618 | 0.342 | 0.591 | 0.848 | 0.645 | 0.609 |
| Gemini-2.5-Flash | **0.622** | **0.391** | 0.590 | **0.866** | 0.663 | **0.626** |
| GPT-Image-1 | 0.508 | 0.299 | **0.633** | 0.856 | **0.670** | 0.593 |
| Qwen-Image | 0.464 | 0.286 | 0.620 | 0.801 | 0.589 | 0.552 |

### Key Findings

- **Significant OCR Robustness Gaps**: Qwen3-VL-8B shows the smallest Simple$\rightarrow$Hard OCR gap ($\Delta=0.156$), indicating superior robustness to visual interference, though its font matching is near-random (0.063), reflecting weak fine-grained typographic perception.
- **Systemic Gap Between Closed and Open Source**: GPT-5 and Gemini-2.5-Pro consistently lead open-source models in advanced design understanding (style/composition/intent). Gemini is strongest in composition (0.802), while GPT-5 excels in intent interpretation (0.824).
- **Font Awareness is a Universal Weakness**: Models other than GPT-5 and Claude-4.5 perform near random levels in font matching, suggesting current MLLMs severely lack fine-grained recognition of typographic styles.
- **Font Diversity Bottleneck in Generation**: All models score $<0.4$ in font generation richness (best is Gemini at 0.391), whereas composition scores reach 0.866, marking font control as the weakest link in generative models.
- **Global-Local Trade-off**: Models proficient in composition and intent (global planning) often perform average in style and font (local precision), reflecting a inherent trade-off in current training paradigms.

## Highlights & Insights

- **Design Theory-Driven Benchmark Construction** is the primary highlight. Unlike technology-oriented benchmarks, PosterIQ defines capability dimensions based on design principles, ensuring every task is supported by explicit design theory.
- **Decoupled Evaluation vs. Holistic Scoring**: Experiments show that holistic model scores correlate poorly with human judgment (max $sim=0.483$), but specific sub-tasks provide meaningful diagnostic information.
- **VLM$\rightarrow$T2I Iterative Loop**: The experiments demonstrate how understanding capabilities directly improve generation quality—VLMs analyze design issues $\rightarrow$ modify prompts $\rightarrow$ regenerate $\rightarrow$ improved quality—offering a practical paradigm for design assistance.

## Limitations & Future Work

- Using MLLMs as automatic reviewers for generation quality faces a "sensitivity" issue, as MLLMs struggle to distinguish between good and bad designs in scoring tasks.
- Dataset scale is relatively limited (7,765 understanding annotations) and primarily targets English posters, with insufficient coverage of complex scripts like Chinese or Japanese.
- Lack of fine-grained user studies to verify if benchmark tasks truly reflect the dimensions designers care about.
- Composition generation evaluation uses coarse-grained concept matching, which may fail to capture subtle spatial design nuances.

## Related Work & Insights

- **vs. Creation-MMBench**: While Creation-MMBench evaluates general creativity, PosterIQ focuses on the highly constrained domain of poster design with deeper task alignment to design theory.
- **vs. GenEval/DPG-Bench**: Unlike general text-image alignment benchmarks, PosterIQ adds design-specific dimensions: typographic hierarchy, compositional techniques, and visual rhetoric.
- **vs. PosterLLaVA/COLE**: PosterIQ provides a standardized framework for evaluating these generative methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First benchmark to construct MLLM evaluation from a design theory perspective with deep task definitions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 24 task categories with 8 understanding and 4 generation models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear task definitions and deep analysis, though information density is very high.
- **Value**: ⭐⭐⭐⭐ Provides a systematic framework for evaluating design cognition in MLLMs and reveals key shortcomings like typographic perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SkyReels-Text: Fine-Grained Font-Controllable Text Editing for Poster Design](skyreels-text_fine-grained_font-controllable_text_editing_for_poster_design.md)
- [\[CVPR 2026\] SimplePoster: A Simple Baseline for Product Poster Generation](simpleposter_a_simple_baseline_for_product_poster_generation.md)
- [\[CVPR 2026\] InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation](innoads-composer_efficient_condition_composition_for_e-commerce_poster_generatio.md)
- [\[CVPR 2026\] GenColorBench: A Color Evaluation Benchmark for Text-to-Image Generation](gencolorbench_a_color_evaluation_benchmark_for_text-to-image_generation.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)

</div>

<!-- RELATED:END -->
