---
title: >-
  [Paper Note] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][UI Code Generation] This paper is the first to formally define the Widget-to-Code task, constructing the first image-only widget dataset and a multi-dimensional evaluation framework. It proposes a modular baseline built upon a Perceptual Agent and the WidgetFactory infrastructure, achieving high-fidelity widget reconstruction through component decomposition, icon retrieval, reusable visualization templates, and adaptive rendering.
tags:
  - CVPR 2026
  - Multimodal VLM
  - UI Code Generation
  - Widget Reconstruction
  - Multimodal Large Language Models
  - Domain-Specific Language
  - Visual Perception
date: 2026-05-08
content_hash: 3d262dd080a3f933
---

# Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs

**Conference**: CVPR 2026
**arXiv**: [2512.19918](https://arxiv.org/abs/2512.19918)
**Code**: [https://djanghao.github.io/widget2code](https://djanghao.github.io/widget2code)
**Area**: Multimodal VLM
**Keywords**: UI Code Generation, Widget Reconstruction, Multimodal Large Language Models, Domain-Specific Language, Visual Perception

## TL;DR
This paper is the first to formally define the Widget-to-Code task, constructing the first image-only widget dataset and a multi-dimensional evaluation framework. It proposes a modular baseline built upon a Perceptual Agent and the WidgetFactory infrastructure, achieving high-fidelity widget reconstruction through component decomposition, icon retrieval, reusable visualization templates, and adaptive rendering.

## Background & Motivation

1. **State of the Field**: UI-to-Code (UI2Code) is a key direction in automated front-end development, aiming to generate executable code from visual designs. With advances in multimodal large language models (MLLMs), the field has shifted from rule-based/supervised pipelines to MLLM-driven approaches. Existing work primarily targets web and mobile UIs, which benefit from rich hierarchical context and available annotated data.

2. **Limitations of Prior Work**: Widgets (small UI components such as weather cards and calendar widgets) represent a distinct class of micro-interfaces—they are compact, context-free, and convey information through dense typography and icons under strict spatial constraints. However, widget designs are proprietary, with no publicly available source code, annotations, or (image, code) paired datasets. Existing UI2Code methods optimized for web/mobile UIs transfer poorly to widgets.

3. **Root Cause**: Widgets exhibit high visual complexity (containing icons, data visualizations, and artistic color schemes) while being extremely spatially compact, with no available training supervision. Although general-purpose MLLMs outperform specialized UI2Code methods, they still produce visually inconsistent and structurally unreliable code—content overflow, size mismatches, and color deviation are pervasive issues.

4. **Paper Goals**: (a) Formally define the Widget2Code task and establish evaluation standards; (b) bridge the gap between pixel-level perception and geometry-aware code generation; (c) establish a unified baseline framework and infrastructure.

5. **Starting Point**: At the perception level—decompose widgets into atomic components following Apple widget design principles; at the system level—design a widget-specific DSL and compiler to replace direct generation of verbose code.

6. **Core Idea**: By combining component decomposition via a Perceptual Agent with a DSL intermediate representation in WidgetFactory, widget reconstruction is transformed from end-to-end MLLM code generation into a structured, controllable pipeline.

## Method

### Overall Architecture
The system consists of three parts: (1) **Data Curation**—collecting and processing widget images from platforms such as Figma and Dribbble to construct a benchmark; (2) **Perceptual Agent**—decomposing input widgets into atomic components and extracting visual, semantic, and stylistic cues; (3) **WidgetFactory**—an end-to-end infrastructure comprising DSL generation, compilation, and adaptive rendering, converting WidgetDSL into executable code.

### Key Designs

1. **Perceptual Agent**:

    - **Function**: Analyzes input widgets at the perceptual level and decomposes them into atomic components.
    - **Mechanism**: Comprises four sub-modules. **Component Extraction**: an MLLM detects and classifies visual components (icons, buttons, text, charts, etc.), with each component represented as $e=[r, b, t, c]$ (cropped region, bounding box, text description, category). **Icon Retrieval**: a 50k SVG icon library is constructed; SigLIP is used to extract visual and text embeddings; coarse retrieval of Top-50 candidates via visual similarity is followed by re-ranking with text similarity to obtain Top-5, avoiding semantic hallucinations in direct MLLM icon generation. **Reusable Component Templates**: a DSL-formatted template library is defined; an MLLM matches detected components to templates and fills in parameters. **Color Extraction**: the image is converted to a perceptually uniform color space, and K-means clustering extracts dominant colors and their proportions $\mathcal{P} = \{(\mu_k, w_k)\}_{k=1}^K$.
    - **Design Motivation**: End-to-end MLLM code generation leads to icon errors, chart distortions, and color deviations. Decomposing the problem into independent modules—each using the most suitable method—significantly improves perceptual accuracy.

2. **WidgetDSL and Code Generation**:

    - **Function**: Replaces direct generation of verbose HTML/CSS/JS with a compact, interpretable intermediate representation.
    - **Mechanism**: WidgetDSL encodes a widget as a parameterized component tree, where nodes correspond to functional units (icons, charts, text blocks) and attributes specify geometry, color, and style. DSL generation applies multi-stage constraint composition, progressively injecting into a base prompt: (1) layout bounding-box constraints, (2) color palette, (3) component specifications (to prevent chart data hallucinations), (4) icon candidate sets, and (5) dynamically inferred component types. A two-stage compiler pipeline (DSL → AST → target code) deterministically converts the representation into HTML/CSS or React code.
    - **Design Motivation**: Direct MLLM code generation produces redundant, uncontrollable outputs. The DSL intermediate representation strikes a balance between conciseness and expressiveness, reducing hallucinations and redundancy while supporting cross-platform compilation.

3. **Adaptive Rendering**:

    - **Function**: Preserves aspect ratio and prevents content overflow while maintaining spatial compactness.
    - **Mechanism**: Given the input widget aspect ratio $r = w/h$, a feedback-guided binary search optimizes the width $w$ such that the violation function $\Psi(w) \leq 0$. $\Psi(w)$ aggregates layout feedback from the browser rendering engine, including whether content exceeds the viewport and whether child elements overflow their bounds. Each rendering pass reports $\Psi(w_t)$, and the width is iteratively adjusted until convergence at $|\Psi(w_t)| < \epsilon$ (approximately one pixel of tolerance).
    - **Design Motivation**: Existing methods broadly neglect the rendering stage; directly rendering MLLM-generated code leads to aspect ratio distortion, size mismatches, and content overflow. The closed-loop optimization of adaptive rendering addresses this critical issue.

### Evaluation Metric Design
Five fine-grained evaluation dimensions are designed based on Apple's Human Interface Guidelines:
- **Layout**: margin symmetry, content aspect-ratio similarity, area-ratio similarity
- **Legibility**: text Jaccard, contrast similarity, local contrast similarity
- **Style**: palette distance, vibrancy consistency, polarity consistency
- **Perceptual**: SSIM, LPIPS, CLIP Score
- **Geometry**: aspect ratio and normalized size comparison

## Key Experimental Results

### Main Results

| Method | Layout-Margin | Layout-Content | Layout-Area | Text | Palette | SSIM | Geometry |
|------|-------------|---------------|------------|------|---------|------|----------|
| GPT-4o | 63.48 | 59.04 | 64.41 | 60.20 | 47.03 | 0.698 | 91.93 |
| Gemini2.5-Pro | 65.35 | 62.74 | 79.66 | 59.48 | 48.99 | 0.701 | 90.25 |
| Qwen3-VL | 64.75 | 60.15 | 69.53 | 61.17 | 47.44 | 0.703 | 95.15 |
| Design2Code | 36.34 | 47.81 | 49.68 | 17.50 | 30.92 | 0.512 | 15.72 |
| DCGen | 43.17 | 40.14 | 64.55 | 50.36 | 35.56 | 0.598 | 31.59 |
| **Widget2Code (Ours)** | **72.15** | **66.08** | **82.24** | **70.60** | **58.09** | **0.721** | **100.00** |

### Ablation Study

| Configuration | Margin | Content | Area | Text | Palette | SSIM | Geometry |
|------|--------|---------|------|------|---------|------|----------|
| Qwen3-VL baseline | 64.75 | 60.15 | 69.53 | 61.17 | 47.44 | 0.703 | 95.15 |
| + WidgetFactory | 69.97 | 64.60 | 82.46 | 67.99 | 42.36 | 0.683 | 100 |
| + Components | 70.83 | 64.90 | 82.30 | 67.49 | 42.61 | 0.676 | 100 |
| + Color analysis | 71.29 | 65.43 | 83.03 | 68.62 | 57.56 | 0.705 | 100 |
| + Layout | 71.49 | 65.33 | 81.92 | 68.89 | 56.10 | 0.710 | 100 |
| + Icon (Full) | 72.15 | 66.08 | 82.24 | 70.60 | 58.09 | 0.721 | 100 |

### Key Findings
- Specialized UI2Code models (Design2Code, DCGen, etc.) perform substantially worse than general-purpose MLLMs on widgets, indicating that widgets require dedicated design.
- WidgetFactory contributes most to the Geometry metric—lifting it directly from 95.15 to 100, achieving perfect widget size reproduction.
- The color analysis module contributes most to the Palette metric (42.61 → 57.56, +35%), demonstrating that MLLMs have significant deficiencies in color reproduction.
- Icon retrieval yields notable gains in Text and SSIM, confirming that accurate icons are critical to visual fidelity.

## Highlights & Insights
- Formally defining the Widget2Code task is an important contribution—widget micro-interfaces are fundamentally distinct from web/mobile UIs (compact, context-free, icon-dense) and merit study as an independent task.
- The strategy of replacing direct icon generation with retrieval is highly practical and effective—MLLMs are inherently unreliable at generating fine-grained icons, whereas retrieval is both accurate and controllable.
- WidgetDSL serves as an intermediate representation bridging perceptual understanding and code generation, analogous to IR in compiler design; this pattern is transferable to other complex code generation tasks.
- The browser-feedback closed-loop optimization in adaptive rendering is a simple yet universally overlooked critical step.

## Limitations & Future Work
- The dataset scale is moderate (2,825 widgets, 1,000 test samples); while comparable to similar benchmarks, a larger scale may surface additional issues.
- The work focuses on static widgets and does not address interactive or dynamic widget behaviors.
- Reliance on the Qwen3-VL API as the backbone model may pose cost and latency bottlenecks for practical deployment.
- Although the evaluation metrics are fine-grained, functional correctness assessment is absent (i.e., whether the generated code is truly interactive).
- WidgetFactory could be used as a data engine to synthesize (image, code) training pairs for end-to-end fine-tuning.

## Related Work & Insights
- **vs Design2Code**: Design2Code targets full-page website design; the compactness of widgets renders it ineffective. Widget2Code's component decomposition strategy is better suited to dense layouts.
- **vs DCGen**: DCGen applies divide-and-conquer prompting to web pages, but widgets lack a divisible hierarchical structure. WidgetDSL provides a more appropriate structured representation.
- **vs GPT-4o / Gemini**: General-purpose MLLMs offer stronger visual understanding but generate redundant and uncontrollable code. Widget2Code amplifies their strengths through DSL constraints and modular perception.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to formally define the Widget2Code task; proposes a dedicated evaluation framework and DSL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons (general MLLMs + specialized UI2Code methods) with clear ablations, though the dataset is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and detailed system design descriptions.
- **Value**: ⭐⭐⭐⭐ Opens a new task direction and provides a complete benchmark + baseline + infrastructure.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)

<!-- RELATED:END -->
