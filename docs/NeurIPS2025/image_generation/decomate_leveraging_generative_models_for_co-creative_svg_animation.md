---
title: >-
  [Paper Note] Decomate: Leveraging Generative Models for Co-Creative SVG Animation
description: >-
  [NeurIPS 2025 (GenProAI Workshop)][Image Generation][SVG animation] This paper proposes Decomate, an interactive system that leverages multimodal large language models (MLLMs) to automatically decompose unstructured SVG graphics into semantic components. Designers specify animation behaviors for each component via natural language, and the system generates production-ready HTML/CSS/JS animation code, supporting iterative co-creative workflows.
tags:
  - NeurIPS 2025 (GenProAI Workshop)
  - Image Generation
  - SVG animation
  - semantic decomposition
  - MLLM
  - co-creative design
  - natural language interaction
date: 2026-05-08
content_hash: 8659ab58f3f8df26
---

# Decomate: Leveraging Generative Models for Co-Creative SVG Animation

**Conference**: NeurIPS 2025 (GenProAI Workshop)  
**arXiv**: [2511.06297](https://arxiv.org/abs/2511.06297)  
**Code**: [HuggingFace Demo](https://huggingface.co/spaces/PrompTartLAB/Decomate)  
**Area**: Human-Computer Co-Creation / Creative Computing / SVG Animation  
**Keywords**: SVG animation, semantic decomposition, MLLM, co-creative design, natural language interaction  

## TL;DR
This paper proposes Decomate, an interactive system that leverages multimodal large language models (MLLMs) to automatically decompose unstructured SVG graphics into semantic components. Designers specify animation behaviors for each component via natural language, and the system generates production-ready HTML/CSS/JS animation code, supporting iterative co-creative workflows.

## Background & Motivation

**State of the Field**: SVG animation is an important yet technically demanding skill in UI/UX design. Designers must switch between multiple platforms, manually edit code, or rely on developers to implement motion effects, resulting in workflows that lack flexibility and fine-grained control.

**Limitations of Prior Work** (derived from interviews with 11 professional designers):
   - **(a) Time constraints**: Animation is frequently deprioritized due to tight schedules; many designers have little opportunity to explore and integrate motion effects.
   - **(b) Reliance on developers for fine-grained control**: Adjusting details such as timing and easing requires developer involvement, creating a disconnect between creative intent and final output.
   - **(c) Absence of AI tools**: All interviewees expressed interest in AI-assisted animation, yet none had used such tools — intuitive, designer-oriented solutions are lacking.

**Limitations of Prior Work**: Tools such as Keyframer support natural language–driven SVG animation generation but assume well-structured SVG input (with explicit class labels and groupings), whereas real-world design assets are typically flat and disorganized.

**Root Cause**: A significant gap exists between designers' creative intent (e.g., "make the wings flap slowly") and the underlying technical implementation (e.g., splitting SVG paths and writing keyframe animations).

## Method

### Overall Pipeline (6-Step Interaction Flow)

**Step 1: SVG Input**
- The user uploads an SVG file or pastes SVG code.
- The system accepts SVG structures ranging from fully flat to well-organized, requiring no preprocessing.
- The user provides a high-level object name (e.g., "dog") to guide semantic decomposition.

**Step 2: Semantic Decomposition + Animation Suggestions**
- An MLLM (Claude Sonnet 4) analyzes both the SVG code and the rendered image simultaneously.
- The SVG is restructured into semantically meaningful component groups (e.g., "ears," "nose," "legs").
- Visual interpretation is prioritized over syntactic hierarchy.
- Natural language animation suggestions are generated for each group.

**Step 3: Preview + Structural Refinement**
- The grouping scheme is visualized for the user.
- If the groupings do not align with creative intent, the user provides natural language feedback to refine them (e.g., "separate the left foot and right foot").
- The system iteratively revises the groupings based on this feedback.

**Step 4: Prompt-Driven Animation Authoring**
- Building on the system's suggestions, the user describes animation behaviors in natural language (e.g., "make the wings flap slowly with elastic easing").
- No knowledge of CSS animation syntax is required.

**Step 5: Code Generation and Rendering**
- An LLM (Claude Sonnet 4) translates the final grouping structure and animation prompts into HTML/CSS/JS code.
- The output is a deployment-ready animated SVG along with its source code.

**Step 6: Interactive Preview and Iteration**
- The user previews the animation in real time, inspecting timing, easing, and spatial behavior.
- Further refinements are made via follow-up prompts (e.g., "increase the bounce on landing").
- Each modification triggers code regeneration and preview updates.

### Technical Core

- **Semantic Decomposition**: The system does not rely on existing `<g>` tags or class names in the SVG — the MLLM leverages visual semantics to re-group SVG elements.
- **Dual-Model Architecture**: An MLLM handles visual and code understanding for decomposition, while a separate LLM handles code generation, with clearly separated responsibilities.
- **Iterative Co-Creative Loop**: Human–AI collaboration alternates at two levels — semantic structure and animation behavior.

## Key Experimental Results

### Formative Interviews (11 Professional Designers)

| Participant | Experience | Animation Tools | Attitude toward AI-Assisted Animation |
|-------------|------------|-----------------|---------------------------------------|
| P1–P4 | 1–10 years | Figma, After Effects, Code | Interested but no experience |
| P5–P8 | 3–10 years | Figma, Lottie, After Effects | Interested but no experience |
| P9–P11 | 1–10 years | Figma, Lottie | Interested but no experience |

100% of interviewees expressed interest in AI-assisted animation, while 0% had prior experience using such tools.

### User Study (Qualitative Feedback from 6 Participants)

| Dimension | Positive Feedback | Key Issues |
|-----------|-------------------|------------|
| Semantic Decomposition | Effective on flat/unstructured SVGs with minimal manual intervention | Incomplete grouping of some elements (e.g., whale eye detached from body) |
| Natural Language Animation | Reasonable intent-to-animation alignment overall | Users uncertain how to phrase prompts; learning curve present |
| Visual Quality | Correct directionality; basic effects acceptable | Insufficient fine-grained control (e.g., motion intensity, hierarchical timing) |
| Iterative Interaction | Natural language refinement effective | Some motions linearized and unnatural (e.g., swing vs. linear) |

### Representative User Feedback

- P2: "Object decomposition was good, but animation behavior was unexpected (the whale's eye detached)."
- P5: "Generating SVG animations directly from text is convenient, but without AI suggestions the motion effects can be very rigid."
- P6: "Direct UI controls (curves, duration) would be more usable than pure text input."

## Highlights & Insights

- **Addresses a core assumption flaw in tools like Keyframer**: No predefined SVG structure is required; MLLM-based visual understanding enables automatic decomposition.
- **Two-level iterative co-creation**: Natural language–driven iteration is supported at both the semantic structure and animation behavior levels, offering greater flexibility than single-level iteration.
- **Well-grounded requirements**: A closed-loop validation chain — 11-person interviews → system design → 6-person user study — grounds the design decisions in observed needs.
- **Practically deployable**: A live demo is available on HuggingFace, and outputs are deployment-ready HTML/CSS/JS.

## Limitations & Future Work

- **Prompt articulation difficulty**: Users are often uncertain how to phrase animation prompts — templates, examples, or autocomplete mechanisms are needed.
- **Insufficient fine-grained control**: Pure natural language struggles to precisely express subtle motion parameters (e.g., easing curves, exact durations) — structured controls such as sliders and toggles should be integrated.
- **Occasional semantic decomposition errors**: The MLLM may incorrectly group elements (e.g., body parts that should move together are separated), requiring stronger global consistency constraints.
- **Workshop paper scope**: Evaluation relies primarily on qualitative feedback; quantitative comparative experiments (e.g., controlled studies against Keyframer) are absent.
- **Dependency on a specific LLM (Claude Sonnet 4)**: Performance under model substitution is unknown, and prompt engineering is model-specific.

## Related Work & Insights

- **vs. Keyframer**: Keyframer assumes well-structured SVG input; Decomate adds a semantic decomposition step to handle unstructured SVGs.
- **vs. Kolthoff et al. (zero-shot GUI prototyping)**: Focuses on GUI generation rather than SVG animation and does not include semantic decomposition.
- **vs. Xing et al. (SVG understanding/generation)**: Provides datasets to enhance LLM SVG capabilities but does not address animation authoring workflows.
- **vs. Traditional tools (Figma + Lottie + After Effects)**: Traditional tools demand manual operation and technical expertise; Decomate lowers the barrier but sacrifices fine-grained control.

## Rating
- **Novelty**: ⭐⭐⭐ — The combination of semantic decomposition and prompt-driven animation constitutes incremental innovation, with core capabilities relying on LLM capacity.
- **Experimental Thoroughness**: ⭐⭐⭐ — Includes needs assessment and user study, but evaluation is qualitative only, with a small sample (n=6) and no quantitative comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; pipeline diagrams and interface screenshots are detailed; user feedback is presented verbatim.
- **Value**: ⭐⭐⭐ — Meaningful as a workshop demonstration, but considerable distance remains from a reliable production design tool.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[ICCV 2025\] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models](../../ICCV2025/image_generation/revelio_interpreting_and_leveraging_semantic_information_in_diffusion_models.md)
- [\[ICCV 2025\] CAP: Evaluation of Persuasive and Creative Image Generation](../../ICCV2025/image_generation/cap_evaluation_of_persuasive_and_creative_image_generation.md)
- [\[AAAI 2026\] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](../../AAAI2026/image_generation/pase_leveraging_the_phonological_prior_of_wavlm_for_low-hallucination_generative.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)

<!-- RELATED:END -->
