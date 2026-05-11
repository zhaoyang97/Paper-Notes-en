---
title: >-
  [Paper Note] Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions
description: >-
  [ACL 2026][Multimodal VLM][automatic slide updating] This paper formalizes a novel task of dynamic slide updating on user-defined templates guided by natural language instructions…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "automatic slide updating"
  - "dynamic templates"
  - "natural language instructions"
  - "multimodal agent"
  - "data-driven reporting"
date: 2026-05-08
content_hash: 0ea1eedf2f8eccd5
---

# Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions

**Conference**: ACL 2026
**arXiv**: [2604.17894](https://arxiv.org/abs/2604.17894)
**Code**: [github](https://github.com/XiaoZhou2024/SlideAgent)
**Area**: Multimodal/VLM
**Keywords**: automatic slide updating, dynamic templates, natural language instructions, multimodal agent, data-driven reporting

## TL;DR

This paper formalizes a novel task of dynamic slide updating on user-defined templates guided by natural language instructions, constructs the DynaSlide benchmark comprising 20,036 instruction-execution triplets, and proposes SlideAgent as a strong reference baseline.

## Background & Motivation

**Background**: Presentation slides serve as the primary medium for data-driven reporting, yet maintaining complex analytical slide decks remains extremely labor-intensive. Existing automation approaches predominantly adopt a fixed-template filling paradigm, which cannot accommodate diverse user-defined slides.

**Limitations of Prior Work**: (1) In periodic business reports, updates typically involve only partial data replacement and minor revision of conclusions, yet substantial human effort is consumed by low-value copy-paste-modify workflows; (2) existing methods are limited to injecting information from structured data sources into fixed templates and cannot handle complex slide structures authored by end users.

**Key Challenge**: The bring-your-own-template (BYO-template) setting requires a system to comprehend the multimodal structure of arbitrary slides—titles, tables, charts, summaries, and their spatial layout and inter-element dependencies—while accurately mapping natural language update instructions to executable operations, which goes far beyond simple value substitution.

**Goal**: To formally define the dynamic slide updating task, construct a large-scale benchmark dataset, and propose an agent-based baseline system.

**Key Insight**: The task is grounded in real-world real-estate business analytics data, where a controlled family of templates is constructed to generate large numbers of instruction-execution triplets that support reproducible evaluation.

**Core Idea**: Slide updating is modeled as a closed-loop perceive–reason–execute process: the system first parses the semantic structure and data logic of the slide, then updates data queries, recomputes statistical results, redraws charts, and rewrites summaries according to natural language instructions, while preserving the original layout and style.

## Method

### Overall Architecture

SlideAgent adopts a two-stage architecture. Stage 1 (slide understanding) parses the input slide into a structured representation capturing element positions, data sources, and functional logic. Stage 2 (instruction-driven updating) interprets user instructions, retrieves updated data, executes transformations, and regenerates content.

### Key Designs

1. **Multimodal Slide Layout Parsing**:

    - Function: Identifies the semantic roles and spatial structure of elements within a slide.
    - Mechanism: The slide is rendered as a PNG image; Qwen2.5-VL-72B is used to predict semantic labels and bounding boxes. Simultaneously, python-pptx extracts precise coordinates and style metadata. VLM predictions are aligned with PPTX shapes via IoU matching (threshold 0.5).
    - Design Motivation: The VLM provides semantic understanding (distinguishing titles, table headers, summaries, etc.) while python-pptx supplies precise geometric and style information, making the two sources complementary.

2. **Table and Chart Logic Extraction (Closed-Domain / Open-Domain)**:

    - Function: Reverse-engineers the underlying data queries and aggregation logic from the slide.
    - Mechanism: In the closed-domain setting, an LLM identifies the corresponding function and parameters from a predefined library of 11 statistical functions (function-calling paradigm). In the open-domain setting, a general-purpose `synthesize_analytical_table` interface is designed to reconstruct logic from five atomic components: table structure type, column headers, constraint specifications, source fields, and aggregation operations.
    - Design Motivation: The closed-domain mode covers known templates while the open-domain mode handles arbitrary user-defined analyses; the two modes are mutually complementary.

3. **Instruction-Driven Content Synchronization Pipeline**:

    - Function: Executes end-to-end slide updates according to natural language instructions.
    - Mechanism: A four-step pipeline comprising instruction parsing (mapping instructions to parameter state updates) → SQL generation and data retrieval → tool invocation and data recomputation → fact-aware summary rewriting and final rendering.
    - Design Motivation: Decomposing the complex update process into independently evaluable sub-modules facilitates precise identification of error bottlenecks.

### Loss & Training

The proposed method relies primarily on LLM inference rather than model training. Evaluation employs task Success Rate (SR)—the proportion of generated slides that exactly match the ground truth in both content and layout—along with element-level accuracy.

## Key Experimental Results

### Main Results

| Model | Closed-Domain SR (%) | Open-Domain SR (%) |
|-------|---------------------|-------------------|
| GPT-OSS-120B | 80.64 | 68.86 |
| Qwen3-80B | 75.33 | 63.91 |
| GPT-OSS-20B | 69.20 | 56.25 |
| Qwen3-30B | 71.40 | 59.69 |
| Qwen3-14B | 45.48 | 31.13 |

### Ablation Study

| Module (GPT-OSS-120B, Open-Domain) | Accuracy (%) | Note |
|------------------------------------|-------------|------|
| Layout parsing | 99.5 | Most stable module |
| Function logic extraction | 88.34 | High accuracy |
| Data source extraction | 90.37 | High accuracy |
| Summary updating | 68.44 | Primary bottleneck |
| End-to-end task SR | 68.86 | Error accumulation effect |

### Key Findings
- Model scale is strongly correlated with task performance: GPT-OSS-120B outperforms the 20B variant by 11–12 percentage points, and Qwen3-80B surpasses Qwen3-14B by approximately 30 points.
- The open-domain setting consistently degrades performance, with a more pronounced effect on smaller models (Qwen3-14B experiences a relative drop of 31.5%).
- Summary updating is the primary bottleneck (68.44%), substantially below logic extraction (88.34%), indicating that while models can effectively extract computational logic, translating quantitative updates into coherent natural language conclusions remains a fundamental challenge.
- Task difficulty varies considerably across themes: simpler tabular structures (Theme 1: 90.12%) versus complex cross-dimensional aggregations (Theme 4: 77.03%).

## Highlights & Insights
- The newly defined task carries strong practical value, as periodic report updating is a genuine and frequent enterprise need.
- The DynaSlide benchmark is elegantly designed: the controlled template family ensures verifiable ground truth, and YAML metadata supports reproducible end-to-end evaluation.
- The closed-domain vs. open-domain comparative design effectively reveals the boundaries of model generalization capability.
- The module-level evaluation protocol provides a clear diagnostic framework for identifying error bottlenecks.

## Limitations & Future Work
- Coverage is limited to the real-estate domain, although the core mechanism is domain-agnostic.
- The use of controlled templates rather than fully in-the-wild slides sacrifices some stylistic diversity in exchange for verifiability.
- The approach assumes that slide elements can be associated with a structured database and does not address the cold-start problem of reconstructing a database from static slides.
- Decorative graphics and conceptual diagrams are not handled.

## Related Work & Insights
- **vs. AutoPresent/PPTAgent**: These works focus on one-shot document-to-slide generation, whereas this paper addresses dynamic updating on user-defined templates.
- **vs. traditional template filling methods**: Such methods rely on fixed predefined templates and cannot accommodate complex layouts created by end users.
- **vs. LLM agent approaches (e.g., Yao et al.)**: These update surface-level content but cannot reconstruct the underlying computational dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal definition of the dynamic slide updating task, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-theme, and module-level evaluation, though restricted to a single domain.
- Writing Quality: ⭐⭐⭐⭐ Task definition is clear and the dataset construction process is described in thorough detail.
- Value: ⭐⭐⭐⭐ Strong practical utility; the benchmark dataset offers a lasting contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)
- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)

</div>

<!-- RELATED:END -->
