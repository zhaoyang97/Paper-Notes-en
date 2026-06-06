---
title: >-
  [Paper Note] Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions
description: >-
  [ACL 2026][Multimodal VLM][Slide Automatic Updating] Defining a new task of "dynamic slide updating on user-defined templates based on natural language instructions," this work constructs the DynaSlide benchmark with 20…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Slide Automatic Updating"
  - "Dynamic Templates"
  - "Natural Language Instructions"
  - "Multimodal Agent"
  - "Data-driven Reports"
date: 2026-05-08
content_hash: a8a58dc764cce8ff
---

# Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions

**Conference**: ACL 2026  
**arXiv**: [2604.17894](https://arxiv.org/abs/2604.17894)  
**Code**: [github](https://github.com/XiaoZhou2024/SlideAgent)  
**Area**: Multimodal/VLM  
**Keywords**: Slide Automatic Updating, Dynamic Templates, Natural Language Instructions, Multimodal Agent, Data-driven Reports

## TL;DR

Defining a new task of "dynamic slide updating on user-defined templates based on natural language instructions," this work constructs the DynaSlide benchmark with 20,036 instruction-execution triplets and proposes SlideAgent as a strong reference baseline.

## Background & Motivation

**Background**: Presentation slides serve as the core medium for data-driven reporting, yet maintaining complex analytical slides remains extremely labor-intensive. Existing automation methods primarily adopt fixed-template filling paradigms, failing to support diverse user-defined slides.

**Limitations of Prior Work**: (1) In periodic business reports, updates usually involve only local data substitution and conclusion fine-tuning, but significant manpower is consumed by low-value "copy-paste-modify" workflows; (2) Existing methods are limited to injecting information from structured data sources into fixed templates, failing to handle complex slide structures created by users.

**Key Challenge**: The Bring-Your-Own-template (BYO-template) scenario requires systems to understand the multimodal structure of arbitrary slides (titles, tables, charts, summaries, as well as their layouts and dependencies), while precisely mapping natural language update instructions to executable operations—which far exceeds simple value replacement.

**Goal**: To formally define the dynamic slide updating task, construct a large-scale benchmark dataset, and propose an Agent baseline system.

**Key Insight**: Based on real-world real estate business analysis data, a controllable template family is constructed to generate a large number of instruction-execution triplets, supporting reproducible evaluation.

**Core Idea**: Slide updating is modeled as a closed-loop process of perception-reasoning-execution: first parsing the semantic structure and data logic of the slide, then updating data queries, recalculating statistical results, redrawing charts, and rewriting summaries based on natural language instructions, while maintaining the original layout and style.

## Method

### Overall Architecture

SlideAgent adopts a two-stage architecture: Stage 1 (Slide Understanding) parses the input slide into a structured representation, capturing element positions, data sources, and functional logic; Stage 2 (Instruction-Driven Updating) interprets user instructions, retrieves updated data, executes transformations, and regenerates content.

### Key Designs

1.  **Multimodal Slide Layout Parsing**:
    - **Function**: Identifies the semantic roles and spatial structures of various elements in the slide.
    - **Mechanism**: Renders the slide as a PNG image and uses Qwen2.5-VL-72B to predict semantic labels and bounding boxes; simultaneously, `python-pptx` parses precise coordinates and style metadata; VLM predictions are aligned with PPTX shapes through IoU matching (threshold 0.5).
    - **Design Motivation**: VLM provides semantic understanding (distinguishing titles/table titles/summaries, etc.), while `python-pptx` provides precise geometry and style information; the two complement each other.

2.  **Logic Extraction for Tables and Charts (Closed-Domain/Open-Domain)**:
    - **Function**: Reverse-engineers underlying data queries and aggregation logic from slides.
    - **Mechanism**: Closed-domain—LLM identifies corresponding functions and parameters from a library of 11 predefined statistical functions (function calling paradigm); Open-domain—a general `synthesize_analytical_table` interface is designed to reconstruct logic from five atomic components: table structure type, headers, constraint specifications, source fields, and aggregation operations.
    - **Design Motivation**: Closed-domain covers known templates, while open-domain handles arbitrary user-defined analyses; the two modes are complementary.

3.  **Instruction-Driven Content Synchronization Pipeline**:
    - **Function**: Executes end-to-end slide updates based on natural language instructions.
    - **Mechanism**: A four-step pipeline—Instruction Parsing (mapping instructions to parameter state updates) → SQL Generation and Data Retrieval → Tool Calling and Data Recalculation → Fact-aware Summary Rewriting and Final Rendering.
    - **Design Motivation**: Decomposes the complex updating process into independently evaluable sub-modules, facilitating the localization of error bottlenecks.

### Loss & Training

The proposed method is primarily based on LLM inference rather than training. Evaluation utilizes the success rate (SR, the proportion of generated slides that perfectly match the ground truth in content and layout) and element-level accuracy.

## Key Experimental Results

### Main Results

| Model | Closed-Domain SR (%) | Open-Domain SR (%) |
|------|-------------|-------------|
| GPT-OSS-120B | 80.64 | 68.86 |
| Qwen3-80B | 75.33 | 63.91 |
| GPT-OSS-20B | 69.20 | 56.25 |
| Qwen3-30B | 71.40 | 59.69 |
| Qwen3-14B | 45.48 | 31.13 |

### Ablation Study

| Module (GPT-OSS-120B, Open-Domain) | Accuracy (%) | Description |
|---------------------------|-----------|------|
| Layout Parsing | 99.5 | Most stable module |
| Function Logic Extraction | 88.34 | High accuracy |
| Data Source Extraction | 90.37 | High accuracy |
| Summary Update | 68.44 | Largest bottleneck |
| End-to-End Task SR | 68.86 | Error accumulation effect |

### Key Findings
- Model scale is strongly correlated with task performance: GPT-OSS-120B is 11-12 percentage points higher than 20B, and Qwen3-80B is approximately 30 percentage points higher than 14B.
- Open-domain scenarios consistently lead to performance degradation, with a more significant impact on smaller models (Qwen3-14B relative drop of 31.5%).
- Summary updating is the largest bottleneck (68.44%), significantly lower than logic extraction (88.34%)—models effectively extract calculation logic, but translating quantitative updates into coherent natural language conclusions remains a fundamental challenge.
- Task difficulty varies significantly by theme: Simple table structures (Theme 1: 90.12%) vs. complex cross-dimensional aggregations (Theme 4: 77.03%).

## Highlights & Insights
- The definition of the new task has strong practical value—periodic report updating is a real and high-frequency demand in enterprises.
- DynaSlide benchmark design is exquisite: controllable template families ensure verifiable ground truth, and YAML metadata supports reproducible end-to-end evaluation.
- The comparative design of closed-domain/open-domain effectively reveals the boundaries of model generalization capabilities.
- The module-level evaluation protocol provides a clear diagnostic framework for identifying error bottlenecks.

## Limitations & Future Work
- Only covers the real estate domain, although the core mechanism is domain-independent.
- Uses controllable templates rather than completely wild slides, sacrificing some stylistic diversity for verifiability.
- Assumes slide elements can be linked to structured databases and does not handle "cold start" problems (rebuilding databases from static slides).
- Does not handle decorative graphics or conceptual diagrams.

## Related Work & Insights
- **vs AutoPresent/PPTAgent**: They focus on one-time document-to-slide generation, whereas this work focuses on dynamic updates on user-defined templates.
- **vs Traditional template filling methods**: They use fixed predefined templates and cannot handle user-created complex layouts.
- **vs LLM Agent methods (e.g., Yao et al.)**: They update surface content but cannot reconstruct underlying computational dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally defines the dynamic slide updating task for the first time, opening a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-theme, module-level evaluation, but limited to a single domain.
- Writing Quality: ⭐⭐⭐⭐ Task definition is clear, and the dataset construction process is detailed.
- Value: ⭐⭐⭐⭐ The task is highly practical, and the benchmark dataset provides a continuous contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)
- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)
- [\[ACL 2026\] Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding](do_mllms_capture_how_interfaces_guide_user_behavior_a_benchmark_for_multimodal_u.md)
- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](../../AAAI2026/multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)

</div>

<!-- RELATED:END -->
