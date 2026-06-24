---
title: >-
  [Paper Note] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting
description: >-
  [ACL 2025][LLM (Other)][Document Parsing] This work proposes Dolphin, a lightweight (322M) document image parsing model that adopts an "analyze-then-parse" two-stage paradigm. It first performs page-level layout analysis to generate an element sequence in reading order, and then utilizes heterogeneous anchor prompting to parse the content of each element in parallel. With only 322M parameters, it outperforms 7B+ models and commercial systems on both page-level and element-lev…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Document Parsing"
  - "Layout Analysis"
  - "Multimodal VLM"
  - "Parallel Decoding"
  - "Anchor Prompting"
date: 2026-05-08
content_hash: 73afd4c5726dc538
---

# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

**Conference**: ACL 2025  
**arXiv**: [2505.14059](https://arxiv.org/abs/2505.14059)  
**Code**: [https://github.com/ByteDance/Dolphin](https://github.com/ByteDance/Dolphin)  
**Area**: Document Understanding / OCR  
**Keywords**: Document Parsing, Layout Analysis, Multimodal VLM, Parallel Decoding, Anchor Prompting

## TL;DR
This work proposes Dolphin, a lightweight (322M) document image parsing model that adopts an "analyze-then-parse" two-stage paradigm. It first performs page-level layout analysis to generate an element sequence in reading order, and then utilizes heterogeneous anchor prompting to parse the content of each element in parallel. With only 322M parameters, it outperforms 7B+ models and commercial systems on both page-level and element-level parsing tasks.

## Background & Motivation

**Background**: Document image parsing (extracting structured content from images) has two main technical routes: (a) Integrated pipelines (e.g., MinerU), which combine expert models for layout detection, OCR, table/formula recognition, etc., providing high accuracy but complex integration; (b) End-to-end autoregressive schemes (e.g., Nougat, GOT), which directly generate page-level content using VLMs, offering simplicity but suffering from efficiency and layout degradation issues.

**Limitations of Prior Work**: Integrated pipelines require independent optimization of each model, making coordination across components difficult; autoregressive schemes are prone to layout structure degradation when handling long documents with complex layouts (e.g., mixed tables, formulas, and images), and token-by-token generation seriously hampers efficiency.

**Key Challenge**: How to maintain the simplicity of end-to-end training while avoiding layout degradation and efficiency bottlenecks of pure autoregressive methods on complex documents?

**Goal**: (1) A unified document parsing framework supporting multi-granularity tasks; (2) Parallel parsing to improve efficiency; (3) A lightweight architecture.

**Key Insight**: Decompose document parsing into two stages: layout analysis (global structure understanding) and element parsing (local content recognition), using layout elements as "anchors" to guide parallel parsing.

**Core Idea**: A two-stage "analyze-then-parse" approach—the first stage generates a sequence of layout elements (type + position + reading order), and the second stage crops the elements and feeds them into the same model for parallel parsing, adapting to different element types using type-specific prompts (heterogeneous anchor prompting).

## Method

### Overall Architecture
Input: Document image → Output: Structured Markdown/HTML text. Both stages share a unified encoder-decoder VLM (Swin Transformer encoder + mBart decoder, totaling 322M parameters).

### Key Designs

1. **Stage 1: Page-level Layout Analysis**:

    - **Function**: Given the full-page image, generate a sequence of layout elements $L = \{l_1, l_2, ..., l_n\}$ arranged in reading order, where each element contains its type (paragraph/table/formula/figure/heading, etc.) and bounding box.
    - **Mechanism**: A Swin Transformer encodes the page image into visual features, and an mBart decoder autoregressively generates the structured layout sequence with the prompt "Parse the reading order of this document.", preserving hierarchical relationships (e.g., figure-caption, table-caption, section heading-paragraph correspondences).
    - **Design Motivation**: Obtain global layout information first to provide "anchors" for subsequent parallel parsing; predicting reading order is a key advantage of end-to-end methods.

2. **Stage 2: Element-level Content Parsing (Parallel)**:

    - **Function**: Crop local image regions of each element according to the layout results from Stage 1, and parse all elements in parallel.
    - **Mechanism**: For each element $l_i$, crop the corresponding region $I_i$ from the original image, encode it using the same Swin Transformer, and then use **type-specific prompts** (tables → HTML format; formulas and paragraphs → Markdown/LaTeX format) to guide the decoder in content generation. Multiple elements are processed concurrently via batch inference.
    - **Design Motivation**: Cropping and focusing ("what you see is what you get") is simpler and more accurate than direct localization and recognition on the full-page image; parallel decoding is 1.8× faster than pure autoregressive decoding.

3. **Heterogeneous Anchor Prompting**:

    - **Function**: Design distinct parsing prompts for different types of elements.
    - **Mechanism**: Tables use a dedicated $P_{table}$ prompt to parse HTML structure; text paragraphs and formulas share a $P_{paragraph}$ prompt (since formulas are often embedded in paragraphs); type information comes from the layout analysis results of Stage 1.
    - **Design Motivation**: Ablation studies show that type-specific prompts vs. generic prompts reduce the Edit Distance (ED) from 0.1613 to 0.1028 (a 36% decrease). Generic prompts often cause misclassifications (e.g., recognizing tables as LaTeX formulas).

### Training Data
Constructed a large-scale dataset with 30M+ samples:
- Page-level: 0.12M mixed documents (with layout annotations), 4.37M HTML renders, 0.5M LaTeX, 0.71M Markdown.
- Element-level: 1.57M tables (PubTabNet + PubTab1M), 23M formulas (arXiv LaTeX renders).
- Data advantage of the decoupled element strategy: Collecting isolated element images is much easier than collecting fully annotated complex pages.

### Loss & Training
- Standard seq2seq cross-entropy loss.
- AdamW optimizer, learning rate 5e-5, cosine decay.
- 40 A100 GPUs, 2 epochs, batch size of 16 per GPU.
- Input images are scaled to 896×896 maintaining aspect ratio (longest side 896 + padding).

## Key Experimental Results

### Main Results (Page-level Parsing, Edit Distance ↓)

| Model | Parameters | Plain Doc EN | Plain Doc ZH | Complex Doc | Avg ED | FPS ↑ |
|------|--------|-------------|-------------|-------------|--------|-------|
| MinerU (Integrated) | 1.2B | 0.0685 | 0.0702 | 0.2770 | 0.1732 | 0.035 |
| GOT | 580M | 0.035 | 0.038 | 0.2459 | 0.1411 | 0.060 |
| Qwen2.5-VL | 7B | 0.0135 | 0.0270 | 0.2025 | 0.1112 | 0.034 |
| Mistral-OCR | - | 0.0138 | 0.0252 | 0.1283 | 0.0737 | 0.100 |
| **Dolphin** | **322M** | **0.0114** | **0.0131** | **0.1028** | **0.0575** | **0.173** |

### Ablation Study (Dolphin-Page, ED ↓)

| Configuration | ED | FPS |
|------|-----|-----|
| **Full model** | **0.1028** | **0.1729** |
| Parallel → Sequential | 0.0971 | 0.0971 (↓44%) |
| Type-specific → Generic prompts | 0.1613 (↑57%) | - |
| Element cropping → Box query | 0.1849 (↑80%) | - |

### Key Findings
- **Lightweight yet powerful**: The 322M model comprehensively outperforms 7B+ VLMs (Qwen2.5-VL, GPT-4o, Claude 3.5) and commercial systems (Mathpix), with is particularly pronounced advantages on complex documents.
- **Parallel decoding yields 1.8× speedup**: Maintains Edit Distance (ED) while increasing FPS from 0.0971 to 0.1729, with further potential for acceleration using off-the-shelf parallel decoding techniques.
- **Type-specific prompts are crucial**: Removing them degrades the ED by 57%; generic prompts result in misclassifications of element types (e.g., table → formula).
- **Cropping > Box Query**: Focusing on local views is more effective than requiring the model to localize and recognize simultaneously within the global image (ED 0.1028 vs 0.1849).
- **Element-level SOTA as well**: Formulas achieve CDM of 0.9850 (SPE), tables achieve TEDS of 0.9515 (PubTabNet), and text achieves ED of 0.0029 (Fox-Block EN).

## Highlights & Insights
- **An elegant design for the "analyze-then-parse" paradigm**: Decoupling "global structure understanding" from "local content recognition" through layout analysis preserves the simplicity of end-to-end setups while overcoming the efficiency and quality issues of pure autoregressive models. This two-stage shared-model design can be transferred to other vision tasks requiring two-level global-to-local understanding.
- **An efficiency miracle of 322M beating 7B**: This indicates that for document parsing, a tailored architecture combined with domain-specific training data is much more efficient than using general-purpose large models. It represents a classic case of "small and specialized" vs. "large and generalist."
- **Data construction strategy**: Decoupling elements makes data collection much more flexible—for instance, 23M formula images can be rendered and collected independently without needing complex layout page annotations.

## Limitations & Future Work
- Supports only standard horizontal text layouts; support for vertical text (e.g., ancient books) is limited.
- Only supports Chinese and English; multilingual capabilities need extension.
- Parallel decoding is constrained by GPU VRAM (with a maximum batch size of 16 elements); pages with excessive elements still require multiple inference runs.
- Insufficient handwritten text recognition capabilities.
- Errors in Stage 1 layout analysis cascade to Stage 2 (error propagation).

## Related Work & Insights
- **vs MinerU (Integrated)**: MinerU combines multiple expert models, which is effective but complex to integrate. Dolphin uses a single 322M model to handle everything in a unified manner, which is simpler and more effective.
- **vs GOT/Nougat (End-to-end Autoregressive)**: These models generate full-page content directly in an autoregressive fashion, leading to much higher ED than Dolphin on complex documents. Dolphin's two-stage design successfully mitigates the layout degradation problem of autoregressive approaches.
- **vs Qwen2.5-VL/GPT-4o (General VLMs)**: General VLMs have 7B+ parameters but still underperform the 322M Dolphin on complex documents, illustrating the importance of task-specific designs.

## Rating
- Novelty: ⭐⭐⭐⭐ The analyze-then-parse paradigm and heterogeneous anchor prompting are cleverly designed, though two-stage parsing itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across both page-level and element-level tasks on multiple benchmarks, accompanied by thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, rich in diagrams and tables, with highly persuasive visual case studies.
- Value: ⭐⭐⭐⭐⭐ Highly practical, with open-source code and models, directly advancing the field of document understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)
- [\[ACL 2026\] Prefix Parsing is Just Parsing](../../ACL2026/llm_nlp/prefix_parsing_is_just_parsing.md)
- [\[ACL 2025\] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)
- [\[ACL 2025\] P3: Prompts Promote Prompting](p3_prompts_promote_prompting.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)

</div>

<!-- RELATED:END -->
