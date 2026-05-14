---
title: >-
  [Paper Note] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing
description: >-
  [CVPR 2026][Multimodal VLM][Document Parsing] PaddleOCR-VL proposes a coarse-to-fine document parsing architecture: the coarse stage employs a lightweight Valid Region Focusing Module (VRFM) to localize effective visual…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Document Parsing"
  - "Vision-Language Model"
  - "Coarse-to-Fine Processing"
  - "Visual Redundancy Elimination"
  - "OCR"
date: 2026-05-08
content_hash: 2416b54f449d5261
---

# PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing

**Conference**: CVPR 2026
**arXiv**: [2603.24326](https://arxiv.org/abs/2603.24326)
**Code**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
**Area**: Multimodal VLM
**Keywords**: Document Parsing, Vision-Language Model, Coarse-to-Fine Processing, Visual Redundancy Elimination, OCR

## TL;DR

PaddleOCR-VL proposes a coarse-to-fine document parsing architecture: the coarse stage employs a lightweight Valid Region Focusing Module (VRFM) to localize effective visual regions and predict reading order, while the fine stage applies a compact 0.9B vision-language model to perform detailed recognition on cropped regions, achieving state-of-the-art document parsing performance with minimal visual tokens and parameters.

## Background & Motivation

1. **Background**: Document parsing approaches fall into three categories — pipeline methods (cascading expert components), general-purpose VLMs (end-to-end but heavyweight), and specialized VLMs (unified architecture but low efficiency). High-resolution input is critical for document parsing but causes quadratic growth in visual token count.
2. **Limitations of Prior Work**: General-purpose VLMs frequently produce hallucinations and recognition errors on handwritten or complex documents; specialized VLMs (e.g., MinerU2-VLM) suffer from high latency due to large parameter counts and long decoding sequences; uniform visual token compression methods (e.g., DeepSeek-OCR) compromise fine-grained layout precision.
3. **Key Challenge**: Effective visual regions in document images are highly non-uniform — valid regions occupy only 39% of PPT documents and approximately 60% of information-dense documents. Large amounts of background and decorative regions waste computational resources.
4. **Goal**: Eliminate visual redundancy while preserving high-resolution accuracy, achieving both high precision and high efficiency.
5. **Key Insight**: Motivated by the observed sparsity of effective visual regions, the paper localizes such regions via a detector and performs fine-grained recognition exclusively on them.
6. **Core Idea**: Decouple layout analysis from element recognition — a lightweight detector performs coarse-grained localization and reading order prediction, while a compact VLM conducts fine-grained recognition on cropped regions, avoiding the need to process the entire high-resolution page.

## Method

### Overall Architecture

PaddleOCR-VL operates in two stages. The coarse stage (VRFM) receives the full document image and outputs the location, category, and reading order of effective regions. The fine stage (PaddleOCR-VL-0.9B) receives the cropped effective regions and produces detailed recognition results (text, formulas, tables, etc.). Results are finally reassembled into a structured document according to the predicted reading order.

### Key Designs

1. **Valid Region Focusing Module (VRFM)**:
    - **Function**: Efficiently localizes effective visual elements in documents and predicts reading order.
    - **Mechanism**: Built upon the RT-DETR detector for layout element detection and classification, generating region-level representations. A Pointer Network is further incorporated to model pairwise relationships among detected regions and predict an $N \times N$ matrix encoding relative reading order. The module is lightweight and jointly accomplishes region localization, category prediction, and reading order estimation.
    - **Design Motivation**: Task-specific detectors are more efficient and yield more accurate coordinates than generative VLMs for layout analysis; Pointer Networks are well-suited for sequence ordering tasks.

2. **PaddleOCR-VL-0.9B**:
    - **Function**: Performs fine-grained element recognition on cropped effective regions.
    - **Mechanism**: A compact vision-language model with 0.9B parameters. It processes only the effective regions cropped by VRFM — rather than the full page — substantially reducing the number of visual tokens. It supports recognition of diverse element types including text, formulas, tables, and charts across 109 languages.
    - **Design Motivation**: The decoupled recognition module handles only small region images, enabling a smaller model to achieve superior performance.

3. **High-Quality Data Pipeline**:
    - **Function**: Constructs large-scale, diverse training data.
    - **Mechanism**: Over 30 million broadly distributed samples are collected from public sources and synthetic data, covering diverse document types, languages, and complexity levels. Data diversity is identified as one of the key contributors to model performance.
    - **Design Motivation**: Data quality and diversity influence VLM performance no less than model architecture.

### Loss & Training

- VRFM: Standard object detection loss + Pointer Network ranking loss.
- PaddleOCR-VL-0.9B: Autoregressive generation loss.
- The two modules are optimized independently, each focused on its respective sub-task.
- Training utilizes over 30 million large-scale samples.

## Key Experimental Results

### Main Results

| Method | Parameters | Visual Tokens | OmniDocBench v1.5 Overall |
|--------|-----------|---------------|--------------------------|
| MinerU2-VLM | Large | Many | Runner-up |
| Dolphin | Large | Many | Runner-up |
| DeepSeek-OCR | Medium | Medium (compressed) | Runner-up |
| **PaddleOCR-VL** | **Fewest (0.9B)** | **Fewest** | **SOTA** |

### Ablation Study

| Configuration | Key Metric | Description |
|--------------|-----------|-------------|
| End-to-end VLM | Baseline | Processes full page; many tokens |
| Coarse stage (VRFM) | Efficient localization | Filters 39–60% redundant regions |
| + Fine stage (VL-0.9B) | SOTA | Fine-grained recognition on cropped regions |
| w/o Pointer Network | Poor reading order | Validates necessity of the ordering module |

### Key Findings

- PaddleOCR-VL achieves state-of-the-art performance on all four key metrics: text, formulas, tables, and reading order.
- It uses the fewest parameters and visual tokens, with significantly lower inference latency and higher throughput compared to competing methods.
- The high-quality data pipeline is identified as one of the key contributors to performance.
- The model demonstrates strong robustness on challenging content such as handwritten and historical documents.
- Multilingual document parsing is supported across 109 languages.

## Highlights & Insights

- **Statistical analysis of document visual redundancy** provides compelling motivation: only 39% of PPT document area is effective, directly justifying selective processing.
- **Decoupled design** allows each module to be independently optimized — a practical advantage enabling separate upgrades to the detector or recognition model.
- **Achieving SOTA with 0.9B parameters and minimal tokens** demonstrates that "intelligently selecting where to allocate computation" is more effective than "processing everything with a larger model."

## Limitations & Future Work

- The two-stage pipeline introduces cascading errors — detection failures in VRFM propagate to the recognition stage.
- VRFM localization accuracy may be limited on densely packed pages.
- Reading order prediction may be unreliable under extremely complex layouts (e.g., multi-column mixed with floating elements).
- Validation is restricted to document parsing scenarios and has not been extended to broader VLM applications.

## Related Work & Insights

- **vs. MinerU2.5 / Dolphin**: Unified end-to-end VLMs with large parameter counts and low efficiency; PaddleOCR-VL achieves higher efficiency through coarse-to-fine decoupling.
- **vs. DeepSeek-OCR**: Uniform visual token compression damages layout precision; PaddleOCR-VL selectively discards invalid regions rather than applying uniform compression.
- **vs. Pipeline Methods**: Traditional pipelines rely on multiple independent expert models, leading to complexity and error accumulation; PaddleOCR-VL requires only two modules, yielding a simpler and cleaner design.

## Rating

- Novelty: ⭐⭐⭐⭐ — The coarse-to-fine decoupling with effective region focusing is a clear and effective idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across multiple benchmarks with broad public and private dataset coverage.
- Writing Quality: ⭐⭐⭐⭐ — Motivation analysis is well supported by quantitative evidence.
- Value: ⭐⭐⭐⭐⭐ — Open-source code and models ensure strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)
- [\[CVPR 2026\] VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)

</div>

<!-- RELATED:END -->
