---
title: >-
  [Paper Note] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing
description: >-
  [CVPR 2026][Multimodal VLM][Document parsing] This paper proposes PaddleOCR-VL, a coarse-to-fine document parsing framework. The coarse stage employs a lightweight VRFM module to identify effective visual regions, while the fine stage applies a compact 0.9B VLM to process only those regions. With minimal visual tokens and parameters, the framework achieves state-of-the-art performance on OmniDocBench v1.5, substantially reducing latency and resource consumption.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Document parsing
  - coarse-to-fine visual processing
  - vision-language model
  - OCR
  - visual token compression
date: 2026-05-08
content_hash: 6abe9a1f31a1f836
---

# PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing

**Conference**: CVPR 2026
**arXiv**: [2603.24326](https://arxiv.org/abs/2603.24326)
**Code**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Document parsing, coarse-to-fine visual processing, vision-language model, OCR, visual token compression

## TL;DR

This paper proposes PaddleOCR-VL, a coarse-to-fine document parsing framework. The coarse stage employs a lightweight VRFM module to identify effective visual regions, while the fine stage applies a compact 0.9B VLM to process only those regions. With minimal visual tokens and parameters, the framework achieves state-of-the-art performance on OmniDocBench v1.5, substantially reducing latency and resource consumption.

## Background & Motivation

1. **State of the Field**: Document parsing is a critical technology for constructing LLM training corpora and RAG systems. High-resolution inputs are essential for document parsing but cause quadratic growth in the number of visual tokens.
2. **Limitations of Prior Work**: Pipeline-based methods (detection + recognition + structure reconstruction) are prone to error propagation; general-purpose VLMs hallucinate on handwritten or highly structured documents; specialized VLMs either have large parameter counts or suffer from coordinate drift.
3. **Root Cause**: High resolution is necessary for fine-grained recognition → visual token count explodes → high computational cost. Yet effective information in documents is highly non-uniform in distribution—PPT slides contain only ~39% effective regions, and newspapers approximately ~60%.
4. **Paper Goals**: Eliminate visual redundancy while preserving high-resolution accuracy, thereby improving efficiency.
5. **Starting Point**: The sparsity of effective visual regions—large portions of background and decorative elements carry no useful information.
6. **Core Idea**: A coarse stage rapidly identifies effective regions (localization + contextual relation prediction), and a fine stage processes only those regions.

## Method

### Overall Architecture

The framework consists of two stages: (1) **Coarse stage**—VRFM (Valid Region Focusing Module) rapidly localizes document elements (text, formulas, tables, etc.); (2) **Fine stage**—PaddleOCR-VL-0.9B performs detailed recognition on the extracted effective regions. The decoupled design allows independent optimization of each stage.

### Key Designs

1. **VRFM (Valid Region Focusing Module)**:

    - **Function**: Rapidly locates semantically effective regions within a document.
    - **Mechanism**: A lightweight detector that simultaneously predicts region locations and inter-region contextual relationships (reading order). Compared to general-purpose object detectors, VRFM is optimized for document elements, achieving higher speed and accuracy.
    - **Design Motivation**: Standard VLMs that directly process full-page images waste substantial computation on background regions. Locating regions prior to recognition significantly reduces the input volume for the fine stage.

2. **PaddleOCR-VL-0.9B**:

    - **Function**: Performs detailed text/formula/table recognition on effective visual regions.
    - **Mechanism**: A compact VLM with only 0.9B parameters, guided by VRFM outputs to process cropped effective regions rather than full-page images. The deficit in model scale is compensated through a high-quality data pipeline (30M+ samples).
    - **Design Motivation**: Small model + precise input > large model + coarse-grained input. The 0.9B scale enables edge deployment.

3. **Large-Scale High-Quality Data Pipeline**:

    - **Function**: Provides sufficient training signal for the small model.
    - **Mechanism**: Over 30 million broadly distributed samples are collected from public sources and synthetic data, constituting one of the key factors behind model performance.
    - **Design Motivation**: Smaller models have lower data efficiency and require more high-quality data to compensate.

### Loss & Training

VRFM: standard detection loss + reading order prediction loss. VLM: standard language modeling loss.

## Key Experimental Results

### Main Results

| Metric | PaddleOCR-VL | GOT-OCR | Qwen2.5-VL-7B | InternVL3 |
|--------|-------------|---------|---------------|-----------|
| Text Score | **SOTA** | 2nd | 3rd | 4th |
| Formula Score | **SOTA** | 2nd | 3rd | 4th |
| Table Score | **SOTA** | 2nd | 3rd | 4th |
| Reading Order | **SOTA** | 2nd | 3rd | 4th |
| Visual Tokens | **Fewest** | More | Many | Most |
| Parameters | **0.9B** | Larger | 7B | Larger |

### Ablation Study

| Configuration | Overall Score | Notes |
|---------------|--------------|-------|
| Full PaddleOCR-VL | **SOTA** | VRFM + 0.9B VLM |
| w/o VRFM (direct full-page) | Drops + slower | Token count explodes |
| w/o data pipeline | Significant drop | Data is critical for small models |

### Key Findings

- VRFM reduces the number of effective tokens by 40–60% while simultaneously improving performance.
- A 0.9B model focused on effective regions can outperform a 7B general-purpose model.
- The data pipeline is one of the key factors enabling the 0.9B model to achieve SOTA performance.

## Highlights & Insights

- The design philosophy of "small model + precise input > large model + coarse-grained input" has important implications for resource-constrained deployment.
- The coarse-stage cost of VRFM is far lower than that of the VLM, yet it yields substantial savings in fine-stage computation—yielding an ROI of approximately 10–100×.
- The decoupled design allows VRFM and the VLM to be independently optimized and upgraded.

## Limitations & Future Work

- Detection errors in VRFM propagate to the fine stage (missed regions cannot be recognized).
- Robustness to extremely complex layouts (e.g., nested tables + formulas + images) requires further improvement.
- Reading order prediction may fail under multi-column layouts.

## Related Work & Insights

- **vs. GOT-OCR**: GOT-OCR processes entire pages end-to-end with high computational cost; PaddleOCR-VL reduces redundancy through the coarse-to-fine strategy.
- **vs. Qwen2.5-VL**: General-purpose VLMs have large parameter counts (7B+) and are over-engineered for document scenarios; PaddleOCR-VL achieves superior results with only 0.9B parameters.
- **vs. Traditional Pipelines**: Traditional pipelines lack global semantic understanding; the VLM fine stage in PaddleOCR-VL provides semantic comprehension capability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The coarse-to-fine document parsing framework and VRFM design demonstrate practical innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on OmniDocBench across four dimensions, achieving SOTA on all.
- **Writing Quality**: ⭐⭐⭐⭐ Efficiency analysis is intuitive and comparisons are clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ The open-source framework offers significant value to the document AI community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

<!-- RELATED:END -->
