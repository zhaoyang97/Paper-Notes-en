---
title: >-
  [Paper Note] DOGR: Towards Versatile Visual Document Grounding and Referring
description: >-
  [ICCV 2025][Multimodal VLM][Document Understanding] This paper proposes DOGR-Engine, a data engine for document grounding and referring, constructs DOGR-Bench — the first comprehensive benchmark evaluating document grounding and referring capabilities across 7 task types × 3 document types — and develops DOGR, the first document understanding MLLM that integrates precise text localization with interactive grounding and referring capabilities.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Document Understanding"
  - "Visual Grounding"
  - "Multimodal Large Language Models"
  - "Data Engine"
  - "OCR"
date: 2026-05-08
content_hash: 4795b8da7d6e4f3d
---

# DOGR: Towards Versatile Visual Document Grounding and Referring

**Conference**: ICCV 2025
**arXiv**: [2411.17125](https://arxiv.org/abs/2411.17125)  
**Code**: [https://github.com/zyinan99/DOGR](https://github.com/zyinan99/DOGR)  
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Document Understanding, Visual Grounding, Multimodal Large Language Models, Data Engine, OCR

## TL;DR

This paper proposes DOGR-Engine, a data engine for document grounding and referring, constructs DOGR-Bench — the first comprehensive benchmark evaluating document grounding and referring capabilities across 7 task types × 3 document types — and develops DOGR, the first document understanding MLLM that integrates precise text localization with interactive grounding and referring capabilities.

## Background & Motivation

With the advancement of multimodal large language models (MLLMs), grounding and referring capabilities are critical for fine-grained document understanding. However, three major gaps exist in this area:

**Data Scarcity**: Existing document data suffer from poor quality — OCR tools produce inaccurate text recognition and bounding box annotations, and extracting semantically coherent text blocks from complex layouts is challenging. Existing instruction tuning data cover only basic referring tasks (region OCR, summarization, translation), with no grounding tasks included.

**Lack of Evaluation**: No comprehensive benchmark exists for evaluating document grounding and referring capabilities, and task definitions remain unclear.

**Insufficient Model Capabilities**: Existing MLLMs (GPT-4o, Gemini, etc.) perform poorly on basic text localization and region recognition, and are unable to integrate grounding and referring capabilities into conversational and reasoning pipelines.

These issues have left the potential of MLLMs for fine-grained document understanding largely untapped.

## Method

### Overall Architecture

The system consists of three components: (1) the DOGR-Engine data engine, (2) the DOGR-Bench evaluation benchmark, and (3) the DOGR model.

The DOGR model adopts a general MLLM architecture: InternViT-300M-448px visual encoder + projection layer + Qwen2-7B-Instruct LLM. It supports dynamic tiling for high-resolution images, uses pixel shuffle for efficiency, and discretizes coordinates into integers in the range 0–999.

### Key Designs

1. **DOGR-Engine Data Engine**: Generates two categories of high-quality data.

    - **Multi-granularity Parsing Data (2.1M)**: Covers five granularity levels — word, phrase, line, paragraph, and full-page — across three document types: poster, chart, and PDF.
        - **Poster**: Leverages metadata from the Crello dataset with a Re-rendering Strategy (modifying individual text block attributes, re-rendering, and using pixel differencing to obtain precise bounding boxes).
        - **Chart**: Extracts information from ChartQA and redraws charts using Matplotlib; bounding boxes are again obtained via re-rendering. One-third of the data removes text and another third randomly occludes half the text to prevent over-reliance on textual content.
        - **PDF**: Applies a Merge Strategy combining MinerU (reading-order-aware but incomplete) and PyMuPDF (complete but unordered) to eliminate duplicates, repair truncated blocks, and sort by column-first order.

    - **Instruction Tuning Data (700K)**: Covers four categories — grounding, referring, grounding-and-referring, and plain QA.
        - For Poster/Chart: full-page parsing data is fed into GPT-4o to generate QA pairs with grounding annotations.
        - For PDF: a Post-annotating Strategy is designed — GPT-4o generates QA from document images (rather than full text), then original text is marked with `<ocr></ocr>` tags, bounding boxes are retrieved via PyMuPDF, and `<bbox></bbox>` tags are inserted accordingly.
        - A rule-based filter removes samples with formatting errors or inaccurate annotations.

2. **DOGR-Bench**: The first comprehensive benchmark for document grounding and referring, comprising 3.6K samples.

    - Seven task types are defined based on input/output format combinations:
        - **Grounding**: $G_a$ (short answer + bbox), $G_r$ (reasoning + grounding), $G_o$ (open-ended + grounding)
        - **Referring**: $R_t$ (answer given a bbox)
        - **Grounding + Referring**: $GR_a$, $GR_r$, $GR_o$
    - Evaluation metrics: grounding is evaluated with $F1_{all}$ (IoU > 0.5 and text match); text is evaluated with Accuracy/BLEU.

3. **Three-Stage Training Strategy**:

    - **Pre-aligning**: Only the projection layer is trained; lr = 1e-3; uses LLaVA-558K.
    - **Pre-training**: Full model is trainable; uses DocStruct4M + 2.1M multi-granularity parsing data.
    - **Fine-tuning**: Full model training using 700K instruction tuning data plus filtered datasets from other sources, totaling ~2M samples.

### Loss & Training

Standard autoregressive language modeling loss (next-token prediction). Learning rates decrease progressively across the three stages (2e-6 for the visual encoder, 1e-5 for other components), and sequence length is extended from 4096 to 6144.

## Key Experimental Results

### Main Results (DOGR-Bench)

| Model | $G_a$ Acc | $G_a$ F1 | $G_r$ Acc | $G_r$ F1 | $R_t$ Acc | $GR_a$ Acc | $GR_a$ F1 |
|------|----------|---------|----------|---------|----------|-----------|----------|
| GPT-4o | 79.0 | 8.8 | 47.0 | 3.8 | 39.8 | 50.0 | 0.5 |
| Gemini 1.5 Pro | 77.7 | 9.4 | 62.0 | 5.8 | 37.2 | 46.0 | 5.7 |
| Gemini 2.5 Flash | 80.2 | 38.8 | 59.3 | 25.4 | 40.7 | 55.0 | 22.8 |
| Qwen2.5-VL-7B | 63.8 | 19.5 | 35.0 | 9.8 | 43.0 | 40.0 | 1.5 |
| **DOGR** | **83.2** | **73.0** | **67.7** | **52.5** | **60.3** | **82.8** | **66.9** |

DOGR substantially outperforms all baselines on grounding: $F1_{all}$ reaches 73.0 on $G_a$, compared to 38.8 for the strongest closed-source model, Gemini 2.5 Flash. On the referring task $R_t$, DOGR achieves 60.3% accuracy.

Text localization performance (DocLocal4K):

| Model | Localization ALL IoU@0.5 | Recognition ALL BLEU-4 |
|------|-----------------|----------------|
| GPT-4o | 5.27 | 4.38 |
| Qwen2.5-VL-7B | 21.51 | 28.98 |
| **DOGR** | **86.64** | **77.88** |

### Ablation Study

| Setting | DocVQA | InfoVQA | DeepForm | ChartQA | VisualMRC |
|------|--------|---------|----------|---------|-----------|
| Baseline (DocOwl data) | 87.57 | 64.58 | 66.13 | 80.88 | 265.9 |
| + Multi-granularity Parsing Data (MG) | - | - | - | - | - |
| + Instruction Tuning Data (IT) | 89.24 | 67.45 | 68.89 | 81.92 | 287.25 |

Multi-granularity parsing data improves localization and recognition performance across all granularity levels; instruction tuning data yields a 2.87% gain on InfoVQA and a 21.35 CIDEr improvement on VisualMRC.

### Key Findings

- Existing MLLMs, including GPT-4o and Gemini, are severely deficient in document grounding, with F1 scores near 0–10%.
- Text localization is a fundamental capability underlying grounding and referring, yet existing models essentially lack it.
- DOGR remains competitive on traditional document understanding tasks (DocVQA 91.7, ChartQA 83.6).
- PDF documents present the greatest challenge for grounding and referring due to complex layouts and dense text.

## Highlights & Insights

- **Elegant Data Engine Design**: The Re-rendering Strategy obtains precise bounding boxes via pixel differencing without OCR; the Merge Strategy combines two PDF parsing tools to leverage complementary strengths; the Post-annotating Strategy uses document images rather than full text as GPT-4o input, substantially reducing cost.
- **Systematic Task Definition**: Seven task types are defined through a combinatorial matrix of input/output formats, covering grounding, referring, and their intersection.
- **Revealing a Critical Gap**: This work is the first to systematically demonstrate the grounding deficiencies of MLLMs in document understanding, with quantitative evidence.
- **PDF Merge Strategy** restores reading order while preserving completeness, offering practical engineering value.

## Limitations & Future Work

- The model relies on high-quality pre-rendered and re-rendered data, which may limit generalization to real-world in-the-wild document images.
- In PDF processing, text that PyMuPDF cannot locate falls back to plain text, resulting in incomplete annotation coverage.
- Only three document types (poster, chart, PDF) are addressed; handwritten documents, scanned materials, and others are not covered.
- Discretizing coordinates to 0–999 may introduce precision loss, particularly for high-resolution, large-scale documents.
- Segmentation-level localization at finer granularities remains an avenue for future exploration.

## Related Work & Insights

- Compared to existing document grounding works such as Kosmos-2.5, mPLUG-1.5, and Fox, DOGR is the first to integrate grounding and referring into conversational and reasoning pipelines.
- The dynamic tiling strategy from the InternVL series is adopted by DOGR, confirming its effectiveness in document scenarios.
- The data engine methodology is generalizable to other multimodal tasks requiring precise spatial annotations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First full-stack solution for document grounding + referring (data engine + benchmark + model) with systematic task definitions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation covering grounding, referring, and traditional VQA tasks, with both closed-source and open-source model comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed pipeline diagrams for the data engine.
- **Value**: ⭐⭐⭐⭐⭐ Fills the gap in document grounding evaluation; the data engine and benchmark represent significant contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](../../CVPR2025/multimodal_vlm/mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](visual_intention_grounding_for_egocentric_assistants.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ICCV 2025\] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)

</div>

<!-- RELATED:END -->
