---
title: >-
  [Paper Note] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating
description: >-
  [ACL 2025][VLM Reasoning][Long Document Understanding] This paper proposes the LongDocURL benchmark, which covers 20 subtasks across three primary task categories: understanding, numerical reasoning, and cross-element locating. It contains 2,325 high-quality QA pairs spanning over 33,000 pages of documents. A systematic evaluation of 26 model configurations exposes key performance gaps of current LVLMs in long document understanding.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Long Document Understanding"
  - "Multimodal Benchmark"
  - "LVLM"
  - "Cross-Element Locating"
  - "Document QA"
date: 2026-05-08
content_hash: 5ed3b43435a1c3f8
---

# LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating

**Conference**: ACL 2025  
**arXiv**: [2412.18424](https://arxiv.org/abs/2412.18424)  
**Code**: [dengc2023/LongDocURL](https://github.com/dengc2023/LongDocURL)  
**Area**: Multimodal VLM / Document Understanding  
**Keywords**: Long Document Understanding, Multimodal Benchmark, LVLM, Cross-Element Locating, Document QA  

## TL;DR

This paper proposes the LongDocURL benchmark, which covers 20 subtasks across three primary task categories: understanding, numerical reasoning, and cross-element locating. It contains 2,325 high-quality QA pairs spanning over 33,000 pages of documents. A systematic evaluation of 26 model configurations exposes key performance gaps of current LVLMs in long document understanding.

## Background & Motivation

**Background**: LVLMs (such as GPT-4o, Qwen2-VL, and InternVL2) have made significant progress in document understanding, demonstrating the capacity to handle complex document elements, longer contexts, and diverse tasks. However, the development of evaluation benchmarks lags far behind model capabilities—models evaluated on single-page benchmarks (e.g., DocVQA) easily achieve over 95% accuracy, whereas multi-page benchmarks (such as MP-DocVQA and DUDE) only cover documents with fewer than 20 pages.

**Limitations of Prior Work**: (1) Existing long document benchmarks like MMLongBench-Doc average only 47.5 pages with about 1k valid samples, and only 33.0% of the questions involve cross-page information, failing to reflect the complexity of real-world long document scenarios; (2) All existing benchmarks focus on understanding and simple QA, completely neglecting the cross-element locating task, which evaluates a model's ability to analyze relationships across different element types (e.g., mapping a paragraph to its corresponding title, or associating a chart with a table); (3) Element coverage is incomplete, with most benchmarks failing to simultaneously cover all document element types such as paragraphs, tables, charts, and titles.

**Key Challenge**: Model capabilities are advancing rapidly (supporting 128K+ context), but evaluation benchmarks remain stuck in the short-document, single-task stage. This ceiling effect of benchmarks makes it impossible to distinguish the performance of models in truly complex long document scenarios, hindering further development of the field.

**Goal**: (1) Build a genuine long-document multimodal benchmark—averaging 85.6 pages, covering 8 document types, with a total of 33,000+ pages; (2) Pioneer the cross-element locating task category to assess the model's capacity to analyze relationships between different element types; (3) Provide 20 fine-grained subtasks to support an in-depth analysis of model capability gaps across various dimensions.

**Key Insight**: Three primary task categories are defined: understanding (extracting information directly), reasoning (numerical counting/computation/comparison/summarization), and locating (analyzing relationships across element types), subdivided into 20 subtasks based on task category $\times$ answer evidence page count $\times$ evidence element type. A semi-automated pipeline (GPT-4o generation + automatic validation + human review) is used to efficiently construct high-quality data.

**Core Idea**: Construct a long-document multimodal benchmark spanning three dimensions—understanding, reasoning, and locating—with an average of 85.6 pages, pioneering the cross-element locating task and systematically revealing the performance gaps of LVLMs.

## Method

### Overall Architecture

The construction of LongDocURL consists of a four-module semi-automated pipeline: (1) **Extract & Filter**—crawled 200K PDFs from CommonCrawl, filtered them by page count (50-150 pages) and language (English), categorized document types using GPT-4o, and ultimately retained 396 documents covering 8 types (research reports, user manuals, books, papers, etc.), averaging 85.6 pages/43,622.6 tokens; (2) **QA Generation**—used PyMuPDF and Docmind to parse PDFs and extract "text-type-bbox" triples as Symbolic Document Representations, and designed multi-step iterative prompts to query GPT-4o to generate QA pairs and corresponding evidence sources; (3) **Automated Verification**—automatically validated the quality of QA pairs based on three criteria: task relevance, format correctness, and faithfulness; (4) **Human Verification**—conducted human review for negative sample recovery (some can be corrected to valid samples), visual document consistency checks (using original PDFs instead of parsed text), and cross-validation. This workflow yielded 2,325 high-quality QA pairs.

### Key Designs

1. **Three Primary Task Categories and 20 Subtasks**:
    - **Function**: Provides a multi-dimensional and fine-grained evaluation framework for LVLMs.
    - **Mechanism**: Understanding (53.5%)—directly extracting information from documents (e.g., keyphrase identification, table parsing); Reasoning (16.6%)—numerical counting, calculation, comparison, and summarization; Locating (29.9%)—analyzing relations across different element types (e.g., Para-Title Locating: finding the corresponding title given a paragraph summary; Cross-Table Locating: associating information across tables).
    - **Design Motivation**: Cross-element locating is a completely new task category—prior benchmarks completely ignored the model's ability to understand relationships between elements in the document structure, which is a core requirement in real-world document usage.

2. **Semi-Automated Quality Control Pipeline**:
    - **Function**: Efficiently constructs a large-scale benchmark while ensuring data quality.
    - **Mechanism**: GPT-4o generation → rule-based automated verification (task relevance + format + faithfulness) → human review for negative sample recovery and cross-checking.
    - **Design Motivation**: Roughly 75.2% of the initial samples for the Cross-Title Locating task were negative, compared to only 19.6% for Cross-Table Locating. Automated verification effectively filters low-quality samples, while human review further recovers rectifiable samples.

3. **Cut-off Input Paradigm**:
    - **Function**: Fairly evaluates models with different context length capabilities.
    - **Mechanism**: For LVLMs unable to process the entire document, a continuous window of 30 pages around the evidence is cropped as the input.
    - **Design Motivation**: In reality, most open-source models cannot handle a full 150-page input; the cut-off paradigm ensures the feasibility of the evaluation.

### Evaluation Protocol

A three-stage evaluation protocol: (1) Models freely generate answers (temperature = 0.0); (2) GPT-4o extracts concise answers; (3) Standardized scores are calculated based on 5 answer formats (Integer/Float/String/List/None).

## Key Experimental Results

### Main Results

| Model | Type | Parameters | Total Score |
|------|------|--------|------|
| **GPT-4o** | Closed-source | - | **64.5** |
| Claude-3.5-Sonnet | Closed-source | - | 41.8 |
| InternVL2-Pro | Open-source | - | **30.6** |
| Qwen2.5-Instruct | Open-source | 32B | 26.6 |
| Qwen2-VL | Open-source | 7B | 25.0 |
| LLaVA-OneVision-Chat | Open-source | 7B | 24.6 |

### LongDocURL vs. Existing Benchmarks

| Benchmark | Avg Pages | QA Count | Multi-page Q% | Cross-element Q% | Contains Locating |
|------|---------|--------|----------|------------|----------|
| DocVQA | 1.0 | - | 0% | - | ✗ |
| MP-DocVQA | 8.3 | - | 0% | - | ✗ |
| MMLongBench-Doc | 47.5 | 1,082 | 33.0% | 22.6% | ✗ |
| **LongDocURL** | **85.6** | **2,325** | **52.9%** | **37.1%** | **✓** |

### Dataset Statistics

| Metric | Value |
|--------|------|
| Total Documents | 396 documents, 8 types |
| Avg Pages / Tokens | 85.6 pages / 43,622.6 tokens |
| Total QA Pairs | 2,325 |
| Understanding/Reasoning/Locating Ratio | 53.5% / 16.6% / 29.9% |
| Multi-page Questions Ratio | 52.9% |
| Cross-element Questions Ratio | 37.1% |

### Key Findings

- **Huge Closed-Open Source Gap**: GPT-4o achieved a score of 64.5, while the best open-source model achieved only 30.6, leaving a gap of over **33.9 points**.
- **Cross-Element Locating is the Hardest Task**: Even GPT-4o performs poorly on locating subtasks, indicating that current LVLMs lack cross-element relationship reasoning capabilities.
- **Multi-page Questions are More Challenging than Single-page**: The 52.9% of multi-page questions are the primary driver of overall performance decline.
- **OCR+LLM Text Input Method Competes with Image Input in Some Scenarios**: This indicates that OCR pipeline methods still hold value in long document scenarios.

## Highlights & Insights

- Pioneered the cross-element Locating task category, filling a gap in document understanding benchmarks.
- The document scale significantly exceeds existing benchmarks (average 85.6 pages vs. 47.5 pages in MMLongBench-Doc).
- The semi-automated construction pipeline strikes a good balance between quality and efficiency.
- 20 fine-grained subtasks support a multi-dimensional, in-depth analysis of model capabilities.

## Limitations & Future Work

- Only covers English documents, lacking multilingual evaluation.
- Using a cropped 30-page input is a compromise on model capabilities, failing to evaluate true full-text comprehension.
- Document sources lean heavily toward academic and business types, with insufficient coverage of daily-life documents.
- Relies on GPT-4o for answer extraction, which potentially introduces evaluation bias.
- The quality of document parsing (PyMuPDF/Docmind) affects the fairness of comparison with text-based input methods.

## Related Work & Insights

- **Single-page Benchmarks**: DocVQA, ChartQA — Have hit performance ceilings.
- **Short Multi-page Benchmarks**: MP-DocVQA, DUDE — Limited to under 20 pages.
- **Long Document Benchmarks**: MMLongBench-Doc (47.5 pages), M-Longdoc (210.8 pages) — Lack locating tasks.
- **Insight**: The concept of cross-element locating can be extended to webpage understanding (cross-DOM element relationships) and multimodal RAG evaluations.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/vlm_reasoning/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/vlm_reasoning/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICCV 2025\] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging](../../ICCV2025/vlm_reasoning/finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal.md)
- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](../../ACL2026/vlm_reasoning/scimdr_advancing_scientific_multimodal_document_reasoning.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](../../NeurIPS2025/vlm_reasoning/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)

</div>

<!-- RELATED:END -->
