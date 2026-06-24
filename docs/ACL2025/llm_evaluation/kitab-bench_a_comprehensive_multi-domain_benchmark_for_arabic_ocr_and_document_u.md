---
title: >-
  [Paper Note] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding
description: >-
  [ACL 2025][LLM Evaluation][Arabic OCR] KITAB-Bench is a comprehensive Arabic OCR benchmark covering 8,809 samples across 9 major domains and 36 sub-domains. Evaluation results indicate that modern vision-language models (such as GPT-4o and Gemini) outperform traditional OCR methods by an average of 60% in terms of character error rate (CER), yet the best model only achieves a 65% accuracy in PDF-to-Markdown conversion, highlighting the massive challenges of Arabic document un…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Arabic OCR"
  - "Document Understanding"
  - "Benchmark"
  - "Vision-Language Models"
  - "Multi-domain Evaluation"
date: 2026-05-08
content_hash: 9d3b25bebc4caadc
---

# KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding

**Conference**: ACL 2025  
**arXiv**: [2502.14949](https://arxiv.org/abs/2502.14949)  
**Code**: [Available](https://mbzuai-oryx.github.io/KITAB-Bench/)  
**Area**: LLM Evaluation  
**Keywords**: Arabic OCR, Document Understanding, Benchmark, Vision-Language Models, Multi-domain Evaluation

## TL;DR

KITAB-Bench is a comprehensive Arabic OCR benchmark covering 8,809 samples across 9 major domains and 36 sub-domains. Evaluation results indicate that modern vision-language models (such as GPT-4o and Gemini) outperform traditional OCR methods by an average of 60% in terms of character error rate (CER), yet the best model only achieves a 65% accuracy in PDF-to-Markdown conversion, highlighting the massive challenges of Arabic document understanding.

## Background & Motivation

With the widespread application of RAG (Retrieval-Augmented Generation) in document processing, high-quality text recognition has become increasingly critical for knowledge extraction. While OCR for English and other languages has benefited from large-scale datasets and mature benchmarks, **Arabic OCR faces unique challenges**:

**Peculiarities of the Writing System**: Arabic is written in cursive script from right to left (RTL), possessing complex layout and calligraphic characteristics.

**Limitations of Prior Work**: KHATT and IFN/ENIT focus solely on handwritten text, APTI covers only specific aspects of printed text, and CAMEL-Bench and LAraBench show limited attention to document understanding tasks.

**Lack of Comprehensive Evaluation**: There is no Arabic benchmark capable of simultaneously evaluating high-level tasks such as table parsing, font detection, digit recognition, and layout analysis.

## Method

### Overall Architecture

KITAB-Bench comprises three core components: (1) a multi-source data collection strategy; (2) an LLM-assisted human-in-the-loop data generation pipeline; and (3) an evaluation framework covering 9 specialized tasks.

### Key Designs

1. **Multi-source Data Collection**:

    - **PDF Data**: 33 complex PDFs are curated from academic, medical, legal, and literary domains, featuring rich formats such as tables, merged cells, watermarks, and handwritten annotations.
    - **Integration of Existing Datasets**: Sourced from multiple datasets, including KHATT (handwritten), HistoryAr (historical documents), EvAREST (scene text), and DocLayNet (layout).
    - **Synthetic Data**: Charts, flowcharts, tables, and VQA data generated through an LLM pipeline.
    - Design Motivation: To ensure coverage of real-world complexities.

2. **Five-Stage LLM-Assisted Data Generation Pipeline**:

    - Phase I: **Topic Generation** — The LLM generates diverse topics across domains (academic, legal, medical, technical roles).
    - Phase II: **Data Generation** — Translates topics into structured raw data complying with Arabic language and formatting norms.
    - Phase III: **Code Generation** — Converts data into plotting code, handling Arabic text rendering and RTL content specifically.
    - Phase IV: **Image Rendering** — Uses rendering engines like Mermaid, Plotly, Vegalite, and HTML to create visual representations.
    - Phase V: **Human Evaluation** — Native Arabic reviewers validate the quality.
    - Design Motivation: To balance data diversity and quality.

3. **Novel Evaluation Metrics Design**:

    - **MARS (Markdown Recognition Score)**: Combines chrF and TEDS to evaluate PDF-to-Markdown conversion.
    - **CharTeX (Chart Extraction Score)**: Combines chart type chrF, topic chrF, and Jaccard data similarity.
    - **CODM (Code-Oriented Diagram Metric)**: Extends SCRM to evaluate flowchart/technical diagram conversion to JSON.
    - Design Motivation: Existing metrics cannot adequately capture the structural complexity of Arabic documents.

4. **Nine Evaluation Tasks**:

    - PDF-to-Markdown, Layout Detection, Line Detection, Line Recognition, Table Recognition, Image-to-Text, Chart-to-DataFrame, Flowchart-to-JSON, VQA.
    - Each task utilizes specialized evaluation metrics.

### Dataset Statistics

| Area | Samples |
|------|--------|
| Image-to-Text | 3,760 |
| Layout Detection | 2,100 |
| VQA | 902 |
| Chart-to-DataFrame | 576 |
| Table Recognition | 456 |
| Line Detection | 378 |
| Line Recognition | 378 |
| Flowchart-to-JSON | 226 |
| PDF-to-Markdown | 33 |
| **Total** | **8,809** |

## Key Experimental Results

### Main Results I: Image-to-Text (OCR)

| Model Group | Model | chrF ↑ | CER ↓ | WER ↓ |
|-------------|-------|--------|-------|-------|
| Proprietary VLM | GPT-4o | 61.01 | 0.31 | 0.55 |
| Proprietary VLM | Gemini-2.0-Flash | 77.95 | 0.13 | 0.32 |
| Open-source VLM | AIN-7B | **78.33** | **0.20** | **0.28** |
| Open-source VLM | Qwen2.5VL-7B | 49.23 | 1.20 | 1.41 |
| Traditional OCR | EasyOCR | 45.47 | 0.58 | 0.89 |
| Traditional OCR | Tesseract | 39.62 | 0.54 | 0.84 |

### Main Results II: Table Extraction and PDF Conversion

| Model | TEDS(HTML) | Jaccard(CSV) | MARS(PDF) |
|-------|-----------|-------------|-----------|
| GPT-4o | **85.76** | **66.36** | 65.12 |
| Gemini-2.0-Flash | 83.08 | 65.55 | **65.65** |
| AIN-7B | 75.94 | 64.83 | 52.92 |
| Qwen2-VL-7B | 57.83 | 40.20 | 40.43 |

### Ablation Study

| Dimension | Findings |
|-----------|----------|
| VLM vs. Traditional OCR | VLMs outperform traditional methods by 60% on average in CER |
| Open-source vs. Closed-source | AIN-7B performs comparably to Gemini 2.0 Flash on Image-to-Text |
| PDF-to-Markdown | The best model (Gemini) achieves only 65.65% MARS, leaving significant room for improvement |
| Layout Detection | DETR (Docling) performs best on mAP@0.5 (BCE: 0.750, DocLayNet: 0.758) |

### Key Findings

1. **VLMs Significantly Outperform Traditional OCR**: In Arabic document understanding, VLMs like GPT-4o and Gemini significantly outperform traditional methods like EasyOCR and Tesseract.
2. **PDF-to-Markdown Remains a Hard Nut to Crack**: The best model achieves only around 65% accuracy, indicating that structured conversion of complex Arabic documents is far from resolved.
3. **Open-Source Models are Catching Up to Closed-Source**: AIN-7B surpasses GPT-4o on Image-to-Text (CER 0.20 vs. 0.31).
4. **Digit Recognition and Table Structure Detection** are prominent weaknesses in Arabic OCR.
5. Complex fonts, word elongation, and diacritical marks remain primary challenges.

## Highlights & Insights

- **The Most Comprehensive Arabic OCR Benchmark**: Covers 9 major domains and 36 sub-domains, from basic OCR to advanced document understanding.
- **LLM-Assisted Data Generation Pipeline** is exemplary: The five-stage human-in-the-loop workflow balances scale and quality.
- **Three Novel Metrics** (MARS, CharTeX, CODM) provide standardized tools for Arabic document evaluation.
- Direct practical guidance value for document processing pipelines in RAG systems: reveals which types of models are most reliable for processing Arabic documents.

## Limitations & Future Work

1. The PDF-to-Markdown subset is small, with only 33 samples.
2. Although synthetically generated data is human-verified, there is still a distribution shift compared to real-world documents.
3. Evaluation does not cover cross-lingual aspects or regional Arabic dialects.
4. Lacks analysis of how OCR quality affects downstream tasks in end-to-end RAG pipelines.
5. Evaluations are primarily based on pre-trained models; the potential improvements through Arabic-specific fine-tuning have not been explored.

## Related Work & Insights

- Forms a cross-lingual complement with English document understanding benchmarks (PubLayNet, DocBank, DocLayNet).
- While MIDAD (Bhatia et al., 2024) focuses on training data, KITAB-Bench focuses on evaluation.
- The five-stage LLM-assisted data pipeline can be transferred to benchmark construction for other low-resource languages.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Fills the gap in comprehensive Arabic OCR evaluation, with valuable designs of new metrics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Systematically and comprehensively evaluates 9 tasks using multiple types of models and multi-dimensional metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Structured clearly with rich tables, although certain sections are somewhat plain in narrative style.
- **Value**: ⭐⭐⭐⭐ — Possesses significant infrastructural value for the Arabic NLP and document processing communities, driving forward research in low-resource language document understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](../../ECCV2024/llm_evaluation/visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)
- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
