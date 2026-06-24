---
title: >-
  [Paper Note] READoc: A Unified Benchmark for Realistic Document Structured Extraction
description: >-
  [ACL 2025][LLM Evaluation][Document Structured Extraction] READoc proposes the first unified benchmark that defines Document Structured Extraction (DSE) as an end-to-end PDF-to-Markdown conversion. It includes 3,576 realistic documents from arXiv, GitHub, and Zenodo, along with a three-module evaluation suite (Standardization, Segmentation, and Scoring), revealing for the first time the gap between current DSE systems and real-world requirements.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Document Structured Extraction"
  - "Unified Benchmark"
  - "PDF-to-Markdown"
  - "Evaluation Suite"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: b85ba4c2d18ff387
---

# READoc: A Unified Benchmark for Realistic Document Structured Extraction

**Conference**: ACL 2025  
**arXiv**: [2409.05137](https://arxiv.org/abs/2409.05137)  
**Code**: [Yes](https://github.com/icip-cas/READoc)  
**Area**: Document AI / Multimodal  
**Keywords**: Document Structured Extraction, Unified Benchmark, PDF-to-Markdown, Evaluation Suite, Vision-Language Models

## TL;DR

READoc proposes the first unified benchmark that defines Document Structured Extraction (DSE) as an end-to-end PDF-to-Markdown conversion. It includes 3,576 realistic documents from arXiv, GitHub, and Zenodo, along with a three-module evaluation suite (Standardization, Segmentation, and Scoring), revealing for the first time the gap between current DSE systems and real-world requirements.

## Background & Motivation

### Problem Background

Document Structured Extraction (DSE) is the task of converting raw documents into machine-readable structured text, which is crucial for building knowledge bases, high-quality corpora, and RAG systems. However, existing evaluations suffer from two critical issues:

**Fragmentation**: Existing benchmarks break DSE into isolated subtasks—document layout analysis (PubLayNet), OCR (Robust Reading), table recognition (PubTabNet), formula conversion (Im2Latex-100K), table of contents extraction (HRDoc), and reading order detection (ReadingBank). These tasks use different data sources and inconsistent formats, making unified evaluation impossible.

**Localization**: Existing research focuses on local regions within a single page (layout blocks or tables), ignoring the multi-page nature, hierarchical headings, and long-range dependencies of realistic documents.

### Core Motivation

There is a need for a unified benchmark to quantify the gap between existing works and real-world DSE objectives, promoting more comprehensive and practical solutions.

## Method

### Overall Architecture

READoc consists of two main parts:

1. **READoc Dataset**: 3,576 PDF-Markdown pairs from three heterogeneous sources.
2. **Evaluation S³uite**: Standardization $\rightarrow$ Segmentation $\rightarrow$ Scoring.

### Key Designs

1. **Task Definition—PDF-to-Markdown Paradigm**:

    - **Input**: Complete raw PDF documents (most general and inherently unstructured).
    - **Output**: Markdown text (a lightweight markup language supporting structural elements like headings and lists, and supporting tables and formulas via LaTeX syntax).
    - The output can be directly chunked, indexed, or consumed by LLMs.

2. **Dataset Construction**:

   | Subset | Source | Documents | Avg. Pages | Characteristics |
   |------|------|-------|---------|------|
   | READoc-arXiv | arXiv preprints | 1,009 | 11.67 | Complex academic structures (formulas/tables), multi-column layout |
   | READoc-GitHub | GitHub README | 1,224 | 6.54 | Basic elements only, diverse heading styles |
   | READoc-Zenodo | Zenodo open repository | 1,343 | 14.93 | Longer documents (>30 pages), 27 languages |

    - **arXiv**: LaTeX $\rightarrow$ HTML (LaTeXML) $\rightarrow$ Markdown (improved Nougat process)
    - **GitHub**: Raw Markdown $\rightarrow$ PDF (Pandoc + Eisvogel template)
    - **Zenodo**: DOCX/HTML $\rightarrow$ Markdown (Markitdown/Pandoc) $\rightarrow$ PDF

3. **Evaluation S³uite Three Modules**:

   **Standardization Module**: Unifies formatting differences from different system outputs.
    - Align formula boundaries (`$$`, `\begin{equation}`, `\[`)
    - Unify Markdown heading styles
    - Align Markdown tables with LaTeX table formats
    - Remove image and hyperlink syntax

   **Segmentation Module**: Segments Markdown text into four types of semantic units.
    - Headings (at different levels)
    - Formulas (inline and display)
    - Tables
    - Plain text (basic text + simple formatting such as bold/italic/lists)

   **Scoring Module**: Comprises two sub-modules:
    - **Semantic Unit Evaluation**: Text extraction (concatenated EDS + lexical F1), heading detection (concatenated EDS + Tree Edit Distance on Hierarchical Headings TEDS), formula conversion (inline/display EDS), table recognition (concatenated EDS + optimal bipartite matching TEDS).
    - **Reading Order Detection**: Chunks text based on semantic unit boundaries and computes Kendall's Tau Distance Similarity (block-level and token-level).

## Key Experimental Results

### Main Results on READoc-arXiv (Table 3, Average Metric)

| System Type | Method | Text(C) | Head(T) | Formula(E) | Formula(I) | Table(T) | RO(B) | **Avg** |
|---------|------|---------|---------|-----------|-----------|---------|-------|---------|
| Baseline | PyMuPDF4LLM | 74.27 | 20.77 | 0.07 | 0.02 | 15.83 | 89.09 | 40.55 |
| Pipeline | **MinerU** | 91.22 | 41.97 | 62.77 | 70.76 | 52.85 | 97.90 | **73.07** |
| Pipeline | Marker | 82.71 | 39.39 | 3.47 | 48.74 | 72.36 | 97.74 | 64.98 |
| Expert Model | Nougat-base | 92.29 | 88.50 | 76.19 | 79.47 | 52.30 | 98.41 | **81.42** |
| VLM | InternVL-Chat-V1.5 | 68.44 | 13.57 | 33.13 | 24.37 | 34.35 | 91.31 | 47.83 |
| VLM | GPT-4o-mini | 84.37 | 18.65 | 42.23 | 41.67 | 39.85 | 96.35 | 57.98 |

### Cross-Subset Generalization (Nougat-base Average)

| Subset | Average |
|------|---------|
| READoc-arXiv | **81.42** |
| READoc-GitHub | 74.12 |
| READoc-Zenodo | 49.94 |

### Ablation Study / Detailed Analysis

| Dimension | Finding |
|---------|------|
| Single vs. Multi-column | All systems suffer a performance drop on multi-column documents; InternVL drops the most (-7.13) |
| Impact of Document Length | VLMs degrade in performance as length increases; pipeline tools and expert models remain stable |
| Impact of Document Depth | Pipeline tools and expert models degrade in performance as depth increases; VLMs remain stable |
| Single-page vs. Multi-page Paradigm | The multi-page paradigm significantly enhances global ToC construction (+28 Tree), but harms local capabilities for formulas and tables |
| Efficiency | Marker is the fastest (24s/doc), while InternVL is the slowest (1182s/doc) |

### Key Findings

1. **Pipeline tools face complex engineering challenges**: Docling cannot recognize formulas, Marker lacks support for inline formulas, and Pix2Text experiences crashes.
2. **Expert models exhibit poor generalization**: Nougat achieves 81.42 on arXiv but drops to 74.12 on GitHub, and plunges to 49.94 on Zenodo (multilingual).
3. **VLMs generally lag behind on complex academic documents**: The best open-source VLM (InternVL) scores only 47.83, far below pipeline tools.
4. **Global ToC construction remains a major challenge**: All single-page systems perform far worse on hierarchical heading trees compared to concatenated evaluations.
5. **Reading order detection is relatively easy**: The baseline tool Tesseract already achieves 96.70 KTDS.
6. **Different systems exhibit distinct bottlenecks**: VLMs are stable across depth but degrade with length, whereas pipeline tools show the opposite trend.

## Highlights & Insights

- **Value of a Unified Paradigm**: The PDF-to-Markdown paradigm unifies 6 subtasks into an end-to-end framework for the first time, revealing interactions among subtasks and system-level bottlenecks.
- **Exquisite Three-Source Data Design**: arXiv (complex structures), GitHub (simple layout with complex headings), and Zenodo (multilingual long documents) complementarily cover diverse challenges.
- **Thoughtful S³uite Design**: The standardization module eliminates formatting differences, the segmentation module quantifies multidimensional capabilities, and the scoring module balances local and global evaluation.
- **Revealing Multi-Page Challenges for the First Time**: The single-page image-to-markup paradigm has inherent limitations; multi-page processing is the key direction for future research.
- **Directly Beneficial for RAG Systems**: The Markdown outputs can be directly utilized for LLM consumption, chunking, and indexing.

## Limitations & Future Work

- The automatically constructed PDF-Markdown pairs contain noise, which is difficult to completely eliminate.
- The standardization module cannot cover all formatting discrepancy scenarios.
- Harder cases like handwritten documents or scanned documents are not included.
- Future Work: Introduce more format unification modules and explore more efficient multi-page modeling paradigms.
- Future Work: Construct subsets containing realistic documents with poorer OCR quality.

## Related Work & Insights

- Inherits and unifies classic DSE subtask benchmarks such as PubLayNet, PubTabNet, and Im2Latex-100K.
- Compared to OmniDocBench (Ouyang et al., 2024), READoc handles multi-page documents and end-to-end pipelines.
- Nougat (Blecher et al., 2023) defines a single-page image-to-Markdown paradigm, which READoc extends to full multi-page documents.
- Provides an objective comparative evaluation of currently popular tools like MinerU and Marker, serving as a valuable reference for selecting document processing tools.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first unified, end-to-end, multi-page DSE benchmark, filling a crucial gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive, evaluating 14 systems across 3 subsets with multidimensional metrics and fine-grained analyses (layout, length, depth, and efficiency).
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured with rich tables and figures, providing thorough comparative analysis.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for promoting progress in both the Document AI community and practical RAG engineering, filling an evaluation vacuum.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)
- [\[ACL 2025\] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)
- [\[ACL 2025\] StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following](structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)

</div>

<!-- RELATED:END -->
