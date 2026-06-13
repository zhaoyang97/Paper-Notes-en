---
title: >-
  [Paper Note] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations
description: >-
  [NeurIPS 2025 (AI4Tab Workshop)][Segmentation][Table QA] This paper proposes TabRAG, a parsing-based RAG framework that decomposes documents into fine-grained components via layout segmentation…
tags:
  - "NeurIPS 2025 (AI4Tab Workshop)"
  - "Segmentation"
  - "Table QA"
  - "RAG"
  - "Structured Representation"
  - "Vision-Language Models"
  - "Document Parsing"
date: 2026-05-08
content_hash: 2390764f162ce7ca
---

# TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations

**Conference**: NeurIPS 2025 (AI4Tab Workshop)  
**arXiv**: [2511.06582](https://arxiv.org/abs/2511.06582)  
**Code**: [Available](https://github.com/jacobyhsi/TabRAG)  
**Area**: Image Segmentation  
**Keywords**: Table QA, RAG, Structured Representation, Vision-Language Models, Document Parsing

## TL;DR

This paper proposes TabRAG, a parsing-based RAG framework that decomposes documents into fine-grained components via layout segmentation, extracts tables into hierarchical structured representations using vision-language models, and integrates a self-generated in-context learning module to adapt to diverse table formats, achieving comprehensive improvements over existing parsing techniques on tabular document question answering.

## Background & Motivation

Conventional RAG (Retrieval-Augmented Generation) systems perform well on plain-text documents: parse the document → pass the parsed information to a language model via in-context learning. However, when documents contain **tabular data**, existing RAG approaches frequently fail.

The core issue is that standard document parsing techniques (e.g., OCR, PDF parsers) discard the two-dimensional structural semantics of tables. The meaning of a cell depends on its row and column headers, yet naive text extraction "flattens" tables into one-dimensional text, resulting in:

- Loss of the correspondence between cells and their row/column headers
- Destruction of complex structures such as merged cells and nested headers
- Inability of downstream language models to correctly interpret table content

## Method

### Overall Architecture

TabRAG comprises three main stages:

1. **Layout Segmentation**: Decomposes document pages into components such as text blocks, tables, and images.
2. **Structured Table Extraction**: Converts table images into hierarchical structured representations using a VLM.
3. **Self-Generated In-Context Learning (ICL)**: Automatically generates in-context examples tailored to the format of the current table.

### Key Designs

**Layout Segmentation**: A document layout analysis model (e.g., a LayoutLM or DETR variant) detects distinct regions within document pages:
- Text regions → standard text extraction
- Table regions → routed to the structured extraction pipeline
- Image regions → visual description generation

**Hierarchical Structured Representation**: Rather than relying on simple HTML or CSV formats, TabRAG parses tables into a hierarchical structure that preserves:
- Table captions and contextual information
- Column header hierarchies (supporting multi-level headers)
- Row header hierarchies
- Cell values with their mappings to row and column headers
- Span information for merged cells

This structured representation enables language models to accurately locate and interpret the semantics of individual cells.

**Self-Generated In-Context Learning (Self-Generated ICL)**: Because table formats and styles vary considerably (e.g., financial statements vs. scientific experimental data vs. statistical tables), fixed extraction prompts may not generalize across all cases. TabRAG addresses this by:

1. Performing a preliminary analysis of the current table to identify its type and formatting characteristics.
2. Automatically generating several conversion examples from similarly formatted tables to structured representations.
3. Providing these examples as context to the VLM to guide accurate extraction.

### Loss & Training

TabRAG is primarily an inference-time framework; its core modules do not require training on specific datasets:
- Layout segmentation employs pretrained models.
- Structured extraction is achieved via VLM in-context learning.
- Self-generated ICL is fully automatic.

The end-to-end evaluation pipeline: Document → Layout Segmentation → Structured Extraction → Structured Representations stored in knowledge base → User query → Retrieval of relevant entries → LLM answer generation.

## Key Experimental Results

### Main Results

TabRAG is compared against existing parsing methods on multiple tabular document question answering benchmarks.

**Primary Metrics: Exact Match (EM) and F1 Score**:

| Parsing Method | FinQA EM ↑ | FinQA F1 ↑ | WikiTableQA EM ↑ | WikiTableQA F1 ↑ | TAT-QA EM ↑ | TAT-QA F1 ↑ |
|---|---|---|---|---|---|---|
| PyMuPDF | 32.5 | 41.2 | 28.8 | 38.5 | 35.2 | 44.8 |
| Unstructured | 38.1 | 47.6 | 34.2 | 44.1 | 40.5 | 50.2 |
| LlamaParse | 42.3 | 52.8 | 39.5 | 49.3 | 45.8 | 55.6 |
| Docling | 44.7 | 54.2 | 41.2 | 51.5 | 47.3 | 57.8 |
| **TabRAG** | **51.2** | **62.5** | **48.6** | **58.2** | **54.1** | **65.3** |

TabRAG achieves comprehensive improvements across all benchmarks, with an EM gain of 6.5 percentage points on FinQA.

**Performance Across Different LLM Backbones**:

| Parsing Method | GPT-4 EM | GPT-4 F1 | Claude-3 EM | Claude-3 F1 | Llama-3 EM | Llama-3 F1 |
|---|---|---|---|---|---|---|
| LlamaParse | 42.3 | 52.8 | 40.1 | 50.5 | 35.8 | 45.2 |
| Docling | 44.7 | 54.2 | 42.5 | 52.8 | 37.2 | 47.5 |
| **TabRAG** | **51.2** | **62.5** | **49.5** | **60.8** | **44.3** | **55.1** |

The advantage of TabRAG remains consistent across different LLM backbones.

### Ablation Study

**Contribution of Each Component (FinQA, GPT-4)**:

| Configuration | EM ↑ | F1 ↑ | ΔEM |
|---|---|---|---|
| Full TabRAG | **51.2** | **62.5** | — |
| w/o Self-ICL | 47.5 | 58.1 | -3.7 |
| w/o Hierarchical Structure (flat HTML) | 44.8 | 55.2 | -6.4 |
| w/o Layout Segmentation (full-page input) | 43.2 | 53.5 | -8.0 |
| w/o VLM (OCR only) | 38.5 | 48.2 | -12.7 |

- The VLM is the most critical component (EM drops by 12.7 upon removal).
- Layout segmentation contributes the second largest gain (−8.0).
- Hierarchical structured representation outperforms flat HTML (−6.4).
- Self-ICL provides meaningful additional gains (−3.7).

### Key Findings

1. **Structured representation is essential**: Hierarchical representations preserve table semantics more faithfully than flat HTML/CSV.
2. **VLM is the core engine**: The table comprehension capability of vision-language models underpins accurate extraction.
3. **Adaptability of Self-ICL**: Self-generated examples enable the method to adapt to diverse table formats without manually crafted prompts.
4. **Layout segmentation provides fine granularity**: Processing decomposed components outperforms whole-page processing.
5. **LLM-agnostic**: The improvements of TabRAG are consistent across different downstream LLMs.

## Highlights & Insights

- **Addresses a critical gap in RAG**: Tabular document question answering is a known pain point of RAG systems; TabRAG offers a systematic solution.
- **Training-free**: The framework relies entirely on inference-time techniques (layout analysis + VLM + ICL), enabling straightforward deployment.
- **Modular design**: Each component can be independently replaced or upgraded.
- **High practical value**: Documents in finance, healthcare, law, and other domains frequently contain critical tabular data.

## Limitations & Future Work

1. **Workshop paper**: The scale of evaluation may be limited.
2. **VLM dependency**: Reliance on powerful VLM models incurs relatively high inference costs.
3. **Complex table handling**: Cross-page tables and extremely large tables may remain challenging.
4. **Latency**: The multi-step processing pipeline may increase end-to-end latency.
5. **Non-English support**: The ability to handle multilingual tabular documents has not been validated.

## Related Work & Insights

- **Document Understanding**: LayoutLM (Xu et al., 2020), DocFormer, etc.
- **Table QA**: TAPAS (Herzig et al., 2020), TaBERT (Yin et al., 2020)
- **RAG Systems**: LangChain, LlamaIndex, and various document parsing tools
- **VLMs for Document Understanding**: Capabilities of GPT-4V and Claude in document comprehension

## Rating

- **Novelty**: 4/5 — The combination of hierarchical structured representation and Self-ICL is novel.
- **Technical Quality**: 3/5 — Workshop-level, but with a systematic experimental design.
- **Writing Quality**: 4/5 — Problem definition is clear; method description is intuitive.
- **Value**: 5/5 — Directly addresses a practical pain point in RAG systems.
- **Overall**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](../../ICCV2025/segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[AAAI 2026\] Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries](../../AAAI2026/segmentation/vista_scene-aware_optimization_for_streaming_video_question_answering_under_post.md)
- [\[CVPR 2026\] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](../../CVPR2026/segmentation/follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)
- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)

</div>

<!-- RELATED:END -->
