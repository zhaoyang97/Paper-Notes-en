---
title: >-
  [Paper Note] Docling Technical Report
description: >-
  [ECCV 2024][PDF Parsing] Docling is an open-source PDF document conversion tool that integrates DocLayNet-based layout analysis models and TableFormer table structure recognition models. It efficiently converts PDFs into structured JSON or Markdown formats on standard hardware.
tags:
  - "ECCV 2024"
  - "PDF Parsing"
  - "Document Layout Analysis"
  - "Table Structure Recognition"
  - "Open Source Tools"
  - "DocLayNet"
date: 2026-05-08
content_hash: 0e3506111b2af7aa
---

# Docling Technical Report

**Conference**: ECCV 2024  
**arXiv**: [2408.09869](https://arxiv.org/abs/2408.09869)  
**Code**: [GitHub](https://github.com/DS4SD/docling)  
**Area**: Document AI / PDF Conversion  
**Keywords**: PDF Parsing, Document Layout Analysis, Table Structure Recognition, Open Source Tools, DocLayNet

## TL;DR

Docling is an open-source PDF document conversion tool that integrates DocLayNet-based layout analysis models and TableFormer table structure recognition models. It efficiently converts PDFs into structured JSON or Markdown formats on standard hardware.

## Background & Motivation

**Background**: Due to diverse formats and low standardization, PDF documents are optimized for printing and lose most structural information. Consequently, converting them into machine-readable formats efficiently has long been difficult. With the rise of LLM and RAG applications, the demand for high-quality PDF content extraction has become increasingly urgent.

**Limitations of Prior Work**: (1) Most powerful document understanding solutions on the market are commercial software or cloud services; (2) existing open-source tools (such as pymupdf, restricted by licensing, or pypdfium, which suffers from quality issues) have a significant capability and quality gap compared to commercial solutions; (3) although multimodal vision-language models can process documents, they are slow and expensive, making them unsuitable for large-scale batch processing.

**Key Challenge**: The open-source community lacks a permissively licensed, feature-complete, and reliable PDF document conversion tool.

**Goal**: To provide an MIT-licensed, open-source PDF conversion library that integrates SOTA layout analysis and table structure recognition capabilities, which can run entirely locally.

**Key Insight**: Rather than pursuing end-to-end multimodal approaches, this work combines specialized AI models (layout detection + table recognition) with traditional PDF parsing to achieve the optimal balance between efficiency and quality.

**Core Idea**: Connecting PDF parsing, layout analysis, table recognition, and post-processing via a modular pipeline architecture to provide an easily extensible open-source framework.

## Method

### Overall Architecture

Docling implements a linear processing pipeline: PDF Backend → Page Rendering + Text Extraction → Layout Analysis Model → Table Structure Recognition Model → Post-Processing Assembly → JSON/Markdown Output. Each stage can be independently configured and replaced.

### Key Designs

1. **In-house PDF Backend (docling-parse)**:

    - **Function**: Extracts text content and its coordinates from PDFs, and renders page images
    - **Mechanism**: An in-house parser built on top of the underlying qpdf library, resolving the AGPL license restrictions of pymupdf and the quality issues of pypdfium/PyPDF (such as cross-column text merging). It also provides pypdfium as an alternative backend
    - **Design Motivation**: PDF parsing is the foundation of the entire pipeline, requiring high-quality text coordinate extraction and reliable page rendering

2. **RT-DETR-based Layout Analysis Model**:

    - **Function**: Detects the bounding boxes and categories of elements such as paragraphs, headings, lists, figures, and tables within the page
    - **Mechanism**: The architecture is based on the RT-DETR object detector, retrained on DocLayNet (a large-scale human-annotated document layout dataset) and private datasets. It inputs 72dpi page images and achieves sub-second inference on a single CPU. Predicted bounding boxes are de-duplicated, filtered by confidence, cross-matched with PDF text tokens, and grouped into semantic units
    - **Design Motivation**: Document layout understanding is the core of structured conversion; RT-DETR offers a good balance between speed and accuracy

3. **TableFormer Table Structure Recognition**:

    - **Function**: Restores logical row-column structures, merged cells, and header hierarchies from table images
    - **Mechanism**: A Vision Transformer architecture that describes table structures using a custom structure-token language. It can handle complex cases such as borderless tables, empty cells, row-column merging, and irregular indentations. Predicted structures are matched with PDF text cells directly, avoiding re-running OCR
    - **Design Motivation**: Tables are the densest yet structurally most complex elements in documents; TableFormer significantly outperforms traditional methods in handling edge cases

### Loss & Training

The technical report does not detail the training processes; the training methods for layout analysis and TableFormer are described in their respective independent papers. Inference relies on ONNX Runtime (for layout) and PyTorch (for tables).

## Key Experimental Results

### Main Results

| Configuration | TTS (225 pages) | Pages/sec | Peak Memory |
|------|-----------|------|---------|
| M3 Max, 4 threads, native | 177s | 1.27 | 6.20 GB |
| M3 Max, 16 threads, native | 167s | 1.34 | 6.20 GB |
| Xeon E5, 4 threads, native | 375s | 0.60 | 6.16 GB |
| M3 Max, 4 threads, pypdfium | 103s | 2.18 | 2.56 GB |

### Ablation Study

| Component | Time Share | Description |
|------|---------|------|
| PDF Parsing | ~10% | Basic text extraction |
| Layout Analysis | ~30% | Sub-second/page |
| Table Recognition | ~50% | 2-6 sec/table, main bottleneck |
| Post-processing | ~10% | Reading order, metadata |

### Key Findings

- Table recognition is the primary performance bottleneck (2-6 seconds per table), being even slower for complex tables
- The pypdfium backend is about 2x faster but has poorer conversion quality (especially for tables)
- Increasing the number of threads from 4 to 16 only improves performance by about 6% (on M3 Max), indicating that it is mainly limited by single-threaded model inference
- OCR mode (EasyOCR) significantly increases latency (30+ seconds/page) and is currently only suitable for scanned documents

## Highlights & Insights

- **Engineering Completeness**: A complete pipeline from PDF parsing to structured output, bundled with two SOTA AI models, MIT licensed, and installable via a single click (`pip install`). It holds extremely high practical value for landing RAG applications
- **Modular Design**: Designed with the `BaseModelPipeline` base class and `Callable` interfaces, allowing third parties to easily replace or add models. This architectural design is highly exemplary
- **Pragmatic Technical Choices**: Instead of striving for end-to-end multimodality, it combines specialized models with traditional parsing to achieve a reasonable balance between speed and quality on commodity hardware

## Limitations & Future Work

- GPU acceleration is not yet fully optimized, currently relying primarily on CPU inference
- OCR support (EasyOCR) is slow and offers limited accuracy
- Lacks specialized models for formula recognition and code block identification
- Does not support non-PDF input formats such as DOCX and HTML
- There is still room for improvement in table recognition on highly complex tables
- Plans exist to add figure classifiers, formula recognizers, code recognizers, etc.

## Related Work & Insights

- **vs Commercial Solutions (e.g., Azure Document Intelligence)**: A capability and quality gap remains, but running completely locally, under an MIT license, with zero cost serve as its core advantages
- **vs Multimodal VLMs**: Docling is 10-100x faster and has a low resource footprint, making it suitable for batch processing, whereas VLMs are better suited for a small number of documents requiring deep understanding
- **vs pymupdf**: The AGPL license restricts commercial usage; Docling's MIT license encounters no such issues

## Rating

- Novelty: ⭐⭐⭐ Since this is a technical report and the models/methods originate from prior work, its main contribution lies in engineering integration.
- Experimental Thoroughness: ⭐⭐⭐ Performance benchmarks are provided, but accuracy comparisons with competing products are lacking.
- Writing Quality: ⭐⭐⭐⭐ The technical report is well-structured and contains comprehensive information.
- Value: ⭐⭐⭐⭐⭐ It fills the gap in open-source document conversion tools and contributes significantly to the RAG ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)
- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[ECCV 2024\] CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)
- [\[ECCV 2024\] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)

</div>

<!-- RELATED:END -->
