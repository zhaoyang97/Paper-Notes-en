---
title: >-
  [Paper Note] Multimodal OCR: Parse Anything from Documents
description: >-
  [CVPR 2026][Multimodal VLM][Document Parsing] This paper proposes the Multimodal OCR (MOCR) paradigm, which unifies the parsing of text and graphics (charts, diagrams, UI components, etc.) in documents into structured textual representations (plain text + SVG code). The trained 3B-parameter dots.mocr model ranks second only to Gemini 3 Pro on OCR Arena, achieves a state-of-the-art score of 83.9 on olmOCR Bench, and surpasses Gemini 3 Pro on the image-to-SVG benchmark.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Document Parsing
  - Graphics Parsing
  - SVG Generation
  - Vision-Language Model
  - Multimodal OCR
date: 2026-05-08
content_hash: 5829abc255adeac3
---

# Multimodal OCR: Parse Anything from Documents

**Conference**: CVPR 2026
**arXiv**: [2603.13032](https://arxiv.org/abs/2603.13032)
**Code**: [https://github.com/rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr)
**Area**: Document Parsing / Multimodal VLM / OCR
**Keywords**: Document Parsing, Graphics Parsing, SVG Generation, Vision-Language Model, Multimodal OCR

## TL;DR
This paper proposes the Multimodal OCR (MOCR) paradigm, which unifies the parsing of text and graphics (charts, diagrams, UI components, etc.) in documents into structured textual representations (plain text + SVG code). The trained 3B-parameter dots.mocr model ranks second only to Gemini 3 Pro on OCR Arena, achieves a state-of-the-art score of 83.9 on olmOCR Bench, and surpasses Gemini 3 Pro on the image-to-SVG benchmark.

## Background & Motivation
In the era of large language models and multimodal models, document parsing serves as a core data engine for pretraining and retrieval. However, documents convey information not only through text but also through graphics such as charts, diagrams, flowcharts, and UI elements. Existing document parsing pipelines are fundamentally text-centric: they focus on text recognition and organization while simply cropping non-textual elements as pixel images. This leads to the discarding of substantial structured and semantic information embedded in document graphics, making current document parsing inherently lossy and limiting the amount of supervisory signal extractable from documents. Recent advances in vision-language models have made it possible to recover structured representations from visual elements in documents—going beyond mere description to generate executable representations (e.g., SVG code) that enable reconstruction of the original structure.

## Core Problem
How can document parsing be extended from "text extraction only" to "parsing everything in a document"—including text, layout structure, tables, and information-dense graphics such as charts, diagrams, icons, and UI components? Key challenges include: (1) scarcity of supervisory signals for graphics, as real-world documents rarely provide aligned programmatic representations; (2) the inherent non-uniqueness of renderable programs, where different code can produce visually identical outputs; and (3) the requirement for precise visual localization combined with long-sequence structured generation.

## Method

### Overall Architecture
MOCR defines document page parsing as the generation of an ordered sequence of parsed elements $S = [(B_1, c_1, p_1), \ldots, (B_k, c_k, p_k)]$, where $B_k$ denotes a spatial region, $c_k$ denotes a semantic category, and $p_k$ denotes a type-specific payload. For text regions, $p_k$ is a textual transcription (plain text, table markup, or LaTeX); for graphic regions, $p_k$ is a renderable structured representation (SVG code). The sequence is generated in human reading order.

### Key Designs

1. **High-Resolution Visual Encoder**: A 1.2B-parameter visual backbone trained entirely from scratch, supporting native high-resolution inputs of up to ~11M pixels. Training from scratch ensures the encoder develops feature representations natively optimized for document parsing, capable of simultaneously handling dense text and geometrically sensitive visual symbols (chart markers, diagram strokes, etc.).

2. **Structured Language Decoder**: Qwen2.5-1.5B is used as the autoregressive decoder. The 1.5B scale represents a trade-off between capacity and cost for unified MOCR parsing—smaller models struggle to simultaneously handle heterogeneous page content and generate long structured outputs, while larger decoders increase training and inference costs. A base model rather than a chat model is used as initialization, providing a neutral starting point for learning non-natural, highly structured target sequences.

3. **Staged Training Strategy**: Three-stage large-scale pretraining followed by instruction fine-tuning:

    - **Stage 1**: General visual training to establish a stable vision-language interface.
    - **Stage 2**: Joint training on general visual data and plain-text document parsing supervision, building a foundation for text parsing.
    - **Stage 3**: Introduction of MOCR-specific objectives (reducing the proportion of general visual data while increasing multimodal document parsing and graphics parsing / image-to-SVG tasks), with progressively increasing input resolution to match growing task difficulty.
    - **Supervised Fine-Tuning (SFT)**: Uses a high-quality curated dataset, prioritizing supervision reliability and task usability. Two checkpoints are released: dots.mocr and dots.mocr-svg (the latter with an increased SVG data share and higher weighting of complex SVG programs).

4. **Large-Scale Data Engine**: Four complementary data sources:

    - **PDF Documents**: Automatically annotated using dots.ocr, with stratified sampling by language, domain, and layout complexity.
    - **Web Pages**: Crawled and rendered as page images; HTML/DOM provides aligned structured signals and natively contains SVG icons and charts.
    - **SVG Graphics**: Native SVG assets collected from the web, processed via svgo for cleaning, deduplication, and complexity-based balanced sampling.
    - **General Visual Data**: Maintained to preserve broad visual capabilities (grounding, counting, etc.).

5. **OCR Arena Automated Evaluation**: An automatic evaluation framework based on the LLM-as-Judge paradigm, using Gemini 3 Flash as the judge for pairwise comparisons. A symmetric evaluation protocol (presenting comparisons in both orders to eliminate position bias) is adopted, with an Elo rating system for ranking and 1,000 rounds of bootstrap resampling for statistical robustness.

### Loss & Training
- A unified autoregressive objective is used throughout: given an input image and task instruction, the model predicts the structured parsing sequence.
- Optimization stability is controlled via mixture ratio reweighting and curriculum scheduling.
- SVG-related preprocessing—including normalization, viewBox normalization, and complexity reduction—is incorporated as part of the data engine.
- Input resolution is increased progressively across training stages.

## Key Experimental Results

| Model | olmOCR-Bench Total | Elo Score |
|---|---|---|
| Gemini 3 Pro | — | 1210.7 |
| **dots.mocr** | **83.9** | **1124.7** |
| dots.ocr | 79.1 | 1086.2 |
| HunyuanOCR | — | 984.2 |
| MonkeyOCR-pro-3B | 75.8 | 781.1 |

**Image-to-SVG (ISVGEN Reconstruction Scores)**:

| Model | UniSVG Total | ChartMimic | Design2Code | ChemDraw |
|---|---|---|---|---|
| Gemini 3 Pro | 0.735 | 0.788 | 0.760 | 0.783 |
| OCRVerse | 0.763 | 0.799 | — | — |
| dots.mocr | 0.894 | 0.772 | 0.801 | 0.660 |
| **dots.mocr-svg** | **0.902** | **0.905** | **0.834** | **0.797** |

### Ablation Study
- dots.mocr achieves TextEdit = 0.031 and ReadOrderEdit = 0.029 on OmniDocBench v1.5, both state-of-the-art results.
- Per-category analysis on olmOCR-Bench shows that dots.mocr performs best on ArXiv, Old scans math, Tables, and Multi-column categories, while room for improvement remains in Old scans and Headers & footers.
- dots.mocr-svg shows significant gains over dots.mocr on graphics parsing (UniSVG +0.008, ChartMimic +0.133), confirming the effectiveness of increasing SVG training data during the SFT stage.
- The 3B-parameter model achieves strong performance on CharXiv description and reasoning tasks (77.4 / 55.3), surpassing Qwen3-VL-4B.

## Highlights & Insights
- **Paradigm innovation**: MOCR elevates document graphics from "cropped pixels" to "first-class parsing targets," converting them into renderable SVG code and opening a new direction for documents as multimodal pretraining data sources.
- **End-to-end system engineering**: The work presents industrial-grade completeness, spanning data engine design (four sources + quality control), training strategy (three-stage pretraining + SFT), and evaluation framework (OCR Arena).
- **Efficiency**: The compact 3B-parameter model competes with Gemini 3 Pro on document parsing and even surpasses it on SVG reconstruction, underscoring the importance of data and training strategy.
- **Evaluation methodology**: OCR Arena's symmetric dual-trial evaluation, Elo ranking, and bootstrap resampling constitute a meaningful complement to traditional character-matching metrics.
- **Reproducibility**: Code and model weights are publicly released.

## Limitations & Future Work
- The current MOCR formulation is task-conditioned; document parsing and SVG parsing must be run separately, and unified single-pass output has not yet been achieved.
- Complex real-world images and natural photographs lack concise programmatic descriptions and are retained as raster content.
- The non-uniqueness of SVG targets, though mitigated by normalization, remains a fundamental challenge during training.
- Performance on certain categories (formulas, tables, headers and footers) still has room for improvement.
- Evaluation relies on an LLM judge (Gemini 3 Flash), which may introduce evaluation bias.
- The current work uses SVG exclusively as the graphical representation; future extensions could cover TikZ, D3.js, CAD formats, and others.

## Related Work & Insights
- **vs. Traditional OCR pipelines (PaddleOCR, Marker, etc.)**: Cascaded pipelines handle only text and do not support graphics parsing. MOCR also outperforms these on text parsing.
- **vs. End-to-end OCR VLMs (GOT-OCR, DeepSeek-OCR)**: These models achieve end-to-end text parsing but do not support decoding graphics into structured code.
- **vs. OCRVerse**: OCRVerse aims to unify multiple OCR tasks, but MOCR exceeds it by +0.139 on UniSVG and covers a broader set of downstream benchmarks.
- **vs. Specialized graphics parsers (StarVector, OmniSVG)**: These are task-specific systems, whereas MOCR handles both document parsing and graphics parsing within a single model.
- **vs. Gemini 3 Pro**: dots.mocr ranks second on Elo (behind only Gemini 3 Pro), while comprehensively surpassing it on SVG benchmarks, with a parameter count of 3B—potentially two orders of magnitude smaller than Gemini.

**Broader Implications**:
- The "document graphics → executable code" paradigm offers a new pathway for constructing large-scale multimodal pretraining corpora: each chart parsed into SVG yields an (image, code, text) triplet.
- The data engine design (multi-source + quality control + complexity-balanced sampling) provides a valuable reference for large-scale training data construction.
- The LLM-as-Judge evaluation methodology of OCR Arena is generalizable to the evaluation of other structured generation tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — MOCR represents a genuine paradigm shift, extending OCR from text to all structurally representable document elements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers document parsing, graphics parsing, and general VQA; comprehensively evaluated across multiple benchmarks; OCR Arena provides complementary assessment.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear and system design is complete, though the paper is lengthy and some details are deferred to the appendix.
- **Value**: ⭐⭐⭐ — Document parsing is not directly aligned with current research directions, but the data engine concept of "converting everything into structured representations for pretraining" is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive](../../AAAI2026/multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[CVPR 2026\] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](paddleocr_vl_coarse_to_fine_document_parsing.md)

</div>

<!-- RELATED:END -->
