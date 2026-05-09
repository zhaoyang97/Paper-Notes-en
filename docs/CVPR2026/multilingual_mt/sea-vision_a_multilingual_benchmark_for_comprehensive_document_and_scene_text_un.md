---
title: >-
  [Paper Note] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia
description: >-
  [CVPR 2026][Multilingual benchmark] This paper introduces SEA-Vision, a benchmark that unifies evaluation of document parsing (15,234 pages) and text-centric VQA (7,496 QA pairs) across 11 Southeast Asian languages. A re-rendering strategy eliminates visual–textual misalignment in multilingual VQA, revealing severe performance degradation of 3–7× for MLLMs on low-resource SEA languages.
tags:
  - CVPR 2026
  - Multilingual benchmark
  - Southeast Asia
  - document parsing
  - text VQA
  - low-resource languages
  - MLLM evaluation
date: 2026-05-08
content_hash: f002f6c454f8a36f
---

# SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia

**Conference**: CVPR 2026  
**arXiv**: [2603.15409](https://arxiv.org/abs/2603.15409)  
**Code**: None  
**Area**: Multilingual Document Understanding  
**Keywords**: Multilingual benchmark, Southeast Asia, document parsing, text VQA, low-resource languages, MLLM evaluation

## TL;DR

This paper introduces SEA-Vision, a benchmark that unifies evaluation of document parsing (15,234 pages) and text-centric VQA (7,496 QA pairs) across 11 Southeast Asian languages. A re-rendering strategy eliminates visual–textual misalignment in multilingual VQA, revealing severe performance degradation of 3–7× for MLLMs on low-resource SEA languages.

## Background & Motivation

**State of the Field**: Multilingual document and scene text understanding has become a core capability in search, finance, and public services. Leading MLLMs such as GPT-4o and the Qwen-VL series perform well on English and Chinese, yet existing benchmarks (DocVQA, TextVQA, MTVQA, etc.) are heavily skewed toward high-resource languages.

**Limitations of Prior Work**: (1) Document parsing and text-centric VQA are typically evaluated independently, precluding unified measurement of combined OCR and reasoning ability. (2) Multilingual VQA datasets commonly rely on OCR/translation-based annotation, where the text referenced by translated questions does not exist in the original image, causing severe visual–semantic misalignment. (3) The 11 SEA languages span four major script families—Latin, Brahmic, Arabic, and logographic—yet existing benchmarks provide extremely sparse coverage.

**Root Cause**: Southeast Asia is one of the most linguistically diverse regions in the world. Real-world applications involve dense text layouts, complex scripts, and heterogeneous document types coexisting, yet no benchmark simultaneously covers the major SEA languages or supports cross-task and cross-script evaluation. MTVQA covers only 9 languages (2 low-resource) and only VQA; CC-OCR covers 10 languages but only 1 low-resource language.

**Paper Goals**: (1) Construct the first benchmark that jointly evaluates document parsing and TEC-VQA for SEA multilingual settings. (2) Design an annotation methodology that resolves visual–textual misalignment. (3) Quantify the true capabilities of MLLMs on low-resource SEA languages.

**Starting Point**: Design a hybrid annotation pipeline (automatic filtering + MLLM-assisted annotation + native-speaker verification) and employ a re-rendering strategy that "paints" translated text back into images, eliminating visual–textual misalignment at the source.

**Core Idea**: By ensuring that visible text and QA language are fully consistent through re-rendering, construct a high-quality benchmark covering 11 SEA languages that unifies evaluation of document parsing and scene text VQA.

## Method

### Overall Architecture

SEA-Vision comprises two sub-tasks: (1) **Document Parsing**—extracting structured content from document images; 15,234 pages across 9 document types (academic papers, books, exam papers, magazines, newspapers, notes, research reports, slides, and textbooks), annotated with hierarchical page/block/line-level labels totaling 243,643 region annotations; (2) **TEC-VQA**—1,839 scene images with 7,496 QA pairs covering five reasoning capabilities (text recognition, numerical computation, comparative analysis, logical reasoning, and spatial understanding). The 11 languages are: EN, ZH, VI, TH, FIL, MS, ID, LO, KM, MY, and PT.

### Key Designs

1. **Four-Stage Document Parsing Annotation Pipeline**

    - **Function**: Filter high-quality multilingual pages from approximately 3M web-crawled PDFs and produce fine-grained annotations.
    - **Mechanism**: Four stages — (i) *Metadata annotation*: a layout detection model segments 10 region categories, followed by MLLM-based block-level language identification and page-type classification; (ii) *Rule-based scoring and ranking*: a weighted composite score $\text{Score} = 30 \cdot S_1 + 30 \cdot S_2 + 20 \cdot S_3 + 10 \cdot S_4 + 10 \cdot S_5$, where $S_1$ is block count, $S_2$ is text area ratio, $S_3$ is type diversity, and $S_4/S_5$ indicate the presence of figures/tables; the top 200 pages are selected per language × page-type group (200 × 11 × 9 = 19,800 pages total); (iii) *Region correction*: MLLM corrects OCR errors, UniMERNet re-parses formulas, and Intsig API corrects table structure; (iv) *Human verification*: inspects layout completeness, OCR reliability, sensitive content filtering, and cross-validates table and formula re-rendering.
    - **Design Motivation**: Balances large-scale automatic annotation with language balance for low-resource languages; after human filtering, 15,234 pages are retained to ensure annotation quality.

2. **TEC-VQA Re-rendering and Multi-round Verification Pipeline**

    - **Function**: Resolve visual–textual misalignment in multilingual VQA annotation.
    - **Mechanism**: (i) OCR detection extracts text regions from images → translate to target language → **font-matched inpainting renders translated text back into the image**, ensuring full consistency between visible text and QA language; (ii) MLLM generates English QA → translated to Chinese QA → independently answered → cross-lingual consistency check (pairs with inconsistent answers are discarded) → translated into the image-language version; (iii) back-translation verification + native-speaker review (removing unanswerable/trivial questions, standardizing numbers/units, checking language–image alignment, and annotating capability labels).
    - **Design Motivation**: Prior translation-based VQA extension only translates text without modifying images, causing QA-referenced text to be absent from the visible image. Re-rendering fundamentally eliminates this misalignment.

3. **Unified Evaluation Framework**

    - **Function**: Evaluate document parsing and TEC-VQA within a single framework.
    - **Mechanism**: Document parsing uses end-to-end NED (Normalized Edit Distance, ↓ lower is better) across 13 models spanning Pipeline, Expert, and General paradigms; TEC-VQA uses a zero-shot accuracy protocol.
    - **Design Motivation**: Each paradigm has distinct strengths and weaknesses; a unified framework enables fair comparison and precise bottleneck identification.

### Loss & Training

This is a benchmark paper; no model training is conducted. Standardized evaluation protocols and publicly released datasets are provided for community use.

## Key Experimental Results

### Document Parsing (End-to-End NED ↓)

| Model Type | Model | EN | KM | LO | MY | Avg (11 languages) |
|------------|-------|-----|-----|-----|-----|------|
| Pipeline | PaddleOCR-VL | 0.108 | 0.634 | 0.648 | 0.456 | 0.238 |
| Expert | dots.ocr | 0.144 | 0.311 | 0.386 | 0.313 | **0.186** |
| General | Qwen3-VL-32B | 0.133 | 0.727 | 0.406 | 0.479 | 0.225 |
| General | Gemini2.5-Pro | 0.154 | 0.278 | 0.195 | 0.214 | **0.159** |
| General | GPT-4o | 0.197 | 0.611 | 0.610 | 0.423 | 0.313 |

### Cross-Dimensional Analysis

| Comparison Dimension | Observation |
|----------------------|-------------|
| High-resource vs. Low-resource | EN/ZH accuracy ≈ 60–70%; KM/MY/LO only 10–20%; gap of **5–7×** |
| Script type | Latin/Chinese scripts NED < 0.2; Brahmic/Burmese/Khmer scripts NED > 0.5; gap of **3–5×** |
| Document type | Newspapers hardest (NED = 0.313); academic papers moderate (0.244); slides easiest (0.159) |
| Capability dimension | Spatial understanding and logical reasoning substantially weaker than text recognition |

### Key Findings

- Gemini2.5-Pro achieves the best overall performance (Avg NED 0.159) with a clear advantage on low-resource languages such as LO and KM.
- Even the strongest closed-source models exhibit substantial performance gaps on KM and MY.
- Models transfer reasonably well to Latin-script languages but show almost no effective generalization to languages with unique writing systems.

## Highlights & Insights

- **First unified benchmark for document parsing and scene VQA across SEA multilingual settings**: covers 11 languages including 7 low-resource ones, whereas the closest prior work CC-OCR includes only 1 low-resource language, filling a significant evaluation gap.
- **The re-rendering methodology contributes beyond the dataset itself**: font-matched inpainting of translated text back into images is directly transferable to data construction for other multilingual vision tasks.
- **Multi-round cross-lingual consistency verification**: independent answering in English and Chinese → consistency filtering → back-translation → native-speaker review effectively suppresses MLLM hallucinations and translation errors.
- **Quantifies multilingual bottlenecks of MLLMs**: NED gaps of 3–5× and accuracy gaps of 5–7× provide clear direction for model improvement.

## Limitations & Future Work

- For extremely low-resource languages (LO/KM/MY), approximately 100–200 pages per language per type may yield insufficiently precise statistical estimates.
- Typographic artifacts introduced by re-rendering may affect evaluation fairness; this bias is not analyzed.
- Only 9 printed document types are covered; handwriting, receipts, and other categories are not included.
- As a purely evaluative benchmark, no training set is provided, precluding direct use for training low-resource models.
- No model improvement strategies or data augmentation approaches targeting low-resource SEA languages are proposed.

## Related Work & Insights

- **vs. MTVQA**: 9 languages / 2 low-resource / 6,778 QA / VQA only. SEA-Vision: 11 languages / 7 low-resource / 7,496 QA + 15,234 document pages / dual tasks — substantially broader coverage.
- **vs. CC-OCR**: 10 languages but only 1 low-resource / 800 parsing pages. SEA-Vision: 7 low-resource / 15K parsing pages.
- **vs. OmniDocBench/Fox**: EN+ZH bilingual only, with no low-resource language coverage.
- The re-rendering strategy can inspire large-scale construction of multilingual document pre-training data—re-rendering English documents into multilingual versions for continual pre-training of models.

## Rating

- **Novelty**: ⭐⭐⭐ — Primarily a benchmark contribution; methodological innovation lies in the annotation pipeline design (re-rendering + cross-lingual consistency verification); no new model is proposed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 13 models across Pipeline/Expert/General paradigms with comprehensive evaluation across 11 languages.
- **Writing Quality**: ⭐⭐⭐⭐ — Scoring mechanism and annotation pipeline are described clearly; statistical analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a major gap in multilingual document understanding evaluation for Southeast Asia.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[AAAI 2026\] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](../../AAAI2026/multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative for Perplexity](../../ICLR2026/multilingual_mt/prior-based_noisy_text_data_filtering_fast_and_strong_alternative_to_perplexity.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](../../NeurIPS2025/multilingual_mt/mergebench_a_benchmark_for_merging_domain-specialized_llms.md)

<!-- RELATED:END -->
