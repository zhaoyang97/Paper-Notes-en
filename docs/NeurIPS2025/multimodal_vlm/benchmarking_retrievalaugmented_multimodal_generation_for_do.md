---
title: >-
  [Paper Note] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering
description: >-
  [NeurIPS 2025][Multimodal VLM][DocRAG] This paper introduces the MMDocRAG benchmark (4,055 expert-annotated QA pairs) to systematically evaluate 60 VLMs/LLMs and 14 retrievers on quote selection and interleaved text-image answer generation in multimodal document RAG. Results reveal that the strongest model, GPT-4.1, achieves only 70.2% Quote Selection F1, while fine-tuning yields substantial performance gains.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "DocRAG"
  - "multimodal QA"
  - "quote selection"
  - "interleaved text-image generation"
  - "benchmark"
date: 2026-05-08
content_hash: b1220342c7bf97f9
---

# Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering

**Conference**: NeurIPS 2025
**arXiv**: [2505.16470](https://arxiv.org/abs/2505.16470)  
**Code**: [https://github.com/MMDocRAG/MMDocRAG](https://github.com/MMDocRAG/MMDocRAG)  
**Area**: Multimodal VLM / Document Question Answering / Retrieval-Augmented Generation
**Keywords**: DocRAG, multimodal QA, quote selection, interleaved text-image generation, benchmark

## TL;DR
This paper introduces the MMDocRAG benchmark (4,055 expert-annotated QA pairs) to systematically evaluate 60 VLMs/LLMs and 14 retrievers on quote selection and interleaved text-image answer generation in multimodal document RAG. Results reveal that the strongest model, GPT-4.1, achieves only 70.2% Quote Selection F1, while fine-tuning yields substantial performance gains.

## Background & Motivation
Document Visual Question Answering (DocVQA) requires locating evidence in long documents containing text, figures, tables, and other multimodal content, followed by cross-modal reasoning. Existing DocRAG approaches suffer from two core limitations: (1) **Single-modal bias**—generated answers heavily rely on plain text and neglect visual information such as charts and tables, which are critical for user comprehension; (2) **Evaluation gap**—existing benchmarks only assess retrieval recall or text answer quality, with no benchmark capable of evaluating a model's ability to select relevant multimodal evidence from noisy candidates or to integrate multimodal content into coherent answers. These two gaps significantly hinder the development of multimodal RAG systems.

## Core Problem
How can one systematically evaluate a model's ability to select multimodal evidence and generate interleaved text-image answers in document-level RAG scenarios? Specifically, given a question and a set of candidate quotes (text + images) containing both gold and noisy entries, can a model (1) correctly identify relevant evidence and (2) produce high-quality, coherent answers that fuse text and images? This is the first DocVQA/DocRAG benchmark focused on evaluating multimodal interleaved generation.

## Method

### Overall Architecture
MMDocRAG is built around a four-stage annotation pipeline and a comprehensive task-and-evaluation framework:

- **Input**: User question + document corpus (containing text quotes $T$ and image quotes $I$)
- **Stage 1: Multimodal Retrieval**—retrieve top-$k$ relevant quotes from documents
- **Stage 2: Quote Selection and Multimodal Generation**—given a fixed candidate set (15 or 20 quotes, containing gold and noisy entries), the model filters relevant evidence and generates an interleaved text-image answer
- **Output**: Multimodal answer containing both text and image quotes

### Key Designs

1. **Four-Stage Annotation Pipeline**:

    - **Document Parsing and Evidence Selection**: MinerU (based on LayoutLMv3) is used for layout detection, segmenting documents into content-aware chunks ("quotes"); text quotes are stored as text, while image quotes are stored in three formats: original image, OCR-text, and VLM-text.
    - **Multimodal Answer Generation**: 943 existing questions are filtered and 3,349 new questions are annotated across eight question types (descriptive / comparative / procedural / explanatory / causal / analytical / inferential / applicative); VLMs generate initial interleaved text-image answers, which are subsequently revised by human annotators.
    - **Gold Quote Annotation**: A dense retriever retrieves top-20 relevant text quotes; an LLM selects among them and inserts citation markers into answers; expert verification follows.
    - **Hard Negative Augmentation**: Retrieved high-similarity but irrelevant quotes serve as noise, constructing candidate sets of 15 (5 images + 10 text) or 20 (8 images + 12 text) quotes, with gold quotes comprising only 13.5%–18.0% of each set.

2. **Novel Evaluation Framework**:

    - **Quote Selection Metrics**: Precision, Recall, and F1 are computed separately for text and image quotes, with the mean taken as the overall F1.
    - **Surface-level Similarity**: BLEU and ROUGE-L.
    - **LLM-as-Judge**: Scoring (0–5) across five dimensions: fluency, citation quality, text-image coherence, reasoning logic, and factual accuracy.

3. **Quality Assurance**:

    - Semi-automatic validation: VLMs check visual content coherence; LLMs check textual accuracy; retention rates are 90.2% (answers) and 93.5% (gold quotes).
    - Human cross-validation: two annotator groups validate each other's work; inter-annotator Quote Selection F1 reaches 89.7/91.4 and answer quality 4.23/4.17.

### Dataset Statistics
- 222 documents, 10 domains, averaging 67 pages / 33k words
- 4,055 QA pairs (dev: 2,055 / eval: 2,000)
- 52.0% cross-page questions, 39.2% multi-image questions, 61.7% cross-modal questions
- 48,618 text quotes (4,640 gold), 32,071 image quotes (6,349 gold)

## Key Experimental Results

### Main Results (20 quotes, pure-text input)

| Model | Quote F1 | Answer Quality Avg |
|------|----------|-------------|
| GPT-4.1 | **68.3** | **4.07** |
| Qwen2.5-72B-Inst | 59.1 | 3.75 |
| DeepSeek-V3 | 61.1 | 3.74 |
| Gemini-2.5-Pro | 65.1 | 3.79 |
| Grok-3-beta | 57.9 | 3.83 |

### Main Results (20 quotes, multimodal input VLM)

| Model | Quote F1 | Answer Quality Avg |
|------|----------|-------------|
| GPT-4.1 (MM) | **70.2** | **4.14** |
| Gemini-2.5-Pro (MM) | 65.4 | 3.88 |
| Gemini-2.5-Flash (MM) | 62.4 | 3.76 |
| Claude-3.5-Sonnet (MM) | 62.5 | 3.65 |
| Qwen2.5-VL-72B (MM) | 57.5 | 3.47 |

### Fine-Tuning Results

| Model | F1 (before→after) | Avg (before→after) |
|------|-------------------|-------------------|
| Qwen2.5-72B-Inst | 59.1→64.9 | 3.75→3.97 |
| Qwen2.5-32B-Inst | 58.9→65.1 | 3.63→3.93 |
| Qwen2.5-14B-Inst | 54.7→59.4 | 3.49→3.84 |
| InternVL3-9B (VLM) | 50.9→60.3 | 3.12→3.87 |

### Retrieval Results (Recall@20)

| Method | Text Recall | Image Recall |
|------|------------|--------------|
| BGE (text) | 47.0 | 74.2 |
| ColQwen (visual) | 36.0 | 84.3 |
| ColQ+BGE (hybrid) | 47.7 | 85.2 |

### Ablation Study
- **Multimodal vs. pure-text input**: Top closed-source VLMs (GPT-4.1, Gemini) perform marginally better with multimodal input, but the gap is small (GPT-4.1: F1 70.2 vs. 68.3); open-source / smaller VLMs perform worse with multimodal input than with pure-text input (Qwen2.5-VL-7B: F1 16.6 vs. LLM 45.8, a 176% improvement in favor of text).
- **VLM-text vs. OCR-text**: VLM-generated image descriptions substantially outperform OCR-extracted text, with an average F1 gain of 6.5 points and an answer quality gain of 0.14.
- **Thinking mode**: Thinking mode consumes 3× the output tokens but does not significantly improve performance, suggesting that step-by-step reasoning offers limited benefit for multimodal quote selection and integration.
- **Positional bias**: Model accuracy in selecting gold quotes is highest for the first position and declines progressively thereafter, exhibiting a "Lost in the Middle" effect.
- **End-to-end RAG**: Generation quality correlates positively with retrieval quality; reducing BGE recall from 100% to 71% causes GPT-4.1's F1 to drop by 22.5% and answer quality by 14.7%; query expansion and multi-retriever ensemble partially close the gap.

## Highlights & Insights
- **First DocRAG benchmark evaluating multimodal interleaved generation**: Fills the gap in assessing models' ability to select multimodal evidence from noisy quotes and generate interleaved text-image answers.
- **Carefully designed hard negatives**: Gold quotes constitute only 13.5%–18.0% of each candidate set, effectively probing models' ability to distinguish relevant from irrelevant information.
- **Unprecedented experimental scale**: 60 models + 14 retrievers + 9 fine-tuned models, providing exceptionally broad coverage.
- **Counter-intuitive yet important findings**: Smaller VLMs perform better with pure-text input than with multimodal input; thinking mode yields no notable gains.
- **Fine-tuning yields substantial improvements**: Even simple LoRA fine-tuning markedly improves quote selection and generation quality, providing a clear practical direction.

## Limitations & Future Work
- **Limited document sources**: Based on 313 documents from MMDocIR, with insufficient domain coverage (e.g., code documentation and legal documents are absent).
- **LLM-as-Judge limitations**: Five-dimension scoring relies on LLM evaluation, which may introduce assessment bias.
- **Selection rather than generation of multimodal content**: Models are required to select images from existing quotes rather than generate them from scratch; end-to-end multimodal generation capability is not assessed.
- **Fixed candidate set sizes**: Only 15 and 20 candidate quotes are tested; larger-scale scenarios (e.g., 50 or 100 quotes) are not covered.
- **Visual token compression unexplored**: The Gemini series incurs token counts similar to pure-text input under multimodal settings, hinting that visual token compression may be a key factor, yet this is not analyzed in depth.

## Related Work & Insights
- **vs. MuRAR / M2RAG**: MuRAR and M2RAG support multimodal answer generation but are limited to the web domain and lack evidence annotation. MMDocRAG covers 10 domains and provides quote-level evidence annotation and hard negatives.
- **vs. MMLongBench-Doc / DocBench**: These benchmarks treat DocVQA as a long-context task but evaluate only text answers. MMDocRAG is the first to support evaluation of multimodal interleaved output.
- **vs. M3DocVQA / M-Longdoc / MMDocIR**: Although these benchmarks include retrieval sub-tasks, answers remain purely textual. MMDocRAG extends them with quote selection evaluation and multimodal answer generation evaluation.

This work highlights the substantial gap in current VLMs' ability to integrate multimodal evidence, providing important reference for designing future multimodal RAG systems. The finding that VLM-text substantially outperforms OCR-text suggests that VLM-based image description should be prioritized over simple OCR in document RAG pipelines. The significant fine-tuning gains for smaller models (e.g., fine-tuned Qwen2.5-14B approaches GPT-4o) demonstrate that task-specific fine-tuning remains an efficient and practical optimization strategy. The observation that smaller VLMs underperform their LLM counterparts with text input highlights a capability bottleneck in current open-source VLMs when processing long multimodal sequences.

## Rating
- Novelty: ⭐⭐⭐⭐ — First DocRAG benchmark focused on multimodal interleaved generation; the quote selection evaluation design is novel, though the overall contribution remains a benchmark paper.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 60 models + 14 retrievers + fine-tuning + multi-dimensional fine-grained analysis; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich tables, though the main text is dense and some analyses are relegated to the appendix.
- Value: ⭐⭐⭐⭐ — Provides essential evaluation infrastructure and empirical findings for multimodal DocRAG, offering clear guidance for future research in this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](../../CVPR2025/multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multimodal_vlm/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)

</div>

<!-- RELATED:END -->
