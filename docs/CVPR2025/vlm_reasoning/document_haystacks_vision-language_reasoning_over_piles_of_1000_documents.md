---
title: >-
  [Paper Note] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents
description: >-
  [VLM Reasoning] Proposes two large-scale document retrieval benchmarks, DocHaystack and InfoHaystack (1000+ documents per question), and V-RAG, a vision-centric retrieval-augmented generation framework, which improves Recall@1 by 9%-11% over the best baseline.
tags:
  - "VLM Reasoning"
date: 2026-05-08
content_hash: 40e3534f8ac7338e
---

# Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents

| Attribute | Value |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2411.16740](https://arxiv.org/abs/2411.16740) |
| Code | [GitHub](https://github.com/Vision-CAIR/dochaystacks) |
| Area | Human Understanding / Document Understanding / Multimodal Reasoning |
| Keywords | document retrieval, visual QA, RAG, large multimodal model, benchmark |

## TL;DR

Proposes two large-scale document retrieval benchmarks, DocHaystack and InfoHaystack (1000+ documents per question), and V-RAG, a vision-centric retrieval-augmented generation framework, which improves Recall@1 by 9%-11% over the best baseline.

## Background & Motivation

### Background

Large Multimodal Models (LMMs) have made significant progress in vision-language understanding, but still face difficulties when handling large-scale image/document collections. Existing multi-image VQA benchmarks are limited in scale, with at most ~30 images paired per question, which falls far short of real-world requirements.

### Limitations of Prior Work

1. **Insufficient Benchmark Scale**: Benchmarks such as RetVQA and WebQA only associate $\le 30$ images per question, failing to simulate real-world large-scale document retrieval scenarios.
2. **Answer Ambiguity**: In existing datasets like DocVQA and InfographicVQA, a large number of "generic questions" (e.g., "What is the table number?") can be answered by multiple documents, leading to unreliable evaluation.
3. **Limited Context Length of LMMs**: Current LMMs cannot directly handle hundreds or thousands of high-resolution document images.
4. **Insufficient Precision of Retrieval Methods**: A single visual encoder struggles to comprehensively capture multi-scale information such as text, symbols, and charts in documents.

### Goal

1. Establish a large-scale document retrieval benchmark with **1000 documents** per question while ensuring the uniqueness of the answers.
2. Design an effective visual retrieval framework to enable LMMs to retrieve and reason across hundreds or thousands of documents.

### Key Insight & Core Idea

Ensures benchmark quality through a three-stage data filtering pipeline (LLM filtering of generic questions $\rightarrow$ manual review $\rightarrow$ filtering of general knowledge questions); proposes V-RAG, which combines ensemble retrieval of multiple visual encoders (CLIP + SigLIP + OpenCLIP) with a two-stage LMM-filter.

## Method

### Overall Architecture

V-RAG consists of three steps: (1) Visual Encoder Ensemble: utilizes three encoders—CLIP, SigLIP, and OpenCLIP—to compute and average the question-document similarity, selecting the top-m documents; (2) LMM-Filter Module: uses an LMM to determine sequentially whether each candidate document can answer the question, filtering out irrelevant documents; (3) LMM-VQA Module: inputs the top-k relevant documents along with the question into the LMM to generate the final answer.

### Key Design 1: Three-Stage Data Filtering Pipeline (Benchmark Construction)

- **Function**: Ensures that each question in the benchmark has a unique answer across the entire document set.
- **Mechanism**:
    - Step 1: Uses GPT-4o to filter out "generic questions" (questions that can be answered by multiple documents).
    - Step 2: Conducts manual review to verify the existence of unique identifiers (names, dates, titles, etc.) and uses OCR + full-text search to ensure the answer does not appear in other documents.
    - Step 3: Filters out "general knowledge questions"—questions that GPT-4o can answer without images (26.4% of questions in DocVQA and 54.9% in InfographicVQA can be answered directly by GPT-4o without viewing the image).
- **Design Motivation**: The core challenge of large-scale retrieval benchmarks is not the scale itself, but the ambiguity of answers. Benchmarks without rigorous filtering cannot reliably evaluate models.

### Key Design 2: Visual Encoder Ensemble

- **Function**: Combines the complementary capabilities of multiple visual encoders to improve retrieval accuracy.
- **Mechanism**: For each question-document pair, CLIP (ViT-L/14@336), SigLIP (ViT-SO400M/14@384), and OpenCLIP (ConvNeXt-XXL@1024) are used to compute cosine similarities $Sim_c$, $Sim_s$, and $Sim_o$, respectively, which are averaged to obtain $Sim_{avg}$.
- **Design Motivation**: Different encoders have distinct advantages—ConvNeXt is powerful for high-resolution processing, CLIP is strong for text descriptions, and SigLIP offers more stable global matching. Experimental verification shows that the three-encoder ensemble outperforms any single encoder.

### Key Design 3: Two-Stage LMM-Filter

- **Function**: Leverages the reasoning capability of LMMs to further refine retrieval results.
- **Mechanism**: For the top-m candidate documents, each is paired with the question and input into the LMM (LLaVA-OneVision), with the prompt "Can this image answer the question? Answer only Yes or No". Only documents with a "Yes" response are retained.
- **Design Motivation**: Similarity matching by visual encoders captures shallow semantics, whereas LMMs can perform deeper question-document relational reasoning. The two stages complement each other, achieving high efficiency in coarse filtering and high accuracy in fine filtering.

## Key Experimental Results

### Retrieval Results (Recall@1)

| Method | DocH-100 | DocH-1000 | InfoH-100 | InfoH-1000 |
|------|----------|-----------|-----------|------------|
| BM25 (OCR) | 63.30 | 56.88 | 56.77 | 38.71 |
| CLIP | 46.79 | 23.85 | 69.68 | 45.81 |
| OpenCLIP | 58.72 | 34.86 | 72.26 | 53.55 |
| **V-RAG** | **81.65** | **66.06** | **79.35** | **64.52** |

V-RAG achieves a Recall@1 on DocHaystack-1000 that is +31.2 percentage points higher than the best single encoder (OpenCLIP).

### VQA Results

| Method | DocH-100 | DocH-1000 | InfoH-100 | InfoH-1000 |
|------|----------|-----------|-----------|------------|
| GPT-4o (Direct) | 27.52 | - | 23.87 | - |
| GPT-4o+V-RAG | 81.65 | 66.97 | 65.16 | 56.77 |
| Qwen2-VL-f.t.+V-RAG | **86.24** | **73.39** | **67.10** | **60.00** |

GPT-4o directly processing 200 documents achieves an accuracy of only 23.85%, which surges to 72.48% (+48.63%) after adding V-RAG.

### Ablation Study

| CLIP | SigLIP | OpenCLIP | VLM-filter | DocH-1000 R@1 |
|------|--------|----------|------------|---------------|
| ✓ | | | | 23.85 |
| | | ✓ | | 34.86 |
| ✓ | ✓ | ✓ | | 56.88 |
| ✓ | ✓ | ✓ | ✓ | **66.06** |

The encoder ensemble contributes +22 percentage points, and the LMM-Filter contributes an additional +9 percentage points.

### Key Findings

- 54.9% of questions in InfographicVQA can be answered by GPT-4o without viewing the images, exposing severe language bias.
- LLaVA-OneVision cannot run in scenarios with more than 100 documents due to context length limitations.
- Fine-tuning Qwen2-VL (trained with 1-10 distractor images) can further improve robustness by approximately 4-7 percentage points.
- Question type distribution: DocHaystack focuses on tables/lists, while InfoHaystack focuses on charts/texts.

## Highlights & Insights

1. **Profound Benchmark Design Philosophy**: The three-stage filtering pipeline ensures answer uniqueness; in particular, the "general knowledge filtering" step reveals the language bias issues in existing benchmarks.
2. **Engineering Wisdom of V-RAG**: Achieves massive improvements purely through modular combinations (encoder ensemble + LMM filtering) without training new models or changing architectures.
3. **1000-Document Scale**: Extends multi-image retrieval to the thousand-scale for the first time, exposing the shortcomings of current LMMs' long-context capabilities.
4. **Significant Complementary Effects of Encoders**: The three-encoder ensemble achieves over 30 percentage points higher performance than the strongest single encoder.

## Limitations & Future Work

1. The final retained dataset is relatively small (109 questions for DocVQA / 155 questions for InfographicVQA), making the testbed size somewhat limited.
2. The LMM-Filter in V-RAG requires performing one LMM inference for each candidate document (top-60), leading to non-trivial latency.
3. The "needle in a haystack" scenario might be overly artificial compared to real-world question distributions, which are far more complex.
4. The benchmark only covers English documents, leaving multilingual document retrieval scenarios unaddressed.

## Related Work & Insights

- **RetVQA** (ECCV 2022): A small-scale retrieval benchmark with ≤30 images per question.
- **MIRAGE** (ICML 2024): CLIP retriever + LMM reasoning; V-RAG significantly outperforms it using a multi-encoder ensemble.
- **Success of RAG in NLP**: V-RAG systematically applies RAG concepts to visual document retrieval.
- **Insights**: Future large-scale multimodal reasoning may require a "hierarchical retrieval" strategy—coarse filtering with lightweight encoders and fine filtering with heavy LMMs.

## Rating

⭐⭐⭐⭐ — Rigorous benchmark design, clear methodology, and comprehensive experiments. Although the technical novelty of V-RAG itself is not cutting-edge (primarily an engineering combination), the contributions of the benchmark and the issues it exposes (LMM long-context weaknesses, language bias) are highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/vlm_reasoning/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/vlm_reasoning/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/vlm_reasoning/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](../../ACL2025/vlm_reasoning/mammoth_vl_multimodal_reasoning.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/vlm_reasoning/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)

</div>

<!-- RELATED:END -->
