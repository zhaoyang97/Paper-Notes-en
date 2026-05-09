---
title: >-
  [Paper Note] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables
description: >-
  [NeurIPS 2025][Multimodal VLM][Long-context understanding] This paper proposes NeedleInATable (NIAT), a benchmark that treats each table cell as a "needle" to evaluate the fine-grained perception capability of LLMs over long structured tables. It reveals that strong performance of existing models on complex downstream tasks may stem from dataset shortcuts rather than genuine table understanding.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Long-context understanding
  - structured tables
  - LLM evaluation benchmark
  - table perception
  - data synthesis
date: 2026-05-08
content_hash: 0ca4b15ef73bdcdb
---

# NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables

**Conference**: NeurIPS 2025
**arXiv**: [2504.06560](https://arxiv.org/abs/2504.06560)
**Code**: [GitHub](https://github.com/wlr737/NeedleInATable)
**Area**: Multimodal / VLM / Table Understanding
**Keywords**: Long-context understanding, structured tables, LLM evaluation benchmark, table perception, data synthesis

## TL;DR

This paper proposes NeedleInATable (NIAT), a benchmark that treats each table cell as a "needle" to evaluate the fine-grained perception capability of LLMs over long structured tables. It reveals that strong performance of existing models on complex downstream tasks may stem from dataset shortcuts rather than genuine table understanding.

## Background & Motivation

**State of the Field**: Long-context LLMs have advanced rapidly, and benchmarks such as Needle-in-a-Haystack have been used to assess long-context processing over unstructured text; however, structured table scenarios remain largely overlooked.

**Limitations of Prior Work**: Existing table benchmarks (e.g., WTQ, TabFact) primarily target high-level reasoning, neglecting models' fundamental fine-grained perception of individual table cells—which is precisely the cornerstone of reliable table applications.

**Root Cause**: Models achieve reasonable performance on complex table reasoning tasks, yet this may result from dataset-specific correlations or shortcuts rather than a genuine understanding of the two-dimensional table structure.

**Paper Goals**: Construct a benchmark to evaluate LLMs' ability to perceive every individual cell in long tables, and verify whether improving this capability can transfer to downstream table tasks.

**Starting Point**: Analogizing tables to "haystacks" and individual cells to "needles," the paper designs two atomic task types: Cell-Locating and Cell-Lookup.

**Core Idea**: If a model fails at the most basic cell localization, its strong results on complex tasks become suspect; enhancing NIAT capability can improve table understanding at a fundamental level.

## Method

### Overall Architecture

The NIAT benchmark comprises 750 tables and 287K test samples, covering three table structures (flat, horizontal, and hierarchical), three formats (Markdown, HTML, and image), and a variety of table sizes.

### Key Designs

1. **Cell-Locating Task**: Given row and column indices, the model is required to extract the content of the corresponding cell. → Evaluates the model's understanding of basic two-dimensional table structure. → Queries are automatically constructed via predefined templates without GPT-4o involvement. → Complements Cell-Lookup by probing different facets of capability.

2. **Cell-Lookup Task**: Given a simple lookup question whose answer corresponds to a specific cell (no aggregation required), the model is asked to retrieve the target cell. → Evaluates the model's ability to perform row–column cross-retrieval using semantic cues. → GPT-4o in-context learning is used to generate lookup questions, with invalid questions filtered via self-consistency. → Results show that LLMs excel at semantic matching but struggle with structural localization.

3. **strong2weak Data Synthesis Method**: GPT-4o is used to generate NIAT queries and chain-of-thought (CoT) reasoning responses on training-set tables, which are then used to fine-tune weaker models. → Direct fine-tuning with short answers risks overfitting to shortcuts. → Six more challenging lookup subtasks (e.g., cell-retrieval requiring full-table search) are designed to increase data diversity. → Only 12K synthetic samples yield significant improvements on both NIAT and downstream tasks.

### Loss & Training

- Fine-tuning is performed on Llama3.1-8B-Instruct and Qwen2.5-7B-Instruct.
- GPT-4o-synthesized CoT reasoning traces serve as target responses.
- Training set: 6K Cell-Locating + 6K Cell-Lookup = 12K samples in total.

## Key Experimental Results

### Main Results

| Model | Cell-Locating Avg | Cell-Lookup Avg | Overall |
|------|:-:|:-:|:-:|
| Llama3.1-8B-Instruct | 6.16 | 65.74 | 35.95 |
| Qwen2.5-7B-Instruct | 9.46 | 47.60 | 28.53 |
| TableGPT2 | 8.84 | 73.87 | 41.36 |
| GPT-4o | 26.00 | 68.30 | 47.15 |
| DeepSeek-R1 | 65.91 | 80.99 | 73.45 |
| Qwen3-30B-A3B | 16.49 | 78.62 | 47.55 |

### Downstream Task Gains (after fine-tuning with synthetic data)

| Model | WTQ | TabFact | HiTab | TABMWP | Avg |
|------|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-7B (original) | 52.90 | 70.00 | 30.50 | 54.42 | 51.96 |
| Qwen2.5-7B + NIAT | 60.28 | 61.28 | **62.28** | 72.39 | **64.06** |
| Llama3.1-8B (original) | 49.90 | 62.80 | 26.10 | 54.78 | 48.40 |
| Llama3.1-8B + NIAT | **67.43** | **78.57** | **49.41** | 66.15 | **65.39** |

### Ablation Study

| Fine-tuning Data Type | WTQ | TabFact | HiTab | TABMWP |
|------|:-:|:-:|:-:|:-:|
| Cell-Locating + Cell-Lookup | **67.43** | **78.57** | **49.41** | 66.15 |
| Cell-Locating only | 67.33 | 67.45 | 33.44 | **70.50** |
| Cell-Lookup only | 59.00 | 53.50 | 35.00 | 69.44 |
| Direct fine-tuning on 4 downstream datasets | 64.78 | 61.35 | 53.76 | 67.15 |

### Key Findings

- **Lost-in-the-Middle-Table Phenomenon**: All LLMs (including GPT-4o) perceive rows at the beginning and end of a table significantly better than middle rows, with performance degrading sharply as table size increases.
- **Large Gap between Cell-Locating and Cell-Lookup**: Models perform far better on Cell-Lookup than on Cell-Locating, indicating reliance on semantic co-occurrence rather than genuine understanding of table structure.
- **Attention Pattern Analysis**: LLMs exhibit two attention patterns: Multi-Slash (attending to the same column) and Local-Triangle (attending to row headers).
- **Fine-tuning on only 12K NIAT samples surpasses TableGPT2 trained on millions of samples**: This validates the critical importance of fundamental perception capability.

## Highlights & Insights

- The paper presents a concise yet compelling perspective: using the most fundamental cell localization and lookup tasks to verify whether LLMs' table understanding is "genuine."
- The Lost-in-the-Middle-Table phenomenon is a table-domain extension of Lost-in-the-Middle and offers valuable insight.
- The strong2weak data synthesis strategy is efficient, achieving significant gains with only 12K samples.
- The work exposes potential data leakage and shortcut issues in existing table benchmarks.

## Limitations & Future Work

- The NIAT benchmark currently covers only English tables, lacking multilingual evaluation.
- The Cell-Locating task is relatively simple; more complex structural understanding tasks (e.g., cross-table association) could be designed.
- The data synthesis method relies on GPT-4o, incurring non-trivial cost; cheaper data generation approaches warrant exploration.
- Only zero-shot settings are evaluated; comparisons under few-shot and fine-tuning conditions are insufficient.
- Evaluation on extremely long tables (>120K tokens) remains limited.

## Related Work & Insights

- The Needle-in-a-Haystack family (RULER, InfiniBench, LongBench) focuses on unstructured text; NIAT serves as its structured-table counterpart.
- Table-specialized LLMs such as TableGPT2 and StructLLM improve performance through large-scale table instruction tuning, yet NIAT reveals that their fundamental perception remains inadequate.
- DeepSeek-R1's test-time scaling achieves the best NIAT performance, suggesting that reasoning chains are highly beneficial for structural table understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ — Fresh perspective; probes the "authenticity" of table understanding from the most fundamental cell perception level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers a broad range of open-source and closed-source models across multiple table structures and formats with thorough analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, coherent argumentation.
- Value: ⭐⭐⭐⭐ — Provides an important benchmark contribution and methodological insight for the table understanding community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)
- [\[NeurIPS 2025\] FlySearch: Exploring how vision-language models explore](flysearch_exploring_how_vision-language_models_explore.md)

<!-- RELATED:END -->
