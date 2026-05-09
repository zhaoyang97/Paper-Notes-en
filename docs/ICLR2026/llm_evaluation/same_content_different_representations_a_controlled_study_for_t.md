---
title: >-
  [Paper Note] Same Content, Different Representations: A Controlled Study for Table QA
description: >-
  [ICLR 2026][LLM Evaluation][Table QA] The first controlled study that systematically evaluates the robustness of NL2SQL, LLM, and hybrid approaches under varying table size, schema quality, and query complexity by changing only the representation format (structured vs. semi-structured) while holding table content constant, demonstrating that representation format is a first-order factor in Table QA performance.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Table QA
  - structured tables
  - semi-structured tables
  - representation format
  - diagnostic benchmark
date: 2026-05-08
content_hash: 335d02fd1c9d2f7d
---

# Same Content, Different Representations: A Controlled Study for Table QA

**Conference**: ICLR 2026
**arXiv**: [2509.22983](https://arxiv.org/abs/2509.22983)
**Code**: [https://github.com/megagonlabs/RePairTQA](https://github.com/megagonlabs/RePairTQA)
**Area**: LLM Evaluation
**Keywords**: Table QA, structured tables, semi-structured tables, representation format, diagnostic benchmark

## TL;DR
The first controlled study that systematically evaluates the robustness of NL2SQL, LLM, and hybrid approaches under varying table size, schema quality, and query complexity by changing only the representation format (structured vs. semi-structured) while holding table content constant, demonstrating that representation format is a first-order factor in Table QA performance.

## Background & Motivation

**Background**: Table QA methods fall into three main paradigms: NL2SQL (translating natural language into SQL for execution), direct LLM reasoning, and hybrid methods (SQL retrieval combined with LLM reasoning). Existing benchmarks fix the table format, and models are optimized for a single representation.

**Limitations of Prior Work**: In practice, tables appear in both strictly-schematized structured form and semi-structured form with irregular columns and free-text cells. However, no existing benchmark systematically studies the effect of representation format itself on model performance, leaving cross-format generalization unknown.

**Key Challenge**: A fair comparison of representation formats requires that table content remain identical while only the format varies. Existing datasets cannot satisfy this requirement, as structured and semi-structured benchmarks differ in their underlying data.

**Goal**: (a) How to generate paired structured/semi-structured tables with controlled content? (b) How do table size, join operations, query complexity, and schema quality each affect different paradigms? (c) How should practitioners select the best method for deployment?

**Key Insight**: A verbalization pipeline converts columns in structured tables into natural language descriptions, producing semantically equivalent but structurally distinct paired tables.

**Core Idea**: Representation format is a first-order variable in Table QA — NL2SQL is strongest on structured input but drops 30–45% on semi-structured input; LLMs are most robust but less accurate; hybrid methods perform best in semi-structured settings.

## Method

### Overall Architecture
Given a table and a natural language question, the system outputs an answer. Rather than proposing a new QA model, this work constructs the diagnostic benchmark **RePairTQA**: a verbalization pipeline generates semantically equivalent semi-structured versions of existing structured tables, and four diagnostic dimensions (table size, join operations, query complexity, schema quality) partition the data into diagnostic splits for systematic evaluation of three method families.

### Key Designs

1. **Verbalization Pipeline**:

    - Function: Converts structured tables into semi-structured tables while preserving semantic content.
    - Mechanism: A three-step process — (1) **Column selection**: GPT-4o selects columns suitable for verbalization, with random combination sampling to increase diversity; (2) **Template construction**: natural language templates are generated for each selected column combination, with 5 distinct templates per group; (3) **Serialization**: templates are instantiated with actual cell values, merged into a free-text column, and the original structured columns are removed.
    - Design Motivation: Semantic preservation is a prerequisite for fair comparison. By changing only form while preserving content, any performance difference can be attributed solely to representation format.

2. **Diagnostic Splits**:

    - Function: Partitions the benchmark into 7 subsets (S1–S5, M1–M2) along four dimensions.
    - Mechanism: Constructed from three complementary datasets — BIRD (clean schema), MMQA (multi-table reasoning), and TableEval (noisy schema). Each subset holds all other variables fixed while varying one: S1 vs. S4 for table size, S1 vs. S2 for schema quality, S1 vs. S3 for query complexity, and S1–S5 vs. M1–M2 for join operations.
    - Design Motivation: Isolating each factor individually avoids confounding effects.

3. **LLM-as-Judge Evaluation Protocol**:

    - Function: Replaces conventional Exact Match evaluation with GPT-4o-based judgment.
    - Mechanism: GPT-4o compares model predictions against gold answers to assess semantic correctness, tolerating surface-form variation (e.g., different numeric formats, synonymous expressions).
    - Design Motivation: Traditional EM/PM metrics are overly strict for paraphrase-equivalent answers. Human annotation on 100 examples confirms 96% agreement.

### Loss & Training
This work does not train models but evaluates existing methods, including:
- **LLM**: GPT-4o, Gemini-2.5-flash, Qwen3-235B (direct reasoning)
- **NL2SQL**: LLM-NL2SQL (two-stage pipeline), XiYan (multi-generator ensemble)
- **Hybrid**: H-STAR (SQL+LLM routing), Weaver (step-by-step workflow)

## Key Experimental Results

### Main Results (RQ1: Structured vs. Semi-Structured Overall Comparison)

| Model | Structured Acc. (%) | Semi-Structured Acc. (%) | Drop (%) |
|------|---------------|-----------------|---------|
| GPT-4o | 45.37 | 41.93 | 3.44 |
| Gemini-2.5-flash | 52.07 | 50.78 | 1.29 |
| LLM-NL2SQL | 69.14 | 38.65 | **30.49** |
| XiYan | 69.55 | 24.08 | **45.47** |
| H-STAR | 49.48 | 47.14 | 2.34 |
| Weaver | 62.19 | 57.70 | 4.49 |

### Ablation Study (Analysis by Dimension)

| Factor | Key Finding | Most Affected Method |
|------|---------|---------------|
| Table size | Longer tables degrade all methods; LLMs are most sensitive | GPT-4o: 70% → 28.9% |
| Join operations | NL2SQL benefits from structured multi-table input (+10%) but collapses in semi-structured settings | LLM-NL2SQL: 82.3% on structured multi-table |
| Query complexity | LLMs achieve ~70% on lookup; large drops on compositional reasoning | All methods affected |
| Schema quality | Noisy schemas severely impact NL2SQL; hybrid methods are most robust | XiYan: large drop from structured to semi-structured |

### Key Findings
- **Representation format is a first-order factor**: NL2SQL methods drop 30–45% on semi-structured input, making this the most fragile paradigm.
- **No universal best method**: NL2SQL is preferred for structured data, hybrid methods for semi-structured data, and LLMs for simple queries.
- **Verbalization can benefit LLMs**: Natural language descriptions in semi-structured tables more closely resemble LLM pre-training data; hybrid methods sometimes achieve higher accuracy on semi-structured lookup tasks.
- **Long tables are a universal bottleneck**: All methods degrade significantly on long tables, though NL2SQL maintains 62.9% on long structured tables.
- **Model scale does not resolve representation sensitivity**: Gemini-2.5-Pro exhibits the same structured vs. semi-structured performance gap.

## Highlights & Insights
- The **controlled experimental design** is particularly elegant: holding information content constant while varying only representation format ensures that all performance differences are attributable to format alone. This methodology is transferable to comparative studies in other modalities (e.g., knowledge graphs vs. documents).
- The **decision-tree method selection guide** is highly practical, recommending optimal methods based on data conditions (structured/semi-structured × table size × schema quality × query complexity) to directly inform deployment decisions.
- The finding that verbalization sometimes benefits LLM reasoning challenges the intuition that structured formats are always superior.

## Limitations & Future Work
- Only three benchmark datasets are covered, limiting domain diversity (e.g., finance and biomedicine are absent).
- Table size is constrained to what fits within the context window; extremely long tables requiring chunking or retrieval are not addressed.
- Verbalization templates are generated by GPT-4o, potentially introducing model bias.
- Emerging methods (e.g., RAG-based table QA systems) are not evaluated.
- Hybrid method support for multi-table scenarios is limited; future work should design improved cross-table reasoning modules.

## Related Work & Insights
- **vs. BIRD/Spider**: These benchmarks fix the structured format; RePairTQA introduces a semi-structured dimension via verbalization, constituting the first controlled-variable benchmark of its kind.
- **vs. H-STAR/Weaver**: This work does not propose new methods but systematically evaluates the representational robustness of existing approaches.
- **Implications for system design**: Practitioners should adaptively select reasoning paradigms based on data conditions rather than applying a single universal approach.

## Rating
- Novelty: ⭐⭐⭐⭐ First controlled study of table representation, with a novel experimental design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 diagnostic splits, 7 models, and 5 research questions analyzed systematically
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures and tables, practical decision-tree summary
- Value: ⭐⭐⭐⭐ Provides important guidance for Table QA method selection

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/llm_evaluation/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](../../ACL2026/llm_evaluation/odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](../../NeurIPS2025/llm_evaluation/beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](../../ACL2026/llm_evaluation/capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)

<!-- RELATED:END -->
