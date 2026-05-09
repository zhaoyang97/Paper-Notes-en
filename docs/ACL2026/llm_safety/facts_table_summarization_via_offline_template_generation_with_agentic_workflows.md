---
title: >-
  [Paper Note] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows
description: >-
  [ACL 2026][LLM Safety][Table Summarization] This paper proposes FACTS (Fast, Accurate, and Privacy-Compliant Table Summarization), a three-stage agentic workflow that automatically generates reusable offline templates (SQL queries + Jinja2 templates) for fast, accurate, and privacy-compliant query-focused table summarization, achieving state-of-the-art performance across FeTaQA, QTSumm, and QFMTS benchmarks.
tags:
  - ACL 2026
  - LLM Safety
  - Table Summarization
  - Offline Templates
  - Agentic Workflow
  - SQL Generation
  - Privacy Compliance
date: 2026-05-08
content_hash: fab5c30cc169ae73
---

# FACTS: Table Summarization via Offline Template Generation with Agentic Workflows

**Conference**: ACL 2026
**arXiv**: [2510.13920](https://arxiv.org/abs/2510.13920)
**Code**: [GitHub](https://github.com/BorealisAI/FACTS)
**Area**: Data Analysis / Table Understanding
**Keywords**: Table Summarization, Offline Templates, Agentic Workflow, SQL Generation, Privacy Compliance

## TL;DR

This paper proposes FACTS (Fast, Accurate, and Privacy-Compliant Table Summarization), a three-stage agentic workflow that automatically generates reusable offline templates (SQL queries + Jinja2 templates) for fast, accurate, and privacy-compliant query-focused table summarization, achieving state-of-the-art performance across FeTaQA, QTSumm, and QFMTS benchmarks.

## Background & Motivation

**Background**: Query-focused table summarization requires generating natural language summaries from tabular data conditioned on user queries, distinct from simple table QA (which returns short answers) and generic table summarization (which captures all salient content). In domains such as finance, healthcare, and law, professionals rely on customized summaries for decision-making.

**Limitations of Prior Work**: (1) Table-to-text models (e.g., TAPEX, ReasTAP) require costly fine-tuning and underperform on numerical reasoning and logical faithfulness; (2) prompt-based methods (e.g., DirectSumm) query LLMs directly, suffer from token limitations, expose sensitive data, and must regenerate outputs for every new table; (3) existing agentic frameworks (e.g., Binder, Dater) rely on decomposed planning or hand-crafted templates, lacking robustness and scalability.

**Key Challenge**: A practical solution must simultaneously satisfy four properties—fast (reusable), accurate (execution-based rather than free-form), scalable (no need to pass all rows), and privacy-compliant (no raw data exposed to the LLM)—yet no existing method satisfies all four.

**Goal**: Design the first automated agentic framework for offline template generation that is generated once, reused many times, and satisfies all four properties.

**Key Insight**: Table summarization is decomposed into SQL queries (for precise value extraction) and Jinja2 templates (for natural language rendering), forming offline templates that are independent of specific data values.

**Core Idea**: Offline templates are bound to the table schema and query semantics rather than concrete data values. Once generated, they can be directly applied to any new table sharing the same schema, eliminating repeated LLM inference.

## Method

### Overall Architecture

FACTS consists of three interconnected stages, each iteratively validated and refined by an LLM Council (a multi-model ensemble). The final output is an offline template comprising a set of SQL queries and a Jinja2 rendering template. The LLM only ever receives schema information and never has access to raw data.

### Key Designs

1. **Schema-Guided Specification and Filtering**:

    - **Function**: Clarifies user query intent and generates filtering rules.
    - **Mechanism**: Given the user query and table schema, the agent produces two types of output: (a) guided questions—identifying which columns, relationships, and operations are relevant; and (b) filtering rules—specifying rows or categorical values to exclude. The LLM never accesses raw data; it formulates abstract filtering rules based solely on the schema (e.g., "exclude rows where category='expense'"), which are subsequently converted into SQL WHERE clauses.
    - **Design Motivation**: User queries are typically high-level natural language expressions that must first be "translated" into concrete schema-level operational specifications.

2. **SQL Queries Generation**:

    - **Function**: Generates executable SQL queries for precise data extraction.
    - **Mechanism**: Based on the Stage 1 specification, the agent generates candidate SQL queries that encode filtering rules as constraints. Each query is executed against a local database for validation—if execution fails or returns empty results, the error message is passed to the LLM Council for feedback, and the agent iteratively refines the query until it executes successfully. The maximum patience is set to 3 rounds.
    - **Design Motivation**: Grounding summarization in executable programs rather than free-form text generation fundamentally eliminates hallucinations.

3. **Jinja2 Template Generation and Alignment**:

    - **Function**: Renders SQL results into natural language summaries.
    - **Mechanism**: The agent generates Jinja2 templates that reference exact column names, correctly iterate over returned rows, and handle empty results gracefully. The LLM Council checks the alignment between SQL output and template references—if field mismatches or shape incompatibilities are detected, the SQL and template are co-revised.
    - **Design Motivation**: Decoupling data extraction (SQL) from text rendering (Jinja2) enables each component to be independently verified and reused.

### Loss & Training

FACTS is a training-free method. The primary agent uses GPT-4o-mini as its backbone. The LLM Council consists of GPT-4o-mini, Claude-4 Sonnet, and DeepSeek v3, with majority voting to accept or reject outputs and aggregated feedback to guide refinement. On average, each sample requires 2.47 guided questions/filtering rules, 1.36 rounds of SQL revision, and 1.84 rounds of template revision.

## Key Experimental Results

### Main Results

| Method | FeTaQA BLEU/RL/MET | QTSumm BLEU/RL/MET | QFMTS BLEU/RL/MET |
|--------|---------------------|---------------------|---------------------|
| CoT | 28.2/51.0/56.9 | 19.3/39.0/47.2 | 31.5/54.3/58.1 |
| DirectSumm | 29.8/51.7/58.2 | 20.7/40.2/50.3 | 33.6/57.0/62.8 |
| SPaGe | 33.8/55.7/62.3 | 20.9/41.3/47.7 | 45.7/68.3/73.4 |
| FACTS (GPT-Only) | 30.8/55.7/66.0 | 20.1/43.1/50.5 | 45.4/70.5/73.2 |
| **FACTS** | **32.6/58.9/67.7** | **21.9/45.8/51.3** | **46.0/70.8/73.2** |

### Ablation Study

| Evaluation Dimension | FACTS Score |
|----------------------|-------------|
| Intent Matching | 97% |
| SQL Execution Accuracy | 94% |
| Template Rendering Accuracy | 98% |
| Council Consensus Error Rate | ~3% |
| Overall Factual Accuracy | ~92% |

### Key Findings

- FACTS achieves the best or second-best results on all three datasets, with particularly notable advantages on ROUGE-L and METEOR.
- Human preference study: FACTS vs. SPaGe—55% prefer FACTS for completeness, 59% for correctness, and 60% for reduced hallucinations.
- Reusability test: With 100 tables sharing the same schema, FACTS achieves substantial speedup via template reuse (requiring only SQL execution + Jinja2 rendering).
- The GPT-Only variant still outperforms most baselines, validating the core workflow design; Council diversity provides further gains.
- Each sample consumes an average of 9,922 input tokens and 1,045 output tokens, keeping computational cost manageable.

## Highlights & Insights

- The concept of "offline templates" represents an elegant engineering innovation—amortizing one-time LLM inference costs over unlimited reuse, making it particularly suited for enterprise scenarios such as recurring annual financial report summarization.
- The LLM Council's majority voting and aggregated feedback mechanism provides a lightweight self-correction capability; the ~3% consensus error rate demonstrates the effectiveness of multi-model ensembling.
- Privacy-compliant design is a core advantage of this approach—the LLM only accesses the schema, while raw data values remain entirely within the local SQL engine.
- The SQL + Jinja2 combination decouples "correctness" from "readability"—the former is guaranteed by program execution, the latter achieved through template rendering.

## Limitations & Future Work

- The method assumes complete template reusability under the same schema, without accounting for schema drift or column renaming.
- Complex multi-table JOINs and nested queries may require additional revision rounds.
- A SQL execution accuracy of 94% implies a 6% error rate, which may be insufficient for high-stakes decision-making.
- Natural language expressions in Jinja2 templates may require adaptation across different languages and cultural contexts.

## Related Work & Insights

- **vs. DirectSumm**: DirectSumm passes the entire table and query to the LLM in a single call, exposing data and precluding reuse; FACTS addresses both issues through offline templates.
- **vs. SPaGe**: SPaGe employs graph-structured planning to improve reliability, but its plans are only partially reusable; FACTS offline templates are fully reusable.
- **vs. Binder/Dater**: These methods translate queries into executable programs but lack templatization and reuse capabilities; FACTS adds a Jinja2 rendering layer to produce natural language output.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The offline template generation concept is novel and practical, though individual components (SQL generation, Jinja2, LLM Council) have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, automatic and human evaluation, reusability/scalability analysis, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem definition is clear, the four-property comparison table is intuitive, and examples are concrete.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical—the privacy-compliant and reusable design directly addresses enterprise deployment pain points.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](../../NeurIPS2025/llm_safety/trap_targeted_redirecting_of_agentic_preferences.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](../../NeurIPS2025/llm_safety/llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)

</div>

<!-- RELATED:END -->
