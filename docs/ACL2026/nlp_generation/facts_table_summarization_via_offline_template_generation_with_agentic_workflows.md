---
title: >-
  [Paper Note] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows
description: >-
  [ACL 2026][Text Generation][Table Summarization] This paper proposes FACTS (Fast, Accurate, and Privacy-Compliant Table Summarization)…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Table Summarization"
  - "Offline Templates"
  - "Agentic Workflows"
  - "SQL Generation"
  - "Privacy Compliance"
date: 2026-05-08
content_hash: 969e39976b7f710f
---

# FACTS: Table Summarization via Offline Template Generation with Agentic Workflows

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.13920](https://arxiv.org/abs/2510.13920)  
**Code**: [GitHub](https://github.com/BorealisAI/FACTS)  
**Area**: Data Analysis / Table Understanding  
**Keywords**: Table Summarization, Offline Templates, Agentic Workflows, SQL Generation, Privacy Compliance

## TL;DR

This paper proposes FACTS (Fast, Accurate, and Privacy-Compliant Table Summarization), which leverages a three-stage Agentic workflow to automatically generate reusable offline templates (SQL queries + Jinja2 templates). This approach enables fast, accurate, and privacy-compliant query-focused table summarization, consistently outperforming baselines across FeTaQA, QTSumm, and QFMTS benchmarks.

## Background & Motivation

**Background**: Query-focused table summarization requires generating natural language summaries from tabular data based on specific user queries. This task differs from simple table QA (which returns short answers) and general table summarization (which captures all salient information). Professionals in fields such as finance, healthcare, and law rely on customized summaries for decision-making.

**Limitations of Prior Work**: (1) Table-to-text models (e.g., TAPEX, ReasTAP) require expensive fine-tuning and struggle with numerical reasoning and logical faithfulness; (2) Prompt-based methods (e.g., DirectSumm) query LLMs directly, which are constrained by token limits, risk exposing sensitive data, and require regeneration for every new table; (3) Existing Agentic frameworks (e.g., Binder, Dater) rely on decomposition planning or manual templates, lacking robustness and scalability.

**Key Challenge**: A practical solution must simultaneously satisfy four properties—Fast (reusable), Accurate (execution-based rather than free generation), Scalable (no need to pass all rows), and Privacy-compliant (raw data not exposed to the LLM)—yet no existing method fulfills all four.

**Goal**: To design the first Agentic framework for automated offline template generation that achieves "generate once, reuse many times" while satisfying all four properties.

**Key Insight**: Decompose table summarization into SQL queries (for precise data extraction) + Jinja2 templates (for natural language rendering), forming offline templates that are independent of specific data values.

**Core Idea**: Offline templates are bound to the table schema and query semantics rather than specific data values. Once generated, they can be directly applied to any new table sharing the same schema, avoiding redundant LLM inference.

## Method

### Overall Architecture

FACTS consists of three interconnected stages, where the output of each stage is iteratively verified and improved by an LLM Council (multi-model ensemble). The final output is an offline template consisting of a set of SQL queries and a Jinja2 rendering template. Throughout the process, the LLM only accesses schema information and never sees raw data.

### Key Designs

1.  **Schema-Guided Specification and Filtering**:
    - **Function**: Clarifies user query intent and generates filtering rules.
    - **Mechanism**: Given a user query and table schema, the Agent generates two types of outputs: (a) guided questions—identifying relevant columns, relationships, and operations; (b) filtering rules—specifying rows or category values to be excluded. The LLM does not touch raw data but proposes abstract filtering rules based on the schema (e.g., "exclude rows where category='expense'"), which are subsequently converted into SQL `WHERE` clauses.
    - **Design Motivation**: User queries are typically high-level natural language and need to be "translated" into specific operational specifications at the schema level.

2.  **SQL Queries Generation**:
    - **Function**: Generates executable SQL queries to extract data precisely.
    - **Mechanism**: Based on the specifications from Stage 1, the Agent generates candidate SQL queries, translating filtering rules into constraints. Each query is verified against a local database; if it fails or returns empty results, the error message is passed to the LLM Council for feedback, and the Agent iterates until the query is executable. The maximum patience is set to 3 rounds.
    - **Design Motivation**: Grounding the summary on executable programs rather than free-text generation fundamentally eliminates hallucinations.

3.  **Jinja2 Template Generation and Alignment**:
    - **Function**: Renders SQL results into natural language summaries.
    - **Mechanism**: The Agent generates a Jinja2 template, requiring it to reference exact column names, correctly iterate over returned rows, and handle empty results gracefully. The LLM Council checks the alignment between the SQL output and the template references—if field mismatches or shape inconsistencies exist, the SQL and template are corrected collaboratively.
    - **Design Motivation**: Decoupling data extraction (SQL) from text rendering (Jinja2) allows both components to be independently verified and reused.

### Loss & Training

FACTS is a training-free method. The primary Agent uses `GPT-4o-mini` as the backbone. The LLM Council consists of `GPT-4o-mini`, `Claude-4 Sonnet`, and `DeepSeek v3`, deciding on acceptance/rejection via majority voting and providing aggregated feedback for guidance. On average, each sample involves $2.47$ guided questions/filtering rules, $1.36$ rounds of SQL refinement, and $1.84$ rounds of template refinement.

## Key Experimental Results

### Main Results

| Method | FeTaQA BLEU/RL/MET | QTSumm BLEU/RL/MET | QFMTS BLEU/RL/MET |
| :--- | :--- | :--- | :--- |
| CoT | 28.2 / 51.0 / 56.9 | 19.3 / 39.0 / 47.2 | 31.5 / 54.3 / 58.1 |
| DirectSumm | 29.8 / 51.7 / 58.2 | 20.7 / 40.2 / 50.3 | 33.6 / 57.0 / 62.8 |
| SPaGe | 33.8 / 55.7 / 62.3 | 20.9 / 41.3 / 47.7 | 45.7 / 68.3 / 73.4 |
| FACTS (GPT-Only) | 30.8 / 55.7 / 66.0 | 20.1 / 43.1 / 50.5 | 45.4 / 70.5 / 73.2 |
| **FACTS** | **32.6 / 58.9 / 67.7** | **21.9 / 45.8 / 51.3** | **46.0 / 70.8 / 73.2** |

### Ablation Study

| Evaluation Dimension | FACTS Score |
| :--- | :--- |
| Intent Matching | $97\%$ |
| SQL Execution Accuracy | $94\%$ |
| Template Rendering Accuracy | $98\%$ |
| Council Consensus Error Rate | $\sim 3\%$ |
| Overall Factual Correctness | $\sim 92\%$ |

### Key Findings

- FACTS achieves state-of-the-art or second-best results across all three datasets, with significant advantages in ROUGE-L and METEOR.
- Human Preference Study: Comparing FACTS vs. SPaGe, $55\%$ preferred FACTS for completeness, $59\%$ for correctness, and $60\%$ for reduced hallucinations.
- Reusability Testing: When processing 100 tables with the same schema, FACTS accelerates significantly due to template reuse (requiring only SQL execution + Jinja2 rendering).
- The GPT-Only variant still outperforms most baselines, proving the effectiveness of the core workflow, while Council diversity provides further enhancement.
- Each sample consumes an average of $9,922$ input tokens and $1,045$ output tokens, maintaining controllable computational costs.

## Highlights & Insights

- The "Offline Template" concept is an elegant engineering innovation—it amortizes one-time LLM inference costs over infinite reuses, making it ideal for enterprise scenarios (e.g., recurring annual financial report summaries).
- The majority voting and aggregated feedback mechanism of the LLM Council provides a lightweight self-correction capability. The $\sim 3\%$ consensus error rate indicates the effectiveness of the multi-model ensemble.
- Privacy compliance is a core advantage—the LLM only interacts with the schema, while raw data values remain entirely within the local SQL engine.
- The combination of SQL and Jinja2 decouples "correctness" from "readability"—the former is guaranteed by programmatic execution, while the latter is achieved through template rendering.

## Limitations & Future Work

- It assumes templates are fully reusable under the same schema and does not account for schema drift or column renaming.
- Complex multi-table JOINs and nested queries may require more refinement rounds.
- A $94\%$ SQL execution accuracy implies a residual $6\%$ error rate, which may be insufficient for high-risk decision-making.
- The natural language expression of Jinja2 templates might require adjustment across different languages or cultural contexts.

## Related Work & Insights

- **vs. DirectSumm**: The latter passes the entire table and query to the LLM at once, exposing data and preventing reuse; FACTS addresses both issues through offline templates.
- **vs. SPaGe**: SPaGe uses graph-structured planning to improve reliability, but its plans are only partially reusable; FACTS offline templates are fully reusable.
- **vs. Binder/Dater**: These methods convert queries into executable programs but lack templating and reuse capabilities; FACTS adds a Jinja2 rendering layer for natural language output.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of offline template generation is novel and practical, though individual components (SQL generation, Jinja2, LLM Council) have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three benchmarks, including automated and human metrics, reusability/scalability analysis, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, intuitive comparison tables for the four properties, and concrete examples.
- Value: ⭐⭐⭐⭐⭐ Highly practical—the privacy-compliant and reusable design directly addresses pain points in enterprise deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)
- [\[ACL 2026\] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)
- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)

</div>

<!-- RELATED:END -->
