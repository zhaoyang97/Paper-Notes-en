---
title: >-
  [Paper Note] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction
description: >-
  [NeurIPS 2025][LLM Reasoning][text-to-SQL] This paper proposes SQL-of-Thought, a multi-agent Text-to-SQL framework that decomposes the task into schema linking → subproblem identification → CoT query plan generation → SQ…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "text-to-SQL"
  - "multi-agent"
  - "error taxonomy"
  - "chain-of-thought"
  - "Spider benchmark"
date: 2026-05-08
content_hash: e18136110611b40f
---

# SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction

**Conference**: NeurIPS 2025
**arXiv**: [2509.00581](https://arxiv.org/abs/2509.00581)  
**Code**: None  
**Area**: LLM Reasoning / NLP
**Keywords**: text-to-SQL, multi-agent, error taxonomy, chain-of-thought, Spider benchmark

## TL;DR
This paper proposes SQL-of-Thought, a multi-agent Text-to-SQL framework that decomposes the task into schema linking → subproblem identification → CoT query plan generation → SQL generation → guided correction loop based on a 31-category error taxonomy. Using Claude 3 Opus on the Spider benchmark, it achieves 91.59% execution accuracy, outperforming the previous best Chase SQL (87.6%) by nearly 4 percentage points.

## Background & Motivation

**Background**: Text-to-SQL has evolved from sequence-to-sequence models to LLM prompting methods (DIN-SQL, DAIL-SQL), with multi-agent approaches (MAC-SQL, Chase SQL) further improving modularity and accuracy.

**Limitations of Prior Work**: (a) Existing error correction relies solely on execution feedback — 95–99% of generated SQL is syntactically correct, yet logical errors (e.g., wrong JOIN types, missing aggregations) cannot be detected via execution signals; (b) unguided reasoning may introduce new errors; (c) there is no systematic error taxonomy to guide correction.

**Key Challenge**: Syntactic correctness ≠ semantic correctness — structured error diagnosis beyond execution feedback is required.

**Key Insight**: Design a 31-category error taxonomy, combined with CoT reasoning in the correction loop to precisely locate and repair logical errors.

## Method

### Overall Architecture
Five specialized agents execute sequentially with a guided correction loop: Schema Linking → Subproblem → Query Plan (CoT) → SQL Generation → execution test → [on failure] Correction Plan (CoT + error taxonomy) → Correction SQL → re-execution.

### Key Designs

1. **Staged Reasoning**:

    - Schema Linking Agent: identifies relevant tables, columns, and primary/foreign keys.
    - Subproblem Agent: decomposes the query into clause-level subproblems (WHERE/GROUP BY/JOIN, etc.) and outputs structured JSON.
    - Query Plan Agent: generates a step-by-step execution plan via CoT (SQL generation is prohibited at this stage).
    - SQL Agent: generates executable SQL based on the plan.

2. **31-Category Error Taxonomy**:

    - Covers 9 major categories: syntax errors, schema linking errors, JOIN errors, filter condition errors, aggregation logic errors, value representation errors, subquery errors, set operation errors, and structural omissions.
    - Uses concise error codes (rather than verbose descriptions) to conserve context window space.
    - The Correction Plan Agent is prompted to reference the taxonomy, diagnosing the error type before formulating a repair strategy.

3. **Guided Correction Loop**:

    - Unlike DIN-SQL (regeneration only) and DAIL-SQL (execution feedback only), the correction loop provides error type + CoT repair plan.
    - A two-step correction process (Correction Plan Agent → Correction SQL Agent) is used instead of direct correction.
    - The loop continues until execution succeeds or the maximum number of attempts is reached.

## Key Experimental Results

### Spider Benchmark

| Method | Spider EA | Spider-Realistic EA |
|--------|----------|-------------------|
| DIN-SQL + GPT-4 | 82.8% | 78.1% |
| DAIL-SQL + GPT-4 + SC | 83.6% | 75.2% |
| MAC-SQL + GPT-4 | 86.8% | - |
| Tool-SQL + GPT-4 | 86.9% | 82.9% |
| Chase SQL | 87.6% | - |
| **SQL-of-Thought + Claude 3 Opus** | **91.59%** | **90.16%** |

### Ablation Study (100 samples)

| Configuration | Accuracy | Note |
|---------------|----------|------|
| SQL-of-Thought (full) | 95% | Complete framework |
| w/o correction loop | 85% | −10%; correction is critical |
| w/o Query Plan | 90% | −5%; CoT planning is beneficial |

### Model Comparison

| Model | SQL-of-Thought EA |
|-------|-----------------|
| Claude 3 Opus | 95% |
| GPT-5 | 89% |
| GPT-4o-mini | 87% |
| GPT-3.5 | 67% |
| Llama-3.1-8B | ~45% |

### Key Findings
- **Correction loop contributes +10%**: SQL that is syntactically correct but logically erroneous requires structured diagnosis to repair.
- **CoT Query Plan contributes +5%**: Planning before generating SQL is more reliable than direct generation.
- **Claude 3 Opus performs best**: It demonstrates the strongest reasoning capability and SQL generation accuracy among all evaluated models.
- **Cost trade-off**: A single Spider run costs ~$42; a hybrid model strategy can reduce this to ~$30 (85% EA).

## Highlights & Insights
- **Value of the error taxonomy**: The structured 31-category taxonomy transforms LLM behavior from "blind retry" to "targeted repair" — removing the taxonomy exacerbates repeated corrections of identical errors.
- **Two-step correction > direct correction**: Generating a correction plan before generating SQL is more effective than directly feeding error information to the SQL Agent — LLMs benefit from structured intermediate reasoning steps.
- **Lessons from failed ablations**: Assigning multiple repair agents to each fix a different error type and then merging results leads to conflicts; carrying correction history causes context bloat and performance degradation.
- **Large gap for open-source models**: Llama-3.1-8B achieves only 45%, exposing the limitations of smaller models on complex structured generation tasks.

## Limitations & Future Work
- **Evaluation limited to the Spider series**: Spider does not reflect the complexity of real-world databases (the TAG framework indicates it covers only ~20% of real queries).
- **High API cost**: The multi-agent framework costs ~$42 per run, limiting practical deployment.
- **Error taxonomy coverage unverified**: Coverage across diverse query structures remains unknown.
- **Future directions**: (1) Evaluation on BIRD-SQL and real-world databases; (2) fine-tuning smaller models to replace API calls and reduce cost; (3) automatic learning and updating of the error taxonomy.

## Related Work & Insights
- **vs. DIN-SQL**: DIN-SQL's correction merely regenerates the prompt without specific error information; SQL-of-Thought provides taxonomy-guided, targeted correction.
- **vs. Chase SQL**: Chase SQL employs a multi-candidate selection strategy, whereas SQL-of-Thought uses a single-path iterative correction approach — making it more efficient.
- **vs. Think2SQL**: Think2SQL finds that reasoning offers mixed benefits for SQL generation; SQL-of-Thought eliminates the "unguided reasoning can be harmful" problem through staged reasoning (plan → SQL) and taxonomy-guided correction.
- **Insight**: For structured output generation (SQL/code), a "diagnose → plan → repair" paradigm is more effective than "detect → retry."

## Rating
- Novelty: ⭐⭐⭐⭐ Taxonomy-guided correction is a novel and practical design.
- Experimental Thoroughness: ⭐⭐⭐ Limited to the Spider series; lacks evaluation on BIRD-SQL and real-world databases.
- Writing Quality: ⭐⭐⭐⭐ Architecture is clearly presented; ablation analysis is thorough, including lessons from failed ablations.
- Value: ⭐⭐⭐⭐ Spider SOTA results are convincing, but generalization to harder benchmarks requires further validation.

## Rating
- Novelty: ⭐⭐⭐ Combination of multi-agent framework and error taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ Standard benchmark evaluation.
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐ Practical reference value for text-to-SQL applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[NeurIPS 2025\] SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)

</div>

<!-- RELATED:END -->
