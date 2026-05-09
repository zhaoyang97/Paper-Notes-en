---
title: >-
  [Paper Note] AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation
description: >-
  [NeurIPS 2025][Text-to-SQL] This paper proposes AgentiQL, a multi-expert agent framework for Text-to-SQL: a reasoning agent decomposes questions into sub-problems, a coding agent generates sub-queries, a refinement step corrects column selection, and an adaptive router intelligently routes between a baseline parser and the modular pipeline. Using a 14B open-source model, AgentiQL achieves 86.07% EX on Spider, approaching GPT-4 SOTA (89.65%).
tags:
  - NeurIPS 2025
  - Text-to-SQL
  - multi-expert
  - question decomposition
  - adaptive routing
  - Spider benchmark
date: 2026-05-08
content_hash: badb8bec49810700
---

# AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.10661](https://arxiv.org/abs/2510.10661)
**Code**: To be released
**Area**: Interpretability
**Keywords**: Text-to-SQL, multi-expert, question decomposition, adaptive routing, Spider benchmark

## TL;DR
This paper proposes AgentiQL, a multi-expert agent framework for Text-to-SQL: a reasoning agent decomposes questions into sub-problems, a coding agent generates sub-queries, a refinement step corrects column selection, and an adaptive router intelligently routes between a baseline parser and the modular pipeline. Using a 14B open-source model, AgentiQL achieves 86.07% EX on Spider, approaching GPT-4 SOTA (89.65%).

## Background & Motivation

**Background**: LLMs have significantly advanced NL2SQL capabilities, yet monolithic LLM architectures still struggle with complex reasoning and diverse schema handling. Static ensemble methods incur high computational overhead, and deploying large-scale LLMs remains costly in practice.

**Limitations of Prior Work**: (1) Monolithic LLMs perform poorly on complex multi-table joins and nested aggregations; (2) existing systems lack interpretability—users cannot trace how SQL is generated; (3) serving large models (e.g., GPT-4) is costly and impractical for real-world deployment.

**Key Challenge**: Large models deliver strong performance but at high cost, while small models are economical but insufficient for complex queries.

**Goal**: Achieve large-model-level NL2SQL performance using a small model (14B parameters) through multi-expert collaboration, while maintaining interpretability and efficiency.

**Key Insight**: Decompose SQL generation into three specialized components—reasoning (question decomposition), coding (sub-query generation), and refinement (column selection correction)—and use an adaptive router to select the execution path based on query complexity.

**Core Idea**: Apply Divide-and-Merge to decompose complex SQL generation into sub-problem–sub-query pairs, combined with Column Selection refinement and Adaptive Routing, enabling a 14B open-source model to approach GPT-4-level performance.

## Method

### Overall Architecture
Query input → Adaptive Router assesses complexity → simple queries follow the baseline for direct generation / complex queries follow the AgentiQL pipeline (Table Selection → Question Decomposition → Sub-Query Generation → Merge → Column Selection).

### Key Designs

1. **Divide**:

    - **Function**: Decompose complex natural language queries into manageable sub-problems.
    - **Mechanism**: Three steps—(1) Table Selection: a reasoning LLM filters irrelevant tables to produce a reduced schema $\tilde{s}$; (2) Question Decomposition: a reasoning LLM decomposes the query into sub-problems $\{x_1, ..., x_k\}$; (3) Query Generation: a coding LLM generates a SQL sub-query for each sub-problem, with up to $R=3$ retries on failure.
    - **Design Motivation**: Complex queries require multi-step reasoning; decomposition simplifies each step and exposes intermediate reasoning, improving interpretability.

2. **Merge**:

    - **Function**: Combine sub-queries into a final complete SQL statement.
    - **Mechanism**: Two strategies—(1) Last Sub-query: takes the final sub-query as the complete answer (assuming the reasoning agent orders sub-problems such that the last corresponds to the full solution); (2) Planner & Executor: a reasoning LLM acts as planner to decide how to merge, while a coding LLM acts as executor to implement the merge. The latter is more general but computationally costlier.
    - **Design Motivation**: Last Sub-query is effective in many cases where sub-problems are naturally progressive, while Planner & Executor handles complex scenarios requiring genuine merging of multiple sub-queries.

3. **Column Selection**:

    - **Function**: Ensure that the columns and their ordering in the SELECT clause match user intent.
    - **Mechanism**: A reasoning LLM inspects the SELECT clause of the merged SQL and adjusts column selection and ordering to precisely match query requirements.
    - **Design Motivation**: The merge process may introduce extraneous columns or incorrect ordering; this refinement step corrects such issues and consistently yields 2–5% EX improvement.

4. **Adaptive Routing**:

    - **Function**: Route queries to either baseline direct generation or the AgentiQL pipeline based on query complexity.
    - **Mechanism**: An XGBoost classifier or a reasoning agent acting as judge can be used, leveraging simple signals such as schema size (number of tables) as a complexity proxy.
    - **Design Motivation**: Simple queries are handled more efficiently and accurately by the baseline; the full pipeline is activated only for complex queries, balancing efficiency and accuracy.

### Loss & Training
No training is required (inference-time method). The Qwen2.5-Coder series (7B/14B/32B) is used as the coding LLM with few-shot prompting.

## Key Experimental Results

### Main Results (Spider Benchmark)

| Method | 7B EX | 14B EX | 32B EX |
|--------|-------|--------|--------|
| Baseline (Qwen2.5-Coder) | 80.69 | 82.10 | 84.57 |
| Last Sub-query + CS | 74.44 | 80.21 | 83.16 |
| Planner & Executor + CS | **75.85** | **83.40** | **86.07** |
| GPT-4 SOTA (CHASE-SQL) | - | - | **89.65** |

### Ablation Study

| Configuration | 7B EX | 14B EX |
|---------------|-------|--------|
| Last Sub-query w/o CS | 72.26 | 78.38 |
| Last Sub-query + CS | 74.44 (+2.18) | 80.21 (+1.83) |
| Planner & Executor w/o CS | 66.77 | 79.39 |
| Planner & Executor + CS | 75.85 (+9.08) | 83.40 (+4.01) |

### Key Findings
- **Column Selection contributes consistently**: improvements of 2–9% EX across all configurations, and is especially critical for the Planner & Executor strategy (correcting column errors introduced by the planner).
- **14B + Planner & Executor + CS reaches 86.07%**: closing the gap with GPT-4 SOTA (89.65%) to 3.5% using an open-source small model.
- **Routing improves robustness**: the baseline performs better on simple queries while AgentiQL excels on complex ones; routing combines the strengths of both.
- **Interpretability is enhanced**: intermediate sub-problems and sub-queries expose the reasoning process.

## Highlights & Insights
- **The Column Selection refinement step**, though seemingly simple, yields substantial gains (up to +9%), suggesting that the last mile of NL2SQL lies in SELECT clause alignment—an aspect that is frequently overlooked.
- **The Adaptive Routing design** is highly practical: not all queries require a complex pipeline; allocating compute on demand is an effective strategy.
- **Extensible to parallel execution**: sub-query generation can be parallelized to improve throughput.

## Limitations & Future Work
- **Evaluation limited to Spider**: validation on additional benchmarks such as BIRD and SQL-Eval is needed.
- **Large model overhead remains prohibitive**: a 235B model requires approximately 60 minutes per query on 4 A100 GPUs.
- **Decomposition failure is the primary failure mode**: if the reasoning agent decomposes a question incorrectly, all subsequent steps fail.
- **Suggested direction**: Incorporate RL to optimize query generation quality (e.g., SkyRL-SQL).

## Related Work & Insights
- **vs. DIN-SQL**: DIN-SQL also performs question decomposition, but AgentiQL additionally introduces Column Selection and Adaptive Routing.
- **vs. CHASE-SQL**: CHASE-SQL achieves SOTA using GPT-4; AgentiQL approaches its performance with a 14B open-source model.
- **vs. CodeS**: CodeS employs fine-tuning, whereas AgentiQL is a pure inference-time method requiring no training.

## Rating
- Novelty: ⭐⭐⭐ The multi-expert decomposition idea is not novel per se, but the Column Selection + Routing combination is effective
- Experimental Thoroughness: ⭐⭐⭐ Limited to Spider, but ablations are thorough
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured
- Value: ⭐⭐⭐⭐ A practical solution for approaching GPT-4 SOTA with small open-source models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](../../ACL2026/interpretability/pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)

</div>

<!-- RELATED:END -->
