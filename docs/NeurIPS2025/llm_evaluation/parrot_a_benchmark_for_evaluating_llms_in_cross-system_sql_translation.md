---
title: >-
  [Paper Note] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation
description: >-
  [NeurIPS 2025][LLM Evaluation][SQL translation] This paper presents PARROT, a practical and realistic benchmark for cross-system SQL translation (SQL-to-SQL), comprising 598 core translation pairs (expanded to 28…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "SQL translation"
  - "cross-database systems"
  - "SQL dialects"
  - "benchmark"
date: 2026-05-08
content_hash: 2c8a7bcbffdaa7c7
---

# PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation

**Conference**: NeurIPS 2025
**arXiv**: [2509.23338](https://arxiv.org/abs/2509.23338)  
**Code**: [https://github.com/weAIDB/PARROT](https://github.com/weAIDB/PARROT)  
**Area**: LLM Evaluation
**Keywords**: SQL translation, cross-database systems, SQL dialects, LLM evaluation, benchmark

## TL;DR

This paper presents PARROT, a practical and realistic benchmark for cross-system SQL translation (SQL-to-SQL), comprising 598 core translation pairs (expanded to 28,003 pairs) sourced from 38 open-source benchmarks and real-world business scenarios, covering 22 production-grade database systems. The benchmark reveals that the strongest current LLMs achieve an average accuracy below 38.53%.

## Background & Motivation

**Background**: LLMs have demonstrated increasingly strong performance on Text-to-SQL tasks (natural language to SQL), making it a popular research direction for database interaction. However, a closely related and practically significant problem—cross-system SQL translation (SQL-to-SQL), i.e., converting SQL queries written for one database system (e.g., MySQL) into equivalent queries for another (e.g., ClickHouse)—has been severely neglected.

**Limitations of Prior Work**: (1) Existing SQL benchmarks (e.g., Spider, WikiSQL) focus on a very limited number of database systems (typically only SQLite) and fail to reflect the system diversity found in real production environments; (2) These benchmarks use highly standardized SQL syntax, making them unable to capture the extensive system-specific SQL dialect differences (e.g., custom functions, data types, syntactic rules) that constitute the core challenge of SQL-to-SQL translation; (3) The absence of dedicated SQL-to-SQL evaluation benchmarks has left this practically important task (e.g., database migration, multi-database system interoperability) without systematic study.

**Key Challenge**: While Text-to-SQL research flourishes, it remains confined to standardized SQL and a handful of systems. In contrast, real-world cross-system SQL migration is an essential need involving substantial dialect variation—a challenge that existing benchmarks are entirely unable to evaluate.

**Goal**: (1) Construct a SQL-to-SQL translation benchmark covering multiple systems with authentic dialect differences; (2) Systematically evaluate the cross-system SQL translation capabilities of current LLMs; (3) Provide benchmark variants at different granularities to accommodate diverse evaluation needs.

**Key Insight**: Translation pairs are collected from 38 open-source SQL benchmarks and real-world business services, with data specifically designed to challenge system-specific SQL comprehension. Two evaluation metrics are proposed: dialect compatibility (AccEX, syntactic correctness) and result consistency (AccRES, execution result equivalence).

**Core Idea**: Construct a practical SQL-to-SQL translation benchmark covering 22 production-grade database systems to expose critical deficiencies in LLMs' understanding of system-specific SQL dialects.

## Method

### Overall Architecture

The PARROT data construction pipeline proceeds as follows: SQL queries are collected from 38 open-source database benchmarks and real-world business scenarios → cross-system translation pairs are identified → system-specific dialect features (custom functions, data types, syntactic rules, etc.) are annotated → translation equivalence is verified → pairs are organized into core and extended sets. The final output consists of three variants: PARROT Core Set (598 pairs, high difficulty), PARROT-Diverse (28,003 pairs, broad coverage), and PARROT-Simple (5,306 pairs, focused stress testing).

### Key Designs

1. **Comprehensive Coverage of System-Specific SQL Dialects**:

    - Function: Ensures the benchmark challenges LLMs' understanding of the unique syntactic features of each database system.
    - Mechanism: Covers 22 production-grade database systems (including MySQL, PostgreSQL, Oracle, SQL Server, ClickHouse, Hive, Spark SQL, Presto, etc.), with translation pairs designed around each system's distinctive features (e.g., ClickHouse Array functions, advanced window function usage in PostgreSQL, Hive UDFs). Data sources include official documentation test cases, open-source benchmarks, and real-world business SQL.
    - Design Motivation: The difficulty of real-world SQL migration lies precisely in dialect differences—the same query intent may require entirely different functions, type conversions, and syntactic structures across systems.

2. **Multi-Variant Benchmark Design**:

    - Function: Accommodates evaluation needs at different levels of granularity.
    - Mechanism:
        - **PARROT (Core Set)**: 598 carefully selected translation pairs focusing on challenging dialect comprehension, intended for core capability evaluation.
        - **PARROT-Diverse**: 28,003 translation pairs covering a wide range of syntactic variants, designed for comprehensive syntactic capability testing.
        - **PARROT-Simple**: 5,306 representative samples for rapid and focused stress testing.
    - Design Motivation: Different use cases require different levels of granularity—researchers need the depth of the core set, while system developers require the broad coverage of the Diverse variant.

3. **Dual-Dimensional Evaluation Metrics**:

    - Function: Evaluates translation quality from both syntactic and semantic perspectives.
    - Mechanism:
        - **AccEX (Dialect Compatibility)**: Whether the translated SQL executes successfully on the target system (syntactic correctness).
        - **AccRES (Result Consistency)**: Whether the execution result of the translated SQL is equivalent to that of the original SQL (semantic correctness).
    - Design Motivation: A SQL query may be syntactically valid on the target system yet semantically incorrect (e.g., precision loss due to type conversion), necessitating evaluation along both dimensions.

### Evaluation Strategy

Multiple LLMs are evaluated on the core set, spanning different licensing types (open-source/proprietary), parameter scales (7B–671B), and task scopes (general-purpose/code-specialized). A public leaderboard is provided to support ongoing evaluation.

## Key Experimental Results

### Main Results (PARROT Core Set Leaderboard)

**Dialect Compatibility (AccEX) Leaderboard:**

| Rank | Model | Parameters | AccEX |
|------|-------|------------|-------|
| 🏆 | GPT-4o | UNK | 53.32% |
| 🥈 | DeepSeek-V3 | 671B | 50.64% |
| 🥉 | Claude 3.7 Sonnet | UNK | 48.09% |
| 4 | DeepSeek-R1 | 671B | 44.42% |
| 5 | DeepSeek-R1-32B | 32B | 41.98% |
| — | o3-mini | UNK | 27.94% |
| — | DeepSeek-R1-7B | 7B | 17.03% |

**Result Consistency (AccRES) Leaderboard:**

| Rank | Model | AccRES |
|------|-------|--------|
| 🏆 | o3-mini | 54.23% |
| 🥈 | o1-preview | 48.69% |
| 🥉 | DeepSeek-R1 671B | 40.52% |
| 4 | DeepSeek-V3 671B | 32.65% |
| — | GPT-4o | 21.87% |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Overall Difficulty | Average accuracy of all LLMs < 38.53%, far below human DBA level (> 90%) |
| AccEX vs. AccRES Ranking Discrepancy | GPT-4o ranks first in AccEX but achieves only 21.87% in AccRES, indicating syntactic correctness does not imply semantic correctness |
| Divergence Among Reasoning Models | o3-mini achieves the highest AccRES but weaker AccEX, suggesting that reasoning ability aids semantic understanding but is insufficient for dialect mastery |
| Scale Effect | DeepSeek-R1 shows consistent performance improvement from 7B→32B→671B, yet remains far below human-level |
| Core Set vs. Extended Set | The core set, focused on high-difficulty dialect comprehension, is more discriminative of model capability |

### Key Findings

- **LLMs are severely deficient in SQL-to-SQL translation**: The strongest model, GPT-4o, achieves only 53.32% AccEX, while human DBAs assisted by translation tools can exceed 90%.
- **The two evaluation dimensions reveal distinct capabilities**: Models leading in AccEX may perform modestly in AccRES (GPT-4o being the most prominent example), demonstrating that "generating executable SQL" and "generating semantically correct SQL" are fundamentally different abilities.
- **An interesting divergence among reasoning models**: o3-mini achieves the highest AccRES but weaker AccEX, suggesting that chain-of-thought reasoning aids query intent understanding but may lead to the use of constructs unsupported by the target system.
- **Small models struggle severely**: 7B-scale models (e.g., DeepSeek-R1-7B) achieve only 17% AccEX, indicating that SQL dialect knowledge requires substantial parameter capacity.

## Highlights & Insights

- **Fills an important gap**: PARROT is the first benchmark dedicated to cross-system SQL translation, with coverage of 22 database systems unprecedented in database research.
- **Exposes the blind spot behind the Text-to-SQL boom**: The research community has focused on NL-to-SQL while overlooking the equally important SQL-to-SQL problem.
- **The AccEX vs. AccRES separation**: The benchmark introduces a valuable evaluation framework in which syntactic executability and semantic equivalence are treated as orthogonal dimensions.
- **Strong practical relevance**: Database migration (e.g., from MySQL to PostgreSQL) and multi-system interoperability are high-frequency enterprise needs.
- **Continuous evaluation infrastructure**: A public leaderboard and GitHub repository support ongoing community contributions and evaluation.

## Limitations & Future Work

- The full paper content was not fully accessible (arXiv HTML unavailable); some details—such as quality assurance procedures in the data construction pipeline and fine-grained dialect type distribution analysis—may be missing.
- The core set contains only 598 pairs, which may provide insufficient coverage for certain database systems.
- Evaluation focuses solely on accuracy, without analysis of specific error type distributions (e.g., function mapping errors vs. type conversion errors vs. syntactic structure errors).
- The methodology and scale of the human DBA baseline are not described in detail.
- The impact of enhancement strategies such as few-shot prompting or RAG (retrieving system documentation) on model performance remains unexplored.

## Related Work & Insights

- Spider and WikiSQL are classic Text-to-SQL benchmarks but cover only SQLite; PARROT fills the gap in the SQL-to-SQL direction.
- CrackSQL (a follow-up work by the same team) proposes an LLM-based hybrid SQL dialect translation system.
- Differences among SQL dialects represent a long-standing practical challenge in the database systems domain; this paper provides the first systematic LLM-based evaluation of this challenge.
- PARROT complements robustness-oriented benchmarks in the NL2SQL field, such as BIRD and Dr.Spider.

## Rating

⭐⭐⭐⭐ (4/5)

The research topic is well-chosen—SQL-to-SQL is a severely neglected yet practically important direction. The data scale and system coverage are impressive (22 production-grade databases, 28K+ translation pairs), and the dual-dimensional evaluation design is well-motivated. The benchmark reveals a striking gap: the strongest LLMs fall far short of human-level performance in SQL dialect translation. The public leaderboard design is well-suited to sustaining progress in this area.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](../../ACL2026/llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)

</div>

<!-- RELATED:END -->
