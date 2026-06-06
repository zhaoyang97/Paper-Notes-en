---
title: >-
  [Paper Note] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency
description: >-
  [ACL 2026][Code Intelligence][SQL Selection] DPC transforms Text-to-SQL candidate selection from "guessing on hidden data" into "deterministic verification on observable data": it constructs a Minimum Distinguishing Data…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "SQL Selection"
  - "Dual-Paradigm Consistency"
  - "Minimum Distinguishing Database"
  - "Training-free"
  - "Adversarial Environment Synthesis"
date: 2026-05-08
content_hash: f0c540c413b464bd
---

# DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency

**Conference**: ACL 2026  
**arXiv**: [2604.15163](https://arxiv.org/abs/2604.15163)  
**Code**: [GitHub](https://github.com/HKUSTDial/DPC)  
**Area**: LLM Reasoning  
**Keywords**: SQL Selection, Dual-Paradigm Consistency, Minimum Distinguishing Database, Training-free, Adversarial Environment Synthesis

## TL;DR
DPC transforms Text-to-SQL candidate selection from "guessing on hidden data" into "deterministic verification on observable data": it constructs a Minimum Distinguishing Database (MDD) to force conflicting SQLs to yield different results, then employs Python/Pandas solutions as reference anchors to select the correct candidate through cross-paradigm consistency, outperforming Self-Consistency by up to 2.2% on BIRD and Spider.

## Background & Motivation

**Background**: Text-to-SQL adopts a "generate-then-select" paradigm—generating $K$ candidate SQLs and selecting the best one. However, a significant "generation-selection gap" exists: Pass@K is much higher than Pass@1 (e.g., 58.8% vs. ~50% on BIRD), indicating the correct SQL is often present but not selected.

**Limitations of Prior Work**: (1) Self-Consistency (majority voting) fails when models have systematic biases—converging consistently on an incorrect answer; (2) LLM-as-Judge suffers from "symbolic blindness," unable to mentally simulate the execution state of complex SQL; (3) Trained verifiers require expensive labeling and demonstrate poor domain robustness.

**Key Challenge**: The triple challenge of SQL verification—partial observability (real databases are too large for context), symbolic blindness (LLM cannot internally simulate SQL execution), and inherent confirmation bias (models favor their own generated candidates).

**Goal**: Design a training-free SQL selection framework to transform verification from probabilistic guessing into deterministic judgment.

**Key Insight**: Construct a meticulously designed small database that forces conflicting SQLs to produce different outcomes, then use Python code as an independent reasoning path for cross-verification.

**Core Idea**: Adversarial Environment Synthesis (MDD) + Dual-Paradigm Execution (SQL vs. Python) + Consistency Voting.

## Method

### Overall Architecture
The DPC pipeline consists of four stages: (1) Candidate Clustering & Pairing: Cluster by execution results, then select the champion (largest cluster) and challenger (second largest); (2) Adversarial Environment Synthesis: Slicer Agent prunes the schema and Tester Agent generates adversarial data to populate the MDD, ensuring champion and challenger yield different results; (3) Dual-Paradigm Execution: Execute SQL on the MDD while the Solver Agent generates and executes Python/Pandas scripts; (4) Consistency Verification: Use the BS-F1 metric to compare the semantic equivalence of SQL results against Python references.

### Key Designs

1. **Minimum Distinguishing Database (MDD) Construction**:
    - **Function**: Transforms SQL verification from partially observable to fully observable.
    - **Mechanism**: Constructed in two steps. The Slicer Agent iteratively prunes the schema to include only tables and columns required by candidate SQLs, verifying structural integrity via dry runs. The Tester Agent adversarially generates data through a discriminative feedback loop, ensuring the champion and challenger produce different execution results on the MDD. For instance, distinguishing `INNER JOIN` from `LEFT JOIN` requires specific records with no matching keys.
    - **Design Motivation**: Random data sampling is insufficient to differentiate between SQLs that are semantically similar but logically distinct—adversarial construction targeting specific differences between candidates is required.

2. **Dual-Paradigm Consistency Verification**:
    - **Function**: Breaks the confirmation bias of LLMs through independent reasoning paths.
    - **Mechanism**: Exploits the "capability gap" where LLMs perform better in imperative languages (Python) than declarative languages (SQL)—Python has broader coverage in pre-training corpora, and imperative code forces explicit planning of data manipulation steps. The Solver Agent generates Python/Pandas solutions on the MDD, and their execution results serve as proxy ground truth to verify SQL candidates.
    - **Design Motivation**: Solving the same problem using two different paradigms with the same LLM—if results align, the answer is likely correct; if they diverge, Python (the more reliable paradigm) acts as an arbiter.

3. **Bipartite Soft F1 Metric (BS-F1)**:
    - **Function**: Robustly quantifies semantic equivalence between SQL and Python results.
    - **Mechanism**: Addresses two major cross-paradigm comparison challenges: type incompatibility (SQL `DECIMAL` vs. Python `float`, SQL `NULL` vs. Python `NaN`) and sorting ambiguity (SQL results without `ORDER BY` are unordered sets). It normalizes types and uses the Hungarian algorithm for optimal row-level matching to calculate column overlap for a Soft-F1 score.
    - **Design Motivation**: Standard Execution Accuracy (EX) requires strict equality, leading to many false negatives in cross-paradigm scenarios.

### Loss & Training
DPC is a pure inference-time framework and involves no training. All agents are implemented via prompt engineering using the same LLM backbone.

## Key Experimental Results

### Main Results

| Dataset | Method | Execution Accuracy | vs. Self-Consistency |
|---------|--------|---------------------|----------------------|
| BIRD    | DPC    | State-of-the-Art    | +2.2%                |
| Spider  | DPC    | State-of-the-Art    | +1-2%                |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| W/O MDD (using sample data) | Significant Drop | Adversarial data construction is critical |
| W/O Python Paradigm | Drop | Single-paradigm verification is inferior to dual-paradigm |
| W/O BS-F1 (using strict EX) | Drop | Cross-paradigm comparison requires soft matching |

### Key Findings
- The discriminative feedback loop of MDD is core—random data sampling fails to distinguish conflicting SQLs in most cases.
- The Python paradigm is more reliable as a verification anchor than SQL itself, confirming the hypothesis that LLMs possess stronger capabilities in imperative languages.
- BS-F1 significantly reduces misjudgments compared to strict EX in cross-paradigm verification.
- DPC consistently outperforms Self-Consistency across multiple LLM backbones.

## Highlights & Insights
- The approach of **transforming a selection problem into a constructive verification problem** is elegant—rather than guessing which SQL is correct, it constructs an "experiment" to reveal differences.
- **Cross-Paradigm Consistency** leverages the performance variance of LLMs across different programming languages—using Python as a "second opinion" for SQL is analogous to independent replication in science.
- The adversarial construction logic of MDD can be generalized to any selection task necessitating the differentiation of similar candidates.

## Limitations & Future Work
- MDD construction requires multiple LLM calls, increasing inference latency and cost.
- Focusing only on the champion-challenger pair may miss correct candidates ranked lower.
- The quality of the Python solution depends on the LLM's Python programming ability, which is not always flawless.
- Success rates of MDD construction may decrease for highly complex SQL (e.g., multi-level nested subqueries).

## Related Work & Insights
- **vs. Self-Consistency**: SC relies on majority voting and fails under systematic bias; DPC replaces probabilistic voting with deterministic execution evidence.
- **vs. LLM-as-Judge**: Judge modes are limited by symbolic blindness and cannot simulate execution; DPC obtains definitive evidence through actual execution.
- **vs. Trained Verifiers**: Training requires annotations and is domain-brittle; DPC is entirely training-free.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of adversarial environment synthesis and dual-paradigm consistency is a novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on BIRD and Spider benchmarks with multiple LLM backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formalization and smooth pipeline descriptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)
- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

</div>

<!-- RELATED:END -->
