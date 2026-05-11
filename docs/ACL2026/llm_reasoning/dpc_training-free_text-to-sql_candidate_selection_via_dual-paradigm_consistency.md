---
title: >-
  [Paper Note] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency
description: >-
  [ACL 2026][LLM Reasoning][SQL selection] DPC reframes Text-to-SQL candidate selection from "guessing over hidden data" to "deterministic verification over constructed data": it builds a Minimal Discriminative Database (M…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "SQL selection"
  - "dual-paradigm consistency"
  - "minimal discriminative database"
  - "training-free"
  - "adversarial environment synthesis"
date: 2026-05-08
content_hash: e026d0079e70043f
---

# DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency

**Conference**: ACL 2026
**arXiv**: [2604.15163](https://arxiv.org/abs/2604.15163)
**Code**: [GitHub](https://github.com/HKUSTDial/DPC)
**Area**: LLM Reasoning
**Keywords**: SQL selection, dual-paradigm consistency, minimal discriminative database, training-free, adversarial environment synthesis

## TL;DR
DPC reframes Text-to-SQL candidate selection from "guessing over hidden data" to "deterministic verification over constructed data": it builds a Minimal Discriminative Database (MDD) that forces conflicting SQL candidates to produce different execution results, then uses Python/Pandas solutions as reference anchors to select the correct candidate via cross-paradigm consistency, outperforming Self-Consistency by up to 2.2% on BIRD and Spider.

## Background & Motivation

**State of the Field**: Text-to-SQL has adopted a generate-then-select paradigm—generating $K$ candidate SQL queries and selecting the best one. However, a significant "generation–selection gap" exists: Pass@K far exceeds Pass@1 (e.g., 58.8% vs. ~50% on BIRD), indicating that correct SQL candidates are already present among the $K$ outputs but fail to be selected.

**Limitations of Prior Work**: (1) Self-Consistency (majority voting) fails under systematic model bias—the model consistently converges to a wrong answer; (2) LLM-as-Judge suffers from "symbolic blindness," being unable to mentally simulate the execution state of complex SQL; (3) trained verifiers require expensive annotation and exhibit poor cross-domain generalization.

**Root Cause**: SQL verification faces a triple challenge—partial observability (real databases are too large to fit in context), symbolic blindness (LLMs cannot internally simulate SQL execution), and inherent confirmation bias (models favor candidates they themselves generated).

**Paper Goals**: Design a training-free SQL selection framework that transforms verification from probabilistic guessing into deterministic judgment.

**Starting Point**: Construct a carefully designed small database in which conflicting SQL candidates are guaranteed to produce different results, and use Python code as an independent reasoning path for cross-validation.

**Core Idea**: Adversarial environment synthesis (MDD) + dual-paradigm execution (SQL vs. Python) + consistency voting.

## Method

### Overall Architecture
DPC operates as a four-stage pipeline: (1) **Candidate clustering and pairing**: cluster candidates by execution results, selecting a champion (largest cluster) and a challenger (second-largest); (2) **Adversarial environment synthesis**: a Slicer agent trims the schema, and a Tester agent generates adversarial data to populate the MDD such that champion and challenger produce different results; (3) **Dual-paradigm execution**: execute SQL on the MDD and have a Solver agent generate and execute a Python/Pandas script; (4) **Consistency verification**: use the BS-F1 metric to compare SQL results with Python results for semantic equivalence.

### Key Designs

1. **Minimal Discriminative Database (MDD) Construction**

    - **Function**: Transforms SQL verification from partially observable to fully observable.
    - **Mechanism**: Constructed in two steps. A Slicer Agent iteratively prunes the schema to include only the tables and columns required by the candidate SQL queries, verifying structural integrity via dry-run execution. A Tester Agent then adversarially generates data through a discriminative feedback loop, ensuring that the champion and challenger yield different execution results on the MDD—e.g., distinguishing `INNER JOIN` from `LEFT JOIN` requires records with unmatched keys.
    - **Design Motivation**: Random data sampling is insufficient to differentiate SQL queries that are semantically similar but logically distinct; adversarial data must be constructed to specifically target the differences between candidates.

2. **Dual-Paradigm Consistency Verification**

    - **Function**: Breaks LLM confirmation bias through an independent reasoning path.
    - **Mechanism**: Exploits the "capability gap" whereby LLMs perform better on imperative languages (Python) than on declarative languages (SQL)—Python enjoys broader coverage in pretraining corpora, and imperative code forces the model to explicitly plan data manipulation steps. A Solver Agent generates a Python/Pandas solution on the MDD, whose execution result serves as a proxy ground truth to validate SQL candidates.
    - **Design Motivation**: Having the same LLM solve the same problem in two different paradigms—if both paradigms agree, the answer is very likely correct; if they disagree, the Python solution (the more reliable paradigm) serves as the arbiter.

3. **Bipartite Soft F1 Metric (BS-F1)**

    - **Function**: Robustly quantifies semantic equivalence between SQL and Python execution results.
    - **Mechanism**: Addresses two key challenges in cross-paradigm comparison: type incompatibility (SQL `DECIMAL` vs. Python `float`, SQL `NULL` vs. Python `NaN`) and ordering ambiguity (SQL results without `ORDER BY` are unordered sets). Types are first normalized, then the Hungarian algorithm is applied for optimal row-level matching, and column overlap among matched rows is computed to yield Soft-F1.
    - **Design Motivation**: Standard execution accuracy (EX) requires strict equality, which produces large numbers of false negatives in cross-paradigm comparison scenarios.

### Loss & Training
DPC is a purely inference-time framework with no training involved. All agents are implemented via prompt engineering using a shared LLM backbone.

## Key Experimental Results

### Main Results

| Dataset | Method | Execution Accuracy | vs. Self-Consistency |
|---------|--------|--------------------|----------------------|
| BIRD | DPC | Best | +2.2% |
| Spider | DPC | Best | +1–2% |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| w/o MDD (use sampled data directly) | Significant drop | Adversarial data construction is critical |
| w/o Python paradigm | Drop | Single-paradigm verification is inferior to dual-paradigm |
| w/o BS-F1 (use strict EX) | Drop | Cross-paradigm comparison requires soft matching |

### Key Findings
- The discriminative feedback loop in MDD construction is central—random data sampling fails to distinguish conflicting SQL in the majority of cases.
- The Python paradigm as a verification anchor is more reliable than SQL itself, corroborating the hypothesis that LLMs exhibit stronger capability on imperative languages.
- BS-F1 significantly reduces false negatives compared to strict EX in cross-paradigm verification.
- DPC consistently outperforms Self-Consistency across multiple LLM backbones.

## Highlights & Insights
- The idea of **recasting a selection problem as a constructive verification problem** is highly elegant—rather than guessing which SQL is correct, the method constructs an "experiment" that reveals the difference.
- **Cross-paradigm consistency** exploits the capability gap between LLMs in different programming languages—Python serves as a "second opinion" to cross-validate SQL, analogous to independent replication in science.
- The adversarial construction paradigm underlying MDD generalizes to any selection task requiring discrimination among similar candidates.

## Limitations & Future Work
- MDD construction requires multiple rounds of LLM calls, increasing inference latency and cost.
- The framework focuses solely on champion-vs.-challenger binary selection, potentially overlooking correct candidates ranked lower.
- The quality of Python solutions depends on the LLM's Python programming ability, which is not always reliable.
- For highly complex SQL (e.g., deeply nested subqueries), the success rate of MDD construction may degrade.

## Related Work & Insights
- **vs. Self-Consistency**: SC relies on majority voting and fails under systematic bias; DPC replaces probabilistic voting with deterministic execution evidence.
- **vs. LLM-as-Judge**: The judge paradigm is constrained by symbolic blindness and cannot simulate execution; DPC obtains deterministic evidence through actual execution.
- **vs. Trained Verifiers**: Training requires annotation and suffers from domain brittleness; DPC is entirely training-free.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of adversarial environment synthesis and dual-paradigm consistency is a genuinely novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two standard benchmarks (BIRD + Spider) with multiple LLM backbones.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formalization is clear; pipeline description is logically coherent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](../../NeurIPS2025/llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](../../NeurIPS2025/llm_reasoning/sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[AAAI 2026\] A Reasoning Paradigm for Named Entity Recognition](../../AAAI2026/llm_reasoning/a_reasoning_paradigm_for_named_entity_recognition.md)

</div>

<!-- RELATED:END -->
