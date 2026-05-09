---
title: >-
  [Paper Note] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation
description: >-
  [ACL 2026][Model Compression][literature-review table generation] This paper proposes the arXiv2Table benchmark (1,957 tables, 7,158 papers) and introduces distractor papers, schema-agnostic user requests, and an annotation-free QA-based evaluation framework to enable more realistic assessment of LLM-based literature-review table generation, along with an iterative batch generation method.
tags:
  - ACL 2026
  - Model Compression
  - literature-review table generation
  - benchmark evaluation
  - LLM
  - distractor papers
  - QA-based evaluation
date: 2026-05-08
content_hash: fdf2be51cfc6ade9
---

# arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation

**Conference**: ACL 2026
**arXiv**: [2504.10284](https://arxiv.org/abs/2504.10284)
**Code**: [GitHub](https://github.com/JHU-CLSP/arXiv2Table)
**Area**: Model Compression
**Keywords**: literature-review table generation, benchmark evaluation, LLM, distractor papers, QA-based evaluation

## TL;DR

This paper proposes the arXiv2Table benchmark (1,957 tables, 7,158 papers) and introduces distractor papers, schema-agnostic user requests, and an annotation-free QA-based evaluation framework to enable more realistic assessment of LLM-based literature-review table generation, along with an iterative batch generation method.

## Background & Motivation

**Background**: Literature-review tables are essential tools for organizing and comparing papers in scientific research. Recent work has applied LLMs to automatically generate such tables, typically following a pipeline where, given a pre-selected paper set and a table title, an LLM generates the table schema (column names) and values (cell content).

**Limitations of Prior Work**: (1) Existing methods assume all input papers are relevant, whereas in practice many semantically related but off-topic distractor papers exist; (2) Ground-truth table titles are used as generation targets, but titles are often too brief and may leak schema/value information, introducing evaluation bias; (3) Evaluation relies on static semantic embeddings or costly human annotation, failing to capture fine-grained differences.

**Key Challenge**: Existing task formulations and evaluation protocols are overly idealized and substantially disconnected from real-world literature review workflows, hindering practical deployment of generation methods.

**Goal**: To construct a more realistic benchmark and evaluation framework for literature-review table generation that simulates the noise and user demands present in real-world scenarios.

**Key Insight**: A three-pronged approach — introducing human-verified distractor papers, rewriting schema-agnostic user requests, and designing an automatic QA-based evaluation.

**Core Idea**: Decompose the literature-review table generation task into three sub-tasks — paper retrieval (T1) → paper filtering (T2) → table synthesis (T3) — and introduce realistic noise at each stage.

## Method

### Overall Architecture

Given a user request $p$, an IR engine first retrieves a candidate paper set $C$ (containing distractor papers); an LLM then filters a relevant subset $R \subseteq C$; finally, an $m$-row $N$-column literature-review table is generated. Evaluation is conducted by measuring the information coverage of the generated table via LLM-synthesized QA pairs.

### Key Designs

1. **Schema-Agnostic User Request Construction**:

    - Function: Replaces the original table title to provide a more realistic generation target.
    - Mechanism: GPT-4o rewrites table titles into self-contained, goal-oriented user requests that do not leak schema or value information, accompanied by automatic leakage checks and human spot-checks.
    - Design Motivation: Original titles are too brief (e.g., "Performance comparison of different approaches") to be understood independently and may inadvertently reveal the gold schema.

2. **Human-Verified Distractor Papers**:

    - Function: Simulates noise present in real retrieval scenarios.
    - Mechanism: SentenceBERT retrieves the top-10 semantically similar distractor papers; annotators with computer science research experience perform binary relevance judgments (IAA = 94%, Fleiss' $\kappa$ = 0.73).
    - Design Motivation: Real retrieval systems exhibit low precision and recall, requiring LLMs to possess the capability to filter out distractor papers.

3. **Annotation-Free QA-Based Evaluation Framework**:

    - Function: Evaluates table quality across three dimensions: schema, cell values, and cell-value relations.
    - Mechanism: QA pairs (schema-level / unary value / pairwise value) are synthesized from ground-truth tables; recall is computed by answering with the generated table, and the process is reversed to compute precision.
    - Design Motivation: Semantic embeddings cannot capture context-specific variations, and human annotation is costly and not scalable.

### Iterative Batch Generation Method

Input papers are processed in batches; within each batch, paper filtering and schema definition are performed simultaneously. Multiple rounds of iteration progressively refine the output, overcoming the limitations of single-pass processing with respect to context window constraints and schema inconsistency.

## Key Experimental Results

### Main Results

| Model | Method | Paper F1 | Schema F1 | Unary F1 | Pairwise F1 | Avg |
|-------|--------|----------|-----------|----------|-------------|-----|
| LLaMA-3.3-70B | Newman et al. | 60.9 | 38.3 | 37.8 | 29.8 | 35.3 |
| LLaMA-3.3-70B | Ours | **67.9** | **47.7** | **51.1** | **41.0** | **46.6** |
| Mistral-Large-123B | Newman et al. | - | 33.8 | - | - | - |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Baseline 1 (single-step generation) | Context window overflow; poor performance |
| Baseline 2 (paper-by-paper processing) | Schema inconsistency; sparse tables |
| Newman et al. (two-stage) | Schema derived from abstracts only; misses details |
| + CoT augmentation | Marginal improvement for strong baselines and the proposed method |

### Key Findings

- The proposed iterative method significantly outperforms all baselines in both paper filtering and table generation.
- Absolute scores remain low (Avg ~47), underscoring the high difficulty of the task.
- LLMs exhibit weak capability in retrieving relevant papers from large corpora (validated by a pilot study).
- QA-based evaluation shows high agreement with human expert judgments (validated via cross-evaluator analysis).

## Highlights & Insights

- This is the first work to introduce distractor papers and schema-agnostic user requests, bringing literature-review table generation evaluation closer to real-world scenarios.
- The QA-based evaluation framework requires no additional annotation and systematically measures table quality across three dimensions (schema, unary, pairwise).
- The leakage analysis comparing user requests vs. table titles (Table 1) is compelling: schema overlap decreases from 5.2% to 0.7% with user requests.
- The iterative batch generation method achieves a 7+ point improvement in paper selection F1.

## Limitations & Future Work

- Only one user request is collected per table; exploring diverse requests remains future work.
- QA evaluation relies on GPT-4o, which may introduce model bias.
- Exact numerical matching within table cells is not considered.
- The computational cost of the iterative method increases with the number of rounds.
- Future work may extend the benchmark to cross-domain and multilingual settings.

## Related Work & Insights

- ArxivDigesTables (Newman et al., 2024): The original data source and two-stage baseline.
- Text-to-table generation: sequence-to-sequence and QA-based approaches.
- Scientific table datasets: TableBank, SciGen, SciTabQA.
- The distractor paper design proposed in this work can inspire other information extraction and automated survey tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic introduction of distractor papers and realistic user requests.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five LLMs, multiple baselines, human validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear task formulation and well-motivated arguments.
- Value: ⭐⭐⭐⭐ Provides practical guidance for literature review automation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](../../ICLR2026/model_compression/beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[NeurIPS 2025\] Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users](../../NeurIPS2025/model_compression/offline_policy_evaluation_of_multi-turn_llm_health_coaching_with_real_users.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)

<!-- RELATED:END -->
