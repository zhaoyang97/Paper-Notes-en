---
title: >-
  [Paper Note] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation
description: >-
  [ACL 2026][LLM Evaluation][Literature Review Table Generation] This paper proposes the arXiv2Table benchmark (1,957 tables, 7,158 papers), which achieves more realistic evaluation of LLM-based literature review table gen…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Literature Review Table Generation"
  - "Benchmark Evaluation"
  - "LLM"
  - "Distractor Papers"
  - "QA Evaluation"
date: 2026-05-08
content_hash: ee41f99d408cd5f5
---

# arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation

**Conference**: ACL 2026  
**arXiv**: [2504.10284](https://arxiv.org/abs/2504.10284)  
**Code**: [GitHub](https://github.com/JHU-CLSP/arXiv2Table)  
**Area**: Model Compression  
**Keywords**: Literature Review Table Generation, Benchmark Evaluation, LLM, Distractor Papers, QA Evaluation

## TL;DR

This paper proposes the arXiv2Table benchmark (1,957 tables, 7,158 papers), which achieves more realistic evaluation of LLM-based literature review table generation by introducing distractor papers, schema-agnostic user requirements, and a QA-based reference-free evaluation framework. It also introduces an iterative batch generation method.

## Background & Motivation

**Background**: Literature review tables are essential tools for organizing and comparing papers in scientific research. Recently, LLMs have been utilized to automatically generate these tables. The typical workflow involves generating a table schema (column names) and values (cell contents) given a pre-selected set of papers and a table title.

**Limitations of Prior Work**: (1) Existing methods assume all input papers are relevant, whereas real scenarios contain many semantically related but irrelevant distractor papers; (2) Using ground-truth table titles as generation targets often leads to evaluation bias, as titles are too brief and may leak schema/value information; (3) Evaluation relies on static semantic embeddings or expensive human annotation, failing to capture fine-grained differences.

**Key Challenge**: Existing task definitions and evaluation protocols are overly idealized and disconnected from actual literature review workflows, hindering the practical application of generation methods.

**Goal**: Construct a more realistic benchmark and evaluation framework for literature review table generation that simulates noise and user requirements in real-world scenarios.

**Key Insight**: A three-pronged approach—introducing human-verified distractor papers, rewriting schema-agnostic user requirements, and designing QA-based automatic evaluation.

**Core Idea**: Decompose the literature review table generation task into three sub-tasks: paper retrieval (T1) $\rightarrow$ paper filtering (T2) $\rightarrow$ table induction (T3), introducing realistic noise at each stage.

## Method

### Overall Architecture

Given a user requirement $p$, an IR engine first retrieves a candidate paper set $C$ (including distractor papers). Then, an LLM filters a relevant subset $R \subseteq C$. Finally, a literature review table with $m$ rows and $N$ columns is generated. Evaluation measures the information coverage of the table via LLM-generated QA pairs.

### Key Designs

1.  **Schema-Agnostic User Requirement Construction**:
    *   Function: Replaces original table titles to provide more realistic generation targets.
    *   Mechanism: GPT-4o rewrites table titles into self-contained, goal-oriented user requirements that do not leak schema/values, complemented by automatic leakage checks and human spot checks.
    *   Design Motivation: Original titles are often too brief (e.g., "Performance comparison of different approaches"), cannot be understood independently, and may leak the gold schema.

2.  **Human-Verified Distractor Papers**:
    *   Function: Simulates noise in real-world retrieval.
    *   Mechanism: SentenceBERT retrieves top-10 distractor papers by semantic similarity, which are then verified through binary judgment by an annotation team with CS research experience (IAA = 94%, Fleiss' $\kappa$ = 0.73).
    *   Design Motivation: Actual retrieval precision and recall are low, requiring LLMs to possess the ability to filter distractor papers.

3.  **QA-Based Reference-Free Evaluation Framework**:
    *   Function: Evaluates table quality across three dimensions: schema, cells, and cell relationships.
    *   Mechanism: Synthesizes QA pairs from the ground-truth table (schema-level / unary value / pairwise value). Recall is calculated using answers from the generated table; precision is calculated via the reverse operation.
    *   Design Motivation: Semantic embeddings fail to capture context-specific variations, and human annotation is costly and non-scalable.

### Iterative Batch Generation Method

Input papers are processed in batches. Each batch simultaneously performs paper filtering and schema definition. Through multiple iterations of refinement, this method overcomes context window limitations and schema inconsistency issues associated with single-run processing.

## Key Experimental Results

### Main Results

| Model | Method | Paper F1 | Schema F1 | Unary F1 | Pairwise F1 | Avg |
|-------|--------|----------|-----------|----------|-------------|-----|
| LLaMA-3.3-70B | Newman et al. | 60.9 | 38.3 | 37.8 | 29.8 | 35.3 |
| LLaMA-3.3-70B | Ours | **67.9** | **47.7** | **51.1** | **41.0** | **46.6** |
| Mistral-Large-123B | Newman et al. | - | 33.8 | - | - | - |

### Ablation Study

| Configuration | Change |
|---------------|--------|
| Baseline 1 (One-step generation) | Context window overflow, poor performance |
| Baseline 2 (Paper-by-paper) | Inconsistent schema, sparse tables |
| Newman et al. (Two-stage) | Schema based only on abstracts, misses details |
| + COT Enhancement | Slight improvement for both strong baselines and the proposed method |

### Key Findings

*   The proposed iterative method significantly outperforms all baselines in both paper filtering and table generation.
*   Absolute scores remain low (Avg ~47), highlighting the high difficulty of the task.
*   LLMs exhibit weak performance in retrieving relevant papers from large-scale corpora (verified by pilot study).
*   QA evaluation shows high alignment with expert human judgment (verified by cross-evaluator).

## Highlights & Insights

*   First to systematically introduce distractor papers and schema-agnostic user requirements, bringing literature table generation evaluation closer to real scenarios.
*   The QA-based evaluation framework requires no additional annotation and systematically measures table quality across three dimensions (schema, unary, pairwise).
*   Leakage analysis (Table 1) of user requirements vs. titles is compelling: schema overlap dropped from 5.2% to 0.7% in user requirements.
*   The iterative batch processing method improves paper selection F1 by more than 7 points.

## Limitations & Future Work

*   Only one user requirement was collected per table; diverse requirements remain to be explored.
*   QA evaluation relies on GPT-4o, which may introduce model bias.
*   Exact matching of numerical values in tables was not considered.
*   Computational cost of the iterative method increases with the number of rounds.
*   Future work could extend to cross-domain and multilingual scenarios.

## Related Work & Insights

*   ArxivDigesTables (Newman et al., 2024): Original data source and two-stage baseline.
*   Text-to-table generation: Sequence-to-sequence and QA-based methods.
*   Scientific table datasets: TableBank, SciGen, SciTabQA.
*   The design of distractor papers can inspire other information extraction and literature automation tasks.

## Rating

*   Novelty: ⭐⭐⭐⭐ Systematic introduction of distractor papers and realistic user requirements.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Five LLMs, multiple baselines, human verification.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear task definition and well-argued motivation.
*   Value: ⭐⭐⭐⭐ Practical significance for automating literature reviews.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)

</div>

<!-- RELATED:END -->
