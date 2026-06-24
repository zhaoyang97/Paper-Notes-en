---
title: >-
  [Paper Note] Enhancing Retrieval Systems with Inference-Time Logical Reasoning
description: >-
  [ACL 2025][Reasoning][Logical Reasoning] Proposes an Inference-Time Logical Reasoning (ITLR) framework, which leverages LLMs to decompose natural language queries into logical expressions (AND/OR/NOT), and then composes the cosine similarity scores of individual sub-terms based on fuzzy logic. It consistently outperforms traditional dense retrieval and the BRIGHT baseline on synthetic data and three real-world datasets (NFCorpus/SciFact/ArguAna)…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Logical Reasoning"
  - "Inference-Time Reasoning"
  - "Fuzzy Logic"
  - "Compositional Queries"
  - "Dense Retrieval"
  - "Negation Handling"
date: 2026-05-08
content_hash: fcd1099ffc53d3bf
---

# Enhancing Retrieval Systems with Inference-Time Logical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2503.17860](https://arxiv.org/abs/2503.17860)  
**Code**: Undisclosed  
**Authors**: Felix Faltings, Wei Wei, Yujia Bao  
**Affiliations**: MIT, Accenture Center for Advanced AI  
**Area**: LLM Inference / Information Retrieval  
**Keywords**: Logical Reasoning, Inference-Time Reasoning, Fuzzy Logic, Compositional Queries, Dense Retrieval, Negation Handling

## TL;DR

Proposes an Inference-Time Logical Reasoning (ITLR) framework, which leverages LLMs to decompose natural language queries into logical expressions (AND/OR/NOT), and then composes the cosine similarity scores of individual sub-terms based on fuzzy logic. It consistently outperforms traditional dense retrieval and the BRIGHT baseline on synthetic data and three real-world datasets (NFCorpus/SciFact/ArguAna), showing significant improvements particularly on complex queries containing negations.

## Background & Motivation

**Background**: Modern retrieval systems map user queries into a vector space and retrieve documents using cosine similarity. Although efficient and scalable, they perform poorly when handling logical constructs (negations, conjunctions, disjunctions).

**Core Problem**:
   - Traditional dense retrieval cannot effectively represent negation semantics—for example, when querying "Vitamin D Benefits, but not including bone health," the system still tends to retrieve documents related to bone health.
   - As the number of logical terms in a query increases, the performance of traditional methods degrades significantly.
   - Existing inference-enhanced retrieval works (such as BRIGHT) are mainly used for knowledge question answering, and inference-time reasoning for general retrieval systems has not been fully explored.

**Design Motivation**: To explicitly introduce inference-time logical reasoning into the retrieval workflow, decomposing queries into logical expressions and composing sub-term scores, thus allowing retrieval systems to handle complex logical queries while maintaining computational efficiency.

## Method

### Overall Architecture

The ITLR method consists of three key steps:

1. **Logical Query Transformation**: Utilizes LLMs (Llama3-70b) to parse natural language queries into logical forms, such as `"Vitamin D Benefits" AND NOT "Bone Health"`.
2. **Term Embedding and Similarity Computation**: Separately encodes each sub-term in the logical expression into vectors, and computes their respective cosine similarity with the documents.
3. **Score Composition Based on Logical Relations**: Composes sub-term scores based on logical relations to obtain the final document ranking score.

### Query Syntax

Defines a concise query syntax supporting three operators: AND, OR, NOT, as well as bracket precedence:

- Precedence rules: NOT > AND > OR
- Strictly defined using a Context-Free Grammar (CFG):
```text
T → U OR U | U; U → V AND V | V; V → NOT W | W; W → string | (T)
```
- Each query is parsed into a syntax tree, then directly converted into a score operation tree.

### Score Operations

Defines various candidate operations for AND, OR, NOT respectively:

| Operator | Candidate Implementations |
|--------|----------|
| OP_AND(x,y) | $x \times y$, $x+y$, $\min(x,y)$ |
| OP_OR(x,y) | $x+y$, $\max(x,y)$ |
| OP_NOT(x) | $1-x$, $1/x$ |

**Default Choice** (Experimental verification shows this is optimal):
- $\text{OP\_AND}(x,y) = x \times y$
- $\text{OP\_OR}(x,y) = x + y$  
- $\text{OP\_NOT}(x) = 1 - x$

**Key Designs**:
- The commonly used $\text{AND}=\min$, $\text{OR}=\max$ compositions in traditional fuzzy logic perform poorly in this scenario (nDCG@10 of only 0.86 vs 0.97 for the default).
- Multiplication operations can better penalize documents with low scores on any sub-terms, which aligns with the AND semantics.

### Computational Efficiency

- Embedding computations for each sub-term can be executed in parallel.
- Introduces only minimal extra overhead compared to the baseline (vector encoding scaled by the number of sub-terms).
- Operator compositions are simple numerical operations, which do not affect retrieval latency.

## Experiments

### Experimental Setup

- **Embedding Models**: NV-Embed-V1, nv-embedqa-mistral-7b-v2, text-embedding-v3-large, text-embedding-v3-small
- **LLMs**: Llama3-70b (used for query transformation and data generation)
- **Evaluation Metrics**: nDCG@10
- **Compared Methods**: Baseline (Direct Query), BRIGHT (Inference-then-Query), ITLR (Ours)

### Synthetic Data Experiments

**Three-term Query Experiments**: Tested on 32 possible three-term logical query templates, with 100 queries per template:

| Number of Negations | Base | ITLR |
|----------|------|------|
| 0 | 0.95 | 0.99 |
| 1 | 0.77 | 0.97 |
| 2 | 0.65 | 0.96 |
| 3 | 0.52 | **1.00** |

**Key Findings**: The more negations there are, the larger the relative improvement of ITLR. With 3 negations, the baseline achieves only 0.52 (close to random at 0.7), while ITLR reaches 1.00.

**Scaling to More Terms**: As the number of query terms increases (from 2 upwards), the performance gap between ITLR and the baseline continually widens, with a more significant gap observed on AND queries.

### Ablation on Operator Combinations (Table 2, NV-Embed-V1)

| NOT | AND | OR | nDCG@10 |
|-----|-----|----|---------|
| $1-x$ | $x \times y$ | $x+y$ | **0.97** |
| $1-x$ | $x+y$ | $x+y$ | **0.97** |
| $1/x$ | $x \times y$ | $x+y$ | 0.90 |
| $1-x$ | $\min$ | $\max$ | 0.86 |
| $1/x$ | $\min$ | $\max$ | 0.86 |

**Multiplication AND + Addition OR + Inversion NOT** is the optimal combination.

### Main Results (Table 3)

Generated 960 compositional reasoning queries using Llama3-70b on NFCorpus, SciFact, and ArguAna:

| Method | NFCorpus | SciFact | ArguAna |
|------|----------|---------|---------|
| **NV-Embed-V1** | | | |
| Baseline | 0.56 | 0.51 | 0.51 |
| BRIGHT | 0.67 | 0.59 | 0.58 |
| ITLR | **0.74** | **0.64** | **0.64** |
| **text-embedding-v3-large** | | | |
| Baseline | 0.63 | 0.59 | 0.63 |
| BRIGHT | 0.70 | 0.63 | 0.66 |
| ITLR | **0.73** | **0.64** | **0.67** |
| **nv-embedqa-mistral-7b-v2** | | | |
| Baseline | 0.54 | 0.50 | 0.40 |
| BRIGHT | 0.48 | 0.39 | 0.29 |
| ITLR | **0.67** | **0.61** | **0.59** |

ITLR achieves the best performance in 11 out of 12 (Model × Dataset) combinations.

### Breakdown by Number of Negations (NFCorpus, NV-Embed-V1, Table 4)

| Number of Negations | Base | Reasoning | Logical |
|--------|------|-----------|---------|
| 0 | 0.81 | 0.81 | 0.76 |
| 1 | 0.60 | 0.68 | 0.71 |
| 2 | 0.51 | 0.64 | **0.76** |
| 3 | 0.36 | 0.56 | **0.73** |

**Findings**: Without negations, ITLR is slightly lower than the baseline (0.76 vs 0.81), but its advantage significantly expands when negations are present. ITLR maintains stable performance on queries with 2+ negations (0.76/0.73), whereas the baseline drops drastically.

## Highlights & Insights

1. **Elegant Problem Decomposition**: Reformulates the retrieval problem of complex queries into a logical expression solving problem, bridging discrete logic and continuous similarity scores via fuzzy logic.
2. **Plug-and-Play**: The method does not modify the underlying embedding model and can be used on top of any dense retrieval model.
3. **Computationally Efficient**: Sub-term embedding calculations can be parallelized, and score composition involves only simple arithmetic operations, introducing negligible actual latency.
4. **Systematic Findings on Negation Handling**: Negation is the Achilles' heel of traditional retrieval—the base model's performance drops to 0.52 with 3 negations, while ITLR perfectly maintains performance; this finding has direct practical implications for RAG system design.
5. **Counter-intuitive Findings on Operator Selection**: The min/max combination of traditional fuzzy logic underperforms in this scenario, while the multiplication/addition combination is superior.

## Limitations & Future Work

1. **Slight Performance Drop on Queries Without Negations**: On simple queries without negations, ITLR is about 5 percentage points lower than the baseline (0.76 vs 0.81), indicating that logical decomposition introduces a slight loss of information.
2. **Dependency on LLM Query Transformation Quality**: The query transformation is implemented using only simple prompts without fine-tuning a dedicated model, which may result in unstable transformation quality for complex natural language queries.
3. **Score Calibration Problem**: Different sub-terms may have different score distributions, leading the AND operation to favor documents that score high on certain sub-terms; the paper admits that no simple calibration method has been found yet.
4. **Evaluation Limitations**: Since both queries and labels in real-world data experiments are generated by Llama3-70b, there is a risk of evaluation circular dependency.
5. **Evaluation Restricted to Three-Term Queries**: Only three-term logical query templates were tested on real-world data, and the effectiveness on more complex nested logical structures remains unknown.

## Related Work & Insights

- **Dense Retrieval**: Sentence-BERT (Reimers, 2019), NV-Embed (Lee et al., 2024), text-embedding-v3 (Wang et al., 2023)
- **Reasoning-Enhanced Retrieval**: BRIGHT (Su et al., 2024) optimizes queries using LLM reasoning trajectories, StructGPT (Jiang et al., 2023) structured reasoning, ChatKBQA (Luo et al., 2023) integrates generation and retrieval.
- **Logical Retrieval**: Term logic-based retrieval model by Meghini et al. (1993), RELIEF (Ounis & Paşca, 1998).
- **LLM Inference-Time Reasoning**: Chain-of-Thought (Wei et al., 2022), Tree of Thoughts (Yao et al., 2024).

## Rating

⭐⭐⭐⭐ — The method is simple and elegant, showing remarkable effectiveness in handling negations and high practical value (with direct guidance for RAG/search systems). However, the slight performance degradation on queries without negations and the reliance on LLM-generated data for evaluation are minor drawbacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures](on_generalization_across_measurement_systems_llms_entail_more_test-time_compute_.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)
- [\[ACL 2025\] Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](aristotle_logical_reasoning.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](../../AAAI2026/llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)

</div>

<!-- RELATED:END -->
