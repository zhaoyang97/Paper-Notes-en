---
title: >-
  [Paper Note] Self-Correction Distillation for Structured Data Question Answering
description: >-
  [AAAI 2026][Graph Learning][Knowledge Distillation] This paper proposes Self-Correction Distillation (SCD), which transfers structured data question answering capabilities from large-scale LLMs (GPT-4) to small-scale LLMs (8B) via an Error Prompting Mechanism (EPM) and a two-stage distillation strategy, achieving state-of-the-art distillation performance across five benchmarks.
tags:
  - AAAI 2026
  - Graph Learning
  - Knowledge Distillation
  - Structured Data QA
  - CoT Distillation
  - Error Correction
  - Small Language Models
date: 2026-05-08
content_hash: bdef0876180608a9
---

# Self-Correction Distillation for Structured Data Question Answering

**Conference**: AAAI 2026
**arXiv**: [2511.07998](https://arxiv.org/abs/2511.07998)
**Code**: None
**Area**: Graph Learning
**Keywords**: Knowledge Distillation, Structured Data QA, CoT Distillation, Error Correction, Small Language Models

## TL;DR

This paper proposes Self-Correction Distillation (SCD), which transfers structured data question answering capabilities from large-scale LLMs (GPT-4) to small-scale LLMs (8B) via an Error Prompting Mechanism (EPM) and a two-stage distillation strategy, achieving state-of-the-art distillation performance across five benchmarks.

## Background & Motivation

Structured Data Question Answering (Structured Data QA) — encompassing Table QA, Knowledge Graph QA (KGQA), and Temporal KGQA — is an important research direction in NLP. Recent unified frameworks such as TrustUQA have achieved notable progress by leveraging LLMs to generate structured queries for answering natural language questions.

However, these frameworks face two core challenges:

**Deployment Constraints**: Many real-world scenarios lack the hardware resources required to deploy 100B+ parameter models, and users often prefer locally deployed small models due to data privacy concerns and API reliability issues.

**Insufficient Small-Model Capability**: Adapting unified QA frameworks to small-scale LLMs (< 10B parameters) is highly challenging — small models frequently produce errors when generating structured queries, including calls to undefined functions, illegal arguments, and nested function calls.

The authors observe that the error types of small models exhibit clear patterns (as shown in Figure 1), which motivates a "identify error type first, then correct accordingly" approach. Existing CoT distillation methods either learn only from correct outputs (Naive-SFT) or have limited error sampling coverage (PERsD), and thus cannot adequately address structured query generation failures in small models.

## Method

### Overall Architecture

SCD builds upon TrustUQA's two-level query framework. Its core contributions are: (1) an Error Prompting Mechanism (EPM) embedded within the query executor; and (2) a two-stage training strategy combining teacher distillation and self-distillation to jointly improve the student model's query generation and error correction capabilities.

### Key Designs

#### 1. Error Prompting Mechanism (EPM)

EPM is embedded within the query executor and is responsible for detecting errors in LLM-generated queries and providing customized error messages. Errors are categorized into two classes:

**Parsing Errors**: Detected during the pre-execution parsing stage, comprising five subtypes:
- Undefined function names: e.g., calling `subtract()` instead of a valid function
- Illegal arguments: e.g., `key` being invalid in `max(set=…, key=…)`
- Inconsistent arguments: simultaneously assigning values to mutually exclusive parameters
- Illegal comparison operators: use of invalid operators such as `<<`
- Non-atomic operations: nested function calls such as `sum(set=set_negation(…))`

**Execution Errors**: Detected after a query passes parsing, during the execution stage:
- Python executor exceptions: e.g., type-mismatched operations
- Empty intermediate results: an intermediate query returns an empty set, likely due to incorrect relation or entity mapping

Each error type is associated with a customized error message template that provides the LLM with specific corrective guidance. EPM is implemented via regular expression matching and achieves 100% parsing accuracy.

#### 2. Multi-Round Correction Process

Given a question and data schema, the LLM generates an initial query → the query executor parses and executes it → if no error, output the result → if an error occurs, EPM reports the error message → the LLM analyzes the error and produces a correction → re-parse and re-execute → repeat until no error or the maximum correction count $MCT$ is reached.

Let $CoT^{(i)}$ denote the error analysis after the $i$-th correction round and $q_{upd}^{(i)}$ the updated query; after $n$ rounds, a correctly executable query $q_{cor}$ is obtained.

#### 3. Two-Stage Distillation Strategy

**Stage 1 — Teacher Distillation**:

The student learns both query generation and error correction from the teacher. The teacher (GPT-4) and student generate initial queries simultaneously; erroneous queries from both are collected and subjected to multi-round correction by the teacher.

Query generation loss:

$$\mathcal{L}_q = -\sum_j \log P_\mathcal{M}(q_{cor(j)}^t | \text{P\_Q}(Q,S); q_{cor(<j)}^t)$$

Error correction loss (key design — each round targets the final correct query):

$$\mathcal{L}_c = -\sum_{i=1}^{n}\sum_j \log P_\mathcal{M}([CoT_t^{(i)};q_{cor}^t]_{(j)} | \text{P\_C}(Q,S,q_{upd}^{t(i-1)},err^{(i-1)}))$$

Total loss: $\mathcal{L}_1 = \mathcal{L}_q + \mathcal{L}_c$

**Key Insight**: Using the final correct query (rather than the per-round updated query) as the learning target in every training round naturally induces a curriculum of increasing difficulty — in later rounds $CoT^{(i)}$ aligns more closely with $q_{cor}$, making correction easier; in earlier rounds the alignment is weaker, requiring deeper understanding from the student.

**Stage 2 — Self-Distillation**:

The student iteratively improves using its own outputs. For a given question, the student generates a query and self-corrects, increasing the probability of producing correct queries and decreasing the probability of erroneous ones:

$$\mathcal{L}_2 = -\sum_{i=1}^{n}(\mathcal{S}(q_{cor}^s) - \mathcal{S}(q_{upd}^{s(i-1)}))$$

where $\mathcal{S}(q) = \sum_j \log P_\mathcal{M}(q_{(j)}|\text{P\_Q}(Q,S);q_{(<j)})$. This is essentially contrastive learning that steers the model away from its own characteristic errors.

### Loss & Training

- LoRA fine-tuning of Llama3.1-8B-Instruct
- 2× NVIDIA A100 40GB GPUs, batch size = 1, gradient accumulation = 8, 3 training epochs
- AdamW optimizer, learning rate = 0.0001, cosine schedule, warmup ratio = 0.1
- SentenceBERT for demonstration retrieval; retrieval count = 15, query generation demonstrations = 8
- Maximum correction count MCT = 3, correction demonstrations = 8

## Key Experimental Results

### Main Results

Evaluated on five datasets (WikiSQL, WTQ, MetaQA, WebQSP, CronQuestion):

| Method | WikiSQL | WTQ | MetaQA-1hop | MetaQA-3hop | WebQSP | CronQ-Complex |
|--------|---------|-----|-------------|-------------|--------|---------------|
| Prev. SOTA | 89.5 | 66.7 | 98.4 | 99.4 | 85.7 | 95.4 |
| GPT-4 w/ EPM | **91.1** | 53.2 | **98.6** | **99.3** | **91.7** | 97.3 |
| Llama3.1 | 74.5 | 35.1 | 94.6 | 83.4 | 75.2 | 80.9 |
| Naive-SFT | 79.3 | 42.2 | 96.1 | 88.1 | 78.2 | 86.4 |
| PERsD | 84.2 | 44.3 | 97.2 | 97.3 | 79.7 | 89.1 |
| KPOD | 86.1 | 46.5 | 97.1 | 97.8 | 82.3 | 92.4 |
| **SCD (Ours)** | **86.9** | **48.9** | **97.4** | **98.2** | **83.7** | **95.1** |

**GPT-4 + EPM surpasses the previous SOTA on most datasets**, and SCD achieves the best performance among all distillation methods.

### Ablation Study

| Configuration | WikiSQL | WTQ | MetaQA-3hop | WebQSP | CronQ-Complex | Note |
|---------------|---------|-----|-------------|--------|---------------|------|
| SCD (full) | 86.9 | 48.9 | 98.2 | 83.7 | 95.1 | Complete method |
| − Self-Distillation | 86.1 | 46.2 | 97.2 | 82.2 | 93.6 | Stage 2 removed |
| − EPM Error Messages | 83.7 | 44.8 | 97.7 | 79.5 | 90.9 | EPM reports error presence only, without detailed messages |

### Key Findings

1. **Correction Rate Discrepancy**: Parsing errors exhibit higher correction rates (teacher: 40.2%, student: 45.5%) than execution errors (teacher: 27.6%, student: 15.7%), since parsing errors are more surface-level while the root causes of execution errors are harder to localize.
2. **Generalization Validation**: On the unseen dataset TabFact, the SCD-trained 8B model achieves 83.4% accuracy, approaching GPT-3.5-based StructGPT (87.6%).
3. **Hyperparameter Analysis**: Large models require only 2 correction rounds to resolve most errors, while small models need 3 or more; increasing the number of correction demonstrations and rounds is effective for models trained with SCD but not for those trained without error correction.

## Highlights & Insights

1. **Error Taxonomy as Design Principle**: Systematically classifying LLM query errors into two major categories (parsing and execution) with six subtypes, each paired with a customized error message, represents a highly practical engineering design.
2. **Targeting the Final Correct Query in Multi-Round Correction** is an elegant design — it naturally induces a curriculum of increasing difficulty without requiring any explicit difficulty scheduling.
3. **Complementary Two-Stage Distillation**: Teacher distillation addresses capability transfer from scratch, while self-distillation provides targeted prevention of the student's own characteristic errors; the design rationale is logically coherent.
4. **Generality of EPM**: EPM benefits not only small models but also enables GPT-4 to surpass existing SOTA, demonstrating that precise error feedback is intrinsically valuable.

## Limitations & Future Work

1. When the correct answer is "null/none," EPM may falsely report an "empty intermediate step" execution error (false negative).
2. For complex reasoning requiring "LLM functions," the 8B model is constrained by its intrinsic capability ceiling.
3. Entity/relation alignment errors fall outside the scope of EPM.
4. The method depends on TrustUQA's specific query language; generalization to other query languages such as SQL requires additional adaptation.

## Related Work & Insights

- TrustUQA's Condition Graph representation and two-level query method provide the foundational architecture for unified structured QA.
- PERsD achieves personalized distillation by having the teacher correct student-generated code, but suffers from limited error sampling coverage.
- KPOD simulates human-like progressive learning; SCD's multi-round difficulty-increasing design offers an alternative realization of curriculum learning.
- The proposed method is extensible to other tasks requiring structured output, such as NL2SQL and code generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of EPM and two-stage distillation is highly innovative
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, three data types, comprehensive ablation and analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables
- Value: ⭐⭐⭐⭐ — A practical solution for deploying structured QA on small models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)

</div>

<!-- RELATED:END -->
