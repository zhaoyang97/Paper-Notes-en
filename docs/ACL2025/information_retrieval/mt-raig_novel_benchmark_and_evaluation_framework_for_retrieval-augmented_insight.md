---
title: >-
  [Paper Note] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables
description: >-
  [ACL 2025][Information Retrieval & RAG][Multi-Table Reasoning] This paper introduces MT-RAIG Bench—the first large-scale benchmark for retrieval-augmented insight generation over multiple tables—and MT-RAIG Eval—a decomposition-based, fine-grained automatic evaluation framework. Experiments demonstrate that even frontier LLMs underperform on multi-table reasoning (achieving only around 40% faithfulness and 60% completeness).
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Multi-Table Reasoning"
  - "Retrieval-Augmented Generation"
  - "Table Question Answering"
  - "Insight Generation"
  - "Automated Evaluation Framework"
date: 2026-05-08
content_hash: 527e47dd8514f1fa
---

# MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables

**Conference**: ACL 2025  
**arXiv**: [2502.11735](https://arxiv.org/abs/2502.11735)  
**Authors**: Kwangwook Seo, Donguk Kwon, Dongha Lee (Yonsei University)  
**Code**: [https://kwondu.github.io/mt-raig](https://kwondu.github.io/mt-raig)  
**Area**: Information Retrieval  
**Keywords**: Multi-Table Reasoning, Retrieval-Augmented Generation, Table Question Answering, Insight Generation, Automated Evaluation Framework  

## TL;DR

This paper introduces MT-RAIG Bench—the first large-scale benchmark for retrieval-augmented insight generation over multiple tables—and MT-RAIG Eval—a decomposition-based, fine-grained automatic evaluation framework. Experiments demonstrate that even frontier LLMs underperform on multi-table reasoning (achieving only around 40% faithfulness and 60% completeness).

## Background & Motivation

### Background
Tables are ubiquitous structured information carriers in real-world data. Existing research on table reasoning has evolved from simple fact-level question answering (e.g., WikiSQL, WTQ) to insight-level tasks (e.g., QTSumm, InsTaSumm). The latter requires systems to synthesize implicit knowledge from tables and provide interpretable analyses.

### Limitations of Prior Work

**Closed-domain assumption**: Existing methods provide pre-defined gold tables during testing. However, in real-world scenarios, users often do not know which tables are relevant to their queries, and manually specifying tables is both expensive and impractical.

**Single-table restriction**: Most works assume all information is contained within a single table, failing to handle user requirements that necessitate synthesizing insights across multiple tables.

**Coarse evaluation methods**: Existing automatic evaluation metrics (such as BLEU, ROUGE, G-Eval, etc.) analyze outputs at a coarse grain, making it difficult to detect factual alignment between insights and multiple tables, as well as complete coverage of multi-hop queries.

### Design Motivation
The goal is to construct an end-to-end benchmark that requires systems to "first retrieve multiple evidence tables, and then synthesize insights across tables," while designing a fine-grained evaluation framework highly aligned with human judgment.

## Method

### MT-RAIG Bench Construction

#### Task Definition
Given a natural language query $q$ and an external table repository $T$, the system needs to: (1) retrieve a set of evidence tables $\hat{T_q} \subset T$; (2) generate an insight $i = Gen(q, \hat{T_q})$ based on the retrieval results.

#### Multi-Table Set Construction
Multi-table sets are collected from two sources:
- **Joinable tables**: Leveraging foreign key relationships in the SPIDER dataset, these are directly connected through common key columns.
- **Topic-related tables**: Leveraging table titles and column headers in Open-WikiTable as semantic indicators, tables with thematic correlations but no direct key column connections are clustered.

#### Question Annotation (Human-in-the-loop)
GPT-4o mini is used as an annotator agent to generate 10 insight-level questions for each table set. Three key strategies are employed:
1. **Decontextualization**: Embedding table title keywords into questions to ensure semantic alignment.
2. **Relationship Enhancement**: Manually identifying shared attributes and data points across tables to guide the agent in generating questions that reflect relationships between tables.
3. **Human-guided Demonstration Iteration**: Categorizing questions into 4 classes (Analysis & Summary (A&S), Comparison & Relation (C&R), Performance & Outcomes (P&O), Trends & Patterns (T&P)) and iteratively optimizing seed examples through a human review-and-feedback loop.

#### Insight Annotation
1. **Programmatic Fact Expansion**: Instructuring the agent to generate a Python function `expand_facts` executed on the tables to systematically extract rich facts.
2. **Query-Relevant Knowledge Extraction**: Filtering out irrelevant content, keeping only facts related to the question, and then having the agent generate the final insight.

#### Two-Stage Quality Control
- Stage 1: Agent self-verification, checking for relevance, faithfulness, and completeness.
- Stage 2: Human verification, achieving a Cohen's Kappa of 0.78-0.86, ensuring high quality.

### MT-RAIG Eval Evaluation Framework

#### Faithfulness Scoring
Through **table-aware insight decomposition**, the insight is broken down into verifiable atomic claims, with each claim explicitly associated with its source table:
$$S_{Faith.} = \frac{1}{|C|} \sum_{k=1}^{|C|} \mathcal{V}(c_k, \hat{T_q})$$
where $\mathcal{V}(c_k, \hat{T_q}) \in \{0,1\}$ verifies whether each claim is supported by the tables.

#### Completeness Scoring
Through **query-aware insight decomposition**, the ground truth and predicted insights are respectively decomposed into atomic topics, and precision, recall, and F1 of semantic matching are calculated:
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

## Key Experimental Results

### Experiment 1: Multi-Table Retrieval Performance

| Type | Retriever | R@2 | R@5 | R@10 | R@20 |
|------|--------|-----|-----|------|------|
| General | BM25 | 17.26 | 27.19 | 33.72 | 41.16 |
| General | DPR | 44.58 | 68.24 | 80.83 | 88.45 |
| General | Contriever | 23.47 | 35.67 | 44.27 | 52.12 |
| Table-specific | DTR | 37.77 | 59.60 | 74.50 | 86.22 |
| Table-specific | TableLlama | 36.93 | 59.48 | 72.44 | 81.56 |

The general text embedding model DPR surprisingly outperforms table-specific models, indicating that the MT-RAIG task prioritizes insight-level semantic connections over structural table features.

### Experiment 2: Insight Generation Performance (Using DPR top-10 retrieved tables)

| Generator | Faithfulness Avg | Faithfulness Gold | Completeness Avg | Completeness Gold |
|--------|-----------|------------|-----------|------------|
| o3-mini | 38.85 | 42.57 | 60.20 | 60.81 |
| GPT-4o | 36.98 | 41.46 | 61.52 | 63.28 |
| Claude 3.5 Sonnet | 39.68 | 43.35 | 58.63 | 59.70 |
| DeepSeek-R1-8B | 35.55 | 40.12 | 60.96 | 63.41 |
| Qwen2-7B | 33.40 | 40.59 | 59.21 | 61.57 |
| Llama 3.1-8B | 31.59 | 37.62 | 56.90 | 58.14 |
| Dater (TQA) | 27.92 | 32.43 | 59.12 | 62.32 |
| Chain-of-Table (TQA) | 31.90 | 37.71 | 57.63 | 62.15 |
| TaPERA (TQA) | 19.25 | 20.68 | 56.59 | 55.28 |

### Experiment 3: MT-RAIG Eval Meta-Evaluation

| Evaluation Metric | Faithfulness Correlation | Completeness Correlation |
|---------|------------|------------|
| SacreBLEU | 31.33 | 33.01 |
| ROUGE-L | 27.69 | 43.43 |
| BERTScore | 24.82 | 43.29 |
| G-Eval | 47.82 | 26.35 |
| **MT-RAIG Eval** | **64.94** | **67.67** |
| Inter-annotator Agreement | 84.81 | 75.70 |

MT-RAIG Eval substantially outperforms existing metrics across both dimensions, achieving a Pearson correlation of 64.94 / 67.67 with human judgment.

## Highlights & Insights

- **First-of-its-kind benchmark**: MT-RAIG Bench is the first large-scale benchmark that requires retrieval + multi-table synthesis + insight generation (containing 18,532 test samples, 19,563 unique tables, with an average of 2.88 gold tables per sample).
- **Fine-grained evaluation framework**: MT-RAIG Eval employs a dual-path decomposition strategy (table-aware and query-aware), achieving human alignment that far exceeds traditional metrics.
- **In-depth analytical findings**: (1) Retrieving more tables is not always beneficial, as performance drops after exceeding a threshold; (2) Faithfulness is highly sensitive to noisy tables, whereas completeness is relatively robust; (3) Even without noise, model performance drops significantly as reference table quantity increases.
- **Effective test-time scaling**: DeepSeek-R1-8B, with only an 8B parameter size, reaches a performance level comparable to closed-source models, validating the efficacy of test-time scaling in table-insight tasks.

## Limitations & Future Work

- **Inherent risks of synthetic data**: Despite employing a human-in-the-loop setting, the queries generated by LLMs may still suffer from limited linguistic diversity and unnatural phrasing.
- **Limited domain coverage**: The dataset currently only includes general Wikipedia tables and relational databases, lacking coverage of specialized domains such as finance, science, or medicine.
- **Evaluation dependency on LLMs**: Both the decomposer and verifier in MT-RAIG Eval rely on LLMs, making the framework susceptible to potential biases in backbone models.
- **Decoupled retrieval and generation**: The experimental design of using a fixed DPR top-10 setting may underestimate the potential of joint end-to-end retrieval and generation optimization.
- **Table size limitations**: The average table size is only 10.54 rows by 6.04 columns, which does not represent large-scale industrial table scenarios.

## Related Work & Insights

- **WikiSQL/SPIDER/WTQ**: Fact-level single-table tasks. MT-RAIG comprehensively upgrades these in terms of reasoning depth (insight-level), table quantity (multi-table), and retrieval requirements.
- **QTSumm/InsTaSumm**: Though focused on insight-level generation, they only support single tables and closed-domain setups. MT-RAIG simultaneously requires retrieval and multi-table synthesis.
- **Open-WikiTable**: Though it supports retrieval, it remains a fact-level QA task. MT-RAIG mandates the generation of long-text insights (averaging 189.87 words vs. 1.90 words).
- **G-Eval**: A coarse-grained LLM evaluation method, yielding a faithfulness correlation of only 47.82 vs. MT-RAIG Eval's 64.94.
- **TAPAS-Acc**: A trained verifier, which exhibits a negative correlation (-10.40) on the MT-RAIG task, rendering it entirely inapplicable.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first multi-table retrieval-augmented insight generation benchmark, filling a critical gap in task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive, covering 5 types of retrievers, 11 types of generators, meta-evaluations, and multi-dimensional analytical experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with detailed dataset construction processes and rich diagrams.
- Value: ⭐⭐⭐⭐ — Provides a high-quality, challenging benchmark and a reliable evaluation tool for the table reasoning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)

</div>

<!-- RELATED:END -->
