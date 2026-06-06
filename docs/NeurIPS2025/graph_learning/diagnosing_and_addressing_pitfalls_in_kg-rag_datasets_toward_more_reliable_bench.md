---
title: >-
  [Paper Note] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking
description: >-
  [NeurIPS 2025][Graph Learning][KG-RAG] A systematic audit of 16 KGQA datasets reveals an average factual correctness of only 57% (WebQSP: 52%, MetaQA: 20%). The paper proposes KGQAGen…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "KG-RAG"
  - "KGQA Benchmarking"
  - "Dataset Quality Auditing"
  - "SPARQL Validation"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: daaf427ae85a189e
---

# Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking

**Conference**: NeurIPS 2025
**arXiv**: [2505.23495](https://arxiv.org/abs/2505.23495)  
**Code**: [https://github.com/liangliang6v6/KGQAGen](https://github.com/liangliang6v6/KGQAGen)  
**Area**: NLP Understanding / Knowledge Graphs
**Keywords**: KG-RAG, KGQA Benchmarking, Dataset Quality Auditing, SPARQL Validation, Multi-hop Reasoning

## TL;DR
A systematic audit of 16 KGQA datasets reveals an average factual correctness of only 57% (WebQSP: 52%, MetaQA: 20%). The paper proposes KGQAGen, a framework that constructs high-quality multi-hop QA datasets via LLM-guided subgraph expansion and automatic SPARQL validation, yielding KGQAGen-10k with 96.3% accuracy. The study further demonstrates that the primary bottleneck in KG-RAG lies in retrieval rather than reasoning.

## Background & Motivation

**Background**: Knowledge graph question answering (KGQA) and KG-augmented RAG (KG-RAG) are active research directions, with canonical datasets such as WebQSP and CWQ widely adopted for evaluation. Reported Hit@1 scores on these benchmarks have been steadily climbing.

**Limitations of Prior Work**: A systematic audit reveals serious quality issues in these canonical datasets—annotated answers are incorrect, outdated, or incomplete (Freebase is no longer maintained), questions are ambiguous or overly simple, and exact match (EM) evaluation is overly strict. WebQSP achieves a factual correctness rate of only 52%, and MetaQA only 20%.

**Key Challenge**: Evaluating on incorrectly annotated data means that a model reporting 97% Hit@1 may be genuinely correct only 48% of the time—rendering progress assessments across the KGQA community potentially unreliable.

**Goal**: (a) Quantify and categorize quality issues in existing datasets; (b) construct a high-quality, verifiable KGQA benchmark.

**Core Idea**: LLM-guided subgraph expansion generates multi-hop questions → SPARQL-based automatic answer verification → KGQAGen-10k with 96.3% accuracy.

## Method

### Overall Architecture
KGQAGen operates in three stages: (1) seed entities are sampled from Wikipedia Vital Articles, and 1-hop subgraphs (15 neighbors) are constructed; (2) an LLM assesses whether the subgraph is sufficient for multi-hop question generation—if not, an exploration set is selected and expanded, followed by QA and SPARQL generation; (3) SPARQL queries are executed to verify answer consistency, with up to three rounds of correction; inconsistent instances are discarded.

### Key Designs

1. **LLM-Guided Iterative Subgraph Expansion**:

    - Function: Starting from a seed entity, iteratively expand the subgraph until it contains sufficient information to support multi-hop question generation.
    - Mechanism: After each expansion round, the LLM evaluates whether the subgraph is adequate (requiring at least 2-hop reasoning support). If not, it outputs an Exploration Set, prioritizing semantically specific entities (e.g., celebrities, events) over generic ones (e.g., countries), and performs 1-hop expansion on each entity in the set (sampling 10–15 neighbors per entity).
    - Design Motivation: Unguided BFS/DFS expansion produces excessively large and noisy subgraphs (high-degree nodes can yield thousands of nodes within a few hops). LLM guidance ensures that expansion remains semantically relevant and computationally tractable.

2. **SPARQL Validation Loop**:

    - Function: Ensure every generated QA pair has a traceable grounding in the knowledge graph.
    - Mechanism: The LLM jointly generates the question $q_e$, answer set $\mathcal{A}_e$, supporting subgraph $\mathcal{P}_e$, and SPARQL query $\mathcal{Q}_e$. The query is executed to verify $\hat{\mathcal{A}}_e = \mathcal{A}_e$; inconsistencies trigger SPARQL correction via GPT-4o-mini for up to three rounds.
    - Design Motivation: LLMs may hallucinate answers absent from the KG; SPARQL execution serves as a hard-constraint verification mechanism.

3. **LASM Evaluation Protocol (LLM-Assisted Semantic Match)**:

    - Function: When EM fails, GPT-4o-mini is employed to judge semantic equivalence.
    - Design Motivation: Traditional EM evaluation is overly sensitive to surface-form variation (e.g., "AUD" vs. "Australian dollar," "Germany" vs. "Federal Republic of Germany"), producing a large number of false negatives.

## Key Experimental Results

### Dataset Audit (16 Datasets, 1,000+ Samples Manually Inspected)

| Dataset | KG | Year | Factual Correctness |
|--------|-----|------|-----------|
| WebQSP | Freebase | 2016 | 52.00% |
| CWQ | Freebase | 2018 | 49.33% |
| MetaQA | WikiMovies | 2018 | 20.00% |
| GrailQA | Freebase | 2020 | 30.00% |
| LC-QuAD 1.0&2.0 | DB/Wiki | 2017/19 | 38.34% |
| Dynamic-KGQA | YAGO | 2025 | 45.00% |
| FreebaseQA | Freebase | 2019 | 98.67% (but questions are overly simple) |
| **Average** | - | - | **57%** |

Three categories of annotation issues are identified: incorrect annotations (answers inconsistent with question intent), outdated answers (e.g., the president of Peru still annotated with a 2011 entry), and incomplete annotations (only partial answers provided for set-type questions).
Three categories of question quality issues are identified: ambiguous wording (e.g., "Which George Wilson?"), low complexity (single-hop factual questions), and unanswerable or subjective questions (e.g., "What to do today in Atlanta?").

### KGQAGen-10k Benchmark Evaluation

| Type | Method | LASM Acc | LASM Hit@1 | LASM F1 |
|------|------|---------|------------|---------|
| LLM-only | LLaMA-3.1-8B | 11.91% | 12.42% | 11.98% |
| LLM-only | Mistral-7B | 32.34% | 34.38% | 33.20% |
| LLM-only | GPT-4o | 54.21% | 57.46% | 54.93% |
| LLM-only | GPT-4.1 | 56.96% | 59.96% | 57.72% |
| KG-RAG | RoG (LLaMA2) | 27.28% | 28.92% | 24.26% |
| KG-RAG | PoG (GPT-4o) | ~60% | - | - |
| LLM-SP | GPT-4o + GT Subgraph | **84.89%** | - | - |
| LLM-SP | LLaMA2 + GT Subgraph | 73.79% | - | - |

### Key Findings
- **Retrieval is the primary bottleneck**: GPT-4o achieves 54.21% without retrieval and 84.89% with ground-truth subgraphs—a 30+ point gap attributable entirely to retrieval quality.
- **Limited gains from KG-RAG models**: The best KG-RAG system (PoG, ~60%) outperforms the pure LLM baseline by only ~4%, indicating that current KG retrieval strategies fail to effectively leverage the KG.
- **High discriminative power of KGQAGen-10k**: Models that achieve 85–92% Hit@1 on WebQSP drop to 21–54% on KGQAGen-10k.
- **Substantial LASM vs. EM gap**: LASM scores exceed EM scores by an average of 5–10%, indicating that pure EM evaluation systematically underestimates true model capability.

## Highlights & Insights
- **A warning to the community**: The paper exposes a "emperor's new clothes" phenomenon in KGQA research—a large body of work reports scores on datasets with less than 50% factual correctness, with 30+ papers from 2022–2025 relying on these flawed benchmarks.
- **Elegant design of SPARQL validation**: The automated and traceable verification pipeline is more scalable and reliable than manual annotation, and the dataset can be re-validated as the KG is updated.
- **Systematic error taxonomy**: Annotation errors are classified into three types (incorrect annotation / outdated answers / incomplete annotation), and question quality issues into three types (ambiguity / excessive simplicity / unanswerable), providing a reusable framework for future dataset quality research.
- **Statistical properties of KGQAGen-10k**: 98% of questions require 2–5 hop reasoning, 84% contain 5–30 entities, and 61% of questions are 16–30 words in length.

## Limitations & Future Work
- KGQAGen relies on Wikidata; adaptation to other KGs (e.g., medical or financial domain KGs) requires additional engineering.
- SPARQL generation is itself error-prone (though multi-round correction mitigates this risk); approximately 30% of instances are filtered out, reducing the initial 15,451 generated instances to 10,787 validated ones.
- Seed entity selection from Wikipedia Vital Articles may introduce topical bias toward Arts (42.3%).
- LLM overhead for subgraph expansion is non-trivial: each QA instance requires multiple GPT-4.1 calls, making large-scale generation costly.

## Related Work & Insights
- **vs. Dynamic-KGQA**: Also uses LLMs for question generation, but suffers heavily from KG sparsity and hallucinations (correctness rate only 45%); KGQAGen's SPARQL validation effectively eliminates hallucinated answers.
- **vs. Maestro**: A rule-based automatic construction framework that depends on manually defined predicate rules, limiting generalizability.
- **vs. CHATTY-Gen**: Introduces conversational-style questions but does not address answer correctness.
- **Implications for RAG evaluation**: A robust benchmark should incorporate a formal verification mechanism (e.g., SPARQL execution) rather than relying solely on human annotation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of dataset auditing and automated construction offers significant value to the community.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Audits 16 datasets, introduces a 10k benchmark, and evaluates 9 models.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is thorough and well-supported by empirical evidence.
- Value: ⭐⭐⭐⭐⭐ — A foundational contribution to the KGQA/KG-RAG community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[NeurIPS 2025\] When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning](when_no_paths_lead_to_rome_benchmarking_systematic_neural_relational_reasoning.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)

</div>

<!-- RELATED:END -->
