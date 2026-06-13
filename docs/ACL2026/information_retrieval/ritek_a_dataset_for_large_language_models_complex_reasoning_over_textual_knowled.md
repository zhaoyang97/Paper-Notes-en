---
title: >-
  [Paper Note] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine
description: >-
  [ACL 2026][Information Retrieval & RAG][Textual Knowledge Graph] RiTeK constructs two large-scale medical Textual Knowledge Graphs (TKG) and corresponding complex reasoning QA datasets covering 6 topological structures a…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Textual Knowledge Graph"
  - "Medical QA"
  - "Complex Reasoning"
  - "Retrieval System"
  - "Topological Structure"
date: 2026-05-08
content_hash: ce4ae21f59c06095
---

# RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine

**Conference**: ACL 2026 Findings  
**arXiv**: [2410.13987](https://arxiv.org/abs/2410.13987)  
**Code**: [https://github.com/ToneLi/Medical-Textual-KG-Reasoning-Benchmark](https://github.com/ToneLi/Medical-Textual-KG-Reasoning-Benchmark)  
**Area**: Medical Imaging  
**Keywords**: Textual Knowledge Graph, Medical QA, Complex Reasoning, Retrieval System, Topological Structure

## TL;DR

RiTeK constructs two large-scale medical Textual Knowledge Graphs (TKG) and corresponding complex reasoning QA datasets covering 6 topological structures and rich textual descriptions. It evaluates 11 retrieval methods and reveals significant deficiencies in existing LLM-driven retrieval systems for medical TKG reasoning.

## Background & Motivation

**Background**: Answering complex medical questions requires precise retrieval of relational path information from medical Textual Knowledge Graphs (TKG) to enhance LLM reasoning capabilities. TKG combines structured entity relations with unstructured textual descriptions, expressing richer semantics than traditional KGs.

**Limitations of Prior Work**: (1) Existing medical TKGs are scarce with limited topological expressiveness (e.g., STaRK-Prime covers only 3 structures); (2) Existing datasets often require only 1-2 hop reasoning paths, which are too simple to reflect real-world medical complexity; (3) Textual description coverage for nodes is low (STaRK-Prime is only 15.29%), limiting semantic understanding; (4) There is a lack of comprehensive evaluation of existing LLM retrieval systems on medical TKGs.

**Key Challenge**: Real medical queries involve multi-hop reasoning, multiple constraints, and complex topological structures, yet existing datasets and retrieval systems cannot effectively handle this complexity.

**Goal**: (1) Construct medical TKGs with rich topological structures and complete textual descriptions; (2) Generate complex query datasets merging relational information and textual attributes; (3) Comprehensively evaluate the capabilities and deficiencies of current retrieval systems.

**Key Insight**: Enhance dataset quality from two dimensions: topological diversity and textual information richness. Generate queries using 6 topological templates validated by medical experts for naturalness, diversity, and utility.

**Core Idea**: Build a medical TKG benchmark that tests both relational reasoning and semantic alignment capabilities, exposing bottlenecks in existing methods and identifying directions for improvement.

## Method

### Overall Architecture

The construction of RiTeK consists of two parts: (1) Medical TKG Construction—enriching node textual descriptions from databases like Ensembl, UMLS, and Mondo Disease Ontology based on PharmKG and ADint; (2) QA Dataset Construction—generating complex queries via a five-step pipeline: relation template design → textual attribute extraction → information synthesis → extra answer filtering → expert evaluation.

### Key Designs

1.  **Medical TKG Construction**:
    *   **Function**: Provide a foundation of medical knowledge graphs rich in structure and text.
    *   **Mechanism**: Based on PharmKG (3 entity types, 29 relations, 500K+ triples) and ADint (102 entity types, 15 relations, 1M+ triples), textual descriptions are integrated from multiple medical databases. RiTeK-PharmKG achieves a 95.61% node text coverage rate (vs. 15.29% for STaRK-Prime). A Textual Triple Graph is defined where each node is $(h, r, t, T(h), T(t))$, unifying relational and textual information.
    *   **Design Motivation**: High text coverage allows queries to include both structural and semantic constraints, closely mimicking complex questions from patients/doctors in real medical scenarios.

2.  **Five-step QA Dataset Construction Pipeline**:
    *   **Function**: Generate high-quality medical queries merging relational and textual information.
    *   **Mechanism**: (a) Medical experts design relation templates for 6 topological structures (e.g., multi-hop, multi-hop with constraints), RiTeK-PharmKG features 68 templates with an instance rate of 11.33 (surpassing STaRK's 1.25-9.3); (b) Select candidate answer entities and extract textual attributes using GPT-4; (c) Synthesize relational info and text attributes into natural language queries, simulating three roles (medical scientist, doctor, patient) for diversity; (d) Use multiple LLMs to verify if other candidates satisfy query conditions; (e) Four medical experts evaluate 1000 samples for naturalness/diversity/utility.
    *   **Design Motivation**: Higher instance rates imply denser topological coverage; multi-role simulation increases linguistic style diversity; multi-round LLM filtering ensures the completeness of answer sets.

3.  **Comprehensive Retrieval System Evaluation**:
    *   **Function**: Systematically evaluate existing methods on complex medical TKG reasoning.
    *   **Mechanism**: Evaluate 11 retrieval methods under zero-shot, few-shot, and supervised settings, including direct generation (GPT-4), graph search (Random Walk, MCTS), prompting strategies (CoT, ToT, GoT, ToG), RAG methods (G-retriever, KAR), and supervised methods (GCR, GNN-RAG), using Exact Match and ROUGE-1 metrics.
    *   **Design Motivation**: Provide systemic baseline comparisons to identify strengths and weaknesses in relational reasoning and semantic alignment.

### Loss & Training

The dataset construction phase does not involve model training. Supervised methods (G-retriever, GCR, GNN-RAG) are fine-tuned on 80% of the training set. Query generation utilizes GPT-4o-mini.

## Key Experimental Results

### Main Results

**RiTeK-PharmKG zero-shot results (Exact Match F1 %)**

| Method | EM F1 |
|------|-------|
| GPT-4 | 11.03 |
| GPT-4 + COT | 13.70 |
| GPT-4 + TOT | 7.22 |
| GPT-4 + GOT | 3.75 |
| GPT-4 + TOG | 31.14 |
| GPT-4 + KAR | 25.18 |
| G-retriever (supervised) | 37.62 |
| GCR (supervised) | 47.71 |
| GNN-RAG (supervised) | **49.72** |

**RiTeK-ADint zero-shot results (Exact Match F1 %)**

| Method | EM F1 |
|------|-------|
| GPT-4 | 8.03 |
| GPT-4 + KAR | 27.29 |
| GNN-RAG (supervised) | **50.55** |

### Ablation Study

**Human Evaluation Results (Positive/Acceptable %)**

| Dimension | RiTeK-PharmKG | RiTeK-ADint |
|------|---------------|-------------|
| Naturalness | 81.80/99.60 | 81.20/99.20 |
| Diversity | 81.60/99.40 | 74.80/100 |
| Utility | 67.40/97.80 | 68.60/96.60 |

**Dataset Statistics Comparison**

| Dataset | Queries | Topologies | Instance Rate |
|--------|--------|-----------|--------|
| STaRK-Prime | 11,204 | 3 | 9.3 |
| RiTeK-PharmKG | 10,235 | 6 | 11.33 |
| RiTeK-ADint | 5,322 | 6 | 9.67 |

### Key Findings

- All zero-shot methods perform poorly on complex medical TKG reasoning; GPT-4 direct generation averages only ~11% EM F1, indicating insufficient internal knowledge for complex relational reasoning.
- ToT and GoT underperform relative to simple CoT, suggesting structured prompting logic backfires without external knowledge access.
- KAR (knowledge-aware method combining text semantics and structural relations) performs best in zero-shot, validating the importance of dual structural and textual information.
- Even the strongest supervised method, GNN-RAG, achieves less than 50% EM F1, highlighting the immense challenge of complex reasoning in medical TKGs.

## Highlights & Insights

- Clear dataset design philosophy: Elevates TKG QA from simple lookup to true complex reasoning through enriched topology and high text coverage.
- Multi-role simulation (scientist/doctor/patient) aligns generated queries with real-world usage scenarios.
- 95.61% text coverage is a massive improvement over STaRK-Prime (15.29%).
- Evaluation results provide a clear direction for the community: need for retrieval systems capable of simultaneous structural reasoning and textual semantic alignment.

## Limitations & Future Work

- Query generation via GPT-4 may introduce synthetic bias, remaining distant from real user queries.
- Evaluates only English medical queries, lacking coverage of multilingual medical scenarios.
- Latency issues in large-scale TKGs were not explored in depth.
- Future work could explore efficient hybrid retrieval architectures and pre-training strategies specialized for medical TKGs.

## Related Work & Insights

- Compared to the STaRK series, RiTeK provides richer topologies and higher text coverage in the medical domain.
- ToG performs excellently in few-shot settings, indicating that beam search on KGs + few-shot examples is a promising direction.
- The relative success of GNN-RAG demonstrates the advantage of GNNs in processing structured path information.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills the dataset gap in the medical TKG field, though methodology is primarily dataset construction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systemic evaluation of 11 retrieval methods, including human evaluation and cross-dataset comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed construction process, though some table information is redundant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](../../ICLR2026/information_retrieval/synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->
