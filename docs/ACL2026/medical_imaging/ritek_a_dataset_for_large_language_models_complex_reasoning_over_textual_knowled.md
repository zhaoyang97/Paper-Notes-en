---
title: >-
  [Paper Note] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine
description: >-
  [ACL 2026][Medical Imaging][Textual Knowledge Graph] RiTeK constructs two large-scale medical textual knowledge graphs (TKGs) and corresponding complex reasoning QA datasets…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Textual Knowledge Graph"
  - "Medical QA"
  - "Complex Reasoning"
  - "Retrieval System"
  - "Topological Structure"
date: 2026-05-08
content_hash: 0024d46366fd5bf5
---

# RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine

**Conference**: ACL 2026
**arXiv**: [2410.13987](https://arxiv.org/abs/2410.13987)
**Code**: [https://github.com/ToneLi/Medical-Textual-KG-Reasoning-Benchmark](https://github.com/ToneLi/Medical-Textual-KG-Reasoning-Benchmark)
**Area**: Medical Imaging
**Keywords**: Textual Knowledge Graph, Medical QA, Complex Reasoning, Retrieval System, Topological Structure

## TL;DR

RiTeK constructs two large-scale medical textual knowledge graphs (TKGs) and corresponding complex reasoning QA datasets, covering 6 topological structures with rich textual descriptions. It evaluates 11 retrieval methods and reveals critical deficiencies in existing LLM-driven retrieval systems for medical TKG reasoning.

## Background & Motivation

**Background**: Answering complex medical questions requires precise retrieval of relational path information from medical textual knowledge graphs (TKGs) to augment LLM reasoning. TKGs integrate structured entity relations with unstructured textual descriptions, enabling richer semantic expression than traditional KGs.

**Limitations of Prior Work**: (1) Existing medical TKGs are scarce with limited topological expressiveness (e.g., STaRK-Prime covers only 3 structure types); (2) Existing datasets typically require only 1–2 hop reasoning paths, which is overly simplistic and fails to reflect the complexity of real medical scenarios; (3) Node text description coverage is low (STaRK-Prime: only 15.29%), limiting semantic understanding; (4) There is no comprehensive evaluation of existing LLM retrieval systems on medical TKGs.

**Key Challenge**: Real-world medical queries involve multi-hop reasoning, multiple constraints, and complex topological structures, yet existing datasets and retrieval systems are ill-equipped to handle such complexity.

**Goal**: (1) Construct medical TKGs with diverse topological structures and complete textual descriptions; (2) Generate complex query datasets that integrate relational and textual attribute information; (3) Comprehensively evaluate the capabilities and limitations of existing retrieval systems.

**Key Insight**: Simultaneously improving dataset quality along two dimensions—topological diversity and textual richness—by generating queries from 6 topological structure templates, with naturalism, diversity, and utility validated by medical experts.

**Core Idea**: Construct a medical TKG benchmark that challenges both relational reasoning and semantic alignment capabilities, exposing bottlenecks in existing methods and pointing toward directions for improvement.

## Method

### Overall Architecture

RiTeK construction is divided into two major components: (1) Medical TKG construction—based on the PharmKG and ADint knowledge graphs, with node textual descriptions enriched from databases such as Ensembl, UMLS, and Mondo Disease Ontology; (2) QA dataset construction—complex queries are generated through a five-step pipeline: relational template design → textual attribute extraction → information synthesis → additional answer filtering → expert evaluation.

### Key Designs

1. **Medical TKG Construction**:

    - **Function**: Provides a medically grounded knowledge graph with rich structural and textual information.
    - **Mechanism**: Built upon PharmKG (3 entity types, 29 relation types, 500K+ triples) and ADint (102 entity types, 15 relation types, 1M+ triples), with textual descriptions integrated from multiple medical databases. RiTeK-PharmKG achieves 95.61% node text coverage (compared to STaRK-Prime's 15.29%). A Textual Triple Graph is defined, where each node is represented as $(h, r, t, T(h), T(t))$, unifying relational and textual information.
    - **Design Motivation**: High text coverage enables queries to incorporate both structural and semantic constraints, more closely resembling the complex questions posed by patients and clinicians in real medical settings.

2. **Five-Step QA Dataset Construction Pipeline**:

    - **Function**: Generates high-quality medical queries that integrate relational and textual information.
    - **Mechanism**: (a) Medical experts design relational templates for 6 topological structures (e.g., multi-hop, constrained multi-hop); RiTeK-PharmKG contains 68 templates with an instantiation rate of 11.33, far exceeding STaRK's 1.25–9.3; (b) Candidate answer entities are selected and textual attributes are extracted via GPT-4; (c) Relational information and textual attributes are synthesized into natural language queries simulating three roles (medical scientist, physician, patient) to increase diversity; (d) Multiple LLMs verify whether additional candidate entities also satisfy the query conditions; (e) Four medical experts evaluate 1,000 samples across naturalism, diversity, and utility dimensions.
    - **Design Motivation**: Higher instantiation rates indicate denser topological coverage; multi-role simulation enriches linguistic style diversity; multi-round LLM filtering ensures completeness of the answer set.

3. **Comprehensive Retrieval System Evaluation**:

    - **Function**: Systematically benchmarks existing methods on complex medical TKG reasoning.
    - **Mechanism**: Evaluates 11 retrieval methods under zero-shot, few-shot, and supervised settings, including direct generation (GPT-4), graph search (Random Walk, MCTS), prompting strategies (COT, TOT, GOT, TOG), RAG methods (G-retriever, KAR), and supervised methods (GCR, GNN-RAG), using Exact Match and ROUGE-1 metrics.
    - **Design Motivation**: Provides systematic baseline comparisons to identify the strengths and weaknesses of different approaches in relational reasoning and semantic alignment.

### Loss & Training

No model training is involved during dataset construction. Supervised methods (G-retriever, GCR, GNN-RAG) are fine-tuned on 80% of the training split. Query generation employs GPT-4o-mini.

## Key Experimental Results

### Main Results

**RiTeK-PharmKG Zero-shot Results (Exact Match F1 %)**

| Method | EM F1 |
|--------|-------|
| GPT-4 | 11.03 |
| GPT-4 + COT | 13.70 |
| GPT-4 + TOT | 7.22 |
| GPT-4 + GOT | 3.75 |
| GPT-4 + TOG | 31.14 |
| GPT-4 + KAR | 25.18 |
| G-retriever (supervised) | 37.62 |
| GCR (supervised) | 47.71 |
| GNN-RAG (supervised) | **49.72** |

**RiTeK-ADint Zero-shot Results (Exact Match F1 %)**

| Method | EM F1 |
|--------|-------|
| GPT-4 | 8.03 |
| GPT-4 + KAR | 27.29 |
| GNN-RAG (supervised) | **50.55** |

### Ablation Study

**Human Evaluation Results (Positive/Acceptable %)**

| Dimension | RiTeK-PharmKG | RiTeK-ADint |
|-----------|---------------|-------------|
| Naturalism | 81.80/99.60 | 81.20/99.20 |
| Diversity | 81.60/99.40 | 74.80/100 |
| Utility | 67.40/97.80 | 68.60/96.60 |

**Dataset Statistics Comparison**

| Dataset | # Queries | # Topologies | Instantiation Rate |
|---------|-----------|---------------|--------------------|
| STaRK-Prime | 11,204 | 3 | 9.3 |
| RiTeK-PharmKG | 10,235 | 6 | 11.33 |
| RiTeK-ADint | 5,322 | 6 | 9.67 |

### Key Findings

- All zero-shot methods perform poorly on complex medical TKG reasoning; GPT-4 direct generation achieves only ~11% EM F1, demonstrating that LLMs' intrinsic knowledge is insufficient for complex relational reasoning.
- TOT and GOT underperform even simple COT, suggesting that structured prompting strategies are counterproductive in the absence of external knowledge access.
- KAR, which combines textual semantics with structural relations, achieves the best zero-shot performance, validating the importance of joint structural and textual information.
- Even the strongest supervised method, GNN-RAG, achieves less than 50% EM F1, underscoring the substantial challenge of complex reasoning over medical TKGs.

## Highlights & Insights

- The dataset design philosophy is well-motivated: by enriching topological structures and achieving high text coverage, TKG QA is elevated from simple entity lookup to genuine complex reasoning.
- Multi-role simulation (scientist/physician/patient) grounds the generated queries in realistic usage scenarios.
- The 95.61% text coverage represents a substantial improvement over STaRK-Prime's 15.29%.
- Evaluation results provide a clear direction for the community: retrieval systems capable of jointly handling structural reasoning and textual semantic alignment are urgently needed.

## Limitations & Future Work

- Using GPT-4 for query generation may introduce synthetic bias, leaving a gap relative to real user queries.
- Only English medical queries are evaluated; multilingual medical scenarios are not covered.
- Latency issues on large-scale TKGs are not thoroughly investigated.
- Future work may explore more efficient hybrid retrieval architectures and pretraining strategies specifically tailored for medical TKGs.

## Related Work & Insights

- Compared to the STaRK series, RiTeK provides richer topological structures and higher text coverage in the medical domain.
- TOG performs well in few-shot settings, indicating that beam search over knowledge graphs combined with few-shot demonstrations is a promising direction.
- The relative success of GNN-RAG highlights the advantage of GNNs in processing structured path information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Fills a dataset gap in the medical TKG domain, though the methodology is primarily centered on dataset construction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic evaluation of 11 retrieval methods, including human evaluation and multi-dataset comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and the dataset construction pipeline is detailed, though some tables contain redundant information.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/medical_imaging/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)

</div>

<!-- RELATED:END -->
