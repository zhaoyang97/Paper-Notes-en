---
title: >-
  [Paper Note] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications
description: >-
  [ACL 2025][Medical LLM][RAG] This paper proposes MedOmniKB, a multi-source medical knowledge base, and the Source Planning Optimisation (SPO) method. By enabling an expert model to explore multi-source retrieval plans and training smaller models to learn source alignment, this work significantly enhances multi-source retrieval planning capabilities, allowing a 7B small model to outperform a 72B large model.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "RAG"
  - "medical QA"
  - "source planning"
  - "multi-source retrieval"
  - "knowledge base"
date: 2026-05-08
content_hash: 3568e0d27771546d
---

# Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications

**Conference**: ACL 2025  
**arXiv**: [2501.02460](https://arxiv.org/abs/2501.02460)  
**Code**: [https://github.com/Jack-ZC8/Omni-RAG-Medical](https://github.com/Jack-ZC8/Omni-RAG-Medical)  
**Area**: Medical NLP  
**Keywords**: RAG, medical QA, source planning, multi-source retrieval, knowledge base

## TL;DR

This paper proposes MedOmniKB, a multi-source medical knowledge base, and the Source Planning Optimisation (SPO) method. By enabling an expert model to explore multi-source retrieval plans and training smaller models to learn source alignment, this work significantly enhances multi-source retrieval planning capabilities, allowing a 7B small model to outperform a 72B large model.

## Background & Motivation

**Background**: Large language models show great promise in medical reasoning and clinical decision-making tasks, but their internal knowledge remains limited, leading to hallucinations. Retrieval-Augmented Generation (RAG) mitigates this issue by integrating external knowledge, serving as an important paradigm in medical QA.

**Limitations of Prior Work**: Existing medical RAG methods typically unify and process all knowledge sources in a pool, querying them using raw questions directly, without custom retrieval strategies tailored to different source characteristics. Although subsequent approaches introduce prompting or reflection techniques to guide LLMs in using multiple sources, the mismatch between the models' source expectations and actual contents persists due to a lack of genuine content awareness.

**Key Challenge**: The core difficulty of multi-source retrieval is "source planning" — that is, how to generate tailored queries for each knowledge source that match its unique attributes, instead of uniformly using the same query across all sources. Existing methods either neglect this issue or fail to plan effectively due to insufficient source awareness.

**Goal**: 1) Address the lack of sufficiently rich and diverse medical knowledge bases to support source planning research; 2) Teach models how to automatically construct optimal query plans customized for different knowledge sources.

**Key Insight**: Medical questions require obtaining information from diverse source types, such as textbooks, guidelines, research papers, wikis, and structured knowledge graphs, each featuring distinct structures and retrieval methods. Permitting LLMs to explore various retrieval plans and evaluate their effectiveness generates high-quality training data, enabling small models to learn efficient source planning.

**Core Idea**: Build a multi-genre, multi-structured medical knowledge base, MedOmniKB, and train small models via a three-stage SPO (exploration-judging-learning) framework to achieve highly efficient multi-source retrieval planning.

## Method

### Overall Architecture

Given a medical question $x$, the system retrieves relevant documents from five knowledge sources $K = \{K^i\}$ (Book / Guideline / Research / Wiki / Graph). A Reader model then reasons over these documents to answer the question. The core lies in training a Planner model $\mathcal{M}_\theta$, which generates customized queries for each knowledge source, producing a comprehensive source plan $P$.

### Key Designs

1. **MedOmniKB Multi-source Medical Knowledge Base**:

    - **Function**: Provides a rich and diverse foundation of multi-genre, multi-structured medical knowledge for retrieval.
    - **Mechanism**: Integrates five representative knowledge sources: Book (27.7k documents covering medical textbooks), Guideline (45.7k clinical guidelines), Research (25.3M PubMed abstracts), Wiki (6.4M Wikipedia entries), and Graph (UMLS+DrugBank structured knowledge graph with 1.7M concepts and 2.9M relations). Text-based sources are chunked to $\le 1000$ characters, encoded with MedCPT, and stored in Qdrant. For the network graph source, concepts are queried to extract definitions and one-hop relations, followed by reranking.
    - **Design Motivation**: Prior medical knowledge bases were either limited in scale or lacked structured knowledge graphs, hindering the systematic study of source planning problems.

2. **Planning Exploration + Planning Judging**:

    - **Function**: Automatically annotates positive and negative queries to build high-quality training pairs.
    - **Mechanism**: Utilizes a Qwen2.5-72B expert model to generate six candidate queries for each source (abiding by the principles of intra-source diversity and cross-source alignment). Once documents are retrieved, the same expert model serves as an LLM-as-a-judge to determine whether the returned text supports answering the question correctly, indexing positive queries as $q^{i,+}$ and negative queries as $q^{i,-}$.
    - **Design Motivation**: Utilizing an LLM-as-a-judge delivers precise training signals, evaluating search relevance more accurately than downstream accuracy or rerank scores.

3. **Planning Learning (SFT + DPO Two-stage Learning)**:

    - **Function**: Trains the smaller model to master source planning.
    - **Mechanism**: Selects up to three positive queries per source to form a positive plan $P^+$, using SFT to optimize $\mathcal{L}_{\text{SFT}} = -\mathbb{E} \log \mathcal{M}_\theta(P^+ | x)$. Subsequently, preferred-rejected plan pairs are leveraged in DPO for alignment: $\mathcal{L}_{\text{DPO}} = -\mathbb{E} \log \sigma(r_\theta(x, P^+) - r_\theta(x, P^-))$, where $r_\theta$ denots the implicit reward derived from the SFT model.
    - **Design Motivation**: SFT builds the foundational capability of plan generation, whereas DPO aligns queries closely with multi-source properties. Experiments show that skipping SFT and training DPO directly yields poor results.

### Loss & Training

- **SFT Stage**: Standard cross-entropy loss, training the model to output positive plans.
- **DPO Stage**: Preference alignment loss trained on positive-negative plan pairs, with $\beta$ controlling the strength of the KL-divergence constraint.
- **Training Base**: Qwen2.5-7B-Instruct.
- **Data Filtering**: Samples without any positive queries are filtered out; a maximum of three positive queries per source are retained to control context length.

## Key Experimental Results

### Main Results

| Reader | Planner/Method | MedQA | MedMCQA | MMLU-Med | PubMedQA | Avg. |
|--------|-------------|-------|---------|----------|----------|------|
| Qwen2.5-7B | No Retrieval | 60.80 | 56.17 | 76.95 | 34.60 | 56.95 |
| Qwen2.5-7B | Original Question | 62.45 | 63.25 | 80.90 | 47.00 | 62.71 |
| Qwen2.5-7B | Prompting (72B) | 72.11 | 65.33 | 81.73 | 53.80 | 66.30 |
| Qwen2.5-7B | SeRTS (72B) | 70.70 | 66.83 | 82.55 | 55.60 | 67.06 |
| Qwen2.5-7B | **SPO (7B)** | **76.98** | **71.08** | **85.49** | **60.20** | **70.93** |

### Ablation Study

| Configuration | MedQA | PubMedQA | SEER | Description |
|------|-------|----------|------|------|
| Full (SFT+DPO) | 76.98 | 60.20 | 61.90 | Full model |
| SFT only | 74.08 | 59.20 | 58.50 | Performance drops by ~3% without DPO |
| DPO only | 67.48 | 55.80 | 54.30 | Skipping SFT and training DPO directly yields poor results |
| Frozen (7B) | 64.10 | 53.20 | 53.50 | Untrained baseline |
| - Book | 70.38 | 52.60 | 56.80 | Performance drops by 8.57% without Book source |
| - Guideline | 72.35 | 56.40 | 52.10 | Performance drops by 15.83% on SEER without Guideline source |
| - Research | 71.72 | 35.40 | 51.20 | Performance drops by 41.2% on PubMedQA without Research source |

### Key Findings

- Under nearly all settings, the SPO-trained 7B model outperforms the frozen 72B planners (e.g., Prompting, Reflexion, SeRTS), proving the extreme efficacy of targeted alignment.
- SFT is a crucial step: direct DPO training provides limited improvements (67.48 vs. 64.10), whereas SFT boosts performance significantly (74.08 vs. 64.10), upon which DPO further aligns and fine-tunes.
- Distinct knowledge sources show highly skewed contributions across tasks: the Research source is critical for PubMedQA (dropping from 60.20 to 35.40 upon omission), while Guideline is paramount for SEER.
- The model exhibits good generalization to unseen sources (OOD corpus trials), indicating that SPO-derived source awareness transfers effectively.

## Highlights & Insights

- **The elegant "exploration-judging-learning" paradigm**: The large model explores and judges, and the small model learns, achieving knowledge distillation for source planning. This methodology is applicable to any scenario involving multi-source retrieval.
- **MedOmniKB's multi-source architecture**: Aligning structured knowledge graphs (UMLS+DrugBank) and unstructured documents in a unified repository represents a very comprehensive knowledge environment in medical RAG.
- **The "7B outperforming 72B" conclusion**: This provides highly practical value, underscoring that for domain-specific tasks, fine-tuning smaller models systematically is more efficient than basic prompt engineering on larger models.

## Limitations & Future Work

- MedOmniKB does not yet cover all types of medical resources (e.g., clinical images, EHR narratives, etc.), leaving space for expansion.
- The exploration and judging phases during SPO training depend heavily on massive 72B model inference, which incurs high costs.
- Evaluation remains limited to multiple-choice benchmarks and professional ratings, lacking assessments of user satisfaction and treatment outcomes in real clinical setups.
- The contribution of the Graph resource (UMLS) is relatively minor, potentially because knowledge graph queries rely heavily on exact entity matching.

## Related Work & Insights

- **vs MedRAG**: MedRAG uniformly searches all sources without any planning. SPO introduces differentiated source planning, yielding an approximate 14% improvement on MedQA.
- **vs Reflexion/SeRTS**: Contrastingly, reflective techniques are constrained by models' native self-correction capacities under multi-source environments; SPO circumvents this restriction through explicit training signals.
- **vs RaFe Planning**: RaFe adopts reranking scores as training objectives, which may lack precision. In contrast, SPO's LLM-as-a-judge provides more accurate estimates regarding whether retrieved documents support final answers.

## Rating

- Novelty: ⭐⭐⭐⭐ The formulation of source planning and the three-stage SPO paradigm are quite novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 distinct datasets, various Reader structures, comprehensive ablation, and OOD evaluations.
- Writing Quality: ⭐⭐⭐⭐ Solid presentation with definitive problem formulations.
- Value: ⭐⭐⭐⭐ Delivers an effective and scalable solution for handling multi-source planning inside medical RAG frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[NeurIPS 2025\] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](../../NeurIPS2025/medical_nlp/demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](../../ACL2026/medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2025\] A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [\[ICML 2025\] On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](../../ICML2025/medical_nlp/on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)

</div>

<!-- RELATED:END -->
