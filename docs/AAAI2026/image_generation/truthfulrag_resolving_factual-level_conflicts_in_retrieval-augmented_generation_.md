---
title: >-
  [Paper Note] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs
description: >-
  [AAAI 2026][Image Generation][knowledge conflict] This paper proposes TruthfulRAG, a framework that, for the first time, leverages knowledge graphs (KGs) to resolve conflicts between retrieved knowledge and LLM parametric knowledge at the factual level in RAG systems. The framework improves generation accuracy and trustworthiness through triple extraction, query-aware graph retrieval, and an entropy-based conflict filtering mechanism.
tags:
  - AAAI 2026
  - Image Generation
  - knowledge conflict
  - retrieval-augmented generation
  - knowledge graph
  - factual-level reasoning
  - entropy filtering
date: 2026-05-08
content_hash: 742ec1c9c167a9d9
---

# TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs

**Conference**: AAAI 2026
**arXiv**: [2511.10375](https://arxiv.org/abs/2511.10375)
**Code**: None
**Area**: image_generation (actually NLP/RAG)
**Keywords**: knowledge conflict, retrieval-augmented generation, knowledge graph, factual-level reasoning, entropy filtering

## TL;DR

This paper proposes TruthfulRAG, a framework that, for the first time, leverages knowledge graphs (KGs) to resolve conflicts between retrieved knowledge and LLM parametric knowledge at the factual level in RAG systems. The framework improves generation accuracy and trustworthiness through triple extraction, query-aware graph retrieval, and an entropy-based conflict filtering mechanism.

## Background & Motivation

1. **Background**: Retrieval-Augmented Generation (RAG) has become the dominant paradigm for enhancing LLM capabilities by incorporating external knowledge to compensate for limitations in parametric knowledge, such as staleness and incompleteness.
2. **Limitations of Prior Work**: As external knowledge bases continue to expand and parametric knowledge grows increasingly outdated, conflicts between retrieved external information and the LLM's internal knowledge become unavoidable, seriously degrading generation quality.
3. **Key Challenge**: Existing conflict resolution methods operate either at the token level (by adjusting output probability distributions) or at the semantic level (through semantic alignment and integration). These coarse-grained strategies rely on fragmented data representations, making it difficult to accurately capture complex factual dependencies and fine-grained factual inconsistencies.
4. **Goal**: To precisely identify and resolve knowledge conflicts in RAG at the factual level.
5. **Key Insight**: Leveraging the structured triple representations of knowledge graphs to construct reliable reasoning paths and enhance LLM confidence in external knowledge.
6. **Core Idea**: Construct a knowledge graph to extract structured triples, perform query-aware graph traversal to obtain relevant reasoning paths, detect and filter conflicting paths via entropy-change signals, and guide the LLM to prioritize accurate external knowledge.

## Method

### Overall Architecture

TruthfulRAG comprises three sequentially connected modules: (1) **Graph Construction**, which extracts structured knowledge triples from retrieved content to build a knowledge graph; (2) **Graph Retrieval**, which identifies reasoning paths highly relevant to the query through query-aware graph traversal; and (3) **Conflict Resolution**, which employs an entropy-based filtering mechanism to detect and resolve factual conflicts between parametric knowledge and external information.

### Key Designs

1. **Graph Construction**:
    - **Function**: Converts unstructured text retrieved by RAG into a structured knowledge graph.
    - **Mechanism**: The retrieved content $C$ is semantically segmented into text spans $\mathcal{S}=\{s_1,...,s_m\}$. An LLM extracts triples $\mathcal{T}_{i,j}=(h,r,t)$ (head entity, relation, tail entity) from each span, which are aggregated to construct the graph $\mathcal{G}=(\mathcal{E},\mathcal{R},\mathcal{T}_{all})$.
    - **Design Motivation**: Structured triple representations filter low-information noise, capture fine-grained factual associations, and provide a semantically rich foundation for subsequent query-aware retrieval.

2. **Graph Retrieval**:
    - **Function**: Retrieves reasoning paths from the knowledge graph that are strongly relevant to the user query.
    - **Mechanism**: Key query elements $\mathcal{K}_q$ are extracted, and semantic similarity matching identifies the top-k important entities $\mathcal{E}_{imp}$ and relations $\mathcal{R}_{imp}$. Starting from the important entities, a two-hop graph traversal collects candidate paths, which are then ranked by a fact-aware scoring function: $\text{Ref}(p)=\alpha \cdot \frac{|e \in p \cap \mathcal{E}_{imp}|}{|\mathcal{E}_{imp}|} + \beta \cdot \frac{|r \in p \cap \mathcal{R}_{imp}|}{|\mathcal{R}_{imp}|}$
    - **Design Motivation**: Query-driven traversal ensures that the retrieved knowledge is factually aligned with the query, avoiding the introduction of irrelevant information.

3. **Conflict Resolution**:
    - **Function**: Detects and filters reasoning paths that exhibit factual conflicts with the LLM's parametric knowledge.
    - **Mechanism**: The output entropy under purely parametric generation and under augmented generation (with a reasoning path) are computed as $H(P_{param}(ans|q))$ and $H(P_{aug}(ans|q,p))$, respectively. The entropy shift is defined as $\Delta H_p = H(P_{aug}) - H(P_{param})$. When $\Delta H_p > \tau$, the path is identified as challenging the LLM's parametric knowledge (potentially correcting its errors) and is marked as a *corrective path*.
    - **Design Motivation**: A positive entropy shift indicates that external knowledge increases model uncertainty, likely because it conflicts with erroneous parametric knowledge. These corrective paths are precisely the ones capable of rectifying the LLM's internal misconceptions.

### Loss & Training

TruthfulRAG is an inference-time framework and requires no additional training. Key hyperparameters include:
- Entropy threshold $\tau$: $\tau=1$ for GPT-4o-mini and Mistral; $\tau=3$ for Qwen2.5.
- All Top-K values are set to 10.
- Temperature is set to 0 to ensure reproducibility.

## Key Experimental Results

### Main Results

| Dataset | Metric | TruthfulRAG (GPT-4o-mini) | FaithfulRAG | Standard RAG | Gain (vs RAG) |
|--------|------|--------------------------|-------------|------------|------|
| FaithEval | ACC | 69.5 | 67.2 | 61.3 | +8.2 |
| MuSiQue | ACC | 79.4 | 79.3 | 72.6 | +6.8 |
| RealtimeQA | ACC | 85.0 | 78.8 | 67.3 | +17.7 |
| SQuAD | ACC | 81.1 | 80.8 | 73.1 | +8.0 |

Results with Mistral-7B are even more pronounced, achieving an average ACC of 81.3 (Imp 66.1), significantly outperforming all baselines.

### Ablation Study

| Configuration | FaithEval (ACC/CPR) | MuSiQue (ACC/CPR) | Notes |
|------|---------|---------|------|
| Standard RAG | 61.3/0.51 | 72.6/1.86 | Baseline |
| w/o Knowledge Graph | 64.8/0.52 | 78.9/1.15 | CPR drops; harder to extract precisely |
| w/o Conflict Resolution | 69.3/0.59 | 77.8/2.79 | High CPR but limited accuracy |
| Full Method | 69.5/0.56 | 79.4/2.25 | Both modules cooperate optimally |

### Key Findings

- **Structured reasoning paths vs. natural language context**: Structured paths yield higher logprob values across all datasets, indicating stronger LLM confidence in KG-based representations.
- **Performance in non-conflicting contexts**: The method also outperforms baselines on MuSiQue-golden (93.2) and SQuAD-golden (98.3), demonstrating its generalizability.
- The KRE method suffers a severe performance drop (−45.8) in non-conflicting scenarios, whereas TruthfulRAG remains stable across both settings.

## Highlights & Insights

- **Factual-level conflict resolution**: Compared to token- or semantic-level approaches, KG triples provide more precise knowledge alignment and conflict detection.
- **Entropy shift as a conflict signal**: This constitutes an elegant unsupervised conflict detection mechanism—if external knowledge increases model uncertainty, it is precisely because it challenges (and may correct) the model's erroneous parametric beliefs.
- **Structured representations enhance model confidence**: Experiments reveal that converting unstructured text into KG triple representations substantially increases the LLM's trust in external knowledge, which is a noteworthy finding.
- **Plug-and-play framework**: No training is required; the framework can be directly integrated into existing RAG systems.

## Limitations & Future Work

- KG construction relies on the LLM's own triple extraction capability; inaccurate extraction may introduce new errors.
- Entropy-based detection requires access to the LLM's token probability distributions, which may be restricted for some closed-source APIs.
- The fixed two-hop graph traversal depth may not be suitable for all scenarios; multi-hop reasoning tasks may require deeper traversal.
- Computational overhead is non-trivial, as multiple LLM calls are required (triple extraction, parametric generation, and augmented generation), leading to increased latency.

## Related Work & Insights

- **vs. FaithfulRAG**: FaithfulRAG performs semantic-level integration via a self-reflection mechanism, whereas TruthfulRAG detects conflicts at the factual level through KG-structured reasoning, achieving greater precision.
- **vs. KRE**: KRE relies on prompt optimization strategies and suffers a significant performance drop in non-conflicting scenarios, while TruthfulRAG maintains stable performance in both settings.
- **vs. COIECD**: COIECD steers the LLM toward external knowledge by modifying the decoding strategy, but lacks precise localization of conflicts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce KGs into RAG conflict resolution; factual-level granularity represents a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers four datasets, three LLMs, multiple baselines, ablation studies, and confidence analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, method description is comprehensive, and figures are informative.
- **Value**: ⭐⭐⭐⭐ — Knowledge conflicts in RAG are a critical challenge in practical deployment; the method offers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Improved Masked Image Generation with Knowledge-Augmented Token Representations](improved_masked_image_generation_with_knowledge-augmented_token_representations.md)
- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](../../NeurIPS2025/image_generation/can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/image_generation/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](../../NeurIPS2025/image_generation/imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)

<!-- RELATED:END -->
