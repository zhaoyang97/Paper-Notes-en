---
title: >-
  [Paper Note] Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG
description: >-
  [AAAI 2026][Graph Learning][GraphRAG] This paper proposes a paradigm shift in GraphRAG from "build-then-reason" to "reason-and-construct," introducing the Relink framework that dynamically constructs query-specific evidence graphs—combining a high-precision KG backbone with a high-recall latent relation pool, unified via a query-driven ranker to assess relevance on demand, complete missing paths, and filter distractor facts—achieving average gains of 5.4% EM and 5.2% F1 across 5 multi-hop QA benchmarks.
tags:
  - AAAI 2026
  - Graph Learning
  - GraphRAG
  - Dynamic Knowledge Graph
  - Multi-hop Reasoning
  - Evidence Graph Construction
  - Query-Driven Retrieval
date: 2026-05-08
content_hash: 7fba82100cdc97db
---

# Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG

**Conference**: AAAI 2026
**arXiv**: [2601.07192](https://arxiv.org/abs/2601.07192)
**Code**: [GitHub](https://github.com/DMiC-Lab-HFUT/Relink)
**Area**: NLP Understanding / Knowledge Graphs
**Keywords**: GraphRAG, Dynamic Knowledge Graph, Multi-hop Reasoning, Evidence Graph Construction, Query-Driven Retrieval

## TL;DR
This paper proposes a paradigm shift in GraphRAG from "build-then-reason" to "reason-and-construct," introducing the Relink framework that dynamically constructs query-specific evidence graphs—combining a high-precision KG backbone with a high-recall latent relation pool, unified via a query-driven ranker to assess relevance on demand, complete missing paths, and filter distractor facts—achieving average gains of 5.4% EM and 5.2% F1 across 5 multi-hop QA benchmarks.

## Background & Motivation

**Background**: GraphRAG has emerged as a key approach to mitigate hallucination by leveraging knowledge graph (KG) structure to enhance LLM multi-hop reasoning. All existing GraphRAG methods follow a "build-then-reason" paradigm—first constructing a static KG, then performing reasoning over it.

**Limitations of Prior Work**:
- **Inherent KG incompleteness**: Static KGs are naturally incomplete due to knowledge evolution and extraction errors, leading to broken reasoning chains. Existing KG completion methods perform "global completion," which often fails to provide the "local" facts required by specific queries.
- **Low signal-to-noise ratio / distractor facts**: General-purpose KGs contain numerous facts that are topically related to a query but unhelpful for answering it (distractor facts). For example, when asking "where was someone buried," the relation "died in" is highly relevant but not the required "buried in."

**Key Challenge**: A static KG designed to serve all possible queries inevitably suffers from both incompleteness and noise.

**Goal**: Achieve query-driven dynamic evidence graph construction that simultaneously addresses KG incompleteness and distractor facts.

**Key Insight**: Transform the paradigm from "build-then-reason" to "reason-and-construct"—rather than navigating a static graph, construct a compact evidence graph in real time according to query demands.

**Core Idea**: Dynamically construct query-specific evidence graphs via heterogeneous knowledge sources (high-precision KG + high-recall latent relation pool) unified by a query-driven ranker.

## Method

### Overall Architecture
Relink consists of three core components: (1) heterogeneous knowledge source construction—a high-precision factual KG $\mathcal{G}_b$ and a high-recall latent relation pool $\mathcal{R}_c$; (2) query-driven dynamic path exploration—beam search + coarse-grained ranking + LLM-based fine-grained reranking + on-demand instantiation; (3) evidence-based answer generation.

### Key Designs

1. **Heterogeneous Knowledge Sources**:

    - **Function**: Fuse two complementary knowledge sources with high precision and high recall to ensure coverage.
    - **Mechanism**: $\mathcal{G}_b$ is a high-confidence factual KG extracted by an LLM from a corpus—high precision but incomplete. $\mathcal{R}_c$ is a latent relation pool based on entity co-occurrence, filtered by PMI to retain meaningful co-occurring pairs: $\text{PMI}(e_i, e_j) > \tau_c$, with relation representations $\mathbf{r}_{ij}$ obtained by encoding co-occurrence context sentences with a pre-trained language model.
    - **Design Motivation**: The KG provides a reliable skeleton (high precision → low noise), while the latent relation pool provides raw material for repairing broken paths (high recall → high coverage). The two are complementary.

2. **Unified Semantic Space + Query-Driven Ranking**:

    - **Function**: Compare and rank candidate edges from both the KG and the latent relation pool within a unified space.
    - **Mechanism**: KG triples are encoded into vectors $\mathbf{v}_f$ via $\text{Encoder}_F$; latent relations have precomputed representations $\mathbf{v}_l$. The two encoders are aligned into a unified space via a contrastive loss $\mathcal{L}_{\text{contra}}$ (InfoNCE). The query-driven Ranker is trained with a pairwise ranking loss $\mathcal{L}_{\text{rank}}$ to discriminate whether a given edge is useful for answering a specific query.
    - **Design Motivation**: The ranking criterion is not "is this fact related to the query" but "does this fact help answer the query"—this is the key to filtering distractor facts. General semantic similarity cannot distinguish "relevant" from "useful."

3. **Dynamic Path Exploration and Repair**:

    - **Function**: Iterative beam search to construct reasoning paths with on-demand instantiation of latent relations.
    - **Mechanism**: Starting from topic entities in the query, each step expands all one-hop neighbors (from $\mathcal{G}_b$ and $\mathcal{R}_c$) → coarse ranking (lightweight Ranker for fast filtering) → fine ranking (LLM evaluates semantic contribution) → retain top-K paths. When a selected path contains a latent relation $\mathbf{r}_{ij}$, an LLM combines the source context and query to generate a concrete triple (dynamic instantiation).
    - **Design Motivation**: The two-stage coarse-to-fine ranking balances efficiency and accuracy. Query-aware instantiation ensures newly constructed facts align with user intent rather than producing generic relations.

### Loss & Training
- **Ranking Loss** $\mathcal{L}_{\text{rank}}$: Pairwise margin loss to train the Ranker to distinguish high-quality paths from low-quality ones.
- **Contrastive Alignment Loss** $\mathcal{L}_{\text{contra}}$: InfoNCE to align KG facts and latent relations in a unified space.
- **Alternating Training**: Freeze encoders and train Ranker for one epoch → freeze Ranker and train encoders for one epoch → repeat until convergence.

## Key Experimental Results

### Main Results

Performance on 5 multi-hop QA benchmarks:

| Method | 2Wiki EM | 2Wiki F1 | HotpotQA EM | HotpotQA F1 | MuSiQue-Full EM |
|--------|----------|----------|-------------|-------------|-----------------|
| GPT-4o | 0.292 | 0.358 | 0.330 | 0.424 | 0.106 |
| HippoRAG | 0.578 | 0.684 | 0.498 | 0.647 | 0.190 |
| GraphRAG | 0.318 | 0.379 | 0.450 | 0.569 | 0.138 |
| **Relink** | **0.628** | **0.722** | **0.558** | **0.704** | **0.252** |

Relative gains vs. HippoRAG: EM +8.7% on 2Wiki, +12.0% on HotpotQA, +32.6% on MuSiQue-Full.

### Ablation Study

| Configuration | 2Wiki EM | HotpotQA EM | Note |
|---------------|----------|-------------|------|
| Full Model | **0.628** | **0.558** | Complete Relink |
| w/o $\mathcal{G}_b$ | 0.582 | 0.486 | Remove KG backbone; EM drops 12.9% (HotpotQA) |
| w/o $\mathcal{R}_c$ | 0.616 | 0.526 | Remove latent relation pool; EM drops 5.7% |
| w/o Query-Driven Ranker | 0.552 | 0.450 | Replace with general cosine similarity; EM drops 19.4% |
| w/o $\mathcal{L}_{\text{contra}}$ | 0.603 | 0.518 | Remove contrastive alignment loss; EM drops 7.2% |

### Key Findings
- **Query-driven Ranker contributes most**: Removing it drops EM by 19.4% on HotpotQA, demonstrating that distinguishing "useful" from "relevant" is the core capability.
- **Dynamic repair remains robust under extreme sparsity**: After removing 90% of KG edges, Relink's F1 drops only from 0.722 to 0.669 (−7.3%), while static methods drop 34.7%.
- **KG backbone is more critical than the latent relation pool**: Removing the KG causes a larger drop than removing the latent relation pool (12.9% vs. 5.7%), confirming that a high-precision skeleton is the foundation for reliable reasoning.
- **Unified semantic space is indispensable**: Without contrastive alignment, the Ranker cannot compare across sources, resulting in a 7.2% EM drop.

## Highlights & Insights
- **Systematic articulation of the paradigm shift**: Rather than an incremental method improvement, this work presents a paradigm-level innovation from "build-then-reason" to "reason-and-construct." The motivation systematically analyzes the fundamental flaws of the static paradigm from two angles: incompleteness and distractor facts.
- **PMI + context encoding for the latent relation pool**: Combining entity co-occurrence, PMI filtering, and contextual sentence encoding to construct a high-recall candidate relation repository at minimal cost—a technique transferable to other knowledge-completion scenarios.
- **Sparsity robustness experiments**: Maintaining strong performance after removing 90% of KG edges provides the most compelling empirical validation of the "reason-and-construct" paradigm.

## Limitations & Future Work
- **LLM call overhead**: Each exploration step requires LLM-based fine-grained reranking and dynamic instantiation, resulting in a large number of LLM calls during multi-hop reasoning.
- **Dependence on the underlying LLM**: All methods use DeepSeek-V3 as the backbone; whether Relink's improvements generalize to other LLMs remains to be verified.
- **Only 500 sampled questions**: Each dataset is evaluated on only 500 questions to reduce computational cost; full-scale evaluation may yield different conclusions.
- **Latent relation pool quality depends on the PMI threshold**: The choice of $\tau_c$ governs the recall-precision trade-off, and different domains may require different settings.

## Related Work & Insights
- **vs. HippoRAG**: HippoRAG is a representative hybrid RAG approach combining graph and text retrieval but still relies on a static graph. Relink outperforms it across all datasets, with an EM improvement of 32.6% on the most challenging MuSiQue-Full.
- **vs. GraphRAG (Edge et al. 2024)**: Microsoft's GraphRAG enhances retrieval via community summaries but remains fundamentally static. Relink directly addresses this limitation through dynamic construction.
- **vs. KG completion methods**: Traditional KGC performs global completion without guaranteeing that completed facts are useful for specific queries. Relink performs query-driven, on-demand completion, yielding greater precision.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "reason-and-construct" paradigm shift, unified ranking over heterogeneous knowledge sources, and query-driven dynamic instantiation are all original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, 9 baselines, detailed ablations, sparsity robustness tests, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ RQ-driven experimental organization is clear and well-structured; Figures 1/4 provide intuitive and compelling paradigm comparisons.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the GraphRAG field; the dynamic construction paradigm is broadly applicable to RAG systems.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[NeurIPS 2025\] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation](../../NeurIPS2025/graph_learning/spot-trip_dual-preference_driven_out-of-town_trip_recommendation.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)

<!-- RELATED:END -->
