---
title: >-
  [Paper Note] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling
description: >-
  [ICML 2026][Information Retrieval & RAG][Hypergraph Memory] This paper reformulates working memory in multi-step RAG from a "flat list of facts" into a **hypergraph**. Each hyperedge serves as a memory point that can be…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Hypergraph Memory"
  - "Multi-step RAG"
  - "High-order Relational Modeling"
  - "Global Sense-making"
  - "Long Document Understanding"
date: 2026-05-08
content_hash: a9edd12b253d0e2b
---

# HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling

**Conference**: ICML 2026  
**arXiv**: [2512.23959](https://arxiv.org/abs/2512.23959)  
**Code**: https://github.com/Encyclomen/HGMem (Available)  
**Area**: Information Retrieval / RAG / Working Memory / Long-Context Reasoning  
**Keywords**: Hypergraph Memory, Multi-step RAG, High-order Relational Modeling, Global Sense-making, Long Document Understanding

## TL;DR
This paper reformulates working memory in multi-step RAG from a "flat list of facts" into a **hypergraph**. Each hyperedge serves as a memory point that can be updated, inserted, or merged. Leveraging the inherent capability of hyperedges to connect $n \geq 2$ entities, memory can continuously merge low-order facts into high-order concepts during interactions, significantly improving long-context QA performance that requires "global sense-making."

## Background & Motivation

**Background**: When dealing with complex QA for long documents of 100k+ tokens, single-step RAG is insufficient. The mainstream approach is multi-step RAG (IRCOT, ReAct, DeepRAG, ComoRAG, etc.), which alternates between retrieval and reasoning while maintaining a working memory to store intermediate states. Memory forms have evolved from unstructured natural language descriptions to structured relational tables, knowledge graphs (KGs), and event logs.

**Limitations of Prior Work**: Existing memories are treated as **static fact repositories** that only append primitive facts. However, human working memory actively **reorganizes scattered facts into higher-order concepts** (Baddeley 2000; Oberauer 2019). This "reorganization capability" is critical for global sense-making tasks, where complex latent connections among events scattered across segments must be aggregated into a unified perspective.

**Key Challenge**: The "edges" in structured memories like KGs or event logs are inherently **binary relations**, which cannot directly express $n$-ary relations where multiple facts constitute a comprehensive proposition. While unstructured descriptions are flexible, they lack traceability to the original text and precision for cross-step operations. There is a systemic trade-off between expressiveness, operability, and the gap between low-order facts and high-order concepts.

**Goal**: Design a working memory that simultaneously satisfies: (1) precise traceability to original chunks; (2) precise cross-step operations (update/insert/merge); (3) direct expression of $n$-ary high-order relations; (4) ability to drive retrieval strategies to switch adaptively between "local investigation" and "global exploration."

**Key Insight**: Hypergraphs naturally generalize "edges" to "hyperedges"—a single hyperedge can connect any number of vertices. This allows complex events, which would require multiple binary edges in a KG, to be collapsed into a single $n$-ary unit. By treating hyperedges as memory points, the hypergraph topology naturally carries the semantics of how facts are organized into concepts.

**Core Idea**: Replace KGs with a **hypergraph as the working memory** for multi-step RAG. Memory points (hyperedges) evolve dynamically through update, insertion, and **merging** operations, with merging being the key step for explicitly constructing high-order relations. Simultaneously, an adaptive "local investigation vs. global exploration" retrieval strategy is introduced, allowing the hypergraph to both store information and guide retrieval paths.

## Method

### Overall Architecture

Input: Long document $\mathcal{D}$ and target query $\hat{q}$.
The preprocessing stage offline partitions $\mathcal{D}$ into 200-token chunks (50-token overlap). Entities and binary relations are extracted using GPT-4o + LightRAG tools to form an entity graph $\mathcal{G}=(\mathcal{V}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}})$. All entities, relations, and chunks are encoded into a vector database using bge-m3.

The online stage maintains a **hypergraph memory** $\mathcal{M}=(\mathcal{V}_{\mathcal{M}}, \tilde{\mathcal{E}}_{\mathcal{M}})$, sharing vertices with $\mathcal{G}$ ($\mathcal{V}_{\mathcal{M}}\subseteq \mathcal{V}_{\mathcal{G}}$), but its hyperedges $\tilde{e}_j=(\Omega^{rel}_{\tilde{e}_j}, \mathcal{V}_{\tilde{e}_j})$ can connect $n \geq 2$ vertices. Each hyperedge is a "memory point" carrying a description from a specific perspective.

At each interaction step $t$:

1. The LLM determines if $\mathcal{M}^{(t)}$ is sufficient to answer $\hat{q}$, generating the answer if so.
2. Otherwise, the LLM analyzes $\mathcal{M}^{(t)}$ to generate sub-queries $\mathcal{Q}^{(t)}$, each automatically assigned to "local investigation" or "global exploration" modes.
3. Vector retrieval is performed on the corresponding scope to obtain candidate entities $\mathcal{V}_{\mathcal{Q}^{(t)}}$. Adjacent relations $\mathcal{E}(\mathcal{V}_{\mathcal{Q}^{(t)}})$ and original chunks $\mathcal{D}(\mathcal{V}_{\mathcal{Q}^{(t)}})$ are pulled via the graph index.
4. The LLM evolves $\mathcal{M}^{(t)}$ into $\mathcal{M}^{(t+1)}$: $\mathcal{M}^{(t+1)}\leftarrow \mathrm{LLM}(\mathcal{M}^{(t)}, \mathcal{V}_{\mathcal{Q}^{(t)}}, \mathcal{E}(\mathcal{V}_{\mathcal{Q}^{(t)}}), \mathcal{D}(\mathcal{V}_{\mathcal{Q}^{(t)}}))$.
5. Upon reaching the maximum step limit or memory sufficiency, all hyperedge descriptions and corresponding chunks are fed to the LLM for the final answer.

At step $t=0$, let $\mathcal{Q}^{(0)}=\{\hat{q}\}$.

### Key Designs

1. **Hypergraph Memory Storage (Each hyperedge = one addressable memory point)**:
    - **Function**: Replaces knowledge graphs/event logs as working memory, allowing each hyperedge to connect $n\geq 2$ entities to natively express "comprehensive propositions formed by multiple facts."
    - **Mechanism**: Vertices $v_i=(\Omega^{ent}_{v_i}, \mathcal{D}_{v_i})$ bind entity descriptions to source chunks; hyperedges $\tilde{e}_j=(\Omega^{rel}_{\tilde{e}_j}, \mathcal{V}_{\tilde{e}_j})$ bind relational descriptions to connected vertex sets. Each hyperedge has an independent embedding in the vector database for direct retrieval; since vertices must exist in $\mathcal{G}$, hyperedges can trace back to original chunks, solving the traceability issue of unstructured memory.
    - **Design Motivation**: In a KG, describing "the respective roles of three characters in an event" requires splitting into multiple binary edges, losing holistic semantics. A hypergraph compresses this into an atomic memory unit that preserves $n$-ary semantics and facilitates future referencing, modification, or merging.

2. **Adaptive Memory-Driven Retrieval (Local Investigation + Global Exploration)**:
    - **Function**: Selects different retrieval scopes for each sub-query based on the LLM's diagnosis of current memory, avoiding over-fixation on existing clues or aimless diffusion.
    - **Mechanism**: If a sub-query $q$ aims to deepen an existing hyperedge $\tilde{e}_j$ (**Local Investigation**), the neighborhood of vertices connected to $\tilde{e}_j$ is used as the candidate set: $\mathcal{N}(\mathcal{V}_{\tilde{e}_j})=\bigcup_{v\in\mathcal{V}_{\tilde{e}_j}}(\mathcal{N}_{\mathcal{M}^{(t)}}(v)\cup \mathcal{N}_{\mathcal{G}}(v))$. If the sub-query seeks new perspectives (**Global Exploration**), the entity set outside memory $\mathcal{C}(\mathcal{M}^{(t)})=\mathcal{V}_{\mathcal{G}}-\mathcal{V}_{\mathcal{M}^{(t)}}$ is used.
    - **Design Motivation**: The hypergraph acts as a retrieval scaffold—the LLM can judge "untouched areas" or "areas worth deepening" from the hyperedge set. This decoupled yet linked design between memory and retrieval fits multi-step reasoning needs better than fixed top-k retrieval.

3. **Three Types of Memory Evolution Operations (Update / Insertion / Merging)**:
    - **Function**: Enables active reorganization of hypergraph memory after each retrieval step, particularly through merging to combine low-order memory points into high-order ones.
    - **Mechanism**: After retrieving information, the LLM executes three operations. **Update** revises descriptions of existing hyperedges (e.g., correcting understanding with new evidence); **Insertion** adds newly discovered facts as new hyperedges; **Merging** scans hyperedges after update/insertion to combine those that are "semantically/logically a whole": $\Omega^{rel}_{\tilde{e}_k}\leftarrow \mathrm{LLM}(\Omega^{rel}_{\tilde{e}_i}, \Omega^{rel}_{\tilde{e}_j}, \hat{q})$. Merging is anchored by $\hat{q}$ to avoid random aggregation.
    - **Design Motivation**: This is the key step in turning memory from a "passive repository" into an "active cognitive structure." Without merging, memory remains at the level of primitive facts. The ablation study confirms that merging is significantly more critical than update.

### Loss & Training

No training is involved. HGMem is a pure prompt-engineering framework. All decisions (sub-query mode, update/insertion/merging, which hyperedges to merge) are performed by the LLM guided by prompts. The backbone used is GPT-4o or Qwen2.5-32B-Instruct (temperature 0.8, max tokens 2048). Hypergraph management uses the `hypergraph-db` package, with hyperedges and chunks encoded by `bge-m3`.

## Key Experimental Results

### Main Results

Comparison across 4 long-context sense-making tasks: Longbench V2 subset (Generative QA, rated by GPT-4o on Comprehensiveness/Diversity 0-100), NarrativeQA, NoCha, Prelude (all reporting Accuracy %).

| Dataset | Metric | Ours HGMem (GPT-4o) | Prev. SOTA | Gain |
|--------|------|--------------------|--------------------|------|
| Longbench | Comprehensiveness | **65.73** | 63.62 (DeepRAG) | +2.11 |
| Longbench | Diversity | **69.74** | 65.98 (DeepRAG) | +3.76 |
| NarrativeQA | Acc (%) | **55.00** | 54.00 (ComoRAG) | +1.00 |
| NoCha | Acc (%) | **73.81** | 72.22 (HippoRAG v2) | +1.59 |
| Prelude | Acc (%) | **62.96** | 61.48 (LightRAG) | +1.48 |

Using Qwen2.5-32B-Instruct, HGMem achieved 51.00 on NarrativeQA / 70.63 on NoCha / 62.22 on Prelude. **The open-source 32B model can match or even exceed several GPT-4o driven baselines**.

### Ablation Study

Backbone fixed to Qwen2.5-32B-Instruct.

| Configuration | NarrativeQA Acc | NoCha Acc | Prelude Acc | Description |
|------|-----------------|-----------|-------------|------|
| HGMem (full) | 51.00 | 70.63 | 62.22 | Full Model |
| w/. GE Only | 47.00 | 68.25 | 59.26 | Global Exploration only, drops 2-4 points |
| w/. LI Only | 43.00 | 63.49 | 60.00 | Local Investigation only, largest drop in NarrativeQA |
| w/o. Update | 50.00 | 68.25 | 60.00 | Without update, drops 1-2 points |
| **w/o. Merging** | **43.00** | **61.11** | **57.78** | **Without merging, drops 8-9 points (most impactful)** |

### Key Findings

- **Merging is the core operation**: Removing it caused a drop of 9.52 on NoCha and 8.00 on NarrativeQA, proving that the "emergence of high-order relations" relies on merging rather than simple fact insertion.
- **Structural gains from merging on sense-making queries**: Analyzing 120 annotated queries (primitive vs. sense-making), the average connected entities per hyperedge (Avg-$N_v$) for sense-making queries jumped from 3.74-4.10 (w/o Merging) to 5.25-7.97, with accuracy rising synchronously. Primitive queries showed almost no change in Avg-$N_v$ and flat accuracy, providing evidence of "on-demand" high-order memory.
- **Optimal at 3 steps**: Peak performance was reached at $t=3$; further steps increased cost without performance gains.
- **Robustness to offline graphs**: Performance remained stable even after randomly removing 50% of entities/relations or using LLM-free Stanford OpenIE for graph construction.

## Highlights & Insights

- **Hypergraph as memory vs. Hypergraph as index**: Unlike HypergraphRAG or PropRAG which use hypergraphs as static offline indices, HGMem treats it as **online dynamic working memory**. This places the expressive power of hypergraphs into the "reasoning scene," allowing the structure to be reshaped per step based on the problem.
- **Merging-as-cognition insight**: Any multi-step agent needing "observation then synthesis" can benefit—storing intermediate states as mergeable discrete units anchored by the target question is more controllable than relying on self-reflection over long histories.
- **Adaptive "local vs. global" retrieval routing**: Many multi-step RAG systems fixate on top-k or expand to the whole database. HGMem uses the "neighborhood of existing hyperedges" for local search and "memory complement" for global search, naturally aligning retrieval costs with cognitive stages.
- **Avg-$N_v$ as a lightweight metric for "depth of understanding"**: Using the average entities per hyperedge as an implicit cognitive complexity metric is a diagnostic tool worth adopting for similar systems.

## Limitations & Future Work

- **Heavy dependency on prompts + strong LLMs**: All judgments rely on zero-shot LLM prompts, which might fail with weaker models. Verification on 7B-level models is missing.
- **Lack of explicit cost/latency data**: Reaching optimal performance at 3 steps implies multiple LLM calls and merging prompts, leading to cost expansion.
- **Redundancy in primitive queries**: For simple facts, the model still tends to aggregate, sometimes introducing unnecessary noise. A more precise "merging trigger" is needed.
- **Static hypergraph logic**: Merging decisions are temporary and not learned across tasks. Future work could integrate GNNs to score merging candidates.
- **Offline graph bottleneck**: Performance still partially depends on the initial offline graph quality. Incremental updates to hypergraph memory for new document content were not discussed.

## Related Work & Insights

- **vs. ComoRAG / DeepRAG**: These also use working memory, but their memories are essentially fact lists or reasoning traces. HGMem's **structural operations (merge/update)** led to a ~10 point lead over ComoRAG on NoCha.
- **vs. ERA-CoT / KnowTrace**: These use "graph memory" but are limited to binary relations. HGMem's native $n$-ary support provides a structural advantage for queries requiring holistic fact synthesis.
- **vs. GraphRAG / LightRAG**: These are single-step graph-enhanced RAG systems using offline indexing (communities, PageRank). HGMem shifts graph memory to online interaction, growing "query-specific" high-order concepts dynamically.
- **Combinatorial potential**: Long-term memory (static hypergraph index) + working memory (dynamic hypergraph) could be a natural future combination point for RAG research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying hypergraphs as working memory is a theoretically sound extension of human cognitive models. The explicit merging operation for high-order relations is a highlight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across 4 datasets, 2 backbones, and 5 baseline categories, including layer-wise analysis of query types and robustness studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation (passive storage → high-order correlation), rigorous notation, and intuitive diagrams. The Avg-$N_v$ metric provides a mechanistic explanation for gains.
- **Value**: ⭐⭐⭐⭐ Provides a specific, reproducible structural memory paradigm for long-context multi-step RAG. The 32B model matching GPT-4o is of significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ACL 2026\] HyperMem: Hypergraph Memory for Long-Term Conversations](../../ACL2026/information_retrieval/hypermem_hypergraph_memory_for_long-term_conversations.md)
- [\[ICML 2026\] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)
- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)

</div>

<!-- RELATED:END -->
