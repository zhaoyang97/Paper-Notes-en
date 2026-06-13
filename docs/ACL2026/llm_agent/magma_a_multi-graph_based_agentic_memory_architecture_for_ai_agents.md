---
title: >-
  [Paper Note] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents
description: >-
  [ACL 2026][LLM Agent][Multi-graph Memory] MAGMA decomposes LLM agent memory into four orthogonal relationship graphs: semantic, temporal, causal…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Multi-graph Memory"
  - "Agentic Memory"
  - "Intent-Aware"
  - "Dual-Stream Writing"
  - "LoCoMo"
date: 2026-05-08
content_hash: ed21ee66fb13294d
---

# MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents

**Conference**: ACL 2026  
**arXiv**: [2601.03236](https://arxiv.org/abs/2601.03236)  
**Code**: https://github.com/FredJiang0324/MAGMA (Available)  
**Area**: LLM Agent / Long-term Memory / Graph Retrieval  
**Keywords**: Multi-graph Memory, Agentic Memory, Intent-Aware, Dual-Stream Writing, LoCoMo

## TL;DR
MAGMA decomposes LLM agent memory into four orthogonal relationship graphs: semantic, temporal, causal, and entity. It employs intent routing and adaptive beam search for policy-guided traversal across appropriate graphs, complemented by a "fast-path synchronous storage + slow-path asynchronous LLM consolidation" dual-stream writing mechanism. MAGMA achieves an overall Judge score of 0.700 on LoCoMo, outperforming A-MEM, Nemori, and MemoryOS, while maintaining a query latency of only 1.47s (40% faster than the runner-up).

## Background & Motivation

**Background**: LLMs are limited by fixed context windows and cannot maintain memory across sessions. This has led to the Memory-Augmented Generation (MAG) paradigm, where external memory $\mathcal{M}_t$ evolves with interactions: $o_t = \mathrm{LLM}(q_t, \mathrm{Retrieve}(q_t, \mathcal{M}_t))$ and $\mathcal{M}_{t+1} = \mathrm{Update}(\mathcal{M}_t, q_t, o_t)$. Representative systems include MemGPT, A-MEM (Zettelkasten-style chains), Nemori (event segmentation), MemoryOS, GraphRAG, and Zep.

**Limitations of Prior Work**: ① Most solutions cram memory into a single repository (vector store or monolithic KG) using cosine similarity for retrieval, causing temporal, causal, and entity relationships to be entangled. ② This "associative proximity" identifies *what* happened but fails to explain *why*, hindering causal chain reasoning. ③ A-MEM's note network relies primarily on semantic embeddings, missing temporal/causal links; Nemori utilizes event segmentation but lacks explicit differentiation of relationship dimensions in its narrative structure. ④ Writing and retrieval are often coupled on a synchronous path, where complex structural reasoning blocks agent responses.

**Key Challenge**: Memory must support both "fast recall" and "deep reasoning"—the synchronous path must be fast to avoid blocking the user, yet structural relationship reasoning requires slow and expensive LLM calls. Furthermore, retrieval must be precise across different query types (why/when/entity), which a single similarity metric cannot achieve.

**Goal**: (a) Replace monolithic storage with a multi-view relationship graph; (b) enable retrieval to dynamically select graph views based on query intent; (c) decouple "fast" writing from "consolidation."

**Key Insight**: Drawing from the Complementary Learning Systems (CLS) theory in cognitive science—fast hippocampus vs. slow cortex—and the "multi-view read path + asynchronous indexing" concept in system design.

**Core Idea**: Four orthogonal relationship graphs + intent-aware policy-guided graph traversal + dual-stream read/write architecture.

## Method

### Overall Architecture

The system consists of three layers: (1) **Query Process** — Intent-Aware Router → Adaptive Topological Retrieval → Context Synthesizer; (2) **Data Structure** — a time-varying directed multigraph $\mathcal{G}_t=(\mathcal{N}_t,\mathcal{E}_t)$, where nodes store events and edges are categorized into four semantic subspaces, integrated with a Vector DB for hybrid retrieval; (3) **Write/Update** — a Fast Path for synchronous storage (event segmentation + vector indexing + temporal backbone updates) and a Slow Path for asynchronous LLM reasoning to derive causal and entity edges. Each event node $n_i = \langle c_i, \tau_i, \mathbf{v}_i, \mathcal{A}_i\rangle$ records content, timestamp, dense vector, and structured attributes.

### Key Designs

1.  **Four Orthogonal Relationship Graphs as Data Structure**:
    - **Function**: Decomposes the previously monolithic memory into four independently accessible relationship views.
    - **Mechanism**: The edge set $\mathcal{E}$ is partitioned into four semantic subspaces: (i) **Temporal Graph** — an immutable chronological chain formed strictly by $\tau_i < \tau_j$; (ii) **Causal Graph** — directed edges derived by the consolidation module when the conditional score $S(n_j\mid n_i, q) > \delta$, supporting "why" queries; (iii) **Semantic Graph** — undirected edges triggered by $\cos(\mathbf{v}_i, \mathbf{v}_j) > \theta_{\mathrm{sim}}$; (iv) **Entity Graph** — connects events to abstract entity nodes to resolve "same object" identification across time periods. The Vector DB is retained as a coarse anchor entry point.
    - **Design Motivation**: "Similar events" retrieved from a purely semantic graph are not necessarily "causal events." The temporal chain ensures chronological correctness, the causal chain answers "why," and the entity chain ensures object permanence—the four complement each other rather than being redundant.

2.  **Intent-Aware Adaptive Traversal Policy**:
    - **Function**: Conducts policy-guided beam search on the appropriate relationship graphs based on query type rather than rigid traversal.
    - **Mechanism**: A lightweight classifier maps the query to an intent $T_q \in \{\mathrm{Why}, \mathrm{When}, \mathrm{Entity}\}$, and a temporal parser resolves phrases like "last Friday" into absolute time windows. Reciprocal Rank Fusion $S_{\mathrm{anchor}} = \mathrm{TopK}\!\sum_{m\in\{vec,key,time\}}\frac{1}{k+r_m(n)}$ identifies entry anchor points. This is followed by a heuristic beam search, where the transition score at each step is $S(n_j\mid n_i, q) = \exp\big(\lambda_1\phi(\mathrm{type}(e_{ij}), T_q) + \lambda_2 \mathrm{sim}(\vec n_j, \vec q)\big)$. Here $\phi(r, T_q) = \mathbf{w}_{T_q}^\top \mathbf{1}_r$ represents the edge type weight corresponding to the intent (e.g., a weight of 3–5 for causal edges in "why" queries). Top-$k$ nodes are selected per layer, with a decay factor $\gamma$ to suppress depth explosion.
    - **Design Motivation**: Single cosine similarity naturally favors "similar topics," causing adversarial queries to be misled by semantically similar but causally irrelevant distractors. Intent routing first selects the graph then performs the beam search, explicitly encoding the human intuition of "deciding what to look for before searching."

3.  **Dual-Stream Writing (Fast Path + Slow Path)**:
    - **Function**: Prevents agent responses from being blocked by structural reasoning while continuously deepening memory structures.
    - **Mechanism**: The **Fast Path** (Algo 2) follows a synchronous route, performing only event segmentation, vector encoding, appending temporal backbone edges $n_{t-1}\to n_t$, indexing in the vector DB, and queuing node IDs—all without LLM reasoning. The **Slow Path** (Algo 3) uses an asynchronous worker to pull nodes from the queue, retrieve a 2-hop neighborhood $\mathcal{N}_{\mathrm{local}}$, use an LLM $\Phi$ to infer implicit causal and entity edges $\mathcal{E}_{\mathrm{new}} = \Phi_{\mathrm{reason}}(\mathcal{N}(n_t), \mathcal{H}_{\mathrm{history}})$, and write them back to the graph.
    - **Design Motivation**: In deployment, the bottleneck is synchronous latency, while causal reasoning can take several seconds. Decoupling ensures user-facing latency only involves vector encoding (~ms), while structural reasoning is completed in the background, analogous to the CLS theory's complementary relationship between the hippocampus (fast) and neocortex (slow).

### Loss & Training
- MAGMA is a **training-free** architecture for retrieval and LLM prompting. All LLM calls use gpt-4o-mini ($T=0$); embeddings use all-MiniLM-L6-v2 (384-dim) or OpenAI text-embedding-3-small (1536-dim). Hyperparameters $\lambda_1=1.0, \lambda_2=0.3$–$0.7$, beam width, $\mathrm{MaxDepth}=5$, and $\mathrm{Budget}=200$ were empirically tuned on LoCoMo.
- Through token budgeting, low-scoring nodes are compressed into summaries like "...3 intermediate events...", while high-scoring nodes retain full text to keep context within the prompt window.

## Key Experimental Results

### Main Results

**LoCoMo (LLM-as-Judge, gpt-4o-mini)**

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial | Overall |
|---|---|---|---|---|---|---|
| Full Context | 0.468 | 0.562 | 0.486 | 0.630 | 0.205 | 0.481 |
| A-MEM | 0.495 | 0.474 | 0.385 | 0.653 | 0.616 | 0.580 |
| MemoryOS | 0.552 | 0.422 | 0.504 | 0.674 | 0.428 | 0.553 |
| Nemori | 0.569 | **0.649** | 0.485 | 0.764 | 0.325 | 0.590 |
| **MAGMA** | **0.528** | **0.650** | **0.517** | **0.776** | **0.742** | **0.700** |

Overall relative improvement of +18.6%~45.5%. The most significant gain is in the Adversarial category (0.742 vs. 0.616), indicating that intent routing + causal edges effectively resist "semantically similar but structurally irrelevant" distractors.

**LongMemEval (Avg. context >100K tokens)**: MAGMA's average accuracy of 61.2% > Nemori 56.2% > Full-context 55.0%. Notably, MAGMA uses only 0.7–4.2K tokens per query compared to 101K for Full-context, achieving **>95% token savings**.

### Ablation Study

| Configuration | Judge | F1 | BLEU-1 |
|---|---|---|---|
| MAGMA (Full) | **0.700** | 0.467 | 0.378 |
| w/o Adaptive Policy | 0.637 | 0.413 | 0.357 |
| w/o Causal Links | 0.644 | 0.439 | 0.354 |
| w/o Temporal Backbone | 0.647 | 0.438 | 0.349 |
| w/o Entity Links | 0.666 | 0.451 | 0.363 |
| Causal Only | 0.590 | – | – |
| Temporal Only | 0.577 | – | – |
| Entity Only | 0.531 | – | – |

The removal of the Adaptive Policy caused the largest drop (-0.063), proving intent routing is core. Causal and temporal links are also critical (each > -0.05 reduction); entity links had the smallest but still positive contribution.

### Key Findings
- The largest improvement occurred in the Adversarial dimension (+12.6 over A-MEM), showing that distractor resistance relies on "structural alignment" rather than "semantic matching"—a blind spot for single-graph systems.
- MAGMA's total query latency of 1.47s is 65% of A-MEM's (2.26s), attributed to early pruning in the adaptive policy and the dual-stream offloading of heavy tasks. Although A-MEM is more token-efficient (2.62k), its Judge score is only 0.580, suggesting excessive pruning loses critical evidence.
- Single-graph-only configurations all scored < 0.60, proving the four relationship types are mutually indispensable. The Causal-only configuration excelled in the Adversarial category (0.680), highlighting the robustness of causal edges.
- In ultra-long context, MAGMA simplifies context to < 5K tokens while improving performance, proving that the multi-graph + adaptive policy effectively performs "structural compression."

## Highlights & Insights
- The abstraction of "memory = multi-view relationship graph" is elegant: four independent graphs with unified node identities allow retrieval to be routed by query dimension, providing easier intent-specific path control than the monolithic heterogeneous KGs used in GraphRAG.
- Dual-stream writing is a highly practical engineering design—it elegantly decouples the conflicting goals of low agent response latency and deep long-term memory reasoning using a producer-consumer queue.
- The hybrid retrieval paradigm, which fuses vector, keyword, and time anchors with intent-weighted edge type bonuses, can be directly migrated to any graph-based RAG system.
- Salience-based token budgeting, which compresses low-relevance nodes into brevity codes like "...3 events...", maintains prompt conciseness without losing link continuity—an insightful prompting strategy for LLMs acting as interpreters.

## Limitations & Future Work
- Heavy reliance on LLMs for structural extraction in the Slow Path—if the LLM yields hallucinated edges, errors propagate to retrieval. A conservative threshold is used but cannot completely eliminate this.
- Multi-graph storage and dual-stream processing introduce engineering complexity and storage overhead, which may be prohibitive for resource-constrained scenarios (edge devices/personal agents).
- Evaluation was limited to dialogue-heavy long contexts (LoCoMo/LongMemEval) and did not cover multimodal, tool-use, or code agent scenarios. The three-way intent classification may also be too coarse for complex queries.
- No analysis of retrieval time growth as the memory graph scales; sparsification strategies for entity and causal graphs in long-running scenarios are missing.
- All parameters are empirically tuned; future work could use RL to learn the traversal weights $\mathbf{w}_{T_q}$.

## Related Work & Insights
- **vs A-MEM (Xu 2025)**: Both use structured memory, but A-MEM employs Zettelkasten-style linear notes and semantic retrieval without differentiating relationship dimensions. MAGMA's explicit four-graph approach yields a 0.742 Adversarial score vs A-MEM's 0.616.
- **vs Nemori (Nan 2025)**: Nemori excels in event segmentation and representation alignment, but its internal structure consists of narrative chunks without explicit causal/entity edges.
- **vs GraphRAG / Zep**: Both are graph-based. GraphRAG uses offline community summaries, while Zep is a single temporal KG. MAGMA's combination of split graphs, asynchronous consolidation, and intent routing is a more lightweight, online-evolvable design.
- **vs MemGPT / MemoryOS**: These focus on OS analogies and paged memory management, whereas MAGMA aligns more with reasoning structures and cognitive science (CLS).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of orthogonal multi-graphs, intent routing, and dual-stream writing is unique in agentic memory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with two benchmarks, four baselines, two types of ablations, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clean formulas and a walkthrough that makes the abstract pipeline concrete.
- Value: ⭐⭐⭐⭐ The dual-stream and intent routing patterns are directly applicable for deploying production-grade agentic memory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](how_adversarial_environments_mislead_agentic_ai.md)
- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](../../NeurIPS2025/llm_agent/a-mem_agentic_memory_for_llm_agents.md)

</div>

<!-- RELATED:END -->
