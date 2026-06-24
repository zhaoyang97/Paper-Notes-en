---
title: >-
  [Paper Note] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph] This paper proposes ARK: a training-free Knowledge Graph (KG) retrieval agent that exposes only two minimal tools—"global lexical search" and "single-hop neighbor expansion"—allowing the LLM to autonomously switch between breadth and depth without seed nodes or fixed hop counts. It pushes the average Hit@1 on three STaRK graphs to 59.1%, achieving up to a 31.4% improvement over training-free baselines…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph"
  - "Adaptive Retrieval"
  - "Breadth-Depth Trade-off"
  - "Tool Use"
  - "Trajectory Distillation"
date: 2026-05-08
content_hash: 6647c350861598ef
---

# Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval

**Conference**: ACL 2026  
**arXiv**: [2601.13969](https://arxiv.org/abs/2601.13969)  
**Code**: https://github.com/mims-harvard/ark  
**Area**: Graph Learning / Knowledge Graph Retrieval / RAG / LLM Agent  
**Keywords**: Knowledge Graph, Adaptive Retrieval, Breadth-Depth Trade-off, Tool Use, Trajectory Distillation

## TL;DR
This paper proposes ARK: a training-free Knowledge Graph (KG) retrieval agent that exposes only two minimal tools—"global lexical search" and "single-hop neighbor expansion"—allowing the LLM to autonomously switch between breadth and depth without seed nodes or fixed hop counts. It pushes the average Hit@1 on three STaRK graphs to 59.1%, achieving up to a 31.4% improvement over training-free baselines, and enables label-free strategy distillation into Qwen3-8B.

## Background & Motivation
**Background**: Integrating Knowledge Graphs (KG) into RAG has become mainstream because KGs organize evidence into "entities + typed edges," enabling reuse and relational constraints. Current KG retrieval follows two paths: (i) similarity retrieval (BM25, ada-002, GritLM, KAR), which is broad but shallow; (ii) multi-hop traversal (Think-on-Graph, GraphFlow), which follows relational chains but requires pre-identified seed entities and often task-specific training.

**Limitations of Prior Work**: Similarity methods "freeze" local neighborhoods into embeddings after encoding; multi-hop queries face complexity explosions when expanding context or stacking message passing. Traversal methods are sensitive to seeds—if the seed is wrong or incomplete, the search remains trapped locally, never reaching the correct evidence. Worse, many methods rely on graph-specific training, hindering generalization.

**Key Challenge**: KG queries inherently require both **breadth** (covering multiple entities or loose concepts via global coverage) and **depth** (reaching evidence hidden across multi-hop paths). Existing systems typically excel at only one mode and cannot adaptively switch within a single trajectory.

**Goal**: To build a training-free framework that allows an LLM to freely choose between "global search" and "relational expansion" within one trajectory, without being locked to specific seeds or preset hop counts, while designing a tool interface suitable for label-free distillation into smaller models.

**Key Insight**: Reframe retrieval from "scoring on a fixed index" to "interactive agent decision-making with tools." The toolset is deliberately kept minimal (only 2 primitives), shifting the burden of capability from tool complexity to the LLM's tool-use proficiency.

**Core Idea**: Use two primitives—"Global BM25 Search" and "Type-filtered Single-hop Expansion"—with a ReAct-style agent to express multi-hop traversal as "multiple combinations of neighbor calls," leaving the breadth-depth switching to the LLM.

## Method

### Overall Architecture
ARK transforms KG retrieval from "scoring on a fixed index" into "interactive agent decision-making with tools," centered on letting the LLM decide whether to "broadcast" or "deep-dive" within a trajectory. The graph is formalized as $G = \langle V, E, \phi_V, \phi_E, d_V \rangle$. Given a query $Q$, the agent $\mathcal{A} = \langle \text{LLM}, \mathcal{T} \rangle$ produces a trajectory $\tau = ((s_1, A_1, o_1), \dots, (s_T, A_T, o_T))$. At each step, it selects one of two tools, receives a new observation, and maintains an ordered candidate list $\mathcal{R}$ (appending nodes returned by tools or terminating with `finish`). Textual relevance is computed throughout using BM25's $\operatorname{rel}(q, d_V(v))$, which is fast and stable for repeated short queries. Optionally, $n$ independent agents run concurrently, with results aggregated via voting. The framework is inherently training-free, with its power derived from the LLM's tool-use ability rather than tool complexity.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["Query Q + Knowledge Graph G"] --> AGENT
    subgraph AGENT["Minimal Dual-Tool Interface (Single Agent ReAct Loop)"]
        direction TB
        R["ReAct Agent: Select tool per step"]
        R -->|Text-driven query| GS["Global Search<br/>Whole-graph BM25 Top-k"]
        R -->|Relation-driven query| NB["Neighbor Expansion<br/>1-hop neighborhood + Type filtering"]
        GS --> CAND["Ordered Candidate List R"]
        NB --> CAND
        CAND -->|Continue exploration| R
    end
    AGENT -->|Concurrent n independent agents| VOTE["Concurrent Self-Consistency<br/>Frequency Voting + First-discovery priority"]
    VOTE --> OUT["Output Ranked Candidate List"]
    AGENT -. GPT-4.1 teacher trajectories .-> DISTILL["Label-free Trajectory Distillation<br/>SFT Student Qwen3-8B"]
    DISTILL -. Replace backbone post-distillation .-> AGENT
```

### Key Designs

**1. Minimal Dual-Tool Interface: Two primitives for multi-hop combinations.** 

A common failure in traditional traversal agents is "total loss if the seed is wrong"—once the starting point is incorrect, the agent is trapped. ARK exposes only two tools: Global Search $\operatorname{Search}_G(q, k)$ performs BM25 on all node descriptions to provide "global anchors into the graph"; Neighborhood Exploration $\operatorname{Neighbors}(v, q, F)$ ranks the 1-hop neighborhood $N_F(v) = \{u \in N(v) \mid \phi_V(u) \in F_V, \phi_E(\{u,v\}) \in F_E\}$ by BM25, supporting node/edge type filtering $F = (F_V, F_E)$ and sub-queries $q$. The LLM alternates calls via ReAct prompting: text-heavy queries (e.g., AMAZON) use consecutive global searches for candidates; relation-heavy queries (e.g., MAG for an author's papers) first global search an anchor and then use multiple `Neighbors` to construct multi-hop paths. Crucially, global search remains available throughout the trajectory, serving as an "exit" to re-examine the whole graph and curing the seed anchoring problem.

**2. Concurrent Self-Consistency: Voting for stable consensus rather than more candidates.** 

A single agent can easily drift on high-branching graphs. Borrowing from LLM reasoning self-consistency, ARK runs $n$ independent agents with stochastic decoding. Each produces its own $\mathcal{R}^{(i)}$, and nodes are finally ranked by frequency across all trajectories. Ties are broken by the "earliest appearance" position—rewarding both majority consensus and early discovery. This rank fusion, implemented as simple "concatenation + frequency sorting," significantly outperforms basic ordering or random selection. The value of concurrency lies in "signal stability" rather than "candidate volume"; since agents are independent, latency is determined by the slowest agent rather than the sum.

**3. Label-free Trajectory Distillation: Embedding strategies into an 8B student.** 

To reduce the cost of large closed-source models, ARK distills the strategy into smaller models. GPT-4.1 acts as a teacher to generate trajectories on the training set (3 trajectories per query, max 20 steps, no rejection sampling). These trajectories consist of complete dialogues with tool calls and returns. The student, Qwen3-8B, undergoes SFT using next-token loss only on assistant tokens, learning "which tool to call and how to fill parameters" without ever seeing ground-truth relevance labels. Training uses LoRA with a 16384 context window for one epoch. The label-free nature is key—a student can be trained for any new graph as long as the teacher can run on it, eliminating the need for manual relevance annotations.

### Loss & Training
ARK is primarily training-free; SFT occurs only during distillation. The budget is approximately 18,000 trajectories per graph (~94.4M tokens). During training, user messages and tool outputs are masked; loss is calculated only on the assistant’s tool-call tokens. The aggregation rule (frequency + first-discovery priority) is hardcoded and not learned.

## Key Experimental Results

### Main Results (STaRK, GPT-4.1 backbone)

| Category | Method | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | Avg Hit@1 | Avg MRR |
|---|---|---|---|---|---|---|
| Training-free / Retrieval | BM25 | 44.94 | 25.85 | 12.75 | 27.85 | 36.68 |
| Training-free / Retrieval | KAR | 54.20 | 50.47 | 30.35 | 45.01 | 52.67 |
| Training-free / Agent | Think-on-Graph + GPT-4o | 20.67 | 23.33 | 16.67 | 20.22 | 31.43 |
| **Training-free / Agent** | **ARK** | **55.82** | **73.40** | **48.20** | **59.14** | **67.44** |
| Trained / Retrieval | mFAR | 53.0 | 55.9 | 40.0 | 49.63 | 60.20 |
| Trained / Retrieval | MoR | 52.19 | 58.19 | 36.41 | 48.93 | 58.77 |
| Trained / Agent | GraphFlow | 47.85 | 39.09 | 51.39 | 46.11 | 54.89 |
| Trained (Distilled) | ARK distilled (Qwen3-8B) | 54.99 | 61.66 | 31.87 | 49.51 | 58.47 |

The distilled student nearly matches the teacher on AMAZON (55.82 → 54.99) and outperforms base Qwen3-8B by +26.6 / +13.5 points on MAG/PRIME, respectively, retaining up to 98.5% of teacher performance.

### Ablation Study (10% Test Subset)

| Configuration | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | Description |
|---|---|---|---|---|
| Full ARK | 58.5 | 79.2 | 49.2 | Complete dual-tool |
| w/o Neighbors | 54.5 | **30.5** | **23.1** | Removing neighbors causes catastrophic drops on MAG/PRIME |
| Neighbors w/o $q$ | 56.0 | 72.1 | 44.7 | Neighbors not ranked by sub-query; moderate drop |
| Neighbors w/o $F$ | 55.5 | 79.2 | 42.2 | Disabling type filtering drops PRIME (biomedical) by 7 pts |

Aggregation comparison: Voting achieves 73.40 Hit@1 on MAG vs. 71.24 for Ordering and 38.04 for Random, proving concurrency provides stable consensus rather than just more candidates.

### Key Findings
- **ARK adaptively selects tools by query**: On AMAZON, 87.7% of calls are global search (text-driven), while on MAG / PRIME, 65.3% / 52.3% of calls shift to neighbor expansion (relation-driven), without explicit instructions.
- **Type filtering is critical for heterogeneous graphs**: PRIME is a biomedical graph with many types; disabling $F$ drops Hit@1 by 7 points. For homogeneous graphs (AMAZON), the impact is minimal.
- **Successful trajectories use neighbors sparingly**: Failed cases either never call neighbors (missing multi-hop context) or call them over 10 times (drift). Successful trajectories typically stop selective expansion within 10 calls, validating the "adaptive stop" mechanism.
- **BM25 can outperform dense retrieval**: Within ARK, BM25 outperforms text-embedding-3-large by 5–9 Hit@1 points on AMAZON/PRIME. Agents can "repair" lexical mismatches through iteration, whereas dense retrieval's advantage in single-shot recall is diluted in multi-step search.

## Highlights & Insights
- **"Minimal Toolset + Strong LLM" beats "Complex Toolset + Training"**: Using only 2 primitives outperforms RL-trained agents like GraphFlow (except on PRIME), emphasizing that framework simplicity can be more effective than tool complexity if decision-making is left to the LLM.
- **Global Search as a Perpetual Safety Net**: Keeping global search available throughout the trajectory provides an "anchor" to the whole graph, curing the seed-anchoring pathology. This design is applicable to any retrieval task requiring "global vision + local refinement" (DB queries, code navigation).
- **Reusability of Label-free Distillation**: Directly distilling teacher tool-use trajectories into students bypasses the cost of relevance annotation. This paradigm is applicable to any "LLM agent + tool" system.

## Limitations & Future Work
- Latency Overhead: Multiple LLM calls per query result in significantly higher end-to-end latency compared to single-shot dense retrieval, making high-concurrency/online scenarios difficult.
- BM25 Dependency: Relies on lexical matching, which is less effective for paraphrases, cross-lingual aliases, or domain synonyms. Performance may drop in graphs with low text density.
- Model Gap: The strongest configuration relies on GPT-4.1; open-source backbones still lag significantly behind the teacher on long-horizon tasks like PRIME even after distillation.
- Scale & Noise: Tested on STaRK graphs; validity on billion-node industrial graphs or graphs with significant schema noise remains to be verified.

## Related Work & Insights
- **vs Think-on-Graph**: ToG is also training-free but relies on beam search to expand paths without a global search "exit," leading to easy anchoring errors. ARK internalizes global search as a tool.
- **vs GraphFlow**: GraphFlow uses GFlowNets for RL-trained traversal; while stronger on PRIME, it requires graph-specific training. ARK is training-free and uses a smaller tool interface.
- **vs KAR / mFAR / MoR**: These are retriever-style methods focusing on field fusion or query expansion. ARK frames retrieval as agent interaction, raising the capability ceiling beyond ranker training.
- **vs ReAct / AvaTaR**: Specifically tailors the ReAct paradigm to KG retrieval, offering a complete "minimal tools + voting + distillation" recipe applicable to general tool-using agent design.

## Rating
- Novelty: ⭐⭐⭐⭐ Minimalist tool design is simple but the "training-free + label-free distillation" combination is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ STaRK graphs + multiple backbones + distillation curves + multi-factor ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization; the breadth-depth trade-off is intuitively explained.
- Value: ⭐⭐⭐⭐⭐ Provides a clean "baseline + distillation" solution for KG-RAG; directly forkable for industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](../../ACL2025/graph_learning/kg_rag_recommendation.md)

</div>

<!-- RELATED:END -->
