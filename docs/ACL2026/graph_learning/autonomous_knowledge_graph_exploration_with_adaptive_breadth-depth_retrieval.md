---
title: >-
  [Paper Note] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval
description: >-
  [ACL 2026][Graph Learning][Knowledge Graphs] This paper proposes ARK: a training-free knowledge graph (KG) retrieval agent. By exposing two minimal primitives—"global lexical search" and "single-hop neighborhood expansio…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graphs"
  - "Adaptive Retrieval"
  - "Breadth-Depth Trade-off"
  - "Tool Use"
  - "Trajectory Distillation"
date: 2026-05-08
content_hash: f1d90bc03680b851
---

# Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval

**Conference**: ACL 2026  
**arXiv**: [2601.13969](https://arxiv.org/abs/2601.13969)  
**Code**: https://github.com/mims-harvard/ark  
**Area**: Graph Learning / Knowledge Graph Retrieval / RAG / LLM Agent  
**Keywords**: Knowledge Graphs, Adaptive Retrieval, Breadth-Depth Trade-off, Tool Use, Trajectory Distillation

## TL;DR
This paper proposes ARK: a training-free knowledge graph (KG) retrieval agent. By exposing two minimal primitives—"global lexical search" and "single-hop neighborhood expansion"—it allows the LLM to autonomously switch between breadth and depth without requiring seed nodes or fixed hop counts. On the STaRK datasets, it improves Hit@1 to an average of 59.1% (up to a 31.4% improvement over training-free baselines) and enables label-free strategy distillation into Qwen3-8B.

## Background & Motivation
**Background**: Integrating Knowledge Graphs (KGs) into RAG is common because KGs organize evidence into "entities + typed edges," enabling reuse and relational constraints. Current KG retrieval follows two paths: (i) similarity retrieval (BM25, ada-002, GritLM, KAR), which is broad but shallow; (ii) multi-hop traversal (Think-on-Graph, GraphFlow), which follows relation chains but requires predefined seed entities and often necessitates task-specific training.

**Limitations of Prior Work**: Similarity methods "freeze" local neighbors into embeddings, limiting flexibility. Multi-hop methods either suffer from context explosion or require complex message-passing modules. Traversal methods are sensitive to seeds—if the initial seeds are incorrect or incomplete, the search remains trapped in local regions and fails to find the correct evidence. Furthermore, many methods rely on graph-specific training, hindering generalization.

**Key Challenge**: KG queries inherently require both **breadth** (covering multiple entities or loose concepts) and **depth** (evidence hidden across multi-hop paths). Existing systems typically excel in only one mode and cannot adaptively switch within a single trajectory.

**Goal**: To build a training-free framework where an LLM can freely choose between "global search" and "relational expansion" within a single trajectory, without locked seeds or preset hop counts, while designing a tool interface suitable for label-free distillation into smaller models.

**Key Insight**: Instead of scoring fixed indices, KG retrieval is reframed as an interactive agent decision-making process. The toolset is kept minimal (only two primitives) so that the retrieval capability arises from the LLM’s tool-use performance rather than the complexity of the tools themselves.

**Core Idea**: Use "global BM25 search + type-filtered single-hop neighbors" as the two primitives within a ReAct-style agent. Multi-hop traversal is expressed as "compositions of neighbor calls," leaving the breadth-depth trade-off to the LLM's autonomous decision-making.

## Method

### Overall Architecture
The KG is formalized as $G = \langle V, E, \phi_V, \phi_E, d_V \rangle$. Given a query $Q$, the agent $\mathcal{A} = \langle \text{LLM}, \mathcal{T} \rangle$ produces a trajectory $\tau = ((s_1, A_1, o_1), \dots, (s_T, A_T, o_T))$. At each step, it selects one of two tools to call and receives a new observation. Simultaneously, it maintains an ordered list of candidates $\mathcal{R}$, which can be appended with nodes returned by the tools until the `finish` action is invoked. Optionally, $n$ independent agents run in parallel for voting aggregation. Text relevance is consistently measured using BM25: $\operatorname{rel}(q, d_V(v))$, which is efficient and stable for repeated short queries.

### Key Designs

1.  **Minimal Dual-Tool Interface (Global Search + Neighborhood Exploration)**:
    - **Function**: Global Search $\operatorname{Search}_G(q, k)$ performs BM25 ranking over all node descriptions to provide "global anchors." Neighborhood Exploration $\operatorname{Neighbors}(v, q, F)$ ranks nodes within the one-hop neighborhood $N_F(v) = \{u \in N(v) \mid \phi_V(u) \in F_V, \phi_E(\{u,v\}) \in F_E\}$ based on BM25, supporting node/edge type filtering $F = (F_V, F_E)$ and sub-queries $q$.
    - **Mechanism**: The LLM alternates tool calls via ReAct prompting. For text-led queries (e.g., AMAZON), it uses global search to gather candidates. For relation-led queries (e.g., MAG author searches), it anchors a starting node via global search and then constructs multi-hop paths through multiple `Neighbors` calls. It can re-anchor at any time to avoid local traps. Depth is not preset; the agent stops when it deems sufficient.
    - **Design Motivation**: Traditional traversal failure is rooted in "seed selection errors." ARK makes global search available throughout the trajectory, providing an "escape hatch" to re-examine the whole graph. BM25 is used because it was found to be more accurate than text-embedding-3-large for short agent queries in the experiments.

2.  **Parallel Voting Aggregation**:
    - **Function**: Simultaneously runs $n$ independent agents (with stochastic decoding). Each produces $\mathcal{R}^{(i)}$, and final results are ranked by node frequency across all trajectories. Ties are broken by "earliest appearance."
    - **Mechanism**: Inspired by self-consistency in LLM reasoning, multi-agent rank fusion is implemented via simple "concatenation + frequency sorting."
    - **Design Motivation**: A single agent may drift on graphs with high branching factors. Voting extracts stable signals. Since agents are independent, the end-to-end latency is determined by the slowest agent rather than the sum.

3.  **Label-free Trajectory Distillation to 8B Student**:
    - **Function**: Uses GPT-4.1 as a teacher to collect trajectories on a training set. A Qwen3-8B student is fine-tuned using next-token loss on assistant tokens (tool calls and parameters), without requiring ground-truth relevance labels.
    - **Mechanism**: Teacher trajectories serve as complete dialogues with tool interactions. The student is trained using LoRA for one epoch with a 16,384 context window.
    - **Design Motivation**: ARK's inference with large closed-source models is expensive; distillation to an 8B model reduces costs. Label-free distillation means a new graph only requires teacher runs for training data, removing the need for human relevance annotation.

### Loss & Training
ARK is training-free by design, only undergoing SFT during the distillation phase. The distillation budget is 18,000 trajectories per graph. Only assistant tool-call tokens are used for calculating the loss, with user messages and tool outputs masked.

## Key Experimental Results

### Main Results (STaRK, GPT-4.1 backbone)

| Category | Method | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | Avg Hit@1 | Avg MRR |
|---|---|---|---|---|---|---|
| Training-free / Retrieval | BM25 | 44.94 | 25.85 | 12.75 | 27.85 | 36.68 |
| Training-free / Retrieval | KAR | 54.20 | 50.47 | 30.35 | 45.01 | 52.67 |
| Training-free / Agent | Think-on-Graph + GPT-4o | 20.67 | 23.33 | 16.67 | 20.22 | 31.43 |
| **Training-free / Agent** | **ARK** | **55.82** | **73.40** | **48.20** | **59.14** | **67.44** |
| Supervised / Retrieval | mFAR | 53.0 | 55.9 | 40.0 | 49.63 | 60.20 |
| Supervised / Retrieval | MoR | 52.19 | 58.19 | 36.41 | 48.93 | 58.77 |
| Supervised / Agent | GraphFlow | 47.85 | 39.09 | 51.39 | 46.11 | 54.89 |
| Supervised (Distilled) | ARK distilled (Qwen3-8B) | 54.99 | 61.66 | 31.87 | 49.51 | 58.47 |

The distilled student nearly matches the teacher on AMAZON (55.82 → 54.99). It outperforms the base Qwen3-8B model by significant margins (+26.6 on MAG, +13.5 on PRIME), with teacher retention rates up to 98.5%.

### Ablation Study (10% Test Subset)

| Configuration | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | Meaning |
|---|---|---|---|---|
| Full ARK | 58.5 | 79.2 | 49.2 | Full dual tools |
| w/o Neighbors | 54.5 | **30.5** | **23.1** | Removing neighbors causes catastrophic drops on MAG/PRIME |
| Neighbors w/o $q$ | 56.0 | 72.1 | 44.7 | Neighbors not ranked by sub-query; moderate drop |
| Neighbors w/o $F$ | 55.5 | 79.2 | 42.2 | Turning off type filtering; PRIME (heterogeneous) drops 7 points |

Aggregation strategy comparison: Voting (73.40 Hit@1) vs. Ordering (71.24) vs. Random (38.04) on MAG proves that concurrency builds consensus rather than just increasing candidate size.

### Key Findings
- **ARK adaptively calls tools according to the query**: 87.7% of calls on AMAZON are global searches (text-led), whereas on MAG/PRIME, 65.3%/52.3% of calls shift to neighborhood expansion (relation-led).
- **Type filtering is critical for heterogeneous graphs**: PRIME is a biomedical graph with many types; disabling $F$ drops Hit@1 by 7 points.
- **Successful trajectories use neighbors temperately**: Failure cases often either fail to invoke neighbors at all or call them excessively (>10 times), leading to drift. Success cases typically converge within 10 calls.
- **BM25 out-performs dense embeddings in the loop**: On AMAZON/PRIME, BM25 Hit@1 is 5–9 points higher than text-embedding-3-large within the ARK framework. Agents can "repair" lexical mismatches through iterations, whereas dense embedding advantages are diluted in multi-step search.

## Highlights & Insights
- **"Minimal Toolset + Strong LLM" beats "Complex Tools + Training"**: Primitives provide the foundation, but leaving search decisions to the LLM proves more effective than RL-trained architectures like GraphFlow.
- **Global search never "retires"**: Keeping global search available throughout the trajectory provides an escape hatch, curing the "seed anchoring" dependency of traversal agents. This design is applicable to any retrieval task requiring "global vision + local refinement."
- **Reusability of Label-free Distillation**: Fine-tuning students on teacher trajectories avoids the cost of relevance labeling. This paradigm is applicable to any "LLM agent + tool" system.

## Limitations & Future Work
- Latency cost: ARK requires multiple LLM calls, making end-to-end latency significantly higher than single-pass retrieval.
- Global search is lexical (BM25): It handles paraphrasing and domain synonyms poorly compared to semantic methods.
- Dependence on closed-source LLMs: Even after distillation, open-source backbones lag behind the teacher on long-horizon tasks like PRIME.
- Evaluation on industrial-scale, dynamic graphs (billions of nodes) remains to be verified.

## Related Work & Insights
- **vs Think-on-Graph**: ToG relies on beam search for paths without a global search "reset," making it prone to bad seeds.
- **vs GraphFlow**: GraphFlow uses RL (GFlowNets) for traversal strategies; it is stronger on PRIME but requires graph-specific training.
- **vs KAR / mFAR / MoR**: These are retriever-style methods focused on field fusion or query expansion; ARK shifts retrieval to an agentic interaction.
- **vs ReAct / AvaTaR**: ARK adapts the ReAct paradigm specifically for KG retrieval, providing a complete "minimal tools + voting + distillation" recipe.

## Rating
- Novelty: ⭐⭐⭐⭐ (Simplifying tools is an elegant approach; label-free distillation is highly practical)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Tested on STaRK, multiple backbones, distillation curves, and extensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear formalization and intuitive explanation of trade-offs)
- Value: ⭐⭐⭐⭐⭐ (Provides a clean "baseline + distillation" solution for KG-RAG)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)

</div>

<!-- RELATED:END -->
