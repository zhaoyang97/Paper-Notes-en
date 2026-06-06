---
title: >-
  [Paper Note] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning
description: >-
  [ACL 2026][Graph Learning][Reinforcement Learning] AgentGL is the first RL-based agentic graph learning (AGL) framework that enables LLM agents to autonomously navigate text-attributed graphs (TAGs) via graph-native sear…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Reinforcement Learning"
  - "Agent Navigation"
  - "Text-Attributed Graph"
  - "Tool Use"
content_hash: c48d7eded3dbe745
---

# AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2604.05846](https://arxiv.org/abs/2604.05846)  
**Code**: [https://github.com/sunyuanfu/AgentGL](https://github.com/sunyuanfu/AgentGL)  
**Area**: Graph Learning / LLM Agent
**Keywords**: Graph Learning, Reinforcement Learning, Agent Navigation, Text-Attributed Graph, Tool Use

## TL;DR
AgentGL is the first RL-based agentic graph learning (AGL) framework that enables LLM agents to autonomously navigate text-attributed graphs (TAGs) via graph-native search tools, achieving up to 17.5% and 28.4% absolute accuracy gains on node classification and link prediction respectively.

## Background & Motivation

**Key Challenge**: Evidence on graphs is multi-scale — some clues exist in tight local neighborhoods, others emerge only from broader structural patterns. Agents must decide "where to go next" in a combinatorial space while avoiding redundant or uninformative regions. Effective graph reasoning requires multi-step exploration, but labeled search trajectories are extremely scarce.

**Core Idea**: Drive LLM agents with RL to learn graph-native search strategies, using search-constrained thinking to suppress over-retrieval and graph-conditioned curriculum learning to stabilize long-horizon policy optimization.

## Method

### Key Designs

1. **Graph-Native Search Toolkit**: Four complementary tools covering local-vs-global and structural-vs-semantic dimensions: $\tau_{1hop}$, $\tau_{2hop}$, $\tau_{ss}$ (PPR-based structural saliency), $\tau_{dense}$ (cosine similarity bridging semantically related but topologically disconnected nodes).

2. **Search-Constrained Thinking**: Backtrack termination triggers, cognitive density regularization (penalizing sparse reasoning fragments), and adaptive reward transitions to achieve "think more, search less."

3. **Graph-Conditioned Curriculum Learning (GCCL)**: Leverages intrinsic graph attributes to quantify sample difficulty at zero cost, enabling progressive training from easy to hard.

## Key Experimental Results

| Task | Dataset | AgentGL | Strongest Baseline | Gain |
|------|---------|---------|-------------------|------|
| Node Classification | OGB-Arxiv | 66.3 | 54.1 | +12.2 |
| Link Prediction | PubMed | 75.8 | 62.5 | +13.3 |
| Zero-shot Transfer (LP) | Reddit | 83.2 | 62.0 | +21.2 |

## Highlights & Insights
- The AGL paradigm itself is the core contribution — redefining graph learning from "static encoding" to "interactive navigation + reasoning"
- Zero-cost curriculum learning via intrinsic graph properties avoids manual annotation bottlenecks
- Search-constrained thinking is transferable to any tool-augmented LLM scenario

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](../../ICML2026/graph_learning/learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)

</div>

<!-- RELATED:END -->
