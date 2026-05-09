---
title: >-
  [Paper Note] Graph-Based Alternatives to LLMs for Human Simulation
description: >-
  [ACL 2026][Graph Learning][Human Simulation] GEMS models closed-form human behavior simulation as link prediction on heterogeneous graphs, matching or surpassing strong LLM baselines with 1000x fewer parameters.
tags:
  - ACL 2026
  - Graph Learning
  - GNN
  - Human Simulation
  - Link Prediction
  - Heterogeneous Graph
content_hash: 987dc5fe8763a8bf
---

# Graph-Based Alternatives to LLMs for Human Simulation

**Conference**: ACL 2026
**arXiv**: [2511.02135](https://arxiv.org/abs/2511.02135)
**Code**: [GitHub](https://github.com/schang-lab/gems)
**Area**: Graph Learning / Human Behavior Simulation
**Keywords**: GNN, Human Simulation, Link Prediction, Heterogeneous Graph, Survey Prediction

## TL;DR
GEMS models closed-form human behavior simulation as link prediction on heterogeneous graphs with three node types (subgroups, individuals, choices) and two bidirectional relations, matching or surpassing strong LLM baselines across three datasets and three evaluation settings while using 1000x fewer parameters.

## Method

### Key Designs

1. **Heterogeneous Graph Construction and Link Prediction**: Individuals use uniform features (non-identifiable); GNN learns node embeddings through relation-aware message passing; decoder computes $p(c|u,q) = \text{softmax}(\text{Dot}(z_u^O, z_c^O) / \tau)$.

2. **Three Evaluation Settings**: Imputation (missing answers), new individual prediction, and new question prediction — covering core application scenarios.

3. **LLM-to-GNN Projection Layer (Setting 3 only)**: Linear projection from frozen LLM hidden states to GNN embedding space for unseen questions.

## Key Experimental Results

| Method | OpinionQA | Twin-2K | Dunning-Kruger |
|--------|-----------|---------|----------------|
| Few-shot FT (best LLM) | 55.98 | 66.36 | 57.21 |
| **GEMS (SAGE)** | **57.00** | **66.62** | **57.89** |

## Highlights & Insights
- Core insight: closed-form human simulation is essentially a recommendation system problem; relational structure matters more than language understanding
- GEMS can be trained from scratch on domain data, avoiding LLM pretraining data leakage and bias concerns

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[AAAI 2026\] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving](../../AAAI2026/graph_learning/human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)

<!-- RELATED:END -->
