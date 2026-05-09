---
title: >-
  [Paper Note] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context
description: >-
  [ACL 2026][Graph Learning][GNN Explainability] Gspell is a lightweight post-hoc explanation framework that projects GNN node embeddings into LLM embedding space and constructs hybrid prompts (soft prompts + text), enabling LLMs to directly reason over GNN internal representations and generate natural language explanations with explanation subgraphs, achieving a good balance of faithfulness and interpretability on text-attributed graphs.
tags:
  - ACL 2026
  - Graph Learning
  - GNN Explainability
  - LLM Explainer
  - Soft Prompt
  - Text-Attributed Graph
  - Natural Language Explanation
content_hash: 910eb65eaea47d4f
---

# From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context

**Conference**: ACL 2026
**arXiv**: [2508.07117](https://arxiv.org/abs/2508.07117)
**Code**: N/A
**Area**: Graph Learning / Interpretability
**Keywords**: GNN Explainability, LLM Explainer, Soft Prompt, Text-Attributed Graph, Natural Language Explanation

## TL;DR
Gspell is a lightweight post-hoc explanation framework that projects GNN node embeddings into LLM embedding space and constructs hybrid prompts (soft prompts + text), enabling LLMs to directly reason over GNN internal representations and generate natural language explanations with explanation subgraphs, achieving a good balance of faithfulness and interpretability on text-attributed graphs.

## Method

### Key Designs

1. **GNN-LLM Embedding Projector**: Maps GNN embeddings to $k$ soft prompt tokens via context alignment loss (cosine similarity) and contrastive loss (preserving GNN similarity structure).

2. **Hybrid Prompt Construction**: Interleaves soft prompt embeddings with text descriptions for each node in the GNN computation tree.

3. **Explanation Subgraph Extraction with Hallucination Mitigation**: LLM predicts support/oppose/neutral labels for each computation tree node; post-processing verifies referenced nodes exist.

## Key Experimental Results

| Method | Fidelity+ ↑ | Sparsity ↑ | Insightfulness ↑ |
|--------|------------|-----------|-----------------|
| GNNExplainer | 0.12 | 0.65 | — |
| **Gspell** | **0.22** | **0.72** | **3.5** |

## Highlights & Insights
- Bypassing traditional GNN explainers to let LLMs directly interpret GNN internal representations reduces information loss and bias
- Plug-and-play (no GNN or LLM fine-tuning needed)

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)

<!-- RELATED:END -->
