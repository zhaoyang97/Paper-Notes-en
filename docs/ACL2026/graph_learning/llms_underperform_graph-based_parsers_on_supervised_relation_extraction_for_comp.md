---
title: >-
  [Paper Note] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
description: >-
  [ACL 2026][Graph Learning][Relation Extraction] Across six RE datasets comparing four LLMs (7B-70B) against a lightweight graph parser (124M parameters), graph parsers consistently and significantly outperform LLMs when average relation graph edges exceed ~18, with F1 gaps reaching 13.2 points on the most complex ERFGC dataset, revealing fundamental LLM limitations in complex linguistic graph structure extraction.
tags:
  - ACL 2026
  - Graph Learning
  - Relation Extraction
  - Graph Parser
  - LLM Limitations
  - Graph Complexity
  - Supervised Learning
content_hash: 66bdf9341ee7da35
---

# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**Conference**: ACL 2026
**arXiv**: [2604.08752](https://arxiv.org/abs/2604.08752)
**Code**: N/A
**Area**: Information Extraction / Relation Extraction
**Keywords**: Relation Extraction, Graph Parser, LLM Limitations, Graph Complexity, Supervised Learning

## TL;DR
Across six RE datasets comparing four LLMs (7B-70B) against a lightweight graph parser (124M parameters), graph parsers consistently and significantly outperform LLMs when average relation graph edges exceed ~18, with F1 gaps reaching 13.2 points on the most complex ERFGC dataset, revealing fundamental LLM limitations in complex linguistic graph structure extraction.

## Method

### Key Designs

1. **Graph Complexity Gradient Dataset Selection**: Six datasets ranging from $\bar{k}$=1.42 (CoNLL04) to $\bar{k}$=49.19 (ERFGC), precisely locating the LLM performance degradation tipping point.

2. **Prompt Ablation and Irrelevant Prompt Experiments**: UUID prompts achieve best average performance on Qwen3-14B-Base, and adversarial prompts are overridden after just 10 fine-tuning steps — proving prompt content is irrelevant under supervised fine-tuning.

## Key Experimental Results

| Dataset | Avg Relations | Graph Parser (124M) | Best LLM | Gap |
|---------|-------------|---------------------|----------|-----|
| ADE | 1.59 | 0.697 | **0.836** | +13.9 |
| enEWT | 17.83 | **0.865** | 0.851 | -1.4 |
| ERFGC | 49.19 | **0.713** | 0.606 | **-13.2** |

## Highlights & Insights
- Clean negative result paper — precisely locates LLM capability boundary through experiments
- UUID prompt experiment is particularly insightful: under supervised fine-tuning, models learn task semantics through gradients, completely ignoring prompt content
- 14M trainable parameters dominate 32B/70B LLMs on complex graphs, demonstrating inductive bias (biaffine attention) outweighs model scale for specific tasks

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)

<!-- RELATED:END -->
