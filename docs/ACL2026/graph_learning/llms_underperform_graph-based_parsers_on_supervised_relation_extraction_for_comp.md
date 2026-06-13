---
title: >-
  [Paper Note] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
description: >-
  [ACL 2026][Graph Learning][Relation Extraction] This paper compares four LLMs (7B-70B) with a lightweight graph parser (124M parameters) across six relation extraction datasets. It discovers that when the average number…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Relation Extraction"
  - "Graph Parsers"
  - "LLM Limitations"
  - "Linguistic Graph Complexity"
  - "Supervised Learning"
date: 2026-05-08
content_hash: ee36554f855c3b5c
---

# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**Conference**: ACL 2026  
**arXiv**: [2604.08752](https://arxiv.org/abs/2604.08752)  
**Code**: None  
**Area**: Information Extraction / Relation Extraction  
**Keywords**: Relation Extraction, Graph Parsers, LLM Limitations, Linguistic Graph Complexity, Supervised Learning

## TL;DR

This paper compares four LLMs (7B-70B) with a lightweight graph parser (124M parameters) across six relation extraction datasets. It discovers that when the average number of edges in a document's relation graph exceeds approximately 18, the graph parser consistently and significantly outperforms LLMs. On the highly complex ERFGC dataset, the F1 gap reaches 13.2 points, revealing fundamental limitations of LLMs in extracting complex linguistic graph structures.

## Background & Motivation

**Background**: Relation Extraction (RE) is a core step for Knowledge Graph construction. Leading paradigms include graph-based parsers (directly modeling token relationships via GNNs/Attention) and LLM-based methods (extracting RDF triples via In-Context Learning or Supervised Fine-Tuning). LLMs have been extensively explored for RE tasks in recent years.

**Limitations of Prior Work**: (1) RE research on LLMs mainly focuses on the In-Context Learning (ICL) setting; direct comparisons with traditional graph parsers in supervised settings remain unexplored. (2) Existing evaluations mostly use datasets with simple relation graphs (only 1-5 relations per document), failing to consider the impact of graph complexity on performance. (3) LLMs have parameter counts two orders of magnitude larger than graph parsers, yet it remains unclear if they provide value in all scenarios.

**Key Challenge**: LLMs must format relation extraction results as text output. This serialization process introduces noise into causal attention and increases the distance between related tokens. When the graph structure is complex (high edge count), this formatting overhead grows linearly with graph size, whereas graph parsers bypass this issue by modeling relations directly on token embeddings.

**Goal**: To systematically study LLM performance in RE under varying relation graph complexities and provide a fair comparison with lightweight graph parsers.

**Key Insight**: Select six datasets with vastly different relation graph complexities (average edges ranging from 1.42 to 49.19) for comparative experiments under a unified supervised setting.

**Core Idea**: The serialization output mechanism of LLMs is the fundamental reason for their inferiority to graph parsers on complex relation graphs—formatted text dilutes attention and increases the distance between tokens that need to be predicted.

## Method

### Overall Architecture

The experimental framework includes two types of models: (1) Graph Parser—based on the biaffine attention architecture of Dozat & Manning, using BERT (110M, frozen) as the encoder with a 14M trainable parameter parsing head. (2) Four LLMs—Mistral-7B, Qwen3-14B-Base, Qwen3-32B, and Llama-3.3-70B, fine-tuned via LoRA. Models are trained and evaluated on six datasets using an exact-match evaluation for micro-F1 (triples must be entirely correct).

### Key Designs

1.  **Dataset Selection for Graph Complexity Gradient**:
    - Function: Construct a gradient of graph structures from simple to complex to systematically test model performance.
    - Mechanism: Six datasets are ordered by average relation count $\bar{k}$—CoNLL04 ($\bar{k}$=1.42, 98.5% of samples $k \leq 5$), ADE ($\bar{k}$=1.59), and SciERC ($\bar{k}$=2.38) represent simple graphs. enEWT ($\bar{k}$=17.83, only 25% of samples $k \leq 5$) and SciDTB ($\bar{k}$=23.41) represent medium complexity. ERFGC ($\bar{k}$=49.19, only 3.3% of samples $k \leq 5$) represents highly complex directed acyclic process graphs.
    - Design Motivation: Most RE evaluations use simple graph datasets, which easily overestimate LLM capabilities. A graph complexity gradient can precisely locate the inflection point of LLM performance degradation.

2.  **Prompt Ablation & Irrelevant Prompts**:
    - Function: Verify the impact of prompt content on performance during supervised fine-tuning.
    - Mechanism: Four prompt configurations were designed—NoDesc (category names only), Desc (category names + descriptions), UUID (meaningless UUIDs instead of instructions), and Adversarial prompts (instructing the model not to perform the task). Results show that the UUID prompt achieved the best average performance on Qwen3-14B-Base (F1=0.692), and the model began executing the task rather than following the instruction after only 10 steps of fine-tuning with the adversarial prompt.
    - Design Motivation: To prove that under the supervised fine-tuning setting, prompt design is nearly irrelevant—the model learns the task through LoRA weights rather than through instruction comprehension. This is a counter-intuitive but significant finding.

3.  **Fair Computational Resource Comparison**:
    - Function: Ensure the fairness of the comparison.
    - Mechanism: The graph parser is trained for 3K steps; LLMs are trained for a single epoch (with some models trained for an additional 3K steps for comparison). Variance is evaluated using multiple random seeds—5 seeds for Mistral and Qwen3-14B, 3 seeds for Qwen3-32B. F1 variance across all LLMs was low (average $\sigma \leq 0.012$), proving the reliability of the results.
    - Design Motivation: Avoid unfair comparisons resulting from differences in training steps or randomness.

### Loss & Training

The graph parser is trained using cross-entropy loss for the biaffine attention head. LLMs use standard language modeling loss with LoRA ($r=a=16$), fine-tuning only the Q/K/V weights of the attention layers. All optimizers are AdamW.

## Key Experimental Results

### Main Results

**Comparison of Best Micro-F1 Across Datasets**

| Dataset | Avg Relations ($\bar{k}$) | Graph Parser (124M) | Best LLM | LLM Model | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CoNLL04 | 1.42 | 0.668 | 0.674 | Llama-70B | +0.6 |
| ADE | 1.59 | 0.697 | **0.836** | Qwen3-14B | +13.9 |
| SciERC | 2.38 | 0.351 | **0.444** | Qwen3-14B | +9.3 |
| enEWT | 17.83 | **0.865** | 0.851 | Llama-70B | -1.4 |
| SciDTB | 23.41 | **0.918** | 0.886 | Qwen3-14B | -3.2 |
| ERFGC | 49.19 | **0.713** | 0.606 | Llama-70B | **-13.2** (3K steps) |

### Ablation Study

| Configuration | Finding | Explanation |
| :--- | :--- | :--- |
| Pearson r (Qwen3-14B, ERFGC) | -0.639 | Edge count is strongly negatively correlated with F1. |
| Pearson r (Graph Parser, ERFGC) | -0.206 | The correlation for the graph parser is much weaker. |
| UUID prompt vs NoDesc | UUID is better on average | Prompt content is irrelevant under supervised settings. |
| 1 epoch vs 3K steps | ERFGC: 0.514→0.581 | More training steps help with complex graphs but are still insufficient. |
| Qwen3-14B vs Qwen3-32B | 14B is usually better | Instruction tuning might introduce harmful chatbot bias. |

### Key Findings

- Relation count threshold $\approx 18$: Graph parsers begin to consistently outperform LLMs when the average relation count $\bar{k} > 18$.
- Performance gap widens rapidly with complexity: From 1.4 points on enEWT to 13.2 points on ERFGC.
- The base version of Qwen3-14B outperformed the instruction-tuned version, indicating that chatbot inductive biases are harmful to RE.
- The inference speed gap is also massive: LLMs require thousands of forward passes to extract large graphs, while graph parsers require only one.

## Highlights & Insights

- This is a clean negative-result paper—it precisely locates the capability boundaries of LLMs through experimentation rather than simply claiming LLMs are inadequate.
- The UUID prompt experiment is particularly insightful: in supervised fine-tuning, the model learns task semantics through gradients, while the actual prompt content is completely ignored. This challenges the practice of "carefully engineering prompts to improve fine-tuning."
- A graph parser with only 14M trainable parameters crushed 32B/70B LLMs on complex graphs, demonstrating that inductive bias (directly modeling token-pair relations via biaffine attention) is far more important than model scale for specific tasks.

## Limitations & Future Work

- Only one graph parser architecture was used; comparing more graph parsers could provide a more comprehensive picture.
- Lack of qualitative attention analysis to directly verify the hypothesis of "formatting noise diluting attention."
- Potential solutions to mitigate LLM disadvantages, such as prompt compression or reducing formatted text, were not explored.
- Computational constraints led to some large models being evaluated only on subsets.

## Related Work & Insights

- **vs Graph Parsers (Dozat & Manning 2017)**: This study directly adopts this classic architecture, proving its advantage actually increases as graph complexity grows.
- **vs LLM-based RE (Gajo & Barrón-Cedeño 2025)**: Previous work examined the impact of natural language vs programming language output on RE, whereas this study focuses on the dimension of graph complexity.
- **vs ICL-based RE (Wan et al. 2023)**: ICL methods do not fine-tune the model and have lower performance ceilings; this study finds LLM limitations even in supervised settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic comparison of LLMs and graph parsers under supervised settings to precisely locate the graph complexity threshold.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with four LLMs, six datasets, multiple prompts, and varying training steps.
- Writing Quality: ⭐⭐⭐⭐ Logical argumentation and clear data presentation.
- Value: ⭐⭐⭐⭐ Provides empirical guidance for model selection in RE tasks: use LLMs for simple graphs and graph parsers for complex ones.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)

</div>

<!-- RELATED:END -->
