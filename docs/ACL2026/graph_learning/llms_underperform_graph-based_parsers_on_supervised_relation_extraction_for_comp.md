---
title: >-
  [Paper Note] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
description: >-
  [ACL 2026][Graph Learning][Paper Note] This paper evaluates four LLMs (7B-70B) against a lightweight graph parser (124M parameters) across six relation extraction datasets. The study finds that graph parsers consistently and significantly outperform LLMs when the average number of edges in a document's relation graph exceeds approximately 18. On the most co
tags:
  - ACL 2026
  - Graph Learning
date: 2026-05-08
content_hash: 4afe92b8ccc98e50
---
# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**Conference**: ACL 2026  
**arXiv**: [2604.08752](https://arxiv.org/abs/2604.08752)  
**Code**: None  
**Area**: Information Extraction / Relation Extraction  
**Keywords**: Relation Extraction, Graph Parsers, LLM Limitations, Linguistic Graph Complexity, Supervised Learning

## TL;DR

This paper evaluates four LLMs (7B-70B) against a lightweight graph parser (124M parameters) across six relation extraction datasets. The study finds that graph parsers consistently and significantly outperform LLMs when the average number of edges in a document's relation graph exceeds approximately 18. On the most complex ERFGC dataset, the F1 gap reaches 13.2 points, revealing fundamental limitations of LLMs in extracting complex linguistic graph structures.

## Background & Motivation

**Background**: Relation Extraction (RE) is a core step in knowledge graph construction. Dominant paradigms include graph-based parsers (which directly model inter-token relations via GNNs or attention mechanisms) and LLM-based methods (which extract RDF triples through in-context learning or supervised fine-tuning). Recently, LLMs have been extensively explored for RE tasks.

**Limitations of Prior Work**: (1) Research on LLMs for RE has focused primarily on In-Context Learning (ICL) settings, leaving a gap in direct comparisons with traditional graph parsers under supervised settings; (2) Existing evaluations mostly utilize datasets with simple relation graphs (only 1-5 relations per document), failing to consider the impact of graph complexity on performance; (3) While LLMs have parameter counts two orders of magnitude larger than graph parsers, it remains unclear if they provide value for money in all scenarios.

**Key Challenge**: LLMs must format relation extraction results as serialized text. This serialization process introduces noise into causal attention and increases the distance between related tokens. When the graph structure is complex (with many edges), this formatting overhead grows linearly with graph size, whereas graph parsers bypass this issue by modeling relations directly on token embeddings.

**Goal**: To systematically investigate the RE performance of LLMs under varying relation graph complexities and provide a fair comparison with lightweight graph parsers.

**Key Insight**: Select six datasets with vast differences in relation graph complexity (average edges ranging from 1.42 to 49.19) to conduct comparative experiments under a unified supervised setting.

**Core Idea**: The serialized output mechanism of LLMs is the fundamental reason for their inferiority to graph parsers on complex relation graphs—formatted text dilutes attention and increases the distance between tokens that need to be predicted.

## Method

### Overall Architecture

This is a controlled experiment paper designed to answer: "Under a supervised setting, are LLMs truly more suitable for relation extraction than lightweight graph parsers?" The framework pits two types of models against each other in the same supervised pipeline: a graph parser following the biaffine attention architecture of Dozat & Manning, using a frozen BERT (110M) as an encoder with a 14M trainable parser head to model relations directly on token embeddings; and four LLMs fine-tuned via LoRA (Mistral-7B, Qwen3-14B-Base, Qwen3-32B, Llama-3.3-70B), which serialize extraction results into RDF triple text. Both model types are trained on six datasets with varying graph complexity (average edges $1.42 \rightarrow 49.19$) and evaluated using exact-match micro-F1 (triples are correct only if perfectly matched) to characterize the performance curve relative to graph complexity.

### Key Designs

**1. Dataset Selection with a Gradient of Graph Complexity: Taking "Edge Count" as a Controllable Independent Variable**

Most RE evaluations use simple graphs with 1-5 relations per document, which can easily overestimate LLM capabilities and mask performance degradation. This paper deliberately creates a gradient from simple to complex based on the average number of relations $\bar{k}$: CoNLL04 ($\bar{k}=1.42$, 98.5% of samples $k \le 5$), ADE ($\bar{k}=1.59$), and SciERC ($\bar{k}=2.38$) represent simple graphs; enEWT ($\bar{k}=17.83$, only 25% of samples $k \le 5$) and SciDTB ($\bar{k}=23.41$) represent medium complexity; and ERFGC ($\bar{k}=49.19$, only 3.3% of samples $k \le 5$) represents highly complex directed acyclic process graphs. This continuous gradient allows for precise identification of the inflection point for LLM performance degradation—empirically observed at $\bar{k} > 18$, where graph parsers begin to dominate.

**2. Prompt Ablation and Irrelevant Prompt Experiments: Verifying the Importance of Prompt Content under Supervision**

To clarify whether LLM performance stems from instruction following or gradient learning, the authors designed four prompt configurations: NoDesc (labels only), Desc (labels + descriptions), UUID (random UUIDs replacing instructions), and Adversarial prompts (explicitly telling the model not to perform the task). The results are counter-intuitive: the UUID prompt achieved the best average F1 of 0.692 on Qwen3-14B-Base, and the model began following the task while ignoring the adversarial instructions after just 10 steps of fine-tuning. This suggest that in supervised fine-tuning, the model learns the task through LoRA weights rather than instructions, rendering prompt content nearly irrelevant—challenging the common practice of "prompt engineering for fine-tuning."

**3. Fair Computational Resource Comparison: Ensuring Results are not due to Training Budgets or Randomness**

To prevent bias from training duration or random fluctuations, computational resources were aligned: graph parsers were trained for 3K steps, and LLMs were trained for a single epoch (with some models trained for an additional 3K steps for comparison). Multiple random seeds were used to estimate variance—5 seeds for Mistral and Qwen3-14B, and 3 seeds for Qwen3-32B. All LLMs showed very low F1 variance (average $\sigma \le 0.012$), indicating that the observed performance gaps are stable and reliable rather than artifacts of training steps or luck.

### Loss & Training

The graph parser is trained using cross-entropy loss for the biaffine attention head. LLMs are trained using standard language modeling loss with LoRA ($r=a=16$), fine-tuning only the Q/K/V weights in attention layers. Both use the AdamW optimizer.

## Key Experimental Results

### Main Results

**Comparison of Best Micro-F1 Across Datasets**

| Dataset | Avg. Relations | Graph Parser (124M) | Best LLM | LLM Model | Delta |
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
| Pearson r (Qwen3-14B, ERFGC) | -0.639 | Strong negative correlation between edge count and F1 |
| Pearson r (Graph Parser, ERFGC) | -0.206 | Much weaker correlation for graph parsers |
| UUID prompt vs NoDesc | UUID average better | Prompt content is irrelevant under supervision |
| 1 epoch vs 3K steps | ERFGC: 0.514 $\rightarrow$ 0.581 | More training steps help with complex graphs but are still insufficient |
| Qwen3-14B vs Qwen3-32B | 14B usually better | Instruction tuning might introduce harmful chatbot bias |

### Key Findings

- **Relation Threshold $\approx 18$**: Graph parsers begin to consistently outperform LLMs when the average number of relations $\bar{k} > 18$.
- **Gap Widens with Complexity**: The performance gap expands rapidly from 1.4 points on enEWT to 13.2 points on ERFGC.
- **Base Models Beat Instruct**: Qwen3-14B-Base outperformed the instruction-tuned version, suggesting that chatbot inductive biases are detrimental to RE.
- **Inference Speed**: The speed gap is enormous; LLMs require thousands of forward passes to extract large graphs, whereas a graph parser requires only one.

## Highlights & Insights

- This is a clean "negative result" paper—it uses experiments to precisely locate the capability boundaries of LLMs rather than simply claiming they are inadequate.
- The UUID prompt experiment is particularly insightful: in supervised fine-tuning, models learn task semantics through gradients, effectively ignoring prompt content. This challenges the practice of meticulous prompt engineering for fine-tuning.
- A graph parser with only 14M trainable parameters crushed 32B/70B LLMs on complex graphs, demonstrating that inductive bias (biaffine attention directly modeling token pairs) can be far more important than model scale for specific tasks.

## Limitations & Future Work

- Only one graph parser architecture was used; comparing more types could provide a more comprehensive picture.
- Qualitative analysis of the attention mechanism is needed to directly verify the hypothesis of "formatting noise diluting attention."
- Strategies to mitigate LLM disadvantages, such as prompt compression or reduced formatting, were not explored.
- Computational constraints meant some large models were only evaluated on subsets.

## Related Work & Insights

- **vs. Graph Parsers (Dozat & Manning 2017)**: This paper adopts this classic architecture and proves its advantage actually increases as graphs become more complex.
- **vs. LLM-based RE (Gajo & Barrón-Cedeño 2025)**: While prior work looked at natural language vs. code output for RE, this work focuses on the dimension of graph complexity.
- **vs. ICL-based RE (Wan et al. 2023)**: ICL methods do not fine-tune and have lower performance ceilings; this paper finds LLM limitations even in supervised settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic comparison of LLMs and graph parsers under supervised settings to locate the complexity threshold.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four LLMs, six datasets, multiple prompt types, and training durations.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear and data presentation is robust.
- **Value**: ⭐⭐⭐⭐ Provides empirical guidance for model selection in RE: use LLMs for simple graphs and graph parsers for complex ones.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)

</div>

<!-- RELATED:END -->
