---
title: >-
  [Paper Note] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
description: >-
  [ACL 2026][Graph Learning][Relation Extraction] By comparing four LLMs (7B-70B) with a lightweight graph parser (124M parameters) across six relation extraction datasets, this study finds that when the average number of edges in a document's relation graph exceeds approximately 18, the graph parser consistently and significantly outperforms LLMs. On the most complex ERFGC dataset, the F1 gap reaches 13.2 points, revealing fundamental limitations of LLMs in extracting complex…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Relation Extraction"
  - "Graph Parsers"
  - "LLM Limitations"
  - "Linguistic Graph Complexity"
  - "Supervised Learning"
date: 2026-05-08
content_hash: ff3d5cfb56457799
---

# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**Conference**: ACL 2026  
**arXiv**: [2604.08752](https://arxiv.org/abs/2604.08752)  
**Code**: None  
**Area**: Information Extraction / Relation Extraction  
**Keywords**: Relation Extraction, Graph Parsers, LLM Limitations, Linguistic Graph Complexity, Supervised Learning

## TL;DR

By comparing four LLMs (7B-70B) with a lightweight graph parser (124M parameters) across six relation extraction datasets, this study finds that when the average number of edges in a document's relation graph exceeds approximately 18, the graph parser consistently and significantly outperforms LLMs. On the most complex ERFGC dataset, the F1 gap reaches 13.2 points, revealing fundamental limitations of LLMs in extracting complex linguistic graph structures.

## Background & Motivation

**Background**: Relation Extraction (RE) is a core step in knowledge graph construction. Dominant paradigms include graph-based parsers (directly modeling token relationships via GNNs/attention mechanisms) and LLM-based methods (extracting RDF triplets through In-Context Learning or Supervised Fine-Tuning). In recent years, LLMs have been widely explored for RE tasks.

**Limitations of Prior Work**: (1) Research on LLMs for RE has primarily focused on In-Context Learning (ICL) settings, leaving a gap in direct comparisons with traditional graph parsers under supervised settings. (2) Existing evaluations mostly use datasets with simple relation graphs (only 1-5 relations per document), failing to account for the impact of graph complexity on performance. (3) While LLMs have parameter counts two orders of magnitude larger than graph parsers, it remains unclear whether they provide value for money in all scenarios.

**Key Challenge**: LLMs must format relation extraction results as serialized text output. This process introduces noise into causal attention and increases the distance between relevant tokens. As graph structures become complex (increasing edge counts), this formatting overhead grows linearly with graph size, whereas graph parsers bypass this issue by modeling relations directly on token embeddings.

**Goal**: To systematically investigate the RE performance of LLMs under varying graph complexities and provide a fair comparison with lightweight graph parsers.

**Key Insight**: Select six datasets with vastly different relation graph complexities (average edges ranging from 1.42 to 49.19) and conduct comparative experiments under a unified supervised setting.

**Core Idea**: The serialized output mechanism of LLMs is the fundamental reason for their inferiority to graph parsers on complex relation graphs—formatted text dilutes attention and increases the distance for predicting relevant tokens.

## Method

### Overall Architecture

This paper presents a controlled experiment to answer whether LLMs are truly more suitable for relation extraction than lightweight graph parsers in supervised settings. The framework pits two classes of models against each other in the same supervised pipeline: one class comprises graph parsers using the Dozat & Manning biaffine attention architecture, utilizing a frozen BERT (110M) as an encoder with a 14M trainable parameter parsing head to model relations directly between token pairs; the other class includes four LLMs fine-tuned via LoRA (Mistral-7B, Qwen3-14B-Base, Qwen3-32B, Llama-3.3-70B), which serialize extraction results into RDF triplet text. Both classes are trained on six datasets with varying graph complexities (average edges from 1.42 to 49.19) and evaluated using exact-match micro-F1 (triplets must be entirely correct), thereby mapping the performance curve against graph complexity.

### Key Designs

**1. Dataset Selection with Graph Complexity Gradients: Using "Edge Count" as a Controlled Variable**

Most RE evaluations use simple graphs with only 1-5 relations per document, which can easily overestimate LLM capabilities and mask performance degradation points. Ours intentionally creates a gradient from simple to complex based on the average relation count $\bar{k}$: CoNLL04 ($\bar{k}=1.42$, 98.5% of samples $k \le 5$), ADE ($\bar{k}=1.59$), and SciERC ($\bar{k}=2.38$) represent simple graphs; enEWT ($\bar{k}=17.83$, only 25% of samples $k \le 5$) and SciDTB ($\bar{k}=23.41$) represent medium complexity; and ERFGC ($\bar{k}=49.19$, only 3.3% of samples $k \le 5$) represents highly complex directed acyclic flowcharts. This continuous gradient allows for precise identification of the inflection point for LLM performance degradation, which is empirically observed to be around $\bar{k} > 18$.

**2. Prompt Ablation and Irrelevant Prompt Experiments: Verifying the Importance of Prompt Content in Supervised Settings**

To clarify whether LLM performance stems from instruction following or gradient learning, the authors designed four prompt configurations: NoDesc (labels only), Desc (labels + descriptions), UUID (replacing instructions with meaningless UUIDs), and Adversarial prompts (explicitly instructing the model not to perform the task). The results are counter-intuitive: the UUID prompt achieved the best average F1 of 0.692 on Qwen3-14B-Base, while models followed the task and ignored instructions within just 10 steps of fine-tuning on adversarial prompts. This suggests that in supervised fine-tuning, models learn tasks via LoRA weights rather than by reading instructions, making prompt content nearly irrelevant—challenging the common practice of "prompt engineering for fine-tuning."

**3. Fair Computational Resource Comparison: Ensuring Results are Not Due to Training Budgets or Randomness**

To avoid bias from training duration or random fluctuations, the authors aligned computational power: graph parsers were trained for 3K steps, and LLMs were trained for a single epoch (with some models receiving an additional 3K steps for comparison). Multiple random seeds were used to estimate variance—5 seeds for Mistral and Qwen3-14B, and 3 seeds for Qwen3-32B. The F1 variance for all LLMs was low (average $\sigma \le 0.012$), indicating that the observed gaps are stable and reliable.

### Loss & Training

Graph parsers are trained using cross-entropy loss for the biaffine attention head. LLMs are trained using standard language modeling loss paired with LoRA ($r=a=16$), fine-tuning only the Q/K/V weights of the attention layers. Both use the AdamW optimizer.

## Key Experimental Results

### Main Results

**Comparison of Best Micro-F1 Across Datasets**

| Dataset | Avg. Relations | Graph Parser (124M) | Best LLM | LLM Model | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CoNLL04 | 1.42 | 0.668 | 0.674 | Llama-70B | +0.6 |
| ADE | 1.59 | 0.697 | **0.836** | Qwen3-14B | +13.9 |
| SciERC | 2.38 | 0.351 | **0.444** | Qwen3-14B | +9.3 |
| enEWT | 17.83 | **0.865** | 0.851 | Llama-70B | -1.4 |
| SciDTB | 23.41 | **0.918** | 0.886 | Qwen3-14B | -3.2 |
| ERFGC | 49.19 | **0.713** | 0.606 | Llama-70B | **-13.2** (3K steps) |

### Ablation Study

| Configuration | Finding | Description |
| :--- | :--- | :--- |
| Pearson r (Qwen3-14B, ERFGC) | -0.639 | Strong negative correlation between edge count and F1. |
| Pearson r (Graph Parser, ERFGC) | -0.206 | Much weaker correlation for graph parser. |
| UUID prompt vs NoDesc | UUID better on avg | Prompt content is irrelevant under supervised fine-tuning. |
| 1 epoch vs 3K steps | ERFGC: 0.514 $\to$ 0.581 | More training steps help with complex graphs but are still insufficient. |
| Qwen3-14B vs Qwen3-32B | 14B often better | Instruction tuning might introduce harmful chatbot biases. |

### Key Findings

- **Relation Count Threshold $\approx$ 18**: Graph parsers begin to consistently outperform LLMs when the average number of relations $\bar{k} > 18$.
- **Gap Widens with Complexity**: The performance gap expands rapidly from 1.4 points on enEWT to 13.2 points on ERFGC.
- **Base vs. Instruct**: The non-instruction-tuned Qwen3-14B-Base outperformed the instruct version, suggesting chatbot inductive biases may be detrimental to RE.
- **Inference Speed**: The speed gap is enormous; LLMs require thousands of forward passes to extract large graphs, while graph parsers require only one.

## Highlights & Insights

- This is a clean "negative results" paper—carefully identifying the boundaries of LLM capabilities through experimentation rather than simply claiming LLMs are inadequate.
- The UUID prompt experiment is particularly insightful: in supervised fine-tuning, models learn task semantics through gradients, effectively ignoring prompt content. This challenges the practice of meticulous prompt design to improve fine-tuning results.
- The fact that a graph parser with only 14M trainable parameters crushes 32B/70B LLMs on complex graphs demonstrates that inductive bias (directly modeling token-pair relations via biaffine attention) is far more important than model scale for specific tasks.

## Limitations & Future Work

- Only one graph parser architecture was used; comparing more types could provide a more comprehensive picture.
- A lack of qualitative attention-level analysis to directly verify the "formatting noise dilutes attention" hypothesis.
- Potential mitigations for LLM disadvantages, such as prompt compression or reduced formatting text, were not explored.
- Computational constraints limited the evaluation of some large models to subsets of data.

## Related Work & Insights

- **vs. Graph Parsers (Dozat & Manning 2017)**: Ours adopts this classic architecture, proving its advantages in complex graphs are actually increasing over time.
- **vs. LLM-based RE (Gajo & Barrón-Cedeño 2025)**: While previous work studied the impact of natural language vs. programming language output formats on RE, ours focuses on the dimension of graph complexity.
- **vs. ICL-based RE (Wan et al. 2023)**: ICL methods do not fine-tune models and have lower performance ceilings; ours identifies LLM limitations even in supervised settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic comparison of LLMs and graph parsers under supervised settings to pinpoint graph complexity thresholds.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage with four LLMs, six datasets, multiple prompts, and training step variations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical progression and robust data presentation.
- **Value**: ⭐⭐⭐⭐ Provides empirical guidance for model selection in RE: use LLMs for simple graphs and graph parsers for complex ones.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](../../ACL2025/graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)

</div>

<!-- RELATED:END -->
