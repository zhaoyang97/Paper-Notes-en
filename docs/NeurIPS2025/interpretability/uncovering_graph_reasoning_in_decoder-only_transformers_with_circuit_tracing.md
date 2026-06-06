---
title: >-
  [Paper Note] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing
description: >-
  [NeurIPS 2025 (Workshop on Efficient Reasoning)][Interpretability][Graph Reasoning] This paper applies a circuit tracing framework to analyze the internal mechanisms of decoder-only Transformers on graph reasoning tasks…
tags:
  - "NeurIPS 2025 (Workshop on Efficient Reasoning)"
  - "Interpretability"
  - "Graph Reasoning"
  - "Transformer"
  - "Circuit Tracing"
  - "Token Merging"
  - "Structural Memorization"
date: 2026-05-08
content_hash: 2498b15ef9fdf1df
---

# Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing

**Conference**: NeurIPS 2025 (Workshop on Efficient Reasoning)  
**arXiv**: [2509.20336](https://arxiv.org/abs/2509.20336)  
**Code**: None  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Graph Reasoning, Transformer, Circuit Tracing, Token Merging, Structural Memorization

## TL;DR

This paper applies a circuit tracing framework to analyze the internal mechanisms of decoder-only Transformers on graph reasoning tasks, uncovering two core reasoning mechanisms: token merging and structural memorization.

## Background & Motivation

Transformer-based LLMs have demonstrated strong performance on graph reasoning tasks (e.g., path finding, subgraph extraction), yet their internal reasoning mechanisms remain a black box. Limitations of existing interpretability work:

**Lack of a unified perspective**: Mechanistic analyses of different graph reasoning tasks lack coherent connections.

**Methodological limitations**: Traditional approaches such as attention visualization are insufficient to reveal deeper mechanisms.

**Model scope**: Most analyses target encoder or encoder-decoder architectures, leaving decoder-only architectures understudied.

This paper employs the circuit tracing framework to conduct a unified analysis of the internal mechanisms underlying graph reasoning in a basic decoder-only Transformer.

## Method

### Overall Architecture

1. Train small-scale decoder-only Transformers on graph reasoning data.
2. Apply circuit tracing techniques to track information flow.
3. Visualize reasoning trajectories to identify core computational patterns.
4. Quantitatively analyze the relationship between these patterns and task performance.

### Key Designs

1. **Circuit Tracing Framework**:

    - Based on the circuit discovery methodology proposed by Anthropic.
    - Traces causal information flow from input to output.
    - Identifies critical attention heads and MLP neurons.

2. **Two Core Mechanisms Discovered**:

   **Token Merging**:
    - Specific attention heads aggregate graph-structural information (nodes and edges) into a single token position.
    - Functions as an "information aggregation point," concentrating distributed graph-structural encodings.
    - Particularly critical in path reasoning tasks.

   **Structural Memorization**:
    - MLP layers store "templates" of common graph structural patterns.
    - During inference, relevant structures are retrieved via pattern matching.
    - Plays a dominant role in subgraph extraction tasks.

3. **Analysis Dimensions**:

    - Effect of graph density on the mechanisms.
    - Effect of model scale on the mechanisms.
    - Contribution ratio of the two mechanisms across different task types.

### Loss & Training

Training uses the standard autoregressive language modeling loss:
$$\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, G)$$

where $G$ denotes the textualized representation of the input graph.

## Key Experimental Results

### Main Results (Graph Reasoning Task Accuracy)

| Model Scale | Path Detection ↑ | Shortest Path ↑ | Cycle Detection ↑ | Subgraph Matching ↑ | Connected Components ↑ |
|-------------|-----------------|-----------------|-------------------|---------------------|------------------------|
| 2L-4H | 72.3 | 58.2 | 68.5 | 65.1 | 71.8 |
| 4L-8H | 89.5 | 75.8 | 84.2 | 82.6 | 87.3 |
| 6L-8H | 95.2 | 86.5 | 91.8 | 90.3 | 93.5 |
| 8L-16H | 97.8 | 92.1 | 95.5 | 94.7 | 96.2 |

### Mechanism Contribution Quantification

| Task Type | Token Merging (%) | Structural Memorization (%) | Other (%) |
|-----------|------------------|-----------------------------|-----------|
| Path Detection | 62.5 | 25.3 | 12.2 |
| Shortest Path | 58.8 | 28.5 | 12.7 |
| Cycle Detection | 45.2 | 42.8 | 12.0 |
| Subgraph Matching | 32.1 | 55.6 | 12.3 |
| Connected Components | 55.3 | 32.5 | 12.2 |

### Graph Density Impact Analysis

| Graph Density | Token Merging Effectiveness ↑ | Structural Memorization Effectiveness ↑ | Overall Accuracy |
|---------------|------------------------------|-----------------------------------------|-----------------|
| Sparse (d=0.1) | 0.92 | 0.85 | 96.5 |
| Medium (d=0.3) | 0.85 | 0.78 | 91.2 |
| Dense (d=0.5) | 0.72 | 0.65 | 82.8 |
| Very Dense (d=0.7) | 0.58 | 0.52 | 71.3 |

### Key Findings

1. **Token merging dominates path-type tasks**: Tasks requiring information propagation along paths rely more heavily on token merging.
2. **Structural memorization dominates matching-type tasks**: Tasks requiring the identification of specific patterns rely more heavily on structural memorization.
3. **Graph density has a significant impact**: The efficiency of both mechanisms degrades in dense graphs due to information overload.
4. **Model scale positively correlates with mechanism complexity**: Larger models learn more refined token merging strategies.

## Highlights & Insights

- **Unified explanatory framework**: This is the first work to reveal the graph reasoning mechanisms of decoder-only Transformers from a unified perspective.
- **Dual-mechanism discovery**: Token merging and structural memorization together provide an intuitive understanding of the reasoning process.
- **Actionable insights**: Understanding these mechanisms can inform the design of more efficient graph reasoning models.

## Limitations & Future Work

1. Experiments are conducted only on small-scale Transformers; mechanisms in large models (e.g., GPT-4) may differ.
2. The textualization scheme for graphs may influence the conclusions drawn.
3. As a workshop paper, the experimental depth warrants further extension.
4. The reliability of the circuit tracing methodology itself remains a subject of debate.

## Related Work & Insights

- **Anthropic Circuit Tracing**: The methodological foundation of this work.
- **Mechanistic Interpretability**: The series of works by Elhage et al.
- **GraphQA**: A graph reasoning benchmark.
- **Theoretical Analysis of Transformers**: Yun et al.'s analysis of Transformer expressivity.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 3 |
| Writing Quality | 4 |
| Value | 3 |
| Overall Recommendation | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers](what_happens_during_the_loss_plateau_understanding_abrupt_learning_in_transforme.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)

</div>

<!-- RELATED:END -->
