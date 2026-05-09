---
title: >-
  [Paper Note] Towards Improved Sentence Representations using Token Graphs
description: >-
  [ICLR 2026][Graph Learning][Sentence Representation] This paper proposes Glot, a lightweight structure-aware pooling module that constructs a latent similarity graph from token-level hidden states of a frozen LLM, refines them via a GNN, and aggregates them into a sentence representation. Glot achieves competitive performance with fine-tuning-based methods on GLUE/MTEB while requiring 20× fewer parameters and 100× faster training.
tags:
  - ICLR 2026
  - Graph Learning
  - Sentence Representation
  - Graph Neural Networks
  - Token Graph
  - Pooling
  - Frozen LLM
date: 2026-05-08
content_hash: 655b5f2ff492496d
---

# Towards Improved Sentence Representations using Token Graphs

**Conference**: ICLR 2026
**arXiv**: [2603.03389](https://arxiv.org/abs/2603.03389)
**Code**: [https://github.com/ipsitmantri/GLOT](https://github.com/ipsitmantri/GLOT)
**Area**: NLP / Graph Learning
**Keywords**: Sentence Representation, Graph Neural Networks, Token Graph, Pooling, Frozen LLM

## TL;DR
This paper proposes Glot, a lightweight structure-aware pooling module that constructs a latent similarity graph from token-level hidden states of a frozen LLM, refines them via a GNN, and aggregates them into a sentence representation. Glot achieves competitive performance with fine-tuning-based methods on GLUE/MTEB while requiring 20× fewer parameters and 100× faster training.

## Background & Motivation

**State of the Field**: LLMs produce token-level hidden states, yet many downstream tasks require a single-vector sentence representation. Standard practice relies on mean/max/[CLS] pooling, which treats tokens as an unordered set.

**Limitations of Prior Work**: (a) Standard pooling discards the rich relational structure captured by self-attention layers; (b) mean pooling is overwhelmed by noise when only a few tokens carry task-relevant signals; (c) the causal attention of decoder-only LLMs is optimized for next-token prediction rather than sentence understanding. Full model fine-tuning is prohibitively expensive.

**Root Cause**: How can high-quality sentence representations be obtained from a frozen model's outputs without fine-tuning the LLM?

**Paper Goals**: Reformulate pooling as "relational learning first, then aggregation" — treating tokens as a graph rather than an independent set.

**Starting Point**: Token hidden states from LLMs naturally carry similarity structure (cosine similarity), enabling the construction of a latent graph. GNNs that propagate information over such graphs before aggregation are strictly more expressive than the DeepSets framework.

**Core Idea**: Glot = Token similarity graph construction + Token-GNN refinement + Learnable readout. The LLM backbone is frozen; only the lightweight GNN head is trained.

## Method

### Overall Architecture

Frozen LLM produces $\mathbf{X} \in \mathbb{R}^{L \times d}$ → construct token similarity graph $\mathcal{G}$ → Token-GNN refinement → weighted aggregation readout → sentence vector $\mathbf{z}$.

### Key Designs

1. **Token Graph Construction**:

    - **Function**: Construct a sparse graph based on cosine similarity.
    - **Mechanism**: $\mathbf{S}_{ij} = \cos(\mathbf{x}_i, \mathbf{x}_j)$; an edge is created only when $\mathbf{S}_{ij} > \tau$, where $\tau$ is a hyperparameter.
    - **Design Motivation**: Preserve connections between semantically related tokens while discarding irrelevant ones. The threshold controls graph sparsity.

2. **Token-GNN Refinement**:

    - **Function**: Propagate information over the token graph.
    - **Mechanism**: A $K$-layer GNN with $\mathbf{a}_i^{(\ell)} = \text{AGGREGATE}_{j \in \mathcal{N}_i}(\mathbf{h}_j^{(\ell)})$ and $\mathbf{h}_i^{(\ell+1)} = \sigma(\mathbf{W}^{(\ell)} \text{CONCAT}(\mathbf{h}_i^{(\ell)}, \mathbf{a}_i^{(\ell)}))$.
    - **Design Motivation**: GNNs capture inter-token dependencies, such as the negation of "good" by "not" in "not good." DeepSets ($K=0$) cannot model such interactions.

3. **Learnable Readout**:

    - **Function**: Adaptively aggregate refined token representations.
    - **Mechanism**: $m_i = \mathbf{v}^\top \tanh(\mathbf{W}_m \mathbf{u}_i + \mathbf{b}_m)$, $\pi = \text{softmax}(\mathbf{m})$, $\mathbf{z} = \sum_i \pi_i \mathbf{u}_i$.
    - **Design Motivation**: Adaptive weights outperform fixed mean/max aggregation. Theoretically, Glot is shown to generalize mean/max/CLS pooling and AdaPool.

### Loss & Training

Task-specific losses are used (cross-entropy for classification, cosine loss for similarity). Only the GNN head and task classifier are trained; the backbone is fully frozen. The number of trainable parameters is 20× smaller than PEFT methods such as LoRA.

## Key Experimental Results

### Main Results (GLUE + Frozen BERT)

| Method | CoLA (MCC) | SST-2 (Acc) | STS-B (Spea) | MRPC (F1) | QQP (F1) |
|--------|-----------|-------------|--------------|-----------|---------|
| [CLS] | 22.66 | 83.83 | 61.08 | 79.58 | 19.70 |
| Mean | 19.55 | 82.91 | 74.96 | 80.28 | 29.01 |
| AdaPool | 29.20 | 87.72 | 80.01 | 77.99 | 40.15 |
| **Glot** | **47.49** | **90.25** | **83.86** | **82.58** | **62.19** |

### Ablation Study (Signal Dilution Stress Test)

| Method | 0% Noise | 50% Noise | 90% Noise |
|--------|----------|-----------|-----------|
| Mean | ~92% | ~70% | ~52% |
| AdaPool | ~93% | ~78% | ~60% |
| **Glot** | ~95% | ~94% | **97%+** |

### Key Findings
- **Glot achieves +25 MCC over [CLS] on CoLA** (47.49 vs. 22.66) — relational modeling is critical for linguistic understanding.
- **Signal dilution stress test**: With 90% randomly injected noise tokens, Mean and AdaPool collapse (~50–60%), while Glot maintains 97%+.
- **Decoder-only LLMs benefit most**: Glot yields substantially larger gains over Mean pooling on SmolLM2 and TinyLlama.
- **Extreme parameter efficiency**: 20× fewer parameters than PEFT methods such as LoRA, with 100×+ faster training.
- **Theoretical guarantee**: Glot strictly generalizes DeepSets; GNN message passing is provably more expressive than pure set functions.

## Highlights & Insights
- **A new paradigm of "pooling as relational learning"**: Tokens are treated not as an independent set but as a graph, enabling relation-based compression.
- **Practical value of frozen LLM + lightweight GNN**: Expensive fine-tuning is avoided, with only a small number of trainable parameters required.
- **Diagnostic value of stress testing**: The robustness gap at 90% noise clearly exposes the fundamental difference between relational learning and independent aggregation.

## Limitations & Future Work
- Graph construction relies on a cosine similarity threshold $\tau$, requiring hyperparameter tuning.
- The GNN introduces additional memory and computational overhead (though far less than fine-tuning).
- The possibility of pre-training the GNN head for cross-task transfer has not been explored.
- For long texts such as documents, the token graph may become prohibitively large.

## Related Work & Insights
- **vs. AdaPool (Brothers, 2025)**: AdaPool learns token weights but operates within the DeepSets framework, precluding token interaction modeling. Glot has a structural advantage via GNNs.
- **vs. TextGCN**: TextGCN constructs word co-occurrence graphs at the corpus level for classification, whereas Glot constructs token graphs at the sentence level for representation learning.
- **vs. ColBERT**: ColBERT retains multi-vector representations, while Glot compresses them into a single vector.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining pooling as graph-based relational learning represents a novel and theoretically grounded paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation spanning GLUE, MTEB, IMDB, stress tests, and 6 backbone architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the logical flow from theory to practice is complete.
- Value: ⭐⭐⭐⭐⭐ Highly efficient and practical, with immediate value for downstream applications of frozen LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](../../NeurIPS2025/graph_learning/learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](revisiting_node_affinity_prediction_in_temporal_graphs.md)
- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)

<!-- RELATED:END -->
