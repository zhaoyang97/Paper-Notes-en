---
title: >-
  [Paper Note] Revisiting Node Affinity Prediction in Temporal Graphs
description: >-
  [ICLR 2026][Graph Learning][Temporal Graph Neural Networks] This paper analyzes why simple heuristics (persistent forecasting, moving average) consistently outperform complex TGNNs on temporal graph node affinity predict…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Temporal Graph Neural Networks"
  - "Node Affinity Prediction"
  - "State Space Models"
  - "Ranking Loss"
  - "Global State"
date: 2026-05-08
content_hash: 044eee7baef016c1
---

# Revisiting Node Affinity Prediction in Temporal Graphs

**Conference**: ICLR 2026
**arXiv**: [2510.06940](https://arxiv.org/abs/2510.06940)  
**Code**: [https://github.com/orfeld415/NAVIS](https://github.com/orfeld415/NAVIS)  
**Area**: Graph Learning / Temporal Graphs
**Keywords**: Temporal Graph Neural Networks, Node Affinity Prediction, State Space Models, Ranking Loss, Global State

## TL;DR
This paper analyzes why simple heuristics (persistent forecasting, moving average) consistently outperform complex TGNNs on temporal graph node affinity prediction. It proves that these heuristics are special cases of linear SSMs and that standard RNNs/LSTMs/GRUs cannot express even the most basic persistent forecasting. Based on these findings, the paper proposes NAViS — a linear SSM architecture with a virtual global state and a ranking loss — which surpasses all baselines on TGB benchmarks.

## Background & Motivation

**Background**: Temporal graph node affinity prediction on CTDGs requires, given a query node $u$, predicting its affinity ranking with all other nodes at future timestamps. TGNNs such as TGN, TGAT, and DyGFormer achieve strong performance on link prediction tasks.

**Limitations of Prior Work**: Simple heuristics (persistent forecasting, moving average) consistently outperform all SOTA TGNNs on affinity prediction — a puzzling and largely unexplained phenomenon.

**Key Challenge**: Complex TGNN models fail to match even the simplest heuristics due to: insufficient expressiveness (nonlinear updates cannot maintain linear memory), loss function mismatch (cross-entropy is unsuitable for ranking), loss of global temporal dynamics from local neighborhood sampling, and information loss introduced by mini-batching.

**Goal**: (a) Provide a theoretical explanation for the deficiencies of TGNNs; (b) design a stronger architecture that generalizes the heuristics; (c) address the loss function mismatch.

**Key Insight**: Theorem 1 proves that heuristics (PF/EMA/SMA) are special cases of linear SSMs, while Theorem 2 shows that standard RNNs/LSTMs/GRUs cannot express even PF due to bounded outputs $\in (-1,1)$. This motivates designing architectures that maintain linear input-output mappings.

**Core Idea**: NAViS = learnable linear SSM + virtual global state + LambdaLoss ranking loss. A gating mechanism ensures the output is a convex combination of inputs (linear), while gate values adapt based on the current event.

## Method

### Overall Architecture

Each node maintains a state $\mathbf{h} \in \mathbb{R}^d$ and a virtual global state $\mathbf{g} \in \mathbb{R}^d$ (where $d = |\mathcal{V}|$). The previous affinity vector and current state are aggregated into a new state via linear transformation and gating. The predicted affinity vector is computed based on both the node state and the global state.

### Key Designs

1. **Gated Linear SSM Architecture**:

    - **Function**: Generalizes EMA while maintaining the output as a linear combination of inputs.
    - **Mechanism**: $\mathbf{z}_h = \sigma(W_{xh}\mathbf{x} + W_{hh}\mathbf{h}_{i-1} + \mathbf{b}_h)$, $\mathbf{h}_i = \mathbf{z}_h \odot \mathbf{h}_{i-1} + (1-\mathbf{z}_h) \odot \mathbf{x}$. The output follows similarly: $\mathbf{s} = \mathbf{z}_s \odot \mathbf{h}_i + (1-\mathbf{z}_s) \odot \mathbf{x}$, where $\mathbf{z}_s$ also depends on the global state $\mathbf{g}$.
    - **Design Motivation**: Sigmoid gating ensures $\mathbf{z} \in [0,1]$, so the output is a convex combination of the previous state and the current input. EMA is a special case where $\mathbf{z}$ is constant. NAViS is compatible with t-Batch, avoiding information loss within batches.

2. **Virtual Global State**:

    - **Function**: Captures network-level trends (e.g., new song releases, regime changes).
    - **Mechanism**: A buffer of recent affinity vectors is maintained and aggregated to compute $\mathbf{g}$. The global state $\mathbf{g}$ participates in computing the output gate $\mathbf{z}_s$.
    - **Design Motivation**: Affinity is often governed by global trends that locally-sampled TGNNs cannot capture.

3. **Lambda Ranking Loss + Pairwise Margin Regularization**:

    - **Function**: Replaces cross-entropy with a ranking-oriented loss.
    - **Mechanism**: Theorem 3 proves that cross-entropy is suboptimal for ranking — a correct ranking can incur higher CE loss than an incorrect one. LambdaLoss approximates the gradients of non-differentiable ranking metrics via pairwise "lambda" weights. A regularization term $\ell_{Reg} = \sum \max(0, -(s_{\pi_i} - s_{\pi_j}) + \Delta)$ prevents the model from collapsing affinity scores.
    - **Design Motivation**: Downstream applications depend on relative rankings rather than absolute values.

### Loss & Training

$\ell = \ell_{Lambda} + \ell_{Reg}$. Models are trained for 50 epochs with a batch size of 200 and a 70/15/15 temporal split. For large-scale graphs, sparsification retains only entries corresponding to candidate target nodes; for tgbn-token with 60,000+ nodes, approximately 5,000 parameters suffice.

## Key Experimental Results

### Main Results

| Method | tgbn-trade (Test) | tgbn-genre (Test) | tgbn-reddit (Test) | tgbn-token (Test) |
|--------|-------------------|-------------------|--------------------|--------------------|
| Moving Avg | 0.777 | 0.497 | 0.480 | 0.414 |
| TGNv2 | ~0.68 | ~0.50 | ~0.38 | ~0.39 |
| DyGFormer | 0.388 | 0.365 | 0.316 | - |
| **NAViS** | **0.874** | **0.553** | **0.503** | **0.416** |

NAViS outperforms TGNv2 (the best-performing TGNN) by +12.8% on tgbn-trade.

### Ablation Study

| Configuration | tgbn-trade | tgbn-genre | Notes |
|---------------|-----------|-----------|-------|
| NAViS (full) | 0.874 | 0.553 | Full model |
| w/o global state | ~0.84 | ~0.52 | Global state contributes |
| w/o Lambda Loss | ~0.82 | ~0.50 | Ranking loss is critical |
| Standard CE loss | ~0.79 | ~0.48 | CE is suboptimal |

### Key Findings
- **Standard RNNs/LSTMs/GRUs cannot express persistent forecasting** (Theorem 2): outputs are constrained to $(-1,1)$ by tanh/sigmoid activations.
- **Heuristics are special cases of SSMs** (Theorem 1): EMA corresponds to $\mathbf{A}=\alpha\mathbf{I}$; SMA corresponds to $\mathbf{A}=\frac{w-1}{w}\mathbf{I}$.
- **Cross-entropy is suboptimal for ranking** (Theorem 3): a correct ranking can yield higher CE loss than an incorrect one.
- **Global trends matter**: the virtual global state substantially reduces error in synthetic experiments.
- **NAViS achieves state-of-the-art across all TGB datasets** and is the first method to surpass heuristics.

## Highlights & Insights
- **Theory-driven design**: Three theorems precisely identify the deficiencies of TGNNs (expressiveness, loss function, information utilization) and directly guide architectural decisions.
- **Minimal yet effective**: NAViS uses very few parameters (~5,000 for tgbn-token), far fewer than standard TGNNs.
- **Unifying heuristics via SSMs**: seemingly ad-hoc heuristics are subsumed under a unified dynamical systems framework.

## Limitations & Future Work
- The model only captures local/global linear trends; complex multi-hop dependencies remain unaddressed.
- The global state is aggregated via a simple buffer; more sophisticated global modeling may yield further gains.
- Extensions to nonlinear SSMs (e.g., Mamba-style) are not explored.
- Parameter count scales linearly with the number of nodes, requiring sparsification for very large graphs.

## Related Work & Insights
- **vs. TGNv2**: TGNv2 updates node states with GRU, which cannot express PF. NAViS's gated linear design is more suitable for this task.
- **vs. DyGMamba / DyGFormer**: Memory-free methods rely on fixed-size buffers, truncating long-term history. NAViS's EMA-style decay retains unbounded memory.
- **vs. Mamba/S4**: SSMs have proven successful in sequence modeling; this work is the first to apply their theoretical analysis to temporal graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Three theoretical results precisely identify the problem; the SSM–TGNN connection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ TGB standard benchmarks + synthetic experiments + ablation studies, though the number of datasets is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, problem formulation is precise, and motivation is logically rigorous.
- Value: ⭐⭐⭐⭐⭐ Resolves the long-standing mystery of why heuristics outperform TGNNs on temporal graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)

</div>

<!-- RELATED:END -->
