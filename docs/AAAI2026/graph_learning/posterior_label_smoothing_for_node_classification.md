---
title: >-
  [Paper Note] Posterior Label Smoothing for Node Classification
description: >-
  [AAAI 2026][Graph Learning][Label Smoothing] This paper proposes PosteL (Posterior Label Smoothing), which derives soft labels from neighborhood label distributions via Bayesian posterior inference for node classification. The method naturally adapts to both homophilic and heterophilic graphs, achieving accuracy improvements in 76 out of 80 combinations across 8 backbone architectures and 10 datasets.
tags:
  - AAAI 2026
  - Graph Learning
  - Label Smoothing
  - Node Classification
  - Posterior Distribution
  - Homophilic/Heterophilic Graphs
  - Iterative Pseudo-labeling
date: 2026-05-08
content_hash: d933ac1367482d81
---

# Posterior Label Smoothing for Node Classification

**Conference**: AAAI 2026
**arXiv**: [2406.00410](https://arxiv.org/abs/2406.00410)
**Code**: [https://github.com/ml-postech/PosteL](https://github.com/ml-postech/PosteL)
**Area**: Graph Learning
**Keywords**: Label Smoothing, Node Classification, Posterior Distribution, Homophilic/Heterophilic Graphs, Iterative Pseudo-labeling

## TL;DR
This paper proposes PosteL (Posterior Label Smoothing), which derives soft labels from neighborhood label distributions via Bayesian posterior inference for node classification. The method naturally adapts to both homophilic and heterophilic graphs, achieving accuracy improvements in 76 out of 80 combinations across 8 backbone architectures and 10 datasets.

## Background & Motivation
**State of the Field**: Label smoothing (adding uniform noise to one-hot labels) is widely adopted in CV and NLP, yet remains largely unexplored for graph node classification. Soft labels from knowledge distillation are known to encode "dark knowledge" that improves student model performance.

**Limitations of Prior Work**: Existing graph label smoothing methods (SALS, ALS) assume that nodes tend to share labels with their neighbors, and directly aggregate neighborhood labels as soft labels. While effective on homophilic graphs, this strategy is harmful on heterophilic graphs, where neighboring nodes tend to belong to different classes from the target node.

**Root Cause**: A label smoothing method is needed that simultaneously accommodates homophilic graphs (neighbors share the same label) and heterophilic graphs (neighbors do not share the same label), whereas existing methods only handle the former.

**Starting Point**: "You can tell a person by the company they keep" — posterior distributions are derived from global statistics of neighborhood labels, such that the posterior favors the majority neighbor label in homophilic graphs and the minority neighbor label in heterophilic graphs.

**Core Idea**: Soft labels derived via Bayesian posterior distributions, conditioned on global label co-occurrence statistics, naturally adapt to both homophilic and heterophilic graphs.

## Method

### Overall Architecture
Given a graph $\mathcal{G}=(\mathcal{V},\mathcal{E},X)$ and training node labels, PosteL proceeds in two stages:
1. Derive soft labels for each training node from neighborhood labels and global statistics via Bayesian posterior inference.
2. Iteratively apply pseudo-labeling: update unlabeled node labels using model predictions, re-estimate global statistics, and re-derive improved soft labels.

### Key Designs

1. **Posterior Label Smoothing**:

    - **Function**: Derive neighborhood-based soft labels for each node.
    - **Mechanism**: $P(\hat{Y}_i=k|\{Y_j\}_{j\in\mathcal{N}(i)}) \propto P(\{Y_j\}|\hat{Y}_i=k) \cdot P(\hat{Y}_i=k)$. Assuming conditional independence among neighbor labels, the likelihood factorizes into a product of per-neighbor conditional probabilities. Both the conditional probabilities and the prior are estimated from global label co-occurrence statistics of the graph. The final soft label is $\alpha \cdot$ posterior $+ (1-\alpha) \cdot$ one-hot $+ \beta \cdot$ uniform.
    - **Design Motivation**: Lemma 1 proves that in homophilic graphs, majority neighbor labels elevate the posterior probability of the corresponding class; Lemma 2 proves that in heterophilic graphs, minority neighbor labels elevate the posterior probability instead. This provides principled adaptation to both graph types.

2. **Iterative Pseudo-labeling**:

    - **Function**: Augment label information using model predictions to improve global statistical estimates.
    - **Mechanism**: Train GNN → predict unlabeled nodes → update likelihoods and priors with pseudo-labels → re-derive soft labels → retrain.
    - **Design Motivation**: In sparse graphs, many nodes lack labeled neighbors; pseudo-labeling fills this information gap.

### Loss & Training
- Soft labels derived by PosteL replace one-hot labels for training any GNN backbone.
- Cross-entropy loss; 1000 training epochs with 200-epoch early stopping.
- 68/20/20 train/val/test split.

## Key Experimental Results

### Main Results
Across 8 backbone architectures × 10 datasets (80 combinations), PosteL yields improvements in 76 cases (95%). Representative results:

| Model + PosteL | Cora | CiteSeer | Chameleon | Squirrel | Texas |
|----------------|------|----------|-----------|----------|-------|
| GCN | Gain | Gain | Gain | Gain | Gain |
| GPR-GNN | Gain | Gain | Gain | Gain | Gain |
| BernNet | Gain | Gain | Gain | Gain | Gain |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| w/o iterative pseudo-labeling | Effective but below full model | Global statistics less accurate |
| SALS (naive aggregation) | Effective on homophilic, degrades on heterophilic | Does not adapt to heterophilic graphs |
| Uniform label smoothing | Marginal improvement | Does not exploit graph structure |
| PosteL (full) | 76/80 improvements | Best overall |

### Key Findings
- PosteL yields **more substantial gains on heterophilic graphs** than on homophilic ones — because naive methods completely fail on heterophilic graphs, while PosteL adapts naturally via posterior inference.
- Iterative pseudo-labeling contributes most on **sparse graphs** (e.g., Cornell, where 26% of nodes have no labeled neighbors).
- PosteL is effective across **all 8 backbone architectures**, demonstrating that it is a genuinely model-agnostic regularization technique.

## Highlights & Insights
- **Elegant use of posterior inference**: Label smoothing is elevated from "adding noise" to "deriving a posterior," with theoretical guarantees for dual adaptability to homophilic and heterophilic graphs.
- **Extremely simple implementation**: Only requires computing global label co-occurrence frequencies → applying Bayes' theorem → obtaining soft labels, with no additional parameters or training overhead.
- **95% success rate** (76/80) demonstrates exceptional robustness, suggesting that PosteL can serve as a standard add-on technique for GNN training.

## Limitations & Future Work
- The conditional independence assumption may not hold in dense graphs.
- Only first-order neighbors are considered; information from multi-hop neighborhoods is not exploited.
- Pseudo-label quality depends on the initial model performance; a poor initial model may introduce noisy pseudo-labels.

## Related Work & Insights
- **vs. SALS**: Directly aggregates neighborhood labels and is only applicable to homophilic graphs. PosteL handles heterophilic graphs naturally via posterior inference.
- **vs. ALS**: Similar to SALS but with adaptive refinement. PosteL is grounded in a probabilistic model with more rigorous theoretical foundations.
- **vs. Knowledge Distillation**: KD requires training a teacher model, whereas PosteL directly derives soft labels from graph structure at zero additional cost.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of posterior label smoothing is simple, elegant, and theoretically well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 models × 10 datasets with 80 combinations provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Theory and experiments are well integrated, with intuitive toy examples.
- **Value**: ⭐⭐⭐⭐ A simple and universally applicable GNN regularization technique with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](../../NeurIPS2025/graph_learning/geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

<!-- RELATED:END -->
