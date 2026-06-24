---
title: >-
  [Paper Note] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?
description: >-
  [ICLR 2026][Graph Learning][oversmoothing] This work points out that the widely used Dirichlet energy metric fails to correctly capture the oversmoothing phenomenon in practical GNN scenarios. It proposes using the numerical/effective rank (Erank) of feature representations as an alternative metric. Under the setting of independent training for each depth (from 2 to 24), Erank achieves an average correlation of 0.91 with accuracy (consistent positive direction)…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "oversmoothing"
  - "graph neural networks"
  - "Dirichlet energy"
  - "numerical rank"
  - "effective rank"
  - "rank collapse"
  - "GNN depth"
date: 2026-05-08
content_hash: b95075d7177e5c89
---

# Are We Measuring Oversmoothing in Graph Neural Networks Correctly?

**Conference**: ICLR 2026  
**arXiv**: [2502.04591](https://arxiv.org/abs/2502.04591)  
**Area**: Graph Neural Networks / Theoretical Analysis  
**Keywords**: oversmoothing, graph neural networks, Dirichlet energy, numerical rank, effective rank, rank collapse, GNN depth  

## TL;DR

This work points out that the widely used Dirichlet energy metric fails to correctly capture the oversmoothing phenomenon in practical GNN scenarios. It proposes using the numerical/effective rank (Erank) of feature representations as an alternative metric. Under the setting of independent training for each depth (from 2 to 24), Erank achieves an average correlation of 0.91 with accuracy (consistent positive direction), whereas Dirichlet energy averages only −0.72 and its correlation direction fluctuates across datasets (failing particularly on large-scale OGB-Arxiv). Furthermore, it theoretically proves that for linear GNNs and a family of non-linear GNNs with non-negative weights, the numerical rank of the feature matrix converges to 1 (rank collapse), thereby redefining oversmoothing as rank collapse rather than eigenvector alignment.

## Background & Motivation

**Oversmoothing is the core bottleneck of GNNs**: As the number of layers increases, node representations tend to become uniform and lose discriminability. This is considered the primary reason why GNNs cannot benefit from depth like other deep networks.

**Dominance of Dirichlet energy**: Existing literature almost universally uses Dirichlet energy (the sum of squared differences between neighboring node representations) as the standard for measuring oversmoothing, and numerous mitigation methods have been designed based on this.

**Disconnect between metric and performance**: In practice, it has been observed that changes in Dirichlet energy are sometimes inconsistent with changes in model performance — a decrease in Dirichlet energy does not necessarily imply a drop in performance, and vice versa.

**Lack of theoretical understanding**: Existing oversmoothing theories mainly analyze the convergence behavior of Dirichlet energy under specific graph topologies but have not questioned the validity of the metric itself.

**Failure on heterophilic graphs**: In heterophilic graphs, neighboring nodes often have different labels. High Dirichlet energy is not necessarily undesirable; it may instead indicate that the model has learned discriminative representations. This renders the simple interpretation of "lower energy equals more oversmoothing" completely invalid.

**Need for better metrics**: The GNN community requires an oversmoothing metric that is more consistent with downstream task performance to correctly guide architectural design and regularization strategies.

## Method

### Overall Architecture

Rather than proposing a new model, this paper redefines oversmoothing from a linear algebra perspective: it views it as a rank collapse of the node feature matrix. As depth increases, the effective rank of the feature matrix approaches 1, and all node representations are compressed into a single low-dimensional subspace. Centered on this definition, the paper replaces the mainstream Dirichlet energy with rank metrics (numerical rank / effective rank). It first demonstrates that these metrics align better with performance across multiple architectures and datasets, and then theoretically proves that rank collapse is an inevitable result in linear GNNs and families of non-linear GNNs with non-negative weights.

### Key Designs

**1. Replacing Dirichlet energy with continuous relaxations of rank: Directly characterizing dimension collapse in representation space**

Dirichlet energy measures the sum of squared differences between adjacent node representations. Essentially, it measures the deviation of features from the principal eigenspace of the message-passing matrix $A$. A low value only indicates that features are aligning with the principal eigenvector, failing to show if the entire representation space is actually collapsing. Furthermore, it only truly equates to oversmoothing when it converges exactly to 0; in reality, a "drop by two orders of magnitude but far from 0" is uninterpretable. This paper uses continuous relaxations of matrix rank. Since rank itself is a discrete count of non-zero singular values and is non-differentiable, three relaxations are used: given singular values $\sigma_1>\sigma_2>\dots$, let $p_k = \sigma_k / \sum_i \sigma_i$,

$$\text{StabRank}(X)=\frac{\|X\|_*^2}{\|X\|_F^2},\quad \text{NumRank}(X)=\frac{\|X\|_F^2}{\|X\|_2^2},\quad \text{Erank}(X)=\exp\!\left(-\sum_k p_k\log p_k\right).$$

Effective rank (Erank) is the exponential of the normalized entropy of the singular value spectrum. It intuitively counts the "number of meaningful dimensions" in the feature matrix: it approaches 1 when all node representations collapse onto a single direction (complete oversmoothing), and increases as the spectrum spreads and representations become more diverse. Compared to Dirichlet energy, rank metrics offer three fundamental advantages: (a) scale invariance, remaining meaningful even if features go to zero or explode; (b) independence from pre-defined fixed feature subspaces, capturing convergence to **any** low-dimensional subspace; (c) the ability to diagnose low rank in shallow networks before reaching rank 1, providing an early warning of oversmoothing.

**2. Correlation diagnosis under the realistic setting of "independent training per depth": Debunking the failure of Dirichlet energy**

A qualified oversmoothing metric should be strongly and consistently correlated with downstream performance. The paper points out that previous evidence mostly came from observing Dirichlet decay layer-by-layer within a single, very deep, **untrained or undertrained** network. Such decay merely reflects magnitude contraction caused by small initialization, not the oversmoothing of a trained model. This paper adopts a more realistic setting: training a GNN independently for each depth $l=2,\dots,24$, and calculating the correlation coefficient between the metrics (in log scale) and node classification accuracy. (Erank and NumRank are shifted by -1 to ensure they also approach zero during oversmoothing for easier comparison). Results show Erank has an average correlation of approximately 0.91 with consistent positive direction. Dirichlet energy averages only $-0.72$, with a direction opposite to Erank, and oscillates between datasets (e.g., $-0.79$ on Cora-GCN but $+0.77$ on OGB-Arxiv-GCN), being particularly unreliable on heterophilic graphs (Squirrel / Chameleon / Amazon-Ratings) and large-scale graphs. This indicates that Dirichlet energy can no longer serve as a credible proxy for oversmoothing.

**3. Theoretical proof of rank collapse: Establishing "depth implies oversmoothing" as a theorem (within provable architecture families)**

To ground the new definition of "rank collapse = oversmoothing," this paper proves that for linear GCNs (Thm 5.1) and non-linear GNNs where weights are element-wise non-negative and the activation function shares the principal eigenvector with the message-passing matrix (Thm 5.3, using non-linear Perron–Frobenius theory), the numerical rank of the feature matrix converges to 1 as the number of layers $l\to\infty$. The intuition is that repeated application of the normalized adjacency matrix $A$ causes features to grow fastest along the principal eigenvector $u$ while other directions are relatively suppressed, leading the non-aligned component $\|(I-\mathcal{P})X\|_F / \|X\|_2 \to 0$ (where $\mathcal{P}=uu^\top$), and thus numerical rank $\to 1$. Crucially, this convergence is **independent of weight magnitude** — even if features do not approach zero and Dirichlet-style metrics show "no change," the rank will still collapse. This also explains why simple scaling or normalization techniques cannot fundamentally cure oversmoothing.

> ⚠️ This theorem is restricted to linear GNNs or non-linear GNNs with "non-negative weights + shared eigenvectors" and does not apply to all GNNs (see Appendix D of the original paper for counterexamples and discussion on general non-linear cases).

## Key Experimental Results

### Main Results: Correlation between Metrics and Accuracy (Depths 2–24, Independent Training)

Correlation coefficients measure the relationship between accuracy and the log of each metric. The expected direction is positive (lower metric implies more oversmoothing and lower accuracy). Erank and NumRank are shifted by -1 for directional consistency. The table below shows representative datasets for GCN; the average behavior covers all 14 groups (7 datasets × {GCN, GAT}).

| Dataset (GCN) | $E_{\mathrm{Dir}}$ | MAD | Erank | NumRank |
|---------------|-------------------|-----|-------|---------|
| Cora | −0.79 | −0.25 | **0.97** | 0.59 |
| Citeseer | −0.84 | −0.72 | **0.97** | 0.68 |
| Pubmed | −0.91 | 0.62 | **0.95** | 0.93 |
| Squirrel (Heterophilic) | −0.78 | −0.82 | 0.63 | **0.96** |
| Chameleon (Heterophilic) | −0.92 | −0.88 | **0.94** | 0.90 |
| Amazon-Ratings (Heterophilic) | −0.93 | 0.92 | **0.93** | 0.80 |
| OGB-Arxiv | **+0.77** (Wrong direction) | 0.28 | **0.97** | 0.91 |
| **Average (7 Datasets × GCN/GAT)** | **−0.72** | 0.16 | **0.91** | 0.84 |

### Accuracy Retention Ratio: 2 → 24 Layers

Retention ratio = Accuracy at 24 layers / Accuracy at 2 layers. Lower values indicate more severe performance degradation (and thus more severe oversmoothing).

| Dataset (GCN) | Accuracy Retention Ratio (24L / 2L) |
|---------------|------------------------------------|
| OGB-Arxiv | 0.09 (Most severe degradation) |
| Cora | 0.27 |
| Citeseer | 0.44 |
| Pubmed | 0.52 |
| Chameleon (Heterophilic) | 0.62 |
| Squirrel (Heterophilic) | 0.85 |
| Amazon-Ratings (Heterophilic) | 0.86 |

### Key Findings

1. **Dirichlet energy direction is inconsistent and averaged "backward"**: Its correlation with accuracy averages $-0.72$, which is opposite to expectations (e.g., Erank), and its sign switches between datasets (Cora-GCN $-0.79$ vs OGB-Arxiv-GCN $+0.77$). It is particularly ineffective on heterophilic and large-scale graphs.
2. **Erank shows strong consistency**: It maintains a high positive correlation (average 0.91) across almost all datasets and architectures; NumRank follows (average 0.84). Both significantly outperform Dirichlet energy and MAD (average only 0.16).
3. **Previous evidence settings were biased**: Past work mostly observed Dirichlet decay across layers in untrained or undertrained very deep networks, which was merely magnitude contraction from small initializations. When trained independently at each depth, Dirichlet decouples from performance, while rank metrics continue to track performance closely.
4. **Degradation correlates with graph homophily**: Homophilic graphs (OGB-Arxiv / Cora) show low accuracy retention (0.09–0.27) as they deepen, whereas some heterophilic graphs (Squirrel / Amazon-Ratings ≈ 0.85) are more resistant to depth.
5. **Metric choice influences research conclusions**: Using Dirichlet energy as a criterion might lead researchers to misperceive an architecture as "not having oversmoothing," thereby directing improvements toward the wrong path.

## Highlights & Insights

1. **Challenging fundamental assumptions**: The work questions the widely accepted Dirichlet energy metric in the GNN community, representing an "Emperor’s New Clothes" style contribution.
2. **Superior alternative**: Erank is significantly better than Dirichlet energy both theoretically and empirically, while remaining simple to implement.
3. **Conceptual redefinition**: Redefining oversmoothing from "neighboring node features converging" (eigenvector alignment) to "feature space rank collapse" provides a more fundamental perspective.
4. **Practical guidance**: Correct metrics will guide the community toward developing more effective oversmoothing mitigation strategies.
5. **Criticality of large-scale datasets**: Differences between the metrics are minor on small datasets but become massive on large-scale real-world graphs.

## Limitations & Future Work

1. **Causality not fully established**: High correlation does not imply causality. It remains debatable whether rank collapse is the sole cause of performance degradation.
2. **Bounds of theoretical and empirical coverage**: The convergence theorem only covers linear GNNs and non-linear GNNs with "non-negative weights + shared eigenvectors." Whether rank collapse is inevitable for general non-linear architectures remains an open question. While heterophilic graphs were tested, Erank occasionally showed lower correlation (e.g., 0.63 for Squirrel-GCN), and its mechanism there requires further exploration.
3. **Lack of new mitigation strategies**: The paper focuses on diagnostic metrics and does not propose specific mitigation algorithms based on the new rank perspective.
4. **Graph-level tasks not addressed**: All experiments targeted node classification; the behavior of oversmoothing in graph-level classification tasks was not explored.

## Related Work & Insights

- **Oversmoothing Theory**: Li et al. (2018) first formally defined oversmoothing; the Dirichlet energy framework of Cai & Wang (2020).
- **Oversmoothing Mitigation**: DropEdge (Rong et al., 2020), PairNorm (Zhao & Akoglu, 2020), DeeperGCN (Li et al., 2020).
- **Effective Rank**: The definition of effective rank proposed by Roy & Vetterli (2007).
- **Deep GNNs**: Design of deep architectures such as GCNII (Chen et al., 2020) and RevGNN (Li et al., 2021).

## Rating

- Novelty: ⭐⭐⭐⭐ Challenges fundamental metric assumptions and introduces the rank collapse perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple datasets, architectures, and theoretical/empirical fronts.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical argumentation and well-designed experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a necessary correction to foundational metrics in the GNN community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ICLR 2026\] WATS: Wavelet-Aware Temperature Scaling for Reliable Graph Neural Networks](wats_wavelet-aware_temperature_scaling_for_reliable_graph_neural_networks.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] Learning from Historical Activations in Graph Neural Networks](learning_from_historical_activations_in_graph_neural_networks.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
