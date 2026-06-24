---
title: >-
  [Paper Note] Forest-Based Graph Learning for Semi-Supervised Node Classification
description: >-
  [ICLR 2026][Graph Learning][Spanning trees] This work reinterprets message passing on graphs as "transmission across multiple spanning trees (forests)." By using homophily-guided sampling to select high-quality trees and a linear-time tree aggregator, the method achieves a global receptive field with $O(n+m)$ complexity, outperforming both deep GNNs and Graph Transformers in semi-supervised node classification.
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Spanning trees"
  - "Forest"
  - "Long-range info propagation"
  - "Homophily"
  - "Linear complexity"
  - "Graph Transformer"
date: 2026-05-08
content_hash: 71550b22e35b4e10
---

# Forest-Based Graph Learning for Semi-Supervised Node Classification

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5asbtzIVpS](https://openreview.net/forum?id=5asbtzIVpS)  
**Code**: [https://anonymous.4open.science/r/FGL/](https://anonymous.4open.science/r/FGL/)  
**Area**: Graph Learning / Semi-Supervised Node Classification  
**Keywords**: Spanning trees, Forest, Long-range info propagation, Homophily, Linear complexity, Graph Transformer  

## TL;DR
This work reinterprets message passing on graphs as "transmission across multiple spanning trees (forests)." By using homophily-guided sampling to select high-quality trees and a linear-time tree aggregator, the method achieves a global receptive field with $O(n+m)$ complexity, outperforming both deep GNNs and Graph Transformers in semi-supervised node classification.

## Background & Motivation
**Background**: To capture long-range knowledge, current graph neural networks follow two main paths: deep local models (e.g., GCNII) expand receptive fields by stacking many layers, while shallow global models (Graph Transformers like Graphormer/DIFFormer) cover all node pairs via 1-2 layers of global attention.

**Limitations of Prior Work**: Both paths are bottlenecked by complexity. Deep GNNs suffer from many layers, serial execution, and over-smoothing; Graph Transformers face quadratic complexity $O(n^2)$, leading to OOM on large graphs. Attempted sparsification techniques (adaptive selection, graph rewiring) often sacrifice global coverage or rely on complex strategies that may lose vital node interactions.

**Key Challenge**: The authors abstract this into a formula: **Total Cost = Single Structure Cost × Number of Structures**. Local primitives (1-hop neighborhoods, short random walks) have cheap single-structure costs but require a massive number of structures to cover long distances; global operators use few structures but have explosive single-structure costs. Balancing these two factors is an inherent flaw in existing paradigms.

**Goal**: Find an **intermediate hierarchical structure** that suppresses both factors simultaneously to break the "cost-efficiency vs. global receptive field" deadlock.

**Core Idea**: **[Spanning trees as the sparsest global structures]** A spanning tree is the smallest subgraph connecting all nodes. When the number of structures is limited, it is the "simplest yet globally covering" structure. Since a single tree contains incomplete knowledge, a **forest (multiple complementary spanning trees)** is used. Graph learning is reformulated as "information transmission over forests," utilizing **homophily-guided sampling** to ensure tree quality and a **linear-time tree aggregator** for efficiency.

## Method

### Overall Architecture
The FGL (Forest-based Graph Learning) pipeline consists of four steps: **Preprocessing** to augment the graph for connectivity and homophily → **Tree Sampler** to draw $N_T$ spanning trees from a distribution biased toward high homophily → **Tree Aggregator** to propagate global messages in linear time on each tree → **Tree Fuser** to merge multi-tree results with local information for final classification.

```mermaid
flowchart LR
    G[Original Graph G] --> P[Preprocessing: <br/>Pseudo-labels + kNN Augmentation]
    P --> Ghat[Augmented Graph Ĝ: <br/>Connected & More Homophilous]
    Ghat --> HE[Homophily Estimator: <br/>Local Attention Edge Scoring]
    HE --> S[Tree Sampler: <br/>Wilson's Algo for N_T Trees]
    S --> A[Tree Aggregator: <br/>Two Recursions O(n)]
    A --> F[Tree Fuser: <br/>RowNorm + Mean + Local Residual]
    F --> H[Final Embedding H'': <br/>Linear Prediction]
```

### Key Designs

**1. Homophily-guided Tree Sampler: Biasing the forest toward "homophilous connections."** For node classification, a "good" tree has a **high homophily rate** (edges connecting similar nodes) and **diversity** (avoiding excessive overlap). The tree score is defined as the product of edge scores, inducing a distribution $P_{\hat G}(T)=\prod_{e\in T}s(e)\big/\sum_{T\subseteq\hat G}\prod_{e\in T}s(e)$. By assigning high scores to homophilous edges and low scores to heterophilous ones, the distribution biases toward high-quality trees. Edge scores are provided by a **local attention homophily estimator**: $\alpha_{i\to j}=\mathrm{softmax}_j(Q_iK_j^\top/\sqrt c)$, trained with pseudo-labels $Y'$. Edge scores are $s(e)=(\alpha_{i\to j}+\alpha_{j\to i})/2$. Finally, $N_T$ trees are independently sampled using a weighted **Wilson’s algorithm** at nearly $O(n)$ cost per tree. Theoretically (Theorem 2), the authors prove that as the score ratio $\Delta=p/q$ (homophilous/heterophilous score) increases, the expected homophily rate of the distribution rises monotonically toward an upper bound—**a more accurate estimator induces a higher quality tree distribution**.

**2. Linear-time Tree Aggregator: Compressing quadratic interactions via dual recursions.** The core observation is that for adjacent nodes $u,v$ on a tree, global aggregated messages **differ only in the direction of one edge**. If the aggregator $f_{\mathrm{Agg}}$ satisfies "composability/decomposability" properties—existing operators $\mathcal M_+/\mathcal M_-$ such that merged results can be reversed (Property I/II)—then global propagation can be completed via two recursions (Theorem 1). Specifically, a bottom-up recursion computes subtree aggregations $S_u=f_{\mathrm{Agg}}(\{S_v\}_{v\in \mathrm{Child}(u)}\cup\{g(H_u)\})$ (Recursion I), followed by a top-down recursion that removes self-contribution via $\mathcal M_-$ and adds parent information via $\mathcal M_+$ to obtain $H'_v=\mathcal M_+(S_v,\mathcal M_-(H'_{\mathrm{Fa}(v)},S_v))$ (Recursion II). The implementation uses weighted sum/difference as linear variants:

$$S_u=\sum_{v\in\mathrm{Child}(u)}(\alpha_{v\to u}W_A)S_v+W_B H_u,\quad H'_v=S_v+\alpha_{\mathrm{Fa}(v)\to v}W_A\big(H'_{\mathrm{Fa}(v)}-\alpha_{v\to\mathrm{Fa}(v)}W_A S_v\big)$$

This design achieves **global propagation equivalent to quadratic node-pair interactions with linear runtime**. It supports parallelism across trees and within single trees (by restructuring trees to be "short and wide").

**3. Tree Fuser: Recovering local knowledge + stabilizing representations.** Since single trees are sparse, local information is first calculated as $H=(\beta_1\hat A_{\hat G}+\beta_2\alpha+(1-\beta_1-\beta_2)I)^{K_L}XW_H$ (low-order propagation with $K_L\le2$). After running the aggregator on $N_T$ trees to get $H'^{(k)}$, RowNorm is applied for numerical stability, and the results are averaged for global information $H'=\mathrm{Mean}\{\mathrm{RowNorm}(H'^{(k)})\}$. Finally, a residual coefficient $\gamma$ balances global and local components: $H''=(1-\gamma)H'+\gamma H$. Training consumes only $O((n+m)Kd)$ time and space per epoch.

## Key Experimental Results

### Main Results
On 9 datasets (Homophilous: Cora/Citeseer/Pubmed/ArXiv; Heterophilous: Actor/Cornell/Texas/Wisconsin/Flickr), FGL achieved an **average rank of 1.22** against 26 baselines:

| Method | Category | Cora | Citeseer | Pubmed | Actor | Cornell | Texas | Wisconsin | ArXiv | Flickr | Avg.Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GCNII | DeepGNN | 85.34 | 73.24 | 79.88 | 34.64 | 74.61 | 69.19 | 70.31 | 51.91 | 41.79 | 8.78 |
| SGFormer | GT | 82.38 | 71.82 | 80.64 | 37.80 | 68.65 | 78.92 | 80.00 | 45.73 | 40.13 | 7.22 |
| DIFFormer | GT | 83.32 | **74.46** | 78.16 | 34.51 | 60.00 | 68.11 | 63.92 | 53.60 | 44.25 | 10.56 |
| GraphMamba | Mamba | 54.36 | 58.98 | 70.90 | 36.05 | 74.05 | 77.29 | 80.39 | 33.59 | 42.30 | 13.89 |
| **Ours** | **Forest** | **85.46** | 74.42 | **81.00** | **39.88** | **83.24** | **91.89** | **86.27** | **56.47** | **47.22** | **1.22** |

**Gain**: Average accuracy improved by 16.2%, 16.1%, and 11.9% compared to GT, DIFFormer, and GCNII, respectively. On Wisconsin, the lead over these baselines reached 20-35%.

### Ablation Study

| No. | Variant | Cora | Pubmed | Actor | Texas | Wisconsin | Flickr |
|---|---|---|---|---|---|---|---|
| (1) | w.o. Global Sub-module | 80.00 | 76.13 | 34.73 | 82.88 | 83.92 | 39.63 |
| (2) | w.o. Local Sub-module | 82.18 | 77.48 | 35.08 | 69.93 | 75.49 | 32.17 |
| (3) | Uniform Tree Sampling | 83.63 | 78.45 | 36.13 | 82.58 | 84.80 | 42.77 |
| (4) | Single Homophily-guided Tree | 83.73 | 78.55 | 36.32 | 84.83 | 85.29 | 42.96 |
| (5) | **FGL-Full** | **85.46** | **81.00** | **39.88** | **91.89** | **86.27** | **47.22** |

Findings: (4) vs. (3) shows homophily guidance is critical; (5) vs. (4) shows forests capture complementary topology better than single trees.

### Key Findings
- **Efficiency (sec/epoch)**: On ArXiv, Ours takes 0.246s vs. ANS-GT's 24.5s; it is 2-5x faster than DIFFormer/GCNII. Several GTs suffered OOM on large graphs.
- **Tree Count**: 6–10 trees is the optimal range across datasets. A small number of trees captures the intrinsic graph structure, while more trees bring diminishing returns.
- **Estimator Accuracy**: Accuracy increases monotonically with the homophily edge score $p$, validating Theorem 2.

## Highlights & Insights
- **Paradigm Abstraction**: The "Total Cost" formula simplifies the GNN vs. Transformer trade-off, identifying spanning trees as the optimal "low structure count, low structure cost" solution.
- **Theoretical-Empirical Loop**: Theorem 2 (accuracy vs. distribution quality) and Theorem 1 (linear propagation) create a rare "provable, runnable, and verifiable" trinity.
- **Aggregator Versatility**: The tree aggregator can swap in linear attention, linear RNNs, or SSMs, providing high extensibility.
- **Heterophily Strength**: Exceptional performance on heterophilous graphs suggests that homophily-guided sampling effectively identifies useful structural paths in "difficult" graphs.

## Limitations & Future Work
- **Dependence on Pseudo-labels**: If pseudo-labels are extremely noisy or sparse, the homophily estimator may degrade, draggin down the tree quality.
- **Hyperparameter Sensitivity**: $N_T$, $\beta$, $K_L$, $\gamma$, and $k$ require tuning; cross-dataset robustness requires more validation.
- **Task Scope**: Currently limited to semi-supervised node classification; extensions to graph classification and link prediction are needed.

## Related Work & Insights
- **Deep Local Models**: Receptive fields expand by stacking layers, but serial bottlenecks remain.
- **Shallow Global Models (GTs)**: Direct global modeling is $O(n^2)$; sparsification often hurts inductive bias.
- **Generative/Random Walk Methods**: Using spanning trees as carriers for long-range propagation leverages classical graph theory (Wilson's algorithm) to innovate GNN architectures.
- **Insight**: When a field faces a "Performance vs. Efficiency" trade-off, rather than approximating at the extremes, it is effective to find an intermediate structure—like the "forest"—that minimizes both cost factors by design.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A new paradigm; homophily-guided forest transmission is a distinct departure from incremental improvements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Deep baseline comparison across 9 datasets; lacks only extremely large-scale (billion-edge) graph testing.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation via the "Cost Formula" is highly compelling; clear theoretical-to-empirical flow.
- **Value**: ⭐⭐⭐⭐⭐ Significant improvements in performance and efficiency suggests this could become a primary methodology for long-range graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](../../NeurIPS2025/graph_learning/geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[ICLR 2026\] Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors](learning_posterior_predictive_distributions_for_node_classification_from_synthet.md)
- [\[ICLR 2026\] Topological Anomaly Quantification for Semi-Supervised Graph Anomaly Detection](topological_anomaly_quantification_for_semi-supervised_graph_anomaly_detection.md)
- [\[ICLR 2026\] Low-Rank Few-Shot Node Classification by Node-Level Graph Diffusion](low-rank_few-shot_node_classification_by_node-level_graph_diffusion.md)
- [\[ICLR 2026\] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding](harmonygnns_harmonizing_heterophily_and_homophily_in_gnns_via_self-supervised_no.md)

</div>

<!-- RELATED:END -->
