---
title: >-
  [Paper Note] Adjusted Count Quantification Learning on Graphs
description: >-
  [NeurIPS 2025][Quantification Learning] This paper extends the classical Adjusted Classify & Count (ACC) quantification method to graph-structured data…
tags:
  - "NeurIPS 2025"
  - "Quantification Learning"
  - "Graph Data"
  - "Covariate Shift"
  - "Importance Sampling"
  - "Non-Homophily"
date: 2026-05-08
content_hash: 061936c9a2f8ac58
---

# Adjusted Count Quantification Learning on Graphs

**Conference**: NeurIPS 2025
**arXiv**: [2503.09395](https://arxiv.org/abs/2503.09395)  
**Code**: N/A  
**Area**: Others
**Keywords**: Quantification Learning, Graph Data, Covariate Shift, Importance Sampling, Non-Homophily

## TL;DR

This paper extends the classical Adjusted Classify & Count (ACC) quantification method to graph-structured data, proposing two techniques — Structural Importance Sampling (SIS) and Neighborhood-aware ACC (N-ACC) — to address structural covariate shift and non-homophilous edges in graph quantification, respectively.

## Background & Motivation

**Quantification Learning** is a task closely related to, yet fundamentally distinct from, classification: given a set of instances, the goal is not to predict each instance's label, but rather to estimate the **label distribution over the entire set**. For example, in a social network, one may wish to know the proportion of users holding different political views within a community, rather than classifying each individual user.

**Quantification on Graphs**: When instances correspond to nodes in a graph, quantification presents unique challenges:
- Edges (relations) exist between nodes, violating the i.i.d. assumption of conventional quantification methods
- Graph topology influences the distribution of node features
- Homophily — the tendency of connected nodes to share the same label — is a prevalent property of graph data

**Limitations of Prior Work**:

**Inapplicability of ACC's prior probability shift assumption**: Classical ACC assumes only prior probability shift between the test and training sets; however, in graph data, the influence of graph structure introduces **covariate shift**, making this assumption invalid.

**Limitations of node clustering approaches**: Prior graph quantification methods rely solely on node clustering and are unable to leverage label information.

**Impact of non-homophilous edges**: The presence of a large proportion of non-homophilous (heterophilous) connections degrades quantification accuracy.

## Method

### Overall Architecture

The proposed method builds upon the classical **Classify & Count (CC)** paradigm:
1. Train a node classifier
2. Run the classifier on the target set to obtain predicted labels
3. Aggregate the predicted label distribution as the quantification result
4. Apply an adjustment step to correct for systematic bias in the classifier

### Key Designs

**1. Analysis of ACC's Failure on Graphs**

The adjustment formula of classical ACC relies on the prior probability shift assumption:

$$\hat{p}(y) = M^{-1} \hat{p}(\hat{y})$$

where $M$ is the confusion matrix. This formula assumes $p_{test}(x|y) = p_{train}(x|y)$ (prior probability shift). In graph data, however, sampling bias induced by graph structure and community organization may cause the feature distribution $p_{test}(x)$ of the test subgraph to exhibit **structural covariate shift** relative to the training graph — i.e., $p_{test}(x|y) \neq p_{train}(x|y)$.

**2. Structural Importance Sampling (SIS)**

SIS constitutes the core contribution of the paper and addresses structural covariate shift. The central idea is:

$$\hat{p}(y) = \frac{1}{|S|} \sum_{v \in S} w(v) \cdot \mathbb{1}[\hat{y}_v = y]$$

where the weight $w(v)$ is based on the density ratio of node $v$ under the test and training distributions:

$$w(v) = \frac{p_{test}(x_v)}{p_{train}(x_v)}$$

The key challenge in SIS lies in estimating this density ratio on graphs. The paper proposes using graph topological features (e.g., degree centrality, PageRank, clustering coefficient) to estimate structural importance weights, such that:
- Nodes more "typical" of the target subgraph receive higher weights
- Nodes that deviate substantially from the training distribution are appropriately down-weighted

**3. Neighborhood-aware ACC (N-ACC)**

When the graph contains a large proportion of non-homophilous edges, neighbor label information may be misleading. N-ACC addresses this by:

- Introducing a neighborhood label consistency metric: $h(v) = \frac{|\{u \in \mathcal{N}(v): y_u = y_v\}|}{|\mathcal{N}(v)|}$
- Applying distinct confusion matrices for adjustment to nodes with different levels of neighborhood consistency
- Using standard ACC in high-homophily regions and a modified ACC in low-homophily regions

### Loss & Training

The underlying node classifier is trained using a standard graph neural network (GNN) objective:

$$L_{cls} = -\sum_{v \in V_{train}} \log p(y_v | x_v, G)$$

The quantification step itself requires no additional training and operates as a post-processing procedure applied to classifier outputs. Density ratio estimation in SIS is performed via kernel density estimation or classification-based methods (e.g., logistic regression) over graph topological features.

## Key Experimental Results

### Main Results

Quantification error (MAE, lower is better) across multiple graph datasets:

| Method | Cora | CiteSeer | PubMed | Amazon | ogbn-arxiv | Avg. MAE |
|--------|------|----------|--------|--------|------------|----------|
| CC (Baseline) | 0.142 | 0.168 | 0.125 | 0.189 | 0.156 | 0.156 |
| ACC | 0.098 | 0.127 | 0.091 | 0.152 | 0.124 | 0.118 |
| Node Clustering | 0.135 | 0.151 | 0.118 | 0.173 | 0.148 | 0.145 |
| SIS (Ours) | 0.072 | 0.095 | 0.068 | 0.114 | 0.093 | **0.088** |
| N-ACC (Ours) | 0.085 | 0.108 | 0.079 | 0.128 | 0.105 | 0.101 |
| SIS + N-ACC (Ours) | **0.065** | **0.088** | **0.061** | **0.106** | **0.087** | **0.081** |

### Ablation Study

Performance under varying degrees of covariate shift (ogbn-arxiv dataset):

| Shift Level | CC MAE | ACC MAE | SIS MAE | Gain (vs. ACC) |
|-------------|--------|---------|---------|----------------|
| Mild (KL<0.1) | 0.085 | 0.062 | 0.058 | 6.5% |
| Moderate (KL 0.1–0.5) | 0.156 | 0.124 | 0.093 | 25.0% |
| Severe (KL>0.5) | 0.234 | 0.198 | 0.127 | 35.9% |
| Extreme (KL>1.0) | 0.312 | 0.275 | 0.164 | 40.4% |

### Key Findings

1. **SIS consistently outperforms ACC across all datasets**: Average MAE decreases from 0.118 to 0.088, a relative reduction of 25.4%.
2. **The advantage of SIS grows with covariate shift severity**: Under extreme shift, the relative improvement reaches 40.4%.
3. **N-ACC excels on heterophilous graphs**: The largest gains are observed on graphs with lower homophily (e.g., Amazon).
4. **The SIS + N-ACC combination achieves the best performance**: The two techniques are complementary, yielding further error reduction when combined.
5. **Conventional node clustering methods perform poorly**: They underperform even simple ACC, underscoring the necessity of directly leveraging label information.

## Highlights & Insights

1. **Rigorous theoretical analysis**: The paper formally characterizes why ACC fails on graphs by demonstrating the inapplicability of the prior probability shift assumption.
2. **Principled method design**: SIS directly addresses covariate shift, while N-ACC targets non-homophily, each with a clearly defined scope of applicability.
3. **Fills a research gap**: This work presents the first effective graph quantification method under structural covariate shift.
4. **High practical value**: In applications such as social network analysis and public opinion monitoring, group-level quantification is often of greater practical relevance than individual-level prediction.

## Limitations & Future Work

1. **Accuracy of density ratio estimation**: The performance of SIS depends on the quality of density ratio estimation, which may be unreliable in high-dimensional feature spaces.
2. **Computational overhead**: SIS requires an additional density estimation step, which may be costly for very large-scale graphs.
3. **Dependence on homophily estimation**: The improvements offered by N-ACC rely on accurate estimation of the local homophily level.
4. **Restricted to node-level quantification**: Edge-level and subgraph-level quantification are not addressed.
5. **Temporal graphs not covered**: In dynamic graphs, distributional shift may evolve over time, which the current methods cannot directly handle.

## Related Work & Insights

- **Classify & Count family**: Classical quantification methods including CC, ACC, and PCC
- **Graph Neural Networks**: GCN, GAT, and GraphSAGE serve as underlying classifiers
- **Covariate shift correction**: Importance weighting has been widely adopted in conventional machine learning
- **Distribution shift on graphs**: Out-of-distribution (OOD) generalization is an active research direction in graph learning
- **Insight**: Adapting classical statistical learning concepts — such as importance sampling and covariate shift correction — to graph-structured data constitutes a fruitful and promising research paradigm

## Rating

- Novelty: ⭐⭐⭐⭐ (First work to address covariate shift in graph quantification)
- Technical Depth: ⭐⭐⭐⭐⭐ (Rigorous theoretical analysis with complete methodological derivations)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated across multiple datasets with analysis under varying shift levels)
- Writing Quality: ⭐⭐⭐⭐ (Complete 19-page paper with clear organization)
- Overall: ⭐⭐⭐⭐ (Solid theoretical contribution with potential impact on the quantification learning community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Optimized Learned Count-Min Sketch](optimized_learned_count-min_sketch.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics](uncertainty_quantification_for_reduced-order_surrogate_models_applied_to_cloud_m.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)

</div>

<!-- RELATED:END -->
