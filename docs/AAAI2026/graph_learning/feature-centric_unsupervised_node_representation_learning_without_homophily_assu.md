---
title: >-
  [Paper Note] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption
description: >-
  [AAAI 2026][Graph Learning][unsupervised node representation learning] This paper proposes FUEL, a method that adaptively learns the degree of graph convolution usage through a node-feature-centric clustering scheme…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "unsupervised node representation learning"
  - "graph convolution"
  - "non-homophilic graphs"
  - "clustering"
  - "feature-centric"
date: 2026-05-08
content_hash: 72db66f858b51876
---

# Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption

**Conference**: AAAI 2026
**arXiv**: [2512.15112](https://arxiv.org/abs/2512.15112)  
**Authors**: Sunwoo Kim, Soo Yong Lee, Kyungho Kim, Hyunjin Hwang, Jaemin Yoo, Kijung Shin (KAIST)
**Code**: [GitHub](https://github.com/kswoo97/unsupervised-non-homophilic)  
**Area**: Graph Learning
**Keywords**: unsupervised node representation learning, graph convolution, non-homophilic graphs, clustering, feature-centric

## TL;DR

This paper proposes FUEL, a method that adaptively learns the degree of graph convolution usage through a node-feature-centric clustering scheme, achieving high-quality unsupervised node representations on both homophilic and non-homophilic graphs without any homophily assumption.

## Background & Motivation

### State of the Field
Unsupervised node representation learning aims to obtain meaningful node embeddings without relying on node labels. Graph convolution is its core operation—encoding node features and graph topology by aggregating neighbor information. On **homophilic graphs** (where similar nodes tend to connect), graph convolution performs well; however, on **non-homophilic graphs** (where dissimilar nodes frequently connect), excessive graph convolution causes nodes that should be distinct to acquire overly similar embeddings, degrading representation quality.

### Limitations of Prior Work
- **Supervised methods already have solutions**: GPR-GNN, GADC, and others learn the degree of graph convolution usage via downstream task losses (e.g., cross-entropy), adaptively adjusting the weight of each-order adjacency matrix.
- **Unsupervised methods lack analogous mechanisms**: Without label supervision, graph convolution weights cannot be directly optimized. Existing non-homophilic unsupervised methods (e.g., PolyGCL, HeterGCL) largely rely on contrastive learning with multiple graph filters but do not fundamentally address the adaptive selection of graph convolution degree.
- **Memory efficiency issues**: Many existing methods (GREET, PolyGCL, HeterGCL, etc.) encounter OOM (GPU memory overflow) on larger graphs, limiting scalability.

### Root Cause
Can an effective proxy metric for "class separability" be identified in the absence of labels, thereby guiding adaptive learning of graph convolution degree? FUEL's key insight is that **node features inherently carry class information**—nodes of the same class are naturally similar in feature space, so clustering in feature space can serve as a proxy for true class membership (latent classes). By maximizing the separability of these latent classes, true class separability can be indirectly optimized.

## Method

### Core Idea: Latent-Class Separability

FUEL's theoretical foundation rests on a key finding: **latent-class separability is strongly positively correlated with true class separability**.

Specifically, the authors establish this connection in three steps:
1. On four datasets—Cora, Citeseer (homophilic) and Actor, Cornell (non-homophilic)—100 random sets of graph convolution weights $(\alpha_0, \alpha_1, \alpha_2)$ are sampled to generate embeddings with varying degrees of graph convolution.
2. For each set of embeddings, both **true class separability** (linear classifier test accuracy) and **latent-class separability** (Calinski-Harabász index) are measured.
3. Results show that the Spearman rank correlation between the two exceeds 0.82 across all datasets, with positive linear regression slopes.

Theoretically, under a Gaussian distribution assumption (two equal-sized classes, features $x_i \sim \mathcal{N}(\mu, \sigma^2)$ or $\mathcal{N}(-\mu, \sigma^2)$), the authors prove **Theorem 1**: under certain conditions, the class separability ranking between two degrees of graph convolution is fully consistent with the latent-class separability ranking. This provides rigorous theoretical justification for using clustering quality as an optimization objective.

### Step 1: Clustering-Based Adaptive Graph Convolution Learning

**Adaptive graph convolution model**: A weighted combination form is adopted:

$$\mathbf{X}^* = \alpha_0 \mathbf{X} + \alpha_1 \tilde{\mathbf{A}}\mathbf{X} + \alpha_2 \widetilde{\mathbf{A}^2}\mathbf{X}$$

where $\alpha_0 + \alpha_1 + \alpha_2 = 1$, $\alpha_k = \text{softmax}(c_k)$, and $c_0, c_1, c_2$ are learnable parameters. $\mathbf{X}$ denotes the original features (0th order), $\tilde{\mathbf{A}}\mathbf{X}$ the 1-hop aggregation (1st order), and $\widetilde{\mathbf{A}^2}\mathbf{X}$ the 2-hop aggregation (2nd order).

**Clustering training loss** consists of three components:

- $\mathcal{L}_1$ (node-cluster assignment entropy): Encourages each node to be assigned to a single cluster with high confidence, sharpening the assignment distribution.
- $\mathcal{L}_2$ (cluster balance): Prevents all nodes from collapsing into a single cluster, ensuring relatively uniform cluster sizes.
- $\mathcal{L}_3$ (distance loss): Directly optimizes the ratio of inter-cluster to intra-cluster distance, enhancing cluster separability.

The final loss is $\mathcal{L}_{clus} = \mathcal{L}_1 + \mathcal{L}_2 + \lambda \mathcal{L}_3$. After optimization, the learned optimal weights $\alpha_0^*, \alpha_1^*, \alpha_2^*$ are used to generate the intermediate embedding $\mathbf{H}$.

### Step 2: Embedding Refinement for Enhanced Separability

The intermediate embedding $\mathbf{H}$ is derived from a simple linear combination model with limited expressive power. FUEL refines it via a feed-forward neural network $f_\theta$ with a skip connection:

$$\mathbf{Z} = f_\theta(\mathbf{H}) + \mathbf{H}$$

During training, $\mathbf{H}$ is fixed and only $\theta$ is optimized. The refinement loss is based on nearest-neighbor relationships in the $\mathbf{H}$ space: pulling together the embeddings of $N$ nearest neighbors (likely same-class pairs) while pushing apart non-nearest-neighbor pairs. The loss function takes an exponential form:

$$\mathcal{L}_{dist} = \exp\left(\frac{\sum_{v_i,v_j \in \mathcal{V}'_+} d(\mathbf{z}_i, \mathbf{z}_j)}{\tau|\mathcal{V}'_+|} - \frac{\sum_{v_k,v_\ell \in \mathcal{V}'_-} d(\mathbf{z}_k, \mathbf{z}_\ell)}{\tau|\mathcal{V}'_-|}\right)$$

The advantage of the exponential function is that its logarithm is equivalent to a geometric mean, making it more robust to outliers.

## Key Experimental Results

### Table 1: Node Classification Accuracy (×100, 14 datasets)

| Method | Type | Texas (0.057) | Cornell (0.111) | Actor (0.220) | Cora (0.825) | Photo (0.849) | Avg. Rank |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive X | Baseline | 75.1 | 71.1 | 35.4 | 58.5 | 88.5 | 14.0 |
| DGI | CL | 65.9 | 48.9 | 29.8 | 82.4 | 92.8 | 11.7 |
| BGRL | CL | 64.3 | 49.2 | 30.9 | 82.1 | 93.2 | 8.4 |
| GREET | Non-homophilic | 81.1 | 76.8 | 37.6 | 83.3 | 92.8 | 7.8 |
| PolyGCL | Non-homophilic | 80.5 | 74.8 | 35.0 | 82.6 | 91.3 | 9.1 |
| HeterGCL | Non-homophilic | 78.9 | 71.6 | 36.5 | 82.5 | 93.0 | 9.2 |
| **FUEL** | **Ours** | **84.6** | **78.1** | **38.2** | **83.8** | **94.2** | **2.4** |

FUEL achieves the best performance on 10 out of 14 datasets, with an average rank of 2.4 substantially surpassing the second-best GREET at 7.8. It ranks first on both the lowest-homophily (Texas, h=0.057) and highest-homophily (Photo, h=0.849) graphs.

### Table 2: Node Clustering NMI (×100, 14 datasets)

| Method | Type | Wisconsin | Cornell | Actor | Cora | Photo | Avg. Rank |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HGRL | Non-homophilic | 36.2 | 40.1 | 5.9 | 51.3 | 50.2 | 8.5 |
| GREET | Non-homophilic | 43.7 | 40.0 | 5.9 | 57.3 | 55.5 | 8.1 |
| GraphMAE | Gen | 16.7 | 16.5 | 1.4 | 56.5 | 53.4 | 9.2 |
| w/o Step 1 | Variant | 17.4 | 12.4 | 0.8 | 55.9 | 71.0 | 9.5 |
| w/o Step 2 | Variant | 43.4 | 35.1 | 5.4 | 48.4 | 66.2 | 9.3 |
| **FUEL** | **Ours** | **50.1** | **43.8** | **6.7** | **59.0** | **71.3** | **2.4** |

FUEL also ranks first on the clustering task (average rank 2.4), with particularly pronounced advantages on non-homophilic graphs (Wisconsin 50.1 vs. runner-up 47.7; Cornell 43.8 vs. runner-up 43.5).

## Highlights & Insights

- **Solid integration of theory and practice**: The effectiveness of latent-class separability as a proxy for true class separability is supported by both empirical correlation analysis (Spearman > 0.82) and a formal proof (Theorem 1), rather than being a simple heuristic.
- **Adaptive behavior is intuitive**: On homophilic graphs (Cora, Citeseer), the model automatically assigns large weights to the graph convolution terms; on non-homophilic graphs (Actor, Cornell), graph convolution weights are suppressed near zero, relying almost entirely on raw features. This behavior requires no manual tuning.
- **Scalability advantage**: FUEL runs successfully on all 14 datasets, while most non-homophilic baselines (GREET, PolyGCL, HeterGCL, etc.) encounter OOM on larger graphs such as Penn94 (42k nodes) and Flickr (89k nodes).
- **Simple yet effective design**: Only three scalar parameters, cluster centers, and one feed-forward network are required. The algorithm has low complexity but achieves strong adaptive capability through carefully designed loss functions.

## Limitations & Future Work

- **Fixed graph convolution order at 2**: Only a weighted combination of 0th, 1st, and 2nd order adjacency matrices is used, which may fail to capture long-range dependencies. Extension to higher or learnable orders is worth exploring.
- **Cluster count must be pre-specified**: Experiments set the number of clusters equal to the true class count; although the appendix shows insensitivity to this choice, fully unsupervised cluster number estimation remains an open problem.
- **Theoretical analysis relies on simplified assumptions**: Theorem 1 assumes two equal-sized classes with Gaussian distributions, which diverges from practical multi-class imbalanced scenarios.
- **Limited to node-level tasks**: Effectiveness on graph-level or edge-level downstream tasks has not been verified.
- **Heterogeneous and temporal graphs not considered**: The authors identify these as future directions in the conclusion.

## Related Work & Insights

- **PolyGCL** (Chen et al., 2024): Uses multiple graph filters and contrasts embeddings across different filters, performing well on non-homophilic graphs (average rank 9.1), but the requirement for multiple filter sets increases computational overhead and prevents scaling to large graphs.
- **GREET** (Liu et al., 2023): Performs self-supervised learning after filtering non-homophilic edges based on feature/topology similarity, achieving the best average rank (7.8) among baselines, but the edge-filtering strategy may discard useful cross-class connection information.
- **HeterGCL** (Wang et al., 2024): A contrastive learning method specifically designed for non-homophilic graphs, but shows no clear advantage on homophilic graphs and is similarly memory-constrained.
- **GPR-GNN / GADC** (supervised): Learn per-order adjacency matrix coefficients in a manner conceptually similar to FUEL Step 1, but require label supervision. FUEL achieves an analogous unsupervised version by using clustering as a proxy.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of latent-class separability proxy and adaptive graph convolution in an unsupervised setting is novel and theoretically well-grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 14 datasets, 15 baselines, two downstream tasks (classification + clustering), complete ablation studies, and design objective verification.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with a smooth narrative flow from motivation through analysis, method, and validation; the theoretical sections are accessible.
- Value: ⭐⭐⭐⭐ — Provides a concise and effective solution to the practical pain point of label-free learning on non-homophilic graphs, with substantially superior average rankings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](../../ICLR2026/graph_learning/graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)

</div>

<!-- RELATED:END -->
