---
title: >-
  [Paper Note] Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction
description: >-
  [ICML 2026][Graph Learning][Graph Neural Networks] ICGNN defines the "Influence Contradiction Score" (ICS) over graph diffusion matrices to measure node label suspiciousness from both structural and attribute levels. It…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Noisy Labels"
  - "Influence Contradiction"
  - "Graph Diffusion"
  - "GMM"
date: 2026-05-08
content_hash: 7e9c0f8715f52b7c
---

# Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction

**Conference**: ICML 2026  
**arXiv**: [2601.17469](https://arxiv.org/abs/2601.17469)  
**Code**: https://github.com/wayc04/ICGNN  
**Area**: Graph Learning / GNN Robustness / Noisy Labels  
**Keywords**: Graph Neural Networks, Noisy Labels, Influence Contradiction, Graph Diffusion, GMM

## TL;DR
ICGNN defines the "Influence Contradiction Score" (ICS) over graph diffusion matrices to measure node label suspiciousness from both structural and attribute levels. It employs a GMM soft threshold to identify dirty labels and performs convex-combination-based soft correction using neighbor predictions, outperforming specialized methods like NRGNN, RTGNN, CGNN, and ProCon across 6 graph benchmarks.

## Background & Motivation

**Background**: The effectiveness of Graph Neural Networks (GNNs) in node classification heavily depends on clean training labels. However, real-world graph data (social, recommendation, molecular) often contain significant noise due to manual annotation. Noise reduction methods from CV—sample selection, loss correction, and label correction—mostly ignore topological structures when migrated to graphs.

**Limitations of Prior Work**: Existing graph-specific works (NRGNN, RTGNN, CGNN, ProCon) remain unsatisfactory in two aspects: (i) Detection: NRGNN bypasses detection by simply connecting unlabeled nodes to similar labeled ones, while RTGNN/CGNN use small-loss or simple prediction consistency, failing to fully utilize topological information. (ii) Correction: RTGNN/ProCon only down-weight suspected noisy samples without actual correction, and CGNN uses neighbor majority voting, which is prone to confirmation bias under class imbalance.

**Key Challenge**: To identify noisy labels on graphs, one must simultaneously characterize the opposing forces of "support from same-class neighbors" and "interference from different-class nodes." Previous methods typically focus on only one aspect (loss, neighbor prediction, or feature similarity), leading to significantly overlapping distributions for clean and noisy samples.

**Goal**: (1) Design a noise metric sensitive to both structure and attributes through global propagation paths; (2) Softly separate noisy samples from clean ones using statistical models; (3) Correct labels more robustly than hard voting; (4) Mitigate label scarcity using unlabeled nodes.

**Key Insight**: Treat the columns $\mathbf{T}_{\cdot i}$ of the graph diffusion matrix $\mathbf{T}=\epsilon(\mathbf{I}-(1-\epsilon)\hat{\mathbf{A}})^{-1}$ as the "global influence distribution of all nodes on node $i$," then aggregate the influence of "non-self classes." Under the homophily assumption, if a node receives substantial "heterogeneous influence," its current label is likely incorrect.

**Core Idea**: **Replace local small-loss or neighbor voting with "heterogeneous influence accumulation" under global graph diffusion as a reliability metric. Use a GMM to learn a soft threshold for separating noisy samples and apply soft correction based on neighbor predictions.**

## Method

### Overall Architecture
The input is a semi-supervised graph $\mathcal{G}=\{\mathcal{V},\mathbf{A},\mathbf{X},\mathbf{Y}_L\}$ with noisy labels ($|\mathcal{V}_L|\ll|\mathcal{V}_U|$). The pipeline consists of three stages per epoch:
1. **Noise Detection**: A GCN encoder generates representations $\mathbf{Z}$. Structural diffusion $\mathbf{T}$ and attribute diffusion $\mathbf{R}$ are computed from the adjacency graph $\mathbf{A}$ and K-NN representation graph $\mathbf{A}^r$, respectively. ICS values are aggregated for labeled nodes and fitted to a two-component GMM to output the posterior probability $\hat{\beta}_i$ of being clean.
2. **Noise Correction**: Using $\hat{\beta}_i$ as weight, original one-hot labels are combined with neighbor softmax predictions via convex combination to obtain soft labels $\mathbf{l}_i^{(t)}$. Pseudo-labels are assigned to unlabeled nodes using the same neighbor aggregation strategy.
3. **Optimization**: A GNN classifier is trained jointly on soft and pseudo-labels using cross-entropy, supplemented by NRGNN's negative sampling reconstruction term. $\hat{\beta}_i$, pseudo-labels, and representations are updated synchronously every epoch.

### Key Designs

1. **Influence Contradiction Score (ICS) (Dual-view Structure + Attribute)**:
    - **Function**: Quantifies the contradiction between node $i$'s label and the actual influence it receives. Higher contradiction indicates a higher probability of noise.
    - **Mechanism**: Based on Personalized PageRank, the global influence matrix $\mathbf{T}=\epsilon(\mathbf{I}-(1-\epsilon)\hat{\mathbf{A}})^{-1}$ is defined. Structural ICS $\text{ICS}_i(\mathbf{T})=\sum_{j\neq y_i}\frac{1}{|\mathcal{C}_j|}\sum_{k\in\mathcal{C}_j}\mathbf{T}_{ki}$ aggregates normalized influence from non-self classes. Attribute ICS $\text{ICS}_i(\mathbf{R})$ is calculated similarly using a K-NN graph on GCN representations. The final $\text{ICS}_i=(1-\alpha)\text{ICS}_i(\mathbf{T})+\alpha\text{ICS}_i(\mathbf{R})$ (default $\alpha=0.5$). Theorem 3.1 proves that under a homophily bound $\delta\in(0,1]$, clean and noisy nodes are theoretically separable when $\delta>1/2$.
    - **Design Motivation**: Unlike small-loss (sensitive to loss scales and overlapping) and local voting (affected by imbalance and sparsity), ICS explicitly aggregates "heterogeneous interference" globally. Functional complementarity between views ensures robustness: attribute views are better for sparse graphs (e.g., Pubmed), while structural views suit dense graphs (e.g., Coauthor CS).

2. **GMM Soft Threshold Detection**:
    - **Function**: Automatically partitions the ICS distribution into "clean" and "noisy" clusters to obtain the posterior "clean" probability $\hat{\beta}_i$.
    - **Mechanism**: A two-component Gaussian Mixture Model $\sum_q\pi_q\mathcal{N}(\cdot|\mu_q,\sigma_q)$ is fitted to $\{\text{ICS}_i\}_{i=1}^L$. After EM convergence, the cluster with the smaller mean is identified as "clean," and the posterior $\hat{\beta}_i$ represents the confidence that the label is clean. Complexity is $O(LT)$ with $T\le 10$.
    - **Design Motivation**: $\delta$ in Theorem 3.1 is unknown in practice. EM soft clustering avoids manual thresholding and provides $\hat{\beta}_i \in [0,1]$ as weights for downstream soft correction.

3. **Soft Label Correction & Pseudo-labeling based on Neighbor Predictions**:
    - **Function**: Replaces "hard replacement" of suspected noise with convex combinations based on confidence; also incorporates unlabeled nodes.
    - **Mechanism**: Soft labels are defined as $\mathbf{l}_i^{(t)}=\hat{\beta}_i^{(t)}\mathbf{y}_i+(1-\hat{\beta}_i^{(t)})h^{(t)}(\mathbf{z}_i)$, where $h^{(t)}(\mathbf{z}_i)=\operatorname{softmax}(\sum_{k\in I(i)}\mathbf{T}_{ki}\mathbf{p}_k^{(t)})$ aggregates global neighbor predictions via $\mathbf{T}$. Pseudo-labels use the same $h^{(t)}(\mathbf{z}_i)$. The final loss $\mathcal{L}$ combines cross-entropy on soft and pseudo-labels with negative sampling terms.
    - **Design Motivation**: Hard voting in CGNN causes confirmation bias toward majority classes. Convex combinations trust the original label when $\hat{\beta}_i$ is high and neighbors when it is low. Using PageRank $\mathbf{T}_{ki}$ instead of first-order $\mathbf{A}_{ki}$ captures stable evidence from higher-order structures.

## Key Experimental Results

### Main Results (Node Classification Accuracy %, 20% Noise)

| Dataset | Noise | GCN | RTGNN | CGNN | DND-NET | ProCon | **ICGNN** |
|--------|------|------|-------|------|---------|--------|-----------|
| Coauthor CS | Uniform | 80.3 | 86.7 | 84.1 | 86.2 | 85.4 | **87.4** |
| Amazon Photo | Uniform | 82.2 | 84.8 | 85.3 | 82.3 | 83.5 | **87.3** |
| Cora | Uniform | 70.3 | 79.1 | 76.8 | 76.5 | 78.6 | **80.9** |
| DBLP | Uniform | 71.0 | 79.0 | 78.9 | 77.0 | 77.2 | **80.1** |
| Citeseer | Uniform | 64.9 | 68.2 | 69.7 | 70.4 | 68.4 | **71.5** |
| Amazon Photo | Pair | 80.9 | 84.2 | 85.1 | 80.1 | 81.4 | **86.3** |
| Coauthor CS | Pair | 79.5 | 83.8 | 81.0 | 84.0 | 82.6 | **85.9** |

ICGNN achieves SOTA performance across 12 settings (6 datasets × 2 noise types). On OGBN-Arxiv, it achieves 28.3 vs DND-NET 25.9. On the heterophilous Cornell dataset, it reaches 44.7 vs RTGNN 42.3.

### Ablation Study (Cora / DBLP, 20% Noise)

| Configuration | Cora-Uni | Cora-Pair | DBLP-Uni | Description |
|------|----------|-----------|----------|------|
| Full ICGNN | **80.9** | **79.4** | **80.1** | Full model |
| w/o s-ICS | 79.4 | 78.1 | 78.1 | Remove structural view (-1.5 to 2.0) |
| w/o a-ICS | 79.2 | 77.9 | 78.3 | Remove attribute view (-1.5 to 1.8) |
| w/o NC | 78.7 | 76.9 | 77.4 | Remove soft correction (-2.7) |
| w/o PL | 79.5 | 77.2 | 77.5 | Remove pseudo-labels (-1.4 to 2.6) |
| w $\mathbf{A}$ for $\mathbf{T}$ | 79.6 | 78.2 | 78.2 | Replace diffusion with first-order (-1.3 to 1.9) |

### Key Findings
- **Soft Correction (NC)** is the most significant contributor (2-3% drop if removed), proving its advantage over re-weighting or hard voting.
- **Global Diffusion $\mathbf{T}$** is superior to first-order $\mathbf{A}$ (>1% drop), suggesting PageRank paths are vital for both detection and aggregation.
- Distribution comparison at 40% noise shows severe overlap for loss statistics, while ICS clearly separates clean and noisy clusters.
- **Robustness**: ICGNN exhibits the slowest decay as noise rates increase (10% to 40%) and maintains stability even at extremely low label rates (0.5%).
- **Sensitivity to $\alpha$**: Dense graphs favor structural views; sparse graphs favor attribute views. A fixed $\alpha=0.5$ remains robust without fine-tuning.

## Highlights & Insights
- **Re-framing Noise Detection**: Instead of loss (scalar, topology-agnostic) or first-order voting (local), detection is treated as "global influence contradiction." This replaces CV-centric "small-loss" heuristics with a metric utilizing global graph topology.
- **Two-stage Soft GMM + Convex Correction**: The model avoids hard binary classification of noise. Posterior probabilities $\hat{\beta}_i$ serve as soft gates, preventing irreversible errors from hard thresholds.
- **Dual-view Diffusion**: Constructing K-NN graphs in the representation space allows "feature similarity" to be treated as a graph. This dual-diffusion approach is generalizable to self-supervised learning or community detection tasks.
- **Scalability**: Unlike NRGNN or RTGNN which often OOM on large graphs, ICGNN does not explicitly expand edges or store dense similarity matrices, making it friendly for OGBN-Arxiv.

## Limitations & Future Work
- **Heterophily Dependence**: Theorem 3.1 relies on homophily. While Cornell shows good results, performance on strongly heterophilous graphs (e.g., Squirrel, Chameleon) remains to be investigated.
- **Bootstrapping Dependency**: Detection and correction rely on an initially noisy GCN encoder. At extreme noise rates (>40%), representation quality may degrade ICS reliability.
- **Class Imbalance in Priors**: Calculating $|\mathcal{C}_j|$ using noisy labels might introduce bias in long-tail distributions.
- **Dynamic Scenarios**: Recalculating diffusion matrices per epoch is efficient but might be non-trivial for very large or dynamic graphs.
- **Task Generality**: The experiments focused on node classification; transferability to graph classification or link prediction is unverified.

## Related Work & Insights
- **vs NRGNN (KDD'21)**: NRGNN adds edges for unlabeled nodes to "bypass" noise but lacks active detection. ICGNN provides explicit metrics + GMM and leverages unlabeled nodes via pseudo-labels, leading by 1–3%.
- **vs RTGNN (WSDM'23) / ProCon (IJCAI'25)**: These use small-loss or consistency for re-weighting. ICGNN replaces local consistency with global diffusion and "re-weighting" with "convex combination correction."
- **vs CGNN (ICASSP'23)**: CGNN uses hard neighbor voting. ICGNN's soft convex combination avoids confirmation bias in imbalanced classes.
- **vs DND-NET (KDD'24)**: ICGNN integrates detection, correction, and pseudo-labeling into a unified loop, outperforming DND-NET by 1–4%.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Re-interpreting graph diffusion as a global influence matrix for noise detection is a logical and theoretically grounded alternative to small-loss heuristics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 6 standard graphs, OGBN-Arxiv, and Cornell, including multiple noise types, ablation studies, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Method is well-structured and theoretical conditions are clear.
- **Value**: ⭐⭐⭐⭐ Provides a robust, interpretable, and scalable baseline for noisy labels on graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] An Approximation Algorithm for Graph Label Selection](an_approximation_algorithm_for_graph_label_selection.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](../../AAAI2026/graph_learning/posterior_label_smoothing_for_node_classification.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] Fixed Aggregation Features Can Rival GNNs](fixed_aggregation_features_can_rival_gnns.md)
- [\[ACL 2026\] Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion](../../ACL2026/graph_learning/collaboration_of_fusion_and_independence_hypercomplex-driven_robust_multi-modal_.md)

</div>

<!-- RELATED:END -->
