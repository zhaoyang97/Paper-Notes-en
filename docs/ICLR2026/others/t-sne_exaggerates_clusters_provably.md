---
title: >-
  [Paper Note] t-SNE Exaggerates Clusters, Provably
description: >-
  [ICLR 2026][Others][t-SNE] This work provides a rigorous theoretical proof that t-SNE suffers from two fundamental failure modes: (1) inability to infer the strength of input clustering from the output, and (2) inability to faithfully represent extreme outliers. Even when the input lacks cluster structure or contains extreme outliers, t-SNE can
tags:
  - ICLR 2026
  - Others
  - t-SNE
date: 2026-05-08
content_hash: a293aa3041e64714
---
# t-SNE Exaggerates Clusters, Provably

**Conference**: ICLR 2026  
**arXiv**: [2510.07746](https://arxiv.org/abs/2510.07746)  
**Code**: [https://github.com/njbergam/tsne-exaggerates-clusters](https://github.com/njbergam/tsne-exaggerates-clusters)  
**Area**: Data Visualization / Theoretical Analysis  
**Keywords**: t-SNE, Cluster Exaggeration, Dimensionality Reduction, Misleading Visualization, Outliers

## TL;DR

This work provides a rigorous theoretical proof that t-SNE suffers from two fundamental failure modes: (1) inability to infer the strength of input clustering from the output, and (2) inability to faithfully represent extreme outliers. Even when the input lacks cluster structure or contains extreme outliers, t-SNE can produce visualizations with perfect clustering.

## Background & Motivation

- **Widespread Use of t-SNE**: t-SNE is a standard tool for exploratory data analysis, widely used in fields such as single-cell genomics and language model interpretability.
- **Existing Theory**: Prior work has proven that t-SNE can produce outputs that preserve cluster structures for well-separated input clusters (true positive guarantees).
- **Theoretical Gap**: Theoretical analysis regarding false positives (clustered output from non-clustered input) and false negatives (non-clustered output from clustered input) has been missing.
- **Scientific Impact**: The output of t-SNE directly influences hypothesis generation, experimental design, and scientific conclusions.

## Method

### Overall Architecture

This paper performs a rigorous negative theoretical analysis of standard t-SNE rather than proposing a new algorithm. It formulates t-SNE as an optimization problem minimizing the KL divergence between input affinities $P$ and output affinities $Q$. By characterizing the stationary point structure regarding "what kind of input is mapped to what kind of output," the authors construct counterexamples to prove two types of failures. Specifically, input affinities are defined by a Gaussian kernel with adaptive bandwidth $\sigma_i$ as $P_{j|i}(X;\sigma_i)=\frac{\exp(-\|x_j-x_i\|^2/(2\sigma_i^2))}{\sum_{k\neq i}\exp(-\|x_k-x_i\|^2/(2\sigma_i^2))}$, and output affinities use the heavy-tailed t-distribution $Q_{ij}(Y)=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}$. The optimization objective is $\mathcal{L}_X(Y):=\text{KL}(P(X)\|Q(Y))$, and the analysis is built entirely upon the stationary points of this objective.

### Key Designs

This paper is a purely theoretical negative analysis and does not involve a pipeline. The following four design points follow a logical chain: "Identifying the root cause (Additive Invariance) -> Deriving two classes of failures -> Turning existential counterexamples into actionable attacks."

**1. Additive Invariance: The Root Cause of Misleading Behavior**

The reason t-SNE can be deceptive lies in a long-overlooked property re-derived in this paper: it is not only scale-invariant to squared input distances but also possesses **additive translation invariance**. If a constant $C$ is added to all pairwise squared distances, i.e., $\|x'_i-x'_j\|^2=\|x_i-x_j\|^2+C$, the adaptive bandwidth $\sigma_i$ for each point will absorb this translation during re-normalization. Consequently, $\text{t-SNE}_\rho(X)=\text{t-SNE}_\rho(X')$ holds for any perplexity $\rho$ (Lemma 17). In other words, "raising or lowering the global distance structure" has no effect on the output—there exists an entire equivalence class of inputs that look drastically different but yield identical t-SNE outputs. Both subsequent failure modes are essentially counterexamples constructed within this equivalence class.

**2. Failure Mode 1: Cluster Strength Cannot be Inferred from Output**

The first failure mode states that a visually clean cluster plot neither allows one to infer how separated the inputs are nor is robust to tiny input perturbations.

In the forward direction, additive invariance allows the construction of "impostor" datasets where any weakly clustered input generates the same visualization as a strongly clustered one. By increasing pairwise distances, an input can be pushed towards a regular simplex (nearly no cluster structure) while maintaining the same visual output. Theorem 3 shows that for any $0<\epsilon\leq 1$, there exists $X_\epsilon$ such that its clustering significance is reduced to $\bar{\mathcal{S}}(X_\epsilon;C_{m\in[k]})=\epsilon\cdot\bar{\mathcal{S}}(X;C_{m\in[k]})$, yet it satisfies $\text{t-SNE}_\rho(X)=\text{t-SNE}_\rho(X_\epsilon)$. Corollary 4 further provides a family of datasets with silhouette coefficients ranging continuously from $\epsilon$ to $1$ that share **identical sets of t-SNE stationary points**.

In the reverse direction, tiny differences in input are magnified into wildly different plots. Theorem 5 constructs two datasets $X, X'$ where all pairwise distance ratios fall within $[1-\epsilon, 1+\epsilon]$ (nearly indistinguishable), yet their t-SNE outputs differ completely. Lemma 6 goes further: a family of datasets $\Delta_\epsilon$ consisting only of approximate regular simplices is sufficient to cover **all possible t-SNE stationary point outputs**. Thus, "unstructured" input can be interpreted as any structure. Together, these show that no stable mapping exists between input cluster strength and output patterns, especially in high dimensions where measure concentration often lands data in the near-simplex regime.

**3. Failure Mode 2: Extreme Outliers Cannot be Faithfully Represented**

The second failure mode targets the representation of outliers. Theorem 9 provides a hard upper bound independent of the input: for **any** t-SNE output $Y$, the metric measuring "how far the most extreme outlier is from the main cluster," $\alpha(Y)$, is bounded by $\alpha(Y)\leq 3.266+o_n(1)$. This means no matter how far an outlier actually is in the input space, its visual outlier-ness in the output is capped at approximately $3.3$. This stems from the asymmetric normalization of affinities—$P$ is normalized per row (adaptive) while $Q$ is normalized globally. This asymmetry prevents any point from being truly isolated in the low-dimensional map. In practice, distant outliers are often absorbed into the main clusters, providing a theoretical explanation for why t-SNE is unsuitable for outlier detection.

**4. Point Poisoning: Destroying the Entire Plot with One Point at the Mean**

These vulnerabilities can be exploited adversarially. Adding **one** "poisoning point" placed at the data mean is sufficient to collapse the entire cluster visualization. This is particularly lethal in high dimensions; due to measure concentration, high-dimensional Gaussian mixtures themselves approximate regular simplices. A poisoning point at the mean becomes the nearest neighbor for the majority of samples, rewriting the input affinity matrix $P$ and flattening the cluster structure. This turns the first two failures into an actionable attack: in experiments, adding one poisoning point to a 400-point $\times$ 2000-dimensional Gaussian mixture makes the two-cluster structure completely disappear, whereas injecting up to 50% ordinary outliers has almost no effect.

## Key Experimental Results

### Main Results (Impostor Dataset Experiment)

| Metric | Original PBMC3k | Impostor Dataset |
|------|-----------|-----------|
| t-SNE Visualization | Clear Clustering | **Nearly Identical Clustering** |
| Silhouette Coefficient | High (Original) | **Extremely Low** |
| Nearest Neighbor Ranking | Normal | **Maintained** |

### Poisoning Attack Experiment

- 400 points $\times$ 2000-dim Gaussian mixture $\rightarrow$ add 1 poisoning point $\rightarrow$ cluster structure **completely disappears**.
- BBC News dataset: injecting 10% poisoning points $\rightarrow$ Silhouette coefficient **halved**.
- Comparison: injecting 50% outliers has **almost no impact** on cluster structure.

### Outlier Experiment

| Dataset | $\alpha$ in t-SNE | $\alpha$ in PCA |
|--------|-------------|-----------|
| Financial Fraud Data | ~0.2 | Remains Separated |
| Gaussian + Outliers | ~0.1 | Faithfully Restored |

## Highlights & Insights

1. **First Theoretical Analysis of t-SNE Failure Modes**: While empirical observations existed, this work provides rigorous proofs.
2. **Discovery of Additive Invariance**: Reveals the fundamental reason behind t-SNE's misleading behavior.
3. **Practical Implications**:
    - One cannot infer input cluster strength from t-SNE visualizations.
    - t-SNE is unsuitable for outlier detection.
    - t-SNE is particularly unstable on high-dimensional data (due to proximity to regular simplices).
4. **PCA as a Complement**: PCA significantly outperforms t-SNE in terms of stability and outlier detection.

## Limitations & Future Work

- Theoretical results are based on stationary point analysis; actual t-SNE output depends on the optimization path (which might avoid certain stationary points).
- The contribution is primarily mathematical theory, with few suggestions for algorithmic improvement.
- Focuses mainly on t-SNE, with only preliminary experiments on methods like UMAP.

## Related Work & Insights

- **t-SNE Theory**: Arora et al. 2018 (cluster preservation guarantees), Cai & Ma 2022 (optimization phase analysis).
- **t-SNE Criticism**: Chari & Pachter 2023 (unreliable tool for exploratory analysis).
- **General Dimensionality Reduction Theory**: Snoeck et al. 2026 (any constant-dimensional embedding inevitably distorts).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First rigorous theoretical analysis of t-SNE failure modes.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Elegant proofs; profound discovery of additive invariance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Strong alignment between theory and experiments.
- **Value**: ⭐⭐⭐⭐ — Important warning for researchers using t-SNE in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](../../ICML2025/others/provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/others/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)

</div>

<!-- RELATED:END -->
