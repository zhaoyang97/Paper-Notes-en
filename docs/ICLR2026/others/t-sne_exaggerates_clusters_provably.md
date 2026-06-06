---
title: >-
  [Paper Note] t-SNE Exaggerates Clusters, Provably
description: >-
  [ICLR 2026][t-SNE] This paper provides rigorous theoretical proofs of two fundamental failure modes of t-SNE: (1) the strength of input clusters cannot be inferred from the output…
tags:
  - "ICLR 2026"
  - "t-SNE"
  - "cluster exaggeration"
  - "dimensionality reduction"
  - "misleading visualization"
  - "outliers"
date: 2026-05-08
content_hash: 4191b4814808ce58
---

# t-SNE Exaggerates Clusters, Provably

**Conference**: ICLR 2026
**arXiv**: [2510.07746](https://arxiv.org/abs/2510.07746)  
**Code**: [https://github.com/njbergam/tsne-exaggerates-clusters](https://github.com/njbergam/tsne-exaggerates-clusters)  
**Area**: Data Visualization / Theoretical Analysis
**Keywords**: t-SNE, cluster exaggeration, dimensionality reduction, misleading visualization, outliers

## TL;DR

This paper provides rigorous theoretical proofs of two fundamental failure modes of t-SNE: (1) the strength of input clusters cannot be inferred from the output, and (2) extreme outliers cannot be faithfully represented — even when the input has no cluster structure or contains extreme outliers, t-SNE may produce perfectly clustered visualizations.

## Background & Motivation

- **Background**: t-SNE is a standard tool for exploratory data analysis, widely used in single-cell genomics, language model interpretability, and beyond.
- **Existing Theory**: Prior work has proven that t-SNE produces cluster-preserving outputs for well-separated input clusters (true-positive guarantees).
- **Limitations of Prior Work**: Theoretical analysis of false positives (clustered output from unstructured input) and false negatives (unstructured output from clustered input) has been absent.
- **Key Challenge**: t-SNE outputs directly influence hypothesis generation, experimental design, and scientific conclusions, making its failure modes practically consequential.

## Method

### Formalization of t-SNE

The input affinity matrix $P$ is constructed via a Gaussian kernel:
$$P_{j|i}(X; \sigma_i) := \frac{\exp(-\|x_j - x_i\|^2 / (2\sigma_i^2))}{\sum_{k \neq i} \exp(-\|x_k - x_i\|^2 / (2\sigma_i^2))}$$

The output affinity matrix $Q$ is based on the $t$-distribution:
$$Q_{ij}(Y) := \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k,l; k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

Objective: minimize $\mathcal{L}_X(Y) := \text{KL}(P(X) \| Q(Y))$

### Core Finding 1: Cluster Strength Is Not Identifiable

**Theorem 3** (Different inputs, identical outputs): For any $0 < \epsilon \leq 1$, there exists a dataset $X_\epsilon$ such that:
$$\bar{\mathcal{S}}(X_\epsilon; C_{m \in [k]}) = \epsilon \cdot \bar{\mathcal{S}}(X; C_{m \in [k]})$$
yet for any perplexity $\rho$:
$$\text{t-SNE}_\rho(X) = \text{t-SNE}_\rho(X_\epsilon)$$

That is, an "impostor" dataset with arbitrarily weak cluster structure can produce exactly the same t-SNE output as a strongly clustered dataset.

**Corollary 4**: For any balanced two-class dataset, there exists a family of datasets with silhouette coefficients ranging from $\epsilon$ to 1 that share an **identical set of t-SNE stationary points**.

### Core Finding 2: Tiny Perturbations Cause Drastic Changes

**Theorem 5**: For any $\epsilon > 0$, there exist datasets $X, X'$ such that all pairwise distance ratios lie within $[1-\epsilon, 1+\epsilon]$ (i.e., distances are nearly identical), yet the t-SNE outputs are completely different.

**Lemma 6** (Surprising result): The set $\Delta_\epsilon$ of datasets that approximately form a regular simplex suffices to generate **all possible t-SNE stationary point outputs**.

### Key Mechanism: Additive Invariance

Beyond multiplicative scale invariance, t-SNE also exhibits **additive shift invariance** with respect to squared input distances. That is, if $\|x'_i - x'_j\|^2 = \|x_i - x_j\|^2 + C$, then $\text{t-SNE}_\rho(X) = \text{t-SNE}_\rho(X')$. This property is the fundamental cause of the failure modes described above.

### Core Finding 3: Outliers Are Suppressed

**Theorem 9**: For **any** t-SNE output $Y$, the outlierness $\alpha(Y) \leq 3.266 + o_n(1)$.

Regardless of how extreme the outliers are in the input, t-SNE cannot represent outlierness exceeding approximately 3.6 in the output. This is caused by the asymmetry between the input and output affinity matrices.

### Single-Point Poisoning Attack

Adding a single "poisoning point" placed at the data mean suffices to destroy the entire cluster visualization structure. This effect is particularly severe in high-dimensional data, where the poisoning point becomes the nearest neighbor of most points, drastically altering the affinity matrix.

## Experimental Validation

### Impostor Dataset Experiment

| Metric | Original PBMC3k | Impostor Dataset |
|--------|----------------|-----------------|
| t-SNE visualization | Clear clusters | **Nearly identical clusters** |
| Silhouette coefficient | High (original) | **Extremely low** |
| Nearest-neighbor ranking | Normal | **Preserved unchanged** |

### Poisoning Attack Experiment

- 400 points × 2000-dimensional Gaussian mixture → add 1 poisoning point → cluster structure **completely disappears**
- BBC News dataset: inject 10% poisoning points → silhouette coefficient **halved**
- By contrast: injecting 50% outliers has **almost no effect** on cluster structure

### Outlier Experiment

| Dataset | $\alpha$ in t-SNE | $\alpha$ in PCA |
|---------|------------------|----------------|
| Financial fraud data | ~0.2 | Separation preserved |
| Gaussian + outliers | ~0.1 | Faithfully recovered |

## Highlights & Insights

1. **First theoretical analysis of t-SNE failure modes**: Prior work offered only empirical observations; this paper provides rigorous proofs.
2. **Discovery of additive invariance**: Reveals the fundamental cause of t-SNE's misleading behavior.
3. **Practical implications**:
    - The strength of input clusters cannot be inferred from t-SNE visualizations.
    - t-SNE is unsuitable for outlier detection.
    - t-SNE is particularly unstable on high-dimensional data (which tends to approximate a regular simplex).
4. **PCA as a complement**: PCA significantly outperforms t-SNE in outlier detection and stability.

## Limitations & Future Work

- Theoretical results are based on stationary point analysis; actual t-SNE outputs depend on the optimization trajectory and may avoid certain stationary points.
- Contributions are primarily mathematical; concrete algorithmic improvements are limited.
- The paper focuses mainly on t-SNE; analysis of methods such as UMAP is only preliminary.

## Related Work & Insights

- **t-SNE theory**: Arora et al. 2018 (cluster-preservation guarantees); Cai & Ma 2022 (analysis of optimization phases)
- **Critiques of t-SNE**: Chari & Pachter 2023 (t-SNE as an unreliable exploratory analysis tool)
- **General dimensionality reduction theory**: Snoeck et al. 2026 (any constant-dimensional embedding necessarily incurs distortion)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First rigorous theoretical analysis of t-SNE failure modes
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Elegant proofs; the discovery of additive invariance is profound
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Theory and experiments are tightly integrated
- **Writing Quality**: ⭐⭐⭐⭐ — Important cautionary findings for researchers using t-SNE in practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](../../ICML2026/others/provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)
- [\[ICLR 2026\] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning](cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)

</div>

<!-- RELATED:END -->
