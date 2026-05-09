---
title: >-
  [Paper Note] Unsupervised Feature Selection Through Group Discovery
description: >-
  [AAAI 2026][Unsupervised feature selection] This paper proposes GroupFS, the first end-to-end differentiable unsupervised feature selection framework that simultaneously discovers latent feature groups and selects the most informative ones, requiring neither predefined groupings nor label supervision.
tags:
  - AAAI 2026
  - Unsupervised feature selection
  - feature grouping
  - graph Laplacian
  - Gumbel-Softmax
  - sparse regularization
date: 2026-05-08
content_hash: 4e665d4a924d807b
---

# Unsupervised Feature Selection Through Group Discovery

**Conference**: AAAI 2026
**arXiv**: [2511.09166](https://arxiv.org/abs/2511.09166)
**Code**: None
**Area**: Machine Learning / Feature Selection
**Keywords**: Unsupervised feature selection, feature grouping, graph Laplacian, Gumbel-Softmax, sparse regularization

## TL;DR

This paper proposes GroupFS, the first end-to-end differentiable unsupervised feature selection framework that simultaneously discovers latent feature groups and selects the most informative ones, requiring neither predefined groupings nor label supervision.

## Background & Motivation

1. **Background**: In high-dimensional data, feature selection (FS) is a critical technique for reducing noise and improving generalization and interpretability. Unsupervised FS is particularly important yet challenging in label-free settings, as no supervision is available to guide the selection process.
2. **Limitations of Prior Work**: Most unsupervised FS methods evaluate features independently, ignoring inter-feature relationships. In practice, features frequently exhibit synergistic behavior—neighboring pixels, functionally connected brain regions, correlated financial indicators, etc. Some methods attempt to capture group structure but rely on predefined groupings or label supervision.
3. **Key Challenge**: Feature grouping and feature selection are inherently coupled problems—group-level selection cannot be performed without knowledge of the group structure, yet discovering meaningful groups is difficult without a selection signal.
4. **Goal**: To simultaneously discover latent feature group structures and select the most informative feature groups in a fully unsupervised setting.
5. **Key Insight**: Two graph structures are constructed—a sample graph and a feature graph—with Gumbel-Softmax enabling differentiable group assignment, STG (stochastic gates) performing group-level selection, and Laplacian smoothness constraints ensuring semantic coherence within groups.
6. **Core Idea**: Feature groups are discovered and selected end-to-end by imposing Laplacian smoothness constraints on both the sample graph and the feature graph, combined with group-sparse regularization.

## Method

### Overall Architecture

Given a data matrix $X \in \mathbb{R}^{N \times d}$, GroupFS learns a soft assignment matrix $M \in \mathbb{R}^{d \times C}$ mapping features to $C$ groups via Gumbel-Softmax, while simultaneously learning a per-group importance weight $z_j$ via STG gating. The total loss $\mathcal{L} = \mathcal{L}_s + \lambda_1 \mathcal{L}_f + \lambda_2 \mathcal{L}_{reg}$ consists of three components: sample smoothness, feature smoothness, and group sparsity.

### Key Designs

1. **Sample-Level Smoothness Loss $\mathcal{L}_s$**:
    - **Function**: Selects features that vary smoothly on the data manifold, preserving the intrinsic geometric structure among samples.
    - **Mechanism**: The assignment matrix $M$ is first obtained via Gumbel-Softmax. Combined with STG gating, per-feature weights are computed as $\hat{z}_i = \sum_{j=1}^C M_{ij} \cdot z_j$, and the input is masked as $\widetilde{X} = X_B \odot \hat{Z}$. A random walk matrix $P_{\widetilde{X}}$ is constructed on the masked data, and the loss is maximized as $\mathcal{L}_s = -\frac{1}{Bd} \text{tr}(\widetilde{X}^\top P_{\widetilde{X}}^t \widetilde{X})$.
    - **Design Motivation**: The smoothness of feature values on the sample manifold is a natural criterion for feature quality (inspired by Laplacian Score), here extended to group-level selection.

2. **Feature-Level Smoothness Loss $\mathcal{L}_f$**:
    - **Function**: Ensures that similar features receive similar group assignments, so that the learned groupings respect the intrinsic structure of the feature space.
    - **Mechanism**: The assignment matrix is embedded via a trainable projection $F = MQ \in \mathbb{R}^{d \times C}$, and a smoothness constraint is imposed on the normalized Laplacian $L_{\text{feat}}$ of the feature graph: $\text{tr}(F^\top L_{\text{feat}} F)$. An orthogonality regularizer $\|F^\top F - I\|_F^2$ prevents degenerate solutions: $\mathcal{L}_f = \frac{1}{dC}[\text{tr}(F^\top L_{\text{feat}} F) + \beta \|F^\top F - I\|_F^2]$.
    - **Design Motivation**: Two features connected by a high-weight edge in the feature graph should be assigned to the same group; the orthogonality constraint ensures diverse and non-redundant embeddings across groups.

3. **Group-Sparse Regularization $\mathcal{L}_{reg}$**:
    - **Function**: Encourages retaining only a small number of highly informative feature groups, yielding compact and interpretable models.
    - **Mechanism**: $\mathcal{L}_{reg} = \frac{1}{C} \sum_{j=1}^C \mathbb{P}(z_j > 0) \cdot \frac{1}{d} \sum_{i=1}^d M_{ij}$, jointly penalizing both the activation probability of each group and its size.
    - **Design Motivation**: Groups with high activation probability and large membership incur greater penalties, driving the model toward a small number of compact active groups.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_s + \lambda_1 \mathcal{L}_f + \lambda_2 \mathcal{L}_{reg}$
- Gumbel-Softmax temperature is annealed linearly from 10 to $10^{-2}$, gradually sharpening assignments toward one-hot
- Assignment matrix logits are initialized via spectral clustering ($p_{\text{main}} = 0.7$)
- Columns of $F$ are zero-mean centered and $\ell_2$-normalized at each step to prevent convergence to trivial solutions
- STG gates initialized with $\mu_j = 0.5$ and noise $\sigma = 0.5$
- Self-tuning kernel with $K=7$ nearest neighbors; diffusion time $t=2$

## Key Experimental Results

### Main Results

| Dataset | GroupFS | Best Baseline | Gain | # Features |
|--------|---------|-------------|------|--------|
| Lung500 | **93.0±6.8** | 91.3±6.7 (CAE) | +1.7% | 234 |
| HeartDisease | **83.1±0.5** | 82.6±0.4 (MCFS) | +0.5% | 10 |
| AR10P | **32.5±4.1** | 24.9±2.9 (MGAGR) | +7.6% | 362 |
| PIE10P | **38.4±2.5** | 35.0±3.6 (DUFS) | +3.4% | 49 |
| NMNIST 3-8 | **83.3±0.1** | 77.3±1.0 (LS) | +6.0% | 51 |
| ALLAML | **70.6±1.4** | 70.6±1.4 (LS) | Tie | 274 |

Under fixed-budget settings, GroupFS ranks first or ties for first on 6 of 9 datasets, outperforming the second-best method by an average of +3.84%.

### Ablation Study

| Configuration | RGsim | TPR | FDR | Notes |
|------|-------|-----|-----|------|
| C ≤ 2+(d-10) | ~1.0 | ~1.0 | ~0 | Perfect recovery of informative groups with noise separation |
| C = 2 | Low | High | High | Too few groups; noise mixed in |
| C > 2+(d-10) | Decreasing | High | Low | Informative groups over-split |

### Key Findings

- GroupFS also achieves the best accuracy on 6/9 datasets under the adaptive-budget setting
- On NMNIST 3-8, it discovers 7 spatially coherent pixel groups; the most important group corresponds to the key region distinguishing digits 3 and 8
- On the Student Performance dataset, the most important group contains alcohol-related features, while the second group captures motivation-related features (absences, past failures, etc.), exhibiting strong semantic coherence
- Higher intra-group correlation strength $\rho$ yields lower loss, confirming that stronger within-group consistency is beneficial
- The method demonstrates robustness to moderate levels of additive noise

## Highlights & Insights

- This is the first work to unify feature group discovery and group-level selection within a single end-to-end differentiable framework, without requiring manually predefined groupings
- The dual-graph design (sample graph + feature graph) is highly natural, with each graph capturing structural information along a different dimension
- Interpretability experiments are convincing: pixel groups discovered on NMNIST are spatially coherent and aligned with discriminative regions
- The combination of Gumbel-Softmax and STG elegantly embeds discrete grouping and gated selection simultaneously into a differentiable optimization framework

## Limitations & Future Work

- Graph construction relies on Euclidean distance, which may yield inaccurate representations for curved non-Euclidean manifolds
- The method learns a single global group importance score, failing to account for conditionally dependent or temporally varying relationships
- The choice of the number of groups $C$ depends on heuristics and exhibits notable sensitivity
- Computational complexity is $O(N^2 d)$ (full-batch) and $O(tN^3)$ (multi-step diffusion), which may become a bottleneck at scale
- No comparison is made against deep feature selection methods (e.g., VAE-based or contrastive learning-based approaches)

## Related Work & Insights

- **vs. Laplacian Score (LS)**: LS scores each feature independently, whereas GroupFS leverages group structure for joint selection, achieving significant improvements over LS on most datasets.
- **vs. DUFS**: DUFS also employs stochastic gating and a sample graph but operates feature-wise without considering group structure; GroupFS elevates gating to the group level.
- **vs. CompFS**: CompFS requires label supervision to learn groups, whereas GroupFS is fully unsupervised; moreover, groups found by CompFS exhibit weaker semantic coherence.
- **vs. MGAGR**: MGAGR relies on predefined groupings and incurs extremely long runtimes (>10,000 CPU hours on large datasets); GroupFS is more flexible and efficient.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First end-to-end unsupervised joint grouping and selection framework with an elegant design
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 datasets, synthetic data, and interpretability experiments across two evaluation settings, though runtime comparisons are absent
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, experimental setups are thorough, and the appendix is comprehensive
- Value: ⭐⭐⭐⭐ Practically valuable for feature selection on high-dimensional unlabeled data; interpretability is a notable strength

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](../../NeurIPS2025/interpretability/tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[AAAI 2026\] HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning](hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)

<!-- RELATED:END -->
