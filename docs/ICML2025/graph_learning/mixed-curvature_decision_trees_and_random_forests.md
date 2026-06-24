---
title: >-
  [Paper Note] Mixed-Curvature Decision Trees and Random Forests
description: >-
  [ICML 2025][Graph Learning][Decision Trees] Extends classical decision tree and random forest algorithms from Euclidean space to mixed-curvature product manifolds (hyperbolic $\times$ spherical $\times$ Euclidean). By utilizing angular reformulation to construct split criteria that respect manifold geometry, the proposed method achieves outstanding performance across 57 classification, regression, and link prediction tasks (ranking 1st in 29 tasks and top-2 in 41 tasks).
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Decision Trees"
  - "Random Forests"
  - "Product Manifolds"
  - "Mixed Curvature"
  - "Hyperbolic Space"
  - "Spherical Space"
  - "Non-Euclidean Geometry"
date: 2026-05-08
content_hash: d40813a3223014a0
---

# Mixed-Curvature Decision Trees and Random Forests

**Conference**: ICML 2025  
**arXiv**: [2410.13879](https://arxiv.org/abs/2410.13879)  
**Authors**: Philippe Chlenski, Quentin Chu, Raiyan R. Khan, Kaizhu Du, Antonio Khalil Moretti, Itsik Pe'er  
**Affiliations**: Columbia University  
**Code**: [pchlenski/manify](https://github.com/pchlenski/manify)  
**Area**: Graph Learning  
**Keywords**: Decision Trees, Random Forests, Product Manifolds, Mixed Curvature, Hyperbolic Space, Spherical Space, Non-Euclidean Geometry  

## TL;DR

Extends classical decision tree and random forest algorithms from Euclidean space to mixed-curvature product manifolds (hyperbolic $\times$ spherical $\times$ Euclidean). By utilizing angular reformulation to construct split criteria that respect manifold geometry, the proposed method achieves outstanding performance across 57 classification, regression, and link prediction tasks (ranking 1st in 29 tasks and top-2 in 41 tasks).

## Background & Motivation

**The Rise of Non-Euclidean Embeddings**: Extensive research demonstrates that hierarchical tree-like data is well-suited for embedding in hyperbolic spaces (negative curvature), while cyclical or periodic data is better suited for spherical spaces (positive curvature). Real-world data often exhibits diverse topological structures, making product manifolds $\mathcal{P} = \mathcal{M}_1 \times \mathcal{M}_2 \times \cdots \times \mathcal{M}_k$ (where each component has distinct curvature) a more appropriate embedding space.

**Limitations of Prior Work**: Classical DT/RF are designed only for Euclidean space; directly applying them to non-Euclidean spaces violates the underlying geometric structures. Prior work (such as HyperDT) operates solely in pure hyperbolic spaces and cannot handle mixed curvatures.

**Scarcity of ML Tools on Product Manifolds**: Although perceptrons, SVMs, and GCNs have been adapted for product manifolds, non-parametric methods that are simple, interpretable, and gradient-free remain scarce. DT/RF are ideal candidates due to their interpretability and robustness.

**Key Challenge**: How to define "split hyperplanes" on each component manifold such that they both respect the manifold geometry and preserve the recursive, greedy splitting framework of decision trees.

## Method

### Overall Architecture

Coordinates on each component manifold of the product manifold $\mathcal{P}$ are converted into angular representations, and splitting is then performed in the angular space. The overall process is as follows:

1. **Preprocessing**: Map ambient coordinates $\mathbf{x} \in \mathbb{R}^{d_{\text{ambient}}}$ to angular vectors $\boldsymbol{\theta} \in [-\pi, \pi)^{d_{\text{intrinsic}}}$.
2. **Recursive Splitting**: Select the optimal splitting dimension and threshold in the angular space.
3. **Geodesic Midpoint**: Use manifold-specific geodesic midpoints as the split boundaries.
4. **Random Forest Ensemble**: Perform random subsampling of features and samples to construct multiple trees.

### Key Designs

#### 1. Angular Reformulation

For each component manifold $\mathcal{M}_i$, points are projected onto two-dimensional subspaces to compute angles:

$$\theta(\mathbf{x}, d) = \arctan\!\left(\frac{x_0}{x_d}\right)$$

Where $x_0$ is the reference dimension coordinate, and $x_d$ is the target dimension coordinate.

- **Hyperbolic Space $\mathbb{H}^n$**: $x_0$ represents the time dimension (the special dimension), and the projected angle naturally corresponds to the direction in the Lorentz model.
- **Spherical Space $\mathbb{S}^n$**: Similarly uses the first dimension, where the angle corresponds to the longitude/latitude on the sphere.
- **Euclidean Space $\mathbb{E}^n$**: Uses a dummy dimension (dummy dimension = 1), $\theta = \arctan(1/x_d)$, which degenerates to a standard Euclidean split.

#### 2. Angular Split Criterion

Given an angular threshold $\theta^*$, the split rule is:

$$S(\mathbf{x}, d, \theta^*) = \begin{cases} 1 & \text{if } \theta(\mathbf{x},d) \in [\theta^*, \theta^* + \pi) \\ 0 & \text{otherwise} \end{cases}$$

The core comparison operator `_angular_greater` computes $(\theta_{\text{key}} - \theta_{\text{query}} + \pi) \bmod 2\pi \geq \pi$, ensuring periodic correctness.

#### 3. Geodesic Midpoints

To position the decision boundaries optimally, manifold-specific geodesic midpoints are utilized:

| Manifold Type | Midpoint Formula |
|---------|---------|
| Euclidean $\mathbb{E}^n$ | $m_{\mathbb{E}}(\theta_u, \theta_v) = \arctan\!\left(\frac{2}{u_d + v_d}\right)$ |
| Hyperbolic $\mathbb{H}^n$ | $m_{\mathbb{H}}(\theta_u, \theta_v) = \arctan\!\left(\frac{u_0 + v_0}{u_d + v_d}\right)$ (involving sinh/cosh) |
| Spherical $\mathbb{S}^n$ | $m_{\mathbb{S}}(\theta_u, \theta_v) = \arctan\!\left(\frac{u_0 + v_0}{u_d + v_d}\right)$ (involving sin/cos) |

Ablation studies confirm that using manifold-specific midpoints outperforms simple angular averaging.

#### 4. Information Gain and Optimal Split Selection

- **Classification Tasks**: Use Gini impurity as the information gain metric.
- **Regression Tasks**: Use MSE as the splitting criterion.
- Iterate through all dimensions and candidates (the angles of each training sample) to select $(d^*, \theta^*)$ that maximizes the information gain.
- Then, find the closest angle to $\theta^*$ in the opposing class, and compute the geodesic midpoint as the final splitting threshold.

#### 5. Random Forest Extension (ProductSpaceRF)

- **Feature Subsampling**: `max_features` supports `sqrt` and `log2`.
- **Sample Subsampling**: `max_samples` controls the bootstrap ratio.
- **Ensemble Prediction**: Majority voting (probability averaging) is used for classification, and the mean is used for regression.

### Loss & Training

This method is non-parametric and does not utilize gradient-based optimization. Split selection is achieved by greedily maximizing information gain:

$$\text{IG}(d, \theta) = H(Y) - \sum_{s \in \{L, R\}} \frac{|S_s|}{|S|} H(Y_s)$$

Where $H$ represents Gini impurity (for classification) or MSE (for regression).

## Key Experimental Results

### Main Results: Absolute Ranking across 57 Tasks

| Method | No. 1 Count | Top 2 Count| Task Coverage |
|------|----------|----------|------------|
| **Product RF (Ours)** | **29** | **41** | Classification + Regression + Link Prediction |
| Product DT | ~12 | ~22 | Classification + Regression + Link Prediction |
| Tangent RF | ~8 | ~15 | Classification + Regression |
| Sklearn RF | ~5 | ~12 | Classification + Regression |
| $\kappa$-GCN | ~4 | ~10 | Classification + Link Prediction |
| Product SVM | ~3 | ~8 | Classification |
| Product Perceptron | ~2 | ~5 | Classification |

### Detailed Results by Task Type

| Task Type | Number of Datasets | Product RF Rank 1 | Primary Competitors |
|---------|-----------|-----------------|------------|
| Classification (Single Manifold) | 15 | ~7 | Sklearn RF, Tangent RF |
| Classification (Product Manifold) | 15 | ~10 | $\kappa$-GCN, Tangent RF |
| Regression | 12 | ~6 | Tangent RF, Sklearn RF |
| Link Prediction | 15 | ~6 | $\kappa$-GCN, Fermi-Dirac |

### Ablation Study

1. **Midpoint Ablation**: Removing geodesic midpoints (using simple angular averaging) leads to a 1-3% drop in classification accuracy and an increase in regression MSE.
2. **Feature Selection Mode**: `d_choose_2` (all coordinate pairs) outperforms `d` (only the reference dimension) on high-dimensional manifolds, albeit with higher computational overhead.
3. **Single Manifold Degeneration**: Yields $\approx$ HyperDT performance on pure hyperbolic data, and $\approx$ scikit-learn DT on pure Euclidean data, validating the correctness of the method.
4. **Sensitivity to Curvature Estimation**: Using greedy signature selection to estimate manifold signatures shows that incorrect signatures still yield reasonable performance, though inferior to correct signatures.
5. **Number of Trees**: Performance converges after 100 trees, consistent with classical RF behavior.

## Highlights & Insights

1. **Elegance of a Unified Framework**: Through angular reformulation, the splitting rules for hyperbolic, spherical, and Euclidean spaces are unified into a single form, differing only in the midpoint computation based on the manifold type. This is a rare "one formula for all curvatures" solution in non-Euclidean ML.

2. **Gradient-Free Optimization**: Unlike methods requiring backpropagation (e.g., GCNs, perceptrons), DT/RF is purely greedy and recursive. It offers stable training without hyperparameter tuning (e.g., learning rates), making it suitable for resource-constrained scenarios.

3. **Preservation of Interpretability**: Inheriting the interpretability of DTs, each split node corresponds to an angular threshold on a specific manifold component, which can be directly interpreted as "whether the direction of this dimension exceeds a certain geodesic angle."

4. **High Practicality**: The code is open-source (the `manify` library) with an API design compatible with scikit-learn, supporting classification, regression, and link prediction tasks.

5. **Experimental Thoroughness**: 57 datasets, 13 baselines, and 3 task types, offering coverage far exceeding a single benchmark.

## Limitations & Future Work

1. **Computational Complexity**: The complexity for the angle comparison matrix is $O(n^2 \times d)$ (in batched mode), which poses memory and time constraints on large datasets.
2. **Dependency on Embedding Quality**: The method assumes that the data is already appropriately embedded in the product manifold; the quality of the embeddings directly impacts classification and regression performance.
3. **Prior Curvature Signature Required**: The correct product manifold signature (curvature and dimension of each component) must be known or estimated; incorrect signatures degrade performance.
4. **Lorentz/Ambient Representation Only**: Stereographic projection representations (such as the Poincaré ball) are not directly supported, requiring coordinate conversion.
5. **Indirect Support for Link Prediction**: Converts distances into classification problems via a Fermi-Dirac decoder, which is not an end-to-end process.

## Related Work & Insights

- **HyperDT (Chlenski et al., 2024)**: Directly extends HyperDT from pure hyperbolic space to product manifolds.
- **Product Space Forms (Tabaghi et al., 2021)**: Explores linear classifiers in product spaces, from which this work adapts greedy signature selection.
- **$\kappa$-GCN (Bachmann et al., 2020)**: GCNs on product manifolds, serving as a primary baseline.
- **HGCN (Chami et al., 2019)**: Hyperbolic Graph Convolutional Networks, the origin of the Fermi-Dirac decoder.

**Insight**: This work showcases a paradigm of "geometrizing classical ML methods." Instead of designing entirely new non-Euclidean algorithms, it identifies appropriate coordinate transformations to naturally adapt classical methods to non-Euclidean geometry. This approach can be generalized to other classical methods such as $k$-means and Gaussian Mixture Models.

## Rating

- **Novelty**: ★★★★☆ — Elegant angular reformulation, but fundamentally a natural extension of HyperDT.
- **Value**: ★★★★★ — Robust codebase, user-friendly API, broad task coverage, and no hyperparameter tuning.
- **Experimental Thoroughness**: ★★★★★ — 57 datasets + 13 baselines + 3 task types.
- **Writing Quality**: ★★★★☆ — A comprehensive 30-page paper with clear mathematical derivations and thorough ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] WILTing Trees: Interpreting the Distance Between MPNN Embeddings](wilting_trees_interpreting_the_distance_between_mpnn_embeddings.md)
- [\[ICML 2025\] Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [\[ICML 2026\] RADE: Random Add-Drop Edge as a Regularizer](../../ICML2026/graph_learning/rade_random_add-drop_edge_as_a_regularizer.md)
- [\[ICLR 2026\] Graph Random Features for Scalable Gaussian Processes](../../ICLR2026/graph_learning/graph_random_features_for_scalable_gaussian_processes.md)
- [\[ICLR 2026\] Rethinking the Gold Standard: Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs?](../../ICLR2026/graph_learning/rethinking_the_gold_standard_why_discrete_curvature_fails_to_fully_capture_over-.md)

</div>

<!-- RELATED:END -->
