---
title: >-
  [Paper Note] Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces
description: >-
  [CVPR 2026][LLM Evaluation][Hierarchical Classification] This paper proposes Hier-COS, a framework that assigns orthogonal basis vectors to each node in a label hierarchy tree to construct a theoretically guaranteed Hier…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Hierarchical Classification"
  - "Orthogonal Subspaces"
  - "Hierarchy-aware Features"
  - "Evaluation Metrics"
  - "Label Hierarchy"
date: 2026-05-08
content_hash: 390df58e10f10694
---

# Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces

**Conference**: CVPR 2026
**arXiv**: [2503.07853](https://arxiv.org/abs/2503.07853)  
**Code**: [Project Page](https://sites.google.com/iiitd.ac.in/hier-cos)  
**Area**: LLM Evaluation
**Keywords**: Hierarchical Classification, Orthogonal Subspaces, Hierarchy-aware Features, Evaluation Metrics, Label Hierarchy

## TL;DR
This paper proposes Hier-COS, a framework that assigns orthogonal basis vectors to each node in a label hierarchy tree to construct a theoretically guaranteed Hierarchy-Aware Vector Space (HAVS). It is the first to unify "hierarchy-aware fine-grained classification" and "hierarchical multi-level classification" within a single framework, while introducing a new evaluation metric HOPS, achieving comprehensive state-of-the-art performance across four datasets.

## Background & Motivation

**Background**: Traditional classifiers treat all categories as mutually exclusive, ignoring the semantic hierarchical structure among classes. Hierarchy-aware representation learning aims to place semantically similar classes closer in feature space, thereby reducing the severity of misclassifications.

**Limitations of Prior Work**: (a) Existing methods constrain feature representations to a one-dimensional space along weight vector directions, resulting in insufficient angular separation; (b) different classes have varying classification complexity, yet existing methods allocate identical learning capacity to all; (c) existing evaluation metrics (MS, AHD@k) have critical flaws—AHD is permutation-invariant and cannot distinguish between the optimal and worst top-k prediction orderings.

**Key Challenge**: Existing methods either support only hierarchy-aware multi-class classification (without multi-level classification), or require additional classifiers with consistency constraints for multi-level classification, increasing training complexity without guaranteeing consistency.

**Goal**: To construct a theoretically guaranteed hierarchy-consistent feature space that unifies both classification paradigms and adaptively allocates learning capacity.

**Key Insight**: Orthogonal basis vectors are used to construct subspaces, where each node's subspace is spanned by the basis vectors of all its ancestors and descendants. Distance is defined as the orthogonal projection distance to the subspace—classes sharing more ancestors have greater subspace overlap and thus smaller distances.

**Core Idea**: Each class corresponds to an orthogonal subspace spanned by the basis vectors of its ancestors, itself, and its descendants; the composition of subspaces naturally encodes the hierarchical structure.

## Method

### Overall Architecture
Given a label hierarchy tree $\mathcal{T}$ with $n$ nodes, an $n$-dimensional orthogonal basis $\mathcal{E}$ is defined with each basis vector corresponding to one node. For each class $v_i$, its subspace is $V_i = \text{span}(\mathcal{E}_i^a \cup \{e_i\} \cup \mathcal{E}_i^d)$ (basis vectors of ancestors, self, and descendants). A lightweight transformation module maps backbone features into $V_\mathcal{T}$.

### Key Designs

1. **Hierarchy-Aware Vector Space (HAVS) Definition and Construction**:

    - Function: Defines a feature space with theoretical guarantees of hierarchy consistency.
    - Mechanism: Theorem 1 proves that if a feature vector $\mathbf{x} \in V_{y_i}$ has non-zero projections along all basis directions, then $V_\mathcal{T}$ is a HAVS—i.e., tree distance $D_\mathcal{T}(y_i, y_j) < D_\mathcal{T}(y_i, y_k)$ implies feature distance $|D_i - D_j| < |D_i - D_k|$. The distance to a subspace reduces to the projection norm onto the orthogonal complement: $d_S^2(\mathbf{x}, V_{y_j}) = \sum_{e \in \neg\mathcal{E}_{y_j}} x_e^2$.
    - Design Motivation: Classes sharing more ancestors have greater subspace overlap and smaller orthogonal complements, resulting in smaller distances—which directly corresponds to the definition of hierarchical distance.

2. **Adaptive Learning Capacity**:

    - Function: More complex classes automatically receive higher-dimensional subspaces.
    - Mechanism: Subspace dimensionality = number of ancestors + 1 + number of descendants. Classes sharing more ancestors (e.g., $\{D6,\ldots,D10\}$) obtain higher-dimensional subspaces capable of encoding finer discriminative features; simpler classes (e.g., $\{A2, A3\}$) require only low-dimensional subspaces.
    - Design Motivation: Existing methods assign a uniform one-dimensional representation space to all classes, failing to handle classes with varying complexity in imbalanced hierarchies.

3. **Unified Classification**:

    - Function: A single model simultaneously performs hierarchy-aware multi-class classification and hierarchical multi-level classification.
    - Mechanism: At inference, $\hat{y} = \arg\max_{y_i \in \mathcal{V}_\ell} \|\mathbb{P}_{\mathcal{E}_{y_i}} \mathbf{x}\|$ (leaf-node classification). Proposition 1 guarantees that the predicted path $\{\hat{y}^{(1)}, \ldots, \hat{y}^{(H)}\}$ forms a valid path in the tree, making multi-level classification naturally consistent—without additional classification heads or consistency constraints.
    - Design Motivation: Existing indirect methods require one classifier per level plus additional consistency losses, adding training complexity without guaranteeing consistency.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{kl} + \alpha \mathcal{L}_{reg}$. $\mathcal{L}_{kl}$: KL divergence aligns the distribution of feature projections onto basis vectors with the target distribution (using exponentially increasing weights, with the highest weight on leaf-node directions). $\mathcal{L}_{reg}$: L1 regularization enforces sparsity (activating only one basis direction per level). The transformation module follows HAFrame and supports either end-to-end training or training of the transformation module alone.

## Key Experimental Results

### Main Results

**CIFAR-100 (5-level hierarchy)**

| Method | Accuracy↑ | MS↓ | AHD@20↓ | HOPS↑ | HOPS@5↑ |
|--------|-----------|-----|---------|-------|---------|
| Cross Entropy | 77.77 | 2.33 | 3.19 | 0.54 | 0.05 |
| HAFrame | 80.55 | 2.00 | 2.45 | 0.86 | 0.81 |
| **Hier-COS** | **81.75** | 2.09 | **2.44** | **0.89** | **0.84** |

**iNaturalist-19 (7-level hierarchy)**: Hier-COS significantly outperforms HAFrame on HOPS, demonstrating its advantage in deep-hierarchy, large-category-count settings.

### Ablation Study

| Configuration | FPA↑ | Accuracy↑ | Note |
|---------------|------|-----------|------|
| Cross Entropy | 77.11 | 77.77 | Large Accuracy–FPA gap = inconsistency |
| HAFrame | 77.0 | 80.55 | FPA lower than CE |
| **Hier-COS** | **82.91** | **81.75** | FPA > Accuracy — strong consistency |

### Key Findings
- Hier-COS achieves Full Path Accuracy (FPA) improvements of 1.36–3.64% over HAFrame across all datasets, with the smallest Accuracy–FPA gap, confirming the theoretical guarantee of hierarchical consistency.
- The HOPS metric effectively distinguishes scenarios that AHD cannot: AHD@20 assigns the same score (2.06) to both optimal and worst-case orderings, while HOPS shows clear differentiation.
- Training only the transformation module on a frozen ViT backbone yields a 2.42% top-1 improvement, demonstrating that Hier-COS can efficiently convert pretrained features into hierarchy-aware representations.
- As $K$ increases, the proportion of correctly ordered predictions drops sharply for existing methods; Hier-COS maintains 64–74% at $K=20$, far exceeding other methods (~0%).

## Highlights & Insights
- **Elegant Unification of Theory and Practice**: The composition of orthogonal subspaces naturally encodes hierarchical structure. Theorem 1 provides rigorous theoretical guarantees, and Proposition 1 ensures inference-time hierarchical consistency—far more elegant than existing methods that approximate consistency via auxiliary loss terms.
- **HOPS Evaluation Metric**: Exposes the permutation-invariance flaw of AHD@k and proposes HOPS, which simultaneously accounts for top-1 accuracy and the ranking preference for error severity. The special case HOPS@1 = top-1 accuracy is a particularly elegant property.
- **"Subspace Dimensionality = Learning Capacity" Insight**: The topological structure of the hierarchy tree automatically determines the representation space dimensionality for each class, requiring no manual design or hyperparameter tuning.

## Limitations & Future Work
- The feature space dimensionality $n$ equals the number of nodes in the hierarchy tree; extremely large hierarchies (e.g., tens of thousands of nodes) may lead to prohibitively high dimensionality.
- Validation is limited to image classification; extension to hierarchical classification in NLP or multimodal settings has not been explored.
- Although HOPS outperforms AHD, the choice of weight function $\eta_j$ (multi-step exponential linear decay) involves a degree of arbitrariness.
- The effect of orthogonal basis assignment (bijective but arbitrary) on results has not been thoroughly analyzed.

## Related Work & Insights
- **vs. HAFrame**: HAFrame also employs fixed frames for hierarchy-aware classification, but restricts features to a one-dimensional direction along weight vectors. Hier-COS introduces subspace composition, providing multi-dimensional representation spaces and adaptive capacity.
- **vs. Flamingo**: Flamingo uses label embeddings to learn hierarchical similarity but does not guarantee hierarchical consistency.
- **vs. Hyperbolic Embeddings**: Hyperbolic spaces naturally encode hierarchies but require manifold optimization. Hier-COS achieves analogous effects in Euclidean space via orthogonal subspaces, with considerably less complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The idea of encoding hierarchical structure through orthogonal subspace composition is original and elegant, with complete theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets with multi-metric comparisons, though NLP and large-scale settings are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, problem-driven exposition, and particularly outstanding critical analysis of evaluation metrics.
- Value: ⭐⭐⭐⭐ — Contributes both methodologically and in terms of evaluation metrics to the hierarchical classification field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](../../ICLR2026/llm_evaluation/deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[ICCV 2025\] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](../../ICCV2025/llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/llm_evaluation/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)

</div>

<!-- RELATED:END -->
