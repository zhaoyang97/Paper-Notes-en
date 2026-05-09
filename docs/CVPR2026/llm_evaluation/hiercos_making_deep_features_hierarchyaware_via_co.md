---
title: >-
  [Paper Note] Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces
description: >-
  [CVPR 2026][LLM Evaluation][hierarchical classification] This paper proposes Hier-COS, a framework that assigns orthogonal basis vectors to each node in a label hierarchy tree and constructs a Hierarchy-Aware Vector Space (HAVS) via subspace composition (ancestor bases + self basis + descendant bases). The approach provides theoretical guarantees that the distance structure of the feature space is consistent with the hierarchy tree, while also introducing the HOPS evaluation metric to address the permutation-invariance deficiency of existing hierarchical evaluation metrics.
tags:
  - CVPR 2026
  - LLM Evaluation
  - hierarchical classification
  - orthogonal subspaces
  - hierarchy-aware features
  - evaluation metric
  - HOPS
date: 2026-05-08
content_hash: 6373c64e9f29c0a5
---

# Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces

**Conference**: CVPR 2026
**arXiv**: [2503.07853](https://arxiv.org/abs/2503.07853)
**Code**: [https://sites.google.com/iiitd.ac.in/hier-cos](https://sites.google.com/iiitd.ac.in/hier-cos)
**Area**: Representation Learning / Hierarchical Classification
**Keywords**: hierarchical classification, orthogonal subspaces, hierarchy-aware features, evaluation metric, HOPS

## TL;DR
This paper proposes Hier-COS, a framework that assigns orthogonal basis vectors to each node in a label hierarchy tree and constructs a Hierarchy-Aware Vector Space (HAVS) via subspace composition (ancestor bases + self basis + descendant bases). The approach provides theoretical guarantees that the distance structure of the feature space is consistent with the hierarchy tree, while also introducing the HOPS evaluation metric to address the permutation-invariance deficiency of existing hierarchical evaluation metrics.

## Background & Motivation
Conventional classifiers treat all class labels as mutually independent and regard all errors as equivalent. In practice, however, class labels typically exhibit semantic hierarchy (e.g., animal → bird → eagle), and misclassifying a semantically distant class (e.g., predicting "fish" for "eagle") is far more severe than misclassifying a semantically close class (e.g., predicting "falcon" for "eagle"). Although existing hierarchy-aware methods (HAFrame, Flamingo, HAFeat) achieve competitive performance on MS/AHD metrics, the authors identify two critical issues: (1) the existing evaluation metric AHD@k is a permutation-invariant statistic — the optimal and worst top-k rankings can yield identical AHD@k scores, failing to genuinely measure hierarchical performance; (2) existing methods assign each class a 1-dimensional feature direction, which may result in small angular separation for semantically similar classes and cannot adaptively adjust learning capacity across classes.

## Core Problem
How to construct a feature space with theoretical guarantees of consistency with the hierarchy tree structure, capable of unifying "hierarchy-aware fine-grained classification" and "hierarchical multi-level classification," while adapting to the varying complexity of different classes?

## Method

### Overall Architecture
Given a hierarchy tree $\mathcal{T}$ with $n$ nodes, Hier-COS assigns each node an orthogonal basis vector $e_i$, defining an $n$-dimensional orthogonal vector space $V_\mathcal{T}$. The subspace for each node $v_i$ is spanned by its ancestor bases, self basis, and descendant bases: $V_i = \text{span}(\mathcal{E}_i^a \cup \{e_i\} \cup \mathcal{E}_i^d)$. Features extracted by the backbone are mapped to $V_\mathcal{T}$ via a lightweight transformation module, and classification is performed by computing the projection distance from the feature vector to each leaf-node subspace.

### Key Designs
1. **Hierarchy-Aware Vector Space (HAVS)**: A formal definition of HAVS is introduced — the distance from a point to a subspace in the vector space must maintain a partial order consistent with the LCA tree distance. Theorem 1 proves that the space constructed by Hier-COS satisfies the HAVS definition. Key property: semantically closer classes share more basis vectors → greater subspace overlap → smaller distances.

2. **Adaptive Learning Capacity**: Subspace dimensionality equals the number of ancestors + 1 + the number of descendants, automatically scaling with the complexity of a class in the hierarchy tree. Deeper classes (e.g., $\{D_6,\ldots,D_{10}\}$, which share more ancestors) require larger dimensions to learn discriminative features, while shallower classes (e.g., $\{A_2, A_3\}$) have smaller dimensions. This property is entirely absent in existing methods.

3. **Unified Hierarchical Multi-Level Classification**: The same classifier can make predictions at any hierarchical level — it suffices to compute distances to the subspace of nodes at that level. Proposition 1 proves that the prediction path is guaranteed to be consistent within the tree (i.e., all ancestor predictions of the leaf-node prediction are correct ancestor nodes).

4. **HOPS Evaluation Metric**: A novel metric based on preference ranking. For each ground-truth class, an expected preference ranking $z$ is constructed based on LCA distances and compared with the predicted ranking $\hat{z}$ using exponential-linear decay weights to compute a weighted discrepancy. HOPS@1 is equivalent to top-1 accuracy, and HOPS@k naturally extends to top-k evaluation. This overcomes the permutation-invariance deficiency of AHD@k.

### Loss & Training
- Tree path KL divergence loss $\mathcal{L}_{kl}$: The target distribution $P$ concentrates more energy toward leaf nodes using exponentially increasing weights $w_l = \exp(1/(h+1-l))$.
- Regularization $\mathcal{L}_{reg}$: Enforces feature vector sparsity — only one basis direction is activated per level, and basis directions not on the class path should be zero.
- Transformation module: 5 layers of Linear + BN + PReLU, with the final layer fixed as orthogonal bases.

## Key Experimental Results

| Dataset | Method | Accuracy↑ | MS↓ | AHD@1↓ | HOPS↑ | HOPS@5↑ |
|---|---|---|---|---|---|---|
| CIFAR-100 | Cross Entropy | 77.77 | 2.33 | 2.25 | 0.54 | 0.05 |
| CIFAR-100 | HAFrame | 77.53 | 2.24 | 1.12 | 0.92 | 0.72 |
| CIFAR-100 | **Hier-COS** | **77.79** | **2.21** | **1.09** | **0.93** | **0.76** |
| iNat-19 | HAFrame | 71.13 | 2.05 | 1.14 | 0.89 | 0.70 |
| iNat-19 | **Hier-COS** | 71.15 | 2.06 | **1.13** | **0.96** | **0.71** |
| iNat-19 (ViT) | Cross Entropy | 78.39 | 1.72 | 1.38 | 0.53 | 0.52 |
| iNat-19 (ViT) | **Hier-COS** | **80.81** | 1.73 | **0.97** | **0.98** | **0.80** |
| FGVC-Aircraft | HAFrame | 80.55 | 2.00 | 1.74 | 0.86 | 0.81 |
| FGVC-Aircraft | **Hier-COS** | **81.75** | 2.09 | **1.73** | **0.89** | **0.84** |

- Full Path Accuracy (FPA): +3.64% on FGVC-Aircraft, +1.36% on CIFAR-100, +1.51% on iNat-19.
- With a frozen ViT backbone training only the transformation module: accuracy on iNat-19 improves by 2.42% (78.39 → 80.81), and HOPS improves from 0.53 → 0.98.

### Ablation Study
- Regularization $\mathcal{L}_{reg}$ improves cosine similarity from 0.87 to 0.97, which is critical for ensuring feature sparsity.
- Expanding the subspace dimensionality from 1D ($V_i = \text{span}(\{e_i\})$) to full ancestor + descendant subspaces yields significant HOPS gains with little change in level-wise accuracy — indicating that the additional dimensions primarily help reduce the severity of errors.
- The direction of the weight distribution $w_l$ is critical: concentrating on the coarse-grained end renders leaf classes indistinguishable (accuracy drops to 51%); uniform distribution yields 70% accuracy; concentrating on the fine-grained end is optimal (78% accuracy).
- The hyperparameter $\alpha$ is insensitive, with stable performance across different values.

## Highlights & Insights
- **Theoretical elegance**: The hierarchical classification problem is fully formalized as subspace relationships in a vector space, with a proof that the construction satisfies HAVS.
- **Implicit hierarchical consistency**: No additional constraints are needed to enforce consistency, as orthogonal subspace composition inherently guarantees it.
- **Adaptive capacity**: Subspace dimensionality automatically adapts to class complexity without manual design.
- **HOPS metric** fills a critical gap in hierarchical evaluation — the permutation-invariance problem of AHD@k is effectively resolved.
- **Lightweight**: Only the transformation module needs to be trained; the backbone can be frozen.

## Limitations & Future Work
- Top-1 accuracy on tieredImageNet-H (12-level, imbalanced tree) is slightly below HAFrame (72.22 vs. 73.70); deep imbalanced trees remain a challenge.
- The space dimensionality $n$ grows linearly with the number of nodes, which may lead to dimensionality explosion for large-scale hierarchies (e.g., ImageNet-21k).
- The authors propose using the kernel trick to implicitly map to a high-dimensional Hier-COS space, but this has not yet been implemented.
- The current approach is limited to tree structures; DAG extension is discussed as feasible but not experimentally validated.

## Related Work & Insights
- **vs. HAFrame**: HAFrame fixes weight vectors as hierarchy-aware frames, but each class is still limited to a 1-dimensional direction; Hier-COS provides adaptive capacity through multi-dimensional subspaces with substantially improved HOPS.
- **vs. Flamingo/HAFeat**: These methods require training independent classifiers for each hierarchical level with complex loss functions; Hier-COS unifies all levels within a single classifier.
- **vs. hyperbolic embedding methods**: These require manifold optimization and have not been validated on deep fine-grained visual hierarchical classification; Hier-COS achieves analogous effects in Euclidean space via orthogonal subspaces.
- The paradigm of orthogonal subspace composition may generalize to multi-task learning (composing subspaces across different tasks).
- The HOPS metric can be directly applied to evaluate any hierarchical system.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Modeling hierarchical classification as a composition of orthogonal subspaces is a genuinely novel and mathematically elegant perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, multiple backbones, detailed ablations and qualitative analysis; however, accuracy on tieredImageNet falls below SOTA.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretically rigorous, with in-depth analysis of evaluation metrics and extremely detailed supplementary material.
- **Value**: ⭐⭐⭐⭐ Advances hierarchical classification both theoretically and methodologically; the HOPS metric has independent value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)
- [\[CVPR 2026\] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)
- [\[CVPR 2026\] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)
- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)

<!-- RELATED:END -->
