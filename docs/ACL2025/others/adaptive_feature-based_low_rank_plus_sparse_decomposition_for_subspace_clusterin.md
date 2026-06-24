---
title: >-
  [Paper Note] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering
description: >-
  [ACL 2025][Subspace Clustering] This paper proposes an adaptive feature-driven low-rank plus sparse matrix decomposition method. By adaptively learning the weights of low-rank and sparse components in the feature space, it addresses the issues of noise robustness and insufficient feature discriminability in subspace clustering.
tags:
  - "ACL 2025"
  - "Subspace Clustering"
  - "Low-Rank Decomposition"
  - "Sparse Decomposition"
  - "Adaptive Features"
  - "Matrix Decomposition"
date: 2026-05-08
content_hash: cbe263d0eec4d65e
---

# Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering

**Conference**: ACL 2025  
**Area**: Others (Machine Learning/Representation Learning)  
**Keywords**: Subspace Clustering, Low-Rank Decomposition, Sparse Decomposition, Adaptive Features, Matrix Decomposition

## TL;DR
This paper proposes an adaptive feature-driven low-rank plus sparse matrix decomposition method. By adaptively learning the weights of low-rank and sparse components in the feature space, it addresses the issues of noise robustness and insufficient feature discriminability in subspace clustering.

## Background & Motivation

**Background**: Subspace clustering is a classic problem in unsupervised learning, where the core assumption is that high-dimensional data lie on a union of multiple low-dimensional subspaces. Streamline methods including Sparse Subspace Clustering (SSC) and Low-Rank Representation (LRR) construct a self-representation matrix of the data and apply spectral clustering to discover the subspace structures.

**Limitations of Prior Work**: Traditional low-rank plus sparse decomposition methods (such as RPCA) impose the same regularization strength on all features, ignoring the differing contributions of different feature dimensions to clustering. Some features may contain a large amount of noise and be detrimental to clustering, while others carry crucial discriminative information. A one-size-fits-all regularization strategy fails to distinguish between these two scenarios.

**Key Challenge**: The low-rank constraint pursues global structural consistency, while the sparse constraint handles local outliers. However, their optimal trade-off points vary across different feature dimensions. A globally uniform balancing strategy leads to over-smoothing on some features and under-constraint on others.

**Goal**: To design a method that can adaptively learn the optimal trade-off of low-rank and sparse decomposition for each feature dimension, thereby improving the robustness of subspace clustering on noisy data.

**Key Insight**: From the perspective of feature selection, generalize the regularization parameters in low-rank plus sparse decomposition from scalars to vectors, allowing each feature dimension to have an independent regularization weight.

**Core Idea**: Replace the fixed regularization parameters with an adaptive feature weight vector, simultaneously learning the decomposition results and feature importances during the optimization process.

## Method

### Overall Architecture
Given a data matrix $X \in \mathbb{R}^{d \times n}$, it is decomposed into a low-rank component $L$ and a sparse component $S$: $X = L + S$. The key innovation is the introduction of a feature weight vector $\mathbf{w} \in \mathbb{R}^d$, which imposes constraints of different strengths on the low-rank and sparse components across different feature dimensions. After the decomposition is completed, the low-rank component $L$ is utilized to construct an affinity matrix for spectral clustering.

### Key Designs

1. **Adaptive Feature-Weighted Decomposition**:

    - **Function**: Implement feature-dimension-by-feature-dimension adaptive regularization in the low-rank plus sparse decomposition.
    - **Mechanism**: Formulate the optimization problem as $\min_{L,S,\mathbf{w}} \|W_L \odot L\|_* + \lambda\|W_S \odot S\|_1 + \gamma R(\mathbf{w})$, where $W_L, W_S$ are weight matrices derived from the feature weights $\mathbf{w}$, and $R(\mathbf{w})$ is a regularization term on the weights (such as entropy regularization to prevent degeneracy). The decomposition results $L$, $S$, and the weights $\mathbf{w}$ are learned simultaneously via alternating optimization.
    - **Design Motivation**: Impose larger weights on the sparse components of highly noisy feature dimensions (encouraging noise to be categorized into the sparse term), while assigning larger weights to the low-rank components of informative feature dimensions (preserving structural information).

2. **ADMM Solver**:

    - **Function**: Efficiently solve the non-convex optimization problem with adaptive weights.
    - **Mechanism**: Decompose the original problem into three sub-problems: updating $L$ and $S$ with fixed weights (weighted nuclear norm and weighted $\ell_1$ minimization), and updating the weights $\mathbf{w}$ with given $L$ and $S$ (closed-form solution). ADMM guarantees convergence and computational efficiency.
    - **Design Motivation**: Directly optimizing the joint objective function is NP-hard. The decomposition strategy of ADMM enables each sub-problem to have an efficient solution.

3. **Affinity Construction**:

    - **Function**: Construct an affinity graph for spectral clustering from the learned low-rank representation.
    - **Mechanism**: Construct the affinity matrix $A = |C| + |C^T|$ using the self-representation coefficient matrix $C$ (such that $L = LC$), and then perform normalized spectral clustering. Since the low-rank component has already removed noise via adaptive weights, the quality of the affinity matrix is higher.
    - **Design Motivation**: Traditional methods that directly construct the affinity matrix on raw data are highly susceptible to noise interference.

### Loss & Training
The overall objective function includes weighted nuclear norm minimization (encouraging low-rank structure), weighted $\ell_1$ norm minimization (encouraging sparsity), and entropy authorization for the weight vector (preventing weight collapse to all-zero or all-one). ADMM is employed to solve the problem, and the convergence criterion is that the relative change between adjacent iterations is smaller than a predefined threshold.

## Key Experimental Results

### Main Results

| Dataset | Metric (ACC) | Ours | LRR | SSC | RPCA+SC | Gain |
|--------|----------|------|-----|-----|---------|------|
| Yale-B | ACC(%) | 89.7 | 83.2 | 85.6 | 84.1 | +4.1 |
| ORL | ACC(%) | 82.4 | 76.8 | 78.3 | 77.5 | +4.1 |
| COIL-20 | ACC(%) | 86.3 | 80.1 | 82.7 | 81.4 | +3.6 |
| Reuters | ACC(%) | 74.8 | 68.5 | 70.2 | 69.1 | +4.6 |

### Ablation Study

| Configuration | Yale-B(ACC) | NMI | Description |
|------|------------|-----|------|
| Full model | 89.7 | 0.912 | Full adaptive weights |
| Fixed equal weights | 84.1 | 0.867 | Degenerates to standard RPCA |
| Adaptive weights for low-rank only | 87.3 | 0.891 | Fixed weights for the sparse component |
| Adaptive weights for sparse only | 86.8 | 0.885 | Fixed weights for the low-rank component |
| Without entropy regularization | 85.9 | 0.876 | Weight collapse leading to degradation |

### Key Findings
- The full model with joint adaptive weighting for both low-rank and sparse components significantly outperforms any single-sided adaptation, demonstrating that the adaptation along both directions is complementary.
- Entropy regularization is critical for preventing weight degradation; removing it results in a performance drop of approximately 4 percentage points in accuracy.
- Under high-noise conditions (such as adding 20% salt-and-pepper noise), the advantage of the proposed method over the baselines is further amplified, indicating that the adaptive feature weights are particularly effective in terms of noise robustness.

## Highlights & Insights
- Incorporates the idea of feature selection into the matrix decomposition framework, implementing "soft feature selection" in an elegant manner that automatically reduces the impact of noisy features without discarding any. This scheme is transferable to any scenario requiring matrix decomposition.
- The ADMM solver framework results in very little computational complexity overhead after introducing adaptive weights, making it highly practical.

## Limitations & Future Work
- The computational overhead is high when the feature dimension is much larger than the number of samples, which limits its scalability; dimensional reduction might be necessary beforehand for million-dimensional text features.
- The convergence speed is affected by initialization; although ADMM guarantees convergence, it may require a relatively large number of iterations.
- The prior assumption on the number of subspaces still needs to be manually set, which can be integrated with methods that automatically determine the number of clusters in the future.
- Weight initialization may affect the final results. Although experiments show acceptable robustness, there is a lack of global optimality guarantees in theory.
- This work focuses on image and text data; its generalization capability to more modalities (such as time series, graph data) is worth exploring.
- In the future, the adaptive weight mechanism can be combined with deep learning feature extraction to realize end-to-end adaptive subspace clustering.

## Related Work & Insights
- **vs LRR**: LRR uses a globally unified nuclear norm constraint. In contrast, the formulated method achieves much finer-grained control through adaptive weights.
- **vs SSC**: SSC focuses on sparsity but overlooks the global low-rank structure. This paper considers both simultaneously.
- **vs RPCA**: The low-rank plus sparse decomposition of RPCA is a special case of this method (with constant weights). The proposed method is a natural generalization of RPCA.
- The adaptive weight mechanism of this method can also be applied to tasks like matrix completion in recommendation systems or anomaly detection.

## Rating
- Novelty: ⭐⭐⭐ The idea of adaptive weighting has precedents in the field of matrix completion, but its application to subspace clustering offers some degree of novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical formulation.
- Value: ⭐⭐⭐ The method is solid, but its impact might be limited to the subspace clustering community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[CVPR 2026\] Scalable Multi-View Subspace Clustering with Tensorized Anchor Guidance](../../CVPR2026/others/scalable_multi-view_subspace_clustering_with_tensorized_anchor_guidance.md)
- [\[ACL 2025\] Understanding Cross-Domain Adaptation in Low-Resource Topic Modeling](understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)

</div>

<!-- RELATED:END -->
