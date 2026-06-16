---
title: >-
  [Paper Note] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion
description: >-
  [NeurIPS 2025][Incomplete multi-view clustering] This paper proposes the HSACC framework, which employs a two-level semantic space design (low-level mutual information consistency + high-level adaptive weighted fusion) c…
tags:
  - "NeurIPS 2025"
  - "Incomplete multi-view clustering"
  - "hierarchical semantic alignment"
  - "dynamic weighted fusion"
  - "MMD"
  - "cooperative completion"
date: 2026-05-08
content_hash: 3f278b19d2e216f0
---

# Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion

**Conference**: NeurIPS 2025
**arXiv**: [2510.13887](https://arxiv.org/abs/2510.13887)  
**Code**: [GitHub](https://github.com/XiaojianDing/2025-NeurIPS-HSACC)  
**Area**: LLM Evaluation
**Keywords**: Incomplete multi-view clustering, hierarchical semantic alignment, dynamic weighted fusion, MMD, cooperative completion

## TL;DR

This paper proposes the HSACC framework, which employs a two-level semantic space design (low-level mutual information consistency + high-level adaptive weighted fusion) combined with cooperatively optimized implicit missing-view recovery, achieving significant improvements over existing incomplete multi-view clustering methods on five benchmark datasets.

## Background & Motivation

Incomplete multi-view data—where certain samples are entirely missing in specific views—is prevalent in practice due to sensor limitations, occlusion, and varying data acquisition conditions. Such missingness disrupts cross-view correlations, amplifies noise, and introduces bias.

Two critical limitations of existing deep Incomplete Multi-View Clustering (IMVC) methods:

1. **Insufficient fusion strategies**: Static fusion (e.g., uniform weighting) cannot adapt to distributional discrepancies across views; dynamic fusion methods lack hierarchical semantic separation and fail to distinguish low-level consistency alignment from high-level semantic fusion, resulting in the loss of multi-granularity information.
2. **Error propagation in two-stage pipelines**: Completing missing views prior to clustering causes errors from the completion stage to propagate into clustering. Although joint optimization has been attempted, existing approaches overlook the varying importance of different views in the fused representation, preventing completion from fully exploiting high-quality view information.

## Method

### Overall Architecture

HSACC comprises three modules:

1. **View Reconstruction**: View-specific autoencoders extract and reconstruct features.
2. **Multi-view Representation Learning**: Low-level mutual information alignment → high-level adaptive weighted fusion → distribution discrepancy minimization.
3. **Data Recovery and Clustering**: Implicit missing-view completion → joint optimization.

### Key Designs

1. **Two-level Semantic Space Alignment**:
    - **Function**: Ensures cross-view consistency via mutual information maximization in the low-level semantic space, and performs adaptive weighted fusion in the high-level semantic space.
    - **Mechanism**: *Low-level*—computes cross-view feature similarity matrix $P_{(m,n)} = \frac{1}{N}\sum_{i=1}^N z_{i,m}^1 \cdot z_{i,n}^2$, and derives mutual information loss $L_{MMI}$ from joint and marginal distributions. *High-level*—obtains an initial fusion $\mathbf{R}$ via concatenation, measures the distributional gap between each view and $\mathbf{R}$ using MMD, and assigns weights via softmax: $W^v = \frac{\exp(-D(\mathbf{Z}^v, \mathbf{R}))}{\sum_v \exp(-D(\mathbf{Z}^v, \mathbf{R}))}$.
    - **Design Motivation**: Low-level alignment captures shared patterns, while high-level weighted fusion dynamically assigns each view's contribution based on its consistency with the fused representation.

2. **Cooperative Completion Mechanism**:
    - **Function**: Projects aligned latent representations to a high-dimensional semantic space via MLP to implicitly recover missing views.
    - **Mechanism**: $\mathbf{Q}^1 = f_\text{MLP1}(\mathbf{Z}^1)$ predicts the latent representation of view 2. An inference consistency loss $L_{INF} = \frac{1}{N}\sum_i \|\mathbf{z}_i^2 - \mathbf{q}_i^1\|_2^2 + \frac{1}{N}\sum_i \|\mathbf{z}_i^1 - \mathbf{q}_i^2\|_2^2$ ensures the inferred representations align with the true latent representations.
    - **Design Motivation**: Completion is performed in latent space rather than input space, avoiding the generation of low-quality raw features; joint optimization enables mutual reinforcement between completion and clustering.

3. **Distribution Alignment Loss**:
    - **Function**: Minimizes the distributional discrepancy between the high-level shared representation $\mathbf{H}$ and each view $\mathbf{Z}^v$ via MMD in RKHS.
    - **Mechanism**: $L_{MMD} = \sum_{v=1}^V \text{MMD}^2(\mathcal{P}_{\mathbf{Z}^v}, \mathcal{Q}_\mathbf{H})$, approximated using kernel matrices.
    - **Design Motivation**: Promotes information exchange between the global representation and individual views, enhancing the consistency and complementarity of cross-view representations.

### Loss & Training

Total loss: $L = \lambda_1 L_{REC} + \lambda_2 L_{INF} + \lambda_3 L_{MMI} + \lambda_4 L_{MMD}$

- $L_{REC}$: Reconstruction loss (MSE)
- $L_{INF}$: Inference consistency loss (introduced from epoch $E_1$ onward)
- $L_{MMI}$: Cross-view mutual information loss
- $L_{MMD}$: Distribution alignment loss

Training strategy: The first $E_1$ epochs train the representation learning module; thereafter, completion and clustering are jointly optimized. Final clustering is performed via k-means on the concatenated complete-view representations.

## Key Experimental Results

### Main Results (Table)

ACC comparison across five datasets at varying missing rates (selected results):

| Missing Rate | Dataset | HSACC | Runner-up | Gain |
|---|---|---|---|---|
| 0.5 | Caltech101-20 | **Best** | DCP/DSIMVC | ACC +5.3%, ARI +8.57% |
| 0.3→0.7 | Noisy MNIST | Drop 6.92% | ICMVC drop 35.19% | **~5× more robust** |
| 0.3 | Hdigit | **Best** | — | — |
| 0.5 | LandUse_21 | **Best** | — | — |
| 0.5 | 100leaves | **Best** | — | — |

### Ablation Study

- Removing hierarchical alignment ($L_{MMI}$): Significant performance degradation across all datasets.
- Removing dynamic weighting (fixed uniform weights): ACC drops by 2–5%.
- Removing cooperative completion ($L_{INF}$): Marked degradation at high missing rates.
- Removing $L_{MMD}$: Cross-view fusion quality degrades.

### Key Findings

- HSACC achieves the best or near-best performance across all five datasets at all missing rates.
- Robustness at high missing rates (0.7) far surpasses competing methods—ACC drops only 6.92% on Noisy MNIST, compared to 35.19% for ICMVC.
- The combination of hierarchical alignment and dynamic weighting contributes most significantly.
- Hyperparameter analysis shows the model is insensitive to variations in $\lambda_1 \sim \lambda_4$.

## Highlights & Insights

- **Hierarchical semantic separation**: Explicitly distinguishes low-level consistency alignment from high-level semantic fusion, enabling more complete preservation of multi-granularity information.
- **Adaptive view weighting**: Dynamically allocates contributions based on distributional affinity, preventing low-quality views from dominating the fusion.
- **Elimination of error propagation**: Joint optimization of completion and clustering allows the completion process to be guided by discriminative features.
- Comprehensively outperforms nine SOTA methods across five datasets with statistical significance.

## Limitations & Future Work

- Experiments are limited to two-view settings; generalization to more than two views requires further validation.
- MLP-based nonlinear mappings for cross-view inference may be insufficient for complex inter-view relationships.
- MMD with a linear kernel may lack sensitivity to complex distributional discrepancies.
- The autoencoder architecture (fully connected 1024-1024-1024) is fixed; the impact of alternative architectures is unexplored.
- Experiments on large-scale datasets (e.g., ImageNet subsets) are absent.
- Graph-structured information is not utilized.

## Related Work & Insights

- **COMP (2023)**: Achieves recovery and consistency learning via contrastive learning and dual prediction, but lacks hierarchical semantic separation.
- **DCP**: A contrastive prediction framework; this work extends it by introducing dynamic weighting and MMD alignment.
- **DSIMVC**: A bilevel optimization framework for dynamic completion, but does not account for view weight differences.
- The hierarchical semantic alignment paradigm is generalizable to other multimodal fusion problems.

## Rating

⭐⭐⭐⭐ — The method is systematically designed and effective, with significant and consistent advantages across multiple datasets and thorough ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/others/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](../../AAAI2026/others/hierarchical_semantic_alignment_for_image_clustering.md)
- [\[NeurIPS 2025\] Graph Alignment via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)

</div>

<!-- RELATED:END -->
