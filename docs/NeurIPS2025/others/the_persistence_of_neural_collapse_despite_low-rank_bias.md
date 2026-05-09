---
title: >-
  [Paper Note] The Persistence of Neural Collapse Despite Low-Rank Bias
description: >-
  [NeurIPS 2025][Neural Collapse] This paper theoretically demonstrates that Deep Neural Collapse (DNC) is globally suboptimal in deep unconstrained feature models due to the low-rank bias induced by L2 regularization, while providing the first theoretical explanation for the persistent empirical occurrence of DNC — its solution-space dimensionality grows faster with network width than that of low-rank solutions.
tags:
  - NeurIPS 2025
  - Neural Collapse
  - Low-Rank Bias
  - Deep Unconstrained Feature Model
  - Loss Landscape
  - Schatten Quasi-Norm
date: 2026-05-08
content_hash: 0d63244bf89c7f2b
---

# The Persistence of Neural Collapse Despite Low-Rank Bias

**Conference**: NeurIPS 2025
**arXiv**: [2410.23169](https://arxiv.org/abs/2410.23169)
**Code**: None
**Area**: Deep Learning Theory
**Keywords**: Neural Collapse, Low-Rank Bias, Deep Unconstrained Feature Model, Loss Landscape, Schatten Quasi-Norm

## TL;DR

This paper theoretically demonstrates that Deep Neural Collapse (DNC) is globally suboptimal in deep unconstrained feature models due to the low-rank bias induced by L2 regularization, while providing the first theoretical explanation for the persistent empirical occurrence of DNC — its solution-space dimensionality grows faster with network width than that of low-rank solutions.

## Background & Motivation

Neural Collapse (NC) is a structured geometric phenomenon observed in the terminal phase of training deep neural network classifiers: last-layer features collapse to their class means, the class means form a simplex equiangular tight frame (ETF), and the classifier weights align with the features. NC has also been observed in earlier layers, a phenomenon referred to as Deep Neural Collapse (DNC).

**Existing theoretical results**: In single-layer unconstrained feature models (UFM), NC has been proven to be the globally optimal solution, with the loss landscape being a strict saddle function (admitting only global optima and non-degenerate saddle points).

**Key Challenge**: Sukenik et al. (2024) proved that in deep UFMs with ReLU activations and MSE loss, DNC is not globally optimal, since the low-rank bias induced by L2 regularization allows solutions of lower rank to achieve lower loss. However, they did not analyze whether DNC or low-rank solutions constitute local optima, nor did they explain why DNC persists empirically despite being suboptimal.

**Key Insight**: This paper conducts a systematic analysis using a deep UFM with cross-entropy (CE) loss and linear layers. Linear layers facilitate theoretical analysis, while the unconstrained feature assumption in the UFM compensates for the limited expressiveness of linear layers. The paper aims to: (1) comprehensively characterize how low-rank bias shapes the loss landscape, and (2) provide the first explanation for the empirical persistence of DNC.

## Method

### Overall Architecture

Consider a $K$-class classification problem with $n$ samples per class. The loss function of the deep UFM is:
$$\mathcal{L}(H_1, W_1, ..., W_L) = g(Z) + \sum_{l=1}^{L} \frac{1}{2}\lambda \|W_l\|_F^2 + \frac{1}{2}\lambda \|H_1\|_F^2$$
where $g(Z)$ denotes the cross-entropy loss and $Z$ is the logit matrix. A key observation is that the regularization terms are equivalent to the Schatten $2/L$ quasi-norm:
$$\frac{1}{2}L\lambda \|X\|_{S_{2/L}}^{2/L} = \min \{\text{regularization terms}\}$$
As $L$ increases, the Schatten $2/L$ quasi-norm approaches the rank of the matrix, so deeper networks inherently favor low-rank solutions.

### Key Designs

1. **Global Suboptimality of DNC (Theorem 1)**: For the deep linear UFM, if $K \geq 4$ and $L \geq 3$ (or $K \geq 6$ and $L = 2$), no solution with DNC structure can be globally optimal. The proof constructs a block-diagonal low-rank logit matrix (each $2\times2$ block of the form $[[1,-1],[-1,1]]$) which, under equal-scaling conditions, achieves lower loss than the DNC solution (rank $K$). This reveals that the optimal structure of single-layer UFMs does not generalize to deep models.

2. **General High-Rank Suboptimality (Theorem 2)**: For any fixed structure whose rank exceeds the minimum rank required to fit the data, when the number of layers $L$ is sufficiently large, that structure is necessarily suboptimal. The key concept introduced is a *diagonally superior matrix*: one in which the correct class score is highest for every sample. Such matrices can achieve arbitrarily small fitting loss through appropriate scaling.

3. **Low-Rank Nature of Global Optima (Theorem 3)**: When the regularization satisfies $\lambda_L = o(L^{-1})$, the global optimal solution $Z_L^*$ has at most $q_K$ (the minimum rank of a diagonally superior matrix, $\leq 2$) singular values that are nonzero or do not decay at an exponential rate. This implies that the optimal solution is approximately low-rank, with most singular values decaying exponentially with $L$. The rank of DNC is $K-1$, a substantial gap from the optimal rank of $\leq 2$.

4. **Explanation for the Persistence of DNC (Theorems 4–5)**:

    - **Theorem 4**: When the regularization $\lambda$ is sufficiently small, the DNC solution is a critical point and the Hessian has no negative eigenvalues (i.e., it is a local minimum or degenerate saddle point). This contrasts with the strict saddle property of single-layer UFMs.
    - **Theorem 5**: The ratio $R(d)$ of the parameter-space dimensionality of DNC solutions $D_{DNC}$ to that of low-rank solutions $D_{Z^*}$ is a monotonically increasing function of width $d$, growing from $<1$ toward $(K-1)/r > 1$. As $d$ increases, DNC occupies an increasingly larger "volume" in the loss landscape, eventually dominating low-rank solutions.

5. **Extension to ReLU (Theorem 7)**: In ReLU UFMs, DNC is likewise globally suboptimal when $K \geq 10$ and $L \geq 5$ (or $K \geq 16$ and $L = 4$). The proof proceeds by showing that the DNC loss in the linear model is a lower bound on the DNC loss in the ReLU model.

### Loss & Training

Cross-entropy loss with L2 regularization (weight decay) is applied to all parameters, including the feature matrix $H_1$ in the UFM framework. The theoretical analysis focuses on how the regularization parameter $\lambda$ and the number of layers $L$ jointly shape the global and local structure of the loss landscape.

## Key Experimental Results

### Main Results: Deep Linear UFM

| Setting | DNC Solution Loss | Low-Rank Solution Loss | Observation |
|---------|------------------|----------------------|-------------|
| $L=2$, $d=70$, $K=10$, $\lambda=2^{-10}$ | Higher | Lower | Low-rank solution outperforms DNC (validates Theorem 1) |
| Logit matrix at convergence | Simplex ETF | Block-diagonal structure | Two distinct convergence structures observed |

### Ablation Study: Effect of Width and Regularization on DNC Frequency

| Parameter Variation | DNC Occurrence Rate | Remark |
|--------------------|--------------------|---------| 
| $d$ increasing | ~0% → ~100% | Validates Theorem 5: wider networks are more prone to DNC |
| $\lambda$ decreasing | Rate increases | DNC and low-rank solution losses become closer under small regularization |
| MNIST, $L=3$, linear head | Low-rank solution outperforms DNC bound | 4 nonzero singular values |
| CIFAR-10, $L=3$, linear head | Low-rank solution outperforms DNC bound | 3 nonzero singular values |
| CIFAR-10, standard regularization | Low-rank structure emerges | Logit matrix does not form a simplex |

### Key Findings

- Low-rank bias is pervasive even in real networks (ResNet-20 with fully connected head)
- Low-rank structure persists when using standard weight decay (non-UFM-style regularization)
- Low-rank bias is also present under ReLU activations; the linear model effectively captures the key phenomenon
- Network width is the critical factor governing the probability of DNC — the "volume" of the DNC region grows exponentially with width
- Hessian analysis indicates that DNC acts as a local attractor (positive semidefinite Hessian), explaining the convergence of gradient descent to DNC

## Highlights & Insights

- **First complete explanation for the persistence of DNC**: Beyond proving DNC is suboptimal, the paper explains why it still occurs frequently — through differences in solution-space dimensionality
- **Fundamental distinction between deep and single-layer models**: Single-layer UFMs exhibit strict saddle geometry, whereas deep UFMs admit degenerate saddle points or local minima, fundamentally altering the geometry of the optimization landscape
- **Quantitative characterization of low-rank bias**: Beyond qualitatively establishing that lower rank is preferable, the paper precisely characterizes the singular value decay rate and the minimum rank of diagonally superior matrices
- **Tight integration of theory and experiment**: Each theoretical result is accompanied by corresponding numerical validation

## Limitations & Future Work

- The analysis focuses on theoretical characterization and does not examine downstream performance (generalization, robustness)
- The UFM assumption requires overparameterization and may not apply to underparameterized regimes
- The interaction of practical factors such as initialization scale, batch size, and optimizer choice with low-rank bias is not considered
- Theorem 5 provides only a heuristic dimensionality argument and does not rigorously establish that dimensionality advantage implies a convergence probability advantage
- Analysis for imbalanced datasets and very large numbers of classes remains limited

## Related Work & Insights

This paper advances several important theoretical frontiers in the NC research literature. (1) It extends the results of Sukenik et al. from the MSE + ReLU setting to CE + linear and CE + ReLU settings. (2) It provides the first analysis of the local properties of DNC on the loss landscape via Hessian analysis. (3) It connects low-rank bias to the Schatten quasi-norm theory from the matrix completion literature. The paper offers new theoretical insights into how hyperparameters such as width, depth, and regularization influence the structure of internal representations in foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First complete explanation of the paradox between DNC suboptimality and its empirical persistence
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical results are well-validated, though experimental scale is limited
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical development proceeds in a logically layered manner with precise theorem statements
- Value: ⭐⭐⭐⭐ Deepens theoretical understanding of training dynamics in deep learning

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] MaxSup: Overcoming Representation Collapse in Label Smoothing](maxsup_overcoming_representation_collapse_in_label_smoothing.md)

<!-- RELATED:END -->
