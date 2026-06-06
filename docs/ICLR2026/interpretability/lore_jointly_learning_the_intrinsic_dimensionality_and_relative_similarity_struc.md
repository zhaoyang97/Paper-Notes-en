---
title: >-
  [Paper Note] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data
description: >-
  [ICLR 2026][Interpretability][Ordinal embedding] This paper proposes LORE — the first framework to jointly learn embeddings and intrinsic dimensionality from ordinal triplet comparisons. It replaces the conventional fixe…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Ordinal embedding"
  - "intrinsic dimensionality recovery"
  - "Schatten-p quasi-norm"
  - "triplet comparisons"
  - "perceptual space"
  - "low-rank regularization"
date: 2026-05-08
content_hash: f47bf7005df91399
---

# LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data

**Conference**: ICLR 2026
**arXiv**: [2602.04192](https://arxiv.org/abs/2602.04192)  
**Code**: [GitHub](https://github.com/vivek2000anand/lore_iclr)  
**Area**: Representation Learning / Perceptual Modeling
**Keywords**: Ordinal embedding, intrinsic dimensionality recovery, Schatten-p quasi-norm, triplet comparisons, perceptual space, low-rank regularization

## TL;DR
This paper proposes LORE — the first framework to jointly learn embeddings and intrinsic dimensionality from ordinal triplet comparisons. It replaces the conventional fixed-dimension strategy with a non-convex Schatten-p quasi-norm regularizer ($p < 1$), solved via an iteratively reweighted nuclear norm (IRNN) algorithm with guaranteed convergence to a stationary point. Evaluated on synthetic data, LLM-simulated perceptual experiments, and three crowdsourced datasets, LORE substantially outperforms all baselines in dimensionality recovery while maintaining high triplet accuracy and semantic interpretability.

## Background & Motivation

**Background**: Ordinal embedding (OE) learns multidimensional representations of perceptual spaces from triplet comparisons of the form "Is A more similar to B or to C?" It is widely used in psychophysics for modeling subjective perception (taste, smell, aesthetic preference, etc.). Compared to absolute rating scales (e.g., Likert scales), triplet comparisons are language-free and immune to individual scale biases.

**Limitations of Prior Work**:
- All existing OE methods (SOE, FORTE, t-STE, CKL, OENN) require the user to pre-specify the embedding dimensionality $d'$.
- No principled criterion exists for determining the "true dimensionality," leading practitioners to default to inflated dimensions.
- Excessive dimensionality obscures the true structure (e.g., a 10-dimensional embedding where only 2 dimensions suffice causes "sweetness" to be fragmented across multiple axes).
- Scientific discovery favors parsimony (Occam's razor): lower-dimensional representations are more interpretable and computationally efficient.
- The only prior attempt at dimensionality recovery (Künstle's method) enumerates candidate dimensions and trains separately for each — not scalable.

**Key Insight**: Integrate dimensionality discovery directly into the OE optimization by using Schatten-p quasi-norm regularization to automatically balance triplet accuracy against embedding rank, eliminating the need to pre-specify dimensionality.

## Method

### Problem Formulation
Given $N$ perceptual objects and a set of triplets $T = \{(a, i, j)\}$ (indicating that $a$ is more similar to $i$ than to $j$), the goal is to learn an embedding matrix $Z \in \mathbb{R}^{N \times d'}$ such that the distance structure in the embedding space is consistent with the triplets and the intrinsic dimensionality $d \ll N$ is automatically recovered.

### LORE Optimization Objective
$$\min_Z \Psi(Z) = \underbrace{\sum_{(a,i,j)\in T} \log(1+\exp(1+d(z_a,z_i)-d(z_a,z_j)))}_{\text{smooth triplet loss}} + \lambda \underbrace{\sum_{i=1}^{\min\{N,d'\}} \sigma_i(Z)^p}_{\text{Schatten-}p\text{ regularization}}$$

**Three Key Design Choices**:

1. **Schatten-p Quasi-Norm ($0 < p < 1$)**:
    - $p = 1$ → nuclear norm (convex but uniformly shrinks all singular values → high bias)
    - $p \to 0$ → rank function (NP-hard)
    - $p = 0.5$ (paper default) → non-convex but more accurate low-rank approximation; imposes smaller penalties on large singular values and larger penalties on small ones, automatically "killing" redundant dimensions.

2. **Softplus Smoothing**: Replaces hinge loss with $\log(1 + \exp(\cdot))$, eliminating zero-gradient plateaus and making the objective differentiable everywhere (except at embedding collapse, which is avoided via wide initialization).

3. **Direct Embedding Optimization**: Optimizes $Z$ rather than the Gram matrix $G = ZZ^\top$, yielding $O(Nd')$ complexity vs. $O(N^2)$, enabling scalability to large datasets.

### Iteratively Reweighted Algorithm (Algorithm 1)
- Perform SVD at each step: $U, S, V^\top = \text{SVD}(Z^k - (1/\mu)\nabla f(Z^k))$
- Update singular values: $S^k = S - (p/\mu)\sigma^{p-1}$, truncating negative values
- Reconstruct embedding: $Z^{k+1} = U \cdot S^k \cdot V^\top$
- Convergence criterion: change in objective value or embedding falls below threshold
- Per-step complexity: $O(d'(T + Nd'))$

### Convergence Guarantee
**Theorem**: The embedding sequence $\{Z^k\}$ generated by LORE converges to a stationary point, i.e., $\sum_{k=1}^{\infty}\|Z^{k+1}-Z^k\|_F < +\infty$.

This is a significant guarantee: although the objective is highly non-convex, empirical and theoretical studies on OE suggest that stationary points are generally close to the global optimum (Bower et al. prove that for $d = 2$, all local optima are global optima).

### Hyperparameter Settings
- $p = 0.5$ (fixed; validated by prior studies)
- $\mu = 0.1$ (fixed; must exceed the Lipschitz constant of the triplet loss)
- $\lambda \approx 0.01$ (the only hyperparameter requiring tuning; stable over a wide range)
- Initialization: Gaussian random with variance $\geq 5$

## Key Experimental Results

### 1. Synthetic Data (Known Ground-Truth Dimensionality)
- Four factors are systematically varied: query ratio, intrinsic rank, number of stimuli, and noise level.
- **LORE is the only method capable of recovering the true intrinsic rank**; all other methods default to the maximum allowed dimensionality.
- $\lambda \approx 0.01$ performs stably across all conditions, requiring no fine-tuning.
- As intrinsic rank increases, LORE tracks the variation while other methods remain entirely unresponsive.

### 2. LLM-Simulated Perceptual Experiment
- SBERT embeddings of 50 food items are truncated via SVD to control intrinsic dimensionality (1–10), then used to generate noisy triplets.
- LORE accurately tracks intrinsic rank, with triplet accuracy significantly outperforming baselines.
- Dim-CV not only produces worse dimensionality estimates but also runs orders of magnitude slower (log-scale difference).

### 3. Crowdsourced Real-World Data (3 Datasets)

| Dataset | LORE Dim. | Other Methods Dim. | LORE Accuracy | Best Baseline Accuracy |
|---|---|---|---|---|
| Food-100 | **3.3** | 15 | 82.45% | 82.79% |
| Materials | **2.23** | 15 | **84.08%** | 83.94% |
| Cars | **3.0** | 15 | 52.12% | 54.06% |

- LORE achieves comparable or superior accuracy at far lower dimensionality (~3 vs. 15).
- Dim-CV severely underfits (Food: 77.67%, Cars: 50.43%), indicating that its conservative hypothesis-testing strategy fails in practice.
- LORE ranks second in runtime (behind only FORTE).

### 4. Semantic Interpretability
- The top three axes learned by LORE on Food-100 correspond to interpretable food attributes:
    - Axis 1: sweet → savory
    - Axis 2: dense → light
    - Axis 3: carbohydrate-rich → protein/vegetable
- These are discovered without any semantic supervision, making the approach highly valuable for scientific inquiry.

## Related Work & Insights

| Method | Optimizes | Recovers Dim. | Scalable | High Accuracy | Interpretable Axes |
|---|---|---|---|---|---|
| GNMDS | Gram matrix | ✗ | ✗ | ✗ | ✗ |
| CKL | Gram matrix | ✗ | ✗ | ✓ | ✓ |
| FORTE | Gram matrix | ✗ | ✓ | ✓ | ✗ |
| t-STE | Embedding | ✗ | — | ✓ | ✗ |
| SOE | Embedding | ✗ | ✓ | ✓ | ✗ |
| Dim-CV | Multi-embedding | Partial | ✗ | ✗ | — |
| **LORE** | **Embedding** | **✓** | **✓** | **✓** | **✓** |

## Limitations & Future Work
- No theoretical guarantee for exact rank recovery or global optimality (convergence to a stationary point only).
- Recovery accuracy degrades at high intrinsic ranks due to a fixed number of triplets and the curse of dimensionality.
- All methods achieve modest accuracy on the Cars dataset (~52–54%), highlighting the challenge of highly noisy data.

## Highlights & Insights
- **Answering a Core Psychophysics Question**: "How many dimensions does a perceptual space have?" is a fundamental question in psychophysics. LORE is the first data-driven, end-to-end method to answer it.
- **Elegant Application of Non-Convex Regularization**: Schatten-p ($p < 1$) introduces additional non-convexity, but the iteratively reweighted algorithm decomposes it into a sequence of convex subproblems, ensuring convergence while substantially outperforming the convex nuclear norm relaxation for low-rank recovery.
- **"Dimensionality as Scientific Discovery"**: Knowing whether a taste space is 2- or 10-dimensional directly reveals the intrinsic structure of human perception — potentially more valuable than the embeddings themselves.
- **Practical Accessibility**: Only one hyperparameter requires tuning ($\lambda \approx 0.01$), which is stable across datasets. Integration into the `cblearn` library is forthcoming, lowering the barrier to adoption.
- **Cross-Domain Potential**: The framework is not limited to psychophysics; it is applicable to any domain with only relative comparison data (no absolute measures), including recommender systems, aesthetic evaluation, and material perception.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First method to jointly learn OE dimensionality and embeddings; first application of Schatten-p regularization to OE.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers synthetic, LLM-simulated, and three real crowdsourced datasets with systematic ablation over four factors.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous mathematical derivation, and information-dense figures.
- Value: ⭐⭐⭐⭐ Significant theoretical and practical contributions to perceptual science and representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](../../ACL2026/interpretability/similarity-distance-magnitude_activations.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](../../ICML2026/interpretability/minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)

</div>

<!-- RELATED:END -->
