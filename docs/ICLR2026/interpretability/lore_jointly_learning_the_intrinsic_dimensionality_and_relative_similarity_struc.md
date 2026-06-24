---
title: >-
  [Paper Note] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data
description: >-
  [ICLR 2026][Interpretability][Ordinal Embedding] The authors propose LORE, the first framework to jointly learn embedding representations and intrinsic dimensionality from ordinal triplet comparisons. By replacing traditional pre-specified dimensionality strategies with a non-convex Schatten-p quasi-norm ($p<1$) regularization, solved via an Iterative Reweighted Nuclear Norm (IRNN) algorithm with guaranteed convergence to a stationary point, LORE significantly outperforms all…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Ordinal Embedding"
  - "Intrinsic Dimensionality Recovery"
  - "Schatten-p Quasi-norm"
  - "Triplet Comparison"
  - "Perceptual Space"
  - "Low-rank Regularization"
date: 2026-05-08
content_hash: a152c8c41ca0d681
---

# LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data

**Conference**: ICLR 2026  
**arXiv**: [2602.04192](https://arxiv.org/abs/2602.04192)  
**Code**: [GitHub](https://github.com/vivek2000anand/lore_iclr)  
**Area**: Representation Learning / Perceptual Modeling  
**Keywords**: Ordinal Embedding, Intrinsic Dimensionality Recovery, Schatten-p Quasi-norm, Triplet Comparison, Perceptual Space, Low-rank Regularization

## TL;DR
The authors propose LORE, the first framework to jointly learn embedding representations and intrinsic dimensionality from ordinal triplet comparisons. By replacing traditional pre-specified dimensionality strategies with a non-convex Schatten-p quasi-norm ($p<1$) regularization, solved via an Iterative Reweighted Nuclear Norm (IRNN) algorithm with guaranteed convergence to a stationary point, LORE significantly outperforms all baseline methods in dimensionality recovery across synthetic data, LLM-simulated perception experiments, and three crowdsourced datasets, while maintaining high triplet accuracy and semantic interpretability.

## Background & Motivation

**Background**: Ordinal Embedding (OE) learns multidimensional representations of perceptual spaces from triplet comparisons ("Is $A$ more similar to $B$ or $C$?"). It is widely used in psychophysics for subjective perceptions such as taste, smell, and aesthetic preferences. Compared to absolute quantitative ratings (e.g., Likert scales), triplet comparisons do not rely on linguistic descriptions and are unaffected by individual scale biases.

**Limitations of Prior Work**:
   - All existing OE methods (SOE, FORTE, t-STE, CKL, OENN) require users to pre-specify the embedding dimensionality $d'$.
   - There is a lack of criteria for determining the "true dimensionality," leading to the common practice of setting excessively high dimensions.
   - Excessive dimensionality masks the true structure (e.g., a 10D embedding that only needs 2D might fragment "sweetness" across multiple axes).
   - Scientific discovery seeks parsimony (Occam's razor): low-dimensional representations are more interpretable and computationally efficient.
   - The only prior attempt to recover dimensionality, the Künstle method, requires enumerating candidate dimensions and training each individually, which is not scalable.

**Key Insight**: Integrate dimensionality discovery into the OE optimization itself by using Schatten-p quasi-norm regularization to automatically balance triplet accuracy and the rank of the embedding, thereby eliminating the need for a preset dimensionality.

## Method

### Overall Architecture

LORE incorporates the determination of the appropriate dimensionality—originally a user-defined hyperparameter—directly into the ordinal embedding optimization objective. Given $N$ objects and a set of triplets $T=\{(a,i,j)\}$ (indicating $a$ is more similar to $i$ than to $j$), it optimizes an embedding matrix $Z\in\mathbb{R}^{N\times d'}$ in a sufficiently wide $d'$-dimensional space. The objective includes a smooth triplet fitting loss and a Schatten-$p$ low-rank regularization term. This allows the model to fit comparison relationships while automatically compressing the singular values of redundant dimensions to zero. The resulting number of non-zero singular values constitutes the recovered intrinsic dimensionality $d\ll N$. The optimization is solved using an iterative reweighting algorithm, where each step involves a Singular Value Decomposition (SVD).

### Key Designs

**1. Schatten-$p$ Quasi-norm Regularization: Approximating Rank with Non-convex Penalty**

To enable "automatic dimensionality selection," the goal is to minimize the rank of the embedding matrix $Z$, but the rank function is NP-hard and cannot be optimized directly. Traditional approaches use the nuclear norm ($p=1$, the sum of all singular values $\sum_i\sigma_i(Z)$) as a convex relaxation. However, the nuclear norm penalizes large and small singular values equally, which can weaken the large singular values that carry the actual structure, introducing significant bias. LORE utilizes the Schatten-$p$ quasi-norm $\sum_{i}\sigma_i(Z)^p$ with $0<p<1$ (defaulting to $p=0.5$). This imposes a light penalty on large singular values and a heavy penalty on those near zero, effectively "killing" redundant dimensions while preserving the main structure. This is a much more accurate low-rank approximation than the nuclear norm, at the cost of non-convexity, which is handled by the iterative algorithm.

**2. Softplus Smooth Triplet Loss: Eliminating Zero-Gradient Plateaus**

Standard hinge-form triplet losses have zero gradients for constraints that are already satisfied, which can cause optimization to stall. LORE replaces this with a everywhere-differentiable softplus form:

$$\sum_{(a,i,j)\in T}\log\big(1+\exp(1+d(z_a,z_i)-d(z_a,z_j))\big),$$

This ensures that there are no zero-gradient plateaus and that gradient signals exist across all triplets. The only non-differentiable point is embedding collapse (all points overlapping), which is avoided using a wide initialization with sufficient variance.

**3. Direct Embedding Optimization: Scalability**

Methods like GNMDS, CKL, and FORTE optimize an $N\times N$ Gram matrix $G=ZZ^\top$, with complexity growing quadratically with the number of objects $N$. LORE directly optimizes the $N\times d'$ embedding $Z$, reducing the per-step complexity to be proportional to $Nd'$. This allows scalability to larger datasets and provides direct access to embedding coordinates for interpreting semantic axes.

**4. Iterative Reweighted Nuclear Norm (IRNN) and Convergence: Decomposing Non-convex Objectives**

Since the non-convex Schatten-$p$ term cannot be handled by simple gradient descent, LORE employs the Iterative Reweighted Nuclear Norm (IRNN) algorithm. Each step takes a gradient step along the triplet loss followed by an SVD: $U,S,V^\top=\mathrm{SVD}\big(Z^k-\tfrac{1}{\mu}\nabla f(Z^k)\big)$. Singular values are then subjected to thresholded shrinkage: $S^k=\max\{S-\tfrac{p}{\mu}\sigma^{p-1},\,0\}$, and $Z^{k+1}$ is reconstructed as $U S^k V^\top$. This iterates until change thresholds are met, with a complexity of $O\!\big(d'(|T|+Nd')\big)$ per step. The paper proves this sequence converges to a stationary point, i.e., $\sum_{k=1}^{\infty}\|Z^{k+1}-Z^k\|_F<+\infty$. While this is a stationary point rather than a global optimum, OE theory suggest these points are typically close to the optimal solution. The number of non-zero singular values after convergence is the recovered intrinsic dimension $d$.

### Loss & Training

The complete objective combines the triplet fitting term and the regularization term: 
$$\min_Z \Psi(Z)=\text{softplus triplet loss}+\lambda\sum_{i=1}^{\min\{N,d'\}}\sigma_i(Z)^p.$$
Three constants generally remain fixed: $p=0.5$, $\mu=0.1$ (must be larger than the Lipschitz constant of the triplet loss), and Gaussian random initialization with variance $\geq 5$. The only parameter requiring tuning is the regularization weight $\lambda$, set to $\lambda\approx 0.01$, which is stable across a wide range.

## Key Experimental Results

### 1. Synthetic Data (Known Intrinsic Dimensionality)
- Four factors were systematically varied: query ratio, intrinsic rank, number of perceptions, and noise level.
- **LORE was the only method capable of recovering the true intrinsic rank**, while all other methods defaulted to the maximum allowed dimension.
- $\lambda\approx 0.01$ performed yield stable results across all conditions.
- LORE successfully tracked increases in intrinsic rank, whereas other methods remained unchanged.

### 2. LLM Simulated Perception
- Used SBERT to embed 50 food items $\rightarrow$ truncated SVD to control intrinsic dimensionality (1-10) $\rightarrow$ generated noisy triplets.
- LORE accurately tracked intrinsic rank and significantly outperformed baselines in triplet accuracy.
- Dim-CV provided poorer dimensionality estimates and had **orders of magnitude** higher runtimes (log-scale difference).

### 3. Crowdsourced Real Data (3 Datasets)

| Dataset | LORE Dim | Other Method Dim | LORE Accuracy | Best Baseline Acc |
|---------|----------|------------------|---------------|-------------------|
| Food-100 | **3.3** | 15 | 82.45% | 82.79% |
| Materials | **2.23** | 15 | **84.08%** | 83.94% |
| Cars | **3.0** | 15 | 52.12% | 54.06% |

- LORE achieved comparable or higher accuracy using much lower dimensions (~3 vs 15).
- Dim-CV suffered from severe underfitting (Food: 77.67%, Cars: 50.43%) due to failed conservative hypothesis testing.
- LORE was the second fastest method (following FORTE).

### 4. Semantic Interpretability
- The first three axes learned by LORE on the Food-100 dataset correspond to interpretable food attributes:
    - Axis 1: Sweet $\rightarrow$ Savory
    - Axis 2: Dense $\rightarrow$ Airy/Light
    - Axis 3: Carbohydrate-rich $\rightarrow$ Protein/Vegetable
- These were discovered automatically without semantic supervision, illustrating high value for scientific discovery.

## Related Work & Insights

| Method | Optimization Target | Dimension Recovery | Scalable | High Accuracy | Interpretable Axes |
|--------|---------------------|--------------------|----------|---------------|--------------------|
| GNMDS | Gram Matrix | ✗ | ✗ | ✗ | ✗ |
| CKL | Gram Matrix | ✗ | ✗ | ✓ | ✓ |
| FORTE | Gram Matrix | ✗ | ✓ | ✓ | ✗ |
| t-STE | Embedding | ✗ | — | ✓ | ✗ |
| SOE | Embedding | ✗ | ✓ | ✓ | ✗ |
| Dim-CV | Multi-Embedding | Partial | ✗ | ✗ | — |
| **LORE** | **Embedding** | **✓** | **✓** | **✓** | **✓** |

## Limitations
- Lack of theoretical guarantees for exact rank recovery or global optimality (only convergence to stationary points).
- Recovery accuracy decreases at high intrinsic ranks due to fixed triplet counts and the curse of dimensionality.
- All methods showed low accuracy on the Cars dataset (~52-54%), highlighting challenges with extremely noisy data.

## Highlights & Insights
- **Addressing the Core Problem of Psychophysics**: "How many dimensions does a perceptual space have?" is a fundamental question. LORE is the first data-driven, end-to-end method to answer it.
- **Sophisticated Use of Non-convex Regularization**: While Schatten-p ($p<1$) introduces non-convexity, the IRNN algorithm decomposes it into convex subproblems, ensuring convergence and outperforming convex nuclear norm relaxation in low-rank recovery.
- **"Dimension as Scientific Discovery"**: Determining if a taste space is 2D or 10D reveals the inherent structure of human perception, which can be more valuable than the embedding itself.
- **Practicality**: Only one hyperparameter ($\lambda\approx 0.01$) needs tuning, and it is stable across datasets. Integration into the `cblearn` library will lower the barrier to entry.
- **Cross-domain Potential**: Applicable to any scenario with relative comparisons (no absolute measures), such as recommendation systems, aesthetic evaluation, and material perception.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First method to jointly learn OE dimension and embedding; first application of Schatten-p in OE.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + LLM simulation + 3 real crowdsourced datasets; systematic ablation of 4 factors.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous mathematical derivation, and informative visualizations.
- Value: ⭐⭐⭐⭐ Significant theoretical and practical contributions to perceptual science and representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Weight Parameters for Training Data Attribution](learning_to_weight_parameters_for_training_data_attribution.md)
- [\[ICLR 2026\] Estimating Dimensionality of Neural Representations from Finite Samples](estimating_dimensionality_of_neural_representations_from_finite_samples.md)
- [\[ICML 2026\] Dimensionality Controls When Modularity Helps in Continual Learning](../../ICML2026/interpretability/dimensionality_controls_when_modularity_helps_in_continual_learning.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](conjuring_semantic_similarity.md)
- [\[ICLR 2026\] Sequences of Logits Reveal the Low Rank Structure of Language Models](sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)

</div>

<!-- RELATED:END -->
