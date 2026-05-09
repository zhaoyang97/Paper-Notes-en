---
title: >-
  [Paper Note] Geometric Data Valuation via Leverage Scores
description: >-
  [NeurIPS 2025][Model Compression][Data Valuation] This paper proposes a geometric data valuation method based on **statistical leverage scores** as an efficient proxy for Data Shapley values. The proposed method satisfies the axioms of symmetry, efficiency, and dummy player, and extends to ridge leverage scores to address the dimensionality saturation problem, providing theoretical guarantees of $O(\varepsilon)$-approximate optimality.
tags:
  - NeurIPS 2025
  - Model Compression
  - Data Valuation
  - Shapley Values
  - Leverage Scores
  - Active Learning
  - Experimental Design
date: 2026-05-08
content_hash: aa599e93a0db825d
---

# Geometric Data Valuation via Leverage Scores

**Conference**: NeurIPS 2025
**arXiv**: [2511.02100](https://arxiv.org/abs/2511.02100)
**Code**: [GitHub](https://github.com/rodrgo/geosh)
**Area**: Model Compression / LLM Efficiency
**Keywords**: Data Valuation, Shapley Values, Leverage Scores, Active Learning, Experimental Design

## TL;DR

This paper proposes a geometric data valuation method based on **statistical leverage scores** as an efficient proxy for Data Shapley values. The proposed method satisfies the axioms of symmetry, efficiency, and dummy player, and extends to ridge leverage scores to address the dimensionality saturation problem, providing theoretical guarantees of $O(\varepsilon)$-approximate optimality.

## Background & Motivation

Data valuation aims to quantify the contribution of each training data point to model performance, with important applications in data cleaning, subset selection, federated learning incentive allocation, and data market pricing. Data Shapley is the most theoretically rigorous approach, but its combinatorial computational complexity—requiring enumeration of marginal contributions over all subsets—renders it infeasible at scale.

Limitations of prior work:
- Computationally expensive, requiring repeated model retraining
- Requires access to model weights and gradients
- Primarily focused on data quality control in pretraining, neglecting uncertainty in data acquisition decisions

This paper approaches the problem from the perspective of **geometry and numerical linear algebra**, proposing a model-agnostic data valuation method.

## Method

### Overall Architecture

Core Idea: **Leverage scores measure the structural importance of each data point in feature space**—high-leverage points span unique directions and are thus valuable, while low-leverage points tend to be redundant.

Given a data matrix $X \in \mathbb{R}^{n \times d}$, the leverage score of the $i$-th data point is defined as the $i$-th diagonal entry of the projection matrix $H = X(X^TX)^{-1}X^T$:

$$\ell_i = x_i^T (X^TX)^{-1} x_i$$

Normalizing yields the valuation function:

$$\pi_i = \frac{\ell_i}{\sum_{j=1}^n \ell_j}$$

### Key Designs

**Theorem 1 (Shapley Axiom Satisfaction)**: When $\text{rank}(X) = d$, the normalized leverage scores $\pi_i$ satisfy the following axioms under the utility function $U(S) = \text{span}\{x_i: i \in S\}$:
- **Symmetry**: Data points with equivalent contributions receive equal valuations
- **Efficiency**: All valuations sum to 1 (since $\text{Tr}(H) = d$, normalization yields unit sum)
- **Dummy Player**: Data points that do not increase the subspace dimension receive zero valuation

**Dimensionality Saturation**: Standard leverage scores assign zero marginal value to all new data points once the subspace reaches full rank $d$, which is a significant practical limitation.

**Ridge Leverage Extension**—regularization is introduced to resolve saturation:

$$\ell_i^{(\lambda)} = x_i^T (X^TX + \lambda I)^{-1} x_i$$

Key properties of ridge leverage scores:
- For any $\lambda > 0$, $\ell_i^{(\lambda)} \in (0,1)$, and the statistical dimension $k_\lambda = \sum_i \ell_i^{(\lambda)}$ lies strictly in $(0, d)$
- Even when $\text{rank}(X_S) = d$, newly added data points retain positive values, as regularization shrinks the inverse matrix without eliminating marginal gains

**Connection to Classical Experimental Design**:
- D-optimality marginal gain: $\log\det(A + xx^T) - \log\det(A) = \log(1 + \ell^{(\lambda)}(x)) > 0$
- A-optimality marginal gain: $\text{Tr}((A+xx^T)^{-1}) - \text{Tr}(A^{-1}) = -\|A^{-1}x\|_2^2 / (1+\ell^{(\lambda)}(x)) < 0$

The marginal gains of both classical criteria are monotone functions of the ridge leverage score.

**Theorem 3 ($\varepsilon$-Approximation Guarantee)**: Sampling $m \geq C \frac{k_\lambda + \log(2d/\delta)}{\varepsilon^2}$ points according to ridge leverage probabilities yields, with probability $\geq 1-\delta$:

$$\|\hat{\theta} - \theta^\star\|_A \leq 4\varepsilon \|\theta_{\text{lin}}\|_A, \quad R(\hat{\theta}) - R(\theta^\star) \leq 8\varepsilon^2 \|\theta_{\text{lin}}\|_A^2$$

That is, the model parameters and prediction risk obtained from training on the sampled subset are $O(\varepsilon)$-close to those of the full-data model.

### Loss & Training

The proposed method does not involve conventional training procedures—valuations are obtained via closed-form computation. In active learning experiments, ridge leverage scores are used to select the most informative samples for annotation, requiring neither gradients nor backpropagation.

## Key Experimental Results

### Main Results (Active Learning)

**Active Learning on MNIST (3-layer MLP, 784→256→64→10)**:
- Initial pool of 100 labeled samples; 40 active learning rounds with 5 samples selected per round

| Method | Final Test Accuracy |
|--------|-------------------|
| Ridge Leverage (adaptive $\lambda$) | **0.846 ± 0.006** |
| K-center | ~0.82 |
| Margin | ~0.83 |
| Entropy | ~0.83 |
| EGL | ~0.81 |
| Random | ~0.80 |

Ridge leverage demonstrates a clear advantage after the pretraining phase and exhibits the lowest cross-trial variance.

**Theoretical vs. Empirical Guarantees**:

| Guarantee Type | Specific Result |
|---------------|----------------|
| Matrix approximation | $(1-\varepsilon)A \preceq A_S \preceq (1+\varepsilon)A$ |
| Parameter approximation | $\|\hat{\theta} - \theta^\star\|_A \leq 4\varepsilon\|\theta_{\text{lin}}\|_A$ |
| Risk approximation | $R(\hat{\theta}) - R(\theta^\star) \leq 8\varepsilon^2\|\theta_{\text{lin}}\|_A^2$ |

### Ablation Study

As a workshop paper, no systematic ablation is included. However, the theoretical analysis examines the effect of different choices of $\lambda$:
- $\lambda = 0$: degenerates to standard leverage scores, subject to dimensionality saturation
- $\lambda > 0$: resolves saturation; the adaptive setting $\lambda = 0.01 \times \text{Tr}(X^TX)/64$ performs best

### Key Findings

1. **Gradient-free data selection**: Ridge leverage scores require neither model gradients nor backpropagation, relying solely on the geometric structure of the data
2. **Axiomatic guarantees**: Satisfies Shapley symmetry and efficiency (the ridge extension does not satisfy the dummy player axiom, which is justified—redundant data still contributes to variance reduction)
3. **Deep connection to experimental design**: The marginal gains of A/D-optimality criteria are both governed by leverage scores

## Highlights & Insights

- **Novel perspective**: Classical concepts from numerical linear algebra (leverage scores) are reinterpreted as data valuation tools, bridging NLA and operations research
- **Computational efficiency**: All leverage scores can be computed in $O(nd^2)$ time, far below the exponential complexity of Shapley values
- **Theoretical elegance**: The $\varepsilon$-approximation theorem is based on matrix Chernoff concentration inequalities, with a compact proof
- **Model-agnostic**: Valuation depends only on the data matrix, requiring no model specification or training

## Limitations & Future Work

1. **Workshop paper**: Experiments are limited in scale (MNIST only); validation on large-scale datasets or LLM settings is absent
2. **Linearity assumption**: Theoretical guarantees are grounded in ridge regression (linear models); extension to nonlinear settings requires further investigation
3. **Linearity axiom not satisfied**: Normalized leverage scores do not satisfy the linearity axiom of Shapley values, limiting multi-task aggregation
4. **Embedding space dependency**: Active learning experiments require penultimate-layer embeddings from a pretrained model; the method's effectiveness depends on embedding quality
5. Comparisons with more advanced active learning methods (BALD, BatchBALD, LESS, TRAK) are not included

## Related Work & Insights

- **Data Shapley**: The theoretical counterpart of this work; leverage scores provide an efficient though slightly weaker alternative
- **A/D-Optimal Experimental Design**: Ridge leverage naturally connects to classical criteria in optimal experimental design
- **Randomized Numerical Linear Algebra**: Leverage score sampling is widely used in randomized least-squares solvers
- **Insights**: Ridge leverage can be extended to feature spaces (e.g., kernel leverage scores) to handle nonlinear problems, or applied to training data selection for LLMs

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Distinctive geometric perspective with elegant connections
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous theoretical proofs with complete axiomatic verification
- **Experimental Thoroughness**: ⭐⭐ — Limited experiments as a workshop paper; MNIST only
- **Practical Value**: ⭐⭐⭐ — Computationally efficient, but applicable scenarios require further exploration
- **Overall**: ⭐⭐⭐⭐

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Insights & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[NeurIPS 2025\] GraSS: Scalable Data Attribution with Gradient Sparsification and Sparse Projection](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)
- [\[NeurIPS 2025\] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling](skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[CVPR 2026\] GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design](../../CVPR2026/model_compression/geofusion-cad_structure-aware_diffusion_with_geometric_state_space_for_parametri.md)

</div>

<!-- RELATED:END -->
