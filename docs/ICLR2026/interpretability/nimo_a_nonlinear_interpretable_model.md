---
title: >-
  [Paper Note] NIMO: a Nonlinear Interpretable MOdel
description: >-
  [ICLR 2026][Interpretability][interpretable model] NIMO proposes a hybrid model $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$ that preserves the global interpretability of linear regression coefficient…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "interpretable model"
  - "marginal effects"
  - "linear regression"
  - "neural networks"
  - "feature effects"
date: 2026-05-08
content_hash: be9a60ddbbae18cf
---

# NIMO: a Nonlinear Interpretable MOdel

**Conference**: ICLR 2026
**arXiv**: [2506.05059](https://arxiv.org/abs/2506.05059)
**Code**: N/A
**Area**: Interpretable Machine Learning
**Keywords**: interpretable model, marginal effects, linear regression, neural networks, feature effects

## TL;DR

NIMO proposes a hybrid model $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$ that preserves the global interpretability of linear regression coefficients (via mean marginal effects, MEM) while leveraging neural networks to provide instance-wise nonlinear corrections. Linear coefficients and network parameters are jointly optimized efficiently through parameter elimination.

## Background & Motivation

**Accuracy vs. Interpretability Dilemma**: Linear regression provides clear feature effect explanations through coefficients but has limited predictive power; neural networks are powerful predictors but lack intrinsic interpretability, being regarded as "black boxes."

**Unreliability of Post-hoc Explanations**: Post-hoc explanation methods such as SHAP and LIME depend on hyperparameter choices and do not guarantee fidelity.

**Limitations of Existing Hybrid Methods**: NAM cannot capture feature interactions; LassoNet has limited global interpretability; IMN predicts different coefficients for each instance, sacrificing global interpretability.

**Importance of Feature Effects**: In high-stakes domains such as healthcare, it is necessary to answer both local questions ("how does an increase in age affect risk for this patient?") and global questions ("how does age affect risk overall?").

**Optimization Challenges**: Jointly optimizing the linear coefficients $\boldsymbol{\beta}$ and neural network parameters $\mathbf{u}$ is non-trivial when the two are tightly coupled.

## Method

### Overall Architecture

NIMO extends linear regression by multiplying each feature's coefficient by a data-dependent nonlinear correction factor:

$$f(\mathbf{x}) = \beta_0 + \sum_{j=1}^d x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$$

A key constraint is imposed: $g_{\mathbf{u}_j}(\mathbf{0}) = 0$ (data is standardized to zero mean), ensuring $\text{MEM}_j = \beta_j$.

### Key Designs

1. **Excluding the Target Feature ($\mathbf{x}_{-j}$)**

    - **Function**: The input to neural network $g_{\mathbf{u}_j}$ excludes the $j$-th feature.
    - **Mechanism**: Feature $x_j$ contributes to predictions only through the linear term $\beta_j$, guaranteeing the interpretability of $\beta_j$.
    - **Design Motivation**: If $g_j$ also depended on $x_j$, the marginal effect could not be expressed concisely in terms of $\beta_j$.

2. **Zero-point Constraint $g_{\mathbf{u}_j}(\mathbf{0}) = 0$**

    - **Function**: Enforced during the forward pass by subtracting $g_{\mathbf{u}}(\mathbf{0})$.
    - **Mechanism**: Since data is standardized to zero mean, the constraint ensures the model reduces to a purely linear form at the mean.
    - **Design Motivation**: $\text{MEM}_j = \frac{\partial f}{\partial x_j}\big|_{\mathbf{x}=\mathbf{0}} = \beta_j$.

3. **Parameter Elimination**

    - **Function**: Derives a closed-form solution $\hat{\boldsymbol{\beta}}(\mathbf{u}) = (B_\mathbf{u}^T B_\mathbf{u} + \lambda I)^{-1} B_\mathbf{u}^T \mathbf{y}$, which is substituted back so that only $\mathbf{u}$ needs to be optimized.
    - **Mechanism**: A profile likelihood approach that eliminates $\boldsymbol{\beta}$ from the optimization.
    - **Design Motivation**: Avoids the difficulty of jointly optimizing $\boldsymbol{\beta}$ and $\mathbf{u}$.

4. **Adaptive Ridge Regression for Sparsity**

    - **Function**: Replaces Lasso with adaptive ridge regression (Grandvalet, 1998).
    - **Mechanism**: Each step admits a closed-form solution, and the method is equivalent to Lasso at the optimum.
    - **Design Motivation**: Lasso lacks a closed-form solution and cannot be used with parameter elimination; adaptive ridge achieves sparsity while retaining a closed form.

5. **Shared Network with Positional Encoding**

    - **Function**: A single shared $g_\mathbf{u}$ augmented with feature-index positional encodings replaces $d$ independent networks.
    - **Design Motivation**: Maintaining $d$ independent networks is infeasible in high-dimensional settings.

6. **Group $\ell_2$ Regularization**

    - **Function**: Applies group $\ell_2$ regularization to each column of the first-layer weight matrix.
    - **Design Motivation**: Encourages feature-level sparsity, providing an additional layer of interpretability.

### Loss & Training

- Regression: $\|\mathbf{y} - B_\mathbf{u}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1$
- Classification: Approximated as weighted least squares via IRLS.
- Supports sub-$\ell_1$ pseudo-norms to mitigate Lasso's over-shrinkage.
- Outer loop: gradient descent over $\mathbf{u}$; inner loop: closed-form update of $\boldsymbol{\beta}$.

## Key Experimental Results

### Main Results

MSE on synthetic regression datasets:

| Method | Setting 1 (5-dim) | Setting 2 (10-dim) | Setting 3 (50-dim) |
|------|---------|----------|----------|
| Lasso | 3.164 | 3.340 | 13.122 |
| NN | 1.109 | 1.482 | 13.718 |
| NAM | 3.427 | 5.126 | 16.543 |
| IMN | 0.137 | 1.188 | 6.308 |
| LassoNet | 0.078 | 2.612 | 1.738 |
| **NIMO** | **0.024** | **0.197** | **0.380** |

NIMO achieves substantial improvements across all settings, with a margin exceeding 4× in the 50-dimensional scenario.

### Ablation Study

| Component | Effect |
|------|------|
| Remove $g_j$ (pure linear) | Accurate coefficients but poor fit |
| Allow $g_j$ to depend on $x_j$ | Coefficients become uninterpretable |
| Remove zero-point constraint | MEM no longer equals $\beta_j$ |
| Remove group $\ell_2$ | Unable to identify non-interacting features |
| Remove sparsity | Zero coefficients not correctly recovered |

Toy example validation (3-dimensional):

| Metric | NIMO | Lasso |
|------|------|-------|
| Recovery of $\beta_1=3, \beta_2=-3$ | Exact | Exact |
| Identification of $\beta_3=0$ | Correctly zero | Non-zero |
| Nonlinear interaction recovery | Matches ground truth | N/A |

### Key Findings

- Remains robust under low data regimes (200 samples), owing to parameter elimination and regularization.
- The network component does not interfere with the recovery of linear coefficients in purely linear settings.
- MEM-based feature rankings are highly consistent with SHAP rankings, but NIMO's explanations are intrinsic rather than post-hoc approximations.
- Predictive performance on the diabetes, Boston, and superconductivity datasets is comparable to or better than the best competing methods.

## Highlights & Insights

1. **Elegant Design**: Three carefully chosen constraints (excluding the target feature, the zero-point constraint, and standardization) guarantee MEM = $\beta$.
2. **Ingenuity of Parameter Elimination**: The profile likelihood idea is applied effectively to mixed-model optimization.
3. **Multi-level Interpretability**: Global interpretability via $\beta_j$, instance-level interpretability via $h_j(\mathbf{x})$, and interaction-level interpretability via sparsity patterns in the first-layer weights.
4. **Natural Extension of GLMs**: Directly applicable to logistic regression and other GLMs via IRLS.
5. **Adaptive Ridge as Lasso Equivalent**: A classical result is exploited to achieve sparsity while preserving closed-form solutions.

## Limitations & Future Work

- Scalability to very high dimensions ($d > 1000$) has not been verified.
- The model assumes nonlinear corrections arise from interactions among other features, neglecting intrinsic nonlinear effects of individual features.
- Experiments are conducted on small-scale datasets (UCI); performance on large-scale data remains unknown.
- Comparison with a broader range of interpretable methods, such as EBM and GAMI-Net, is lacking.
- Currently limited to tabular data.

## Related Work & Insights

- **NAM (Agarwal et al., 2021)**: Additive model without interactions → NIMO supports interactions via $g_j(\mathbf{x}_{-j})$.
- **LassoNet (Lemhadri et al., 2021)**: Achieves sparsity and nonlinearity but with limited global interpretability → NIMO achieves both.
- **IMN (Kadra et al., 2024)**: Instance-wise coefficients sacrifice global interpretability → NIMO unifies global and local interpretability.
- **Grandvalet (1998)**: Theoretical foundation for adaptive ridge as a Lasso equivalent → incorporated into NIMO's optimization.
- **Inspiration**: The framework could be extended to time series (time-varying coefficients) and causal inference (heterogeneous treatment effect correction).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The model design is elegant; the theoretical guarantee MEM=$\beta$ is the core contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic and real-data experiments provide adequate validation, though dataset scales are small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear, toy examples are intuitive, and theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for "accurate and interpretable" modeling with strong application potential in high-stakes domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)

</div>

<!-- RELATED:END -->
