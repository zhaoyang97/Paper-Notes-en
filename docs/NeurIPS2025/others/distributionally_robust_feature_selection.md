---
title: >-
  [Paper Note] Distributionally Robust Feature Selection
description: >-
  [NeurIPS 2025][Feature Selection] This paper proposes a model-agnostic distributionally robust feature selection method that achieves a continuous relaxation of discrete selection by injecting controlled Gaussian noise i…
tags:
  - "NeurIPS 2025"
  - "Feature Selection"
  - "Distributionally Robust Optimization"
  - "Group DRO"
  - "Noise Injection"
  - "Model-Agnostic"
date: 2026-05-08
content_hash: f6fec16b07693c88
---

# Distributionally Robust Feature Selection

**Conference**: NeurIPS 2025
**arXiv**: [2510.21113](https://arxiv.org/abs/2510.21113)  
**Code**: Available (link provided in the paper)  
**Area**: Other
**Keywords**: Feature Selection, Distributionally Robust Optimization, Group DRO, Noise Injection, Model-Agnostic

## TL;DR
This paper proposes a model-agnostic distributionally robust feature selection method that achieves a continuous relaxation of discrete selection by injecting controlled Gaussian noise into covariates, and optimizes the conditional variance of the Bayes-optimal predictor, so that the selected feature subset enables high-quality downstream models to be trained simultaneously across multiple subpopulations.

## Background & Motivation

**Background**: Feature selection is a fundamental problem in machine learning. Classical methods such as Lasso, forward/backward selection, and XGBoost feature importance all optimize for a single distribution. In the area of distributionally robust optimization (DRO), methods such as Group DRO aim to find a single model that performs well across all subgroups, but do not address feature selection.

**Limitations of Prior Work**: In practice, it is often necessary to select a small number of features under budget constraints (e.g., a medical questionnaire can only ask a limited number of questions), and the selected features must support training good models separately for different subpopulations. Existing feature selection methods do not consider cross-distribution robustness, and DRO methods do not perform feature selection — the intersection of these two directions has been entirely unexplored.

**Key Challenge**: Feature selection is a discrete combinatorial optimization problem (NP-hard), while DRO requires minimax optimization. The coupling of these two levels of optimization makes direct solutions extremely difficult — each time the feature subset changes, models for all subpopulations must be retrained.

**Goal**: Select $k$ features such that models trained separately on each subpopulation $P_i$ using only those $k$ features achieve the best possible worst-group performance.

**Key Insight**: (a) Replace hard selection with noise injection to achieve a continuous relaxation; (b) shift to optimizing the performance of the Bayes-optimal predictor to eliminate dependence on any specific model; (c) leverage the Gaussian noise model to derive closed-form kernel weight expressions.

**Core Idea**: By combining feature-level noise injection with variance optimization of the Bayes-optimal predictor, the intractable discrete-minimax problem is transformed into a differentiable continuous optimization, enabling model-agnostic distributionally robust feature selection.

## Method

### Overall Architecture
The input consists of labeled data from multiple subpopulations $\{(X_i^j, Y_i^j)\}$, and the output is the index set of $k$ optimal features. The method proceeds in three steps: (1) fit the conditional expectation $\hat{\mu}_i(X)$ once for each subpopulation; (2) control the information content of each feature via the noise injection parameter $\alpha$, and optimize the minimax objective through gradient descent; (3) select the $k$ features with the smallest $\alpha$ values.

### Key Designs

1. **Continuous Relaxation via Noise Injection**:

    - **Function**: Relaxes the discrete binary mask $\alpha \in \{0,1\}^m$ to a continuous noise parameter $\alpha \in \mathbb{R}_{\geq 0}^m$.
    - **Mechanism**: Rather than directly scaling features ($\alpha \odot X$, which a model can compensate for inversely), Gaussian noise is injected as $S_i(\alpha)|X \sim \mathcal{N}(X_i, \alpha_i)$. Here $\alpha_i = 0$ denotes full information retention, and $\alpha_i \to \infty$ denotes discarding the feature.
    - **Design Motivation**: Deterministic scaling does not alter information content (a model can learn to compensate via $w_i/\alpha_i$), whereas stochastic noise genuinely reduces the signal-to-noise ratio.
    - **Distinction from Lasso**: Lasso's $\ell_1$ regularization is only effective for linear models and assumes a single distribution.

2. **Variance Optimization via the Bayes-Optimal Predictor**:

    - **Function**: Replaces the loss of a specific model with the Bayes-optimal loss, bypassing the inner optimization.
    - **Mechanism (Theorem 1)**: Under MSE loss, the problem is equivalent to $\min_\alpha \max_{P_i} -\mathbb{E}_{S(\alpha)}[\mathbb{E}_X[\mu_i(X)|S(\alpha)]^2] + \lambda \text{Reg}(\alpha)$, where $\mu_i(X) = \mathbb{E}_{P_i}[Y|X]$.
    - **Design Motivation**: (a) Eliminates the need to retrain a model for each $\alpha$; (b) is agnostic to the downstream model architecture; (c) requires fitting $\mu_i$ only once per subpopulation at the outset.

3. **Closed-Form Kernel Representation (Theorem 2)**:

    - **Function**: Transforms the empirical estimate of the conditional expectation into a Gaussian kernel-weighted sum.
    - **Closed-Form Weights**: $w_i^j(S,\alpha) = \frac{\exp(-\frac{1}{2}(X_i^j - S)^T \text{diag}(\alpha)^{-1}(X_i^j - S))}{\sum_k \exp(-\frac{1}{2}(X_i^k - S)^T \text{diag}(\alpha)^{-1}(X_i^k - S))}$
    - **Design Motivation**: $\alpha$ directly controls the kernel bandwidth — a small $\alpha_i$ yields a narrow kernel along that dimension (information preserved), while a large $\alpha_i$ yields a wide kernel (information discarded).

### Loss & Training
The final optimization objective is $\min_\alpha \max_{P_i} -\frac{1}{b}\sum_{\ell=1}^b (\sum_j w_i^j(S^\ell, \alpha) \mu_i(X_i^j))^2$, with the reparameterization trick $S = X + \sqrt{\alpha} \odot \epsilon$ to ensure gradient flow. The inner max is approximated via softmax with temperature $\beta$. The regularizer $\lambda \cdot \text{Reg}(\alpha) = \lambda / \|\alpha\|_1$ encourages sparsity. In practice, summation is restricted to $k$ nearest neighbors for efficiency.

## Key Experimental Results

### Main Results: Synthetic Dataset 1 (Linear Model, 3 Subgroups, 15 Features)

| Method | Budget=5 Group A MSE | Group B MSE | Group C MSE | Budget=10 Worst-Group |
|--------|---------------------|-------------|-------------|----------------------|
| **Ours** | **Lowest** | **Balanced** | **Balanced** | **Tied for best** |
| DRO-Lasso | Moderate | Moderate | Good | Second best |
| DRO-XGBoost | Good | Moderate | Good | Tied with Ours |
| Vanilla Lasso | Poor | Poor | Best | Poor |
| Embedded MLP | Poor | Poor | Poor | Poor |

### Main Results: Real-World Dataset ACS (Income Prediction, 3 States)

| Method | CA MSE↓ | FL MSE↓ | NY MSE↓ | CA R²↑ | FL R²↑ | NY R²↑ |
|--------|---------|---------|---------|--------|--------|--------|
| **Ours** | **Lowest (order-of-magnitude lead)** | **Lowest** | **Lowest** | **Highest** | **Highest** | **Highest** |
| DRO-XGBoost | High | High | High | Low | Low | Low |
| DRO-Lasso | Moderate | Moderate | Moderate | Moderate | Moderate | Moderate |

### Ablation Study: Method Property Comparison

| Property | Ours | Lasso | DRO-Lasso | XGBoost | DRO-XGB | Embedded MLP |
|----------|------|-------|-----------|---------|---------|-------------|
| Model-Agnostic | ✓ | ✗ (linear) | ✗ (linear) | ✗ (tree) | ✗ (tree) | ✗ (MLP) |
| Distributionally Robust | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ |
| Handles Nonlinearity | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Low Variance | ✓ | Medium | Medium | High | High | High |

### Key Findings
- On the ACS dataset, the proposed method achieves MSE one order of magnitude lower than all baselines, with substantially higher R² as well.
- On synthetic data, when coefficients have opposite signs across subgroups (Groups A and B), vanilla Lasso fails completely, while the proposed method maintains balanced performance.
- The proposed method consistently exhibits the lowest variance across random seeds, reflecting stability of the selected features.
- The choice of downstream prediction model (random forest vs. MLP) does not affect the relative ranking of feature selection methods.

## Highlights & Insights
- **Noise Injection vs. Deterministic Scaling**: Deterministic scaling does not alter mutual information (a linear model can compensate by inverting the scale), whereas only stochastic noise genuinely reduces the information content of a feature.
- **From Specific Models to the Bayes Optimum**: Abandoning optimization of a concrete model's loss in favor of the theoretically optimal loss not only circumvents the computational difficulty of bilevel optimization, but also naturally decouples the method from any downstream model architecture.
- **Kernel Bandwidth as Feature Importance**: In the resulting Gaussian kernel weights, $\alpha$ directly controls the kernel bandwidth along each dimension, establishing a direct connection between feature selection and kernel methods.

## Limitations & Future Work
- The theoretical derivation is grounded in the bias-variance decomposition under MSE loss; extension to other losses such as cross-entropy requires non-trivial generalization.
- Fitting $\mu_i(X)$ for each subpopulation in advance means that when subgroup sample sizes are very small, estimation quality may degrade and affect the final results.
- The $k$-nearest-neighbor approximation introduces approximation error; neighbor quality may deteriorate in high-dimensional sparse settings.
- The method does not account for interaction effects among features — certain features may be individually uninformative yet highly informative in combination.

## Related Work & Insights
- **vs. Lasso**: Lasso assumes linearity and a single distribution; the proposed method is model-agnostic and robust across multiple distributions.
- **vs. Group DRO (Sagawa et al. 2019)**: Group DRO trains a single robust model, whereas the proposed method first selects features and then trains separate models per subpopulation — offering greater flexibility.
- **vs. MAML (Finn et al. 2017)**: Both have a bilevel structure, but MAML requires differentiating through inner-loop training, while the proposed method entirely avoids this by appealing to the Bayes optimum.

## Rating
- Novelty: ⭐⭐⭐⭐ First formal study of distributionally robust feature selection; the combination of noise injection and the Bayes-optimal predictor is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of both synthetic and real-world data across regression and classification settings, though large-scale high-dimensional experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, derivations proceed in a logical sequence, and the flow from motivation to method to experiments is coherent throughout.
- Value: ⭐⭐⭐⭐ The problem has strong practical relevance (medical questionnaires, sensor deployment); the method is concise and practically applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Variable Clustering via Distributionally Robust Nodewise Regression](../../ICML2026/others/variable_clustering_via_distributionally_robust_nodewise_regression.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](../../ICLR2026/others/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
