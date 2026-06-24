---
title: >-
  [Paper Note] Prediction via Shapley Value Regression (ViaSHAP)
description: >-
  [ICML2025][Shapley values] Proposes ViaSHAP, which integrates Shapley value computation into the model training process, allowing prediction to be obtained directly via summing the Shapley values during inference. It requires no post-hoc explainer, achieves XGBoost-level predictive accuracy on tabular data, and yields Shapley value approximation quality significantly superior to FastSHAP.
tags:
  - "ICML2025"
  - "Shapley values"
  - "Explainability"
  - "KAN"
  - "Feature attribution"
  - "Tabular data"
  - "Self-interpretable models"
date: 2026-05-08
content_hash: 89d51a85c5f2e72c
---

# Prediction via Shapley Value Regression (ViaSHAP)

**Conference**: ICML2025  
**arXiv**: [2505.04775](https://arxiv.org/abs/2505.04775)  
**Code**: [GitHub](https://github.com/amrmalkhatib/ViaSHAP)  
**Area**: Explainable Prediction / Explainable ML  
**Keywords**: Shapley values, Explainability, KAN, Feature attribution, Tabular data, Self-interpretable models

## TL;DR

Proposes ViaSHAP, which integrates Shapley value computation into the model training process, allowing prediction to be obtained directly via summing the Shapley values during inference. It requires no post-hoc explainer, achieves XGBoost-level predictive accuracy on tabular data, and yields Shapley value approximation quality significantly superior to FastSHAP.

## Background & Motivation

- **Key Challenge**: Shapley values possess desirable explanatory properties such as local accuracy, missingness, and consistency, being the unique feature attribution scheme satisfying these three axioms simultaneously. However, traditional approaches (KernelSHAP, FastSHAP) are post-hoc calculations, introducing substantial computational overhead during inference.
- **KernelSHAP** requires solving a weighted least squares optimization problem for each instance individually, which necessitates sampling a large number of coalitions to converge.
- **FastSHAP** trains a parameterized explainer to amortize inference costs but still requires a pre-trained black-box model as a "teacher," remaining inherently a post-hoc scheme.
- **Research Gap**: Up to now, no study has utilized Shapley value computation as a means of prediction (prediction via Shapley values), where Shapley values are calculated first and aggregated to yield the final prediction.

## Method

### Core Idea

ViaSHAP trains a function $\phi^{\mathcal{V}ia}: X \to \mathbb{R}^{n \times d}$, which outputs an $n \times d$ Shapley value matrix for an input $x$. The prediction is obtained through column summation:

$$\hat{y} = \sigma\!\left(\mathbf{1}^\top \phi^{\mathcal{V}ia}(x;\theta)\right)$$

where $\sigma$ is a link function (e.g., sigmoid or softmax). This implies that the model first computes the contribution of each feature to each output dimension, then aggregates them into the prediction.

### Loss & Training

The training simultaneously optimizes two objectives:

$$\mathcal{L}(\theta) = \sum_{x \in X}\sum_{j \in M}\left(\beta \cdot \mathbb{E}_{p(S)}\left[\left(\mathcal{V}ia_j^{\text{SHAP}}(x^S) - \mathcal{V}ia_j^{\text{SHAP}}(\mathbf{0}) - \mathbf{1}_S^\top \phi_j^{\mathcal{V}ia}(x;\theta)\right)^2\right] - y_j \log(\hat{y}_j)\right)$$

- **Shapley Loss** $\mathcal{L}_\phi$: For a randomly sampled coalition $S$, it requires the sum of Shapley values of the selected features to reconstruct the "model output when only these features are used", forcing the Shapley values to satisfy the optimal solution in the weighted least squares sense.
- **Prediction Loss**: Standard cross-entropy (classification) or MSE (regression).
- The hyperparameter $\beta$ controls the trade-off. The default is $\beta=10$, with 32 coalitions sampled per instance.

### Theoretical Guarantees

The paper proves that when $\phi^{\mathcal{V}ia}(x;\theta^*)$ reaches the global optimum:

| Property | Implication |
|------|------|
| **Local Accuracy** (Lemma 3.1) | $\sum_i \phi_i = f(x) - f(\mathbf{0})$, the sum of Shapley values equals the prediction difference |
| **Missingness** (Lemma 3.2) | Features with no effect on prediction have a Shapley value of 0 |
| **Consistency** (Lemma 3.3) | If a feature's contribution increases, its Shapley value does not decrease |
| **Theorem 3.4** | The optimal solution converges to the exact Shapley values |

### Model Implementations: Four Variants

| Variant | Architecture | Remarks |
|------|------|------|
| **KAN$^{\mathcal{V}ia}$** | Kolmogorov-Arnold Network (spline) | Layer structure $n \to 64 \to 128 \to 64 \to n \times d$ |
| **KAN$_\varrho^{\mathcal{V}ia}$** | KAN + Radial Basis Function (RBF) | Same structure as above, replacing spline with RBF |
| **MLP$^{\mathcal{V}ia}$** | Standard MLP + BatchNorm + ReLU | Same dimensional structure |
| **MLP$_\theta^{\mathcal{V}ia}$** | Widened MLP (parameter size aligned with KAN) | Used for fair comparison |

Additionally, three implementations are provided for image tasks: ResNet50$^{\mathcal{V}ia}$, ResNet18$^{\mathcal{V}ia}$, and U-Net$^{\mathcal{V}ia}$.

## Key Experimental Results

### Tabular Data Prediction Performance (25 Datasets, AUC)

| Method | Average Rank | Difference with XGBoost |
|------|----------|----------------|
| **KAN$^{\mathcal{V}ia}$**| **Best** | Statistically insignificant (Nemenyi p>0.05) |
| KAN$_\varrho^{\mathcal{V}ia}$ | Second | Statistically insignificant |
| XGBoost | Third | — |
| Random Forest | Fourth | — |
| TabNet | Fifth | — |
| MLP$_\theta^{\mathcal{V}ia}$ | Sixth | Significantly different from KAN |
| MLP$^{\mathcal{V}ia}$ | Seventh | Significantly different from KAN |

- There is no statistically significant difference between KAN variants and tree models; KAN significantly outperforms MLP variants.
- KAN$^{\mathcal{V}ia}$ also significantly outperforms the same-structure KAN classifier without Shapley loss, indicating that the Shapley loss has a regularization effect.

### Shapley Value Approximation Quality

| Metric | Best Implementation | Explanation |
|------|----------|------|
| Cosine Similarity | MLP$_\theta^{\mathcal{V}ia}$ 1st, KAN 2nd | No significant difference among the four variants via Friedman test |
| Spearman Rank Correlation | **KAN$^{\mathcal{V}ia}$** 1st | MLP$^{\mathcal{V}ia}$ is significantly different from others |
| vs FastSHAP | ViaSHAP **significantly outperforms** FastSHAP | Holds true on both tabular and image data |

### Image Experiments (CIFAR-10)

| Model | Test Accuracy | Shapley Value Quality |
|------|-----------|---------------|
| ResNet50$^{\mathcal{V}ia}$ | Competitive | Outperforms FastSHAP |
| ResNet18$^{\mathcal{V}ia}$ | Competitive | Outperforms FastSHAP |
| U-Net$^{\mathcal{V}ia}$ | Competitive | Outperforms FastSHAP |

### Ablation Study

- **Effect of $\beta$**: Increasing $\beta$ improves Shapley value accuracy without sacrificing predictive performance, but excessively large values ($\ge 200\times$) lead to training failure.
- **Number of Coalition Samples**: Has minimal impact on performance and explanation accuracy.
- **Link Function**: Removing the link function significantly improves Shapley value accuracy without degrading prediction performance.
- **Efficiency Constraints**: Have no significant impact on performance or explanation accuracy.

## Highlights & Insights

1. **Paradigm Shift**: Converted Shapley values from a "post-hoc explanation tool" into a "prediction mechanism" for the first time, achieving "inference is explanation" with zero additional overhead.
2. **Advantage of KAN**: KAN, based on the Kolmogorov-Arnold representation theorem, is more effective than MLP in learning Shapley value functions, maintaining a significant lead even when parameters are aligned.
3. **Regularization Effect of Shapley Loss**: Integrating Shapley loss improves the model's prediction performance instead of degrading it, indicating that forcing the model to learn feature contributions results in a regularization-like effect.
4. **Theoretical Completeness**: Rigorously proves that the optimal solution satisfies the three axiomatic properties of Shapley values.
5. **Architecture-Agnostic**: The method adapts to various architectures such as KAN, MLP, ResNet, and U-Net, showing good generalizability.

## Limitations & Future Work

1. **Global Optimality Assumption**: Theoretical guarantees depend on global optimality, but only local optima can be achieved in practical training, limiting the precision of the Shapley values.
2. **Increased Output Dimensionality during Inference**: The model output is expanded from $d$ dimensions to $n \times d$ dimensions, which increases parameter and computational costs in high-dimensional feature scenarios.
3. **Lack of Comparison with Latest Tabular Models**: Only compared with XGBoost/RF/TabNet, missing recent SOTA models like FT-Transformer and TabPFN.
4. **Hyperparameter Tuning of $\beta$**: Though authors claim the default value is robust, ablation studies show extreme values can cause training collapse, suggesting tuning may be necessary for different tasks.
5. **Limitations of Causal Interpretation**: Shapley values measure statistical contributions rather than causal effects, which the paper acknowledges could potentially mislead users.
6. **Security Risks**: Real-time output of Shapley values might be exploited by adversaries for model reverse engineering or adversarial attacks.

## Related Work & Insights

- **KernelSHAP** (Lundberg & Lee, 2017): Classic post-hoc Shapley value approximation serving as the theoretical foundation for ViaSHAP.
- **FastSHAP** (Jethani et al., 2022): Trains an explainer to amortize inference costs but remains fundamentally a post-hoc scheme.
- **KAN** (Liu et al., 2024): Kolmogorov-Arnold Neworks; ViaSHAP demonstrates they outperform MLPs in learning Shapley value functions.
- **Self-explaining networks** (Alvarez Melis & Jaakkola, 2018): Generate explanations but do not satisfy the Shapley axioms.
- **Insight**: The ViaSHAP framework can be transferred to stronger tabular models (such as Transformer architectures) or utilized for adversarial robustness research.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Proposes the paradigm of "prediction via Shapley value regression" for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 25 datasets + image experiments + complete ablation, but lacks comparison with the latest tabular SOTA.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical derivation and clear structure.
- Value: ⭐⭐⭐⭐ — Reconciling prediction and explanation is of practical significance, but the global optimality assumption restricts the theory's practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](gradient_aligned_regression_via_pairwise_losses.md)
- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)
- [\[ICML 2025\] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions](learning_safe_strategies_for_value_maximizing_buyers_in_uniform_price_auctions.md)

</div>

<!-- RELATED:END -->
