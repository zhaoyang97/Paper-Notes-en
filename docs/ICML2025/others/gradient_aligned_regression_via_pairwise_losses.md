---
title: >-
  [Paper Note] Gradient Aligned Regression via Pairwise Losses
description: >-
  [ICML2025][Regression Loss Functions] Proposes Gradient Aligned Regression (GAR), which aligns the gradients of the predictive and true functions by introducing two pairwise difference losses (error variance + negative Pearson correlation coefficient) in the label space, and robustly aggregates three sub-losses using DRO. This achieves the same linear complexity as traditional regression losses, while outperforming MAE/MSE and contrastive learning methods on multiple benchmar…
tags:
  - "ICML2025"
  - "Regression Loss Functions"
  - "Pairwise Losses"
  - "Gradient Alignment"
  - "Distributionally Robust Optimization"
  - "Pearson Correlation Coefficient"
date: 2026-05-08
content_hash: be7567c5efba9ec0
---

# Gradient Aligned Regression via Pairwise Losses

**Conference**: ICML2025  
**arXiv**: [2402.06104](https://arxiv.org/abs/2402.06104)  
**Code**: [GitHub](https://github.com/DixianZhu/GAR)  
**Area**: Regression / Robustness  
**Keywords**: Regression Loss Functions, Pairwise Losses, Gradient Alignment, Distributionally Robust Optimization, Pearson Correlation Coefficient

## TL;DR

Proposes Gradient Aligned Regression (GAR), which aligns the gradients of the predictive and true functions by introducing two pairwise difference losses (error variance + negative Pearson correlation coefficient) in the label space, and robustly aggregates three sub-losses using DRO. This achieves the same linear complexity as traditional regression losses, while outperforming MAE/MSE and contrastive learning methods on multiple benchmarks.

## Background & Motivation

### Limitations of Traditional Regression

Traditional regression losses (MAE, MSE, Huber) only focus on the magnitude of individual sample prediction errors $\delta_{\mathbf{x}}^f = f(\mathbf{x}) - y$, **failing to capture relationship patterns between samples**. For instance, given two models with errors $\{1, -1, 1, -1\}$ and $\{1, 1, 1, 1\}$, their MAE/MSE are identical, but the latter has zero error variance, exhibits better order-preservation, and can achieve zero error with a simple bias correction.

### Limitations of Prior Work

Recent methods like RankSim, RNC, and ConR impose pairwise similarity constraints in the feature space:

- **High Computational Overhead**: Requires $O(N^2)$ pairwise computations.
- **Information Loss**: Converting continuous label similarity into discrete rankings or positive/negative pairs leads to irreversible approximation loss.
- **Lack of Theoretical Explanation**: Fails to establish a connection with learning function gradients.

### Design Motivation

Directly modeling pairwise differences $f(\mathbf{x}_i) - f(\mathbf{x}_j) \approx y_i - y_j$ in the label space retains the complete label difference information while reducing the complexity to linear through equivalent transformations.

## Method

### Overall Architecture

GAR consists of three losses:

$$\mathcal{L}_{\text{GAR}} = \text{DRO-Aggregate}(\mathcal{L}_c^{\text{MAE}},\; \mathcal{L}_{\text{diff}}^{\text{MSE}},\; \mathcal{L}_{\text{diffnorm}}^{p=2};\; \alpha)$$

### Loss 1: Conventional MAE Loss

$$\text{L}_c^{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|f(\mathbf{x}_i) - y_i|$$

Responsible for point-wise fitting of predicted values to ground truth.

### Loss 2: Pairwise Difference Loss $\to$ Error Variance

The original formulation is an $O(N^2)$ pairwise MSE:

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \frac{1}{N^2}\sum_{i}\sum_{j} \frac{1}{2}\big[(f(\mathbf{x}_i)-f(\mathbf{x}_j))-(y_i-y_j)\big]^2$$

**Theorem 1** proves it is equivalent to the variance of the prediction error:

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \text{Var}(\delta_{\mathbf{x}}^f)$$

The equivalent linear-complexity form is:

$$\mathcal{L}_{\text{diff}}^{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}\big[(f(\mathbf{x}_i)-\bar{f})-(y_i-\bar{y})\big]^2$$

**Intuition**: Minimizing error variance implies that errors across all samples tend to be consistent, which enhances order-preservation.

### Loss 3: Normalized Pairwise Difference $\to$ Negative Pearson Correlation

Introducing a scaling factor and normalizing the pairwise differences yields the loss under the $\ell_2$ norm:

$$\mathcal{L}_{\text{diffnorm}}^{p=2} = 1 - \rho(f, y) = 1 - \frac{\text{Cov}(f, y)}{\sqrt{\text{Var}(f)\text{Var}(y)}}$$

**Intuition**: Maximizing the Pearson correlation coefficient is equivalent to capturing the "shape" (alignment of direction) of the predictive function and the true function, allowing variations in amplitude.

### Theoretical Insight: Gradient Alignment

**Theorem 4** (Core Theorem): For $K$-order differentiable deterministic functions, exact matching of pairwise label differences is equivalent to exact matching of gradients of all orders:

$$f(\mathbf{x}_1)-f(\mathbf{x}_2)=y_1-y_2,\;\forall (\mathbf{x}_1,\mathbf{x}_2) \iff \nabla^k f(\mathbf{x})=\nabla^k \mathcal{Y}(\mathbf{x}),\;k=1,...,K$$

The proof is based on the mean value theorem and L'Hôpital's rule. This implies that pairwise losses **implicitly learn the gradient field of the true function**.

### DRO Robust Aggregation

The scales of the three sub-losses differ significantly (e.g., MAE can be unbounded, while the Pearson loss $\in[0,2]$); simple arithmetic or geometric averaging has drawbacks. GAR adopts KL-divergence-based DRO aggregation:

$$\mathcal{L}_{\text{GAR}}^{\text{KL}} = \alpha\log\Big(\frac{1}{M}\sum_{i=1}^{M}\mathcal{L}_i^{1/\alpha}\Big)$$

- $\alpha \to 0$: Degenerates to the $\max$ loss.
- $\alpha = 1$: Arithmetic mean.
- $\alpha \to +\infty$: Geometric mean.

Experiments default to $\alpha = 0.5$ to balance attention on smaller losses and numerical stability. Numerical overflow is prevented through normalization using $\mathcal{L}_{\max}$ or $\mathcal{L}_{\min}$.

### Complexity

The overall algorithm requires only $O(B)$ ($B$ represents the batch size) per iteration, matching MAE/MSE, without requiring pairwise computations in the feature space.

## Key Experimental Results

### Synthetic Datasets

| Dataset | MAE (MAE↓) | MSE (MAE↓) | RNC (MAE↓) | **GAR (MAE↓)** |
|--------|-----------|-----------|-----------|-------------|
| Sine | Decent, misses 1-2 peaks | Similar to MAE | Medium | **Captures the most peaks** |
| Squared Sine | Only captures the largest amplitude peak | Similar to MAE | Medium | **Almost completely recovers the true pattern** |

### Real-world Datasets (8 tasks, 5 tabular + 1 image benchmark)

| Dataset | Metric | MAE | Best Competitor | **GAR** | Gain |
|--------|------|-----|---------|---------|------|
| Concrete | MAE↓ | 4.976 | 4.698(Huber) | **4.603** | 7.5%/2.0% |
| Concrete | Pearson↑ | 0.919 | 0.923(RNC) | **0.929** | 1.1%/0.6% |
| Wine | MAE↓ | 0.500 | 0.500(MAE) | **0.494** | 1.1%/1.1% |
| STS-B | Pearson↑ | 0.865 | 0.880(RNC) | **0.882** | 2.0%/0.2% |
| IMDB-WIKI | MAE↓ | 6.685 | 6.468(ConR) | **6.366** | 4.8%/1.6% |

- GAR outperforms or equals all baselines on the MAE metric across **all 8 tasks**.
- It also maintains superiority on Pearson/Spearman correlation coefficients.
- p-value tests show that most of the gains are statistically significant.

### Runtime Comparison

| Method | Relative Time (vs MAE) |
|------|-------------------|
| MAE | 1.0× |
| RankSim | ~2.5× |
| RNC | ~1.8× |
| ConR | ~2.0× |
| **GAR** | **~1.0×** |

The runtime of GAR is almost identical to MAE, which is significantly faster than all methods that perform pairwise computations in the feature space.

## Highlights & Insights

1. **Elegant Equivalent Transformation**: The $O(N^2)$ pairwise loss is proven to be equivalent to the error variance and Pearson correlation coefficient, reducing the complexity to $O(N)$ linear complexity, which is theoretically sound and practical.
2. **Theoretical Insight of Gradient Alignment** (Theorem 4): For the first time, an equivalence relation is established between learning pairwise label differences and function gradient matching, providing profound mathematical intuition for the method.
3. **DRO Aggregation Mechanism**: A single hyperparameter $\alpha$ unifies the trade-off among arithmetic mean, geometric mean, and the maximum value, eliminating the need to manually tune multiple loss weights.
4. **Zero Extra Computational Overhead**: Achieves the same efficiency as MAE/MSE, running 1.8-2.5 times faster than RankSim/RNC/ConR.
5. **Broad Applicability**: Effective across various scenarios, including tabular regression and image age estimation.

## Limitations & Future Work

1. **Clean Data Only**: The authors explicitly restrict the scope of research to settings without noise, outliers, or distribution shifts; robustness to dirty data remains unverified.
2. **Selection of Hyperparameter $\alpha$**: Although reduced to a single hyperparameter, the optimal value of $\alpha$ still requires tuning across different tasks.
3. **Strong Theoretical Assumptions**: Theorem 4 assumes functions are $K$-order differentiable on open domains, which limits its applicability to non-smooth or discrete problems.
4. **Single-Target Regression**: Results are only demonstrated on single-dimensional targets; the efficacy in multi-target regression scenarios needs verification.
5. **Model Architecture Constraints**: Experiments predominantly utilize simple FFNNs and ResNets; the performance on large models like Transformers remains unexplored.
6. **Advantage Scenarios of Contrastive Learning Methods**: Contrastive learning in the feature space may possess unique advantages in representation learning (such as the pre-training paradigm of RNC), which GAR, operating solely in the label space, might struggle to capture.

## Related Work & Insights

- **RankSim** (Gong et al., 2022): Label similarity $\to$ rank regularization, but loses continuous information.
- **RNC** (Zha et al., 2023): Contrastive pre-training + fine-tuning, highly effective but computationally expensive.
- **ConR** (Keramati et al., 2023): Contrastive regularizer, employing a different way of defining positive and negative pairs.
- **Insights**: The approach of GAR (simple mathematical equivalent transformations in the label space) can be extended to other tasks requiring modeling of relationships between samples (e.g., learning to rank, uncertainty quantification).

## Rating

- Novelty: ⭐⭐⭐⭐ (Clever equivalent transformation, novel gradient alignment theory)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 real-world tasks + 2 synthetic + ablation + runtime, but lacks experiments on noisy/dirty data)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, standardized notations, complete theorems, although some formulas are dense)
- Value: ⭐⭐⭐⭐ (Highly practical, zero extra computational overhead, plug-and-play loss function enhancement)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](prediction_via_shapley_value_regression.md)
- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](../../NeurIPS2025/others/rethinking_losses_for_diffusion_bridge_samplers.md)
- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
