---
title: >-
  [Paper Note] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction
description: >-
  [ICML2026][Optimization][conformal prediction] This paper reveals the inherent deficiency of standard pinball loss in optimizing conditional coverage through Taylor expansion—specifically…
tags:
  - "ICML2026"
  - "Optimization"
  - "conformal prediction"
  - "conditional coverage"
  - "quantile regression"
  - "density weighting"
  - "pinball loss"
date: 2026-05-08
content_hash: 2253b42edb1917bc
---

# Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction

**Conference**: ICML2026  
**arXiv**: [2512.24139](https://arxiv.org/abs/2512.24139)  
**Code**: https://github.com/Colorful-Pinball/CPCP  
**Area**: optimization  
**Keywords**: conformal prediction, conditional coverage, quantile regression, density weighting, pinball loss  

## TL;DR

This paper reveals the inherent deficiency of standard pinball loss in optimizing conditional coverage through Taylor expansion—specifically, the neglect of heteroscedastic structures. It proposes density-weighted pinball loss as a tighter surrogate objective for conditional coverage MSE and designs a three-head quantile network to estimate density weights via finite differences, significantly improving conditional coverage performance across eight high-dimensional regression benchmarks.

## Background & Motivation

**Background**: Conformal Prediction (CP) is the current mainstream paradigm for uncertainty quantification, providing distribution-free marginal coverage guarantees $\mathbb{P}(Y \in \mathcal{C}_\alpha(X)) \geq 1-\alpha$ with finite samples. However, standard split CP only ensures marginal coverage at the population level and fails to guarantee conditional coverage $\mathbb{P}(Y \in \mathcal{C}_\alpha(X) \mid X=x)$ for specific inputs $x$, which is often the actual requirement in high-risk scenarios.

**Limitations of Prior Work**: Existing methods for conditional coverage primarily follow two paths: (1) approximating conditional guarantees through grouping/localization (e.g., group-conditional, localized CP), which are limited by the curse of dimensionality; (2) improving non-conformity score functions (e.g., CQR, RCP) by calibrating the heteroscedasticity of scores through quantile regression. However, a systematic bias exists between the optimization objective of standard pinball loss and the conditional coverage MSE.

**Key Challenge**: Prior works (Kiyani et al., 2024; Plassier et al., 2025a) established upper bound connections between MSCE and the excess risk of pinball loss. However, these bounds depend on the global Lipschitz constant $L_F$ of the conditional CDF and are typically loose, as they ignore variations in $f_{S|X}(q_\tau(x))$ across different $x$—i.e., the difference in the "steepness" of the conditional score distribution at the target quantile.

**Goal**: To directly approximate the MSE of conditional coverage (MSCE) rather than relying on loose upper bounds or relaxed definitions of conditional coverage.

**Key Insight**: Through Taylor expansion of MSCE, the authors discover that the dominant term is the **density-weighted pinball excess risk** $\mathbb{E}_X[f_{S|X}(q_\tau(X)) \cdot \mathcal{E}(X)]$. The weights correspond exactly to the values of the conditional density at the true quantiles. Under the location-scale family, this weight is proportional to $1/\sigma(x)$, assigning higher weights to low-variance (steep CDF) regions where conditional coverage is most sensitive—small quantile errors in these regions can cause coverage to plummet from 95% to 80%.

**Core Idea**: Replace standard pinball loss with density-weighted pinball loss to train quantile regression, using finite differences from auxiliary quantiles to estimate density weights, thereby compensating for the standard method's neglect of heteroscedastic structures.

## Method

### Overall Architecture

CPCP (Colorful Pinball Conformal Prediction) divides the calibration set into three subsets $\mathcal{D}_{\text{cal},1}, \mathcal{D}_{\text{cal},2}, \mathcal{D}_{\text{cal},3}$ and executes a three-stage process: (1) joint training of three quantile estimators (target quantile $\tau$ and auxiliary quantiles $\tau \pm \delta$) on $\mathcal{D}_{\text{cal},1}$; (2) construction of finite-difference density weights using auxiliary quantiles to fine-tune the target quantile on $\mathcal{D}_{\text{cal},2}$ with weighted pinball loss; (3) performing RCP rectified score calibration on $\mathcal{D}_{\text{cal},3}$ to ensure marginal validity. The final output is the prediction set $\mathcal{C}_\alpha(x_{\text{test}}) = \{y: S(x_{\text{test}}, y) \leq \hat{q}_\tau(x_{\text{test}}) + \hat{\gamma}\}$.

### Key Designs

1.  **Density-weighted pinball loss (Theoretical Core)**:
    -   **Function**: Construct a tighter surrogate optimization objective for MSCE.
    -   **Mechanism**: Via Taylor expansion, the dominant term of the squared conditional coverage bias $(F_{S|X}(\hat{q}_\tau(x)) - \tau)^2$ is $f_{S|X}(q_\tau(x))^2 \cdot \epsilon_q(x)^2$, while the dominant term of the standard pinball excess risk is $\frac{1}{2} f_{S|X}(q_\tau(x)) \cdot \epsilon_q(x)^2$. These differ by a factor of $f_{S|X}(q_\tau(x))$. Thus, weighting the pinball loss by the density $f_{S|X}(q_\tau(x))$ aligns the optimization objective precisely with the dominant term of MSCE.
    -   **Design Motivation**: Standard pinball loss assigns insufficient weight to regions where the conditional CDF is steep ($f_{S|X}$ is large, $\sigma(x)$ is small), which are precisely the areas where conditional coverage is most sensitive.

2.  **Three-head Quantile Network + Finite-Difference Density Estimation**:
    -   **Function**: Obtain density weights without explicit density estimation.
    -   **Mechanism**: Utilizing the inverse relationship between the quantile function and the CDF, $\partial q_\tau(x)/\partial \tau = 1/f_{S|X}(q_\tau(x))$, the density is approximated via finite difference: $\hat{w}(x) = 2\delta / (\hat{q}_{\tau+\delta}(x) - \hat{q}_{\tau-\delta}(x))$. The network uses a shared backbone $h(x)$ with three projection heads. Auxiliary quantiles are constructed as $\hat{q}_{\tau \pm \delta}(x) = \hat{q}_\tau(x) \pm \text{Softplus}(\phi_{\text{high/low}} \circ h(x))$, where Softplus activation ensures monotonicity and prevents quantile crossing.
    -   **Design Motivation**: Avoid explicitly estimating the full conditional density $f_{S|X}$ (which is harder than regression) by using only two auxiliary quantiles to obtain the required weights.

3.  **Weight Clipping + Loss Mixing (Finite-Sample Stabilization)**:
    -   **Function**: Control the variance explosion of inverse weights.
    -   **Mechanism**: Weight clipping truncates extreme weights to $M$ times the empirical mean; loss mixing sets the optimization objective as a convex combination of weighted and standard pinball loss, equivalent to imposing an artificial lower bound on weights.
    -   **Design Motivation**: The denominator of the finite-difference estimator may approach zero, leading to weight explosion, especially when $\delta$ is small. Clipping and mixing trade slight bias for a significant reduction in variance, similar to classic strategies like inverse propensity score clipping in causal inference.

### Loss & Training

A two-stage training strategy is adopted: in the first stage, three quantile heads are jointly trained on $\mathcal{D}_{\text{cal},1}$ using standard pinball loss (ensuring finite-difference accuracy); in the second stage, the backbone and auxiliary heads are frozen, and only the main head is fine-tuned on $\mathcal{D}_{\text{cal},2}$ using the weighted pinball loss. Finally, the empirical quantile $\hat{\gamma}$ of the rectified scores is computed on $\mathcal{D}_{\text{cal},3}$ to guarantee marginal validity.

## Key Experimental Results

### Main Results: Conditional Coverage Performance (MSCE ↓)

Mean Squared Coverage Error (MSCE) is compared across 8 high-dimensional regression benchmarks with a target coverage $\tau = 90\%$ and 20 repetitions:

| Method | Bike | Diamond | Naval | SGEMM | Transcoding | WEC |
|------|------|---------|-------|-------|-------------|-----|
| Split CP | 0.0031 | 0.0118 | 0.0351 | 0.0039 | 0.0125 | 0.0123 |
| CQR | 0.0011 | 0.0010 | 0.0120 | 0.0012 | 0.0016 | 0.0061 |
| RCP | 0.0010 | 0.0013 | 0.0029 | 0.0007 | 0.0009 | 0.0030 |
| CPCP | 0.0009 | 0.0009 | 0.0019 | 0.0003 | 0.0009 | 0.0025 |
| **CPCP (Clip+Mix)** | **0.0008** | **0.0004** | **0.0019** | **0.0003** | **0.0004** | **0.012** |

### Ablation Study / Worst-Slice Coverage (WSC ↑)

| Method | Bike | Diamond | Naval | SGEMM | Transcoding | WEC |
|------|------|---------|-------|-------|-------------|-----|
| Split CP | 0.8133 | 0.6480 | 0.5428 | 0.7435 | 0.6797 | 0.7623 |
| CQR | 0.8641 | 0.8563 | 0.6997 | 0.8393 | 0.8175 | 0.8149 |
| RCP | 0.8849 | 0.8448 | 0.8002 | 0.8627 | 0.8515 | 0.8516 |
| **CPCP (Clip+Mix)** | **0.8882** | **0.8802** | **0.8320** | **0.8912** | **0.8759** | **0.8715** |

### Key Findings

- CPCP (Clip+Mix) reduces MSCE by approximately 40-60% on average compared to RCP and improves worst-slice coverage (WSC) from ~80% to ~87-89%.
- Ablation experiments show that RCP-MultiHead (joint training without density weights) performs similarly to RCP, proving that improvements stem from the density-weighted objective itself rather than multi-task learning or extra capacity.
- The results are robust for bandwidth $\delta$ in the range of 0.01-0.05, with a default of $\delta=0.02$.
- The stabilization strategies (weight clipping + loss mixing) consistently provide gains across all datasets, particularly in high-dimensional multi-output datasets like SGEMM and Transcoding.

## Highlights & Insights

- **Elegant Theoretical Insight**: Through Taylor expansion, the paper precisely characterizes the "density factor difference" between standard pinball loss and conditional coverage MSCE. It transforms the black-box conditional coverage optimization problem into a weighted regression problem with clear mathematical motivation. The result that weights degenerate to $1/\sigma(x)$ under the location-scale family is also highly intuitive.
- **Clever Design of Finite-Difference Density Estimation**: By leveraging the relationship between quantile functions and the inverse of the CDF, the method avoids the more difficult task of conditional density estimation. The Softplus parameterization simultaneously addresses the practical issues of quantile crossing and negative weights.
- **Strong Theoretical Completeness**: The paper provides a complete non-asymptotic excess risk bound, including precise characterization of the inverse weights in estimated weights. This theoretical tool is also valuable for other problems involving inverse propensity weighting, such as causal inference and off-policy evaluation.

## Limitations & Future Work

- The calibration set is split into three parts, reducing the effective sample size for each step to about 1/3 of the original, which may degrade performance in small calibration set scenarios.
- Density weight estimation depends on the accuracy of auxiliary quantiles and may be unstable at extreme quantiles (e.g., $\tau$ near 0 or 1) or in sparse data regions.
- The theoretical rate $O(n^{-1/3})$ is slower than standard quantile regression. Although the constant factor advantage of density weighting compensates for this in finite samples, standard methods might catch up in very large sample regimes.
- Currently, only regression tasks have been validated. Future work could explore extensions to conditional coverage guarantees for classification and structured output scenarios (e.g., graphs, sequences).

## Related Work & Insights

- **RCP** (Plassier et al., 2025a): The direct foundation of CPCP, providing the rectified score framework; CPCP can be viewed as introducing a superior weighted objective in the quantile regression stage of RCP.
- **CQR** (Romano et al., 2019): The classic quantile regression + conformal calibration method, though it requires replacing the objective function on the training set.
- **PLCP** (Kiyani et al., 2024): Established the link between MSCE and pinball excess risk but used discrete grouping approximations with a slower convergence rate of only $O(n^{-1/4})$.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/optimization/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[CVPR 2026\] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](../../CVPR2026/optimization/conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)

</div>

<!-- RELATED:END -->
