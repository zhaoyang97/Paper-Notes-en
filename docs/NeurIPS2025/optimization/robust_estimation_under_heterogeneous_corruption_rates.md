---
title: >-
  [Paper Note] Robust Estimation Under Heterogeneous Corruption Rates
description: >-
  [NeurIPS 2025][Optimization][robust estimation] This paper studies robust estimation under heterogeneous corruption rates—where each sample is corrupted with a distinct known probability—and establishes tight minimax rat…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "robust estimation"
  - "heterogeneous corruption"
  - "minimax rates"
  - "mean estimation"
  - "linear regression"
date: 2026-05-08
content_hash: 784ea98e16e89abf
---

# Robust Estimation Under Heterogeneous Corruption Rates

**Conference**: NeurIPS 2025
**arXiv**: [2508.15051](https://arxiv.org/abs/2508.15051)
**Code**: Unavailable
**Area**: Optimization
**Keywords**: robust estimation, heterogeneous corruption, minimax rates, mean estimation, linear regression

## TL;DR

This paper studies robust estimation under heterogeneous corruption rates—where each sample is corrupted with a distinct known probability—and establishes tight minimax rates for mean estimation and linear regression under both bounded and Gaussian distributions. A key finding is that the optimal estimator can simply discard samples whose corruption rates exceed a certain threshold.

## Background & Motivation

Classical robust statistics assumes that all data are corrupted with a uniform probability $\epsilon$ (the Huber contamination model), which is theoretically well understood. In practice, however—such as in federated learning, crowdsourced annotation, and sensor networks—data originate from heterogeneous sources with widely varying reliability. For example, large-scale biomedical datasets collected by different institutions at different times naturally exhibit varying proportions of outliers across subsets.

The root cause of the open problem is that all existing robust statistical techniques assume a uniform noise prior, leaving it unclear **whether optimal statistical rates are achievable under heterogeneous corruption**, or even what those optimal rates should be.

Prior work has established minimax rates for robust estimation under uniform corruption (e.g., the rate $d/n + \epsilon^2$ for Gaussian mean estimation), but these results do not extend directly to the heterogeneous setting—naively setting $\epsilon = \max_i \lambda_i$ yields highly conservative estimates.

The authors propose a natural generalization, the $\boldsymbol{\lambda}$**-contamination model** (Definition 1): sample $i$ is corrupted with known probability $\lambda_i$, where the $\lambda_i$ may differ across samples. The central technical challenge is **how to effectively combine information from data points with different noise levels**. A canonical example is semi-verified learning: a small amount of trusted data ($\lambda_i = 0$) combined with a large amount of high-noise data ($\lambda_i \approx 1$), where neither subset alone suffices for accurate estimation, but together they can.

## Method

### Overall Architecture

The authors establish matching upper and lower bounds on the minimax rates for four fundamental estimation problems: mean estimation for bounded distributions, one-dimensional Gaussian mean estimation, multivariate Gaussian mean estimation, and linear regression with Gaussian design. A central finding is that a single "effective corruption rate" function $f(\boldsymbol{\lambda}, k)$ governs all results.

### Key Designs

1. **$\boldsymbol{\lambda}$-Contamination Model (Definition 1)**: Formalizes heterogeneous corruption. The observed sample is $Z_i = (1-B_i)X_i + B_i\tilde{X}_i$, where $B_i \sim \text{Bern}(\lambda_i)$, clean samples satisfy $X_i \overset{iid}{\sim} P$, and outliers $\tilde{X}_i$ may be chosen by an adaptive adversary. A key assumption is that $\boldsymbol{\lambda}$ is known (or that a componentwise upper bound is known).

2. **Effective Corruption Rate Function**: Define $f(\boldsymbol{\lambda}, k) = \min_{t \in [0,1]} \left(\frac{k}{|\{i: \lambda_i \leq t\}|} + t^2\right)$. This function finds the optimal balance over threshold $t$, simultaneously accounting for the number of retained samples (statistical precision term $k/N(t)$) and the maximum permitted corruption rate (bias term $t^2$). This function is central to all results in the paper.

3. **Threshold Estimator (Upper Bound)**: For bounded distributions, the simplest approach is to select a threshold $t$ and compute the mean using only samples with $\lambda_i \leq t$. The optimal threshold $t^* = \arg\min_{t}(1/N(t) + t^2)$ yields the rate $r^2 f(\boldsymbol{\lambda}, 1)$. More refined weighted estimators $\sum w_i Z_i$ minimizing $\|w\|_2^2 + 3(w^T\lambda)^2$ achieve bounds of the same order while potentially performing better in practice. Algorithm 1 provides a near-linear-time exact algorithm for optimal weights.

4. **Le Cam Two-Point Lower Bound**: Two hypotheses $P_0, P_1$ (Bernoulli distributions supported on $\pm r e_1$) are constructed. The adversary's strategy is: for samples with $\lambda_i \geq 2\delta/(1+2\delta)$, the adversary can make the induced distribution identical under both hypotheses ("simulation"), rendering these samples entirely uninformative. Only samples with $\lambda_i < 2\delta/(1+2\delta)$ contribute to hypothesis testing. Choosing $\delta$ appropriately matches the upper bound.

5. **Weighted Tukey Median (Gaussian Setting)**: A weighted depth is defined as $D_w(\eta, \boldsymbol{Z}) = \min_{v \in \mathbb{S}_d} \sum_i w_i \mathbb{I}\{v^T(Z_i - \eta) \geq 0\}$, which downweights highly corrupted samples to improve robustness. For Gaussian mean estimation, the upper bound takes the form $(w^T\lambda)^2 + d\|w\|_2^2$; a threshold variant is also applicable.

6. **Weighted Regression Depth (Linear Regression)**: An analogous weighted regression depth is defined, measuring estimation quality via the uniformity of residual signs across directions, yielding upper and lower bounds of the same form.

### Matching Upper and Lower Bounds

The proof for bounded distributions is the cleanest: a critical threshold $\delta^*$ is identified such that $N(2\delta^*) \geq \frac{1}{12\delta^{*2}}$ while $n(2\delta^*) \leq \frac{1}{12\delta^{*2}}$; substituting into both bounds yields the matching rate $\Theta(r^2\delta^{*2})$.

## Key Experimental Results

### Bounded Distribution Experiments

| Method | Applicability | Exploits Heterogeneity | MSE Performance |
|---|---|---|---|
| Uniform robust estimator (baseline) | Uses $\max \lambda$ | No | Worst |
| Threshold method (Ours) | Selects optimal threshold | Partially | Near-optimal |
| Weighted method (Ours, Algorithm 1) | Per-sample weighting | Fully | Optimal |

### Gaussian Mean Estimation Comparison

| Method | Estimation Error | Notes |
|---|---|---|
| Uniform-weight Tukey median | High | Highly corrupted samples skew depth estimation (Fig. 1A) |
| Threshold-weighted Tukey median | Moderate | Discards high-corruption samples (Fig. 1B) |
| Optimally-weighted Tukey median | Lowest | Smoothly leverages all available information (Fig. 1C) |

Figure 1 intuitively illustrates the Tukey depth contours under three weighting schemes: the deepest point (yellow star) under the optimal weighting scheme is closest to the true mean.

### Key Findings

- The minimax rates for bounded distributions and one-dimensional Gaussians are fully determined, with matching upper and lower bounds.
- For multivariate Gaussians and linear regression, a $\sqrt{d}$ gap remains (upper bound $f(\boldsymbol{\lambda}, d)$, lower bound $f(\boldsymbol{\lambda}, d)/\sqrt{d}$), which is tight in constant dimension.
- Samples whose corruption rates exceed the optimal threshold $t^*$ can be discarded without sacrificing optimality up to constant factors.
- Per-sample weighting can outperform simple thresholding in practice while maintaining theoretical optimality.

## Highlights & Insights

- The function $f(\boldsymbol{\lambda}, k)$ elegantly unifies all results; it essentially identifies the optimal trade-off between statistical precision and corruption bias.
- The threshold estimator is of significant practical value: in heterogeneous settings, no sophisticated algorithm is required—selecting the right threshold and applying a standard robust estimator suffices.
- The adversary construction in the Le Cam lower bound is technically elegant: the adversary can "simulate" the target distribution, rendering highly corrupted samples completely uninformative.
- The paper reframes heterogeneity in robust statistics from a deliberately ignored complication into an exploitable structure.

## Limitations & Future Work

- The $\sqrt{d}$ gap for multivariate Gaussians and linear regression remains unresolved; the authors suggest that closing it may require fundamentally new complexity measures to characterize the contribution of heterogeneity precisely.
- $\boldsymbol{\lambda}$ is assumed known (though constant-factor approximations suffice); in practice, upper bounds on corruption rates may need to be estimated from historical data.
- In the semi-verified learning setting, the upper bound of the threshold method diverges with $d$, whereas prior work has shown dimension-free rates are achievable—indicating that the threshold method is not universally optimal.
- Only the Huber-type (additive) contamination model is considered; other models (e.g., total variation, KL divergence) remain to be explored.

## Related Work & Insights

- The paper is directly related to the semi-verified learning work of Charikar et al. (2017): it provides a natural generalization, and recovers existing results as a special case.
- The paper contrasts with the techniques in Diakonikolas & Kane (2023): standard methods assume uniform corruption, and new lower bound techniques are needed here to jointly capture the effects of dimensionality and heterogeneity.
- How the deep connection between robustness and privacy (Georgiev & Hopkins, 2022) generalizes to the heterogeneous setting remains an interesting open problem.
- Weighted Tukey depth and regression depth serve as the technical core of the paper, extending classical statistics to the heterogeneous weighting setting.
- Prior work on heterogeneous privacy requirements in the privacy literature (Cummings et al., 2023) is limited to the one-dimensional bounded case; this paper achieves substantial progress in the multivariate setting.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic establishment of minimax rate theory under heterogeneous corruption rates.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical contribution; experiments serve as supplementary synthetic validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formalization is clear, results are stated concisely, and the central concept (the $f$ function) is woven consistently throughout.
- Value: ⭐⭐⭐⭐ Fills a theoretical gap in robust statistics for the heterogeneous corruption setting, with important theoretical implications for practical scenarios such as federated learning and crowdsourced annotation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Near-Exponential Savings for Mean Estimation with Active Learning](near-exponential_savings_for_mean_estimation_with_active_learning.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts](on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

</div>

<!-- RELATED:END -->
