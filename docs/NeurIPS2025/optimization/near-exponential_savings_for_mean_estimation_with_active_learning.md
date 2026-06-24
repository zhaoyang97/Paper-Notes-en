---
title: >-
  [Paper Note] Near-Exponential Savings for Mean Estimation with Active Learning
description: >-
  [NeurIPS 2025][Optimization][Active Learning] This paper proposes the PartiBandits algorithm, which combines disagreement-based active learning with UCB-style stratified sampling to achieve near-exponential label savings for mean estimation when auxiliary information $X$ is predictive of the target variable $Y$.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Active Learning"
  - "Mean Estimation"
  - "Stratified Sampling"
  - "UCB Algorithm"
  - "Minimax Optimality"
date: 2026-05-08
content_hash: f834a71f44fda524
---

# Near-Exponential Savings for Mean Estimation with Active Learning

**Conference**: NeurIPS 2025
**arXiv**: [2511.05736](https://arxiv.org/abs/2511.05736)  
**Code**: [R Package: PartiBandits](https://cran.r-project.org/)  
**Area**: Optimization
**Keywords**: Active Learning, Mean Estimation, Stratified Sampling, UCB Algorithm, Minimax Optimality

## TL;DR

This paper proposes the PartiBandits algorithm, which combines disagreement-based active learning with UCB-style stratified sampling to achieve near-exponential label savings for mean estimation when auxiliary information $X$ is predictive of the target variable $Y$.

## Background & Motivation

Estimating the population mean $\mathbb{E}[Y]$ of a $k$-class random variable $Y$ is a fundamental problem in statistics and machine learning. When labels are expensive but auxiliary information (covariates $X$) is abundant, the key question is how to leverage $X$ to reduce the number of required labels.

**Simple Random Sampling (SRS)** achieves an error rate of $\mathcal{O}(\mathrm{Var}(Y)/N)$ without exploiting information in $X$; **Stratified Random Sampling (StRS)** can utilize $X$ but requires a good stratification scheme to be known in advance, and label allocation may be suboptimal when within-stratum variances are unknown.

Existing active learning literature focuses primarily on label efficiency for classification tasks, or on subgroup mean estimation under predefined stratifications. However, no prior work systematically addresses: **how to achieve efficient population mean estimation via active learning when the optimal stratification scheme is unknown**.

There are two core challenges: (1) learning a good stratification scheme from unlabeled data; and (2) adaptively allocating the label budget while simultaneously estimating variances.

## Method

### Overall Architecture

PartiBandits is a two-phase algorithm:

- **Phase 1**: A disagreement-based active learning algorithm $\mathcal{S}$ (e.g., the $A^2$ algorithm) is used to learn a classifier $\hat{h}$ with half the label budget; the pre-image mapping of $\hat{h}$ then induces a stratification scheme $\mathcal{G} = \{A_i = \hat{h}^{-1}(i)\}$.
- **Phase 2**: Given the learned stratification, the WarmStart-UCB subroutine adaptively allocates the remaining label budget to estimate the population mean.

### Key Designs

1. **WarmStart-UCB Subroutine**: Estimates the population mean under a given stratification $\mathcal{G}$. The core idea is to maintain a UCB upper bound on the conditional variance of each stratum, $\mathrm{UCB}_t(\sigma_g) = \hat{\sigma}_{g,t} + C_N(\delta)/\sqrt{n_{g,t}}$, and at each round sample from the stratum with the largest $\mathrm{UCB}_t(\sigma_g)/n_{g,t}$. The key innovation is a "warm start" phase: a fraction $\tau$ of the label budget is first distributed uniformly across all strata, ensuring a minimum number of samples per stratum. This eliminates pathological dependence on $\sigma_{\min}$ (the minimum within-stratum conditional variance) and guarantees non-asymptotic high-probability bounds for **all** label budgets.

   **Theorem 1**: $|\hat{\mu}_{\text{WS-UCB}} - \mathbb{E}[Y]|^2 = \tilde{\mathcal{O}}(\Sigma_1(\mathcal{G})/N)$, where $\Sigma_1(\mathcal{G}) = \sum_g \sigma_g'^2 P_g$ is the average within-stratum variance. By the law of total variance, $\Sigma_1(\mathcal{G}) \leq \mathrm{Var}(Y)$, so the rate is at least as fast as SRS.

2. **Convergence Guarantee of PartiBandits**: The classifier $\hat{h}$ learned via the disagreement-based active learning algorithm has excess risk that decays to the Bayes-optimal risk $\nu$ at a near-exponential rate. By a bias-variance decomposition, the squared loss of the classifier upper-bounds the average within-stratum variance of the induced stratification. Therefore:

   **Theorem 3**: $|\hat{\mu}_{\text{PB}} - \mathbb{E}[Y]|^2 = \tilde{\mathcal{O}}\left(\frac{\nu + \exp(c \cdot (-N/\log N))}{N}\right)$

   When $\nu$ is small (i.e., $X$ is strongly predictive of $Y$), the rate is substantially faster than the SRS rate of $\mathcal{O}(1/N)$.

3. **Minimax Optimality**: Theorems 2 and 4 establish matching lower bounds for WarmStart-UCB and PartiBandits respectively, proving that both rates are minimax optimal in the classical setting. The lower bounds are constructed based on threshold classification problems $Y = \mathbf{1}\{X \geq t\}$ with random label flipping.

### Loss & Training

- The squared loss $\mathrm{er}(h) = \mathbb{E}[(h(X)-Y)^2]$ is used as the classifier loss (asymmetric losses are also supported).
- The key assumption (Assumption 1) is that the joint distribution $(X, Y)$ and hypothesis class $\mathcal{C}$ admit exponential decay of classification excess risk under active learning.
- Corollaries 1–4 show how different subalgorithms $\mathcal{S}$ apply to different problem settings (binary/multiclass, hard-margin/weak noise conditions, etc.).
- Corollary 4 proposes a "heterogeneity-aware" $\mathcal{S}$ that further subdivides high- and low-variance strata by separating the abstention region from the certain region.

## Key Experimental Results

### Main Results

| Setting | Method | 90% Error Bound at $N=50$ | 90% Error Bound at $N=100$ | Trend |
|---------|--------|--------------------------|---------------------------|-------|
| $\nu=0$ (perfectly separable) | PartiBandits | ~0.002 | ~0.0005 | Far superior to SRS |
| $\nu=0$ (perfectly separable) | SRS | ~0.020 | ~0.010 | Standard rate |
| $\nu=0.05$ (5% noise) | PartiBandits | ~0.015 | ~0.004 | Better than SRS |
| $\nu=0.10$ (10% noise) | PartiBandits | ~0.020 | ~0.008 | Slightly better than SRS |

### Ablation Study (WarmStart-UCB)

| Stratification $\Sigma_1(\mathcal{G})$ | $N=100$ | $N=200$ | Note |
|----------------------------------------|---------|---------|------|
| Optimal stratification (lowest $\Sigma_1$) | ~0.003 | ~0.001 | Fastest convergence |
| Suboptimal stratification (moderate $\Sigma_1$) | ~0.006 | ~0.003 | Moderate improvement |
| Poor stratification (high $\Sigma_1$) | ~0.009 | ~0.005 | Still better than SRS |
| SRS ($\Sigma_1 = \mathrm{Var}(Y)$) | ~0.012 | ~0.006 | Slowest |

### Key Findings

- Simulations on real electronic health records (AFC dataset, 6 million records) confirm theoretical predictions: PartiBandits begins to outperform SRS at a label budget of approximately 30–50, with the gap widening substantially as the budget increases.
- Even when $X$ is not highly predictive across the entire population, PartiBandits achieves label savings in tail subregions where predictive power exists.

## Highlights & Insights

- **Bridging two active learning paradigms**: The paper elegantly combines disagreement-based active learning (for classification) with UCB-style adaptive sampling (for mean estimation), demonstrating that exponential label savings in classification can be transferred to mean estimation tasks.
- **Eliminating dependence on $\sigma_{\min}$**: The warm start mechanism resolves an open problem in the stratified active learning literature.
- **Theoretical rigor**: The minimax optimal results with matching upper and lower bounds are particularly elegant.
- **Practical utility**: An R package implementation is provided, with simulations demonstrating performance on real data.

## Limitations & Future Work

- Satisfaction of Assumption 1 (the exponential savings condition) depends on the choice of hypothesis class $\mathcal{C}$ and the data distribution, requiring prior knowledge.
- Using half the label budget for Phase 1 may be wasteful when labels are extremely scarce; an adaptive phase division could be more efficient.
- Experiments are primarily validated on simple threshold models and binary classification; performance on high-dimensional or complex data remains to be examined.
- When $\nu$ is close to $\mathrm{Var}(Y)$ (i.e., $X$ provides no predictive power for $Y$), PartiBandits degenerates to SRS with no additional benefit.

## Related Work & Insights

- Directly extends the multi-group mean estimation framework of Aznag et al. (2023), improving dependence on $\sigma_{\min}$ and the number of strata $G$.
- Deeply connected to the active learning theory for classification in Hanneke (2011) and Puchkin & Zhivotovskiy (2022).
- Suggests a broader principle: label efficiency results from classification can be transferred to estimation problems through stratification mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to transfer exponential label savings from classification active learning to mean estimation, bridging two major paradigms
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both simulated and real data, though experimental settings are relatively simple
- Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous and clear, though heavy notation presents a high entry barrier
- Value: ⭐⭐⭐⭐ Solid theoretical contributions with practical relevance to survey sampling and healthcare equity

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts](on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)
- [\[ICML 2025\] A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation](../../ICML2025/optimization/a_unified_view_on_learning_unnormalized_distributions_via_noise-contrastive_esti.md)
- [\[ICML 2025\] FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation](../../ICML2025/optimization/fsl-sage_accelerating_federated_split_learning_via_smashed_activation_gradient_e.md)

</div>

<!-- RELATED:END -->
