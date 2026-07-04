---
title: >-
  [Paper Note] Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning
description: >-
  [ICML2025 Spotlight][Reinforcement Learning][off-policy learning] A novel non-linear estimator based on the log-sum-exponential (LSE) operator is proposed for off-policy evaluation and learning, which significantly reduces variance and provides theoretical guarantees under heavy-tailed rewards and noisy propensity score scenarios.
tags:
  - "ICML2025 Spotlight"
  - "Reinforcement Learning"
  - "off-policy learning"
  - "off-policy evaluation"
  - "inverse propensity score"
  - "heavy-tailed reward"
  - "variance reduction"
  - "log-sum-exponential"
date: 2026-05-08
content_hash: 4adcd01c347ee1a8
---

# Log-Sum-Exponential Estimator for Off-Policy Evaluation and Learning

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.06873](https://arxiv.org/abs/2506.06873)  
**Code**: [GitHub](https://github.com/armin-behnamnia/lse-offpolicy-learning)  
**Area**: Reinforcement Learning  
**Keywords**: off-policy learning, off-policy evaluation, inverse propensity score, heavy-tailed reward, variance reduction, log-sum-exponential

## TL;DR
A novel non-linear estimator based on the log-sum-exponential (LSE) operator is proposed for off-policy evaluation and learning, which significantly reduces variance and provides theoretical guarantees under heavy-tailed rewards and noisy propensity score scenarios.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**：**Background**：**Background**：Off-policy learning and evaluation (OPL/OPE) utilize existing logged bandit feedback datasets to conduct policy evaluation and learning. This is widely applied in fields such as recommendation systems, personalized medicine, and advertising. The standard method is the inverse propensity score (IPS) based estimator.

**Limitations**:

**High Variance Issue**: IPS estimators suffer from extremely high variance when policy distributions mismatch excessively, leading to unstable evaluations.

**Heavy-Tailed Rewards**: In scenarios like financial markets and online advertising, reward distributions are often heavy-tailed, where variance might even be undefined. Existing estimators (such as PM, ES, and IX) all assume bounded rewards and fail to handle this.

**Noisy Propensity Scores**: In practical scenarios, estimating propensity scores is often required instead of using ground truth, and estimation errors further deteriorate performance.

**Core Idea**: Leverage the inherent robustness of the log-sum-exponential operator—where for $\lambda < 0$, exceptionally large values $z_i$ are automatically suppressed by $e^{\lambda z_i} \to 0$—to construct a novel non-linear estimator that maintains low variance in both heavy-tailed and noisy scenarios.

## Method

### Core Framework: LSE Estimator

Given a logged bandit feedback dataset $S = (x_i, a_i, p_i, r_i)_{i=1}^n$, the LSE estimator is defined as:

$$\hat{V}_{\text{LSE}}^{\lambda}(S, \pi_\theta) = \frac{1}{\lambda} \log\left(\frac{1}{n} \sum_{i=1}^{n} e^{\lambda r_i w_\theta(a_i, x_i)}\right)$$

where $\lambda < 0$ is a tunable parameter, and $w_\theta(a,x) = \pi_\theta(a|x) / \pi_0(a|x)$ is the importance weight.

**Key Properties**:
- As $\lambda \to 0$, it degenerates to the standard IPS estimator.
- For $\lambda < 0$, exceptionally large weighted rewards $r_i w_\theta$ are naturally suppressed through the exponential operation.
- LSE is a monotonically increasing function with respect to $\lambda$.
- Distinct from all existing linear estimators, LSE is a non-linear function of the overall weighted reward samples.

### Heavy-Tailed Assumption

Assume that the $(1+\epsilon)$-th moment of the weighted reward is bounded ($\epsilon \in [0,1]$):

$$\mathbb{E}\left[(w_\theta(A,X) R)^{1+\epsilon}\right] \leq \nu$$

This assumption is much weaker than the traditional bounded reward assumption, allowing for unbounded rewards.

### Theoretical Results

**Regret Upper Bound (Theorem 5.3)**: Under a finite policy set $|\Pi_\theta| < \infty$, with probability $\geq 1-\delta$, the regret is bounded by a function of $\lambda$, $\nu$, and $n$.

**Convergence Rate (Proposition 5.4)**: Choosing $\lambda = -n^{-1/(1+\epsilon)}$ yields a regret upper bound convergence rate of $O(n^{-\epsilon/(1+\epsilon)})$. When $\epsilon=1$ (bounded second moment), it reaches $O(n^{-1/2})$

**Bias Bound (Proposition 5.5)**: The upper bound of the bias is $\frac{|\lambda|^\epsilon}{1+\epsilon}\nu + O(1/(n\lambda))$. Selecting $\lambda(n) = -n^{-\varsigma}$ makes the bias asymptotically zero (asymptotically unbiased).

**Variance Bound (Proposition 5.7)**: $\mathbb{V}(\hat{V}_{\text{LSE}}^{\lambda}) \leq \frac{1}{n}\mathbb{V}(w_\theta R) \leq \frac{\nu_2}{n}$, which does not exceed the IPS variance for all $\lambda < 0$.

**Noise Robustness (Theorem 5.9)**: Under a reward distribution shift $\tilde{P}_{R|X,A}$, the additional regret term is proportional to $\text{TV}(P_{R|X,A}, \tilde{P}_{R|X,A}) / \lambda^2$. Increasing $|\lambda|$ reduces the noise penalty but increases the bias.

## Key Experimental Results

### Main Results: EMNIST Classification

| Scenario | LSE | PM | ES | IX | BanditNet | LS-LIN | OS |
|------|-----|----|----|----|-----------|---------|----|
| τ=1, True PS | **89.29** | 89.08 | 88.45 | 88.14 | 59.90 | 88.30 | 88.74 |
| τ=1, Noisy PS (b=0.01) | **86.07** | 85.62 | 85.71 | 81.39 | 66.55 | 84.64 | 84.59 |
| τ=10, Noisy PS (b=0.01) | **82.15** | 80.85 | 81.07 | 77.49 | 27.02 | 78.43 | 21.70 |
| τ=10, Noisy Reward (Pf=0.1) | **88.29** | 88.22 | 88.19 | 87.93 | 84.89 | 87.50 | 87.68 |

### Key Findings

- **Noisy Propensity Scores**: LSE shows a significant advantage under low-quality propensity scores (b=0.01), leading PM by approximately 1.3% and OS by over 60% when τ=10.
- **Noisy Rewards**: LSE consistently leads at Pf=0.1; it remains competitive under high noise at Pf=0.5.
- **Variance Stability**: LSE generally achieves the lowest standard deviation, demonstrating the variance reduction effect theoretically predicted.
- **Pareto Distribution Experiment** (Table 1): LSE achieves a mean squared error (MSE) of only 0.13 at n=10 (compared to 1.54 for Monte-Carlo), yielding a 15-fold reduction in variance.

## Highlights & Insights

1. **Elegant Design Intuition**: Utilizes $e^{\lambda z} \to 0$ ($\lambda<0, z\to\infty$) to naturally suppress outliers without requiring explicit truncation.
2. **Theoretical Comprehensiveness**: Simultaneously covers both OPE and OPL scenarios, providing a complete theoretical framework for regret, bias, variance, and robustness.
3. **Weaker Assumptions**: Only requires the $(1+\epsilon)$-th moment of weighted rewards to be bounded, allowing for unbounded rewards and heavy-tailed distributions.
4. **Differentiability**: LSE is differentiable with respect to policy parameters, avoiding the optimization difficulties encountered with truncated IPS.
5. **Sub-Gaussian Tail**: Concentration of LSE exhibits sub-Gaussian-type tail behavior.

## Limitations & Future Work

1. **Finite Policy Set Assumption**: The main theorems assume $|\Pi_\theta| < \infty$, requiring VC dimension or PAC-Bayes extensions to generalize to continuous policy spaces.
2. **Parameter $\lambda$ Selection**: The theoretical optimal choice $\lambda = -n^{-1/(1+\epsilon)}$ requires knowledge of $\epsilon$; in practice, cross-validation is needed.
3. **Bias Penalty**: The bias introduced by LSE is non-negligible in small-sample regimes, requiring careful tuning of the bias-variance trade-off.
4. **Model-Free Limitation**: Only the model-free setting is considered; integrating with model-based methods (e.g., doubly robust) is worth exploring (preliminary discussion in Appendix G.3).
5. **Experimental Scale**: Validation is primarily conducted on medium-scale datasets such as EMNIST; applicability to large-scale scenarios (e.g., large-scale recommendation systems) remains to be verified.

## Related Work & Insights

- **IPS Lineage**: IPS → SN-IPS → Truncated IPS → PM → ES → IX → LS; LSE pioneers a non-linear direction within this lineage.
- **Tilted ERM (Li et al., 2023)**: Connection between LSE and tilted empirical risk, drawing robustness analysis inspiration from supervised learning.
- **Heavy-tailed Bandits (Bubeck et al., 2013)**: Source of inspiration for the finite moment assumption.
- **PAC-Bayes Extension** (Appendix D.6): Connection to the PAC-Bayesian analysis framework.

## Rating
- Novelty: ⭐⭐⭐⭐ (The application of the LSE operator to off-policy paradigms is novel, and the non-linear estimator direction is unique)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers multiple scenarios with sufficient ablations, though the scale is relatively small)
- Writing Quality: ⭐⭐⭐⭐ (The paper tightly couples theory and experiments, with a clear system of notations)
- Value: ⭐⭐⭐⭐ (Provides a practical and theoretically guaranteed new tool for offline reinforcement learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](../../ICLR2026/reinforcement_learning/a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](../../NeurIPS2025/reinforcement_learning/bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](../../NeurIPS2025/reinforcement_learning/succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)

</div>

<!-- RELATED:END -->
