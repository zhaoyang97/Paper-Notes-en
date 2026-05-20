---
title: >-
  [Paper Note] Online Linear Regression with Paid Stochastic Features
description: >-
  [AAAI2026][online linear regression] This paper studies a novel setting in online linear regression where features are corrupted by noise and the learner can **pay to reduce noise intensity**. It establishes that the opt…
tags:
  - "AAAI2026"
  - "online linear regression"
  - "noisy features"
  - "paid features"
  - "regret bounds"
  - "covariance estimation"
  - "matrix martingale concentration"
date: 2026-05-08
content_hash: e78b8aac67704093
---

# Online Linear Regression with Paid Stochastic Features

**Conference**: AAAI2026
**arXiv**: [2511.08073](https://arxiv.org/abs/2511.08073)  
**Code**: None  
**Area**: Online Learning / Linear Regression
**Keywords**: online linear regression, noisy features, paid features, regret bounds, covariance estimation, matrix martingale concentration

## TL;DR

This paper studies a novel setting in online linear regression where features are corrupted by noise and the learner can **pay to reduce noise intensity**. It establishes that the optimal regret rate is $\widetilde{\mathcal{O}}(\sqrt{T})$ when the noise covariance is known and $\widetilde{\mathcal{O}}(T^{2/3})$ when unknown, with matching lower bounds; all bounds are order-optimal in $T$.

## Background & Motivation

### Noisy Features in Online Learning

In classical online linear regression, the learner sequentially observes i.i.d. feature vectors $x_t \in \mathbb{R}^d$ and predicts output $y_t = x_t^\top \theta^*$. Most prior work assumes perfect observation of $x_t$, yet in practice only a noisy version is accessible. Examples include:

- **Experimental measurement error**: Features obtained from physical experiments inevitably carry measurement noise.
- **User behavior sampling**: A user's contextual information depends only on random samples of their behavior.
- **Privacy-preserving mechanisms**: Features are deliberately perturbed by differential privacy mechanisms (e.g., Laplace/Gaussian noise).

### The Paid Noise-Reduction Scenario

In many settings, the learner can **allocate more resources** to reduce noise intensity:

**Equipment upgrades**: Renting more precise instruments or hiring more experienced personnel.

**Extended measurement time**: Averaging multiple raw samples over a longer time window.

**Privacy compensation**: Offering paid services to users in exchange for lower privacy protection levels (i.e., less noise).

**Data collection**: Gathering more statistical information before making decisions.

Reducing noise improves prediction but incurs costs (time, money, or service rewards). The learner must therefore **optimally balance prediction error against payment cost**.

### Core Problem Formulation

The learner observes a noisy feature $\hat{x}_t(c_t) = x_t + n_t(c_t)$, where $c_t \in [0,1]$ is the payment chosen at round $t$, and $n_t(c)$ is zero-mean noise whose covariance matrix $\Sigma_n(c)$ decreases as payment increases (higher payment → smaller noise). The learner's goal is to minimize **regret**:

$$\text{Reg}(T) = \mathbb{E}\left[\sum_{t=1}^T \left((\hat{y}_t(\hat{x}_t) - y_t)^2 + \lambda c_t\right)\right] - T \cdot \ell^*$$

where $\ell^*$ is the expected loss of the optimal linear predictor at the optimal payment level.

## Method

### Overall Architecture

The paper treats two settings separately:

1. **Known noise covariance** (Section 4): The learner knows the noise covariance $\Sigma_n(c)$ for all payment levels $c$.
2. **Unknown noise covariance** (Section 5): The learner does not know how payment affects noise level.

The two settings admit fundamentally different optimal regret rates: $\widetilde{\mathcal{O}}(\sqrt{T})$ in the known case and $\widetilde{\mathcal{O}}(T^{2/3})$ in the unknown case.

### Key Design 1: Covariance-Corrected Information Sharing (Known Setting)

The central insight in the known setting is that, since the noise covariance is known, samples collected at different payment levels can **share information**. Specifically, the only term in the loss $\ell(c, \nu)$ that depends on the noise covariance is $\nu^\top \Sigma_{\hat{x}}(c) \nu$. The paper constructs a **covariance-corrected estimator**:

$$\hat{\Sigma}_{\hat{x}}(c) = \sum_{s=1}^t \left(\hat{x}_s(c_s) \hat{x}_s(c_s)^\top + \Sigma_n(c) - \Sigma_n(c_s)\right)$$

This estimator utilizes all historical samples regardless of the payment level at which they were observed, by adding a correction term $\Sigma_n(c) - \Sigma_n(c_s)$ to align observations from different payment levels to the target level $c$. Its expectation equals $t \cdot \Sigma_{\hat{x}}(c)$, making it unbiased.

Based on this corrected estimator, the empirical loss is defined as:

$$\hat{L}_t^{kc}(c, \nu) = \sum_{s=1}^t \left((\hat{x}_s^\top \nu - y_s)^2 + \nu^\top(\Sigma_n(c) - \Sigma_n(c_s))\nu + \lambda c\right)$$

The key advantage of information sharing is that **no exploration of different payment levels is needed** — the estimator provides valid estimates for all payment levels under any historical payment sequence, enabling greedy selection of the optimal payment.

### Key Design 2: Optimistic Exploration Strategy (Unknown Setting)

Without knowledge of the noise covariance, cross-payment-level information sharing is infeasible, and only samples collected **at the same payment level** can be used to estimate the loss at that level. This gives rise to the classic **exploration–exploitation dilemma**:

- **Exploration need**: Sufficient samples must be collected at different payment levels to identify the optimal payment $c^*$.
- **Exploitation need**: The learner should operate as much as possible at the currently estimated optimal payment level.

The paper adopts a **UCB-style optimistic strategy**:

1. Discretize the continuous payment space into a grid $\{k/K\}_{k \in [K]}$.
2. Maintain empirical losses $\hat{L}_t^{uc}(k/K, \nu)$ for each grid point.
3. Select the payment level that minimizes the **optimistic loss** (empirical average minus a confidence width):

$$k_{t-1} \in \arg\min_{k \in [K]} \left\{\frac{\hat{L}_{t-1}^{uc}(k/K, \hat{\nu}_{t-1}(k))}{N_{t-1}(k/K)} - 9S^2 \sqrt{\frac{8\beta_t^2 \ln \frac{3dt(t+1)}{\delta}}{N_{t-1}(k/K)}}\right\}$$

where $N_{t-1}(k/K)$ denotes the number of times that payment level has been selected. Under-observed payment levels receive larger confidence bonuses, encouraging exploration.

### Theoretical Analysis

#### Theorem 1 (Upper Bound, Known Covariance)

Using matrix martingale concentration inequalities, the paper proves **uniform convergence** of the empirical loss over all payment levels and linear predictors (Proposition 2), and derives the regret bound for Algorithm 1:

$$\text{Reg}(T) = \widetilde{\mathcal{O}}\left(S^2(R^2 d + S^2)\sqrt{T} + \lambda\right)$$

#### Theorem 2 (Lower Bound, Known Covariance)

A one-dimensional counterexample is constructed: the noise variance is fixed as $\mathcal{N}(0,1)$ for $c < 1/2$ and zero for $c \geq 1/2$. Small differences in feature variance ($1 \pm \epsilon$) cause the optimal payment to alternate between $c=0$ and $c=1/2$, reducing the problem to a two-armed bandit and establishing an $\Omega(\sqrt{T})$ lower bound.

#### Theorem 3 (Upper Bound, Unknown Covariance)

Via grid discretization and UCB exploration, the paper proves the regret bound for Algorithm 2:

$$\text{Reg}(T) = \widetilde{\mathcal{O}}\left((S^2(R^2 d + S^2))^{2/3} \lambda^{1/3} T^{2/3}\right)$$

Setting the grid size to $K = \lceil T^{1/3} \lambda^{2/3} / (S^2(R^2 d + S^2))^{2/3} \rceil$ optimally balances discretization error and exploration cost.

#### Theorem 4 (Lower Bound, Unknown Covariance)

The paper carefully constructs $K$ noise variance curves, each deviating from the baseline function $f(c) = \frac{1-c}{1+c}$ (which makes the optimal loss identical across all payment levels) on a disjoint small interval of width $\approx 1/K$. Using the pigeonhole principle and information-theoretic tools, an $\Omega(T^{2/3})$ lower bound is established; the technique is analogous to classical lower bound constructions for Lipschitz bandits.

## Key Experimental Results

This paper is a purely theoretical contribution; the experimental section primarily consists of comparisons of theoretical bounds.

### Table 1: Summary of Regret Rates

| Setting | Upper Bound | Lower Bound | Order-Optimal in $T$? |
|---------|-------------|-------------|------------------------|
| Known noise covariance | $\widetilde{\mathcal{O}}(\sqrt{T})$ | $\Omega(\sqrt{T})$ | ✅ Order-optimal (up to log factors) |
| Unknown noise covariance | $\widetilde{\mathcal{O}}(T^{2/3})$ | $\Omega(T^{2/3})$ | ✅ Order-optimal (up to log factors) |
| Noiseless classical setting | $\Theta(\ln T)$ | $\Theta(\ln T)$ | ✅ Exact |

### Table 2: Key Comparison of Algorithms Across Two Settings

| Property | Algorithm 1 (Known) | Algorithm 2 (Unknown) |
|----------|----------------------|------------------------|
| Information sharing | ✅ Across payment levels | ❌ Within same payment level only |
| Exploration mechanism | Not needed (greedy suffices) | UCB-style optimistic exploration |
| Payment grid size $K$ | $\lceil \lambda T \rceil$ | $\lceil T^{1/3} \lambda^{2/3} / (\cdot)^{2/3} \rceil$ |
| Regularization | Required (ensure convexity) | Not required (naturally convex) |
| Regret rate | $\widetilde{\mathcal{O}}(\sqrt{T})$ | $\widetilde{\mathcal{O}}(T^{2/3})$ |
| Per-step computation | $K$ convex optimization problems | $K$ convex optimization problems |

## Highlights & Insights

1. **Covariance correction is the central technical contribution**: By using the known noise covariance difference $\Sigma_n(c) - \Sigma_n(c_s)$ to correct empirical covariance estimates, observations at different payment levels can be unified. This concise algebraic trick simultaneously eliminates the need for exploration and preserves estimator unbiasedness.

2. **The gap between $\sqrt{T}$ and $\ln T$**: Classical noiseless online linear regression achieves optimal regret $\Theta(\ln T)$, yet even when the noise covariance is fully known, the paid-features problem degrades exponentially to $\Theta(\sqrt{T})$. The lower bound construction reveals the root cause: discontinuities in the noise variance make payment selection inherently a bandit problem.

3. **Elegant exploitation of one-sided Lipschitz structure**: The paper proves that the optimal loss $\ell^*(c)$ is $\lambda$-one-sided Lipschitz — increasing payment can never substantially increase the loss (since higher payment always reduces prediction error). This property makes the discretization error controllable without imposing a Lipschitz assumption on the noise covariance.

4. **Avoiding the curse of dimensionality from Lipschitz bandits**: A direct reduction to Lipschitz bandits would yield a regret rate of $\widetilde{\mathcal{O}}(T^{(d+2)/(d+3)})$ in $(d+1)$-dimensional space. By decoupling linear predictor optimization from payment selection and discretizing only the one-dimensional payment space, the paper maintains a $T^{2/3}$ rate that is dimension-independent.

5. **Non-trivial application of matrix martingale concentration**: The paper extends Tropp's (2012) matrix martingale concentration inequality to a heteroscedastic setting where noise covariances vary across time steps, representing a meaningful contribution at the level of theoretical tools.

## Limitations & Future Work

1. **Tightness of dimension dependence unresolved**: All lower bound constructions are based on one-dimensional instances; the optimal dimensional dependence in $d$ dimensions (i.e., whether the factor $d$ in the upper bound is improvable) remains an open problem.

2. **Computational efficiency is suboptimal**: Each time step requires solving $K$ convex optimization problems, where $K$ can be $\Theta(T)$ in the known setting, yielding total complexity $O(T^2 \cdot \text{poly}(d))$, which is heavy for online scenarios. The paper notes that adaptive grids or continuous payment optimization are worth exploring.

3. **Restricted to linear predictors**: The optimal nonlinear predictor requires computing $\mathbb{E}[x_t | \hat{x}_t]^\top \theta^*$, which depends on full distributional information and is generally computationally intractable. Comparing against a linear baseline is reasonable but limits the applicability of the results.

4. **No empirical experiments**: As a purely theoretical paper, no numerical simulations are provided to demonstrate finite-sample performance, nor any empirical comparison with existing methods.

5. **Strong noise model assumptions**: The noise is required to be zero-mean, independent, and sub-Gaussian, with the payment–noise mapping satisfying monotonicity. In some applications, noise may be correlated with features, or the relationship between payment and noise may be more complex.

## Related Work & Insights

- **Errors-in-Variables Models**: The classical statistical treatment of noisy features in regression, focusing primarily on estimation of $\theta^*$ rather than online prediction, and typically assuming high-dimensional sparsity.
- **Cesa-Bianchi et al. (2011)**: The most directly related prior work, studying online linear regression with noisy features but without any payment/cost notion; the regret benchmark is the optimal predictor under noiseless features.
- **van den Heuvel & Kamvar (2023)**: Studies paid noise reduction in a binary classification setting and establishes a regret bound of $d^2 (\ln T) \sqrt{T}$, but is limited to independent binary feature noise; the technique does not extend to regression with correlated noise.
- **Lipschitz Bandits (Kleinberg 2004; Bubeck et al. 2011)**: The one-dimensional problem also achieves $\widetilde{\mathcal{O}}(T^{2/3})$; the lower bound construction in this paper is heavily inspired by this line of work, though the problem structure is fundamentally different (non-Lipschitz, with the linear predictor component being exactly optimizable).
- **Privacy and Incentive Mechanism Design (Ghosh et al. 2011; Hsu et al. 2014)**: Studies how to design payment mechanisms to incentivize users to sell private data, but focuses on optimizing estimation accuracy rather than regret minimization in online prediction.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The paper introduces a novel and practically motivated online learning problem — paying to reduce feature noise. The theoretical results are clean: order-optimal upper and lower bounds in $T$ are established for both settings, and technical contributions such as covariance correction and one-sided Lipschitz properties have independent value. The main weaknesses are the absence of empirical validation, the unresolved optimality of dimensional dependence, and computational efficiency concerns. Overall, this is a solid theoretical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination](../../NeurIPS2025/others/information-computation_tradeoffs_for_noiseless_linear_regression_with_oblivious.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/others/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[AAAI 2026\] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes](detonation_decoupled_torch_network-aware_training_on_interlinked_online_nodes.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](../../ICLR2026/others/lipschitz_bandits_with_stochastic_delayed_feedback.md)

</div>

<!-- RELATED:END -->
