---
title: >-
  [Paper Note] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization
description: >-
  [ICML 2026][Optimization][Momentum methods] This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time, Momentum SGD is systematically inferior to vanilla SGD due to "inertial lag," with performance degradation amplified by a factor of the order $(1 - \beta)^{-2}$. Through information-theoretic lower bounds, it demonstrates that this cost is a fundamental obstacle rather than an analytical artifact.
tags:
  - "ICML 2026"
  - "Optimization"
  - "Momentum methods"
  - "nonstationary optimization"
  - "tracking error"
  - "distribution shift"
  - "information-theoretic lower bounds"
date: 2026-05-08
content_hash: 6fc49fb8f82b555b
---

# On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization

**Conference**: ICML 2026  
**arXiv**: [2601.12238](https://arxiv.org/abs/2601.12238)  
**Code**: To be confirmed  
**Area**: Optimization Theory  
**Keywords**: Momentum methods, nonstationary optimization, tracking error, distribution shift, information-theoretic lower bounds

## TL;DR
This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time, Momentum SGD is systematically inferior to vanilla SGD due to "inertial lag," with performance degradation amplified by a factor of the order $(1 - \beta)^{-2}$. Through information-theoretic lower bounds, it demonstrates that this cost is a fundamental obstacle rather than an analytical artifact.

## Background & Motivation

**Background**: Momentum methods (Heavy-Ball, Nesterov) have been proven to accelerate convergence and reduce gradient noise in static convex optimization and are the default configuration in deep learning. However, in nonstationary environments such as online learning, federated learning, and reinforcement learning, the optimum $\theta_t^*$ moves continuously with distribution shifts, rendering past gradients stale.

**Limitations of Prior Work**: Empirically, momentum often exhibits instability and poorer tracking performance in dynamic environments, but rigorous theoretical explanations are lacking—existing dynamic regret analyses (Zhang 2015; Hardt 2016) only provide general path-length bounds without explicitly characterizing the relationship between the momentum parameter $\beta$ and performance degradation. No information-theoretic lower bounds exist to show whether this is an inherent cost of momentum or a result of loose analysis.

**Key Challenge**: Momentum pushes in opposite directions: (1) reducing variance by averaging historical gradients in static noisy scenarios; (2) generating inertial lag by averaging "stale gradients" under distribution shifts, causing the algorithm to systematically lag behind the moving target.

**Goal**: To quantitatively characterize the performance difference between Momentum SGD and vanilla SGD in nonstationary strongly convex smooth optimization, providing clear boundaries for when momentum helps versus when it hurts.

**Key Insight**: Treating SGDM as a 2D dynamical system composed of "parameters + momentum buffer" and using Lyapunov functions for stability analysis to explicitly reveal amplification factors from $(1 - \beta)^{-1}$ to $(1 - \beta)^{-2}$. Assouad-style constructions under variation budgets are then used to prove these factors are information-theoretically inevitable.

**Core Idea**: Tracking error can be decomposed into three terms: "initialization forgetting + noise floor + drift-induced lag." Momentum amplifies each term by an order of $(1 - \beta)^{-k}$, consistent with tight lower bounds.

## Method

### Overall Architecture
Consider a time-varying strongly convex smooth problem $G_t(\theta) = \mathbb{E}_{X_t \sim \Pi_t}[g(\theta, X_t)]$, where the optimum $\theta_t^*$ drifts over time. The goal is to track $\theta_t^*$ rather than converge to a single point.

SGD: $\theta_{t+1} = \theta_t - \gamma_t \nabla g(\theta_t, X_{t+1})$.

Generalized SGDM: $\psi_t = \theta_t + \beta_1 (\theta_t - \theta_{t-1})$, $\theta_{t+1} = \psi_t - \gamma_t \nabla g(\psi_t, X_{t+1}) + \beta_2 (\psi_t - \psi_{t-1})$; Heavy-Ball takes $\beta_1 = 0, \beta_2 = \beta$, while Nesterov takes $\beta_1 = \beta, \beta_2 = 0$.

### Key Designs

**1. 2D Lyapunov + Three-term Tracking Error Decomposition (Upper Bound): Explicitly revealing momentum amplification factors**

To characterize when momentum helps or hurts, explicit tracking error upper bounds for SGD and SGDM must be established. For SGD:

$$\mathbb{E}\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma\mu/2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\Delta^2}{\gamma^2 \mu^2} + \frac{\sigma^2 \gamma}{\mu}$$

The three terms represent initialization forgetting, drift lag, and noise floor. For SGDM, the recurrence of parameters $\theta_t$ and the momentum buffer are coupled. Decomposing this into a 1D recurrence loses coupling information; this work unifies them into a 2D Lyapunov function for joint tracking. Consequently, each of the three terms is multiplied by an amplification factor of the order $(1-\beta)^{-2}$—a key technique to explicitly derive this factor. The conclusion is direct: while momentum reduces variance by averaging historical gradients in static noisy settings, it averages "stale gradients" under distribution shift, causing systematic lag behind the moving target.

**2. Time-Resolved High-Probability Bounds + Weighted Historical Drift: Removing the need for uniform drift upper bounds**

Uniform drift upper bounds ($\Delta$) assume drift is constant, but real-world drift is often intermittent or localized (e.g., seasonality or abrupt shifts). This step replaces MGF recurrences with optional stopping arguments for martingale differences, yielding a bound at any time $t$ with probability $1-\delta$:

$$\| \theta_t - \theta_t^* \|^2 \lesssim (1 - \gamma\mu/2)^t \| \theta_0 - \theta_0^* \|^2 + \frac{\mathfrak{D}_t}{\gamma\mu} + O(d\sigma^2\gamma/\mu)$$

where $\mathfrak{D}_t = \sum_{\ell=0}^{t-1}(1-\gamma\mu/2)^{t-\ell-1}\|\Delta_\ell\|^2$ is the weighted historical drift rather than a fixed upper bound. It adaptively captures the locality of drift—recent drifts carry high weight while distant drifts decay—directly inspiring restart and windowing strategies.

**3. Information-Theoretic Lower Bounds + Inertial Window: Proving momentum degradation is inherent**

To determine if the degradation is an inherent cost or due to loose analysis, the paper constructs worst-case drift sequences under variation budget constraints $\mathrm{GVar}_{p,q}(g)\leq\mathbb{V}_T$. It provides dynamic regret lower bounds for SGDM: $\mathfrak{M}_T(\Pi_\beta,\mathbb{V}_T)\gtrsim\max\{(1-\beta)^{-2/(\alpha q+2)}\cdot\mathbb{V}_T^{2q/(\alpha q+2)}T^{\alpha q/(\alpha q+2)},\ldots\}$, which explicitly includes factors from $(1-\beta)^{-1}$ to $(1-\beta)^{-2}$, matching the upper bounds. "Block drift" constructions further prove that any SGDM must endure an "inertial window" of $\Omega(\kappa/(1-\beta))$ steps for transient adjustment after a distribution change. Tight upper and lower bounds together show that "inertial lag" is an unavoidable fate for momentum in nonstationary settings.

### Loss & Training
- Constant Step Size: $\gamma^* = \arg\min_\gamma \left[ \frac{192 (2 + \beta)^2}{\mu^2 \gamma^2} \Delta^2 + \frac{96}{\mu (1 - \beta)} \sigma^2 \gamma \right]$.
- Epoch Decay + Momentum Restart: Increasing step sizes over logarithmic time intervals and resetting the momentum buffer to 0 at period boundaries to break the accumulation of stale gradients.

## Key Experimental Results

### Main Results: Strongly Convex Quadratic Target + Random Walk Drift

| Setting | SGD ($\gamma = 0.01$) | HB | NAG | Conclusion |
|------|-------------|-----|-----|------|
| $\beta = 0.50, \sigma^2 = 0.1$ | 1.036 | 0.342 | 0.349 | Moderate momentum helps |
| $\beta = 0.50, \sigma^2 = 0.8$ | 1.305 | 0.961 | 1.019 | Momentum beneficial under high noise |
| $\beta = 0.90, \sigma^2 = 0.1$ | 1.029 | 0.497 | 0.453 | Momentum helps with slight drift + low noise |
| $\beta = 0.90, \sigma^2 = 0.8$ | 1.466 | **3.899** | **3.721** | Momentum worsens with slight drift + high noise |
| $\beta = 0.99, \sigma^2 = 0.8$ | 1.403 | **38.802** | **21.038** | Momentum collapses with strong momentum + high noise |

Tracking error after 5000 steps. As $\beta$ increases from 0.50 to 0.99, HB/NAG degrades sharply, while SGD remains relatively robust.

### Ablation Study: Interaction between Condition Number and Drift Magnitude

| Dataset | Condition Number $\kappa$ | SGD | HB ($\beta = 0.9$) | NAG ($\beta = 0.9$) | HB/SGD Gain |
|--------|---------------|-----|-------------------|-------------------|--------|
| Linear Regression | 10 | 0.31 | 2.47 | 1.73 | 7.97× |
| Linear Regression | 1000 | 1.28 | 12.30 | 9.19 | 9.61× |
| Logistic Regression | 10 | 0.42 | 3.56 | 2.18 | 8.48× |
| Teacher-Student MLP | — | 0.58 | 5.23 | 3.27 | 9.02× |

### Key Findings
- The larger the condition number $\kappa$, the more pronounced the harm of momentum—ill-conditioned problems require smaller step sizes $\gamma \lesssim (1 - \beta)^2 / L$ for stability, further slowing convergence.
- Increasing drift magnitude $\delta_{\text{rw}}$ rapidly widens the gap between HB/NAG and SGD.
- High noise $\sigma^2 = 0.8$ combined with moderate momentum $\beta = 0.9$ represents the most vulnerable region for momentum, where inertial lag and noise amplification overlap.

## Highlights & Insights
- **2D Lyapunov Dynamical System Perspective**: Analyzing the two coupled recurrences (parameters and momentum) together is key to making the $(1 - \beta)^{-2}$ factor explicit; this can be generalized to other optimization algorithms with auxiliary variables.
- **Fundamental Nature of $(1 - \beta)^{-2}$**: Proving that this is an information-theoretic necessity rather than an analytical looseness via tightly matched upper and lower bounds.
- **Time-Resolved Boundaries**: Replacing uniform drift upper bounds $\Delta$ with weighted history $\mathfrak{D}_t$ allows for adaptation to intermittent drift, directly suggesting the "gradient-momentum alignment" $S_t = 1 - \frac{\langle \nabla g, v \rangle}{\|\nabla g\| \|v\|}$ as a change detection signal.
- **Drift-Noise Tradeoff Visualization**: Clearly demonstrating that momentum simultaneously amplifies initialization sensitivity, noise floor, and drift lag, resulting in a narrow tradeoff space.

## Limitations & Future Work
- Restricted by the strong convexity assumption; non-convex scenarios (e.g., PŁ conditions) are analogous but results are not provided.
- Stability condition $\gamma \leq \mu (1 - \beta)^2 / (4 L^2)$ is somewhat conservative; qualitative conclusions are robust, but quantitative predictions require finer analysis.
- Assumes the optimum $\theta_t^*$ is measurable; lacks analysis for stochastic or adversarial drift.
- Future directions: Expanding to non-convex settings; studying adaptive $\beta(t)$ schedules; combining with second-order information to preserve variance reduction benefits.

## Related Work & Insights
- **vs Loizou & Richtárik 2020**: They proved momentum does not reduce MSE in slow-adapting stationary settings; this work extends to full nonstationarity, proving systematic harm.
- **vs Allen-Zhu & Hazan 2016**: They proved acceleration is optimal in deterministic convex settings; this work shows the acceleration advantage vanishes or reverses under combined stochasticity and nonstationarity.
- **vs Zhang 2015 / Hardt 2016 (Dynamic Regret)**: This work provides more refined lower bounds using variation budgets, offering the first quantitative characterization of the information-theoretic cost of momentum.
- **Insight**: Performance for all "history-averaging" methods (e.g., SWA, EMA shadow weights) needs to be re-evaluated in nonstationary scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  The first paper to provide a rigorous quantitative proof of the systematic inferiority of momentum under distribution shift; both the 2D Lyapunov and information-theoretic lower bounds are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐  Progression from strongly convex quadratic to MLP is sound with sufficient ablation; lacks empirical validation in large-scale deep learning scenarios (e.g., nonstationary RL).
- Writing Quality: ⭐⭐⭐⭐⭐  Theorem statements are precise, the $(1 - \beta)^{-2}$ main thread is consistent, and visualizations are intuitive.
- Value: ⭐⭐⭐⭐⭐  Resolves a long-standing practical confusion (why momentum fails in nonstationary settings) and provides theoretical guidance for algorithm design (necessity of restarts, step size scheduling, and momentum decay).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](../../ICLR2026/optimization/sgd_with_adaptive_preconditioning_unified_analysis_and_momentum_acceleration.md)
- [\[ICLR 2026\] DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD](../../ICLR2026/optimization/dnt_a_deeply_normalized_transformer_that_can_be_trained_by_momentum_sgd.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)

</div>

<!-- RELATED:END -->
