---
title: >-
  [Paper Note] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization
description: >-
  [ICML 2026][Optimization][Momentum methods] This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Momentum methods"
  - "nonstationary optimization"
  - "tracking error"
  - "distribution shift"
  - "information-theoretic lower bounds"
date: 2026-05-08
content_hash: 1cd89e93afaa4a2e
---

# On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization

**Conference**: ICML 2026  
**arXiv**: [2601.12238](https://arxiv.org/abs/2601.12238)  
**Code**: To be confirmed  
**Area**: Optimization Theory  
**Keywords**: Momentum methods, nonstationary optimization, tracking error, distribution shift, information-theoretic lower bounds

## TL;DR
This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time, Momentum SGD is systematically inferior to vanilla SGD due to "inertial lag." The performance degradation cost is amplified by a factor of the order $(1 - \beta)^{-2}$. Through information-theoretic lower bounds, it is demonstrated that this cost is a fundamental barrier rather than an artifact of the analysis.

## Background & Motivation

**Background**: Momentum methods (Heavy-Ball, Nesterov) have been proven to accelerate convergence and reduce gradient noise in static convex optimization and are nearly default configurations in deep learning. However, in nonstationary environments such as online learning, federated learning, and reinforcement learning, the optimum $\theta_t^*$ moves continuously with distribution shifts, rendering historical gradients stale.

**Limitations of Prior Work**: Empirically, momentum often exhibits instability and poorer tracking performance in dynamic environments, but rigorous theoretical explanations are lacking. Existing dynamic regret analyses (Zhang 2015; Hardt 2016) typically provide general path-length bounds without explicitly characterizing the relationship between the momentum parameter $\beta$ and performance degradation. Furthermore, there are no information-theoretic lower bounds to clarify whether this is an inherent cost of momentum or a result of loose analysis.

**Key Challenge**: Momentum pushes in two opposite directions simultaneously: (1) it reduces variance by averaging historical gradients in static noisy scenarios; (2) it generates inertial lag by averaging "stale gradients" under distribution shift, causing the algorithm to systematically lag behind the moving target.

**Goal**: To quantitatively characterize the performance difference between Momentum SGD and vanilla SGD in nonstationary strongly convex smooth optimization, and to provide clear boundaries for when momentum helps versus when it hurts.

**Key Insight**: By viewing SGDM as a 2D dynamic system composed of "parameters + momentum buffer" and utilizing Lyapunov function analysis for stability, the amplification factor from $(1 - \beta)^{-1}$ to $(1 - \beta)^{-2}$ is explicitly revealed. Subsequently, Assouad-style constructions under variation budgets are used to prove that these factors are information-theoretically inevitable.

**Core Idea**: Tracking error can be decomposed into three terms: "initialization forgetting + noise floor + drift-induced lag." Momentum amplifies each term by a factor of order $(1 - \beta)^{-k}$, consistently matching the tightly derived lower bounds.

## Method

### Overall Architecture
Consider a time-varying strongly convex smooth problem $G_t(\theta) = \mathbb{E}_{X_t \sim \Pi_t}[g(\theta, X_t)]$, where the optimum $\theta_t^*$ drifts over time. The goal is to track $\theta_t^*$ rather than converging to a single point.

SGD: $\theta_{t+1} = \theta_t - \gamma_t \nabla g(\theta_t, X_{t+1})$.

Generalized SGDM: $\psi_t = \theta_t + \beta_1 (\theta_t - \theta_{t-1})$, $\theta_{t+1} = \psi_t - \gamma_t \nabla g(\psi_t, X_{t+1}) + \beta_2 (\psi_t - \psi_{t-1})$; Heavy-Ball uses $\beta_1 = 0, \beta_2 = \beta$, while Nesterov uses $\beta_1 = \beta, \beta_2 = 0$.

### Key Designs

1.  **2D Lyapunov + Three-term Tracking Error Decomposition (Upper Bound)**:
    - **Function**: Provides explicit tracking error upper bounds for SGD and SGDM under nonstationary strongly convex smooth settings.
    - **Mechanism**: For SGD, $\mathbb{E}\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma \mu / 2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\Delta^2}{\gamma^2 \mu^2} + \frac{\sigma^2 \gamma}{\mu}$, where the terms represent initialization forgetting, drift lag, and noise floor. For SGDM, tracking the concatenated parameters and momentum buffer results in each of the three terms being amplified by factors of $(1 - \beta)^{-2}$ / $(1 - \beta)^{-2}$ / $(1 - \beta)^{-2}$.
    - **Design Motivation**: Formulating the two coupled recursions of SGDM (parameters + momentum) into a unified 2D Lyapunov system avoids the loss of coupling information inherent in 1D decomposition—a key technique for making the $(1 - \beta)^{-2}$ factor explicit.

2.  **Time-Resolved High-Probability Bounds + Weighted Historical Drift**:
    - **Function**: Provides high-probability tracking error bounds for any time $t$ without requiring a uniform drift upper bound.
    - **Mechanism**: Utilizes optional stopping arguments for martingale differences instead of MGF recursions. With probability $1 - \delta$, $\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma \mu / 2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\mathfrak{D}_t}{\gamma \mu} + O(d \sigma^2 \gamma / \mu)$, where $\mathfrak{D}_t = \sum_{\ell = 0}^{t-1} (1 - \gamma \mu / 2)^{t - \ell - 1} \|\Delta_\ell\|^2$ is the **weighted historical drift** rather than a fixed upper bound.
    - **Design Motivation**: Actual drift is often intermittent or local (seasonal, abrupt). Using weighted history adaptively captures this locality, directly inspiring restart and windowing strategies.

3.  **Information-Theoretic Lower Bounds + Inertial Window**:
    - **Function**: Proves that momentum degradation is an information-theoretic necessity rather than an analytical artifact.
    - **Mechanism**: Constructs a worst-case drift sequence under the variation budget constraint $\mathrm{GVar}_{p, q}(g) \leq \mathbb{V}_T$. For SGDM, the dynamic regret lower bound $\mathfrak{M}_T(\Pi_\beta, \mathbb{V}_T) \gtrsim \max\{(1 - \beta)^{-2/(\alpha q + 2)} \cdot \mathbb{V}_T^{2q/(\alpha q + 2)} T^{\alpha q/(\alpha q + 2)}, \ldots\}$ explicitly contains factors of $(1 - \beta)^{-1}$ to $(1 - \beta)^{-2}$, matching the upper bound. The lower bound proof via "block drift" construction shows that any SGDM must spend an "inertial window" of $\Omega(\kappa / (1 - \beta))$ steps for transient adjustment after a change.
    - **Design Motivation**: Tightly matched upper and lower bounds prove that "inertial lag" is the fundamental fate of momentum in nonstationary settings.

### Training Strategy
- **Constant Step Size**: $\gamma^* = \arg\min_\gamma \left[ \frac{192 (2 + \beta)^2}{\mu^2 \gamma^2} \Delta^2 + \frac{96}{\mu (1 - \beta)} \sigma^2 \gamma \right]$.
- **Epoch Decay + Momentum Restart**: Increase step size over logarithmic time and reset the momentum buffer to 0 at epoch boundaries to break the accumulation of stale gradients.

## Key Experimental Results

### Main Results: Strongly Convex Quadratic Objective + Random Walk Drift

| Setting | SGD ($\gamma = 0.01$) | HB | NAG | Conclusion |
|------|-------------|-----|-----|------|
| $\beta = 0.50, \sigma^2 = 0.1$ | 1.036 | 0.342 | 0.349 | Moderate momentum helps |
| $\beta = 0.50, \sigma^2 = 0.8$ | 1.305 | 0.961 | 1.019 | Momentum beneficial under high noise |
| $\beta = 0.90, \sigma^2 = 0.1$ | 1.029 | 0.497 | 0.453 | Momentum still helps with light drift + low noise |
| $\beta = 0.90, \sigma^2 = 0.8$ | 1.466 | **3.899** | **3.721** | Momentum worsens with light drift + high noise |
| $\beta = 0.99, \sigma^2 = 0.8$ | 1.403 | **38.802** | **21.038** | Momentum collapses with strong momentum + high noise |

Tracking error after 5000 steps. As $\beta$ increases from 0.50 → 0.99, HB/NAG deteriorate sharply, while SGD remains relatively robust.

### Ablation Study: Interaction of Condition Number + Drift Magnitude

| Dataset | Condition Number $\kappa$ | SGD | HB ($\beta = 0.9$) | NAG ($\beta = 0.9$) | HB/SGD |
|--------|---------------|-----|-------------------|-------------------|--------|
| Linear Regression | 10 | 0.31 | 2.47 | 1.73 | 7.97× |
| Linear Regression | 1000 | 1.28 | 12.30 | 9.19 | 9.61× |
| Logistic Regression | 10 | 0.42 | 3.56 | 2.18 | 8.48× |
| Teacher-Student MLP | — | 0.58 | 5.23 | 3.27 | 9.02× |

### Key Findings
- The larger the condition number $\kappa$, the more pronounced the damage from momentum—ill-conditioned problems require smaller step sizes $\gamma \lesssim (1 - \beta)^2 / L$ for stability, further slowing convergence.
- As the drift magnitude $\delta_{\text{rw}}$ increases, the gap between HB/NAG and SGD widens rapidly.
- High noise $\sigma^2 = 0.8$ combined with moderate drift $\beta = 0.9$ is the most vulnerable region for momentum, where inertial lag and noise amplification overlap.

## Highlights & Insights
- **2D Lyapunov Dynamic System Perspective**: Analyzing the two coupled recursions of SGDM (parameters + momentum) together is key to making the $(1 - \beta)^{-2}$ factor explicit, offering a transferable technique for analyzing other optimization algorithms with auxiliary variables.
- **Fundamentality of $(1 - \beta)^{-2}$**: Tightly matched upper and lower bounds prove this is an information-theoretic necessity, not an artifact of loose analysis.
- **Time-Resolved Bounds**: Replacing the uniform drift upper bound $\Delta$ with weighted history $\mathfrak{D}_t$ allows for adaptation to intermittent drift and directly inspires the use of "gradient-momentum alignment" $S_t = 1 - \frac{\langle \nabla g, v \rangle}{\|\nabla g\| \|v\|}$ as a change detection signal.
- **Drift-Noise Tradeoff Visualization**: Clearly illustrates how momentum simultaneously amplifies initialization sensitivity, the noise floor, and drift lag, narrowing the viable tradeoff space.

## Limitations & Future Work
- Restricted by the strong convexity assumption; while results for non-convex scenarios (e.g., PŁ conditions) might be analogous, the authors did not provide them.
- Stability conditions $\gamma \leq \mu (1 - \beta)^2 / (4 L^2)$ are somewhat conservative; qualitative conclusions are robust, but quantitative predictions require finer analysis.
- Assumes the optimum $\theta_t^*$ is measurable; lacks analysis for stochastic or adversarial drift.
- Future directions: Extension to non-convex settings; research into adaptive $\beta(t)$ scheduling; combining with second-order information to retain variance reduction advantages.

## Related Work & Insights
- **vs. Loizou & Richtárik 2020**: They proved momentum does not reduce MSE in slow-adapting stationary settings; this paper extends this to fully nonstationary settings, proving systematic harm.
- **vs. Allen-Zhu & Hazan 2016**: They proved acceleration is optimal in deterministic convex settings; this paper shows that under the combination of stochasticity and nonstationarity, the acceleration advantage disappears or even reverses.
- **vs. Zhang 2015 / Hardt 2016 (Dynamic Regret)**: This paper provides finer lower bounds using variation budgets, quantifying the information-theoretic cost of momentum for the first time.
- **Insights**: All "history-averaging" methods (e.g., SWA, EMA shadow weights) need to be re-examined regarding their performance in nonstationary scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first paper to rigorously and quantitatively prove the systematic inferiority of momentum under distribution shift; both 2D Lyapunov and information-theoretic lower bounds are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Progressive verification from strongly convex quadratic to MLP with thorough ablation; lacks empirical validation in real-world deep learning scenarios (e.g., nonstationary RL or Federated Learning).
- Writing Quality: ⭐⭐⭐⭐⭐ Precise theorem statements; the $(1 - \beta)^{-2}$ central theme is consistently maintained; intuitive charts.
- Value: ⭐⭐⭐⭐⭐ Resolves a long-standing empirical puzzle (why momentum fails in nonstationary settings) and provides theoretical guidance for algorithm design (necessity of restarts, step-size scheduling, and momentum decay).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](../../ICLR2026/optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)

</div>

<!-- RELATED:END -->
