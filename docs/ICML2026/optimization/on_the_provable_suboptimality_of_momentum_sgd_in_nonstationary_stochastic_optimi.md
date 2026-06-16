---
title: >-
  [Paper Note] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time, Momentum SGD (MSGD) is systematically inferior to vanilla SGD due to "inertial lag." The performance degradation is characterized by an amplification factor of the order $(1 - \beta)^{-2}$.
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 4a10418942a065d9
---
# On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization

**Conference**: ICML 2026  
**arXiv**: [2601.12238](https://arxiv.org/abs/2601.12238)  
**Code**: To be confirmed  
**Area**: Optimization Theory  
**Keywords**: Momentum methods, nonstationary optimization, tracking error, distribution shift, information-theoretic lower bounds

## TL;DR
This paper theoretically proves that in nonstationary strongly convex stochastic optimization where the optimum drifts over time, Momentum SGD (MSGD) is systematically inferior to vanilla SGD due to "inertial lag." The performance degradation is characterized by an amplification factor of the order $(1 - \beta)^{-2}$. Through information-theoretic lower bounds, the authors demonstrate that this cost is not an artifact of analysis but a fundamental obstacle inherent to any momentum-based method.

## Background & Motivation

**Background**: Momentum methods (Heavy-Ball, Nesterov) have been proven to accelerate convergence and reduce gradient noise in static convex optimization, making them the default configuration in deep learning. However, in nonstationary environments such as online learning, federated learning, and reinforcement learning, the optimum $\theta_t^*$ continuously moves due to distribution shift, rendering past gradients "stale."

**Limitations of Prior Work**: Empirically, momentum often exhibits instability and poorer tracking performance in dynamic environments. However, a rigorous theoretical explanation is lacking. Existing dynamic regret analyses (e.g., Zhang 2015; Hardt 2016) typically provide general path-length bounds without explicitly characterizing the relationship between the momentum parameter $\beta$ and performance degradation. Furthermore, no information-theoretic lower bounds have shown whether this is an inherent cost of momentum or a result of loose analysis.

**Key Challenge**: Momentum simultaneously pushes in two opposite directions: (1) it averages historical gradients to reduce variance in static noise scenarios; (2) it averages "stale gradients" during distribution shifts, creating inertial lag that causes the algorithm to systematically fall behind the moving target.

**Goal**: To quantitatively characterize the performance discrepancy between MSGD and vanilla SGD in nonstationary strongly convex smooth optimization, and to provide a clear boundary for when momentum helps or hurts.

**Key Insight**: By viewing SGDM as a 2D dynamical system composed of the "parameters + momentum buffer," the authors use Lyapunov functions to make the amplification factors $(1 - \beta)^{-1}$ to $(1 - \beta)^{-2}$ explicit. They then use Assouad-style constructions under variation budgets to prove these factors are information-theoretically inevitable.

**Core Idea**: The tracking error can be decomposed into three terms: "initialization forgetting," "noise floor," and "drift-induced lag." Momentum amplifies each term by a factor of the order $(1 - \beta)^{-k}$, matching the derived lower bounds.

## Method

### Overall Architecture
Consider a time-varying strongly convex smooth problem $G_t(\theta) = \mathbb{E}_{X_t \sim \Pi_t}[g(\theta, X_t)]$, where the optimum $\theta_t^*$ drifts over time. The goal is to track $\theta_t^*$ rather than converging to a single point.

SGD: $\theta_{t+1} = \theta_t - \gamma_t \nabla g(\theta_t, X_{t+1})$.

Generalized SGDM: $\psi_t = \theta_t + \beta_1 (\theta_t - \theta_{t-1})$; $\theta_{t+1} = \psi_t - \gamma_t \nabla g(\psi_t, X_{t+1}) + \beta_2 (\psi_t - \psi_{t-1})$. Heavy-Ball sets $\beta_1 = 0, \beta_2 = \beta$, while Nesterov sets $\beta_1 = \beta, \beta_2 = 0$.

### Key Designs

**1. 2D Lyapunov + Three-term Tracking Error Decomposition (Upper Bound): Exposing the Momentum Amplification Factor**

To characterize when momentum helps or hurts, the authors first provide explicit tracking error upper bounds for SGD and SGDM. For SGD:

$$\mathbb{E}\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma\mu/2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\Delta^2}{\gamma^2 \mu^2} + \frac{\sigma^2 \gamma}{\mu}$$

The three terms represent initialization forgetting, drift lag, and noise floor. In SGDM, the updates for parameter $\theta_t$ and the momentum buffer are coupled. Treating them as a 1D recurrence loses coupling information; this work treats them as a 2D Lyapunov function. Consequently, each of the three terms is multiplied by an amplification factor of $(1-\beta)^{-2}$. The conclusion is straightforward: while momentum reduces variance by averaging historical gradients in static scenarios, it averages "stale gradients" under distribution shift, causing a systematic lag.

**2. Time-Resolved High-Probability Bounds + Weighted Historical Drift: Beyond Uniform Drift Bounds**

Typical uniform drift bounds $\Delta$ assume drift is constant, but real-world drift is often intermittent or localized (seasonal or abrupt). This step uses optional stopping arguments for martingale differences instead of MGF recurrences to provide a bound at any time $t$ with probability $1-\delta$:

$$\|\theta_t - \theta_t^*\|^2 \lesssim (1 - \gamma\mu/2)^t \|\theta_0 - \theta_0^*\|^2 + \frac{\mathfrak{D}_t}{\gamma\mu} + O(d\sigma^2\gamma/\mu)$$

Here, $\mathfrak{D}_t = \sum_{\ell=0}^{t-1}(1-\gamma\mu/2)^{t-\ell-1}\|\Delta_\ell\|^2$ is the weighted historical drift. This captures the locality of drift—recent drifts have higher weight while older ones decay—directly inspiring restart and windowing strategies.

**3. Information-Theoretic Lower Bounds + Inertial Window: Proving Degradation is Fated**

To prove that the degradation is an inherent cost rather than analytical looseness, the authors construct a worst-case drift sequence under a variation budget $\mathrm{GVar}_{p,q}(g)\leq\mathbb{V}_T$. They provide a dynamic regret lower bound for SGDM: $\mathfrak{M}_T(\Pi_\beta,\mathbb{V}_T)\gtrsim\max\{(1-\beta)^{-2/(\alpha q+2)}\cdot\mathbb{V}_T^{2q/(\alpha q+2)}T^{\alpha q/(\alpha q+2)},\ldots\}$. This bound explicitly includes factors of $(1-\beta)^{-1}$ to $(1-\beta)^{-2}$, tightly matching the upper bound. The "block-drift" construction further proves that any SGDM must spend an "inertial window" of $\Omega(\kappa/(1-\beta))$ steps for transient adjustment after a distribution change.

### Loss & Training
- **Constant Step-size**: $\gamma^* = \arg\min_\gamma \left[ \frac{192 (2 + \beta)^2}{\mu^2 \gamma^2} \Delta^2 + \frac{96}{\mu (1 - \beta)} \sigma^2 \gamma \right]$.
- **Epoch Decay + Momentum Restart**: Increasing step-size logarithmically over time and resetting the momentum buffer to 0 at epoch boundaries to break the accumulation of stale gradients.

## Key Experimental Results

### Main Results: Strongly Convex Quadratic Target + Random Walk Drift

| Setting | SGD ($\gamma = 0.01$) | HB | NAG | Conclusion |
|------|-------------|-----|-----|------|
| $\beta = 0.50, \sigma^2 = 0.1$ | 1.036 | 0.342 | 0.349 | Moderate momentum helps |
| $\beta = 0.50, \sigma^2 = 0.8$ | 1.305 | 0.961 | 1.019 | Momentum beneficial in high noise |
| $\beta = 0.90, \sigma^2 = 0.1$ | 1.029 | 0.497 | 0.453 | Momentum still helps with light drift/low noise |
| $\beta = 0.90, \sigma^2 = 0.8$ | 1.466 | **3.899** | **3.721** | Momentum worsens with light drift/high noise |
| $\beta = 0.99, \sigma^2 = 0.8$ | 1.403 | **38.802** | **21.038** | Momentum collapses with strong momentum/high noise |

Tracking error after 5000 steps. As $\beta$ increases from 0.50 to 0.99, HB/NAG deteriorate sharply, while SGD remains relatively robust.

### Ablation Study: Interaction of Condition Number + Drift Magnitude

| Dataset | Condition Number $\kappa$ | SGD | HB ($\beta = 0.9$) | NAG ($\beta = 0.9$) | HB/SGD |
|--------|---------------|-----|-------------------|-------------------|--------|
| Linear Regression | 10 | 0.31 | 2.47 | 1.73 | 7.97× |
| Linear Regression | 1000 | 1.28 | 12.30 | 9.19 | 9.61× |
| Logistic Regression | 10 | 0.42 | 3.56 | 2.18 | 8.48× |
| Teacher-Student MLP | — | 0.58 | 5.23 | 3.27 | 9.02× |

### Key Findings
- As the condition number $\kappa$ increases, the damage from momentum becomes more apparent—ill-conditioned problems require smaller step-sizes $\gamma \lesssim (1 - \beta)^2 / L$ for stability, further slowing convergence.
- Increasing drift magnitude $\delta_{\text{rw}}$ rapidly widens the performance gap between HB/NAG and SGD.
- The combination of high noise $\sigma^2 = 0.8$ and moderate drift $\beta = 0.9$ represents the most vulnerable region for momentum, where inertial lag and noise amplification compound.

## Highlights & Insights
- **2D Lyapunov Dynamical System Perspective**: Analyzing the two coupled recurrences (parameters and momentum) together is the key to making the $(1 - \beta)^{-2}$ factor explicit; this approach can be applied to other optimization algorithms with auxiliary variables.
- **Fundamentality of $(1 - \beta)^{-2}$**: The tightly matched upper and lower bounds prove this is an information-theoretic necessity, not an artifact of loose analysis.
- **Time-Resolved Boundaries**: Replacing the uniform drift bound $\Delta$ with weighted history $\mathfrak{D}_t$ allows for adaptation to intermittent drift and inspires using "gradient-momentum alignment" $S_t = 1 - \frac{\langle \nabla g, v \rangle}{\|\nabla g\| \|v\|}$ as a change detection signal.
- **Drift-Noise Trade-off Visualization**: Clearly illustrates how momentum simultaneously amplifies initialization sensitivity, the noise floor, and drift lag, narrowing the viable trade-off space.

## Limitations & Future Work
- Restricted by the strong convexity assumption; non-convex scenarios (e.g., PŁ conditions) might follow similar logic, but results are not provided.
- The stability condition $\gamma \leq \mu (1 - \beta)^2 / (4 L^2)$ is conservative; while the qualitative conclusion is robust, quantitative predictions require finer analysis.
- Assumes the optimum $\theta_t^*$ is measurable; lacks analysis for stochastic or adversarial drift.
- Future directions: Extension to non-convex settings; research into adaptive $\beta(t)$ scheduling; integration with second-order information to retain variance reduction advantages.

## Related Work & Insights
- **vs. Loizou & Richtárik 2020**: They proved momentum does not reduce MSE in slow-adapting stationary settings; this work extends this to fully nonstationary settings, proving systematic harm.
- **vs. Allen-Zhu & Hazan 2016**: They proved acceleration is optimal in deterministic convex settings; this work shows the acceleration advantage disappears or even reverses under the combination of stochasticity and nonstationarity.
- **vs. Zhang 2015 / Hardt 2016 (Dynamic Regret)**: This paper uses variation budgets to provide more refined lower bounds, quantitatively characterizing the information-theoretic cost of momentum for the first time.
- **Insight**: For all "history-averaging" methods (e.g., SWA, EMA shadow weights), their performance in nonstationary scenarios needs to be re-evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first paper to strictly and quantitatively prove the systematic disadvantage of momentum under distribution shift; both the 2D Lyapunov and lower bound analyses are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Progression from quadratic cases to MLP is solid; however, lacks empirical validation in large-scale deep learning scenarios (e.g., nonstationary RL/FL).
- Writing Quality: ⭐⭐⭐⭐⭐ Precise theorem statements; the $(1 - \beta)^{-2}$ theme is consistently integrated; clear visualizations.
- Value: ⭐⭐⭐⭐⭐ Resolves a long-standing empirical confusion (why momentum fails in nonstationary settings) and provides theoretical guidance for algorithm design (necessity of restarts, step-size scheduling, and momentum decay).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](../../ICLR2026/optimization/provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[ICML 2025\] Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent](../../ICML2025/optimization/provable_benefit_of_random_permutations_over_uniform_sampling_in_stochastic_coor.md)

</div>

<!-- RELATED:END -->
