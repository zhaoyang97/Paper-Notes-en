---
title: >-
  [Paper Note] Cost-Aware Stopping for Bayesian Optimization
description: >-
  [ICML2026][Optimization][Bayesian optimization] The authors expand Weitzman's Pandora's Box stopping principle to correlated Bayesian Optimization (BO) scenarios. They demonstrate that under a shared stopping rule where…
tags:
  - "ICML2026"
  - "Optimization"
  - "Bayesian optimization"
  - "cost-aware stopping"
  - "Pandora's Box"
  - "Gittins index"
  - "expected improvement per cost"
date: 2026-05-08
content_hash: 89369612924c5b61
---

# Cost-Aware Stopping for Bayesian Optimization

**Conference**: ICML2026  
**arXiv**: [2507.12453](https://arxiv.org/abs/2507.12453)  
**Code**: https://github.com/QianJaneXie/CostAwareStoppingBayesOpt  
**Area**: Bayesian Optimization / AutoML / Decision Theory  
**Keywords**: Bayesian optimization, cost-aware stopping, Pandora's Box, Gittins index, expected improvement per cost

## TL;DR
The authors expand Weitzman's Pandora's Box stopping principle to correlated Bayesian Optimization (BO) scenarios. They demonstrate that under a shared stopping rule where "the acquisition function value exceeds the current optimum," both PBGI and LogEIPC acquisition functions ensure that the cost-adjusted simple regret is no worse than "stopping after one evaluation." This provides the first adaptive stopping rule with theoretical guarantees for cost-adjusted simple regret.

## Background & Motivation

**Background**: Bayesian Optimization (BO) is the primary framework for optimizing expensive black-box objectives $f:X\to\mathbb{R}$. It uses Gaussian Processes (GP) to fit the posterior and acquisition functions $\alpha_t$ to balance exploration and exploitation when selecting the next point $x_{t+1}$. Common acquisition functions include EI, LCB, KG, and TS. When evaluation costs $c(x)$ are significantly heterogeneous (e.g., training times vary across hyperparameters), cost-aware versions have been proposed, with **PBGI** (Pandora's Box Gittins Index, Xie et al. 2024) and **LogEIPC** (Log EI per cost, Ament et al. 2023) being state-of-the-art.

**Limitations of Prior Work**: The question of "when to stop" in BO has been largely neglected. Existing stopping rules are either heuristic (fixed iterations, no improvement for several rounds) or based on simple regret convergence (UCB-LCB, PRB, SRGap-med). Crucially, **none explicitly model the evaluation cost**. In cost-aware scenarios, these rules often result in minor regret improvements at the expense of massive evaluation overhead, leading to cumulative costs that far exceed marginal utility.

**Key Challenge**: In practice, users aim to minimize the cost-adjusted simple regret $\mathcal{R}_c = \mathbb{E}[\min_{1\le t\le\tau} f(x_t)-\inf_{x\in X} f(x) + \sum_{t=1}^\tau c(x_t)]$. However, existing rules focus solely on the first term and ignore the second. Furthermore, even rules like EI thresholding (Nguyen et al. 2017), which stop when EI falls below a threshold, rely on heuristic thresholds rather than fundamental principles.

**Goal**: To design an adaptive stopping rule that is **explicitly cost-aware**, **theoretically provable**, and requires **minimal user parameters**. It should apply to both uniform and heterogeneous cost scenarios and naturally pair with PBGI and LogEIPC acquisition functions.

**Key Insight**: The authors return to the classic Pandora's Box optimal strategy by Weitzman (1979), where the Gittins-index argument dictates that the selection strategy and stopping time must be **bundled** to achieve Bayesian optimality. By extending this stopping condition from an independent discrete setting to a correlated GP setting, the PBGI stopping rule is naturally derived. Through EI monotonicity, this is **equivocally** rewritten as the LogEIPC stopping rule, revealing that two acquisition functions from different origins share the same stopping rule.

**Core Idea**: The stopping condition $\min_x \alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is equivalent to $\max_x \alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$. This expresses the intuition that "no unevaluated point has a fair value better than the current optimum."

## Method

### Overall Architecture
PBGI treats each candidate point as a "box" in Pandora's Box. Its fair value $\alpha_t^{\mathrm{PBGI}}(x)$ is defined as the threshold where the expected improvement equals the cost: $\mathrm{EI}_{f\mid x_{1:t}, y_{1:t}}(x; \alpha_t^{\mathrm{PBGI}}(x))=c(x)$. Weitzman's classic argument shows that in independent discrete settings, **selecting** $\arg\min_x \alpha^{\mathrm{PBGI}}(x)$ and **stopping** at $\min_x \alpha^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is the Bayesian optimal strategy. The authors maintain this structure in the correlated GP setting: the posterior updates $\alpha_t^{\mathrm{PBGI}}$ each round, followed by the stopping condition check.

### Key Designs

1.  **Stopping Rule Based on PBGI (Using Updated $\alpha_t$ vs. $\alpha_{t-1}$)**:
    - **Function**: Determines at each round $t$ whether any boxes remain worth the cost of opening relative to the current optimum.
    - **Mechanism**: The stopping condition is $\min_{x\in X\setminus\{x_1,\dots,x_t\}} \alpha_t^{\mathrm{PBGI}}(x) \ge y^*_{1:t}$. A subtle but critical design choice is using $\alpha_t$ **after the posterior update** (instead of $\alpha_{t-1}$ used in prior theoretical work like Gergatsouli & Tzamos 2023). Intuitively, $\alpha_t^{\mathrm{PBGI}}(x)$ represents the "fair price" of point $x$ given all current information. Using the latest $\alpha_t$ accurately reflects whether further investment is justified. Experiments in Section C.2 verify that this choice significantly improves cost-adjusted regret.
    - **Design Motivation**: Weitzman's original strategy requires selection and stopping to be paired for optimality. Extending this to correlated GP using $\alpha_{t-1}$ introduces information lag, causing premature or delayed stops.

2.  **Derived Equivalent LogEIPC Stopping Rule via EI Monotonicity**:
    - **Function**: Reforms the PBGI stopping condition into an equivalent LogEIPC form, allowing one rule to support two cost-aware acquisition functions.
    - **Mechanism**: Since $\mathrm{EI}_\psi(x;y)$ is strictly monotonically increasing with respect to $y$, $\alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ holds if and only if $\mathrm{EI}_{f\mid x_{1:t},y_{1:t}}(x;y^*_{1:t}) \le c(x)$. Taking the log yields $\max_{x\in X\setminus\{x_1,\dots,x_t\}}\alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$. When $c(x)\equiv c_0$, this simplifies to $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$, recovering EI thresholding (Nguyen et al. 2017) while replacing heuristic thresholds with the "cost" itself.
    - **Design Motivation**: This provides two perspectives—Pandora's Box decision theory and EI-per-cost economics—broadening applicability and allowing LogEIPC users to benefit from theoretical guarantees without new conditions.

3.  **Cost-Adjusted Regret Guarantee and Finite-Time Termination**:
    - **Function**: Provides the first theoretical guarantee for cost-adaptive stopping concerning cost-adjusted simple regret.
    - **Mechanism**: Lemma 3.1 proves that before stopping, every selected $x_{t+1}$ satisfies $\alpha_t^{\mathrm{EI}}(x_{t+1})\ge c(x_{t+1})$, meaning the expected improvement is at least the cost. Theorem 3.2 establishes $\mathbb{E}[y^*_{1:\tau}-\min_x f(x)+\sum_{t=1}^\tau c(x_t)]\le \mathbb{E}[y_1-\min_x f(x)+c(x_1)]=U+C$, where $U=\mu(x_1)-\mathbb{E}[\min_x f(x)]$ and $C=c(x_1)$. This implies the rule is **guaranteed to be no worse than "stopping immediately after one evaluation,"** a "no-regret" property for cost-adjusted regret. Corollary 3.3 adds an upper bound on cumulative cost ($U+C$) and ensures finite-time termination with probability $1-\delta$ within $\frac{U+C}{\delta c_0}$ steps if $c(x)\ge c_0>0$.
    - **Design Motivation**: Unlike previous rules that ignore cumulative cost, this connects cost and regret in a non-asymptotic bound, ensuring safety in worst-case scenarios.

### Loss & Training
No training is involved; a single condition check is added to the BO loop. Implementation details include: (i) **Stabilization & Moving Average**: To handle GP hyperparameter and acquisition optimization instability in high dimensions, a window of $W=20$ rounds acts as a stabilization period, and a moving average of $W$ rounds is applied to the stopping signal. (ii) **Unknown Costs**: Costs $\ln c(x)$ are modeled as a GP, using $\mathbb{E}[c(x)]=\exp(\mu_{\ln c}+\sigma_{\ln c}^2/2)$ as a plug-in for $c(x)$.

## Key Experimental Results

### Main Results
The authors evaluate the PBGI/LogEIPC stopping rules against seven baselines across three scenarios: 1D Bayesian regret (grid search used to eliminate optimization error), 8D Bayesian regret (three cost types: uniform, linear, periodic), and two AutoML benchmarks—LCBench (35 datasets) and NATS-Bench (32k architectures).

| Scenario | Dim / Cost | PBGI + Ours | LogEIPC + Ours | UCB-LCB | Convergence | Hindsight (Oracle) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bayesian regret 1D | $\lambda=0.1$ | Near Hindsight | Near Hindsight | High | High | Lower Bound |
| Bayesian regret 8D | Linear cost | Near Hindsight | Near Hindsight | Significantly worse | Significantly worse | Lower Bound |
| LCBench (35 datasets) | $\lambda=10^{-3}$ | Top-3 in ~75% datasets | Top-3 in ~75% datasets | High | Moderate | Lower Bound |
| NATS-Bench | $\lambda=10^{-5}$ | Near Hindsight | Slightly < PBGI | Often hits 200-round cap | Moderate | Lower Bound |

### Ablation Study
| Configuration | Key Findings |
| :--- | :--- |
| $\alpha_t$ vs. $\alpha_{t-1}$ | $\alpha_t$ is significantly superior, reflecting the real-time "fair value" after updates. |
| Stabilization + Moving Avg ($W=20$) | Crucial for high-dimensional stability to prevent spurious stops from GP jitter. |
| Known vs. Unknown Cost | Performance is nearly identical using $\mathbb{E}[c(x)]$ from a GP model. |
| Cost model misspecification | Performance degrades gracefully; the rule remains robust even with biased cost models. |

### Key Findings
- Most baseline rules (SRGap-med, UCB-LCB) on NATS-Bench **fail to stop before the 200-round limit**, while the proposed rule terminates appropriately.
- The "partnership" between selection and stopping is vital: PBGI + the proposed rule outperforms LogEIPC on LCBench, likely due to PBGI's superior robustness under GP misspecification.
- Even with biased cost measurements (e.g., proxying runtime as a linear function of parameters), the relative ranking of the stopping rule remains stable.

## Highlights & Insights
- **Bundling selection and stopping** is essential for Bayesian optimality. This core property of Weitzman (1979) was overlooked by most BO research; this paper reintroduces it and generalizes it to correlated GP settings.
- **Shared stopping for different acquisition functions**: The discovery that PBGI (Gittins index) and LogEIPC (one-step lookahead) share a literally equivalent stopping condition via EI monotonicity provides a unified framework.
- **"No worse than stopping immediately"** is a pragmatic worst-case guarantee. While weaker than absolute regret bounds, it is highly functional for cost-adjusted settings, ensuring the user is never worse off by using BO.

## Limitations & Future Work
- Theoretical guarantees are specific to PBGI and LogEIPC; acquisition functions based on value-of-information (KG, MES) would require their own derived rules.
- The stabilization window $W=20$ is an engineering necessity but introduces a hyperparameter that might be too large for very small budgets ($<50$ iterations).
- The "no worse than immediate" bound is tight in worst-case scenarios (where cost > potential improvement) but does not provide tighter regret rates for intermediate regions.
- Assumes $c(x)$ is independent of $f$ observations; if failures incur extra costs, the framework requires adjustment.

## Related Work & Insights
- **vs. Nguyen et al. (2017) / Zhou et al. (2024)**: Their EI thresholding uses a heuristic $c_0$. This paper proves that replacing $c_0$ with actual per-sample cost is the LogEIPC rule under uniform costs, elevating heuristics to principled constants.
- **vs. UCB-LCB (Makarova et al. 2022)**: UCB-LCB ignores costs and over-evaluates in cost-aware settings; this paper explicitly incorporates $c(x)$.
- **vs. PRB (Wilson, 2024)**: PRB provides confidence bounds on simple regret but ignores cost, often spending significantly more than the hindsight oracle.
- **vs. Chick & Frazier (2012)**: They addressed cost-aware stopping for independent sampling; this paper extends the approach to correlated GPs via PBGI, making it suitable for modern BO.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Elegant generalization of Weitzman's rule to BO and the realization of stopping equivalence.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across synthetic regret and large-scale benchmarks (LCBench, NATS-Bench).
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from intuition to theory and validation.
- **Value**: ⭐⭐⭐⭐⭐ Provides a "safe default" stopping rule for AutoML with the first non-asymptotic cost-adjusted regret bound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/optimization/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive $\varepsilon$-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2026\] Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization](asymmetric_perturbation_in_solving_bilinear_saddle-point_optimization.md)

</div>

<!-- RELATED:END -->
