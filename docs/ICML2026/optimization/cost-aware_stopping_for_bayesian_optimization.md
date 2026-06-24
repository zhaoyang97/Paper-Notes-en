---
title: >-
  [Paper Note] Cost-Aware Stopping for Bayesian Optimization
description: >-
  [ICML2026][Optimization][Bayesian optimization] The authors extend Weitzman's Pandora's Box stopping rule to Bayesian Optimization (BO) with correlations. They prove that under a shared "acquisition function value crossing the current best" stopping rule, the PBGI and LogEIPC cost-aware acquisition functions achieve an expected cost-adjusted simple regret no worse than "stopping after one sample." This provides the first adaptive stopping rule with theoretical guarantees for…
tags:
  - "ICML2026"
  - "Optimization"
  - "Bayesian optimization"
  - "cost-aware stopping"
  - "Pandora's Box"
  - "Gittins index"
  - "expected improvement per cost"
date: 2026-05-08
content_hash: 49000fb32aa73a44
---

# Cost-Aware Stopping for Bayesian Optimization

**Conference**: ICML2026  
**arXiv**: [2507.12453](https://arxiv.org/abs/2507.12453)  
**Code**: https://github.com/QianJaneXie/CostAwareStoppingBayesOpt  
**Area**: Bayesian Optimization / AutoML / Decision Theory  
**Keywords**: Bayesian optimization, cost-aware stopping, Pandora's Box, Gittins index, expected improvement per cost

## TL;DR
The authors extend Weitzman's Pandora's Box stopping rule to Bayesian Optimization (BO) with correlations. They prove that under a shared "acquisition function value crossing the current best" stopping rule, the PBGI and LogEIPC cost-aware acquisition functions achieve an expected cost-adjusted simple regret no worse than "stopping after one sample." This provides the first adaptive stopping rule with theoretical guarantees for cost-adjusted simple regret.

## Background & Motivation

**Background**: Bayesian Optimization (BO) is the dominant framework for optimizing expensive black-box objectives $f:X\to\mathbb{R}$. It fits a GP posterior and selects the next point $x_{t+1}$ using an acquisition function $\alpha_t$ to balance exploration and exploitation. Common acquisition functions include EI, LCB, KG, and TS. When evaluation costs $c(x)$ are significantly heterogeneous (e.g., varying training times for different hyperparameters), the community has developed cost-aware versions, with **PBGI** (Pandora's Box Gittins Index, Xie et al. 2024) and **LogEIPC** (log EI per cost, Ament et al. 2023) being two state-of-the-art (SOTA) methods.

**Limitations of Prior Work**: The question of "when to stop" in BO has been largely overlooked. Existing stopping rules are either heuristic (fixed iterations, best value unchanged for several rounds) or based on simple regret convergence criteria (UCB-LCB, PRB, SRGap-med, etc.), but **none explicitly model evaluation costs**. In cost-aware scenarios, these rules often lead to "marginal regret improvements at exorbitant evaluation costs," essentially resulting in cumulative costs far exceeding actual marginal gains.

**Key Challenge**: Users typically aim to minimize the cost-adjusted simple regret $\mathcal{R}_c = \mathbb{E}[\min_{1\le t\le\tau} f(x_t)-\inf_{x\in X} f(x) + \sum_{t=1}^\tau c(x_t)]$. However, existing rules focus solely on the first term while ignoring the second. Even rules like EI thresholding (Nguyen et al. 2017), which "stops when EI is below a threshold," rely on heuristically tuned thresholds rather than principled ones.

**Goal**: To design an **explicitly cost-aware**, **theoretically provable**, and **minimally parameterized** adaptive stopping rule. This rule should be unified for both uniform and heterogeneous cost scenarios and naturally integrate with PBGI and LogEIPC acquisition functions.

**Key Insight**: The authors revisit Weitzman's (1979) optimal strategy for Pandora's Box, where the Gittins-index argument requires the selection policy and stopping time to be **bundled** for Bayesian optimality. By extending this stopping condition from independent discrete settings to correlated GP settings, the stopping rule for PBGI naturally emerges. Then, through the monotonicity of EI, this is **equivalently** rewritten as a stopping rule for LogEIPC, revealing that these two acquisition functions from seemingly different origins share the same stopping rule.

**Core Idea**: The stopping condition $\min_x \alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is equivalent to $\max_x \alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$. This expresses the intuition that "no fair value of an unevaluated point is better than the current best."

## Method

### Overall Architecture
PBGI treats each candidate point as a "box" in Pandora's Box. Its fair value $\alpha_t^{\mathrm{PBGI}}(x)$ is defined as the threshold where the expected improvement equals the cost: $\mathrm{EI}_{f\mid x_{1:t}, y_{1:t}}(x; \alpha_t^{\mathrm{PBGI}}(x))=c(x)$. Weitzman's classical argument shows that in independent discrete settings, **selecting** $\arg\min_x \alpha^{\mathrm{PBGI}}(x)$ and **stopping** when $\min_x \alpha^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is Bayesian optimal. The authors maintain this structure in the correlated GP setting: in each round, the posterior is updated for $\alpha_t^{\mathrm{PBGI}}$ before evaluating the stopping condition.

### Key Designs

**1. PBGI-based Stopping Rule: Using updated $\alpha_t$ rather than $\alpha_{t-1}$ to judge if a "box" is worth opening.**

Existing rules are either heuristic or ignore evaluation costs, often "spending heavily for minimal gains." This paper's stopping condition directly asks: $\min_{x\in X\setminus\{x_1,\dots,x_t\}} \alpha_t^{\mathrm{PBGI}}(x) \ge y^*_{1:t}$, meaning stop when the fair value of all remaining boxes is no better than the current best. A subtle but crucial choice is using the **post-update** $\alpha_t$, unlike prior theoretical work (Gergatsouli & Tzamos 2023) which used $\alpha_{t-1}$. Since $\alpha_t^{\mathrm{PBGI}}(x)$ represents the "fair price" of $x$ given all current information, only the latest $\alpha_t$ truly answers whether it is worth continuing. This is necessary because Weitzman's original "select + stop" must be paired for optimality; using lagged information would result in sub-optimal stopping. Experiments in Section C.2 confirm that using $\alpha_t$ yields significant improvements in cost-adjusted regret.

**2. Deriving equivalent LogEIPC stopping rule via EI monotonicity: One rule for two acquisition functions.**

While PBGI stems from the Gittins index and LogEIPC from cost-normalized EI, the authors discover they share the same stopping condition. Since $\mathrm{EI}_\psi(x;y)$ is strictly monotonically increasing with respect to $y$, $\alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ if and only if $\mathrm{EI}_{f\mid x_{1:t},y_{1:t}}(x;y^*_{1:t}) \le c(x)$. Taking the log yields $\max_{x\in X\setminus\{x_1,\dots,x_t\}}\alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$. For uniform costs $c(x)\equiv c_0$, this simplifies to $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$, recovering the EI thresholding rule of Nguyen et al. (2017) / Zhou et al. (2024), but replacing heuristic thresholds with the principled "per-sample cost." This equivalence provides two perspectives: the decision-theoretic view of Pandora's Box and the economic view of EI-per-cost.

**3. "No worse than immediate stopping" guarantee for cost-adjusted regret + finite-time termination.**

This addresses the "guarantee" of the rule, which is often missing in existing methods. The proof proceeds in two steps: Lemma 3.1 shows that for every round $t<\tau$ before stopping, the selected $x_{t+1}$ satisfies $\alpha_t^{\mathrm{EI}}(x_{t+1})\ge c(x_{t+1})$, meaning the expected improvement is always at least the cost. Theorem 3.2 uses this to show:

$$\mathbb{E}\Big[y^*_{1:\tau}-\min_x f(x)+\sum_{t=1}^\tau c(x_t)\Big]\le \mathbb{E}\big[y_1-\min_x f(x)+c(x_1)\big]=U+C,$$

where $U=\mu(x_1)-\mathbb{E}[\min_x f(x)]$ and $C=c(x_1)$. This implies that the worst-case result using this rule is no worse than "stopping after one sample"—a "no-regret" guarantee in the cost-adjusted sense. Corollary 3.3 bounds the expected cumulative cost within $U+C$, and Corollary 3.5 extends this to budget-constrained scenarios, deriving a principled cost scaling of $\lambda=U/(B-C)$. Figures 2 and 3 demonstrate that many baselines actually perform worse than "immediate stopping."

### Loss & Training
No training is involved; a single check is added to the BO loop. Two engineering details are included for deployment: (i) **Stabilization period + Moving average**: Since GP hyperparameters and acquisition function optimization can be unstable in high-dimensional spaces, the first $W=20$ rounds are a stabilization period where stopping is prohibited, and a moving average of $W$ rounds is applied to the stopping signal. (ii) **Handling unknown costs**: $\ln c(x)$ is modeled as a GP, and $c(x)$ is replaced by $\mathbb{E}[c(x)]=\exp(\mu_{\ln c}+\sigma_{\ln c}^2/2)$, while maintaining theoretical guarantees.

## Key Experimental Results

### Main Results
The authors evaluate the PBGI/LogEIPC stopping rule against 7 baseline rules across three scenarios: 1D Bayesian regret, 8D Bayesian regret (with uniform, linear, and periodic costs), and AutoML benchmarks (LCBench with 35 datasets and NATS-Bench with 32k architectures).

| Scenario | Dim / Cost | PBGI + Ours | LogEIPC + Ours | UCB-LCB | Convergence | Hindsight (oracle) |
|------|----------|--------------|--------------|---------|-------------|------------|
| Bayesian regret 1D | $\lambda=0.1$ | Near hindsight | Near hindsight | High | High | Lower bound |
| Bayesian regret 8D | linear cost | Almost hindsight | Near hindsight | Significantly worse | Significantly worse | Lower bound |
| LCBench (35 datasets) | $\lambda=10^{-3}$ | Top-3 on ~75% datasets | Top-3 on ~75% datasets | High | Medium | Lower bound |
| NATS-Bench | $\lambda=10^{-5}$ | Near hindsight (except 2 tasks) | Slightly weaker than PBGI | Often reaches 200 iter limit | Medium | Lower bound |

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|--------|------|
| $\alpha_t$ vs $\alpha_{t-1}$ | $\alpha_t$ significantly better | Post-update fair value accurately reflects if it's worth continuing |
| Stabilization + moving average ($W=20$) | More stable in high-dim | Suppresses spurious stops caused by GP hyperparameter oscillations |
| Known vs Unknown cost (GP for $\ln c$) | Similar performance | Validates that guarantees hold when replacing $c(x)$ with $\mathbb{E}[c(x)]$ |
| Cost model misspecification | Graceful degradation | Rule remains robust even if the cost model is biased |

### Key Findings
- Most baseline stopping rules (SRGap-med, UCB-LCB) often **fail to stop before the 200-round limit** on NATS-Bench, whereas the proposed rule terminates before the limit, demonstrating cost-aware termination.
- The "pairing" is important: The proposed rule + PBGI is slightly stronger than + LogEIPC on LCBench, likely due to PBGI's robustness to GP misspecification (Figure 10).
- Even with biased cost measurements (e.g., proxying runtime as a linear function of parameters), the relative ranking of the stopping rule remains unchanged, suggesting high precision in cost measurement is not required.
- The few datasets where performance was lower on LCBench are mostly very small datasets ($<10000$ instances), suggesting issues with val/test distribution mismatch rather than the rule itself.

## Highlights & Insights
- **"Select + Stop" must be paired** for Bayesian optimality—a key property from Weitzman (1979) overlooked by most BO work. This paper elegantly extends it to the correlated GP setting.
- **Two acquisition functions from different origins share the same stopping rule**—PBGI (Gittins index) and LogEIPC (one-step lookahead cost-normalized EI) are shown to have **literally equivalent** stopping conditions via EI monotonicity.
- **"No worse than immediate stopping" is a clever worst-case guarantee**—Though weaker than a direct regret bound, it is sufficient for the cost-adjusted setting as it ensures users are never penalized for using BO.

## Limitations & Future Work
- Theoretical guarantees are restricted to PBGI and LogEIPC. Other acquisition functions like KG or MES (value-of-information / entropy search) would require their own compatible stopping rules.
- High-dimensional spaces require a $W=20$ stabilization window. While technically necessary, this introduces a hyperparameter and might prevent triggering for small-budget tasks ($<50$ total iterations).
- The "no worse than immediate stopping" guarantee is tight in the worst case but does not provide a tighter regret rate for intermediate ranges.
- Assumes cost $c(x)$ is a deterministic function or GP-modellable; the framework may fail if cost depends on $f$'s observed value (e.g., failed training triggering extra cleanup costs).

## Related Work & Insights
- **vs Nguyen et al. (2017) / Zhou et al. (2024) EI thresholding**: They stop when $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$, but $c_0$ is heuristic. This paper proves that replacing $c_0$ with the actual "per-sample cost" is a specific case of the LogEIPC rule under uniform costs.
- **vs UCB-LCB (Makarova et al. 2022)**: UCB-LCB stops based only on confidence width, ignoring cost, leading to over-evaluation in cost-aware scenarios.
- **vs PRB (Wilson, 2024)**: PRB provides a $(1-\delta)$ confidence guarantee for simple regret but ignores costs, often spending more than Hindsight.
- **vs Chick & Frazier (2012) cost-aware stopping**: They addressed cost-aware stopping for independent samples; this work generalizes that logic to correlated GPs via PBGI, making it suitable for BO.

## Rating
- Novelty: ⭐⭐⭐⭐ Successfully ports Weitzman’s stopping rule to cost-aware BO and reveals the equivalence between PBGI/LogEIPC rules.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 1D/8D Bayesian regret, 35 LCBench datasets, and NATS-Bench with consistent Top-3 rankings.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from intuition to equivalent reformulations and theoretical guarantees.
- Value: ⭐⭐⭐⭐⭐ Provides a principled "default stopping rule" for AutoML engineers and the first non-asymptotic bound for cost-adjusted regret.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](../../ICLR2026/optimization/symmetry-aware_bayesian_optimization_via_max_kernels.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](../../ICLR2026/optimization/non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/optimization/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](../../CVPR2026/optimization/dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[ICLR 2026\] Combinatorial Bandit Bayesian Optimization for Tensor Outputs](../../ICLR2026/optimization/combinatorial_bandit_bayesian_optimization_for_tensor_outputs.md)

</div>

<!-- RELATED:END -->
