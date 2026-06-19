---
title: >-
  [Paper Note] Cost-Aware Stopping for Bayesian Optimization
description: >-
  [ICML 2026][Optimization & Theory][Bayesian optimization] The authors generalize Weitzman's Pandora's Box stopping principle to correlated Bayesian Optimization scenarios. They prove that under a shared "acquisition function value exceeding current best" stopping rule, the PBGI and LogEIPC cost-aware acquisition functions yield an expected cost-adjusted simple regret no worse
tags:
  - ICML 2026
  - Optimization & Theory
  - Bayesian optimization
  - cost-aware stopping
  - Pandora's Box
  - Gittins index
  - expected improvement per cost
date: 2026-05-08
content_hash: d495accc71716e42
---
# Cost-Aware Stopping for Bayesian Optimization

**Conference**: ICML2026  
**arXiv**: [2507.12453](https://arxiv.org/abs/2507.12453)  
**Code**: https://github.com/QianJaneXie/CostAwareStoppingBayesOpt  
**Area**: Bayesian Optimization / AutoML / Decision Theory  
**Keywords**: Bayesian optimization, cost-aware stopping, Pandora's Box, Gittins index, expected improvement per cost

## TL;DR
The authors generalize Weitzman's Pandora's Box stopping principle to correlated Bayesian Optimization scenarios. They prove that under a shared "acquisition function value exceeding current best" stopping rule, the PBGI and LogEIPC cost-aware acquisition functions yield an expected cost-adjusted simple regret no worse than "stopping after one sample," providing the first theoretical guarantee for adaptive stopping in the cost-adjusted simple regret setting.

## Background & Motivation

**Background**: Bayesian Optimization (BO) is a dominant framework for expensive black-box objectives $f:X\to\mathbb{R}$. It uses a GP to fit the posterior and an acquisition function $\alpha_t$ to trade off between exploration and exploitation to select the next point $x_{t+1}$. Common acquisition functions include EI, LCB, KG, and TS. When evaluation costs $c(x)$ are significantly heterogeneous (e.g., varying training times for different hyperparameters), the community has developed cost-aware versions. Two SOTA methods are **PBGI** (Pandora's Box Gittins Index, Xie et al. 2024) and **LogEIPC** (log EI per cost, Ament et al. 2023).

**Limitations of Prior Work**: The question of "when to stop" in BO has been largely overlooked. Existing stopping rules are either heuristic (fixed iterations, best value unchanged for several rounds) or based on simple regret convergence criteria (UCB-LCB, PRB, SRGap-med, etc.), but **none explicitly model evaluation costs**. In cost-aware scenarios, these rules often lead to "marginal regret improvements at massive evaluation costs," essentially resulting in cumulative costs far higher than the actual marginal benefit.

**Key Challenge**: Users ultimately want to minimize the cost-adjusted simple regret $\mathcal{R}_c = \mathbb{E}[\min_{1\le t\le\tau} f(x_t)-\inf_{x\in X} f(x) + \sum_{t=1}^\tau c(x_t)]$, but existing stopping rules only focus on the former term while ignoring the latter. Furthermore, even rules like EI thresholding (Nguyen et al. 2017) rely on heuristic thresholds rather than principled derivations.

**Goal**: Design an **explicitly cost-aware**, **theoretically provable**, and **parameter-minimal** adaptive stopping rule that applies naturally to both uniform and heterogeneous cost scenarios and integrates with PBGI or LogEIPC acquisition functions.

**Key Insight**: The authors return to the classic optimal strategy for Weitzman’s Pandora’s Box (1979), where the Gittins-index argument requires the selection policy and stopping time to be **coupled** to maintain Bayesian optimality. By generalizing this stopping condition from an independent discrete setting to a correlated GP setting, a stopping rule for PBGI naturally emerges. Using the monotonicity of EI, this is then shown to be **equivalent** to a stopping rule for LogEIPC, revealing that two seemingly distinct acquisition functions share the same underlying stopping principle.

**Core Idea**: The stopping condition $\min_x \alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is equivalent to $\max_x \alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$, which formalizes the intuition that "no unevaluated point has a fair value better than the current best."

## Method

### Overall Architecture
PBGI treats each candidate point as a "box" in the Pandora's Box problem. Its fair value $\alpha_t^{\mathrm{PBGI}}(x)$ is defined as the threshold where the expected improvement equals the cost: $\mathrm{EI}_{f\mid x_{1:t}, y_{1:t}}(x; \alpha_t^{\mathrm{PBGI}}(x))=c(x)$. Weitzman’s classic argument shows that in independent discrete settings, **selecting** $\arg\min_x \alpha^{\mathrm{PBGI}}(x)$ and **stopping** at $\min_x \alpha^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ is Bayesian optimal. The authors retain this structure in the correlated GP setting: updating $\alpha_t^{\mathrm{PBGI}}$ each round and evaluating the stopping condition.

### Key Designs

**1. Stopping rule based on PBGI: Using updated $\alpha_t$ rather than $\alpha_{t-1}$ to judge if a box is "worth opening"**

Existing stopping rules are often heuristic or ignore evaluation costs, leading to "high spending for tiny gains" in cost-aware settings. Ours stopping condition directly addresses this: $\min_{x\in X\setminus\{x_1,\dots,x_t\}} \alpha_t^{\mathrm{PBGI}}(x) \ge y^*_{1:t}$, meaning "stop when the fair value of all remaining unopened boxes is no better than the current best." A subtle but critical design choice is using the **post-posterior update** $\alpha_t$ rather than the $\alpha_{t-1}$ used in previous theoretical work (Gergatsouli & Tzamos 2023). $\alpha_t^{\mathrm{PBGI}}(x)$ represents the "fair price" of $x$ given all current information; it reflects both uncertainty in $f(x)$ and cost $c(x)$. Only by using the latest $\alpha_t$ can the system accurately decide whether to continue investing. This is necessary because Weitzman's original "selection + stopping" must be paired for Bayesian optimality; using a stale $\alpha_{t-1}$ in correlated GPs causes information lag. Experiments in Section C.2 confirm that using $\alpha_t$ yields significant improvements in cost-adjusted regret.

**2. Deriving equivalent LogEIPC stopping rules via EI monotonicity: One rule for two acquisition functions**

PBGI originates from the Gittins index, while LogEIPC stems from cost-normalized EI. Authors found they share the same stopping condition. Since $\mathrm{EI}_\psi(x;y)$ is strictly monotonically increasing with respect to $y$, $\alpha_t^{\mathrm{PBGI}}(x)\ge y^*_{1:t}$ holds if and only if $\mathrm{EI}_{f\mid x_{1:t},y_{1:t}}(x;y^*_{1:t}) \le c(x)$. Taking the log yields $\max_{x\in X\setminus\{x_1,\dots,x_t\}}\alpha_t^{\mathrm{LogEIPC}}(x;y^*_{1:t})\le 0$. Under uniform cost $c(x)\equiv c_0$, this simplifies to $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$—recovering the EI thresholding rule from Nguyen et al. (2017) and Zhou et al. (2024), but replacing the heuristic threshold with the principled "cost per sample." This equivalence provides two perspectives: the decision-theoretic view from Pandora's Box and the economic view of EI-per-cost, allowing LogEIPC users to apply the same theoretical guarantees.

**3. "No worse than immediate stopping" guarantee for cost-adjusted regret + finite-time termination**

This design provides the first non-asymptotic bound for cost-adjusted regret. The proof proceeds in two steps: Lemma 3.1 shows that before stopping, every selected $x_{t+1}$ satisfies $\alpha_t^{\mathrm{EI}}(x_{t+1})\ge c(x_{t+1})$, meaning the "expected improvement always exceeds the cost." Theorem 3.2 uses this to show:

$$\mathbb{E}\Big[y^*_{1:\tau}-\min_x f(x)+\sum_{t=1}^\tau c(x_t)\Big]\le \mathbb{E}\big[y_1-\min_x f(x)+c(x_1)\big]=U+C,$$

where $U=\mu(x_1)-\mathbb{E}[\min_x f(x)]$ and $C=c(x_1)$. This implies that the worst-case result of using this rule is no worse than "sampling once and stopping immediately"—a "no-regret" guarantee in the cost-adjusted sense. Corollary 3.3 further bounds the expected cumulative cost within $U+C$ and provides a finite-time termination guarantee with probability $1-\delta$ within $\frac{U+C}{\delta c_0}$ steps (when $c(x)\ge c_0>0$). Figures 2 and 3 demonstrate the practical significance of this, showing many baselines actually perform worse than "immediate stopping."

### Loss & Training
No separate training is involved; a single logic check is added to the BO main loop. Two engineering details are included: (i) **Stabilization + Moving Average**: To handle instability in GP hyperparameters and acquisition function optimization in high dimensions, a stabilization period of $W=20$ rounds is used where stopping is prohibited, followed by a $W$-round moving average of the stopping signal. (ii) **Handling Unknown Costs**: $\ln c(x)$ is modeled as a GP, and $c(x)$ is replaced by $\mathbb{E}[c(x)]=\exp(\mu_{\ln c}+\sigma_{\ln c}^2/2)$ in the criteria.

## Key Experimental Results

### Main Results
The authors cross-evaluate PBGI/LogEIPC stopping rules against 7 baseline stopping rules in three scenarios: 1D Bayesian regret, 8D Bayesian regret (with uniform, linear, and periodic costs), and AutoML benchmarks (LCBench with 35 datasets and NATS-Bench with 32k architectures).

| Scenario | Dim / Cost | PBGI + Ours Stopping | LogEIPC + Ours Stopping | UCB-LCB | Convergence | Hindsight (oracle) |
|------|----------|--------------|--------------|---------|-------------|------------|
| Bayesian regret 1D | $\lambda=0.1$ | Near hindsight | Near hindsight | High | High | Lower bound |
| Bayesian regret 8D | linear cost | Near hindsight | Near hindsight | Significantly worse | Significantly worse | Lower bound |
| LCBench (35 datasets) | $\lambda=10^{-3}$ | Top-3 on ~75% datasets | Top-3 on ~75% datasets | High | Medium | Lower bound |
| NATS-Bench | $\lambda=10^{-5}$ | Near hindsight (2 exceptions) | Slightly weaker than PBGI | Often hits 200-round limit | Medium | Lower bound |

### Ablation Study

| Configuration | Key Observation | Mechanism |
|------|--------|------|
| $\alpha_t$ vs $\alpha_{t-1}$ | $\alpha_t$ is significantly better | Fair values updated with posterior genuinely reflect if continuing is worth it |
| stabilization + moving avg ($W=20$) | More stable in high dimensions | Suppresses spurious stops caused by GP hyperparameter oscillations |
| Known vs Unknown Costs (GP for $\ln c$) | Performance is similar | Validates that replacing $c(x)$ with $\mathbb{E}[c(x)]$ maintains guarantees |
| Cost model mismatch (proxy vs actual) | Graceful degradation | Even with biased cost models, the relative ranking of the rule remains robust |

### Key Findings
- Most baseline stopping rules (SRGap-med, UCB-LCB) on NATS-Bench **repeatedly fail to stop before the 200-round limit**, whereas Ours rule successfully terminates early, demonstrating cost-awareness.
- The "partner relationship" matters: Ours stopping rule + PBGI is slightly stronger than + LogEIPC on LCBench, likely due to PBGI's robustness under GP misspecification.
- Even when cost measurements are biased (e.g., modeling runtime as a linear function of parameters), the relative ranking of the stopping rule holds.
- Poor performance on a few LCBench datasets is attributed to tiny datasets (instances < 10,000) where distribution mismatch between validation/test is the likely culprit.

## Highlights & Insights
- **"Selection + Stopping" must be coupled**: This key property from Weitzman (1979) is often ignored in BO. Ours reintroduces it and elegantly generalizes it from discrete Pandora's Boxes to correlated GP settings.
- **Equivalent rules for different origins**: Revealing that PBGI (Gittins index) and LogEIPC (one-step lookahead) share an identical stopping condition allows users to switch acquisition functions without changing the stopping logic.
- **"No worse than immediate stopping" is a clever guarantee**: While theoretically weaker than a regret bound, it is perfectly suited for cost-adjusted settings as it ensures the user is never "harmed" by using the BO framework.

## Limitations & Future Work
- Theoretical guarantees are restricted to PBGI and LogEIPC; acquisition functions based on value-of-information (KG) or entropy search (MES) would require their own derived stopping rules.
- High-dimensional settings require a $W=20$ stabilization window. While necessary for engineering, this introduces a hyperparameter; for small-budget tasks (< 50 iterations), this window might prevent the rule from ever triggering.
- The "no worse than immediate stopping" bound is tight in worst-case scenarios but does not provide a tighter regret rate for intermediate intervals.
- Assumes cost $c(x)$ is a deterministic function or can be modeled by a GP; the framework may fail if cost depends dynamically on the observed $f$ value (e.g., failed training triggering cleanup costs).

## Related Work & Insights
- **vs Nguyen et al. (2017) / Zhou et al. (2024) EI thresholding**: They stop when $\max_x \alpha_t^{\mathrm{EI}}(x)\le c_0$, but $c_0$ is heuristic. Ours proves that substituting the true "per-sample cost" for $c_0$ makes this a special case of LogEIPC stopping under uniform costs.
- **vs UCB-LCB (Makarova et al. 2022)**: UCB-LCB stops based on confidence width irrespective of cost; Ours explicitly incorporates $c(x)$ into the condition.
- **vs PRB (Wilson, 2024)**: PRB provides a $(1-\delta)$ confidence guarantee for simple regret but ignores cost, often spending significantly more than the hindsight oracle.
- **vs Chick & Frazier (2012) cost-aware stopping**: They addressed cost-aware stopping for independent sampling in finite domains; Ours generalizes this through PBGI to correlated GPs.

## Rating
- Novelty: ⭐⭐⭐⭐ Successfully ports Weitzman’s stopping rule to cost-aware BO and discovers the functional equivalence between PBGI/LogEIPC.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 1D/8D Bayesian regret, LCBench, and NATS-Bench with consistent top-tier ranking.
- Writing Quality: ⭐⭐⭐⭐ Clear progression (intuition → equivalence → theory) and consistent notation.
- Value: ⭐⭐⭐⭐⭐ Provides a principled "default stopping rule" for AutoML practitioners and provides the first non-asymptotic cost-adjusted regret bound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](../../CVPR2026/optimization/dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/optimization/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)

</div>

<!-- RELATED:END -->
