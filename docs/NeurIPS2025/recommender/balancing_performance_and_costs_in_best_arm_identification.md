---
title: >-
  [Paper Note] Balancing Performance and Costs in Best Arm Identification
description: >-
  [NeurIPS 2025][Recommender Systems][Best Arm Identification] This paper proposes to reformulate Best Arm Identification (BAI) from the fixed-budget/fixed-confidence paradigm into a risk functional minimization problem over misidentification probability (or simple regret) plus sampling cost. It derives lower bounds exhibiting a phase transition phenomenon (when the gap is too small, the optimal strategy is to guess directly), and designs the DBCARE algorithm that achieves optimality within logarithmic factors under a dynamic budget.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - Best Arm Identification
  - Cost-Performance Tradeoff
  - Phase Transition
  - Dynamic Budget
  - Risk Functional
date: 2026-05-08
content_hash: 4a08146531999bf7
---

# Balancing Performance and Costs in Best Arm Identification

**Conference**: NeurIPS 2025
**arXiv**: [2505.20583](https://arxiv.org/abs/2505.20583)
**Code**: To be confirmed
**Area**: Multi-Armed Bandits / Best Arm Identification
**Keywords**: Best Arm Identification, Cost-Performance Tradeoff, Phase Transition, Dynamic Budget, Risk Functional

## TL;DR
This paper proposes to reformulate Best Arm Identification (BAI) from the fixed-budget/fixed-confidence paradigm into a risk functional minimization problem over misidentification probability (or simple regret) plus sampling cost. It derives lower bounds exhibiting a phase transition phenomenon (when the gap is too small, the optimal strategy is to guess directly), and designs the DBCARE algorithm that achieves optimality within logarithmic factors under a dynamic budget.

## Background & Motivation

**Background**: Classical BAI operates under two paradigms — fixed-budget (minimize error rate given $T$ samples) and fixed-confidence (minimize sample count given $\delta$). Practical settings such as A/B testing require simultaneous tradeoffs between accuracy and sampling cost.

**Limitations of Prior Work**: Both paradigms require pre-specifying a constraint ($T$ or $\delta$), yet in practice neither is known a priori and they are mutually coupled. Choosing the wrong $T$ wastes resources; choosing the wrong $\delta$ yields bounds that are either too loose or too tight.

**Key Challenge**: More sampling improves accuracy but increases cost — a unified framework is needed to find the optimal balance along a continuous tradeoff.

**Goal**: Define a new risk functional $\mathcal{R} = \text{performance loss} + c \cdot E[\tau]$, derive optimal lower bounds, and design matching algorithms.

**Key Insight**: Two risk functionals are considered — $\mathcal{R}_{MI}$ (misidentification probability + cost) and $\mathcal{R}_{SR}$ (simple regret + cost) — each admitting distinct optimal strategies and phase transition behavior.

**Core Idea**: Unify BAI as risk functional minimization → derive lower bounds with phase transitions → DBCARE dynamic-budget elimination algorithm matches the lower bounds to logarithmic factors.

## Method

### Overall Architecture
**Risk Functional Definition**: $\mathcal{R}_{MI} = P(\hat{I} \neq I^*) + c \cdot E[\tau]$, or $\mathcal{R}_{SR} = E[\mu^* - \mu_{\hat{I}}] + c \cdot E[\tau]$. **Lower Bound Derivation**: Instance-dependent lower bounds are derived for both the 2-arm and $K$-arm settings as functions of $\Delta$ (gap) and $H$ (complexity), revealing a phase transition. **DBCARE Algorithm**: A dynamic-budget elimination method — all surviving arms are pulled each epoch, Hoeffding confidence bounds are used for elimination, and the budget adapts as the surviving set shrinks.

### Key Designs

1. **Discovery of the Phase Transition**:

    - Function: Derive risk lower bounds and identify that the optimal strategy is random guessing when the arm gap falls below a critical threshold.
    - Mechanism: The 2-arm lower bound for $\mathcal{R}_{MI}$ equals $\frac{\sigma^2 c}{4\Delta^2}\log(\frac{e\Delta^2}{\sigma^2 c})$ when $\Delta \geq \sqrt{\sigma^2 c}$, and degrades to $1/4$ (random guessing) when $\Delta < \sqrt{\sigma^2 c}$. The phase transition point $\Delta^* = \sqrt{\sigma^2 c}$ depends on noise level and per-sample cost.
    - Design Motivation: A fundamental theoretical insight — when the gap is sufficiently small, the cost of sampling outweighs the benefit in accuracy.

2. **DBCARE Algorithm (Dynamic Budget Successive Elimination)**:

    - Function: Adaptively balance sampling and error without prior knowledge of $\Delta$.
    - Mechanism: Maintain a surviving arm set $S$; pull each arm in $S$ once per epoch. Eliminate arm $k$ when $\max_\ell \hat{\mu}_\ell - \hat{\mu}_k > e_n$. Dynamic budget: for $\mathcal{R}_{MI}$, $N^*(k) = (kec)^{-1}$; for $\mathcal{R}_{SR}$, $N^*(k) = \frac{3}{2e}\sigma^{2/3}((k-1)c)^{-2/3}$.
    - Design Motivation: Fixed budgets provide no principled stopping criterion; DBCARE dynamically adjusts based on the number of surviving arms and the cost parameter.

3. **Upper Bound Matching (Optimal to Logarithmic Factors)**:

    - Function: Prove that the risk of DBCARE matches the lower bound up to logarithmic factors.
    - Mechanism: 2-arm $\mathcal{R}_{MI}$ upper bound $\leq 32\text{LB}(\Delta) + 2c$; worst-case $\mathcal{R}_{SR} \leq 9\text{LB}^* + 3c$.
    - Design Motivation: No algorithm can achieve essentially better guarantees in theory.

### Loss & Training
- Pure theory with simulation validation and real-world drug discovery experiments.
- $\sigma$-sub-Gaussian distribution assumption.

## Key Experimental Results

### Main Results (2-arm Gaussian, $\sigma^2=1, c=10^{-4}$)

| Method | Small $\Delta$ | Medium $\Delta$ | Large $\Delta$ | Uniformly Competitive |
|--------|--------------|----------------|----------------|----------------------|
| Fixed-Budget $T=10$ | Poor | Medium | Good | ✗ |
| Fixed-Budget $T=500$ | Good | Good | Poor | ✗ |
| Fixed-Confidence $\delta=0.1$ | Poor | Medium | Poor | ✗ |
| **DBCARE** | **Good** | **Good** | **Good** | **✓** |

### Drug Discovery Experiment (5 arms, 237 patients, binary and ordinal outcomes)
- DBCARE is the most consistently competitive across all costs $c \in [10^{-3}, 10^{-5}]$.
- No method dominates at all values of $c$, but DBCARE is never the worst.

### Key Findings
- The phase transition is clearly observable — all methods degrade to random guessing when $\Delta$ is too small.
- DBCARE is the only globally competitive method.
- Matches the oracle strategy to a constant factor.

## Highlights & Insights
- **The phase transition discovery is profound**: there exists a critical gap below which sampling is not worthwhile — this directly guides resource allocation in A/B testing.
- **The unified BAI framework** eliminates the artificial choice between fixed-budget and fixed-confidence formulations.
- **DBCARE's dynamic budget** is elegantly simple — a single formula $N^*(k)$ achieves adaptivity.

## Limitations & Future Work
- Restricted to $\sigma$-sub-Gaussian distributions.
- A logarithmic gap between upper and lower bounds remains for the $K$-arm setting.
- The simple regret lower bound may be improvable as $H \to \infty$.

## Related Work & Insights
- **vs Fixed-Budget BAI**: Requires pre-specifying $T$; DBCARE is self-stopping.
- **vs Fixed-Confidence BAI**: Requires pre-specifying $\delta$; DBCARE is governed by $c$.
- **vs Thompson Sampling**: TS optimizes cumulative regret; this work targets one-shot identification with cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified framework + phase transition constitute an important theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real drug discovery data.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorem structure is clear and well-organized.
- Value: ⭐⭐⭐⭐ Provides a theoretically optimal framework for practical A/B testing.
- Theoretical work; no training involved. Key parameters of DBCARE are the budget function $N^*$ and the elimination threshold.

## Key Experimental Results

### Simulation Experiments

| Setting | DBCARE Risk | Fixed-Budget | Fixed-Confidence | Remarks |
|---------|-------------|-------------|-----------------|---------|
| Easy (large gap) | **Lowest** | Over-samples | Competitive | DBCARE stops early |
| Hard (small gap) | **Lowest** | Insufficient budget | Over-samples | DBCARE adapts |
| Very hard (tiny gap) | **Lowest** | Fails significantly | Grossly over-samples | DBCARE opts to guess |

### Drug Discovery Experiment
Validated on a real-world drug discovery dataset: DBCARE consistently outperforms fixed-budget/confidence methods across varying cost $c$.

### Ablation Study

| Configuration | Performance | Remarks |
|---------------|-------------|---------|
| Fixed-budget + optimal $T$ | Optimal $T$ unattainable | Requires knowledge of gap |
| Fixed-confidence + optimal $\delta$ | Same issue | Requires knowledge of gap |
| **DBCARE** | Adaptively optimal | No parameter pre-specification required |

### Key Findings
- **Phase transition is a distinctive phenomenon of this paradigm**: guessing is optimal when the gap is too small — this has no counterpart in fixed-budget/confidence settings.
- **DBCARE performs well on both easy and hard instances**: fixed-budget methods waste samples on easy problems; fixed-confidence methods over-sample on hard ones.
- **Drug discovery is a natural application domain**: each arm pull carries a real monetary cost.
- **Minimax optimality**: DBCARE's worst-case risk matches the minimax lower bound for any algorithm.

## Highlights & Insights
- **"Risk = penalty + cost" is a natural modeling choice** — analogous to an advertiser's profit maximization, and more practically grounded than fixed-budget/confidence formulations.
- **The phase transition reveals the value of "knowing when to quit"** — when the gap is too small, the optimal strategy is to forgo sampling and guess directly.
- **DBCARE's adaptive budget function** elegantly unifies the strengths of fixed-budget and fixed-confidence approaches.

## Limitations & Future Work
- An additive gap between upper and lower bounds for $\mathcal{R}_{SR}$ remains as $H \to \infty$.
- The approach assumes $\sigma$-sub-Gaussian distributions with known $\sigma$ and $B$.
- Only expected risk is considered; high-probability guarantees are not discussed.
- The cost $c$ is assumed to be a known constant.

## Related Work & Insights
- **vs Fixed-Budget BAI (Audibert et al.)**: No pre-specified $T$ is required; DBCARE stops adaptively.
- **vs Fixed-Confidence BAI (Jamieson et al.)**: No pre-specified $\delta$ is required; DBCARE adapts its confidence intervals.
- **vs Bayesian Sequential Testing (Arrow/Wald)**: Extended to the $K$-arm, non-parametric setting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ An entirely new BAI paradigm + phase transition discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real experiments + comparison with classical methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clear motivating examples.
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both BAI theory and practice.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[NeurIPS 2025\] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)
- [\[NeurIPS 2025\] PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders](pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)

<!-- RELATED:END -->
