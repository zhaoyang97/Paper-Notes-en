---
title: >-
  [Paper Note] Thompson Sampling for Multi-Objective Linear Contextual Bandit
description: >-
  [NeurIPS 2025][Reinforcement Learning][Thompson Sampling] This paper proposes MOL-TS—the first multi-objective linear contextual bandit Thompson Sampling algorithm with worst-case Pareto regret guarantees. By introducing…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Thompson Sampling"
  - "Multi-Objective Optimization"
  - "Contextual Bandit"
  - "Pareto Optimality"
  - "Effective Pareto Front"
date: 2026-05-08
content_hash: 1ab00a1006d9f728
---

# Thompson Sampling for Multi-Objective Linear Contextual Bandit

**Conference**: NeurIPS 2025
**arXiv**: [2512.00930](https://arxiv.org/abs/2512.00930)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Thompson Sampling, Multi-Objective Optimization, Contextual Bandit, Pareto Optimality, Effective Pareto Front

## TL;DR
This paper proposes MOL-TS—the first multi-objective linear contextual bandit Thompson Sampling algorithm with worst-case Pareto regret guarantees. By introducing the concept of "effective Pareto optimal arms" and an optimistic sampling strategy, MOL-TS achieves a regret upper bound of $\widetilde{O}(d^{3/2}\sqrt{T})$, with the number of objectives $L$ contributing only an $O(\log L)$ factor.

## Background & Motivation

**Practical Demand for Multi-Objective Bandits**: In real-world applications such as recommendation systems, online advertising, and resource allocation, decision-makers frequently need to optimize multiple potentially conflicting objectives simultaneously (e.g., click-through rate, user satisfaction, revenue). Multi-Objective Multi-Armed Bandits (MOMAB) address such trade-offs through the Pareto optimality framework—pulling an arm yields a reward vector rather than a scalar, and no single "best" arm exists.

**UCB-Dominated Theoretical Landscape**: All existing MOMAB algorithms with Pareto regret guarantees are based on UCB policies (e.g., ParetoUCB, MOGLM-UCB). Their theoretical analyses are relatively straightforward because the deterministic upper bounds in UCB naturally ensure optimism. In the single-objective setting, however, Thompson Sampling (TS) has become one of the most popular algorithms, celebrated for its elegant stochastic exploration and superior empirical performance (often surpassing UCB). Extending TS to the multi-objective setting has long been recognized as a fundamental theoretical challenge.

**Core Technical Difficulty in Multi-Objective TS**: Worst-case regret analysis for TS relies on the property that sampled parameters are optimistic with at least a constant probability $\tilde{p}$. In the single-objective case, this probability is constant per sample. In the multi-objective setting, all $L$ objectives must be simultaneously optimistic; naïve TS (one sample per objective, $M=1$) yields a joint optimism probability of $\tilde{p}^L$—decaying **exponentially** in $L$—causing regret to grow exponentially with the number of objectives, rendering theoretical guarantees meaningless.

**Limitations of Standard Pareto Optimality**: The authors also identify a deficiency in the definition of Pareto regret itself: a policy that randomly selects arms solely from the standard Pareto front can achieve zero Pareto regret, yet its cumulative reward may be substantially worse than another policy with the same zero regret. This occurs because standard Pareto optimality only compares pairwise dominance between individual arms, ignoring the potentially superior cumulative performance achievable through mixed selections (convex combinations). This deficiency becomes increasingly pronounced over long horizons.

## Method

### Overall Architecture
MOL-TS is a multi-objective linear Thompson Sampling algorithm based on regularized maximum likelihood estimation. At each round $t$: (1) $M$ parameter vectors are independently sampled for each objective $\ell$; (2) optimistic reward estimates are computed per objective by taking the maximum over samples; (3) the effective Pareto front is constructed and one arm is selected uniformly at random from it; (4) the reward vector is observed and parameter estimates are updated.

### Key Designs

1. **Effective Pareto Optimal Arm**:

    - **Function**: Defines a stricter notion of optimality than standard Pareto optimality, ensuring that repeatedly selecting an arm yields cumulative rewards that are also Pareto optimal.
    - **Mechanism**: Standard Pareto optimality requires that arm $a^*$'s reward vector $\boldsymbol{\mu}_{a^*}$ is not dominated by any **single** arm. Effective Pareto optimality requires that $\boldsymbol{\mu}_{a^*}$ is not dominated by **any convex combination** of other arms: "$\forall \beta \in \mathcal{S}^{|\mathcal{A}|-1}: \boldsymbol{\mu}_{a^*} \not\prec \sum_{a \neq a^*} \beta_a \boldsymbol{\mu}_a$". The effective Pareto front $\mathcal{C}^*$ is thus a subset of the standard Pareto front $\mathcal{P}^*$. A key theorem establishes a duality: an arm is effective Pareto optimal $\Leftrightarrow$ there exists a weight vector $\boldsymbol{w}$ under which it is optimal with respect to the linear scalarization.
    - **Design Motivation**: Through a concrete counterexample, the paper demonstrates the inadequacy of standard Pareto regret—two policies may both select only Pareto optimal arms and achieve zero regret, yet their cumulative rewards differ significantly (e.g., $[1.3, 1.3]$ vs. $[1.6, 1.7]$). Effective Pareto optimality guarantees that no mixed strategy can dominate performance over long horizons.

2. **Optimistic Sampling Strategy**:

    - **Function**: Resolves the core challenge of exponentially decaying joint optimism probability in multi-objective TS by drawing multiple samples per objective and taking the maximum.
    - **Mechanism**: For each objective $\ell$, $M$ parameter vectors $\tilde{\theta}_{t,1}^{(\ell)}, \ldots, \tilde{\theta}_{t,M}^{(\ell)}$ are independently sampled from the Gaussian posterior $\mathcal{N}(\hat{\theta}_t^{(\ell)}, c^2 V_t^{-1})$, and the optimistic reward estimate is $\tilde{\mu}_{t,a}^{(\ell)} = \max_{m} x_{t,a}^\top \tilde{\theta}_{t,m}^{(\ell)}$. When $M \geq 1 - \frac{\log L}{\log(1-\tilde{p})}$ (i.e., $M = O(\log L)$), the probability that all objectives are simultaneously optimistic remains a constant $p \geq 0.15$.
    - **Design Motivation**: With $M=1$, the joint optimism probability is $\tilde{p}^L$ (exponential decay); with $M=O(\log L)$, it becomes $(1-(1-\tilde{p})^M)^L$ (constant). This is because the probability that at least one of $M$ independent samples for each objective is optimistic is $1-(1-\tilde{p})^M$, which approaches 1 as $M$ grows sufficiently large, keeping the $L$-fold product at a constant. This design trades $O(M \cdot L) = O(L \log L)$ sampling cost for an exponential improvement in theoretical guarantees.

3. **Estimation of Effective Pareto Front and Arm Selection**:

    - **Function**: Constructs an empirical effective Pareto front from optimistic reward estimates and uniformly selects one arm at random from it.
    - **Mechanism**: True rewards are replaced by optimistic estimates $\tilde{\boldsymbol{\mu}}_{t,a}$ to construct the estimated effective Pareto front $\tilde{\mathcal{C}}_t = \{a \in \mathcal{A} \mid \tilde{\boldsymbol{\mu}}_{t,a} \not\prec \sum_{a'} \beta_{a'} \tilde{\boldsymbol{\mu}}_{t,a'}, \forall \beta \in \mathcal{S}^{|\mathcal{A}|}\}$. Unlike UCB-based methods that require explicit computation of the empirical Pareto front followed by arm selection, MOL-TS only needs to select uniformly at random from the front, yielding higher computational efficiency.
    - **Design Motivation**: By the duality of Theorem 1, selecting from the effective Pareto front is equivalent to optimizing a linear scalarization under some random weight vector $\boldsymbol{w}_t$, eliminating the need to enumerate all weight directions.

### Loss & Training
Parameter estimation uses regularized least squares (RLS): $V_t = \sum_{s=1}^{t-1} x_{s,a_s} x_{s,a_s}^\top + \lambda I$ and $\hat{\theta}_t^{(\ell)} = V_t^{-1} \sum_{s=1}^{t-1} x_{s,a_s} r_{s,a_s}^{(\ell)}$. All objectives share the same gram matrix $V_t$ each round (since the same arm's context is observed), while separate cumulative reward sums $Z_t^{(\ell)}$ are maintained per objective. The hyperparameter $\lambda$ is the regularization coefficient, and $c$ controls sampling variance, typically set consistent with the confidence interval width $c_{1,t}(\delta)$. Key theoretical parameter settings: confidence interval width $c_{1,t}(\delta) = R\sqrt{d\log\frac{1+(t-1)/(\lambda d)}{\delta/L}} + \lambda^{1/2}$, where the $O(\sqrt{\log L})$ dependence is unavoidable (required to hold jointly across all objectives); the number of samples $M = \lceil 1 - \frac{\log L}{\log(1-p)}\rceil$ (a constant-order quantity when $p=0.15$), ensuring that the optimism probability does not decay exponentially with $L$.

## Key Experimental Results

### Main Results
**Setting**: $K=50$ arms, $d=5$ dimensions, $L=4$ objectives, $T=10000$ rounds, 10 independent instances.

| Algorithm | Pareto Regret | Effective Pareto Regret | Obj1 Cum. Reward | Obj2 Cum. Reward |
|-----------|--------------|------------------------|-----------------|-----------------|
| **MOL-TS ($M=O(\log L)$)** | **Lowest** | **Lowest** | **Highest** | **Highest** |
| MOL-UCB | Moderate | Moderate | Moderate | Moderate |
| MOL-$\epsilon$-Greedy | Highest | Highest | Mixed (some objectives higher but inconsistent) | Mixed |
| MOL-TS ($M=1$) | Slightly above $M=O(\log L)$ | Slightly higher | Slightly lower | Slightly lower |

### Ablation Study

| Configuration | Pareto Regret | Notes |
|---------------|--------------|-------|
| MOL-TS, $M=1$ | Higher | Naïve TS; joint optimism probability decays exponentially |
| **MOL-TS, $M=O(\log L)$** | **Lowest** | Optimistic sampling restores constant probability |
| MOL-UCB | Higher than MOL-TS | UCB is deterministic but less flexible than TS |

### Key Findings
- MOL-TS outperforms MOL-UCB and $\epsilon$-Greedy on both Pareto regret and Effective Pareto Regret.
- Selection from the effective Pareto front yields measurably higher cumulative rewards—MOL-TS achieves the highest cumulative reward across all 4 objectives.
- The $M = O(\log L)$ optimistic sampling strategy proves empirically necessary—the performance of $M=1$ is noticeably inferior to $M=O(\log L)$.
- MOL-$\epsilon$-Greedy exhibits a counterintuitive phenomenon: despite having the highest Pareto regret, it may yield higher instantaneous rewards on certain objectives; this is because it does not differentiate the effective Pareto front, and random selection causes cumulative rewards to be diluted by averaging.

## Highlights & Insights
- The concept of **effective Pareto optimality** addresses a long-overlooked issue: zero standard Pareto regret does not guarantee optimal cumulative performance. This new definition, grounded in convex-combination dominance, precisely characterizes the strict optimality condition of "not being dominated by any mixed strategy over repeated selections." Intuitively, the effective Pareto front corresponds to the "boundary of the convex hull" of the Pareto front—only points on this boundary are worth selecting in the long run.
- **Elegant solution via optimistic sampling**: $O(\log L)$ additional samples per objective restore the exponentially decaying joint optimism probability to a constant, achieving an exponential theoretical improvement at negligible computational cost (sampling complexity increases only from $O(L)$ to $O(L \log L)$). This "logarithmic cost for exponential gain" technique recurs throughout computer science (e.g., the $\log$ factor in union bounds); the authors' concrete instantiation in the bandit context is particularly elegant.
- **Elegance of the duality theorem**: Theorem 1 establishes a perfect correspondence between effective Pareto optimal arms and linear scalarization-optimal arms (a classical result from multi-objective optimization). This not only simplifies algorithm design (selecting from the effective Pareto front $\leftrightarrow$ optimizing a scalarized objective under some weight vector) but also provides a key decomposition tool for regret analysis—reducing multi-objective regret to a weighted sum of single-objective regrets.

## Limitations & Future Work
- Computing the effective Pareto front requires solving convex hull-related problems; computational overhead increases when the number of arms $|\mathcal{A}|$ or objectives $L$ is large, affecting real-time applicability.
- The experimental setup is relatively simple ($K=50, d=5, L=4$), lacking large-scale or high-dimensional empirical validation and evaluation in real-world applications such as recommendation systems.
- The $d^{3/2}$ dependence in the regret bound may not be optimal—certain single-objective TS variants (e.g., OFUL-TS) achieve $d\sqrt{T}$; improving the $d^{3/2}$ factor remains an open problem.
- The analysis assumes independent sub-Gaussian noise across objectives, whereas objectives in practice are often positively or negatively correlated; leveraging correlation structure to improve bounds is a natural extension.
- The concept of the effective Pareto front is defined with respect to mean reward vectors; extension to risk-sensitive settings (concerned with variance or tail risk) requires further development.

## Related Work & Insights
- **vs. MOGLM-UCB (lu2019moglb)**: Also a multi-objective algorithm for linear contextual bandits, but UCB-based methods must explicitly compute the empirical Pareto front each round before selecting an arm, incurring high computational complexity and less flexible exploration than TS. MOL-TS implicitly explores the Pareto front through random sampling and selects uniformly from it, achieving greater efficiency.
- **vs. Hypervolume scalarization (zhang2024)**: Uses random scalarization to explore the entire Pareto front; analysis of Bayes regret requires assuming a known Gaussian prior, and the bound grows with $L$. MOL-TS analyzes worst-case (frequentist) regret with only $O(\log L)$ dependence on $L$.
- **vs. Single-objective TS (abeille2017linear)**: MOL-TS inherits the anti-concentration inequalities and confidence interval framework of single-objective TS, but resolves the exponential decay of joint optimism probability in the multi-objective extension via the optimistic sampling strategy ($M=O(\log L)$ samples with max aggregation). Single-objective TS is a special case with $M=1, L=1$.
- **vs. ParetoUCB (drugan2013momab)**: The earliest MOMAB algorithm, which introduced the basic concept of Pareto regret. MOL-TS not only inherits Pareto regret analysis but also corrects the theoretical deficiency of standard Pareto regret through the new concept of effective Pareto optimality.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First worst-case regret analysis for multi-objective TS, combined with the new concept of effective Pareto optimality; dual theoretical contributions fill an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments validate the theoretical claims but use a relatively simple setup, lacking large-scale validation and real-world application scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous, counterexamples and intuitions are well-presented, though the notation system is somewhat heavy.
- **Value**: ⭐⭐⭐⭐ — Fills an important gap in multi-objective bandit theory; the duality theorem and optimistic sampling technique offer broad inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Thompson Sampling in Function Spaces via Neural Operators](thompson_sampling_in_function_spaces_via_neural_operators.md)
- [\[NeurIPS 2025\] DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads](dccluster-opt_benchmarking_dynamic_multi-objective_optimization_for_geo-distribu.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)

</div>

<!-- RELATED:END -->
