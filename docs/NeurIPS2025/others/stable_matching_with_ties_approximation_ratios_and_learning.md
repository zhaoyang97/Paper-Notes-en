---
title: >-
  [Paper Note] Stable Matching with Ties: Approximation Ratios and Learning
description: >-
  [NeurIPS 2025][stable matching] This paper studies two-sided matching markets with tied preferences, introduces the Optimal Stable Share (OSS) ratio to measure fairness…
tags:
  - "NeurIPS 2025"
  - "stable matching"
  - "preference ties"
  - "approximation ratio"
  - "OSS-ratio"
  - "multi-armed bandit"
date: 2026-05-08
content_hash: 47b9ecbd362ff552
---

# Stable Matching with Ties: Approximation Ratios and Learning

**Conference**: NeurIPS 2025
**arXiv**: [2411.03270](https://arxiv.org/abs/2411.03270)  
**Code**: Unavailable  
**Area**: Other
**Keywords**: stable matching, preference ties, approximation ratio, OSS-ratio, multi-armed bandit

## TL;DR

This paper studies two-sided matching markets with tied preferences, introduces the Optimal Stable Share (OSS) ratio to measure fairness, proves that the OSS-ratio under distributions over stable matchings is $\Omega(N)$ while under general matching distributions it is $O(\log N)$ (asymptotically tight), and extends the offline approximation results to a bandit learning setting.

## Background & Motivation

Two-sided matching markets (students–schools, doctors–hospitals, workers–jobs) are central problems in economics and algorithm design. The classical Gale–Shapley algorithm produces a worker-optimal stable matching under strict preferences. In practice, however, **workers may be indifferent among certain jobs (tied preferences)**—for example, in the conference paper assignment system TPMS, reviewers naturally produce ties when rating papers using discrete levels such as "Eager" or "Willing."

Tied preferences introduce a fundamental difficulty: different tie-breaking rules yield different stable matchings, each of which may be optimal for a different subset of workers. **No single matching can simultaneously guarantee every worker their optimal stable utility.** This challenge is further compounded in the bandit learning setting—when utility estimates are statistically indistinguishable, the standard regret bound $O(\ln T/\Delta^2)$ diverges as $\Delta \to 0$.

The central question of this paper is: **For markets with tied preferences, can a distribution over matchings approximate each worker's Optimal Stable Share (OSS) while guaranteeing every worker at least a constant fraction of their utility?**

## Method

### Overall Architecture

The paper proceeds in three layers: (1) offline setting with known preferences—tight bounds on the OSS-ratio for different matching classes (stable / internally stable / general); (2) robustness under preference uncertainty—guarantees in the presence of utility estimation errors; (3) bandit learning—online preference learning with low regret.

### Key Designs

1. **OSS-ratio Definition and Lower Bounds (Section 3.1)**: The Optimal Stable Share of worker $w$ is defined as $\mathbf{U}^*(w) = \max_{\mu \in \mathcal{S}} \mathbf{U}(w, \mu(w))$, and the OSS-ratio is $R_\mathcal{C} = \min_{D \in \Delta(\mathcal{C})} \max_{w} \frac{\mathbf{U}^*(w)}{\mathbf{U}_D(w)}$. **Theorem 1**: There exist instances where $R_\mathcal{S} = \Omega(N)$ under distributions over stable matchings—constructed with $N/2$ high-skill and $N/2$ ordinary workers such that each stable matching satisfies only one ordinary worker. **Theorem 2**: Even allowing non-stable matchings, $R_\mathcal{M} = \Omega(\log N)$—established via a recursive construction in which the number of workers grows logarithmically faster than the number of desirable jobs.

2. **Offline Approximation Oracle—Algorithm 1 (Section 3.2)**: **Core technical contribution.** Steps: (a) duplicate each job $m$ times (duplication index); (b) workers rank jobs by utility, breaking ties by preferring lower duplication indices; (c) run the GS algorithm on the expanded market to obtain a stable matching $\tilde{\mu}$; (d) decompose $\tilde{\mu}$ into $m$ original matchings $\tilde{\mu}_1,\ldots,\tilde{\mu}_m$ and select one uniformly at random. **Theorem 3**: Setting $m = \lfloor\log_2 N + 2\rfloor$ guarantees each worker utility $\geq \mathbf{U}^*(w)/m$, i.e., $R_\mathcal{I} = O(\log N)$. The key proof technique constructs a directed forest—nodes represent workers who prefer a stable matching in the expanded market, edges represent conflicts over the same job copy—and uses tree structure together with stability constraints for inductive proof. **Theorem 4**: Algorithm 1 satisfies dominant-strategy incentive compatibility (DSIC).

3. **$\epsilon$-Stability and Robustness (Section 4)**: An $\epsilon$-stable matching is defined such that any blocking pair must yield a utility gain exceeding $\epsilon$. Algorithm 2 generalizes Algorithm 1 and guarantees $\mathbf{U}_D(w) \geq \mathbf{U}_\epsilon^*(w)/m - \epsilon$. **Theorem 6**: For a rectangular uncertainty set $\mathcal{U}$, running Algorithm 2 with center $\hat{\mathbf{U}}$ and width $\epsilon$ guarantees every worker utility $\geq \mathcal{U}^*(w)/m - \epsilon$, directly applicable to settings where the utility matrix is estimated from data (Example 2).

4. **Bandit Learning—ETCO Algorithm (Section 5)**: The $\alpha$-approximate stable regret is defined as $Reg_i^\alpha(T) = \alpha T \cdot \mathbf{U}^*(w_i) - \mathbb{E}[\sum_t X_i(t)]$. **Algorithm 3 (ETCO = Explore-Then-Choose-Oracle)**: In the exploration phase, round-robin assignment builds confidence intervals $[LCB_{i,j}, UCB_{i,j}]$; if all top-$N$ job CIs are non-overlapping → GS oracle (no ties); otherwise → approximation oracle (ties present). **Theorem 7**: When $\Delta_{\min}$ is large, $Reg_i = O(K\ln T/\Delta_{\min}^2)$ (matching the lower bound); when $\Delta_{\min}$ is small, $Reg_i^\alpha = O(\alpha T_0 + T\sqrt{K\ln T/T_0})$. **Theorem 8** (trade-off lower bound): If an algorithm guarantees sublinear regret whenever $\Delta_{\text{rel}} \geq cT^{-1/2+\delta}$, then there exists an instance with $\Delta_{\text{rel}}=0$ for which the approximate regret is $\Omega(T^{1-2\delta})$.

### Loss & Training

In the bandit setting, each round yields a 1-sub-Gaussian reward $X_i(t)$ with mean $\mathbf{U}(w_i, \mu_t(w_i))$. The confidence bound is $UCB_{i,j} = \hat{\mathbf{U}}(i,j) + \sqrt{6\ln T/T_{i,j}}$. The exploration length $T_0$ is the key hyperparameter: $T_0 = T^{2/3}(K\ln T)^{1/3}$ yields optimal approximate regret; $T_0 = T/(2\ln T)$ yields optimal standard regret.

## Key Experimental Results

### Summary of Theoretical Results

| Matching Class $\mathcal{C}$ | OSS-ratio Lower Bound | OSS-ratio Upper Bound | Tightness |
|---|---|---|---|
| Stable matchings $\mathcal{S}$ | $\Omega(N)$ (Thm 1) | $O(N)$ (trivial) | Tight |
| Internally stable $\mathcal{I}$ | $\Omega(\log N)$ (Thm 2) | $O(\log N)$ (Thm 3) | Tight |
| General matchings $\mathcal{M}$ | $\Omega(\log N)$ (Thm 2) | $O(\log N)$ (Thm 3) | Tight |

### Bandit Learning Upper and Lower Bounds

| Regime | Upper Bound (ETCO) | Lower Bound | Remarks |
|---|---|---|---|
| $\Delta_{\min} = \tilde{\Omega}(T^{-1/3})$ | $O(K\ln T/\Delta_{\min}^2)$ | $\Omega(N\ln T/\Delta_{\min}^2)$ [52] | Matching (up to constants) |
| $\Delta_{\min} = \tilde{O}(T^{-1/3})$, $T_0=T^{2/3}(\cdot)^{1/3}$ | $O((K\ln T)^{1/3}T^{2/3})$ | — | Optimal ETC rate |
| Trade-off: sublinear regret when $\Delta_{\text{rel}} \geq cT^{-1/2+\delta}$ | — | $\exists$ instance with $\Delta_{\text{rel}}=0$: $\Omega(T^{1-2\delta})$ approx-regret | Fundamental trade-off |

### Key Findings

- **The $\Omega(N)$ lower bound for stable matching distributions reveals that "a single matching is insufficient"**: broader matching classes are needed to guarantee fairness.
- **$O(\log N)$ is a universal upper bound across matching classes**: the duplication index technique enables GS to produce fair matching distributions.
- **The bandit trade-off lower bound is not merely computational but statistical**—a novel finding in the matching bandit literature.
- **DSIC property**: Algorithm 1 is immune to manipulation by workers misreporting preferences, ensuring mechanism design correctness.

## Highlights & Insights

- The OSS-ratio concept elegantly quantifies "how far each worker is from their optimum," capturing fairness more precisely than maximum matching size.
- The duplication index is the core technical innovation—by replicating jobs and having workers break ties by index, the tied-preference problem is reduced to one with strict preferences.
- The directed forest construction and inductive proof are central to the $O(\log N)$ upper bound, with stability constraints bounding the depth of conflict trees.
- The trade-off lower bound of Theorem 8 has independent significance for bandit theory beyond matching, demonstrating a fundamental statistical rather than computational barrier.

## Limitations & Future Work

- Only one-sided ties (worker side) are considered; the two-sided case where jobs also have tied preferences is substantially harder.
- In Algorithm 1, each worker is assigned a job in only one of the $m$ matchings (with probability $1/m$), remaining unmatched most of the time.
- The bandit learning section considers only centralized assignment; the decentralized setting (workers proposing independently) is an important open problem.
- Extensions to many-to-one or many-to-many matching are not addressed.
- No empirical validation is provided; all results are purely theoretical.

## Related Work & Insights

- Complements Irving (1994)'s notions of weak/strong/super-stability—this paper focuses on fair approximation under weak stability.
- Orthogonal to the DEF1/DMMS fairness concepts of Freeman et al. (2021)—the OSS-ratio is tailored to one-to-one markets.
- Directly compared with Kong et al. (2025) at ICLR—the latter uses pessimistic regret, while this paper adopts the more flexible notion of approximate regret.
- The proof technique for the bandit trade-off lower bound (constructing two instances differing in a single entry) may inspire work on other combinatorial bandit problems.

## Core Results at a Glance

- $R_\mathcal{S} = \Theta(N)$: OSS-ratio under distributions over stable matchings is linear.
- $R_\mathcal{M} = R_\mathcal{I} = \Theta(\log N)$: OSS-ratio under general / internally stable matchings is logarithmic.
- Algorithm 1: duplicate jobs $m$ times + GS → uniform distribution over internally stable matchings, DSIC.
- ETCO: $O(K\ln T/\Delta^2)$ when gap is large; $O(T^{2/3})$ approximate regret when gap is small.
- Trade-off: guaranteeing sublinear regret for large gaps implies the existence of an instance with $\Delta_{\text{rel}}=0$ suffering $\Omega(T^{1-2\delta})$ approximate regret.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the OSS-ratio concept and the duplication index technique are original; the trade-off lower bound is an independent theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ A purely theoretical paper with no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Definitions and theorem statements are clear; proofs are largely deferred to the appendix, though the main text provides sufficient intuition.
- Value: ⭐⭐⭐⭐⭐ Completely resolves the approximation ratio question for matching with tied preferences and establishes a foundational framework for the bandit setting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](../../AAAI2026/others/approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[NeurIPS 2025\] Look-Ahead Reasoning on Learning Platforms](look-ahead_reasoning_on_learning_platforms.md)
- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](adjusted_count_quantification_learning_on_graphs.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
