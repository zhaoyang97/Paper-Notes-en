---
title: >-
  [Paper Note] Non-Clairvoyant Scheduling with Progress Bars
description: >-
  [NeurIPS 2025][Non-clairvoyant scheduling] This paper introduces a "progress bar" information model as an interpolation framework between clairvoyant and non-clairvoyant scheduling. It designs scheduling algorithms with optimal consistency–robustness tradeoffs for both adversarial and stochastic progress bars, while advancing the theoretical frontier of learning-augmented scheduling.
tags:
  - NeurIPS 2025
  - Non-clairvoyant scheduling
  - progress bars
  - competitive ratio
  - learning-augmented algorithms
  - exploration-exploitation tradeoff
date: 2026-05-08
content_hash: e2a2c9aa42cfc6a5
---

# Non-Clairvoyant Scheduling with Progress Bars

**Conference**: NeurIPS 2025
**arXiv**: [2509.19662](https://arxiv.org/abs/2509.19662)
**Code**: Unavailable
**Area**: Online Scheduling Algorithms / Learning-Augmented Algorithms / Competitive Analysis
**Keywords**: Non-clairvoyant scheduling, progress bars, competitive ratio, learning-augmented algorithms, exploration-exploitation tradeoff

## TL;DR

This paper introduces a "progress bar" information model as an interpolation framework between clairvoyant and non-clairvoyant scheduling. It designs scheduling algorithms with optimal consistency–robustness tradeoffs for both adversarial and stochastic progress bars, while advancing the theoretical frontier of learning-augmented scheduling.

## Background & Motivation

The classical single-machine scheduling problem involves $n$ jobs each with processing time $p_j$, with the objective of minimizing total completion time $\sum C_j$. In the clairvoyant setting, the optimal policy is SPT (Shortest Processing Time first); in the non-clairvoyant setting, the optimal policy is Round-Robin with a competitive ratio of 2 (known to be optimal). The information gap between these two extremes is substantial.

Learning-augmented algorithms (algorithms with predictions) attempt to bridge this gap using predicted information, but existing models provide only **static a priori predictions** (based on job IDs, etc.) that cannot improve dynamically during execution. This is unnatural in settings such as profiling—as a job executes, estimates of its processing time should become increasingly accurate.

The central question of this paper is: **Does there exist an information model that allows an algorithm to dynamically improve its decisions during execution?** This motivates the introduction of the progress bar model—while processing a job, the scheduler receives an estimate of the fraction completed (possibly inaccurate). This naturally interpolates between the clairvoyant setting ($\varphi_j: x \mapsto x$) and the non-clairvoyant setting ($\varphi_j: x \mapsto \mathbb{1}(x=1)$).

## Method

### Overall Architecture

The paper proceeds in increasing order of information model complexity: (1) single-signal progress bars (granularity $g=1$)—a job emits one signal after processing $\alpha$ fraction; (2) untrusted progress bars (granularity $g$)—multi-level signals with an algorithm combining strategy; (3) stochastic progress bars—exploration-exploitation under a Poisson model.

### Key Designs

1. **Single Signal — Trusted Signal (Section 2.1)**: Each job emits a signal exactly after $\alpha$ fraction of its processing time. Algorithm: Run Round-Robin until some job emits a signal, then prioritize that job until completion. **Theorem**: The competitive ratio is $1+\alpha$, which is also optimal for randomized algorithms. Design Motivation: The signal divides scheduling into two phases—before the signal all jobs are processed in parallel (indistinguishable), and after the signal execution switches to prioritizing the signaling job (exploiting new information).

2. **Single Signal — Untrusted Signal (Algorithm 1)**: The signal may arrive at $\beta_j \neq \alpha$. The core difficulty is that naively trusting the signal yields $(1+\alpha)$-consistency but lacks robustness (Lemma), and naive robustification (time-sharing) cannot achieve perfect consistency. **Algorithm 1** introduces a parameter $\rho \in (0,1]$: upon receiving a signal, the algorithm prioritizes job $j$ for $(1/\alpha\rho - 1) \cdot e_j(t)$ additional time units (where $e_j(t)$ is the elapsed processing time). If the job is not completed, it switches to SETF (Shortest Elapsed Time First) until catching up. **Theorem**: The algorithm is $(1+\alpha)$-consistent and $(1+1/\rho\alpha)$-robust, with smoothness bound $\text{ALG} \leq (1+\alpha)\text{OPT} + \frac{2n}{\rho(1-\rho)\alpha^2}\sum_i |\beta_i - \alpha|p_i$. The parameter $\rho$ controls a **smoothness–robustness tradeoff** (rather than the usual consistency–robustness tradeoff).

3. **Algorithm Combining Strategy (Algorithm 2, Section 3)**: Given $g$ scheduling algorithms $A^{(1)},\ldots,A^{(g)}$ (each possibly using different predictions), the goal is to select the best one. Approach: Randomly sample $m$ pairs of jobs and run them to completion; compute the empirical delay of each algorithm; schedule the remaining jobs using the algorithm with minimum empirical delay. **Theorem**: $\mathbb{E}[\text{ALG}] \leq \min_h A^{(h)} + \frac{9}{4}n^{5/3}(\log g)^{1/3} \max_i p_i$. When $\max_i p_i = o(n^{-5/3}\text{OPT})$, the regret term is $o(\text{OPT})$. This generalizes the permutation-prediction combining of Eliás et al. (2024) to arbitrary algorithms with computable delays.

4. **Stochastic Progress Bars (Algorithm 3, Section 4)**: Progress bar jumps are generated by a Poisson process, creating a structure analogous to multi-armed bandits. **Repeated Explore-then-Commit Algorithm**: Run Round-Robin until some job reaches progress level $k/(g+1)$, commit to it (execute until completion), then continue Round-Robin. **Theorem**: Choosing $k = \lceil(g/2)^{2/3}\rceil + 1$ yields a competitive ratio of $1 + O(g^{-1/3})$. The **lower bound** $1 + \Omega(g^{-1/3})$ shows the algorithm is asymptotically optimal. The lower bound is established by constructing two jobs with nearly identical processing times and arguing indistinguishability via properties of the Poisson process.

### Corollaries for Learning-Augmented Scheduling

By converting prediction $\pi_j$ into a signal $\beta_j = \alpha\pi_j/p_j$, Algorithm 1 applies directly. This yields a $(1+\alpha)$-consistent and $(1+1/\rho\alpha)$-robust algorithm, improving upon the previously best-known $(1/\lambda, 2/(1-\lambda))$ tradeoff (Purohit et al. 2018). The combining strategy (Algorithm 2) further achieves $(1+o(1))$-consistency and $(2+o(1))$-robustness on certain instance families.

## Key Experimental Results

### Main Results ($n=500$, Pareto(1.1) distribution, $\sigma$ = noise level)

| Configuration | Comp. ratio ($\sigma=0$) | Comp. ratio ($\sigma=1$) | Comp. ratio ($\sigma=5$) | Note |
|---|---|---|---|---|
| Alg.1, $\rho\to0$ (smooth) | $1+\alpha\approx1.5$ | ~1.6 | ~1.8 | Smooth degradation |
| Alg.1, $\rho=0.5$ | $1+\alpha\approx1.5$ | ~1.55 | ~2.0 | Intermediate tradeoff |
| Alg.1, $\rho=1$ (robust) | $1+\alpha\approx1.5$ | ~1.5 | ~3.0 | Optimal under no error but clearly brittle |
| Time sharing (baseline) | ~1.6 | ~1.65 | ~2.0 | Poor consistency but robust |
| Combining (Alg.2) | ~1.5 | ~1.55 | ~1.8 | No tuning required, balanced performance |

### Ablation Study (Stochastic Progress Bars)

| Granularity $g$ | Alg.3 ($k=\Theta(g^{2/3})$) | Alg.3 ($k=1$) | Round-Robin | Generic ETC |
|---|---|---|---|---|
| 10 | ~1.35 | ~1.55 | 2.0 | ~1.65 |
| 50 | ~1.15 | ~1.40 | 2.0 | ~1.55 |
| 200 | ~1.08 | ~1.25 | 2.0 | ~1.30 |

### Key Findings

- **Brittleness at $\rho=1$**: Algorithm 1 with $\rho=1$ degrades to competitive ratio 2 under even a tiny signal error (Proposition A.3), validating the importance of the smoothness guarantee.
- **Practical advantage of the combining algorithm**: It requires no parameter tuning and achieves balanced performance across different noise levels; the gap relative to alternatives becomes more pronounced as $n$ grows.
- **$g^{-1/3}$ convergence rate for stochastic progress bars**: Experiments confirm that the competitive ratio approaches 1 as $g$ increases, and $k=\Theta(g^{2/3})$ significantly outperforms both $k=1$ and generic ETC.

## Highlights & Insights

- The progress bar model provides a natural and practically meaningful continuum between clairvoyant and non-clairvoyant scheduling.
- This is the first work to achieve $(1+\alpha)$-consistency while exploiting **numerical prediction values** (rather than only ordinal rankings) for robustification, establishing a separation between numerical and permutation predictions.
- The generality of the combining strategy is impressive—it can combine any scheduling algorithms with computable delays.
- The tight $1+\Theta(g^{-1/3})$ competitive ratio exploits the statistical indistinguishability of Poisson processes, analogous to lower bound arguments in the bandit literature.

## Limitations & Future Work

- A gap of $\Theta(\sqrt{\alpha})$ remains between the lower bound $1+\Omega(1/\sqrt{\alpha})$ and the upper bound $1+1/\alpha$ for the consistency–robustness tradeoff.
- The regret term $O(n^{5/3}\max p_i)$ in the combining strategy may dominate when $\max p_i$ is large.
- The stochastic progress bar model cannot improve upon a competitive ratio of 2 when $g \leq 12$.
- The general multi-machine parallel scheduling setting is not considered (Appendix E contains a preliminary extension).

## Related Work & Insights

- This work is complementary to Benomar & Perchet (2024) on non-clairvoyant scheduling with partial predictions—that work obtains predictions at discrete time steps, while the present paper receives information continuously during execution.
- The combining strategy generalizes the permutation-prediction combiner of Eliás et al. (2024) and is applicable to a broader class of learning-augmented online problems.
- The connection between stochastic progress bars and mortal bandits (Chakrabarti et al. 2008) may inspire further application of bandit-theoretic tools.

## Core Results at a Glance

- Trusted single signal: competitive ratio $1+\alpha$ (optimal)
- Untrusted single signal (Alg.1): $(1+\alpha)$-consistent, $(1+1/\rho\alpha)$-robust
- Combining strategy (Alg.2): regret $O(n^{5/3}(\log g)^{1/3}\max p_i)$
- Stochastic progress bars (Alg.3): competitive ratio $1+O(g^{-1/3})$, lower bound $1+\Omega(g^{-1/3})$
- Learning-augmented corollary: improves the tradeoff from $(1/\lambda, 2/(1-\lambda))$ to $(2/(1+\lambda), 2/(1-\lambda))$

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The progress bar model is a highly natural and insightful new information model; the tight $g^{-1/3}$ rate is particularly satisfying.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments adequately validate theoretical predictions; evaluation on practical application scenarios could be expanded.
- Writing Quality: ⭐⭐⭐⭐⭐ Model definitions are clear, results are presented in a well-structured progression, and proof intuitions are thoroughly explained.
- Value: ⭐⭐⭐⭐⭐ Represents a substantive advance in learning-augmented scheduling theory and improves upon the optimal consistency–robustness tradeoff.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/others/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[NeurIPS 2025\] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning](directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)

<!-- RELATED:END -->
