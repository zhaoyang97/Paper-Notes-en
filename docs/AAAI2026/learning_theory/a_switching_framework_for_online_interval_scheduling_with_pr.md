---
title: >-
  [Paper Note] A Switching Framework for Online Interval Scheduling with Predictions
description: >-
  [AAAI 2026][Online Algorithms][Online interval scheduling] For the irrevocable online interval scheduling problem, this paper proposes the SemiTrust-and-Switch framework and the SmoothMerge randomized algorithm. By switching between or blending a prediction-trusting strategy and a classical greedy algorithm, the approach achieves near-optimal performance when predictions are accurate (consistency) and degrades gracefully when predictions are erroneous (robustness and smoothne…
tags:
  - "AAAI 2026"
  - "Online Algorithms"
  - "Learning-Augmented Algorithms"
  - "Scheduling Optimization"
  - "Online interval scheduling"
  - "competitive ratio"
  - "consistency–robustness tradeoff"
  - "randomized algorithms"
date: 2026-05-08
content_hash: 260fd98eeeea7444
---

# A Switching Framework for Online Interval Scheduling with Predictions

**Conference**: AAAI 2026
**arXiv**: [2511.16194](https://arxiv.org/abs/2511.16194)  
**Code**: None  
**Area**: Online Algorithms / Learning-Augmented Algorithms / Scheduling Optimization
**Keywords**: Online interval scheduling, learning-augmented algorithms, competitive ratio, consistency–robustness tradeoff, randomized algorithms

## TL;DR
For the irrevocable online interval scheduling problem, this paper proposes the SemiTrust-and-Switch framework and the SmoothMerge randomized algorithm. By switching between or blending a prediction-trusting strategy and a classical greedy algorithm, the approach achieves near-optimal performance when predictions are accurate (consistency) and degrades gracefully when predictions are erroneous (robustness and smoothness). Tightness of the framework on specific instances is also established.

## Background & Motivation
Online interval scheduling is a classical problem: intervals arrive in order and an algorithm must immediately and irrevocably accept or reject each one, with the goal of maximizing total length of accepted intervals. Classical results show that no deterministic algorithm achieves a constant competitive ratio in the worst case (Lipton & Irani, 1994), and the lower bound for randomized algorithms is $O(\log \Delta)$, where $\Delta$ is the ratio of the longest to the shortest interval.

In practice, however, predictions about future intervals are often available via historical data or machine learning models. The recent paradigm of *algorithms with predictions* aims to design algorithms that exploit predictions to improve performance while remaining robust when predictions are wrong.

Prior related work includes: (1) unit-weight interval scheduling with predictions (Antoniadis et al., WADS 2023); and (2) proportional-weight scheduling in the revocable setting (Karavasilis et al., 2025). The practically important setting of **irrevocable decisions with proportional weights** has not been studied; this paper fills that gap.

## Core Problem
In irrevocable online interval scheduling, how can (potentially inaccurate) machine-learned predictions be leveraged to improve algorithm performance? Specifically, how should an algorithm be designed to achieve:
1. **Good consistency**: near-optimal performance when predictions are accurate;
2. **Good robustness**: bounded degradation when predictions are entirely wrong;
3. **Good smoothness**: graceful, rather than cliff-like, degradation as prediction error increases.

The fundamental challenge is irrevocability—accepting an incorrectly predicted interval early may permanently block high-value intervals later.

## Method

### Overall Architecture
The paper presents a three-level progressive algorithm design:
1. **Trust-and-Switch (warm-up)**: follow predictions until an error is detected, then switch once to a classical algorithm;
2. **SemiTrust-and-Switch (main framework)**: during the trust phase, apply predictions only to short intervals while always greedily accepting long intervals, combined with the same switching mechanism;
3. **SmoothMerge (randomized algorithm)**: probabilistically blend the Trust and Greedy strategies to achieve smooth transitions.

The input is an online sequence of intervals together with binary predictions $\hat{O} = \{0, 1\}^n$ indicating whether each interval should be accepted. The output is a non-overlapping subset maximizing total length.

### Key Designs

1. **Trust-and-Switch Framework**: The algorithm begins in a trust phase and follows predictions blindly. Upon the arrival of each interval $I_j$, an evaluation point $t_j$ is computed to compare the predicted solution $|Trust(\mathcal{I})|$ against the offline optimum $|Opt(\mathcal{I})|$ up to $t_j$. Once the predicted solution is found to be strictly inferior to optimal, the algorithm irreversibly switches to a classical $\theta$-competitive algorithm. The core guarantee is 1-consistency and $|Opt| \leq \theta \cdot |Alg| + 2k$ (where $k$ is the maximum interval length).

2. **SemiTrust-and-Switch Framework**: A length threshold $\tau$ is introduced. During the trust phase, intervals longer than $\tau$ are accepted greedily (ignoring predictions), while intervals shorter than $\tau$ follow predictions. This ensures that high-value long intervals are not missed even under inaccurate predictions. The tradeoff is that consistency degrades from 1 to $(1 + k/\tau)$, but robustness improves significantly to $|Opt| \leq \max(\theta, \Delta+1) \cdot |Alg| + \tau$. The threshold $\tau$ is tunable and controls the consistency–robustness tradeoff curve.

3. **SmoothMerge Randomized Algorithm**: The Trust and Greedy strategies are simulated in parallel. For each arriving interval, if both strategies agree (both accept or both reject), the consensus decision is executed; if they disagree, the interval is accepted with probability $p_t$ (Trust probability) or $p_g$ (Greedy probability) at random. The core guarantee is:
$$\mathbb{E}[|Alg|] \geq \max\left\{|Opt| \cdot (1-\eta) \cdot p_t \cdot (1-p_g),\quad \frac{p_g - p_t p_g}{1 - p_t p_g} \cdot |Greedy|\right\}$$
where $\eta$ is the prediction error. By tuning $p_t$ and $p_g$, one can continuously interpolate between smoothness and robustness.

### Prediction Models
- **Binary prediction $\hat{O}$**: an accept/reject recommendation for each interval, revealed online as intervals arrive;
- **Full-information prediction $\hat{I}$**: predicted arrival and deadline times for all intervals (a strictly stronger prediction model).
- An interesting finding is that more detailed predictions or offline predictions do not improve the achievable theoretical guarantees.

### Loss & Training
This is a theoretical algorithm design paper and involves no training. The core performance metrics are the consistency and robustness of the competitive ratio.

## Key Experimental Results

Experiments are conducted on real scheduling traces (NASA-iPSC-1993 and CTC-SP2-1996 parallel machine scheduling benchmarks):

| Dataset | # Intervals | Max Deadline |
|---------|-------------|--------------|
| NASA-iPSC-1993-3.1 | 18,065 | 62,643 |
| CTC-SP2-1996-3.1 | 77,205 | 71,998 |

- Different prediction error levels $\eta$ are simulated by corrupting a controlled number of predicted intervals.
- SmoothMerge approaches Opt under low prediction error and transitions smoothly between Trust and Greedy as error increases.
- Under large prediction error, SmoothMerge degrades to near-Greedy performance without collapsing.
- The Trust strategy degrades sharply as prediction error grows, whereas SmoothMerge degrades gracefully.

### Ablation Study
- Setting $p_t = 1, p_g = 0$ recovers pure Trust (perfect smoothness, no robustness).
- Setting $p_t = 0, p_g = 1$ recovers pure Greedy (full robustness, no smoothness).
- Intermediate values (e.g., $p_t = 0.6, p_g = 0.4$) simultaneously achieve both properties at a moderate cost in the coefficients.

## Highlights & Insights
- **Careful design of the switching trigger**: Trust-and-Switch switches at most once and irreversibly upon detecting a prediction error. By carefully choosing evaluation points $t_j$ (with respect to the deadlines of predicted intervals), the switch is never triggered when predictions are accurate, thereby guaranteeing perfect 1-consistency.
- **Threshold idea in SemiTrust**: The simple divide-and-conquer strategy of applying greedy to long intervals and predictions to short ones elegantly improves the robustness additive term from $+2k$ to $+\tau$ (where $\tau < k$), with a controlled loss in consistency.
- **Conflict graph analysis**: Modeling the choices of Opt and Alg as connected components of a conflict graph is a transferable technique for analyzing greedy-type algorithms in other online optimization problems.
- **Constructive adversary for tightness**: Carefully crafted adversarial instances demonstrate the optimality of the framework on two-value instances, revealing a fundamental consistency–robustness tradeoff.

## Limitations & Future Work
- **Tight for two-value instances, gap remains in general**: The upper and lower bounds match on two-value (short/long) instances, but the gap for general instances has not been closed.
- **Interval length as the only weight**: The framework does not extend to arbitrary-weight interval scheduling, where weights may be independent of length.
- **Granularity of prediction error**: The current binary prediction model is coarse; finer-grained prediction information (e.g., confidence scores) may yield stronger guarantees.
- **Limited experimental scale**: Only two benchmarks are tested; systematic experiments across different values of $\Delta$ are lacking.
- **Potential extensions**: revocable settings, multi-resource scheduling, and scheduling with additional constraints.

## Related Work & Insights

| Work | Setting | Decision | Main Contribution |
|------|---------|----------|-------------------|
| Antoniadis et al. (WADS 2023) | Unit weight | Irrevocable | Tight bounds for deterministic algorithms |
| Karavasilis et al. (2025) | Proportional weight | Revocable | Adaptive algorithm under binary predictions |
| **Ours** | **Proportional weight** | **Irrevocable** | **Unified framework + randomized smooth algorithm** |

The key distinctions of this paper are: (1) it addresses the hardest setting of irrevocable decisions with proportional weights; (2) it proposes a general framework rather than a specific algorithm, accepting any classical algorithm as a fallback; (3) SmoothMerge is the first algorithm to achieve smoothness in this setting.

The trust-and-switch paradigm is transferable to other online maximization problems (e.g., online knapsack, matching). The conflict graph analysis technique is broadly useful for analyzing greedy-type algorithms. The definition of prediction error $\eta$ (based on the ratio between Trust and Opt) serves as a general measure of how much predictions benefit an algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐ Fills an important gap in the literature with a novel framework design, though the switching technique itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ Experiments serve as supplementary validation; scale and diversity are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with a clear progression from warm-up to the main framework and randomization; proof sketches are intuitive.
- Value: ⭐⭐⭐⭐ Contributes to the learning-augmented online algorithms literature; the modular framework design has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions](../../ICLR2026/learning_theory/on_smoothness_bounds_for_non-clairvoyant_scheduling_with_predictions.md)
- [\[ICLR 2026\] Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval](../../ICLR2026/learning_theory/discounted_online_convex_optimization_uniform_regret_across_a_continuous_interva.md)
- [\[NeurIPS 2025\] Non-Clairvoyant Scheduling with Progress Bars](../../NeurIPS2025/learning_theory/non-clairvoyant_scheduling_with_progress_bars.md)
- [\[ICLR 2026\] Towards Safe and Optimal Online Bidding: A Modular Look-Ahead Lyapunov Framework](../../ICLR2026/learning_theory/towards_safe_and_optimal_online_bidding_a_modular_look-ahead_lyapunov_framework.md)
- [\[ICLR 2026\] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling](../../ICLR2026/learning_theory/atlas_alibaba_dataset_and_benchmark_for_learning-augmented_scheduling.md)

</div>

<!-- RELATED:END -->
