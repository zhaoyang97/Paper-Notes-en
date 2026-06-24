---
title: >-
  [Paper Note] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors
description: >-
  [ICML2025][Others/Online Algorithms][Metrical Task Systems] In Metrical Task Systems (MTS), when the algorithm can only access $\ell$ heuristics in a bandit fashion (querying only one heuristic per step and requiring $m$ consecutive queries to observe the state), this paper presents an algorithm with $O(\text{OPT}^{2/3})$ regret and proves that this bound is tight.
tags:
  - "ICML2025"
  - "Others/Online Algorithms"
  - "Metrical Task Systems"
  - "Learning-Augmented Algorithms"
  - "Bandit Feedback"
  - "Combining Heuristics"
  - "Competitive Ratio"
  - "Regret Bound"
date: 2026-05-08
content_hash: d2cebf3551a87d7f
---

# Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors

**Conference**: ICML2025  
**arXiv**: [2506.05479](https://arxiv.org/abs/2506.05479)  
**Code**: None  
**Area**: Others/Online Algorithms  
**Keywords**: Metrical Task Systems, Learning-Augmented Algorithms, Bandit Feedback, Combining Heuristics, Competitive Ratio, Regret Bound

## TL;DR

In Metrical Task Systems (MTS), when the algorithm can only access $\ell$ heuristics in a bandit fashion (querying only one heuristic per step and requiring $m$ consecutive queries to observe the state), this paper presents an algorithm with $O(\text{OPT}^{2/3})$ regret and proves that this bound is tight.

## Background & Motivation

**Background**: Metrical Task Systems (MTS) represent one of the most general frameworks in online algorithms, capable of modeling classic problems such as caching, $k$-server, ski rental, and convex body chasing. Under the paradigm of learning-augmented algorithms, "combining heuristics" is a core technique—given $\ell$ heuristics $H_1, \dots, H_\ell$ (each possibly designed for different input types), the goal is to construct an online algorithm with performance close to the best heuristic in hindsight.

**Prior Work**: In the full-feedback setting (where the states of all heuristics can be queried at each step), Blum & Burch (2000) used the HEDGE algorithm to achieve $(1+\epsilon)$-competitiveness. Antoniadis et al. (2023a) studied a "bandit-like" setting, but required heuristics to actively report movement costs.

**Limitations of Prior Work**: 
1. Querying all heuristics is expensive, especially when heavyweight ML models are used.
2. In a true bandit setting, only one heuristic can be queried per step, and the cost of heuristic $H_i$ at time $t$, $c_t(s_t^i) + d(s_{t-1}^i, s_t^i)$, contains both service and movement costs—unless the same heuristic is queried for $m$ consecutive steps, its cost cannot be estimated.
3. Classical bandit methods (such as EXP3) cannot be directly applied due to the lack of unbiased loss estimators.

**Core Idea**: To design an alternating exploration-exploitation algorithm that obtains cost information by querying the same heuristic for $m$ consecutive steps during the exploration phase, while using "improper steps" and MTS-style rounding to control switching costs, achieving a regret dependent on the optimal heuristic cost $\text{OPT}_{\leq 0}$ rather than the time horizon $T$.

## Method

### Problem Formalization

Given a metric space $(M, d)$, diameter $D = \max_{s,s'} d(s,s')$, and $\ell$ heuristics. The algorithm has **$m$-delayed bandit access**:
- Only one heuristic $H_i$ can be queried per step.
- The state $s_t^i$ is obtained only when $H_i$ is queried consecutively from $t-m+2$ to $t$.
- The algorithm retains full access to the input instances (cost function $c_t$).

Heuristic cost: $f_t(i) = c_t(s_t^i) + d(s_{t-1}^i, s_t^i) \in [0, 2D]$

### Algorithm Framework: Alternating Exploration-Exploitation

The core of the algorithm is an MTS adaptation of the classical MAB exploration-exploitation strategy, which internally uses HEDGE/SHARE as the full-information learning algorithm $\bar{A}$:

1. **Initialization**: Sample $\beta_t \sim \text{Bernoulli}(\epsilon)$ to control exploration probability, and $e_t \sim U(\{1,\dots,\ell\})$ to select the exploration target.
2. **Exploitation Step**: Follow the heuristic sampled from the internal distribution $x_t$.
3. **Exploration Trigger**: When $\beta_t = 1$, enter an $m$-step bootstrapping phase, querying $H_{e_{t+m}}$ consecutively.
4. **Exploration Step**: Obtain $f_{t+m}(e_{t+m})$ at step $t+m$, and construct a loss vector to update the distribution.

### Key Designs

**Loss Vector Construction**: After obtaining the cost in the exploration step, construct a normalized loss vector:

$$g_{t+m}^{e_{t+m}}(i) = \begin{cases} f_{t+m}(e_{t+m}) / 2D & \text{if } i = e_{t+m} \\ 0 & \text{otherwise} \end{cases}$$

This ensures $g_{t+m} \in [0,1]^\ell$, satisfying the input requirements of HEDGE.

**MTS-style Rounding (Proposition 2.2)**: Earth Mover Distance is used to control the distribution switching cost, avoiding the $O(T)$ switching cost caused by independent random sampling:

$$\mathbb{E}\left[\sum_t c_t(s_t^{i_t}) + d(s_{t-1}^{i_{t-1}}, s_t^{i_t})\right] \leq \sum_t f_t^T x_t + \frac{D}{2}\|x_{t-1} - x_t\|_1$$

**Improper Steps**: In the bootstrapping phase and exploration steps, the algorithm does not follow any heuristic but executes a greedy step starting from the nearest known exploitation state. This is the key to achieving regret dependent on $\text{OPT}_{\leq 0}$ instead of $T$.

**HEDGE Stability Utilization (Property 2.3)**: The amount of distribution change is controlled by the loss:

$$\|x_{t-1} - x_t\|_1 \leq \eta \cdot g_{t-1}^T x_{t-1}$$

Used to constrain the switching cost.

### Parameter Selection

Exploration rate $\epsilon$ and learning rate $\eta$ are jointly optimized, setting $\epsilon = \Theta(\text{OPT}_{\leq 0}^{-1/3})$ to balance exploration cost, learning accuracy, and switching cost.

## Theoretical Results

| Theorem | Setting | Result | Explanation |
|------|------|------|------|
| Thm 1.1 | Static Benchmark ($k=0$) | $\mathbb{E}[\text{ALG}] \leq \text{OPT}_{\leq 0} + O(\text{OPT}_{\leq 0}^{2/3})$ | Competitive ratio approaches 1 |
| Thm 1.2 | Dynamic Benchmark ($k$ switches) | $\mathbb{E}[\text{ALG}] \leq \text{OPT}_{\leq k} + \tilde{O}(k^{1/3} \cdot \text{OPT}_{\leq k}^{2/3})$ | Allows the benchmark to switch heuristics |
| Thm 1.3 | Lower Bound ($m=2$) | $\mathbb{E}[\text{ALG}] \geq \text{OPT}_{\leq 0} + \tilde{\Omega}(\text{OPT}_{\leq 0}^{2/3})$ | Proved to be tight based on Dekel et al.'s construction |

**Parameter Dependency**: The exact scaling of regret is $(Dk\ell \ln \ell)^{1/3} m^{2/3} \cdot \text{OPT}^{2/3}$, where the $(Dk\ell)^{1/3}$ part is proved to be nearly optimal.

**Bridge to Memory-Bounded Bandits**: The algorithm can adapt to the $m$-memory bounded bandit setting, achieving $O(T^{2/3})$ regret, slightly improving the dependency on $m$ compared to Arora et al. (2012).

## Highlights & Insights

- **Tight Bound**: The upper bound $O(\text{OPT}^{2/3})$ matches the lower bound $\tilde{\Omega}(\text{OPT}^{2/3})$ (up to logarithmic factors), revealing the intrinsic complexity of the problem.
- **Regret Dependent on OPT instead of T**: When the optimal heuristic performs very well ($\text{OPT} \ll T$), the bound is significantly better than the traditional bandit bound of $O(T^{2/3})$.
- **Necessity of Improper Steps**: Even for $m=2$ (minimum delay), improper steps are necessary to obtain OPT-dependent regret, highlighting the fundamental difference between MTS and standard bandit problems.
- **Robustification Application**: By adding a classical $\rho$-competitive online algorithm to the set of heuristics, the combined algorithm remains $(1+o(1))\rho$-competitive in the worst case, while utilizing ML predictions on favorable instances.
- **Lower Bound Construction Techniques**: The classic lower bound by Dekel et al. cannot be directly applied because the MTS algorithm enjoys two additional advantages: improper actions and 1-step look-ahead. Thus, a careful reconstruction is required.

## Limitations & Future Work

- **Purely Theoretical Work**: Lacks experimental validation and has not been evaluated on practical MTS instances (such as caching or k-server).
- **Parameters Require Knowledge of OPT**: The exploration rate $\epsilon$ depends on the unknown $\text{OPT}_{\leq 0}$. Although this can be resolved using the doubling trick, it introduces an extra constant factor.
- **Logarithmic Gap**: There is a logarithmic gap between the upper and lower bounds, and whether it can be eliminated remains unclear.
- **Assumption of Constant $D, \ell, m$**: The main theorems assume these parameters are constants; when $\ell$ or $m$ is large, the actual performance might degrade significantly.
- **Adaptive Adversary**: Only the oblivious adversary is considered; the extension to adaptive adversaries remains unexplored.

## Related Work & Insights

- **Arora et al. (2012)**: Bandit learning with memory-bounded adversaries, achieving $O(\mu T^{2/3})$ regret. This work improves the dependency on $m$ under the MTS setting.
- **Dekel et al. (2013)**: The $\Omega(T^{2/3})$ lower bound construction for bandits with switching costs, which serves as the technical foundation for the lower bound proof in this work.
- **Antoniadis et al. (2023a)**: Dynamic combination of heuristics in MTS, presenting a $(1+\epsilon)$-competitive algorithm under a bandit-like (but simpler) setting.
- **Blum & Burch (2000)**: Classic result on combining heuristics in MTS under full feedback, achieving $(1+\epsilon)$-competitiveness using HEDGE.
- **Lykouris & Vassilvitskii (2021)**: Pioneering work on learning-augmented algorithms, proposing the robustification framework.

## Rating
- Novelty: ⭐⭐⭐⭐ — Combining heuristics in MTS under bandit access is a novel setting, and the necessity of improper steps is a fresh insight.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical, no experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, coherent technical structure, and sufficient comparison with related work.
- Value: ⭐⭐⭐⭐ — Establishes tight bounds and solves a natural and important open problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)
- [\[ICLR 2026\] Online Rounding and Learning Augmented Algorithms for Facility Location](../../ICLR2026/learning_theory/online_rounding_and_learning_augmented_algorithms_for_facility_location.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](../../ICLR2026/learning_theory/decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)

</div>

<!-- RELATED:END -->
