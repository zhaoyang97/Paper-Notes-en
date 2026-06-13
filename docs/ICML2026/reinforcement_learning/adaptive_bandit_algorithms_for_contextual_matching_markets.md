---
title: >-
  [Paper Note] Adaptive Bandit Algorithms for Contextual Matching Markets
description: >-
  [ICML 2026][Reinforcement Learning][contextual bandit] This paper investigates online matching markets with contexts, treating players' linear preferences over dynamic arm contexts as bandit learning objectives. It propo…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "contextual bandit"
  - "matching market"
  - "stable matching"
  - "regret bound"
  - "adaptive exploration"
date: 2026-05-08
content_hash: 08f69474d3ec5f64
---

# Adaptive Bandit Algorithms for Contextual Matching Markets

**Conference**: ICML 2026  
**arXiv**: [2605.28290](https://arxiv.org/abs/2605.28290)  
**Code**: None  
**Area**: Reinforcement Learning / Contextual Bandit  
**Keywords**: contextual bandit, matching market, stable matching, regret bound, adaptive exploration  

## TL;DR
This paper investigates online matching markets with contexts, treating players' linear preferences over dynamic arm contexts as bandit learning objectives. It proposes BARB for stochastic contexts and AdECO for adversarial contexts, providing adaptive upper bounds for player-optimal stable regret and tight $\tilde O(T^{2/3})$ theoretical results.

## Background & Motivation
**Background**: Two-sided matching markets are used to model school admissions, labor markets, medical residency matching, and platform dispatching. The classic Gale-Shapley algorithm assumes preferences on both sides are known to find a stable matching. Recently, research on bandit learning in matching markets has treated one side's preferences as unknown, learning them through repeated matching and feedback.

**Limitations of Prior Work**: Many existing works assume preference profiles are static or require prior knowledge such as the minimum gap, covariance structure, or number of environments. However, in real-world platforms, the context of a job/task changes every round; a worker's utility for a task is determined by both a fixed preference vector and a dynamic context. Small context changes might only slightly alter a player's utility but completely rearrange the stable matching benchmark, causing other players' regret to skyrocket.

**Key Challenge**: Learning in matching markets is not about individual reward maximization. Algorithms must simultaneously learn the linear preferences of each player, maintain stable or near-stable matchings, and control the regret of each player relative to the player-optimal stable matching. Small utility gaps or ties make the stability benchmark itself fragile, especially under adversarial contexts, where traditional regret may become uncontrollable.

**Goal**: To design adaptive algorithms for both stochastic and adversarial context settings that do not require knowledge of the true gap or covariance structure, while providing regret guarantees that hold for every player.

**Key Insight**: The authors define a probabilistic minimum preference gap $\Delta_{min}$ for the stochastic setting, allowing for occasional small gaps rather than requiring a deterministic lower bound in all rounds. For the adversarial setting, they define $\alpha$-approximate $\Delta$-optimal stable regret, which aligns with the exact stable benchmark during large gaps and shifts to an approximate stable benchmark during small gaps.

**Core Idea**: Use the Mahalanobis norm to determine if the current context still requires exploration, combined with maximum cardinality matching for information gathering and Gale-Shapley for exploitation. In adversarial small-gap cases, an approximation oracle is invoked.

## Method
The basic model consists of $N$ players and $K$ arms, with $N \leq K$. Arm preferences for players are fixed, strict, and known; player preferences for arms are unknown and change with the context. In each round $t$, each arm $a_j$ presents a context $x_j(t) \in \mathbb{R}^d$, and the true utility for player $p_i$ is $U_{i,j}(t) = \theta_i^\top x_j(t)$. If matched, the platform observes a noisy reward $y_{i,j}(t) = U_{i,j}(t) + \epsilon_{i,j}(t)$.

### Overall Architecture
Given the true utility matrix $U(t)$ and arm-side preferences, a set of stable matchings $S_t$ exists. The paper uses the player-optimal stable matching $\mu_t^*$ as the benchmark and defines the regret for each player as: $Reg_i(T) = \sum_{t=1}^T U_i^*(t) - \mathbb{E}[\sum_{t=1}^T y_i(t)]$. The goal is not to maximize total reward but to ensure each player stays as close as possible to the utility they would receive in the player-optimal stable matching.

In the stochastic setting, each arm context is sampled independently from a fixed unknown distribution. The authors define the minimum preference gap $\Delta_{min}$ as the largest gap satisfying $P(\delta_{min} \geq \Delta) \geq 1 - \log T / (T\Delta^2)$. This arises from balancing the exploration cost $O(\log T / \Delta^2)$ and the probability cost of exploiting small gaps $O(\zeta T)$.

In the adversarial setting, contexts can be chosen arbitrarily or even adaptively. Since an adversary can keep the gap close to zero for long periods, exact stable regret is no longer feasible. The paper uses $\epsilon$-stable matching and an approximation oracle to define tractable approximate regret: comparing against $U_i^*(t)$ when $\delta_{min}(t) > \Delta$, and against $\alpha U_i^\epsilon(t)$ when the gap is small.

### Key Designs
1.  **BARB: Batched Adaptive Regret Balancing**:
    - **Function**: Adaptively explores to the appropriate precision without knowing the true $\Delta_{min}$ under stochastic contexts.
    - **Mechanism**: The $k$-th batch maintains a candidate gap $\Delta_k$ and a threshold $\xi_k = \Delta_k / \eta$. If a player-arm pair satisfies $\|x_j(t)\|_{V_i(t)^{-1}} > \xi_k$, the estimate in that direction is deemed insufficient, and the algorithm performs maximum cardinality matching on the corresponding bipartite graph for exploration; otherwise, it runs Deferred Acceptance using estimated utilities.
    - **Design Motivation**: The Mahalanobis norm measures the uncertainty of the current context within the player's estimation. Maximum cardinality matching allows for collecting effective samples for as many players as possible within a single round.

2.  **Overlap Counter and Gap Shrinking Mechanism**:
    - **Function**: Detects whether the current candidate gap is too large and enters a more refined batch when necessary.
    - **Mechanism**: The exploitation phase constructs confidence intervals of radius $\Delta_k$ for each estimated utility. If intervals for different arms frequently overlap, it indicates the current precision is insufficient for reliable ranking. Once the counter $N_k$ exceeds a threshold, it sets $\Delta_{k+1} = \Delta_k / \sqrt{2}$.
    - **Design Motivation**: The algorithm does not need prior knowledge of $\Delta_{min}$ but approaches the necessary precision by observing how often rankings cannot be clearly separated.

3.  **AdECO: Adaptive Exploration-Selection Oracle under Adversarial Contexts**:
    - **Function**: Provides meaningful regret even when contexts change arbitrarily and small gaps appear frequently.
    - **Mechanism**: AdECO continues to use the Mahalanobis norm for exploration. If estimates are sufficiently accurate and confidence intervals are separated, it calls Gale-Shapley; if intervals overlap, it calls an $\alpha$-approximation oracle, allowing for an instability tolerance of level $\Delta + \epsilon$.
    - **Design Motivation**: When gaps are too small to distinguish, insisting on an exact stable benchmark leads to unavoidable regret. The oracle branch switches the target to approximate stable utility, making the problem solvable again.

### Loss & Training
This is an online learning theory paper and does not utilize neural network losses. Preference estimation uses ridge regression. For player $i$, updates are made only using rounds $G^{(i)}$ where they participated in exploration: $V_i(t) = \lambda I + \sum_{s < t, s \in G^{(i)}} x_{i_s} x_{i_s}^\top$ and $\hat\theta_i(t) = V_i(t)^{-1} \sum_{s < t, s \in G^{(i)}} x_{i_s} y_{i, i_s}$. Standard linear bandit confidence bounds ensure that $\|\theta_i - \hat\theta_i(t)\|_{V_i(t)} \leq \eta$ holds with high probability, so $\|x_j(t)\|_{V_i(t)^{-1}}$ directly controls the utility estimation error.

Regret analysis relies primarily on the elliptical potential lemma: the number of exploration rounds is controlled by $\sum \|x\|_{V^{-1}}$, and GS does not produce player-optimal stable regret when confidence intervals do not overlap during exploitation. The number of overlapping rounds is controlled by the batch stopping threshold.

## Key Experimental Results

### Main Results
The primary contribution is the theoretical regret guarantee; numerical experiments serve to verify the convergence of BARB/AdECO. Main results categorized by theorem are as follows:

| Setting / Algorithm | Regret Metric | Upper / Lower Bound | Key Conditions | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| Stochastic contexts / BARB | player-optimal stable regret | $O(\log^2 T / \Delta_{min}^2)$ | bounded contexts, sub-Gaussian noise, bounded $\theta_i$ | No prior knowledge of gap or covariance required |
| Stochastic + covariance lower bound | player-optimal stable regret | $O(\log T / (\tilde\lambda^2 \Delta_{min}^2))$ | context covariance min eigenvalue $\geq \tilde\lambda$ | Removing one log factor with better structure |
| Stochastic asymptotic upper | per-player regret | $\tilde O(T^{2/3})$ | small gap CDF is at most linear near 0 | instance-independent upper bound |
| Stochastic lower bound | at least one player regret | $\Omega(T^{2/3})$ | construction of $N=K=3$ instance | Proving the $T^{2/3}$ rate is tight |
| Adversarial contexts / AdECO | $\alpha$-approx. $\Delta$-optimal stable regret | $O(Nd\log^2T/(\Delta-\epsilon)^2 + (\Delta+\epsilon)T/2)$ | arbitrary context, given oracle | $O(T^{2/3})$ by setting $\Delta=O(T^{-1/3}), \epsilon=\Delta/2$ |

Numerical experiments set $T=200k$, 20 independent trials, $N=K=4$, and context dimension $d=3$. In stochastic scenarios where the minimum covariance eigenvalue is small, the cumulative regret of BARB is lower than that of ETC and Batched-ETC. When the covariance structure is favorable, BARB performs similarly to ETC but is more robust as it does not rely on prior knowledge of the covariance.

### Ablation Study
The paper lacks traditional model ablations; the analysis focuses on comparing different algorithms and benchmark selections.

| Algorithm / Design | Applicable Scenarios | Prior Knowledge Needed | Behavior | Major Pros/Cons |
| :--- | :--- | :--- | :--- | :--- |
| ETC | stochastic contexts | fixed exploration length | Explore then exploit | Simple, but sensitive to covariance and gap |
| Batched-ETC | stochastic + PD covariance | covariance structure assumptions | Batched explore-then-commit | Achieves $O(\log T / (\tilde\lambda^2 \Delta_{min}^2))$, but stronger assumptions |
| BARB | stochastic contexts | None ($\Delta_{min}$ or covariance) | Adaptive explore/exploit in batches, shrinking candidate gap | Most robust; theoretically adds one log factor |
| AdECO | adversarial contexts | $\Delta, \epsilon$ and approx. oracle | GS for large gaps, oracle for small gaps | $O(T^{2/3})$ guarantee under arbitrary contexts |

### Key Findings
- The probabilistic definition of the minimum preference gap is crucial. It is more realistic than assuming a deterministic gap in all rounds and explains why the final instance-independent rate is $T^{2/3}$.
- Exploration in BARB is triggered by the uncertainty of the context within the current estimation ellipsoid rather than round count. This allows it to adapt to any context covariance without prior knowledge of which directions are difficult to learn.
- The adversarial setting necessitates relaxing the benchmark. If an adversary continuously creates small gaps, insisting on exact player-optimal stable regret leads to theoretically uncontrollable bounds.
- Numerical results support theoretical intuition: fixed exploration designs tend to fail when covariance is degenerate, while BARB's adaptive exploration is more stable.

## Highlights & Insights
- The paper tightly integrates stability constraints of matching markets with the confidence ellipsoids of contextual linear bandits; the algorithm design corresponds clearly to the regret proof.
- The definition of $\Delta_{min}$ is not a trivial substitution but is derived from the balance between exploration regret and exploitation regret caused by the probability of small gaps.
- AdECO's handling of small gaps is pragmatic: rather than pretending all ties can be resolved, it switches to an approximate stable benchmark, acknowledging the inherent non-identifiability of the problem.
- Player-level regret is more aligned with fairness in matching markets than social regret, as stable matching focuses on whether participants have justified envy rather than a single aggregate utility.

## Limitations & Future Work
- The theory and algorithms are primarily centralized platform versions; while the paper discusses decentralized extensions, a complete analysis is complex, and communication and strategic behaviors in actual platforms are not fully addressed.
- AdECO relies on an offline $\alpha$-approximation oracle. The computational complexity, approximation quality, and practical feasibility of such an oracle affect deployment.
- Experiments are based on synthetic markets, lacking validation on real-world labor or task platform data. Arm-side preferences in real data may also not be fixed or known.
- Linear utility assumptions facilitate theoretical analysis, but real-world preferences may include non-linearities, interaction terms, and strategic responses. Future work could consider generalized linear or representation learning versions.

## Related Work & Insights
- **vs. Static Matching Bandits**: Most existing work learns fixed preference profiles; this paper handles dynamic preferences where contexts change every round.
- **vs. Li et al. 2022 Contextual Matching**: Previous methods required known gap or covariance structures; BARB eliminates these priors through batch gap shrinking and Mahalanobis exploration.
- **vs. Contextual Combinatorial Bandits**: CCB focuses on the total reward of selecting a super arm; this paper focuses on two-sided stable matching and the player-optimal stable regret of each player.
- **Insight**: In bandit problems with equilibrium/stability benchmarks, defining a learnable and identifiable regret benchmark is often more important than directly applying standard bandit regret.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines contextual bandit, stable matching, and adaptive gap learning with unique theoretical settings.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theory paper; numerical experiments verify trends but lack real-world data and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem structure and well-motivated algorithms, though notation and proof sketches are dense.
- Value: ⭐⭐⭐⭐ Clear theoretical value for online platform matching, bandit theory, and stable matching learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MoMa QL: Accelerating Diffusion/Flow Matching Policies for Offline and Offline-to-Online RL via Moment Matching](moment_matching_q-learning.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[AAAI 2026\] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization](../../AAAI2026/reinforcement_learning/provably_efficient_multi-objective_bandit_algorithms_under_preference-centric_cu.md)
- [\[ICML 2026\] Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges](turning_bias_into_bugs_bandit-guided_style_manipulation_attacks_on_llm_judges.md)
- [\[ICML 2026\] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](plug-and-play_benchmarking_of_reinforcement_learning_algorithms_for_large-scale_.md)

</div>

<!-- RELATED:END -->
