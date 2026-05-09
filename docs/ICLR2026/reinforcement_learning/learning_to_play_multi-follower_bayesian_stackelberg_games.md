---
title: >-
  [Paper Note] Learning to Play Multi-Follower Bayesian Stackelberg Games
description: >-
  [ICLR 2026][Reinforcement Learning][Stackelberg Games] This paper provides the first systematic study of online learning in multi-follower Bayesian Stackelberg Games (BSGs). By geometrically partitioning the leader's strategy space into best-response regions, it achieves a regret bound of $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ under type feedback — a bound that does not grow polynomially in the number of followers $n$ — and establishes a nearly matching lower bound of $\Omega(\sqrt{\min\{L, nK\}T})$.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Stackelberg Games
  - Bayesian Games
  - Online Learning
  - Best-Response Regions
  - Regret Bounds
date: 2026-05-08
content_hash: 2d6a981b2613356e
---

# Learning to Play Multi-Follower Bayesian Stackelberg Games

**Conference**: ICLR 2026
**arXiv**: [2510.01387](https://arxiv.org/abs/2510.01387)
**Code**: None
**Area**: Game Theory / Online Learning
**Keywords**: Stackelberg Games, Bayesian Games, Online Learning, Best-Response Regions, Regret Bounds

## TL;DR

This paper provides the first systematic study of online learning in multi-follower Bayesian Stackelberg Games (BSGs). By geometrically partitioning the leader's strategy space into best-response regions, it achieves a regret bound of $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ under type feedback — a bound that does not grow polynomially in the number of followers $n$ — and establishes a nearly matching lower bound of $\Omega(\sqrt{\min\{L, nK\}T})$.

## Background & Motivation

**State of the Field**: Stackelberg games provide a foundational framework for modeling asymmetric interactions in multi-agent systems, where a leader commits to a strategy first and followers best-respond afterward. Applications span security games (e.g., airport patrol resource allocation), platform economics, information design (Bayesian persuasion), and contract design. In the Bayesian variant, followers hold private types, and the leader must know the type distribution to compute the optimal strategy (Stackelberg equilibrium). Existing online learning work (Balcan et al. 2015, 2025; Bollini et al. 2026) addresses only the **single-follower** setting.

**Limitations of Prior Work**: With $n$ followers each having $K$ types, the joint type space grows as $K^n$, causing exponential explosion. Directly estimating the joint distribution incurs estimation error of $\Omega(\sqrt{K^n/t})$, leading to regret of $\Omega(\sqrt{K^n T})$, which is completely unacceptable for large $n$. Moreover, each follower's best response is a piecewise-constant function of the leader's mixed strategy, rendering the leader's utility **discontinuous and non-convex** over the strategy space. Even in the offline single-follower setting, computing the optimal strategy is NP-Hard when the number of leader actions $L$ grows (Conitzer & Sandholm, 2006).

**Root Cause**: Multiple followers introduce exponential explosion in the joint type space. The key question is how to design learning algorithms whose regret bounds avoid exponential dependence on $n$. Intuitively, while the joint distribution of size $K^n$ is hard to learn, the leader truly cares about the **utility function** rather than the distribution itself — the complexity of the utility function may be far smaller than that of the distribution.

**Paper Goals**: (1) Is online learning in multi-follower BSGs feasible? (2) What are the optimal regret bounds under type feedback and action feedback, respectively? (3) Can regret bounds avoid exponential dependence on $n$?

**Starting Point**: The authors observe that although the joint type space $K^n$ is exponentially large, the leader's strategy space $\Delta(\mathcal{L})$ is only $(L-1)$-dimensional. Using classical results from computational geometry on hyperplane arrangements, one can show that the strategy space is partitioned into **at most $O(n^L K^L A^{2L})$ non-empty best-response regions** — a count that grows only polynomially in $n$ for fixed $L$. Crucially, the leader's utility is **linear within each region**, enabling efficient solution via linear programming.

**Core Idea**: Partition the leader's strategy space into polyhedral regions according to the followers' best-response behavior. By exploiting the fact that the number of regions is polynomially controlled in $n$, the paper designs UCB- and concentration-inequality-based learning algorithms that achieve regret bounds free of exponential dependence on $n$.

## Method

### Overall Architecture

The problem setting is as follows: a leader has $L$ actions; $n$ followers each have $K$ types and $A$ actions. At each round, the leader selects a mixed strategy $x \in \Delta(\mathcal{L})$; follower types are sampled from an unknown distribution $\mathcal{D}$; each follower observes $x$ and plays a best-response action. The leader aims to minimize cumulative regret over $T$ rounds. Key assumptions include: (1) followers are myopic best-responders; (2) no externalities — each follower's utility depends only on their own action, type, and the leader's action, not on other followers' actions.

The overall algorithmic approach operates on three levels. At the **bottom level**, the strategy space $\Delta(\mathcal{L})$ is geometrically partitioned into best-response regions, within each of which the utility is linear. At the **middle level**, the region structure enables concentration inequality analysis, proving that empirical utility uniformly approximates true utility. At the **top level**, an empirical-optimum strategy is used under type feedback, while a UCB algorithm explores across regions under action feedback.

### Key Designs

1. **Geometric Characterization of Best-Response Regions**:

    - Function: Discretizes the leader's continuous strategy space according to follower response patterns.
    - Mechanism: Define a mapping $W = (w_1, \ldots, w_n)$, where $w_i: \Theta \to \mathcal{A}$ specifies each follower's action for each type. The corresponding best-response region $R(W) = \{x \in \Delta(\mathcal{L}) \mid \mathbf{br}(\theta, x) = W(\theta), \forall \theta\}$ is the set of all leader strategies that induce followers to respond according to $W$. Each region can be expressed as an intersection of $O(nKA)$ halfspaces: $R(W) = \bigcap_{i, \theta_i, a_i} H(d_{\theta_i, w_i(\theta_i), a_i})$, where $H(d) = \{x: \langle x, d \rangle \geq 0\}$ encodes the constraint that "follower $i$ with type $\theta_i$ prefers action $w_i(\theta_i)$ over $a_i$." Using hyperplane arrangement counting results from computational geometry, the number of non-empty regions satisfies $|\mathcal{W}| = O(n^L K^L A^{2L})$ — polynomial in $n$ for fixed $L$. Regions can be enumerated via BFS graph traversal in $\mathrm{poly}(n^L, K^L, A^L, L)$ time.
    - Design Motivation: Since utility is linear within each region, the optimal strategy within each region can be solved in polynomial time via LP, and the globally optimal strategy is obtained by comparing the optimal values across all $|\mathcal{W}|$ regions. This reduces the offline BSG problem to polynomial time when $L$ is constant, complementing the prior NP-Hardness result (for growing $L$) and completing the computational complexity landscape.

2. **Learning Algorithms under Type Feedback (Algorithms 1 & 2)**:

    - Function: Learn the optimal strategy by observing all followers' types each round.
    - Mechanism: **Algorithm 1 (general type distribution)**: At each round, select the strategy maximizing empirical utility $x^t = \arg\max_x \sum_{s<t} u(x, \mathbf{br}(\theta^s, x))$. The analysis avoids estimating the joint distribution $\mathcal{D}$ (which requires $K^n$ parameters) and instead uses the best-response region structure to prove **uniform concentration** of the empirical utility: within each region the utility is an $L$-dimensional linear function with pseudo-dimension at most $L$; applying a union bound over all $|\mathcal{W}|$ regions yields $|U_\mathcal{D}(x) - \hat{U}^t(x)| \leq O(\sqrt{L \log(nKAT)/t})$, resulting in $\tilde{O}(\sqrt{L \cdot T})$ regret. **Algorithm 2 (independent type distribution)**: Estimate each follower's marginal distribution $\hat{\mathcal{D}}_i$ separately and form the product distribution $\hat{\mathcal{D}} = \prod_i \hat{\mathcal{D}}_i$. Independence allows the TV distance to decompose into a sum of Hellinger distances, reducing the distribution estimation error from $O(\sqrt{K^n/t})$ to $O(\sqrt{nK/t})$, yielding $O(\sqrt{nK \cdot T})$ regret.
    - Design Motivation: The two analyses are complementary — Algorithm 1's bound is superior when $n$ is large and $L$ is small ($\sqrt{L}$ vs. $\sqrt{nK}$ or $\sqrt{K^n}$), while Algorithm 2 is superior when $K$ is small. Taking the minimum of both gives the final bound.

3. **UCB Algorithm under Action Feedback (Algorithm 3)**:

    - Function: Learn the optimal strategy when only followers' actions (not types) are observed.
    - Mechanism: When the leader repeatedly plays within the same region $R(W)$, followers' actions are drawn from the same distribution $\mathcal{P}(\cdot | R(W))$, enabling estimation of the utility of any strategy within that region. The algorithm maintains a UCB score for each region: $\mathrm{UCB}^t(W) = \hat{u}^*_{R(W)} + \sqrt{4(L+1)\log(3T)/N^t(W)}$, and at each round selects the region with the highest UCB score and plays its empirically optimal strategy. This effectively reduces learning over the continuous strategy space to a multi-armed bandit problem over $|\mathcal{W}|$ "arms," with a linear optimization subproblem within each arm. Additionally, the paper provides a linear bandit approach based on techniques from Bernasconi et al. (2023), reformulating BSG as a linear program and applying the OFUL algorithm to obtain $O(K^n \sqrt{T} \log T)$ regret, which is superior when $L$ is large and $n$ is small.
    - Design Motivation: Action feedback contains less information than type feedback but is more accessible in practice (e.g., a platform can observe user behavior but not preference types). The UCB method's regret $O(\sqrt{n^L K^L A^{2L} L \cdot T \log T})$ is exponential in $L$, but this is computationally unavoidable given that BSG is NP-Hard in $L$.

### Loss & Training

This is a purely theoretical online learning work with no training loss functions. The leader's objective is to maximize cumulative utility (equivalently, minimize regret $\mathrm{Reg}(T) = \sum_{t=1}^T [U_\mathcal{D}(x^*) - U_\mathcal{D}(x^t)]$). Type-feedback algorithms are based on empirical utility maximization (the ERM principle); action-feedback algorithms are based on optimism under uncertainty (the UCB principle). The key theoretical tools are uniform concentration inequalities based on the pseudo-dimension of best-response regions, and the relationship between TV distance and Hellinger distance under independent distributions.

## Key Experimental Results

### Main Results: Regret Bound Summary

| Feedback Setting | Type Distribution Assumption | Regret Upper Bound | Regret Lower Bound | Gap Analysis |
|-----------------|-----------------------------|--------------------|--------------------|----|
| Type feedback | Independent types | $\tilde{O}(\sqrt{\min\{L, nK\} \cdot T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | Differs only by logarithmic factors; nearly optimal |
| Type feedback | General types | $\tilde{O}(\sqrt{\min\{L, K^n\} \cdot T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | Gap between $K^n$ in upper bound and $nK$ in lower bound |
| Action feedback | Arbitrary | $\tilde{O}(\min\{\sqrt{n^L K^L A^{2L} L}, K^n\} \cdot \sqrt{T})$ | $\Omega(\sqrt{\min\{L, nK\} \cdot T})$ | Significant gap; remains an open problem |

### Computational Complexity Comparison

| Algorithm | Feedback Type | Per-Round Time Complexity | Applicable Regime |
|-----------|--------------|--------------------------|-------------------|
| Algorithm 1 (General-type ERM) | Type feedback | $\mathrm{poly}((nKA)^L \cdot L \cdot T)$ | Small $L$, large $n$ |
| Algorithm 2 (Independent-type ERM) | Type feedback | $\mathrm{poly}((nKA)^L \cdot L \cdot T \cdot K^n)$ | Independent types, moderate $n$ and $K$ |
| Algorithm 3 (Region UCB) | Action feedback | $\mathrm{poly}((nKA)^L \cdot L \cdot T)$ | Small $L$, large $n$ |
| Linear Bandit (OFUL) | Action feedback | $\mathrm{poly}(K^n \cdot T)$ | Small $n$ and $K$ |

### Key Findings

- **Regret does not grow exponentially in $n$**: Under type feedback with independent types, both the upper and lower bounds are $\Theta(\sqrt{\min\{L, nK\} \cdot T})$, with only linear dependence on $n$. This is the most surprising result — the joint type space $K^n$ is exponentially large, yet learning only $nK$ marginal distribution parameters suffices to reconstruct the required information.
- **The dual role of $L$**: Conventional wisdom holds that growing $L$ causes computational hardness (NP-Hardness), but statistically, small $L$ is beneficial — the number of best-response regions grows exponentially in $L$, so small $L$ implies fewer regions and tighter union bounds in concentration inequalities.
- **Information gap between feedback models**: Type feedback enables near-optimal regret of $\tilde{O}(\sqrt{L \cdot T})$, whereas the optimal regret rate under action feedback remains undetermined, revealing a computation-statistics tradeoff: even if regret of $\mathrm{poly}(n, K, L)\sqrt{T}$ were achievable, the runtime must be exponential in $L$ (unless P=NP).
- **New offline result as a byproduct**: It is shown that BSG can be solved in polynomial time when $L$ is constant, filling a gap in the complexity landscape previously limited to NP-Hardness results for growing $L$.

## Highlights & Insights

- **Bridging computational geometry and game theory**: The key technical contribution is handling the discontinuity in utility caused by followers' best responses as a region-counting problem in hyperplane arrangements. This perspective transforms an otherwise intractable non-convex, discontinuous optimization problem into the standard task of solving a linear program within each of finitely many polyhedral regions. This paradigm of "geometrization → discretization → region-wise optimization" may offer insights for other online learning problems with discontinuous utilities.
- **Learning distributions vs. learning utility functions**: Although the joint type distribution $\mathcal{D}$ requires $K^n$ parameters to describe, the leader's utility function is only an $L$-dimensional linear function within each region. Learning the utility function is therefore far easier than learning the distribution — an insight that challenges the intuition that "the difficulty of learning a distribution determines the difficulty of the learning problem," and one that may transfer to other online optimization settings.
- **Double reduction technique for lower bounds**: The lower bound is established via a two-step reduction: first, distribution learning is reduced to single-follower BSG (yielding $\Omega(\sqrt{\min\{L, K\}T})$); then, single-follower BSG with $nK$ types is reduced to $n$-follower BSG with $K$ types each (yielding $\Omega(\sqrt{\min\{L, nK\}T})$). This two-step reduction construction is a technique worthy of broader adoption.

## Limitations & Future Work

- **Restriction of the no-externalities assumption**: The assumption that each follower's utility depends only on their own action and type — not on other followers' actions — does not hold in some settings (e.g., auctions, congestion games). Introducing externalities requires solving for Nash equilibria among followers, substantially complicating the structure of best-response regions.
- **Myopic follower assumption**: Followers are assumed to best-respond independently each round rather than strategically considering long-term payoffs. In practice, sophisticated adversaries may deliberately conceal type information to gain long-term advantages.
- **Unclosed gap under action feedback**: The optimal regret rate under action feedback is the main open problem left by this work. Whether an algorithm achieving $\mathrm{poly}(n, K, L)\sqrt{T}$ regret exists remains unknown.
- **Extension to adversarial types**: The paper considers only stochastic types. The authors conjecture in the discussion that the "region concentration" technique can be extended to adversarial settings (e.g., replacing ERM with EXP3+FTRL), but this has not been formally verified.

## Related Work & Insights

- **vs. Balcan et al. (2015, 2025)**: Their work designs online learning algorithms for single-follower BSG with regret $\mathrm{poly}(K)\sqrt{T}$. This paper extends the problem to multiple followers, where the core challenge lies in the joint type space growing from $K$ to $K^n$. The present work avoids this exponential difficulty via best-response region geometry.
- **vs. Bernasconi et al. (2023)**: Their work uses adversarial linear bandit techniques for multi-receiver Bayesian persuasion (structurally similar to multi-follower BSG), achieving $\tilde{O}(K^{3n/2}\sqrt{T})$ regret with exponential growth in $n$. The present paper improves to $O(K^n \sqrt{T})$ using stochastic linear bandits (OFUL), and further obtains $O(\sqrt{n^L K^L A^{2L} L \cdot T})$ via the region UCB approach when $L$ is small.
- **vs. Piecewise-linear bandits (Bacchiocchi et al. 2025)**: Their work studies one-dimensional piecewise-linear bandits with unknown breakpoints, while this paper operates in a multi-dimensional setting with known region boundaries. The techniques and results are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic treatment of online learning in multi-follower BSG; the geometric best-response region framework is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐ A purely theoretical work; the appendix includes simple simulations for validation but no large-scale experiments. However, the near-matching upper and lower bounds are themselves highly convincing.
- Writing Quality: ⭐⭐⭐⭐ The narrative progression from single-follower to multi-follower is logically clear, and the key technical ideas are explained with sufficient depth.
- Value: ⭐⭐⭐⭐ Makes important theoretical contributions to the intersection of game theory and online learning; the geometric partitioning methodology is transferable to related problems.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)

<!-- RELATED:END -->
