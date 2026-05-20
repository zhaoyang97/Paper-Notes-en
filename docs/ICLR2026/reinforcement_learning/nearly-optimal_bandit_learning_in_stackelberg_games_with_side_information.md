---
title: >-
  [Paper Note] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information
description: >-
  [ICLR 2026][Reinforcement Learning][Stackelberg games] By linearizing the leader's utility space in Stackelberg games, this paper proposes a reduction to linear contextual bandits that improves the regret bound from $\ti…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Stackelberg games"
  - "online learning"
  - "contextual bandits"
  - "side information"
  - "regret bounds"
date: 2026-05-08
content_hash: 86e8571daa0d07a1
---

# Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information

**Conference**: ICLR 2026
**arXiv**: [2502.00204](https://arxiv.org/abs/2502.00204)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Stackelberg games, online learning, contextual bandits, side information, regret bounds

## TL;DR

By linearizing the leader's utility space in Stackelberg games, this paper proposes a reduction to linear contextual bandits that improves the regret bound from $\tilde{O}(T^{2/3})$ to the nearly-optimal $\tilde{O}(T^{1/2})$ under bandit feedback with side information.

## Background & Motivation

Stackelberg games arise broadly in sequential decision-making scenarios (e.g., airport security, wildlife conservation), where a leader commits to a mixed strategy and a follower best-responds. In practice, payoffs often depend on time-varying **side information** (e.g., weather, airport congestion), making the problem considerably more challenging.

Harris et al. (2024) first formalized online learning in contextual Stackelberg games with side information. Under full feedback, a regret bound of $\tilde{O}(T^{1/2})$ has been achieved; however, under the more realistic **bandit feedback** setting (where only the follower's action is observed), the best known regret bound is $\tilde{O}(T^{2/3})$, leaving a notable gap with the lower bound.

The root cause lies in the fact that the leader's utility is a **nonlinear function** of her mixed strategy (due to the follower's best response), making direct application of standard bandit algorithms difficult. The paper's starting point is the observation that, although utility is nonlinear in strategy space, it possesses a **linear structure in utility space**, which can be exploited to achieve nearly-optimal learning.

## Method

### Overall Architecture

The algorithmic core is a **reduction framework** (Algorithm 1): it reduces the Stackelberg game learning problem with side information to a linear contextual bandit problem. At each round, the linear contextual bandit algorithm recommends a vector in utility space, which is then inverted back to the leader's mixed strategy.

### Key Designs

1. **Utility Space Linearization**:

    - Function: Express the leader's utility in inner-product form.
    - Mechanism: Define the utility vector $\mathbf{u}(\mathbf{z},\mathbf{x}) = [u(\mathbf{z},\mathbf{x},b_1(\mathbf{z},\mathbf{x})),\dots,u(\mathbf{z},\mathbf{x},b_K(\mathbf{z},\mathbf{x}))]^\top$, so that the leader's utility can be written as $u(\mathbf{z}_t,\mathbf{x}_t,b_{f_t}(\mathbf{z}_t,\mathbf{x}_t)) = \langle \mathbf{u}(\mathbf{z}_t,\mathbf{x}_t), \mathbf{1}_{f_t} \rangle$.
    - Design Motivation: Exploit the structure induced by the finite number $K$ of follower types to convert the nonlinear problem into a linear form, enabling direct application of mature linear contextual bandit algorithms.

2. **Strategy Space Discretization**:

    - Function: Reduce the infinite leader mixed-strategy space to a finite set of approximate extreme points $\mathcal{E}_t$.
    - Mechanism: For each context $\mathbf{z}_t$, compute the $\delta$-approximate extreme point set $\mathcal{E}_{\mathbf{z}_t}(1/T)$ over all contextual best-response regions.
    - Design Motivation: Ensure a finite action set to satisfy the requirements of linear bandit algorithms, while incurring only $O(1)$ additional regret.

3. **Two Instantiations**:

    - **Adversarial context + stochastic follower**: Uses the OFUL algorithm (Abbasi-Yadkori et al., 2011), employing the optimism principle to learn the follower type distribution $\mathbf{p}^*$, achieving $\tilde{O}(K\sqrt{T})$ regret.
    - **Stochastic context + adversarial follower**: Uses the logdet-FTRL algorithm (Liu et al., 2023) with log-determinant regularization to handle the adversarial follower, achieving $\tilde{O}(K^{2.5}\sqrt{T})$ regret.

### Extension: Unknown Utility Functions

When the leader's utility function is unknown but satisfies the linear assumption $u(\mathbf{z},a_l,a_f) = \langle \mathbf{z}, U(a_l,a_f) \rangle$, by constructing a high-dimensional feature vector $\mathbf{h}(\mathbf{z},\mathbf{x}) \in \mathbb{R}^{d \times K \times A_l \times A_f}$, the $\tilde{O}(\sqrt{T})$ regret bound is maintained at the cost of polynomial dimensional factors.

## Key Experimental Results

### Main Results

| Setting | Metric | Alg1-OFUL | Random Baseline | Optimal Policy |
|---------|--------|-----------|-----------------|----------------|
| 4 types / 4 actions / d=2 | Cumulative utility (T=200) | Near-optimal | Far below optimal | Highest |
| Same | Cumulative regret (T=200) | Sublinear growth | Linear growth | 0 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| Alg1-OFUL vs. Harris Algorithm 3 | Cumulative utility | Proposed method significantly outperforms the prior method |
| Follower utility depends on context | Applicability | Prior method inapplicable; proposed method applicable |

### Key Findings
- The algorithm empirically outperforms prior methods in synthetic experiments.
- The linearization reduction significantly simplifies algorithm design, eliminating the need to apply Carathéodory's theorem at each round.
- Results apply to secondary-price auctions and Bayesian information design, among other settings.

## Highlights & Insights

- **Elegant reduction**: Operating in utility space rather than strategy space perfectly linearizes an otherwise nonlinear problem.
- **Closing the theoretical gap**: Improves bandit-feedback regret from $\tilde{O}(T^{2/3})$ to $\tilde{O}(T^{1/2})$, matching the lower bound.
- **Broad applicability**: The framework extends to auctions, information design, and other problems with Stackelberg structure.

## Limitations & Future Work

- Worst-case per-round runtime is exponential, inheriting the NP-hardness of Stackelberg games.
- The follower's utility function is assumed to be known (though this requirement has been relaxed for the leader's utility).
- Experiments are small-scale (T=200, 4 types), lacking large-scale validation.

## Related Work & Insights

- Conceptually related to Bernasconi et al. (2023), but addresses a more complex contextual setting.
- The discretization technique may inspire online learning in other continuous action space settings.
- The linearity assumption in the unknown-utility extension may be further relaxed.

## Rating
- Novelty: ⭐⭐⭐⭐ The reduction idea is concise and effective, though the underlying tools (OFUL, etc.) are existing.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical contribution; experiments are relatively simple.
- Writing Quality: ⭐⭐⭐⭐⭐ Exposition is clear and theoretical treatment is rigorous.
- Value: ⭐⭐⭐⭐ Fully resolves an open theoretical problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](../../ACL2026/reinforcement_learning/the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)

</div>

<!-- RELATED:END -->
