---
title: >-
  [Paper Note] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits
description: >-
  [AAAI 2026][Reinforcement Learning][Multi-objective bandits] Two elimination-based algorithms, LexElim-Out and LexElim-In, are proposed to simultaneously address regret minimization (RM) and best arm identification (BAI) in lexicographic multi-objective bandits for the first time. LexElim-In breaks the known lower bound of single-objective problems through cross-objective information sharing.
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Multi-objective bandits"
  - "lexicographic preference"
  - "regret minimization"
  - "best arm identification"
  - "elimination algorithm"
date: 2026-05-08
content_hash: 8ca579b1157c24f9
---

# Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits

**Conference**: AAAI 2026
**arXiv**: [2511.05802](https://arxiv.org/abs/2511.05802)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Multi-objective bandits, lexicographic preference, regret minimization, best arm identification, elimination algorithm

## TL;DR

Two elimination-based algorithms, LexElim-Out and LexElim-In, are proposed to simultaneously address regret minimization (RM) and best arm identification (BAI) in lexicographic multi-objective bandits for the first time. LexElim-In breaks the known lower bound of single-objective problems through cross-objective information sharing.

## Background & Motivation

The multi-armed bandit (MAB) is a foundational framework for sequential decision-making, encompassing two major paradigms: regret minimization (RM, minimizing cumulative regret) and best arm identification (BAI, identifying the optimal arm with as few samples as possible). Traditional research focuses on scalar rewards under a single objective, whereas real-world applications frequently involve multiple objectives with hierarchical priorities—e.g., patient safety before cost in clinical trials, or fairness before user engagement in recommender systems.

Lexicographic Bandits are designed precisely for such hierarchical decision-making scenarios: the agent optimizes multiple objectives in priority order, and higher-priority objectives must be satisfied first. Existing work on lexicographic bandits primarily addresses RM, leaving BAI unexplored in this framework. Yet in many settings (e.g., clinical trials), it is necessary to accurately identify the optimal arm while minimizing regret during learning. This motivates the core research question: **Can one design a lexicographic bandit algorithm that unifies both RM and BAI?**

## Method

### Overall Architecture

The paper proposes an elimination-based algorithmic framework whose core idea is to progressively eliminate suboptimal arms using confidence upper bounds. Key definitions include:

- **Lexicographic dominance**: Arm $a_1$ lexicographically dominates $a_2$ if and only if there exists an objective $i$ such that the two arms are tied on the first $i-1$ objectives and $a_1$ is superior on objective $i$.
- **Lexicographically optimal arm** $a_*$: the unique arm not dominated by any other arm.
- **Regret**: $R^i(T) = T \cdot \mu^i(a_*) - \sum_{t=1}^T \mu^i(a_t)$, defined separately for each objective $i$.

### Key Designs

#### Algorithm 1: LexElim-Out (Outer Elimination)

LexElim-Out adopts a layer-by-layer elimination strategy, processing objectives from highest to lowest priority:

1. For each objective $i = 1, \ldots, m$, repeatedly eliminate arms until the active set shrinks to the known size of the optimal set $|\mathcal{O}_*(i)|$.
2. In each round, pull the arm with the highest uncertainty (largest confidence width).
3. Elimination condition: the gap between the empirical best arm's mean and another arm's mean exceeds $2c(a_t)$ (twice the confidence width).
4. After each pull, update the empirical means for all objectives.

The confidence width is defined as: $c(a) = \sqrt{\frac{4}{n(a)} \log\frac{6Km \cdot n(a)}{\delta}}$

**Limitation**: Layer-by-layer processing prevents the algorithm from leveraging information from higher-priority objectives when optimizing lower-priority ones, making early exploration of lower-priority objectives essentially uninformed.

#### Algorithm 2: LexElim-In (Inner Elimination)

LexElim-In employs an innovative cross-objective elimination strategy, simultaneously utilizing information from all objectives in each round:

1. In each round, perform nested filtering across all $m$ objectives: $\mathcal{A}_t^0 = \mathcal{A}_t$, for $i = 1, \ldots, m$.
2. The elimination threshold grows geometrically with objective level: $(2 + 4\lambda + \cdots + 4\lambda^{i-1}) \cdot c(a_t)$.
3. The parameter $\lambda \geq 0$ characterizes the degree of conflict between objectives.
4. The updated active set is $\mathcal{A}_{t+1} = \mathcal{A}_t^m$.

The design of the scaling factor ensures that lower-priority objectives can accelerate elimination when they provide stronger elimination signals, without compromising optimization of higher-priority objectives.

### Loss & Training

This work is driven by theoretical analysis rather than empirical training; there is no loss function in the conventional sense. The core theoretical guarantees are as follows:

**Regret bound of LexElim-Out** (Theorem 1): For any objective $i$,
$$R^i(t) \leq \sum_{j=1}^{i} \sum_{a \in \mathcal{S}(j)} \frac{\gamma^j(\delta) \cdot \Delta^i(a)}{(\Delta^j(a))^2}$$

**Regret bound of LexElim-In** (Theorem 3):
$$R^i(t) \leq \sum_{\Delta^i(a) > 0} \min_{j \in [m]} \left\{ \frac{(\Lambda^j(\lambda))^2 \cdot \Delta^i(a) \cdot \gamma^j(\delta)}{(\Delta^j(a))^2} \right\}$$

The presence of $\min_{j \in [m]}$ allows the algorithm to adaptively exploit auxiliary objectives to accelerate learning, breaking the single-objective lower bound.

**Minimax regret bound of LexElim-In** (Corollary 1): $R^i(t) \leq \widetilde{O}(\Lambda^i(\lambda) \cdot \sqrt{Kt})$, matching the single-objective optimal rate.

## Key Experimental Results

### Main Results (RM)

| Algorithm | Obj. 1 Regret | Obj. 2 Regret | Obj. 3 Regret | Growth Trend |
|-----------|--------------|--------------|--------------|--------------|
| LexElim-In | Lowest | Lowest | Lowest | Sublinear |
| LexElim-Out | Low | Low | Low | Sublinear |
| PF-LEX | Medium | Medium | Medium | $\widetilde{O}(T^{2/3})$ |
| UCBα | Low | Linear growth | Linear growth | Sublinear on Obj. 1 only |

Setting: $K=10$ arms, $m=3$ objectives, $T=10000$ rounds, averaged over 10 independent runs.

### BAI Sample Complexity

| Algorithm | K=10 | K=20 | K=30 | Trend |
|-----------|------|------|------|-------|
| LexElim-In | Best | Best (advantage grows) | Best (advantage significant) | Advantage increases with $K$ |
| LexElim-Out | 2nd best | 2nd best | 2nd best | — |
| UCBα | Moderate | Moderate | Moderate | — |
| EGE | Worst | Worst | Worst | — |

### Ablation Study

**Cross-objective acceleration analysis**: Varying the inter-objective conflict parameter $\lambda$:
- $\lambda = 0$ (no conflict): the large gap on the second objective efficiently eliminates suboptimal arms.
- $\lambda = 1$ (with conflict): only arms far from the optimal arm can be eliminated quickly, as the confidence threshold for the second objective is scaled to $2 + 4\lambda = 6$.

### Key Findings

1. Both LexElim-Out and LexElim-In achieve sublinear regret on all objectives, whereas the single-objective method UCBα performs well only on the first objective.
2. The BAI advantage of LexElim-In grows more pronounced as the number of arms increases, since large gaps on lower-priority objectives provide stronger elimination signals.
3. PF-LEX, despite being a multi-objective method, converges slowly ($O(T^{2/3})$ vs. $O(\sqrt{T})$).

## Highlights & Insights

1. **Breaking the single-objective lower bound**: By sharing information across objectives, LexElim-In surpasses the classical single-objective lower bound $\Omega(\sum 1/\Delta(a))$ on both RM and BAI—the first demonstration that multi-objective structure yields a positive algorithmic benefit in this field.
2. **Theoretical elegance**: The $\min_{j \in [m]}$ term in the regret bound naturally captures the cross-objective acceleration mechanism: a large gap on one objective can "assist" in accelerating the elimination of arms for other objectives.
3. **First unified framework**: This is the first algorithmic framework to jointly address RM and BAI in lexicographic bandits, filling an important theoretical gap.

## Limitations & Future Work

1. LexElim-In requires prior knowledge of $\lambda$ (characterizing inter-objective conflict); eliminating this dependency is an important future direction.
2. LexElim-Out requires prior knowledge of $|\mathcal{O}_*(i)|$ (the size of the optimal arm set at each level), which may be unavailable in practice.
3. Experiments are conducted only on synthetic data, lacking validation in real-world application scenarios.
4. Tight lower bounds for lexicographic RM and BAI have yet to be established to fully characterize inter-objective interactions.

## Related Work & Insights

- **Relation to UCBα**: UCBα unifies single-objective RM and BAI; this paper extends that approach to the multi-objective lexicographic setting.
- **Relation to PF-LEX**: PF-LEX addresses only lexicographic RM with $O(T^{2/3})$ regret; this paper achieves $O(\sqrt{T})$.
- **Inspiration**: The cross-objective acceleration mechanism may generalize to other multi-objective online learning problems, such as Pareto bandits or contextual bandits.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to unify lexicographic RM+BAI and break the single-objective lower bound
- Experimental Thoroughness: ⭐⭐⭐ — Only synthetic experiments; real-world scenarios absent
- Writing Quality: ⭐⭐⭐⭐ — Theoretically rigorous and clearly structured
- Value: ⭐⭐⭐⭐ — Strong theoretical contributions that fill an important gap

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)
- [\[ICLR 2026\] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](../../ICLR2026/reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)
- [\[AAAI 2026\] Perturbing Best Responses in Zero-Sum Games](perturbing_best_responses_in_zero-sum_games.md)
- [\[NeurIPS 2025\] Simultaneous Swap Regret Minimization via KL-Calibration](../../NeurIPS2025/reinforcement_learning/simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [\[NeurIPS 2025\] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](../../NeurIPS2025/reinforcement_learning/comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)

</div>

<!-- RELATED:END -->
