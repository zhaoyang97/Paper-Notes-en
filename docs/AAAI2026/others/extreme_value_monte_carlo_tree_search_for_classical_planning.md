---
title: >-
  [Paper Note] Extreme Value Monte Carlo Tree Search for Classical Planning
description: >-
  [AAAI 2026][MCTS] This paper applies Peaks-Over-Threshold Extreme Value Theory (POT EVT) to provide a statistical foundation for Full Bellman Backup in MCTS for classical planning. It proposes the UCB1-Uniform bandit alg…
tags:
  - "AAAI 2026"
  - "MCTS"
  - "Extreme Value Theory"
  - "UCB1-Uniform"
  - "Generalized Pareto"
  - "Classical Planning"
  - "Heuristic Search"
  - "Full Bellman Backup"
date: 2026-05-08
content_hash: f38b00d8dd7da129
---

# Extreme Value Monte Carlo Tree Search for Classical Planning

**Conference**: AAAI 2026
**arXiv**: [2405.18248](https://arxiv.org/abs/2405.18248)  
**Code**: [https://github.com/guicho271828/pyperplan-mcts](https://github.com/guicho271828/pyperplan-mcts)  
**Area**: Classical Planning / Search Algorithms
**Keywords**: MCTS, Extreme Value Theory, UCB1-Uniform, Generalized Pareto, Classical Planning, Heuristic Search, Full Bellman Backup

## TL;DR

This paper applies Peaks-Over-Threshold Extreme Value Theory (POT EVT) to provide a statistical foundation for Full Bellman Backup in MCTS for classical planning. It proposes the UCB1-Uniform bandit algorithm, which uses MLE under a uniform distribution (a special case of the Generalized Pareto distribution) to guide action selection, outperforming GBFS by 67.8 instances and Softmin-Type(h) by 33.2 instances under a $10^4$ node budget on Pyperplan.

## Background & Motivation

**Background**: MCTS combined with multi-armed bandits (MAB) has achieved success in game-playing and reinforcement learning, yet has long underperformed in classical planning. Prior work (Wissow & Asai 2024) identified that UCB1's bounded support assumption $[0, c]$ is incompatible with planning heuristic values, and proposed a Gaussian bandit (UCB1-Normal2) to address this.

**Limitations of Prior Work**: Although UCB1-Normal2 no longer violates the support assumption, the Gaussian support $(-\infty, +\infty)$ constitutes an **under-specification** of heuristic value distributions, which are in fact semi-bounded: $h \in [h^+, \infty)$ (e.g., $h^{FF}$) or $h \in [0, h^+]$ (e.g., $h^{max}$), suggesting a tighter support is warranted.

**Key Challenge**: Full Bellman Backup—propagating the minimum/maximum value among child nodes—works well in practice, yet lacks statistical justification within the theoretical frameworks of UCB1 and UCB1-Normal2, whose regret bounds are derived around the **mean** rather than **extreme values**, creating a theory–practice disconnect.

**Goal**: (a) Identify a theoretically accurate distributional model for heuristic values that is neither over- nor under-specified; (b) provide a rigorous statistical foundation for Full Bellman Backup; (c) address the destructive effect of dead-end nodes ($h = \infty$) on mean estimation.

**Key Insight**: The paper draws on Extreme Value Theory (EVT), leveraging the POT limit theorem—which states that exceedances over a high threshold converge to a Generalized Pareto (GP) distribution—to unify the focus on extreme values (minimum $h$) in planning and to justify discarding dead-end samples.

**Core Idea**: The GP distribution is specialized to the uniform distribution $U(l, u)$ (corresponding to the short-tailed GP with $\xi = -1$), whose MLE is naturally the sample min/max, directly corresponding to Full Bellman Backup. This motivates the UCB1-Uniform bandit, which admits a polynomial–constant regret bound.

## Method

### Overall Architecture

Within the MCTS framework (Trial-Based Heuristic Tree Search), UCB1-Uniform replaces UCB1/UCB1-Normal2 as the MAB algorithm for action selection. The backup strategy employs Full Bellman Backup (propagating the min/max of leaf node values in the subtree), which is theoretically consistent with the MLE estimator of the uniform distribution. The search targets the agile setting, focusing on the speed of finding a solution rather than solution quality.

### Key Design 1: POT EVT Statistical Framework

- **Function**: Models the distribution of heuristic values using Peaks-Over-Threshold extreme value theory, treating exceedances over threshold $\theta$ as following a GP distribution $\text{GP}(\theta, \sigma, \xi)$.
- **Mechanism**: The POT limit theorem is analogous to the Central Limit Theorem—CLT states that sample means converge to a Gaussian; POT states that sample extremes converge to a GP—neither assumption requires knowledge of the original distribution. Short-tailed GP ($\xi < 0$) has an upper bound $\theta - \sigma/\xi$, enabling estimation of $-h^+$; furthermore, by conditioning on $x > \theta$, GP naturally excludes dead ends.
- **Design Motivation**: Search naturally concentrates on low-$h$ regions (experiments confirm only 3.3% of nodes satisfy $h(s) > h(I)$), so no explicit threshold is required; the search process itself provides implicit filtering.

### Key Design 2: UCB1-Uniform Bandit

- **Function**: Proposes a new MAB algorithm that assumes each arm's reward follows a uniform distribution $U(l_i, u_i)$ with unknown support, selecting the arm minimizing LCB1-Uniform.
- **Mechanism**: $\text{U/LCB1-Uniform}_i = \frac{\hat{u}_i + \hat{l}_i}{2} \pm (\hat{u}_i - \hat{l}_i)\sqrt{6 t_i \log T}$, where $\hat{l}_i = \min_j r_{ij}$ and $\hat{u}_i = \max_j r_{ij}$. The mean estimate $(\hat{u}_i + \hat{l}_i)/2$ is derived from the expectation of the uniform distribution, and the exploration term is proportional to the support width $\hat{u}_i - \hat{l}_i$.
- **Design Motivation**: (a) The uniform distribution is the $\xi = -1$ special case of GP, avoiding the numerical difficulties of three-parameter GP estimation; (b) the MLE corresponds to min/max, perfectly aligning with Full Bellman Backup and eliminating the theory–practice gap; (c) for plateaus where the support collapses ($\hat{u}_i = \hat{l}_i$), the exploration term vanishes, inducing depth-first traversal through the plateau.

### Key Design 3: Spread-Aware Exploration and Plateau Escape

- **Function**: When two subtrees are equally "informative" (identical $u, l$) but have been explored to different extents, UCB1-Uniform favors continuing to explore the more-visited subtree.
- **Mechanism**: The exploration term contains $\sqrt{t_i}$ (rather than $\sqrt{1/t_i}$), so more pulls increase the exploration bonus, inducing a preference to continue—effectively implementing a depth-first plateau traversal strategy.
- **Design Motivation**: In regions where heuristic values are flat (plateaus), distributing search effort evenly prevents breakthrough in any direction; concentrating resources on one direction accelerates escape.

### Key Design 4: Theoretical Grounding for Dead-End Handling

- **Function**: Excludes dead-end nodes ($h = \infty$) from the sample set.
- **Mechanism**: The GP distribution is defined conditionally on $x > \theta$, naturally excluding values below the threshold (including $-\infty$).
- **Design Motivation**: Prior work (Schulte & Keller 2014) discarded dead ends heuristically without statistical justification; the POT framework provides a principled theoretical basis for this practice.

## Key Experimental Results

### Table 1: Pyperplan Experiments (10,000 node budget, $h^{FF}$, averaged over 5 seeds)

| Algorithm | $h^{FF}$ | $h^{add}$ | $h^{max}$ | $h^{GC}$ | $h^{FF}$+PO |
|-----------|---------|----------|----------|---------|------------|
| GBFS | 538 | 518 | 224 | 354 | — |
| Softmin-Type(h) | 576 | 542.6 | 297.2 | 357.6 | 575.8 |
| GUCT-Normal2 | 582.95 | 538 | 316.6 | 380.6 | 623.2 |
| **GUCT-Uniform** | **606.4** | **563.4** | **455.6** | **492.2** | **635.6** |
| GUCT*-Normal2 | 567.2 | 533.8 | 263 | 341.2 | 619.8 |
| MaxSearch | 253.75 | 243.4 | 260 | 255.2 | 368.6 |

GUCT-Uniform achieves the best performance across all heuristics, with a particularly large advantage on the weak heuristic $h^{max}$ (+139 over GUCT-Normal2).

### Table 2: Fast Downward IPC2018 Experiments ($h^{FF}$, 5min/8GB, averaged over 3 seeds)

| Algorithm | Problems Solved | IPC Score |
|-----------|----------------|-----------|
| GBFS | 67.0 | 39.5 |
| Softmin-Type(h) | 79.6 | 47.0 |
| GUCT-Normal2 | 81.6 | 48.6 |
| **GUCT-Uniform** | **80.0** | **53.2** |

The number of problems solved is comparable (subject to C++ implementation differences), but GUCT-Uniform achieves a substantially higher IPC score, which measures solution speed.

## Key Findings

1. **Full Bellman Backup is theoretically consistent with the uniform distribution**: min/max propagation is precisely the MLE of $U(l, u)$, unlike UCB1's $[0, c]$ or Normal2's $\mathcal{N}(\mu, \sigma)$, resolving the longstanding theory–practice disconnect.
2. **A non-asymptotically optimal bandit can outperform an asymptotically optimal one**: UCB1-Uniform (polynomial regret bound) significantly outperforms CHK-Uniform (asymptotically optimal), mirroring the phenomenon of UCB1-Normal2 outperforming UCB1-Normal.
3. **GUCT*-Normal2 underperforms GUCT-Normal2**: Full Bellman Backup is actually harmful when combined with a Gaussian bandit, since min/max propagation contradicts mean-based estimation.
4. **The Max-$k$ bandit paradigm is ill-suited for planning**: All three Max-$k$ algorithms perform far worse than GUCT-Uniform (a gap exceeding 300 instances), as they target heavy-tailed rather than short-tailed distributions.
5. **Advantages are larger under weaker heuristics**: GUCT-Uniform's gains on $h^{max}$ and $h^{GC}$ are on the order of hundreds of instances, indicating that superior exploration strategies are especially critical when heuristic information is weak.

## Highlights & Insights

- **Theory-driven algorithm design**: The paper starts from the question "which distribution correctly models heuristic values?" and derives the bandit algorithm via EVT—prioritizing theory over ad hoc design, rather than engineering an algorithm and then proving its properties.
- **Elegant specialization**: Restricting GP to the uniform case ($\xi = -1$) trades one degree of freedom for tractable parameter estimation, and the resulting MLE is exactly min/max, seamlessly connecting to the existing Full Bellman Backup.
- **Plateau escape mechanism**: The exploration term in UCB1-Uniform scales with $\sqrt{t_i}$ rather than inversely, naturally inducing depth-first behavior in flat heuristic regions—an inversion of the conventional "exploration = diversification" intuition.
- **Unified resolution of three independent problems**: Support range modeling, theoretical justification of Full Bellman Backup, and dead-end handling are all addressed within a single POT framework.

## Limitations & Future Work

1. **Limited to the agile setting**: Performance in satisficing or optimal planning scenarios is not evaluated; the behavior of UCB1-Uniform when solution quality matters remains unknown.
2. **Uniform distribution is a coarse approximation of GP**: Fixing $\xi = -1$ constrains tail behavior; estimating $\xi$ could yield greater precision, though numerical difficulties arise—robust GP parameter estimation methods warrant future investigation.
3. **C++ implementation gap**: The large advantage observed in Pyperplan narrows in Fast Downward (80 vs. 81.6 problems solved), suggesting that low-level implementation efficiency influences comparisons.
4. **No combination with novelty/BFWS**: The current work employs single-heuristic MCTS; combining with Bilevel MCTS and novelty measures (as in Nεbula) is identified as a natural future direction.
5. **Implicit threshold selection**: The approach relies on the search process itself to filter high-$h$ states, without an adaptive threshold mechanism.

## Related Work & Insights

- **Wissow & Asai (2024)**: The precursor work proposing the Gaussian bandit UCB1-Normal2; the present paper refines the distributional assumption therein.
- **Schulte & Keller (2014, THTS)**: The classical MCTS framework for planning that introduced Full Bellman Backup without providing MAB statistical justification.
- **Softmin-Type(h) (Kuroiwa & Beck 2022)**: State-of-the-art diversified search based on a queue-competition mechanism.
- **Max-$k$ Bandit (Cicirello & Smith 2004)**: Although also grounded in EVT, it targets heavy-tailed distributions for maximum-value optimization, which is contrary to the short-tailed minimum-value objective in planning.
- **Insights**: MAB algorithm design must be aligned with the statistical properties of the reward distribution—incorrect distributional assumptions are more damaging than incorrect algorithmic structure; EVT has broad applicability in non-machine-learning domains such as planning and combinatorial optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic application of POT extreme value theory to MCTS for planning, unifying three theoretical deficiencies in a single framework
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across multiple heuristics and bandit algorithms, though C++ experiments are limited to IPC2018
- **Writing Quality**: ⭐⭐⭐⭐⭐ The CLT→EVT analogy is textbook-quality exposition, balancing rigorous derivation with intuitive explanation
- **Value**: ⭐⭐⭐⭐ Substantially advances the theoretical foundations of MCTS in planning; UCB1-Uniform has practical value, though integration with additional techniques is needed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning](bilevel_mcts_for_amortized_o1_node_selection_in_classical_planning.md)
- [\[ICML 2026\] NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](../../ICML2026/others/nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)
- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](../../ICLR2026/others/compositional_diffusion_long_horizon_planning.md)
- [\[ICML 2026\] Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach](../../ICML2026/others/markov_chain_monte_carlo_without_evaluating_the_target_an_auxiliary_variable_app.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)

</div>

<!-- RELATED:END -->
