---
title: >-
  [Paper Note] Decision Tree Learning on Product Spaces
description: >-
  [ICML 2026][top-down greedy heuristic] This work extends the theoretical guarantees of "top-down greedy decision tree heuristics" from Blanc et al. (ITCS'20) from the uniform distribution to **arbitrary product distribut…
tags:
  - "ICML 2026"
  - "top-down greedy heuristic"
  - "influence split"
  - "product distribution"
  - "PAC learning"
  - "parameter-free"
date: 2026-05-08
content_hash: d0ce32867f4ac76f
---

# Decision Tree Learning on Product Spaces

**Conference**: ICML 2026  
**arXiv**: [2605.12983](https://arxiv.org/abs/2605.12983)  
**Code**: None (theoretical paper)  
**Area**: Learning Theory / Decision Tree Learning  
**Keywords**: top-down greedy heuristic, influence split, product distribution, PAC learning, parameter-free

## TL;DR
This work extends the theoretical guarantees of "top-down greedy decision tree heuristics" from Blanc et al. (ITCS'20) from the uniform distribution to **arbitrary product distributions**, providing an upper bound of $\exp(\Delta_\mathrm{opt} D_\mathrm{opt}\log(e/\epsilon))$ (strictly tighter than ITCS'20 in the full binary tree case), and is **completely parameter-free**—it can be run without prior knowledge of the optimal tree size or depth.

## Background & Motivation

**Background**: Decision trees (ID3 / C4.5 / CART) dominate numerous practical tasks with "top-down greedy + influence (or equivalently entropy/Gini) splits," but theoretical analysis has long lagged behind practice—algorithms by Ehrenfeucht-Haussler, Mehta-Raghavan, Blanc, etc., are either brute-force searches or require prior knowledge of $s$ (optimal tree size), making them far from real algorithms.

**Limitations of Prior Work**: (a) Blanc et al. ITCS'20 first provided strict guarantees for top-down greedy, but their analysis heavily relies on **uniform distribution + Boolean Fourier analysis**, limiting applicability; (b) Real-world feature distributions are highly non-uniform, so theoretical guarantees do not effectively explain practice; (c) Even Blanc et al.'s algorithm requires prior knowledge of $s$ to select hyperparameters, making it impractical.

**Key Challenge**: The "adaptive, locally maximal influence split, parameter-free" nature of practical algorithms vs. the "global optimization + uniform distribution dependence + need for $s$" of theoretical algorithms.

**Goal**: (1) Extend top-down greedy guarantees to arbitrary product distributions $\mu=\mu_1\times\cdots\times\mu_n$; (2) Strictly tighten Blanc et al.'s upper bound in the full binary tree case; (3) Provide a parameter-free implementation and a robust version (tolerant to sample estimation errors).

**Key Insight**: Avoid Fourier-analytic tools and instead use "two depth parameters"—maximum depth $D_\mathrm{opt}$ (for the total influence ≤ depth × variance inequality, Lemma 4.2) and average depth $\Delta_\mathrm{opt}$ (for O'Donnell 2005's max-influence ≥ variance / average depth inequality, Lemma C.1). The product $\Delta_\mathrm{opt} D_\mathrm{opt}$ is the mixed driving term.

**Core Idea**: Use "cost = $\sum_\mathrm{leaves} p_v \cdot \mathrm{Inf}(f_v)$" as a potential function, proving (a) error ≤ cost, (b) each split strictly reduces cost by the score of that leaf, (c) provide lower bounds on the score in two cost intervals, thus bounding the total number of steps.

## Method

This is a purely theoretical paper; "method" = algorithm (from Blanc et al. ITCS'20) + new analysis and parameter-free implementation.

### Overall Architecture
Algorithm `BuildTopDownDT(f, ε)`: Start from a single-leaf tree; in each round, compute score = $p_v \cdot \max_i \mathrm{Inf}^\mu_i(f_v)$ for each leaf, select the leaf with the highest score, and split on its most influential variable; stop when the $f$-completion (by majority label) $\epsilon$-approximates $f$. The analysis framework centers on the two-phase descent of the cost potential function: first from $\mathrm{Inf}(f)$ down to $\epsilon D_\mathrm{opt}$ (Phase 1, Lemma 4.6), then from $\epsilon D_\mathrm{opt}$ down to error ≤ $\epsilon$ (Phase 2, Lemma 4.7).

### Key Designs

1. **Influence and Cost Potential under Product Distributions**:

    - Function: Translates "what the algorithm does at each step" into a monotonically decreasing scalar; provides the connection error ≤ cost (Lemma 4.1).
    - Mechanism: Define $\mathrm{Inf}^\mu_i(f)=\Pr_{x\sim\mu}[f(x)\neq f(x^{(i)})]$, where $x^{(i)}$ is obtained by resampling the $i$-th coordinate according to $\mu_i$; leaf score is $\mathrm{Score}(v)=p_v\cdot \max_i \mathrm{Inf}_i(f_v)$; tree cost is $\mathrm{cost}(T^\circ)=\sum_{v\in\mathrm{leaves}} p_v\cdot \mathrm{Inf}(f_v)$. Lemma 4.3 proves that splitting leaf $v$ strictly decreases cost by $\mathrm{Score}(v)$, translating "algorithmic greediness" into "cost descent rate."
    - Design Motivation: Does not rely on Fourier coefficients, only on probability-defined influence and product structure, naturally extending the analysis to arbitrary product distributions.

2. **Upper Bound Driven by Two Depth Parameters**:

    - Function: Achieves the tighter upper bound $\max\bigl((e\Delta_\mathrm{opt}/(\epsilon D_\mathrm{opt}))^{\Delta_\mathrm{opt} D_\mathrm{opt}}, e^{\Delta_\mathrm{opt} D_\mathrm{opt}}\bigr)$ (Theorem 1.1).
    - Mechanism: $D_\mathrm{opt}$ enters via Lemma 4.2's $\mathrm{Inf}(f)\le D(T)\cdot \mathrm{Var}(f)$; $\Delta_\mathrm{opt}$ enters via O'Donnell et al.'s max-influence inequality $\max_i \mathrm{Inf}_i(f)\ge \mathrm{Var}(f)/\Delta(T)$. Two score lower bounds (Lemma 4.4 for cost ≤ ε$D_\mathrm{opt}$ phase, Lemma 4.5 for cost ≥ ε$D_\mathrm{opt}$ phase) respectively bound the number of steps in each phase; their sum yields the mixed bound. For path-like trees ($\Delta_\mathrm{opt}$ constant, $D_\mathrm{opt}=n$), $\Delta_\mathrm{opt} D_\mathrm{opt}$ is much smaller than $D_\mathrm{opt}^2$, giving an exponentially better bound than using only $D_\mathrm{opt}$; for balanced trees ($D_\mathrm{opt}=\Delta_\mathrm{opt}=\log s$), the bound is $s^{\log s\log(e/\epsilon)}$, slightly better than Blanc et al.'s $s^{O(\log(s/\epsilon)\log(1/\epsilon))}$.
    - Design Motivation: Under uniform distribution, $D_\mathrm{opt}=\Delta_\mathrm{opt}$ so prior work did not need to separate them; under non-uniform distributions, the two can differ exponentially, so tracking them separately is key to tightening the bound.

3. **Parameter-free + Robust Approximate Implementation**:

    - Function: Can be run without prior knowledge of $s$ or $D_\mathrm{opt}$; tolerates score estimation errors (as long as the selected leaf's score is at least 1/4 of the true maximum).
    - Mechanism: Theorem 5.1 proves that as long as in each step the selected leaf satisfies $\mathrm{Score}(l')\ge \frac14 \max_l \mathrm{Score}(l)$, the upper bound only degrades to exponent $4\Delta_\mathrm{opt} D_\mathrm{opt}$ (effectively treating $\Delta$ as 4 times larger). The score is estimated using the unbiased sample estimator $\widehat{\mathrm{Score}}(l,i,E_i)=\frac{1}{|E_i|}\sum_{(x,x^{(i)})}\mathbf 1[x,x^{(i)}\to l]\mathbf 1[f(x)\neq f(x^{(i)})]$ plus Chernoff bound, giving per-step sample complexity $M_S(j,\delta,\epsilon,n)=\frac{12(j+1)n}{\epsilon}\log\frac{4j^2(j+1)n}{\delta}$; majority-vote ERM is used to estimate tree error as the stopping criterion.
    - Design Motivation: Previous theoretical algorithms all required "knowing $s$ in advance to set stopping conditions/hyperparameters," which is meaningless in practice; this work provides the first directly runnable version.

### Loss & Training
N/A (theoretical paper). The algorithm's optimization objective is to "split the leaf with the highest score," equivalent to greedily reducing cost; the stopping condition is estimated error ≤ ε (ERM majority label + Chernoff sample size).

## Key Experimental Results

### Main Results (Theoretical, Not Empirical Data)

| Setting | Upper Bound | Comparison with Blanc et al. ITCS'20 |
|---|---|---|
| Arbitrary product distribution, general tree | $\max((e\Delta_\mathrm{opt}/(\epsilon D_\mathrm{opt}))^{\Delta_\mathrm{opt} D_\mathrm{opt}}, e^{\Delta_\mathrm{opt} D_\mathrm{opt}})$ | Extended to non-uniform distributions |
| Uniform distribution + full binary tree ($\Delta_\mathrm{opt}=D_\mathrm{opt}=\log s$) | $s^{\log s\cdot\log(e/\epsilon)}$ | Slightly tighter than $s^{O(\log(s/\epsilon)\log(1/\epsilon))}$ |
| Balanced tree ($D_\mathrm{opt},\Delta_\mathrm{opt}\in O(\log s)$) | $s^{O(\log s\cdot\log(e/\epsilon))}$ | Same as above |
| Path-like tree ($\Delta_\mathrm{opt}$ constant, $D_\mathrm{opt}=n$) | Exponentially better than "only $D_\mathrm{opt}^2$" bound | Essential gain from parameter separation |

### Key Robustness Results

| Configuration | Upper Bound | Notes |
|---|---|---|
| Exact score | Exponent in $\Delta_\mathrm{opt} D_\mathrm{opt}$ | Theorem 1.1 |
| Selected leaf score ≥ ¼ maximum | Exponent in $4\Delta_\mathrm{opt} D_\mathrm{opt}$ | Theorem 5.1, tolerant to sample estimation |
| Per-step sample complexity | $\tilde O((j+1)n/\epsilon)$ | $\delta$/total failure probability = δ/2 |

### Key Findings
- $D_\mathrm{opt}$ and $\Delta_\mathrm{opt}$ must be tracked separately; for path-like trees, this separation yields exponential improvement, otherwise the symmetry of the uniform distribution masks this phenomenon.
- The greedy algorithm is highly robust—selecting a ¼-approximately optimal leaf suffices, making sample estimation naturally feasible; this explains why practical CART/C4.5 still work on noisy data.
- Koch et al. (2023) have shown that decision tree learning has no poly-size algorithm, so quasi-polynomial dependence on $s$ is nearly tight—this upper bound is "close to" the lower bound.

## Highlights & Insights
- Completely avoids Boolean Fourier tools—this means all future theoretical analyses of decision trees on product spaces (noise stability, agnostic learning, etc.) now have a "non-Fourier path" to follow.
- Separately bounding cost with $D_\mathrm{opt}$ and $\Delta_\mathrm{opt}$ in two inequalities yields a rare "mixed parameter exponential bound"—this technique can transfer to any "score-greedy + monotone potential function" algorithm analysis.
- Parameter-free implementation + 1/4-approximate robustness truly brings "theoretical algorithms" to "runnable algorithms"—a step often omitted in previous theoretical work.

## Limitations & Future Work
- Still a quasi-polynomial bound ($s^{\log s}$), not polynomial; Koch et al.'s lower bound shows this is unavoidable in the worst case, but real data may be much better than worst-case, and no distribution-specific tight bounds are given.
- Only applies to product distributions $\mu=\mu_1\times\cdots\times\mu_n$—real data features are **correlated**, so this work does not apply; extension to non-product (Markov/general) distributions is open.
- Error metric is 0-1 loss (Boolean function $\{\pm 1\}$); does not directly cover regression trees or soft labels.
- Sample complexity in the worst case still depends on $n$ (number of features), so sampling cost is high in high-dimensional sparse scenarios.

## Related Work & Insights
- **vs Blanc et al. (ITCS'20)**: Also analyzes top-down greedy, but only supports uniform distribution and relies on Fourier; this work circumvents that with product-space influence + variance-depth inequalities.
- **vs Mehta-Raghavan (TCS'02)**: Provides $n^{O(\log(s/\epsilon))}$ DP algorithm, but only covers uniform distribution and deviates from practical algorithms.
- **vs Blanc et al. (FOCS'22)**: Designs polylog-influential variable algorithms with $n^{O(\log\log n)}$ runtime, more complex and non-greedy; this work, conversely, gives strict analysis for the real greedy algorithm.
- **vs Koch et al. (SODA'23, COLT'24)**: Provides superpolynomial / NP-hard lower bounds, showing this work's quasi-poly upper bound is nearly tight.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but uses "two-depth mixed + non-Fourier" techniques to extend guarantees to arbitrary product distributions, and the parameter-free implementation grounds the theory.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical paper, no empirical experiments; but theoretical analysis aligns with upper and lower bounds.
- Writing Quality: ⭐⭐⭐⭐ Lemma chain is clear and progressive, proof sketches provide intuition; a few symbols are dense.
- Value: ⭐⭐⭐⭐ A meaningful bridge for decision tree theory—the first to provide strict guarantees for real greedy algorithms in settings close to real data distributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](../../NeurIPS2025/others/product_distribution_learning_with_imperfect_advice.md)
- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](../../AAAI2026/others/dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)

</div>

<!-- RELATED:END -->
