---
title: >-
  [Paper Note] Near-Optimal Decision Trees in a SPLIT Second
description: >-
  [ICML2025 Oral][Interpretability][Decision Tree Optimization] A family of SPLIT algorithms is proposed, which implements a hybrid scheme of global optimal search near the root of the decision tree and greedy strategies near the leaf nodes, achieving decision tree construction that is over 100 times faster than global optimization methods with almost no loss in accuracy.
tags:
  - "ICML2025 Oral"
  - "Interpretability"
  - "Decision Tree Optimization"
  - "Branch-and-bound"
  - "Dynamic Programming"
  - "Greedy Search"
  - "Rashomon Set"
  - "Sparsity"
date: 2026-05-08
content_hash: 9148aa73724e5bb4
---

# Near-Optimal Decision Trees in a SPLIT Second

**Conference**: ICML2025 Oral  
**arXiv**: [2502.15988](https://arxiv.org/abs/2502.15988)  
**Code**: TBD  
**Area**: Decision Trees / Interpretable Machine Learning  
**Keywords**: Decision Tree Optimization, Branch-and-bound, Dynamic Programming, Greedy Search, Rashomon Set, Interpretability, Sparsity

## TL;DR

A family of SPLIT algorithms is proposed, which implements a hybrid scheme of global optimal search near the root of the decision tree and greedy strategies near the leaf nodes, achieving decision tree construction that is over 100 times faster than global optimization methods with almost no loss in accuracy.

## Background & Motivation

Decision trees are core models in interpretable machine learning. Traditional methods are polarized:

- **Greedy methods** (CART, C4.5): Fast (linear complexity), but without optimality guarantees, resulting in suboptimal accuracy and sparsity.
- **Globally optimal methods** (GOSDT, MurTree, DL8.5): Solved using dynamic programming and branch-and-bound, provably optimal, but scale extremely poorly to high-dimensional features or deep trees.

Empirical studies show an average accuracy gap of 1.1 to 2.2 percentage points between greedy and optimal trees, with gaps on individual datasets reaching as high as 10 percentage points. An ideal scheme should combine the accuracy of optimal methods with the speed of greedy methods.

**Key Insight**: By analyzing the Rashomon set (the set of near-optimal trees), the authors observe that **splits closer to the leaf nodes tend to favor greedy splits**. This implies that global optimization near the leaf nodes yields minimal utility, whereas optimization near the root is critical. This discovery forms the theoretical foundation of the SPLIT algorithm.

## Method

### Problem Formalization

Given a dataset $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i \in \{0,1\}^K$, the objective is to solve:

$$\mathcal{L}^*(D, d, \lambda) = \min_{T \in \mathcal{T}} \sum_{i=1}^{|D|} \frac{1}{N} \left( \ell(T(\mathbf{x}_i), y_i) + \lambda S(T) \right) \quad \text{s.t. depth}(T) \leq d$$

where $S(T)$ is the number of leaves (sparsity penalty), and $\lambda$ is the regularization coefficient.

### SPLIT Algorithm (Algorithm 2)

SPLIT takes a **look-ahead depth** $d_l$ parameter to divide the search into two levels:

1. **Root to look-ahead depth** ($d_l$ layers): Globally optimized using branch-and-bound and dynamic programming.
2. **Look-ahead depth to leaf nodes** ($d - d_l$ layers): Split using greedy information gain.

The recursive optimization equation is:

$$\mathcal{L}(D, d', \lambda) = \begin{cases} \min\left\{\lambda + \frac{|D^-|}{N},\; \min_{f \in \mathcal{F}} \left\{ L(T_g(D(f), d', \lambda)) + L(T_g(D(\bar{f}), d', \lambda)) \right\} \right\} & d' = d - d_l \\ \min\left\{\lambda + \frac{|D^-|}{N},\; \min_{f \in \mathcal{F}} \left\{ \mathcal{L}(D(f), d'-1, \lambda) + \mathcal{L}(D(\bar{f}), d'-1, \lambda) \right\} \right\} & d' > d - d_l \end{cases}$$

The key innovation lies in the `get_bounds` process: at the look-ahead depth, the upper and lower bounds of subproblems are directly set to the loss of the greedy tree (rather than loose bounds), thereby pruning the search space significantly. After solving the prefix tree, a **post-processing optimization** can further replace greedy subtrees with optimal subtrees.

### LicketySPLIT (Algorithm 3)

A polynomial-time variant. It recursively invokes SPLIT with a look-ahead depth of $d_l = 1$: finding the current optimal split first (assuming greedy splits for all subsequent steps), and then recursively invoking LicketySPLIT on the left and right subtrees. The complexity is $O(nk^2 d^2)$, which is vastly superior to the NP-hard complexity of global optimization.

### RESPLIT: Rashomon Set Approximation

Using SPLIT as a subroutine, RESPLIT first generates a set of prefix trees, and then invokes TreeFARMS at each leaf node to compute a shallow near-optimal subtree set, rapidly approximating the complete Rashomon set.

## Key Experimental Results

### Speed Comparison (SPLIT vs GOSDT)

| Dataset | GOSDT Time | SPLIT Time | LicketySPLIT Time | Speedup |
|--------|-----------|-----------|-------------------|--------|
| Bike | ~1000s | ~10s | ~1s | **100-1000×** |
| COMPAS | ~1s | ~1s | ~1s | Comparable |
| Netherlands | ~1s | ~1s | ~1s | Comparable |

- When the sparsity penalty $\lambda$ is small (i.e., when the tree is more complex), SPLIT is 2-3 orders of magnitude faster than GOSDT.
- When $\lambda$ is large (i.e., for ultra-sparse trees), all methods complete within sub-seconds.

### Rashomon Set Computation Acceleration

| Dataset | Full Computation (s) | RESPLIT (s) | Speedup | Variable Importance Correlation Coefficient $\tau$ |
|--------|------------|-----------|--------|------------------------|
| COMPAS | 152 | 18 | 8.4× | 1.000 |
| Spambase | 2659 | 154 | 17.3× | 0.930 |
| Netherlands | 4255 | 216 | 19.7× | 0.932 |
| HELOC | 5564 | 337 | 16.5× | 0.979 |
| HIV | 9273 | 388 | 23.9× | 0.959 |
| Bike | 14330 | 194 | 73.9× | 0.999 |

RESPLIT achieves a 10-74× speedup, and the variable importance ranking is highly consistent with the complete Rashomon set (Pearson correlation $\geq 0.93$).

### Frontier Characteristics

In the three-dimensional space of test loss, sparsity, and training time, SPLIT and LicketySPLIT consistently occupy the ideal regions of the Pareto frontier (low loss, low complexity, and short time), outperforming methods like CART, MAP-Tree, and Thompson Sampling.

## Highlights & Insights

1. **Simple yet profound observation**: Splits near the leaves in near-optimal trees are almost always greedy. This empirical discovery directly drives the algorithm design.
2. **Comprehensive theoretical guarantees**:
    - SPLIT complexity is $O(n(d-d_l)k^{d_l+1} + nk^{d-d_l})$, saving $O(k^{(d-1)/2} \cdot (d/2)!)$ times compared to global optimization at the optimal look-ahead depth of $d_l = (d-1)/2$.
    - LicketySPLIT achieves a polynomial-time complexity of $O(nk^2d^2)$.
    - SPLIT is proven to be arbitrarily better than purely greedy methods (Theorem 6.5).
3. **High practicality**: Plug-and-play modification to the existing GOSDT framework, requiring only changes to how initialization bounds are set.
4. **Rashomon set application**: Extends large-scale Rashomon set computation to deeper and wider feature spaces for the first time.

## Limitations & Future Work

1. **Support for binary features only**: Requires pre-discretization of continuous features (using threshold guessing), which itself introduces suboptimality.
2. **Classification tasks only**: Not yet extended to regression decision trees.
3. **Look-ahead depth requires manual setting**: Although the theory provides an optimal value of $d_l = (d-1)/2$, parameter tuning on specific datasets may still be necessary in practice.
4. **Post-processing still invokes globally optimal solvers**: In extreme cases, post-processing can become a bottleneck.
5. **Multi-class problems**: The paper only discusses binary classification. Although trivial extension is claimed, multi-class experiments are not provided.

## Related Work & Insights

- **GOSDT** (gosdt_guesses): The current fastest globally optimal decision tree method, which SPLIT directly modifies.
- **TreeFARMS**: A Rashomon set computation framework, with RESPLIT acting as its acceleration front-end.
- **MurTree / DL8.5**: Other branch-and-bound methods, to which the core idea of SPLIT can be transferred.
- **Top-k methods**: Share a similar hybrid concept but do not optimize for sparsity, nor do they utilize the insight that "greedy is better near the leaf nodes".

## Rating

- Novelty: ⭐⭐⭐⭐ — Simple and powerful core observation, elegantly designed hybrid strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple datasets, metrics, and baselines, including Rashomon set experiments.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, tightly coupling theory and experiments.
- Value: ⭐⭐⭐⭐ — Highly impactful for the interpretable machine learning community, bridging the gap between greedy and optimal methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Leveraging Predictive Equivalence in Decision Trees](leveraging_predictive_equivalence_in_decision_trees.md)
- [\[NeurIPS 2025\] Empowering Decision Trees via Shape Function Branching](../../NeurIPS2025/interpretability/empowering_decision_trees_via_shape_function_branching.md)
- [\[ICLR 2026\] TreeGrad-Ranker: Feature Ranking via O(L)-Time Gradients for Decision Trees](../../ICLR2026/interpretability/treegrad-ranker_feature_ranking_via_ol-time_gradients_for_decision_trees.md)
- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](../../ICML2026/interpretability/from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)

</div>

<!-- RELATED:END -->
