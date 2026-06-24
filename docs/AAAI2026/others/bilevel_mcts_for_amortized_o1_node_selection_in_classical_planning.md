---
title: >-
  [Paper Note] Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning
description: >-
  [AAAI 2026][MCTS] This paper proposes Bilevel MCTS, which launches a depth-proportional budgeted best-first search at the leaf node selected by MCTS, reducing the amortized node-selection complexity from $O(\log N)$ to $O(1)$. Complemented by Tree Collapsing to reduce the number of action-selection steps, these components are integrated into the Nεbula planner, which solves 192.2/230.6 problems on IPC2018/2023 benchmarks (5min/30min), outperforming all prior SOTA planners inc…
tags:
  - "AAAI 2026"
  - "MCTS"
  - "Classical Planning"
  - "O(1) Node Selection"
  - "Tree Collapsing"
  - "UCB1-Normal2"
  - "IPC Benchmark"
  - "Agile Planning"
date: 2026-05-08
content_hash: c15dc6c6b95016e2
---

# Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning

**Conference**: AAAI 2026
**arXiv**: [2508.08385](https://arxiv.org/abs/2508.08385)  
**Code**: [https://github.com/guicho271828/downward-mcts](https://github.com/guicho271828/downward-mcts)  
**Area**: Classical Planning / Search Algorithms
**Keywords**: MCTS, Classical Planning, O(1) Node Selection, Tree Collapsing, UCB1-Normal2, IPC Benchmark, Agile Planning

## TL;DR

This paper proposes Bilevel MCTS, which launches a depth-proportional budgeted best-first search at the leaf node selected by MCTS, reducing the amortized node-selection complexity from $O(\log N)$ to $O(1)$. Complemented by Tree Collapsing to reduce the number of action-selection steps, these components are integrated into the Nεbula planner, which solves 192.2/230.6 problems on IPC2018/2023 benchmarks (5min/30min), outperforming all prior SOTA planners including LAMA, DecStar, NOLAN, and SM-Type-LAMA.

## Background & Motivation

**Background**: MCTS has achieved great success in game-playing search (Go, Backgammon), but remains unpopular in the classical planning community. The dominant approaches are queue-based forward best-first search methods (GBFS/A*), and recent methods addressing exploration–exploitation trade-offs (e.g., Softmin-type) do not employ MAB statistical tools.

**Limitations of Prior Work**: MCTS uses a tree-structured OPEN list where node selection requires recursive descent from the root to a leaf, with complexity $O(BD)$ ($B$ = branching factor, $D$ = depth), equivalent to $O(\log N)$ in balanced trees. In contrast, popmin on a standard array-based priority queue (Dial's algorithm) costs only $O(1)$. Scatter plots show that GUCTN2's node processing speed can be up to **three orders of magnitude** slower than GBFS.

**Key Challenge**: In game search, depth is bounded (Go: $d \leq 361$), making selection overhead negligible. However, in classical planning, search depth can grow arbitrarily (the $k$-disk Tower of Hanoi has solution length $2^k - 1$), making selection cost the primary bottleneck. Log-log scatter plots confirm a clear negative correlation between nodes-per-second and average search depth.

**Goal**: Eliminate the node-selection efficiency bottleneck caused by growing depth in MCTS, enabling it to match the raw efficiency of queue-based search while retaining the exploration–exploitation advantages of UCB1-Normal2, and ultimately surpass existing SOTA planners.

**Key Insight**: Rather than improving heuristic quality, the paper starts from **complexity bottleneck analysis**—launching a budgeted BFS sub-search at the leaf node selected by MCTS, with budget proportional to current depth $D$, thereby amortizing the $O(D)$ tree traversal cost.

**Core Idea**: Three techniques working in concert: (a) Bilevel MCTS achieves amortized $O(1)$ selection via BFS with $b=D$ (Thm. 3); (b) Tree Collapsing bypasses nodes whose child count falls below threshold $\theta$, reducing uninformative action-selection steps; (c) Dynamic Tree Collapsing sets $\theta$ to current depth, eliminating hyperparameter tuning. These are orthogonally combined with a novelty metric and alternating queues to form Nεbula.

## Method

### Overall Architecture

A standard MCTS iteration consists of: (1) Selection—recursively select actions from the root to a leaf; (2) Expansion—generate successors of the leaf; (3) Evaluation—compute $h(s_n)$; (4) Backpropagation—update ancestor statistics. Bilevel MCTS modifies steps (1)+(2): upon reaching a leaf at depth $D$, a priority queue $Q$ (sorted by NEC $f$) is initialized at that leaf and BFS is run for $b=D$ expansions, after which all expanded nodes are backpropagated together. If $Q$ uses an array-based priority queue (Dial 1969), popmin costs $O(1)$, and $D$ selection steps + $D$ BFS steps yield constant amortized selection cost per node. Nεbula ultimately combines Bilevel GUCTN2 + DTC with the $w_{\#l,h^{FF}}$ novelty metric, LAMA-style 4-queue alternation (boost=10), and eager evaluation.

### Key Design 1: Depth-Proportional Dynamic Budget

- **Function**: The BFS budget $b$ equals the number of action-selection steps taken to reach the leaf (i.e., tree depth $D$), rather than a fixed hyperparameter.
- **Mechanism**: Deeper nodes incur greater tree traversal overhead; more BFS expansions amortize this cost. Shallow nodes have small budgets, preserving MAB policy fidelity. Theoretically, $D$ selection steps + $D$ BFS steps process $D+1$ nodes in $2D$ steps, yielding amortized $O(1)$.
- **Design Motivation**: Thm. 2 proves amortized $O(\log\log N)$ with a heap; Thm. 3 proves amortized $O(1)$ with an array queue. Compared to fixed budgets ($b \in \{10, 100, 1000, 10000\}$), fixed strategies either waste too much time on BFS at shallow depths or fail to amortize sufficiently at large depths. The dynamic budget outperforms all fixed alternatives in total problems solved (347 vs. best fixed 327) and IPC score.

### Key Design 2: Tree Collapsing

- **Function**: After expanding node $p$, if the grandparent $p'$ satisfies $|S(p')| + |S(p)| - 1 < \theta$, all children of $p$ are re-attached directly to $p'$ and $p$ is discarded. A separate plan-path tracking table is maintained independently.
- **Mechanism**: Internal nodes with few children provide little information during MAB action selection—they cannot effectively narrow the candidate leaf set. Skipping these nodes reduces effective tree depth and thus the number of NEC computations.
- **Design Motivation**: Synergizes with Bilevel MCTS. During BFS, backpropagation is deferred; collapsing causes multiple expanded nodes to share the same grandparent $p'$, enabling deduplication in the backpropagation queue $B$ (an ordered set sorted by $g$), significantly reducing backpropagation overhead. Experiments confirm Tree Collapsing alone has limited effect on base GUCTN2, but combined with Bilevel it improves from 328 to 347.2 (+5.8%).

### Key Design 3: Dynamic Tree Collapsing (DTC)

- **Function**: The collapsing threshold $\theta$ is not fixed but dynamically set to the current depth of the node being collapsed.
- **Mechanism**: Inspired by the success of dynamic budgets—deeper nodes tolerate wider collapsing, while shallower nodes retain fine-grained MAB selection.
- **Design Motivation**: Eliminates $\theta$ tuning. Experiments show DTC (347 problems solved, 182.4 IPC score) closely matches the best fixed $\theta=40$ (347.2 problems, 184.1 IPC score), removing the burden of hyperparameter selection.

### Key Design 4: Nεbula Configuration

- **Function**: Combines Bilevel GUCTN2 + DTC with three orthogonal exploration techniques: (1) novelty metric $w_{\#l,h^{FF}}$ ($w_{max}=2$, computing the minimum novel state–proposition combination size within $h$-value subsets); (2) LAMA-style 4-queue alternation $[\text{n2}([w,h^{FF}]), \text{pr}(h^{FF}), \text{n2}([w,\#l]), \text{pr}(\#l)]$; (3) boosted preferred operators (boost=10).
- **Mechanism**: MAB handles exploration/exploitation for a single heuristic; novelty encourages visits to novel states (acting as a soft closed list); alternating queues diversify heuristics—the three components address different dimensions of exploration and are orthogonally complementary.
- **Design Motivation**: Eager evaluation is used (not lazy), as MCTS requires accurate $h$ values to maintain MAB statistics. The tree-structured OPEN replaces the standard queue in LAMAe directly (rather than adding a new queue as in SM-Type-LAMA), preserving the simplicity of 4-queue alternation. Ablation studies confirm that removing any component degrades performance.

## Key Experimental Results

### Table 1: Bilevel + Tree Collapsing Ablation (IPC18+23 Total, 5min, summed over $h^{CEA}+h^{CG}+h^{FF}$)

| Configuration | GBFS | Softmin | GUCTN2 base | Bilevel $\theta$=0 | Bilevel $\theta$=40 | Bilevel DTC | Fixed $b$=100 |
|---|---|---|---|---|---|---|---|
| Problems Solved | 252 | 316.8 | 247 | 328 | 347.2 | 347 | 326.8 |
| IPC Score | 144.2 | 169.9 | 141.0 | 173.8 | 184.1 | 182.4 | 175.7 |

- Bilevel improves GUCTN2 from 247 to 328 (+33%); adding Tree Collapsing reaches 347.2 (+40.5%)
- DTC achieves 347 without tuning, closely matching the best fixed $\theta=40$ (347.2)
- Best fixed budget $b=100$ achieves only 326.8, far below the dynamic strategy

### Table 2: Final Comparison with SOTA Planners (IPC18+23, 5min and 30min)

| Method | ApxNovelty | LAMA | SM-Type-LAMA | DecStar | NOLAN | LAMAe+BFWS | Nεbula |
|---|---|---|---|---|---|---|---|
| 5min Solved | 166 | 164 | 186.4 | 163 | 177 | 177 | **192.2** |
| 5min IPC Score | 90.3 | 90.2 | 99.3 | 89.2 | 95.4 | 92.8 | **99.4** |
| 30min Solved | 201 | 208 | 213.4 | 174 | 204 | 226 | **230.6** |
| 30min IPC Score | 113.9 | 113.7 | 123.4 | 108.6 | 117.6 | 118.7 | **126.8** |

- At 5min, Nεbula leads SM-Type-LAMA (186.4) by +3.1% in problems solved, with comparable IPC score
- At 30min, the advantage grows substantially: 230.6 vs. SM-Type-LAMA 213.4 (+8.1%), IPC score 126.8 vs. 123.4
- At 30min, LAMAe+BFWS (226) closes the gap due to Nεbula's tree management exhausting available memory

### Table 3: Nεbula Ablation (5min, IPC18+23)

| Configuration | Full Nεbula | w/o MAB (LAMAe+BFWS) | w/o Novelty (N2DTC+LAMAe) | w/o Alternation (N2DTC+BFWS) |
|---|---|---|---|---|
| Problems Solved | 192.2 | 177 | 156.2 | 167.4 |
| IPC Score | 99.4 | 92.8 | 79.1 | 86.8 |

Removing any single component causes clear degradation, confirming the orthogonal complementarity of the three exploration mechanisms.

## Key Findings

1. **Node selection is the core bottleneck for MCTS in classical planning**: Scatter plots show GUCTN2 nodes-per-second can be over 1000× slower than GBFS, with a linear negative correlation between speed and log search depth, consistent with the $O(\log N)$ prediction of Thm. 1.
2. **Dynamic budget substantially outperforms fixed budget**: The depth-proportional strategy adaptively performs fewer BFS steps at shallow depths and more at greater depths, outperforming any fixed value overall—the best fixed $b=100$ (IPC score 175.7) still trails the dynamic approach (182.4).
3. **Synergy between Bilevel and Tree Collapsing**: Tree Collapsing alone has almost no effect on base GUCTN2 (improvement only for $h^{FF}$+IPC23), but combined with Bilevel it yields additional speedup due to deduplication of entries in backpropagation queue $B$.
4. **Three exploration mechanisms are orthogonally complementary**: Ablation shows removing MAB/novelty/alternating queues reduces problems solved from 192.2 to 177/156.2/167.4 respectively, with novelty contributing the most.
5. **Longer runtime reveals Bilevel MCTS's true potential**: At 5min, Nεbula leads only marginally; at 30min, the advantage exceeds 8%, though tree management memory overhead becomes a limiting factor toward the end.
6. **UCB1-Normal2 is the correct MAB choice for planning**: Since heuristic rewards are unbounded (unlike binary game outcomes), UCB1—which assumes finite support $[0,c]$—is inapplicable; UCB1-Normal2 assuming a Gaussian distribution (unbounded support $\mathbb{R}$) is required.

## Highlights & Insights

- **Closed-loop theory–experiment argumentation**: Thm. 1 identifies the bottleneck → scatter plots validate it → Thms. 2–3 guide the solution → experiments confirm throughput improvement. Every theoretical prediction has a corresponding empirical counterpart.
- **Surprising finding that "sparse MAB suffices"**: At large depths, most nodes are expanded by BFS rather than driven by MAB, indicating that UCB1-Normal2 needs only a small number of samples to provide sufficient exploration–exploitation signal.
- **Elegant unification via DTC**: The dynamic collapsing threshold $\theta=\text{depth}$ shares the "depth-proportional" intuition with dynamic budget $b=D$, requires no tuning, and achieves near-optimal performance—a highly attractive engineering design.
- **Modular composition paradigm**: Nεbula demonstrates a methodology of orthogonally stacking MAB + novelty + alternating queues, with each technique addressing a different dimension of exploration, providing a reusable framework for future planner design.
- **Incremental writing style**: The paper alternates between introducing improvements and presenting small experiments, providing immediate validation at each step, allowing readers to progressively understand the necessity of each design decision.

## Limitations & Future Work

1. **Memory overhead**: Tree management consumes significantly more memory than queue-based search. At the end of 30min runs, memory exhaustion (8GB limit) narrows LAMAe+BFWS's gap to 226 vs. 230.6; optimizing tree data structures could expand the advantage.
2. **Restricted to unit-cost problems**: The $O(1)$ amortized guarantee relies on Dial's array priority queue (handling only dense integer keys); non-unit-cost problems require falling back to a heap, degrading complexity to $O(\log\log N)$.
3. **Incompatibility with lazy evaluation**: MCTS requires accurate $h$ values to maintain MAB statistics; experiments with lazy evaluation are predominantly negative (Appendix Table 4), limiting potential savings in evaluation time.
4. **No approximate novelty**: The current implementation uses exact computation with $w_{max}=2$, incurring storage overhead quadratic in the number of propositions. Integrating Bloom filter approximations from ApxNovelty or NOLAN's dynamic $w_{max}$ selection is worth exploring.
5. **Some hyperparameters still require domain knowledge**: Although DTC eliminates $\theta$ tuning, the boost value (10 vs. LAMA's 1000), eager vs. lazy evaluation, and queue count remain empirically determined.

## Related Work & Insights

- **PN2 (Kishimoto et al. 2012)**: A pioneer of bilevel search in games, but motivated by memory savings (similar to IDA*) and discards nodes expanded during sub-search—contrasting with the present work, which retains all expanded nodes.
- **Df-UCT (Yoshizoe et al. 2011)**: Also targets action-selection efficiency, but focuses on communication overhead in distributed search rather than single-machine complexity.
- **THTS (Keller & Helmert 2013; Schulte & Keller 2014)**: Theoretical foundations for applying MCTS to classical planning; this paper implements the Bilevel modification within that framework.
- **Wissow & Asai 2024**: Establishes that UCB1-Normal2 is the correct MAB choice for classical planning (due to unbounded rewards); GUCTN2 in this paper builds directly on that work.
- **LAMA / SM-Type-LAMA**: The standard framework of alternating queues + boosted preferred operators; Nεbula demonstrates that MCTS trees can seamlessly replace standard queues within this framework with additional gains.
- **BFWS (Lipovetzky & Geffner 2017)**: Application of novelty metrics in width-based search; this paper integrates novelty as the primary MCTS ranking criterion, with UCB1-Normal2 as a tiebreaker.
- **Burns et al. 2012**: Efficient implementation techniques for heuristic search, inspiring the key design of using Dial's array queue for $O(1)$ popmin.
- **Insights**: Complexity bottleneck analysis (rather than focusing solely on heuristic quality) is an effective entry point for improving search algorithms; amortized analysis, a classical algorithm design tool, is transferable to other tree search settings (e.g., PUCT in AlphaZero, VSIDS in SAT solving).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Bilevel architecture achieving amortized $O(1)$ is derived from rigorous complexity analysis, backed by three theorems and empirically validated; the DTC design is elegant and hyperparameter-free
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ IPC18+23 standard benchmarks, 3 heuristics, 7 baselines/variants, detailed ablations (Bilevel/TC/DTC/fixed budget/novelty/alternating queues), 5min+30min settings, 5 random seeds
- **Writing Quality**: ⭐⭐⭐⭐ The incremental structure of alternating improvements with immediate experimental validation is very clear, though information density is extremely high; the rich appendix (≥12 sections) warrants multiple readings
- **Value**: ⭐⭐⭐⭐⭐ Nεbula sets a new SOTA on IPC standard benchmarks; the modular orthogonal composition methodology generalizes to other search problems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](../../ICML2026/others/amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)
- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)
- [\[CVPR 2026\] Differentiable Stroke Planning with Dual Parameterization for Efficient and High-Fidelity Painting Creation](../../CVPR2026/others/differentiable_stroke_planning_with_dual_parameterization_for_efficient_and_high.md)

</div>

<!-- RELATED:END -->
