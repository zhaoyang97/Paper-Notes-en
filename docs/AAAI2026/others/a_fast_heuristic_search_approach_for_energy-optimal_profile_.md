---
title: >-
  [Paper Note] A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles
description: >-
  [AAAI 2026][Electric vehicle routing] This paper proposes Pr-A*, a label-setting method based on multi-objective A* search for efficiently solving energy-optimal profile routing for electric vehicles (EVs) when the initi…
tags:
  - "AAAI 2026"
  - "Electric vehicle routing"
  - "energy-optimal search"
  - "A* search"
  - "energy profile"
  - "multi-objective search"
date: 2026-05-08
content_hash: 676b006173c20bb2
---

# A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles

**Conference**: AAAI 2026
**arXiv**: [2512.01331](https://arxiv.org/abs/2512.01331)  
**Code**: The paper states code is publicly available (footnote 3); no link is attached on the arXiv page  
**Area**: Other
**Keywords**: Electric vehicle routing, energy-optimal search, A* search, energy profile, multi-objective search

## TL;DR
This paper proposes Pr-A*, a label-setting method based on multi-objective A* search for efficiently solving energy-optimal profile routing for electric vehicles (EVs) when the initial state of charge (SoC) is unknown. By using profile dominance pruning, the method avoids the complex profile merge operations required by traditional approaches, achieving performance close to standard A* with known initial SoC on large-scale road networks.

## Background & Motivation
In EV routing, energy can be both consumed (positive cost) and recovered via regenerative braking (negative cost), resulting in potentially negative edge weights. When the initial SoC is known, the problem can be solved efficiently via variants of Dijkstra/A* combined with Johnson reweighting to eliminate negative-weight edges. However, in many practical scenarios the initial SoC is unknown — for example, in multi-leg trip planning where each leg's starting SoC depends on the previous leg's terminal state, or in bidirectional search where the destination SoC cannot be predetermined. In such cases, optimal routes must be computed for all possible initial SoC values — this is the **energy profile routing** problem.

Existing profile routing methods (Baum et al.; Schönfelder et al.) are based on **label-correcting** strategies that merge multiple energy profiles associated with the same state. However, merge operations are costly: while a single-path profile has at most two breakpoints, the merged profile can have $O(|S|)$ breakpoints, requiring a full sweep over the SoC range to compute the lower envelope. This is complex to implement and computationally expensive, limiting practical applicability.

## Core Problem
How can energy-optimal EV profile routing be performed efficiently **without complex profile merge operations**? That is, for all possible initial SoC values, how can optimal routes be found with efficiency close to standard A*?

## Method

### Overall Architecture
Energy profile routing is reformulated as a **multi-objective shortest path problem**: in addition to minimizing energy cost $g$, the minimum required energy $\mathcal{E}_{\min}$ and the maximum energy cost $\bar{g}$ are introduced as additional objectives. Under this framework, a **label-setting** (rather than label-correcting) strategy is adopted, expanding nodes in non-decreasing order of $f$ values and applying profile dominance pruning to avoid explicit profile merging.

The algorithm (Pr-A*) maintains a priority queue Open and a set of non-dominated nodes $X(u)$ for each state $u$. Starting from the source, the node with the smallest $f$ value is extracted and its successors are expanded. For each generated node, the three breakpoint values of the energy profile $(g, \bar{g}, \mathcal{E}_{\min})$ are updated, and feasibility and dominance checks are performed. Nodes reaching the goal update the global upper bound $\bar{f}$. The search terminates when Open is empty or all remaining nodes exceed $\bar{f}$, returning $X(\text{goal})$ as a superset of the optimal profile.

### Key Designs

1. **Energy Profile Representation**: Each path's energy profile is compactly represented using two breakpoints — $(\mathcal{E}_{\min}, g)$ and $(\mathcal{E}_{\max}, \bar{g})$. Here $\mathcal{E}_{\min}$ is the minimum initial energy required to traverse the path, $g$ is the minimum energy cost, and $\bar{g}$ is the maximum cost when departing at full charge (since excess recovered energy cannot be stored). The slope between the two breakpoints is 1, and the profile is constant at $g$ below $\mathcal{E}_{\min}$, forming a piecewise-linear shape. This avoids the breakpoint explosion caused by merging.

2. **Profile Dominance Pruning**: The core contribution. Node $y$ is dominated by node $x$ if $\mathcal{E}_{\min}(x) \leq \mathcal{E}_{\min}(y)$, $g(x) \leq g(y)$, and $\bar{g}(x) \leq \bar{g}(y)$. Since A* expands nodes in non-decreasing $f$ order, $g(x) \leq g(y)$ is guaranteed by the expansion order, and only the $\mathcal{E}_{\min}$ and $\bar{g}$ dimensions need additional checking. Dominated nodes can be safely pruned without affecting optimality. This pairwise dominance is a relaxation compared to constructing a global lower envelope, but experiments show it is sufficiently effective. A lazy checking strategy is also employed: a fast comparison against the most recently expanded node at the same state is performed at expansion time, with a full dominance check deferred to extraction time.

3. **Feasibility Pruning and Upper Bound**: Three pruning conditions are applied: (i) $\mathcal{E}_{\min}(y) > \mathcal{E}_{\max}$ (path infeasible); (ii) $\bar{g}(y) > \mathcal{E}_{\max}$ (infeasible even at full charge); (iii) $\bar{g}(y) + h(s(y)) > \mathcal{E}_{\max}$ (maximum suffix cost plus remaining estimate exceeds capacity). Upon reaching the first goal node, a global upper bound $\bar{f} = \max\{\mathcal{E}_{\min}, \bar{g}\}$ is set using the node's values; subsequent nodes with $f > \bar{f}$ are pruned.

4. **Four Variants**: (i) Pr-A*$^{\text{fw}}$ — forward search (primary algorithm); (ii) Pr-A*$^{\text{bw}}$ — backward search from goal to start, with a slightly different profile concatenation formula; (iii) Pr-BA* — interleaved bidirectional search, where forward and backward nodes meeting at the same state form a complete path; (iv) Pr-BA*$^{\text{par}}$ — parallel bidirectional search, with forward and backward searches running on separate CPU threads.

5. **Consistent Heuristic Functions**: Two families of heuristic functions are designed based on physical energy models. One is based on potential energy differences (elevation change) plus distance estimates; the other is based on a simplified longitudinal dynamics model. Both are proven to satisfy consistency and admissibility, maintaining monotone $f$ values even on negative-weight graphs with battery capacity constraints, without requiring Johnson reweighting.

### Loss & Training
Not applicable — this is a search algorithm paper with no training process. The core contribution is the theoretically proven optimality guarantee (Theorem 1: Pr-A* returns a superset containing all optimal profiles).

## Key Experimental Results
Experiments use 10 DIMACS large-scale road networks (the largest, Central USA, contains 14 million nodes and 34 million edges), with 200 random queries per network (2,000 total). A Nissan Leaf with an 85 kWh battery is simulated.

| Energy Range (kWh) | Metric (avg. ms) | Dijkstra | A* (known SoC) | Pr-A*$^{\text{fw}}$ | Pr-A*$^{\text{bw}}$ | Speedup (Pr-A*$^{\text{fw}}$ vs. Dijkstra) |
|---|---|---|---|---|---|---|
| 0–20 | Avg. runtime | 27.6 | 15.6 | 20.6 | 19.6 | 1.51× |
| 20–40 | Avg. runtime | 79.5 | 44.0 | 55.3 | 53.1 | 1.63× |
| 40–60 | Avg. runtime | 160.7 | 79.8 | 101.4 | 101.8 | 1.77× |
| 60–85 | Avg. runtime | 242.9 | 121.7 | 151.2 | 152.8 | 1.69× |

- Pr-A*$^{\text{fw}}$ is only **within ~30%** slower than A* with known SoC, while solving the more general profile problem.
- The bidirectional variant Pr-BA* is slower — even slower than Dijkstra — due to the overhead of path matching.
- The number of nodes expanded by Pr-A*$^{\text{fw}}$ is nearly identical to standard A* (less than 10% more), demonstrating highly effective pruning.

### Ablation Study
- **Forward vs. backward**: The forward variant is more stable; the backward variant exhibits high variance (node expansion counts fluctuate by over 100%).
- **Unidirectional vs. bidirectional**: Bidirectional search does not outperform unidirectional search on this problem (roughly 50% more node expansions, with high matching overhead).
- **Parallel vs. serial bidirectional**: Pr-BA*$^{\text{par}}$ is approximately 35% faster than serial bidirectional, but still slower than unidirectional search.
- **vs. Schönfelder et al. (2014)**: Their profile A* is an order of magnitude slower than standard A*; the proposed method is only 30% slower — a substantial improvement.
- **vs. Baum et al. (2020)** (indirect): Their profile Dijkstra is 50% slower than standard Dijkstra; the proposed profile A* is 70% faster than Dijkstra, suggesting an estimated speedup of more than 2×.
- The vast majority of queries return only 1 optimal path; at most 4 are returned.

## Highlights & Insights
1. **Elegant simplicity**: Reformulating profile routing as multi-objective search with dominance pruning completely avoids the complex merge operations of label-correcting methods; the algorithm requires only a few additional lines of code beyond standard A*.
2. **Theoretical completeness**: Full correctness proofs are provided — safety of dominance relations, convergence, and the optimality-superset guarantee — with $f$-value monotonicity established on negative-weight graphs.
3. **Strong practicality**: On road networks with 14 million nodes, profile routing latency is on the order of hundreds of milliseconds, approaching the performance of A* with known SoC — effectively obtaining planning results for all SoC values "for free."
4. **Ease of integration**: The C++ implementation uses standard data structures (binary heap) and can be readily integrated into existing EV routing systems.

## Limitations & Future Work
1. **Suboptimal bidirectional search**: The bidirectional variants are slower due to path-matching overhead; more efficient bidirectional profile search strategies are needed.
2. **No preprocessing acceleration**: Contraction Hierarchies and similar preprocessing techniques are not employed; further speedups on very large networks remain feasible.
3. **Simplified energy model**: Only slope- and mass-driven energy consumption is modeled; dynamic factors such as traffic conditions, temperature effects on the battery, and air conditioning loads are not considered.
4. **Non-tight profile superset**: The algorithm returns a superset rather than the exact lower envelope of optimal profiles, requiring post-processing to extract the result. Although experiments show the superset is typically small ($\leq 4$), it could theoretically be larger.
5. **Single vehicle type**: Only the Nissan Leaf is simulated; generalization to other vehicle types and battery capacities is not validated.

## Related Work & Insights
- **Artmeier et al. (2010) / Sachenbacher et al. (2011)**: Address energy-optimal routing with known SoC. This paper extends the setting to unknown SoC while maintaining comparable efficiency.
- **Baum et al. (2013, 2019)**: Label-correcting + Dijkstra + CH preprocessing. The proposed method achieves better performance without preprocessing and with a simpler implementation.
- **Schönfelder et al. (2014)**: The closest baseline; their profile A* is 10× slower than standard A*. The proposed method is only 30% slower — a dramatic improvement.
- **Multi-objective search (LTMOA*, BOA*)**: The proposed method draws on dominance pruning and lazy checking from multi-objective A*, innovatively applied to energy profile routing.

**Transferability**: The paradigm of reformulating "parameterized optimal search" as "multi-objective search with dominance pruning" is general and may apply to other routing problems with uncertain parameters (e.g., uncertain payload, uncertain speed profiles). **Integration with charging station planning**: Profile routing is naturally suited for joint optimization of multi-leg trips and charging station selection, since each leg's starting SoC is directly obtained from the profile. **Real-time applicability**: Sub-second latency makes the method suitable for online navigation; when SoC sensors are imprecise, profile routing is more robust than point queries.

## Rating
- Novelty: ⭐⭐⭐⭐ — Reformulating the label-correcting problem as label-setting with multi-objective search is a clean and effective idea, though the core techniques (dominance pruning, heuristic functions) build on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10 large-scale real-world road networks, 2,000 queries, and 4 variants are compared with detailed statistics; however, direct comparison with Baum et al. and experiments on different vehicle types/battery capacities are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, theoretical proofs are complete, and the appendix provides pseudocode for backward/bidirectional variants and profile concatenation examples; reproducibility is strong.
- Value: ⭐⭐⭐⭐ — Practically valuable for EV routing; the algorithm is concise and efficient, though the research direction is relatively niche (the EV profile routing community is small).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[AAAI 2026\] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets](intermediate_n-gramming_deterministic_and_fast_n-grams_for_large_n_and_large_dat.md)

</div>

<!-- RELATED:END -->
