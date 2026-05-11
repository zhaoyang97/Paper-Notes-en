---
title: >-
  [Paper Note] Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity
description: >-
  [AAAI 2026][Robotics][grid-based storage] For the problem of uncertain retrieval order in fully loaded 2D grid-based storage systems, this paper proposes the k-bounded perturbation uncertainty model…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "grid-based storage"
  - "out-of-order retrieval"
  - "maximum capacity"
  - "k-bounded perturbation"
  - "robust storage"
  - "relocation minimization"
date: 2026-05-08
content_hash: 479d3de85a53fdc1
---

# Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity

**Conference**: AAAI 2026
**arXiv**: [2601.19144](https://arxiv.org/abs/2601.19144)
**Code**: None
**Area**: Robotics
**Keywords**: grid-based storage, out-of-order retrieval, maximum capacity, k-bounded perturbation, robust storage, relocation minimization

## TL;DR

For the problem of uncertain retrieval order in fully loaded 2D grid-based storage systems, this paper proposes the k-bounded perturbation uncertainty model, proves that $\Theta(k)$ columns is both necessary and sufficient for zero relocation, and presents an efficient robust storage solver and greedy retrieval strategy. The approach nearly eliminates relocations when $k \leq 0.5c$ and still reduces relocations by 50%+ when $k$ reaches $c$.

## Background & Motivation

1. **Tension between full-capacity storage and efficiency**: In high-density grid-based storage systems (e.g., Puzzle-Based Storage, PBS), higher storage density leads to worse accessibility of individual items; the problem is especially severe at full capacity.
2. **Unpredictable retrieval order**: Prior work assumes storage and retrieval orders are fully known in advance, but in real logistics scenarios (last-mile distribution centers, container terminals), the retrieval order is determined after the storage phase and may change dynamically.
3. **High cost of relocation**: When a target item is blocked by others, relocation operations are required (moving blocking items aside and returning them), each of which incurs additional time, energy, and I/O row usage.
4. **Fragility of existing zero-relocation solutions**: With $c \geq 3$ columns, zero relocation can be guaranteed for known retrieval orders (BaseS), but this fails under even slight perturbations to the retrieval order (as illustrated in Figure 1, minor perturbations cause multiple relocations).
5. **Theoretical gap**: No theoretical analysis (upper and lower bounds on required column count) has been established for grid-based storage that simultaneously addresses order uncertainty and full-capacity conditions.
6. **Lack of practical design guidance**: Warehouse designers need to know "how much retrieval order deviation can a given grid width tolerate," in order to trade off space utilization against operational robustness.

## Method

### Problem Formulation (R-StoRMR)

- The storage area $W$ is an $r \times c$ grid with an open I/O row at the bottom; items enter and exit through the I/O row.
- $n = r \cdot c$ labeled items arrive according to a known sequence $A$ and are retrieved according to $\tilde{D}$, a k-bounded perturbation of $D = (1, 2, \ldots, n)$.
- **k-bounded perturbation**: A sequence $\tilde{S}$ is a k-bounded perturbation of $S$ if and only if all inversions $(s_i, s_j)$ satisfy $j - i \leq k$ (i.e., the retrieval order of any two items deviates by at most $k$ positions).
- Objective: minimize the number of relocations, I/O row occupancy, and total travel distance.

### Theoretical Analysis: Required Columns for Zero Relocation

- **Robust adjacency condition (Proposition 1)**: An arrangement $\mathcal{A}$ is k-robust with respect to $D$ if and only if every non-bottom-row item $d_i$ is adjacent to some item $d_j$ that is retrieved at least $k+1$ positions earlier.
- **Upper bound (Theorem 2)**: $3k+3$ columns suffice. Items are divided into $k+1$ groups by label modulo $(k+1)$, and each group is assigned 3 columns for independent StoRMR solving.
- **Lower bound (Theorem 3)**: $2k+3$ columns are necessary. A constructive proof shows that the first $k+1$ items in retrieval order and the first $k+2$ items in arrival order must all occupy the bottom row.
- **Design guidance**: Zero relocation is guaranteed when $k/c \leq 1/3$; relocations cannot be avoided when $k/c \geq 1/2$.

### Robust Storage Algorithm (RobustS)

The core idea is to fill adjacent column pairs $(L, R)$ bottom-up, jointly iterating over $D$ and $A^R$, and greedily searching for item pairs that simultaneously satisfy adjacency conditions in both directions.

- **Column pair initialization**: The bottom cell of $L$ receives the next unassigned item from $A^R$; the bottom cell of $R$ receives the smallest unassigned item (ensuring the first $k+1$ items are in the bottom row).
- **Main loop**: The next D-valid item $x$ from $D$ is placed in column $R$; a matching item $y$ is found in $A^R$ for column $L$, subject to $x$ and $y$ simultaneously satisfying all adjacency conditions.
- **Last-column handling**: The 3rd (and 4th) columns are reserved for last filling, leveraging horizontal adjacency from neighboring columns to increase success rate.
- **Load-skipping enhancement**: Different starting offsets of $A^R$ are explored exhaustively (parallelizable), significantly improving the probability of finding a robust arrangement.

### Retrieval Strategy (ImpR)

When $k$ exceeds the theoretical upper bound and relocations are unavoidable:

1. Compute the retrieval path $\pi$ with the fewest blockers for target item $x$.
2. Relocate each blocker $b$ along $\pi$ from outside to inside:
   - Compute the current set $U$ of unblocked items (those with direct I/O access).
   - Preferentially relocate $b$ within $W$, choosing the nearest empty cell that does not block items in $U$.
   - If no empty cell exists in $W$, temporarily move $b$ to the I/O row, retrieve $x$, and then return $b$ to its original position.
3. Constraint: the I/O row must be cleared after each retrieval.

## Key Experimental Results

Performance is evaluated on square grids at 100% density, comparing four combinations: BaseS+BaseR, RobustS+BaseR, BaseS+ImpR, and RobustS+ImpR. Each configuration is tested over 50 random trials.

### Table 1: Average Number of Relocations at Different $k/c$ Ratios ($17 \times 17$ Grid)

| $k/c$ | BaseS+BaseR | RobustS+BaseR | BaseS+ImpR | RobustS+ImpR | Reduction |
|-------|-------------|---------------|------------|--------------|-----------|
| 0.25 | ~60 | ~2 | ~40 | **~0** | ~100% |
| 0.50 | ~130 | ~15 | ~85 | **~8** | ~94% |
| 0.75 | ~190 | ~80 | ~120 | **~55** | ~71% |
| 1.00 | ~240 | ~120 | ~150 | **~90** | ~63% |

### Table 2: Algorithm Runtime (seconds, by grid size)

| Grid Size | RobustS Time | ImpR Time |
|-----------|-------------|-----------|
| $9\times9$ | <1 | <0.1 |
| $13\times13$ | ~5 | <0.5 |
| $17\times17$ | ~20 | <1 |
| $21\times21$ | ~50 | <1 |

Both RobustS and ImpR complete within 1 minute and 1 second, respectively, across all grid sizes.

## Key Findings

- When $k \leq 0.5c$, RobustS nearly eliminates all relocations, consistent with the theoretical prediction of Theorem 3.
- As $k$ grows to $c$, RobustS+ImpR still reduces relocations by 60–70% compared to the baseline.
- I/O row occupancy is substantially reduced: ImpR prioritizes in-warehouse relocation, and RobustS further reduces occupancy to zero when $k/c \leq 0.5$.
- Almost all items relocated within $W$ are not relocated again, avoiding cascading relocations.
- The load-skipping enhancement enables RobustS to maintain an 80%+ success rate even when $k$ exceeds the theoretical upper bound.

## Highlights & Insights

- **Solid theoretical contributions**: This work is the first to establish tight bounds of $\Theta(k)$ on the number of columns required for robust zero-relocation in fully loaded grid-based storage, with upper and lower bounds differing by at most a factor of 1.5.
- **Elegant problem formulation**: The k-bounded perturbation model precisely captures "locally disordered" uncertainty in practice, more realistic than fully random permutations.
- **Unified theory and practice**: The theoretical bounds not only provide existence proofs but directly inform the design of RobustS and guide engineering choices for warehouse width.
- **Computational efficiency**: The storage algorithm runs in nearly linear time in $n$; the retrieval strategy is applicable in real time.
- **Convincing experiments**: Results cover multiple grid sizes and $k/c$ ratios, with comprehensive improvement over baselines across three metrics (relocations, I/O occupancy, travel distance).

## Limitations & Future Work

- **Square grid assumption**: Experiments are conducted only on square grids; non-square scenarios ($r \ll c$ or $r \gg c$) are not validated.
- **RobustS is suboptimal**: The algorithm is a greedy heuristic and cannot guarantee finding a robust arrangement when $2k+3 \leq c < 3k+3$ (the theoretical gap is not closed).
- **Single-robot assumption**: Only one robot/manipulator performing sequential operations is considered; extension to multi-robot parallel scenarios is not addressed.
- **Static $k$ value**: $k$ must be fixed at the storage phase and cannot adapt dynamically; real perturbation distributions may be non-uniform.
- **NP-hard relocation subproblem**: ImpR is a greedy approximation without optimality guarantees; greedy choices may be suboptimal at large scale.
- **No dynamic arrivals**: The model assumes all items are stored before any retrieval begins and does not support online interleaving of storage and retrieval operations.

## Related Work & Insights

- **Puzzle-Based Storage (PBS)**: gue2007puzzle and related work study item movement in grids with a small number of empty cells, but do not address full retrieval sequences.
- **Block Relocation Problem (BRP)**: Minimizing relocations in stack-based storage given a retrieval order (NP-hard); boge2020robust studies priority uncertainty (Γ inversions); galle2018scrp reveals retrieval sequences in batches.
- **Multi-Agent Path Finding (MAPF)**: Multi-robot path planning in grids; Li et al. 2021 targets warehouse settings but assumes the presence of aisles.
- **Sequenced storage/retrieval**: Ordered entry and exit problems in train scheduling and container terminals, but mostly based on stack structures rather than 2D grids.
- **Predecessor work rss25**: The BaseS algorithm guaranteeing zero relocation with $c \geq 3$ under known retrieval order; the present paper extends it to the uncertainty setting.

## Rating

- Novelty: ⭐⭐⭐⭐ The k-bounded perturbation uncertainty model is novel; the $\Theta(k)$ tight bound is established for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three metrics × four combinations × multiple grid sizes, including ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise; the theory–algorithm–experiment logical chain is complete and clear.
- Value: ⭐⭐⭐⭐ Provides theoretical guidance for warehouse design; the algorithm is directly deployable in fully loaded storage systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)
- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[AAAI 2026\] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](../../NeurIPS2025/robotics/efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)

</div>

<!-- RELATED:END -->
