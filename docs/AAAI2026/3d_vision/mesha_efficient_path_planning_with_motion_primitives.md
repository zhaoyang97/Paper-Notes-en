---
title: >-
  [Paper Note] MeshA*: Efficient Path Planning With Motion Primitives
description: >-
  [AAAI 2026][3D Vision][Motion Primitives] This paper proposes the MeshA* algorithm, which shifts lattice-based path planning from "searching at the motion primitive level" to "searching at the grid cell level while simultaneously fitting primitive sequences." By defining a new search space called the "extended cell," MeshA* achieves a 1.5x-2x runtime speedup compared to standard LBA* while ensuring completeness and optimality.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Motion Primitives"
  - "lattice-based planning"
  - "A*"
  - "expanded grid"
  - "branching factor reduction"
  - "mobile robots"
date: 2026-05-08
content_hash: 7b957ac506753492
---

# MeshA*: Efficient Path Planning With Motion Primitives

**Conference**: AAAI 2026  
**arXiv**: [2412.10320](https://arxiv.org/abs/2412.10320)  
**Code**: [https://github.com/PathPlanning/MeshAStar](https://github.com/PathPlanning/MeshAStar)  
**Area**: Path Planning / Heuristic Search  
**Keywords**: Motion Primitives, lattice-based planning, A*, expanded grid, branching factor reduction, mobile robots

## TL;DR

This paper proposes the MeshA* algorithm, which shifts lattice-based path planning from "searching at the motion primitive level" to "searching at the grid cell level while simultaneously fitting primitive sequences." By defining a new search space called the "extended cell," MeshA* achieves a 1.5x-2x runtime speedup compared to standard LBA* while ensuring completeness and optimality.

## Background & Motivation

**Background**: Lattice-based planning based on motion primitives is one of the mainstream methods for mobile robot motion planning. It constructs paths using precomputed, kinematically feasible trajectory segments and searches for the optimal path in the discretized $(x, y, \theta)$ space using A*.

**Limitations of Prior Work**: When there is a large number of motion primitives (which is common in practice), the branching factor of the A* search on the lattice graph becomes very large. Each expansion requires checking collision and costs for all available primitives, leading to significant computational overhead. Even with lazy collision checking, extracting and backtracking a large number of invalid states in complex obstacle environments still wastes time.

**Key Challenge**: Lattice-based search operates at the granularity of "primitives," where one primitive corresponds to multiple grid cells. Even if a primitive becomes invalid within the first few cells due to obstacles, the entire primitive must be processed before making a decision. This results in a search granularity that is too coarse to facilitate early pruning.

**Goal**: To reduce the search overhead of lattice-based planning and achieve faster path planning without sacrificing the guarantees of completeness and optimality.

**Key Insight**: Refine the search granularity from the "primitive level" to the "grid cell level." By advancing cell-by-cell during the search and simultaneously tracking which primitives still traverse the current cell, the proposed method achieves: (a) fine-grained pruning—enabling determination of promising directions without waiting for the primitive to complete; and (b) merging primitives that share initial cells—thereby reducing the effective branching factor.

**Core Idea**: Define the "extended cell" $u = (i, j, \Psi)$, where $\Psi$ records all primitives passing through this cell and their position indices in the collision trajectory. Successor relationships are defined at the cell level: primitives sharing the next-step cell are merged into a single successor, where the cost of non-terminal steps is 0, and the cost of the terminal step is the cost of the primitive itself.

## Method

### Overall Architecture

MeshA* is not a new search algorithm, but rather standard A* applied to a new search space: the "mesh graph." The nodes of this graph are extended cells $(i, j, \Psi)$, and the edges are defined by successor relationships that progressively advance the collision trajectories of the primitives. After finding the shortest path between initial extended cells on the mesh graph, the complete primitive sequence is recovered through a trajectory reconstruction algorithm (Algorithm 3). It is theoretically proven that the optimal path on the mesh graph is equivalent to the optimal trajectory on the lattice graph (Theorem 4).

### Key Design 1: Extended Cell

- **Function**: Defines the search node as $(i, j, \Psi)$, where $\Psi = \{(prim_1, k), (prim_2, k), \ldots\}$ is a set of (primitive, step index) pairs representing all primitives passing through cell $(i, j)$ and their current positions.
- **Mechanism**: Multiple primitives share the same collision trajectory over the first few cells (e.g., multiple primitives with different terminal orientations may match in their first few steps). Bundling them in the same search node avoids redundant searches.
- **Design Motivation**: Reduces the effective branching factor—the number of successors depends on the number of different displacement directions in the next step, rather than the total number of primitives. In practice, multiple primitives frequently share initial steps.

### Key Design 2: Dual Successor Types

- **Function**: Defines two types of successors—(a) Regular Successor: the primitive is incomplete, the step index is incremented by 1, and the cost is 0; (b) Initial Successor: the primitive is completed, initiating a new initial configuration $\Psi_\theta$ with a cost of $c_{prim}$.
- **Mechanism**: Postpones the computation of the primitive cost until its completion, making intermediate steps cost-free. This ensures that the path cost is equivalent to the sum of the costs of the sequentially completed primitives.
- **Design Motivation**: Ensures a one-to-one correspondence between the path cost on the mesh graph and the trajectory cost on the lattice graph (Lemma 1 + Theorem 2/3), so that the optimality guarantee of A* is automatically inherited.

### Key Design 3: Fine-Grained Heuristics and Early Pruning

- **Function**: Defines the heuristic for non-initial extended cells as $h(u) = \min_{(u_0, c_0) \in \text{Finals}(u)} \{h(u_0) + c_0\}$, taking the minimum heuristic value of the endpoints of all possible completing primitives.
- **Mechanism**: Progress can be evaluated midway through primitive execution. If the terminal heuristic of a primitive upon completion is very high, the search naturally deprioritizes that direction.
- **Design Motivation**: LBA* can only evaluate heuristics at the start and end of primitives, preventing midway pruning. MeshA* evaluates cell-by-cell to achieve earlier elimination of unpromising directions.

### Key Design 4: Terminal Pruning Rules

- **Function**: If all primitive endpoints $\text{Finals}(u)$ of a non-initial extended cell $u$ have already been expanded, $u$ can be safely skipped.
- **Mechanism**: Any path starting from $u$ must pass through one of its endpoints. If all endpoints already have optimal paths, there is no need for further exploration.
- **Design Motivation**: Leverages the structural features of the mesh graph to achieve pruning that is impossible in LBA*, which treats whole primitives as the minimum unit and cannot skip steps in the middle.

## Key Experimental Results

### Table 1: Runtime Comparison (MovingAI Benchmark Maps, 25000+ Instances)

| Algorithm | w=1 | w=2 | w=5 | w=10 |
|------|-----|-----|-----|------|
| LBA* | 100% | 100% | 100% | 100% |
| LazyLBA* | >100% | >100% | >100% | >100% |
| **MeshA*** | **60-80%** | **~55%** | **~50%** | **~50%** |

MeshA* achieves a 1.5x speedup under optimal search (w=1) and a 2x speedup under weighted search (w=5-10). LazyLBA* is actually slower than LBA* (the simple collision checking makes the additional queue operations of the lazy strategy cost more than it saves).

### Table 2: Solution Quality Comparison (ht_0_hightown Map, Median Cost / Optimal Cost %)

| Algorithm | w=1 | w=2 | w=5 | w=10 |
|------|-----|-----|-----|------|
| LBA* | 100.0% | 105.7% | 110.0% | 113.2% |
| MeshA* | 100.0% | 109.2% | 117.6% | 122.3% |

Both are optimal when w=1. Under weighted search, MeshA* has slightly worse solution quality (due to more frequent heuristic evaluations magnifying the weight effect), but achieves greater speedup in return.

## Key Findings

1. **Fine-grained search granularity is the key to acceleration**: The number of grid cells processed by MeshA* is only about 50% of LazyLBA*, indicating that cell-by-cell progression effectively terminates unpromising primitives early.
2. **Lazy collision checking is not a silver bullet**: In scenarios where collision checking is inherently simple, the overhead of extra queue operations in the lazy strategy slows down the search, showing that acceleration strategies must match problem characteristics.
3. **Weighted heuristics magnify the advantage of MeshA***: The speedup increases from 1.5x at w=1 to 2x at w=10, because more frequent heuristic evaluations make the weighted pruning more aggressive.
4. **Theoretical equivalence holds strictly**: Theorems 2-4 prove that mesh graph search is strictly equivalent to lattice graph search in terms of optimal path cost and collision trajectories.

## Highlights & Insights

- **Methodology of redefining the search space**: Instead of modifying the search algorithm itself (which remains A*), significant acceleration is achieved solely by redefining the search space, illustrating that "how the problem is modeled" can be more important than "which algorithm is used."
- **Primitive bundling to reduce branching factor**: Multiple primitives sharing an initial path are automatically merged into the same extended cell. The effective branching factor depends on the number of displacement directions rather than the total number of primitives.
- **Elegant cost assignment deferral**: Setting the cost of intermediate steps to 0 and the completion step to the primitive's cost makes the proof of cost equivalence concise and intuitive.
- **Efficient implementation via configuration indexing**: Pre-indexing all possible primitive configurations $\Psi$ as single integer indices avoids the overhead of runtime set operations.

## Limitations & Future Work

1. **Only validated in 2D scenarios**: Current experiments are limited to the $(x, y, \theta)$ state space, and extensions to 3D or higher-dimensional states (e.g., acceleration, curvature) have not yet been implemented.
2. **Greater decline in solution quality under weighted search**: The frequent heuristic evaluations in MeshA* magnify the weighted effects, requiring more sophisticated heuristic designs to balance speed and quality.
3. **Experimental setup with simple collision checking**: The current collision checking only involves grid queries. If collision checking itself is expensive (e.g., 3D object collision detection), the lazy strategy might outperform MeshA*.
4. **Potential state-space explosion of configurations**: When the number of primitives and the length of collision trajectories increase, the number of possible combinations of configurations $\Psi$ may grow too rapidly, affecting the indexing efficiency.
5. **Incomplete integration with incremental/replanning methods**: Integrations with methods like D* Lite or Lifelong Planning A* have not been explored; thus, performance in dynamic environments is unknown.

## Related Work & Insights

- **Lattice-based Planning (Pivtoraiko et al. 2005/2009)**: The classic framework for motion primitive planning, which is the direct target for improvement of MeshA*.
- **Jump Point Search (JPS)**: Accelerates A* on uniform grids by utilizing jump points. The concept of reducing search nodes is similar, but JPS is only applicable to uniform grids rather than motion primitives.
- **WA* (Weighted A*)**: MeshA* is orthogonal to WA*; weights can be superimposed on MeshA* to achieve further acceleration.
- **RRT*/Informed RRT***: Competing sampling-based planning methods, which are suitable for high dimensions but lack deterministic guarantees.
- **Insight**: "Recoding" the search space (from primitive level to cell level) is a general acceleration strategy. Seeking similar search space reconstructions in other combinatorial optimization problems could be equally effective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of redefining the search space is novel, but the core is still A*, meaning the algorithm itself lacks structural innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Tested on MovingAI standard benchmarks with multiple maps and weight settings, but limited to 2D scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The chain of definitions, theorems, and proofs is rigorous and comprehensive; the formalization of successor relationships and cost designs is exemplary.
- **Value**: ⭐⭐⭐⭐ Provides direct practical value to lattice-based path planning, and the 1.5-2x speedup is directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](../../CVPR2026/3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[CVPR 2026\] MoRGS: Efficient Per-Gaussian Motion Reasoning for Streamable Dynamic 3D Scenes](../../CVPR2026/3d_vision/morgs_efficient_per-gaussian_motion_reasoning_for_streamable_dynamic_3d_scenes.md)
- [\[CVPR 2026\] GHPT: Real-Time Relightable Gaussian Splatting using Hybrid Path Tracing](../../CVPR2026/3d_vision/ghpt_real-time_relightable_gaussian_splatting_using_hybrid_path_tracing.md)
- [\[CVPR 2026\] D-Prism: Differentiable Primitives for Structured Dynamic Modeling](../../CVPR2026/3d_vision/d-prism_differentiable_primitives_for_structured_dynamic_modeling.md)
- [\[AAAI 2026\] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios](epsegfz_efficient_point_cloud_semantic_segmentation_for_few-_and_zero-shot_scena.md)

</div>

<!-- RELATED:END -->
