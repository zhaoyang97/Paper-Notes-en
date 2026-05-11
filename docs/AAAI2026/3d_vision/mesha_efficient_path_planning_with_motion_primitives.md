---
title: >-
  [Paper Note] MeshA*: Efficient Path Planning With Motion Primitives
description: >-
  [AAAI 2026][3D Vision][motion primitives] This paper proposes MeshA*, an algorithm that reformulates lattice-based path planning from "searching at the motion primitive level" to "searching at the grid cell level while s…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "motion primitives"
  - "lattice-based planning"
  - "A*"
  - "extended cell"
  - "branching factor reduction"
  - "mobile robots"
date: 2026-05-08
content_hash: 18c0d3879a41c537
---

# MeshA*: Efficient Path Planning With Motion Primitives

**Conference**: AAAI 2026
**arXiv**: [2412.10320](https://arxiv.org/abs/2412.10320)
**Code**: [https://github.com/PathPlanning/MeshAStar](https://github.com/PathPlanning/MeshAStar)
**Area**: Path Planning / Heuristic Search
**Keywords**: motion primitives, lattice-based planning, A*, extended cell, branching factor reduction, mobile robots

## TL;DR

This paper proposes MeshA*, an algorithm that reformulates lattice-based path planning from "searching at the motion primitive level" to "searching at the grid cell level while simultaneously fitting primitive sequences." By defining a novel search space based on *extended cells*, MeshA* achieves 1.5×–2× runtime speedup over standard LBA* while preserving completeness and optimality guarantees.

## Background & Motivation

**Background**: Lattice-based planning with motion primitives is one of the dominant approaches to mobile robot motion planning. It composes paths from precomputed kinematically feasible trajectory segments and uses A* to search for optimal paths in a discretized $(x, y, \theta)$ state space.

**Limitations of Prior Work**: When the number of motion primitives is large—as is common in practice—the branching factor of A* on the lattice graph becomes substantial. Each expansion requires collision checking and cost evaluation for all applicable primitives, incurring significant computational overhead. Even with lazy collision checking, extracting and rolling back large numbers of invalid states in cluttered environments wastes considerable time.

**Key Challenge**: Lattice-based search operates at the granularity of individual primitives, where each primitive spans multiple grid cells. Even if a primitive becomes infeasible at its very first cell due to an obstacle, the planner cannot reject it until the full primitive is processed—making the search granularity too coarse for early pruning.

**Goal**: To reduce the computational cost of lattice-based planning without sacrificing completeness or optimality, enabling faster path planning.

**Key Insight**: The search granularity is refined from the primitive level to the grid cell level. During search, the planner advances cell by cell while tracking which primitives still pass through the current cell, enabling (a) fine-grained pruning—unpromising directions can be identified before a primitive completes—and (b) merging of primitives that share initial cells, reducing the effective branching factor.

**Core Idea**: Define an *extended cell* $u = (i, j, \Psi)$, where $\Psi$ records all primitives passing through cell $(i, j)$ along with their current step indices in the collision trajectory. Successor relations are defined at the cell level—primitives sharing the same next cell are merged into a single successor node, with zero cost for non-terminal steps and primitive cost for terminal steps.

## Method

### Overall Architecture

MeshA* is not a new search algorithm; rather, it applies standard A* to a novel search space called the *mesh graph*. Nodes in this graph are extended cells $(i, j, \Psi)$, and edges are defined by successor relations that advance primitives through their collision trajectories step by step. After finding the shortest path between initial extended cells on the mesh graph, a trajectory reconstruction algorithm (Algorithm 3) recovers the full primitive sequence. Theorem 4 formally establishes that optimal paths on the mesh graph correspond exactly to optimal trajectories on the lattice graph.

### Key Design 1: Extended Cell

- **Function**: The search node is defined as $(i, j, \Psi)$, where $\Psi = \{(\text{prim}_1, k), (\text{prim}_2, k), \ldots\}$ is a set of (primitive, step-index) pairs representing all primitives passing through cell $(i, j)$ at their current step.
- **Mechanism**: Multiple primitives may share identical collision trajectories over their initial cells (e.g., primitives terminating at different headings may agree on their first few steps). Bundling them into a single search node avoids redundant exploration.
- **Design Motivation**: To reduce the effective branching factor—the number of successors depends on the number of distinct displacement directions at the next step, not the total number of primitives, since many primitives commonly share initial steps.

### Key Design 2: Two Types of Successor Relations

- **Function**: Two successor types are defined: (a) *Regular Successor*—the primitive has not yet completed; the step index increments by one with cost 0; (b) *Initial Successor*—the primitive has completed; a new initial configuration $\Psi_\theta$ is opened with cost $c_{\text{prim}}$.
- **Mechanism**: Primitive costs are deferred until completion; intermediate steps are cost-free. The total path cost thus equals the sum of costs of sequentially completed primitives.
- **Design Motivation**: To establish a one-to-one correspondence between mesh graph path costs and lattice graph trajectory costs (Lemma 1 + Theorems 2/3), ensuring that A*'s optimality guarantee carries over directly.

### Key Design 3: Fine-Grained Heuristic and Early Pruning

- **Function**: For non-initial extended cells, the heuristic is defined as $h(u) = \min_{(u_0, c_0) \in \text{Finals}(u)} \{h(u_0) + c_0\}$, i.e., the minimum over all possible primitive endpoints of the sum of endpoint heuristic value and completion cost.
- **Mechanism**: The promise of a partially executed primitive can be evaluated mid-execution—if the heuristic value at a primitive's projected endpoint is high, the search naturally deprioritizes that direction.
- **Design Motivation**: LBA* can only evaluate the heuristic at the start and end of each primitive and cannot prune mid-execution. MeshA* evaluates the heuristic at every cell, enabling earlier elimination of unpromising directions.

### Key Design 4: Terminal Pruning Rule

- **Function**: If all primitive endpoints in $\text{Finals}(u)$ for a non-initial extended cell $u$ have already been expanded, $u$ can be safely skipped.
- **Mechanism**: Any path through $u$ must pass through one of its terminal cells; if optimal paths to all terminals are already known, further exploration from $u$ is unnecessary.
- **Design Motivation**: This pruning exploits the structural properties of the mesh graph and is impossible in LBA*, which treats complete primitives as atomic units with no intermediate skip points.

## Key Experimental Results

### Table 1: Runtime Comparison (MovingAI Benchmark Maps, 25,000+ Instances)

| Algorithm | w=1 | w=2 | w=5 | w=10 |
|-----------|-----|-----|-----|------|
| LBA* | 100% | 100% | 100% | 100% |
| LazyLBA* | >100% | >100% | >100% | >100% |
| **MeshA*** | **60–80%** | **~55%** | **~50%** | **~50%** |

MeshA* achieves 1.5× speedup under optimal search ($w=1$) and 2× speedup under weighted search ($w=5$–$10$). LazyLBA* is actually slower than LBA* due to the simplicity of the collision check: with cheap per-cell queries, the overhead from additional queue operations in the lazy scheme outweighs any savings.

### Table 2: Solution Quality Comparison (ht_0_hightown Map, Median Cost / Optimal Cost %)

| Algorithm | w=1 | w=2 | w=5 | w=10 |
|-----------|-----|-----|-----|------|
| LBA* | 100.0% | 105.7% | 110.0% | 113.2% |
| MeshA* | 100.0% | 109.2% | 117.6% | 122.3% |

Both algorithms are optimal at $w=1$. Under weighting, MeshA* yields slightly lower solution quality (due to more frequent heuristic evaluations amplifying the weight effect), trading solution quality for greater speedup.

## Key Findings

1. **Fine-grained search granularity is the key to speedup**: MeshA* expands roughly 50% fewer grid cells than LazyLBA*, demonstrating that cell-by-cell advancement effectively terminates unpromising primitives early.
2. **Lazy collision checking is not universally beneficial**: When collision checking itself is cheap, the additional queue-management overhead of the lazy strategy is counterproductive—illustrating that acceleration techniques must be matched to problem characteristics.
3. **Weighted heuristics amplify MeshA*'s advantage**: The speedup grows from 1.5× at $w=1$ to 2× at $w=10$, as more frequent heuristic evaluations make weighted pruning increasingly aggressive.
4. **Theoretical equivalence holds rigorously**: Theorems 2–4 formally prove that mesh graph search and lattice graph search are fully equivalent in terms of optimal solution cost and collision trajectories.

## Highlights & Insights

- **Methodology of search space redefinition**: Significant speedup is achieved solely by redefining the search space—without modifying the search algorithm itself (A* is used throughout). This suggests that *how the problem is modeled* may matter more than *which algorithm is applied*.
- **Primitive bundling reduces the branching factor**: Primitives sharing initial trajectory segments are automatically merged into the same extended cell; the effective branching factor depends on the number of distinct displacement directions rather than the total primitive count.
- **Elegant deferred cost assignment**: Assigning zero cost to intermediate steps and full primitive cost to terminal steps yields a clean and intuitive cost equivalence proof.
- **Efficient indexing of configurations**: Precomputing integer indices for all possible primitive configurations $\Psi$ avoids runtime set-operation overhead.

## Limitations & Future Work

1. **Evaluation limited to 2D scenarios**: Experiments are restricted to the $(x, y, \theta)$ state space; extension to 3D or higher-dimensional states (e.g., incorporating acceleration or curvature) has not been explored.
2. **Greater solution quality degradation under weighting**: The frequent heuristic evaluations in MeshA* amplify the effect of the weight parameter; more sophisticated heuristic design is needed to better balance speed and solution quality.
3. **Simplified collision checking setting**: The current experiments use grid-cell queries as the collision check. If collision checking were expensive (e.g., 3D rigid body collision detection), lazy strategies might outperform MeshA*.
4. **Potential configuration space explosion**: As the number of primitives or collision trajectory length increases, the number of possible configurations $\Psi$ may grow too rapidly, undermining the efficiency of the indexing scheme.
5. **No integration with incremental or replanning methods**: Performance in dynamic environments, e.g., in combination with D* Lite or Lifelong Planning A*, remains unexplored.

## Related Work & Insights

- **Lattice-based Planning (Pivtoraiko et al. 2005/2009)**: The classical motion primitive planning framework and the direct predecessor of MeshA*.
- **Jump Point Search (JPS)**: Accelerates A* on uniform grids by skipping intermediate nodes—conceptually similar in spirit (reducing the number of expanded nodes), but applicable only to uniform grids rather than motion primitive graphs.
- **WA* (Weighted A*)**: Orthogonal to MeshA*; the weighting strategy can be layered on top of MeshA* for additional speedup.
- **RRT*/Informed RRT***: Sampling-based planning methods that scale to high dimensions but lack deterministic guarantees.
- **Insight**: "Re-encoding" the search space (from the primitive level to the cell level) is a general acceleration strategy. Analogous search space restructuring in other combinatorial optimization problems may yield similar benefits.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The search space redefinition perspective is original, though the underlying algorithm remains standard A* with no algorithmic innovation per se.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ MovingAI standard benchmarks with multiple maps and weight settings are used, though evaluation is limited to 2D scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The definition–lemma–theorem–proof chain is rigorous and complete; the formalization of successor relations and cost design is exemplary.
- **Value**: ⭐⭐⭐⭐ Directly applicable to lattice-based path planning in practice; a 1.5–2× speedup is deployment-ready.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](../../CVPR2026/3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[AAAI 2026\] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios](epsegfz_efficient_point_cloud_semantic_segmentation_for_few-_and_zero-shot_scena.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[ICLR 2026\] CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives](../../ICLR2026/3d_vision/crisp_contact-guided_real2sim_from_monocular_video_with_planar_scene_primitives.md)

</div>

<!-- RELATED:END -->
