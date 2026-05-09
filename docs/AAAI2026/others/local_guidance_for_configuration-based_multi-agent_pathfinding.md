---
title: >-
  [Paper Note] Local Guidance for Configuration-Based Multi-Agent Pathfinding
description: >-
  [AAAI 2026][MAPF] This paper introduces the concept of Local Guidance (LG) to improve solution quality in LaCAM-based multi-agent pathfinding. By constructing local space-time paths for each agent at every configuration generation step, LG mitigates congestion and reduces solution cost by up to 50%, while maintaining completion within a few seconds for 1,000 agents.
tags:
  - AAAI 2026
  - MAPF
  - Local Guidance
  - LaCAM
  - Congestion Mitigation
  - Space-Time Search
date: 2026-05-08
content_hash: c5a2dfc266d48f78
---

# Local Guidance for Configuration-Based Multi-Agent Pathfinding

**Conference**: AAAI 2026
**arXiv**: [2510.19072](https://arxiv.org/abs/2510.19072)
**Code**: [https://github.com/allegorywrite/lg_lacam](https://github.com/allegorywrite/lg_lacam)
**Area**: Other
**Keywords**: MAPF, Local Guidance, LaCAM, Congestion Mitigation, Space-Time Search

## TL;DR

This paper introduces the concept of Local Guidance (LG) to improve solution quality in LaCAM-based multi-agent pathfinding. By constructing local space-time paths for each agent at every configuration generation step, LG mitigates congestion and reduces solution cost by up to 50%, while maintaining completion within a few seconds for 1,000 agents.

## Background & Motivation

**State of the Field**: Multi-Agent Path Finding (MAPF) aims to find collision-free paths for all agents on a graph. Configuration-based solvers such as LaCAM can handle thousands of agents efficiently, but the resulting solution quality is highly suboptimal.

**Limitations of Prior Work**: Existing guidance methods alleviate congestion by globally considering the entire environment and all agents. However, global guidance is time-agnostic, discarding space-time information and yielding limited effectiveness in bottleneck regions.

**Root Cause**: Global guidance is too coarse—it identifies less congested routes but lacks awareness of when and where congestion occurs. Exact solvers, on the other hand, are computationally prohibitive due to the full collision constraint search space.

**Starting Point**: Local guidance occupies the middle ground—constructing guidance paths within a local space-time window around each agent. This approach is more precise than global guidance (with space-time awareness) while remaining more lightweight than exact solvers.

## Method

### Overall Architecture

Within the LaCAM framework, local guidance is invoked at each configuration generation step. For each agent, a short-term collision-aware path $\Phi[i]$ is planned within a window of size $w$. The first step $\Phi[i][1]$ is then used to guide PIBT's preference ordering.

### Key Designs

1. **Local Guidance Construction (Algorithm 1)**:

    - **Function**: Plans an independent $w$-step short-term path for each agent, accounting for collisions with other agents' paths.
    - **Mechanism**: Each agent runs a cost-based windowed space-time A*, with cost function $c(\pi[t], \pi[t+1]) = \langle 1 + \alpha \cdot \mathsf{Ind}[\chi > 0], \chi \rangle$, where $\chi$ is the number of collisions with other paths and $\alpha=3$ is the collision penalty weight.
    - **Design Motivation**: Soft collision constraints (rather than hard constraints) allow agents to find compromise paths in congested regions by penalizing rather than prohibiting collisions.

2. **Path Caching and Initialization (Algorithm 2)**:

    - **Function**: Initializes the current step's guidance using the previous step's planned path, avoiding replanning from scratch.
    - **Mechanism**: If agent $i$'s current position matches the first step of its previous guidance path, the path is simply shifted: $\Phi_k[i][t] \leftarrow \Phi_{k-1}[i][t+1]$.
    - **Design Motivation**: Reduces the required number of planning iterations from 2 to 1, substantially lowering computational overhead.

3. **Agent Ordering**:

    - **Function**: Plans agents in descending order of their collision count from the previous round.
    - **Mechanism**: Agents with the most collisions are planned first, granting them greater freedom in path selection.
    - **Design Motivation**: Mitigates the inequity inherent in sequential planning, where earlier-planned agents face fewer constraints than later ones.

4. **Global + Local Guidance Fusion**:

    - **Mechanism**: Incorporates a global guidance offset into the cost function: $c = \langle 1 + \alpha \cdot \mathsf{Ind}[\chi > 0], \delta(\pi[t+1]), \chi \rangle$, where $\delta(v)$ denotes the distance to the global guidance path.

### Time Complexity

Single guidance construction: $O(nw|V|\log(w|V|))$, where $n$ is the number of agents, $w$ is the window size, and $|V|$ is the number of graph vertices.

## Key Experimental Results

### Main Results: Solution Cost Comparison Across Maps (1,000 Agents)

| Method | maze-128 | random-64 | warehouse | den312d | Avg. Reduction |
|--------|----------|-----------|-----------|---------|----------------|
| LaCAM | baseline | baseline | baseline | baseline | — |
| GG (Global Guidance) | -14% | -8% | -15% | -12% | ~12% |
| LG (Local Guidance) | **-38%** | **-20%** | -12% | **-25%** | ~24% |
| LG+GG | **-40%** | **-22%** | **-18%** | **-28%** | ~27% |
| LNS2 | -30% | -15% | -10% | failed | — |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| LG w/ caching + 1 iteration (default) | Best cost-performance tradeoff |
| LG w/o caching (replan from scratch) | Significant quality degradation |
| LG w/ 2 iterations | Marginal improvement, doubled runtime |
| Infrequent updates (every 2/3 steps) | May perform worse than no guidance |
| $\alpha=3, w=20$ | Sweet spot |

### Key Findings
- **Local guidance outperforms global guidance on most maps**, particularly in bottleneck-heavy environments (e.g., 38% reduction on maze vs. 14% for global guidance).
- **The only exception is the warehouse map**: global guidance is more effective in long narrow corridors, where it suppresses oscillatory movement.
- **"Live" guidance is critical**: updating the guidance path at every step is essential; reducing update frequency can degrade performance below the no-guidance baseline.
- When combined with anytime methods (LNS), **a better initial solution matters more than additional optimization time in high-density scenarios**.
- Scalability is maintained at 10,000 agents with approximately 30% cost reduction.

## Highlights & Insights
- **Spectrum view of local vs. global guidance**: Framing MAPF solvers as a continuum from coarse global guidance to exact collision-free solutions—with local guidance as a practical midpoint—is a conceptually illuminating perspective.
- **Caching and incremental update**: Exploiting the temporal continuity of DFS search to reuse path caches reduces the overhead of two iterations to one, representing an elegant engineering contribution.
- **Practical value of soft collision constraints**: Compared to hard constraints that induce overly conservative behavior, soft constraints achieve a favorable balance between congestion mitigation and path efficiency.

## Limitations & Future Work
- LG underperforms global guidance on warehouse-style maps with long narrow corridors, as local field-of-view is insufficient to optimize global traffic flow.
- The window size $w$ and collision penalty $\alpha$ require per-map tuning.
- Validation is limited to PIBT as the configuration generator; other generators remain unexplored.
- Per-step update overhead, while acceptable, is still several times slower than the guidance-free baseline.

## Related Work & Insights
- **vs. GG (Global Guidance, SUO)**: Global guidance mitigates congestion in a time-agnostic manner, whereas LG is space-time aware and more precise in bottleneck regions.
- **vs. LNS2**: Both are suboptimal MAPF methods; LG achieves better solution quality and faster runtime on most maps.
- **vs. lacam3 (SOTA)**: LG+LNS matches or surpasses lacam3 on non-warehouse maps.
- **vs. Learning-Based Methods (Imitation Learning / RL)**: LG is a purely search-based method with stronger generalization and no training data requirement, though it may be outperformed by environment-specific learned policies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The local guidance concept is clear and practical, though it is essentially a variant of windowed prioritized planning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple maps, scales, and includes extensive ablation and design validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logically structured with intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐ Advances the Pareto frontier of real-time MAPF.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)
- [\[CVPR 2026\] AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](../../CVPR2026/others/assistmimic_physics_grounded_humanoid_assistance.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)

<!-- RELATED:END -->
