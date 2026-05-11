---
title: >-
  [Paper Note] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization
description: >-
  [AAAI 2026][Optimization][Multi-Objective Optimization] This paper proposes MPaGE, a framework that integrates LLMs with a Pareto Front Grid mechanism and semantic clustering to automatically generate heuristics for mult…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Multi-Objective Optimization"
  - "LLM"
  - "Pareto Front Grid"
  - "SEMO"
  - "Heuristic Generation"
  - "Semantic Diversity"
date: 2026-05-08
content_hash: 94ff10e06e3f9dad
---

# Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization

**Conference**: AAAI 2026
**arXiv**: [2507.20923](https://arxiv.org/abs/2507.20923)
**Code**: [GitHub](https://github.com/langkhachhoha/MPaGE)
**Area**: Combinatorial Optimization / LLM-based Automatic Heuristic Design
**Keywords**: Multi-Objective Optimization, LLM, Pareto Front Grid, SEMO, Heuristic Generation, Semantic Diversity

## TL;DR
This paper proposes MPaGE, a framework that integrates LLMs with a Pareto Front Grid mechanism and semantic clustering to automatically generate heuristics for multi-objective combinatorial optimization problems (MOCOPs), jointly optimizing solution quality and runtime efficiency. MPaGE achieves significant improvements in HV and IGD over baselines such as EoH and MEoH on Bi-TSP, Tri-TSP, Bi-CVRP, and Bi-KP benchmarks.

## Background & Motivation
MOCOPs arise broadly in domains such as vehicle routing and production scheduling, requiring the identification of Pareto-optimal fronts across conflicting objectives. Traditional evolutionary algorithms (e.g., NSGA-II, MOEA/D) depend heavily on domain expertise and parameter tuning, limiting their generalizability.

Recent advances have demonstrated the strong potential of LLMs for automatic heuristic design (EoH, FunSearch, ReEvo), leveraging code generation to directly produce executable optimization algorithms. However, existing LLM-based methods predominantly target single-objective settings and suffer from three key limitations in multi-objective scenarios: (1) runtime efficiency is neglected — only solution quality is optimized; (2) heuristics in the population exhibit highly similar logic — yielding a false sense of diversity through different implementations of equivalent strategies; (3) no structured guidance is provided over the objective space.

## Core Problem
How to systematically leverage LLMs to automatically design a set of Pareto-optimal heuristics for MOCOPs, simultaneously optimizing solution quality and runtime efficiency while maintaining **semantic diversity** among heuristics?

## Method

### Overall Architecture
MPaGE operates under the Simple Evolutionary Multiobjective Optimization (SEMO) paradigm, iteratively evolving a population of heuristics. Each heuristic is encoded as a natural language description paired with Python code and is evaluated on two objectives: $e_1$ (negative hypervolume, measuring solution quality) and $e_2$ (runtime).

### Key Designs
**Pareto Front Grid (PFG)**: The objective space $[0,1]^2$ is partitioned into a grid, with non-dominated sorting applied within each cell to retain elite individuals:

$$G(h_i) = \left(\left\lfloor \frac{e_1(h_i)}{\delta_1} \right\rfloor, \left\lfloor \frac{e_2(h_i)}{\delta_2} \right\rfloor\right)$$

The selection strategy chooses parents from neighboring grid cells with probability $\epsilon=0.9$ (local exploitation), and from the global elite set otherwise (global exploration).

**Semantic Clustering**: An LLM (GPT-4o) analyzes the semantic structure of elite heuristics and groups those with equivalent logic but different implementations into the same cluster:

$$\text{SemClust}(P) = \{C_1, C_2, \ldots, C_m\}, \quad \bigcup C_i = P, \quad C_i \cap C_j = \emptyset$$

Mutation is performed within clusters (exploring local variants), while crossover is performed across clusters (combining distinct behaviors), with probability $\gamma=0.3$ governing the choice between the two operators.

**Feedback Reflection**: For each pair of parent heuristics, the LLM first analyzes their respective strengths and weaknesses to produce textual suggestions, which then guide offspring generation — effectively embedding LLM reasoning within the evolutionary operators.

### Automatic Generation Pipeline
1. LLM initializes the population (descriptions + code)
2. PFG partitions the objective space and constructs the elite pool
3. Semantic clustering → inter-cluster crossover / intra-cluster mutation
4. Feedback reflection guides offspring generation
5. Non-dominated sorting updates the population
6. Iteration until stopping criterion is met

## Key Experimental Results

| Method | Bi-TSP20 HV↑ | Bi-TSP20 IGD↓ | Tri-TSP20 HV↑ | Bi-CVRP50 HV↑ | Bi-KP50 HV↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| EoH | 0.756 | 0.117 | 0.755 | 0.957 | 0.602 |
| MEoH | 0.724 | 0.067 | 0.884 | 0.322 | 0.748 |
| ReEvo | 0.541 | 0.435 | 0.694 | 0.658 | 0.996 |
| **MPaGE** | **0.911** | **0.010** | **0.936** | **0.980** | **0.932** |

Generalization (Bi-TSP, out-of-distribution problem sizes):
- TSP20: HV 0.629 (MPaGE best) vs. 0.603 (MEoH best), 165× speedup
- TSP50: HV 0.542 vs. 0.480, 126× speedup
- TSP100: HV 0.442 vs. 0.395, 79× speedup

Compared to conventional MOEAs (e.g., NSGA-II, PFG-MOEA), MPaGE-generated heuristics achieve **50–200×** speedups with comparable or superior HV.

## Highlights & Insights
- First systematic evaluation of LLM-generated heuristics on standard MOCOP benchmarks, jointly optimizing quality and efficiency
- Semantic clustering addresses the limitation of AST-based diversity measures, which cannot distinguish heuristics that are logically equivalent but syntactically distinct
- The PFG grid mechanism simultaneously promotes diversity and guides convergence in the objective space
- The best-generated heuristics generalize well to out-of-distribution problem sizes
- Experiments are conducted on lightweight hardware (M1 Mac + 8GB RAM), demonstrating practical accessibility

## Limitations & Future Work
- Dependence on GPT-4o/4o-mini introduces LLM API costs and latency as potential bottlenecks
- Population size and iteration count are relatively small (10 individuals, 20 generations); scalability to larger search spaces remains untested
- Only two optimization objectives (quality and runtime) are considered; extensions to additional objectives (e.g., code complexity, generalizability) are left for future work
- The quality of semantic clustering depends on the LLM's code comprehension capability, which may be unstable for complex heuristics
- Evaluation instances are of modest scale (20–200); performance on large-scale instances has yet to be verified

## Related Work & Insights
- **EoH/ReEvo/HSEvo**: Single-objective frameworks that optimize solution quality only, incapable of balancing efficiency; MPaGE addresses this through bi-objective optimization
- **MEoH**: Also a multi-objective framework but relies on AST-based diversity metrics, which fail to distinguish semantically equivalent heuristics; MPaGE replaces this with LLM-based semantic clustering
- **Conventional MOEAs (NSGA-II/MOEA/D)**: Yield higher solution quality but incur 50–200× longer runtimes; MPaGE achieves comparable quality with substantial acceleration
- **Original SEMO**: Relies on random neighborhood sampling; MPaGE uses LLM-guided selection and neighborhood exploration, yielding significant quality improvements

The paradigm of using LLMs as evolutionary operators has broad applicability and can be extended to automatic algorithm design for other NP-hard problems. The combination of PFG partitioning and semantic clustering is generalizable to any evolutionary search scenario requiring multi-objective balance with maintained diversity. The feedback reflection mechanism resembles self-play, leveraging LLM reasoning to guide the search trajectory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](../../NeurIPS2025/optimization/constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Bayes](../../NeurIPS2025/optimization/large_language_bayes.md)
- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](../../NeurIPS2025/optimization/training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)

</div>

<!-- RELATED:END -->
