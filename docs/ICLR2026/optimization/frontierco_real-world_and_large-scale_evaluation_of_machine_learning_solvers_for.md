---
title: >-
  [Paper Note] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization
description: >-
  [ICLR 2026][Optimization & Theory][combinatorial optimization] FrontierCO is a large-scale real-world benchmark covering 8 categories of combinatorial optimization (CO) problems (TSP, MIS, CVRP, etc.). It evaluates 16 ML solvers (neural methods + LLM Agents) against SOTA traditional solvers, finding that ML methods still lag significantly behind traditional approaches on structura
tags:
  - ICLR 2026
  - Optimization & Theory
  - combinatorial optimization
  - ML solver
  - benchmark
  - real-world instances
  - LLM agent
date: 2026-05-08
content_hash: dc26ec190716120e
---
# FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization

**Conference**: ICLR 2026  
**arXiv**: [2505.16952](https://arxiv.org/abs/2505.16952)  
**Code**: [HuggingFace](https://huggingface.co/datasets/CO-Bench/FrontierCO)  
**Area**: Agent / Combinatorial Optimization  
**Keywords**: combinatorial optimization, ML solver, benchmark, real-world instances, LLM agent

## TL;DR
FrontierCO is a large-scale real-world benchmark covering 8 categories of combinatorial optimization (CO) problems (TSP, MIS, CVRP, etc.). It evaluates 16 ML solvers (neural methods + LLM Agents) against SOTA traditional solvers, finding that ML methods still lag significantly behind traditional approaches on structurally complex and extreme-scale instances, despite showing potential in specific scenarios.

## Background & Motivation
**Background**: The use of ML for CO has developed rapidly in recent years, including end-to-end neural solvers (GNN, RL, diffusion models) and LLM Agent methods (FunSearch, ReEvo), demonstrating promising results on small-scale synthetic benchmarks.

**Limitations of Prior Work**: Three major limitations—① **Scale**: Most evaluations are conducted on toy-level instances (e.g., TSP $\le$ 10K nodes), whereas practical applications require handling millions of nodes; ② **Realism**: Synthetic datasets fail to capture the structural diversity of the real world (e.g., non-Euclidean graphs, irregular competition-grade instances); ③ **Coverage**: A lack of unified evaluation protocols across different problem types.

**Key Challenge**: The "progress" of ML methods on synthetic benchmarks might stem from problems being too simple or regular rather than the methods being truly effective. Testing under real structures and extreme scales is necessary.

**Goal**: Provide a rigorous, real-world-based CO benchmark suite to uniformly evaluate the gap between ML solvers and traditional SOTA solvers.

**Key Insight**: Collect real-world instances from competition libraries (DIMACS, TSPLib, PACE Challenge), categorized into easy (currently solvable) and hard (open problems) test sets, pushing scales up to 10 million nodes for TSP and 8 million nodes for MIS.

**Core Idea**: Progress in ML for CO must be verified against real-world structures and extreme scales rather than merely optimization on synthetic data.

## Method

### Overall Architecture
FrontierCO is a real-world evaluation benchmark covering 8 categories of CO problems across routing (TSP, CVRP), graphs (MIS, MDS), facility location (CFLP, CPMP), scheduling (FJSP), and tree structures (STP). Each problem includes an easy set (historically difficult but now solvable), a hard set (instances that remain open or are extremely computationally expensive), and standardized training/validation data. All methods are compared under a uniform time budget and hardware using identical metrics to establish a reproducible baseline for how far ML solvers lag behind traditional ones.

The benchmark uses primal gap to measure solution quality. For a solution $x$ on instance $s$, it is defined as $\text{pg}(x;s) = |cost(x;s) - c^*| / \max(|cost(x;s)|, |c^*|)$, where $c^*$ is the known optimal or best reference solution. The value falls within $[0,1]$, where closer to 0 is better. This normalization allows for horizontal comparison across different scales and magnitudes and directly exposes the performance degradation of ML methods.

### Key Designs

**1. Real Data + Extreme Scale: Pushing Evaluation Beyond Toy Problems**
Previous evaluations of ML for CO mostly stayed on small-scale synthetic instances—TSP maxing at ~10K nodes and MIS at ~1.1K nodes, with nodes following uniform distributions and overly regular structures. FrontierCO collects real instances directly from competition-grade sources (TSPLib, DIMACS Challenge, PACE Challenge 2025, BHOSLib, etc.), pushing scales to 10 million nodes for TSP and 8 million nodes for MIS. The key logic is that real-world CO instances are often non-Euclidean and irregular graphs that synthetic uniform distributions cannot represent; ML methods' true ability to "learn to solve" is only revealed under real structures and extreme scales.

**2. Hard Set Emphasizing "Difficulty" over just "Size": Blocking Heuristic Hacking**
The construction of the hard set deliberately emphasizes structural complexity rather than just increasing scale. Examples include PUC hypercube graphs in STP and graphs induced from SAT instances in MIS. Such structures are particularly lethal to neural methods that rely on local patterns. Crucially, many instances in the hard set lack known optimal solutions, which mechanistically prevents "heuristic hacking" and memorization of answers—methods cannot cheat by "remembering" reference solutions and must demonstrate true generalization.

**3. Unified Evaluation Protocol: 16 ML Solvers vs. SOTA**
The evaluation includes 16 ML solvers, ranging from neural solvers (DiffUCO, SDDS, LEHD, DIFUSCO, SIL, DeepACO, etc.) to LLM Agent methods (FunSearch, Self-Refine, ReEvo). Traditional SOTA for each problem serves as the reference: KaMIS for MIS, LKH-3 for TSP, HGS for CVRP, Gurobi for MDS/CFLP, and CPLEX for FJSP. To ensure comparability, all methods are constrained to the same limits: 1 hour per instance, using a single CPU core and a single GPU, avoiding false advantages gained through computational scaling.

**4. Standardized Training/Validation Data: Ending "Apples to Oranges" Comparisons**
Comparisons across papers have long been hindered by inconsistent synthetic data. FrontierCO provides uniform training/validation sets for every problem, accompanied by data loaders, evaluation functions, and LLM Agent templates. This ensures methods are trained on the same data and scored by the same functions. Evaluation functions are hidden from the LLMs to prevent agents from reading the scoring logic and causing data leakage, ensuring the credibility of LLM Agent results.

## Key Experimental Results

### Main Results — Primal Gap (%) on Easy Set

| Area | SOTA Traditional | Best Neural | Best LLM |
|------|----------|------------|---------|
| TSP | **0.00** (LKH-3) | 0.16 (LEHD) | 3.82 (ReEvo) |
| MIS | **0.00** (KaMIS) | 0.37 (SDDS) | 7.21 |
| CVRP | **0.14** (HGS) | 1.73 (SIL) | 12.5 |
| CFLP | **0.00** (Gurobi) | 0.91 (SORREL) | 5.4 |
| FJSP | **0.00** (CPLEX) | 8.2 (MPGN) | 15.3 |

### Gap Analysis on Hard Set
- The gap **widens drastically** on the Hard Set.
- TSP 10M nodes: Traditional methods gap ~1%, best Neural gap ~15%.
- MIS structured instances: Neural methods almost completely fail on SAT-induced graphs.
- LLM Agent methods generally show high variance, occasionally outperforming traditional methods by chance but remaining unstable.

### Ablation Study

| Dimension | Finding |
|------|------|
| Scale Expansion | Neural performance degrades exponentially with scale; traditional methods degrade linearly. |
| Structural Complexity | Non-Euclidean/irregular structures impact Neural methods most severely. |
| Generalization | Distribution shift from synthetic to real results in significant performance drops. |
| LLM Stability | Methods like ReEvo show extreme variance across different runs of the same problem. |

### Key Findings
- **ML methods are competitive on easy instances but fall behind entirely on hard instances**, with the gap increasing alongside structural complexity and scale.
- **Neural methods can enhance simple heuristics** but cannot yet replace finely engineered specialized solvers.
- **LLM Agents occasionally outperform SOTA traditional methods** but suffer from high variance because they do not understand which parts of their generated algorithms are truly effective.
- **Distribution shift** from synthetic to real data remains the core bottleneck for Neural methods.
- Complex constraint problems like **Scheduling and Facility Location** are particularly difficult for ML methods.

## Highlights & Insights
- **Stark Scale Comparison**: Pushing TSP from 10K to 10M and MIS from 11K to 8M exposes fundamental scaling flaws in ML methods.
- **"Hard $\neq$ Large" Design Philosophy**: Emphasizing structural complexity (e.g., PUC hypercubes and SAT-induced graphs) over simple size is more meaningful for evaluation.
- **Double-edged Sword of LLM Agents**: High variance yet occasional SOTA-beating performance suggests LLM code generation is creative but lacks stability and deep understanding.
- **Value of Standardization**: Providing unified training data and evaluation protocols ends the "apples to oranges" confusion in cross-paper comparisons.

## Limitations & Future Work
- The 1-hour time limit per instance might be insufficient for some methods to converge.
- Only single GPU settings were evaluated, without considering distributed parallelism.
- Neural baselines for certain problems (e.g., STP) are relatively weak, potentially underestimating ML potential.
- Emerging methods (e.g., recent progress in Neural-guided Branch-and-Bound) were not included.
- The BKS (Best Known Solution) for the hard set might not be the true optimum; primal gap is affected by the quality of reference solutions.

## Related Work & Insights
- **vs. CO-Bench**: FrontierCO focuses on real-world instances and extreme scale, while CO-Bench is more oriented toward LLM Agent evaluation.
- **vs. Existing Neural CO Evaluations**: Previous evaluations relied on synthetic data and small scales; FrontierCO exposes the unreliability of those evaluation methods.
- **vs. DIMACS Competition**: Adopts competition-style evaluation but targets the ML community, lowering the barrier to entry.
- Provides a significant guide for ML for CO research: Focus should shift toward generalization and structural adaptability rather than leaderboard climbing on synthetic datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ The first truly large-scale real-world ML for CO benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 problem types, 16 ML solvers, and SOTA traditional solvers.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive data presentation.
- Value: ⭐⭐⭐⭐⭐ Acts as a "reality check" for the ML for CO community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Sequential to Parallel: Reformulating Dynamic Programming as GPU Kernels for Large-Scale Stochastic Combinatorial Optimization](from_sequential_to_parallel_reformulating_dynamic_programming_as_gpu_kernels_for.md)
- [\[ICLR 2026\] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization](neuclip_efficient_large-scale_clip_training_with_neural_normalizer_optimization.md)
- [\[ICLR 2026\] Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers](beyond_the_heatmap_a_rigorous_evaluation_of_component_impact_in_mcts-based_tsp_s.md)
- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)

</div>

<!-- RELATED:END -->
