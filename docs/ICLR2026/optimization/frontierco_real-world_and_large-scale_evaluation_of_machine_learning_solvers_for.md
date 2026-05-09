---
title: >-
  [Paper Note] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization
description: >-
  [ICLR 2026][Optimization][combinatorial optimization] FrontierCO is a large-scale, real-world benchmark covering 8 categories of combinatorial optimization problems (TSP, MIS, CVRP, etc.), evaluating 16 ML solvers (neural methods + LLM agents) against state-of-the-art classical solvers. The benchmark reveals that ML methods remain significantly behind classical approaches on structurally complex and extremely large-scale instances, though they show potential to surpass classical methods in certain scenarios.
tags:
  - ICLR 2026
  - Optimization
  - combinatorial optimization
  - ML solver
  - benchmark
  - real-world instances
  - LLM agent
date: 2026-05-08
content_hash: 7dafedabc177a540
---

# FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization

**Conference**: ICLR 2026
**arXiv**: [2505.16952](https://arxiv.org/abs/2505.16952)
**Code**: [HuggingFace](https://huggingface.co/datasets/CO-Bench/FrontierCO)
**Area**: Agent / Combinatorial Optimization
**Keywords**: combinatorial optimization, ML solver, benchmark, real-world instances, LLM agent

## TL;DR
FrontierCO is a large-scale, real-world benchmark covering 8 categories of combinatorial optimization problems (TSP, MIS, CVRP, etc.), evaluating 16 ML solvers (neural methods + LLM agents) against state-of-the-art classical solvers. The benchmark reveals that ML methods remain significantly behind classical approaches on structurally complex and extremely large-scale instances, though they show potential to surpass classical methods in certain scenarios.

## Background & Motivation
**Background**: ML for combinatorial optimization (CO) has advanced rapidly in recent years, encompassing end-to-end neural solvers (GNN, RL, diffusion models) and LLM agent methods (FunSearch, ReEvo, etc.), which have demonstrated promising results on small-scale synthetic benchmarks.

**Limitations of Prior Work**: Three major limitations exist: ① **Scale**: most evaluations are conducted on toy-level instances (e.g., TSP ≤ 10K nodes), whereas real-world applications require handling millions of nodes; ② **Authenticity**: synthetic datasets fail to capture the structural diversity of real-world instances (e.g., non-Euclidean graphs, competition-grade irregular instances); ③ **Coverage**: a unified evaluation protocol across problem types is lacking.

**Key Challenge**: The apparent "progress" of ML methods on synthetic benchmarks may stem from the simplicity and regularity of those problems rather than genuine methodological effectiveness. Validation under real-world structure and extreme scale is therefore necessary.

**Goal**: To provide a rigorous, real-world-grounded CO benchmark suite that enables unified evaluation of ML solvers against classical state-of-the-art solvers.

**Key Insight**: Real-world instances are collected from competition repositories (DIMACS, TSPLib, PACE Challenge) and partitioned into two test sets—easy (solvable with existing methods) and hard (open problems)—scaling up to 10 million nodes for TSP and 8 million nodes for MIS.

**Core Idea**: Progress in ML for CO must be validated under real-world structure and extreme scale, rather than measured solely by performance on synthetic data.

## Method

### Overall Architecture
FrontierCO covers 8 CO problem categories (routing: TSP/CVRP; graph: MIS/MDS; facility location: CFLP/CPMP; scheduling: FJSP; tree: STP), with primal gap adopted as the unified evaluation metric: $\text{pg}(x;s) = |cost(x;s) - c^*| / \max(|cost(x;s)|, |c^*|)$, ranging in $[0,1]$ where 0 denotes optimality. Each problem provides: an easy set (historically challenging but now solvable), a hard set (open or computationally intensive instances), and standardized training/validation splits.

### Key Designs

1. **Instance Scale and Data Sources**:

    - TSP: up to 10 million nodes (prior ML evaluations maxed at 10K)
    - MIS: up to 8 million nodes (prior maximum was 11K)
    - Data sources: TSPLib, DIMACS Challenge, PACE Challenge 2025, BHOSLib, etc.
    - Design Motivation: Real-world CO instances exhibit irregular structure that uniform synthetic distributions cannot represent.

2. **Construction Logic of the Hard Set**:

    - "Hard" is not synonymous with "larger"—structural complexity is emphasized
    - Examples include PUC hypercube graphs for STP and SAT-induced instances for MIS
    - Many instances have no known optimal solution, preventing heuristic hacking and memorization
    - Design Motivation: Forces ML methods to demonstrate genuine generalization capability.

3. **Evaluation Protocol**:

    - 16 ML solvers: neural solvers (DiffUCO, SDDS, LEHD, DIFUSCO, SIL, DeepACO, etc.) and LLM agents (FunSearch, Self-Refine, ReEvo)
    - Classical SOTA baselines: KaMIS (MIS), LKH-3 (TSP), HGS (CVRP), Gurobi (MDS/CFLP), CPLEX (FJSP), etc.
    - Unified time limit: 1 hour per instance
    - Unified hardware: single CPU core + single GPU

4. **Standardized Training/Validation Data**:

    - Resolves inconsistencies in synthetic data across publications
    - Includes data loaders, evaluation functions, and LLM agent templates
    - Evaluation functions are withheld from LLMs to prevent data leakage

## Key Experimental Results

### Main Results — Primal Gap (%) on the Easy Set

| Area | Classical SOTA | Best Neural | Best LLM |
|------|---------------|-------------|----------|
| TSP | **0.00** (LKH-3) | 0.16 (LEHD) | 3.82 (ReEvo) |
| MIS | **0.00** (KaMIS) | 0.37 (SDDS) | 7.21 |
| CVRP | **0.14** (HGS) | 1.73 (SIL) | 12.5 |
| CFLP | **0.00** (Gurobi) | 0.91 (SORREL) | 5.4 |
| FJSP | **0.00** (CPLEX) | 8.2 (MPGN) | 15.3 |

### Gap Analysis on the Hard Set
- Gaps **widen dramatically** on the Hard Set.
- TSP at 10M nodes: classical methods achieve ~1% gap; the best neural method reaches ~15%.
- MIS on structured instances: neural methods nearly completely fail on SAT-induced graphs.
- LLM agent methods exhibit consistently high variance—occasionally surpassing classical methods but in an unstable manner.

### Ablation Study

| Dimension | Finding |
|-----------|---------|
| Scale Scaling | Neural performance degrades exponentially with scale; classical methods degrade linearly |
| Structural Complexity | Non-Euclidean and irregular structures most severely impact neural methods |
| Generalization | Distribution shift from synthetic to real instances causes significant performance drops |
| LLM Stability | Methods such as ReEvo exhibit extremely high variance across runs on the same problem |

### Key Findings
- **ML methods are competitive on easy instances but comprehensively lag on hard instances**, with gaps amplified on structurally complex and large-scale instances.
- **Neural methods can augment simple heuristics** but cannot replace carefully engineered, specialized solvers.
- **LLM agents occasionally surpass classical SOTA**, but with high variance, as they lack the ability to identify which components of their generated algorithms are genuinely effective.
- **Distribution shift** from synthetic to real-world data is the core bottleneck for neural methods.
- **Scheduling and facility location** problems with complex constraints are particularly challenging for ML methods.

## Highlights & Insights
- **Striking Scale Contrast**: TSP pushed from 10K to 10M nodes, and MIS from 11K to 8M—exposing fundamental scalability deficiencies of ML methods.
- **"Hard ≠ Large" Design Philosophy**: The hard set emphasizes structural complexity rather than scale alone, as exemplified by PUC hypercube and SAT-induced graphs—a more meaningful challenge than simply increasing instance size.
- **Double-Edged Nature of LLM Agents**: High variance yet occasional surpassing of SOTA—indicating that LLM code generation is creative but lacks stability and deep understanding.
- **Value of Standardization**: Providing unified training data and evaluation protocols eliminates the "apples-to-oranges" comparisons prevalent across publications.

## Limitations & Future Work
- The 1-hour time limit per instance may be insufficient for certain methods to converge.
- Evaluation is restricted to a single-GPU setup without consideration of distributed parallel execution.
- Neural baselines for certain problems (e.g., STP) are relatively weak, potentially underestimating ML's potential.
- Several emerging methods (e.g., recent advances in neural-guided Branch-and-Bound) are not included.
- The best-known solutions (BKS) in the hard set may not represent true optima, making primal gap sensitive to reference solution quality.

## Related Work & Insights
- **vs. CO-Bench**: FrontierCO focuses on real-world instances and extreme scale, whereas CO-Bench is more oriented toward LLM agent evaluation.
- **vs. Existing Neural CO Evaluations**: Prior evaluations rely on synthetic data and small scales; FrontierCO exposes the unreliability of such evaluation paradigms.
- **vs. DIMACS Competition**: Borrows the competition-style evaluation philosophy but targets the ML community with a lower participation barrier.
- The benchmark provides important guidance for the ML for CO research agenda: generalization capability and structural adaptability should take priority over performance gains on synthetic benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ First truly large-scale, real-world benchmark for ML for CO
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 problem types × 16 ML solvers + classical SOTA baselines
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive data presentation
- Value: ⭐⭐⭐⭐⭐ Serves as a revealing "litmus test" with significant cautionary value for the ML for CO community

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[ICLR 2026\] SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICLR 2026\] Celo2: Towards Learned Optimization Free Lunch](celo2_towards_learned_optimization_free_lunch.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)

<!-- RELATED:END -->
