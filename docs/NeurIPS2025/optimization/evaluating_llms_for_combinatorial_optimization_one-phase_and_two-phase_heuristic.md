---
title: >-
  [Paper Note] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing
description: >-
  [NeurIPS 2025][Optimization][LLM Evaluation] This paper proposes a systematic evaluation framework combining LLMs with evolutionary algorithms to assess the capability of LLMs in generating and optimizing heuristics for…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "LLM Evaluation"
  - "Combinatorial Optimization"
  - "2D Bin-Packing"
  - "Evolutionary Algorithms"
  - "Heuristic Generation"
date: 2026-05-08
content_hash: 39ea30aa5acf0508
---

# Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing

**Conference**: NeurIPS 2025
**arXiv**: [2509.22255](https://arxiv.org/abs/2509.22255)  
**Code**: [Provided in Appendix](https://arxiv.org/abs/2509.22255)  
**Area**: Optimization / LLM Applications
**Keywords**: LLM Evaluation, Combinatorial Optimization, 2D Bin-Packing, Evolutionary Algorithms, Heuristic Generation

## TL;DR
This paper proposes a systematic evaluation framework combining LLMs with evolutionary algorithms to assess the capability of LLMs in generating and optimizing heuristics for the 2D bin-packing problem. GPT-4o achieves optimal solutions within 2 iterations, reducing the average number of bins from 16 to 15 and improving space utilization from 0.76–0.78 to 0.83.

## Background & Motivation

**Background**: The evaluation of LLMs is expanding beyond traditional NLP tasks into specialized domains such as mathematical reasoning and algorithm design. The 2D Bin-Packing Problem (2D-BPP) is a classical NP-hard combinatorial optimization problem requiring rectangular items to be packed into the minimum number of fixed-size bins.

**Limitations of Prior Work**: (a) Traditional heuristics such as Finite First-Fit (FFF) and Hybrid First-Fit (HFF) are efficient but yield solutions of limited quality; (b) Prior work such as FunSearch has demonstrated the potential of LLMs to generate high-performance heuristics within evolutionary loops, yet a **systematic evaluation framework** for quantifying LLM capabilities in such tasks remains absent.

**Key Challenge**: It remains unclear how to systematically evaluate whether LLMs genuinely understand complex algorithmic constraints, whether they can continuously improve solution quality, and what the quantitative gap is relative to traditional methods.

**Goal**: To establish a structured methodology for evaluating the effectiveness of LLMs in combinatorial optimization, encompassing evaluation metrics, iterative procedures, and baseline comparisons.

**Key Insight**: Inspired by FunSearch, the paper introduces an "island model" evolutionary loop — LLM generation → correctness verification → scoring → clustering (islands) → feedback-driven refinement.

**Core Idea**: An iterative evolutionary framework enables LLMs to automatically generate, evaluate, and improve heuristics for 2D-BPP, with multi-dimensional metrics used to systematically assess their optimization capability.

## Method

### Overall Architecture
The input consists of a structured prompt (containing problem constraints, function prototypes, and I/O format). GPT-4o generates Python heuristic scripts → correctness verification (no overlap, within bounds, complete packing) → multi-dimensional scoring (number of bins, space utilization, runtime) → island clustering → Top-3 islands each contribute one script as feedback to the next round, iterated over 6 rounds.

### Key Designs

1. **Mathematical Formulation**:

    - **Function**: Formalizes 2D-BPP as an integer program.
    - **Mechanism**: $n$ items with dimensions $(w_i, h_i)$, bin dimensions $(W, H)$, decision variable $z_{ij}=1$ indicating item $i$ is placed in bin $j$, and $u_j = 1$ indicating bin $j$ is used. Objective: $\min \sum_{j=1}^n u_j$, subject to: each item assigned to exactly one bin $\sum_j z_{ij} = 1$, coordinate constraints $0 \leq x_{ij} \leq (W-w_i)z_{ij}$, and non-overlap constraints.
    - Total utilization: $\rho_{\text{total}} = \frac{\sum_{i} w_i h_i}{(\sum_j u_j) WH}$

2. **Six-Step Iterative Evolutionary Process**:

    - **Step 1 – Structured Prompt**: Precisely describes problem constraints, function signatures, and I/O format. Template functions are provided to ensure syntactic consistency.
    - **Step 2 – Code Generation and Verification**: 20 scripts are generated per round and strictly validated for constraint satisfaction (no overlap, within bounds, no duplicate assignment); non-compliant scripts are discarded.
    - **Step 3 – Scoring**: Primary metric is number of bins; secondary metric is space utilization; tertiary metric is execution time.
    - **Step 4 – Island Clustering**: High-scoring scripts are grouped into "islands" by logical similarity, maintaining diversity to prevent premature convergence.
    - **Step 5 – Feedback Refinement**: One script is randomly sampled from each of the Top-3 islands and embedded as "best-shot" examples in the next round's prompt, guiding the LLM to learn from successful strategies.
    - **Step 6 – Termination Condition**: The process terminates after 6 rounds, based on resource constraints and diminishing marginal returns.

3. **Baseline Methods**:

    - **FFF (Finite First-Fit)**: Items sorted by height and placed into the first available position; $O(n^2)$ complexity. Single-phase greedy strategy.
    - **HFF (Hybrid First-Fit)**: Two-phase approach — FFDH first performs strip packing to form horizontal layers, then FFD packs strips into bins. $O(n\log n)$ complexity.

### Experimental Setup
- Dataset: 20 problem instances, each with 50 randomly generated squares (side lengths 10–50 units), bin size 200×100.
- Model: GPT-4o with BPE tokenization.
- Hardware: Intel Core i5-8250U, 8 GB RAM.
- All methods are evaluated on identical datasets.

## Key Experimental Results

### Main Results

| Method | Avg. Bins ↓ | Execution Time (s) | Space Utilization ↑ |
|--------|------------|-------------------|---------------------|
| FFF | 16.05 | 0.002446 | 0.76 |
| HFF | 16.00 | 0.024438 | 0.78 |
| **LLM (GPT-4o)** | **15.00** | 0.0103 | **0.83** |

- Bin count reduced by 6.25%; space utilization improved by 6.4% (vs. HFF).
- Execution time falls between FFF and HFF, demonstrating strong competitiveness.

### Space Utilization Distribution Analysis

| Method | Utilization (Early Bins) | Utilization (Late Bins) | Variation |
|--------|--------------------------|-------------------------|-----------|
| FFF | 87.50% | 68.00% | High (significant late-stage decline) |
| HFF | 86.83% | 63.54% | High (significant late-stage decline) |
| LLM | ~83% | ~83% | **Low (consistently uniform)** |

### Key Findings
- The LLM achieves the optimal solution within **only 2 iterations**, converging far faster than the 6-round limit, indicating strong constraint-satisfaction learning ability.
- LLM-generated heuristics maintain more uniform utilization (~83%) across bins, whereas traditional methods exhibit significant utilization drop in later bins.
- The LLM demonstrates an "optimization intuition" beyond explicit programming — generated packing strategies exhibit complex patterns not explicitly specified.
- Solution quality stabilizes after 6 evolutionary rounds, with diminishing marginal improvements.

## Highlights & Insights
- **Effectiveness of the Evolutionary Feedback Mechanism**: The island model preserves strategy diversity, preventing the LLM from converging to a single strategy. "Best-shot" feedback enables incrementally improving output quality across rounds. This LLM + evolutionary framework is transferable to other combinatorial optimization problems.
- **LLM Constraint Understanding**: GPT-4o successfully internalizes geometric and logical constraints (non-overlap, within-bounds); the 100% validity rate of generated code demonstrates a reliable level of comprehension of complex constraints.
- **Rapid Convergence under Low-Resource Conditions**: Achieving optimality in just 2 iterations suggests that the LLM's algorithmic design capability may substantially exceed the complexity demands of this task.

## Limitations & Future Work
- **Limited Scale**: Evaluation is conducted only on medium-scale instances (50 squares, 200×100 bins); performance on industrial-scale problems (thousands of items, irregular shapes) remains unknown.
- **LLM Non-Determinism**: Non-deterministic outputs affect reproducibility, necessitating multiple runs for statistical analysis.
- **API Cost Constraints**: Limits the number of iterations and candidate scripts; a larger search space may yield superior solutions.
- **Square Items Only**: Practical 2D-BPP involves rectangular and even irregular shapes; constraints such as rotation support remain unexplored.
- **Single Model Evaluation**: Only GPT-4o is evaluated; comparisons across different LLMs (Claude, Gemini, etc.) are absent.

## Related Work & Insights
- **vs. FunSearch [Romera-Paredes et al., 2024]**: FunSearch uses LLM + evaluator to discover novel heuristics (e.g., for online bin packing) but does not systematically evaluate LLM capabilities. This paper establishes a more complete evaluation methodology.
- **vs. Traditional Metaheuristics (SA, GA) [Ferreira, 2017]**: Traditional methods require hand-crafted search operators. LLMs directly generate complete algorithm code, reducing manual involvement.
- **vs. Deep Reinforcement Learning [Lee, 2025]**: DRL requires extensive training and problem-specific state/action design. The LLM-based approach requires zero training and is immediately deployable.
- **Relationship to Prompt Engineering**: The success of this approach is highly dependent on prompt design quality; systematic prompt optimization represents an important direction for future work.

## Rating
- Novelty: ⭐⭐⭐ The methodological framework is adapted from FunSearch; the primary contribution lies in the evaluation methodology.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation metrics are comprehensive, but problem scale is limited and broader comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly with a complete methodological presentation.
- Value: ⭐⭐⭐ Provides a preliminary methodological reference for evaluating LLMs in combinatorial optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)

</div>

<!-- RELATED:END -->
