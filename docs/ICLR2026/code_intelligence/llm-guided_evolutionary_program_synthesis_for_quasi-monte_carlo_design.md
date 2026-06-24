---
title: >-
  [Paper Note] LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design
description: >-
  [ICLR 2026][Code Intelligence][Program Synthesis] Two long-standing Quasi-Monte Carlo (QMC) design problems—constructing finite point sets with low star discrepancy and selecting Sobol' direction numbers—are reformulated as "program synthesis" tasks. An LLM acts as an intelligent mutation operator within an evolutionary loop to search for generating code. Without any task-specific training, this approach reproduces known optimal solutions and sets new benchmarks for several f…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "Program Synthesis"
  - "Evolutionary Search"
  - "Low-Discrepancy Point Sets"
  - "Star Discrepancy"
  - "Sobol' Sequences"
  - "AlphaEvolve"
date: 2026-05-08
content_hash: 98c2dc1235a1661c
---

# LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=6L8fgclOTS](https://openreview.net/forum?id=6L8fgclOTS)  
**Code**: [https://github.com/hockeyguy123/openevolve-star-discrepancy](https://github.com/hockeyguy123/openevolve-star-discrepancy)  
**Area**: Code Intelligence / LLM Scientific Discovery / Quasi-Monte Carlo  
**Keywords**: Program Synthesis, Evolutionary Search, Low-Discrepancy Point Sets, Star Discrepancy, Sobol' Sequences, AlphaEvolve  

## TL;DR
Two long-standing Quasi-Monte Carlo (QMC) design problems—constructing finite point sets with low star discrepancy and selecting Sobol' direction numbers—are reformulated as "program synthesis" tasks. An LLM acts as an intelligent mutation operator within an evolutionary loop to search for generating code. Without any task-specific training, this approach reproduces known optimal solutions and sets new benchmarks for several finite-scale and high-dimensional financial pricing scenarios.

## Background & Motivation

**Background**: High-dimensional numerical integration is a cornerstone of scientific computing. Standard Monte Carlo is limited by the Central Limit Theorem to a convergence rate of $O(N^{-1/2})$. In contrast, QMC replaces pseudo-random sampling with deterministic high-uniformity point sets, lowering error bounds via the Koksma–Hlawka inequality $\left|\int_{[0,1]^d} f\,du - \frac{1}{N}\sum_i f(x_i)\right| \le V(f)\cdot D_N^*(P)$, where a smaller star discrepancy $D_N^*$ leads to higher integration accuracy.

**Limitations of Prior Work**: Constructing point sets with minimal star discrepancy is a highly complex combinatorial problem. Mathematical programming has only proven optimality for 2D with $N\le21$ and 3D with $N\le8$; larger $N$ remains largely open. For infinite sequences, the quality of Sobol' sequences depends heavily on integer parameters called "direction numbers." Joe–Kuo (2008) provided a universal benchmark through massive computational search, but it optimizes the t-value of 2D projections, which may not be optimal for specific downstream integration tasks.

**Key Challenge**: Traditional number-theoretic constructions (Halton/Sobol') guarantee asymptotic uniformity but are non-optimal at finite $N$. Optimization-based methods (mathematical programming, GNN-based MPMC) can directly optimize coordinates but solve only for a fixed $(N,d)$; they require re-computation for different scales and are difficult to randomize or extend to high dimensions.

**Goal**: To verify whether LLM-driven evolutionary program search can become a new paradigm for "discovery" in computational mathematics—reproducing classic designs at known optima while improving them where finite structures are effective.

**Core Idea**: **Treating the construction of mathematical objects as a code evolution problem.** Instead of directly optimizing coordinates, let an LLM act as a mutation operator to rewrite "Python programs that generate point sets/direction numbers." Use discrepancy or MSE as fitness feedback to drive selection, utilizing the open-source OpenEvolve (an open-source implementation of AlphaEvolve) for scheduling.

## Method

### Overall Architecture
The method is built upon OpenEvolve, formulating the search as an evolutionary process over a "population of programs capable of generating mathematical constructions." Each individual is a snippet of Python code. An LLM (Gemini 2.0 Flash, temperature 0.7, top-p 0.95, no training or fine-tuning throughout) serves as the mutation operator to rewrite core functions. A fitness function returns a scalar score. The loop follows five steps: Initialization → Evaluation → Selection & Prompting → Mutation Generation → Population Iteration. The population size is 60, split into four islands with ring migration, involving approximately 2,000 LLM calls per run.

```mermaid
flowchart LR
    A[Initial Population<br/>Classic Construction Seeds] --> B[Execute Program<br/>Get Candidate Sets/Direction Numbers]
    B --> C[Fitness Evaluation<br/>1/(1+D*) or 1/(1+MSE)]
    C --> D[Select High-Score Parents<br/>+Island/Elite Archive]
    D --> E[Construct Prompt<br/>Parent Source+Score+Inspiration Program]
    E --> F[LLM Mutation<br/>Rewrite Generation Code]
    F --> B
```

**1. Encoding construction and tuning as "executable programs"**: Two seemingly different problems are unified within the same framework. In the point set problem, the program is a `construct_star()` function returning an $N\times d$ coordinate array. In the Sobol' problem, the program returns a list of dictionaries containing Sobol' parameters $(s, a, m_i)$ for each dimension. The key is that fitness remains a "cheap, deterministic quantity calculated by a short script"—$\frac{1}{1+D_N^*}$ for point sets and $\frac{1}{1+\text{MSE}}$ for direction numbers. Thus, RL training is unnecessary; search in the program space driven by numerical signals is sufficient. Star discrepancy $D_N^*$ is calculated exactly via coordinate-induced grid scanning of extreme anchor boxes with complexity $O(N^{d/2+1})$.

**2. Two-stage strategy to balance "broad exploration" and "fine-tuning"**: Point set discovery is deliberately split into two phases. **Phase I: Direct Construction** asks the LLM to write code that "explicitly generates an $N$-point set." For 2D, it starts with a shifted Fibonacci lattice; for 3D, it starts with scrambled Sobol' sequences, encouraging the model to traverse various constructive heuristics. After sufficient iterations, it switches to **Phase II: Iterative Optimization**, prompting the LLM to rewrite code that "refines initial guesses using numerical routines like `scipy.optimize.minimize`." This shifts the search from "finding explicit formulas" to "directly optimizing coordinates." The $N=16$ case is illustrative: the initial Fibonacci lattice has a discrepancy of 0.0962. Phase I reduced this to 0.0924 via optimal shifting after 243 iterations, and Phase II further dropped it to 0.0744 using "jittered Fibonacci initialization + SLSQP with random restarts," coming within 0.68% of the known optimum (0.0739).

**3. Maintaining diversity and encouraging crossover with Island Models + Elite Archiving**: The population is divided into four islands in a ring topology (15 programs per island). Migration occurs every 25 iterations—one elite from each island is sent clockwise to replace the worst individual in the target island. Additionally, a global elite archive is maintained both for parent selection and LLM context construction. During mutation, parents are sampled from the elite archive with a 0.7 probability and from within the island with a 0.3 probability. Prompts include code from other high-scoring "inspiration programs" to encourage crossover. This structure makes evolution more stable than repeated one-shot prompting: on $N=100$ with 16 seeds, evolutionary search variance is much lower than pure prompting baselines.

**4. Control experiment: "Why the LLM is not just tuning SLSQP"**: A natural skepticism is whether Phase II relies purely on `scipy` for local coordinate optimization without LLM contribution. The authors ran SLSQP on Sobol' sets, Fibonacci lattices, and the best Phase I sets using the same budget as a baseline. While these baselines significantly improved their respective seeds, the LLM-evolved Phase II programs still achieved lower discrepancy in 22 out of 24 tested $N$. The relative reduction reached up to 15% over pure SLSQP, indicating that gains stem from non-trivial initializations and restart strategies discovered by the LLM.

## Key Experimental Results

### Main Results

**2D Star Discrepancy Comparison (Selected $N$, lower is better, bold indicates best)**

| $N$ | Fibonacci | MPMC | LLM | Clément (Proven optimal for $N\le20$) |
|----|-----------|------|-----|------|
| 16 | 0.1486 | 0.0768 | 0.0745 | **0.0739** |
| 20 | 0.1188 | 0.0640 | 0.0611 | **0.0604** |
| 40 | 0.0638 | N/A | **0.0331** | 0.0332 |
| 50 | 0.0531 | N/A | **0.0278** | 0.0280 |
| 60 | 0.0442 | 0.0273 | **0.0234** | 0.0244 |
| 100 | 0.0275 | 0.0188 | **0.0150** | 0.0193 |

For $N\le10$, the method fully reproduces known optima. For $N\ge40$, it begins to outperform the best values in previous literature, e.g., improving 0.0188 to 0.0150 at $N=100$. In 3D, it matches known optima for $N=1,2,3,5,6,7$ and sets new best benchmarks for $N>8$ beyond the proven range. In 2D, it achieves lower reported values up to $N=1020$.

**Sobol' Direction Numbers → Asian Option rQMC MSE (32 dimensions, vs. Joe–Kuo, selected cases)**

| $N$ | Sobol(JK) | LLM | p-value | Scenario |
|----|-----------|-----|---------|------|
| 8192 | 4.52e-05 | **4.10e-05** | 7.0e-06 | Training Example |
| 8192 | 1.17e-04 | **1.07e-04** | 1.0e-06 | Out-of-Money |
| 8192 | 5.66e-04 | **5.32e-04** | 0.0156 | High Volatility |
| 1024 | 0.00146 | **0.00131** | 3.5e-08 | At-the-Money |

Ours only modified parameters for dimensions 4, 5, and 6 (keeping Joe–Kuo for the others). For $N\ge512$, it significantly reduced MSE according to a one-sided Wilcoxon signed-rank test + FDR correction. Compared to LatNetBuilder digital nets, it often achieved 2–10x lower MSE across six Asian-type payoff scenarios for $N\le1024$.

### Ablation Study

| Ablation Dimension | Setting | Conclusion |
|---------|------|------|
| Initialization Source | Joe–Kuo warm start vs. Random init | Random init remains inferior to Joe–Kuo for most $N$, indicating search refines strong priors rather than solving from scratch. |
| LLM Scale | Gemini Flash vs. Flash-Lite | Both achieved similar MSE reductions over Joe–Kuo for $N\gtrsim2048$. |
| Evolution vs. Prompting | Full evolution vs. One-shot / Multi-turn | Evolution yielded lower discrepancy and much smaller variance across 16 seeds. |
| Phase II vs. Pure Opt | LLM Evolution vs. Direct SLSQP baseline | LLM was better in 22 out of 24 $N$ cases, with up to 15% further reduction. |

### Key Findings
- **Generalization**: Evolved direction numbers generalized to many high-dimensional exotic options (Lookback, Basket, Bermudan) for $N\ge512$. The sole exception is highly discontinuous Barrier options—indicating this is "problem-dependent Sobol' design" where smooth path-dependent integrals benefit while strong discontinuities may not.
- **Computational Cost**: A single Sobol' experiment involves ~2000 LLM calls and ~$1.6\times10^{10}$ 32D QMC samples and payoff evaluations, taking ~96 hours on a CPU workstation. However, this is a one-time offline cost; the resulting direction numbers are reusable across tasks.

## Highlights & Insights
- **Elegance of Unified Formulation**: Reformulating two classic number theory problems into a single "program synthesis + evolution" framework, where the only differences lie in program output and fitness, is methodologically minimal.
- **Structural Advantages over Direct Optimization**: Outputting direction numbers that define an entire Sobol' sequence, rather than fixed coordinates for a specific $(N,d)$, inherently supports arbitrary $N$ scalability, incremental integration, standard randomization (Owen scrambling), and high-dimensional applicability—capabilities structurally missing in fixed-point methods like MPMC or math programming.
- **Honest Control Design**: The inclusion of SLSQP-only baselines and random initialization ablations quantifies exactly what the LLM contributes—concluding it performs "non-trivial refinement on strong priors" without exaggerating the LLM's ability to discover everything from zero.

## Limitations & Future Work
- **Single Evolutionary Run per Problem**: Limited by compute, only one evolution run was performed per problem. While significance is quantified via paired tests, the authors acknowledge this statistical limitation.
- **No Improvement for Discontinuous Integrals**: Evolved direction numbers only help smooth, low-to-medium variation path-dependent integrals; they do not help, and may even slightly harm, Barrier-type discontinuous payoffs.
- **No New Operators Methodologically**: The contribution is empirical and conceptual—proving that general LLM program search can reproduce and improve long-studied QMC constructions nearly out-of-the-box, without introducing new evolutionary operators.
- **Future Work**: Exploring better prompting techniques, evolutionary hyperparameter optimization, and "meta-model" approaches to generate optimal parameters directly for any $d$ or $N$.

## Related Work & Insights
- **Classic Constructions & Optimization**: Number-theoretic sequences (Halton/Sobol'), mathematical programming by Clément et al. (proven optimal but limited), heuristics like GA/Threshold Accepting, and Rusch et al.'s MPMC using GNNs for coordinate optimization. The fundamental difference here is "program synthesis" versus "coordinate optimization."
- **Sobol' Parameter Optimization**: Joe–Kuo (2008) standard, improvements for graphics projection properties, and LatNetBuilder for lattice rules/digital nets.
- **LLM Scientific Discovery**: Directly inspired by AlphaEvolve (LLM + evolution for algorithm discovery, e.g., matrix multiplication, math conjectures), implemented via OpenEvolve. Discusses more engineered subsequent systems like ShinkaEvolve—though for this paper's cheap deterministic fitness, vanilla OpenEvolve sufficed.
- **Insight**: When a search problem's "solution can be written as a short program and evaluation is cheap and deterministic," letting an LLM evolve code with numerical fitness is a more efficient and transferable paradigm than training specialized models. Meanwhile, "warm starting from strong human priors" is often more stable than searching from zero.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — No innovation in the algorithm itself (uses AlphaEvolve/OpenEvolve), but the first to systematically apply LLM program synthesis to QMC design and unify two classic problems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 2D/3D point sets + six Asian scenarios + multiple exotic option generalizations + multiple ablations (SLSQP, initialization, model scale, prompting). Robust paired significance tests. Points deducted for single evolution runs.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic across background, method, and controls. Honest discussion of the LLM's contribution.
- **Value**: ⭐⭐⭐⭐ — Refreshes several finite-$N$ star discrepancy benchmarks and provides reusable, improved Sobol' direction numbers. Directly valuable to high-dimensional numerical integration practice and provides a compelling case for "LLM evolution for computational math discovery."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](../../AAAI2026/code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)
- [\[ICLR 2026\] RefineStat: Efficient Exploration for Probabilistic Program Synthesis](refinestat_efficient_exploration_for_probabilistic_program_synthesis.md)
- [\[ICLR 2026\] Gradient-Based Program Synthesis with Neurally Interpreted Languages](gradient-based_program_synthesis_with_neurally_interpreted_languages.md)
- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)

</div>

<!-- RELATED:END -->
