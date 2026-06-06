---
title: >-
  [Paper Note] AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions
description: >-
  [ICML2026][symbolic regression] AutoNumerics-Zero is proposed as a zero-prior evolutionary symbolic regression method that discovers arithmetic programs for approximating transcendental functions (e.g., exponential…
tags:
  - "ICML2026"
  - "symbolic regression"
  - "evolutionary search"
  - "transcendental function approximation"
  - "program discovery"
  - "numerical computation"
date: 2026-05-08
content_hash: fe080150e669c8f4
---

# AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions

**Conference**: ICML2026  
**arXiv**: [2312.08472](https://arxiv.org/abs/2312.08472)  
**Code**: https://github.com/google-deepmind/autonumerics_zero  
**Area**: others  
**Keywords**: symbolic regression, evolutionary search, transcendental function approximation, program discovery, numerical computation

## TL;DR
AutoNumerics-Zero is proposed as a zero-prior evolutionary symbolic regression method that discovers arithmetic programs for approximating transcendental functions (e.g., exponential, cosine) from scratch. It surpasses classical approximation methods designed by mathematicians over centuries by achieving higher accuracy with fewer operations under finite precision targets.

## Background & Motivation

**Background**: Transcendental functions (exponential, logarithmic, trigonometric, etc.) are cornerstones of scientific computing. However, digital hardware cannot compute them natively and must approximate them through finite combinations of basic operations $\{+,-,\times,\div\}$. Classical methods include Taylor expansion, Chebyshev approximation, Padé approximation, and the Remez minimax algorithm, which are highly mature after centuries of development.

**Limitations of Prior Work**: Classical mathematical approximation methods pursue "arbitrary precision"—where accuracy can be infinitely improved by adding terms. However, finite-precision data types like float32 used in modern computers suffice for most applications. Accuracy exceeding the data type's precision limit is effectively wasted. Furthermore, classical methods are restricted to specific functional forms (e.g., polynomials, rational functions) and cannot explore all possible combinations of operations.

**Key Challenge**: There is a mismatch between the design objective of classical methods (arbitrary precision) and actual requirements (finite precision). If the optimization goal shifts from "infinite precision" to "sufficiently high but finite precision," can approximation schemes with fewer operations be discovered?

**Goal**: To explore whether more efficient transcendental function approximation programs can be automatically discovered via large-scale evolutionary search under finite precision targets compared to classical methods.

**Key Insight**: Symbolic regression can search through arbitrary combinations of operations (unconstrained by fixed forms like polynomials or rational functions) and optimize non-differentiable objectives (such as the number of operations). Representing functions as programs rather than formulas allows for the reuse of intermediate values, further reducing computational cost.

**Core Idea**: Large-scale evolutionary symbolic regression is used to search for efficient finite-precision approximation programs for transcendental functions within the $\{+,-,\times,\div\}$ operation space starting from empty programs. This "zero mathematical prior" approach aims to surpass centuries of human design.

## Method

### Overall Architecture
AutoNumerics-Zero is a multi-objective evolutionary search system. The inputs are precise values of the target transcendental function $f(x)$ (e.g., $2^x$) on a specified interval as training data. The output is a set of approximation programs, each achieving the highest possible accuracy with the minimum number of basic operations. The search process starts from an empty program (identity function) and improves the population through nested two-layer optimization: outer genetic programming discovers program structures, while inner CMA-ES optimizes floating-point coefficients. After the search, the best programs are verified with rigorous error bounds using interval arithmetic.

### Key Designs

1. **Distributed multi-objective population selection (dNSGA-II)**:

    - **Function**: Efficiently maintains and filters program populations on large-scale distributed clusters.
    - **Mechanism**: A distributed variant based on the classical NSGA-II algorithm. Each worker receives $2S$ programs from random other workers and selects $S$ parent programs near the Pareto front (accuracy vs. operation count) via a SelectNearPareto operation. Each parent is mutated twice to generate $2S$ offspring, which are sent to other workers. Accuracy is measured as $a = -\log_{10}(E)$ (where $E$ is the maximum error). This decentralized design avoids centralized population bottlenecks and supports asynchronous updates.
    - **Design Motivation**: Classical NSGA-II requires centralized population management and is unsuitable for large-scale parallel search. dNSGA-II enables distributed population evolution through random inter-worker communication while maintaining a uniform distribution of programs on the Pareto front.

2. **Mutation Strategy**:

    - **Function**: Explores the program structure space to generate new candidate programs.
    - **Mechanism**: Programs are represented as computational graphs (directed graphs), where input nodes are program inputs and coefficients, intermediate nodes are $\{+,-,\times,\div\}$ operations, and output nodes are program results. Mutation randomly executes one of three operations: (1) inserting a random operation node at a random position; (2) deleting a random node; (3) randomly reconnecting a randomly selected edge. All choices are random without introducing any mathematical prior knowledge.
    - **Design Motivation**: This "zero-knowledge" mutation ensures the search process is not biased toward any known approximation forms (e.g., polynomial forms), allowing evolution to discover entirely new, unconventional functional expressions, such as a nested division-addition combination raised to the 8th power.

3. **CMA-ES coefficient optimization**:

    - **Function**: Optimizes floating-point coefficients for a fixed program structure to maximize accuracy.
    - **Mechanism**: For each mutated offspring program, CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is used to optimize its coefficients. CMA-ES is a gradient-free continuous optimization algorithm that performs well with small parameter counts. The optimization goal is to minimize the maximum error on a batch of randomly sampled input-output examples.
    - **Design Motivation**: Mutation only changes the program structure; coefficients require fine-tuning. The gradient-free nature of CMA-ES avoids local optima traps and aligns with the black-box nature of the entire system—meaning non-differentiable operations (like bit-shift instructions) can be added in the future without modifying the optimizer.

### Error Bound Proof
Once the search is complete, interval arithmetic (using the IBEX library) is used on the best programs to iteratively subdivide the domain and prove error upper bounds for each sub-interval. The global bound is the maximum of all sub-intervals. For hardware-aware float32 programs, the Gappa prover is used to handle intermediate rounding errors. This ensures the discovered programs have rigorous mathematical guarantees.

## Key Experimental Results

### Main Results: Exponential Function Approximation

| Method | Operations | Precision (Decimal Digits) | Gap to Baseline |
|------|---------|-------------------|-----------|
| AutoNumerics-Zero (Best) | 10 | 14.3 | Surpasses baseline by 6+ orders of magnitude |
| Ratio/Minimax | 10 | ~8 | Best baseline |
| Ratio/Padé | 10 | ~7 | — |
| Poly/Taylor (Horner) | 10 | ~6 | — |
| Chebyshev | 10 | ~7 | — |
| Poly/Minimax | 10 | ~8 | — |

The discovered 10-operation program $f(x) = \left(\frac{c_4}{\frac{c_1}{\frac{c_3}{x}+x}+c_2+\frac{c_3}{x}+x}-c_5\right)^8$ approximates $2^x$ with a guaranteed 14 decimal digits across all real numbers. The maximum relative error is rigorously proven via interval arithmetic to be below $5.4\times 10^{-15}$.

### Hardware-aware Exponential and Other Functions

| Target Function | Scenario | AutoNumerics-Zero | Best Baseline | Gain |
|---------|------|-------------------|---------|------|
| $2^x$ (Hardware-aware, Skylake) | float32 Throughput | 3x+ Faster | Poly/Minimax | Error <1 ULP, migrates across 6 generations of Intel/AMD |
| $\cos(x)$ | Absolute Error | Higher Precision | Chebyshev | Better precision for same ops |
| Airy $\text{Ai}(-7x)$ | Oscillatory Function | 19 ops, 4.2 Precision | 20 ops Baseline | Error reduced by 2 orders of magnitude |
| Bessel $I_{1/2}(x)$ | Contains $\sqrt{\cdot}$ | 8 ops, 8.1 Precision | 9 ops, 7.8 Precision | 1 fewer op and more accurate |
| $\text{erf}(x)$ | $(0,2]$ | Better in low precision | Padé (Better in high precision) | Padé dominates at high precision |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Complete Method | Best Pareto front | dNSGA-II + Mutation + CMA-ES |
| Without dNSGA-II (Random Search) | Significant degradation | Lack of population selection pressure leads to poor efficiency |
| Without CMA-ES | Significant degradation | Unoptimized coefficients limit precision |
| Random Graph instead of Mutation | Significant degradation | Equivalent to purely random search |

### Key Findings
- In exponential function approximation, evolved programs surpass all classical baselines (including Taylor, Padé, Chebyshev, and Minimax) at all operation count levels, with advantages confirmed by mathematical proof.
- Hardware-aware search discovered code that triggers compiler anomalies but results in beneficial execution paths; manually writing such code is nearly impossible.
- Evolved programs maintain at least 80% speedup across Intel and AMD architectures spanning 8 years, demonstrating excellent hardware portability.
- Convergent evolution was observed across different search experiments: optimal programs exhibit similar structural characteristics, analogous to the independent evolution of fins in nature.

## Highlights & Insights
- **Zero-knowledge discovery surpassing human design**: The entire search process uses no mathematical priors (no asymptotic expansions, no derivatives, no pre-training), yet discovers approximation methods superior to those accumulated by mathematicians over centuries. This proves the powerful exploration capability of evolutionary search for structured mathematical problems.
- **Program representation superior to formula representation**: By representing functions as programs (computational graphs) rather than mathematical formulas, intermediate values can be reused to achieve higher precision with fewer operations. The discovered optimal program forms are neither polynomials nor continued fractions but entirely new nested structures.
- **Extensibility of black-box optimization**: Since both dNSGA-II and CMA-ES are black-box methods, the framework can easily extend to new operation sets (e.g., $\sqrt{\cdot}$), new optimization objectives (e.g., hardware execution speed), and new target functions, which is unachievable with gradient-based methods.

## Limitations & Future Work
- **Dependency on range reduction**: The search process is zero-knowledge within bounded intervals, but extension to the full real line still relies on standard range reduction techniques, a limitation shared by all baseline methods.
- **High computational cost for search**: Requires 100-10k processes running for 1-4 days. However, this is a one-time cost, amortized over all future uses of the discovered programs.
- **Not dominant for all functions**: For the erf function in high-precision ranges, the method is inferior to Padé approximation, indicating it is not universally superior for every target function.
- **Interpretability-quality tradeoff**: The discovered program forms are unconventional and lack intuitive mathematical interpretations like Taylor expansions. Post-hoc analysis could be attempted to extract structural patterns.
- Future directions include incorporating range reduction into the search process, adding non-floating-point instructions like bit manipulation, and exploring backward stability optimization.

## Related Work & Insights
- **AlphaTensor** (Fawzi et al., 2022) and **AlphaDev** (Mankowitz et al., 2023) use RL to discover optimal algorithms for matrix multiplication and sorting, respectively, but they generalize analytically after searching small-scale problems; AutoNumerics-Zero searches directly at practical scales.
- Unlike traditional symbolic regression work, this paper discovers previously unknown mathematical relationships verified by mathematical proof rather than recovering known formulas.
- Complementary to superoptimization: while superoptimization improves existing correct programs, AutoNumerics-Zero discovers entirely new programs from scratch.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](../../NeurIPS2025/others/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)
- [\[AAAI 2026\] Agent-SAMA: State-Aware Mobile Assistant](../../AAAI2026/others/agent-sama_state-aware_mobile_assistant.md)
- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/others/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[AAAI 2026\] Automated Reproducibility Has a Problem Statement Problem](../../AAAI2026/others/automated_reproducibility_has_a_problem_statement_problem.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](../../ICLR2026/others/bayesian_influence_functions_for_hessian-free_data_attribution.md)

</div>

<!-- RELATED:END -->
