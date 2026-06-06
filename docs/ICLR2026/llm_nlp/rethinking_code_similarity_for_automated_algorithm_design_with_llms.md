---
title: >-
  [Paper Note] Rethinking Code Similarity for Automated Algorithm Design with LLMs
description: >-
  [ICLR 2026][LLM/NLP][LLM-AAD] This paper proposes BehaveSim, an algorithm similarity metric based on Problem-Solving Trajectories (PSTrajs) and Dynamic Time Warping (DTW). BehaveSim measures algorithmic differences at th…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "LLM-AAD"
  - "code similarity"
  - "behavioral similarity"
  - "dynamic time warping"
  - "FunSearch"
  - "EoH"
  - "algorithm diversity"
date: 2026-05-08
content_hash: d08b9d5ae4f5d035
---

# Rethinking Code Similarity for Automated Algorithm Design with LLMs

**Conference**: ICLR 2026
**arXiv**: [2603.02787](https://arxiv.org/abs/2603.02787)  
**Code**: [https://github.com/RayZhhh/behavesim](https://github.com/RayZhhh/behavesim)  
**Area**: LLM / Automated Algorithm Design / Code Similarity
**Keywords**: LLM-AAD, code similarity, behavioral similarity, dynamic time warping, FunSearch, EoH, algorithm diversity

## TL;DR
This paper proposes BehaveSim, an algorithm similarity metric based on Problem-Solving Trajectories (PSTrajs) and Dynamic Time Warping (DTW). BehaveSim measures algorithmic differences at the level of execution behavior rather than syntax or output, and when integrated into LLM-AAD frameworks such as FunSearch and EoH, yields significant performance improvements.

## Background & Motivation
**Background**: LLM-driven automated algorithm design (LLM-AAD) has emerged as a new paradigm for algorithm development. Frameworks such as FunSearch and EoH can autonomously generate code implementations of expert-level algorithms, achieving remarkable results on classical problems including Online Bin Packing, Cap Set, and TSP.

**Limitations of Prior Work**: In LLM-AAD, the core design principles of algorithms are implicit in the generated code rather than expressed as explicit mathematical formulations or pseudocode. Existing code similarity metrics—such as syntax-tree edit distance, BLEU, and cosine similarity of code embeddings—capture only surface-level syntactic differences and cannot determine whether two code snippets implement fundamentally distinct algorithmic logic.

**Key Challenge**: Two syntactically dissimilar code snippets may realize the same algorithmic idea (differing only in variable names or loop structure), while syntactically similar code may embody entirely different solving strategies. Existing metrics cannot distinguish genuine algorithmic innovation from superficial code variation.

**Key Gap**: LLM-AAD frameworks rely on similarity measures for deduplication and diversity management in population maintenance. Inaccurate similarity metrics cause frameworks to retain redundant pseudo-innovations while crowding out genuinely diverse algorithms, reducing search efficiency.

**Limitations of Output Equivalence**: Another class of approaches compares the final outputs of algorithms (e.g., objective function values), but different algorithms may coincidentally produce identical outputs on the same inputs while behaving very differently on others. Output equivalence cannot reveal differences in the solving process.

**Core Idea**: By recording the sequence of intermediate solutions produced during algorithm execution (problem-solving trajectories), BehaveSim applies DTW to align these trajectories and measures behavioral similarity between algorithms—focusing on *how* a problem is solved rather than *what* solution is produced.

## Method

### Overall Architecture
The core idea of BehaveSim is to evaluate algorithms not by their code text or final outputs, but by their *behavior* during execution—specifically, the trajectory of intermediate solutions generated step by step. The more similar two algorithms' behaviors are (i.e., the better their trajectories align), the more similar their underlying solving strategies are considered to be.

### Key Designs

1. **Problem-Solving Trajectory (PSTraj) Extraction**:

    - **Function**: During algorithm execution, the state of each intermediate solution is recorded at each time step (e.g., packing decisions at each step in bin packing, the next city selected at each step in TSP).
    - **Formalization**: For algorithm $A$ executing on problem instance $p$, the trajectory is recorded as $\tau_A(p) = (s_1, s_2, \ldots, s_T)$, where $s_t$ denotes the intermediate solution state at time step $t$.
    - **Design Motivation**: Trajectory granularity is problem-dependent—placement decisions per item for bin packing, incremental path construction for TSP.
    - **Distinction from Traditional Profiling**: Program profiling focuses on resource consumption (CPU, memory), whereas PSTrajs track the evolution of solutions, capturing the algorithm's "reasoning process."

2. **DTW-Based Trajectory Alignment**:

    - **Why DTW**: Different algorithms may execute for different numbers of steps (e.g., fewer steps for greedy algorithms, more for backtracking), necessitating elastic alignment. Dynamic Time Warping (DTW) permits nonlinear time-axis stretching to find the optimal alignment between two trajectories.
    - **Similarity Computation**: $\text{BehaveSim}(A_1, A_2) = \frac{1}{|P|} \sum_{p \in P} \text{DTW}(\tau_{A_1}(p), \tau_{A_2}(p))$, averaged over multiple problem instances.
    - **Distance Measure**: The per-state distance function $d(s_i, s_j)$ is defined according to the specific problem (e.g., Jaccard distance for bin packing configurations, edit distance for paths).

3. **Integration into LLM-AAD Frameworks**:

    - **FunSearch Integration**: BehaveSim replaces the original deduplication and diversity maintenance mechanism in FunSearch's population management, ensuring that algorithms in the population are genuinely diverse at the behavioral level.
    - **EoH Integration**: Within the Evolution of Heuristics framework, BehaveSim guides crossover and mutation operations by prioritizing individuals with greater behavioral divergence.
    - **Effect**: The introduction of behavioral diversity enables the framework to explore a broader solution space and avoid premature convergence to local optima.
    - **Plug-and-Play**: BehaveSim functions as a modular component requiring no modifications to the core framework logic—only the similarity measurement interface needs to be replaced.

4. **Algorithm Clustering and Analysis**:

    - **Function**: BehaveSim constructs a pairwise distance matrix among algorithms, which is used with hierarchical clustering to group AI-generated algorithms by behavioral pattern.
    - **Value**: This enables researchers to systematically analyze the types of solving strategies generated by LLMs and to identify genuinely novel algorithmic designs.
    - **Visualization**: Clustering results can be rendered as dendrograms or heatmaps, providing an intuitive view of the diversity distribution across algorithm families.

## Key Experimental Results

### Main Results: AAD Performance Gains After BehaveSim Integration

| Task | Framework | Baseline Performance | +BehaveSim | Gain | Notes |
|------|-----------|---------------------|------------|------|-------|
| Online Bin Packing | FunSearch | baseline score | significant improvement | diversity-driven | classical NP-hard problem |
| Cap Set | FunSearch | baseline score | significant improvement | more strategies discovered | mathematical combinatorial optimization |
| TSP (Travelling Salesman) | EoH | baseline heuristic | significant improvement | strategy redundancy avoided | path optimization |

### BehaveSim vs. Existing Code Similarity Metrics

| Similarity Metric | Distinguishes Syntactic Variants? | Distinguishes Algorithmic Logic? | AAD Improvement? | Notes |
|-------------------|----------------------------------|----------------------------------|------------------|-------|
| Syntax-tree edit distance | ✓ | ✗ | Limited | Code structure only |
| Code embeddings (CodeBERT, etc.) | Partial | ✗ | Limited | Representation-level semantics |
| Output equivalence | N/A | Partial | Limited | Ignores solving process |
| **BehaveSim (Ours)** | ✓ | **✓** | **Significant** | Captures execution behavior |

### Key Findings
- BehaveSim effectively distinguishes syntactically similar code with different algorithmic logic, and identifies syntactically dissimilar but behaviorally equivalent "pseudo-innovations."
- Across all three AAD tasks, integrating BehaveSim yields significant performance gains, demonstrating that behavioral diversity is a key factor in improving LLM-AAD effectiveness.
- Clustering analysis reveals that LLM-generated algorithms can be grouped into several distinct strategy families based on behavioral patterns, facilitating a deeper understanding of AI "algorithmic design thinking."

## Highlights & Insights
- **Paradigm Shift from "Reading Code" to "Observing Behavior"**: BehaveSim advances an elegant insight—measuring algorithm similarity should focus on what an algorithm *does* rather than what it *says*. This perspective has broad implications for code comprehension, software engineering, and program synthesis.
- **Elegant Application of DTW**: Importing a classical tool from time-series analysis into the code similarity domain, elastic time alignment resolves the inconsistency in execution step counts across different algorithms.
- **Analyzability of LLM-Generated Algorithms**: Through behavioral clustering, BehaveSim provides the first systematic tool for analyzing the strategic lineage of AI-generated algorithms, offering important insights into the capabilities and limitations of LLM-based code generation.
- **Infrastructure Contribution to the AAD Ecosystem**: As the volume of LLM-generated algorithms grows explosively, BehaveSim provides a much-needed "algorithm quality inspection" tool for filtering genuine innovations from redundant variants.

## Limitations & Future Work
- **Problem-Specific Trajectory Definitions**: Extracting PSTrajs requires defining the format of "intermediate solution states" for each problem. Generalizing to arbitrary programming tasks requires additional engineering effort. For problems without a clear notion of intermediate solutions (e.g., classifier design), defining trajectories is itself an open problem.
- **Computational Overhead**: DTW has time complexity $O(T_1 \times T_2)$; for long trajectories or large populations, pairwise similarity computation may become a bottleneck.
- **Choice of Trajectory Granularity**: Recording at too coarse a granularity loses behavioral distinctions; too fine a granularity introduces noise. Currently, granularity selection relies on domain knowledge.
- **Evaluation Limited to Three Tasks**: Although representative combinatorial optimization problems are covered, applicability to a broader range of AAD tasks—such as ML hyperparameter optimization, program synthesis, and constraint satisfaction—remains to be verified.
- **Handling Stochastic Algorithms**: For algorithms with randomness (e.g., randomized greedy, simulated annealing), trajectories from multiple runs of the same algorithm may vary considerably, requiring repeated sampling and statistical aggregation to stabilize the metric.
- **Extreme Trajectory Length Discrepancies**: When two algorithms differ drastically in execution steps (e.g., $O(n)$ vs. $O(n^2)$), DTW may produce unreliable alignments, necessitating additional normalization strategies.
- **Multi-Objective Scenarios**: When algorithms simultaneously optimize multiple objectives (e.g., latency and throughput), a single trajectory may insufficiently characterize behavioral differences, requiring joint alignment of multi-dimensional trajectories.

## Related Work & Insights
- **vs. FunSearch (Romera-Paredes et al., 2024)**: FunSearch is Google DeepMind's LLM-AAD framework that evolves algorithms via LLMs. BehaveSim is integrated into FunSearch as a plug-and-play diversity module, enhancing its population diversity management.
- **vs. EoH (Liu et al., 2024)**: EoH is another LLM-AAD framework that designs heuristics through evolutionary algorithms. BehaveSim replaces EoH's original syntax-based deduplication mechanism.
- **vs. Code Clone Detection Literature**: Traditional code clone detection asks "are these two pieces of code doing the same thing?" BehaveSim asks "are these two pieces of code doing it the same *way*?"—a finer-grained question that is more valuable for AAD.
- **vs. Program Equivalence Verification**: Program equivalence determines whether two programs always produce identical outputs (an undecidable problem); BehaveSim instead measures the *degree* of behavioral similarity, offering a more practical approximation.
- **vs. Algorithm Selection / AutoML**: Algorithm selection focuses on "which algorithm best suits a given problem," while BehaveSim focuses on "how similar are these algorithms"—the latter provides a more fine-grained structured representation of the algorithm space for the former.
- **vs. Neural Program Embedding**: Methods such as Code2Vec embed programs into vector spaces but capture static semantics. BehaveSim's dynamic behavioral perspective is complementary—combining the two could yield a more comprehensive algorithm representation.

## Rating
- Novelty: ⭐⭐⭐⭐ Defining code similarity from a behavioral perspective is a highly creative starting point
- Experimental Thoroughness: ⭐⭐⭐⭐ Three AAD tasks + multiple similarity baselines + clustering analysis
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and methodology is described intuitively
- Value: ⭐⭐⭐⭐ Practically valuable for the LLM-AAD ecosystem; the DTW + trajectory approach is transferable to other problems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](../../ACL2026/llm_nlp/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[ICML 2026\] Rethinking LLM Ensembling from the Perspective of Mixture Models](../../ICML2026/llm_nlp/rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](../../NeurIPS2025/llm_nlp/adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)
- [\[ICLR 2026\] WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)

</div>

<!-- RELATED:END -->
