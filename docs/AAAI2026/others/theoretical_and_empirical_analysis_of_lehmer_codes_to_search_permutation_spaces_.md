---
title: >-
  [Paper Note] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms
description: >-
  [AAAI 2026][Lehmer Code] This work presents the first rigorous mathematical runtime analysis of Lehmer codes (inversion tables) for searching permutation spaces with evolutionary algorithms. It proves that Lehmer-code-based EAs achieve expected runtimes of $O(n^2 \log n)$ or $O(n^2)$ on most benchmark functions, matching or improving upon classical representations, and validates practical utility on LOP and QAP instances.
tags:
  - "AAAI 2026"
  - "Lehmer Code"
  - "Permutation Space"
  - "Evolutionary Algorithms"
  - "Runtime Analysis"
  - "Combinatorial Optimization"
date: 2026-05-08
content_hash: 0f442a856b2863ee
---

# Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms

**Conference**: AAAI 2026
**arXiv**: [2511.19089](https://arxiv.org/abs/2511.19089)  
**Code**: [https://github.com/TrendMYX/LehmerEA](https://github.com/TrendMYX/LehmerEA)  
**Area**: Evolutionary Algorithms / Combinatorial Optimization
**Keywords**: Lehmer Code, Permutation Space, Evolutionary Algorithms, Runtime Analysis, Combinatorial Optimization

## TL;DR

This work presents the first rigorous mathematical runtime analysis of Lehmer codes (inversion tables) for searching permutation spaces with evolutionary algorithms. It proves that Lehmer-code-based EAs achieve expected runtimes of $O(n^2 \log n)$ or $O(n^2)$ on most benchmark functions, matching or improving upon classical representations, and validates practical utility on LOP and QAP instances.

## Background & Motivation

**Background**: Permutation problems (e.g., shortest path, linear ordering, quadratic assignment) are central to combinatorial optimization, with search spaces growing factorially. Evolutionary algorithms (EAs) are widely used metaheuristics for such NP-hard problems. The classical linear encoding of permutations (a vector of $n$ distinct elements) is the most common representation in EAs, but requires specialized constraint handling and mutation/crossover operators to maintain the mutual exclusivity property.

**Limitations of Prior Work**: The mutual exclusivity constraint of classical encodings precludes direct application of standard mutation operators (e.g., single-point mutation), necessitating tailored neighborhood structures such as transpositions, adjacent swaps, and insertions. Random key representations circumvent mutual exclusivity but suffer from severe redundancy (many-to-one mappings). Existing theoretical runtime analyses almost exclusively target classical permutation representations or binary search spaces, leaving the theoretical properties of alternative representations largely unexplored.

**Key Challenge**: The constrained nature of permutation spaces limits the applicability of standard EA tools, while Lehmer codes—though naturally unconstrained and bijective with permutations—lack theoretical analysis to guide their selection and use within EAs.

**Goal**: (1) Establish benchmark functions and simple EA algorithms over the Lehmer code space; (2) derive runtime bounds for these algorithms on the benchmarks; (3) compare results against known bounds for classical permutation representations; (4) empirically validate practical utility on real-world problems.

**Key Insight**: The paper employs classical tools from theoretical runtime analysis—including drift analysis and the coupon collector argument—to derive tight bounds, while establishing structural correspondences between operations in Lehmer space and permutation space.

**Core Idea**: Lehmer codes provide an unconstrained, bijective representation for permutation-based EAs whose asymptotic runtime on standard benchmark functions is no worse than, and in some cases superior to, that of classical representations.

## Method

### Overall Architecture

The Lehmer encoding maps a permutation $\sigma \in S_n$ to a vector $L(\sigma) = (L(\sigma)_n, \ldots, L(\sigma)_1)$, where $L(\sigma)_{n-i+1} = \#\{j > i \mid \sigma(j) < \sigma(i)\}$ denotes the number of elements after position $i$ that are smaller than $\sigma(i)$. The Lehmer space $L_n = [n] \times [n-1] \times \cdots \times [1]$ assigns position $i$ a domain of $[0, i-1]$, naturally satisfying domain constraints without mutual exclusivity handling. The authors define simple EAs (RLS and (1+1)-EA) over this space along with three families of benchmark functions, and derive corresponding runtime bounds.

### Key Designs

1. **Step Operators and Probability Vectors**:

    - Function: Define mutation operations over the Lehmer space.
    - Mechanism: Two step operators are considered—(a) uniform step: uniformly sample a new value from $[0, i-1] \setminus \{x_i\}$ at position $i$; (b) $\pm 1$ step: increment or decrement $x_i$ by 1 with probability $1/2$ each (with boundary truncation). Two position selection distributions are considered—uniform probability (each position selected with probability $1/(n-1)$) and proportional probability ($p_i = 2(i-1)/(n(n-1))$, proportional to domain size). RLS modifies one position per step; (1+1)-EA mutates each position independently with probability $1/(n-1)$.
    - Design Motivation: Since domain sizes vary across dimensions in Lehmer space (position $i$ has $i$ possible values), it is natural to investigate whether mutation probabilities should be allocated proportionally. The choice between uniform and $\pm 1$ steps reflects a trade-off between global and local search.

2. **Benchmark Functions and Equivalence Relations**:

    - Function: Establish a theoretical analysis framework over Lehmer space and relate it to classical permutation space.
    - Mechanism: Three function families are defined—(a) $\mathcal{L}$-OneMax (sum of position values) $\leftrightarrow$ INV (number of inversions), fully equivalent via the Lehmer bijection; (b) $\mathcal{L}$-LeadingZeros (number of consecutive leading zeros from the high-order position) $\leftrightarrow$ PLeadingOnes (number of prefix fixed points), likewise equivalent; (c) FacVal (factorially weighted value) $\leftrightarrow$ LexVal (lexicographic rank), also equivalent. Lemma 2 establishes the precise mapping of adjacent swaps in permutation space to operations in Lehmer space.
    - Design Motivation: By establishing these equivalences, runtime results in Lehmer space can be directly compared against known results in classical permutation space, making the comparison meaningful.

3. **Core Runtime Theorems**:

    - Function: Provide precise or asymptotic runtime bounds for each algorithm–benchmark combination.
    - Mechanism: Key analytical tools include the variable drift theorem, the multiplicative drift theorem, and the non-uniform coupon collector. Principal results include: RLS with uniform step on $\mathcal{L}$-OneMax achieves expected runtime $(n-1)^2 \ln n + \Theta(n^2)$ (Theorem 1); RLS with $\pm 1$ step on $\mathcal{L}$-OneMax achieves $\Theta(n^2)$ (Theorem 5); (1+1)-EA with uniform step on $\mathcal{L}$-OneMax achieves $\Theta(n^2 \log n)$, and for the equivalent multi-valued NVal problem improves the known upper bound from $O(n^4 \log \log n)$ to $\Theta(n^2 \log n)$ (Theorem 10)—an improvement by a factor of nearly $\Theta(n^2)$.
    - Design Motivation: Tight runtime bounds enable fine-grained comparison of different representations and operators, thereby informing algorithm design.

### Structural Correspondences

A key finding is that an adjacent swap in permutation space corresponds to swapping two position values in the Lehmer encoding plus a $\pm 1$ adjustment (Lemma 2), and that $\sigma_i > \sigma_{i+1}$ is equivalent to $L(\sigma)_{n-i+1} > L(\sigma)_{n-i}$ (Lemma 3). These structural relationships establish a clear correspondence between operations in classical permutation space and those in Lehmer space.

## Key Experimental Results

### Main Results (Theoretical Benchmarks, $n = 50 \sim 350$)

| Benchmark | Best Lehmer Algorithm | Best Classical Algorithm | Comparison |
|---|---|---|---|
| $\mathcal{L}$-OneMax / INV | Lehmer-Harmonic | Perm-AdjSwap | Lehmer slightly better |
| $\mathcal{L}$-LeadingZeros / PLeadingOnes | Lehmer-Harmonic | Perm-Trans | Lehmer competitive |
| FacVal / LexVal | Lehmer-Harmonic | Perm-Jump | Lehmer significantly better |

### LOP / QAP Real-World Problems ($n = 10$, success rate and empirical runtime)

| Problem | Lehmer-Harmonic | Lehmer-Uniform | Perm-Jump | Perm-Trans |
|---|---|---|---|---|
| LOP success rate | ~85% | ~80% | ~90% | ~75% |
| QAP success rate | ~60% | ~55% | ~73% | ~65% |

### Key Findings

- On all theoretical benchmarks, at least one Lehmer algorithm outperforms all classical algorithms.
- Lehmer-Harmonic (harmonic mutation strength) performs well across all benchmarks and is the most robust choice.
- On real-world LOP/QAP instances, Lehmer-Harmonic and Lehmer-Uniform perform comparably to classical algorithms but do not uniformly dominate them.
- The $\pm 1$ step operator is $\Theta(n)$ times slower than classical methods on $\mathcal{L}$-LeadingZeros due to random walk behavior, but the uniform step operator corrects this issue.

## Highlights & Insights

- This work provides the first rigorous theoretical runtime analysis of Lehmer codes within EAs, filling an important gap in the runtime analysis of permutation spaces.
- The known upper bound for the NVal problem is improved by a factor of nearly $\Theta(n^2)$, representing an independent theoretical contribution.
- The primary advantage of Lehmer codes lies in eliminating the need for constraint handling, allowing standard EA operations to be applied directly.
- Harmonic mutation, as a compromise between uniform and unit-step mutation, achieves the best performance in both theory and experiment.

## Limitations & Future Work

- Theoretical analysis is restricted to the simplest EAs (RLS, (1+1)-EA), with population-based algorithms and adaptive mutation left unaddressed.
- On NP-hard real-world problems, Lehmer-EA does not yet match the performance of carefully engineered operators for classical representations.
- Crossover operator design for Lehmer codes is not considered; only mutation is analyzed.
- The effects of adaptive mutation strength and heavy-tailed mutation on Lehmer codes remain to be explored.

## Related Work & Insights

- **vs. Scharnow et al.**: A pioneering work in permutation space runtime analysis, establishing $O(n^2 \log n)$ bounds for a combined jump+transposition operator on INV; the $\pm 1$ step in Lehmer space directly yields the asymptotically tight bound of $\Theta(n^2)$.
- **vs. Doerr 2023**: Analyzes transposition on PLeadingOnes with a $\Theta(n^3)$ bound; the present work obtains the same polynomial degree on $\mathcal{L}$-LeadingZeros with precise constants.
- **vs. Doerr & Pohl**: Analyzes the runtime of (1+1)-EA on multi-valued spaces $[r+1]^n$, with an upper bound of $O(n^4 \log \log n)$ for NVal; this work improves it to $\Theta(n^2 \log n)$.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to establish a runtime theory for Lehmer codes in EAs; the research direction is entirely pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Both theoretical benchmarks and real-world problems (LOP/QAP) are evaluated, though instance sizes for real-world problems are small.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear, but the paper is lengthy and proof-dense, posing a high barrier for readers outside the field.
- Value: ⭐⭐⭐⭐ — Provides important reference for the EA theory community; the practical advantages of Lehmer codes require further investigation in future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Runtime Analysis of Evolutionary NAS for Multiclass Classification](../../ICML2025/others/runtime_analysis_of_evolutionary_nas_for_multiclass_classification.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICML 2026\] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](../../ICML2026/others/theoretical_analysis_of_sparse_optimization_with_reparameterization_weight_decay.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/others/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[AAAI 2026\] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis](bayesian_network_structural_consensus_via_greedy_min-cut_analysis.md)

</div>

<!-- RELATED:END -->
