---
title: >-
  [Paper Note] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs
description: >-
  [AAAI 2026][Travelling Salesman Problem] This paper proposes improved FPT approximation algorithms for the Travelling Salesman Problem (TSP) on non-metric graphs…
tags:
  - "AAAI 2026"
  - "Travelling Salesman Problem"
  - "parameterized approximation algorithms"
  - "non-metric graphs"
  - "fixed-parameter tractability"
  - "triangle inequality"
date: 2026-05-08
content_hash: d66da98e7f061bc7
---

# Parameterized Approximation Algorithms for TSP on Non-Metric Graphs

**Conference**: AAAI 2026
**arXiv**: [2503.03642](https://arxiv.org/abs/2503.03642)  
**Code**: None  
**Area**: Theoretical Computer Science / Combinatorial Optimization
**Keywords**: Travelling Salesman Problem, parameterized approximation algorithms, non-metric graphs, fixed-parameter tractability, triangle inequality

## TL;DR

This paper proposes improved FPT approximation algorithms for the Travelling Salesman Problem (TSP) on non-metric graphs, parameterized by $p$ (the number of vertices involved in triangle inequality violations) and $q$ (the size of a minimum violation set), improving the approximation ratio under parameter $p$ from 2.5 to 1.5 and under parameter $q$ from 11 to 3.

## Background & Motivation

- **Background**: TSP is a classical NP-hard problem in combinatorial optimization with broad applications in logistics, manufacturing, and telecommunications. For metric graphs satisfying the triangle inequality, the Christofides–Serdyukov algorithm achieves an approximation ratio of 1.5, further improved to $1.5 - 10^{-36}$ by Karlin et al. For general (non-metric) graphs, however, TSP cannot be approximated within any computable function $f(n)$ in polynomial time.
- **Limitations of Prior Work**: A large gap in approximation performance exists between metric and general graphs, motivating the study of graphs that are "close" to metric. Two natural parameters have been introduced in the parameterized complexity literature to measure this distance: parameter $p$ (the number of vertices involved in triangle inequality violations) and parameter $q$ (the minimum number of vertices whose removal makes the graph metric). Under parameter $p$, Zhou et al. achieved a 3-approximation FPT algorithm, later improved to 2.5 by Bampis et al.; under parameter $q$, the best known FPT result by Zhou et al. was an 11-approximation.
- **Key Challenge**: Non-metric graphs lack the shortcutting property that underpins metric TSP algorithms, making direct extensions of Christofides-style methods non-trivial.
- **Goal**: To answer the open question posed by Bampis et al. on whether these approximation ratios can be further improved, and to provide tighter FPT approximation algorithms under both parameters.
- **Key Insight**: The practical motivation comes from city tour bus route planning, where most landmarks satisfy the triangle inequality but a small number of high-profile direct routes violate it—implying that $p$ and $q$ are small in practice.

## Method

### Overall Architecture

The paper proposes multiple algorithms for the two parameters. The core mechanism is to decompose the non-metric TSP instance into subproblems on "bad vertices" (those involved in triangle inequality violations) and "good vertices" (the remaining vertices), and to exploit the metric structure of good vertices via shortcutting to reduce solution cost.

**Key Property (Property 1)**: Any triangle containing only good vertices satisfies the triangle inequality. This is the foundation of all algorithm designs—shortcutting through good vertices does not increase path weight.

### Key Designs

1. **ALG.1 ($(\alpha+1)$-approximation, parameter $p$)**: The simplest algorithm. A good vertex $o$ is selected; dynamic programming is applied on $G[V_b \cup \{o\}]$ to find an optimal TSP tour $T_b$, and a metric-TSP $\alpha$-approximation algorithm is applied on $G[V_g]$ to find $T_g$. The final TSP tour is obtained by shortcutting $T_b \cup T_g$. The key observation is that $\text{OPT} \geq w(T_b^*)$ and $\text{OPT} \geq w(T_g^*)$, yielding an overall approximation ratio of $(\alpha + 1)$. The running time is $2^{O(p)} + n^{O(1)}$, simultaneously improving both the approximation ratio and running time.

2. **ALG.2 (1.5-approximation, parameter $p$)**: A more refined algorithm. The algorithm first guesses the subgraph of the optimal tour $T^*$ induced on bad vertices (i.e., "bad chains" $\mathcal{A}$), constructs a constrained spanning tree (CST) $F_{\mathcal{A}}$, and then corrects odd-degree vertices via minimum-weight matching on an auxiliary graph. The core innovations are:
    - Constructing an auxiliary graph $\widetilde{G}$ by contracting bad chains and finding a minimum spanning tree on it.
    - Constructing an auxiliary graph $G'$ to handle the matching of odd-degree vertices (since the original graph may be non-metric, minimum-weight matching cannot be applied directly).
    - Proving that $w(F_{\mathcal{A}}) \leq \text{OPT}$ and $w(\mathcal{M}_{\mathcal{A}}) \leq \frac{1}{2} \text{OPT}$.

   Via a carefully established shortcutting lemma (Lemma 3), an approximation ratio of $\frac{3}{2} \cdot \text{OPT}$ is achieved.

3. **ALG.3 ($(\alpha+\varepsilon)$-approximation, when $p$ is constant)**: Reduces TSP to metric $k$-TSPP ($k$-TSP path problem) and applies the $\Phi$-TSP algorithm of Traub et al. When $p = O(1)$, the approximation ratio nearly matches that of metric TSP. Running time is $n^{O(p/\varepsilon)}$.

4. **ALG.4 (3-approximation, parameter $q$)**: The core algorithm for parameter $q$. The key challenge is that, unlike the $p$-parameter setting, triangles containing one bad vertex and two good vertices may also violate the triangle inequality, so Property 1 does not directly apply. The algorithm consists of three subroutines:
    - **LIMB**: Uses a potential set technique to guess anchors and limb edges such that $w(\mathcal{B}') \leq w(\mathcal{B})$.
    - **CONNECT**: Connects disconnected components by guessing a partition, with complexity controlled at $2^{O(q \log q)}$.
    - **SHORTCUT**: Handles shortcutting operations on the Eulerian graph.

### Loss & Training

This is a purely theoretical work with no training involved. The core analytical techniques include:

- **Shortcutting analysis**: Property 1 is used to prove that shortcutting through good vertices does not increase weight.
- **Matching lower bound**: The TSP tour is decomposed into two matchings, proving that the minimum matching weight is at most $\frac{1}{2} \text{OPT}$.
- **Potential set technique**: In the LIMB subroutine, a candidate set of size $O(q)$ is maintained for each anchor to ensure that the guessed anchor cost does not exceed the true anchor cost.

## Key Experimental Results

### Main Results

This is a purely theoretical work with no experimental data. A complete comparison of theoretical results is provided:

| Parameter | Approx. Ratio | Running Time | Source |
|-----------|---------------|--------------|--------|
| $p$ | 3 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Zhou et al. |
| $p$ | 2.5 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Bampis et al. |
| $p$ | **$\alpha+1 \approx 2.5$** | $2^{O(p)} + n^{O(1)}$ | **Ours, ALG.1** |
| $p$ | **1.5** | $2^{O(p \log p)} \cdot n^{O(1)}$ | **Ours, ALG.2** |
| $p$ | **$\alpha + \varepsilon$** | $n^{O(p/\varepsilon)}$ | **Ours, ALG.3** |
| $q$ | 3 | $n^{O(q+1)}$ | Zhou et al. |
| $q$ | 11 | $2^{O(q \log q)} \cdot n^{O(1)}$ | Zhou et al. |
| $q$ | **3** | $2^{O(q \log q)} \cdot n^{O(1)}$ | **Ours, ALG.4** |
| $q$ | **$\alpha + \varepsilon$** | $n^{O(q/\varepsilon)}$ | **Ours** |

### Ablation Study

| Algorithm Variant | Approx. Ratio | Key Improvement |
|-------------------|---------------|-----------------|
| ALG.1 alone | $\alpha + 1$ | Simplest; optimal running time $2^{O(p)} + n^{O(1)}$ |
| ALG.2 with CST + matching | 1.5 | Eliminates the auxiliary edge set $E'$; directly constructs the constrained spanning tree |
| ALG.4 vs. prior $q$-parameter work | 3 vs. 11 | LIMB+CONNECT technique achieves 3-approximation within FPT time |

### Key Findings

- For parameter $p$, the 1.5-approximation ratio matches the best known approximation ratio for metric TSP, suggesting that the proposed algorithm is nearly optimal under this parameterization.
- When $p$ or $q$ is constant, the $(\alpha+\varepsilon)$-approximation ratio nearly matches the best known approximation ratio for metric TSP.
- ALG.4 significantly improves the FPT approximation ratio under parameter $q$ from 11 to 3 while maintaining the same FPT running time.

## Highlights & Insights

1. **Elegant divide-and-conquer design**: ALG.1 connects the bad-vertex sub-tour and good-vertex sub-tour via a single good vertex—an extremely simple construction that already improves upon prior best results.
2. **Constrained Spanning Tree (CST)**: The CST introduced in ALG.2 eliminates the bottleneck of requiring additional edge sets in prior algorithms, representing a key technical innovation.
3. **Potential set technique**: In the LIMB subroutine of ALG.4, maintaining candidate sets of size $O(q)$ for anchor guessing elegantly enables the search to be completed within FPT time.
4. **Reduction to $\Phi$-TSP**: ALG.3 reveals an elegant connection between parameterized TSP and state-of-the-art metric TSP algorithms.
5. **Unified framework**: The techniques developed in this paper apply to both parameters $p$ and $q$, and may generalize to other related problems.

## Limitations & Future Work

- Under parameter $q$, whether the approximation ratio can be further improved from 3 to 1.5 within the FPT framework remains an important open problem.
- Computing $q$ itself is NP-hard and requires $O(3^q n^3)$ time, which may become a bottleneck for large $q$.
- The running time of ALG.3, $n^{O(p/\varepsilon)}$, is not FPT (the parameter appears in the polynomial exponent) and is only meaningful when $p$ is constant.
- All algorithms are combinatorial; experimental evaluation to assess practical performance is lacking.
- The paper does not discuss typical parameter ranges for $p$ and $q$ in real-world applications.

## Related Work & Insights

This paper sits at the intersection of parameterized complexity and approximation algorithms. Key related works include:

- **$\tau$-triangle inequality relaxation**: A series of works by Andreae–Bandelt, Bender–Chekuri, Mömke et al., relaxing the metric condition from different perspectives.
- **Subgroup Path Planning**: Has applications in robotics polishing in AI; complements the parameterized direction of this paper.
- **$\Phi$-TSP framework**: The general framework of Traub et al., directly utilized in ALG.3.

Implications for future research: The CST and potential set techniques introduced in this paper may be valuable for other graph-structure-based parameterized problems, such as parameterized minimum-weight Hamiltonian paths and parameterized vehicle routing problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Multiple algorithms with clever individual designs; significant improvement in approximation ratios (11→3)
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Rigorous and complete proofs; high technical sophistication
- **Practicality**: ⭐⭐⭐ — Strong theoretical contributions, but lacks experimental validation
- **Clarity of Presentation**: ⭐⭐⭐⭐ — Well-structured; notation clearly defined
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Finding Diverse Solutions Parameterized by Cliquewidth](finding_diverse_solutions_parameterized_by_cliquewidth.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/others/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)

</div>

<!-- RELATED:END -->
