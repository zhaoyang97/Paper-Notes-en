---
title: >-
  [Paper Note] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs
description: >-
  [AAAI 2026][3D Vision][Traveling Salesman Problem] This paper proposes improved FPT approximation algorithms for the Traveling Salesman Problem (TSP) on non-metric graphs, parameterized by $p$ (the number of vertices violating the triangle inequality) and $q$ (the size of a minimum violator set). It improves the approximation ratio from 2.5 to 1.5 under parameter $p$, and from 11 to 3 under parameter $q$.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Traveling Salesman Problem"
  - "Parameterized Approximation Algorithm"
  - "Non-Metric Graphs"
  - "Fixed-Parameter Tractable"
  - "Triangle Inequality"
date: 2026-05-08
content_hash: 422a39d126e62b12
---

# Parameterized Approximation Algorithms for TSP on Non-Metric Graphs

**Conference**: AAAI 2026  
**arXiv**: [2503.03642](https://arxiv.org/abs/2503.03642)  
**Code**: None  
**Area**: Theoretical Computer Science / Combinatorial Optimization  
**Keywords**: Traveling Salesman Problem, Parameterized Approximation Algorithm, Non-Metric Graphs, Fixed-Parameter Tractable, Triangle Inequality

## TL;DR

This paper proposes improved FPT approximation algorithms for the Traveling Salesman Problem (TSP) on non-metric graphs, parameterized by $p$ (the number of vertices violating the triangle inequality) and $q$ (the size of a minimum violator set). It improves the approximation ratio from 2.5 to 1.5 under parameter $p$, and from 11 to 3 under parameter $q$.

## Background & Motivation

The Traveling Salesman Problem (TSP) is a classic NP-hard problem in combinatorial optimization with widespread applications in logistics, manufacturing, telecommunications, and more. For metric graphs satisfying the triangle inequality, the classical Christofides-Serdyukov algorithm achieves an approximation ratio of 1.5, which Karlin et al. further improved to $1.5 - 10^{-36}$. However, for general graphs (non-metric graphs), TSP cannot be approximated within any computable function $f(n)$ in polynomial time.

The huge approximation performance gap between metric and general graphs motivates researchers to think: can better approximation algorithms be obtained for graphs that are "close" to being metric? Recently, the parameterized complexity community introduced two natural parameters to measure the "distance" of a graph to being metric:

- **Parameter $p$**: the number of vertices in the graph that participate in triangles violating the triangle inequality.
- **Parameter $q$**: the minimum number of vertices that must be deleted to make the graph metric (the minimum violator set).

The realistic motivation for these two parameters comes from urban tour bus route planning: most landmarks satisfy the triangle inequality, but rapid direct routes between a few high-profile landmarks violate the triangle inequality. These violations typically involve only a small number of landmarks, meaning $p$ and $q$ are small.

In prior work, Zhou et al. gave an FPT 3-approximation algorithm under parameter $p$, which Bampis et al. improved to a 2.5-approximate algorithm; whereas under parameter $q$, the best FPT result by Zhou et al. was an 11-approximation. The core motivation of this paper is precisely to answer the open question raised by Bampis et al.: can these approximation ratios be further improved?

## Method

### Overall Architecture

This paper proposes multiple algorithms for the two parameters, respectively. The core mechanism is to decompose the non-metric TSP into subproblems on "bad vertices" (vertices participating in triangle inequality violations) and "good vertices," leveraging the metric properties of good vertices to perform shortcutting operations to reduce the cost of the solution.

**Property 1 (Key Property)**: Any triangle containing good vertices satisfies the triangle inequality. This is the foundation of all algorithm designs—shorcutting through good vertices does not increase the weight of the path.

### Key Designs

1. **ALG.1 ($(α+1)$-approximation, parameter $p$)**: The simplest algorithm. Select a good vertex $o$, use dynamic programming to find an optimal TSP tour $T_b$ on $G[V_b \cup \{o\}]$, then use an $\alpha$-approximation algorithm for metric TSP to find $T_g$ on $G[V_g]$, and finally shortcut $T_b \cup T_g$ to obtain a complete TSP tour. The key observation is that $\text{OPT} \geq w(T_b^*)$ and $\text{OPT} \geq w(T_g^*)$, thus the overall approximation ratio is $(\alpha + 1)$. The running time is $2^{O(p)} + n^{O(1)}$, which simultaneously improves both the approximation ratio and running time.

2. **ALG.2 (1.5-approximation, parameter $p$)**: A more sophisticated algorithm. First guess the subgraph of the optimal tour $T^*$ on bad vertices (i.e., "bad paths" $\mathcal{A}$), then construct a Constrained Spanning Tree (CST) $F_{\mathcal{A}}$, and subsequently use minimum weight matching on an auxiliary graph to correct odd-degree vertices. The core innovations are:

    - Constructing an auxiliary graph $\widetilde{G}$ by contracting bad paths and finding a minimum spanning tree on it.
    - Constructing an auxiliary graph $G'$ to handle the matching problem for odd-degree vertices (since the original graph might be non-metric, minimum weight matching cannot be applied directly).
    - Proving that $w(F_{\mathcal{A}}) \leq \text{OPT}$ and $w(\mathcal{M}_{\mathcal{A}}) \leq \frac{1}{2} \text{OPT}$.

   Through a delicate shortcutting lemma (Lemma 3), a final approximation ratio of $\frac{3}{2} \cdot \text{OPT}$ is achieved.

3. **ALG.3 ($(α+\varepsilon)$-approximation, when the parameter $p$ is a constant)**: Reducible to the metric $k$-TSPP ($k$-TSP path problem), utilizing the $\Phi$-TSP algorithm by Traub et al. When $p = O(1)$, the approximation ratio almost equals that of the metric TSP. The running time is $n^{O(p/\varepsilon)}$.

4. **ALG.4 (3-approximation, parameter $q$)**: The core algorithm under parameter $q$. The key challenge is that, unlike the parameter $p$, triangles containing one bad vertex and two good vertices may also violate the triangle inequality, so Property 1 does not apply. The algorithm contains three subprocesses:

    - **LIMB**: Guessing anchors and limbs using the potential set technique, ensuring $w(\mathcal{B}') \leq w(\mathcal{B})$.
    - **CONNECT**: Connecting the disconnected components via partition guessing, maintaining the complexity within $2^{O(q \log q)}$.
    - **SHORTCUT**: Handling the shortcutting operations on Eulerian graphs.

### Loss & Training

This work is purely theoretical and does not involve training. The core analysis techniques include:

- **Shortcutting analysis**: Using Property 1 to prove that shortcutting through good vertices does not increase weight.
- **Matching lower bound**: Decomposing the TSP tour into two matchings, proving that the minimum matching weight does not exceed $\frac{1}{2} \text{OPT}$.
- **Potential set technique**: In the LIMB sub-algorithm, maintaining a candidate set of size $O(q)$ for each anchor to ensure that the guessed anchor cost does not exceed the real anchor cost.

## Key Experimental Results

### Main Results

This work is purely theoretical, without experimental data, but provides a comprehensive theoretical results comparison:

| Parameter | Approximation Ratio | Running Time | Source |
|-----------|---------------------|--------------|--------|
| $p$ | 3 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Zhou et al. |
| $p$ | 2.5 | $2^{O(p \log p)} \cdot n^{O(1)}$ | Bampis et al. |
| $p$ | **$\alpha+1 \approx 2.5$** | $2^{O(p)} + n^{O(1)}$ | **Ours ALG.1** |
| $p$ | **1.5** | $2^{O(p \log p)} \cdot n^{O(1)}$ | **Ours ALG.2** |
| $p$ | **$\alpha + \varepsilon$** | $n^{O(p/\varepsilon)}$ | **Ours ALG.3** |
| $q$ | 3 | $n^{O(q+1)}$ | Zhou et al. |
| $q$ | 11 | $2^{O(q \log q)} \cdot n^{O(1)}$ | Zhou et al. |
| $q$ | **3** | $2^{O(q \log q)} \cdot n^{O(1)}$ | **Ours ALG.4** |
| $q$ | **$\alpha + \varepsilon$** | $n^{O(q/\varepsilon)}$ | **Ours** |

### Ablation Study

| Algorithm Variant | Approximation Ratio | Key Improvement |
|-------------------|---------------------|-----------------|
| ALG.1 Alone | $\alpha + 1$ | Simplest, optimal running time of $2^{O(p)} + n^{O(1)}$ |
| ALG.2 using CST + matching | 1.5 | Avoids the extra edge set $E'$, directly constructing the constrained spanning tree |
| ALG.4 vs prior $q$ parameter | 3 vs 11 | LIMB + CONNECT techniques achieve 3-approximation in FPT time |

### Key Findings

- For parameter $p$, the 1.5-approximation ratio reaches the best-known approximation ratio for metric TSP, demonstrating that the proposed algorithm is almost optimal under this parameterization.
- When $p$ or $q$ is a constant, the $(α+\varepsilon)$-approximation ratio almost matches the optimal approximation ratio of metric TSP.
- ALG.4 substantially improves the FPT approximation ratio under parameter $q$ from 11 to 3, while preserving the same FPT running time.

## Highlights & Insights

1. **Simple and effective divide-and-conquer**: ALG.1 connects the bad-vertex sub-tour and the good-vertex sub-tour via one good vertex. It is extremely simple yet already improves on prior state-of-the-art results.
2. **Constrained Spanning Tree (CST) concept**: The CST introduced in ALG.2 avoids the bottleneck of requiring extra edge sets in prior algorithms, which is a key technical innovation.
3. **Potential set technique**: In the LIMB sub-algorithm of ALG.4, anchors are guessed by maintaining a candidate set of size $O(q)$, cleverly completing the search in FPT time.
4. **Reduction to $\Phi$-TSP**: ALG.3 shows an elegant connection between parameterized TSP and the latest metric TSP algorithms.
5. **Unified framework**: The techniques in this paper are applicable to both parameters $p$ and $q$, and could potentially be generalized to other related problems.

## Limitations & Future Work

- For parameter $q$, can the approximation ratio be further improved from 3 to 1.5 in the FPT framework? This remains an important open question.
- Computing $q$ itself is NP-hard, requiring $O(3^q n^3)$ time, which may become a bottleneck when $q$ is large.
- The running time of ALG.3, $n^{O(p/\varepsilon)}$, is not FPT (the parameter appears in the polynomial exponent), which is only meaningful when $p$ is a constant.
- All algorithms are combinatorial, with no experimental evaluations to verify actual performance.
- The paper does not discuss the typical parameter ranges of these parameterized results in real-world applications.

## Related Work & Insights

This paper lies at the intersection of parameterized complexity and approximation algorithms. Key related work includes:

- **$\tau$-triangle inequality relaxation**: A series of works by Andreae-Bandelt, Bender-Chekuri, Mömke et al., which relax the metric condition from different perspectives.
- **Subgroup Path Planning**: Used in robotics polishing in AI, which is complementary to the parameterized direction in this paper.
- **$\Phi$-TSP framework**: A general framework by Traub et al., which ALG.3 directly exploits.

Insights for future research: The CST and potential set techniques proposed in this paper could be valuable for other graph-structure-based parameterized problems, such as parameterized minimum-weight Hamiltonian paths and parameterized vehicle routing problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Multiple algorithms each have clever designs, with substantial improvements in approximation ratios (11 to 3).
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Rigorous and complete proofs, high technical content.
- **Utility**: ⭐⭐⭐ — Outstanding theoretical contribution, but lacking experimental verification.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-defined notation.
- **Overall Rating**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Special Unitary Parameterized Estimators of Rotation](../../ICLR2026/3d_vision/special_unitary_parameterized_estimators_of_rotation.md)
- [\[NeurIPS 2025\] Fully Dynamic Algorithms for Chamfer Distance](../../NeurIPS2025/3d_vision/fully_dynamic_algorithms_for_chamfer_distance.md)
- [\[CVPR 2026\] Learning to Infer Parameterized Representations of Plants from 3D Scans](../../CVPR2026/3d_vision/learning_to_infer_parameterized_representations_of_plants_from_3d_scans.md)
- [\[CVPR 2026\] The Midas Touch for Metric Depth](../../CVPR2026/3d_vision/the_midas_touch_for_metric_depth.md)
- [\[CVPR 2026\] X-band Radar Non-Line-of-Sight Imaging](../../CVPR2026/3d_vision/x-band_radar_non-line-of-sight_imaging.md)

</div>

<!-- RELATED:END -->
