---
title: >-
  [Paper Note] The Structural Complexity of Matrix-Vector Multiplication
description: >-
  [NeurIPS 2025][matrix-vector multiplication] This paper proves that for Boolean matrices $\mathbf{M} \in \{0,1\}^{m \times n}$ with corrupted VC-dimension $d$…
tags:
  - "NeurIPS 2025"
  - "matrix-vector multiplication"
  - "VC dimension"
  - "structured matrices"
  - "dynamic algorithms"
  - "minimum spanning tree"
date: 2026-05-08
content_hash: 9c5caffa6e722e9e
---

# The Structural Complexity of Matrix-Vector Multiplication

**Conference**: NeurIPS 2025
**arXiv**: [2502.21240](https://arxiv.org/abs/2502.21240)  
**Code**: Unavailable  
**Area**: Theoretical Computer Science / Algorithm Complexity
**Keywords**: matrix-vector multiplication, VC dimension, structured matrices, dynamic algorithms, minimum spanning tree

## TL;DR

This paper proves that for Boolean matrices $\mathbf{M} \in \{0,1\}^{m \times n}$ with corrupted VC-dimension $d$, matrix-vector multiplication can be performed in $\widetilde{O}(nm^{1-1/d}+m)$ time. This is the first truly sub-quadratic upper bound for structured matrices, refuting the applicability of the OMv conjecture on structured inputs, and yields the first high-accuracy sub-quadratic algorithms for dynamic Laplacian solving, effective resistance, triangle detection, and related problems.

## Background & Motivation

Matrix-vector multiplication is one of the most fundamental operations in machine learning—optimization, graph neural networks, online algorithms, and forward/backward passes in neural networks all rely on it. Strong theoretical lower bounds exist: Grønlund & Larsen (2015) showed that any polynomial-space data structure requires $\Omega(n^2/\log n)$ time, and this bound holds even in the average case.

In practice, however, heuristic-based sparse methods are far faster than the theoretical lower bounds suggest, creating a large gap between theory and practice. Alves et al. (2024) independently discovered a minimum spanning tree-based acceleration method in the GNN community and validated its practical efficiency, but without theoretical justification.

The authors' core insight is that since the average case (i.e., random unstructured inputs) is hard, the efficiency of practical algorithms must stem from the intrinsic structure of real-world data. This motivates studying the complexity of structured inputs, with VC dimension as the measure of structural complexity.

**Why VC dimension?** Many theoretical objects naturally have low VC dimension (star graphs and interval graphs have VC dimension 2; planar graphs have 4), and empirical observations suggest that real-world data typically exhibits low VC dimension. More importantly, the authors establish an elegant structural characterization (Theorem 1.1): every non-trivial hereditary matrix class has a constant upper bound on its VC dimension.

## Method

### Overall Architecture

The algorithm proceeds in two phases: a preprocessing phase that constructs a $\Delta$-labeled spanning tree of the matrix, and a query phase that exploits the spanning tree to efficiently compute matrix-vector products. The core idea is **differential compression**—linking similar columns via difference vectors $\Delta$.

### Key Designs

1. **Differential Compression and Minimum Spanning Tree**: Each column of $\mathbf{M}$ is treated as a point in $\{0,1\}^m$, with Hamming distance defining edge weights between columns. An approximate minimum spanning tree (MST) is constructed, with each edge $(x, y)$ labeled by $\Delta_{x,y} = \mathbf{M}_y - \mathbf{M}_x$. To answer a query $\mathbf{M}v$, the algorithm performs a DFS from the root, computing $(\mathbf{M}v)_y = (\mathbf{M}v)_x + \Delta_{y,x}^\top v$ recursively. The running time is $O(\text{weight}(T) + n + m)$. This idea originates from Björklund & Lingas (2001) and was independently rediscovered by Alves et al. (2024) in the GNN context.

2. **Low-Weight MST Existence (Core Contribution)**: Using the crossing lemma of Welzl (1988) from computational geometry, the authors prove that when the VC dimension of $\mathbf{M}^\top$ is at most $d$, the MST weight is at most $O(mn^{1-1/d}\log^2 n)$. The key steps are:

    - **Lemma 3.11** (Welzl's Lemma): In a range space with dual VC dimension $\leq d$, there exists a point pair $(x, y)$ with crossing number at most $O(mn^{-1/d}\log^2 n)$.
    - **Iterative Construction**: At each step, an edge with low crossing number is identified and one endpoint is removed; after $n$ steps, a spanning tree of weight $O(mn^{1-1/d}\log^2 n)$ is obtained.
    - **Robustness to Corruption**: The spanning tree is converted to a path (Lemma 3.14, with weight at most doubled), so that each corruption affects at most 2 edges.

3. **Definition of Corrupted VC-dimension**: Real-world data may not perfectly satisfy low VC dimension but may do so approximately. The corrupted VC-dimension $d$ is defined as the minimum $d$ such that there exists a set system $\mathcal{F}'$ with VC dimension $\leq d$, where $\mathcal{F}$ can be obtained from $\mathcal{F}'$ by adding or removing each element from at most $O(m^{1-1/d})$ sets. This definition is robust to small perturbations of regular structures and naturally accommodates noisy real-world data.

4. **Dynamic Extension**: Row and column insertions and deletions are handled via an **exponential layering** technique. The matrix is partitioned into $\log n$ sub-matrices $\mathbf{M}^{(0)}, \ldots, \mathbf{M}^{(\log n)}$, where $\mathbf{M}^{(i)}$ has at most $2^i$ columns. New columns enter $\mathbf{M}^{(0)}$; when a layer is full, it is merged into the next, yielding amortized update time $\widetilde{O}(m)$.

### Extension to Non-Boolean Matrices

The results are extended to real-valued matrices via Pollard's pseudo-dimension. If each column has at most $T$ distinct values and the pseudo-dimension is $d$, the query time is $\widetilde{O}(Tnm^{1-1/d}+m)$.

## Key Experimental Results

This is a purely theoretical paper with no experimental section. The core theoretical results are summarized below.

### Main Results

| Setting | Preprocessing | Query Time | Update Time |
|---------|--------------|------------|-------------|
| Static (Theorem 1.3) | $\widetilde{O}(mn)$ | $\widetilde{O}(nm^{1-1/d}+m)$ | — |
| Dynamic (Theorem 1.4) | $\widetilde{O}(mn)$ | $\widetilde{O}(nm^{1-1/d^*}+m)$ | Row $\widetilde{O}(n)$, Col $\widetilde{O}(m)$ |
| Naive | — | $O(mn)$ | — |

### Application Corollaries

| Dynamic Problem | Ours | Prior Lower Bound (OMv/BMM conjecture) | Notes |
|----------------|------|----------------------------------------|-------|
| Laplacian Solving | $\widetilde{O}(n^{2-1/d}\log(1/\epsilon))$ | $\Omega(n^2)$ (high accuracy) | First high-accuracy sub-quadratic algorithm |
| Effective Resistance | $\widetilde{O}(n^{2-1/d}\log(1/\epsilon))$ | $\Omega(n^2)$ | $(1\pm\epsilon)$-approximation |
| Triangle Detection | $\widetilde{O}(n^{2-1/d})$ | $\Omega(n^2)$ (BMM/OMv) | Breaks lower bound on structured graphs |
| Approximate Shortest Path | $\widetilde{O}(kn^{2-1/2d}/\epsilon)$ | — | $(1+\epsilon)$-approximation |

### Key Findings

- For VC dimension $d=2$ (star graphs, interval graphs), query time drops from $\Theta(n^2)$ to $\widetilde{O}(n^{3/2})$.
- For planar graphs ($d=4$), it drops to $\widetilde{O}(n^{7/4})$.
- The algorithm does not require knowledge of the VC dimension and adapts to structure automatically.

## Highlights & Insights

- **Bridging Theory and Practice**: The paper provides the first theoretical justification for the MST-based heuristic of Alves et al. (2024), explaining why MST-based matrix multiplication is far faster in practice than worst-case lower bounds suggest.
- **Elegant Definition of Corrupted VC-dimension**: The definition simultaneously captures structure and noise, and the algorithm requires no knowledge of the specific corruption locations.
- **Refuting the Scope of the OMv Conjecture**: The OMv conjecture underlies numerous conditional lower bounds for dynamic algorithms; this paper demonstrates that it fails for structured inputs, opening new algorithmic possibilities.
- **Cross-Domain Transfer**: The crossing number theory of range spaces from Welzl (1988) in computational geometry is applied to matrix-vector multiplication, representing a notable interdisciplinary technique transfer.

## Limitations & Future Work

- The linear dependence on the number of distinct values $T$ in the non-Boolean setting may not be optimal (the authors conjecture it should be logarithmic).
- Matching lower bounds are absent to verify whether the $\widetilde{O}(n^{2-1/d})$ scaling with respect to VC dimension is tight.
- As a purely theoretical result, no experiments validate practical speedups on real-world data.
- In the dynamic setting, $d^*$ tracks the historical maximum corrupted VC dimension and cannot improve over time.

## Related Work & Insights

- The MST idea of Björklund & Lingas (2001) is foundational, but their work lacks a VC-dimension-based analysis.
- This work is complementary to Alman & Vassilevska Williams (2020) on orthogonal vector matrices.
- The results have broad implications for the graph algorithms community: all conditional lower bounds based on OMv/BMM conjectures need to be revisited for structured inputs.

## Technical Details

- The hereditary matrix class characterization in Theorem 1.1 implies that whenever the matrix structure of real-world data is closed under row/column deletion (which naturally holds in most settings), the VC dimension must have a constant upper bound.
- The relationship between the VC dimension of $\mathbf{M}$ and that of $\mathbf{M}^\top$ (VC dimension of $\mathbf{M}^\top$ being $d$ implies VC dimension of $\mathbf{M}$ is at most $2^{d+1}$) is resolved via Lemma 3.4, avoiding exponential blowup.
- The approximate MST uses the $(1+\epsilon)$-approximation algorithm of Indyk & Motwani (1998); setting $\epsilon = \log n$ allows preprocessing in $\widetilde{O}(mn)$ time.
- The extension to non-Boolean matrices via Pollard's pseudo-dimension is natural—the linear dependence on $T$ (the number of distinct values per column) may potentially be improved to logarithmic through a more refined analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The corrupted VC-dimension framework and cross-domain technical integration are highly original.
- Experimental Thoroughness: ⭐⭐⭐ — A purely theoretical paper with no experiments, though application corollaries are abundant.
- Writing Quality: ⭐⭐⭐⭐ — Definitions are clear and proofs are well-motivated, though notation density is high.
- Value: ⭐⭐⭐⭐⭐ — Provides a structural explanation for a fundamental theory-practice gap with far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](the_parameterized_complexity_of_computing_the_vc-dimension.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](../../AAAI2026/others/structural_approach_to_guiding_a_present-biased_agent.md)
- [\[ICCV 2025\] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](../../ICCV2025/others/on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)

</div>

<!-- RELATED:END -->
