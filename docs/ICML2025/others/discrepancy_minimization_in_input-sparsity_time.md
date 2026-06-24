---
title: >-
  [Paper Note] Discrepancy Minimization in Input-Sparsity Time
description: >-
  [ICML2025][discrepancy minimization] Proposed is the first input-sparsity time algorithm for real-valued discrepancy minimization—a combinatorial version running in $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ time and a fast matrix multiplication (FMM) version in $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$ time. The logarithmic approximation guarantee relative to $\mathrm{herdisc}$ remains unchanged, nearly bridging the computational gap between real-valued and binary matrices.
tags:
  - "ICML2025"
  - "discrepancy minimization"
  - "input-sparsity time"
  - "leverage score sampling"
  - "sketching"
  - "Edge-Walk"
  - "partial coloring"
  - "fast matrix multiplication"
date: 2026-05-08
content_hash: c581f19f400dddd6
---

# Discrepancy Minimization in Input-Sparsity Time

**Conference**: ICML2025  
**arXiv**: [2210.12468](https://arxiv.org/abs/2210.12468)  
**Code**: None (purely theoretical work)  
**Area**: Other/Theory (Combinatorial Optimization, Randomized Linear Algebra)  
**Keywords**: discrepancy minimization, input-sparsity time, leverage score sampling, sketching, Edge-Walk, partial coloring, fast matrix multiplication

## TL;DR

Proposed is the first input-sparsity time algorithm for real-valued discrepancy minimization—a combinatorial version running in $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ time and a fast matrix multiplication (FMM) version in $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$ time. The logarithmic approximation guarantee relative to $\mathrm{herdisc}$ remains unchanged, nearly bridging the computational gap between real-valued and binary matrices.

## Background & Motivation

**Discrepancy Theory** is a fundamental topic in combinatorics and theoretical computer science: given a set system $S_1,\dots,S_m \subseteq [n]$ (equivalently, an $m\times n$ matrix $A$), the goal is to find a two-coloring $x\in\{-1,+1\}^n$ that minimizes the maximum imbalance $\mathrm{disc}(A,x)=\|Ax\|_\infty$. This problem has wide applications in computational geometry, differential privacy, bin-packing, machine learning, and other fields.

The classic Spencer's Theorem guarantees the existence of a coloring with $\mathrm{disc}(A)\le 6\sqrt{n}$, but efficient constructive algorithms were long missing. Bansal (FOCS 2010) provided the first polynomial-time SDP algorithm with a running time of $\widetilde{O}(mn^{4.5})$. Larsen (SODA 2023) proposed a combinatorial algorithm that improved the time to $\widetilde{O}(mn^2+n^3)$, which is still far from optimal for sparse matrices. For **binary matrices**, JSS (SODA 2023) achieved the input-sparsity time of $\widetilde{O}(\mathrm{nnz}(A))$, but their approach cannot be generalized to real-valued matrices.

**Core Problem**: Can discrepancy minimization for real-valued matrices be solved in input-sparsity time? This is one of the main open questions proposed at the 2018 Discrepancy Theory and Integer Programming workshop.

## Method

### Overall Architecture

The algorithm follows the Iterated Partial Coloring framework of Lovett-Meka, consisting of two core subroutines:

1. **ProjectToSmallRows**: Finds a "hereditary projection" subspace $V$ such that the $\ell_2$-norm of each row of $A$ projected onto $V^\perp$ is bounded by $O(\mathrm{herdisc}(A)\log(m/n))$.
2. **PartialColoring**: Performs an Edge-Walk random walk on $V^\perp$, rounding a constant fraction of the coordinates to $\{-1,+1\}$ in each round.

### Key Designs

### Key Design 1: JL Sketch for Accelerating Row Norm Estimation

Larsen's algorithm requires explicitly computing the norm of each row of $B_t = A(I-V_t^\top V_t)$, costing $O(mn^2)$. This work constructs a sketched matrix using a JL random matrix $R\in\mathbb{R}^{n\times O(\log n)}$:

$$\widehat{B}_t = A(I-V_t^\top V_t)R$$

By the JL Lemma, it is guaranteed that for all rows $j$:

$$\|e_j^\top \widehat{B}_t\|_2 \in (1\pm\epsilon_0)\|e_j^\top B_t\|_2$$

Setting $\epsilon_0=\Theta(1)$ suffices to preserve the algorithm's correctness. The row norm query complexity is reduced from $O(mn^2)$ to $\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$.

### Key Design 2: Implicit Leverage Score Sampling

To avoid explicitly computing the SVD of $\bar{B}_t^\top\bar{B}_t$ (which costs $O(mn^2)$), this work proposes the **ImplicitLeverageScore** algorithm:

- It takes the original matrix $A$ and the orthogonal basis $V$ as input, and indirectly computes leverage scores via sparse embedding matrices $S_1, S_2$.
- It generates a diagonal sampling matrix $\widetilde{D}$ and constructs $\widetilde{B}_t = \widetilde{D}_t B_t$.
- It proves that $\widetilde{B}_t^\top\widetilde{B}_t$ is a spectral approximation of $\bar{B}_t^\top\bar{B}_t$. Appending its eigenvectors to $V$ still satisfies the "oversampling" precondition.

The entire process avoids explicitly storing $B_t$ (of size $m\times n$), in $\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$ time.

### Key Design 3: Robustness Analysis

The original algorithm requires exact row norms and exact SVD. This work demonstrates that:

- **Approximate norms suffice**: The constant-factor error induced by the JL sketch does not affect correctness. The norms of unselected rows remain bounded by $(1+\epsilon_0)\cdot C_0\cdot T\cdot\mathrm{herdisc}(A)$.
- **Approximate SVD suffices**: It is only required that the eigenvectors satisfy orthogonality and spectral approximation. A spectral approximation with $\epsilon_B=\Theta(1)$ preserves the row norm guarantees of ProjectToSmallRows.

### Key Design 4: Breaking the Cubic Barrier—The Guess-and-Correct Data Structure

In each round, PartialColoring needs to compute $(I-V_t^\top V_t)\mathsf{g}$. Since $V_t$ is dynamically updated, doing this over $O(n)$ rounds takes $O(n^3)$ time in total (constrained by the Online Matrix-Vector conjecture). This work designs a "Guess-and-Correct" data structure:

1. **Precomputation**: In the Init phase, a Gaussian matrix $\mathsf{G}\in\mathbb{R}^{n\times N}$ is generated, and the projection $\widetilde{G}=V^\top V\cdot\mathsf{G}$ is batch-precomputed.
2. **Lazy Update**: The columns are partitioned into batches of size $K=n^a$. Counters $k_q,k_u$ are maintained; on reaching the threshold, a Restart is triggered, accumulating low-rank corrections.
3. **Query**: Outputs $g - \widetilde{g} - \sum_{i=1}^{k_u} w_i w_i^\top g$, utilizing known low-rank corrections instead of fully recomputing the matrix.

For $K=n^a$, the total running time is:

$$\widetilde{O}\!\left(\mathrm{nnz}(A)+n^\omega+n^{2+a}+n^{1+\omega(1,1,a)-a}\right)$$

Choosing $a\approx 0.53$ balances these terms, yielding $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$.

### Key Design 5: Removing the Validation Step

Larsen's original algorithm requires validating the event $\mathcal{E}_\tau$ (checking if any of the $m$ rows exceed the threshold $\tau$) in each round, costing $O(mn)$ per round. By slightly increasing the threshold to $\tau'=\tau(\delta)$, this work utilizes concentration inequalities to prove that $\mathcal{E}_{\tau'}$ holds with probability $1-\delta$, thereby completely bypassing the validation step.

## Key Findings

### Main Theorem

| Method | Running Time | Approximation Ratio |
|------|---------|--------|
| Bansal (SDP) | $\widetilde{O}(mn^{4.5})$ | $O(\log(mn)\cdot\mathrm{herdisc}(A))$ |
| Larsen (Combinatorial) | $\widetilde{O}(mn^2+n^3)$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |
| **Ours (Combinatorial)** | $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |
| **Ours (FMM)** | $\widetilde{O}(\mathrm{nnz}(A)+n^{2.53})$ | $O(\mathrm{herdisc}(A)\cdot\log n\cdot\log^{1.5}m)$ |

### Key Theoretical Discoveries

- **Optimal for Tall Matrices**: When $m=\mathrm{poly}(n)$, the combinatorial version running in $\widetilde{O}(\mathrm{nnz}(A)+n^3)$ is optimal for tall matrices.
- **Breaking the Cubic Barrier**: The FMM version running in $n^{2.53}$ is the first to break the $n^3$ barrier for square matrix discrepancy, going beyond the limits of LP-based methods.
- **Fast Hereditary Projection (Theorem 1.5)**: Finding the hereditary projection matrix in $\widetilde{O}(\mathrm{nnz}(A)+n^\omega)$ time is optimal in a certain sense (reading $A$ requires $O(\mathrm{nnz}(A))$ and computing the projection matrix takes $n^\omega$).
- **Insensitive to the FMM Exponent**: Even if $\omega=2$, the final exponent only drops from $2.53$ to $2.5$, showing that the bottleneck is no longer matrix multiplication.

## Highlights & Insights

1. **Achieves the first input-sparsity time discrepancy minimization for real-valued matrices**, answering the open question from the 2018 workshop.
2. **Implicit leverage score sampling** is an independently valuable contribution: it performs oblivious dimension reduction without explicitly storing the $m\times n$ projection matrix.
3. **The guess-and-correct data structure** cleverly bypasses the lower bound of the OMv conjecture, turning online matrix-vector multiplication into batch operations via precomputation and lazy low-rank corrections.
4. **Robustness analysis** proves that constant-factor approximations of the norm/SVD do not compromise accuracy, offering a paradigm for applying sketching techniques to iterative algorithms.
5. **Nearly bridges the computational complexity gap** between real-valued and binary matrix discrepancy algorithms.

## Limitations & Future Work

1. **Logarithmic Approximation Ratio**: Compared to Bansal's $O(\log(mn))$ approximation, this work achieves $O(\log n\cdot\log^{1.5}m)$, which is slightly weaker.
2. **Reliance on FMM**: The $n^{2.53}$ version requires fast matrix multiplication, which is impractical; the combinatorial $n^3$ version, although more practical, does not break the cubic barrier.
3. **Limits of the Method**: Even with $\omega=2$, this method only reaches $n^{2.5}$, suggesting that a fundamentally new idea is needed for further improvements.
4. **Purely Theoretical Work**: Although there is experimental verification of efficiency in the appendix, evaluations in large-scale practical application scenarios are absent.
5. **Inevitable Projection Step**: For real-valued matrices, all known algorithms require projection operations, leading to an $n^\omega$ lower bound; whether projection-free techniques for binary matrices can be generalized remains an open question.

## Related Work & Insights

- **Bansal (FOCS 2010)**: Pioneering SDP discrepancy algorithm, $O(mn^{4.5})$
- **Lovett & Meka (SICOMP 2015)**: The Edge-Walk partial coloring framework, from which the core architecture of this paper is derived.
- **Larsen (SODA 2023)**: Combinatorial discrepancy algorithm of $O(mn^{2}+n^{3})$, which this work directly improves.
- **Jain, Sah & Sawhney (SODA 2023)**: Input-sparsity time algorithm for binary matrices; this work represents its generalization to the real-valued case.
- **Eldan & Singh (RS&A 2018)**: LP-based method. Combining this work's framework with theirs yields faster LP variants.
- **JL Sketch / Leverage Score Sampling**: This work generalizes standard sketching tools to iterative algorithm scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First to answer the open problem of real-valued matrix input-sparsity discrepancy minimization)
- Experimental Thoroughness: ⭐⭐⭐ (Primary focus is theoretical, with verification in the appendix)
- Writing Quality: ⭐⭐⭐⭐ (High technical depth but with clear presentation, assisted by flowcharts)
- Value: ⭐⭐⭐⭐⭐ (Significant progress in discrepancy theory, with implicit leverage score sampling holding independent value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[ICML 2025\] Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs](parity_requires_unified_input_dependence_and_negative_eigenvalues_in_ssms.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[ICML 2025\] Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification](fixing_the_loose_brake_exponential-tailed_stopping_time_in_best_arm_identificati.md)

</div>

<!-- RELATED:END -->
