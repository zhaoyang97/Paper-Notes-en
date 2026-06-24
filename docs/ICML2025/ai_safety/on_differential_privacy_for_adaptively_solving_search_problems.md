---
title: >-
  [Paper Note] On Differential Privacy for Adaptively Solving Search Problems via Sketching
description: >-
  [ICML 2025][AI Safety][differential privacy] This work extends differential privacy from numerical estimation to search problems (which require returning a solution vector rather than a single scalar) for the first time. Under a mild sparse nearest neighbor assumption, it proposes an algorithm that correctly answers $T$ adaptive approximate nearest neighbor queries using $\tilde{O}(\sqrt{T} \cdot s)$ copies of the data structure, while also providing adaptive regression data…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "differential privacy"
  - "adaptive queries"
  - "approximate nearest neighbor"
  - "regression"
  - "sketching"
date: 2026-05-08
content_hash: 8df4444773a69e5d
---

# On Differential Privacy for Adaptively Solving Search Problems via Sketching

**Conference**: ICML 2025  
**arXiv**: [2506.05503](https://arxiv.org/abs/2506.05503)  
**Code**: None  
**Area**: AI Safety / Differential Privacy / Theoretical Algorithms  
**Keywords**: differential privacy, adaptive queries, approximate nearest neighbor, regression, sketching

## TL;DR

This work extends differential privacy from numerical estimation to search problems (which require returning a solution vector rather than a single scalar) for the first time. Under a mild sparse nearest neighbor assumption, it proposes an algorithm that correctly answers $T$ adaptive approximate nearest neighbor queries using $\tilde{O}(\sqrt{T} \cdot s)$ copies of the data structure, while also providing adaptive regression data structures that depend on the condition number.

## Background & Motivation

**Background**: In modern algorithms, data structures are often embedded in iterative optimization or online learning pipelines, where the output of the current iteration affects the subsequent inputs. This requires data structures to be robust against adaptive adversaries (who can design future queries based on past outputs). However, traditional randomized data structures only guarantee correctness against oblivious adversaries (where the query sequence is fixed in advance, independent of the randomness).

**Limitations of Prior Work**: Three existing paradigms for handling adaptive queries each have their bottlenecks: (1) Preparing $T$ independent copies and using a fresh one for each query, inducing an unacceptable $O(T)$ factor overhead in space and preprocessing; (2) Utilizing $\varepsilon$-net arguments, requiring $O(d)$ copies, which is ineffective when $d \approx T$ or $d \gg T$; (3) Differential privacy (DP) methods (Hassidim et al., Ben-Eliezer et al.) can reduce the number of copies to $O(\sqrt{T})$, but only apply to estimation problems outputting a single numerical value (e.g., distance estimation, regression loss).

**Key Challenge**: The output of search problems is a full vector (e.g., coordinates of nearest neighbors, regression solution vectors), which leaks significantly more information about the internal randomness of the data structure than a single scalar. Directly generalizing the DP median mechanism to high-dimensional spaces fails because the median mechanism cannot guarantee that the output is a valid point in the original dataset.

**Goal**: In adaptive query scenarios, can efficient algorithms using $o(\min\{d, T\})$ copies of the data structure be designed for search problems (which return a solution vector instead of a scalar)?

**Key Insight**: The approach reduces search problems to DP selection problems: instead of computing a median over numbers, the method conducts private voting/selection among candidate answers. The key insight is that if the size of the candidate answer set per query is bounded ($\leq s$), the underlying sparsity can be leveraged to design an efficient private selection mechanism.

**Core Idea**: Leveraging the sparsity of candidate solution sets in search problems, the number of copies is reduced from $\min\{d, T\}$ to $\tilde{O}(\sqrt{T} \cdot s)$ via DP selection mechanisms (rather than median mechanisms).

## Method

### Overall Architecture

Algorithms are designed for two classic search problems: (1) Approximate Nearest Neighbor (ANN) search—returning a sufficiently close point from the dataset given a query point; (2) Adaptive Regression—where the design matrix and response vector can be adaptively updated, returning a regression solution vector. The shared core idea is to treat the internal randomness as a private database to be protected, while employing different technical strategies tailored to the characteristics of each problem.

### Key Designs

1. **Differential Privacy Selection Mechanism (for ANN)**:

    - **Function**: Correctly select approximate nearest neighbors from $\tilde{O}(\sqrt{T} \cdot s)$ LSH data structure copies under adaptive query scenarios.
    - **Mechanism**: For each query $v_t$, a subset of data structure copies is sampled and queried, with each copy returning zero or more candidate nearest neighbors. The occurrences of all candidate nearest neighbors form a count vector, to which Laplace noise is added, and the candidate with the highest noisy count is returned. Crucially, the Advanced Composition Theorem ensures that sampling $\tilde{O}(\sqrt{T})$ independent copies is sufficient, and the assumption that the candidate solution set size is $\leq s$ (Assumption 1.2) keeps the privacy budget bounded.
    - **Design Motivation**: Unlike the median mechanism used in numerical problems, ANN requires outputting an actual point from the dataset. The voting-and-noise selection mechanism naturally satisfies this constraint, while Laplace noise provides differential privacy guarantees.

2. **Sparse Argmax Mechanism**:

    - **Function**: Overcome the $O(n)$ noise generation bottleneck in the naive selection mechanism.
    - **Mechanism**: The naive approach requires generating Laplace noise for all $n$ points in the dataset and finding the maximum, even though most points have zero counts. Sparse Argmax leverages the $s$-sparse structure of the count vector: noise is added only to the $s$ non-zero positions, while the "maximum noise among all zero positions" is simulated by sampling the $n$-th order statistic from the Laplace distribution. This reduces the selection time to $O(s \log n)$.
    - **Design Motivation**: A direct implementation takes $O(n)$ time, which becomes a bottleneck when the dataset is large ($n \gg s$). The sparse structure is a natural corollary of the ANN assumption, and failing to exploit it would waste the structural properties of the problem.

3. **Coordinate-wise Private Median + $\ell_\infty$ Guarantee (for Regression)**:

    - **Function**: Return regression solution vectors under adaptive update scenarios.
    - **Mechanism**: Create $\tilde{O}(\sqrt{Td})$ independent sketch matrices $S_i$, and store $S_i U$ and $S_i b$. For each query, sample $\tilde{O}(1)$ sketches and solve the sketched regression problem to obtain multiple candidate solutions $x_{(1)}, ..., x_{(m)}$. Compute the private median independently for each coordinate of the solution vector (by adding Laplace noise and then finding the median). The key innovation is utilizing the unique properties of SRHT (Subsampled Randomized Hadamard Transform) to obtain an $\ell_\infty$ guarantee—the error for each coordinate is $O(\frac{\alpha}{\sqrt{d}} \cdot \frac{\|Ux^* - b\|_2}{\sigma_{\min}(U)})$, which keeps the overall error controlled after aggregating coordinate-wise medians.
    - **Design Motivation**: Search problems cannot directly use scalar medians like numerical estimation problems. However, the regression solution is a continuous vector that can be decomposed coordinate-wise—turning each coordinate into a one-dimensional numerical estimation problem, which allows reusing the scalar private median. The $\ell_\infty$ guarantee of SRHT ensures that this coordinate-wise decomposition does not blow up the error.

### Loss & Training

Purely theoretical work, no training involved. The core metrics are complexity measures: space, preprocessing time, query time, and amortized cost.

## Key Experimental Results

### Complexity Comparison Table (ANN Problem)

| Method | Space | Amortized Preprocessing/Query | Query Time |
|------|------|-------------|---------|
| $T$ Copies | $Tn^{1+\rho}+nd$ | $n^{1+\rho}d$ | $n^\rho d$ |
| $d$ Copies ($\varepsilon$-net) | $n^{1+\rho}d$ | $\frac{d}{T}n^{1+\rho}d$ | $n^\rho d$ |
| **Ours** | $\sqrt{T} \cdot s \cdot n^{1+\rho}+nd$ | $\frac{s}{\sqrt{T}} n^{1+\rho}d$ | $s \cdot n^\rho d$ |

When $s = o(\sqrt{T})$, Ours achieves better space and amortized preprocessing than the $T$-copy baseline.

### Complexity Comparison Table (Regression Problem)

| Method | Space | Query Time | Key Dependency |
|------|------|---------|---------|
| $T$ Sketches | $Td^2/\alpha^2$ | $d^{\omega+1}/\alpha^2$ | Linear dependency on $T$ |
| $nd$ Sketches ($\varepsilon$-net) | $nd^3/\alpha^2$ | $d^{\omega+1}/\alpha^2$ | Linear dependency on $nd$ |
| **Ours (Thm 1.8)** | $\sqrt{T} \cdot d^{2.5}\kappa^2/\alpha^2$ | $d^{\omega+1}\kappa^2/\alpha^2$ | $\sqrt{T}$, including condition number $\kappa$ |
| **Ours (Thm 1.9, only update $b$)** | — | $d^2$ | Preconditioning eliminates $\kappa$ |

### Key Findings

- **The improvement from $T$ to $\sqrt{T}$ is non-trivial**: For search problems, this is the first time the linear lower bound on the number of copies has been broken.
- **Mild assumptions**: The sparse nearest neighbor assumption ($s \leq n^\rho$) can be met via preprocessing clustering, and a bounded condition number is a standard assumption in regression.
- **Fundamentally different technical approaches for ANN and regression**: ANN uses discrete selection (voting) while regression uses coordinate decomposition (median), reflecting the structural differences between these two search problems.
- **Theorem 1.9 eliminates dependency on the condition number**: When only the response vector $b$ is updated, a preconditioner for $U$ can be precomputed, reducing the query time to $O(d^2)$.

## Highlights & Insights

- **Resolves a major open question**: Extending the DP framework from numerical estimation to search problems represents a conceptual breakthrough. This paper provides an affirmative answer to the core open question left by prior work: "Can search problems achieve $\sqrt{T}$?"
- **The Sparse Argmax Mechanism** bypasses the bottleneck of generating noise for all $n$ candidate points. By cleverly exploiting the sparsity of the candidate set, the running time is reduced from $O(n)$ to $O(s \log n)$.
- **Innovative application of SRHT's $\ell_\infty$ guarantee** in regression sketches: The combined approach of coordinate-wise decomposition and private medians can be generalized to other private problems requiring vector outputs.

## Limitations & Future Work

- Purely theoretical contribution without empirical verification; the practical impact of constant factors and polylogarithmic terms remains unclear.
- The ANN assumption requires at most $s$ approximate nearest neighbors, which might be violated in high-density datasets (e.g., image feature spaces).
- The regression results depend on the condition number $\kappa$ (Theorem 1.8) and may perform poorly on ill-conditioned problems.
- ANN results under general metric spaces (non-normed spaces) remain an open problem.

## Related Work & Insights

- **vs. Hassidim et al. / Ben-Eliezer et al.**: The former applied DP to numerical estimation to achieve $\sqrt{T}$ copies. This work extends it to search problems, which is technically more challenging since the output dimension increases from 1 to $d$.
- **vs. $\varepsilon$-net methods**: $\varepsilon$-net approaches require $O(d)$ copies, yielding no improvement when $d \approx T$. The proposed method consistently outperforms both baselines when $s = o(\sqrt{T})$.
- **Applications**: The results can be directly applied to two downstream problems: online weighted matching (requiring adaptive nearest neighbor queries during matching) and terminal embedding (preserving distances after dimension reduction). Quantitative complexity improvements are detailed in Section 4 of the paper.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Resolves a core open question in adaptive search problems, achieving a conceptual breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical; replaced empirical experiments with complexity tables, lacking validation on real datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly defined problems, reasonable assumptions, and complete proof structures.
- **Value**: ⭐⭐⭐⭐ Outstanding theoretical contribution, opening new directions for the intersection of DP and adaptive data structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](../../NeurIPS2025/ai_safety/sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](../../NeurIPS2025/ai_safety/multi-class_support_vector_machine_with_differential_privacy.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](../../NeurIPS2025/ai_safety/unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](../../NeurIPS2025/ai_safety/differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)

</div>

<!-- RELATED:END -->
