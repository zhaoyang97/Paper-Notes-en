---
title: >-
  [Paper Note] Graph Alignment via Birkhoff Relaxation
description: >-
  [NeurIPS 2025][graph alignment] This paper provides the first theoretical guarantees for the Birkhoff relaxation (the tightest convex relaxation of QAP) under the Gaussian Wigner model: when $\sigma = o(n^{-1})$, the relaxed solution approximates the true permutation; when $\sigma = \Omega(n^{-0.5})$, the relaxed solution is far from the true permutation, revealing a phase transition phenomenon.
tags:
  - "NeurIPS 2025"
  - "graph alignment"
  - "quadratic assignment"
  - "Birkhoff polytope"
  - "convex relaxation"
  - "phase transition"
  - "Gaussian Wigner Model"
date: 2026-05-08
content_hash: 6f7bd47b54a6112d
---

# Graph Alignment via Birkhoff Relaxation

**Conference**: NeurIPS 2025
**arXiv**: [2503.05323](https://arxiv.org/abs/2503.05323)  
**Code**: [smv30/convex_rel_for_graph_alignment](https://github.com/smv30/convex_rel_for_graph_alignment)  
**Area**: Graph Algorithms / Combinatorial Optimization
**Keywords**: graph alignment, quadratic assignment, Birkhoff polytope, convex relaxation, phase transition, Gaussian Wigner Model

## TL;DR

This paper provides the first theoretical guarantees for the Birkhoff relaxation (the tightest convex relaxation of QAP) under the Gaussian Wigner model: when $\sigma = o(n^{-1})$, the relaxed solution approximates the true permutation; when $\sigma = \Omega(n^{-0.5})$, the relaxed solution is far from the true permutation, revealing a phase transition phenomenon.

## Core Problem

**Graph Alignment**: Given two undirected graphs $G_1, G_2$ each with $n$ vertices, find a vertex mapping that maximizes edge overlap. This is mathematically equivalent to the Quadratic Assignment Problem (QAP):

$$\Pi^\star = \arg\min_{X \in \mathcal{P}_n} \|AX - XB\|_F^2$$

where $\mathcal{P}_n$ is the set of $n \times n$ permutation matrices. QAP is NP-hard in the worst case and difficult to approximate.

**Limitations of Prior Work**:

- **GRAMPA** (Fan et al.): Succeeds at $\sigma = O(1/\log n)$, but requires tuning regularization parameter $\eta$ and achieves weaker empirical performance than Birkhoff relaxation.
- **Simplex relaxation** (Araya & Tyagi): Theoretical guarantees only at $\sigma = 0$ (noiseless).
- **EIG1** (spectral method): Threshold is $\sigma = \Theta(n^{-7/6})$, a more restrictive condition.
- The Birkhoff relaxation achieves the best empirical performance but previously lacked any theoretical guarantees.

## Key Designs

### Core Idea

Replace the feasible set of QAP from the permutation matrix set $\mathcal{P}_n$ with its convex hull — the **Birkhoff polytope** $\mathcal{B}_n$ (the set of all doubly stochastic matrices) — yielding the Birkhoff relaxation:

$$X^\star = \arg\min_{X \in \mathcal{B}_n} \|AX - XB\|_F^2$$

This is the **tightest convex relaxation** of QAP. The core contribution is proving that the distance between the relaxed solution $X^\star$ and the true permutation $\Pi^\star$ exhibits a **phase transition**:

| Noise Condition | Behavior |
|---|---|
| $\sigma = o(n^{-1})$ | $\|X^\star - \Pi^\star\|_F^2 = o(n)$, small perturbation |
| $\sigma = \Omega(n^{-0.5})$ | $\|X^\star - \Pi^\star\|_F^2 = \Omega(n)$, far from true permutation |

## Method

### Gaussian Wigner Model Setup

The (weighted) adjacency matrices are drawn from correlated Gaussian Orthogonal Ensembles (GOE):

- $A$ is a GOE matrix: $A_{ii} \sim N(0, 2/n)$, $A_{ij} = A_{ji} \sim N(0, 1/n)$
- $B_{\pi^\star(i), \pi^\star(j)} = A_{ij} + \sigma Z_{ij}$, where $Z$ is an independent GOE matrix
- Correlation coefficient is $1/\sqrt{1 + \sigma^2}$; $\sigma$ controls noise intensity

### Part I: Well-Separation ($\sigma = \Omega(n^{-0.5+\epsilon})$)

The proof establishes upper and lower bounds on $\|AX^\star - X^\star B\|_F^2$:

1. **Upper bound**: Using $J/n$ (uniform matrix) as a feasible solution yields $\|AX^\star - X^\star B\|_F^2 \leq 9n^\epsilon$.
2. **Lower bound**: Expanding $\|AX^\star - X^\star B\|_F^2$ and using $B = A + \sigma Z$ gives a lower bound in terms of $\|I - X^\star\|_F$:
   $$\|AX^\star - X^\star B\|_F^2 \geq \frac{\sigma^2 n}{4} - 2c\sigma^2 \sqrt{n} \|I - X^\star\|_F$$
3. **Combining**: The two bounds together yield $\|I - X^\star\|_F \geq \sqrt{n}/(16c)$.

Key technical ingredients: concentration of $\|Z\|_F^2 \geq n/2$ for GOE matrices, and the bound $\max_{i \neq j} |(AZ - ZA)_{ij}| \leq 8n^{\epsilon/2 - 0.5}$ on off-diagonal commutator entries.

### Part II: Small-Perturbation ($\sigma = o(n^{-1-\epsilon})$)

This is the most technically involved part, based on a **dual certificate construction**:

1. **Simplified case**: Consider $\sigma = 0$, where $X = I$ is optimal and $AX - XA = 0$.
2. **Dual problem**: Introduce dual variables $(R, \mu, \tilde{\mu}, M)$ and construct an approximately feasible dual solution.
3. **Key construction**:
    - Set $R = J - I$ (off-diagonal entries 1, diagonal entries $-1$).
    - Construct $M = \sum_{i \neq j} \frac{\tilde{w}_{ij}}{\lambda_i - \lambda_j} u_i u_j^T$ using the eigenvectors $\{u_i\}$ of the GOE matrix.
    - $\mu$ is also constructed from eigenvectors, ensuring $\langle \mu, \mathbf{1} \rangle = 0$ (strong duality).
4. **Core inequality (Lemma 6)**: For any $X \in \mathcal{B}_n$:
   $$\sum_{j \neq i} X_{ij} \leq 2n^{3/2 + 7\epsilon/8} \|AX - XA\|_F + 4n^{1 - \epsilon/32}$$
5. **Auxiliary bound (Lemma 7)**: $\|AX^\star - X^\star A\|_F \leq c\sigma\sqrt{n}$.
6. **Combining**: When $\sigma = n^{-1-\epsilon}$, this yields $\sum_{i \neq j} X_{ij}^\star \leq 5n^{1-\epsilon/32}$, and hence $\|X^\star - I\|_F^2 \leq 10n^{1-\epsilon/32}$.

### Refined Eigenvalue Gap Analysis

The proof requires controlling $\sum_{i \neq j} \frac{1}{(|\lambda_i - \lambda_j| + n^{-1-\epsilon})^2}$. A direct bound gives $n^{4+2\epsilon}$ (too loose). By separately treating cases with large and small $|i-j|$, combined with Nguyen-Vu eigenvalue spacing tail bounds and Markov's inequality, a tighter bound of $n^{3+3\epsilon/2}$ is obtained. The regularization term $n^{-1-\epsilon}$ is a key device for ensuring finite expectations.

### Rounding

- **Simple rounding**: $\hat{\pi}(i) = \arg\max_j X_{ij}^\star$ recovers $1 - o(1)$ fraction of correct alignments (Corollary 2).
- **Hungarian projection**: $\tilde{\Pi} = \arg\max_{\Pi \in \mathcal{P}_n} \langle X^\star, \Pi \rangle$ also succeeds (Corollary 3).

## Loss & Training

This is a purely theoretical work with no training procedure. In practice, the Birkhoff relaxation is solved using:

- **Convex optimization solver**: cvxpy + SCS (Splitting Conic Solver) with `use_indirect=True` for large instances.
- **Post-processing**: Hungarian algorithm to project the doubly stochastic matrix onto a permutation matrix.
- **Complexity**: The Birkhoff relaxation is a convex quadratic program solvable in polynomial time; however, computational cost becomes prohibitive for large $n$ (above $n \approx 500$).

## Key Experimental Results

### Experimental Setup

- **Model**: Gaussian Wigner Model
- **Graph size**: $n = 400$ (default); $n \in \{100, 200, 300, 400, 500\}$ (scaling experiments)
- **Noise range**: $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$
- **Baselines**: GRAMPA ($\eta = 0.2$), Simplex relaxation, Birkhoff relaxation
- **Repetitions**: 10 trials per configuration, averaged
- **Hardware**: CPU + 50GB memory; maximum 3 hours per instance

### Main Results

#### Comparison of Three Relaxation Methods ($n = 400$)

| Method | Maximum $\sigma$ for 100% alignment | $\sigma$ at onset of degradation |
|---|---|---|
| **Birkhoff** | **0.5** | 0.6 |
| Simplex | 0.3 | 0.4 |
| GRAMPA | 0.1 | 0.2 |

The Birkhoff relaxation significantly outperforms the other two methods at all noise levels.

### Ablation Study

#### Phase Transition Behavior

- For $\sigma \in [0.3, 0.5]$, even when $\|X^\star - \Pi^\star\|_F / \sqrt{n}$ approaches 1 (i.e., the relaxed solution is far from the true permutation), Hungarian rounding still achieves perfect alignment.
- This indicates that $X^\star$, while globally distant from $\Pi^\star$, still maintains a weak preference toward $\Pi^\star$ sufficient to support successful rounding.

### Key Findings

#### Empirical Estimate of Phase Transition Threshold

- Log-log regression over $n \in \{100, \ldots, 500\}$ gives a slope of approximately $-0.45$ for the transition point where $\|X^\star - \Pi^\star\|_F / \sqrt{n} = 0.5$.
- This is consistent with the theoretical prediction of $\sigma = \Theta(n^{-0.5})$ (Theorem 1).

#### Performance Across Different $n$

As $n$ increases, the rate at which alignment success decays with $\sigma$ accelerates, yet the transition remains gradual, suggesting that Birkhoff relaxation with rounding may succeed at nearly constant $\sigma$.

## Limitations & Future Work

1. **Gap between theory and practice**: Theoretical small-perturbation guarantees require $\sigma = o(n^{-1})$, while experiments show success for $\sigma = O(1)$; a gap of $n^{-0.5}$ to $n^{-1}$ remains between the population phase transition and the current theory.
2. **Model limitations**: Analysis is restricted to the Gaussian Wigner model (continuous weights); the more practically relevant Erdős-Rényi graph (binary edges) is not covered.
3. **Eigenvector concentration**: The Small-Perturbation proof relies heavily on the uniform distribution of GOE eigenvectors on the sphere, which is a bottleneck for extending to general Wigner matrices.
4. **Well-separation is not a failure condition**: $\|X^\star - \Pi^\star\|_F = \Omega(\sqrt{n})$ does not imply rounding failure; the paper does not characterize the precise threshold for rounding success.
5. **Computational scalability**: Solving instances with $n > 500$ via SCS exceeds 3 hours; the method is impractical at large scale.
6. **No real-world validation**: Experiments are conducted solely on synthetic data, without testing in practical scenarios such as social network de-anonymization or protein alignment.

## Highlights & Insights

- **Solid theoretical contribution**: This paper provides the first non-trivial theoretical guarantees for the Birkhoff relaxation under noise (prior results were limited to $\sigma = 0$); the dual certificate construction is technically elegant.
- **Intuition behind dual construction**: Setting $R = J - I$ penalizes all off-diagonal entries, while $M$ and $\mu$ compensate for approximate feasibility residuals. This construction paradigm may transfer to convex relaxation analyses of other combinatorial optimization problems.
- **The population version phase transition is elegant**: $\bar{X}^\star = \epsilon I + \frac{1-\epsilon}{n}J$ with $\epsilon = \frac{2}{2 + \sigma^2(n+1)}$ directly reveals the phase transition at $\sigma\sqrt{n}$.
- **The "redemption effect" of rounding in experiments** merits further investigation: $X^\star$ can be far from $\Pi^\star$ in $\|\cdot\|_F$ yet Hungarian projection still succeeds. Analyzing the post-rounding performance (rather than the relaxed solution itself) may yield a sharper threshold.
- **Practical relevance**: For graph matching applications in machine learning (e.g., graph matching layers in GNNs), the Birkhoff relaxation is a standard component; this paper's theoretical analysis helps characterize its robustness boundary.
- **Future directions**: (1) Close the gap between $n^{-1}$ and $n^{-0.5}$; (2) analyze the precise threshold for successful rounding; (3) extend to the Erdős-Rényi model.

## Rating

- Novelty: ⭐⭐⭐⭐ First phase transition analysis for Birkhoff relaxation under a noise model; novel dual certificate construction methodology.
- Experimental Thoroughness: ⭐⭐⭐ Theory and synthetic experiments are mutually corroborating, but real-world data validation is absent.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, clearly structured, with sufficient intuitive explanations.
- Value: ⭐⭐⭐⭐ An important advance in the theory of convex relaxations for graph matching/QAP, likely to inspire subsequent analyses with sharper thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Topological Descriptors for Graph Products](on_topological_descriptors_for_graph_products.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](../../ICLR2026/others/learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)

</div>

<!-- RELATED:END -->
