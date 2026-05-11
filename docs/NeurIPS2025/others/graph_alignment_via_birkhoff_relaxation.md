---
title: >-
  [Paper Note] Graph Alignment via Birkhoff Relaxation
description: >-
  [NeurIPS 2025][graph alignment] This paper provides the first theoretical guarantees for the Birkhoff relaxation (the tightest convex relaxation of QAP) under the Gaussian Wigner model: when $\sigma = o(n^{-1})$…
tags:
  - "NeurIPS 2025"
  - "graph alignment"
  - "quadratic assignment"
  - "Birkhoff polytope"
  - "convex relaxation"
  - "phase transition"
  - "Gaussian Wigner Model"
date: 2026-05-08
content_hash: ab83ff018fdb4b55
---

# Graph Alignment via Birkhoff Relaxation

**Conference**: NeurIPS 2025
**arXiv**: [2503.05323](https://arxiv.org/abs/2503.05323)
**Authors**: Sushil Mahavir Varma (University of Michigan), Irène Waldspurger (CNRS/INRIA, Université Paris Dauphine), Laurent Massoulié (INRIA, PSL Research University)
**Code**: [GitHub](https://github.com/smv30/convex_rel_for_graph_alignment)
**Area**: Graph Matching / Combinatorial Optimization / Convex Relaxation

---

## TL;DR

This paper provides the first theoretical guarantees for the Birkhoff relaxation of the graph alignment problem (relaxing the permutation matrix constraint to doubly stochastic matrices), proving a phase transition in the Gaussian Wigner model: when $\sigma = o(n^{-1})$, the relaxed solution approximates the true permutation; when $\sigma = \Omega(n^{-0.5})$, the relaxed solution is far from the true permutation.

---

## Background & Motivation

### Problem Formulation

The goal of graph alignment is to find a vertex mapping between two graphs that maximizes edge overlap. This is formalized as a Quadratic Assignment Problem (QAP):

$$\Pi^{\star} = \arg\min_{X \in \mathcal{P}_n} \|AX - XB\|_F^2$$

where $\mathcal{P}_n$ is the set of $n \times n$ permutation matrices and $A, B$ are the (weighted) adjacency matrices of the two graphs. QAP is NP-hard in the worst case, and even approximation is intractable.

### Limitations of Prior Work

Graph alignment has broad applications in network de-anonymization, computational biology, and pattern recognition. Existing convex relaxation methods include:

**GRAMPA** (Fan et al., 2023): Further relaxes the Birkhoff constraint and adds a quadratic regularization term. Theoretically succeeds at $\sigma = O(1/\log n)$, but requires tuning $\eta$ and empirically underperforms the Birkhoff relaxation.

**Simplex relaxation** (Araya & Tyagi, 2024): Retains non-negativity constraints but relaxes row/column sum constraints; theoretical guarantees only hold at $\sigma = 0$ (noiseless setting).

**Spectral method EIG1** (Ganassali et al., 2022): Noise threshold is only $\sigma = \Theta(n^{-7/6})$.

### Core Motivation

The Birkhoff relaxation achieves the best empirical performance but lacks theoretical analysis. This paper fills this gap by replacing the permutation matrix set $\mathcal{P}_n$ with its convex hull — the Birkhoff polytope $\mathcal{B}_n$ (the set of all doubly stochastic matrices):

$$X^{\star} = \arg\min_{X \in \mathcal{B}_n} \|AX - XB\|_F^2$$

---

## Method

### Gaussian Wigner Model

The input matrices $A, B$ are drawn from correlated Gaussian Orthogonal Ensembles (GOE): $B = (\tilde{\Pi}^{\star})^T (A + \sigma Z) \tilde{\Pi}^{\star}$, where $A, Z$ are independent GOE matrices and $\sigma$ controls noise intensity. The correlation is $1/\sqrt{1+\sigma^2}$.

### Main Theorem: Phase Transition (Theorem 1)

For any fixed $\epsilon > 0$, when $n$ is sufficiently large, with probability $1-o(1)$:

- **Well-Separation**: When $\sigma \geq n^{-0.5+\epsilon}$, $\|X^{\star} - \Pi^{\star}\|_F^2 \geq \delta n$, i.e., the relaxed solution is far from the true permutation.
- **Small-Perturbation**: When $\sigma \leq n^{-1-\epsilon}$, $\|X^{\star} - \Pi^{\star}\|_F^2 \leq 10 n^{1-\epsilon/32}$, i.e., the relaxed solution is a small perturbation of the true permutation.

### Corollary: Recovery via Simple Rounding

When $\sigma = o(n^{-1})$, both row-wise argmax rounding ($\hat{\pi}(i) = \arg\max_j X_{ij}^{\star}$) and Hungarian projection correctly match $1-o(1)$ fraction of vertices.

### Key Proof Techniques

**Proof of Part I (Well-Separation)**:
1. Upper-bound the optimal Birkhoff relaxation value using $J/n$ (the uniform matrix): $\|AX^{\star} - X^{\star}B\|_F^2 \leq 9n^{\epsilon}$.
2. Construct a lower bound as a function of $\|X^{\star} - I\|_F$, exploiting $\|Z\|_F^2 \geq n/2$ and the concentration inequality $\max_{i \neq j}|(AZ-ZA)_{ij}| \leq 8n^{\epsilon/2-0.5}$.
3. Combine upper and lower bounds to conclude $\|I - X^{\star}\|_F \geq \sqrt{n}/(16c)$.

**Proof of Part II (Small-Perturbation)**:
1. Consider the dual problem at $\sigma=0$ and construct a dual feasible solution $(R, M, \mu, \tilde{\mu})$.
2. Set $R = J - I$ (all-ones matrix minus identity), and construct $M$ and $\mu$ using the eigenvalue/eigenvector structure of the GOE matrix.
3. Establish a connection between the off-diagonal sum and $\|AX - XA\|_F$ via Lemma 6: $\sum_{j \neq i} X_{ij} \leq 2n^{3/2+7\epsilon/8}\|AX-XA\|_F + 4n^{1-\epsilon/32}$.
4. Prove $\|AX^{\star} - X^{\star}A\|_F \leq c\sigma\sqrt{n}$ via Lemma 7.
5. Refined eigenvalue gap analysis (Claim 10): $\sum_{i \neq j} \frac{1}{(|\lambda_i - \lambda_j| + n^{-1-\epsilon})^2} \leq n^{3+3\epsilon/2}$.

### Intuition from the Population Version

When sample objectives are replaced by their expectations, the optimal solution has a closed form: $\bar{X}^{\star} = \epsilon I + \frac{1-\epsilon}{n}J$, where $\epsilon = \frac{2}{2+\sigma^2(n+1)}$. This reveals a phase transition at $\sigma \sim n^{-0.5}$, suggesting the sufficient condition $\sigma = o(n^{-1})$ may not be tight.

---

## Key Experimental Results

### Table 1: Noise Threshold Comparison Across Convex Relaxation Methods

| Method | Algorithm Type | Noise Threshold |
|------|----------|----------|
| EIG1 (Ganassali et al.) | Top eigenvector alignment | $\sigma = \Theta(n^{-7/6})$ |
| GRAMPA (Fan et al.) | Regularized convex relaxation | $\sigma = O(1/\log n)$ |
| Simplex (Araya & Tyagi) | Simplex relaxation | $\sigma = 0$ |
| **Birkhoff (Ours)** | Birkhoff relaxation | $\sigma = O(n^{-1})$ |

> Although GRAMPA has the most permissive theoretical threshold, it requires tuning $\eta$ and is outperformed in practice by the Birkhoff relaxation.

### Table 2: Summary of Experimental Results ($n=400$, averaged over 10 trials)

| $\sigma$ Range | Birkhoff | GRAMPA | Simplex |
|---------------|----------|--------|---------|
| $0 \sim 0.2$ | 100% matching | Begins to degrade | 100% matching |
| $0.3 \sim 0.5$ | 100% matching | Significantly degraded | Degraded |
| $0.5 \sim 0.7$ | Begins to degrade | Near failure | Near failure |
| $0.8 \sim 1.0$ | Degraded | Fails | Fails |

> The Birkhoff relaxation achieves exact alignment for all vertices at $\sigma$ up to 0.5, far exceeding GRAMPA (degrades at $\sigma \approx 0.2$) and Simplex (degrades at $\sigma \approx 0.4$).

**Key Experimental Findings**:
- Log-log regression shows the phase transition slope is approximately $-0.45$, supporting the conjecture of a phase transition at $\sigma = \Theta(n^{-0.5})$.
- Even when $\|X^{\star} - \Pi^{\star}\|_F / \sqrt{n}$ approaches 1 (i.e., $\sigma \in [0.3, 0.5]$), post-processing via Hungarian projection still succeeds — indicating that $X^{\star}$, while globally far from $\Pi^{\star}$, still retains a preference toward $\Pi^{\star}$ over other permutation matrices.

---

## Highlights & Insights

1. **First theoretical guarantees for Birkhoff relaxation**: Prior to this work, no theoretical results existed for this "tightest" convex relaxation. This paper is the first to prove conditions for success under nonzero noise, bridging the gap between theory and practice.
2. **Elegant phase transition characterization**: The gradual behavior from $\sigma = o(n^{-1})$ to $\sigma = \Omega(n^{-0.5})$ reveals the fundamental nature of relaxation tightness — not a binary success/failure, but a smooth transition.
3. **Technical ingenuity in dual certificate construction**: Using the GOE eigenvector decomposition as a basis, setting $R = J - I$ to bound off-diagonal element sums, and the refined eigenvalue gap analysis (treating $|i-j|$ magnitudes separately) constitute the key technical contributions.
4. **Practical implications**: The Birkhoff relaxation requires no hyperparameter tuning (unlike GRAMPA, which requires selecting $\eta$), and the post-processing step (Hungarian algorithm) remains effective beyond the theoretical boundary, suggesting the practical operating range is wider than the theoretical guarantees.

---

## Limitations & Future Work

1. **Gap between theory and practice**: The sufficient condition $\sigma = o(n^{-1})$ is a factor of $\sqrt{n}$ away from the conjectured optimal threshold $\sigma = o(n^{-0.5})$; the population analysis suggests improvement is possible.
2. **Strong model assumptions**: Analysis is limited to the Gaussian Wigner model (GOE matrices); extension to more practical Erdős-Rényi graph models is hindered by eigenvector concentration requirements (Claim 7 relies on the uniform distribution of GOE eigenvectors).
3. **No theoretical analysis of post-processing gains**: The Well-Separation result ($X^{\star}$ is far from $\Pi^{\star}$ when $\sigma = \Omega(n^{-0.5})$) does not imply failure — experiments show Hungarian projection may still succeed in this regime, but no theoretical analysis is provided.
4. **Computational scalability**: The SCS solver requires up to 50GB of memory and up to 3 hours for instances with $n=500$, limiting practical applicability at large scale.

---

## Related Work & Insights

- **Convex relaxation methods**: GRAMPA (Fan et al., 2023) trades tighter theoretical guarantees for weaker empirical performance via regularization; Simplex relaxation (Araya & Tyagi, 2024) is limited to $\sigma=0$; SDP relaxation (Ling, 2024) analyzes exactness for QAP.
- **Spectral methods**: Umeyama's (1988) eigenvector decomposition approach; spectral alignment of Feizi et al. (2019); top eigenvector alignment EIG1 of Ganassali et al. (2022).
- **Random graph matching**: Exact matching in the Erdős-Rényi model (Mao et al., 2023), tree matching (Ganassali & Massoulié, 2020), degree-based analysis (Ding et al., 2021).
- **Application-driven**: Shape matching (Aflalo et al., 2015), image ordering (Dym et al., 2017), gene sequencing (Fogel et al., 2013), computer vision (Birdal & Simsekli, 2019).

---

## Rating

| Dimension | Score (1-10) | Remarks |
|------|:-----------:|------|
| Theoretical Depth | 9 | Elegant dual certificate construction; rigorous eigenvalue gap analysis |
| Novelty | 8 | First analysis of Birkhoff relaxation behavior under noise |
| Experimental Thoroughness | 7 | Numerical validation limited to the GOE model; no real-world datasets |
| Writing Quality | 8 | Clear structure; sufficient intuition provided via population version analysis |
| Value | 6 | Theoretically elegant but practically limited in scalability; $n=500$ is near the computational ceiling |
| **Overall** | **7.5** | A theoretically solid analysis of convex relaxation for graph matching, with an elegant phase transition characterization |

<!-- Generated by src/gen_stubs.py -->
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
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](../../ICLR2026/others/learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
