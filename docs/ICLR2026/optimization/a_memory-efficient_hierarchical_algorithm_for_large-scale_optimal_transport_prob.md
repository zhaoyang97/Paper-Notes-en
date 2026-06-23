---
title: >-
  [Paper Note] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper proposes HALO—a multiscale hierarchical framework for large-scale optimal transport (OT) problems. By combining "coarse-to-fine warm-start," "active support set pruning," and a "factorization-free first-order LP solver," the framework reduces memory requirements to $O(n)$. On $1024^2$ pixel images, it achiev
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 3eda3d96045be4e0
---
# A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CkOBcyntGd](https://openreview.net/forum?id=CkOBcyntGd)  
**Code**: TBD  
**Area**: optimization  
**Keywords**: Optimal Transport, Wasserstein Distance, Linear Programming, Multiscale Algorithm, Memory-Efficient, GPU Parallelism, PDHG  

## TL;DR
This paper proposes HALO—a multiscale hierarchical framework for large-scale optimal transport (OT) problems. By combining "coarse-to-fine warm-start," "active support set pruning," and a "factorization-free first-order LP solver," the framework reduces memory requirements to $O(n)$. On $1024^2$ pixel images, it achieves an 8.9× speedup and a 70.5% reduction in GPU memory compared to the strongest baselines, while providing a scale-invariant upper bound on iteration complexity.

## Background & Motivation
**Background**: The Wasserstein distance (the Kantorovich formulation of OT) is a powerful tool for measuring the similarity between two distributions, widely used in generative modeling, color transfer, texture synthesis, registration, and domain adaptation. Discrete OT is standardized as a linear program (LP): $\min_{x\ge 0} c^\top x,\ \text{s.t.}\ Ax=q$, where $x=\mathrm{vec}(X)$ is the vectorized coupling matrix, and the constraint matrix is $A\in\mathbb{R}^{(m+n)\times mn}$.

**Limitations of Prior Work**: This LP involves $mn$ variables and $m+n$ constraints. For OT between two images of size $n=r\times r$, the number of variables grows with $n^2$—approaching $10^{12}$ when $r=1024$, making computation and memory overhead unbearable. Existing methods have distinct drawbacks: (1) **Approximate methods** (Sinkhorn entropy regularization, low-rank, sliced Wasserstein) sacrifice accuracy for scalability; (2) **LP acceleration methods** (factorization-free first-order methods like PDHG and its variants) are suitable for GPU parallelism but still face severe memory bottlenecks at ultra-large scales; (3) **Multiscale methods** (ShortCut, multiscale semi-smooth Newton MSSN) can reduce costs in practice but lack iteration complexity bounds and rely on CPU-based LP solvers, limiting parallelism.

**Key Challenge**: It is difficult to simultaneously satisfy memory efficiency, large-scale parallelism, and high precision. GPU-friendly first-order methods are memory-heavy, while memory-efficient multiscale methods are often bottlenecked by CPU solvers and lack theoretical guarantees.

**Goal**: Design a large-scale OT solver that is simultaneously memory-efficient, highly parallelizable, accurate, and possesses iteration complexity guarantees.

**Core Idea**: **[Hierarchical + Sparse]** Leverage two structures of OT—the multiscale pyramid structure (coarse levels warm-start fine levels) and the sparsity of the optimal transport plan (the optimal solution has at most $m+n$ non-zeros). By alternating between "updating the active support set" and "solving the restricted LP on the active set," the workload at each level is compressed into sparse sub-problems, reducing memory requirements to $O(n)$.

## Method

### Overall Architecture
HALO employs a two-level loop structure: the **outer loop** solves OT level-by-level from the coarsest level $L$ to the finest level $0$, using the solution from the previous coarse level for initialization via prolongation (warm-start). The **inner loop** alternates between `updateActive` (updating the active support set $N$) and `solveRestricted` (solving the restricted OT on $N$) at a fixed level until convergence criteria are met. Two structural observations—the coarse-to-fine multiscale representation and the sparsity of transport plans—work together to compress computational complexity and memory needs to $O(n)$.

```mermaid
fixed code blocks
flowchart TD
    A[Construct L+1 levels of<br/>hierarchical OT problems] --> B[Coarsest level L<br/>Solve for x_L,f_L,g_L directly]
    B --> C{Outer Loop: ℓ = L-1 … 0}
    C --> D[Prolongation<br/>Initialize fine level with coarse solution]
    D --> E[Inner Iteration @ Level ℓ]
    E --> F[updateActive<br/>Shielding + Dual violation correction]
    F --> G[solveRestricted<br/>cuPDLPx first-order LP solver]
    G --> H{Converged?}
    H -- No --> F
    H -- Yes --> C
    C -- ℓ=0 Finished --> I[Output optimal coupling x_0]
```

### Key Designs

**1. Coarse-to-fine Multiscale Hierarchical Construction: High-quality initialization via geometric aggregation**  
The efficiency of inner iterations relies heavily on the quality of initial values. HALO utilizes the geometry of OT to generate these. Support points at level $\ell$ consist of representative points from neighborhood groupings at level $\ell-1$, marginal distributions $u^{(\ell)}, v^{(\ell)}$ are aggregated from these groups, and the cost $c^{(\ell)}$ is defined by the distance between representatives. For regular grid images, the hierarchy is built by recursively merging $2\times 2$ pixels using block centroids; for non-grid scenarios like point clouds, spatial partitioning (e.g., 2d-tree/kd-tree) is used. Injecting coarse solutions into fine levels via prolongation allows each level to perform lightweight refinement from a starting point that is already "near-optimal," providing the practical basis for the $O(1)$ iteration bound.

**2. Restricted OT on Active Support Sets: Reducing massive LPs to $m+n$ scale via sparsity**  
Classical LP theory states that an optimal solution $x^*$ has at most $m+n$ non-zero entries. Thus, if $\mathrm{supp}(x^*)$ were known, the problem could be solved only on that set. HALO uses an active support set $N\subset S\times D$ as an estimate of $\mathrm{supp}(x^*)$ and defines a restricted OT: $\min_x c_N^\top x_N,\ \text{s.t.}\ A_N x_N=q,\ x_N\ge 0,\ x_{N^\complement}=0$, where $A_N$ and $c_N$ are sub-blocks of the constraint matrix and cost vector for columns in $N$. Since the true support is unknown, the algorithm alternates between "updating $N$ based on the current solution" and "resolving the restricted OT on the new $N$," gradually approaching the true support. This step cuts the variable scale from $mn$ to $O(n)$, serving as the root of memory reduction.

**3. `updateActive`: Dual support expansion via shielding geometry and dual violation correction**  
How the active support set is updated determines the efficiency of the inner loop. HALO builds upon the shielding mechanism of ShortCut (which uses the shielding condition $c(s,d)+c(s',d')>c(s,d')+c(s',d)$ to determine if $(s',d')$ "shields" the connection from $s$ to $d$) with two key modifications: (i) **Expansion to the entire previous support set** rather than just keeping the support of the previous coupling. While this increases the estimated set size, it yields a scale-invariant iteration complexity bound and ensures that set growth does not destroy sparsity; (ii) **Dual violation correction**—solving the pricing problem $C=\{(s_i,d_j): f_i+g_j>c_{ij}\}$ to collect pairs violating KKT conditions, then using a TopK operator to add the $K=\beta|S|$ entries with the largest dual violations to $N$. Compared to threshold-dependent parameters in MSSN, TopK selection accelerates convergence for difficult instances while strictly controlling sparsity. Ablation shows this correction reduces maximum runtime to 24.3% of the uncorrected version at $r=1024$.

**4. Factorization-free First-order LP Solver + Pock–Chambolle Constant Step Size: GPU-friendly execution without norm estimation**  
Restricted OT is handled by `solveRestricted`, which dominates computation and memory overhead. HALO defaults to cuPDLPx (a PDHG-like first-order method), where the primary operations are matrix-vector multiplications, naturally suited for GPU parallelism without requiring matrix factorization. The paper further proves (Proposition 1) that after Pock–Chambolle rescaling $\tilde B=D_r^{-1}BD_c^{-1}$ (where $D_r=\mathrm{diag}(\sqrt{r_i},\sqrt{c_j})$ and $D_c=\sqrt{2}I$), the constraint matrix $B$ of the restricted OT satisfies $\|\tilde B\|_2=1$. This allows PDHG to use constant step sizes, avoiding norm (power iteration) estimation in each round. Ablations show constant step sizes bring a 1.97× speedup at $r=1024$.

**Theoretical Guarantee**: Under five assumptions—direction coverage, bounded radius, bounded density, uniform Lipschitz, and coupling stability—Theorem 1 provides a **scale-invariant upper bound on iteration complexity**. There exists a constant $C$ such that $x^k$ is globally optimal for all $k\ge C$, and $|N^k|\le C|S|$ (the active set remains sparse). This aligns with numerical observations: the average number of inner iterations per scale never exceeds 2 and actually decreases at finer scales.

## Key Experimental Results

### Main Results (DOTmark 2D Images, $n=m=r^2$)

Comparison with three SOTA solvers: HOT, ShortCut, and M3S (Time in seconds / Memory in GB / Relative gap):

| Metric | Method | r=64 | r=128 | r=256 | r=512 | r=1024 |
|---|---|---|---|---|---|---|
| time(s) | **HALO** | 1.50 | 2.20 | 4.31 | **11.17** | **27.73** |
| | HOT | 1.56 | 2.12 | 14.32 | 78.43 | OOM |
| | ShortCut | 0.25 | 2.41 | 25.74 | 438.14 | TO |
| | M3S | 1.95 | 3.44 | 8.51 | 39.32 | 247.22 |
| mem(GB) | **HALO** | 0.38 | 0.48 | 0.76 | **2.07** | **6.25** |
| | HOT | 0.84 | 1.09 | 3.10 | 19.25 | OOM |
| | M3S | 0.71 | 0.95 | 1.92 | 5.78 | 21.21 |
| gap | **HALO** | 1.2e-6 | 1.5e-5 | 1.4e-5 | – | – |
| | HOT | 6.8e-4 | 6.0e-3 | 3.3e-2 | – | – |
| | M3S | 1.9e-4 | 1.9e-4 | 3.3e-4 | – | – |

At $r=512$, compared to HOT, Ours achieves a 7.02× speedup and 89.2% memory savings with gaps 2–3 orders of magnitude tighter. At $r=1024$, compared to M3S, Ours achieves an 8.92× speedup and 70.5% memory savings with gaps 1–2 orders of magnitude tighter. The runtime curve slope is ≈1 (near-linear growth).

### Ablation Study

**Multiscale Framework + cuPDLPx are both essential** (Time in seconds):

| Multiscale | PDHG | r=32 | r=64 | r=128 | r=256 |
|---|---|---|---|---|---|
| ✓ | ✓ | 0.88 | 1.50 | 2.19 | 4.31 |
| ✓ | ✗(Gurobi) | 0.91 | 2.56 | 27.68 | 159.06 |
| ✗ | ✓ | 2.87 | 128.4 | OOM | OOM |

Disabling cuPDLPx results in a 36.9× slowdown at $r=256$; removing the multiscale framework results in a 85.6× slowdown at $r=64$ and OOM at higher resolutions.

**Dual Violation Correction** (Max runtime in seconds on DOTmark): At $r=1024$, 154.20s with correction vs. 633.34s without, reducing the maximum time to 24.3%.

### Key Findings
- **Verification of $O(1)$ Inner Iteration Bound**: The average number of inner iterations per scale never exceeds 2 and decreases at finer scales, empirically supporting Theorem 1.
- **Runtime Correlation with Pixel Sparsity**: Convergence is fast for low-intensity sparsity (ClassicImages 0.01%), while high sparsity (Shapes 45.3% / Microscopy 42.0%) requires 44–56s, which can be explained by a larger Lipschitz constant $L$ in Theorem 1.
- **Generalization to 3D Point Clouds (ModelNet10)**: Sinkhorn OOMs at $n=2^{16}$, and HiRef fails at $n=2^{19}$, whereas HALO scales to $n=2^{19}$ using only 2.99 GB. At $n=2^{18}$, compared to HiRef, Ours achieves a 1.84× speedup, 83.2% memory savings, and a 24.9% lower transport cost.

## Highlights & Insights
- **Unifying "Theoretical Complexity Bounds" and "GPU Engineering"**: Previous multiscale methods either had theory without parallelism (MSSN relying on CPU) or parallelism without guarantees (single-level PDHG). HALO uses the design "expand to the entire previous support set" to exchange a seemingly larger set for a scale-invariant iteration bound, and uses a factorization-free solver to ensure GPU friendliness.
- **Structural Utilization via Pock–Chambolle Rescaling**: Proving $\|\tilde B\|_2=1$ is a elegant use of structure. It eliminates the overhead of norm estimation per round in PDHG-like methods, fully exploiting the specific structure of the OT constraint matrix.
- **TopK Replacing Threshold Parameters**: Compared to threshold tuning in MSSN, TopK selection for dual violations maintains sparsity while avoiding sensitive hyperparameters, improving engineering robustness.

## Limitations & Future Work
- **Focus on Squared Euclidean Cost + Moderate Dimensions**: Both the theory and method are built on squared Euclidean costs; applicability to general costs or ultra-high dimensional scenarios has not been fully verified.
- **Strong Assumptions for Theorem 1 (Assumptions 4–5)**: The authors admit that uniform Lipschitz and coupling stability are stronger than Assumptions 1–3 and are only "intuitively reasonable" under multiscale warm-start, lacking rigorous characterization for general cases.
- **Slow Performance on High-pixel Sparsity Instances**: Cases like Shapes/Microscopy exhibit an increase in $L$ due to non-convex support singularities, leading to significantly higher execution times, indicating room for optimization in difficult geometries.
- **Dependency on Specific GPU Solver Implementations**: The default binding to cuPDLPx means performance and ecosystem support are tied to the evolution of the underlying solver, though the appendix demonstrates solver flexibility.

## Related Work & Insights
- **Approximate Methods** (Sinkhorn 2013, low-rank HiRef, sliced Wasserstein): HALO avoids sacrificing precision, using restricted LPs to control memory while maintaining accuracy.
- **LP Acceleration Methods** (PDHG/cuPDLPx, semi-smooth Newton, Douglas–Rachford, Halpern iteration HOT): HALO reuses the GPU advantages of factorization-free first-order methods but reduces them from "full-scale" to "sparse sub-problems" via active sets, solving their memory bottlenecks.
- **Multiscale Methods** (ShortCut, MSSN): HALO builds on the shielding mechanism of ShortCut, complements it with missing iteration complexity bounds, and replaces CPU solvers with GPU-friendly first-order methods.
- **Insight**: Beyond the "approximation vs. exact" dichotomy, "leveraging structural sparsity + multiscale initialization" is a general path for scaling exact solvers to massive scales, which could be transferred to other structured LP or convex optimization problems with sparse optimal solutions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematically integrates multiscale warm-start, active set pruning, and factorization-free GPU solvers, providing the first scale-invariant iteration complexity bound for this type of multiscale OT.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 2D images (to $1024^2$) and 3D point clouds (to $2^{19}$), compares against 5 SOTA methods, and includes four sets of ablations (multiscale, solver, dual correction, step size), with empirical support for the theory.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagrams, strong correspondence between theory and numerical results, and distinct contributions. The authors honestly discuss the strength of their assumptions.
- **Value**: ⭐⭐⭐⭐ Memory bottlenecks in large-scale OT are real pain points in downstream tasks like generative modeling and registration. $O(n)$ memory and near-linear scaling are practically significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport](a_scalable_constant-factor_approximation_algorithm_for_w_p_optimal_transport.md)
- [\[ICLR 2026\] HOTA: Hamiltonian Framework for Optimal Transport Advection](hota_hamiltonian_framework_for_optimal_transport_advection.md)
- [\[ICLR 2026\] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization](neuclip_efficient_large-scale_clip_training_with_neural_normalizer_optimization.md)
- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[CVPR 2026\] HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](../../CVPR2026/optimization/hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r.md)

</div>

<!-- RELATED:END -->
