---
title: >-
  [Paper Note] Scalable Second-Order Riemannian Optimization for K-means Clustering
description: >-
  [ICLR 2026][Optimization][K-means] This paper reformulates the non-negativity constrained low-rank K-means SDP relaxation as an **unconstrained smooth optimization problem on a product manifold**. It is solved using a cubic-regularized Riemannian Newton method. Through manifold decomposition, the cost per Newton subproblem is reduced to linear complexity with respect to the number of samples $n$. This allows convergence to a second-order critical point (globally optimal under…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "K-means"
  - "Riemannian optimization"
  - "Semidefinite programming relaxation"
  - "Second-order critical point"
  - "Burer-Monteiro decomposition"
date: 2026-05-08
content_hash: 033d8e6c11957bdf
---

# Scalable Second-Order Riemannian Optimization for K-means Clustering

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FNWNG1ftuw](https://openreview.net/forum?id=FNWNG1ftuw)  
**Area**: Optimization / Clustering  
**Keywords**: K-means, Riemannian optimization, Semidefinite programming relaxation, Second-order critical point, Burer-Monteiro decomposition

## TL;DR
This paper reformulates the non-negativity constrained low-rank K-means SDP relaxation as an **unconstrained smooth optimization problem on a product manifold**. It is solved using a cubic-regularized Riemannian Newton method. Through manifold decomposition, the cost per Newton subproblem is reduced to linear complexity with respect to the number of samples $n$. This allows convergence to a second-order critical point (globally optimal under benign non-convexity assumptions) within $n\cdot\epsilon^{-3/2}\cdot\mathrm{poly}(r,d)$ time, which is 2–4 times faster than first-order SOTA methods while maintaining comparable statistical accuracy.

## Background & Motivation
**Background**: K-means clustering aims to partition $n$ points into $K$ groups to maximize intra-cluster similarity. Heuristic methods like Lloyd's algorithm and spectral clustering lack guarantees for local or global optima. A theoretically grounded path involves writing K-means as a semidefinite programming (SDP) relaxation. When data comes from a well-separated Gaussian Mixture Model (GMM), Peng & Wei (2007) proved the SDP relaxation can recover globally optimal clusters in polynomial time as $\mu\to 0^+$, achieving exact recovery once cluster separation exceeds the information-theoretic threshold.

**Limitations of Prior Work**: SDP (2) requires optimization over an $n\times n$ matrix $Z$ with $O(n^2)$ variables, which is impractical for $n$ samples. A natural alternative is the Burer-Monteiro (BM) decomposition $Z=UU^\top$, reducing variables to factor matrices $U\in\mathbb{R}^{n\times r}$ at $O(n)$ cost. However, this sacrifices convexity, generally only finding critical points that might be spurious local minima or saddle points. Moreover, the non-negative BM form for K-means adds element-wise constraints $U\ge 0$ (for doubly non-negative $Z$). Non-negative BM decomposition is **generally considered to lack benign non-convexity**—testing for copositivity/complete positivity is NP-hard, and spurious local minima often emerge.

**Key Challenge**: Although theory suggests the non-negative form should be riddled with spurious solutions, the authors **empirically and repeatedly observed** that in average cases where SDP solves K-means globally, all second-order critical points of the low-rank non-negative form (3) lie within a neighborhood of some global optimum (formulated as "Benign Non-convexity" Assumption 1). If this assumption holds, solving K-means globally is **equivalent** to finding a second-order critical point of (3). However, in constrained non-convex scenarios, no general-purpose algorithm guarantees convergence to a second-order critical point. The core obstacle is maintaining the **feasibility** of non-convex constraints $UU^\top\mathbf{1}_n=\mathbf{1}_n$ and $\mathrm{tr}(UU^\top)=K$ at each iteration while advancing toward the optimum. Solvers like `fmincon` or `knitro` only guarantee convergence to critical points of a merit function (potentially infeasible for the original problem), and augmented Lagrangian only guarantees local first-order convergence.

**Goal**: Design an algorithm that **guarantees convergence to a second-order critical point** with low per-step cost and scalability to large $n$, while maintaining strict feasibility.

**Key Insight**: In unconstrained non-convex optimization, algorithms like gradient descent and trust-region Newton can globally converge to second-order critical points from any initialization. If the constraints in (3) can be "internalized" into a manifold structure, turning the problem into **unconstrained** smooth optimization on a manifold, one can directly apply these mature algorithms along with their first/second-order convergence guarantees.

**Core Idea**: Rewrite the constraint set as a **product manifold** (projected hypersphere × orthogonal group) that admits cheap second-order retractions. Then, run a cubic-regularized Riemannian Newton method on it, leveraging the "block-diagonal + low-rank" structure of the Riemannian Hessian to compress the Newton subproblem to linear time.

## Method

### Overall Architecture
The method follows a rewriting chain: "relaxation → decomposition → manifoldification → second-order solving." The goal is to transform (3), a **non-negative low-rank problem with non-convex constraints**, into a form efficiently solvable by existing Riemannian second-order algorithms.

First step: The constraint set
$$\mathcal{M}=\{U\in\mathbb{R}^{n\times r}: UU^\top\mathbf{1}_n=\mathbf{1}_n,\ \mathrm{tr}(UU^\top)=K\}$$
is verifiable as a manifold (satisfying Linear Independence Constraint Qualification, LICQ). Running Riemannian optimization directly on $\mathcal{M}$ yields an algorithm similar to Carson et al. (2017), but its fatal flaw is the lack of an efficient retraction—which must be computed every step to keep $U\in\mathcal{M}$. The exponential retraction used by Carson requires $O(n^2)$, severely limiting scalability.

Second step: This is the key turning point. Instead of working directly on $\mathcal{M}$, the authors construct a submersion $\varphi$ from a **product manifold** $\widetilde{\mathcal{M}}=\mathcal{V}\times O_r$ to $\mathcal{M}$, where $\mathcal{V}$ is a projected hypersphere and $O_r$ is the $r\times r$ orthogonal group. Retractions for each component on the product manifold are simple Euclidean projections, reducing the total cost to $O(nr+r^3)$. The problem is then equivalently reformulated as optimizing objective (13) on $\widetilde{\mathcal{M}}$.

Third step: Run a **cubic-regularized Riemannian Newton method** (Theorem 1, converges to a second-order critical point in $O(\epsilon^{-3/2})$ iterations) on $\widetilde{\mathcal{M}}$. While a naive implementation is expensive, this work exploits the block-diagonal + low-rank structure of the Riemannian Hessian and uses binary search to compress each step to linear time in $n$, obtaining an $\epsilon$-second-order critical point in $(n/\epsilon^{1.5})\cdot\mathrm{poly}(r,d,K)$ time.

### Key Designs

**1. Formulating constrained K-means as unconstrained smooth optimization on manifolds: Bringing second-order convergence guarantees**

Addressing the "lack of general second-order convergence in constrained non-convexity," the feasibility constraints are fully absorbed into the manifold geometry. For $\min_{U\in\mathcal{M}} f(U)$ where $\mathcal{M}=\{U: \mathcal{A}(UU^\top)+\mathcal{B}(U)=c\}$, under LICQ (8), every local minimum is an $\epsilon$-second-order critical point (satisfying first-order (9) and tangent space second-order (10) conditions). Riemannian algorithms use problem-specific retraction operators to keep iterates on the feasible set $\mathcal{M}$. Starting from a feasible point $U$, the algorithm moves along a smooth curve $\gamma(t)=R_U(t\dot U)$ and performs a local second-order Taylor expansion (11) using the Riemannian gradient $\mathrm{grad}\,f$ and Riemannian Hessian $\mathrm{Hess}\,f$ to select descent directions. This opens the possibility of "global convergence to first/second-order optima from arbitrary initialization" for K-means—something Carson or NLR could not achieve (the former uses non-smooth objectives + penalties and may get stuck in saddles; the latter only has local linear convergence).

**2. Product manifold submersion: Reducing $O(n^2)$ retraction to $O(nr+r^3)$**

This design is the key to scalability. The authors prove (Theorem 2) there exists a submersion from the product manifold $\widetilde{\mathcal{M}}=\mathcal{V}\times O_r$ to $\mathcal{M}$:
$$\varphi(V,Q)=\hat V Q,\quad \hat V=\begin{bmatrix}\hat{\mathbf 1}_n & V\end{bmatrix},\ \hat{\mathbf 1}_n=n^{-1/2}\mathbf 1_n,$$
where the projected hypersphere $\mathcal{V}=\{V\in\mathbb{R}^{n\times(r-1)}: \mathbf 1_n^\top V=0,\ \mathrm{tr}(VV^\top)=K-1\}$ and the orthogonal group $O_r=\{Q: QQ^\top=I_r\}$. Standard submersion properties ensure that every $\epsilon$-second-order point on $\widetilde{\mathcal{M}}$ is also a $c\epsilon$-second-order point on $\mathcal{M}$ ($c$ being a constant scaling), making the optimization equivalent. The beauty of the product manifold lies in its second-order retraction being the Euclidean projections of its two components:
$$\mathrm{Proj}_{\mathcal V}(V)=\sqrt{K-1}\,\frac{V-n^{-1}\mathbf 1_n\mathbf 1_n^\top V}{\lVert V-n^{-1}\mathbf 1_n\mathbf 1_n^\top V\rVert},\qquad \mathrm{Proj}_{O_r}(Q)=(QQ^\top)^{-1/2}Q,$$
requiring only $O(nr+r^3)$, a massive improvement over $O(n^2)$. The equivalent objective becomes:
$$\min_{(V,Q)\in\widetilde{\mathcal{M}}}\ \langle C, VV^\top\rangle-\mu\sum_{i,j}\log\big(\varphi_{i,j}(V,Q)\big)_+,\qquad C=-XX^\top.$$
The log-penalty necessitates starting from a strictly feasible point $\varphi(V_0,Q_0)>0$ (the authors provide a suitable initialization and prove such points exist only when the search rank is **over-parameterized** $r>K$).

**3. Cubic-regularized Riemannian Newton + Linear-time Newton subproblem: Trading speed for block-diagonal and low-rank structure**

With high-speed retractions, Riemannian gradient descent could reach first-order points in $(n/\epsilon)\cdot\mathrm{poly}$ time. However, the log-penalty makes the landscape **extremely ill-conditioned**, causing standard methods like CG or Riemannian Trust Regions (RTR) to converge poorly. The authors use cubic-regularized Riemannian Newton (Theorem 1, $O(\epsilon^{-3/2})$ iterations), solving the Newton subproblem:
$$\min_{Ap=0}\ g^\top p+\tfrac12 p^\top H p+\tfrac{L}{6}\lVert p\rVert^3$$
efficiently. Here $g, H$ are the vectorized Riemannian gradient and Hessian, and $A$ enforces tangent space constraints. By recognizing that the Riemannian Hessian $H$ has a "block-diagonal + low-rank" structure, the KKT system is solved via **binary search** on the regularization parameter $\lambda$, reducing the complexity of each step to $n\cdot\mathrm{poly}(r,d)$. Combining this with Theorem 1, the total time to reach an $\epsilon$-second-order point is $(n/\epsilon^{1.5})\cdot\mathrm{poly}(r,d,K)$.

### Loss & Training
The objective is (13): the data term $\langle C,VV^\top\rangle$ ($C=-XX^\top$ is the negative Gram matrix) plus a log-barrier penalty $-\mu\sum_{i,j}\log(\varphi_{i,j})_+$. Core hyperparameters: penalty coefficient $\mu$ and search rank $r$. Practice suggests taking $\mu$ small enough to stay in the "benign" phase transition region. $r$ is less sensitive when $\mu$ is small; setting $r=K+1$ is usually sufficient. After convergence, the cluster assignment matrix is uniquely recovered from the block-diagonal membership matrix $Z=UU^\top$ via Lemma 1.

## Key Experimental Results

Datasets: Synthetic Gaussian Mixture Models (GMM) and real-world CyTOF mass cytometry (265,627 cells × 32 markers; sampled as $K=4$ imbalanced clusters with $n=1800$ per Zhuang et al. 2024).

### Main Results

Comparison with first-order SOTA (NLR) and classical baselines on CyTOF data (misclustering rate and Frobenius distance to oracle $Z^\star$):

| Method | Misclustering Rate | Frobenius Dist. to $Z^\star$ | Description |
|------|---------|------------------------------|------|
| Ours (Riemannian Newton) | ≈0, lowest variance | **Lowest** | Same model but more accurate recovery via 2nd-order updates |
| NLR (Zhuang 2024) | ≈0 | Higher | First-order SOTA; low error but poorer ground truth recovery |
| SC / NMF / KM++ | Significantly higher | — | Classical baselines |

Convergence efficiency comparison on GMM ($n=100$, $\gamma=0.8$, $\mu=0.1$):

| Metric | Ours | NLR |
|------|------|-----|
| Iterations to converge | **152** | ≈80,000 |
| Per-step cost | 25–100x NLR step | 1× (single matrix-vector product) |
| Per-step complexity vs $n$ | $O(n)$ | $O(n)$ |
| Total wall-clock time | **2–4x faster** | Baseline |

### Ablation Study

| Configuration / Baseline | Key Results | Description |
|------|---------|------|
| Verifying Assumption 1 | Loss drops to $10^{-10}$, min. eigenvalue of Hessian becomes positive | 2nd-order point ≈ Global optimum; zero error |
| vs Carson et al. 2017 | Misclustering > 30%, $\lVert U_-\rVert$ remains high | Penalty $\lambda$ faces a tradeoff between feasibility and clustering |
| vs Classical RTR | Ours ≈ 360 iterations; RTR stalls > 21,000 | Log-penalty causes extreme ill-conditioning |
| Hyperparameter $\mu$ | Sharp phase transition threshold | $\mu$ must be small enough |
| Search rank $r$ | Insensitive if $\mu$ is small; larger $r$ helps escape with large $\mu$ | Recommend $r=K+1$ |
| Misspecified $K_{\mathrm{mis}}=3,5$ | High NMI maintained | Robust to cluster number misspecification |

### Key Findings
- **Newton's iterative advantage offsets per-step overhead**: A single Newton step is 25–100x more expensive than NLR, but iterations drop by ~3 orders of magnitude. The net gain is a 2–4x total speedup.
- **Strong empirical support for Benign Non-convexity**: Across 50 random initializations, the loss consistently converged to zero and Hessian eigenvalues were non-negative; second-order points are almost always global optima.
- **Ill-conditioning stems from log-barrier**: Generic methods fail; cubic regularization combined with block-diagonal structure is essential for stability.
- **Hyperparameter Robustness**: $r=K+1$ is sufficient when $\mu$ is well-chosen; the method is robust to misspecifying $K$.

## Highlights & Insights
- **Replacing expensive retraction with product manifold submersion**: Lifting $\mathcal{M}$ to "Projected Hypersphere × Orthogonal Group" reduces retraction complexity from $O(n^2)$ to $O(nr+r^3)$. This idea is transferable to other low-rank SDPs with complex constraints.
- **Block-diagonal + Low-rank Hessian → Linear-time Newton step**: This allows second-order methods to achieve per-step costs comparable to first-order methods, providing a counter-example to "second-order methods are impractical."
- **Converting combinatorial clustering to guaranteed continuous optimization**: Under benign non-convexity, global K-means is reduced to "finding a second-order critical point," which is strictly guaranteed by Riemannian algorithms.
- **Binary search for cubic regularization**: Lemma 2 guarantees monotonicity, turning the subproblem into a clean 1D search for $\lambda$.

## Limitations & Future Work
- **Dependence on benign non-convexity assumption**: The global guarantee relies on Assumption 1. While empirically supported, a rigorous proof for the non-negative BM form of K-means is still needed.
- **Dependence on well-separated cases**: Theoretically bound to cases where the SDP can solve the problem; it cannot resolve clusters below the information-theoretic threshold.
- **Sensitivity to $\mu$**: A sharp phase transition exists; $\mu$ must be tuned to locate the benign region.
- **Limited scale validation**: Real-world experiments were at $n=1800$. Performances on massive scales or very high rank $r$ require further investigation.

## Related Work & Insights
- **vs Carson et al. (2017, 1st-order Riemannian)**: Uses non-smooth objectives and $O(n^2)$ retractions. Ours provides feasibility, 2nd-order guarantees, and scalability.
- **vs NLR (Zhuang et al. 2024)**: An augmented Lagrangian 1st-order method with only local linear convergence. Ours is 2-4x faster and more accurate in recovery.
- **vs Hybrid methods**: Methods mixing Riemannian and augmented Lagrangian often break the strict feasibility required for K-means membership; Ours maintains strict feasibility throughout.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)
- [\[ICLR 2026\] Riemannian Optimization on Relaxed Indicator Matrix Manifold](riemannian_optimization_on_relaxed_indicator_matrix_manifold.md)
- [\[ICLR 2026\] Landing with the Score: Riemannian Optimization through Denoising](landing_with_the_score_riemannian_optimization_through_denoising.md)
- [\[ICLR 2026\] Angle k-means: Accelerating Exact k-means via Angular Relationships](angle_k-means.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)
- [\[ICLR 2026\] Angle k-means: Accelerating exact k-means via angular relationships](angle_k-means.md)
- [\[ICLR 2026\] Riemannian Optimization on Relaxed Indicator Matrix Manifold](riemannian_optimization_on_relaxed_indicator_matrix_manifold.md)
- [\[ICLR 2026\] Landing with the Score: Riemannian Optimization through Denoising](landing_with_the_score_riemannian_optimization_through_denoising.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)

</div>

<!-- RELATED:END -->
