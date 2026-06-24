---
title: >-
  [Paper Note] A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints
description: >-
  [ICLR2026][Optimization][Orthogonality constraints] OBCD is proposed as a block coordinate descent algorithm for solving "smooth + nonsmooth" composite optimization under orthogonality constraints (Stiefel manifold). By updating only $k\ge 2$ rows of the solution matrix and reducing the problem to a small-scale $k\times k$ orthogonal subproblem for exact solution, the method ensures strict feasibility and low per-iteration cost. It establishes "block-$k$ stationarity…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Orthogonality constraints"
  - "Stiefel manifold"
  - "Block Coordinate Descent"
  - "Nonsmooth composite optimization"
  - "Sparse PCA"
date: 2026-05-08
content_hash: 33350e292f84fb8f
---

# A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=L3Or2mhuCH](https://openreview.net/forum?id=L3Or2mhuCH)  
**Code**: To be confirmed  
**Area**: optimization  
**Keywords**: Orthogonality constraints, Stiefel manifold, Block Coordinate Descent, Nonsmooth composite optimization, Sparse PCA

## TL;DR
OBCD is proposed as a block coordinate descent algorithm for solving "smooth + nonsmooth" composite optimization under orthogonality constraints (Stiefel manifold). By updating only $k\ge 2$ rows of the solution matrix and reducing the problem to a small-scale $k\times k$ orthogonal subproblem for exact solution, the method ensures strict feasibility and low per-iteration cost. It establishes "block-$k$ stationarity," a stronger optimality guarantee than classic critical points, alongside an $O(1/\epsilon)$ iteration complexity and last-iterate convergence rates under KL conditions.

## Background & Motivation

**Background**: Many models in statistical learning and data science appear as "nonsmooth composite optimization under orthogonality constraints": $\min_{X\in\mathbb{R}^{n\times r}} F(X)=f(X)+h(X)$, s.t. $X^\top X=I_r$, where $f$ is smooth and $h$ is nonsmooth and coordinate-wise separable (e.g., $\ell_0$, $\ell_1$, capped-$\ell_1$, non-negative indicator functions). Typical examples include sparse PCA, non-negative PCA, dictionary learning, orthogonal non-negative matrix factorization, K-indicators clustering, and orthogonal regularization in deep networks. The constraint set $\{X:X^\top X=I_r\}$ is the Stiefel manifold $\mathrm{St}(n,r)$.

**Limitations of Prior Work**: This problem is hampered by two issues: the non-convexity of orthogonality constraints with high projection/retraction costs, and the presence of nonsmooth terms in the objective. Existing methods have drawbacks: geodesic methods require solving ODEs; projection/retraction methods require SVD or polar decomposition at each step; multiplier correction and landing methods often satisfy feasibility only at the limit (remaining "nearly orthogonal" during iterations). Proximal gradient, Riemannian subgradient, and operator splitting (ADMM, RADMM, PSM) methods either rely on full gradients with high per-step costs or lack rigorous last-iterate convergence guarantees.

**Key Challenge**: Almost all existing methods rely on **full gradient information** and handle orthogonality constraints and nonsmooth terms across the **entire $n\times r$ space**, making single steps expensive and subproblems difficult to solve exactly. Furthermore, their optimality guarantees are mostly "critical points" (first-order stationary), which are **weak** and fail to characterize how far a solution is from the global optimum.

**Goal**: Design an algorithm that is (i) computationally cheap per step by updating few coordinates, (ii) strictly feasible on the manifold throughout iterations (descent method), (iii) capable of solving subproblems exactly, and (iv) equipped with stronger optimality characterizations and rigorous convergence rates.

**Key Insight**: The authors adapt classic Block Coordinate Descent (BCD) to the Stiefel manifold. Since the full space is difficult to handle, the algorithm updates only $k$ rows at a time and employs an **orthogonality-preserving** update format. This "updating $k$ rows" approach is equivalent to solving a subproblem on a small $k\times k$ Stiefel manifold $\mathrm{St}(k,k)$. Consequently, both nonsmooth terms and orthogonality constraints are compressed into a low-dimensional space, allowing for exact solutions.

**Core Idea**: Replace full-space updates with a "row-block update $X^{t+1}(B,:)=V\,X^{t}(B,:)$ ($V\in\mathrm{St}(k,k)$)" to transform high-dimensional challenges into exactly solvable $k\times k$ orthogonal subproblems. Based on this, a stronger optimality hierarchy called block-$k$ stationarity is established.

## Method

### Overall Architecture

OBCD (Orthogonality-constrained Block Coordinate Descent) is an iterative algorithm that sequentially minimizes the objective along block-coordinate directions on a sub-manifold of the Stiefel manifold. The inputs are an initial feasible point $X^0\in\mathrm{St}(n,r)$, block size $k\ge 2$, and proximal parameter $\alpha>0$; the output is a sequence $\{X^t\}$ converging to a block-$k$ stationary point.

Each iteration $t$ consists of four steps: (S1) Select a **working set** $B$ of size $k$ from $\{1,\dots,n\}$ (random or cyclic); (S2) Construct a curvature matrix $Q=(Z^\top\!\otimes U_B)^\top H (Z^\top\!\otimes U_B)$ or a diagonal approximation $Q=\varsigma I$; (S3) Construct an upper surrogate function $K(V;X^t,B)$ using Majorization-Minimization (MM) and find the global minimizer $\bar V^t$ over $V\in\mathrm{St}(k,k)$; (S4) Update only these $k$ rows via $X^{t+1}(B,:)\leftarrow \bar V^t X^t(B,:)$. The key is that for any $V\in\mathrm{St}(k,k)$, the updated $X^{t+1}$ automatically remains in $\mathrm{St}(n,r)$, ensuring feasibility and descent without retraction/projection steps.

This is a **partial gradient method**: computing the linear term $\langle [\nabla f(X^t)(X^t)^\top]_{BB}, V\rangle = \langle [\nabla f(X^t)]_{B,:}[X^t]_{B,:}^\top, V\rangle$ in $K$ requires only $k$ rows of $\nabla f$ and $X^t$, making the per-step cost significantly lower than full-gradient methods.

### Key Designs

**1. Orthogonality-preserving row-block update format: Natural feasibility**

Non-convex orthogonality constraints and expensive retraction/projections are primary obstacles. The proposed update format skips retractions entirely. Given a selection matrix $U_B\in\mathbb{R}^{n\times k}$ for the working set $B$ ($|B|=k$), the update is:

$$X^{+}=X+U_B(V-I_k)U_B^\top X,\qquad V\in\mathrm{St}(k,k).$$

This replaces the $k$ rows in $B$ with $V X(B,:)$ while keeping others fixed. Lemma 2.1 proves that for any $X$, $[X^{+}]^\top X^{+}=X^\top X$. Thus, **if $X$ is orthogonal, the updated $X^{+}$ remains orthogonal**. Intuitively, since $V$ is a $k\times k$ orthogonal matrix, an orthogonal linear combination of a set of orthogonal rows does not destroy global orthogonality. This "feasible method" avoids the "feasibility only at the limit" issue of landing/multiplier methods. A supporting lemma provides $\|X^{+}-X\|_F^2=2\langle I_k-V,\,U_B^\top X X^\top U_B\rangle\le \|V-I_k\|_F^2$, facilitating sufficient descent proofs.

**2. MM Majorization reduces subproblems to small $k\times k$ orthogonal optimization**

Directly minimizing $F(X_B^t(V))$ over $V$ remains difficult due to smooth $f$ and nonsmooth $h$. The method uses a quadratic upper bound for the smooth part based on $H$-smoothness while keeping $h$ unchanged to construct the surrogate:

$$K(V;X^t,B)\triangleq \tfrac12\|V-I_k\|_{Q+\alpha I}^2+\langle V,[\nabla f(X^t)(X^t)^\top]_{BB}\rangle+h(VU_B^\top X^t)+\ddot c,$$

satisfying $F(X_B^t(V))\le K(V;X^t,B)$. Utilizing the coordinate-wise separability of $h$, $h(X_B^t(V))=h(U_{B^c}^\top X^t)+h(VU_B^\top X^t)$, leaving only the term $h(VU_B^\top X^t)$ dependent on $V$. The subproblem $\bar V^t\in\arg\min_{V\in\mathrm{St}(k,k)} K(V;X^t,B)$ is simplified to the standard form $\min_{V\in\mathrm{St}(k,k)} \tfrac12\|V\|_{\tilde Q}^2+\langle V,P\rangle+h(VZ)$ (where $\tilde Q=Q+\alpha I$). This handles both nonsmooth $h$ and orthogonality constraints within a low-dimensional $k \times k$ space, which is more robust than full-space proximal steps yet exactly solvable. When $h\equiv 0$ and $Q$ is diagonal, the solution is $\bar V^t=-\mathcal{P}_M(P)$ (the nearest orthogonal matrix of $P$).

**3. $k=2$ Rotation + Reflection + Breakpoint Search: Exact subproblem solution**

The subproblem is cleanest when $k=2$. Lemma 2.5 states that any $V\in\mathrm{St}(2,2)$ can be expressed as a Givens rotation $V_\theta^{\mathrm{rot}}=\big(\begin{smallmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{smallmatrix}\big)$ ($\det=1$) or a Jacobi reflection $V_\theta^{\mathrm{ref}}=\big(\begin{smallmatrix}-\cos\theta & \sin\theta\\ \sin\theta & \cos\theta\end{smallmatrix}\big)$ ($\det=-1$), reducing the subproblem to a **one-dimensional** problem over $\theta$. Even with nonsmooth $h$, the authors use a **breakpoint search** method to locate the optimal $\bar\theta$ (finding extrema across segments where the nonsmooth function is differentiable, implemented with efficient C++ inner loops).

Crucially, both rotation and reflection families must be used. Prior works (e.g., symmetric eigenvalues, sparse PCA) often used only rotations $\{V_\theta^{\mathrm{rot}}\}$, but this paper provides counterexamples showing that rotations alone can miss optimal solutions. Reflective matrices cannot be represented by rotations; they cover the two disconnected components of $\mathrm{St}(2,2)$.

**4. Block-$k$ Stationary Points: Stronger optimality + Rigorous convergence rates**

The theoretical core defines **block-$k$ stationary points (BS$k$-point)**: $\ddot X$ is a BS$k$-point if and only if for all working sets $B$ of size $k$, the identity $I_k$ is a global minimizer of the surrogate $K(V;\ddot X,B)$. This implies the objective cannot be further reduced by optimizing any $k$ rows. Using the fact that any orthogonal matrix is a composition of $2\times 2$ rotations/reflections (Theorem 3.1 / Corollary 3.2), the following optimality hierarchy is proven (Theorem 3.8):

$$\{\text{Critical points}\}\ \supseteq\ \{\text{BS}_2\}\ \supseteq\ \cdots\ \supseteq\ \{\text{BS}_k\}\ \supseteq\ \{\text{BS}_{k+1}\}\ \supseteq\ \{\text{Global Optima}\},$$

meaning larger $k$ indicates stronger stability, and BS$2$ is already strictly stronger than standard critical points. For convergence, sufficient descent $\tfrac{\alpha}{2}\|X^{t+1}-X^t\|_F^2\le F(X^t)-F(X^{t+1})$ is first established, leading to an $O(1/\epsilon)$ iteration complexity for finding an $\epsilon$-BS$k$-point (Theorem 4.2). Complexity for $\epsilon$-critical points under Riemannian subgradient measures is also given (Theorem 4.6). Finally, **last-iterate convergence rates** are established under Kurdyka-Łojasiewicz (KL) conditions (Theorem 4.10–4.11): depending on the KL exponent $\sigma$, rates range from finite convergence ($\sigma=0$) to linear convergence ($\sigma\in(0,\tfrac12]$) and sublinear convergence ($\sigma\in(\tfrac12,1)$).

### Loss & Training

No training required. The algorithm has two main hyperparameters: proximal parameter $\alpha>0$ (controlling descent step size) and block size $k\ge 2$ (larger $k$ offers stronger subproblems and optimality but higher per-step cost; $k=2$ is often preferred for exact one-dimensional solutions). Working sets are selected via random or cyclic strategies.

## Key Experimental Results

### Main Results

The task is $\ell_0$-regularized sparse PCA: $\min_{X\in\mathrm{St}(n,r)} -\langle X,CX\rangle+\lambda\|X\|_0$. Comparison targets include three operator splitting methods: LADMM, RADMM, and PSM, with identity/random initializations. OBCD uses a random working set and identity initialization (OBCD-R(id)). **Metric: Objective value (lower is better), 40s time limit.** Results for $r=20, \lambda=10$:

| Dataset | LADMM(id) | RADMM(id) | PSM(id) | OBCD-R(id) |
|--------|-----------|-----------|---------|------------|
| w1a (2477×300) | 199.897 | 219.698 | 199.897 | **199.667** |
| TDT2 (500×1000) | 199.997 | 359.382 | 199.997 | **199.258** |
| 20News (8000×1000) | 199.995 | 219.673 | 199.995 | **199.222** |
| MNIST (60000×784) | 199.985 | 379.893 | 199.985 | **199.896** |
| Cifar (1000×1000) | 199.979 | 479.979 | 199.979 | **199.974** |

Under a harder setting ($\lambda=50$), the gap widens (e.g., MNIST: LADMM 999.985 vs OBCD **999.896**).

### Ablation Study

| Configuration | Function | Description |
|------|------|------|
| Rotation $V^{\mathrm{rot}}$ only | $k=2$ subproblem | Misses optimal solutions in specific $2\times 2$ cases; higher objective. |
| Rotation + Reflection | $k=2$ subproblem | Covers both branches of $\mathrm{St}(2,2)$; achieves lower objective. |
| Increasing block size $k$ | Optimality strength | BS$k$ hierarchy becomes monotonically stronger (BS$_{k}\supseteq$ BS$_{k+1}$). |

### Key Findings

- OBCD-R(id) achieves the **lowest objective values** across almost all datasets, especially when $\lambda$ is large and the problem is more non-convex, indicating that block-$k$ stationarity captures better solutions than the critical points of ADMM-like methods.
- RADMM/PSM exhibit high variance and often get stuck in poor solutions under random initialization; OBCD is stable due to its "feasible descent" nature.
- Reflections are necessary: using only rotations systematically overestimates the objective.

## Highlights & Insights

- **Orthogonality-preserving row-block updates are the fulcrum**: The update $X^{+}=X+U_B(V-I_k)U_B^\top X$ makes "updating $k$ rows" and "staying on the Stiefel manifold" identical. This eliminates all retraction steps, enabling a feasible descent method that simplifies high-dimensional constraints into low-dimensional subproblems.
- **Simultaneous handling of nonsmoothness and orthogonality**: By stuffing both into a $k\times k$ space, the subproblem becomes exactly solvable. Reducing $k=2$ to a 1D angle problem with breakpoint search is an elegant solution for nonsmoothness.
- **Adjustable optimality hierarchy**: The block-$k$ stationary point provides a controllable "ladder" between critical points and global optima based on block size $k$.
- **Necessity of reflections**: A subtle point often missed—rotations alone cannot cover the negative determinant branch of $\mathrm{St}(2,2)$. The authors prove this necessity with counterexamples.

## Limitations & Future Work

- **Global solvability for $k>2$**: When $k>2$ and $h\ne 0$, the $k\times k$ subproblem may not have a global solution, requiring a local minimizer such that $K(\bar V^t)\le K(I_k)$, which yields weaker guarantees.
- **Implementation overhead**: Breakpoint search for nonsmooth subproblems requires element-wise loops, necessitating C++ for performance; pure MATLAB is slow. Scalability and parallelization (compared to column-wise BCD) require further work.
- **Cost of verifying BS$k$**: Checking if a solution is a BS$k$-point definitively requires solving all $C_n^k$ subproblems; it can only be verified in expectation via random sampling.

## Related Work & Insights

- **vs. Projection/Retraction methods**: Those methods take steps along (Riemannian) gradients and project/retract back. OBCD uses row-block updates that are naturally feasible, requiring only partial gradients and fewer computations.
- **vs. Operator Splitting (ADMM/PSM)**: Splitting methods involve auxiliary variables and linear constraints, are often only feasible at the limit, and only reach critical points. OBCD is strictly feasible, has $O(1/\epsilon)$ complexity, and reaches stronger block-$k$ stationary points.
- **vs. Previous Column-wise BCD**: Prior works were limited to smooth objectives, $k=2$, and $r=n$ using only rotations. This work provides a **row-wise** BCD supporting nonsmooth $h$, $k\ge 2$, $r\le n$, and uses both rotations and reflections.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Orthogonality-preserving updates + block-$k$ stationarity hierarchy).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Solid results for PCA, though tasks are specific to PCA variants).
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous three-part proof: algorithm-optimality-convergence).
- Value: ⭐⭐⭐⭐ (Provides a solid new framework for exactly solvable nonsmooth optimization on Stiefel manifolds).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[ICLR 2026\] Composite Optimization with Error Feedback: the Dual Averaging Approach](composite_optimization_with_error_feedback_the_dual_averaging_approach.md)
- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](../../ICML2026/optimization/mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](../../ICML2026/optimization/multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[ICLR 2026\] Arbitrary-Order Block SignSGD for Memory-Efficient LLM Fine-Tuning](arbitrary-order_block_signsgd_for_memory-efficient_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->
