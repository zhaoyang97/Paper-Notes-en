---
title: >-
  [Paper Note] Riemannian Optimization on Relaxed Indicator Matrix Manifold
description: >-
  [ICLR 2026][Optimization & Theory][Ratio Cut] Ours proposes a new relaxation for indicator matrices by relaxing column sum constraints from "equal to a fixed value" to "falling within an interval $(l,u)$." It is proven that this relaxation set forms an $(n-1)c$-dimensional embedded submanifold (RIM manifold). A complete Riemannian optimization toolbox is provided,
tags:
  - ICLR 2026
  - Optimization & Theory
  - Ratio Cut
date: 2026-05-08
content_hash: f16d4b41403f41ab
---
# Riemannian Optimization on Relaxed Indicator Matrix Manifold

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ERJd7dMN6U](https://openreview.net/forum?id=ERJd7dMN6U)  
**Code**: See Appendix H of the paper  
**Area**: Optimization Theory / Riemannian Optimization / Clustering  
**Keywords**: Indicator Matrix, Riemannian Optimization, Manifold, Doubly Stochastic Matrix, Ratio Cut

## TL;DR
Ours proposes a new relaxation for indicator matrices by relaxing column sum constraints from "equal to a fixed value" to "falling within an interval $(l,u)$." It is proven that this relaxation set forms an $(n-1)c$-dimensional embedded submanifold (RIM manifold). A complete Riemannian optimization toolbox is provided, reducing the computational complexity of gradients and Hessians from $O(n^3)$ (required by doubly stochastic manifolds) to $O(n)$. On image denoising and Ratio Cut clustering with millions of variables, the method is 70–200 times faster and achieves superior results compared to doubly stochastic manifold methods.

## Background & Motivation
**Background**: The indicator matrix $F\in\mathrm{Ind}_{n\times c}=\{X\mid X_{ij}\in\{0,1\},\,X\mathbf{1}_c=\mathbf{1}_n\}$ is a core object in machine learning tasks such as clustering and classification. However, direct optimization is a 0-1 programming problem and is NP-hard. In practice, the common approach is to **relax** this discrete constraint into a continuous manifold and optimize over it. Historically, three mainstream relaxations exist: the Stiefel manifold $\{X\mid X^TX=I\}$ (the basis of spectral clustering), the row-stochastic manifold $\{X\mid X\mathbf{1}_c=\mathbf{1}_n,X>0\}$ (Fuzzy K-means), and the recent doubly stochastic manifold $\{X\mid X\mathbf{1}_c=\mathbf{1}_n,X^T\mathbf{1}_n=r,X>0\}$.

**Limitations of Prior Work**: These three relaxations have significant drawbacks. Stiefel manifold optimization requires $O(n^2c)$ and only provides analytical optimal solutions for the form $\mathrm{tr}(F^TLF)$, necessitating K-means post-processing for clustering, which is difficult to scale. Row-stochastic manifolds impose no constraints on column sums, leading to empty clusters or extreme class imbalances, failing to incorporate priors like "number of samples per class." Doubly stochastic manifolds **fix** column sums strictly to a prior distribution $r$; however, the true class distribution of unknown datasets is almost never known beforehand, making the constraint over-strict. More critically, its Riemannian gradient involves $n\times n$ pseudoinverses like $(I-FF^T)^\dagger$, leading to an optimization complexity of $O(n^3)$, which is nearly impossible for millions of variables.

**Key Challenge**: The relaxation must simultaneously satisfy two conflicting requirements: it must allow **flexible injection of category priors** (which row-stochastic cannot do) and be **computationally scalable** (which doubly stochastic cannot do). Tighter constraints (doubly stochastic) allow priors but are expensive and fragile to unknown distributions; looser constraints (row-stochastic) are cheaper but lose class balance information.

**Goal**: Construct a relaxation that allows the amount of injected prior information to be adjustable (transitioning continuously from "completely unknown" to "precisely known column sums") while compressing the core computational complexity to linear.

**Key Insight**: The authors observe that column sums do not need to be a **point** constraint; they can be an **interval** constraint—relaxing $X^T\mathbf{1}_n=r$ to $l<X^T\mathbf{1}_n<u$. The width of the interval corresponds to the strength of the prior: tight intervals for abundant priors and wide intervals for scarce priors. Extreme cases include $l=u$ (doubly stochastic) and $l<0,u>n$ (row-stochastic). This set happens to be an embedded submanifold of Euclidean space, allowing the use of the full Riemannian optimization machinery.

**Core Idea**: Replace "pointwise column sum constraints" with "interval-based column sum constraints" to obtain a unified, adjustable, and $O(n)$ complexity Relaxed Indicator Matrix (RIM) manifold.

## Method

### Overall Architecture
The method follows a chain: "Define Manifold → Equip Riemannian Structure → Provide Retraction Mappings → Apply to Specific Problems." The input is an indicator matrix objective function $H(F)$ to be optimized (e.g., graph cut loss), and the output is the (approximate) optimal solution $F^*$ on the RIM manifold.

The first step is defining the relaxation set:
$$M=\{X\mid X\mathbf{1}_c=\mathbf{1}_n,\ l<X^T\mathbf{1}_n<u,\ X>0\}$$
and proving it is an $(n-1)c$-dimensional embedded submanifold (Lemma 1). The second step is equipping $M$ with a Riemannian metric restricted from the Euclidean inner product. This allows expressing the tangent space, Riemannian gradient, Riemannian connection, and Riemannian Hessian via a unified "projection by subtracting column-wise means" operation (Lemma 2–4), which is the fundamental reason the complexity drops from $O(n^3)$ to $O(n)$. The third step provides three retraction methods to map tangent vectors back to the manifold—Dykstra projection (yielding geodesics), dual gradient ascent, and Sinkhorn variants—and compares their efficiency. The fourth step applies this toolbox to graph cut problems like Max Cut / Ratio Cut, providing closed-form expressions for Euclidean gradients/Hessians and proving Lipschitz continuity and convergence. Based on the toolbox, three solvers are implemented: Riemannian Gradient Descent (RIMRGD), Riemannian Conjugate Gradient (RIMRCG), and Riemannian Trust Region (RIMRTR).

The core value lies in the closed-form derivations at each layer; thus, the key designs are detailed below with formulas.

### Key Designs

**1. Interval Relaxation and RIM Manifold: Unifying Three Relaxations with Adjustable Intervals**

Addressing the pain point that "point constraints on column sums are too strict and fragile for unknown distributions," ours replaces $X^T\mathbf{1}_n=r$ with $l<X^T\mathbf{1}_n<u$, yielding $M=\{X\mid X\mathbf{1}_c=\mathbf{1}_n,\ l<X^T\mathbf{1}_n<u,\ X>0\}$. The brilliance of this design is that "one parameterized framework subsumes three old relaxations": when $l<0,u>n$, the interval constraint is inactive, and $M$ degrades to a row-stochastic manifold; when $u=l=r$, the interval shrinks to a point, and $M$ degrades to a doubly stochastic manifold. Thus, the injection of priors is **continuously adjustable**—tighten $(l,u)$ if much is known, or relax if nothing is known.

The authors further prove (Lemma 1) that this set is an embedded submanifold of Euclidean space with dimension $\dim M=(n-1)c$. "Being a manifold" is crucial: it means one can legitimately invoke Riemannian geometric tools (tangent spaces, gradients, connections, Hessians, geodesics) rather than just performing projections on a general convex constraint set. This is the foundation for all subsequent linear complexity conclusions.

**2. Riemannian Toolbox Restricted from Euclidean Inner Product: Reducing Gradient/Hessian to $O(n)$**

Small double-stochastic manifolds are slow because their Riemannian metric (Fisher information metric) causes $(I-FF^T)^\dagger$, an $n\times n$ pseudoinverse, to appear in the gradient, costing $O(n^3)$ per calculation. Ours takes the opposite approach: **instead of using the Fisher metric, it directly restricts the Euclidean inner product $\langle U,V\rangle=\sum_{ij}U_{ij}V_{ij}$ to the manifold**. This choice collapses all Riemannian quantities into cheap operations like "original Euclidean quantity minus column-wise means."

Specifically, the tangent space is $T_XM=\{U\mid U\mathbf{1}_c=\mathbf{0}\}$ (Lemma 2); the Riemannian gradient is just the Euclidean gradient minus its row-mean component:
$$\mathrm{grad}\,H(F)=\mathrm{Grad}\,H(F)-\tfrac{1}{c}\,\mathrm{Grad}\,H(F)\mathbf{1}_c\mathbf{1}_c^T$$
The Riemannian connection is $\nabla_VU=\bar\nabla_VU-\tfrac{1}{c}\bar\nabla_VU\,\mathbf{1}_c\mathbf{1}_c^T$ (Lemma 3, selecting the unique compatible connection that makes the Hessian self-adjoint), and the Riemannian Hessian is similarly $\mathrm{hess}\,H[V]=\mathrm{Hess}\,H[V]-\tfrac{1}{c}\mathrm{Hess}\,H[V]\mathbf{1}_c\mathbf{1}_c^T$ (Lemma 4). These formulas involve only "summing by columns, dividing by $c$, and copying back to $c$ columns," totaling $2nc$ additions and $n$ divisions. Therefore, the Riemannian gradient and Hessian both require only $O(n)$ (see complexity comparison table), which is $O(n^2)$ times faster than the $O(n^3)$ of doubly stochastic manifolds. This is the engineering core of the paper—theoretically "changing the metric" eliminates the cubic term.

**3. Three Retraction Mappings: Balancing Geodesics and Speed between Projection, Dual, and Sinkhorn**

Given a tangent space, a "retraction" is needed to map the tangent vector $tV$ back to the manifold $M$. Ours provides three paths. The first is **Dykstra projection**: noting $M=\Omega_1\cap\Omega_2\cap\Omega_3$ (intersection of row-sum, column-sum lower bound, and column-sum upper bound closed convex sets), it solves $\arg\min_{F\in M}\|F-(X+tV)\|_F^2$ via alternating projections. The row-sum projection is $\mathrm{Proj}_{\Omega_1}(X)=(X_{ij}+\eta_i)_+$, and the column-sum bound projection is piecewise (unchanged if satisfying constraints, shifted if boundary is crossed, see Eq. 7). Theorem 1 further proves that this orthogonal projection retraction satisfies $R_X(0)=X$, $\frac{d}{dt}R_X(tV)|_{t=0}=V$, and $\frac{D}{dt}R_X'(tV)|_{t=0}=0$ for sufficiently small $t$, meaning it provides **geodesics**—offering better convergence guarantees for optimization.

The second is **dual gradient ascent** (Theorem 3), writing the projection as a dual problem concerning Lagrange multipliers $\nu,\omega,\rho$ and performing gradient ascent. The third is a **Sinkhorn variant retraction** (Theorem 4), using two diagonal scaling matrices to push $X\odot\exp(tV\oslash X)$ onto the manifold, equivalent to solving a bi-bounded optimal transport problem with entropy regularization—though the authors admit the selection of the entropy coefficient lacks strong justification. Experiments show: Sinkhorn is faster for small matrices, but **Dykstra significantly dominates and yields geodesics** at scale. Thus, Dykstra is recommended for large-scale scenarios. This trio allows users to trade off scale and mathematical properties.

**4. Graph Cut Application and Convergence Guarantees: Connecting the Toolbox to Max/Ratio Cut**

To be useful, the toolbox must connect to real problems. Taking graph cuts as an example: Max Cut loss $H_m(F)=-\mathrm{tr}(F^TSF)$ and Ratio Cut loss $H_r(F)=\mathrm{tr}(F^TLF(F^TF)^{-1})$ are solved by relaxing $F\in\mathrm{Ind}_{n\times c}$ to the RIM manifold. Using the unified formulas from Design 2, the Riemannian gradient for Max Cut is directly $\mathrm{grad}\,H_m(F)=-SF+\tfrac1cSF\mathbf{1}_c\mathbf{1}_c^T$, and the Hessian is written in one line. The Euclidean gradient for Ratio Cut (Theorem 5) is $2\big(LF(F^TF)^{-1}-F(F^TF)^{-1}(F^TLF)(F^TF)^{-1}\big)$; although it contains $(F^TF)^{-1}$, $F^TF\in\mathbb{R}^{c\times c}$ with $c\ll n$, so the inversion takes only $O(c^3)$, maintaining overall linear complexity.

Crucially, Theorem 6 proves that for any graph cut problem of the form $\mathrm{tr}((F^TLF)(F^TWF)^{-1})$, the spectral norm of the Euclidean gradient is bounded $\|\mathrm{Grad}\,H(F)\|_{\mathbb{S}}\le 2(\tfrac{\|L\|_{\mathbb{S}}\sqrt n}{\alpha}+\tfrac{\|W\|_{\mathbb{S}}\|L\|_{\mathbb{S}}n^{3/2}}{\alpha^2})$ where $\alpha=\sigma_{\min}(W)\cdot\tfrac{l^2}{n}$, ensuring $H$ is Lipschitz continuous. Further Lipschitz smoothness and convergence rate analyses are provided (Appendix A.11). This ensures that running Ratio Cut on the RIM manifold is not just empirically fast but has rigorous convergence guarantees.

### Loss & Training
Ours does not train a neural network but directly minimizes the objective function on the RIM manifold. Three Riemannian solvers are implemented based on the toolbox: RIMRGD, RIMRCG, and RIMRTR. When comparing with doubly stochastic manifold methods, the authors deliberately maintain identical line search methods and stopping criteria, so the only difference is the manifold toolbox itself, isolating the gains brought by the "manifold choice."

## Key Experimental Results

### Main Results
**Time and Loss for Ratio Cut Optimization ($l=u$, excerpt from Table 5)**: The solvers on the RIM manifold significantly lead doubly stochastic manifold methods in time, with lower losses.

| Dataset | DSRGD Time/Loss | RIMRCG Time/Loss | RIMRGD Time/Loss |
|--------|------|------|------|
| COIL20 | 8.978 / 28.17 | **0.685** / 27.46 | 1.145 / **24.83** |
| JAFFE | 2.224 / 30.06 | **0.119** / 29.92 | 0.149 / 29.56 |
| USPS20 | 9.238 / 25.52 | **0.735** / 19.91 | 5.257 / **16.46** |
| PalmData25 | 43.39 / 737.1 | 18.77 / 516.3 | 9.506 / **456.0** |

**Convex Optimization (Norm Approximation, RTR Solver, excerpt Table 3)**: RIM manifold crushes the doubly stochastic manifold in both time and final loss, with loss several orders of magnitude lower.

| Size $n$ ($c{=}100$) | RIM Cost | RIM Time | DSM Cost | DSM Time |
|------|------|------|------|------|
| 5000 | 1.31E-19 | 0.990 | 1.78E-09 | 91.40 |
| 10000 | 1.26E-17 | 1.721 | 2.83E-09 | 241.4 |

The authors state that with millions of variables and $l=u$, the RIM manifold is **70–200 times faster** than doubly stochastic manifold methods with lower loss.

### Ablation Study
**Comparison of Execution Time for Three Retraction Methods ($l=u$, excerpt Table 2, in seconds)**: Dykstra's advantage becomes significant as scale increases.

| Config | $n{=}1000,c{=}1000$ | $n{=}10000,c{=}1000$ | Description |
|------|------|------|------|
| Sinkhorn | 0.082 | 3.614 | Fastest for small matrices, slows down for large ones |
| Dual | 0.178 | 56.56 | Slowest for large scale |
| Dykstra | **0.067** | **0.789** | Fastest for large scale and yields geodesics |

**Denoising Metrics (RIM vs DSM, excerpt Table 4)**: RIM outperforms doubly stochastic across various noise/regularization combinations.

| Metric | RIM (0.3,0.3) | DSM (0.3,0.3) |
|------|------|------|
| PSNR↑ | **19.27** | 18.33 |
| SSIM↑ | **0.502** | 0.412 |
| LPIPS↓ | **0.561** | 0.671 |

In clustering (Table 6, ACC/NMI/ARI metrics across 8 real datasets, comparing 10+ algorithms), the Ratio Cut accuracy on the RIM manifold generally exceeds the latest clustering algorithms.

### Key Findings
- **Metric Choice is Key to Speed**: Swapping the Fisher metric for the Euclidean inner product restriction reduces Riemannian gradient/Hessian from $O(n^3)$ to $O(n)$. This is the root cause of the 70–200x speedup, rather than mere implementation optimization.
- **Retraction Methods Shift with Scale**: Sinkhorn is fast for small matrices, but Dykstra is both fast and provides geodesics for large matrices; thus, the authors explicitly recommend Dykstra for large-scale use.
- **Looser Constraints Yield Better Results**: RIM results on denoised images are smoother with fewer artifacts than doubly stochastic (29.77s/loss 1.05e5 vs 85.33s/loss 1.17e5), suggesting that relaxing column sum constraints not only saves time but also helps escape poor local minima caused by over-strict doubly stochastic constraints.

## Highlights & Insights
- **"Interval instead of Point" is a Minimalist yet Unifying Idea**: Simply relaxing an equality constraint to an interval unifies row-stochastic and doubly stochastic manifolds and makes prior injection adjustable—elegantly aligning "prior strength" with "constraint tightness" via the $(l,u)$ parameters.
- **Metric Change Erases the Cubic Term**: The $O(n^3)$ in doubly stochastic comes from the pseudoinverse induced by the Fisher metric. Restricting the Euclidean inner product collapses Riemannian operations into "subtracting column-wise means." The insight that "choosing the right metric is easier than choosing the right algorithm" can transfer to other manifolds with row/column sum constraints.
- **Dykstra Projection is Geodesic**: Proving that orthogonal projection retraction is precisely a geodesic for small steps combines "low-cost alternating projection" with "convergence-guaranteed geodesics," which is the most elegant part of the toolbox.
- **Transferable logic**: Any problem with "fixed row sum + interval column sum + non-negativity" (Optimal Transport, Soft Assignment, Balanced Clustering) can directly utilize this $O(n)$ toolbox.

## Limitations & Future Work
- The authors admit that as a relaxation of an NP-hard indicator matrix problem, ours can only prove the RIM manifold is superior in various aspects but **cannot provide a precise bound between the relaxed optimal and the original discrete optimal**.
- RIM manifold uses "projection to a closed set" as a retraction; for problems strictly requiring an open set ($X>0$ rather than $X\ge0$), an $\varepsilon$ correction term is needed, which is a theoretical imperfection.
- Self-identified limitation: Experiments are concentrated on graph cuts/denoising/clustering. Universality for other indicator matrix tasks like classification and non-graph-cut non-convex objectives requires further verification. A systematic scheme for adaptive selection of $(l,u)$ based on data is not provided.

## Related Work & Insights
- **vs Doubly Stochastic Manifold Methods (DSRGD/DSRCG)**: They nail column sums to prior distributions and require $O(n^3)$ pseudoinverses for Riemannian gradients. Ours relaxes them to intervals and uses a Euclidean metric restriction to reduce the same operations to $O(n)$. When $l=u$, ours can serve as a fast alternative, measuring 70–200x faster with lower loss.
- **vs Row-Stochastic Manifolds (Fuzzy K-means types)**: They do not constraint column sums, risking empty clusters or class imbalance. Ours incorporates category priors via the $(l,u)$ interval, leading to better clustering metrics.
- **vs Stiefel Manifold (Spectral Clustering)**: They require $O(n^2c)$ and only give analytical solutions for $\mathrm{tr}(F^TLF)$, often needing K-means post-processing. Ours optimizes Ratio Cut directly on the relaxed manifold with convergence proofs, avoiding the scale bottleneck of spectral decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "Interval Relaxation + Euclidean Metric Restriction" unifies three relaxations and drops complexity from $O(n^3)$ to $O(n)$, making it a theoretically substantial new construction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers convex/non-convex, denoising/graph cut/clustering, millions of variables, and many baselines, though focused on graph-cut-style tasks.
- Writing Quality: ⭐⭐⭐⭐ The chain of Lemma-Theorem is clear, with complete complexity comparisons and convergence proofs. Some derivations are dense and require a strong background from the reader.
- Value: ⭐⭐⭐⭐⭐ Provides a universal linear-complexity toolbox for a large class of manifold optimization problems with row/column sum constraints, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Landing with the Score: Riemannian Optimization through Denoising](landing_with_the_score_riemannian_optimization_through_denoising.md)
- [\[ICLR 2026\] Scalable Second-Order Riemannian Optimization for K-means Clustering](scalable_second-order_riemannian_optimization_for_k-means_clustering.md)
- [\[ICLR 2026\] FedMC: Federated Manifold Calibration](fedmc_federated_manifold_calibration.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)

</div>

<!-- RELATED:END -->
