---
title: >-
  [Paper Note] Strongly Convex Sets in Riemannian Manifolds
description: >-
  [ICLR 2026][Optimization & Theory][Frank-Wolfe] This paper systematically generalizes the "strong convexity of sets" from Euclidean (Hilbert) space to Riemannian manifolds for the first time. It provides three non-equivalent definitions of strongly convex sets, their implication relationships, a set of constructive identification tools (lower level sets of smooth st
tags:
  - ICLR 2026
  - Optimization & Theory
  - Frank-Wolfe
date: 2026-05-08
content_hash: 4aa7cfedaa2e039f
---
# Strongly Convex Sets in Riemannian Manifolds

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4sDszSYKP6](https://openreview.net/forum?id=4sDszSYKP6)  
**Area**: Optimization Theory / Riemannian Optimization  
**Keywords**: Strongly convex sets, Riemannian manifold, scaling inequality, Frank-Wolfe, linear convergence

## TL;DR
This paper systematically generalizes the "strong convexity of sets" from Euclidean (Hilbert) space to Riemannian manifolds for the first time. It provides three non-equivalent definitions of strongly convex sets, their implication relationships, a set of constructive identification tools (lower level sets of smooth strongly convex functions are strongly convex sets), and proves that the Riemannian Frank-Wolfe (RFW) algorithm achieves **linear convergence** under the (Approximate) Riemannian Scaling Inequality (RSI).

## Background & Motivation

**Background**: In Euclidean/Hilbert spaces, strong convexity is a core structure in convex optimization. Whether applied to norms, functions, or **sets**, it leads to faster convergence, tighter generalization bounds, and stronger concentration inequalities. Specifically, when the constraint set is a strongly convex set, projection-free methods like Frank-Wolfe (FW) can be accelerated from $O(1/T)$ to $O(1/T^2)$, or even achieve linear convergence directly when the optimal point lies outside the set. The strong convexity of sets has multiple **equivalent** characterizations in Hilbert spaces (inner ball definitions, scaling inequalities, etc.), which have been thoroughly studied.

**Limitations of Prior Work**: Modern machine learning increasingly places problems in non-Euclidean spaces—Wasserstein geometry in optimal transport, distributions on manifolds, SPD matrix data, hyperbolic embeddings, and robotic movement skills all rely on Riemannian optimization. However, while "strong convexity of functions" generalizes relatively smoothly to geodesic spaces, there is no universally accepted definition for "**strong convexity of sets**" outside Euclidean space. Several alternative formulations exist in CAT(0)/non-positively curved spaces (Aleksandrov 1957 and subsequent work), but they are **not equivalent** to each other, lead to different corollaries, and fail to directly provide the structural properties required by modern optimization algorithms.

**Key Challenge**: The equivalence of multiple definitions of set strong convexity in Euclidean space depends on "flat" tools such as straight lines, norms, and normal cones. Once moved to curved manifolds, "straight lines" become geodesics, "addition" must go through the exponential map, and parallel transport introduces curvature distortion. Original equivalent definitions split apart, and none can simultaneously satisfy "geometric intuition" and "algorithmic utility."

**Goal**: On uniquely geodesic sets, (1) provide several definitions of strongly convex sets that collapse to classical definitions in the Euclidean limit; (2) clarify the implication relationships between them; (3) provide easy-to-verify identification tools; (4) prove that these structures can be translated into algorithm convergence acceleration.

**Key Insight**: The authors "Riemannianize" the three elements of the Euclidean definition one by one—replacing the line segment $(1-t)x+ty$ with a geodesic $\gamma(t)$, replacing "adding a ball" with adding a ball in the tangent space $T_{\gamma(t)}M$ and pushing it back to the manifold via the exponential map, and replacing the normal cone scaling inequality with a logarithmic map version. This results in a family of definitions. They then use the residual term of the "double exponential map" to quantify the deviation between Euclidean and Riemannian structures.

**Core Idea**: By rewriting three characterizations of Euclidean strongly convex sets using geodesics, exponential/logarithmic maps, and tangent spaces, the authors derive **Geodesic / Riemannian / Double Geodesic** types of Riemannian strongly convex sets. Coupled with the **Riemannian Scaling Inequality (RSI) and its approximate version** as an "algorithm-friendly" bridge, they translate the geometric structure of sets directly into the linear convergence of RFW.

## Method

### Overall Architecture

This is a pure theory paper without a data pipeline. The logic chain follows four steps: "**Definition → Relationship → Identification → Algorithm**."

First, it reviews two classic equivalent definitions of an $\alpha$-strongly convex set in Euclidean space (Proposition 1.1): (a) for any point on a line segment connecting two points $x,y$ in the set, a ball with radius $\propto \alpha t(1-t)\|x-y\|^2$ remains within the set; (b) the scaling inequality $\langle w; v-x\rangle \ge \alpha\|w\|\|v-x\|^2$ (where $v$ is the support point in direction $w$). This work is built on a standard assumption—the set $C$ is compact and uniquely geodesic convex (Assumption 1.2, ensuring the logarithmic map $\mathrm{Exp}_x^{-1}$ is well-defined everywhere), plus a distance equivalence assumption (Assumption 1.3, allowing the use of $d$ different from the Riemannian distance, with constants $\ell_M, L_M$ characterizing the deviation).

Building on this, Section 2 "Riemannianizes" the Euclidean definitions into three types of strongly convex sets (see Key Design 1). The end of Section 2 uses RSI to connect geometric properties to algorithmic analysis (Key Design 2). Section 3 introduces the double exponential map and Approximate RSI to bridge the gap between "Double Geodesic strong convexity" and RSI. Section 4 provides constructive criteria—lower level sets of smooth strongly convex functions are strongly convex sets (Key Design 3). Section 5 applies this to algorithms, proving that RFW converges linearly under (Approximate) RSI (Key Design 4), verified by numerical experiments on spheres and SPD manifolds.

The three definitions form an implication chain from "strong to weak" (Eq. 7):

$$\text{Riemannian str. cvx} \xRightarrow{\text{Hadamard}} \text{Geodesic str. cvx} \xLeftrightarrow{\text{A.1.3}} \text{Double Geodesic str. cvx} \Rightarrow \text{Approx. RSI},$$

Meanwhile, Riemannian str. cvx $\Rightarrow$ RSI $\Rightarrow$ Approx. RSI. Riemannian strong convexity is the strongest and most restrictive (requiring the logarithmic image to be a Euclidean strongly convex set), while Approximate RSI is the weakest but sufficient to support algorithmic convergence.

### Key Designs

**1. Three Definitions of Riemannian Strongly Convex Sets: Leveling Euclidean "Segments + Adding Balls" by Geometric Structure Usage**

Euclidean definitions split on manifolds because "inserting a ball" can be understood either using the manifold's metric (geodesic ball) or the tangent space structure (pushed back via exponential map), which are no longer consistent in curved spaces. The authors provide three levels:

- **Geodesic strong convexity (Definition 2.1)**: Uses only the geodesic metric structure of the manifold without touching the tangent space. It requires that for any geodesic $\gamma$ connecting $x,y\in C$ and $t\in[0,1]$, the metric ball $\{z\in M\mid d(\gamma(t),z)\le \alpha t(1-t)d^2(x,y)\}\subseteq C$. This is the formulation closest to "metric spaces."
- **Riemannian strong convexity (Definition 2.2)**: The strongest tier, utilizing the exponential map. it requires that for each $x\in C$, the logarithmic image $\mathrm{Exp}_x^{-1}(C)\subseteq T_xM$ is an $\alpha$-strongly convex set in the tangent space according to the Euclidean definition (1). Intuitively, this holds only when the manifold curvature is bounded, $C$ is not too large, and the logarithmic map only slightly distorts $C$.
- **Double Geodesic strong convexity (Definition 2.3)**: Uses the tangent space and exponential map but stays closer to the Euclidean algebraic form. For any point $\gamma(t)$ on the geodesic $\gamma$, a ball of radius $\alpha t(1-t)d^{2}(x,y)$ is taken in its tangent space $T_{\gamma(t)}M$. Each point $z$ in the ball falls within $C$ after being pushed back by $\mathrm{Exp}_{\gamma(t)}(z)$. "Double" refers to the point being generated by two geodesics—one connecting $x,y$, and one starting from $\gamma(t)$.

The relationships between the three (Prop. 2.5–2.7) are key outputs: Riemannian strong convexity implies Geodesic strong convexity on Cartan-Hadamard (complete non-positively curved) manifolds (with constant scaled by $\alpha\ell_M/L_M^2$). Geodesic and Double Geodesic **imply each other** under Assumption 1.3 (differing by a factor of $\ell_M$ or $1/L_M$), which is noteworthy because the former depends only on metric structure while the latter depends on manifold structure; they converge under mild conditions. When $M$ reduces to Euclidean space, all three collapse back to (1).

**2. Riemannian Scaling Inequality (RSI) and Approximate RSI: Moving Algorithm-Friendly Support Characterizations to Manifolds**

In Euclidean space, the scaling inequality (Prop. 1.1(b)) is the most convenient strong convexity characterization for analyzing algorithm convergence, but it is no longer **equivalent** to global strong convexity on manifolds. The authors define RSI (Definition 2.4) as the Riemannian counterpart of Euclidean (2): for all $x\in C$ and $w\in T_xM$,

$$\langle w; \mathrm{Exp}_x^{-1}(v)\rangle_x \ge \alpha\|w\|_x\,\|\mathrm{Exp}_x^{-1}(v)\|_x^2,\quad v\in\arg\max_{z\in C}\langle w;\mathrm{Exp}_x^{-1}(z)\rangle_x.$$

It replaces Euclidean inner products/norms with tangent space inner products and replaces $v-x$ with the logarithmic map $\mathrm{Exp}_x^{-1}(v)$. Prop. 2.5 proves that Riemannian strong convexity $\Rightarrow$ RSI, thus translating geometric structure into algorithmically useful inequalities.

However, there is no obvious direct link between (Double) Geodesic strong convexity and RSI. The gap lies in the fact that "concatenating two geodesics" does not equal "adding two vectors in the tangent space." The authors use the **Double Exponential Map** (Definition 3.1) $\mathrm{Exp}_x(u,v):=\mathrm{Exp}_{\mathrm{Exp}_x(u)}(\Gamma^{\mathrm{Exp}_x(u)}_x v)$ (moving along $u$ first, then along $v$ after parallel transport) to characterize this concatenation and write it as an "exponential map operator" $h_x(u,v)=u+v+R_x(u,v)$. The residual $R_x$ measures the deviation from Euclidean addition (explicit Taylor expansions exist for constant curvature spaces like spheres or hyperbolic spaces). Based on this, **Approximate RSI** (Definition 3.2) is defined: adding a residual term $\langle w; r(x)\rangle_x$ to the right side of RSI. Prop. 3.3 proves that Double Geodesic strong convexity $\Rightarrow$ Approximate RSI, where the residual $r(x)$ is given by the values of $R_x$ in specific directions. When $x,v$ are close, the residual is negligible relative to the main term $\alpha\|w\|_x d(x,v)^2$, which is key to the subsequent local linear convergence of the algorithm.

**3. Lower Level Set Criterion: Reducing "Hard-to-Verify" Strong Convexity to "Function Smoothness + Strong Convexity"**

Verifying whether a set is strongly convex according to the definition is often difficult. The authors generalize the classic Euclidean result—that the lower level set $Q_s=\{x\mid f(x)-f^*\le s\}$ of an $L$-smooth, $\mu$-strongly convex function is itself a strongly convex set (Journée et al. 2010)—to manifolds (Theorem 4.2). Specifically, if $f$ is real, geodesic $L$-smooth, and $\mu$-strongly convex on $C$ satisfying Assumption 1.2, $x^*$ satisfies $\nabla f(x^*)=0$, and $Q_s$ is geodesically strictly convex, then $Q_s$ is a geodesically strongly convex set with constant:

$$\alpha = \frac{\mu}{2\,(2sL\max\{\ell_M^{-2};1\})^{1/2}}.$$

The proof relies on a Riemannian smoothness lemma (Lemma 4.1): $\|\nabla f(x)\|_x\le\sqrt{2L(f(x)-f(x^*))}$, derived from function duality tools on manifolds. This criterion is "constructive"—providing an operational path to turn strongly convex functions into strongly convex sets. The paper gives two specific examples: the lower level set of $f(x)=d^2(x_0,x)$ on a unit sphere when $s<(\pi/2)^2$, and the lower level set of the squared distance on SPD matrix manifolds (Cartan-Hadamard). Additionally, Theorem D.4 provides another independent criterion: a sufficiently small geodesic ball on a manifold with bounded curvature is a Riemannian strongly convex set (covering spheres, hyperbolic spaces, Grassmann/Stiefel manifolds, special orthogonal groups, and SPD cones).

**4. Linear Convergence of RFW under (Approximate) RSI: Realizing Geometric Structure as Algorithmic Acceleration**

The structure established in the first three steps finally answers "what is it used for." Riemannian Frank-Wolfe (RFW, Algorithm 1) computes a linear minimization oracle $v_t=\arg\min_{v\in C}\langle\nabla f(x_t);\mathrm{Exp}_{x_t}^{-1}(v)\rangle$ at each step, then updates along a geodesic with a short step. Existing work (Weber & Sra 2023) proved it achieves $O(1/T)$ under convex smoothness and linear convergence when the function is strongly convex and the optimum is internal—but these are "function-side" strong convexity. This paper adds the acceleration brought by "**set-side**" strong convexity:

- **Theorem 5.1 (Exact RSI)**: If $C$ satisfies RSI and the unconstrained optimum of $f$ falls outside $C$ (there exists $c>0$ such that $\min_{x\in C}\|\nabla f(x)\|_x>c$), then RFW converges globally linearly, $f(x_{t+1})-f(x^*)\le (f(x_t)-f(x^*))\cdot\max\{1/2,\,1-\alpha c/(2L)\}$. This translates the Euclidean Demyanov-Rubinov (1970) result to the Riemannian setting.
- **Theorem 5.2 (Approximate RSI)**: When only Approximate RSI is satisfied (with residual satisfying $\|R_x(u,w)\|_x\le C\max\{\|u\|^2\|w\|;\|w\|^2\|u\|\}$), after a burn-in phase guaranteed by the $O(1/T)$ global rate until iteration points are sufficiently close ($d(x_t,v_t)^2$ is smaller than a certain curvature-related threshold), RFW switches to **local linear convergence** with the same rate as above.

The key is: the constants do not need to be known in advance; once conditions are met, RFW automatically switches to the linear rate. When the set is Riemannian strongly convex, the residual bound (12) holds automatically, and both exact and approximate RSI are satisfied. On constant curvature spaces (spheres, hyperbolic), even if exact RSI fails, approximate RSI can still hold—extending the applicability beyond CAT(0).

## Key Experimental Results

As a theoretical paper, the experiments consist of numerical examples to "verify theoretical rates" rather than benchmark comparison tables. The core includes two minimization examples and a comparison table for the three definitions.

### Comparison of Three Strongly Convex Set Definitions (Summary of Table 1)

| Definition | USes Tangent Space | Core Tool | Meaning | Typical Example |
|------|------|------|------|------|
| Geodesic str. cvx | No | Geodesic metric | Geodesic ball around geodesic points exists in set | Same as Double Geodesic |
| Double Geodesic str. cvx | Yes | Exp + Parallel Transport | Tangent space ball via Exp map exists in set | Lower level sets of smooth str. cvx functions |
| Riemannian str. cvx | Yes | Log map | Log image of the set is Euclidean str. cvx | Small geodesic balls on bounded curvature manifolds |

### Numerical Examples

| Example | Manifold / Setting | Observed Phenomenon | Corresponding Theory |
|------|------|------|------|
| Spherical Quadratic Minimization | $S^{n-1}$, $n=500$, $f(x)=\|A(x-x^\star)\|^2/2$, $A\in\mathbb{R}^{250\times500}$, constraint $d(x,x_c)\le R<\pi/2$ | Convergence curve shows two segments: global $O(t^{-1})$ rate, followed by **local linear** rate | Theorem 5.2 |
| Finite Temperature Free Energy Minimization | Geodesic ball constraint on SPD manifold, minimizing $F(X)=E_\mu(X)-TS(X)$ (density matrix) | Trust-region subproblems for linearized energy + exact entropy converge at theoretical rate on str. cvx set | Theorem 4.2 + 5.x |

### Key Findings
- **Two-segment convergence curves** provide core evidence: RFW follows a global $O(1/T)$ rate during the burn-in phase and automatically switches to a local linear rate once iteration points enter the "sufficiently close $x_t,v_t$" region, perfectly matching the "Approximate RSI + burn-in" prediction of Theorem 5.2.
- **Optimum at boundary** is the prerequisite for triggering the linear rate: The experiment deliberately sets $R=0.9\,\mathrm{dist}(x^\star,x_c)$ to ensure the solution falls on the constraint boundary and $\arg\min\|\nabla f\|>c$, corresponding to the assumption that the unconstrained optimum is outside the set.
- **Practically applicable scenarios**: The spherical constrained quadratic problem corresponds to neural network training with hierarchical labels ("classifiers for each class fall on a sphere centered on its superclass classifier"); the SPD example corresponds to finite temperature free energy minimization in electronic structure calculations, showing the theory is not just an abstract generalization.

## Highlights & Insights
- **Decoupling "equivalence" into hierarchies on manifolds**: In Euclidean space, multiple definitions of set strong convexity are equivalent. The authors keenly point out that this equivalence depends on flat structures and splits when curved. They choose not to force a single definition but provide three tiers and clarify their implications—a treatment that is more honest and useful than forcing a unified definition.
- **Double exponential map residual as the finishing touch**: By using $h_x(u,v)=u+v+R_x(u,v)$ to quantify the fact that "concatenating two geodesics $\neq$ adding tangent space vectors" as a controllable residual, the authors bridge the weakened Approximate RSI with algorithm convergence—a key bridge between geometry and algorithms.
- **Constant adaptation**: RFW does not need to know constants like $\alpha,c,C$. It automatically switches to the linear rate once conditions are met, allowing theoretical results to be realized in practice without tedious parameter tuning.
- **Transferability**: The authors explicitly state that all scenarios where strongly convex set structures are utilized in the Euclidean side (generalized power method, online learning, strongification techniques) are expected to be transferable to the Riemannian side, providing a clear future direction.

## Limitations & Future Work
- **Strong Assumptions**: The paper requires sets to be compact and uniquely geodesic convex throughout (Assumption 1.2), which excludes large sets on positively curved manifolds (where the exponential map is no longer injective). Riemannian strong convexity further requires bounded curvature and sufficiently small sets.
- **Constant Dependence on Curvature and Scale**: The strong convexity constant $\alpha\propto \mu/\sqrt{sL}$ in the lower level set criterion degrades as the set grows ($s$ increases), and the burn-in threshold for Approximate RSI also depends on the curvature upper bound $K$. Actual gains are sensitive to geometric dimensions.
- **Verification-oriented Experiments**: Numerical examples aim to "confirm theoretical rates." They are limited in scale and lacking horizontal comparisons with other Riemannian optimization methods (e.g., Riemannian gradient/trust-region) in real-world tasks. Engineering value requires more validation.
- **Explicit Form of Approximate RSI Residual**: Whether the residual expansion (Gavrilov 2007 series) is easy to use and how large the constant $C$ is on general manifolds remain open questions.

## Related Work & Insights
- **vs CAT(0) / Aleksandrov (1957) and other metric space formulations**: CAT(0) literature globally derives convexity from the non-positive curvature of the environment space. This paper is a **set-level refinement**, analyzing how local curvature and exponential maps induce strong convexity in specific subsets, recovering metric convexity in non-positive curvature while extending beyond CAT(0).
- **vs Paris (2020/2021) geodesic convexity in online learning**: They use geodesic convexity but do not quantify set curvature or provide strong convexity conditions for algorithms. This paper adds "set curvature characterization + algorithmically usable conditions."
- **vs proximal smoothness (Davis et al. 2025)**: The two are "dual" in structure—proximal smoothness requires a consistent **upper bound** on all outward tangent directions, while RSI provides a one-sided **lower bound** only at the LMO maximization point of a restricted tangent subset, serving different algorithmic mechanisms.
- **vs RFW analysis by Weber & Sra (2023)**: They proved the RFW rate under "**function** strong convexity." This paper adds acceleration from "**set** strong convexity" (linear rate when optimum is outside, Riemannian counterpart of $O(1/T^2)$ for strongly convex functions), complementing their work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first systematic study of set strong convexity on Riemannian manifolds, filling a conceptual gap in non-Euclidean settings.
- Experimental Thoroughness: ⭐⭐⭐ A theoretical paper with numerical examples confirming rates, but limited in scale and horizontal comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain of "Definition → Relationship → Identification → Algorithm," with theorems well-paired with examples, though proofs are entirely in the appendix.
- Value: ⭐⭐⭐⭐ Provides a structural basis for Riemannian optimization that can be utilized by algorithms, offering a clear migration path for subsequent strongification-type work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds](riemannian_zeroth-order_gradient_estimation_with_structure-preserving_metrics_fo.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](../../AAAI2026/optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[ICLR 2026\] Riemannian Optimization on Relaxed Indicator Matrix Manifold](riemannian_optimization_on_relaxed_indicator_matrix_manifold.md)

</div>

<!-- RELATED:END -->
