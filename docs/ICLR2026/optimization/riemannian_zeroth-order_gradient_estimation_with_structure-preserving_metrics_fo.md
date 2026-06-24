---
title: >-
  [Paper Note] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds
description: >-
  [ICLR 2026][Optimization][Riemannian zeroth-order optimization] To address the failure of zeroth-order estimators on Riemannian manifolds where "geodesic incompleteness" causes the exponential map to push perturbed points out of the manifold, this paper constructs a conformally equivalent metric $g'$ that is geodesically complete while preserving the original stationary point structure. From a purely intrinsic perspective (independent of embeddings)…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Riemannian zeroth-order optimization"
  - "geodesic completeness"
  - "structure-preserving metrics"
  - "sectional curvature"
  - "rejection sampling"
date: 2026-05-08
content_hash: c7820552f6d5aa1b
---

# Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yEfKDCbtv1](https://openreview.net/forum?id=yEfKDCbtv1)  
**Area**: Optimization Theory / Riemannian Optimization / Zeroth-Order Optimization  
**Keywords**: Riemannian zeroth-order optimization, geodesic completeness, structure-preserving metrics, sectional curvature, rejection sampling

## TL;DR
To address the failure of zeroth-order estimators on Riemannian manifolds where "geodesic incompleteness" causes the exponential map to push perturbed points out of the manifold, this paper constructs a conformally equivalent metric $g'$ that is geodesically complete while preserving the original stationary point structure. From a purely intrinsic perspective (independent of embeddings), it derives the mean squared error (MSE) upper bound for the two-point symmetric zeroth-order estimator (revealing its relationship with manifold curvature). Combined with unbiased rejection sampling, it generalizes the optimal convergence complexity of Riemannian zeroth-order SGD from Euclidean metrics to general Riemannian metrics.

## Background & Motivation

**Background**: When the objective function contains black-box modules or non-differentiable external solvers (such as PDE solvers), explicit gradients are unavailable, necessitating zeroth-order methods—estimating gradient directions via differences in function values. On a Riemannian manifold, the classical two-point symmetric zeroth-order estimator is defined as:

$$\hat\nabla f(p) = \frac{f(\exp_p(\mu v)) - f(\exp_p(-\mu v))}{2\mu}\, v,$$

where $v$ is a random direction in the tangent space $T_pM$, $\mu>0$ is the perturbation step size, and $\exp_p$ is the exponential map that maps tangent vectors back to the manifold along geodesics (often approximated by a first-order retraction in practice).

**Limitations of Prior Work**: Manifolds are typically treated as submanifolds of the Euclidean space $\mathbb{R}^n$, inheriting a Euclidean metric $g_E$ to simplify numerical calculations. However, $g_E$ is **not necessarily geodesically complete**—the exponential map $\exp_p$ is not guaranteed to be defined on the entire tangent space $T_pM$. Consequently, if a randomly sampled direction $v$ is too long, $\exp_p(\mu v)$ may fall outside the manifold and become undefined, causing the zeroth-order estimation to fail. The paper provides three real-world examples in this category: mesh optimization (vertices cannot be compressed to triangle boundaries), irrigation sprinkler layout (sprinklers cannot be placed outside field boundaries), and covariance matrix estimation (the set of positive definite matrices $\mathcal{S}^{++}_d$ is an open cone).

**Key Challenge**: Theoretically, one could switch to a geodesically complete metric (the Nomizu–Ozeki theorem guarantees the existence of such a metric for any smooth manifold without boundary) and then find an equivalent complete Euclidean metric using the Nash embedding theorem to apply existing convergence analyses for complete cases. However, the constructive proof of Nash embedding is numerically extremely difficult and impractical for actual optimization algorithms. A gap exists between completeness and computability.

**Goal**: Assuming a given metric $g$ may be geodesically incomplete, design a Riemannian zeroth-order optimization algorithm that can still find stationary points with respect to $g$.

**Key Insight**: Instead of constructing a new "ambient Euclidean space," the authors switch to a geodesically complete metric $g'$ that **still preserves the original stationary points**. They rewrite the entire gradient estimation and convergence analysis in a form that depends only on the manifold's own geometry (intrinsic) rather than any embedding.

**Core Idea**: Replace the original metric with a "structure-preserving metric" $g'=hg$ that is conformally equivalent, geodesically complete, and preserves $\epsilon$-stationary points. Then, analyze the error of the zeroth-order estimation and prove convergence within the intrinsic framework—circumventing incompleteness while allowing stationary point conclusions to be mapped back to the original metric $g$.

## Method

### Overall Architecture
The paper addresses the black-box objective $\min_{p\in M} f(p)=\mathbb{E}_\xi[f(p;\xi)]$, where the metric $g$ of the manifold $(M,g)$ (typically a inherited Euclidean metric) is geodesically incomplete, causing zeroth-order perturbations to potentially go out of bounds. The logic follows a three-step sequence:

First, **Metric Transformation**—proving that for any $g$, a structure-preserving metric $g'=h(p)\,g$ (where $h$ is a positive smooth function) can be constructed such that it is geodesically complete (perturbations always fall within the domain of the exponential map), conformally equivalent (the set of stationary points remains unchanged), and preserves $\epsilon$-stationarity. Second, **Intrinsic Estimation**—under the new metric $g'$ (no longer the Euclidean metric of the original ambient space), the mean squared error (MSE) of the two-point symmetric estimator is re-analyzed. An upper bound is derived containing only manifold curvature without embedding information, and SGD convergence is proved accordingly. Third, **Unbiased Sampling**—since the estimator requires the direction $v$ to be uniformly distributed on the $g$-unit sphere, and naive "Gaussian normalization" has systematic bias in non-Euclidean metrics, the paper provides strictly uniform unbiased sampling via rejection sampling. Finally, a corollary maps $\epsilon$-stationarity under $g'$ back to the original metric $g_E$ under appropriate conditions, matching the optimal complexity of the complete case. As this is a pure optimization theory work without a multi-module pipeline, no architecture diagram is provided.

### Key Designs

**1. Structure-Preserving Metric: A Geodesically Complete Conformal Metric that Retains Stationary Points**

The primary pain point is that the original metric $g$ is geodesically incomplete, meaning perturbation $\mu v$ might send points out of the exponential map's domain. The structure-preserving metric $g'$ defined in the paper (Definition 2.5) must satisfy three criteria: (a) **Geodesic Completeness**—there exists $\rho>0$ such that for any $p$, the domain of the exponential map contains the ball $B_p(\rho)=\{v: \|v\|_g\le\rho\}$; thus, as long as the perturbation step $\mu<\rho$, the perturbed point is always valid. (b) **Conformal Equivalence**—there exists a positive smooth function $h$ such that $g'_p(v,w)=h(p)\,g_p(v,w)$. Conformal scaling does not change the locations where the gradient is zero, so the set of stationary points is preserved. (c) **$\epsilon$-Stationarity Preservation**—any $\epsilon$-stationary point under $g$ (i.e., $\sqrt{g_p(\nabla f,\nabla f)}<\epsilon$) is also an $\epsilon$-stationary point under $g'$.

Theorem 2.6 proves that such a $g'$ exists for any $g$. The construction follows the classical approach of Nomizu–Ozeki (1961) but **additionally adjusts the conformal coefficient $h$** to satisfy the $\epsilon$-stationarity preservation. Intuitively (Figure 1 in the paper), under this metric, a geodesic starting from $p$ never leaves the probability simplex regardless of the direction's length, ensuring zeroth-order perturbations are always legal. Notably, the converse does not necessarily hold—an $\epsilon$-stationary point under $g'$ may not be one under $g$; this asymmetry is handled in the convergence analysis.

**2. Intrinsic Zeroth-Order Gradient Estimation and Curvature-Sensitive MSE Bound**

Switching to $g'$ introduces a new problem: $g'$ is generally no longer the Euclidean metric inherited from the ambient space, rendering previous "embedding-dependent" zeroth-order analyses invalid. The authors respond with a purely intrinsic approach—analyzing the estimator using only the manifold's own structure. Under a geodesically complete general metric $g$, Theorem 2.7 provides the MSE upper bound for the symmetric estimator (Eq. 3) using a unit vector $v$:

$$\mathbb{E}_{v\sim\mathrm{Unif}(S^{d-1})}\Big[\big\|\hat\nabla f(p;v)-\tfrac{1}{d}\nabla f(p)\big\|_p^2\Big]\le \frac{1+\mu^2\kappa^2}{d}\,\|\nabla f(p)\|_p^2 + O(\mu^2),$$

where $v$ is sampled uniformly from the $g$-induced unit sphere $S^{d-1}\subset T_pM$, and $\kappa$ is a uniform upper bound on the absolute value of the sectional curvature of the manifold $(M,g')$ (Assumption 2.4), with bounded third/fourth-order derivatives (Assumption 2.3). The key significance of this bound is that the curvature term $\kappa$ directly quantifies the local geometry's impact on estimator variance. When $\kappa=0$ (flat), the bound reduces to the classical variance expression for zeroth-order estimation in Euclidean space. In other words, **the flatter the chosen metric, the more accurate the estimation**—providing a theoretical guide for choosing the conformal metric. Based on this, Theorem 2.9 further proves SGD convergence under general Riemannian metrics.

**3. Unbiased Rejection Sampling on the $g$-Unit Sphere**

The estimator requires direction $v$ to be **uniformly** distributed on the $g$-unit sphere. Since $g$ defines a bilinear form $g_p(u,v)=u^\top A v$ on the tangent space ($A\succ0$, $A_{ij}=g_p(\partial_{x_i},\partial_{x_j})$), the $g$-unit sphere is equivalent to the ellipsoid $C=\{v\in\mathbb{R}^d: v^\top A v=1\}$. While "sample Gaussian and normalize" is simple in Euclidean space, in non-Euclidean metrics, this rescaling **systematically clusters samples near the minor axes of the ellipsoid** (Figure 2 in the paper), introducing biased estimation—which even causes divergence for logistic objectives.

The paper employs rejection sampling (Algorithm 1): first, perform eigendecomposition $A=Q\Lambda Q^\top$, let $L=Q\Lambda^{-1/2}$, $\lambda_{\max}=\max\mathrm{diag}(\Lambda)$; in each round, sample $z\sim\mathcal{N}(0,I_d)$, normalize $s=z/\|z\|$, obtain candidate $v=Ls$, and sample $u\sim U(0,1)$. Accept if $u<\sqrt{v^\top A^2 v}/\lambda_{\max}$. Proposition 2.8 proves the output strictly follows the uniform distribution on ellipsoid $C$. This unbiased sampling is a prerequisite for the validity of the MSE bound and a key for stable convergence in experiments.

**4. Mapping Stationary Points from $g'$ back to the Original Metric**

Theorem 2.9 gives the convergence rate of SGD under a complete metric $g$ as $\min_t\|\nabla f(p_t)\|_{p_t}^2\lesssim\sqrt{d/T}$ (with $\eta\lesssim\sqrt{d/T}$, $\mu\lesssim \tfrac{1}{d^2}\sqrt{d/T}$). However, the actual starting point is the incomplete Euclidean metric $g_E$. After constructing $g=h g_E$ and achieving convergence under $g$, we obtain an $\epsilon$-stationary point in terms of $g$. Corollary 2.10 provides sufficient conditions to map this back to $g_E$: (a) $g_E$ itself is geodesically complete; or (b) the set of $\epsilon$-stationary points under $g_E$, $K=\{p:\|\nabla_{g_E}f(p)\|_{p,g_E}\le\epsilon\}$, is compact. Under either condition, the conformal coefficient $h$ is uniformly bounded, meaning an $\epsilon$-stationary point under $g$ (subject to constant scaling) is also one under $g_E$, resulting in a final complexity of $T\le O(d/\epsilon^4)$. This generalizes the optimal complexity of Riemannian zeroth-order SGD under Euclidean metrics to the broader class of general Riemannian metrics. Even if neither condition is met, Theorem 2.9 still guarantees convergence, albeit with potentially worse complexity.

## Key Experimental Results

The experiments aim to verify the theory: how sampling bias affects convergence, how curvature affects estimation accuracy, and the effectiveness in real-world geodesically incomplete tasks (mesh optimization).

### Main Results

| Experiment | Comparison | Key Phenomenon | Theoretical Basis |
|------|--------|----------|----------|
| Synthetic · Sampling Bias | Rejection sampling vs. Rescaling (avg. 16 runs) | Rejection sampling is consistently superior; rescaling diverges directly on logistic objectives with same hyperparams | Prop. 2.8 (Unbiasedness) |
| Synthetic · Curvature Impact | 4 conformally equivalent metrics with different curvatures (50,000 trials for MSE) | Smaller sectional curvature $|K(p_0)|$ leads to lower zeroth-order estimation MSE | Thm. 2.7 ($\kappa$ term) |
| Real · Mesh Optimization | Structure-preserving vs. Unconstrained/Reversion/Soft Projection (20,000 steps) | Structure-preserving method reduces error stably throughout; achieves lowest final MSE | Thm. 2.6 + Framework |

Objectives for synthetic experiments: The sampling bias experiment used $f_{\text{quadratic}}$ and $f_{\text{logistic}}$ (with non-Euclidean metric $g_A(u,v)=u^\top A v$). The curvature experiment used KL objective $f_{\text{KL}}(p)=\mathrm{KL}(p\|q)$ and Euclidean objective $f_{\text{Euclidean}}(p)=\tfrac12\|p-q\|^2$ on the probability simplex ($q=\tfrac1d\mathbf{1}_d$).

### Ablation Study (Baseline Comparison for Mesh Optimization)

The task is to optimize $20\times20$ coarse grid node positions to approximate the true solution of the Helmholtz equation $\nabla^2 f=-k^2 f$ ($k=10$) on a $200\times200$ fine grid. Nodes are represented as convex combinations of their six neighbors (residing on a probability simplex, geodesically incomplete).

| Method | Handling Out-of-Bounds | Performance |
|------|----------------|------|
| Unconstrained | None | Frequently violates grid validity; severe oscillations around step 16,000 |
| Reversion | Revert to previous position if OOB | Prevents illegal updates but stagnates after ~8,000 steps |
| Soft Projection | Stepwise shrink $\mu$ until valid | Stable but slow progress; almost no improvement after 14,000 steps |
| Structure-Preserving (Ours) | Warps Riemannian structure so perturbations never OOB | Continuously reduces error; lowest final MSE without instability |

### Key Findings
- Unbiased sampling is the "foundation": Rescaling bias does more than just degrade performance; it causes training to diverge on logistic objectives with the same hyperparameters, indicating that the choice of sampling scheme on non-Euclidean metrics is the watershed between convergence and failure, rather than just an refinement.
- The curvature term $\kappa$ in the MSE bound was directly verified—MSE decreased as curvature decreased across the four conformal metrics, aligning empirical results with theory and supporting the metric design principle of "choosing a flatter manifold."
- Native methods for handling out-of-bounds (reversion/soft projection) are either stagnant or glacial because they are "post-hoc remedies." The structure-preserving metric builds legality into the geometry itself (ex-ante guarantee), thus ensuring both feasibility and convergence efficiency.

## Highlights & Insights
- The work brings "geodesic incompleteness"—a frequently ignored yet ubiquitous issue in real-world problems like mesh, PSD matrices, and layouts—to the forefront and provides a computable solution (constructing $g'$ rather than the impractical Nash embedding).
- The MSE upper bound $\tfrac{1+\mu^2\kappa^2}{d}\|\nabla f\|^2$ cleanly attributes zeroth-order estimation error to the manifold's sectional curvature. The fact that it recovers the classical Euclidean conclusion when $\kappa=0$ provides an elegant "curvature-variance" link and an actionable criterion for metric selection.
- "Uniform sampling on a non-Euclidean unit sphere" is simplified to rejection sampling on an ellipsoid $v^\top A v=1$, which is simple and provably unbiased. This sampling subroutine can be reused in any zeroth-order or stochastic algorithm requiring directional sampling under general Riemannian metrics.
- Conformal equivalence to preserve stationary points plus tuning $h$ to preserve $\epsilon$-stationarity is the key technique for making the "metric switch" lossless: changing the foundation without moving the solution set.

## Limitations & Future Work
- Mapping $\epsilon$-stationary points under $g'$ back to $g_E$ requires additional conditions (completeness of $g_E$ or compactness of the $\epsilon$-stationary set); otherwise, complexity may worsen—"flattening has no free lunch," and there is a trade-off between estimation precision and optimization dynamics.
- The analysis relies on several strong assumptions: bounded Hessian ($L$-smoothness), bounded 3rd/4th-order derivatives, globally bounded sectional curvature, and retraction regularity ($C_{\text{Ret}}$ in Assumption 2.2), which real-world black-box objectives may not satisfy.
- The experimental scale is relatively small (synthetic problems + $20\times20$ mesh), leaning towards theoretical verification. Empirical evidence on high-dimensional, large-scale black-box tasks is missing. Data on how the rejection sampling acceptance rate changes with dimensionality or the condition number of $A$ is also absent.
- While the construction of structure-preserving metrics guarantees existence, the design space for $h$ is large. How to select the optimal metric between "sufficiently flat (small $\kappa$)" and "controllable constant scaling when mapping back to $g_E$" remains an open question.

## Related Work & Insights
- **vs. Standard Riemannian Zeroth-Order Optimization (Li et al. 2023a;b, etc.)**: They typically assume geodesic completeness or rely on Euclidean embeddings to define estimators. This paper does not require completeness and performs purely intrinsic analysis, extending the scope from "manifolds with inherited Euclidean metrics" to "manifolds with general Riemannian metrics."
- **vs. Nomizu–Ozeki + Nash Embedding Route**: Theoretically, one could switch to a complete metric and then use Nash embedding to apply existing analyses, but Nash construction is numerically unfeasible. This paper adapts the Nomizu–Ozeki construction and modifies the conformal coefficient $h$ to additionally preserve $\epsilon$-stationarity, bypassing embedding.
- **vs. Euclidean Zeroth-Order Estimation (Nesterov & Spokoiny 2017)**: The MSE upper bound in this paper reduces to their classical conclusion when $\kappa=0$, making it a generalization of Euclidean zeroth-order variance analysis to Riemannian settings with curvature.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic treatment of Riemannian zeroth-order optimization under geodesic incompleteness; structure-preserving metrics and intrinsic curvature-sensitive bounds are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐ Experiments closely follow the theory but are small-scale and purely justificatory, lacking large-scale black-box tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain from motivation to construction, analysis, and mapping back; theorems and assumptions are well-documented.
- Value: ⭐⭐⭐⭐ Provides provable and computable tools for zeroth-order optimization on incomplete manifolds such as meshes, PSD matrices, and black-box layouts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[ICLR 2026\] Scalable Second-Order Riemannian Optimization for K-means Clustering](scalable_second-order_riemannian_optimization_for_k-means_clustering.md)
- [\[ICLR 2026\] Riemannian Federated Learning via Averaging Gradient Streams](riemannian_federated_learning_via_averaging_gradient_streams.md)
- [\[ICLR 2026\] Unbiased Gradient Estimation for Event Binning via Functional Backpropagation](unbiased_gradient_estimation_for_event_binning_via_functional_backpropagation.md)

</div>

<!-- RELATED:END -->
