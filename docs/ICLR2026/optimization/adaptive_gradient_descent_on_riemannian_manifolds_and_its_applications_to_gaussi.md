---
title: >-
  [Paper Note] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference
description: >-
  [ICLR2026][Optimization][Riemannian optimization] This paper proposes RAdaGD—a family of Riemannian adaptive gradient descent methods that do not require line searches. By automatically adjusting step sizes through online estimation of local smoothness constants, RAdaGD achieves a non-ergodic convergence rate of $f(x_k)-f(x^\star)\le O(1/k)$ under the weak assumptions of "local geodesic smoothness + generalized geodesic convexity." Based on this…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Riemannian optimization"
  - "adaptive step size"
  - "convergence rate"
  - "Gaussian variational inference"
  - "Bures-Wasserstein geometry"
date: 2026-05-08
content_hash: d75c25e72648ae4a
---

# Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=2TTQjRkgFn](https://openreview.net/forum?id=2TTQjRkgFn)  
**Code**: https://github.com/wldyddl5510/RAdaGD  
**Area**: optimization  
**Keywords**: Riemannian optimization, adaptive step size, convergence rate, Gaussian variational inference, Bures-Wasserstein geometry

## TL;DR
This paper proposes RAdaGD—a family of Riemannian adaptive gradient descent methods that do not require line searches. By automatically adjusting step sizes through online estimation of local smoothness constants, RAdaGD achieves a non-ergodic convergence rate of $f(x_k)-f(x^\star)\le O(1/k)$ under the weak assumptions of "local geodesic smoothness + generalized geodesic convexity." Based on this, it provides the first convergence guarantee for Gaussian variational inference when the target log-density **does not satisfy global L-smoothness**.

## Background & Motivation
**Background**: Riemannian optimization generalizes Euclidean space optimization to curved manifolds (e.g., positive definite matrix manifolds, hyperbolic spaces, Gaussian measure spaces) and is widely used in machine learning, computer vision, and statistical inference. The standard method for solving $\min_{x\in N} f(x)$ is Riemannian Gradient Descent (RGD): $x_{k+1}=\exp_{x_k}(-s_k\,\mathrm{Grad}\,f(x_k))$, where $\exp$ is the exponential map and $\mathrm{Grad}$ is the Riemannian gradient. When $f$ is geodesically $L$-smooth and geodesically convex, RGD achieves an $O(1/k)$ convergence rate with a fixed step size $s_k=1/L$.

**Limitations of Prior Work**: RGD with a fixed step size $1/L$ has two major drawbacks: (i) "geodesic $L$-smoothness + geodesic convexity" are strong assumptions, and many natural functions (e.g., the squared distance function $x\mapsto d^2(x,p)$ on non-Euclidean Hadamard manifolds) do not satisfy global geodesic $L$-smoothness; (ii) even if satisfied, the smoothness constant $L$ must be known in advance, which is often unknown or difficult to estimate in practice.

**Key Challenge**: In Euclidean spaces, several "adaptive" algorithms have emerged (Malitsky–Mishchenko, Suh–Ma, etc.) that automatically adjust step sizes without line searches or prior knowledge of $L$. However, porting this adaptive mechanism to Riemannian manifolds is hindered by the additional dimension of manifold **curvature**. Curvature introduces an additional factor $\zeta$ related to curvature into key inequalities like "co-coercivity," preventing a direct translation of Euclidean analysis. Consequently, Riemannian adaptive methods were previously largely unexplored.

**Goal**: (1) Develop a family of Riemannian adaptive gradient descent methods that require no line search and no prior knowledge of $L$; (2) Analyze them under weaker assumptions than $L$-smoothness; (3) Apply them to a practical application—Gaussian Variational Inference (GVI)—to provide convergence guarantees for non-$L$-smooth targets.

**Key Insight**: The authors note that the core of the Euclidean adaptive method by Suh–Ma (2025) is "estimating the local smoothness constant $L_{k+1}$ using adjacent steps to adjust the step size." This approach essentially relies on **local** information, making it naturally suitable for relaxation to "local geodesic smoothness"—a condition much weaker than global $L$-smoothness and **satisfied by all $C^2$ functions**. The difficulty lies in cleanly incorporating the curvature factor $\zeta$ into the Lyapunov analysis.

**Core Idea**: Replace the fixed $1/L$ step size with an online-estimated local smoothness constant $L_{k+1}$, and introduce a third auxiliary sequence $\tilde B_k$ specifically to absorb the curvature term $\zeta_k$. This allows migrating the $O(1/k)$ guarantee of Euclidean adaptive methods to Riemannian manifolds and subsequently addressing the non-smooth case in GVI.

## Method

### Overall Architecture
RAdaGD (Riemannian Adaptive Gradient Descent) overall follows the iterative skeleton of RGD, $x_{k+1}=\exp_{x_k}(-s_k\,\mathrm{Grad}\,f(x_k))$, where the sole innovation lies in "how to determine the step size $s_k$." It avoids line searches (which are expensive on manifolds as they require repeated exponential map calculations). Instead, it uses information from the step just completed to **infer** a proxy for the local smoothness constant $L_{k+1}$, and then combines $L_{k+1}$, the previous step size $s_k$, and the curvature factor $\zeta_k$ into the next step size $s_{k+1}$ using three scalar sequences $A_k, B_k, \tilde B_k$.

The inputs are the initial point $x_0$, initial step size $s_0$, and three sequences $\{A_k\}, \{B_k\}, \{\tilde B_k\}$; the output is the iterative sequence $\{x_k\}$ converging to the minimum. The logical chain of the paper is: define "relaxed assumptions" (local geodesic smoothness + generalized geodesic convexity) and prove their broader applicability (Theorem 3.3: all $C^2$ functions are locally geodesically smooth) $\rightarrow$ provide adaptive step size rules (Algorithm 1) $\rightarrow$ prove $O(1/k)$ convergence using a Lyapunov function (Theorems 4.1, 4.5) $\rightarrow$ provide corollaries for three scenarios (known, unknown, or unbounded curvature) $\rightarrow$ apply to GVI on Bures-Wasserstein manifolds (Corollary 5.4, BWAdaGVI). This is a theoretical paper centered on "algorithms + convergence proofs," thus it does not feature a multi-module pipeline diagram.

### Key Designs

**1. Weakening assumptions: Local geodesic smoothness + generalized geodesic convexity replacing global L-smoothness**

Traditional Riemannian analysis requires $f$ to be globally geodesically $L$-smooth, meaning the difference of gradients after parallel transport along a geodesic is uniformly bounded by $L \cdot d(x,y)$. This paper replaces "uniform" with "having a constant on each compact set": $f$ is called **locally geodesically smooth** if for every compact set $K \subseteq M$, there exists $L_K$ such that $\|\Gamma_x^y\,\mathrm{Grad}\,f(x)-\mathrm{Grad}\,f(y)\|_y\le L_K\,d(x,y),\ \forall x,y\in K$ (where $\Gamma_x^y$ is parallel transport). The value of this relaxation is supported by Theorem 3.3: **any $C^2$ function on a complete Riemannian manifold is locally geodesically smooth**. While nearly trivial in Euclidean space, the authors point out that the Riemannian version was not formally stated previously, serving as the "entry ticket" for all subsequent applications. The aforementioned squared distance function $d^2(x,p)$ is not globally $L$-smooth on non-Euclidean Hadamard manifolds but is locally geodesically smooth because it is $C^2$, falling precisely under the new assumptions.

For convexity, **generalized geodesic convexity** is used (convexity along generalized geodesics, originating from Wasserstein spaces): $f(y)\ge f(x)+\langle\Gamma_x^z\,\mathrm{Grad}\,f(x),\,\log_z y-\log_z x\rangle_z$ holds for some base point $z$. Setting $z=x$ reverts to standard geodesic convexity, making this a stronger condition; however, it is natural in sampling and variational inference contexts and provides the flexibility needed for clean RGD analysis. Combining these two assumptions, the authors prove a Riemannian version of **local co-coercivity** (Proposition 3.4), a key lemma for porting Euclidean adaptive analysis.

**2. Adaptive step size rule without line search: Online estimation of local smoothness**

In each step, the algorithm first takes a normal step to obtain $x_{k+1}$, then uses the actual "gradient change vs. function value change" from this step to infer a local smoothness proxy:

$$L_{k+1}=\frac{-\tfrac12\,\big\|\Gamma_{x_{k+1}}^{x_k}\mathrm{Grad}\,f(x_{k+1})-\mathrm{Grad}\,f(x_k)\big\|_{x_k}^2}{f(x_{k+1})-f(x_k)+s_k\big\langle\Gamma_{x_{k+1}}^{x_k}\mathrm{Grad}\,f(x_{k+1}),\,\mathrm{Grad}\,f(x_k)\big\rangle_{x_k}}.$$

Intuitively, $L_{k+1}$ estimates "how steep the current locality is" using adjacent steps while correcting for curvature effects via parallel transport $\Gamma$. After obtaining $L_{k+1}$, the step size update proceeds through two paths: first, calculating two scaling factors

$$r_k^s=\min\Big\{\tfrac{A_{k-1}+1}{A_k},\,\tfrac{A_k}{\tilde B_{k+1}}\Big\},\qquad r_k^L=\begin{cases}\tfrac{A_k}{\tilde B_{k+1}} & s_k<1/L_{k+1}\\[4pt]\big(\tfrac{A_k}{B_k}+\tfrac{\tilde B_{k+1}}{A_k}-1\big) & s_k\ge 1/L_{k+1}\end{cases}$$

then setting $s_{k+1}=\min\{r_k^s\,s_k,\ r_k^L/L_{k+1}\}$. The dual $\min$ design ensures the step size does not grow too rapidly (kept stable by $r_k^s s_k$) while remaining pulled by the local smoothness proxy $1/L_{k+1}$ (matching the current curvature and steepness). The entire process only requires calculating geometric quantities already needed by RGD (gradients, exponential maps, parallel transport), **performing no additional line search**. Compared to line searches that often test multiple candidate step sizes and calculate exponential maps for each, this approach only adds two metric evaluations and one parallel transport per step, maintaining the same asymptotic time complexity as RGD.

**3. Three sequences $A_k, B_k, \tilde B_k$ and absorption of the curvature factor $\zeta$**

RAdaGD is a "family" rather than a single algorithm because the step size rule is parameterized by three scalar sequences $A_k, B_k, \tilde B_k$. These are not arbitrary but are constrained by a set of coupling inequalities (Equation (10) in the paper): $\tfrac{A_k+1}{A_{k+1}}\ge1,\ \tfrac{A_k}{\tilde B_{k+1}}\ge1,\ \big(\tfrac{A_k}{B_k}+\tfrac{\tilde B_{k+1}}{A_k}-1\big)\ge r$. Satisfying these allows a carefully designed Lyapunov function

$$V_k=s_{k+1}A_k(f(x_k)-f^\star)+\tfrac12 s_k^2 B_k\|\mathrm{Grad}\,f(x_k)\|^2+\tfrac12\|\log_{x_{k+1}}x^\star\|^2$$

to be monotonically non-increasing (Proposition 4.2), which in turn implies convergence. The most critical innovation here is the third sequence $\tilde B_k$: it is **newly added** compared to the Euclidean version of Suh–Ma to specifically absorb the curvature factor $\zeta_k=\zeta(d(x_k,x^\star))$. $\zeta$ is defined by the lower bound of sectional curvature $K_{\min}$—$\zeta=1$ for non-negative curvature, and $\zeta(\rho)=\rho\sqrt{-K_{\min}}\coth(\rho\sqrt{-K_{\min}})>1$ for negative curvature, characterizing the additional relaxation caused by "geodesic triangles being fatter than Euclidean ones." $\tilde B_k$ must satisfy $\tilde B_{k+1}\ge B_{k+1}+\zeta_{k+1}$, effectively dedicating a specific portion of the Lyapunov budget to the curvature term.

Setting $A_k=\alpha(k+1)+1+\bar\zeta,\ B_k=\alpha(k+1),\ \tilde B_k=\alpha(k+1)+\bar\zeta$ (where $\alpha\in(0,1]$ and $\bar\zeta=\sup_x\zeta$) yields a representative instance (Theorem 4.1) achieving a non-ergodic convergence rate of $f(x_k)-f^\star\le O(1/k)$, equivalent to fixed-step RGD, with the squared gradient norm converging at $O(1/k^2)$. The authors provide corollaries based on curvature information: $\bar\zeta$ bounded and **known** (Corollary 4.6); $\bar\zeta$ **unbounded** but with bounded initial distance $d(x_0,x^\star)$ (Corollary 4.7, controlling $\zeta_k$ via induction); and $\bar\zeta$ bounded but **unknown** (Corollary 4.8, setting $\tilde B_k-B_k\to\infty$ so Inequality (12) eventually holds). These cover scenarios from positive to strongly negative curvature, and from known to completely unknown curvature. The trade-off is an additional $O(\bar\zeta)$ constant factor in the convergence rate compared to non-adaptive RGD—harmless for positive curvature ($\bar\zeta=1$) but potentially large for strongly negative curvature. The authors admit whether this is an "inherent cost of adaptation or an analytical artifact" remains an open question.

**4. Landing on Gaussian Variational Inference: BWAdaGVI**

The core application is Gaussian Variational Inference (GVI): finding the Gaussian that best approximates the target distribution $\pi\propto e^{-V}$ in the KL sense among all non-singular Gaussians, i.e., $\min_{\mu\in\mathrm{BW}(\mathbb R^n)}F(\mu)=\mathbb E_{X\sim\mu}[V(X)]+\mathbb E_{X\sim\mu}[\log\mu(X)]$. The Gaussian space $\mathrm{BW}(\mathbb R^n)$ can be parameterized by mean and covariance as $\mathbb R^n\times\mathrm{SPD}(n)$, which under **Bures-Wasserstein geometry** is a non-negatively curved product Riemannian manifold, thus $\zeta=1$. Applying RAdaGD (with objective $f=F$) is natural, but the assumptions must be verified: the authors prove that when the potential $V$ is convex (i.e., $\pi$ is log-concave) and satisfies a mild growth condition $|V(X)|\le C(1+\|X\|^p)\exp(a\|X\|^\beta),\ \beta\in[0,2)$, then $F$ is both generalized geodesically convex (Lemma 5.2) and locally geodesically smooth (Proposition 5.3). The latter is non-trivial because $\mathrm{BW}(\mathbb R^n)$ is geodesically incomplete, preventing direct use of Theorem 3.3, which the authors prove separately.

Due to geodesic incompleteness, Corollary 4.6 cannot be directly used; the algorithm requires a slight modification—adding an extra step size clipping $s_{k+1}\leftarrow\min\{s_{k+1},\ (1-\delta)/\max_i\lambda_i(\mathbb E_{X\sim\mu_{k+1}}[\nabla^2 V(X)]-\Sigma_{k+1}^{-1})\}$—and a requirement that the minimum eigenvalue of the iterative covariance is uniformly bounded below $\lambda_{\min}(\Sigma_k)\ge\epsilon$. This yields **BWAdaGVI** (Corollary 5.4), maintaining the same convergence guarantees as Corollary 4.6. The significance is that previous GVI algorithms (Lambert 2022, Diao 2023, etc.) rely on global $L$-smoothness of $V$, which many practical models (e.g., Bayesian Poisson regression, where $V(\theta)=\sum_i(\exp(X_i^T\theta)-Y_iX_i^T\theta)$, which is not globally $L$-smooth due to exponential dependence) do not satisfy. BWAdaGVI replaces $L$-smoothness with "convexity + mild growth," offering what the authors claim is the **first** provable convergence for GVI under non-$L$-smooth targets.

### Loss & Training
The paper presents an optimization algorithm itself rather than a training objective; the "objective function" is the $f$ being optimized (KL divergence $F$ in GVI). In practice, the authors recommend the sequence selection (Lemma 5.1) $A_k=2(k+2)^{0.1}+2,\ B_k=2(k+2)^{0.1},\ \tilde B_k=2(k+2)^{0.1}+1$, which satisfies Corollary 4.6 for non-negative curvature. This gives a more conservative but practically faster $O(1/k^{0.1})$ rate—as the theoretical parameters in Theorem 4.1, while sound, lead to overly conservative step sizes.

## Key Experimental Results

### Main Results
The experiments focus on a representative task that highlights the value of "non-$L$-smoothness": GVI for Bayesian Poisson regression ($\ell=50$ samples, $n=25$ dimensions, $X_i \sim N(0,I_n), Y_i \sim \mathrm{Poisson}(\exp(\theta^TX_i))$). The comparison is against the SOTA FBGVI (Diao et al., 2023). Because $V$ for Poisson regression is not globally $L$-smooth, FBGVI **lacks theoretical guarantees** for this task, whereas RAdaGD/BWAdaGVI has them.

| Task | Metric | Ours (BWAdaGVI) | Prev. SOTA (FBGVI) | Conclusion |
|------|------|------|----------|------|
| Poisson Regression GVI | $F(\mu_k)-F_{\min}$ (step size $s=1, 1/2, 1/4$) | Lower, faster convergence | Slower | Outperforms FBGVI across all step sizes (Fig 1 left) |
| Poisson Regression GVI | $\lambda_{\min}(\Sigma_k)$ over iterations | Always away from 0 | — | Empirically validates the mild eigenvalue lower bound assumption (Fig 1 right) |

### Ablation Study
As this is a theoretical paper, there is no traditional module ablation; the closest equivalent is the **curvature dependency analysis**—explicitly comparing adaptive vs. non-adaptive convergence constants under a simplified $f$ that is geodesically $L$-smooth with $\alpha=1$.

| Setting | Convergence Rate Upper Bound (Leading Constant) | Description |
|------|---------|------|
| RAdaGD (Ours, Theorem 4.1) | $\sim\frac{3+\bar\zeta}{2(k+2+\bar\zeta)}(\cdots)Ld(x_0,x^\star)^2$, includes extra $O(\bar\zeta)$ factor | Adaptive, no prior knowledge of $L$ |
| Classic Fixed-Step RGD (Zhang–Sra 2016) | $\frac{\bar\zeta L}{2(k-1+\bar\zeta)}d(x_0,x^\star)^2$ | Requires prior knowledge of $L$ |

### Key Findings
- **Curvature factor $\zeta$ is the "cost center" of adaptation**: For non-negative curvature $\bar\zeta=1$, RAdaGD is nearly equivalent to fixed-step RGD. For strongly negative curvature, the additional $O(\bar\zeta)$ factor may grow. Whether this factor is an inherent cost or an analytical artifact remains open.
- **Non-L-smoothness is the primary selling point**: On tasks where global $L$-smoothness fails, like Poisson regression, RAdaGD not only provides theoretical guarantees but also empirically outperforms FBGVI. This suggests that relaxing assumptions does not sacrifice performance for universality.
- **Eigenvalue lower bound assumption is mild**: BWAdaGVI theoretically requires $\lambda_{\min}(\Sigma_k)\ge\epsilon$. Figure 1 (right) shows that in practice, the minimum eigenvalue remains far from zero, making the assumption non-restrictive.

## Highlights & Insights
- **Online estimation of local $L$ is clever**: Replacing the fixed $1/L$ with $1/L_{k+1}$ inferred from adjacent steps resolves both the "unknown $L$" and the "too strong global $L$-smoothness" problems. It eliminates line searches while expanding applicability to all $C^2$ functions. This "online calibration of second-order constants via first-order information" is transferable to many algorithms.
- **Third sequence $\tilde B_k$ is key to handling curvature**: The main obstacle in porting Euclidean methods to manifolds is the curvature term $\zeta$. Instead of crudely adjusting the step size, the authors introduce a new sequence to dedicate Lyapunov budget to curvature ($\tilde B_{k+1}\ge B_{k+1}+\zeta_{k+1}$). This design pattern of "dedicating an absorption variable for new coupling terms" is highly insightful.
- **Theorem 3.3 ($C^2 \Rightarrow$ local geodesic smoothness) is a critical pivot**: A seemingly trivial Euclidean proposition that was never formally proven for Riemannian manifolds serves as the foundation for the "relaxed assumptions" and GVI guarantees. It serves as a reminder that filling in "obvious but unproven bridges" can unlock new applications.
- **Three-tier curvature corollaries cover reality**: Providing guarantees for known, unbounded, or unknown $\bar\zeta$ makes the "theoretically beautiful" algorithm "practically usable" under various curvature and prior information conditions.

## Limitations & Future Work
- **Generalized geodesic convexity remains a strong assumption**: The authors admit it is stronger than standard geodesic convexity; while natural in sampling/GVI, it limits the class of applicable functions.
- **$O(\bar\zeta)$ factor may scale with negative curvature**: The curvature constant penalty relative to non-adaptive RGD could become significant in strongly negative manifolds.
- **BWAdaGVI relies on technical assumptions**: It requires a uniform lower bound on eigenvalues $\lambda_{\min}(\Sigma_k)\ge \epsilon$ and that the BW-gradient is analytically or stochastically approximable; the current theory does not cover the stochastic approximation version.
- **Small experimental scale**: The main results only include one $50 \times 25$ Poisson regression GVI case, with more tasks relegated to the appendix.
- **Future Work**: Extending Riemannian adaptive acceleration (Nesterov-type), proximal variants, stochastic versions with controlled error, and introducing retractions/inexact parallel transport for more practical computing are natural next steps.

## Related Work & Insights
- **vs. Fixed-Step RGD (Zhang–Sra 2016)**: The latter requires known $L$ and global geodesic $L$-smoothness; this work estimates $L_{k+1}$ online and only requires local smoothness, significantly expanding the function class at the cost of an $O(\bar\zeta)$ factor in the constant.
- **vs. Suh–Ma (2025) Euclidean adaptive method**: The direct inspiration; the difference lies in handling the curvature term $\zeta$ in both the rule and analysis, necessitating the new third sequence $\tilde B_k$—a "highly non-trivial" extension from Euclidean to Riemannian settings.
- **vs. FBGVI / existing GVI algorithms**: Convergence proofs for previous methods require global $L$-smoothness of the target log-density $V$; this work replaces it with "convexity + mild growth," providing the first provable convergence for non-$L$-smooth GVI.
- **Insights**: This work demonstrates the theoretical paradigm of "filling a basic, seemingly obvious but unproven proposition ($C^2 \Rightarrow$ local smoothness) to relax assumptions and open new applications," as well as the analytical trick of dedicated absorption sequences for cross-setting coupling terms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First Riemannian adaptive method to achieve $O(1/k)$ non-ergodic rate and first convergence guarantee for non-$L$-smooth GVI.
- Experimental Thoroughness: ⭐⭐⭐ Sound theory, but the main text only contains one small-scale Poisson regression case.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from assumption to application; notations are dense, posing a threshold for non-Riemannian experts.
- Value: ⭐⭐⭐⭐ Cleanly ports Euclidean adaptive optimization to manifolds and directly benefits variational inference, bridging theory and application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICLR 2026\] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds](riemannian_zeroth-order_gradient_estimation_with_structure-preserving_metrics_fo.md)
- [\[ICLR 2026\] Corner Gradient Descent](corner_gradient_descent.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](../../NeurIPS2025/optimization/natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

</div>

<!-- RELATED:END -->
