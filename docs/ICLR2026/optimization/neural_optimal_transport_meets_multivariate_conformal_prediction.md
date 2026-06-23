---
title: >-
  [Paper Note] Neural Optimal Transport Meets Multivariate Conformal Prediction
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] The authors utilize **Neural Optimal Transport** to learn a continuous, cyclically monotone "vector quantile function" (transporting a reference distribution to a conditional distribution), then use the induced **multivariate rank** as a conformal score to construct multivariate prediction regions that possess finite-s
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 93aa130633075041
---
# Neural Optimal Transport Meets Multivariate Conformal Prediction

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ylaKpd7tmA](https://openreview.net/forum?id=ylaKpd7tmA)  
**Code**: To be confirmed (authors promise open-source upon acceptance)  
**Area**: Optimization / Optimal Transport / Uncertainty Quantification  
**Keywords**: Vector Quantile Regression, Neural Optimal Transport, Multivariate Conformal Prediction, Input Convex Neural Networks, Amortized Optimization  

## TL;DR
The authors utilize **Neural Optimal Transport** to learn a continuous, cyclically monotone "vector quantile function" (transporting a reference distribution to a conditional distribution), then use the induced **multivariate rank** as a conformal score to construct multivariate prediction regions that possess finite-sample coverage guarantees and adapt to the geometric shape of the conditional distribution.

## Background & Motivation
- **Background**: Scalar quantile regression (Koenker) is a cornerstone for characterizing heteroskedasticity, skewness, and tail behaviors; conformal prediction provides distribution-free, finite-sample marginal coverage guarantees. However, both tools are difficult to apply to **multivariate responses** $Y \in \mathbb{R}^d$.
- **Limitations of Prior Work**: There is no natural total ordering in $\mathbb{R}^d$, making quantiles difficult to define. Existing multivariate conformal methods either operate **coordinate-wise** (resulting in conservative rectangular boxes that ignore correlations), **compress multidimensional problems into 1D scalar scores** (with limited ball/box shapes), or rely on **heuristic scores** from deep generative embeddings, which lack theoretical guidance and fail to explicitly exploit the geometry of the joint conditional distribution.
- **Key Challenge**: Optimal transport theory provides the "correct" definition for multivariate ranks/quantiles (viewing quantiles as transport maps from a reference distribution to the law of $Y$, recovering center-outward ranks and nested quantile regions). However, **no prior work has scaled Vector Quantile Regression (VQR) using Neural OT**—existing continuous VQR is restricted by the assumption that the "quantile map is affine to $X$ embeddings," leading to limited expressivity and only providing discrete pointwise solutions rather than a continuous rank function.
- **Goal**: To learn a **continuous, parameterized, and cyclically monotone** Conditional Vector Quantile Function (CVQF) along with its inverse (multivariate rank), and to seamlessly integrate the rank function into conformal prediction to obtain geometrically adaptive and valid prediction regions.
- **Key Insight**: Viewing quantiles as a "transport mapping from a reference distribution to a target distribution" naturally recovers center-outward ranks and nested quantile regions, cleanly extending 1D order statistics to high dimensions.
- **Core Idea**: (1) Extend Neural OT to "conditional" VQR using **Partially Input Convex Neural Networks (PICNN) + Amortized Optimization**, learning convex potentials directly from joint samples; (2) Use the learned multivariate rank norm $\|\hat Q^{-1}_{Y|X}(y,x)\|$ as the **conformal score**, and prove that under radial structures, this "pullback ball" constitutes the volume-optimal Highest Density Region (HPD).

## Method

### Overall Architecture
The method is divided into two stages: first, Neural OT is used to learn a vector quantile mapping $Q_{Y|X}(u,x)=\nabla_u\varphi(u,x)$ that transports a reference distribution $F_U$ (e.g., uniform ball or Gaussian) to the conditional distribution $F_{Y|X=x}$. This mapping is the gradient of a convex potential $\varphi$ (ensuring cyclical monotonicity/invertibility); its inverse $Q^{-1}_{Y|X}(y,x)=\nabla_y\psi(y,x)$ follows $F_U$ conditional on $X$. In the second stage, the rank norm $S=\|\hat Q^{-1}_{Y|X}(Y,X)\|$ is treated as the conformal score. A radius $\rho_{1-\alpha}$ is determined using the $1-\alpha$ quantile of the calibration set, and the prediction set is defined as all $y$ whose rank falls within the radius ball.

```mermaid
flowchart LR
    A[Reference Distribution F_U<br/>Uniform Ball/Gaussian] -->|Gradient of Convex Potential φθ<br/>PICNN Parameterization| B[Vector Quantile Map<br/>Q_Y|X = ∇φ]
    B -->|Legendre Conjugate<br/>c-transform| C[Multivariate Rank<br/>Q⁻¹ = ∇ψ]
    C -->|Rank Norm as Conformal Score<br/>S = ‖Q⁻¹‖| D[Compute 1-α<br/>Quantile ρ on Cal Set]
    D --> E[Pullback Ball Prediction Set<br/>‖Q⁻¹y,x‖ ≤ ρ]
    C -.Optional Reranking R.-> E
```

### Key Designs

**1. Semi-dual + PICNN for Conditional VQR (C-NQR):** Based on the OT duality for conditional vector quantiles provided by Carlier et al., the mapping is determined by a pair of Legendre-conjugate convex potentials $\varphi(u,x)$ and $\psi(y,x)$, where $Q_{Y|X}=\nabla_u\varphi$ and $Q^{-1}_{Y|X}=\nabla_y\psi$. This work reformulates the dual problem to optimize only a **single** convex potential $\varphi_\theta$, with the conjugate $\varphi^*_\theta(y,x)=\max_u\{u^\top y-\varphi_\theta(u,x)\}$ obtained via the c-transform. $\varphi_\theta$ is parameterized using **PICNN**, maintaining convexity with respect to $u$. By Danskin's theorem, the gradient of the conjugate only requires the derivative of $\varphi_\theta$. Training involves **L-BFGS for the inner argmax** and SGD for the outer PICNN.

**2. Amortized Optimization (AC-NQR):** To eliminate the cost of solving the argmax repeatedly, an **amortized network** $\tilde u_\vartheta(y,x)\approx\check u_{\varphi_\theta(\cdot,x)}(y)$ is introduced to directly predict the approximate maximizing point. It is trained via a **two-time-scale** approach. AC-NQR is significantly faster in training and inference (approx. 8.9 sec/epoch and 1.1 sec to infer 8192 points) and serves as the default base model for conformal experiments.

**3. Entropy-Regularized Scalable Variant (EC-NQR):** For higher dimensions where computing the convex conjugate remains expensive, an **entropy regularization term** is added to the primal OT problem. This smooths the objective and turns the inner argmax into a closed-form softmax, allowing for pure stochastic gradient solving. The trade-off is that entropy introduces bias and may distort the quantile mapping geometry. The authors argue that **cyclical monotonicity cannot be discarded**, and thus non-convex normalizing flows cannot replace the convex potential.

**4. Rank-Norm Conformal Score + Volume Optimality:** Using the multivariate rank norm $S_i=\|\hat Q^{-1}_{Y|X}(Y_i,X_i)\|$ as the conformal score, the prediction set $\hat C^{pb}_\alpha(x)=\{y:\hat Q^{-1}_{Y|X}(y,x)\in B(0,\rho_{1-\alpha})\}$ provides a finite-sample marginal coverage $\ge 1-\alpha$. **Theorem 3 (Theoretical Highlight)**: When the Jacobian determinant of the inverse transport has a radial structure, this pullback ball achieves the **minimum volume** among all sets satisfying conditional coverage $\ge 1-\alpha$, effectively recovering the Highest Density Region (HPD).

**5. Reranking for Anisotropy (RPB):** Pullback balls implicitly assume that the rank $U=\hat Q^{-1}(Y,X)$ is radially symmetric. If the model is misspecified and the rank is anisotropic, the Euclidean radius may be unreliable. This work applies the OT-CP reranking operator $R$ to the vector rank to correct for deviations from the reference distribution $F_U$.

## Key Experimental Results

### Main Results (Generation Quality, S-W2, lower is better; Synthetic Data)

| Dataset | AC-NQRU (Ours) | VQR | FN-VQR | CPQ | CVQR |
|---|---|---|---|---|---|
| Star | **0.182** | 0.270 | 0.271 | 0.274 | 0.443 |
| Glasses | **0.771** | 1.964 | 2.017 | 0.931 | 1.170 |
| Banana | 0.073 | 0.389 | 0.398 | 0.237 | 0.401 |
| Convex Glasses | **0.657** | 1.961 | 1.954 | 0.793 | 0.953 |

- AC-NQRU achieves the fastest training and inference times while maintaining the best or second-best S-W2 across most datasets.

### Ablation Study (L2-UV for recovering the true quantile operator, lower is better)

| Function | Dataset | C-NQRU | C-NQRY | AC-NQRU | AC-NQRY | CPF |
|---|---|---|---|---|---|---|
| $Q^{-1}_{Y|X}$ | Convex Banana | 3.784 | 0.212 | **0.106** | 0.206 | 9.479 |
| $Q^{-1}_{Y|X}$ | Convex Glasses | 0.332 | **0.068** | 0.203 | 0.109 | 2.268 |
| $Q_{U|X}$ | Convex Banana | 7.665 | 0.660 | **0.545** | 0.569 | 16.537 |

- The proposed model shows orders of magnitude improvement in reconstructing the true quantile operator compared to baselines such as CPF.

### Key Findings
- **Conformal Experiments (scm20d/sgemm/blog/bio, $\alpha=0.1$)**: PB/PBS variants simultaneously achieve competitive conditional coverage and the smallest prediction set volumes (measured by $\log V/d_y$), significantly outperforming OT-CP, OT-CP+, and local ellipsoids (ELL).
- **Residual Version Stability**: Fitting VQR on the signal residuals $s=y-\hat f(x)$ (where $\hat f$ is a Random Forest) further improves performance, indicating that VQR can be orthogonally combined with point predictors.
- **Adaptivity**: Unlike concurrent discrete OT-CP methods, this continuous neural VQR does not depend on conditional density estimation and is expected to be more robust in high dimensions.
- **Multimodal Scalability**: The appendix demonstrates a density-based score derived via the change-of-variables formula, which can characterize **disconnected** geometries for multimodal distributions, addressing the limitations of spherical pullback sets.

## Highlights & Insights
- **Unity of Three Fields**: The work integrates Neural OT (learning potentials), Vector Quantile Regression (multivariate ranks), and Conformal Prediction (coverage guarantees), leveraging the strengths of each—geometry from OT, continuous invertibility from VQR, and finite-sample guarantees from CP.
- **Theoritical Solidness**: Theorem 3 proves that "Pullback Ball = HPD = Volume Optimal," providing a clean optimality justification for using the rank norm as a score rather than relying on engineering heuristics.
- **Amortization as an Engineering Pivot**: Using a forward network to replace inner-loop convex optimization in AC-NQR is the practical prerequisite for scaling OT-VQR to real-world multi-target regression benchmarks.
- **Importance of Convexity**: The counter-examples emphasize that arbitrary flows cannot replace convex potentials, identifying "cyclical monotonicity" as the fundamental property that makes multivariate ranks statistically meaningful.
- **Decoupling from Point Predictors**: Current models can fit $y$ directly or residuals $y-\hat f(x)$, allowing VQR to serve as an "uncertainty shell" for any regressor.

## Limitations & Future Work
- **Model Class Restricted by Convexity**: The convexity of PICNN is both a guarantee and a constraint; expressivity might be limited when handling highly complex multimodal conditional distributions.
- **Entropy Bias**: EC-NQR offers better scalability but at the cost of distorting the quantile geometry, requiring a finer balance in high dimensions.
- **Response Dimensionality**: Experiments focus on dimensions up to 16; the "high-dimensional advantage" is more of a theoretical argument rather than a large-scale validation.
- **Dependence on Exchangeability**: Conformal guarantees rely on the exchangeability of calibration and test samples; coverage may fail under distribution shift or in time-series scenarios.

## Related Work & Insights
- **Multivariate Quantile Lineage**: Transitioning from spatial quantiles to the OT measure transport perspective, this work inherits from Carlier's CVQF while removing the "affine to $X$" constraint.
- **Neural OT**: Compared to biased entropy-regularized Sinkhorn methods, the ICNN-based potential approach used here preserves monotonicity and invertibility, localized to conditional potentials.
- **Multivariate Conformal**: Offers an explicit generative model and optimality theory compared to heuristic coordinate-wise or scalarized scores.
- **Relation to CQR**: This is the most natural high-dimensional extension of the CQR principle, where the rank norm acts as the multivariate equivalent of "residuals."

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Scaling Neural OT to conditional VQR and integrating it with CP is a solid contribution; Theorem 3 provides rigorous justification.
- **Experimental Thoroughness**: ⭐⭐⭐ — Detailed benchmarks on synthetic data and multi-target regression, though the response dimensionality is relatively low for a "high-dimensional" claim.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical flow from 1D intuition to multivariate theory and conformal application.
- **Value**: ⭐⭐⭐⭐ — Provides a scalable, theoretically backed tool for multivariate uncertainty quantification with volume-optimal guarantees.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[ICLR 2026\] HOTA: Hamiltonian Framework for Optimal Transport Advection](hota_hamiltonian_framework_for_optimal_transport_advection.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)
- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)
- [\[ICLR 2026\] A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport](a_scalable_constant-factor_approximation_algorithm_for_w_p_optimal_transport.md)

</div>

<!-- RELATED:END -->
