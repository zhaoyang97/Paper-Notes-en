---
title: >-
  [Paper Note] A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation
description: >-
  [NEURIPS2025][grokking] This paper rigorously proves the mechanism behind grokking from a purely optimization-theoretic perspective. Gradient flow with small weight decay exhibits two-phase dynamics in the $\lambda\to 0$…
tags:
  - "NEURIPS2025"
  - "grokking"
  - "weight decay"
  - "gradient flow"
  - "Riemannian optimization"
  - "implicit regularization"
  - "two-timescale dynamics"
date: 2026-05-08
content_hash: 6f54292d12380036
---

# A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation

**Conference**: NEURIPS2025
**arXiv**: [2505.20172](https://arxiv.org/abs/2505.20172)
**Code**: None
**Area**: Other
**Keywords**: grokking, weight decay, gradient flow, Riemannian optimization, implicit regularization, two-timescale dynamics

## TL;DR

This paper rigorously proves the mechanism behind grokking from a purely optimization-theoretic perspective. Gradient flow with small weight decay exhibits two-phase dynamics in the $\lambda\to 0$ limit: rapid convergence to the critical manifold $\mathcal{M}$ of the training loss, followed by a Riemannian gradient flow along the manifold minimizing the $\ell_2$ norm at timescale $t\approx 1/\lambda$, thereby inducing delayed generalization.

## Background & Motivation

**Grokking phenomenon**: First reported by Power et al. (2022) — training loss quickly drops to zero, while test loss remains on a long plateau before suddenly decreasing. This phenomenon has been observed in modular addition, image classification, matrix factorization, and other tasks.

**Limitations of prior work**: Previous studies (Liu et al. 2022, Lyu et al. 2023) attributed grokking to a lazy-to-rich regime transition, but lacked rigorous optimization-theoretic characterization of the slow drift phase and relied on assumptions specific to certain architectures (homogeneous parameterization, large initialization).

**Ambiguous role of weight decay**: Grokking is typically more pronounced and occurs earlier with weight decay, yet has also been reported without it (classification implicit bias). A rigorous analysis of how weight decay drives norm reduction and thereby improves generalization has been absent.

**Drift on the interpolation manifold**: Li et al. (2021) studied SGD-noise-driven drift using the Katzenberger framework, a stochastic effect; by contrast, this paper identifies a **deterministic** mechanism driven by regularization.

**Technical gap**: Existing analyses either treat the Katzenberger result as a black box or only handle the case where initialization is already near the manifold. This work requires characterizing the complete coupling between fast and slow phases from arbitrary initialization.

**Regression vs. classification**: In regression tasks, grokking cannot be observed without weight decay (unbounded dynamics, no implicit bias), further underscoring the central role of weight decay.

## Method

### Overall Architecture

For a sufficiently smooth loss function $F:\mathbb{R}^d\to\mathbb{R}_+$, the regularized gradient flow is:

$$\dot{w}^\lambda(t) = -\nabla F(w^\lambda(t)) - \lambda w^\lambda(t)$$

In the $\lambda\to 0$ limit, the trajectory $w^\lambda(t)$ is shown to decompose into two coupled dynamical phases.

### Key Designs

**1. Fast dynamics (Proposition 1)**

- On any finite time interval $[0,T]$, the regularized gradient flow $w^\lambda$ converges uniformly to the unregularized gradient flow $w^{\mathrm{GF}}$.
- Technical tool: Grönwall's inequality, yielding the error bound $\|w^\lambda(t)-w^{\mathrm{GF}}(t)\|\leq \lambda t e^{ct}\sup_{K}\|w\|$.
- Mathematical implication: Training loss decreases rapidly, but parameters have large norm and poor generalization (analogous to the lazy/NTK regime).

**2. Slow dynamics (Proposition 2)**

- After time rescaling $\tilde{w}^\lambda(t)=w^\lambda(t/\lambda)$, on $[\varepsilon,T]$, $\tilde{w}^\lambda$ converges uniformly to the Riemannian gradient flow $\tilde{w}^\circ$:
$$\dot{\tilde{w}}^\circ(t)=-\mathrm{grad}_{\mathcal{M}}\ell_2(\tilde{w}^\circ(t)), \quad \tilde{w}^\circ(0)=\Phi(w_0)$$
- Here $\mathrm{grad}_{\mathcal{M}}\ell_2(w)=P_{\mathrm{Ker}(\nabla^2 F(w))}(w)$ is the Riemannian gradient of the $\ell_2$ norm on the critical manifold $\mathcal{M}$.
- Key lemma: The differential $D\Phi_w$ of the flow map $\Phi$ is exactly the orthogonal projection onto the null space of the Hessian (using results from Li et al. 2021).
- Core of the proof: $D\Phi(w)\cdot\nabla F(w)=0$, so the $\frac{1}{\lambda}\nabla F$ term in the regularization is annihilated, leaving only the norm gradient.

**3. Fast-slow matching (Lemma 2)**

- A matching time $t(\lambda)=-\frac{\lambda\ln\lambda}{2c}$ is constructed so that $\tilde{w}^\lambda(t(\lambda))\to\Phi(w_0)$.
- This ensures the initial condition of the slow dynamics starts from the limit point $\Phi(w_0)\in\mathcal{M}$ of the unregularized flow.
- Technically guarantees seamless coupling between the two phases in the limit.

### Key Assumptions and Loss Functions

- **Assumption 1** (Regularity): $F$ is $\mathcal{C}^3$ and definable in an o-minimal structure (ensuring convergence of bounded gradient flows), covering all standard architectures including polynomials and exponentials.
- **Assumption 2** (Morse-Bott property): The critical manifold $\mathcal{M}$ is a smooth submanifold, and the nonzero eigenvalues of the Hessian are bounded below by $\eta>0$.
- Both assumptions are standard, covering all neural networks and Transformers with differentiable activation functions.

### Characterization of Convergence Points

- **Proposition 3**: Limit points of $w^\lambda$ are contained in the KKT points of $\min_{w\in\mathcal{M}}\|w\|_2^2$.
- **Proposition 4**: If the Riemannian flow converges to a strict local minimum, then $\lim_{\lambda\to 0}\lim_{t\to\infty}w^\lambda(t)=w^\star$.
- The essence of grokking = non-commutativity of two limits: $\lim_{\lambda\to 0}\lim_{t\to\infty}\neq\lim_{t\to\infty}\lim_{\lambda\to 0}$.

## Key Experimental Results

| Setting | Training Loss | Test Loss | Grokking | Key Finding |
|---|---|---|---|---|
| Linear regression $F(w)=\|Xw-y\|^2$ | Rapidly → 0 | Plateau → decrease | ✓ | Slow phase converges to $w^\star=X^+y$ (minimum-norm solution), analytically solvable |
| Low-rank matrix completion ($20\times20$, rank 3) | Rapidly → 0 @ $t\approx 1$ | High plateau → decrease starting $t\approx 10^2$ | ✓ | Slow phase minimizes $\|U\|_F^2+\|V\|_F^2$ (equivalent to nuclear norm); singular values transition from all large to only 3 nonzero |
| Two-layer ReLU network ($m=100$, $n=10$) | Rapidly → 0 @ $t_2=1$ | High plateau → decrease @ $t_3\approx 10^5$ | ✓ | Gradual norm reduction promotes simpler functions (fewer kinks), $\lambda=10^{-3}$ |
| Diagonal linear network | Rapidly → 0 | Plateau → decrease | ✓ | Slow phase promotes sparse estimators |

**Key Findings:**

1. **The grokking transition is not abrupt**: On a linear timescale, the descent persists for $O(1/\lambda)$ time, comparable to the plateau length; it only appears "sudden" on a logarithmic timescale.
2. **Role of initialization scale**: Large initialization → large norm of $\Phi(w_0)$ → pronounced grokking; small initialization → already low norm → no grokking.
3. **Matrix completion experiment** validates theoretical predictions: the top 3 singular values converge to the true values $\sigma_1^\star,\sigma_2^\star,\sigma_3^\star$, while the rest approach zero.
4. Despite ReLU networks not satisfying the $\mathcal{C}^3$ assumption, experiments perfectly match the theoretically predicted two-phase behavior.

## Highlights & Insights

1. **Theoretical elegance**: Grokking is unified as a purely optimization phenomenon (no statistical assumptions required); the core result is clear — fast phase = unregularized gradient flow, slow phase = Riemannian norm minimization on the manifold.
2. **Self-contained proof**: Avoids the heavy stochastic differential equation machinery of Katzenberger (1990), providing a concise deterministic proof based on Falconer (1983).
3. **Broad applicability**: Mild assumptions ($\mathcal{C}^3$ + o-minimal + Morse-Bott) without restricting to specific architectures or initialization distributions.
4. **Deep insight**: The observation that "grokking is not abrupt" corrects a common misconception in the literature; the analysis offers inspiration for extensions to other regularizers such as SAM.
5. **Complete fast-slow matching**: The matching time construction $t(\lambda)=-\lambda\ln\lambda/(2c)$ fills a technical gap left by prior work.

## Limitations & Future Work

1. **Regression only**: The assumption that the unregularized flow is bounded excludes classification tasks where parameters diverge — yet classification is precisely where grokking was originally discovered.
2. **Asymptotic analysis $\lambda\to 0$**: In practice $\lambda$ is fixed (e.g., $10^{-3}$); the paper provides no quantitative guarantees for finite $\lambda$ (only heuristic discussion in the appendix).
3. **Requires $\mathcal{C}^3$ smoothness**: Excludes non-differentiable activations such as ReLU, although experiments suggest the conclusions still hold.
4. **Morse-Bott assumption**: Requires a positive lower bound on nonzero Hessian eigenvalues at the critical manifold; handling degenerate points is left as an open problem.
5. **No generalization guarantee from a purely optimization perspective**: The association between norm reduction and improved generalization is empirical; a theoretical proof that test loss decreases is not provided.
6. **Discrete time / SGD**: The analysis is based on continuous gradient flow; the noise effects of SGD and finite learning rates are not covered.

## Related Work & Insights

- **Power et al. (2022)**: Coined the term grokking; Transformer + modular addition + weight decay.
- **Lyu et al. (2023)**: Grokking = lazy-to-rich regime transition; homogeneous parameterization + large initialization; no characterization of the slow drift phase.
- **Kumar et al. (2024)**: Similar lazy-to-rich perspective; emphasizes that grokking can occur without weight decay.
- **Liu et al. (2022, OmniGrok)**: Proposed the intuition that weight decay drives norm reduction, but without theoretical proof — this paper formalizes that intuition.
- **Li et al. (2021)**: Noise-driven drift after SGD reaches the interpolation manifold (Katzenberger framework); this paper identifies **deterministic regularization-driven** drift instead.
- **Chizat et al. (2020)**: Implicit bias of infinite-width two-layer networks; delayed generalization was already observed early on.
- **Fatkullin & Vanden-Eijnden (2010)**: Drift SDEs in energy landscapes under small stochastic perturbations; this paper provides a cleaner deterministic counterpart.

## Rating

- Novelty: ⭐⭐⭐⭐ — Elevates grokking from empirical observation tied to specific architectures/tasks to a general two-timescale optimization theorem; the Riemannian gradient flow perspective is original.
- Experimental Thoroughness: ⭐⭐⭐ — Coverage across linear regression / matrix completion / ReLU networks / diagonal networks is broad, but all settings are synthetic; real datasets and large-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear exposition; the triple correspondence of intuition, formalization, and experiment is outstanding; proof sketches are highly accessible.
- Value: ⭐⭐⭐⭐ — Provides a solid theoretical foundation for understanding the deeper role of weight decay, with broad implications for implicit regularization and related directions such as SAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)
- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)
- [\[NeurIPS 2025\] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering](mutualvpr_a_mutual_learning_framework_for_resolving_supervision_inconsistencies_.md)

</div>

<!-- RELATED:END -->
