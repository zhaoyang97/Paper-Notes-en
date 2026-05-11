---
title: >-
  [Paper Note] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning
description: >-
  [NeurIPS 2025][Optimization][nonlinear preconditioning] This paper proposes a unified sufficient condition connecting the $(L_0,L_1)$-smoothness and anisotropic smoothness frameworks…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "nonlinear preconditioning"
  - "saddle point escaping"
  - "anisotropic smoothness"
  - "$(L_0"
  - "L_1)$-smoothness"
  - "gradient clipping"
date: 2026-05-08
content_hash: 9a81216b30639cd9
---

# Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning

**Conference**: NeurIPS 2025
**arXiv**: [2509.15817](https://arxiv.org/abs/2509.15817)
**Code**: None
**Area**: Optimization Theory
**Keywords**: nonlinear preconditioning, saddle point escaping, anisotropic smoothness, $(L_0,L_1)$-smoothness, gradient clipping

## TL;DR
This paper proposes a unified sufficient condition connecting the $(L_0,L_1)$-smoothness and anisotropic smoothness frameworks, proves that nonlinear preconditioned gradient methods (including gradient clipping variants) retain saddle-point avoidance under these relaxed conditions, and establishes that a perturbed variant achieves second-order stationary points with polylogarithmic dimension dependence.

## Background & Motivation

**Background**: Classical gradient descent analysis relies on the Lipschitz smoothness assumption ($\|\nabla^2 f(x)\| \leq L$), under which GD is known to avoid strict saddle points. Nonlinear preconditioned gradient methods $x^{k+1} = x^k - \gamma \nabla\phi^*(\lambda \nabla f(x^k))$ provide a flexible optimization framework that subsumes gradient clipping.

**Limitations of Prior Work**: (a) The Lipschitz smoothness assumption **fails** in many practical problems (e.g., phase retrieval, matrix factorization), holding only locally on compact sets; (b) The $(L_0,L_1)$-smoothness assumption, while empirically reasonable for LSTMs and Transformers, lacks theoretical validation on concrete problems; (c) Anisotropic smoothness has been developed independently, with its structural relationship to $(L_0,L_1)$-smoothness left unexplored.

**Key Challenge**: Existing saddle-point escaping results rely on classical Lipschitz smoothness, whereas practical applications such as matrix factorization naturally violate this assumption, rendering the theoretical guarantees inapplicable.

**Goal**: (i) Under what conditions do practical problems satisfy generalized smoothness? (ii) Does nonlinear preconditioning retain saddle-point avoidance under relaxed conditions?

**Key Insight**: A new sufficient condition (Assumption 2.8) is proposed—the Hessian norm is bounded above by a degree-$R$ polynomial in $\|x\|$, and the gradient norm is bounded below by a degree-$(R+1)$ polynomial—unifying the derivation of both generalized smoothness frameworks.

**Core Idea**: When the gradient of the objective grows faster than the Hessian, $(L_0,L_1)$-smoothness and anisotropic smoothness hold automatically, and nonlinear preconditioning preserves saddle-point avoidance.

## Method

### Overall Architecture
The paper consists of three major parts: (1) proposing the new sufficient condition and verifying it on key applications; (2) proving asymptotic saddle-point avoidance of nonlinear preconditioned GD (including gradient clipping variants); (3) analyzing the finite-time complexity of perturbed preconditioned GD.

### Key Designs

1. **Unified Sufficient Condition for Generalized Smoothness (Assumption 2.8)**:

    - Function: characterizes the asymptotic behavior of the Hessian and gradient via polynomial growth rates.
    - Mechanism: requires the existence of $R \in \mathbb{N}$ such that $\|\nabla^2 f(x)\|_F \leq p_R(\|x\|)$ (degree-$R$ polynomial) and $\|\nabla f(x)\| \geq q_{R+1}(\|x\|)$ (degree-$(R+1)$ polynomial with leading coefficient $b_{R+1} > 0$), so that gradient growth exceeds Hessian growth by one polynomial degree.
    - Design Motivation: Classical $(L_0,L_1)$-smoothness requires $\|\nabla^2 f\| \leq L_0 + L_1\|\nabla f\|$, which does **not necessarily hold for multivariate polynomials**—the paper illustrates this with the counterexample $f(x,y) = \frac{1}{4}x^4 + \frac{1}{4}y^4 - \frac{1}{2}x^2 y^2$, which admits paths where $\nabla f = 0$ yet $\|\nabla^2 f\| \to \infty$. The new condition circumvents this issue.
    - **Theorem 2.9**: Under Assumption 2.8, for any $L_1 > 0$ there exists $L_0$ such that $f$ is $(L_0, L_1)$-smooth.
    - **Theorem 2.11**: Under appropriate kernel conditions (Assumption 2.10—e.g., $h^{*\prime}(y)/y$ is decreasing), the second-order characterization of anisotropic smoothness likewise holds.

2. **Application Verification**:

    - **Phase retrieval** $f(x) = \frac{1}{4}\sum_i(y_i^2 - (a_i^\top x)^2)^2$: satisfies Assumption 2.8 when the measurement vectors $\{a_i\}$ span $\mathbb{R}^n$ (Theorem 2.12).
    - **Symmetric matrix factorization** $f(U) = \frac{1}{2}\|UU^\top - Y\|_F^2$: holds unconditionally (Theorem 2.13).
    - **Asymmetric matrix factorization** (with regularization $\kappa > 0$): regularization is required to prevent the gradient from vanishing along the solution manifold (Theorem 2.14).
    - **Burer–Monteiro MaxCut augmented Lagrangian**: generalized smoothness holds with respect to the primal variable (Theorem 2.15).

3. **Asymptotic Saddle-Point Avoidance**:

    - **Theorem 3.1** (smooth preconditioner): Let $f \in \mathcal{C}^2$ satisfy the second-order characterization of anisotropic smoothness. For preconditioned GD iterates $x^{k+1} = x^k - \gamma \nabla\phi^*(\bar{L}^{-1} \nabla f(x^k))$ with $\gamma < 1/L$, the iterates converge to a strict saddle point $\mathcal{X}^\star$ with probability zero under random initialization.
    - Proof strategy: employs the stable-center manifold theorem, crucially using $\nabla^2\phi^*(0) = I$ to ensure that the Jacobian eigenvalues at saddle points coincide with those of $\nabla^2 f$.
    - **Theorem 3.2** (hard gradient clipping): The clipping map $\min(1/\|\nabla f\|, \bar{L}^{-1})\nabla f$ is non-differentiable, but via a generalization of the non-smooth stable-center manifold theorem, it suffices that the iteration map be **differentiable almost everywhere**—since the set where $\|\nabla f(x)\| = \bar{L}$ has measure zero.

4. **Finite-Time Analysis of Perturbed Preconditioned GD (Algorithm 1)**:

    - When the first-order stationarity measure $\lambda^{-1}\phi(\nabla\phi^*(\lambda \nabla f(x^k)))$ is sufficiently small, a uniform perturbation $\xi^k \sim \mathbb{B}_0(r)$ is injected, followed by $\lceil\mathcal{T}\rceil$ unperturbed iterations.
    - Requires Lipschitz continuity of the map $H_\lambda(x) = \nabla^2\phi^*(\lambda \nabla f(x)) \nabla^2 f(x)$ (Assumption 3.3)—a strictly weaker requirement than Lipschitz continuity of $\nabla^2 f$.
    - **$\varepsilon$-second-order stationary point**: defined by $\lambda^{-1}\phi(\nabla\phi^*(\lambda \nabla f(x))) \leq \epsilon^2$ and $\lambda_{\min}(H_\lambda(x)) \geq -\sqrt{\rho\epsilon}$.
    - Iteration complexity has only **polylogarithmic dependence** on the dimension $n$.

### Key Kernel Functions
Three kernel functions are examined: $h_1(x) = \cosh(x) - 1$, $h_2(x) = \exp(|x|) - |x| - 1$, and $h_3(x) = -|x| - \ln(1-|x|)$, corresponding to different forms of gradient clipping/normalization. $h_3$ corresponds to the classical $(L_0,L_1)$-smoothness framework.

## Key Experimental Results

### Application Verification Summary

| Problem | Lipschitz Smooth | $(L_0,L_1)$-Smooth | Anisotropic Smooth | Condition |
|---------|:---:|:---:|:---:|------|
| Phase retrieval | ✗ | ✓ | ✓ | $\{a_i\}$ spans $\mathbb{R}^n$ |
| Symmetric matrix factorization | ✗ | ✓ | ✓ | None |
| Asymmetric matrix factorization | ✗ | ✓ | ✓ | $\kappa > 0$ |
| BM-MaxCut ALM | ✗ | ✓ | ✓ | Fixed multiplier $y$ |
| Multivariate polynomials | ✗ | Not guaranteed | Not guaranteed | Growth rates must be checked |

### Theoretical Results Comparison

| Method / Condition | Saddle Avoidance | Smoothness Assumption | Dimension Dependence |
|---------|:---:|---------|---------|
| Classical GD [Lee et al.] | Asymptotic | Lipschitz smooth | — |
| Perturbed GD [Jin et al.] | Finite-time | Lipschitz + Hessian Lipschitz | $\tilde{O}(\log n)$ |
| Ours Theorem 3.1 | Asymptotic | Anisotropic smooth (2nd-order) | — |
| Ours Theorem 3.2 | Asymptotic | Same (clipping variant) | — |
| Ours Algorithm 1 | Finite-time | Anisotropic smooth + $H_\lambda$ Lipschitz | $\tilde{O}(\log n)$ |

### Key Findings
- The polynomial growth-rate condition in Assumption 2.8 is practically verifiable—the paper provides constructive proofs for four classes of problems.
- Asymmetric matrix factorization **fails** to satisfy $(L_0,L_1)$-smoothness without regularization: along the manifold $WD, D^{-1}H$, the gradient norm can remain zero while $\|x\| \to \infty$.
- Lipschitz continuity of $H_\lambda$ is a substantially weaker requirement than Lipschitz continuity of $\nabla^2 f$—the "saturation" effect of the kernel function naturally suppresses higher-order terms.

## Highlights & Insights
- **Deep insight from the unified framework**: The $(L_0,L_1)$-smoothness and anisotropic smoothness theories, which appeared to develop independently, are unified in this paper through algebraic relationships between gradient and Hessian growth rates, revealing their intrinsic connection.
- **Complete loop from theory to practice**: Rather than proving abstract theorems alone, the paper explicitly verifies that **concrete problems** such as phase retrieval and matrix factorization satisfy the proposed assumptions—an approach that is relatively uncommon in optimization theory papers.
- **Elegant treatment of hard clipping**: By applying a non-smooth stable manifold theorem, the paper handles the non-differentiability introduced by clipping under the minimal requirement of almost-everywhere differentiability, elegantly covering the hard gradient clipping most commonly used in practice.

## Limitations & Future Work
- The analysis is primarily deterministic; the **stochastic gradient** setting is not addressed—handling SGD combined with gradient clipping requires additional treatment of variance.
- The complexity of perturbed preconditioned GD depends on the Lipschitz constant $\rho$ of $H_\lambda$, which is difficult to estimate in practice.
- No numerical experiments are provided—this is a purely theoretical contribution, and the practical effects of different kernel function choices are not empirically compared.
- The conditions on kernel functions in Assumption 2.10 (e.g., $\lim_{y\to\infty} h^{*\prime}(s_d(y))/y = 0$) are technically involved and must be verified on a case-by-case basis.

## Related Work & Insights
- **vs. Zhang et al. (2020)**: First proposed $(L_0,L_1)$-smoothness and analyzed convergence of clipped GD, but did not address saddle-point avoidance. The present paper completes the second-order guarantees.
- **vs. Oikonomidis et al. (2023)**: Studied convergence of preconditioned GD under anisotropic smoothness, but only to first-order stationary points. The present paper extends the results to second order.
- **vs. Cao et al.**: Studied saddle-point avoidance under second-order self-bounding conditions; the assumptions are orthogonal and complementary to those of the present paper.
- **vs. Jin et al. (2017, 2021)**: Classical finite-time analysis of perturbed GD under Lipschitz smoothness and Hessian Lipschitz continuity. The present paper relaxes these to anisotropic smoothness and Lipschitz continuity of $H_\lambda$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies two major generalized smoothness frameworks and establishes saddle-point escaping theory beyond Lipschitz smoothness; contributions are substantial.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no numerical experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-structured; counterexamples and application verifications prevent the theory from being dry.
- Value: ⭐⭐⭐⭐ Makes important advances in non-convex optimization theory, particularly in bridging the gap between theoretical assumptions and practical problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems](extragradient_method_for_l_0_l_1-lipschitz_root-finding_problems.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](../../ICLR2026/optimization/saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)

</div>

<!-- RELATED:END -->
