---
title: >-
  [Paper Note] On the Relation between Rectified Flows and Optimal Transport
description: >-
  [NeurIPS 2025][Image Generation][Rectified Flow] This paper presents a rigorous theoretical investigation of the relationship between rectified flows (flow matching) and optimal transport (OT). Through the construction o…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Rectified Flow"
  - "Optimal Transport"
  - "Flow Matching"
  - "Counterexample Construction"
  - "Wasserstein Distance"
date: 2026-05-08
content_hash: 0957be9503446a64
---

# On the Relation between Rectified Flows and Optimal Transport

**Conference**: NeurIPS 2025
**arXiv**: [2505.19712](https://arxiv.org/abs/2505.19712)
**Code**: Unavailable
**Area**: Diffusion Models / Flow Matching / Optimal Transport
**Keywords**: Rectified Flow, Optimal Transport, Flow Matching, Counterexample Construction, Wasserstein Distance

## TL;DR

This paper presents a rigorous theoretical investigation of the relationship between rectified flows (flow matching) and optimal transport (OT). Through the construction of multiple counterexamples, it demonstrates that previously published claims asserting the asymptotic equivalence between gradient-constrained rectified flows and OT do not hold in general, and that stronger assumptions than those previously identified are required to guarantee such equivalence.

## Background & Motivation

Optimal Transport (OT) is a classical framework for measuring and transforming between probability distributions, with broad applications in clustering, domain adaptation, and generative modeling. The Benamou–Brenier dynamic formulation recasts OT as finding a minimum $L^2$-norm velocity field that transports a source distribution to a target distribution via the continuity equation.

Flow matching and rectified flows are recently emerged generative modeling paradigms whose core idea is to learn a velocity field by projecting the linear interpolation of $X_0$ and $X_1$—starting from an arbitrary coupling—onto a conditional expectation. Iterative rectification straightens transport trajectories and reduces transport cost. Liu (2022) claimed that, when the velocity field is constrained to be a gradient field (i.e., $v_t = \nabla \varphi_t$), fixed points of rectified flow are equivalent to OT couplings.

**Key Challenge**: This equivalence claim has been widely cited in the literature to justify rectified flow as a reliable approach for computing OT. However, the authors identify that the claim lacks critical assumptions and fails in multiple settings.

**Key Insight**: Through rigorous mathematical construction, the paper presents two classes of counterexamples—disconnected support sets and non-rectifiable couplings—demonstrating that fixed points of gradient-constrained rectified flow need not correspond to OT maps, thereby clarifying the theoretical boundaries.

## Method

### Overall Architecture

The paper takes a theoretical analysis perspective and examines rectified flows at three levels: (1) affine invariance of unconstrained rectification and closed-form solutions in the Gaussian case; (2) existence of optimal velocity fields under gradient constraints; (3) counterexamples to the claimed equivalence between fixed points of gradient-constrained rectification and OT.

### Key Designs

1. **Affine Invariance Analysis (Theorem 2)**: Establishes the equivariance of the rectification operator $\mathcal{R}$ under affine transformations. Specifically:

    - If $(Z_0, Z_1) = \mathcal{R}(X_0, X_1)$, then $\mathcal{R}(AX_0+b, AX_1+b) = (AZ_0+b, AZ_1+b)$
    - Analogous equivariance holds for translations and scalings of the target distribution
    - **Design Motivation**: These properties mirror the invariances of OT; however, part (i) of the affine equivariance no longer holds once the gradient constraint is imposed, hinting at a fundamental divergence between the two approaches.

2. **Closed-Form Solution in the Gaussian Case (Theorem 3)**: When $(X_0, X_1)$ follows a joint Gaussian distribution, the optimal velocity field admits an analytic form:
    $$v_t(x) = \frac{1}{1-t}\left(((1-t)\Sigma_{01} + t\Sigma_1)\Sigma_t^{-1} - \text{Id}\right)x$$
   where $\Sigma_t = (1-t)^2\Sigma_0 + (1-t)t(\Sigma_{01}+\Sigma_{10}) + t^2\Sigma_1$. In particular, when $\Sigma_0$ and $\Sigma_1$ are jointly diagonalizable, a single rectification step recovers the optimal coupling. Explicit velocity fields for Gaussian mixture models are also derived (Theorem 5).

3. **Existence of Weak Solutions under Gradient Constraint (Proposition 8)**: Proves that the constrained problem $\min_{w_t = \nabla\varphi_t} \mathcal{L}(w_t | X_0, X_1)$ always admits a solution in the weak sense. Key conclusions:

    - The gradient-constrained optimal velocity field $v_t^p$ is the orthogonal projection of the unconstrained solution $v_t$ onto the space $T_{\mu_t}$
    - $v_t^p$ is the minimum-norm solution to the continuity equation
    - When $X_0 \sim \mathcal{N}(0, I_d)$ under an independent coupling, the unconstrained solution is itself a gradient field (Corollary 9), so both problems share the same solution.

4. **Counterexample I: Disconnected Support (Section 4.1, Proposition 10)**: Constructs an example in which $\mu_0$ and $\mu_1$ have supports consisting of two separated regions. A coupling $(\tilde{X}_0, \tilde{X}_1)$ that is locally optimal on each subdomain but globally suboptimal is shown to be a fixed point of $\mathcal{R}_p$ with zero loss, yet satisfies $\mathbb{E}[\|\tilde{X}_1 - \tilde{X}_0\|^2] > \mathbb{E}[\|X_1 - X_0\|^2]$, disproving the original equivalence claim (8).

5. **Counterexample II: Non-Rectifiable Coupling (Section 4.2, Proposition 13)**: Setting $X_0 = -X_1 \sim \mathcal{N}(0, I_d)$ yields a velocity field $v_t(x) = -\frac{2}{1-2t}x$, whose ODE is singular at $t=1/2$ and admits non-unique solutions. This coupling achieves zero loss but is not optimal. It is further shown (Corollary 17) that even when the loss is arbitrarily small and the $W_2$ distance is near-optimal, the transport cost can remain bounded away from optimality.

### Corrected Conditions and Noise Injection

The paper identifies sufficient conditions for equivalence (Theorem 11): the additional assumption that $\text{supp}(X_t) = \mathbb{R}^d$ (full support) is required. A **smoothed rectification** scheme is also proposed (Section 5): at each rectification step, a small amount of Gaussian noise is injected via $X_0^{(i)} = \sqrt{1-c_i} Z_0^{(i)} + \sqrt{c_i} W^{(i)}$, ensuring that couplings remain rectifiable at every iteration (Theorem 14) while preserving the monotone decrease of the loss function.

## Key Experimental Results

### Main Results: Numerical Validation of Counterexamples

| Configuration | Transport Cost $\mathbb{E}[\|X_1-X_0\|^2]$ | Loss $\mathcal{L}$ | Optimal? |
|---|---|---|---|
| Optimal coupling $(X_0,X_1)$ (vertical transport) | 4 | 0 | ✓ |
| Suboptimal fixed point $(\tilde{X}_0,\tilde{X}_1)$ (horizontal transport) | 16 | 0 | ✗ |
| Non-rectifiable coupling $X_1=-X_0$ | 4 | 0 | ✗ |

### Analytical Validation in the Gaussian Case

| Setting | Optimal after One Rectification Step? | Condition |
|---|---|---|
| $\Sigma_0, \Sigma_1$ jointly diagonalizable | ✓ | Theorem 3(ii) |
| $\Sigma_0, \Sigma_1$ not jointly diagonalizable | ✗ | Affine invariance (i) fails |
| One-dimensional case | ✓ (any rectifiable coupling) | Proposition 4 |

### Key Findings

- Disconnected supports are extremely common in real-world datasets, severely limiting the applicability of rectified flow as an OT solver.
- The interpolated distribution $\mu_{1/2}$ can degenerate at $t=1/2$ (e.g., all trajectories crossing the origin in Counterexample II), rendering the velocity field non-unique.
- Entropy regularization (as in DSBM) guarantees convergence but may converge arbitrarily slowly as the regularization parameter approaches zero.

## Highlights & Insights

- This is a purely theoretical contribution featuring elegant counterexample constructions. Counterexample I exploits disconnected supports to show that local optimality does not imply global optimality; Counterexample II exploits path crossing at a single point to produce a non-unique ODE.
- Proposition 8 establishes an equivalence between gradient-constrained solutions and minimum-norm solutions to the continuity equation, which is a deeper result than a direct treatment of gradient field existence.
- Smoothed rectification via noise injection is a practical remedy that maintains theoretical guarantees while introducing only controlled approximation error.

## Limitations & Future Work

- Counterexamples are primarily constructed in low-dimensional spaces ($\mathbb{R}^2$); whether subtler failure modes arise in high dimensions remains an open question.
- In practice, the generalization error introduced by neural network approximations of the velocity field may provide a form of implicit regularization that mitigates the severity of these issues.
- No practical guidance is provided for selecting the noise magnitude $c_i$ in smoothed rectification.

## Related Work & Insights

- Complementary to methods that initialize rectified flow with mini-batch OT (PBDALC2023, TFMH2024), which also do not guarantee global optimality but are empirically effective.
- Relation to Schrödinger bridge methods (DSBM): DSBM corresponds to entropy-regularized OT and enjoys convergence guarantees, but convergence may be slow as the regularization parameter vanishes.
- The corrected equivalence theorem (Theorem 11) can guide future rectified flow training by motivating the monitoring of support connectivity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Refutes a published equivalence theorem through carefully constructed counterexamples; theoretical contribution is outstanding.
- **Experimental Thoroughness**: ⭐⭐⭐ Primarily theoretical; numerical validation is provided in the appendix, but large-scale experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematically rigorous and clearly written; counterexample constructions are intuitive and accessible.
- **Value**: ⭐⭐⭐⭐ Corrects an important error in the literature with far-reaching implications for the theory of rectified flows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](../../CVPR2026/image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)

</div>

<!-- RELATED:END -->
