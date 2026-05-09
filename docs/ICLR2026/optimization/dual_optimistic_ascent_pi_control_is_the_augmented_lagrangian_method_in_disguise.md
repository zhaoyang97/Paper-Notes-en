---
title: >-
  [Paper Note] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise
description: >-
  [ICLR 2026][Optimization][Augmented Lagrangian] This paper proves that dual optimistic ascent (PI control), widely used in constrained deep learning, is mathematically equivalent to the classical Augmented Lagrangian Method (ALM) under the single-step first-order update regime. This equivalence transfers ALM's robust convergence guarantees (linear convergence to all strict local solutions) to PI control, and provides principled tuning guidance for the optimism coefficient $\omega$.
tags:
  - ICLR 2026
  - Optimization
  - Augmented Lagrangian
  - Dual Optimistic Ascent
  - PI Control
  - Constrained Optimization
  - Non-Convex Min-Max
date: 2026-05-08
content_hash: 2672c7633b298b12
---

# Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise

**Conference**: ICLR 2026
**arXiv**: [2509.22500](https://arxiv.org/abs/2509.22500)

**Code**: [GitHub](https://github.com/juan43ramirez/pi-control-is-alm)
**Area**: Optimization Theory
**Keywords**: Augmented Lagrangian, Dual Optimistic Ascent, PI Control, Constrained Optimization, Non-Convex Min-Max

## TL;DR

This paper proves that dual optimistic ascent (PI control), widely used in constrained deep learning, is mathematically equivalent to the classical Augmented Lagrangian Method (ALM) under the single-step first-order update regime. This equivalence transfers ALM's robust convergence guarantees (linear convergence to all strict local solutions) to PI control, and provides principled tuning guidance for the optimism coefficient $\omega$.

## Background & Motivation

- **Background**: Numerous DL applications (fairness, safety, RLHF alignment, etc.) require imposing constraints during training. The standard approach applies first-order gradient descent-ascent (GDA) on the Lagrangian $\mathcal{L}(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\mu}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^\top \boldsymbol{g}(\boldsymbol{x}) + \boldsymbol{\mu}^\top \boldsymbol{h}(\boldsymbol{x})$, favored for its scalability and compatibility with optimizers such as Adam.

- **Limitations of Prior Work**: GDA suffers from two fundamental drawbacks: (1) it cannot converge to all local constrained optima in the non-convex setting — only to local min-max points of the Lagrangian; (2) multipliers and constraint values exhibit oscillation, with iterates alternating between feasible and infeasible regions, leading to slow convergence that is unacceptable in safety-critical scenarios.

- **Key Challenge**: ALM resolves these issues by adding a quadratic penalty term $\frac{c}{2}\|\boldsymbol{h}(\boldsymbol{x})\|^2$, making the augmented Lagrangian strictly convex at all strictly regular local solutions, thus guaranteeing convergence to all local solutions and suppressing oscillation. Yet in practice, the community favors applying dual optimistic ascent directly on the standard Lagrangian.

- **Core Problem**: PI control / dual optimistic ascent (stooke2020responsive; sohrabi2024nupi) empirically suppresses oscillation effectively in RL, unsupervised, and supervised learning, but its convergence properties are almost entirely unformalized. Existing OGDA results either rely on overly strong assumptions (strong convexity-strong concavity) or have mismatched algorithmic structures (applying optimism to both players).

- **Goal**: Motivated by observations from Gallego-Posada and Mitliagkas, this work explores whether dual optimistic ascent and ALM share a deeper connection. The central finding is that under the single-step first-order update regime, dual optimistic ascent and ALM-GDA produce identical primal iterates for equality constraints (Theorem 1) and converge to the same set of locally stable stationary points for inequality constraints (Theorem 2), enabling a complete transfer of theoretical guarantees.

## Method

### 1. Problem Setup and Three Algorithms

Consider the constrained optimization problem:

$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad \boldsymbol{g}(\boldsymbol{x}) \preceq \boldsymbol{0}, \; \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}$$

**Standard Lag-GDA** (baseline, prone to oscillation):

$$\boldsymbol{\mu}_{t+1} \leftarrow \boldsymbol{\mu}_t + \eta_{\text{dual}} \boldsymbol{h}(\boldsymbol{x}_t), \quad \boldsymbol{\lambda}_{t+1} \leftarrow [\boldsymbol{\lambda}_t + \eta_{\text{dual}} \boldsymbol{g}(\boldsymbol{x}_t)]_+$$

**Dual Optimistic Ascent (Lag-GD-OA)** (PI control): augments the dual update with an optimistic term $\omega[\boldsymbol{h}(\boldsymbol{x}_t) - \boldsymbol{h}(\boldsymbol{x}_{t-1})]$:

$$\boldsymbol{\mu}_{t+1} \leftarrow \boldsymbol{\mu}_t + \eta_{\text{dual}} \boldsymbol{h}(\boldsymbol{x}_t) + \omega[\boldsymbol{h}(\boldsymbol{x}_t) - \boldsymbol{h}(\boldsymbol{x}_{t-1})]$$

**ALM-GDA** (GDA on the augmented Lagrangian): applies primal-first GDA on $\mathcal{L}_c$:

$$\boldsymbol{x}_{t+1} \leftarrow \boldsymbol{x}_t - \eta_{\boldsymbol{x}} \nabla_{\boldsymbol{x}} \mathcal{L}_c(\boldsymbol{x}_t, \boldsymbol{\lambda}_t, \boldsymbol{\mu}_t)$$

where the augmented Lagrangian is:

$$\mathcal{L}_c(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\mu}) = f(\boldsymbol{x}) + \frac{1}{2c}\left[\|\boldsymbol{\mu} + c\boldsymbol{h}(\boldsymbol{x})\|^2 - \|\boldsymbol{\mu}\|^2 + \|[\boldsymbol{\lambda} + c\boldsymbol{g}(\boldsymbol{x})]_+\|^2 - \|\boldsymbol{\lambda}\|^2\right]$$

### 2. Core Equivalence Theorems

**Theorem 1 (Exact Equivalence for Equality Constraints)**: When $\omega = c > 0$ and initialization satisfies $\boldsymbol{\mu}_0^{\text{OGA}} = \boldsymbol{\mu}_0^{\text{ALM}} + (c - \eta_{\text{dual}})\boldsymbol{h}(\boldsymbol{x}_0)$, ALM-GDA and Lag-GD-OA produce identical primal iterate sequences $\{\boldsymbol{x}_t\}_{t=0}^{\infty}$.

The key insight is that the ALM primal gradient employs a "look-ahead" multiplier $\boldsymbol{\mu}_t + c\boldsymbol{h}(\boldsymbol{x}_t)$, which coincides exactly with the effective multiplier accumulated via the dual optimistic ascent update. This equivalence holds for any first-order primal optimizer, including Adam.

**Theorem 2 (LSSP Equivalence for Inequality Constraints)**: Under strict complementary slackness, ALM-GDA (penalty $c$) converges locally to $(\boldsymbol{x}^*, \boldsymbol{\lambda}^*)$ if and only if Lag-GD-OA (optimism coefficient $\omega = c$) also converges locally to that point. The spectral radii of the two algorithms' Jacobians satisfy:

$$\rho(\mathcal{J}_{\text{AL}}) = \max\{\rho(\mathcal{J}_{\text{OG}}), 1 - \eta_{\text{dual}}/c\}$$

### 3. Theoretical Corollaries and Practical Guidance

**Theorem 3 (Dual Optimistic Ascent Recovers All Local Solutions)**: $\boldsymbol{x}^*$ is a strict local constrained minimum of the CMP $\Leftrightarrow$ there exists $\bar{\omega} \geq 0$ such that for all $\omega \geq \bar{\omega}$, $\boldsymbol{x}^*$ is an LSSP of Lag-GD-OA. This strictly dominates standard Lag-GDA.

**Corollary 2 (Local Linear Convergence)**: Lag-GD-OA exhibits local linear convergence to all strict local minima satisfying regularity conditions under appropriate hyperparameters. When $\eta_{\text{dual}}$ is sufficiently close to $\omega = c$, the convergence rate exactly matches that of ALM-GDA.

**Corollary 3 (Global Linear Convergence for Convex Equality Constraints)**: For convex smooth objectives with affine equality constraints, Lag-GD-OA converges globally at a linear rate.

### Key Designs: Three-Way Trade-off of $\omega$

| Effect | Increasing $\omega$ | $\omega$ Too Large |
|--------|:---:|:---:|
| Reachable solution set | Monotonically non-decreasing (Corollary 4) | Covers all local solutions |
| Oscillation suppression | Eigenvalues tend toward purely real (Proposition 5) | Oscillation fully eliminated |
| Condition number | — | Tends to $\infty$ (Corollary 5) |

**Proposition 5 (Oscillation Suppression)**: There exists a finite threshold $\bar{\omega}$ such that for $\omega \geq \bar{\omega}$, all Jacobian eigenvalues are real; as $\omega \to \bar{\omega}^{-}$, the maximum imaginary part decays as $\mathcal{O}(\sqrt{\bar{\omega} - \omega})$.

**Practical Recommendation**: The classical ALM penalty-increase schedule can be directly adopted to adaptively tune $\omega$, e.g., set $\omega_{t+1} = \gamma \omega_t$ when $\|h(\boldsymbol{x}_t)\| > \beta \|h(\boldsymbol{x}_{t-1})\|$.

## Key Experimental Results

### Experimental Setup

1D equality-constrained problem: $\min_x \frac{1}{2}x^2 \;\text{s.t.}\; e^x = e$ (solution $x^* = 1$), with constraint written as $h(x) = e^x - e$ to induce a nonlinear landscape.

| Hyperparameter | Value |
|--------|-----|
| Primal optimizer | GD + Polyak Momentum |
| $\eta_x$ | 0.01 |
| Momentum | 0.5 |
| $\eta_{\text{dual}}$ | 0.1 |
| $\omega / c$ | 1.0 |
| $x_0$ | 2.0 |
| ALM multiplier initialization | 0.0 |

### Experiment 1: Primal Iterate Matching (Validating Theorem 1)

| Method | Primal iterates $\{x_t\}$ | Effective multiplier | Convergence to $x^*=1$ |
|------|:---:|:---:|:---:|
| ALM-GDA | Sequence A | $\mu_t^{\text{ALM}} + c \cdot h(x_t)$ | ✓ |
| Dual Optimistic Ascent | Sequence B ≡ Sequence A | $\mu_t^{\text{OGA}}$ | ✓ |

Figure 1 numerically verifies that both methods produce identical primal iterate trajectories with exactly matching effective multipliers. While the internal multiplier values differ, the primal gradients are identical.

### Experiment 2: $\omega$ Scheduling Strategy (Validating §4.3 Practical Guidance)

| Configuration | $\omega$ Strategy | Multiplier Overshoot | Convergence Quality |
|------|:---:|:---:|:---:|
| Fixed $\omega$ | Constant | Large overshoot | Convergent but oscillatory |
| Adaptive $\omega$ ($\gamma=2, \beta=0.99$, tol=$10^{-2}$) | ALM increase schedule | Significantly reduced | Smooth convergence without overshoot |

Figure 2 demonstrates the advantages of the adaptive scheduling strategy in reducing multiplier overshoot and primal iterate oscillation.

## Key Findings

- **Core mechanism of equivalence**: The ALM primal gradient uses a "look-ahead multiplier" $\boldsymbol{\mu}_t + c \boldsymbol{h}(\boldsymbol{x}_t)$, which coincides exactly with the effective multiplier accumulated through the optimistic term in dual optimistic ascent.
- The equivalence is **exact for equality constraints** (iterate-level matching) and **stability-level for inequality constraints** (identical LSSP sets).
- The gap in the inequality constraint case stems from the differing placement of the projection $[\cdot]_+$: Lag-GD-OA projects once whereas ALM-GDA projects twice.
- **Equivalence holds only under the single-step first-order update regime**: multi-step primal updates or second-order methods break the equivalence — in such cases, ALM should be used directly.
- Applying dual optimistic ascent on top of the augmented Lagrangian produces a compounded effect, equivalent to ALM on $\mathcal{L}_{c+\omega}$ (Proposition 6).

## Highlights & Insights

- **The "in disguise" perspective is remarkably elegant**: it unifies two independently developed communities — ALM from optimization theory and PI control from control theory/RL — under a single algorithmic identity, analogous to work connecting Adam with natural gradient methods.
- **Bridge from theory to practice**: Decades of ALM hyperparameter tuning experience (penalty-increase schedules) can be directly transferred to tuning $\omega$ in PI control.
- **Clear methodological boundaries**: The paper explicitly identifies when the equivalence holds (single-step, first-order) and when it breaks down (multi-step / second-order), providing actionable recommendations rather than vague claims.
- Dual optimistic ascent is proven to strictly dominate naive GDA: it not only suppresses oscillation but also recovers local solutions unreachable by GDA.
- The results have direct implications for safety/alignment constrained optimization in RLHF, providing theoretical grounding for the empirical success of works such as stooke2020responsive and sohrabi2024nupi.

## Limitations & Future Work

- **Limited experimental scale**: Only a 1D synthetic experiment validates the equivalence; large-scale constrained DL experiments (e.g., RLHF, fairness training) demonstrating convergence improvements are absent.
- **Inequality constraints: only local equivalence**: The global behaviors of the two algorithms may differ, and global convergence analysis for inequality constraints is not addressed.
- **Limitation of single-step first-order regime**: The equivalence breaks under multi-step primal updates and second-order methods, yet multi-step inner loops are sometimes employed in deep learning.
- **Stochastic / mini-batch setting not discussed**: In practice, $\boldsymbol{g}(\boldsymbol{x}_t)$ and $\boldsymbol{h}(\boldsymbol{x}_t)$ may be estimated from mini-batches; the effect of noise on the equivalence is not analyzed.
- **Optimal $\omega$ depends on the unknown solution $\boldsymbol{x}^*$**: The threshold $\bar{\omega}$ cannot be directly computed in practice, leaving only heuristic tuning.

## Related Work & Insights

- **vs. Standard Lag-GDA**: Lag-GDA can only converge to local min-max points of the Lagrangian, potentially missing solutions that are non-convex yet reachable within the constraint tangent space. Dual optimistic ascent / ALM renders the augmented Lagrangian strictly convex at all regular local solutions, enabling convergence to all local solutions satisfying LICQ and second-order sufficient conditions.
- **vs. OGDA (two-sided optimism)**: OGDA applies optimism to both players and yields results such as global linear convergence under strong convexity-strong concavity. However, applying optimism solely to the dual side (as in this work) preserves flexibility in the primal optimizer (e.g., Adam), and existing OGDA assumptions are mismatched with the non-convex-linear structure of constrained optimization.
- **vs. Method of Multipliers (classical ALM)**: Classical ALM requires approximate minimization of $\mathcal{L}_c$ via multi-step inner loops at each iteration. The AL-GDA variant studied here replaces this with a single step and decouples $\eta_{\text{dual}}$ from $c$, aligning more naturally with deep learning practice.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Reveals the mathematical equivalence of two independently developed methods; the core result is concise and surprising.

- Experimental Thoroughness: ⭐⭐⭐ — Primarily theoretical, with only a 1D synthetic experiment for validation; large-scale DL experiments are absent.

- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous; the "in disguise" title is highly engaging; scope claims are honest and precise.

- Value: ⭐⭐⭐⭐ — Unifies two directions in constrained optimization theory and provides principled guidance for RL/RLHF practitioners.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](../../NeurIPS2025/optimization/memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)
- [\[NeurIPS 2025\] Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](../../NeurIPS2025/optimization/optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)
- [\[NeurIPS 2025\] Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems](../../NeurIPS2025/optimization/extragradient_method_for_l_0_l_1-lipschitz_root-finding_problems.md)
- [\[NeurIPS 2025\] Revisiting Orbital Minimization Method for Neural Operator Decomposition](../../NeurIPS2025/optimization/revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)
- [\[NeurIPS 2025\] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](../../NeurIPS2025/optimization/mdns_masked_diffusion_neural_sampler_via_stochastic_optimal_control.md)

<!-- RELATED:END -->
