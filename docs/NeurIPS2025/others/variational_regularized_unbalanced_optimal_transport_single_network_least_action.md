---
title: >-
  [Paper Note] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action
description: >-
  [NeurIPS 2025][Regularized unbalanced optimal transport] This paper proposes Var-RUOT, which incorporates the necessary optimality conditions of the Regularized Unbalanced Optimal Transport (RUOT) problem into the parameterization and loss design, enabling the solution of RUOT by learning a single scalar field. The approach yields solutions with lower action and improves training stability, while also analyzing the effect of growth penalty functions on biological priors.
tags:
  - NeurIPS 2025
  - Regularized unbalanced optimal transport
  - variational methods
  - least action principle
  - single scalar field
  - single-cell trajectory inference
date: 2026-05-08
content_hash: e21aefe9aa0442c9
---

# Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action

**Conference**: NeurIPS 2025
**arXiv**: [2505.11823](https://arxiv.org/abs/2505.11823)
**Code**: [GitHub](https://github.com/ZerooVector/VarRUOT)
**Area**: Other
**Keywords**: Regularized unbalanced optimal transport, variational methods, least action principle, single scalar field, single-cell trajectory inference

## TL;DR

This paper proposes Var-RUOT, which incorporates the necessary optimality conditions of the Regularized Unbalanced Optimal Transport (RUOT) problem into the parameterization and loss design, enabling the solution of RUOT by learning a single scalar field. The approach yields solutions with lower action and improves training stability, while also analyzing the effect of growth penalty functions on biological priors.

## Background & Motivation

Recovering continuous dynamics of high-dimensional systems from limited snapshot data is a central challenge in statistical physics and computational biology. In single-cell RNA sequencing, only a few snapshot measurements at sparse time points are available, necessitating the reconstruction of continuous cellular trajectories.

Various frameworks have been proposed to address this problem: dynamic optimal transport (Benamou–Brenier), Schrödinger Bridge, unbalanced dynamic OT (Wasserstein–Fisher–Rao), etc. The RUOT framework unifies stochasticity and particle birth–death processes. However, existing deep learning solvers face two major challenges:

**Optimality conditions are not explicitly enforced**: Existing methods typically parameterize the velocity field $u$ and growth rate $g$ with independent neural networks, without exploiting the optimality relationship between them. This causes solutions to deviate from the least action principle and leads to unreliable convergence.

**Lack of guidance for choosing the growth penalty function**: The standard WFR metric uses $\psi(g) = g^2/2$, but different choices of $\psi$ implicitly encode different biological priors — a point that has not been sufficiently studied.

## Method

### Overall Architecture

The core idea of Var-RUOT is to derive the necessary optimality conditions of RUOT via variational methods, revealing that both the velocity field $u$ and growth rate $g$ can be fully determined by a single scalar field $\lambda(x, t)$. Consequently, only one neural network is needed to parameterize $\lambda$, substantially simplifying the problem.

### Key Designs

1. **Derivation of Necessary Optimality Conditions (Theorem 4.1)**: For RUOT problems with isotropic time-invariant diffusion, the variational method yields three necessary conditions:

   - $u = \nabla_x \lambda$ (the velocity field is the gradient of the scalar field)
   - $\alpha \cdot \psi'(g) = \lambda$ (the growth rate is implicitly determined by the scalar field)
   - HJB equation: $\partial\lambda/\partial t + \frac{1}{2}\|\nabla\lambda\|^2 + \frac{1}{2}\sigma^2 \nabla^2\lambda + \lambda g - \alpha\psi(g) = 0$

   Key insight: once $\lambda$ is determined, $u$ and $g$ are automatically fixed, and the evolution governed by the Fokker–Planck equation is fully determined. This reduces the problem from learning $u$ and $g$ separately to learning a single scalar field.

2. **Growth Penalty Function and Biological Priors (Theorem 4.2)**: The sign of $\psi''(g)$ is shown to govern the monotonicity of $g$ along the velocity field direction:

   - $\psi''(g) > 0$ (e.g., $g^2/2$ in standard WFR): $g$ increases along $u$ — downstream cells proliferate faster.
   - $\psi''(g) < 0$ (e.g., $g^{2/15}$ proposed in this paper): $g$ decreases along $u$ — upstream stem cells proliferate fastest.

   The latter better matches biological priors: stem cells reside at the upstream end of the trajectory, exhibit the highest proliferative and differentiation capacity, and $g$ should decrease along the differentiation direction. Accordingly, the paper proposes using $\psi_2(g) = g^{2/15}$ as a more biologically grounded alternative.

3. **Weighted Particle Method (Theorem 5.1)**: The solution to the Fokker–Planck equation is approximated by $N$ weighted particles. Each particle position follows an SDE and each weight follows an ODE:

   - $dX_i = u(X_i, t)\,dt + \sigma\,dW_t$
   - $dw_i = g(X_i, t)\,w_i\,dt$

   The empirical measure $\mu^N$ converges to the true density $\rho$ as $N \to \infty$.

4. **Three-Component Loss Function**:

   - **Reconstruction Loss $L_\text{Recon}$**: Comprises a mass-matching loss ($\hat{M}(T_k) \approx M(T_k)$) and a Wasserstein-2 distributional distance.
   - **HJB Loss $L_\text{HJB}$**: Integrates the violation of the HJB equation along particle trajectories, enforcing the optimality condition on $\lambda$.
   - **Action Loss $L_\text{Action}$**: Directly minimizes the transport action (since necessary conditions are insufficient, explicit optimization is still required).

### Loss & Training

The joint objective $L = L_\text{Recon} + \gamma_\text{HJB} \cdot L_\text{HJB} + \gamma_\text{Action} \cdot L_\text{Action}$ is minimized. The Euler–Maruyama method is used to discretize the SDE, and automatic differentiation computes $\nabla\lambda$ and $\nabla^2\lambda$. Weight normalization is applied during training for the weighted HJB loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | Var-RUOT | DeepRUOT | Other Baselines |
|---------|--------|----------|----------|-----------------|
| Three-gene simulation ($t=1$) | $W_1$ | **0.0452** | 0.0569 | TIGON: 0.0519 |
| Three-gene simulation ($t=2$) | $W_1$ | **0.0385** | 0.0811 | OTCFM: 0.2078 |
| Three-gene simulation ($t=4$) | $W_1$ | **0.0572** | 0.1538 | UOTCFM: 0.4129 |
| Three-gene simulation | Path action | **1.1105** | 1.4058 | TIGON: 1.2442 |
| EMT dataset (10D) | Trajectory morphology | Near-straight | Curved | — |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Standard WFR $\psi_1(g)=g^2/2$ | $g$ increases along $u$ | Inconsistent with the biological prior of stem cells at the upstream end |
| Modified $\psi_2(g)=g^{2/15}$ | $g$ decreases along $u$ | Consistent with biological prior |
| Without HJB loss | Higher action | HJB constraint is critical for finding least-action paths |
| Without action loss | May converge to saddle point | Necessary conditions alone are insufficient; explicit optimization is required |
| Convergence comparison | Training epochs | Var-RUOT converges faster and more stably |

### Key Findings

- Var-RUOT achieves 21% lower action than DeepRUOT on three-gene simulation data (1.11 vs. 1.41), while also attaining higher distributional matching accuracy.
- On the EMT dataset, trajectories learned by Var-RUOT approach straight lines (corresponding to least action), whereas DeepRUOT learns curved trajectories.
- Single scalar field parameterization substantially simplifies the optimization landscape, leading to more stable and faster convergence.
- The choice of $\psi$ genuinely affects the learned biological dynamics — a factor completely overlooked in prior work.

## Highlights & Insights

- **Elegant theoretical simplification**: Variational analysis reduces the problem of learning two fields ($u$ and $g$) to learning a single scalar field $\lambda$, substantially lowering optimization difficulty.
- **Embedding physical constraints into network design**: Rather than imposing them as external penalties, the parameterization space is directly restructured — $u = \nabla\lambda$ inherently guarantees a curl-free velocity field.
- **Biological interpretation of the growth penalty function**: This work is the first to reveal the connection between the sign of $\psi''(g)$ and the direction of cellular development, offering actionable modeling guidance for computational biology.
- **Generalization of Action Matching**: Extends the framework of Neklyudov et al. (2023, 2024) to simultaneously handle unbalanced and stochastic dynamics.

## Limitations & Future Work

- Only isotropic time-invariant diffusion ($\sigma^2 I$) is considered; anisotropic or time-varying diffusion matrices are not addressed.
- The choice of $\psi_2(g) = g^{2/15}$, while theoretically justified, is somewhat ad hoc; more systematic selection criteria remain to be developed.
- Computing $\nabla^2\lambda$ via automatic differentiation may become a computational bottleneck in high-dimensional settings (Hessian trace estimation).
- Experiments are conducted only in low-to-moderate dimensions (3D, 10D); scalability to full-dimensional single-cell data (thousands of genes) has not been validated.
- Particle degeneracy in the weighted particle method (some particle weights approaching zero) is not discussed.

## Related Work & Insights

- Action Matching (Neklyudov et al., 2023) and WLF (Neklyudov et al., 2024) are the most direct precursors; this paper extends their framework to the full RUOT setting.
- DeepRUOT (Zhang et al., 2025a) parameterizes $u$ and $g$ with independent networks and serves as the primary baseline.
- TIGON (Sha et al., 2024) demonstrates strong performance in computational biology applications, but similarly does not exploit optimality conditions.
- The work provides a new paradigm for applying optimal control and variational methods in machine learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The idea of embedding variational optimality conditions into neural network parameterization is highly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ Core claims are validated, but experimental scope is limited (low dimensions, few datasets).
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and the structure is clear.
- **Value**: ⭐⭐⭐⭐ Directly valuable for trajectory inference and computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[NeurIPS 2025\] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](../../AAAI2026/others/optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[NeurIPS 2025\] Neural Network for Simulating Radio Emission from Extensive Air Showers](neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)

</div>

<!-- RELATED:END -->
