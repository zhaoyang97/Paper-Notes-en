---
title: >-
  [Paper Note] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges
description: >-
  [ICLR2026][Image Generation][Schrödinger bridge] This paper proposes the Non-Conservative Generalized Schrödinger Bridge (NCGSB)—built on contact Hamiltonian mechanics to allow time-varying energy—and introduces the Contact Wasserstein Geodesic (CWG), which reformulates the bridge problem as geodesic computation on a finite-dimensional Jacobi metric. A ResNet parameterization achieves near-linear complexity and supports guided generation. CWG substantially outperforms iterative SB solvers on manifold navigation, molecular dynamics, and image generation tasks.
tags:
  - ICLR2026
  - Image Generation
  - Schrödinger bridge
  - contact Hamiltonian
  - Wasserstein geodesic
  - non-conservative dynamics
  - guided generation
date: 2026-05-08
content_hash: 4fcfb1464daab1ac
---

# Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges

**Conference**: ICLR2026
**arXiv**: [2511.06856](https://arxiv.org/abs/2511.06856)
**Code**: [Project Page](https://sites.google.com/view/c-w-g)
**Area**: Image Generation
**Keywords**: Schrödinger bridge, contact Hamiltonian, Wasserstein geodesic, non-conservative dynamics, guided generation

## TL;DR
This paper proposes the Non-Conservative Generalized Schrödinger Bridge (NCGSB)—built on contact Hamiltonian mechanics to allow time-varying energy—and introduces the Contact Wasserstein Geodesic (CWG), which reformulates the bridge problem as geodesic computation on a finite-dimensional Jacobi metric. A ResNet parameterization achieves near-linear complexity and supports guided generation. CWG substantially outperforms iterative SB solvers on manifold navigation, molecular dynamics, and image generation tasks.

## Background & Motivation

**Background**: The Schrödinger Bridge (SB) provides a principled framework for modeling stochastic processes between two distributions, with broad applications in cell dynamics, weather forecasting, economic modeling, and image generation.

**Limitations of Prior Work**: Existing SB methods assume energy conservation (constant kinetic plus potential energy), which constrains the shape of the bridge and precludes modeling dissipative systems such as gradually weakening storms or cell differentiation. Current SB solvers rely on forward-backward iterative simulation (IPF, matching-based methods, etc.), incurring high computational cost. GSBM restricts expressiveness by assuming Gaussian probability paths, while mmSB suffers from piecewise consistency issues. Momentum SB models damping by augmenting the state with velocity, doubling the state space and computational cost; OU process substitutes lack energy dissipation mechanisms for rotational dynamics.

**Key Insight**: Replace classical Hamiltonians with contact Hamiltonians, introducing only a single scalar state $z^t$ to model energy variation, while exploiting a geometric perspective that reduces SB to geodesic computation, thereby avoiding iterative procedures.

**Core Contributions**: (1) NCGSB non-conservative formulation; (2) CWG near-linear-time solver; (3) guided generation via modification of the Riemannian metric.

## Method

### Overall Architecture
The SB problem is lifted from the infinite-dimensional probability space $\mathcal{P}^+(\mathcal{M})$ to a finite-dimensional parameter space: (1) a contact Hamiltonian introduces a scalar $z^t$ to model energy variation; (2) optimality conditions are shown to correspond to geodesics on the extended space $\mathcal{P}^+(\mathcal{M}) \times \mathbb{R}$; (3) discrete geodesics are parameterized by a ResNet, with each residual block corresponding to one step of distribution transport.

### Key Designs

1. **Non-Conservative Generalized Schrödinger Bridge (NCGSB)**:

    - The cost functional becomes an integral over the time-varying state $z^t$: $\partial_t z^t = \int_\mathcal{M} (\frac{1}{2}\|v^t\|^2 + U(x))\rho^t dx - \gamma z^t$
    - The damping factor $\gamma \in \mathbb{R}$ controls the direction and rate of energy change: $\gamma > 0$ yields energy dissipation, $\gamma < 0$ yields energy growth.
    - The recursive structure endows the system with "memory"—$z^t$ implicitly encodes information about the entire trajectory, enabling modeling of path-dependent non-conservative forces.

2. **Contact Wasserstein Geodesic (CWG) Solver**:

    - Contact Hamiltonian optimality conditions are reformulated as geodesics under the Jacobi metric $\tilde{g}_J = (H - \mathcal{F} - \mathcal{B})g^{\mathcal{W}_2}$.
    - A $(K+1)$-block ResNet parameterizes the discrete geodesic: $T_{\{\theta^k\}} = T_{\theta^K} \circ \cdots \circ T_{\theta^0}$.
    - Complexity is $\mathcal{O}(NK(T_{sh} + D(LW + \log N)))$—linear in dimension $D$ and near-linear in batch size $N$, with no outer iterative loop.

3. **Guided Generation (Guided CWG)**:

    - A guidance term $\|y - f(x^{t_s})\|^2$ is incorporated into the Lagrangian dynamics, equivalent to modifying the Jacobi metric.
    - The modified metric $\tilde{g}'_J = (\Phi^{t_k} + \|y - f(x^{t_s})\|^2) g^{\mathcal{W}_2}$ penalizes geodesics that deviate from the target condition.
    - The model is first trained without guidance, then fine-tuned with the guidance loss—this hybrid approach combines global optimality with local guidance.

### Loss & Training
$$\ell = d_{\mathcal{W}_2}^2(\rho_\theta^{t_K}, \rho_b) + \sum_m d_{\mathcal{W}_2}^2(\rho_\theta^{t_{k_m}}, \rho_m) + \sum_k \Phi^{t_k} d_{\mathcal{W}_2}^2(\rho_\theta^{t_k}, \rho_\theta^{t_{k-1}})$$
The three terms respectively enforce terminal marginal matching, intermediate marginal matching, and minimization of energy-weighted geodesic length.

## Key Experimental Results

### LiDAR Manifold Navigation + Single-Cell Sequencing

| Task / Metric | CWG (Ours) | GSBM | DSBM | SBIRR / DM-SB |
|---|---|---|---|---|
| LiDAR Optimality ↓ | **1.40** | 2.18 | 4.16 | — |
| LiDAR Feasibility ↓ | **0.06** | 0.83 | 0.97 | — |
| LiDAR Training Time (s) | **280** | 1570 | 1340 | — |
| Single-Cell $d_{\mathcal{W}_2}(x^{t_3})$ ↓ | **0.33** | — | — | 1.64 / 1.86 |
| Single-Cell Training Time (s) | **710** | — | — | 38120 / 1740 |

### Image Generation Tasks

| Task / Metric | CWG (Ours) | GSBM | DSBM | SB-Flow |
|---|---|---|---|---|
| Sea Surface Temp. FID($x^{t_1}$) ↓ | **121** | 161 | 242 | 177 |
| Robot Reconstruction FID ↓ | **19** | 40 | 150 | 73 |
| Robot Training Time (h) | **0.5** | 25.3 | 7.6 | 1.4 |
| FFHQ Feasibility ↓ | **4.33** | 6.84 | 7.78 | 21.75 |
| FFHQ Training Time (s) | **930** | 2650 | 2530 | 1490 |

## Highlights & Insights
- **Elegant bridge from contact mechanics to generative modeling**: Introducing only the scalar $z^t$ via contact Hamiltonian mechanics breaks the energy conservation constraint far more efficiently than Momentum SB, which doubles the state space.
- **ResNet as discrete geodesic**: Each residual block is interpreted as a one-step pushforward map on the probability manifold, yielding a theoretically grounded and concise implementation.
- **Remarkable speed advantage**: CWG is **50×** faster than DM-SB on single-cell tasks and **50×** faster than GSBM on robot reconstruction tasks.
- **Geometric interpretation of guided generation**: The guidance term directly modifies the Riemannian metric rather than injecting gradients during sampling, making it more intrinsic than classifier guidance.

## Limitations & Future Work
- Empirical estimation of $d_{\mathcal{W}_2}$ is unstable in high dimensions; image experiments partially rely on VAE latent spaces.
- Guided generation requires pre-training an unguided model before fine-tuning, and is thus not end-to-end.
- The choice of $\gamma$ depends on prior knowledge of dissipation direction and rate; no adaptive tuning mechanism is provided.
- The number of ResNet blocks $K$ determines temporal discretization accuracy: too few blocks yield coarse geodesic approximations, while too many increase parameter count.

## Related Work & Insights
- **vs. GSBM (Liu et al., 2024)**: GSBM assumes Gaussian paths and requires iterative solving; CWG imposes neither restriction and is non-iterative—achieving 36% lower Optimality scores and over 5× faster training.
- **vs. Momentum SB (Blessing et al., 2025)**: Momentum SB doubles the state space to model damping; NCGSB requires only one additional scalar—a more elegant approach to non-conservative modeling.
- **vs. SB-Flow (Bortoli et al., 2024)**: SB-Flow relies on iterative IPF and does not support GSB, mmSB, energy variation, or guided generation; CWG covers all of these.
- **Insight**: Contact geometry is a mathematical tool that has yet to be fully exploited in deep learning, with broad prospects in fields requiring dissipation modeling, such as materials science and climate simulation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing contact Hamiltonian mechanics into SB is an entirely new perspective with deep theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers manifold navigation, molecular dynamics, and multiple image tasks with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though the barrier is high for readers without a geometry background.
- Value: ⭐⭐⭐⭐⭐ Combines theoretical depth with practical speed advantages; a significant advance in the SB literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](branched_schrödinger_bridge_matching.md)
- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](../../NeurIPS2025/image_generation/schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [\[ICLR 2026\] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)
- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](../../ACL2026/image_generation/zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)
- [\[ICLR 2026\] Does FLUX Already Know How to Perform Physically Plausible Image Composition?](does_flux_already_know_how_to_perform_physically_plausible_image_composition.md)

</div>

<!-- RELATED:END -->
