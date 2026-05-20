---
title: >-
  [Paper Note] Adjoint Schrödinger Bridge Sampler
description: >-
  [NeurIPS 2025][Schrödinger Bridge] This paper proposes the Adjoint Schrödinger Bridge Sampler (ASBS), which reinterprets the Schrödinger Bridge problem as a stochastic optimal control (SOC) problem. This eliminates the m…
tags:
  - "NeurIPS 2025"
  - "Schrödinger Bridge"
  - "Diffusion Sampler"
  - "Boltzmann Distribution"
  - "Adjoint Matching"
  - "Stochastic Optimal Control"
date: 2026-05-08
content_hash: 247e49a4c16cd5f7
---

# Adjoint Schrödinger Bridge Sampler

**Conference**: NeurIPS 2025
**arXiv**: [2506.22565](https://arxiv.org/abs/2506.22565)  
**Code**: [https://github.com/facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_samplers)  
**Area**: Diffusion Models / Sampling Methods / Molecular Simulation
**Keywords**: Schrödinger Bridge, Diffusion Sampler, Boltzmann Distribution, Adjoint Matching, Stochastic Optimal Control

## TL;DR
This paper proposes the Adjoint Schrödinger Bridge Sampler (ASBS), which reinterprets the Schrödinger Bridge problem as a stochastic optimal control (SOC) problem. This eliminates the memoryless condition required by prior diffusion samplers, supports arbitrary source distributions (e.g., Gaussian, harmonic priors), and employs a scalable matching objective without importance weight estimation. ASBS consistently outperforms prior methods on multi-particle energy functions and molecular conformation generation.

## Background & Motivation

**Background**: Sampling from the Boltzmann distribution $\nu(x) \propto e^{-E(x)}$ is a core problem in computational science (Bayesian inference, statistical physics, chemistry). Traditional MCMC methods suffer from slow mixing and expensive energy evaluations. Recent diffusion samplers learn an SDE drift $u_t^\theta$ to transport samples toward the target distribution.

**Limitations of Prior Work**: Since the Boltzmann distribution is only accessible through its unnormalized energy function without explicit samples, prior matching-based diffusion samplers (PDDS, iDEM) rely on importance-weighted estimation of target samples, incurring significant computational overhead. Adjoint Sampling (AS) avoids importance weights via Adjoint Matching, but is restricted by the memoryless condition—requiring the source distribution to be a Dirac delta $\mu(x) = \delta$.

**Key Challenge**: The memoryless condition excludes useful source distributions such as Gaussian and harmonic priors. While non-memoryless processes are known to improve transport efficiency, existing methods either require memorylessness or rely on expensive non-matching approaches.

**Goal**: Learn a diffusion sampler using a scalable matching objective, without requiring the memoryless condition or importance weight estimation.

**Key Insight**: The optimality conditions of the Schrödinger Bridge problem are recast as an SOC problem, and a corrector function $\nabla \log \hat{\varphi}_1$ is introduced to eliminate the initial-value bias induced by non-memoryless source distributions.

**Core Idea**: By alternately optimizing Adjoint Matching (learning drift $u$) and Corrector Matching (learning debiasing corrector $h$), the procedure is equivalent to the IPF algorithm and converges to the global optimum of the Schrödinger Bridge.

## Method

### Overall Architecture
ASBS learns an SDE $dX_t = [f_t(X_t) + \sigma_t u_t^\theta(X_t)] dt + \sigma_t dW_t$ that transports samples from a source distribution $\mu$ to the target Boltzmann distribution $\nu$. Unlike AS, the source distribution can be arbitrary (Gaussian, harmonic prior, etc.), with base drift $f_t = 0$ (Brownian motion). The algorithm alternately trains two networks: a drift network $u_\theta$ and a corrector network $h_\phi$.

### Key Designs

1. **SOC Characterization of the SB Problem (Theorem 3.1)**:

    - Function: Proves that the dynamically optimal drift $u_t^*$ of the Schrödinger Bridge can be obtained by solving an SOC problem with a specific terminal cost.
    - Mechanism: The SB optimality equations involve coupled SB potentials $\varphi_t, \hat{\varphi}_t$, which are difficult to solve directly. The key observation is that the integral form of the forward SB potential $\varphi_t$ resembles the SOC optimality condition, making the SB problem equivalent to an SOC problem with terminal cost $g(x) = \log \frac{\hat{\varphi}_1(x)}{\nu(x)}$.
    - Design Motivation: Transforms the intractable SB problem into an SOC problem amenable to Adjoint Matching, preserving the scalability of AS.

2. **Corrector Matching for Debiasing (Eq. 15)**:

    - Function: Learns a corrector function $h_\phi \approx \nabla \log \hat{\varphi}_1$ to eliminate the bias arising from non-memoryless source distributions.
    - Mechanism: $\nabla \log \hat{\varphi}_1$ is the Markovian projection of the time-reversed dynamically optimal drift at $t=1$, and can be learned via regression $\min_h \mathbb{E}_{p_{0,1}^{u^{(k)}}} [\|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1|X_0)\|^2]$. Crucially, this objective depends only on samples from the model itself and requires no samples from the target distribution.
    - Design Motivation: When the source distribution is a Dirac delta, $\nabla \log \hat{\varphi}_1 = \nabla \log p_1^{\text{base}}$ is known analytically and requires no additional learning; for arbitrary source distributions, the corrector must be learned explicitly.

3. **Alternating Optimization = IPF (Theorem 3.2)**:

    - Function: Proves that alternating between Adjoint Matching and Corrector Matching is equivalent to Iterative Proportional Fitting (IPF).
    - Mechanism: AM solves the forward half-bridge—fixing the source distribution $\mu$ and minimizing $D_{KL}(p \| q^{\bar{h}^{(k-1)}})$; CM solves the backward half-bridge—fixing the target distribution $\nu$ and minimizing $D_{KL}(p^{u^{(k)}} \| q)$. Their alternation is equivalent to IPF, guaranteeing global convergence $\lim_{k \to \infty} u^{(k)} = u^*$.
    - Design Motivation: The convergence guarantees of IPF ensure that ASBS converges to the global SB optimum without additional hyperparameter tuning.

### Loss & Training
- **AM Loss**: $\min_u \mathbb{E}[\|u_t(X_t) + \sigma_t(\nabla E + h^{(k-1)})(X_1)\|^2]$, regressing onto energy gradients plus the corrector.
- **CM Loss**: $\min_h \mathbb{E}[\|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1|X_0)\|^2]$.
- Initialized with $h^{(0)} = 0$; the first stage is equivalent to standard AS.
- A replay buffer stores historical samples; Adam optimizer is used throughout.
- Molecular systems use an equivariant graph neural network (EGNN) with a harmonic prior as the source distribution.

## Key Experimental Results

### Main Results

| Energy Function | Metric | ASBS | AS | Best Other | Gain |
|----------------|--------|------|----|-----------|------|
| MW-5 (d=5) | Sinkhorn ↓ | **0.15** | 0.32 | 0.44 (SCLD) | −53% vs. AS |
| DW-4 (d=8) | $\mathcal{W}_2$ ↓ | **0.43** | 0.62 | 0.68 (PIS) | −31% vs. AS |
| LJ-13 (d=39) | $\mathcal{W}_2$ ↓ | **1.59** | 1.67 | 1.61 (iDEM) | −5% vs. AS |
| LJ-55 (d=165) | $\mathcal{W}_2$ ↓ | **4.00** | 4.04 | 4.60 (DDS) | −1% vs. AS |
| Alanine Dipeptide | $D_{KL}(\phi)$ ↓ | **0.02** | 0.09 | 0.03 (DDS) | −78% vs. AS |

### Ablation Study (Conformation Generation)

| Configuration | SPICE Coverage ↑ | GEOM Coverage ↑ | Notes |
|--------------|-----------------|-----------------|-------|
| ASBS + harmonic prior | **73.04%** | **50.23%** | Full model with domain prior |
| ASBS + Gaussian prior | 67.58% | 41.23% | Without domain prior |
| AS (Dirac prior) | 56.75% | 36.23% | Baseline, memoryless |
| RDKit ETKDG | 56.94% | 50.81% | Chemistry heuristic |
| ASBS + harmonic + RDKit warmup | **85.82%** | **66.79%** | Strongest configuration |

### Key Findings
- **ASBS consistently outperforms AS and all other methods across all synthetic energy functions**, with particularly large gains in low dimensions (MW-5, DW-4: 30–50%) and smaller gains in high dimensions (LJ-55).
- **Harmonic prior outperforms both Gaussian prior and Dirac delta**: confirming that domain-specific source distributions improve transport efficiency and validating the practical value of relaxing the memoryless condition.
- **KL divergences across all 5 dihedral angles of alanine dipeptide are near zero**: substantially better than AS and all other baselines; the Ramachandran plot nearly matches the MD ground truth.
- **Computational efficiency**: the number of energy/model evaluations per gradient update is comparable to AS, with only the additional overhead of the corrector network.

## Highlights & Insights
- **The theoretical insight of recasting SB as SOC is elegant**: transforming a seemingly intractable problem with coupled boundary conditions into an SOC problem solvable via Adjoint Matching. The corrector function precisely compensates for non-memoryless bias, and this idea generalizes naturally to other SB application domains.
- **The equivalence between alternating optimization and IPF provides rigorous convergence guarantees**: unlike the empirical convergence typical of deep learning methods, ASBS admits a formal global convergence proof (under the assumption that each stage reaches a critical point).
- **The use of harmonic priors demonstrates the value of integrating domain knowledge**: standard priors in molecular simulation, previously excluded by the memoryless condition, can now be naturally incorporated into the diffusion sampler framework.

## Limitations & Future Work
- **Each SB stage requires full training of both AM and CM**: while convergence is theoretically guaranteed, the number of stages and training steps per stage require tuning in practice.
- **Limited gains in high dimensions**: on LJ-55 (d=165), ASBS improves over AS by only 1%, suggesting that the advantage of non-memoryless source distributions diminishes as dimensionality increases.
- **The corrector network introduces additional memory and computational overhead**: although theoretically negligible, maintaining an extra network represents an engineering burden in deployment.
- **Validation is limited to Brownian motion base processes with $f_t = 0$**: while the theory applies to general $f_t$, alternative choices such as VP-SDE have not been experimentally evaluated.

## Related Work & Insights
- **vs. Adjoint Sampling (AS)**: AS is a special case of ASBS ($\mu = \delta$, $h = \nabla \log p_1^{\text{base}}$). ASBS generalizes it to arbitrary source distributions via corrector matching while retaining all scalability advantages.
- **vs. PDDS/iDEM**: These methods also use matching objectives but rely on importance-weighted estimation of target samples; ASBS requires no such estimation.
- **vs. Sequential SB (SSB)**: SSB also solves the general SB problem, but is based on SMC and requires a large number of energy evaluations per step, limiting scalability.
- **vs. Data-Driven SB (DSB, I²SB, etc.)**: These methods require explicit samples from the target distribution and are therefore inapplicable to Boltzmann sampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The SOC characterization of SB and the alternating matching algorithm are both original contributions, with theoretical elegance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers synthetic energy functions, molecular simulation, and large-scale conformation generation with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear motivation, and intuitive explanations of key equations.
- Value: ⭐⭐⭐⭐⭐ A significant advance in the design space of diffusion samplers, with direct applicability to molecular simulation and related domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)
- [\[AAAI 2026\] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](../../AAAI2026/others/rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
