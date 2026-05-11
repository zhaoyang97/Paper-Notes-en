---
title: >-
  [Paper Note] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models
description: >-
  [NeurIPS 2025][Medical Imaging][Diffusion Models] This paper identifies an inconsistency between sampling and simulation in diffusion models (particularly at small diffusion timesteps)…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Diffusion Models"
  - "Molecular Dynamics Simulation"
  - "Fokker-Planck Equation"
  - "Energy-Based Models"
  - "Coarse-Graining"
date: 2026-05-08
content_hash: e89f86947480aae8
---

# Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.17139](https://arxiv.org/abs/2506.17139)
**Code**: [https://github.com/noegroup/ScoreMD](https://github.com/noegroup/ScoreMD)
**Area**: Molecular Dynamics / Diffusion Models
**Keywords**: Diffusion Models, Molecular Dynamics Simulation, Fokker-Planck Equation, Energy-Based Models, Coarse-Graining

## TL;DR

This paper identifies an inconsistency between sampling and simulation in diffusion models (particularly at small diffusion timesteps), proposes a Fokker-Planck-based regularization term to enforce consistency, and combines it with a time-partitioned Mixture-of-Experts (MoE) strategy to achieve consistent and efficient sampling and molecular dynamics simulation across multiple biomolecular systems.

## Background & Motivation

**Background**: Molecular dynamics (MD) simulation is a fundamental tool for studying biomolecular behavior, but reaching biologically relevant timescales (μs–ms) is computationally prohibitive. Coarse-graining (CG) methods accelerate simulation by reducing system dimensionality, but require learning CG force fields. Diffusion models have recently demonstrated strong performance in molecular generation and sampling, enabling the generation of new conformations by training on equilibrium molecular distributions.

**Limitations of Prior Work**:
- The score $\nabla_x \log p_t(x)$ learned by diffusion models is theoretically equivalent to the force ($-\nabla U / k_BT$) at $t=0$, making it directly applicable to MD simulation.
- In practice, however, while classical diffusion sampling (i.i.d.) correctly recovers the training distribution, using the same model's score for Langevin simulation produces a distribution inconsistent with i.i.d. sampling.
- This inconsistency is observable even on simple 2D toy systems (e.g., a bimodal Gaussian mixture): i.i.d. sampling recovers both modes, yet the model energy at $t=0$ learns a spurious third mode.
- The prior work Two For One mitigates inconsistency by evaluating the model at $t>0$, but introduces additional noise that degrades structural accuracy.

**Key Challenge**: The score of a diffusion model exhibits numerical instability near the data distribution ($t \to 0$), causing the Fokker-Planck equation to be violated and rendering energy estimates and forces inaccurate.

**Core Problem**:
- How can i.i.d. sampling and Langevin simulation from a diffusion model be made to produce consistent distributions?
- How can simulation accuracy be improved while preserving sampling quality?
- How can an energy-consistent diffusion model be trained efficiently?

**Key Insight**: The Fokker-Planck equation, which governs how the score should evolve with diffusion time, is used as the theoretical starting point. If the model violates this equation, its energy estimates are necessarily inconsistent. Training is regularized by minimizing the Fokker-Planck residual.

**Core Idea**: Constrain an energy-parameterized diffusion model with a regularization term derived from the Fokker-Planck equation, so that the score is more self-consistent at small $t$, thereby unifying sampling and simulation capabilities.

## Method

### Overall Architecture

The method comprises three core components:

1. **Conservative Energy Parameterization**: The score is parameterized as the gradient of an energy function, $\nabla_x \log p_t^\theta = \nabla_x \text{NNET}(x, t)$.
2. **Fokker-Planck Regularization**: A residual loss is added to ensure the model satisfies the Fokker-Planck equation.
3. **Mixture of Experts (MoE)**: The diffusion time axis is partitioned into segments, each handled by a dedicated model; regularization is applied only to the small-$t$ segment.

**Input**: Equilibrium molecular conformations sampled from the Boltzmann distribution.
**Output**: A unified model usable for both i.i.d. sampling and Langevin simulation.

### Key Designs

**Conservative Score Parameterization**: Standard diffusion models directly parameterize the score as $s_\theta = \text{NN}(x, t)$, which does not guarantee a conservative (curl-free) field. This work instead uses an energy parameterization, $\nabla_x \log p_t^\theta = \nabla_x \text{NN}(x, t)$, ensuring the score is the gradient of some energy function. This is both physically motivated (molecular forces should be conservative) and necessary for Fokker-Planck regularization (which requires evaluating $\partial_t \log p_t^\theta$). Experiments confirm that conservative parameterization is critical for simulation stability — non-conservative models diverge after a few thousand simulation steps.

**Fokker-Planck Regularization**: For a diffusion process defined by a VP-SDE, the log-form of the Fokker-Planck equation is:

$$\partial_t \log p_t(x) = \frac{1}{2}g^2(t)[\text{div}_x(\nabla_x \log p_t) + \|\nabla_x \log p_t\|^2] - \langle f, \nabla_x \log p_t \rangle - \text{div}_x(f)$$

The Fokker-Planck residual $R(x,t) = \mathcal{F}_{p^\theta}(x,t) - \partial_t \log p_t^\theta(x)$ is defined, and the regularization loss is:

$$\mathcal{L}_{FP}[\log p^\theta](x,t) = \lambda_{FP}(t) D^{-2} \|R(x,t)\|^2$$

The total training objective combines standard denoising score matching with FP regularization:

$$\min_\theta \mathbb{E}_{t,x(0),x(t)}[\mathcal{L}_{DSM}[\nabla_x \log p^\theta](x(t), t) + \alpha \cdot \mathcal{L}_{FP}[\log p^\theta](x(t), t)]$$

**Weak Residual Formulation (Theorem 1)**: Direct computation of the FP residual requires expensive higher-order derivatives (the divergence term requires a Hessian trace). The paper derives a weak residual estimator requiring only first-order derivatives:

$$\tilde{R}(x,t;v) = \frac{1}{2}g^2(t)\left[\left(\frac{v}{\sigma}\right)^\top \frac{s_\theta(x+v,t) - s_\theta(x-v,t)}{2\sigma} + \|s_\theta(x+v,t)\|^2\right] - \langle f(x+v,t), s_\theta(x+v,t)\rangle - \text{div}_x(f(x+v,t))$$

where $v \sim \mathcal{N}(0, \sigma^2 I)$ and $\sigma = 0.0001$. The time derivative $\partial_t \log p_t^\theta$ is approximated via finite differences.

**Mixture of Experts (MoE) Strategy**: The diffusion time interval $(0,1)$ is partitioned into disjoint subintervals $\mathcal{I}_0, \mathcal{I}_1, \ldots$, each handled by an independent expert:

$$s_\theta(x,t) = \sum_i w_i(t) s_i^\theta(x,t), \quad \sum_i w_i(t) = 1$$

A typical configuration uses $(0, 0.1)$, $[0.1, 0.6)$, and $[0.6, 1.0)$. Only the small-$t$ expert uses conservative parameterization and FP regularization; large-$t$ experts use simpler architectures. This provides two advantages:
1. Avoids over-regularization — large-$t$ models do not require accurate forces.
2. Reduces computational cost — the expensive conservative model is applied only to the critical region.

**Physically Constrained Architecture**: A Graph Transformer architecture is used, with node features comprising atom type and diffusion time, and edge features comprising inter-atomic distance vectors. Translation invariance is achieved by using pairwise distances (rather than absolute coordinates), and rotational equivariance is learned via random rotations applied during training. A scalar energy map $\psi: \mathbb{R}^K \to \mathbb{R}$ ensures conservativeness.

### Loss & Training

Complete training objective (for the **Both** model):
- Small-$t$ expert $[0, 0.1)$: $\mathcal{L}_{DSM} + \alpha \cdot \mathcal{L}_{FP}$ (conservative parameterization, $\alpha = 0.0001$)
- Mid-$t$ expert $[0.1, 0.6)$: $\mathcal{L}_{DSM}$ (non-conservative, simpler architecture)
- Large-$t$ expert $[0.6, 1.0)$: $\mathcal{L}_{DSM}$ (non-conservative, minimal architecture)

VP-SDE parameters: $\beta_{min} = 0.1$, $\beta_{max} = 20$. Optimizer: AdamW.

## Key Experimental Results

### Main Results

**Alanine Dipeptide — 5-atom coarse-grained**:

| Method | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|--------|---------|---------|----------|----------|
| Diffusion | 0.0081±0.0003 | 0.0695±0.0517 | 0.095±0.003 | 1.047±0.924 |
| Two For One | 0.0081±0.0003 | 0.0158±0.0002 | 0.098±0.006 | 0.206±0.004 |
| Mixture | 0.0080±0.0004 | 0.0353±0.0117 | 0.092±0.007 | 0.388±0.109 |
| Fokker-Planck | 0.0084±0.0002 | 0.0088±0.0006 | 0.098±0.006 | 0.105±0.011 |
| **Both** | **0.0079±0.0002** | **0.0086±0.0004** | **0.089±0.005** | **0.099±0.003** |

Key observation: the standard Diffusion model's sim JS (0.0695) is an order of magnitude larger than its iid JS (0.0081), confirming the inconsistency. The Both model reduces sim JS to 0.0086, approaching the iid level.

**Transferable model across dipeptides (400 dipeptides, 10-atom coarse-grained)**:

| Method | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|--------|---------|---------|----------|----------|
| Transferable BG | 0.0183±0.0070 | - | 0.230±0.119 | - |
| Diffusion | 0.0155±0.0083 | 0.2256±0.1304 | 0.206±0.159 | 6.515±3.175 |
| Two For One | 0.0153±0.0080 | 0.0466±0.0114 | 0.203±0.149 | 0.741±0.319 |
| Mixture | 0.0155±0.0078 | 0.0444±0.0237 | 0.200±0.127 | 0.658±0.407 |
| Fokker-Planck | 0.0154±0.0060 | 0.0200±0.0106 | 0.192±0.118 | 0.290±0.222 |
| **Both** | **0.0158±0.0077** | **0.0158±0.0052** | **0.197±0.124** | **0.183±0.070** |

The Both model achieves near-perfect iid–sim consistency, with identical JS divergence values for both modes.

### Ablation Study

**Müller-Brown Potential (2D toy system)**:

| Method | iid JS ↓ | sim JS ↓ | iid PMF ↓ | sim PMF ↓ |
|--------|---------|---------|----------|----------|
| Reference | - | - | 0.0119±0.0004 | 0.087±0.002 |
| Diffusion | 0.0122±0.0013 | 0.0448±0.0125 | 0.111±0.006 | 0.504±0.150 |
| Mixture | 0.0109±0.0007 | 0.0254±0.0109 | 0.097±0.004 | 0.247±0.113 |
| Fokker-Planck | 0.0130±0.0010 | 0.0166±0.0009 | 0.122±0.006 | 0.163±0.008 |
| **Both** | **0.0110±0.0007** | **0.0108±0.0008** | **0.098±0.003** | **0.099±0.004** |

**Conservative vs. Non-Conservative Parameterization** (Alanine Dipeptide):
- Conservative model: marginally better i.i.d. sampling, stable simulation.
- Non-conservative model: comparable i.i.d. sampling, but simulation diverges after a few thousand steps — a non-conservative force field leads to energy non-conservation.

**C–N Bond Length Wasserstein Distance**:

| Method | iid Relative W1 ↓ | sim Relative W1 ↓ |
|--------|-----------------|-----------------|
| Diffusion | 1.51±1.28 | 1.70±0.38 |
| Two For One | 0.96±0.34 | **48.92±11.25** |
| Mixture | 1.36±0.21 | 0.94±0.21 |
| Fokker-Planck | 2.05±0.62 | 2.51±0.59 |
| **Both** | **1.00±0.00** | **1.00±0.00** |

Two For One's sim W1 reaches 48.92 — evaluating the model at $t>0$ introduces excessive noise that severely corrupts bond length structure.

**Runtime Comparison**:

| System | Phase | Diffusion | Mixture | Fokker-Planck | Both |
|--------|-------|-----------|---------|---------------|------|
| Alanine Dipeptide | Training | 49 min | 50 min | 4h 39 min | 3h 59 min |
| Alanine Dipeptide | Inference | 3 min | 4 min | 3 min | 4 min |
| Dipeptides | Training | 4h 5 min | 3h 50 min | 28h 39 min | 27h 5 min |
| Dipeptides | Inference | 8 min | **4 min** | 8 min | **4 min** |

MoE halves inference time (only the small model is needed for simulation), while FP regularization substantially increases training time.

### Key Findings

1. **Confirmation of the inconsistency's origin**: The Fokker-Planck residual is largest as $t \to 0$, consistent with theoretical predictions.
2. **FP regularization and MoE improve consistency through distinct mechanisms**: FP directly reduces the equation residual; MoE focuses the model on the critical time interval — their combination yields the best performance.
3. **Structural accuracy cost of Two For One**: While evaluating at $t>0$ improves mode coverage, the introduced noise severely degrades structural features (bond length W1 is 48× higher).
4. **Transferability validation**: A single model trained on 400 dipeptides achieves consistent sampling and simulation across all dipeptide species.

## Highlights & Insights

- **Complete closed loop from theory to practice**: The problem is identified from the Fokker-Planck equation; a weak residual formulation is derived to reduce computational cost; regularization is selectively applied via MoE; and results are validated across multiple molecular systems.
- **Engineering significance of the weak residual formulation**: Higher-order derivatives (Hessian trace) are reduced to first-order derivatives, making regularization tractable in high-dimensional molecular systems.
- **Dual role of MoE**: Beyond improving efficiency, MoE prevents over-regularization of the large-$t$ region from degrading i.i.d. sampling quality.
- **Unification of sampling and simulation**: A single model can simultaneously provide thermodynamic information (i.i.d. sampling) and dynamical information (Langevin simulation).

## Limitations & Future Work

- Validation is currently limited to small coarse-grained molecules (≤10 atoms); scalability to large protein systems remains to be demonstrated.
- FP regularization increases training time by approximately 6× (the weak residual formulation still requires multiple forward passes).
- Perfect consistency cannot be guaranteed — due to fundamental differences between diffusion sampling and Langevin simulation, perfect alignment may require restricting model expressivity.
- The MoE model introduces discontinuities at interval boundaries, though their impact is negligible in experiments.
- Training requires samples from a known Boltzmann distribution and cannot be directly applied to unknown potential energy functions.

## Related Work & Insights

- **Two For One (Arts et al., 2023)**: The most direct predecessor; applies diffusion models to coarse-grained MD but mitigates inconsistency by evaluating the model at $t>0$.
- **Force matching methods (Husic et al., Charron et al.)**: Directly learn force fields rather than distributions, but require force labels.
- **Transferable Boltzmann Generator (Klein & Noé, 2024)**: A transferable Boltzmann generator supporting i.i.d. sampling but not simulation.
- **FP-Diffusion (Lai et al., 2023)**: Also employs FP regularization, but targets improved i.i.d. sampling quality rather than simulation consistency.
- **AlphaFold 3 / RFDiffusion**: Large-scale protein structure prediction models demonstrating that equivariant architectures are not always necessary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of FP regularization and MoE has both theoretical depth and practical value.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — A complete technical chain from SDE theory to weak formulation derivation to experimental validation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Toy system → alanine dipeptide → transferable dipeptide experiments, with progressively increasing complexity.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear; experimental results are presented systematically.
- **Value**: ⭐⭐⭐⭐ — Open-source code (JAX/PyTorch); directly useful to the computational chemistry community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensembles](jamun_bridging_smoothed_molecular_dynamics_and_score-based_learning_for_conforma.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)

</div>

<!-- RELATED:END -->
