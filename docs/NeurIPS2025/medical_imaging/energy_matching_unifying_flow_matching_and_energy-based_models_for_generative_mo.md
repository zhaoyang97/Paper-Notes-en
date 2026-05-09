---
title: >-
  [Paper Note] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling
description: >-
  [NeurIPS 2025][Medical Imaging][Energy-Based Models] This paper proposes Energy Matching, which unifies flow matching and energy-based models via a single time-independent scalar potential field: far from the data manifold, the model performs efficient transport along optimal transport paths; near the manifold, it transitions to a Boltzmann equilibrium distribution for likelihood modeling. The method achieves FID 3.34 on CIFAR-10, substantially outperforming existing EBMs by more than 50%.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Energy-Based Models
  - Flow Matching
  - Optimal Transport
  - Boltzmann Distribution
  - Inverse Problems
  - Local Intrinsic Dimensionality
date: 2026-05-08
content_hash: a228dfd509fbc450
---

# Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2504.10612](https://arxiv.org/abs/2504.10612)
**Code**: [GitHub](https://github.com/m1balcerak/EnergyMatching)
**Area**: Medical Imaging
**Keywords**: Energy-Based Models, Flow Matching, Optimal Transport, Boltzmann Distribution, Inverse Problems, Local Intrinsic Dimensionality

## TL;DR

This paper proposes Energy Matching, which unifies flow matching and energy-based models via a single time-independent scalar potential field: far from the data manifold, the model performs efficient transport along optimal transport paths; near the manifold, it transitions to a Boltzmann equilibrium distribution for likelihood modeling. The method achieves FID 3.34 on CIFAR-10, substantially outperforming existing EBMs by more than 50%.

## Background & Motivation

**Limitations of flow matching / diffusion models**: Current state-of-the-art generative models (flow matching, diffusion) map noise to the data distribution but do not directly capture the data likelihood — they cannot freely navigate the data manifold and cannot naturally incorporate additional observations and priors (e.g., measurement likelihoods in inverse problems).

**Difficulties of traditional EBMs**: Energy-based models define an unnormalized density $p(x) \propto \exp(-E(x))$ via a scalar function $E(x)$. While theoretically elegant, their practical generative quality is poor — MCMC training struggles to adequately explore the energy landscape in high-dimensional spaces, leading to mode collapse and training instability.

**Complexity of existing improvements**: To improve EBM performance, prior methods rely on time-conditioned ensembles, hierarchical latent ensembles, or cooperative training with independent generators — resulting in large parameter counts and complex training pipelines.

**Core Idea**: Use a single time-independent scalar field to simultaneously achieve two objectives: (a) perform optimal transport far from the manifold for efficient sampling; (b) perform Boltzmann density modeling near the manifold to retain likelihood information.

## Method

### Theoretical Foundation: The JKO Scheme

The method is grounded in the Jordan–Kinderlehrer–Otto (JKO) scheme, which describes the discrete-time evolution of probability distributions along energy-minimizing trajectories in Wasserstein space:

$$\rho_{t+\Delta t} = \arg\min_{\rho} \underbrace{\frac{W_2^2(\rho, \rho_t)}{2\Delta t}}_{\text{transport cost}} + \underbrace{\int V_\theta(x) \mathrm{d}\rho(x)}_{\text{potential energy}} + \underbrace{\varepsilon(t) \int \rho(x) \log \rho(x) \mathrm{d}x}_{\text{internal energy (negative entropy)}}$$

where $V_\theta(x)$ is a learnable scalar potential and $\varepsilon(t)$ is a time-dependent temperature parameter.

### Two-Phase Mechanism

By analyzing the first-order optimality conditions, two distinct behavioral regimes emerge:

**Regime 1** (far from the data manifold, $t < \tau^*$): $\varepsilon(t) = 0$, reducing to optimal transport:
$$\frac{1}{\Delta t}(x - y) + \nabla_x V_\theta(x) = 0$$
The system transports efficiently along deterministic OT paths.

**Regime 2** (near the data manifold, $t \geq 1$): $\varepsilon(t) = \varepsilon_{\max}$, and the equilibrium distribution follows Boltzmann:
$$\rho_{\text{eq}}(x) \propto \exp\left(-\frac{V_\theta(x)}{\varepsilon_{\max}}\right)$$
The system enters EBM mode, accurately modeling the data density.

The temperature schedule follows a linear scheme:
$$\varepsilon(t) = \begin{cases} 0, & 0 \leq t < \tau^* \\ \varepsilon_{\max} \frac{t - \tau^*}{1 - \tau^*}, & \tau^* \leq t < 1 \\ \varepsilon_{\max}, & t \geq 1 \end{cases}$$

### Training Objectives

**Phase 1 (warm-up): Flow objective $\mathcal{L}_{\text{OT}}$**

Mini-batch OT couplings $\gamma^*$ are computed, geodesic interpolations $x_t = (1-t)T(x_{\text{data}}) + t \cdot x_{\text{data}}$ are constructed, and the gradient field is required to approximate the velocity field:

$$\mathcal{L}_{\text{OT}} = \mathbb{E}_{x_{\text{data}} \in \mathcal{D}, t \sim U(0, \tau^*)} \left[\|\nabla_x V_\theta(x_t) + x_{\text{data}} - T(x_{\text{data}})\|^2\right]$$

This is equivalent to flow matching with a curl-free constraint (velocity field as the gradient of a scalar potential), which aligns naturally with OT.

**Phase 2 (main): Contrastive objective $\mathcal{L}_{\text{CD}}$**

Fine-tunes $V_\theta$ near the data manifold so that the Boltzmann distribution matches the data distribution:

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{x \sim p_{\text{data}}}\left[\frac{V_\theta(x)}{\varepsilon_{\max}}\right] - \mathbb{E}_{\tilde{x} \sim \text{sg}(p_{\text{eq}})}\left[\frac{V_\theta(\tilde{x})}{\varepsilon_{\max}}\right]$$

Negative samples are approximated via Langevin chains (half initialized from real data, half from noise). The joint objective is $\mathcal{L} = \mathcal{L}_{\text{OT}} + \lambda_{\text{CD}} \mathcal{L}_{\text{CD}}$.

### Inverse Problem Solving

Given measurements $y = A(x) + w$, the posterior decomposes as:
$$p(x|y) \propto \underbrace{\exp\left(-\frac{\|y - A(x)\|^2}{\zeta^2}\right)}_{p(y|x)} \underbrace{\exp\left(-E_\theta(x)\right)}_{p(x)}$$

The learned $E_\theta(x) = V_\theta(x)/\varepsilon_{\max}$ directly serves as the prior; Langevin sampling is performed by adding the measurement fidelity term — no need to shuttle between noise and data distributions. An interaction energy $W(x_1, x_2)$ can further be introduced to encourage diverse reconstructions.

## Key Experimental Results

### Unconditional Generation (FID↓)

**CIFAR-10**:

| Method | Category | Params | FID↓ |
|--------|----------|--------|------|
| ImprovedCD | EBM | - | 25.1 |
| CLEL-large | EBM | 32M | 8.61 |
| Cooperative DRL-large | Ensemble | 145M | 3.68 |
| DDPM | Diffusion | - | 6.45 |
| NCSN++ | Diffusion | 107M | 2.45 |
| OT-CFM | Flow | 37M | 4.04 |
| **Energy Matching** | **EBM** | **50M** | **3.34** |

**ImageNet 32×32**:

| Method | Category | FID↓ |
|--------|----------|------|
| ImprovedCD | EBM | 32.48 |
| CLEL-large | EBM | 15.47 |
| Cooperative DRL | Ensemble | 9.35 |
| DDPM++ | Diffusion | 8.42 |
| Flow-matching | Flow | 5.02 |
| **Energy Matching** | **EBM** | **6.64** |

- FID 3.34 on CIFAR-10, **outperforming all existing EBMs by more than 50%**
- 50M parameters suffice to surpass Cooperative DRL-large with 145M parameters (3.34 vs. 3.68)
- FID 6.64 on ImageNet, the first time an EBM approaches the level of flow-based models

### Local Intrinsic Dimensionality (LID) Estimation

Spearman correlation vs. PNG compression ratio (4096 images):

| Method | MNIST | CIFAR-10 |
|--------|-------|----------|
| ESS | 0.444 | 0.326 |
| FLIPD | 0.837 | 0.819 |
| NB (Diffusion) | 0.864 | 0.894 |
| **Energy Matching** | **0.877** | **0.901** |

LID is computed directly on the data manifold via the Hessian spectrum $\nabla_x^2 V(x_{\text{data}})$, where the number of near-zero eigenvalues reflects the local dimensionality — more accurate than diffusion-based approaches (no approximation required).

### Protein Inverse Design

On the fitness–diversity tradeoff for AAV capsid protein segments: by introducing a tunable repulsion term $\propto 1/\sigma^2$ to explicitly control generative diversity, the method achieves a better Pareto frontier than flow models and score-based models on both Medium and Hard benchmarks.

## Highlights & Insights

- ⭐⭐⭐⭐⭐ **Theoretical Unification**: The first rigorous unification of OT flow matching and EBMs, with the two-phase mechanism derived naturally from the JKO scheme.
- ⭐⭐⭐⭐⭐ **Minimal Design**: A single time-independent scalar field — no auxiliary generators, no time conditioning, no additional networks.
- ⭐⭐⭐⭐ **Performance Breakthrough**: FID 3.34 on CIFAR-10 for the EBM category, the first time approaching the level of diffusion/flow models.
- ⭐⭐⭐⭐ **Versatility**: The same model supports generation, inverse problem solving, and LID estimation — all stemming from explicit likelihood modeling.
- ⭐⭐⭐ **Protein Design**: The interaction energy term straightforwardly extends the method to controlled protein generation, demonstrating its flexibility.

## Limitations & Future Work

1. **Limited resolution**: Experiments are conducted only on CIFAR-10 (32×32) and ImageNet 32×32; high-resolution settings have not been validated.
2. **High sampling step count**: Sampling requires 325 Euler–Heun steps, slower than some accelerated flow matching methods.
3. **MCMC overhead**: Langevin chain negative sample generation in Phase 2 increases training cost.
4. **Absolute FID gap remains**: Despite substantially surpassing EBMs, the absolute FID (3.34) still falls short of the best diffusion models (NCSN++ at 2.45).
5. **Hyperparameter sensitivity**: Hyperparameters such as $\tau^*$, $\varepsilon_{\max}$, and $\lambda_{\text{CD}}$ require per-dataset tuning.

## Overall Rating ⭐⭐⭐⭐⭐

This is an elegant theoretical contribution that naturally unifies two major generative modeling paradigms from the JKO scheme. The idea of using a single scalar field to simultaneously achieve efficient transport and density modeling is both concise and profound. While absolute generation quality still has room for improvement, the work opens an entirely new direction for EBMs — combining explicit likelihood, flexible inverse problem solving, and LID estimation in a single framework. It holds significant implications for advancing EBM applications across broader domains.

## Related Work & Insights

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/medical_imaging/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)

<!-- RELATED:END -->
