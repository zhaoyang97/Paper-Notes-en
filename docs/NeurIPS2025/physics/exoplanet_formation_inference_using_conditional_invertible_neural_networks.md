---
title: >-
  [Paper Note] Exoplanet Formation Inference Using Conditional Invertible Neural Networks
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][exoplanets] A conditional invertible neural network (cINN) trained on 15,777 synthetic planets infers planet formation parameters (disk mass, turbulent $\alpha$…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "exoplanets"
  - "conditional invertible neural networks"
  - "Bayesian inference"
  - "planet formation"
  - "surrogate model"
date: 2026-05-08
content_hash: fddddaf30f270639
---

# Exoplanet Formation Inference Using Conditional Invertible Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2512.05751](https://arxiv.org/abs/2512.05751)  
**Code**: None  
**Area**: Physics / Planetary Science
**Keywords**: exoplanets, conditional invertible neural networks, Bayesian inference, planet formation, surrogate model

## TL;DR
A conditional invertible neural network (cINN) trained on 15,777 synthetic planets infers planet formation parameters (disk mass, turbulent $\alpha$, dust-to-gas ratio) from observables (planet mass, orbital distance), achieving probabilistic parameter retrieval ~10⁶× faster than physical simulations. Multi-planet system data is shown to yield more robust inference than single-planet data.

## Background & Motivation

**Background**: Understanding the origins of exoplanets requires tracing formation parameters from observed planetary properties. Direct MCMC approaches are infeasible — a single run of a global planet formation model takes days to months.

**Limitations of Prior Work**: (a) Physical models are computationally prohibitive for large-scale Bayesian inference; (b) gravitational chaos among planets renders the parameter-to-observable mapping stochastic; (c) data are sparse in the high-dimensional parameter space (disk mass, viscosity, dust-to-gas ratio, inner edge, etc.).

**Key Challenge**: Accurate probabilistic inference is required, yet physical models cannot be run at scale.

**Goal**: Train a fast surrogate model on a limited synthetic dataset to enable practical Bayesian inference of planet formation parameters.

**Key Insight**: cINNs provide exact invertible mappings — the forward pass maps parameters to a standard Gaussian latent space conditioned on observables, while the inverse pass samples the posterior, naturally supporting probabilistic inference.

**Core Idea**: Use a cINN as a surrogate for the planet formation physical model. Individual planets extracted from multi-planet systems are treated as separate training samples to increase data diversity, enabling millisecond-scale probabilistic parameter inference.

## Method

### Overall Architecture
A physical model (dust-to-planet global formation model) generates synthetic planetary data → a cINN is trained to learn an invertible mapping from parameters to latent space conditioned on observables → at inference time, samples drawn from a standard Gaussian are passed through the inverse flow to obtain the posterior distribution.

### Key Designs

1. **Global Planet Formation Model for Data Generation**:

    - **Function**: Generate synthetic planet samples covering the parameter space.
    - **Mechanism**: Tracks the full pipeline of dust grain coagulation, planetesimal formation, protoplanet accretion, and photoevaporation of the gas disk. Comprises 707 single-planet disks and ~15,777 multi-planet systems (up to 100 planets per disk). Four parameters varied in log-space: disk mass fraction ($10^{-3}$ to $10^{-0.5}$), viscosity $\alpha$ ($10^{-3.5}$ to $10^{-2}$), dust-to-gas ratio ($10^{-2.4}$ to $10^{-1}$), and inner-edge orbital period (1–20 days).
    - **Design Motivation**: Extracting one sample per planet from multi-planet systems provides ~22× more diverse parameter–observable combinations.

2. **cINN Architecture**:

    - **Function**: Learn an invertible mapping from parameters to latent space.
    - **Mechanism**: 16 affine coupling blocks with random permutations between blocks; each subnetwork consists of 3 layers × 8 units with ReLU activations. Maps 4D parameters $\vec{x}$ to a 4D latent space $\vec{z}$ (unit Gaussian) conditioned on 2D observables $\vec{c}$ (planet mass, semi-major axis). Loss: $L = \frac{1}{2}\|f(x;c)\|^2 - \log|\det \frac{\partial f}{\partial x}| + \|\hat{x}-x\|^2$
    - **Design Motivation**: The invertible mapping guarantees exact posterior sampling without additional MCMC steps.

3. **Multi-Planet vs. Single-Planet Training Strategy**:

    - **Function**: Compare the effect of different data organization strategies on inference robustness.
    - **Mechanism**: Multi-planet training extracts individual (parameters, observables) pairs from each planet, increasing training sample diversity. Single-planet training uses only 707 simulations.
    - **Design Motivation**: Single-planet training produces unphysical extrapolations in unsampled regions (overestimated $\alpha$ at large orbital distances); multi-planet training is more robust.

### Loss & Training
Combined loss: maximum likelihood (negative log-likelihood under a Gaussian latent space) + reconstruction loss. Adam optimizer ($\beta_1=\beta_2=0.8$, lr=0.001, decay $\gamma=0.99$/epoch); data augmentation with Gaussian noise ($\sigma=0.01$).

## Key Experimental Results

### Main Results

| Training Data | MAP Deviation (σ) | Parameter Space Coverage | Extrapolation Quality |
|---|---|---|---|
| Multi-planet (~15.7k) | 0.2 (good) | Excellent | Physically consistent |
| Single-planet (707) | 0.2 (sampled region) | Poor | Unphysical extrapolation |
| Two-planet systems | Stable | Improved | Good generalization |

### Parameter Inference Analysis

| Parameter | Inference Quality | Correlation Pattern |
|---|---|---|
| Disk mass $M_{disk}$ | Good, narrow posterior | Positively correlated with $\alpha$ |
| Viscosity $\alpha$ | Diagonal pattern in mass–distance space | Affects migration and dust properties |
| Dust-to-gas ratio | Good | Relatively independent |
| Inner-edge period | Recoverable | Weakly constrained |

### Key Findings
- **Multi-planet data is essential**: Single-planet training yields spuriously narrow posteriors (overconfidence) in unsampled regions; multi-planet training eliminates this artifact.
- **Chaos does not impede inference**: Gravitational chaos among planets does not degrade parameter recovery — chaotic effects are orthogonal to the formation parameter imprint.
- **~10⁶× inference speedup**: Millisecond-scale inference vs. month-scale physical model computation.
- **Physical causality reflected in $\alpha$ inference**: The diagonal pattern of $\alpha$ in distance–mass space reflects the dual influence of viscosity on migration and dust evolution.

## Highlights & Insights
- **Data diversity > data volume**: Extracting individual planets from multi-planet systems provides more uniform parameter space coverage and is more effective than increasing the number of single-planet simulations.
- **cINNs are naturally suited for parameter inference**: Invertibility guarantees exact posterior sampling without MCMC or variational approximations.
- **Chaos robustness is surprising**: Gravitational N-body chaos might be expected to destroy a deterministic parameter-to-observable mapping, yet the posterior remains recoverable.

## Limitations & Future Work
- Data volume remains insufficient for higher-dimensional settings (6D observables, three-planet systems).
- Strong dependence on the accuracy of the physical model — violations of model assumptions necessitate retraining.
- Single-planet training is unsuitable for real survey data, limiting applicability in the simplest observational scenario.

## Related Work & Insights
- **vs. MCMC / nested sampling**: cINNs provide instantaneous posterior sampling, whereas MCMC requires thousands of model evaluations.
- **vs. Simulation-Based Inference (SBI)**: cINNs represent one implementation of SBI, with the advantage of exact invertibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of cINNs to planet formation parameter inference; the multi-planet data strategy is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic data evaluation is comprehensive, but validation on real observational data is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Physical motivation is clear; connections between methodology and physics are well articulated.
- **Value**: ⭐⭐⭐⭐ Provides a practical inference tool for exoplanet population demographics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[NeurIPS 2025\] Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)
- [\[ICCV 2025\] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers](../../ICCV2025/physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)

</div>

<!-- RELATED:END -->
