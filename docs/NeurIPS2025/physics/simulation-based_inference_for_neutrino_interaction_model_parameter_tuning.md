---
title: >-
  [Paper Note] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning
description: >-
  [NeurIPS 2025][Physics][simulation-based inference] This work presents the first application of simulation-based inference (SBI) to neutrino interaction model parameter tuning. Using neural posterior estimation (NPE)…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "simulation-based inference"
  - "neutrino scattering"
  - "neural posterior estimation"
  - "GENIE"
  - "parameter tuning"
date: 2026-05-08
content_hash: cde2c661dbdc01d7
---

# Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2510.07454](https://arxiv.org/abs/2510.07454)  
**Code**: [GitHub](https://github.com/karlaTame/Neutrino_SBI/) (open source)  
**Area**: Physics
**Keywords**: simulation-based inference, neutrino scattering, neural posterior estimation, GENIE, parameter tuning

## TL;DR
This work presents the first application of simulation-based inference (SBI) to neutrino interaction model parameter tuning. Using neural posterior estimation (NPE), the method learns the posterior distribution of 4 physical parameters from 200K GENIE-simulated 58-bin histograms, and accurately recovers the ground-truth parameter values on mock data from the MicroBooNE Tune.

## Background & Motivation

**Background**: Neutrino experiments require precise simulations of neutrino–nucleus interactions; however, the underlying theoretical understanding remains incomplete, and simulators rely on semi-empirical approximations. Experimental collaborations typically tune the physical parameters of simulators such as GENIE to reference data in order to obtain reliable predictions.

**Limitations of Prior Work**: (a) Conventional tuning methods employ simple likelihood fits, but MicroBooNE encountered pathological results in initial attempts and was forced to discard inter-bin correlations in T2K data; (b) direct MCMC is infeasible — a single GENIE simulation can take days to months; (c) next-generation experiments such as DUNE will face larger parameter spaces and more complex datasets.

**Key Challenge**: Accurate probabilistic inference with uncertainty quantification is required, yet physical simulators are expensive and the parameter space is high-dimensional.

**Goal**: To validate whether SBI combined with NPE can replace traditional likelihood fitting and achieve amortized inference at low training cost.

**Key Insight**: The MicroBooNE Tune — a 4-parameter tuning problem with known results — serves as the test scenario, with SBI correctness verified on mock data.

**Core Idea**: An embedding network compresses the 58-dimensional histogram into a 24-dimensional summary feature, which is then fed into a Masked Autoregressive Flow for NPE; a single training run supports unlimited fast inference.

## Method

### Overall Architecture
The GENIE simulator takes 4 physical parameters as input → NUISANCE generates a 58-bin histogram → an embedding network compresses it to 24 dimensions → NPE (MAF architecture) learns the inverse mapping from histogram to parameters → after training, inference is completed in seconds.

### Key Designs

1. **Data Generation**:

    - Function: Construct a large-scale training set covering the parameter space.
    - Mechanism: Four parameters — MaCCQE $\in [0.961, 1.39]$ GeV, NormCCMEC $\in [1.0, 3.0]$, XSecShape_CCMEC $\in [0.0, 1.0]$, RPA_CCQE $\in [0.0, 1.0]$ — are uniformly sampled; GENIE and NUISANCE generate corresponding 58-bin histograms in T2K Analysis I format.
    - Scale: 200K training + 1K test samples, within the parameter range near the MicroBooNE Tune.

2. **Embedding Network**:

    - Function: Dimensionality reduction and informative summary extraction.
    - Mechanism: A 3-layer neural network compresses the 58-bin histogram to a 24-dimensional summary feature. The choice of 24 dimensions (rather than fewer) is motivated by the observation that excessively low dimensionality leads to overconfident posteriors; 24 dimensions are found to be a stable choice that maintains calibration.
    - Design Motivation: Direct NPE on the 58-dimensional raw input is less efficient; summary features capture the most informative statistics.

3. **Neural Posterior Estimation (NPE with MAF)**:

    - Function: Learn the posterior distribution $p(\theta | x)$.
    - Mechanism: Masked Autoregressive Flow architecture with 6 transformation layers and 55 hidden features per layer. The embedding network and MAF are jointly trained end-to-end.
    - Design Motivation: MAF can model complex multi-modal posteriors and inter-parameter correlations; joint training ensures the embedding is optimal for the inference task.

### Loss & Training
- Objective: Negative log-likelihood.
- Training configuration: batch size = 512, lr = 1e-2, 90/10 train/val split.
- Early stopping: patience = 45 epochs; convergence at ~150 epochs on average.
- Training time: ~10 minutes (CPU).
- Inference time: seconds (amortized inference — train once, infer unlimited times).

## Key Experimental Results

### Main Results — Posterior Coverage and Parameter Recovery

| Metric | Result |
|--------|--------|
| Residual center | Centered at 0 for all 4 parameters; no systematic bias |
| Residual width | Narrow distribution, low variance |
| $\theta_1$ (MaCCQE) coverage | Within 10% tolerance band |
| $\theta_2$ (NormCCMEC) coverage | Within 10% tolerance band |
| $\theta_3$ (XSecShape) coverage | Within 20% tolerance band (slightly overconfident) |
| $\theta_4$ (RPA_CCQE) coverage | Within 20% tolerance band |

### MicroBooNE Tune Parameter Recovery

| Parameter | MicroBooNE Ground Truth | SBI Inferred Value ($1\sigma$) | Match |
|-----------|------------------------|-------------------------------|-------|
| MaCCQE | MicroBooNE reported value | Near-perfect match | Excellent |
| NormCCMEC | MicroBooNE reported value | Near-perfect match | Excellent |
| XSecShape_CCMEC | MicroBooNE reported value | Near-perfect match | Excellent |
| RPA_CCQE | MicroBooNE reported value | Near-perfect match | Excellent |

### Key Findings
- **All 4 posteriors are unbiased**: Residuals over 1,000 test events are centered at 0, demonstrating no systematic estimation bias.
- **Weak inter-parameter correlations**: Single-event posteriors show near-independence among the 4 parameters, though mild correlations appear across the full test sample.
- **$\theta_3$ is slightly overconfident**: Coverage tests reveal that predicted confidence intervals are too narrow for this parameter; the remaining parameters are slightly underconfident (more conservative).
- **Key validation passed**: Mock data from the MicroBooNE Tune can be accurately recovered, establishing the foundation for application to real experimental data.

## Highlights & Insights
- **First application of SBI to neutrino interaction model tuning**: Establishes a methodological precedent and paves the way for next-generation experiments such as DUNE.
- **Practical value of amortized inference**: 10-minute training → second-level inference, representing a ~$10^6$-fold efficiency gain over MCMC combined with GENIE (months per run).
- **No need to discard data correlations**: The original MicroBooNE tuning discarded inter-bin correlations; SBI naturally avoids this workaround.

## Limitations & Future Work
- **Only a 4-dimensional parameter space**: Next-generation tuning may involve tens of parameters; scalability has not been validated.
- **Mock data validation only**: The method has not yet been applied to real experimental data (T2K measurements); noise and systematic uncertainties in real data may introduce new challenges.
- **Overconfidence issue for $\theta_3$**: Improved calibration methods are needed, potentially via ensembling or better network architectures.
- **Incomplete uncertainty treatment**: Full correlated uncertainty propagation on inputs and outputs is not addressed.

## Related Work & Insights
- **vs. MicroBooNE original tuning**: MicroBooNE used simple likelihood fitting while discarding bin correlations; SBI provides a full posterior distribution with uncertainty quantification.
- **vs. JUNO SBI (Gavrikov2025)**: JUNO applied SBI to detector response tuning; this work is the first to target physical interaction model parameters.
- **vs. collider physics SBI**: SBI is already mature in the collider domain (Higgs potential, CP violation); neutrino physics represents a new application area.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel as a first application to this domain, though the SBI+NPE methodology itself is well established.
- Experimental Thoroughness: ⭐⭐⭐ Coverage tests and MicroBooNE validation are adequate, but limited to mock data.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, with a well-balanced presentation of neutrino physics background and ML methodology.
- Value: ⭐⭐⭐⭐ Directly useful to the neutrino experimental community; lays groundwork for DUNE and beyond.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)
- [\[NeurIPS 2025\] Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)
- [\[CVPR 2026\] QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](../../CVPR2026/physics/qkd_quantum_gated_incremental_learning.md)

</div>

<!-- RELATED:END -->
