---
title: >-
  [Paper Note] Modeling X-ray Photon Pile-up with a Normalizing Flow
description: >-
  [NeurIPS 2025][Medical Imaging][Normalizing Flow] This paper proposes a Simulation-Based Inference (SBI) framework based on Normalizing Flows. A CNN extracts spatially resolved X-ray spectral features…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Normalizing Flow"
  - "Simulation-Based Inference (SBI)"
  - "X-ray pile-up"
  - "eROSITA"
  - "posterior estimation"
date: 2026-05-08
content_hash: 39f13b0d7de86bcd
---

# Modeling X-ray Photon Pile-up with a Normalizing Flow

**Conference**: NeurIPS 2025
**arXiv**: [2511.11863](https://arxiv.org/abs/2511.11863)  
**Code**: None  
**Area**: Astronomy / Medical Imaging / Simulation-Based Inference
**Keywords**: Normalizing Flow, Simulation-Based Inference (SBI), X-ray pile-up, eROSITA, posterior estimation

## TL;DR

This paper proposes a Simulation-Based Inference (SBI) framework based on Normalizing Flows. A CNN extracts spatially resolved X-ray spectral features, which are then passed to a neural spline flow to perform accurate posterior estimation of astrophysical source parameters in the presence of photon pile-up, substantially outperforming the conventional PSF-core excision approach.

## Background & Motivation

In X-ray astronomy, CCD detectors suffer from **photon pile-up** when observing bright X-ray sources:

**Energy pile-up**: Multiple photons strike the same or adjacent pixels within a single readout cycle, causing the reconstructed photon energy to be artificially elevated (spectral "hardening").

**Pattern pile-up**: In extreme cases, the charge distribution cannot be recognized as a valid event pattern (single-, double-, triple-, or quadruple-pixel), leading to complete signal loss.

**Nonlinear distortion**: Pile-up is highly nonlinear, rendering the likelihood function intractable.

Limitations of conventional approaches:
- **Analytic models** (e.g., Davis 2001): Cannot account for signal loss due to pattern pile-up.
- **PSF-core excision**: Discards the brightest central region, resulting in substantially wider posteriors.
- **Simulator grid-matching** (e.g., SIXTE): Computationally expensive and requires specialized expertise.

As a result, a large body of archival observations affected by pile-up remains underexplored, severely limiting studies such as the cosmic X-ray binary population.

## Method

### Overall Architecture

The framework adopts **Simulation-Based Inference (SBI)**:
1. The SIXTE simulator generates forward-model training data incorporating pile-up effects.
2. A CNN compresses spatially resolved spectra into a low-dimensional summary statistic.
3. A Normalizing Flow uses this summary as a conditioning vector to infer the posterior distribution over physical parameters.

### Key Designs

1. **Spatially resolved input design**:

    - Spectra are extracted from four annular regions (radii 30, 60, 120, and 240 arcseconds).
    - Exploits the **radial dependence** of pile-up: the PSF core is most severely affected, with the effect diminishing outward.
    - This radial variation encodes key information about the incident photon flux.
    - Input dimensionality: $4 \times 1024$ channels.

2. **CNN feature extractor**:

    - Architecture: 2 convolutional layers + 2 pooling layers + 1 fully connected layer.
    - Maps four 1024-channel spectra to a 128-dimensional representation.
    - Uses softplus activation (superior training behavior compared to ReLU).

3. **Neural Spline Flow (NF)**:

    - Employs a neural spline flow.
    - 3 transformation layers, each with a hidden layer of 256 nodes.
    - Deforms an initial normal distribution into the target posterior.
    - Output: posterior probability distributions over 3 physical parameters (flux, temperature, absorption).

### Loss & Training

- 40,000 eROSITA observation simulations generated with the SIXTE simulator.
- Based on an absorbed blackbody model; parameters sampled via Latin hypercube sampling.
- Flux sampled on a logarithmic grid spanning 4 orders of magnitude: $10^{-12}$–$10^{-8}\ \text{erg s}^{-1}\text{cm}^{-2}$.
- Data split: 70% training, 15% validation, 15% test.
- Flux parameter log-transformed and standardized.
- Adam optimizer with learning rate $10^{-4}$.
- Trained on 30 Intel i9 CPUs for several hours.

## Key Experimental Results

### Training Data Parameter Ranges

| Parameter | Range |
|-----------|-------|
| Flux | $10^{-12}$–$10^{-8}$ erg s⁻¹ cm⁻² |
| Temperature | 0.03–0.2 keV |
| Absorption | $(0.2$–$2)\times 10^{22}$ cm⁻² |
| No. of simulations | 40,000 (train 28,000 / val 6,000 / test 6,000) |

### NF vs. Conventional MCMC

| Scenario | Method | Data Used | Posterior Constraint |
|----------|--------|-----------|----------------------|
| With pile-up | Conventional MCMC (PSF-core excision) | Outer annulus only (120″–240″, 351 counts) | Wide posterior, weak constraint |
| With pile-up | NF (proposed) | All 4 annuli | **Narrow posterior, substantially stronger constraint** |
| Without pile-up | MCMC | Full source region (233 counts) | Baseline distribution |
| Without pile-up | NF (proposed) | All 4 annuli | Similar to MCMC (validates absence of overconfidence) |

### Coverage and Accuracy Analysis

| Metric | Flux | Temperature | Absorption |
|--------|------|-------------|------------|
| Coverage calibration | Best (close to ideal diagonal) | ~5% overconfident | ~5% overconfident |
| Mean absolute percentage error | **Well below 10%** | **Well below 10%** | **Well below 10%** |
| Systematic uncertainty baseline | ~10% (SIXTE simulator) | ~10% | ~10% |

### Key Findings

1. **NF posteriors are significantly tighter than PSF-core excision**: the full source region is utilized rather than only the PSF periphery.
2. **Parameter correlations are correctly captured**: the NF successfully learns the known absorption–temperature degeneracy.
3. **Absence of overconfidence validated**: at low flux without pile-up, the NF produces distributions consistent with MCMC.
4. **Training set size has a notable impact**: increasing from 14,000 to 28,000 simulations yields substantial improvement in coverage.
5. **Statistical precision far exceeds systematic uncertainty**: mean absolute percentage errors for all parameters are well below the 10% systematic floor.
6. **Strong practical potential**: approximately 36 neutron-star X-ray binaries in the eROSITA catalog exhibit pile-up in at least one all-sky survey pass.

## Highlights & Insights

- **Elegant physical intuition**: the radial dependence of pile-up is treated as an information source rather than a nuisance.
- **Natural fit for the SBI framework**: pile-up renders the likelihood intractable, precisely the regime where SBI excels.
- **High practical value**: enables the scientific recovery of large quantities of archival data previously discarded due to pile-up.
- **Modest computational cost**: training requires only a few hours; inference (sampling 10,000 posterior samples) is very fast.
- **Rigorous coverage analysis**: the probability integral transform is used to systematically evaluate posterior quality.

## Limitations & Future Work

- **Trained solely on a blackbody model**: application to other spectral models (e.g., power law, multi-temperature plasma) is expected to introduce bias.
- **Spectral model scope**: future work should extend the framework to a broader range of astrophysical source types.
- **Simulation-induced bias**: the pile-up implementation for charge clouds and PSF calibration at large off-axis angles may introduce systematic offsets.
- **No hyperparameter search conducted**: as a proof of concept, further optimization remains possible.
- **Validated only for eROSITA**: generalizability to other X-ray telescopes (e.g., Chandra, XMM-Newton) has not been tested.

## Related Work & Insights

- Forward-modeling approaches based on the SIXTE simulator (Dauser 2019, Tamba 2022, König 2022) have previously attempted large-scale simulation grids.
- Applications of Normalizing Flows in astronomy are growing rapidly.
- The SBI framework provides a unified solution for a wide range of astronomical observations with intractable likelihoods.
- The proposed method can naturally extend to future X-ray observatories such as NewAthena/WFI and AXIS.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First application of NF + SBI to the X-ray pile-up problem)
- **Experimental Thoroughness**: ⭐⭐⭐ (Rigorous coverage analysis, but limited to a blackbody-model proof of concept)
- **Writing Quality**: ⭐⭐⭐⭐ (Physical problem clearly articulated; figures are of high quality)
- **Value**: ⭐⭐⭐⭐ (Significant implications for the reuse of archival X-ray astronomy data)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] FOXES: A Framework For Operational X-ray Emission Synthesis](foxes_a_framework_for_operational_x-ray_emission_synthesis.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)

</div>

<!-- RELATED:END -->
