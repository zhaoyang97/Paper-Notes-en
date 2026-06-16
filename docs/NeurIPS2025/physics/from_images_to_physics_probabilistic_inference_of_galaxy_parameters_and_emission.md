---
title: >-
  [Paper Note] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows
description: >-
  [NEURIPS2025][Physics & Scientific Computing][VAE] This work proposes a VAE–Normalizing Flow hybrid framework that jointly infers galaxy physical parameters (stellar mass, SFR, redshift, gas-phase metallicity…
tags:
  - "NEURIPS2025"
  - "Physics & Scientific Computing"
  - "VAE"
  - "Normalizing Flows"
  - "galaxy parameter inference"
  - "emission line prediction"
  - "probabilistic inference"
date: 2026-05-08
content_hash: 098ff98a5cf6228f
---

# From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE & Normalizing Flows

**Conference**: NEURIPS2025
**arXiv**: [2511.12737](https://arxiv.org/abs/2511.12737)  
**Code**: To be confirmed  
**Area**: Scientific Computing
**Keywords**: VAE, Normalizing Flows, galaxy parameter inference, emission line prediction, probabilistic inference

## TL;DR

This work proposes a VAE–Normalizing Flow hybrid framework that jointly infers galaxy physical parameters (stellar mass, SFR, redshift, gas-phase metallicity, central black hole mass) and emission line fluxes (Hα, Hβ, [N II], [O III]) in a probabilistic manner from SDSS gri images and photometric data, achieving over 100× speedup relative to SED fitting while providing well-calibrated posterior distributions.

## Background & Motivation

**Core Task**: Inferring stellar mass, star formation rate (SFR), redshift, gas-phase metallicity, and central black hole mass is fundamental to understanding galaxy formation and evolution, yet traditional methods (e.g., SED fitting tools such as Prospector, Bagpipes, and CIGALE) are computationally prohibitive.

**Importance of Emission Lines**: Emission lines such as Hα, Hβ, [N II], and [O III] are key diagnostics for constraining SFR, metallicity, dust content, and AGN/shock activity (BPT diagrams), but measuring them requires expensive spectroscopic observations.

**Challenges of Large-Scale Surveys**: Upcoming facilities such as the Roman Space Telescope and Rubin LSST will observe billions of galaxies, making spectroscopic follow-up infeasible for all sources and creating an urgent need for efficient methods that infer physical parameters from imaging and photometry alone.

**Limitations of Prior Work**: Contrastive learning approaches such as AstroCLIP and conditional VAE methods largely produce point estimates, lack calibrated uncertainties, and rarely infer both physical parameters and emission line fluxes simultaneously.

**Need for Probabilistic Modeling**: Strong degeneracies exist among galaxy parameters (e.g., stellar mass–SFR–redshift), which cannot be captured by point regression alone; joint posterior distributions are required.

**Gap in Black Hole Mass Inference**: No prior method has probabilistically estimated central black hole mass from imaging and photometry alone.

## Method

### Overall Architecture

A two-stage VAE–Normalizing Flow architecture is adopted. In the first stage, a VAE learns a 32-dimensional latent representation of galaxy images. In the second stage, the VAE latent features are concatenated with photometric colors and magnitudes and fed into a conditional Normalizing Flow that models the joint posterior distribution of physical parameters and emission line fluxes.

### Module 1: Variational Autoencoder (VAE) Image Encoding

- **Function**: Encodes $160 \times 160$ three-channel (g/r/i band) galaxy images into 32-dimensional latent vectors.
- **Mechanism**: The encoder applies three convolutional layers (kernel 4, stride 2, padding 1) followed by fully connected layers to output latent mean $\mu \in \mathbb{R}^{32}$ and log-variance $\log \sigma^2 \in \mathbb{R}^{32}$; samples are drawn via the reparameterization trick as $z \sim \mathcal{N}(\mu, \sigma^2)$. The decoder reconstructs images through transposed convolutions.
- **Design Motivation**: The VAE learns a continuous, structured latent space in which $\mu$ and $\sigma$ encode galaxy morphology and uncertainty respectively, providing information-rich features for subsequent probabilistic inference. The training loss combines MSE reconstruction loss and KL divergence, optimized with Adam (lr=1e-4). The encoder is frozen after training.

### Module 2: Two-Stage Conditional Normalizing Flow (Physical Parameter Inference)

- **Function**: Infers the joint posterior distribution of $M_\star$, SFR, $z$, $M_\mathrm{BH}$, and metallicity from VAE latent features and photometric information.
- **Mechanism**:
    - The input concatenates the 32-dimensional VAE mean, 32-dimensional standard deviation, and photometric colors and apparent magnitudes, encoded by an MLP into a 256-dimensional representation.
    - **First branch**: An MLP predicts mean estimates of four core parameters ($M_\star$, SFR, $z$, $M_\mathrm{BH}$); residuals are then modeled by a 12-layer affine coupling conditional RealNVP flow over their joint distribution.
    - **Second branch**: A separate 1D conditional affine flow models the metallicity residual, conditioned on the true core parameters during training and on samples from the first branch during inference, realizing the chain decomposition $p(y_\mathrm{core}, \mathrm{O/H} \mid x) = p(y_\mathrm{core} \mid x) \cdot p(\mathrm{O/H} \mid y_\mathrm{core}, x)$.
    - An additional sigmoid MLP is trained to predict whether a galaxy has measurable metallicity (binary classification, ~84% accuracy).
- **Design Motivation**: The two-stage flow design explicitly encodes the conditional dependence of metallicity on other physical parameters, yielding higher accuracy than independent modeling. The invertible transformations of RealNVP guarantee exact probability density computation.

### Module 3: Emission Line Flux Inference Normalizing Flow

- **Function**: Infers the joint posterior of Hα, Hβ, [N II] λ6584, and [O III] λ5007 emission line fluxes from the same encoded representation.
- **Mechanism**: An MLP first predicts mean fluxes in log1p space; a 4D conditional RealNVP flow (12 affine coupling layers) then models the residual distribution.
- **Design Motivation**: Emission line fluxes are physically correlated (e.g., the intrinsic Balmer decrement), and joint modeling captures these structures more effectively than independent regression. The log1p transformation addresses the heavy-tailed flux distribution.

### Loss & Training

- VAE stage: MSE reconstruction loss + KL divergence regularization
- NF stage: Negative log-likelihood loss (maximizing the log-probability of data under the flow model), plus MSE loss for the physical parameter MLP and binary cross-entropy for metallicity detectability

## Key Experimental Results

### Dataset

Approximately 250,000 SDSS Main Galaxy Sample galaxies ($z \leq 0.3$), of which ~100,000 are used for VAE training and ~125,000 for the NF with a 70/15/15 split.

### Table 1: $R^2$ Comparison for Physical Parameter Inference

| Method | Redshift $z$ | Stellar Mass | SFR | Black Hole Mass | Metallicity |
|--------|-------------|--------------|-----|----------------|-------------|
| (r,g,z) Photometry + MLP [AstroCLIP] | 0.68 | 0.67 | 0.34 | N/A | 0.41 |
| Image Embedding + MLP [AstroCLIP] | 0.78 | 0.73 | 0.42 | N/A | 0.43 |
| Image Embedding + kNN [AstroCLIP] | 0.79 | 0.74 | 0.44 | N/A | 0.44 |
| Image Embedding [Gagliano] | 0.83 | 0.75 | N/A | N/A | N/A |
| **Image+Phot+NF (Ours)** | **0.80** | **0.85** | **0.76** | **0.67** | **0.76** |
| Photometry+NF (Ours) | 0.72 | 0.80 | 0.75 | 0.62 | 0.65 |

### Table 2: Uncertainty Decomposition (Validation Set)

| Parameter | $M_\mathrm{BH}$ | $\log M_\star$ | $12+\log(\mathrm{O/H})$ | $\log \mathrm{SFR}$ | $z$ | Hα | Hβ | [N II] | [O III] |
|-----------|----------------|----------------|------------------------|---------------------|-----|----|----|--------|---------|
| $\sigma_\mathrm{aleatoric}$ | 0.589 | 0.191 | 0.134 | 0.327 | 0.018 | 0.427 | 0.381 | 0.427 | 0.611 |
| $\sigma_\mathrm{epistemic}$ | 0.034 | 0.012 | 0.010 | 0.019 | 0.001 | 0.027 | 0.026 | 0.027 | 0.045 |

### Key Findings

1. **SFR inference substantially surpasses prior methods**: $R^2 = 0.76$ vs. the previous best of 0.44, a 73% improvement, attributable to the NF's ability to model the stellar mass–redshift–SFR degeneracy.
2. **Emission line predictions**: Balmer lines (Hα, Hβ) achieve $R^2 = 0.79$–$0.80$; [N II] achieves $R^2 = 0.70$; [O III] achieves $R^2 = 0.50$ due to its greater sensitivity to local ionization conditions.
3. **Uncertainty is dominated by aleatoric contributions**: Epistemic uncertainty is far smaller than aleatoric uncertainty across all parameters, indicating sufficient model capacity.
4. **Photometry-only NF** still outperforms prior image-embedding methods on SFR and metallicity.
5. **First probabilistic black hole mass inference from imaging and photometry**: $R^2 = 0.67$; the model yields more conservative predictions for extreme masses ($10^{11}$–$10^{12}\ M_\odot$), which may be more physically reasonable than catalog values.

## Highlights & Insights

1. **Joint probabilistic inference**: A single framework simultaneously infers joint posteriors over 5 physical parameters and 4 emission line fluxes, preserving inter-parameter correlations such as the $M_\star$–SFR main sequence.
2. **Elegant chain decomposition**: The two-stage flow design conditions metallicity on core parameters, explicitly encoding physical dependencies.
3. **Interpretable latent space**: Perturbing specific latent dimensions and decoding visualizes physically consistent morphological changes, such as galaxies appearing smaller at higher redshift and bulges becoming redder at lower SFR.
4. **Speed advantage**: Over 100× faster than SED fitting; VAE training requires 1.5 hours on an A100, while NF training requires only 30 minutes on a T4.

## Limitations & Future Work

1. Only SDSS DR1 data are used (relatively shallow and noisy), without leveraging the higher-quality spectra and broader coverage of DR17.
2. The redshift range is restricted to $z \leq 0.3$, precluding application to high-redshift galaxies.
3. The VAE may smooth small-scale structure under noisy inputs, degrading the retention of morphological information.
4. The "ground truth" black hole masses are derived from the $M_\mathrm{BH}$–$\sigma$ empirical relation rather than direct measurements, introducing additional uncertainty.
5. The relatively low $R^2 = 0.50$ for [O III] reflects the difficulty of capturing local ionization conditions from global properties.

## Related Work & Insights

- **AstroCLIP** (Parker et al.): Aligns images and photometry via multi-modal contrastive learning but outputs only point estimates. The proposed framework surpasses its $R^2$ on all parameters through the NF.
- **Gagliano et al.**: Employs a conditional VAE to infer stellar mass and redshift, but does not address SFR, metallicity, or emission lines.
- **SED fitting (Prospector/Bagpipes/CIGALE)**: Physically well-grounded but computationally expensive; the proposed framework offers a substantial speed advantage.
- **Future directions**: The authors plan to replace the VAE with a diffusion model to better preserve fine-grained structural detail and to extend the framework to broader redshift ranges.

## Rating

- Novelty: ⭐⭐⭐⭐ (First joint probabilistic inference of physical parameters and emission lines; first imaging-based black hole mass estimation)
- Experimental Thoroughness: ⭐⭐⭐ (Single dataset SDSS; limited comparison with additional methods and cross-survey validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with thorough physical discussion)
- Value: ⭐⭐⭐⭐ (Provides a practical probabilistic inference tool for large-scale surveys)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)
- [\[NeurIPS 2025\] Neural Network for Simulating Radio Emission from Extensive Air Showers](neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)
- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)

</div>

<!-- RELATED:END -->
