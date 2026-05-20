---
title: >-
  [Paper Note] Neural Deprojection of Galaxy Stellar Mass Profiles
description: >-
  [NeurIPS 2025][Physics][galaxy mass distribution] A neural network approach is proposed to map Nuker galaxy profile parameters to analytically deprojectable Multi-Gaussian Expansion (MGE) components…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "galaxy mass distribution"
  - "deprojection"
  - "deep learning"
  - "conditional flow matching"
  - "astronomical data"
date: 2026-05-08
content_hash: 5edf19aac0a662d1
---

# Neural Deprojection of Galaxy Stellar Mass Profiles

**Conference**: NeurIPS 2025
**arXiv**: [2511.20746](https://arxiv.org/abs/2511.20746)  
**Code**: None  
**Area**: Physics / Astronomy
**Keywords**: galaxy mass distribution, deprojection, deep learning, conditional flow matching, astronomical data

## TL;DR

A neural network approach is proposed to map Nuker galaxy profile parameters to analytically deprojectable Multi-Gaussian Expansion (MGE) components, enabling stellar mass modeling of galaxies without optical imaging. The method is integrated into the differentiable dynamical modeling pipeline SuperMAGE for Bayesian inference of supermassive black hole (SMBH) masses.

## Background & Motivation

- **Core Scientific Problem**: A key open question in galaxy evolution concerns the origin of tight correlations between galaxy properties and central SMBH masses, which requires high-precision SMBH mass measurements across large galaxy samples.
- **Limitations of Conventional Methods**:
    - Traditional cold molecular gas dynamical modeling relies on optical imaging to obtain the projected stellar surface density, which is then deprojected to 3D via MGE models.
    - **Dust Obscuration**: Optical observations are attenuated by dust, limiting analysis to relatively dust-free galaxies.
    - **Active Galactic Nucleus (AGN) Contamination**: MGE models fit AGN light, leading to underestimation of SMBH masses.
    - **Quasars**: Extremely luminous AGN completely outshine host galaxy starlight, rendering optical fitting infeasible.
- **Goal**: To develop a method independent of optical imaging that constrains the stellar mass distribution using radio data alone, thereby extending the analyzable galaxy sample to dusty, AGN-dominated, and high-redshift gravitationally lensed systems.

## Method

### Overall Architecture

The overall approach consists of three steps:

1. Parameterize the projected stellar surface density using the **Nuker model**—which has fewer parameters and stronger physical constraints than MGE, but lacks an axisymmetric analytic deprojection.
2. Train a **neural network** to map Nuker parameters $(\alpha, \beta, \gamma, r_b)$ to the normalized coefficients $\Sigma_i$ of 64 MGE Gaussian components, thereby combining the physical interpretability of Nuker with the analytic deprojection capability of MGE.
3. Embed this NN into the differentiable dynamical modeling pipeline **SuperMAGE** for end-to-end Bayesian inference on radio interferometric (ALMA) visibility data.

### Key Designs

1. **Neural Mapping from Nuker to MGE**: The Nuker model has 5 physical parameters $(\alpha, \beta, \gamma, r_b, \Sigma_b)$ and enforces physically reasonable stellar mass profiles, but lacks axisymmetric analytic deprojection; MGE supports analytic deprojection but has too many parameters and is prone to degeneracy. The NN bridges the two by mapping Nuker parameters to 64 MGE normalized coefficients on a fixed $\sigma_i$ grid.
2. **Symexp Activation for Large Dynamic Range**: The MGE coefficients $\Sigma_i$ span approximately 15 orders of magnitude ($\sim\pm 10^{-12}$ to $\sim\pm 10^{3}$), making direct prediction numerically unstable. The authors design a symexp activation $\Sigma_i = \text{sgn}(\Sigma_i') \cdot 10^{-12}(10^{|\Sigma_i'|} - 1)$, compressing the output to a learnable range of $\sim\pm 0.3$ to $\sim\pm 15$.
3. **Direct Visibility-Space Modeling**: Radio interferometric data are fundamentally Fourier components of the sky (visibilities). The conventional approach of first converting to images via the CLEAN algorithm and then fitting makes uncertainty estimation difficult. This work models directly in visibility space, where noise is Gaussian-distributed, and uses Kaiser-Bessel window functions to bin visibilities onto a regular grid.

### Loss & Training

- **NN Training Loss**: **MSE loss** between the predicted MGE profile $\Sigma_{\text{MGE}}(r)$ and the ground-truth Nuker profile $\Sigma(r)$.
- **Optimizer**: Adam, learning rate $10^{-5}$.
- **Training Data**: A uniform grid of $6.25 \times 10^6$ Nuker parameter combinations ($50^4$), with parameter ranges:
    - $\alpha \in [0.1, 10]$, $\gamma \in [0.001, 1.2]$, $\beta - \gamma \in [0.3, 3.0]$, $r_b \in [2 \text{ pc}, 2 \text{ kpc}]$.
- **Training Time**: Approximately 10 hours (NVIDIA RTX 4060 Ti).
- **Posterior Sampling** at inference uses **Metropolis-Adjusted Langevin Dynamics (MALD)**, with the mass matrix tuned via the empirical Fisher information matrix and step size adjusted to an acceptance rate of $\approx 0.574$.
- **Likelihood Approximation**: The true likelihood follows a Student's T distribution, approximated by a Gaussian with standard deviation scaled by a factor of 2 (conservative estimate covering the 99% confidence interval).

## Key Experimental Results

### Main Results

Validation on galaxy NGC4697, compared against the state-of-the-art method using optical HST imaging and KinMS:

| Method | Stellar Mass Model | Data Source | BH Mass $\log_{10}(M_\bullet/M_\odot)$ |
|--------|--------------------|-------------|----------------------------------------|
| KinMS + MGE (w/o AGN) | Optical MGE, innermost Gaussian removed | HST + ALMA | Overestimated (AGN over-correction) |
| KinMS + MGE (w/ AGN) | Optical MGE, innermost Gaussian retained | HST + ALMA | Underestimated (AGN contamination) |
| **SuperMAGE + Nuker (Ours)** | Neurally deprojected Nuker | **ALMA only** | Intermediate between the two |

- All three methods converge consistently in stellar mass profiles at radii accessible to ALMA resolution.
- Total mass profiles (as characterized by rotation velocity curves) agree within $3\sigma$.
- The velocity curve scatter of the proposed model is significantly smaller, attributed to the additional fitting flexibility from jointly sampling the stellar mass profile shape.

### Ablation Study

| Component / Design | Effect |
|--------------------|--------|
| Symexp activation vs. direct prediction | Direct prediction of $\Sigma_i$ (dynamic range $\sim 10^{15}$) leads to training instability; symexp compression enables stable training. |
| 64 fixed $\sigma_i$ grid | $\sigma_i$ log-uniformly distributed from 1 pc to 10 kpc, covering all physical scales of the galaxy. |
| NN accuracy | After training, fractional error in mass profiles is $<3\%$, well below the statistical uncertainty of the Nuker profile itself. |
| AGN correction (KinMS) | Removing the innermost Gaussian component causes stellar mass underestimation and SMBH mass overestimation—results from this work suggest that correction is excessive. |

### Key Findings

1. **SMBH mass measurements consistent with SOTA are achievable without optical data**, relying solely on radio (ALMA) observations.
2. The Nuker model break radius is constrained by the posterior to lie beyond the maximum extent of the gas (~3 arcseconds), indicating that NGC4697 is well described by a single power law.
3. The uncertainty intervals on the velocity curves are narrower than those from KinMS+MGE, reflecting improved data fit.
4. The results suggest that prior AGN corrections (removing the innermost Gaussian) constitute an **over-correction**, inadvertently removing stellar light along with AGN light.

## Highlights & Insights

- **Paradigm Shift**: Replacing the conventional optical imaging dependency with a learnable parameter mapping extends dynamical modeling to dust-obscured, AGN-dominated, and even high-redshift gravitationally lensed galaxies.
- **Symexp Activation**: An elegant solution to the dynamic range problem spanning 15 orders of magnitude, and a generalizable technique for handling the large dynamic ranges common in astronomical data.
- **Decoupling Physical Constraints from Flexible Deprojection**: Nuker provides physical constraints (few parameters, physically reasonable profile shapes); MGE provides mathematical convenience (analytic deprojection); the NN bridges the two. This "physical model + differentiable surrogate" paradigm has broad applicability.
- **Direct Visibility-Space Modeling**: Avoids error propagation introduced by the CLEAN algorithm and fully exploits the Gaussian noise properties for probabilistic inference.

## Limitations & Future Work

- Validation on only a single galaxy (NGC4697) limits statistical significance; systematic testing across larger galaxy samples is needed.
- The Nuker model assumes axisymmetry; applicability to non-axisymmetric systems (e.g., barred galaxies) remains untested.
- While NN accuracy at $<3\%$ is currently sufficient, future high-precision data may require further optimization.
- The break radius in NGC4697 is not effectively constrained (degenerating to a single power law); the full parameter space of the model needs validation on other galaxy types.
- Approximating the Student's T likelihood with a Gaussian distribution is conservative but may affect posterior sharpness.
- The SuperMAGE pipeline itself has not been formally published (in prep.), and the complete codebase is unavailable.

## Related Work & Insights

- **MGE Deprojection** [Cappellari, 2002]: The classical method for mapping optical surface brightness to 3D mass density; flexible but parameter-rich and susceptible to AGN contamination.
- **Nuker Model** [Lauer et al., 1995]: A double power-law model for galaxy central surface brightness with fewer physical parameters but lacking axisymmetric deprojection.
- **KinMS** [Davis et al., 2013]: A cold gas dynamical modeling tool and the direct reference method for this work.
- **JamPy** [Cappellari, 2008/2020]: A Jeans anisotropic modeling tool used for cross-validation with SuperMAGE.
- **AI for Science Inspiration**: This work demonstrates the NN as a differentiable mapping between physical models—not replacing physics, but bridging the mathematical gap between different parameterizations.

## Rating

| Dimension | Score (1–10) | Remarks |
|-----------|--------------|---------|
| Novelty | 8 | First use of NN to replace optical deprojection pipeline; paradigm-shifting. |
| Technical Depth | 7 | Solid physical modeling; NN component is relatively simple (fully connected network). |
| Experimental Thoroughness | 5 | Validation on a single galaxy; lacks systematic large-scale experiments. |
| Writing Quality | 8 | Astronomical background clearly explained; methodological motivation well structured. |
| Value | 7 | Opens new pathways for SMBH mass measurement in dusty, AGN-dominated, and high-redshift galaxies. |
| **Overall** | **7.0** | Methodologically innovative with strong physical motivation, but limited experimental scale. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[NeurIPS 2025\] Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)

</div>

<!-- RELATED:END -->
