---
title: >-
  [Paper Note] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems
description: >-
  [NeurIPS 2025][Neural ODE] This paper systematically investigates the extrapolation capability of Neural ODEs (NODEs) on noisy synthetic data…
tags:
  - "NeurIPS 2025"
  - "Neural ODE"
  - "Symbolic Regression"
  - "Dynamical Systems"
  - "Extrapolation"
  - "Scientific Discovery"
date: 2026-05-08
content_hash: dca2415c4dfe08b0
---

# An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems

**Conference**: NeurIPS 2025
**arXiv**: [2601.20637](https://arxiv.org/abs/2601.20637)
**Code**: Available (based on JAX/Diffrax + PySR)
**Area**: Scientific Discovery / Dynamical Systems
**Keywords**: Neural ODE, Symbolic Regression, Dynamical Systems, Extrapolation, Scientific Discovery

## TL;DR

This paper systematically investigates the extrapolation capability of Neural ODEs (NODEs) on noisy synthetic data, and explores a pipeline that employs NODEs as a data augmentation tool combined with symbolic regression (SR) to recover governing equations from limited data. Results demonstrate that this combined approach can recover two of three governing equations—and a strong approximation of the third—using only 10% of the simulation data.

## Background & Motivation

Accurately modeling complex system dynamics or discovering their governing differential equations is central to accelerating scientific discovery. With the explosion of available data and advances in machine learning, inferring the dynamics of complex systems from experimental data has become a critical challenge.

**Neural ODEs** naturally suit differential equation modeling due to their continuous-time formulation, and have been successfully applied in domains such as turbulence prediction and pharmacokinetics. However, existing research has primarily focused on architectural improvements and robustness evaluation, leaving the **boundary-condition extrapolation capability** of NODEs on noisy data insufficiently studied.

**Symbolic Regression (SR)** can directly discover exact governing equations but typically requires large amounts of data, which is often unavailable in experimental science.

The core motivation of this paper is: can a trained NODE serve as a **data augmentation tool**, expanding limited experimental data into a more complete dataset to facilitate SR in recovering underlying physical equations? This offers a practical pathway for scientific discovery in **data-scarce settings**.

## Method

### Overall Architecture

The research framework consists of two interconnected components:

1. **NODE Extrapolation Study**: NODEs are trained on noisy synthetic data and evaluated for generalization to new boundary conditions and longer time horizons.
2. **NODE + SR Pipeline**: NODEs are trained on only 10% of the original simulation data → the NODE generates a complete dataset → SR recovers the governing equations from this dataset.

Two damped oscillatory systems are used as benchmarks:
- **Cart-pole system**: A friction-bearing inverted pendulum without external forcing, described by angle $\theta$ and angular velocity $\omega$.
- **Bio-model**: A biological model of bacterial adaptation to nutrient environment changes, comprising three state variables $(\psi_A, \phi_R, \chi_R)$ and three coupled differential equations.

### Key Designs

**NODE Training and Extrapolation Strategy**:

For the cart-pole system, two models are designed:
- **Model A**: 35 boundary conditions (combinations of initial angle and angular velocity), sampled at 25 Hz, trained on only the first second of data.
- **Model B**: Trained only within a restricted region of initial conditions, used to evaluate extrapolation capability.

For the Bio-model:
- **Model 2A**: Trained exclusively on two up-shift simulation trajectories.
- Multiple models are trained at varying sampling frequencies (5–100 data points/hour).

**Symbolic Regression Analysis**:
- PySR is used as the SR framework.
- SR is performed separately on **ground-truth data** and **NODE-generated data**.
- Equation recovery is compared under noisy and noise-free conditions.
- A key finding is that the choice of input variables has a decisive impact on SR success.

### Loss & Training

- NODEs are implemented using the JAX-based Diffrax library, trained with a standard ODE-fitting objective.
- Synthetic data is corrupted with ±5% uniform noise to simulate realistic experimental conditions.
- Cart-pole simulations run for 10 seconds with a time step of 0.01 seconds.
- Bio-model simulations run for 8 hours with a time step of 0.01 hours.

## Key Experimental Results

### Main Results

**Table 1: SR Analysis Results — Recovery of Three Governing Equations from Different Data Sources**

| Equation | Ground Truth (No Noise) | Ground Truth (5% Noise) | NODE Data (No Noise) | NODE Data (5% Noise) |
|----------|:-:|:-:|:-:|:-:|
| Equation 2 | ✓ | ✗ | ✗* | ✗* |
| Equation 3 | ✓ | ✓ | ✓ | ✓ |
| Equation 4 | ✓ | ✗ | ✓ | ✓ |

*Note: ✗\* indicates that the exact equation was not recovered but a good approximation was found.*

Three key findings emerge from this table:
1. When $\lambda$ is included as an input variable, SR recovers all three equations from noise-free ground-truth data.
2. 5% noise severely degrades SR performance on ground-truth data (only 1/3 equations recovered).
3. SR recovers 2/3 equations from NODE-generated data, with better performance on noisy data—indicating that the NODE acts as a **denoising filter**.

**Table 2: SR Recovery Details for Equation 2**

| Dataset | MSE Loss | SR Result |
|---------|----------|-----------|
| Ground Truth (No Noise) | $2.09\times10^{-8}$ | $2.076 - 3.77\times\phi_R - \lambda - \lambda\times\psi_A$ |
| Ground Truth (5% Noise) | $8.12\times10^{-3}$ | $1.410 - \lambda$ (oversimplified) |
| NODE (No Noise) | $1.91\times10^{-5}$ | $2.024 - 3.54\times\phi_R - \lambda - \psi_A$ |
| NODE (5% Noise) | $2.16\times10^{-4}$ | $2.006 - 3.67\times\phi_R - \lambda$ |

The ground-truth form of Equation 2 is `2.079 - 3.78×ϕ_R - λ - λ×ψ_A`. Observations:
- Noise-free ground-truth data yields near-perfect recovery.
- NODE noise-free data omits the $\lambda$ coefficient in the $\lambda\times\psi_A$ term, substituting $\psi_A$ alone.
- NODE noisy data further loses the $\psi_A$ term, but absorbs its mean into the constant ($2.006$ vs. $2.079$).

### Ablation Study

**Effect of Sampling Frequency on Model Performance**:

Six models at different sampling frequencies (5, 10, 20, 33, 50, 100 data points/hour) are trained on the Bio-model using only the first hour of data. Key findings:

- **8-hour MSE is highly consistent** across sampling frequencies, indicating that high-quality long-term prediction is achievable from very limited training data.
- **1-hour MSE** is significantly higher for the two lowest frequencies (5 and 10 points/hour), as each variable has approximately 6 training points per shift, amplifying the effect of noise.
- Results are averaged over 19 shifts and 10 independent runs.

**Effect of Input Variable Selection**:

When SR input variables include only the three primary state variables $(\psi_A, \phi_R, \chi_R)$ rather than also including $\lambda$, only Equation 4 is recovered. This is because the rational term $\psi_A\cdot\phi_R/(\psi_A + k_a)$ in $\lambda$ is obscured by the data—since $\psi_A$ is on average 8 times larger than $k_a$, the denominator approximates $\psi_A$, reducing the entire rational term to approximately $\phi_R$ and concealing the true structure.

### Key Findings

1. **Phase-space extrapolation of NODEs**: Model B exhibits low MSE regions outside its training domain; analysis reveals these regions **share the same phase-space trajectories** as the training data. This suggests that the key to NODE generalization is capturing diverse dynamical features, not densely sampling initial conditions.

2. **Cross-shift generalization**: Model 2A, trained on only two up-shift trajectories, accurately predicts down-shift responses (error < 5%), demonstrating NODE generalization to unseen dynamical regimes.

3. **Temporal extrapolation**: Model A, trained on only the first second of data, accurately extrapolates dynamics over a horizon 5× longer than the training duration (5 seconds).

4. **Denoising effect of NODEs**: SR recovers only 1/3 of equations from 5%-noisy ground-truth data, whereas SR applied to NODE outputs trained on the same noisy data recovers 2/3 equations plus one approximation, confirming that NODEs effectively filter noise.

## Highlights & Insights

- The core insight that **"diverse dynamics > dense sampling"** directly informs experimental design: when constructing training sets, priority should be given to covering distinct dynamical behaviors rather than simply increasing the number of initial conditions.
- The dual role of **NODEs as both data augmentor and denoiser** is an elegant finding: a single model simultaneously addresses data scarcity and data quality.
- Recovering 2/3 of governing equations via the NODE→SR pipeline using only 10% of simulation data demonstrates the practical potential of this approach for data-scarce experimental sciences.

## Limitations & Future Work

1. SR analysis is based on single-shift simulations only; extending to multi-condition, diversified data may substantially improve equation recovery rates.
2. NODE training data is not optimized to maximize generalization; training set design strategies warrant further exploration.
3. More advanced architectures (e.g., Neural CDEs) may improve data quality and extrapolation range.
4. The SR framework itself could be strengthened through physical priors (e.g., unit consistency) or alternative frameworks (e.g., SINDy).
5. The failure to recover Equation 2 is partly attributable to the low signal-to-noise ratio of the $\lambda\cdot\psi_A$ term (magnitude only 0.01–0.13), requiring more refined signal processing strategies.

## Related Work & Insights

- **Neural ODE** [Chen et al., 2018]: This paper systematically evaluates extrapolation capability built upon this foundation.
- **PySR** [Cranmer, 2023]: The core tool used for symbolic regression.
- **SINDy** [Brunton et al., 2016]: An alternative approach for discovering dynamical equations from data, serving as a potential substitute for the SR framework.
- **Neural CDE** [Kidger et al., 2020]: An advanced NODE variant capable of handling irregular time series.

This paper provides preliminary yet compelling evidence for combining NODEs with SR, inspiring the research direction of using deep learning models as "data amplifiers" to assist interpretable methods.

## Rating

- **Novelty**: ⭐⭐⭐ — Individual components are established, but the NODE→SR pipeline combination and the denoising effect discovery are noteworthy contributions.
- **Technical Depth**: ⭐⭐⭐ — Experiments are systematically designed with careful analysis, though no theoretical analysis is provided.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two systems, multiple ablations, and complete error analysis.
- **Value**: ⭐⭐⭐⭐ — Direct applicability to experimental sciences with limited data.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with highly informative figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics](lagrangian_neural_odes_measuring_the_existence_of_a_lagrangian_with_helmholtz_me.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)

</div>

<!-- RELATED:END -->
