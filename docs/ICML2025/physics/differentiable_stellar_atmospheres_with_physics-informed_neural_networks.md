---
title: >-
  [Paper Note] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks
description: >-
  [ICML 2025][Physics & Scientific Computing] This work proposes Kurucz-a1, a physics-informed neural network (PINN) designed to simulate 1D stellar atmosphere models under the Local Thermodynamic Equilibrium (LTE) assumption. It resolves the key bottleneck of non-differentiable atmospheric structure solvers in differentiable stellar spectroscopy, outperforming the classic ATLAS-12 code in hydrostatic equilibrium and solar spectrum consistency.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
date: 2026-05-08
content_hash: 397ea0e6037288a9
---

# Differentiable Stellar Atmospheres with Physics-Informed Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2507.06357](https://arxiv.org/abs/2507.06357)  
**Code**: None  
**Area**: Scientific Computing  

## TL;DR

This work proposes Kurucz-a1, a physics-informed neural network (PINN) designed to simulate 1D stellar atmosphere models under the Local Thermodynamic Equilibrium (LTE) assumption. It resolves the key bottleneck of non-differentiable atmospheric structure solvers in differentiable stellar spectroscopy, outperforming the classic ATLAS-12 code in hydrostatic equilibrium and solar spectrum consistency.

## Background & Motivation

### Problem Definition
Stellar spectrum modeling consists of two core steps: (1) constructing the atmosphere structure model—solving radiative transfer, radiative equilibrium, and hydrostatic equilibrium equations to obtain the depth-dependent distribution of temperature, pressure, and electron density; (2) synthesizing spectra based on this atmospheric structure. Traditional methods rely on pre-computed grids (e.g., ATLAS, MARCS, PHOENIX), rendering the transition between these two steps fragmented and non-differentiable.

### Limitations of Prior Work
- **Non-differentiability bottleneck**: While existing radiative transfer solvers (e.g., Korg) have been made differentiable, atmospheric structure solvers still rely on legacy codes (e.g., ATLAS-12 written in Fortran 77), making them incompatible with automatic differentiation frameworks.
- **Difficulties in high-dimensional mapping**: ATLAS-12 predicts 6 atmospheric parameters across 80 optical depth points (yielding a 480-dimensional output) from an input space of effective temperature $T_{\text{eff}}$, surface gravity $\log g$, metallicity $\text{[Fe/H]}$, etc. Traditional MLPs lack appropriate inductive biases for this task.
- **Insufficient parameter constraints**: Spectra are influenced by numerous weakly constrained parameters (e.g., oscillator strengths, opacity calculations, convection treatment). A differentiable framework is required to jointly optimize these global physical parameters from large-scale astronomical survey data.

### Design Motivation
Large-scale spectral surveys (e.g., SDSS, LAMOST) provide vast amounts of data. While the fundamental parameters of individual stars differ, the underlying atomic physics remains universal. Developing an end-to-end differentiable modeling framework would allow the optimization of these universal physical parameters by fitting large populations of stellar spectra. This requires transforming the atmospheric structure solver into a differentiable module.

## Method

### 1. Dual-Encoder Architecture

Kurucz-a1 adopts a dual-encoder design to separately process global stellar parameters and local depth information:

- **Stellar Parameter Encoder**: Takes 4 fundamental stellar parameters ($T_{\text{eff}}$, $\log g$, $\text{[Fe/H]}$, and $\text{[\alpha/Fe]}$) as input and encodes them into a 512-dimensional embedding vector via an MLP.
- **Optical Depth Encoder**: Encodes each of the 80 Rosseland optical depth points into a 512-dimensional representation.
- **Feature Fusion**: The stellar parameter embedding is broadcasted and concatenated with each depth embedding, forming 80 vectors of 1024 dimensions.
- **Prediction Head**: A 3-layer MLP (hidden dimensions: 1024-512-256) with GeLU activations, which predicts 6 atmospheric parameters for each depth point:
    - $\rho_x$ (column mass density), $T$ (temperature), and $P$ (gas pressure)
    - $X_{\text{NE}}$ (electron number density), $\kappa_{\text{Ross}}$ (Rosseland mean opacity), and $\text{ACCRAD}$ (radiative acceleration)

### 2. Physics-Constrained Loss Function

The total loss is a weighted combination of the data reconstruction loss and the physics-informed constraint loss:

$$L_{\text{total}} = (1 - \alpha) L_{\text{data}} + \alpha L_{\text{physics}}$$

- **Data Loss $L_{\text{data}}$**: The reconstruction error between the predictions from Kurucz-a1 and the reference outputs from the ATLAS-12 model.
- **Physics Loss $L_{\text{physics}}$**: Enforces the hydrostatic equilibrium constraint $dP/d\tau = g/\kappa$ to ensure that the predicted pressure-depth relationship is physically self-consistent.
- **Weight $\alpha$**: Controls the strength of the physical constraints, balancing data fitting and adherence to fundamental physical laws.

### 3. Design Philosophy

The dual-encoder architecture reflects the underlying physics: global stellar parameters dictate the overall atmospheric structure, while local conditions vary systematically with atmospheric depth. The key innovation of the PINN lies in directly encoding physical laws like hydrostatic equilibrium into the training process, providing inductive biases that standard neural networks fail to capture.

## Key Experimental Results

### Experimental Setup
- **Training Data**: Stellar atmosphere models generated using the ATLAS-12 code.
- **Validation Set**: A diverse parameter range covering Milky Way stellar populations.
- **Baselines**: A standard MLP (without physical constraints) and the original ATLAS-12 code.

### Table 1: Relative Errors of Atmospheric Parameter Predictions

| Atmospheric Parameter | Kurucz-a1 | MLP Baseline | Description |
|---------|-----------|-------------|------|
| Column mass density RHOX | Extremely low | Higher | Low error across the entire optical depth range |
| Temperature T | Extremely low | Higher | Most accurate in the intermediate optical depth region |
| Gas pressure P | Extremely low | Higher | Pressure prediction is significantly improved by physical constraints |
| Rosseland opacity | Lower | Higher | Opacity error distribution is the broadest but remains controlled |

### Table 2: Comparison of Hydrostatic Equilibrium Consistency

| Model | Hydrostatic Equilibrium Loss | Solar Spectrum Consistency | Physical Self-Consistency |
|------|---------------|-------------|-----------|
| ATLAS-12 | Compact and near-zero | Baseline | Good |
| Kurucz-a1 (PINN) | Compact and near-zero | Better than ATLAS-12 | Excellent |
| MLP Baseline | Displaced and higher | Poor | Insufficient |

**Key Findings**: Kurucz-a1 nearly matches ATLAS-12 in hydrostatic equilibrium diagnostics, whereas the MLP baseline exhibits significant deviation. The validation distribution of the hydrostatic equilibrium loss shows that Kurucz-a1 behaves comparably to ATLAS-12, with both tightly clustered near zero.

### Solar Spectrum Validation
The stellar atmosphere model generated by Kurucz-a1 matches solar observations even better than ATLAS-12 itself. This demonstrates the advantages of modern optimization: PINNs can find physically self-consistent solutions that outperform traditional iterative solvers through global optimization.

## Highlights & Insights

1. **Resolving a Key Bottleneck**: For the first time, a stellar atmospheric structure solver is transformed into a differentiable module. When paired with differentiable radiative transfer codes like Korg, this enables end-to-end differentiable stellar spectral modeling.
2. **Physics Outperforming Numerical Methods**: Kurucz-a1 outperforms ATLAS-12 in hydrostatic equilibrium compliance and solar spectrum agreement, demonstrating that PINNs combined with modern optimization techniques can surpass traditional numerical iteration methods.
3. **Elegant Dual-Encoder Design**: Separately encoding global stellar parameters and local optical depths mirrors the physical structure of the system, providing correct inductive biases.
4. **Enabling Data-Driven Astrophysics**: Allows the joint optimization of universal atomic physics parameters directly from large-scale survey data, establishing a foundational capability for next-generation stellar astrophysics.

## Limitations & Future Work

1. **LTE Assumption Limitation**: The model is restricted to Local Thermodynamic Equilibrium (LTE) conditions, failing to accommodate extreme stars where non-LTE (NLTE) effects are significant (e.g., extremely metal-poor stars, supergiants).
2. **1D Atmosphere Approximation**: It employs a 1D atmosphere model and ignores 3D convection effects and inhomogeneities, which may limit the accuracy of detailed abundance measurements.
3. **Dependence on ATLAS-12 Training Data**: Since the model learns from ATLAS-12 outputs, it might inherit systematic biases present in the legacy code.
4. **Parameter Space Coverage**: The current model only includes 4 fundamental parameters ($T_{\text{eff}}$, $\log g$, $\text{[Fe/H]}$, and $\text{[\alpha/Fe]}$), leaving out the feedback of individual element abundances on the atmospheric structure.
5. **Generalization Remains to be Verified**: The model's performance on extreme stellar types outside the training distribution (e.g., white dwarfs, Wolf-Rayet stars) is still unverified.

## Related Work & Insights

- **Traditional Atmosphere Models**: ATLAS (Castelli & Kurucz, 2003), MARCS (Gustafsson et al., 2008), PHOENIX (Allard, 2016) — pre-computed grid methods, non-differentiable.
- **Differentiable Radiative Transfer**: Korg (Wheeler et al., 24) — achieves differentiable spectral synthesis but relies on predetermined, fixed atmospheric structures.
- **PINN Foundations**: Raissi et al. (2019) — foundational work introducing physics-based constraint injection into neural networks.
- **Large-Scale Surveys**: SDSS (York, 2000), LAMOST (Zhao et al., 2012) — provide massive quantities of data for the optimization of universal physical parameters.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

**Overall Rating**: ⭐⭐⭐⭐ (4/5)

This work makes a significant contribution to scientific computing in astrophysics. Applying PINNs to stellar atmosphere modeling is not only technically innovative but also addresses a major bottleneck in end-to-end differentiable stellar spectroscopy. The fact that Kurucz-a1 surpasses its own training target, ATLAS-12, in terms of physical consistency underscores the immense potential of physics-informed neural networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[ICML 2025\] Causal-PIK: Causality-based Physical Reasoning with a Physics-Informed Kernel](causal-pik_causality-based_physical_reasoning_with_a_physics-informed_kernel.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](../../NeurIPS2025/physics/neuro-spectral_architectures_for_causal_physics-informed_networks.md)

</div>

<!-- RELATED:END -->
