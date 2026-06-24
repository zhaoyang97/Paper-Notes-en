---
title: >-
  [Paper Note] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation
description: >-
  [CVPR 2025][Inverse physics simulation] This work proposes UniPhy, the first unified latent-conditioned constitutive model that encodes diverse material properties (such as elastomers, sand, plastics, Newtonian, and non-Newtonian fluids) within a shared latent space. During inference, the latent variables are optimized through a differentiable Material Point Method (MPM) simulator to match observed particle trajectories, reducing reconstruction errors by 1 to 2 orders of magn…
tags:
  - "CVPR 2025"
  - "Inverse physics simulation"
  - "constitutive model"
  - "latent variable optimization"
  - "MPM simulation"
  - "material parameter estimation"
date: 2026-05-08
content_hash: 6a48177bf605804f
---

# UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation

**Conference**: CVPR 2025  
**arXiv**: [2505.16971](https://arxiv.org/abs/2505.16971)  
**Code**: [https://himangim.github.io/UniPhy](https://himangim.github.io/UniPhy)  
**Area**: Others / Physics Simulation  
**Keywords**: Inverse physics simulation, constitutive model, latent variable optimization, MPM simulation, material parameter estimation

## TL;DR

This work proposes UniPhy, the first unified latent-conditioned constitutive model that encodes diverse material properties (such as elastomers, sand, plastics, Newtonian, and non-Newtonian fluids) within a shared latent space. During inference, the latent variables are optimized through a differentiable Material Point Method (MPM) simulator to match observed particle trajectories, reducing reconstruction errors by 1 to 2 orders of magnitude compared to NCLaw.

## Background & Motivation

**Background**: Inverse physics simulation infers material properties (e.g., elastic modulus, viscosity) from observational data. Existing methods either design specialized models for each material type (such as Neo-Hookean for elasticity, Navier-Stokes for fluids) or utilize learning-based methods like NCLaw which require pre-specifying the material type.

**Limitations of Prior Work**: In real-world scenarios, the material type is often unknown; for instance, a soft-looking object could be either an elastomer or a plastic. Existing approaches require classifying the material before inferring its properties, making both steps prone to errors.

**Key Challenge**: The physical behaviors of different materials vary drastically (e.g., elastomeric bouncing vs. sand collapsing vs. fluid flowing), making a unified representation with a single model seemingly impractical.

**Key Insight**: Material properties are encoded into a continuous latent vector $z$, where different materials correspond to distinct regions in the latent space while sharing the same neural network architecture. During inference, optimizing the latent vector automatically "discovers" the material type.

**Core Idea**: Latent-conditioned deformation projection $g_\phi(F,z)$ + constitutive law $f_\theta(F_{proj},z)$ = a unified multi-material physics model.

## Method

### Key Designs

1. **Latent-Conditioned Dual-Network Architecture**:

    - **Function**: Uniformly encodes the physical behavior of various materials
    - **Mechanism**: Two neural networks are employed: a deformation gradient projection function $g_\phi(F,z)$ that projects the deformation gradient onto a material-dependent subspace (e.g., projecting elastomers onto a volume-preserving manifold), and a constitutive law function $f_\theta(F_{proj},z)$ that computes stress from the projected deformation. The latent vector $z$ encodes material properties.
    - **Design Motivation**: To decouple "how deformation is decomposed" from "how stress is computed." Different materials decompose and project deformation differently (e.g., volume-preservation for elasticity vs. yield surfaces for sand), but the network structure for stress computation can be shared.

2. **Latent Variable Optimization via Differentiable MPM**:

    - **Function**: Infers material properties from observed particle trajectories
    - **Mechanism**: The pre-trained $g_\phi$ and $f_\theta$ are embedded into a differentiable Material Point Method (MPM) simulator. While holding the network parameters fixed, the latent vector $z$ is optimized so that the simulated trajectories match the observations, using L2 distance as the loss.
    - **Design Motivation**: By backpropagating through the simulator's gradient stream to the latent space, end-to-end "observation to material" inference is achieved.

### Loss & Training

Training: $L = \sum_n \sum_t \sum_p (L(F_{proj}, \hat{F}_{proj}) + L(S, \hat{S}) + \frac{1}{\sigma^2}\|z_n\|^2)$, which includes deformation projection error + stress error + latent variable regularization. Inference: L2 particle position error. A 32-dimensional latent space is optimal.

## Key Experimental Results

### Main Results

Reconstruction Error ($\times 10^{-5}$):

| Material | UniPhy | NCLaw |
|------|--------|-------|
| Elastomer | **0.052** | 2.40 |
| Sand | **1.50** | 2.60 |
| Plastic | **3.90** | 6.50 |
| Newtonian Fluid | **0.011** | 2.00 |

### Ablation Study

| Configuration | Elastomer Error |
|------|-----------|
| UniPhy (w/ teacher forcing) | 0.052 |
| UniPhy (w/o TF) | 1.10 |
| NCLaw (w/o TF) | 36.0 |
| 4D Latent Space | 7.80 |
| 32D Latent Space | **1.10** |
| 256D Latent Space | 1.50 |

### Key Findings
- UniPhy reduces error by 1-2 orders of magnitude compared to NCLaw—showing that a unified model is not only feasible but actually outperforms dedicated models.
- A 32-dimensional latent space represents the optimal balance—too small cannot distinguish between materials, while too large leads to overfitting.
- The optimized latent variables accurately reflect material types—different materials cluster into distinct regions.

## Highlights & Insights
- **Counter-intuitive Result of Unified > Dedicated**: Handling all materials with a single model outperforms designing specialized models for each, likely due to shared knowledge across materials (e.g., the general structure of continuum mechanics).
- **Latent Vector as Material Fingerprint**: No prior classification of material types is required; the optimized latent vector automatically encodes the material properties.

## Limitations & Future Work
- Only homogeneous materials are supported (no multi-material mixtures).
- Requires known initial geometry and 3D motion observations.
- Trained only on simulated data; generalization to the real world remains unverified.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified multi-material constitutive model.
- Experimental Thoroughness: ⭐⭐⭐⭐ Diverse material types and thorough generalization testing.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for inverse physics simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](../../CVPR2026/others/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](../../NeurIPS2025/others/learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)

</div>

<!-- RELATED:END -->
