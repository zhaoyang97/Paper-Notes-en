---
title: >-
  [Paper Note] Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics
description: >-
  [NeurIPS 2025][3D Vision][cosmological simulation] This paper proposes a hybrid physical-neural cosmological simulator that handles gravitational dynamics via a differentiable particle-mesh (PM) method and parameterizes the effective gas pressure field using a physics-constrained neural network. The model requires only a single reference simulation for training and outperforms the EGD baseline at both the field level and the statistics level.
tags:
  - NeurIPS 2025
  - 3D Vision
  - cosmological simulation
  - hybrid physical-neural method
  - particle-mesh
  - differentiable simulation
  - gas dynamics
date: 2026-05-08
content_hash: 6f8ac344dca9df7a
---

# Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics

**Conference**: NeurIPS 2025
**arXiv**: [2510.26593](https://arxiv.org/abs/2510.26593)
**Code**: Based on JaxPM (open-source); neural pressure model code TBD
**Area**: 3D Vision
**Keywords**: cosmological simulation, hybrid physical-neural method, particle-mesh, differentiable simulation, gas dynamics

## TL;DR
This paper proposes a hybrid physical-neural cosmological simulator that handles gravitational dynamics via a differentiable particle-mesh (PM) method and parameterizes the effective gas pressure field using a physics-constrained neural network. The model requires only a single reference simulation for training and outperforms the EGD baseline at both the field level and the statistics level.

## Background & Motivation

**Background**: Large-scale cosmic structure analysis aims to constrain fundamental cosmological parameters. Field-level inference leverages the full spatial information encoded in the cosmic density and velocity fields to jointly constrain cosmological parameters and initial conditions. Because the parameter space of initial conditions is extremely high-dimensional, inference requires gradient-based sampling methods (e.g., Hamiltonian Monte Carlo), which in turn demand differentiable forward models.

**Limitations of Prior Work**: The majority of matter in the universe is dark matter, whose purely gravitational dynamics can be efficiently solved with differentiable PM methods. However, modern observations now probe small scales where contributions from ordinary (baryonic) matter (gas) must be accounted for, and hydrodynamical simulations of gas (e.g., full Euler equation solvers) are computationally prohibitive. Existing approaches are either physically complete but expensive (diffHydro), analytically approximate but limited in accuracy (HPM/HYPER), or lack self-consistent temporal evolution due to their post-processing nature (EGD).

**Key Challenge**: There is a need for a forward model of joint gas–dark matter co-evolution that is simultaneously differentiable (to support gradient-based sampling), efficient (to avoid full hydrodynamics), and accurate (to match reference simulations).

**Goal**: To construct a solver-in-the-loop hybrid simulator that retains a physics-driven gravitational solver while replacing expensive hydrodynamics computations with a data-driven neural network.

**Key Insight**: The observation that dark matter constitutes the bulk of cosmic matter and already admits an efficient differentiable solver, while gas physics can be regarded as a "correction term" to gravitational dynamics. The gas pressure field is parameterized by a neural network as a function of local physical quantities (density, velocity divergence, velocity dispersion, etc.) and is embedded within the PM solver for joint temporal evolution.

**Core Idea**: Gravity solved by physics + gas pressure learned by a neural network = an efficient, differentiable hybrid cosmological simulator.

## Method

### Overall Architecture
Gas particle species are introduced on top of JaxPM (an open-source dark matter PM simulator). The masses of both particle types are proportional to the cosmological density ratios $\Omega_{\mathrm{cdm}}$ and $\Omega_b$, respectively. The system evolves via ordinary differential equations (ODEs) in comoving coordinates: dark matter particles are subject only to gravity, while gas particles are additionally driven by the learned pressure field. The entire system is implemented in JAX and supports automatic differentiation.

### Key Designs

1. **Hybrid Physical-Neural Equations of Motion**

    - **Function**: Defines the equations of motion for dark matter and gas particles in an expanding universe background.
    - **Mechanism**: Dark matter particles are driven solely by the total gravitational potential $\Phi_{\mathrm{tot}}$, while gas particles additionally experience a pressure gradient force $-\nabla P / \rho_{\mathrm{gas}}$. The gravitational potential is efficiently solved in Fourier space via the Poisson equation $\nabla^2 \Phi_{\mathrm{tot}} = \frac{3}{2} H_0^2 \Omega_m \delta_{\mathrm{tot}}$. Optionally, a neural residual correction $\tilde{\Phi}^{\star} = (1 + f_\theta(a, |\mathbf{k}|)) \tilde{\Phi}$ can be applied to the gravitational potential.
    - **Design Motivation**: Gravity is physically well-understood and can be computed efficiently, so it does not need to be learned. Restricting the learning component to gas pressure substantially reduces the complexity of what must be modeled.

2. **Neural Effective Pressure Field**

    - **Function**: Predicts the effective gas pressure field using a physics-constrained neural network.
    - **Mechanism**: Based on the ideal gas assumption, the pressure is factored as $P_\varphi(a, \mathbf{x}) \propto \rho_{\mathrm{gas}}(\mathbf{x}) \cdot U_\varphi(a, \mathbf{h}(\mathbf{x}))$, where $U_\varphi$ is a fully convolutional network (6 layers, 16 channels, $3\times3\times3$ kernels, 47,265 parameters) predicting internal energy. The network outputs $\log U_\varphi$ to ensure non-negativity and reduce dynamic range. The input feature vector $\mathbf{h}(\mathbf{x}) = (\rho_{\mathrm{gas}}, f_{\mathrm{scalar}}, \nabla \cdot \mathbf{v}, \sigma_\mathbf{v}^2)$ comprises gas density, a scalar force, velocity divergence, and velocity dispersion—all of which can be computed efficiently on the PM grid.
    - **Design Motivation**: Directly predicting $P$ leads to a large dynamic range and training instability. The decomposition $P \propto \rho \cdot U$ exploits the physical relationship of ideal gases, narrowing the prediction target to the internal energy $U$ and reducing learning difficulty. The use of local features decouples the network's receptive field from the simulation box size, enabling generalization to larger volumes.

3. **Solver-in-the-Loop Training**

    - **Function**: End-to-end training of the entire simulator to match a reference full-hydrodynamics simulation.
    - **Mechanism**: The ODE is integrated with diffrax, and automatic differentiation with respect to model parameters is achieved via recursive checkpointing adjoint methods. The loss function minimized is $\mathcal{L} = \sum_s [H_\delta(\mathbf{r}_s) + \lambda H_{\delta'}(\mathbf{v}_s - \mathbf{v}_s^{\mathrm{ref}}) + \mu \|\frac{P_s(|\mathbf{k}|)}{P_s^{\mathrm{ref}}(|\mathbf{k}|)} - 1\|_2^2]$, comprising a Huber loss on particle positions with periodic boundary conditions, a Huber loss on velocities, and a power spectrum ratio loss. The loss is evaluated at only 4 of the 34 simulation snapshots.
    - **Design Motivation**: The Huber loss reduces sensitivity to outlier particles. The power spectrum term ensures statistical-level matching. Extremely sparse supervision (4 out of 34 snapshots from a single simulation) remains effective because the physics-constrained parameterization provides strong inductive bias, the large particle count supplies abundant signal per snapshot, and the physics-driven gravitational component requires no learning from data.

### Loss & Training
- A single Simba reference simulation ($128^3$ particles), 4 snapshots, training on a single A100 GPU.
- Adam optimizer, learning rate $10^{-4}$, 1000 steps, gradient clipping with global norm 1.
- Data augmentation: random $90°$ rotations and axis-aligned flips.
- FiLM conditioning enables the network to be aware of the scale factor $a$ for temporal awareness of cosmic evolution.

## Key Experimental Results

### Main Results — Gas Density Two-Point Statistics

The method is evaluated on the Simba subset of the CAMELS suite against held-out test simulations with different random initial conditions.

| Method | Power Spectrum Match | Cross-Correlation Coefficient | Field-Level Performance |
|---|---|---|---|
| JaxPM (gravity only) | Excess small-scale power | Lower | Lacks gas feedback effects |
| EGD (post-processing) | Comparable to Ours | Lower than Ours | Over-smooths fine details |
| **Ours (hybrid)** | **Good match to reference** | **Highest** | **Preserves more structural detail** |
| CAMELS reference | Baseline | Baseline | Baseline |

The proposed method is on par with EGD in terms of power spectrum (Fourier mode amplitude), but achieves substantially higher cross-correlation coefficients (Fourier phase consistency), indicating superior field-level structural matching.

### Method Comparison

| Property | Ours | EGD | diffHydro |
|---|---|---|---|
| Self-consistent temporal evolution | ✓ (full history) | ✗ (single-snapshot post-processing) | ✓ |
| Differentiable | ✓ | ✓ | ✓ |
| Computational efficiency | High (PM + small CNN) | High (PM + analytic) | Low (full Euler equations) |
| Data efficiency | Extremely high (1 simulation, 4 snapshots) | Requires power spectrum minimization | No training required |
| Physical constraints | Exact gravity + learned pressure | Power-law temperature assumption | Full physics |

### Key Findings
- Dark matter distributions show virtually no difference across methods, confirming that baryonic back-reaction is negligible at this resolution.
- The model generalizes across cosmic variance (i.e., different random initial conditions)—training on a single simulation suffices for application to varied initial conditions.
- Extremely sparse supervision (only 4 out of 34 snapshots) proves sufficient, validating the synergy between physics-constrained parameterization and the large number of particle data points.

## Highlights & Insights
- **Extreme data efficiency**: A single simulation and 4 snapshots suffice to constrain the entire neural pressure model. This stems from the strong inductive bias provided by physics constraints (ideal gas decomposition, exact gravitational solution). The principle is transferable to other physics simulations—retain as much known physics as possible and learn only the unknown or computationally expensive components.
- **Elegant solver-in-the-loop implementation**: Rather than fitting simulation outputs after the fact, the model is embedded within the ODE solver and trained end-to-end. This guarantees temporal consistency between the learned pressure field and gravitational evolution—a fundamental advantage over post-processing methods such as EGD.
- **Potential for direct connection to observational data**: Given the extremely high data efficiency and differentiability, it is in principle possible to bypass simulation training sets entirely and fit the pressure model directly to Sunyaev–Zel'dovich (SZ) effect and weak gravitational lensing observations—thereby circumventing the model misspecification problem that arises from differences between hydrodynamics codes.

## Limitations & Future Work
- The pressure field relies solely on instantaneous local quantities, neglecting the thermodynamic history of particles. A complete thermodynamic treatment would require tracking additional state variables such as internal energy or entropy. The authors note that history-aware pressure prediction can be achieved by introducing latent variables to form a neural ODE.
- The spatial resolution of PM methods is limited by grid size, restricting the ability to model structures at very small scales (e.g., galaxy interiors).
- Validation is performed only on the Simba hydrodynamics code; the strength of gas feedback varies considerably across different codes (Astrid, IllustrisTNG).
- Current experiments are conducted at a relatively small scale ($128^3$ particles, $(25\ \mathrm{Mpc}/h)^3$ box); scaling to cosmological volumes requires further validation.
- The CNN architecture is confined to a grid representation; future work could explore particle-based MLP or graph network architectures.

## Related Work & Insights
- **vs. HYPER (HPM)**: HYPER also incorporates gas pressure into the PM framework but derives the pressure functional form from an analytic halo model, requiring separate treatment of the low-density intergalactic medium and high-density galaxy cluster gas. This work replaces the analytic function with a data-driven neural network, offering greater flexibility.
- **vs. diffHydro**: diffHydro achieves fully differentiable hydrodynamics by solving the complete Euler equations, which is physically more rigorous but computationally far more expensive. This work finds a practical trade-off between accuracy and efficiency through a physics-plus-learning hybrid strategy.
- **vs. EGD**: EGD is a post-processing method that assumes a power-law temperature–density relation and displaces particles along enthalpy gradients. This work avoids the limitations of post-processing through self-consistent temporal evolution and achieves superior field-level performance as measured by the cross-correlation coefficient.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The solver-in-the-loop hybrid physical-neural design is elegant, and the extreme data efficiency is a notable highlight.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments are conducted at a relatively small scale with only one hydrodynamics code and lack quantitative comparison against full hydrodynamics simulations.
- **Writing Quality**: ⭐⭐⭐⭐ Physical background is introduced appropriately and the method derivation is rigorous, though the paper has a high entry barrier for readers outside cosmology.
- **Value**: ⭐⭐⭐⭐ The work is of significant value to field-level cosmological inference, and the hybrid physical-neural methodology offers broader inspiration for other scientific computing domains.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[NeurIPS 2025\] PhysX-3D: Physical-Grounded 3D Asset Generation](physx-3d_physical-grounded_3d_asset_generation.md)
- [\[NeurIPS 2025\] Neural Green's Functions](neural_greens_functions.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)

<!-- RELATED:END -->
