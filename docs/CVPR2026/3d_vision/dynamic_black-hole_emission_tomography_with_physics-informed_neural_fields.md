---
title: >-
  [Paper Note] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields
description: >-
  [CVPR2026][3D Vision][Black hole imaging] This paper proposes PI-DEF, a physics-informed coordinate neural network framework that jointly reconstructs the 4D (temporal + 3D spatial) emissivity field and 3D velocity field of gas near a black hole. Under sparse EHT measurements, PI-DEF significantly outperforms BH-NeRF, which enforces hard Keplerian dynamical constraints.
tags:
  - CVPR2026
  - 3D Vision
  - Black hole imaging
  - neural radiance fields
  - physics-informed constraints
  - 4D tomography
  - Event Horizon Telescope
date: 2026-05-08
content_hash: b27a85dd418ec5d8
---

# Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields

**Conference**: CVPR2026
**arXiv**: [2602.08029](https://arxiv.org/abs/2602.08029)
**Code**: Not released
**Area**: 3D Vision / Computational Imaging / Scientific Imaging
**Keywords**: Black hole imaging, neural radiance fields, physics-informed constraints, 4D tomography, Event Horizon Telescope

## TL;DR

This paper proposes PI-DEF, a physics-informed coordinate neural network framework that jointly reconstructs the 4D (temporal + 3D spatial) emissivity field and 3D velocity field of gas near a black hole. Under sparse EHT measurements, PI-DEF significantly outperforms BH-NeRF, which enforces hard Keplerian dynamical constraints.

## Background & Motivation

1. **Static imaging has succeeded; dynamic 3D imaging is the next frontier**: The EHT has successfully captured static 2D images of M87\* and Sgr A\*, but static images are complex 2D projections of the 3D emissivity distribution and cannot reveal the physical nature of the dynamic 3D environment.
2. **Severely underdetermined inverse problem**: The EHT observes from a single viewpoint with highly sparse and noise-contaminated measurements, making 4D tomographic reconstruction an extremely ill-posed problem.
3. **Dynamic sources further complicate reconstruction**: Radiating gas moves, appears, and disappears over time, precluding simple aggregation of measurements across time to improve reconstruction quality.
4. **Partially unknown forward model**: Light propagation depends on the fluid dynamics near the black hole, which are not fully known, leaving the measurement forward model incomplete.
5. **BH-NeRF relies on overly strong assumptions**: The only prior method, BH-NeRF, assumes Keplerian dynamics; however, near the black hole, strong gravitational and electromagnetic activity causes fluid dynamics to deviate from the Keplerian model, and the method cannot handle newly emerging radiation.
6. **High scientific significance**: Recovering the dynamic 3D emissivity field near a black hole can reveal previously unseen regions of the universe, helping to test general relativity and infer physical parameters such as black hole spin.

## Method

### Overall Architecture

PI-DEF (Physics-Informed Dynamic Emission Fields) represents the 4D emissivity field $e(t, \mathbf{x}; \theta_e)$ and 3D velocity field $\tilde{u}^i(\mathbf{x}; \theta_v)$ using two separate coordinate neural networks, optimized jointly through physics-informed losses. The emissivity network takes $(t, x, y, z)$ as input, while the velocity network takes spatial coordinates as input; both employ positional encoding $\gamma$ to enhance high-frequency representation capacity.

### Loss & Training

The total loss consists of three terms: $\mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$

- **Data fitting loss** $\mathcal{L}_{\text{data}}$: The emissivity field is projected onto the image plane via general-relativistic geodesic ray tracing to simulate EHT visibility measurements, which are then fit to the observed data using a Gaussian likelihood. The redshift factor $g^2$ is derived from the velocity field, so this loss jointly constrains both networks.
- **Dynamics loss** $\mathcal{L}_{\text{dyn}}$ (core contribution): The emissivity field $e(t)$ is propagated by $\Delta t$ using velocities predicted by the velocity network via an ODE solver to obtain $\hat{e}(t+\Delta t)$, which is then compared to the emissivity network's direct prediction $e(t+\Delta t)$ using an L1 loss. A Gaussian blur is applied beforehand to prevent blurry reconstructions from trivially satisfying this loss.
- **Velocity regularization** $\mathcal{L}_{\text{reg}}$: The AART velocity model serves as a soft prior, with L2 regularization between the estimated and theoretical velocities. Crucially, the regularization weight decays exponentially during training ($\lambda_{\text{init}}=10^6 \to \lambda_{\text{final}}=10$), so that the reconstruction is guided by the prior in early stages and becomes primarily data-driven in later stages.

### Key Designs

- Velocities are estimated in the numerically stable normal observer frame, avoiding numerical issues where the four-velocity $u^t$ is undefined.
- The AART velocity model, incorporating sub-Keplerian azimuthal velocity and radial infall, is parameterized by three parameters $(\beta_\phi, \beta_r, \xi)$.
- The emissivity field is time-dependent and can capture newly appearing radiation within the observation window, a capability absent in BH-NeRF.
- General-relativistic geodesics are computed using the kgeo library; EHT measurements are simulated using the eht-imaging library.

## Key Experimental Results

### Emissivity Reconstruction Accuracy (5 random test scenarios, ngEHT measurements)

| Method | PSNR (dB) ↑ | MSE (×10⁻⁵) ↓ |
|------|-------------|----------------|
| **PI-DEF (Ours)** | **37.3 ± 2.3** | **2.3 ± 0.2** |
| 4D-MLP | 35.4 ± 0.5 | 3.8 ± 0.4 |
| BH-NeRF | 34.0 ± 1.9 | 4.9 ± 0.8 |

- PI-DEF significantly outperforms both BH-NeRF and the purely data-driven 4D-MLP even when the velocity prior is misspecified (assuming purely sub-Keplerian dynamics without radial infall).
- BH-NeRF fails severely near the black hole due to its hard Keplerian constraint.

### Ablation Study

- **Measurement sparsity**: ngEHT (23 telescopes) yields substantially better reconstruction quality than EHT 2025 (12 telescopes) and EHT 2017 (8 telescopes); the improvement from EHT 2017 to EHT 2025 is limited.
- **Velocity recovery**: In high-emissivity regions (above the 65th percentile), PI-DEF recovers radial and azimuthal velocities in good agreement with ground truth, even under initial prior mismatch. Velocity recovery in low-emissivity regions is unconstrained.
- **Velocity network parameterization**: Parameterizing the velocity network using only $r$, $(r,\theta)$, or $(x,y,z)$ all yield reasonable results; axisymmetric constraints are not strictly necessary.
- **Realistic noise**: The method remains functional under realistic Gaussian noise at the true total flux density of Sgr A\* (~2.3 Jy).
- **Atmospheric noise**: Using closure phases and amplitudes in place of complex visibilities mitigates atmospheric phase errors, though reconstruction accuracy is reduced.
- **Spin inference**: The data fitting loss is sensitive to the assumed black hole spin; the correct spin $a=0.2$ yields the lowest loss, demonstrating the potential of PI-DEF for physical parameter inference.

## Highlights & Insights

- **Soft vs. hard constraints**: Incorporating the physical velocity model as an exponentially decaying soft regularizer rather than a hard constraint elegantly balances prior guidance with robustness to modeling errors—a design that is both principled and generalizable.
- **Joint dual-field reconstruction**: The simultaneous recovery of the 4D emissivity field and 3D velocity field, with the dynamics loss enforcing physical consistency between the two, constitutes a coherent and well-motivated formulation.
- **Outstanding scientific value**: This work achieves dynamic 3D reconstruction in the near-black-hole region (not merely distant flares) for the first time, and demonstrates the feasibility of spin inference.
- **Computer vision advancing fundamental physics**: The combination of the NeRF paradigm with general-relativistic geodesic ray tracing exemplifies how CV techniques can drive progress at the frontier of astrophysics.

## Limitations & Future Work

- Validation is performed exclusively on simulated data; the method has not yet been applied to real EHT observations.
- The slow-light effect—arising from the finite speed of light when gas moves at relativistic velocities—is neglected.
- Absorption and scattering attenuation of radiation are ignored.
- Very close to the black hole (near the event horizon), the vanishing redshift factor $g^2 \to 0$ renders the emissivity contribution negligible, making velocity recovery difficult in this region.
- Joint optimization of black hole spin and inclination has not been performed; spin inference is demonstrated only as a grid-search proof of concept.
- Gaussian splatting is excluded due to its inability to handle the appearance and disappearance of hotspots, though its efficiency advantages could potentially be recovered in future work through dynamic point management.

## Related Work & Insights

| Method | Temporal Modeling | Velocity Constraint | New Emitters | Reference |
|------|---------|---------|---------|--------|
| BH-NeRF | Initial field + Keplerian propagation | Hard constraint (Keplerian) | ✗ | ECCV 2022 |
| 4D-MLP | 4D coordinate network | Unconstrained | ✓ | Baseline |
| **PI-DEF** | **4D coordinate network** | **Soft constraint (AART with decaying regularization)** | **✓** | **Ours** |

PI-DEF differs fundamentally from dynamic scene NeRF methods in that it operates under the extreme setting of curved light paths (general-relativistic geodesics) + single viewpoint + dynamic sources, rather than the standard setting of linear ray tracing + multi-view observations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to introduce physics-informed soft constraints into 4D black hole tomography; the dual-field joint reconstruction and decaying regularization design are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulation experiments are comprehensive (sparsity, noise, spin inference, ablations), but real-data validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Physical background and methodology are presented with clarity; figures are well-crafted and accessible to readers without an astrophysics background.
- Value: ⭐⭐⭐⭐⭐ — A landmark demonstration of CV techniques advancing fundamental physics, with direct applicability to EHT science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](../../ICLR2026/3d_vision/hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[ICLR 2026\] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](../../ICLR2026/3d_vision/diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](../../NeurIPS2025/3d_vision/learning_neural_exposure_fields_for_view_synthesis.md)
- [\[CVPR 2026\] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation](let_it_snow_animating_3d_gaussian_scenes_with_dynamic_weather_effects_via_physic.md)

</div>

<!-- RELATED:END -->
