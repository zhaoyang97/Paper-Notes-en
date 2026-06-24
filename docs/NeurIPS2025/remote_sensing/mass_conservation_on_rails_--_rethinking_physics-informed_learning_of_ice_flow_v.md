---
title: >-
  [Paper Note] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields
description: >-
  [NeurIPS 2025][Remote Sensing][Divergence-free neural networks] This paper proposes a divergence-free neural network (dfNN) that architecturally enforces exact mass conservation (divergence identically zero) via the symplectic gradient of a stream function, and combines it with a directional guidance learning strategy. The approach significantly outperforms soft-constraint PINNs and unconstrained NNs on ice flux interpolation over Antarctica's Byrd Glacier.
tags:
  - "NeurIPS 2025"
  - "Remote Sensing"
  - "Divergence-free neural networks"
  - "physics-informed neural networks"
  - "ice flow modeling"
  - "mass conservation"
  - "vector field interpolation"
date: 2026-05-08
content_hash: b8715167ac6ef71e
---

# Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields

**Conference**: NeurIPS 2025
**arXiv**: [2510.06286](https://arxiv.org/abs/2510.06286)  
**Code**: [GitHub](https://github.com/kimbente/mass_conservation_on_rails)  
**Area**: Remote Sensing / Physics-Informed Machine Learning
**Keywords**: Divergence-free neural networks, physics-informed neural networks, ice flow modeling, mass conservation, vector field interpolation

## TL;DR

This paper proposes a divergence-free neural network (dfNN) that architecturally enforces exact mass conservation (divergence identically zero) via the symplectic gradient of a stream function, and combines it with a directional guidance learning strategy. The approach significantly outperforms soft-constraint PINNs and unconstrained NNs on ice flux interpolation over Antarctica's Byrd Glacier.

## Background & Motivation

### State of the Field

**Background**: The Antarctic Ice Sheet (AIS) stores ice equivalent to approximately 58 meters of global sea-level rise; accurate modeling of ice flow is essential for projecting sea-level change.

### Limitations of Prior Work

**Limitations of Prior Work**: Extreme environmental conditions yield sparse and noisy ice thickness observations; unconstrained interpolation produces unphysical flux divergence, degrading downstream numerical ice-sheet models.

### Root Cause

**Key Challenge**: Existing PINNs incorporate mass conservation as a soft penalty in the loss function, but cannot guarantee physical consistency (MAD > 0) and exhibit poor generalization.

### Starting Point

**Key Insight**: The core question is: **can mass conservation be enforced as a hard architectural constraint rather than relying on a soft loss-function penalty?**

### Additional Notes

**Additional Notes**: By 2100, sea-level rise is projected to impose annual flood damage costs of approximately 2% of GDP, with roughly 360 million people living in flood-prone areas.

## Method

### Overall Architecture

The dfNN exploits a property from vector calculus: the symplectic gradient of a scalar stream function is inherently divergence-free. The network predicts a scalar stream function $\psi(x,y)$, and the divergence-free vector field is obtained via automatic differentiation as $(\partial\psi/\partial y, -\partial\psi/\partial x)$. The procedure is mesh-free, fully differentiable, and directly implementable in PyTorch. Ice flux is defined as ice thickness times velocity $\mathbf{v} = h \cdot \mathbf{s}$, which satisfies the divergence-free condition under the steady-state incompressible assumption.

### Key Designs

1. **Divergence-Free Architecture (dfNN)**:
    - **Function**: Guarantees architecturally that the output vector field has zero divergence everywhere.
    - **Mechanism**: A feedforward NN takes spatial coordinates $(x,y)$ as input, outputs a scalar stream function $\psi$, and vector components are obtained via the symplectic operator.
    - **Design Motivation**: Exploits the mathematical identity $\nabla \cdot (\partial\psi/\partial y, -\partial\psi/\partial x) = 0$.

2. **Directional Guidance Strategy**:
    - **Function**: Constrains predictions using continent-wide InSAR satellite ice surface velocity directions.
    - **Mechanism**: A cosine similarity loss $\mathcal{L}_{dir} = 1 - \cos(\hat{\mathbf{s}}, \hat{\mathbf{v}})$ aligns predicted and observed directions.
    - **Design Motivation**: Satellite data are spatially dense but provide direction only (no magnitude), offering additional constraints in unobserved regions.

### Loss & Training

$$\mathcal{L} = (1-w_{dir}) \cdot \mathcal{L}_{MSE} + w_{dir} \cdot \mathcal{L}_{dir}$$

The AdamW optimizer (with weight decay) is used. Experiments are conducted over a 200×200 km region of Byrd Glacier, with a 15 km checkerboard pattern for train/test splitting.

## Key Experimental Results

### Main Results

| Model | RMSE ↓ | MAE ↓ | MAD ↓ |
|-------|--------|-------|-------|
| NN | 0.470 ± 0.02 | 0.239 ± 0.01 | 1.269 ± 0.08 |
| PINN | 0.466 ± 0.01 | 0.236 ± 0.01 | 0.471 ± 0.06 |
| dfNN | 0.391 ± 0.03 | 0.199 ± 0.01 | **0.000** |
| dfNN + dir | **0.385 ± 0.02** | **0.193 ± 0.01** | **0.000** |

### Ablation Study

- **Directional guidance**: Improves all models; the relative gain is larger for PINNs and NNs.
- **Auxiliary surface elevation**: Degrades performance, introducing more noise than useful signal.
- **Surface gradient**: Further worsens performance.

### Key Findings

- The dfNN consistently outperforms PINNs and NNs on all metrics, with MAD identically zero (exact mass conservation).
- PINNs substantially reduce MAD but do not converge to zero, confirming that soft constraints cannot guarantee physical consistency.
- Parsimonious models generalize best, supporting Occam's Razor.

## Highlights & Insights

- This work provides a clear demonstration of the "hard constraint vs. soft constraint" paradigm: physical constraints are embedded in the architecture rather than the loss function.
- The directional guidance strategy cleverly exploits dense but incomplete satellite data (direction without magnitude), and is applicable to a variety of geophysical fluid flows.
- The experimental design is rigorous: real Antarctic data, checkerboard train/test splitting, and results averaged over 5 independent runs with standard deviations reported.
- The code is open-source and experiments are reproducible; CodeCarbon is used to monitor carbon emissions, reflecting responsible research practices.

## Limitations & Future Work

- Validation is limited to the 2D steady-state assumption; temporal evolution is not considered.
- Byrd Glacier offers limited representativeness; validation across additional regions is needed.
- The dfNN is restricted to divergence-free fields; extension to more complex flow regimes is required.
- Robustness analysis under varying noise levels and data sparsity is absent.

## Related Work & Insights

- The proposed approach belongs to the same family of hard-constrained physics methods as Hamiltonian NNs and Neural Conservation Laws.
- The method generalizes to other divergence-free vector fields, such as ocean circulation and groundwater flow.
- The directional guidance strategy may inspire other settings that exploit partial observations.
- The underlying technique traces back to Kuroe et al. (1998) and was later rediscovered in the context of Hamiltonian NNs and Neural Conservation Laws.

## Rating

- ⭐⭐⭐⭐ — The method is elegant and effective with clear physical motivation; however, problem scale and generalizability require further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Causal Foundation Models: Disentangling Physics from Instrument Properties](../../ICML2025/remote_sensing/causal_foundation_models_disentangling_physics_from_instrument_properties.md)
- [\[ICML 2026\] The Perception-Physics Paradox: Probing Scientific Alignment with TC-Bench](../../ICML2026/remote_sensing/the_perception-physics_paradox_probing_scientific_alignment_with_tc-bench.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](../../CVPR2026/remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization.md)

</div>

<!-- RELATED:END -->
