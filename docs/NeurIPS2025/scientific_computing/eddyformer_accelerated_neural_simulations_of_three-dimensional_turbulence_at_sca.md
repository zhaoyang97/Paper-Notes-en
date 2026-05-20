---
title: >-
  [Paper Note] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale
description: >-
  [NeurIPS 2025][Scientific Computing][Turbulence Simulation] EddyFormer is a Transformer architecture based on the Spectral Element Method (SEM) that decomposes the flow field into two parallel streams — LES (large-scale)…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "Turbulence Simulation"
  - "Spectral Element Method"
  - "Transformer"
  - "LES"
  - "Neural PDE Solver"
date: 2026-05-08
content_hash: 4ada556585ae16a0
---

# EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale

**Conference**: NeurIPS 2025
**arXiv**: [2510.24173](https://arxiv.org/abs/2510.24173)  
**Code**: [https://github.com/ASK-Berkeley/EddyFormer](https://github.com/ASK-Berkeley/EddyFormer)  
**Area**: Scientific Computing / Fluid Mechanics
**Keywords**: Turbulence Simulation, Spectral Element Method, Transformer, LES, Neural PDE Solver

## TL;DR
EddyFormer is a Transformer architecture based on the Spectral Element Method (SEM) that decomposes the flow field into two parallel streams — LES (large-scale) and SGS (small-scale) — achieving DNS-level accuracy on 3D turbulence at $256^3$ resolution with a 30× speedup, while generalizing well to unseen domains 4× larger.

## Background & Motivation

### State of the Field

**Background**: Turbulence simulation requires $Re^{9/4}$ resolution for DNS, making it prohibitively expensive. LES resolves only large-scale structures and approximates small scales via theoretical models, but struggles with wall-bounded and anisotropic turbulence.

**Limitations of Prior Work**: (a) Neural operators such as FNO are difficult to scale to large turbulence problems; (b) the quadratic complexity of Transformers grows with mesh resolution; (c) most ML methods are validated only on 2D, small-scale flows.

**Key Challenge**: How can one preserve the high accuracy of spectral methods while efficiently capturing multi-scale interactions via attention mechanisms?

**Key Insight**: SEM is adopted as the tokenization strategy: coarse elements serve as tokens, with spectral expansions within each element. The sequence length equals only the number of elements $N^3$, far smaller than the total number of modes $N^3 M^3$.

**Core Idea**: SEM-based tokenization combined with a dual-stream LES/SGS architecture that explicitly encodes the multi-scale nature of turbulence into the model design.

### Mechanism

**Goal**: ### Overall Architecture
EddyFormer interpolates PDE initial conditions using SEM and splits them into a LES stream (global large-scale) and an SGS stream (local small-scale) processed in parallel, yielding the output $\mathbf{u} = \mathbf{u}_{LES} + \mathbf{u}_{SGS}$.

## Method

### Overall Architecture
EddyFormer interpolates PDE initial conditions using SEM and splits them into a LES stream (global large-scale) and an SGS stream (local small-scale) processed in parallel, yielding the output $\mathbf{u} = \mathbf{u}_{LES} + \mathbf{u}_{SGS}$.

### Key Designs

1. **SEM Tokenization**:

    - The domain is divided into $H = N^3$ coarse elements, each expanded with $M^3$-order spectral basis functions.
    - Elements serve as tokens; spectral expansions within each element serve as token features. The sequence length is only $N^3$.

2. **SGS Stream (small-scale local dynamics)**:

    - SEM-based convolution (SEMConv) models local eddy interactions.
    - Based on the Kolmogorov hypothesis, small-scale dynamics exhibit universality.

3. **LES Stream (large-scale global dynamics)**:

    - SEM-based self-attention (SEMAttn) captures global long-range dependencies.
    - Rotary Position Encoding preserves translation invariance.

### Loss & Training
- Time-averaged single-step relative error.

## Key Experimental Results

### Main Results — 3D Homogeneous Isotropic Turbulence

| Model | Parameters | Single-step Error |
|-------|------------|-------------------|
| EddyFormer | **2.3M** | **Lowest** |
| FNO | 17.6M | Medium |

### Speed Comparison

| Method | $256^3$ Simulation Time | L2 Error |
|--------|------------------------|----------|
| DNS ($256^3$) | 152 sec | 16.3% |
| **EddyFormer** | **4.86 sec** | 18.2% |

### Key Findings
- 30× speedup over DNS with comparable accuracy.
- Domain generalization: physical invariant metrics remain accurate on unseen domains 4× larger.
- Resolves turbulence cases on The Well benchmark where other ML models fail to converge.

## Highlights & Insights
- **Physics-inspired architecture**: the LES/SGS dual-stream design directly corresponds to the multi-scale structure of turbulence.
- **Elegant SEM tokenization**: using coarse spectral elements as tokens effectively truncates the attention input size.
- **Domain generalization** is achieved via attention masking.

## Limitations & Future Work
- Only isotropic turbulence is tested; anisotropic and wall-bounded turbulence remain unverified.
- Predictions rely on fixed time steps.

## Related Work & Insights
- **vs. FNO**: FNO employs a global Fourier kernel but is parameter-inefficient; EddyFormer separates global and local dynamics through SEM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ SEM tokenization combined with the LES/SGS dual-stream architecture is highly novel and strongly physics-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3D turbulence, 2D domain generalization, and The Well benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Physical background and method description are both detailed and clear.
- Value: ⭐⭐⭐⭐⭐ Achieving DNS-level accuracy on 3D turbulence with a 30× speedup offers high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[NeurIPS 2025\] Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)

</div>

<!-- RELATED:END -->
