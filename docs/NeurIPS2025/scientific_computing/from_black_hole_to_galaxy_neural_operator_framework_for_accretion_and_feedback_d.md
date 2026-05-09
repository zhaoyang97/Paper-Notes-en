---
title: >-
  [Paper Note] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics
description: >-
  [NeurIPS 2025][Scientific Computing][Neural Operator] A Neural Operator-based "sub-grid black hole" model is proposed to learn the small-scale (GR)MHD time-evolution operator $u_t \to u_{t+\Delta T}$, replacing hand-crafted closure rules embedded in a multi-level direct numerical simulation framework. This work achieves, for the first time, the capture of intrinsic variability in accretion-driven feedback, with a speedup of $\sim 10^5\times$.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - Neural Operator
  - Black Hole Accretion
  - Multi-scale Simulation
  - Sub-grid Model
  - GRMHD
date: 2026-05-08
content_hash: 037ae18c39dd1ba7
---

# From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2512.01576](https://arxiv.org/abs/2512.01576)
**Code**: None
**Area**: Scientific Computing
**Keywords**: Neural Operator, Black Hole Accretion, Multi-scale Simulation, Sub-grid Model, GRMHD

## TL;DR
A Neural Operator-based "sub-grid black hole" model is proposed to learn the small-scale (GR)MHD time-evolution operator $u_t \to u_{t+\Delta T}$, replacing hand-crafted closure rules embedded in a multi-level direct numerical simulation framework. This work achieves, for the first time, the capture of intrinsic variability in accretion-driven feedback, with a speedup of $\sim 10^5\times$.

## Background & Motivation
**State of the Field**: Supermassive black holes (SMBHs) and their host galaxies co-evolve through accretion–feedback (feeding–feedback) cycles, spanning 9 orders of magnitude in scale from milliparsecs (black hole event horizon) to megaparsecs (intergalactic medium).

**Limitations of Prior Work**: End-to-end first-principles simulations are computationally infeasible — accurately resolving the accretion flow requires time steps on the order of the gravitational radius, while capturing galaxy-scale feedback demands spatiotemporal scales $\sim 10^9$ times larger. Existing approaches (direct simulation / nested grids / multi-zone) are either prohibitively expensive or rely on static sub-grid schemes or theoretical prescriptions that cannot capture temporal variability.

**Root Cause**: Small-scale accretion dynamics are chaotic and time-variable, yet existing sub-grid schemes (e.g., FIRE, IllustrisTNG in cosmological simulations) employ fixed analytic prescriptions and cannot dynamically respond to large-scale environmental changes. Unfaithful inner boundary settings (e.g., relativistic jets that cannot switch on/off across boundaries) inject erroneous information into the active simulation domain.

**Paper Goals**: To replace hand-crafted sub-grid closures with a data-driven approach while maintaining long-term stable integration and physical consistency.

**Starting Point**: Sub-grid modeling is reformulated as an operator learning problem — learning small-scale dynamics to provide dynamically updated boundary conditions.

**Core Idea**: A Local Neural Operator is trained to learn the fine-scale (GR)MHD time-evolution semigroup $u_t \to u_{t+\Delta T}$, and is embedded in a multi-level framework to enable bidirectional coupling.

## Method

### Overall Architecture
A two-level Neural Operator–DNS framework:
1. **Domain Decomposition**: Coarse-level domain $(n_L L)^3$ ($n_L=6$) and fine-level domain $L^3$.
2. **Domain Treatment**: The coarse level is evolved by DNS; the fine level (unresolvable) is replaced by the sub-grid model.
3. **Training**: DNS simulations are first run on the fine-level domain for $t \in [0,T]$, producing $N_{\text{data}}=300$ snapshots as the training set.
4. **Inference**: The trained Neural Operator is autoregressively unrolled to simulate fine-scale quasi-steady states, supplying boundary conditions and fluxes to the coarse level.
5. **Coupling**: The coarse-level simulation runs for $t \in [0, NT]$ ($N \sim 10^2$), with the fine-level NO providing hydrodynamic variables and magnetic fields at the inner boundary.

Direct simulation of both fine and coarse levels would reduce the time step by at least $n_L^2$. NO inference requires only a few GPU-seconds versus 400 GPU-hours for 50 direct simulation steps, yielding a $\sim 10^5\times$ speedup.

### Key Designs
1. **Local Neural Operator (LocalNO)**: Employs a 3D DISCO (equidistant discrete-continuous convolution) architecture, taking 8 physical channels $(\rho, P, v_x, v_y, v_z, B_x, B_y, B_z)$ plus 8 shell positional encoding channels as input and outputting 8 physical channels. It learns the mapping $u_t \to u_{t+\Delta T}$.
2. **Magnitude Normalization**: Applies (signed) log transformation + robust z-scoring + soft clipping ($\gamma=6$) to quantities with large dynamic ranges such as density, energy, and magnetic field, compressing the dynamic range.
3. **Radial Scaling Residualization**: Exploits the radial scaling law of density/energy $\log u(\vec{x}) \approx -k|\vec{x}| + b$; the model predicts residuals relative to this baseline rather than absolute values, reducing the dynamic range to be learned.
4. **Shell Positional Encoding**: The domain is divided into 8 radial shells by logarithmic distance, encoded as one-hot vectors ($c \in \{0,1\}^8$) serving as additional input channels. This outperforms standard Fourier positional encoding and allows the model to focus on the central region near the black hole.
5. **Bidirectional Multi-scale Coupling**: The NO rollout provides inner boundary hydrodynamic variables that directly overwrite the coarse-level inner boundary; magnetic fields are handled separately via a constraint transport algorithm to maintain the divergence-free condition.

### Loss & Training
The total loss comprises multiple components:
$$\mathcal{L} = \mathcal{L}_{\mathbf{B}} + \mathcal{L}_{\mathbf{v}} + \mathcal{L}_\rho + \mathcal{L}_e + \lambda_{H^1}\mathcal{L}_{H^1} + \mathcal{L}_{\text{vel,ROI}} + \mathcal{L}_{\text{dissip}} + \mathcal{L}_{\text{env}} + \mathcal{L}_{\text{constr}}$$

- **Component-weighted $L^2$**: Magnetic field weight $\lambda_B=1.2$, velocity weight $\lambda_{\text{vel}}=1.0$.
- **$H^1$ Matching**: $\lambda_{H^1}=0.05$, constraining the gradient field.
- **Velocity ROI**: Applies $8\times$ weight to high-velocity regions (top 20%), with linear ramp over 375 epochs.
- **Dissipation Regularization**: Prevents unphysical amplification of the $L^2$ norm, $\alpha=5\times10^{-4}$.
- **Residual Envelope**: Density/energy predictions are constrained within $\pm 1.5$ of the radial baseline.
- **Physical Constraint Penalty**: Data-driven quantile upper and lower bounds on density/energy.

Training configuration: 1200 epochs, batch size 4 (effective 16), Adam lr=$10^{-3}$, cosine schedule, single RTX 4090, training time only 10 GPU-hours.

## Key Experimental Results

### Main Results: MHD and GRMHD Quality Validation

**MHD (Magnetized Bondi Accretion)**: $64^3$ grid, 8 physical fields, 300 snapshots for training.

| Configuration | Avg Error (%) | $B_x$ (%) | $B_y$ (%) | $B_z$ (%) | $\rho$ (%) | $e$ (%) | $v_x$ (%) | $v_y$ (%) | $v_z$ (%) |
|---|---|---|---|---|---|---|---|---|---|
| **Ours (LocalNO)** | **14.02** | 16.71 | 17.01 | 14.73 | 10.98 | 11.25 | 13.47 | 14.04 | 13.94 |
| CNN backbone | 19.09 | 23.85 | 23.91 | 21.44 | 15.21 | 14.87 | 17.46 | 17.80 | 18.17 |
| Plain $L^2$ | 13.69 | 16.76 | 16.98 | 14.67 | 9.55 | 9.89 | 13.45 | 14.06 | 14.16 |

**GRMHD (Fishbone–Moncrief torus, spin $a=0.9$)**: Qualitative validation confirms that jet structure and central torus morphology are well preserved.

### Ablation Study

| Ablation Configuration | Avg Error (%) | Long-term Rollout Stability |
|---|---|---|
| Ours (full) | 14.02 | ✅ Stable at both 50 and 100 steps |
| Plain $L^2$ | 13.69 | ❌ Dynamics near black hole lost |
| No PE/Radial Shell | 13.93 | ❌ Degraded accuracy in central region |
| PE (Fourier) | 13.87 | ❌ Anisotropic diffusion |
| No radial/constraint | 14.17 | ❌ Density/energy distortion |
| CNN backbone | 19.09 | ❌ Magnetic field ripple artifacts |

### Key Findings
- **The Plain $L^2$ Paradox**: Achieves the lowest single-step validation error (13.69%) but completely loses near-black-hole dynamics during long-term rollout. The region near the singularity occupies a negligible fraction of the total volume and is thus ignored by global MSE, yet is physically critical to the entire system.
- **CNN Failure Mode**: Non-physical ripples appear in the magnetic field, and the velocity field fails to match the torus structure.
- **Analytic Sub-grid Scheme Failure**: Cannot capture the jet, demonstrating the necessity of data-driven dynamic closure.
- **Neural Operator vs. CNN**: Architecture advantage is significant; CNN average error is 36% higher (19.09% vs. 14.02%).
- **Radial Physical Priors Are Critical**: Shell encoding and radial scaling are key to long-term stability, rather than sources of single-step accuracy gains.
- **Observable Matching**: Spherically averaged radial profiles of density $\rho$, temperature $T$, and mass accretion rate $\dot{M}$ from the NO rollout agree well with DNS ground truth.

## Highlights & Insights
- **Operator Learning Redefines Sub-grid Modeling**: Rather than using hand-crafted closure formulas, the small-scale dynamics operator is learned directly — a paradigm shift in astrophysical computation.
- **$10^5\times$ Speedup**: NO inference requires only GPU-seconds versus 400 GPU-hours for DNS, making previously infeasible multi-scale coupled simulations tractable.
- **First Capture of Intrinsic Variability**: Data-driven closure naturally carries temporal variability, which analytic prescriptions cannot achieve.
- **Balance Between Physical Priors and Data-Driven Learning**: Physical priors such as radial scaling laws, shell encoding, and dissipation regularization are essential for long-term stability; purely data-driven approaches (plain $L^2$) fail instead.
- **General Framework**: Applicable to any system with a central accreting body (SMBH, neutron stars, etc.).

## Limitations & Future Work
- The current work demonstrates only a two-level (fine + coarse) setup; a complete multi-level cyclic-zoom/multi-zone framework has not been implemented.
- Quantitative long-term rollout error metrics are lacking, as ground truth is computationally inaccessible.
- Training data originate from a single simulation setup; generalization to different black hole spins, accretion rates, etc. has not been thoroughly validated.
- Magnetic field coupling via EMF reconstruction preserves the divergence-free condition but may introduce numerical diffusion.
- MHD experiments exclude cooling/heating terms, representing a gap relative to realistic galactic environments.
- Integration into actual cosmological-scale simulations has not been addressed.

## Related Work & Insights
- **vs. Multi-zone (Cho et al.)**: Multi-zone iteratively refines and coarsens to ensure consistency but still employs fixed inner boundaries; this work uses NO to provide dynamic inner boundaries.
- **vs. Cyclic-zoom (Guo et al.)**: A similar iterative framework, but inner boundaries are derived from theoretical assumptions; replacing them with NO captures time variability.
- **vs. Duarte et al. (2D black hole accretion surrogate)**: Limited to 2D Newtonian fluid steady-state accretion; this work addresses 3D (GR)MHD non-steady-state dynamics.
- **vs. FNO/DeepONet**: This work employs LocalNO to handle local structures, combined with astrophysics-specific training techniques.
- **Insight**: Decomposing any multi-physics problem with scale separation into "fine-scale operator learning + coarse-scale DNS" is a broadly applicable strategy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First work to embed a Neural Operator as a sub-grid black hole model within a multi-scale astrophysical simulation; pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — MHD + GRMHD experiments with systematic ablations, but lacks quantitative long-term metrics and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; method–physics coupling is described in detail; appendix is comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Has the potential to revolutionize black hole feedback modeling in cosmological simulations, with implications for mainstream frameworks such as FIRE and IllustrisTNG.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/scientific_computing/jpeg_processing_neural_operator_for_backward-compatible_coding.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[NeurIPS 2025\] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE–Normalizing Flows](one-shot_transfer_learning_for_nonlinear_pdes_with_perturbative_pinns.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/scientific_computing/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

<!-- RELATED:END -->
