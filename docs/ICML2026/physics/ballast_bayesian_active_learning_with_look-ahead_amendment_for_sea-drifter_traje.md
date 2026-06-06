---
title: >-
  [Paper Note] BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields
description: >-
  [ICML2026][Physics & Scientific Computing][Active learning] The BALLAST algorithm is proposed to amend active learning utility estimation by sampling vector fields from the GP posterior and simulating the future trajecto…
tags:
  - "ICML2026"
  - "Physics & Scientific Computing"
  - "Active learning"
  - "Gaussian processes"
  - "sea drifters"
  - "spatio-temporal vector fields"
  - "Bayesian experimental design"
date: 2026-05-08
content_hash: 0c82617182ee58fb
---

# BALLAST: Bayesian Active Learning with Look-ahead Amendment for Sea-drifter Trajectories under Spatio-Temporal Vector Fields

**Conference**: ICML2026  
**arXiv**: [2509.26005](https://arxiv.org/abs/2509.26005)  
**Code**: https://github.com/ShuSheng3927/BALLAST  
**Area**: scientific_computing  
**Keywords**: Active learning, Gaussian processes, sea drifters, spatio-temporal vector fields, Bayesian experimental design  

## TL;DR

The BALLAST algorithm is proposed to amend active learning utility estimation by sampling vector fields from the GP posterior and simulating the future trajectories of Lagrangian observers. Simultaneously, the VaSE inference method is developed to accelerate GP posterior sampling efficiency by thousands of times, achieving deployment cost savings of approximately 16%-22% on synthetic and high-fidelity ocean flow fields.

## Background & Motivation

**Background**: Understanding and predicting ocean flow fields is crucial for tracking heat, nutrients, and pollutants in the ocean. Free-floating sea drifters are widely used for their ability to simultaneously collect spatio-temporal flow field properties. Once deployed, drifters are advected by the underlying vector field, performing velocity measurements at different locations and times; they are categorized as Lagrangian observers.

**Limitations of Prior Work**: Current drifter placement strategies either employ standard "space-filling" designs (such as Sobol sequences) or rely on relatively arbitrary expert opinions. While some work has proposed handcrafted design criteria based on travel distance and placement spacing, no formal active learning framework exists to guide the deployment of Lagrangian observers.

**Key Challenge**: Standard active learning methods (such as Expected Information Gain, EIG) only consider the information gain at the initial observation position when estimating the utility of a candidate placement. They completely ignore subsequent observations collected as the drifter is continuously advected by the flow field. Consequently, EIG strategies tend to place observers near boundaries—where initial information gain is high, but the observer quickly exits the study area, resulting in low actual utility. Experiments show that EIG consistently performs worse than a uniform random strategy.

**Goal**: Design an active learning strategy that correctly evaluates the full life-cycle information gain of Lagrangian observers and resolve the computational bottleneck of GP posterior sampling that it entails.

**Key Insight**: Utilize GP posterior sampling to simulate the future trajectories of drifters within hypothesized vector fields, incorporating the information gain of all subsequent observations along the trajectory into the utility calculation.

**Core Idea**: Amend the utility function (look-ahead amendment) by Monte Carlo sampling the posterior vector field and simulating observer trajectories, while proposing the VaSE method to bypass the computational bottleneck of SPDE-GP for non-gridded observations.

## Method

### Overall Architecture

BALLAST is a sequential experimental design framework: At each decision time $t_n$, given existing observations $\mathcal{D}_n$, $J$ vector field samples are drawn from the GP posterior. For each candidate placement position $\bm{s}$, its complete trajectory is simulated under each sampled field until the termination time $T$. The optimal placement position is selected by aggregating the information gain across all samples. Inputs are a spatial grid $R$, time range $[0,T]$, and number of drifters $M$; the output is $M$ sequential optimal placement positions.

### Key Designs

1.  **Trajectory-Aware Utility Amendment (BALLAST Amendment)**:
    - **Function**: Incorporates the future trajectory of Lagrangian observers into utility calculations, replacing the standard EIG strategy that considers only the initial position.
    - **Mechanism**: For any utility function $U$, the BALLAST-amended acquisition function is $\bm{s}_n^* = \arg\max_{\bm{s} \in R} \mathbb{E}_{F \sim p(f|\mathcal{D}_n)}[\mathbb{E}[U(P_F^T(\bm{s}, t_n))]]$, where $P_F^T(\bm{s}, t_n)$ is the projected trajectory of the observer starting from position $\bm{s}$ at time $t_n$ until time $T$ under the sampled vector field $F$. The outer expectation is approximated via Monte Carlo with $J=20$ posterior samples; each trajectory is simulated via numerical integration using the Euler method with step size $\delta_t$.
    - **Design Motivation**: Standard EIG ignores subsequent observations, leading to sub-optimal decisions (placing observers near boundaries where they quickly exit). By simulating the full trajectory, BALLAST correctly evaluates the long-term information contribution of each candidate position.

2.  **Vanilla SPDE Exchange (VaSE) Inference Method**:
    - **Function**: Efficiently samples vector fields from the spatio-temporal GP posterior, solving the computational bottleneck of BALLAST.
    - **Mechanism**: Combines standard GP regression with SPDE methods—first using an augmented GP $\bm{f} = [f, \partial_t f]^T$ to generate SPDE initial conditions at decision time $t_n$ via standard GP regression, then propagating along the temporal direction to termination time $T$ using a Kalman filter/RTS smoother. Standard GP sampling costs $O(N_{\text{pred,s}}^3 N_{\text{pred,t}}^3)$, SPDE costs $O((N_{\text{obs}}+N_{\text{pred,s}})^3 N_{\text{obs,t}})$, while VaSE costs only $O(N_{\text{obs}}^3 + N_{\text{pred,s}}^2 N_{\text{pred,t}})$.
    - **Design Motivation**: SPDE methods experience a surge in cost when observation positions (non-gridded Lagrangian data) and prediction positions (regular grids) do not overlap. VaSE bypasses this via standard GP regression.

3.  **Spatio-Temporal Helmholtz GP Surrogate**:
    - **Function**: Provides a probabilistic surrogate model with physical priors for time-varying ocean vector fields.
    - **Mechanism**: Construct a vector-valued kernel using Helmholtz decomposition $k_{\text{tHelm}}((\bm{s},t),(\bm{s}',t')) = k_{\text{Helm}}(\bm{s},\bm{s}') k_{\text{time}}(t,t')$. The spatial part is based on linear differential operators of potential and stream function kernels, while the temporal part adopts a Matérn 3/2 kernel (consistent with the empirical $\nu \approx 2$ in oceanography).
    - **Design Motivation**: The separable spatio-temporal kernel structure enables SPDE propagation in VaSE, while the Helmholtz decomposition encodes physical constraints of fluid mechanics.

## Key Experimental Results

### Main Results

Comparison of six strategies: Uniform Random (UNIF), Sobol Sequence (SOBOL), Distance-Separation Heuristic (DIST-SEP), Expected Information Gain (EIG), BALLAST-opt (optimized hyperparameters), BALLAST-true (ground-truth hyperparameters).

| Experimental Setup | Metric | BALLAST-true | BALLAST-opt | UNIF | EIG | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Temporal Helmholtz (Synthetic) | Deployment Cost Saving | ~16% | ~16% | baseline | Worse than UNIF | Saves ~3 drifters |
| SUNTANS (High-fidelity Fluid Sim) | Deployment Cost Saving | ~22% | ~22% | baseline | Better than UNIF | Saves ~2 drifters |

### Efficiency Comparison

| Inference Method | Cost Magnitude (Typical Setup) | Sampling Time per Sample | Speedup |
| :--- | :--- | :--- | :--- |
| Standard GP | $10^{17}$ | Infeasible | — |
| SPDE-GP | $10^{11}$ | ~4.5 min | 1× |
| VaSE (Ours) | $10^{8}$ | ~3.8 s | ~70× |

### Ablation Study

| Posterior Sample Count $J$ | Reaches 1% Utility Gap | Decision Time ($J=20$) | Note |
| :--- | :--- | :--- | :--- |
| $J < 20$ | ✓ Consistently met | < 3 min | Converges before $J=20$ across three decision times $t=3,5,7$ |
| $J = 200$ (Ref) | — | — | Used as a baseline approximation for true expected utility |

## Highlights & Insights

- The BALLAST method is generalizable, applicable not only to ocean drifters but also to Lagrangian observation equipment advected by the environment, such as animal tracking collars and weather balloons.
- The "counter-intuitive" finding that standard EIG consistently performs worse than uniform strategies in Lagrangian observation scenarios reveals the fundamental flaw of ignoring observer dynamics.
- The VaSE method is significant independent of BALLAST, usable in any scenario requiring efficient sampling from spatio-temporal GPs with non-gridded observations.

## Limitations & Future Work

- Currently assumes predefined decision times; the deployment timing itself is not optimized.
- Hyperparameter optimization of the GP surrogate might not be robust in high-dimensional or complex flow fields.
- Amortized acquisition optimization could be considered to achieve faster deployment decisions.
- Validated only on 2D spatial flow fields; 3D ocean flow fields have not yet been tested.

## Related Work & Insights

- Berlinghieri et al. (2023) proposed Helmholtz GP kernels for ocean flow field modeling.
- Chen et al. (2024b) proposed handcrafted placement criteria based on a Lagrangian data assimilation framework.
- The SPDE-GP framework by Sarkka et al. (2013) is one of the base components of VaSE.
- Predictive Entropy Search by Hernández-Lobato et al. (2014), utilizing mutual information symmetry, inspired the information gain reconstruction in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to formally introduce active learning to Lagrangian observer placement; the VaSE inference method provides an independent contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual validation via synthetic and high-fidelity simulations, including ablations and comparison with six baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem motivation, rigorous theoretical and algorithmic development, intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Direct application value for actual ocean science deployments; the method is extensible to other Lagrangian observation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)
- [\[ICML 2026\] ANTIC: Adaptive Neural Temporal In-situ Compressor](antic_adaptive_neural_temporal_in-situ_compressor.md)
- [\[ICML 2026\] Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation](distribution_transformers_fast_approximate_bayesian_inference_with_on-the-fly_pr.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](../../NeurIPS2025/physics/vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [\[NeurIPS 2025\] Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios](../../NeurIPS2025/physics/bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)

</div>

<!-- RELATED:END -->
