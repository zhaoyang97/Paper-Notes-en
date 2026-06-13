---
title: >-
  [Paper Note] Graph-based Neural Space Weather Forecasting
description: >-
  [NEURIPS2025][Time Series][space weather] This paper proposes a graph neural network-based neural emulator for space weather, trained on Vlasiator hybrid-Vlasov simulation data…
tags:
  - "NEURIPS2025"
  - "Time Series"
  - "space weather"
  - "graph neural network"
  - "probabilistic forecasting"
  - "hybrid-Vlasov"
  - "ensemble forecasting"
date: 2026-05-08
content_hash: 2c59550013e89384
---

# Graph-based Neural Space Weather Forecasting

**Conference**: NEURIPS2025
**arXiv**: [2509.19605](https://arxiv.org/abs/2509.19605)  
**Code**: [fmihpc/spacecast](https://github.com/fmihpc/spacecast)  
**Area**: Image Generation
**Keywords**: space weather, graph neural network, probabilistic forecasting, hybrid-Vlasov, ensemble forecasting

## TL;DR

This paper proposes a graph neural network-based neural emulator for space weather, trained on Vlasiator hybrid-Vlasov simulation data, enabling both deterministic and probabilistic autoregressive forecasting of near-Earth space conditions. The emulator achieves over 100× speedup relative to the original simulator and quantifies forecast uncertainty through latent-variable ensemble generation.

## Background & Motivation

Space weather describes near-Earth space conditions driven by the solar wind and poses severe threats to modern infrastructure: geomagnetically induced currents can damage power grids, satellite operations are adversely affected, and electromagnetic communications may fail. Current operational forecasting systems rely on global magnetohydrodynamic (MHD) models (e.g., BATS-R-US), driven by real-time solar wind data from L1 Lagrange point satellites. This paradigm faces two major challenges:

1. **Insufficient physical fidelity**: MHD models approximate plasma as a fluid, neglecting ion kinetic processes. Higher-fidelity hybrid-Vlasov models (e.g., Vlasiator) can capture these processes, but their computational cost precludes real-time operational use.
2. **Lack of uncertainty quantification**: Most operational forecasts are deterministic single-point predictions, lacking critical uncertainty information. Although the need for ensemble forecasting has been recognized, existing approaches either perturb solar wind inputs to physical models or apply machine learning as post-processing on deterministic outputs.

Inspired by ML breakthroughs in atmospheric weather forecasting (GraphCast, Pangu-Weather, etc.), this work transfers graph-based limited-area modeling to the space weather domain, aiming to address both challenges simultaneously.

## Core Problem

How can machine learning make computationally expensive hybrid-Vlasov simulations (Vlasiator) operationally viable? Specifically:
- How to achieve fast deterministic forecasting while preserving physical fidelity?
- How to add uncertainty quantification capabilities—currently absent—to space weather forecasting systems?

## Method

### Problem Formulation

Space weather forecasting is formulated as follows: given two consecutive magnetospheric states $X^{-1:0} = (X^{-1}, X^{0})$ (capturing first-order dynamics), predict the future trajectory $X^{1:T} = (X^1, \dots, X^T)$. Each state $X^t \in \mathbb{R}^{N \times d_x}$ represents $d_x$ physical variables at $N$ grid locations. The predicted variables include: magnetic field $(B_x, B_y, B_z)$, electric field $(E_x, E_y, E_z)$, velocity field $(v_x, v_y, v_z)$, particle number density $\rho$, plasma temperature $T$, and plasma pressure $P$, totaling 12 physical quantities.

### Data Source: Vlasiator Hybrid-Vlasov Simulation

Vlasiator is run in a 2D-3V configuration (two spatial dimensions, three velocity dimensions):
- **Time step**: $\Delta t = 1\,\text{s}$, spatial resolution 600 km
- **Simulation domain**: Sun–Earth meridional plane ($x$-$z$ plane in GSE coordinates), $x \in [-60R_E, +30R_E]$, $z \in \pm 30R_E$
- **Solar wind parameters**: density $\rho = 1/\text{cm}^3$, velocity $\mathbf{v} = (-750, 0, 0)\,\text{km/s}$, temperature $T = 0.5\,\text{MK}$
- **Interplanetary magnetic field**: southward $\mathbf{B} = (0, 0, -5)\,\text{nT}$, Alfvén Mach number $M_A = 6.9$
- **Inner boundary**: perfect conductor at $3.7\,R_E$ from Earth's center

### Graph Neural Network Architecture

An encode-process-decode architecture is adopted:
1. **Encoder**: maps grid inputs to a mesh representation
2. **Processor**: multiple GNN layers process latent representations
3. **Decoder**: maps processed data back to the original grid

Key design: the GNN predicts residual updates rather than directly predicting the next state ($\hat{X}^t = X^{t-1} + \tilde{g}(X^{t-2:t-1})$), reducing learning difficulty. Boundary enforcement: in the dayside open boundary region ($x = 27R_E$ to $30R_E$), ground-truth boundary data replaces model predictions after each step via a static binary mask.

### Three Mesh Architectures

| Architecture | Nodes | Edges | Characteristics |
|---|---|---|---|
| Simple | 58,592 | 465,584 | Single coarse mesh layer |
| Multiscale | 58,592 | 522,054 | Multi-level edges connected to the same node set; a single GNN layer processes all scales simultaneously |
| Hierarchical | 65,825 | 587,156 | Three independent graphs; information is passed between levels via inter-level graphs |

The Hierarchical architecture uses an interaction network to retain information and a propagation network to transfer new information, performing full up-down sweeps between coarse and fine levels.

### Deterministic Model: Graph-FM

Autoregressive mapping $\hat{X}^t = f(X^{t-2:t-1})$, trained with a weighted MSE loss:

$$\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{d_x} \omega_i \lambda_i (\hat{X}_{n,i} - X_{n,i})^2$$

where $\lambda_i$ is the inverse variance of temporal differences (normalizing contributions from different physical quantities) and $\omega_i = 1/d_x$.

### Probabilistic Model: Graph-EFM

A latent stochastic variable $Z^t$ is introduced to model forecast uncertainty, in a structure analogous to a conditional VAE:

$$p(X^t | X^{t-2:t-1}) = \int p(X^t | Z^t, X^{t-2:t-1}) \, p(Z^t | X^{t-2:t-1}) \, dZ^t$$

- **Latent Map**: a GNN maps inputs to the mean of an isotropic Gaussian distribution with fixed variance. The latent distribution is defined on nodes of the coarsest level $\mathcal{V}_L$, ensuring randomness is introduced in a low-dimensional space.
- **Predictor**: a deterministic mapping where $Z^t$ is injected at the top level and propagated downward, producing high-resolution, spatially coherent forecasts.

Training proceeds in three stages:
1. Autoencoder pre-training (200 epochs, $\lambda_{\text{KL}} = 0, \lambda_{\text{CRPS}} = 0$)
2. Variational training (300 epochs, $\lambda_{\text{KL}} = 1$)
3. CRPS fine-tuning (50 epochs, $\lambda_{\text{CRPS}} = 10^{-3}$), improving ensemble calibration

## Loss & Training

### Training Strategy

All models use the AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$, weight decay $0.01$) with an effective batch size of 8.

**Graph-FM (deterministic)**: standard two-stage training
- Stage 1: 300 epochs, learning rate $10^{-3}$
- Stage 2: 200 epochs, learning rate $10^{-4}$

**Graph-EFM (probabilistic)**: three-stage progressive training
1. **Autoencoder pre-training** (200 epochs, lr $10^{-3}$): $\lambda_{\text{KL}}=0, \lambda_{\text{CRPS}}=0$; encoder/decoder first learns to reconstruct fields from latent space
2. **Variational training** (300 epochs, lr $10^{-3}$): $\lambda_{\text{KL}}=1$; KL divergence regularization introduced, full variational objective trained
3. **CRPS fine-tuning** (50 epochs, lr $10^{-4}$): $\lambda_{\text{CRPS}}=10^{-3}$; CRPS loss applied to improve ensemble calibration

### Loss Functions

- **Deterministic**: weighted MSE using inverse temporal-difference variance $\lambda_i$ to normalize the dynamic range of different physical variables
- **Probabilistic**: variational objective (negative ELBO) = KL divergence regularization + reconstruction loss (negative log-likelihood), with CRPS loss added during fine-tuning
- **CRPS** uses an unbiased two-sample estimator: $\mathcal{L}_{\text{CRPS}} = \sum \frac{1}{2}(|\hat{X}-X| + |\check{X}-X| - |\hat{X}-\check{X}|)$, where $\hat{X}, \check{X}$ are two independent ensemble members

### Training Cost

Training uses 8 AMD MI250X GPUs:

| Model | Mesh Architecture | Training Time (GPU h) | Inference Time (s/step) |
|---|---|---|---|
| Graph-FM | Simple | 101 | 0.47 |
| Graph-FM | Multiscale | 102 | 0.48 |
| Graph-FM | Hierarchical | 108 | 0.52 |
| Graph-EFM | Simple | 119 | 3.20 |
| Graph-EFM | Multiscale | 122 | 3.31 |
| Graph-EFM | Hierarchical | 131 | 3.45 |

For reference: Vlasiator requires 4–5 minutes on 50 AMD EPYC 7H12 CPUs to simulate 1 second. The deterministic model achieves approximately **500×** speedup; the probabilistic model (ensemble size 5) achieves approximately **80×** speedup.

### Graph Construction Details

The data grid is $671 \times 1006\,(z, x)$, excluding 5,124 inner boundary nodes. Graphs are constructed via recursive downsampling: each coarse-level node is centered on a $3 \times 3$ square of fine-level nodes. The Grid-to-Mesh graph $\mathcal{G}_{G2M}$ has 1,411,687 edges; the Mesh-to-Grid graph $\mathcal{G}_{M2G}$ has 2,679,608 edges. All MLPs use a single hidden layer with Swish activation and Layer Normalization.

## Key Experimental Results

### Experimental Setup

#### Data Split
Due to the extremely high computational cost of Vlasiator, data are limited: training set 10 minutes, validation set 1 minute, test set 1 minute, split in causal (temporal) order to ensure the test set constitutes meaningful generalization to the future.

#### Evaluation Metrics

- **RMSE**: root mean square error between ensemble mean (or deterministic forecast) and ground truth, normalized to a dimensionless score (divided by variable standard deviation)
- **CRPS** (Continuous Ranked Probability Score): evaluates overall quality of probabilistic forecasts, measuring both accuracy and calibration, estimated using a finite-sample estimator
- **Spread**: root mean square deviation of ensemble members relative to the ensemble mean, quantifying forecast uncertainty. Under good calibration, Spread should equal RMSE

### Main Results

#### Deterministic Model Comparison
- **The Hierarchical architecture exhibits the slowest error accumulation**; Simple and Multiscale architectures show faster error growth.
- Probable reason: the hierarchical information propagation in Hierarchical (interaction network retaining + propagation network transferring) more effectively handles multi-scale spatial structure.

#### Probabilistic vs. Deterministic
- **Probabilistic model RMSE is lower than that of the deterministic model**, particularly at longer forecast lead times.
- Reason: averaging over multiple ensemble members mitigates the effect of trajectory drift, yielding more stable estimates.
- The three mesh architectures yield similar probabilistic model performance; differences are less pronounced than for deterministic models.

#### Calibration Analysis
- All probabilistic models exhibit **underdispersion**: ensemble Spread is consistently smaller than RMSE.
- Primary causes: limited training data and inherent properties of the variational objective (a known issue in limited-area modeling).
- CRPS fine-tuning is conservative ($\lambda_{\text{CRPS}}=10^{-3}$, only 50 epochs); increasing the weight could further improve calibration.

#### Per-Variable Analysis
- The model successfully captures complex magnetospheric dynamics; ensemble standard deviation is correctly localized at physically active regions such as the bow shock and magnetotail.
- **Artifact issue**: spurious structures not present in the training data appear in $B_y$ and $v_y$ components in the magnetotail northern lobe region, attributed to limited training samples.
- Detailed per-variable RMSE/CRPS/Spread analysis is provided for all 12 physical variables ($B_{x,y,z}$, $E_{x,y,z}$, $v_{x,y,z}$, $\rho$, $T$, $P$).

## Highlights & Insights

1. **Domain innovation**: This is the first work to transfer the graph neural network weather forecasting framework (GraphCast-style) to space weather, bridging the gap between atmospheric ML and space weather forecasting.
2. **Dual-mode forecasting**: The same framework supports both deterministic and probabilistic forecasting; the probabilistic variant realizes ensemble forecasting via conditional VAE + CRPS fine-tuning.
3. **Uncertainty quantification**: The system adds uncertainty quantification capabilities currently absent from most operational space weather systems, with uncertainty correctly localized at physically active regions (bow shock, magnetotail, etc.).
4. **Computational efficiency**: Over 100× speedup is achieved while maintaining the physical fidelity of hybrid-Vlasov simulations.
5. **Residual prediction + boundary enforcement**: Simple yet effective engineering designs that reduce learning difficulty and handle open boundaries.

## Limitations & Future Work

1. **Extremely limited training data**: Only 10 minutes of training data, which is the primary cause of poor probabilistic calibration and artifact generation.
2. **Two-dimensional simulation**: Validation is limited to the 2D meridional plane ($x$-$z$ plane); extension to three-dimensional global simulation has not been explored.
3. **Underdispersion**: Ensemble Spread is consistently smaller than RMSE; the forecast system is not fully calibrated.
4. **Artifacts**: Spurious structures appear in $B_y$ and $v_y$ in the magnetotail northern lobe region.
5. **Error accumulation**: Inherent error growth in autoregressive forecasting leads to degradation at long lead times.
6. **Single solar wind condition**: Training is conducted under fixed parameters ($\rho=1/\text{cm}^3$, $v=-750\,\text{km/s}$, $B_z=-5\,\text{nT}$), without coverage of diverse real solar activity conditions.
7. **Prediction of VDF moments only**: Training is on moments of the velocity distribution function; the full velocity distribution function is not modeled.

### Future Directions Proposed by the Authors

- **Three-dimensional extension**: Combined with adaptive mesh refinement (AMR) to increase resolution in critical regions; GNNs are naturally suited to irregular grids.
- **Larger datasets**: Diverse conditions spanning multiple solar cycles, supporting larger time steps to mitigate cumulative errors.
- **Physical constraints**: e.g., zero magnetic field divergence ($\nabla \cdot \mathbf{B} = 0$), improving physical realism.
- **Thermalization techniques**: Controlling error growth to extend stable rollout duration.
- **Foundation model direction**: Space weather data can be incorporated into large-scale multi-physics simulation datasets such as The Well, enabling foundation models across scales and systems.
- **Full VDF prediction**: Predicting the complete ion velocity distribution function, following approaches from 5D gyrokinetic surrogate models.

## Related Work & Insights

| Method | Type | Characteristics |
|---|---|---|
| BATS-R-US (MHD) | Physical model | Operational standard, but neglects ion kinetics |
| Vlasiator | Physical model | Highest fidelity, hybrid-Vlasov, but prohibitively expensive for real-time use |
| GraphCast | ML atmospheric weather | Graph structure directly borrowed by this work |
| Graph-FM/EFM (Oskarsson 2024) | ML atmospheric weather | Source of the base architecture, adapted for space weather |
| Solar wind perturbation ensemble | Ensemble forecasting | Perturbs physical model inputs; this work directly generates ensembles via ML |

The core contribution of this paper lies not in simply replicating atmospheric ML methods for space weather, but in adapting them to the specific structure of magnetospheric simulation (irregular grids, inner boundary exclusion, open boundary enforcement), and in being the first to train an ML model on hybrid-Vlasov data (rather than MHD).

## My Notes

- **Generality of GNNs in scientific simulation**: encode-process-decode + residual prediction + multi-scale graph hierarchy has become a general paradigm for physics simulation ML (GraphCast → Graph-FM/EFM → this work's space weather transfer), with strong architectural reusability.
- **Architecture choice for probabilistic forecasting**: Conditional VAE + CRPS fine-tuning is a practical probabilistic forecasting solution; the three-stage progressive training (AE → VAE → CRPS fine-tune) is worth referencing and can be generalized to other spatiotemporal prediction tasks.
- **ML under data-limited regimes**: With only 10 minutes of training data, the GNN still learns meaningful dynamics. This demonstrates the critical importance of inductive biases (graph structure, residual prediction, boundary enforcement) in low-data settings.
- **Foundation model trend**: Large-scale multi-physics simulation datasets such as The Well and Multimodal Universe, combined with foundation models such as Aurora/Surya, make space weather a natural next target domain.
- **Classification note**: This paper is categorized under the image_generation directory, but it is effectively a scientific computing / spatiotemporal prediction task with little connection to image generation. Reclassification to a more appropriate category may be warranted.
- **Differences from atmospheric weather ML**: The distinctive aspects of space weather include irregular inner boundaries (Earth), dayside forcing for open boundaries, and the kinetic complexity of the Vlasov equation, which substantially exceeds that of Navier-Stokes.

## Rating
- Novelty: ⭐⭐⭐⭐ (domain transfer innovation, not methodological innovation)
- Experimental Thoroughness: ⭐⭐⭐ (very limited data, but ablations and comparisons are complete)
- Writing Quality: ⭐⭐⭐⭐ (clear and well-organized)
- Value: ⭐⭐⭐⭐ (opens a new direction for space weather ML)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)

</div>

<!-- RELATED:END -->
