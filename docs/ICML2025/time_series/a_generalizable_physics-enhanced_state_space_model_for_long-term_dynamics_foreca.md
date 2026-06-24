---
title: >-
  [Paper Note] A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments
description: >-
  [ICML 2025][Time Series][State Space Models] This paper proposes Phy-SSM, which integrates partially known physical knowledge into deep state space models (SSMs). Through dynamics decomposition (known/unknown matrices) and physical state regularization, it achieves accurate long-term dynamics forecasting and extrapolation for noisy, irregularly sampled data.
tags:
  - "ICML 2025"
  - "Time Series"
  - "State Space Models"
  - "Partial Physical Knowledge"
  - "Long-term Dynamics Forecasting"
  - "Irregular Sampling"
  - "Physics-Enhanced Machine Learning"
date: 2026-05-08
content_hash: 177fd50bf9317b2c
---

# A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments

**Conference**: ICML 2025  
**arXiv**: [2507.10792](https://arxiv.org/abs/2507.10792)  
**Code**: [GitHub](https://github.com/511205787/Phy_SSM-ICML2025)  
**Area**: Human Understanding  
**Keywords**: State Space Models, Partial Physical Knowledge, Long-term Dynamics Forecasting, Irregular Sampling, Physics-Enhanced Machine Learning

## TL;DR

This paper proposes Phy-SSM, which integrates partially known physical knowledge into deep state space models (SSMs). Through dynamics decomposition (known/unknown matrices) and physical state regularization, it achieves accurate long-term dynamics forecasting and extrapolation for noisy, irregularly sampled data.

## Background & Motivation

Real-world dynamical systems (such as autonomous driving, epidemic spread, and climate science) are typically governed by physical laws, but complete physical equations are often difficult to obtain. Prior methods face the core dilemmas:

**Incomplete Physical Knowledge**: Traditional PINNs or Hamiltonian networks assume that physical laws are fully known, but the complete dynamical equations of complex systems (e.g., vehicle operation in adverse weather) cannot be derived from first principles.

**Poor Data Quality**: Sensor failures and clock unsynchronization lead to noisy and irregularly sampled data.

**Weak Long-term Extrapolation**: Existing partially physics-enhanced methods (e.g., GOKU, PI-VAE) are based on Neural ODEs, which heavily rely on initial conditions and lack dynamic correction mechanisms, leading to a sharp drop in performance during long-term extrapolation tasks.

**Limitations of Prior Work**: Methods like SINDy Autoencoder rely on finite differences to estimate derivatives, making them applicable only to noise-free, regularly sampled data; NODE-based methods face difficulties in capturing long-range correlations in non-linear time-varying systems.

**Core Problem**: How to improve the accuracy and generalization capability of long-term dynamics forecasting when physical knowledge is only partially known and the data is noisy and irregularly sampled?

## Method

### Overall Architecture

Phy-SSM consists of three core components:

1. **Sequential Encoder**: Based on a simplified structured SSM, it encodes observation sequences and estimates the posterior distribution of latent states.
2. **Phy-SSM Unit (Core)**: Decomposes partially known physical dynamics into known and unknown state matrices, and models the unknown dynamics using the HiPPO memory mechanism.
3. **Decoder**: Maps physical latent states to final outputs.

Workflow: The encoder extracts the latent state posterior from noisy, irregular observations $\rightarrow$ The Phy-SSM unit utilizes the previous latent state and control input to predict the physics-consistent latent state for the next step $\rightarrow$ The decoder outputs the predicted trajectory.

### Key Designs

#### 1. Dynamics Decomposition

This is the core innovation of this work. The system dynamics $f(z, u)$ are decomposed into a known part $f_{\text{knw}}$ and an unknown part $f_{\text{unk}}$:

$$\frac{dz(t)}{dt} = f_{\text{knw}}(z(t), u(t)) + f_{\text{unk}}(z(t), u(t))$$

Furthermore, the non-linear system is transformed into a linear SSM form through **state expansion** (introducing a non-linear term $\psi(z)$):

$$\frac{d\bar{z}(t)}{dt} = (A_{\text{knw}}(t) + A_{\text{unk}}(t))\bar{z}(t) + B_{\text{unk}}(t)u(t)$$

where $\bar{z} = [z^\top, \psi(z)^\top]^\top$ represents the expanded state, $A_{\text{knw}}$ encodes the known physics, and $A_{\text{unk}}$ is learned by the network. The benefits of this decomposition are: (i) leveraging matrix operations to improve efficiency; (ii) directly embedding physical knowledge into the matrix structure.

#### 2. Knowledge Mask Mechanism

A binary mask $M \in \{0,1\}^{d_{\bar{z}} \times d_{\bar{z}}}$ is introduced to impose **hard constraints** on the unknown matrix learned by the network:

$$A_{\text{unk}}(t) = M_A \odot \tilde{A}_{\text{unk}}(t)$$

In the mask, 1 indicates that the corresponding position is allowed to be learned (representing unknown dynamics), and 0 indicates that updates are blocked (representing known or unrelated parts). Taking a pendulum system as an example, only the second row describing angular acceleration needs to be learned, and all other rows are masked to zero. This ensures that physical constraints are strictly enforced as hard constraints.

#### 3. Structured SSM for Modeling Unknown Dynamics

Multilayer structured SSMs (based on the S5 architecture) are used with latent state $z$ and control input $u$ as inputs to approximate and learn the unknown continuous functions $\tilde{A}_{\text{unk}}(t)$ and $\tilde{B}_{\text{unk}}(t)$. Key advantages:

- **HiPPO Memory Mechanism**: Efficiently retains historical information and captures long-range dependencies.
- **Continuous-time Modeling**: Naturally supports irregularly sampled data.
- **Parallel Scan Training**: Accelerates training via parallel scans.

#### 4. Posterior Estimation of the Encoder

The encoder is a simplified structured SSM that introduces a memory variable $h(t)$ to capture long-sequence correlations. The posterior of the latent state at each time step depends on the current observation and the memory of the previous step:

$$z(t_i) \mid x(t_i) \sim \mathcal{N}(\hat{\mu}_z(t_i), \text{diag}(\hat{\sigma}_z^2(t_i)))$$

This design allows the model to dynamically correct predictions during the interpolation phase based on subsequent observations, rather than relying entirely on initial conditions.

#### 5. Physical State Regularization

The mean and variance of the prior distribution are generated by the Phy-SSM unit, directly embedding physical knowledge. It dynamically corrects predictions utilizing the posterior during the interpolation phase and implements autoregressive prediction during the extrapolation phase.

#### 6. Theoretical Guarantee — Uniqueness of Decomposition

**Proposition 1**: Under the condition that the support sets of $A_{\text{knw}}$ and $A_{\text{unk}}$ do not overlap (guaranteed by the knowledge mask), the solution to the dynamics decomposition is unique when minimizing the objective function. This ensures that the known and unknown parts do not interfere with each other during training.

### Loss & Training

The overall objective function consists of two parts:

$$\mathcal{L} = \mathcal{L}_{\text{VAE}} + \lambda \mathcal{L}_{\text{reg}}$$

- **VAE Loss** $\mathcal{L}_{\text{VAE}}$: Consists of the reconstruction loss $\mathcal{L}_{\text{recon}}$ (reconstruction quality of observations) and the KL divergence $\mathcal{L}_{\text{KL}}$ (distance between prior and posterior distributions), where the weight coefficient $\beta$ controls the strength of the KL term.
- **Physical State Regularization** $\mathcal{L}_{\text{reg}}$: A penalty on the Euclidean distance between prior samples $z(t_i)$ and posterior samples $z^*(t_i)$, controlled by the weight coefficient $\lambda$.

The regularization term plays a dual role: (1) constraining the encoder output to adhere to physical dynamics; (2) guiding the Phy-SSM unit to learn more accurate unknown dynamics consistent with global trajectories, significantly improving extrapolation performance.

## Key Experimental Results

### Main Results

Evaluated on three real-world applications: UAV (drone) state prediction (high-frequency irregular sampling 573–1915 Hz), COVID-19 epidemic forecasting (10% missing data), and vehicle motion prediction (nuScenes dataset).

**UAV state prediction (Drone)**

| Method | Interpolation MAE(×10⁻¹)↓ | Interpolation MSE(×10⁻¹)↓ | Extrapolation MAE(×10⁻¹)↓ | Extrapolation MSE(×10⁻¹)↓ |
|------|-----|-----|-----|-----|
| S5 | 1.059 | 0.309 | 8.426 | 17.333 |
| ContiFormer | 1.446 | 0.374 | 4.059 | 5.092 |
| GOKU | 3.293 | 2.738 | 3.456 | 3.130 |
| **Phy-SSM (Ours)** | **1.002** | **0.222** | **2.733** | **1.798** |

**COVID-19 epidemic forecasting**

| Method | Interpolation MAE(×10⁻¹)↓ | Interpolation MSE(×10⁻²)↓ | Extrapolation MAE(×10⁻¹)↓ | Extrapolation MSE(×10⁻¹)↓ |
|------|-----|-----|-----|-----|
| S5 | 0.861 | 1.057 | 5.212 | 4.560 |
| ContiFormer | 0.830 | 1.059 | 6.882 | 9.147 |
| GOKU | 1.019 | 1.667 | 6.140 | 7.918 |
| **Phy-SSM (Ours)** | **0.795** | **1.032** | **1.998** | **0.692** |

**Vehicle motion prediction (nuScenes, Out-of-Domain extrapolation)**

| Method | ADE↓ | FDE↓ | Speed Error↓ | Accel Error(×10¹)↓ |
|------|------|------|------|------|
| Wayformer | 8.842 | 8.810 | 46.233 | 76.267 |
| SDVAE | 7.050 | 8.235 | 2.689 | 2.065 |
| PIVAE | 7.569 | 8.519 | 2.381 | 2.081 |
| **Phy-SSM (Ours)** | **6.206** | **7.197** | **2.398** | **2.043** |

### Ablation Study

| Configuration | Extrapolation MAE(×10⁻¹)↓ | Extrapolation MSE(×10⁻¹)↓ | Description |
|------|-----|-----|------|
| W/o Phy-SSM unit + w/o regularization (pure data-driven SSM) | 8.426 | 17.333 | Severe extrapolation degradation |
| W/ Phy-SSM unit + w/o regularization | 3.008 | 2.176 | Physics embedding significantly improves extrapolation |
| **Full Model (Phy-SSM unit + regularization)** | **2.733** | **1.798** | Regularization further constrains the latent state |

### Key Findings

1. **Huge Gap in Extrapolation Performance**: The pure data-driven S5 has an extrapolation MSE of 17.333, whereas Phy-SSM achieves only 1.798, reducing the error by nearly 10 times.
2. **Physics-Enhanced Methods Uniformly Outperform Data-Driven Methods**: In out-of-domain vehicle prediction, all PEML (Physics-Enhanced Machine Learning) methods outperform data-driven SOTAs like Wayformer by a large margin on Speed/Accel/Jerk metrics.
3. **Double-edged Sword Effect of Regularization**: Removing the regularization slightly improves interpolation performance (due to overfitting observations) but severely degrades extrapolation. This indicates that regularization guides the model to learn generalized physical dynamics rather than overfitting the training data.
4. **Most Significant Extrapolation Improvement in COVID-19**: The extrapolation MSE drops from 4.560 (attained by the runner-up S5) to 0.692, a reduction of over 85%, demonstrating the capability to model time-varying dynamics.

## Highlights & Insights

1. **Dynamics Decomposition + Knowledge Mask**: Integrates partial physical knowledge as hard constraints into the SSM in a concise and elegant manner, avoiding the looseness of PINN-style soft constraints, with theoretical guarantees of decomposition uniqueness.
2. **Rationality of Replacing NODE with SSM**: Addressing the inherent defect of NODEs relying heavily on initial conditions and degrading in long-term forecasting, the work opts for the HiPPO memory and continuous-time properties of SSMs to capture long-range dependencies.
3. **Generalized Framework**: The same method is effective across three highly diverse domains: autonomous driving (kinematics), epidemiology (SIR models), and UAV control (rigid body dynamics), demonstrating the generality of the framework.
4. **Physical State Regularization**: Cleverly constrains the prior-posterior alignment within the VAE framework, extending physical consistency from the model architecture down to the training objective.
5. **Pendulum System Walk-Through**: Provides a concrete pendulum system example to clearly demonstrate the entire process of state expansion, matrix decomposition, and mask design, which is of highly valuable engineering reference.

## Limitations & Future Work

1. **Requires Manual Design by Domain Experts**: The $A_{\text{knw}}$ matrix and knowledge mask $M$ must be constructed manually according to specific systems. The degree of automation is insufficient, requiring physical modeling expertise when transferring to new domains.
2. **Linearization Hypothesis**: Translates non-linear dynamics into linear SSMs via state expansion, but highly non-linear systems (e.g., turbulence) might require extremely high-dimensional expanded states.
3. **Limited Evaluation Metrics**: The extrapolation horizon is 60–200 steps; performance over longer time scales is yet to be verified.
4. **Insufficient Discussion on Computational Overhead**: The incremental computational complexity of multilayer structured SSMs + matrix decomposition compared to simple S5 or Mamba is not fully provided.
5. **Extendable Directions**: Future work could attempt to automatically discover partial physical structures (integrating SINDy concepts), introduce attention mechanisms to enhance the representation capability of the time-varying $A_{\text{unk}}$, or explore multi-agent interaction scenarios.

## Related Work & Insights

- **PINN/HNN/LNN Series**: Reliant entirely on fully known physical laws; this work relaxes this to partially known physics, making it more realistic.
- **GOKU / PI-VAE**: Both are partially physics-enhanced VAEs, but they model unknown dynamics based on NODEs, thus exhibiting weak long-term extrapolation. Replacing NODE with SSM in this work is a key improvement.
- **S4/S5/Mamba**: Pure data-driven SSMs lack physical inductive bias, leading to limited extrapolation capability. This work is the first to embed partial physical knowledge into SSMs.
- **SINDy Autoencoder**: Automatically discovers physical laws from data, but relies on finite differences and is unsuitable for noisy data. This is complementary to this work—future directions could combine SINDy automatic discovery with Phy-SSM prediction.

**Insights for Own Research**: The concept of dynamics decomposition + masking can be generalized to any "partial knowledge embedding" scenario—as long as the problem can be split into a known structure and an unknown part, a similar framework can be applied. This has direct application prospects in human understanding tasks such as human motion prediction and social behavior modeling.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The combination of SSM and partial physical knowledge decomposition is novel; the masking mechanism is concise and effective. |
| Theoretical Depth | 4 | Provides a theoretical analysis of the uniqueness of the decomposition. |
| Experimental Thoroughness | 4.5 | Three real-world scenarios + comprehensive ablations + sensitivity analysis. |
| Writing Quality | 4 | Excellent Walk-Through example; the overall structure is clear. |
| Value | 3.5 | Requires manual construction of the physical matrices, which poses a relatively high barrier to engineering deployment. |
| **Overall** | **4** | Solid work with distinct contributions to the direction of physics-enhanced deep learning. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](../../ICML2026/time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](../../NeurIPS2025/time_series/rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)

</div>

<!-- RELATED:END -->
