---
title: >-
  [Paper Note] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning
description: >-
  [ICLR 2026][Scientific Computing][KKL observer] This paper proposes HyperKKL, which uses a hypernetwork to encode exogenous input signals and dynamically generate the transformation mapping parameters of a KKL observer, enabling state estimation for non-autonomous nonlinear systems without retraining or online gradient updates. The method is validated on four classical nonlinear systems: Duffing, Van der Pol, Lorenz, and Rössler.
tags:
  - ICLR 2026
  - Scientific Computing
  - KKL observer
  - state estimation
  - hypernetwork
  - non-autonomous system
  - dynamical system
date: 2026-05-08
content_hash: b7501abccf00ff2c
---

# HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning

**Conference**: ICLR 2026
**arXiv**: [2602.22630](https://arxiv.org/abs/2602.22630)
**Code**: To be confirmed
**Area**: Scientific Computing
**Keywords**: KKL observer, state estimation, hypernetwork, non-autonomous system, dynamical system

## TL;DR

This paper proposes HyperKKL, which uses a hypernetwork to encode exogenous input signals and dynamically generate the transformation mapping parameters of a KKL observer, enabling state estimation for non-autonomous nonlinear systems without retraining or online gradient updates. The method is validated on four classical nonlinear systems: Duffing, Van der Pol, Lorenz, and Rössler.

## Background & Motivation

**Background**: State estimation—reconstructing the full internal state of a dynamical system from partially observable measurements—is a fundamental problem in control and engineering. The KKL (Kazantzis-Kravaris/Luenberger) observer achieves state estimation by immersing a nonlinear dynamical system into a higher-dimensional stable linear latent space, and theoretically guarantees global convergence under the backward distinguishability condition.

**Limitations of Prior Work**:

- The core of the KKL observer requires solving an analytically intractable PDE: $\frac{\partial \mathcal{T}}{\partial x}(x) f(x) = A \mathcal{T}(x) + B h(x)$
- Recent neural network-based approaches (PINNs, autoencoders, etc.) can approximately solve these mappings, but nearly all target **autonomous systems** (no external input $u(t)$)
- Real-world systems are almost never autonomous—robots receive motor commands, biological systems respond to external stimuli, and industrial processes are subject to time-varying disturbances

**Key Challenge**: When extending to non-autonomous systems, the transformation mapping $\mathcal{T}$ must become input-dependent as $\mathcal{T}(x, t)$, satisfying the time-varying PDE:

$$\frac{\partial \mathcal{T}}{\partial x}(x,t) f(x, u(t)) + \frac{\partial \mathcal{T}}{\partial t}(x,t) = A \mathcal{T}(x,t) + B h(x)$$

The additional temporal partial derivative term $\frac{\partial \mathcal{T}}{\partial t}$ couples the transformation with the temporal evolution of the input, rendering static mappings insufficient. Existing learning methods either require retraining for each input scenario or online gradient updates, severely limiting practical applicability.

**Goal**: HyperKKL encodes the history of the input signal into an instantaneous perturbation of the observer parameters via a hypernetwork, enabling inference-time adaptation to different exogenous input conditions without retraining or online optimization.

## Method

### Overall Architecture

HyperKKL employs a two-phase sequential training procedure:

1. **Phase 1 (Autonomous Pretraining)**: Under no external input ($u \equiv 0$), a base encoder $\hat{\mathcal{T}}_{\theta^{\text{base}}}$ and decoder $\hat{\mathcal{T}}^*_{\phi^{\text{base}}}$ are trained with a physics-informed loss to satisfy the autonomous KKL conditions. These parameters are frozen after training.
2. **Phase 2 (Hypernetwork Training)**: The base mappings are frozen, and only the hypernetwork parameters $\psi$ are trained to learn the mapping from input signals to parameter perturbations.

At inference time: a new input signal → LSTM encoding → generation of parameter perturbations $\Delta\theta, \Delta\phi$ → superimposed onto the frozen base parameters → instantaneous input-adaptive observer.

The learning objective combines reconstruction loss and PDE residual:

$$\min_\psi \mathbb{E}_{(x,u) \sim \mathcal{D}} \left[ \underbrace{\| x - \hat{\mathcal{T}}^*(\hat{\mathcal{T}}(x; \theta_u), \phi_u) \|^2}_{\mathcal{L}_{\text{rec}}} + \lambda \underbrace{\left\| \frac{\partial \hat{\mathcal{T}}}{\partial x} f(x,u) + \frac{\Delta \hat{\mathcal{T}}}{\Delta t} - A\hat{\mathcal{T}} - Bh(x) \right\|^2}_{\mathcal{L}_{\text{PDE}}} \right]$$

### Key Design 1: Dynamic HyperKKL

For complex systems where inputs continuously reshape the attractor geometry, a truly time-varying transformation $\mathcal{T}(x, \theta(t))$ is required. Dynamic HyperKKL employs a residual hypernetwork that decouples the base parameters from the input-conditioned perturbation:

$$\theta_{\text{enc}}(t) = \theta_{\text{enc}}^{\text{base}} + \Delta\theta_{\text{enc}}(u_{[t-w, t]})$$
$$\phi_{\text{dec}}(t) = \phi_{\text{dec}}^{\text{base}} + \Delta\phi_{\text{dec}}(u_{[t-w, t]})$$

The hypernetwork consists of three components:
- **Shared LSTM encoder**: processes the input window $u_{[t-w, t]}$ (window size $w = 100$) and outputs hidden state $h_t \in \mathbb{R}^{d_h}$
- **Encoder head MLP**: predicts $\Delta\theta_{\text{enc}}$ from $h_t$
- **Decoder head MLP**: predicts $\Delta\phi_{\text{dec}}$ from $h_t$

**Chunked prediction strategy**: Directly predicting the full parameter perturbation dimensionality is prohibitively high-dimensional. The target weight matrix $W \in \mathbb{R}^{m \times n}$ is partitioned into blocks, with the MLP independently predicting each block, maintaining expressiveness while controlling output dimensionality.

**Residual structure guarantee**: When $u \equiv 0$, the LSTM hidden state produces $\Delta\theta = \Delta\phi = 0$, exactly recovering the autonomous observer—ensuring no degradation in the absence of external input.

The temporal partial derivative is estimated via finite differences:

$$\frac{\Delta \hat{\mathcal{T}}}{\Delta t} \approx \frac{\hat{\mathcal{T}}(x; \theta(u_{[t, t+\Delta t]})) - \hat{\mathcal{T}}(x; \theta(u_{[t-\Delta t, t]}))}{\Delta t}$$

### Key Design 2: Static HyperKKL

For simpler systems where the input acts only as a bounded perturbation, the autonomous transformation $\mathcal{T}(x)$ is retained unchanged, and a learned input injection term is added only to the observer dynamics:

$$\dot{\hat{z}} = A\hat{z} + By + \bar{\varphi}(\hat{z}, u; \xi)$$

where $\bar{\varphi}$ is a small MLP, taking both the LSTM-encoded input context and $\hat{z}$ as inputs. Training constrains $\bar{\varphi}$ to output zero when $u = 0$.

### Key Design 3: Adaptive Curriculum Learning Baseline

As a control condition, the paper also evaluates whether training strategy alone can compensate for the limitations of a static architecture. Curriculum learning is used to grade training data by input complexity ($\mathcal{D}_1$: constants → $\mathcal{D}_2$: low-frequency sinusoids → ... → high-frequency mixtures), advancing to the next level when loss stagnates at the current one. This baseline addresses the question: **can the non-autonomous problem be solved purely through richer training data, without architectural modification?**

## Key Experimental Results

### Main Results: State Estimation Performance on Four Nonlinear Systems (RMSE / SMAPE%)

**Oscillatory systems (Duffing, Van der Pol)**:

| Method | Duffing-Zero | Duffing-Sin | Duffing-Sqr | VdP-Zero | VdP-Sin | VdP-Sqr |
|:-----|:------------|:------------|:------------|:---------|:--------|:--------|
| Autonomous | 0.04 (5.6) | 0.26 (26) | 0.33 (31) | 0.15 (7.0) | 0.23 (9.8) | 0.25 (10.5) |
| Curriculum | 0.27 (33) | 0.44 (41) | 0.57 (46) | 1.10 (51.4) | 1.15 (51.5) | 1.15 (51.7) |
| Static HyperKKL | **0.04** (5.6) | **0.10↓** (9.3) | **0.17↓** (14) | 0.12↓ (5.3) | 0.24 (10.2) | 0.25 (10.8) |
| Dynamic HyperKKL | 0.08 (8.2) | 0.24↓ (25) | 0.27↓ (28) | **0.12↓** (5.0) | **0.21↓** (8.6) | **0.22↓** (9.1) |

**Chaotic systems (Rössler, Lorenz)**:

| Method | Rössler-Zero | Rössler-Sin | Rössler-Sqr | Lorenz-Zero | Lorenz-Sin | Lorenz-Sqr |
|:-----|:------------|:------------|:------------|:------------|:-----------|:-----------|
| Autonomous | 1.14 (6.7) | 1.47 (7.6) | 1.48 (8.3) | **5.56** (18) | **5.58** (18) | **5.55** (18) |
| Curriculum | 5.58 (35) | 5.94 (37) | 5.61 (38) | 11.5 (41) | 11.6 (42) | 11.6 (42) |
| Static HyperKKL | 1.14 (6.7) | 1.70 (10) | 1.75 (12) | 5.56 (18) | 16.3 (52) | 16.2 (51) |
| **Dynamic HyperKKL** | **1.01↓** (5.1) | **1.38↓** (6.0) | **1.36↓** (6.9) | 6.67 (22) | 6.67 (22) | 6.66 (22) |

Key findings:

1. **Static HyperKKL achieves the best performance on low-dimensional oscillatory systems**: RMSE on Duffing with sinusoidal input is reduced by 62% (0.26 → 0.10), consistent with theoretical expectations—the attractor of a low-dimensional oscillator shifts smoothly with the input, making a static transformation sufficient.
2. **Curriculum learning fails across the board**: Performance is **worse** than the autonomous baseline on all systems under all input conditions (e.g., VdP-Zero: 0.15 → 1.10), demonstrating that the bottleneck is **representational rather than educational**.
3. **The Lorenz system exposes a fundamental limitation**: The autonomous baseline achieves the best performance (RMSE ≈ 5.5), Static HyperKKL degrades catastrophically (16.3), and Dynamic HyperKKL also exhibits mild degradation (6.67).

### Ablation Study: Decoupled Analysis of Architecture vs. Training

| Analysis Dimension | Conclusion | Evidence |
|:---------|:-----|:-----|
| Curriculum learning vs. no training | Curriculum learning is harmful | Performance inferior to autonomous baseline on all systems |
| Static vs. Dynamic | System complexity determines the choice | Static for low-dimensional, Dynamic for chaotic |
| Input encoding method | LSTM outperforms MLP | Temporal aggregation is critical for chaotic systems |
| Recovery at $u=0$ | All hypernetwork methods correctly recover autonomous performance | $\Delta\theta \to 0$ verification successful |
| Lorenz specificity | High-sensitivity attractor causes hypernetwork conditioning to introduce noise | Small errors amplify exponentially along unstable manifolds |

## Rating

**Rating**: ⭐⭐⭐⭐

**Strengths**:

- Clearly extends the KKL observer from autonomous to non-autonomous systems, filling a practical gap in learning-based KKL methods
- The two-phase training scheme (autonomous pretraining + hypernetwork fine-tuning) and residual architecture design are well-motivated and guarantee graceful recovery at $u=0$
- The chunked prediction strategy balances the expressive capacity of the hypernetwork with output dimensionality
- Failures on the Lorenz system are honestly reported with in-depth theoretical analysis (unstable manifolds + exponential error amplification)
- The comparison between Static and Dynamic architectures provides a practical selection guide

**Limitations**:

- Validation is limited to four classical low-dimensional systems (up to 3-dimensional state spaces); scalability to high-dimensional real-world systems remains unknown
- The failure on the Lorenz system exposes a fundamental limitation of hypernetwork conditioning on high-sensitivity systems, with no solution currently proposed
- The failure of the curriculum learning baseline may be partially attributable to implementation details (e.g., hyperparameter choices) rather than purely architectural constraints
- Comparisons with other non-autonomous observer methods (e.g., EKF, UKF in non-autonomous settings) are absent
- Computational cost analysis is missing—does the inference latency of the LSTM hypernetwork meet the requirements of real-time control?

**Key Distinctions from Related Work**:

- Unlike Niazi et al. (2025), which handles only autonomous KKL, this paper achieves non-autonomous extension via a hypernetwork
- Unlike meta-RL approaches (e.g., MAML) that require online gradient updates, HyperKKL adapts through purely forward inference
- Unlike static transformation methods, Dynamic HyperKKL explicitly models the temporal partial derivative term in the time-varying PDE

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/scientific_computing/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[ICLR 2026\] Policy Myopia as a Mechanism of Gradual Disempowerment in Post-AGI Governance](policy_myopia_as_a_mechanism_of_gradual_disempowerment_in_post-agi_governance_ci.md)

</div>

<!-- RELATED:END -->
