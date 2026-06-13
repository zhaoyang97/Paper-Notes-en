---
title: >-
  [Paper Note] WARP: Weight-Space Linear Recurrent Neural Networks
description: >-
  [ICLR 2026][Time Series][Weight-space learning] This paper proposes WARP (Weight-space Adaptive Recurrent Prediction), which explicitly parameterizes the hidden state of a linear RNN as the weights and biases of an auxil…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Weight-space learning"
  - "Linear RNN"
  - "Adaptive prediction"
  - "Dynamical system reconstruction"
  - "Gradient-free adaptation"
date: 2026-05-08
content_hash: af5dec63340b7c6b
---

# WARP: Weight-Space Linear Recurrent Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2506.01153](https://arxiv.org/abs/2506.01153)  
**Area**: Time Series
**Keywords**: Weight-space learning, Linear RNN, Adaptive prediction, Dynamical system reconstruction, Gradient-free adaptation

## TL;DR

This paper proposes WARP (Weight-space Adaptive Recurrent Prediction), which explicitly parameterizes the hidden state of a linear RNN as the weights and biases of an auxiliary MLP. Input differences drive a linear recurrence to update these weights, and a nonlinear decoding step enables efficient sequence modeling. WARP achieves state-of-the-art performance on classification, forecasting, and dynamical system reconstruction tasks.

## Background & Motivation

Deep sequence models face two fundamental limitations:

**Insufficient generalization**: They cannot reliably operate outside the training distribution and require gradient descent for adaptation.

**Difficulty incorporating domain priors**: Physical constraints and other domain knowledge cannot be integrated during the forward pass.

Meanwhile, two emerging paradigms each offer distinct advantages but have not been combined:

| Paradigm | Strengths | Limitations |
|----------|-----------|-------------|
| **Weight-space learning** | Treats neural network weights as data points | Used only as inputs/outputs, not as intermediate representations |
| **Linear RNNs** (S4, Mamba) | Hardware-efficient and parallelizable training | Limited expressivity, insufficient information compression |

**Core Insight**: The absence of nonlinearity in linear RNNs limits their expressivity, yet reintroducing nonlinearity sacrifices training efficiency. WARP resolves this tension by defining the hidden state as MLP weights, preserving the efficiency of linear recurrence while introducing nonlinearity at decoding time.

## Method

### Overall Architecture

The core recurrence and decoding process of WARP:

$$\theta_t = A\theta_{t-1} + B\Delta\mathbf{x}_t, \quad \mathbf{y}_t = \text{MLP}_{\theta_t}(\tau)$$

where:
- $\theta_t \in \mathbb{R}^{D_\theta}$ are the flattened weights of an auxiliary MLP (the "root network")
- $\Delta\mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$ is the **input difference** (inspired by neural signal processing in the brain)
- $A \in \mathbb{R}^{D_\theta \times D_\theta}$ is the state transition matrix
- $B \in \mathbb{R}^{D_\theta \times D_x}$ is the input projection matrix
- $\tau$ is a coordinate system (normalized pixel positions, time steps, etc.)

### Key Design 1: Self-Decoding Mechanism

$\theta_t$ serves a dual role as both the **hidden state** and the **decoder parameters**—it decodes itself. This significantly reduces parameter count by eliminating the need for a separate decoder network.

### Key Design 2: Input-Difference Drive

Using $\Delta\mathbf{x}_t$ rather than $\mathbf{x}_t$ to drive the recurrence:
- Weight updates scale proportionally when inputs change slowly
- The model learns to translate input differences into network updates—essentially **gradient-free continual adaptation**

### Key Design 3: Initialization Strategy

- $A$ is initialized as the identity matrix $I$: simulates residual connections and promotes gradient flow
- $B$ is initialized as the zero matrix $\mathbf{0}$: prevents $\theta_t$ from diverging in early training
- $\theta_0 = \phi(\mathbf{x}_0)$: initial weights are generated from the first observation via a hypernetwork $\phi$

### Training and Inference

**Training modes**:
- **Convolutional mode**: Unrolls the linear recurrence into a convolution kernel $K$ for parallel training
- **Recurrent mode**: Distinguishes between autoregressive (AR) and non-AR settings

**Loss function**:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{T}\sum_{t=0}^{T-1}\|\mathbf{y}_t - \hat{\mathbf{y}}_t\|_2^2$$

Negative log-likelihood (NLL) is used for probabilistic forecasting; categorical cross-entropy (CCE) is used for classification.

### Physics Prior Injection (WARP-Phys)

Domain knowledge is injected by replacing the root network's forward pass with a physical formula (e.g., $\tau \mapsto \sin(2\pi\tau + \hat{\varphi})$). This yields over a 10× performance improvement on dynamical system reconstruction tasks.

## Key Experimental Results

### Image Completion (MNIST, $L=300$ context pixels)

| Model | MSE ↓ | BPD ↓ |
|-------|-------|-------|
| GRU | 0.054 | 0.573 |
| LSTM | 0.057 | 0.611 |
| S4 | 0.049 | 0.520 |
| **WARP** | **0.042** | **0.516** |

### Traffic Flow Forecasting (PEMS08)

| Model | MAE ↓ | RMSE ↓ |
|-------|-------|--------|
| STIDGCN (GNN-SOTA) | 13.45 | 23.28 |
| D2STGNN | 14.35 | 24.18 |
| **WARP** | **6.59** | **10.10** |

Without using graph structure, WARP reduces MAE by over 50%, substantially outperforming GNN-based models that explicitly exploit spatial information.

### Dynamical System Reconstruction

| Dataset | GRU MSE | LSTM MSE | Transformer MSE | WARP MSE | WARP-Phys MSE |
|---------|---------|----------|-----------------|----------|---------------|
| MSD | 1.43 | 1.46 | 0.34 | 0.94 | **0.03** |
| MSD-Zero | 0.55 | 0.57 | 0.48 | **0.32** | **0.04** |
| LV | 5.83 | 6.18 | 11.27 | **4.72** | — |
| SINE* | 4.90 | 9.48 | 1728 | 2.77 | **0.62** |

WARP-Phys improves over WARP by more than **30×** on MSD (0.94 → 0.03).

### Multivariate Time Series Classification (6 UEA Datasets)

Among six competing methods, WARP ranks in the top three on four datasets, achieves state-of-the-art on SCP2 and Heartbeat, and demonstrates strong performance on very long sequences (EigenWorms, 17,984 steps).

## Highlights & Insights

1. **Paradigm-level innovation**: WARP is the first to use weight-space features as intermediate hidden-state representations in a recurrent network, unifying weight-space learning and linear recurrence.
2. **Brain-inspired input differencing**: Processing changes rather than absolute inputs naturally supports continual learning and test-time adaptation.
3. **Gradient-free adaptation**: The fast-weight $\theta_t$ is updated via linear recurrence (not gradient descent), enabling efficient runtime adaptation.
4. **Flexible physics prior injection**: Arbitrary domain knowledge can be embedded into the root network's forward pass; WARP-Phys achieves over 10× performance gains.
5. **Striking PEMS08 results**: A 50% MAE reduction without graph structure challenges the dominance of GNNs in traffic forecasting.

## Limitations & Future Work

1. The state transition matrix $A \in \mathbb{R}^{D_\theta \times D_\theta}$ can be very large, constraining the scale of the root network.
2. Physics prior injection (WARP-Phys) requires known domain-specific formulas, limiting its generality.
3. The input-difference formulation assumes uniformly sampled sequences; handling irregular time series is not discussed.
4. The classification experiments cover only six datasets, and statistical significance could be further strengthened.
5. Direct comparisons with recent linear RNNs such as Mamba and Griffin are insufficient.

## Rating ⭐⭐⭐⭐⭐

A paradigm-level contribution with exceptional novelty. WARP elegantly unifies weight-space learning and linear recurrence, achieving strong expressivity and adaptability within a clean framework. The 50% MAE reduction on PEMS08 and the 10× gain from WARP-Phys are impressive results. The primary concern is the scalability of the state transition matrix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](../../NeurIPS2025/time_series/parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](../../ICML2026/time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](../../NeurIPS2025/time_series/graph-based_neural_space_weather_forecasting.md)
- [\[CVPR 2026\] Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](../../CVPR2026/time_series/stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)

</div>

<!-- RELATED:END -->
