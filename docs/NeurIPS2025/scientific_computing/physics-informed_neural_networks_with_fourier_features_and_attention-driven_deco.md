---
title: >-
  [Paper Note] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][Scientific Computing][PINNs] This paper proposes Spectral PINNsformer (S-Pformer), which replaces the encoder of PINNsformer with Fourier feature embeddings and adopts a decoder-o…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "Scientific Computing"
  - "PINNs"
  - "Transformer"
  - "Fourier Features"
  - "Spectral Bias"
  - "PDE Solving"
date: 2026-05-08
content_hash: 7f45067991bba378
---

# Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding

**Conference**: NeurIPS 2025 (AI for Science Workshop)
**arXiv**: [2510.05385](https://arxiv.org/abs/2510.05385)
**Code**: Open-sourced (link provided in the paper)
**Area**: Scientific Computing
**Keywords**: PINNs, Transformer, Fourier Features, Spectral Bias, PDE Solving

## TL;DR

This paper proposes Spectral PINNsformer (S-Pformer), which replaces the encoder of PINNsformer with Fourier feature embeddings and adopts a decoder-only Transformer architecture. S-Pformer achieves superior performance on multiple PDE benchmarks while reducing parameter count by 18.6%, effectively alleviating the spectral bias problem.

## Background & Motivation

**Background**: Numerical solving of partial differential equations (PDEs) is a core problem in science and engineering. Traditional methods (finite differences, spectral methods) rely on fine grid discretization, incurring high computational costs and difficulty adapting to complex geometries. Physics-Informed Neural Networks (PINNs) embed physical constraints into the loss function to enable mesh-free solving.

**Limitations of Prior Work — Spectral Bias**: Mainstream MLP-based PINNs suffer from severe spectral bias, struggling to learn high-frequency components in PDE solutions and performing poorly on problems with rich multi-scale behavior.

**Limitations of Prior Work — Temporal Dependencies**: MLP-PINNs perform pointwise predictions and cannot capture spatiotemporal correlations in PDE solutions, limiting their effectiveness on parabolic and hyperbolic PDEs involving time derivatives.

**PINNsformer's Attempt**: Zhao et al. (2024) proposed PINNsformer, which uses an encoder-decoder Transformer architecture to capture spatiotemporal relationships via self-attention, significantly improving performance.

**Redundancy in PINNsformer**: The encoder-decoder design originates from sequence-to-sequence tasks (e.g., translation), but in the PINN setting the input and output share the same structure, making the encoder a source of unnecessary parameter redundancy and computational overhead.

**Goal**: To simultaneously address spectral bias and parameter redundancy by replacing the encoder with Fourier feature embeddings, yielding a more lightweight and capable Transformer PINN.

## Method

### Overall Architecture

The S-Pformer architecture consists of three components: (1) an input embedding module with Fourier features, (2) a decoder-only multi-head attention module, and (3) a linear output network. The input is first expanded into a temporal sequence via a pseudo-sequence generator, then encoded with Fourier and positional embeddings to capture multi-scale frequency information, and finally processed by the decoder's self-attention to model spatiotemporal dependencies.

### Key Designs

#### Module 1: Fourier Feature Embedding

- Normalized input coordinates $\tilde{\mathbf{z}} = (\tilde{\mathbf{x}}, \tilde{t}) \in [0,1]^{d_{in}}$ are projected into a high-dimensional frequency space via a random projection matrix $\mathbf{B} \sim \mathcal{N}(0, \mathbf{I})$
- Fourier embedding: $E_f(\tilde{\mathbf{z}}) = \theta_f([\sin(2\pi\mathbf{B}\tilde{\mathbf{z}}), \cos(2\pi\mathbf{B}\tilde{\mathbf{z}})])$, where $d_{\text{mapping}}$ controls the number of frequency bands (default 64)
- Positional embedding: $E_p(\tilde{\mathbf{z}}) = \theta_p(\tilde{\mathbf{z}})$, a linear transformation that preserves spatiotemporal locality
- Final embedding: $E(\tilde{\mathbf{z}}) = E_f(\tilde{\mathbf{z}}) + E_p(\tilde{\mathbf{z}})$
- **Core Idea**: Fourier features encode global periodic patterns to capture oscillatory behavior of multi-scale PDE solutions, while positional embeddings preserve local spatiotemporal relationships; the two are complementary in replacing both the original encoder and the spatiotemporal mixer

#### Module 2: Decoder-Only Transformer

- Each layer consists of: Wavelet activation → multi-head self-attention → residual connection → Wavelet activation → feed-forward network → residual connection
- Wavelet activation function: $\text{Wavelet}(z) = \omega_1 \sin(z) + \omega_2 \cos(z)$, where $\omega_1, \omega_2$ are learnable, replacing ReLU/LayerNorm (which are unfriendly to PINNs due to discontinuous derivatives)
- Self-attention is applied directly to the embedded coordinates, maintaining the ability to model time-dependent PDEs without an encoder
- Default configuration: $N=1$ layer, $n_{\text{heads}}=2$, $d_{\text{ff}}=512$, $d_{\text{emb}}=32$

#### Module 3: NTK Adaptive Loss Weighting

- Computes the Jacobian $J_i$ and NTK trace for each loss component: $K_i = \text{Tr}(J_i J_i^\top)$
- Loss weights are inversely proportional to the NTK trace: $\lambda_i = \frac{\sum K}{K_i}$, assigning smaller weights to more sensitive components
- Weights are updated every 50 iterations to ensure balanced convergence among the PDE residual, initial condition, and boundary condition loss terms

### Loss & Training

Standard three-term PINN loss:

$$\mathcal{L}(u_\theta) = \frac{\lambda_1}{N_\mathcal{F}} \sum \|\mathcal{F}(u_\theta)\|^2 + \frac{\lambda_2}{N_\mathcal{I}} \sum \|\mathcal{I}(u_\theta)\|^2 + \frac{\lambda_3}{N_\mathcal{B}} \sum \|\mathcal{B}(u_\theta)\|^2$$

where $\lambda_1, \lambda_2, \lambda_3$ are dynamically adjusted by the NTK scheme. The optimizer is L-BFGS (Strong-Wolfe line search), trained for 1000 iterations.

## Key Experimental Results

### Main Results: Transformer Architecture Comparison (Table 2)

| Model | PDE Type | rMAE | rMSE | Training Time |
|-------|----------|------|------|---------------|
| Pformer | Convection | 0.018 | 0.020 | 0:17:53 |
| Pformer | 1D-Reaction | 7.38e-3 | 0.163 | 0:03:59 |
| Pformer | 1D-Wave | 0.083 | 0.091 | 1:11:45 |
| Pformer | Navier-Stokes | 0.091 | 0.085 | 2:17:09 |
| DO-Pformer | Convection | 0.025 | 0.029 | 0:11:41 |
| DO-Pformer | 1D-Wave | 0.015 | 0.017 | 0:37:48 |
| **S-Pformer** | **Convection** | **0.016** | **0.018** | 0:14:29 |
| **S-Pformer** | **1D-Reaction** | **1.15e-3** | **2.98e-3** | 0:03:48 |
| **S-Pformer** | **1D-Wave** | **6.94e-3** | **7.01e-3** | 0:42:40 |
| **S-Pformer** | **Navier-Stokes** | **0.079** | **0.071** | 1:03:55 |

S-Pformer achieves better rMAE/rMSE than Pformer on all 4 benchmarks, with a 54% reduction in training time on Navier-Stokes.

### Ablation Study & Spectral Analysis (Table 3 + Table 1)

| Frequency Band | S-Pformer MAE | DO-Pformer MAE | Pformer MAE |
|----------------|---------------|----------------|-------------|
| Very Low ($f < 0.3f_n$) | 0.1401 | 0.1940 | 0.1400 |
| Low ($0.3f_n \leq f < 0.5f_n$) | **0.0904** | 0.1683 | 0.1764 |
| Mid ($0.5f_n \leq f < 0.7f_n$) | **0.0302** | 0.0354 | 0.0363 |
| High ($0.7f_n \leq f < 0.9f_n$) | **0.0110** | 0.0157 | 0.0155 |
| Very High ($f \geq 0.9f_n$) | **0.0093** | 0.0136 | 0.0133 |

Parameter count comparison: Pformer 453,561 → S-Pformer 369,039 (**18.6% reduction**). DO-Pformer, which merely replaces the encoder with a linear layer without Fourier features, exhibits noticeably higher errors in the low and mid frequency bands compared to S-Pformer, confirming the role of Fourier features in alleviating spectral bias.

### Optimized S-Pformer vs. Optimized MLP-PINN (Table 4)

| Problem | Model | rMAE | rMSE | Parameters |
|---------|-------|------|------|------------|
| Convection | MLP-PINN | 0.663 | 0.745 | 66,561 |
| Convection | **S-Pformer** | **0.015** | **0.018** | 305,551 |
| 1D-Reaction | MLP-PINN | 0.014 | 0.028 | 1,052,673 |
| 1D-Reaction | **S-Pformer** | **1.09e-3** | **2.15e-3** | 167,471 |
| 1D-Wave | MLP-PINN | 0.023 | 0.023 | 2,365,441 |
| 1D-Wave | **S-Pformer** | **2.89e-3** | **2.94e-3** | 247,823 |
| Navier-Stokes | **MLP-PINN** | **0.045** | **0.046** | 264,706 |
| Navier-Stokes | S-Pformer | 0.057 | 0.062 | 149,680 |

### Key Findings

1. On Convection, S-Pformer's rMAE is only 1/44 that of MLP-PINN, with fewer parameters
2. On 1D-Reaction, S-Pformer achieves a 13× accuracy improvement using only 1/6 of the parameters (167K vs. 1.05M)
3. High-frequency band error is reduced by approximately 30% (compared to DO-Pformer), directly demonstrating Fourier features' effectiveness against spectral bias
4. Navier-Stokes is the only problem where MLP performs slightly better, as it contains a data-driven component rather than purely physical constraints

## Highlights & Insights

- **"Subtractive" Design Philosophy**: Rather than stacking more modules, the method removes the redundant encoder and replaces it with a more targeted Fourier embedding, challenging the "bigger is better" paradigm
- **Explicit Resolution of Spectral Bias**: Random Fourier features project low-dimensional inputs into a multi-frequency space, fundamentally equipping the network with the capacity to represent high-frequency functions
- **Wavelet Activation Function**: A learnable weighted combination of sin/cos replaces ReLU, which is physically more compatible with the periodic characteristics of PDE solutions
- **NTK Adaptive Weighting**: Dynamically balances multi-task losses from a gradient sensitivity perspective, offering a more principled alternative to manual weight tuning

## Limitations & Future Work

1. **Slight Underperformance on Navier-Stokes**: For problems with data-driven components, the pure Transformer architecture has not yet fully demonstrated its advantage
2. **Hyperparameter Sensitivity**: The significant performance gains after optimization (e.g., 1D-Wave rMAE reduced from 6.94e-3 to 2.89e-3) suggest that default hyperparameters are suboptimal, necessitating Bayesian search
3. **Limited Problem Scale**: Evaluation is restricted to classical 1D/2D PDEs, with no coverage of high-dimensional, multi-physics, or complex-geometry problems
4. **Training Efficiency**: The method still relies on the L-BFGS optimizer and NTK weight computation, the latter introducing additional Jacobian computation overhead

## Related Work & Insights

- **PINNsformer (Zhao et al., 2024)**: The direct foundation of this work; an encoder-decoder Transformer PINN
- **Fourier Features (Tancik et al., 2020)**: Demonstrates that random Fourier features enable networks to learn high-frequency functions
- **NTK for PINNs (Wang et al., 2020/2021)**: Analyzes PINN training failures and spectral bias from the neural tangent kernel perspective
- **Insights**: The paradigm of Fourier embedding + decoder-only Transformer can be generalized to other scientific computing settings (molecular dynamics, weather forecasting, etc.)

## Rating

- Novelty: ⭐⭐⭐ — The core idea combines encoder removal with Fourier features; individual components have precedents, but the combination is meaningful
- Experimental Thoroughness: ⭐⭐⭐ — Covers 4 benchmarks, spectral analysis, ablation, and hyperparameter optimization; reasonably thorough but limited in scale
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete derivations, and well-designed ablation studies
- Value: ⭐⭐⭐ — Offers practical reference value to the PINN community, though application scope and broader impact are limited

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](../../ICLR2026/scientific_computing/empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)

</div>

<!-- RELATED:END -->
