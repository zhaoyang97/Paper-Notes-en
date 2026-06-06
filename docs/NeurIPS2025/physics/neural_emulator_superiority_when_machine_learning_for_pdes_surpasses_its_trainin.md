---
title: >-
  [Paper Note] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][PDE emulator] This work challenges the prevailing assumption that the accuracy of neural PDE emulators is bounded by that of their training data (i.e.…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "PDE emulator"
  - "numerical solver"
  - "emulator superiority"
  - "Fourier analysis"
  - "autoregressive rollout"
  - "inductive bias"
date: 2026-05-08
content_hash: 5143fdb0ed52ac9f
---

# Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data

**Conference**: NeurIPS 2025
**arXiv**: [2510.23111](https://arxiv.org/abs/2510.23111)  
**Code**: [tum-pbs.github.io/emulator-superiority](https://tum-pbs.github.io/emulator-superiority)  
**Area**: Scientific Computing / Neural PDE Solvers
**Keywords**: PDE emulator, numerical solver, emulator superiority, Fourier analysis, autoregressive rollout, inductive bias

## TL;DR

This work challenges the prevailing assumption that the accuracy of neural PDE emulators is bounded by that of their training data (i.e., the numerical solver). It discovers and rigorously defines the phenomenon of **emulator superiority**—neural networks trained solely on low-accuracy solver data can, when evaluated against high-accuracy reference solutions, outperform the very solver that generated their training data.

## Background & Motivation

**Rise of neural PDE emulators**: In recent years, replacing traditional numerical solvers with neural networks for simulating partial differential equations (PDEs) has become a prominent direction in scientific computing. Architectures such as FNO, UNet, and Transformer have demonstrated remarkable potential for accelerating simulations, achieving inference speedups of several orders of magnitude.

**Implicit assumption about training data**: Existing work almost universally uses the output of numerical solvers as training data. A widely accepted implicit assumption is that the accuracy of an emulator is capped by the accuracy of its training data—a student cannot surpass its teacher.

**Intrinsic errors in numerical solvers**: Numerical methods (finite difference, finite element, spectral methods, etc.) inherently introduce discretization errors. Different schemes (explicit/implicit, various orders) exhibit systematic biases in different regions of spectral space, a fact that has been largely overlooked in the emulator community.

**Paradox in evaluation**: If emulators can indeed surpass their training solvers, then using the training solver's output as the ground truth for evaluation is inherently unfair—high-quality emulators would be penalized precisely for being too accurate.

**Lack of theoretical explanation**: "Surpassing training data" appears counterintuitive, and prior sporadic observations (e.g., Kochkov et al., 2021) had not been systematically studied or theoretically explained.

**Paper goals**: To rigorously define emulator superiority, provide a complete theoretical explanation for linear PDEs via Fourier analysis, and conduct extensive empirical validation across multiple mainstream architectures on a nonlinear PDE (Burgers' equation).

## Method

### Overall Architecture

The paper constructs a three-level comparison framework:

- **Low-accuracy numerical solver** $P$: generates training data and is itself subject to discretization error
- **High-accuracy reference solution** $\tilde{P}$: serves as the true ground truth (e.g., via an extremely fine grid or analytical solution)
- **Neural emulator** $f_\theta$: trained on outputs of $P$, but evaluated against $\tilde{P}$

The **superiority ratio** is defined as:

$$\xi[t] = \frac{\mathbb{E}[\zeta(f_\theta^t(u),\, \tilde{P}^t(u))]}{\mathbb{E}[\zeta(P^t(u),\, \tilde{P}^t(u))]}$$

where $\zeta$ is an error metric (e.g., MSE) and $t$ denotes the number of time steps. When $\xi < 1$, the emulator outperforms its training solver.

Two distinct forms are further distinguished:

- **State-space superiority**: $\xi[1] < 1$, i.e., superiority is already achieved at a single prediction step
- **Autoregressive superiority**: $\xi[1] \geq 1$ but $\xi[t] < 1$ for some $t \geq 2$, i.e., the advantage emerges only through multi-step rollout

### Key Designs

**Fourier analysis framework**: A complete frequency-domain analysis is conducted for three linear PDEs:

1. **Advection equation**: The explicit upwind scheme introduces excessive numerical dissipation at high-frequency modes, while implicit schemes exhibit different phase/amplitude bias patterns. A simple two-parameter linear emulator (equivalent to a parameterized convolution kernel), when trained on implicit scheme data via MSE minimization, automatically learns a "blended" behavior that more closely approximates the true solution in high-frequency regions. This is termed **forward superiority**—the emulator outperforms the solver precisely in the frequency bands where the solver's error is largest.

2. **Diffusion equation**: A similar forward superiority pattern is observed. The explicit scheme over-attenuates high-frequency components, and the emulator automatically corrects for this bias.

3. **Poisson equation (iterative solver)**: When using a Jacobi iterative solver, low-frequency components converge most slowly. The emulator outperforms the training solver in the low-frequency band, manifesting as **backward superiority**—the advantage appears at the opposite end of the spectrum.

**Core mechanistic explanation**:

- Numerical schemes introduce **non-uniform** errors across different frequency components
- The MSE training objective encourages the emulator to learn the "globally optimal" approximation across all frequencies
- The inductive bias of convolutional structures naturally imposes spectral smoothing regularization
- Consequently, the emulator "sacrifices" accuracy in certain frequency bands but "compensates" more in others, yielding a solution overall closer to the high-accuracy reference

### Loss & Training

The emulator is trained with a standard single-step MSE loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{u \sim \mathcal{D}} \left[ \| f_\theta(u) - P(u) \|^2 \right]$$

The key insight is that although the training objective is to approximate $P$ (the low-accuracy solver), the spectral smoothing bias of convolutional networks combined with MSE's equal treatment of all frequencies causes the learned mapping $f_\theta$ to be closer to $\tilde{P}$ than $P$ in certain frequency bands.

## Key Experimental Results

### Main Results

Five mainstream architectures are evaluated on the **linear advection equation** and the **nonlinear Burgers' equation**:

| Architecture | Advection $\xi[1]$ | Advection $\xi[\text{auto}]$ | Burgers $\xi[1]$ | Burgers $\xi[\text{auto}]$ |
|------|:---:|:---:|:---:|:---:|
| ConvNet | < 1 ✓ | < 1 ✓ | < 1 ✓ | < 1 ✓ |
| Dilated ResNet | ≥ 1 | < 1 ✓ | ≥ 1 | < 1 ✓ |
| FNO | ≥ 1 | < 1 ✓ | ≥ 1 | < 1 ✓ |
| UNet | ≥ 1 | < 1 ✓ | ≥ 1 | < 1 ✓ |
| Transformer | ≥ 1 | < 1 ✓ | ≥ 1 | < 1 ✓ |

Core finding: **Nearly all architectures achieve autoregressive superiority**; ConvNet additionally achieves state-space superiority due to its local convolutional inductive bias.

### Ablation Study

**Spectral decomposition analysis**:

- The implicit upwind scheme is relatively accurate at low frequencies but exhibits systematic bias at high frequencies
- After training, the emulator's error spectrum at high frequencies is significantly lower than that of the training solver
- The theoretically predicted "superiority region" (frequency interval) closely matches experimental observations

**Analytical solution for the linear emulator**:

- For the two-parameter linear emulator $f_\theta(u) = a \cdot u + b \cdot P(u)$, the optimal parameters can be derived analytically
- It is theoretically proven that when the solver's frequency response exhibits systematic bias in certain modes, the optimal linear combination must outperform $P$ in those modes

**Cumulative effect of autoregressive rollout**:

- Errors from numerical solvers accumulate linearly or super-linearly over rollout steps
- Due to implicit regularization, the emulator's error accumulation rate is lower
- This explains why autoregressive superiority is more prevalent than state-space superiority

### Key Findings

1. **Emulator superiority is a universal phenomenon**, not limited to specific architectures or PDE types
2. **The locality bias of ConvNet** is particularly conducive to state-space superiority, as the local structure of PDEs naturally aligns with convolution kernels
3. **FNO operates in the frequency domain**, but due to its design of truncating high-frequency modes, it does not necessarily outperform the solver at a single step; however, advantages accumulate over multi-step rollout
4. **Results on Burgers' equation** (nonlinear) validate the generalizability of the theory to nonlinear settings
5. **The "free" nature of superiority**: no modifications to the training pipeline and no additional high-accuracy data are required—superiority emerges from standard MSE training alone

## Highlights & Insights

- **A paradigm-challenging finding**: The intuition that "a student cannot surpass its teacher" is fundamentally overturned. The systematic bias of numerical schemes combined with the implicit regularization of neural networks yields a free accuracy gain.
- **Elegant theory via Fourier analysis**: Frequency-domain tools precisely characterize the frequency intervals and conditions under which superiority occurs, providing a rigorous theoretical foundation for empirical observations.
- **Forward vs. backward superiority**: The direction of superiority is found to depend on the error characteristics of the numerical scheme—time integrators erring at high frequencies lead to forward superiority; iterative solvers erring at low frequencies lead to backward superiority. This dichotomy is a particularly insightful finding.
- **Profound implications for evaluation**: If emulators can surpass their training solvers, evaluating them using the training solver as ground truth is fundamentally flawed. This implies that the community needs to adopt independent high-accuracy reference solutions as evaluation ground truth.
- **Concise and comprehensive experimental design**: The argument progresses systematically from an analytically tractable two-parameter model to five mainstream deep architectures, and from three linear PDEs to the nonlinear Burgers' equation.

## Limitations & Future Work

1. **Theory covers only linear PDEs**: The Fourier analysis framework relies on the principle of linear superposition; for nonlinear PDEs, only empirical validation rather than rigorous proof is provided
2. **Primarily one-dimensional**: Experiments are conducted mainly in 1D; superiority behavior in 2D/3D settings with complex geometries remains to be investigated
3. **Turbulence and chaotic systems not addressed**: In high-Reynolds-number Navier-Stokes and other chaotic PDEs, numerical errors grow exponentially, and whether superiority still holds is an open question
4. **Training cost not quantified**: While inference accuracy is discussed, the trade-off between training data volume and computational cost versus directly using a high-accuracy solver is not thoroughly analyzed
5. **Adaptive and high-order schemes**: Only low-order numerical schemes are tested; superiority analysis for high-accuracy methods such as AMR and high-order spectral methods is absent
6. **Practical deployment**: How to translate the findings into concrete pipeline improvements in real engineering settings (e.g., actively selecting solvers that are "favorable for being surpassed") remains an open problem

## Related Work & Insights

- **Kochkov et al. (2021)**: Among the first to observe emulator-surpassing-training-data behavior in the context of learned corrections, but without systematic study
- **FNO (Li et al., 2021)**: A frequency-domain PDE emulator; this paper provides detailed analysis of its superiority behavior
- **Neural Operator theory**: Frameworks such as DeepONet and Neural Operator provide theoretical foundations for operator learning; this paper contributes a new perspective from the angle of error analysis
- **Error theory in numerical methods**: The paper elegantly bridges classical numerical analysis (Modified Equation Analysis, Fourier stability analysis) with deep learning theory
- **Directions for future inspiration**: (1) Designing "superiority-friendly" training data generation strategies; (2) using the superiority ratio as an architecture selection criterion; (3) extending the Fourier analysis to local linearization of nonlinear operators

## Rating

- ⭐ Novelty: 5/5 — First systematic definition and theoretical explanation of the emulator superiority phenomenon
- ⭐ Theoretical Depth: 5/5 — The Fourier analysis framework is elegant and verifiable; the forward/backward superiority dichotomy is highly insightful
- ⭐ Experimental Thoroughness: 4/5 — Multi-architecture, multi-PDE validation is comprehensive, though primarily limited to 1D
- ⭐ Value: 4/5 — Directly informs evaluation paradigms and benchmark design
- ⭐ Writing Quality: 5/5 — The argumentation is well-structured and progressively developed from theory to experiment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/physics/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)

</div>

<!-- RELATED:END -->
