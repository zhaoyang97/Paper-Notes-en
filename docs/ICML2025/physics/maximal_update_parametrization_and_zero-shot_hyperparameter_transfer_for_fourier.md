---
title: >-
  [Paper Note] Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators
description: >-
  [ICML2025][Physics & Scientific Computing][μP] This work derives the Maximal Update Parametrization (μP) for the Fourier Neural Operator (FNO) for the first time, enabling zero-shot transfer of hyperparameters tuned on small models to billion-parameter FNOs, which reduces the tuning computational cost for Navier-Stokes problems to 0.30×.
tags:
  - "ICML2025"
  - "Physics & Scientific Computing"
  - "μP"
  - "μTransfer"
  - "FNO"
  - "Hyperparameter Transfer"
  - "PDE Solving"
  - "Fourier Mode Scaling"
date: 2026-05-08
content_hash: 03d7070aeedba5fe
---

# Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators

**Conference**: ICML2025  
**arXiv**: [2506.19396](https://arxiv.org/abs/2506.19396)  
**Code**: [LithiumDA/muTransfer-FNO](https://github.com/LithiumDA/muTransfer-FNO)  
**Area**: Scientific Computing / Fourier Neural Operator  
**Keywords**: μP, μTransfer, FNO, Hyperparameter Transfer, PDE Solving, Fourier Mode Scaling

## TL;DR

This work derives the Maximal Update Parametrization (μP) for the Fourier Neural Operator (FNO) for the first time, enabling zero-shot transfer of hyperparameters tuned on small models to billion-parameter FNOs, which reduces the tuning computational cost for Navier-Stokes problems to 0.30×.

## Background & Motivation

FNOs learn the solution operator mapping of PDEs by performing kernel integration on low-frequency components in the frequency domain, serving as the most dominant neural operaor architecture currently. Its expressivity is directly related to the number of Fourier modes $K$:

- For a $d$-dimensional PDE, the parameter size of the kernel integration is $\mathcal{O}(K^d)$, and increasing $K$ causes parameter explosion.
- Example: A 4-layer FNO-3D increasing $K$ from 3 to 24 results in parameter growth from 1.7M to 906M.
- Hyperparameter tuning (learning rate, batch size, optimizer parameters) on large models is computationally prohibitive.

**Core Problem**: Is it possible to tune hyperparameters on a small FNO and transfer them losslessly to a large FNO?

Existing μP theories cover width scaling ($\Theta(m^{-1})$) and depth scaling ($\Theta(L^{-1/2})$) of MLPs. However, the working mechanism of the kernel integral operator in FNOs is fundamentally different from standard linear layers, and its μP has not been previously derived.

## Method

### 1. FNO Architecture Review

The forward pass of FNO:

$$\mathcal{G}_\theta = \mathcal{Q} \circ \phi(W_L + \mathcal{K}_L) \circ \cdots \circ \phi(W_1 + \mathcal{K}_1) \circ \mathcal{P}$$

where the kernel integral operator is:

$$(\mathcal{K}v)(x) = \mathcal{F}^{-1}\left[\boldsymbol{R} \cdot \mathcal{T}_K(\mathcal{F}v)\right](x)$$

- $\mathcal{F}$/$\mathcal{F}^{-1}$: $d$-dimensional FFT and its inverse
- $\mathcal{T}_K$: Truncation to the lowest $K$ Fourier modes
- $\boldsymbol{R} \in \mathbb{R}^{K^d \times m \times m}$: Learnable parameter tensor

### 2. Generalization of abc-Parametrization

The parametrization of $\boldsymbol{R}$ is defined using three scaling functions $a(K), b(K), c(K)$:

| Function | Meaning |
|------|------|
| $a(K)$ | Parameter scaling: $\boldsymbol{R} = a(K)\boldsymbol{r}$ |
| $b(K)$ | Initialization variance: $\boldsymbol{r}_{ij} \sim \mathcal{N}(0, b(K)^2)$ |
| $c(K)$ | Learning rate scaling: $\eta = c(K)\eta_0$ |

Under standard parametrization, $a=1, b=\Theta(1), c=1$, and the optimal learning rate drifts with $K$.

### 3. Main Theorem: μP for FNO

**Theorem 3.5 (Main Theoretical Result)**: Under tanh/GELU activation and sub-Gaussian gradient update assumptions, the μP for FNO under the Adam optimizer is given by:

$$a(K)=1, \quad b(K) = c(K) = \Theta\!\left(\frac{1}{\sqrt{d\log K}}\right)$$

That is:

- **Initialization variance** scales as $\Theta(1/(d\log K))$
- **Learning rate** scales as $\Theta(1/\sqrt{d\log K})$

This is entirely different from the existing width scaling of $\Theta(m^{-1})$ and depth scaling of $\Theta(L^{-1/2})$. The source of the difference lies in the fact that prior derivations rely on the Central Limit Theorem of random variable means, whereas the spectral norm of the FNO kernel integral depends on the **maximum** of $K^d$ sub-Gaussian random variables, which naturally yields the $\sqrt{d\log K}$ term.

### 4. μTransfer-FNO Algorithm

1. Perform grid search for optimal hyperparameters $\xi^*$ on a small proxy model ($K_{\text{proxy}}$).
2. Transfer them to the target model ($K^*$) according to the scaling rules:
    - Learning rate: $\eta_{K^*} = \sqrt{\frac{\log K_{\text{proxy}}}{\log K^*}} \cdot \eta_{K_{\text{proxy}}}$
    - Initialization variance: $\sigma^2_{K^*} = \frac{\log K_{\text{proxy}}}{\log K^*} \cdot \sigma^2_{K_{\text{proxy}}}$
3. Train the large model directly with the transferred hyperparameters.

Key property: The scaling rules are independent of the input discretization $N_1 \times \cdots \times N_d$, preserving the resolution-invariant property of FNO.

## Key Experimental Results

### Experimental Setup

| PDE Problem | Model | Dimension $d$ | $K$ Range | Parameter Range |
|----------|------|---------|---------|-----------|
| Burgers Equation | FNO-1D | 1 | 3→512 | — |
| Darcy Flow | FNO-2D | 2 | 3→24 | — |
| Navier-Stokes | FNO-3D | 3 | 3→24 | 1.7M→906M |

All models: 4 layers, 64 hidden dimensions, GELU activation, Adam optimizer.

### Learning Rate Transfer Performance

- **Standard Parametrization**: The optimal learning rate drifts significantly with $K$ (e.g., FNO-3D drifts from $1.8\times10^{-3}$ to $7.4\times10^{-4}$).
- **μTransfer-FNO**: The optimal learning rate remains stable across different $K$ (e.g., around $4.2\times10^{-3}$ for FNO-3D).

### End-to-End Performance Comparison

| Method | Darcy Flow $L^2$ Error | Training Cost | NS Equation $L^2$ Error | Training Cost |
|------|----------------------|---------|-------------------|---------|
| Direct Tuning on Large Model | 1.25% | 1× | 5.69% | 1× |
| **μTransfer-FNO** | **1.22%** | 1.38× | **5.34%** | **0.30×** |

Achieved a win-win result of lower error + only 0.30× computational cost on Navier-Stokes.

### More Hyperparameter Transfers

The transfer of batch size and Adam $\beta_2$ was validated on Darcy Flow (FNO-2D):

- **Batch size**: Under μTransfer, the optimal batch size remains consistently 20, whereas standard parametrization requires a larger batch size for larger models.
- **$\beta_2$**: Under μTransfer, the optimal $\beta_2$ is always 0.98, while under standard parametrization it varies with $K$.

### Generalization to PINO

It remains effective under the Physics-Informed Neural Operator (PINO) training mode:
- PINO introduces additional physics-constraint losses, making training dynamics more complex.
- μTransfer-PINO's optimal learning rate stabilizes at $5.6\times10^{-3}$ on Darcy Flow.

## Highlights & Insights

1. **Novel Theory**: Evaluates and extends the μP framework to the FNO kernel integral operator for the first time, discovering that the scaling rate $\Theta(1/\sqrt{d\log K})$ is fundamentally different from existing results (width/depth scaling).
2. **Technical Depth**: The core of the derivation lies in analyzing the maximum rather than the mean of $K^d$ sub-Gaussian variables, which enriches the theoretical toolbox of μP.
3. **Practical Value**: Successfully validates zero-shot hyperparameter transfer on FNOs with nearly 1 billion parameters, saving 70% of computational cost on the Navier-Stokes equations.
4. **Strong Generalization**: Proven effective across multiple hyperparameters like learning rate, batch size, and $\beta_2$, and compatible with the PINO training paradigm.
5. **Resolution Independence**: The scaling rules do not depend on spatial discretization, maintaining the core advantage of FNO.

## Limitations & Future Work

1. **Scaling Only for $K$**: Joint scaling of width $m$ and depth $L$ is not covered, whereas in practice, all three might vary simultaneously.
2. **sub-Gaussian Assumption**: Requires element-wise clipping of gradients (clip=0.01) to satisfy theoretical assumptions, introducing an additional hyperparameter to tune.
3. **Limited PDE Types**: Only validated on three classic PDEs (Burgers, Darcy, NS). Its applicability to more complex or higher-dimensional problems remains to be verified.
4. **Instability at Small $K$**: Experiments show that the optimal learning rate occasionally shifts at very small $K$, suggesting asymptotic theory introduces errors in small models.
5. **Limited to FNO Architecture**: The theory has not yet been extended to other architectures like DeepONet or Transformer-based operators.

## Related Work & Insights

- **μP/μTransfer Series**: Yang & Hu (2021) $\rightarrow$ Yang et al. (2022, 2024); this work represents the first application of this theory to the neural operator field.
- **iFNO** (George et al., 2024): Incrementally increases Fourier modes to improve training efficiency, which is complementary to this work.
- **Insights**: For other frequency-domain parametrized models (such as spectral methods), a similar $\log K$ scaling mechanism might hold true.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of μP theory to neural operators, with a scaling rate fundamentally different from existing results.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three kinds of PDEs, multiple hyperparameters, and PINO extension, though the variety of PDE types could be wider.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivation, clear experiments, and well-defined motivation.
- Value: ⭐⭐⭐⭐ — Directly practical for large-scale FNO training, though limited to the FNO architecture.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators](../../ICLR2026/physics/the_false_promise_of_zero-shot_super-resolution_in_machine-learned_operators.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](../../NeurIPS2025/physics/oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[ICLR 2026\] Extending Fourier Neural Operators for Modeling Parameterized and Coupled PDEs](../../ICLR2026/physics/extending_fourier_neural_operators_for_modeling_parameterized_and_coupled_pdes.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[CVPR 2025\] DiffFNO: Diffusion Fourier Neural Operator](../../CVPR2025/physics/difffno_diffusion_fourier_neural_operator.md)

</div>

<!-- RELATED:END -->
