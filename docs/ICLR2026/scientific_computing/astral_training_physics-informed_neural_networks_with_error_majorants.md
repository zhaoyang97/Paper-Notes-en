---
title: >-
  [Paper Note] Astral: Training Physics-Informed Neural Networks with Error Majorants
description: >-
  [ICLR 2026][Scientific Computing][PiNN] This paper proposes the Astral loss function — based on a functional a posteriori error majorant — as a replacement for the conventional residual loss in training physics-informed neural networks (PiNNs). The approach enables reliable error estimation throughout training and achieves superior or comparable accuracy across multiple PDE types, including diffusion and Maxwell equations.
tags:
  - ICLR 2026
  - Scientific Computing
  - PiNN
  - a posteriori error estimation
  - error upper bound
  - PDE solving
  - loss function design
date: 2026-05-08
content_hash: e43ea2bf28e3bd71
---

# Astral: Training Physics-Informed Neural Networks with Error Majorants

**Conference**: ICLR 2026
**arXiv**: [2406.02645](https://arxiv.org/abs/2406.02645)
**Code**: [https://github.com/4gnskq5g2s-collab/Astral](https://github.com/4gnskq5g2s-collab/Astral)
**Area**: Scientific Computing / Physics-Informed Neural Networks
**Keywords**: PiNN, a posteriori error estimation, error upper bound, PDE solving, loss function design

## TL;DR
This paper proposes the Astral loss function — based on a functional a posteriori error majorant — as a replacement for the conventional residual loss in training physics-informed neural networks (PiNNs). The approach enables reliable error estimation throughout training and achieves superior or comparable accuracy across multiple PDE types, including diffusion and Maxwell equations.

## Background & Motivation
**Background**: PiNNs approximate PDE solutions with neural networks, most commonly by minimizing the PDE residual (the $L_2$ norm evaluated at randomly sampled collocation points).

**Limitations of Prior Work**:
- The correlation between the residual and the true error is extremely poor. The authors demonstrate this with a simple BVP: one can construct examples where the residual is arbitrarily large yet the error is arbitrarily small, and vice versa.
- Statistical experiments over 100 diffusion equation instances show that the average spatial correlation between the residual and the energy-norm error is only $0.22 \pm 0.09$.
- The residual cannot reliably indicate the accuracy of an approximate solution, leaving no principled stopping criterion for training.

**Key Challenge**: The residual is at best an indirect proxy for the error rather than a direct upper bound, depriving PiNNs of reliable a posteriori error control.

**Goal**: Design a new loss function that simultaneously trains PiNNs effectively and provides a rigorous upper bound on the approximation error.

**Key Insight**: Leverage classical functional a posteriori error estimates from numerical analysis. Such estimates are approximation-agnostic and thus naturally suited to neural network approximations.

**Core Idea**: Use the PDE's error majorant (an error-bounding functional) directly as the PiNN training loss, thereby obtaining both a high-quality approximate solution and a reliable error certificate simultaneously.

## Method

### Overall Architecture
Given a PDE $\mathcal{A}[\phi, \mathcal{D}] = 0$, an error majorant $U[\tilde{\phi}, \mathcal{D}, w] \geq E[\tilde{\phi} - \phi]$ is derived in the energy norm, where $\tilde{\phi}$ is the approximate solution and $w$ is an auxiliary free function. Two independent neural networks parameterize $\tilde{\phi}$ and $w$ respectively, and both are jointly trained by minimizing $U$. At the end of training, the value of $U$ directly yields an upper bound on the approximation error.

### Key Designs

1. **Astral Loss Function**:

   - **Function**: Employs the error majorant as the PiNN loss.
   - **Mechanism**: Taking the diffusion equation as an example, an auxiliary variable $\tilde{F}(x,y) \simeq \sigma(x,y)\operatorname{grad}\phi(x,y)$ is introduced to approximate the exact flux. The loss takes the form $U = \alpha \int (f + \operatorname{div}\tilde{F})^2 + \beta \int \|\sigma\operatorname{grad}\tilde{\phi} - \tilde{F}\|^2 / \sigma$, where $\alpha$ and $\beta$ are constants depending on PDE parameters.
   - **Design Motivation**: $U$ is a rigorous upper bound on the energy-norm error; it is saturated if and only if $\tilde{\phi} \to \phi$ and $\tilde{F} \to \sigma\operatorname{grad}\phi$. Minimizing $U$ therefore simultaneously drives the approximate solution toward the exact solution and the auxiliary field toward the exact flux.
   - **Novelty**: The residual loss guarantees only a small residual, not a small error; variational losses require the problem to admit a variational formulation; the Astral loss provides a rigorous upper bound together with high accuracy.

2. **Auxiliary Field Parameterization**:

   - **Function**: Parameterizes the auxiliary field $w$ with an independent Siren network.
   - **Mechanism**: The dimension and interpretation of $w$ depend on the specific PDE — it is a flux vector field for the diffusion equation and a scalar field for Maxwell equations. Each field is represented by a dedicated Siren network.
   - **Design Motivation**: The auxiliary field is a free variable in the error majorant; optimizing it tightens the upper bound.

3. **Error Majorant Derivation for Multiple PDE Classes**:

   - **Function**: Explicit error majorant expressions are derived for seven PDE types.
   - **Coverage**: Isotropic/anisotropic diffusion, diffusion with large mixed derivatives, diffusion on an L-shaped domain, Maxwell equations (both $\alpha > 0$ and $\alpha = 0$), convection-diffusion, and nonlinear elasto-plasticity.
   - **Mechanism**: Rigorous upper bounds are derived from integral identities using tools such as the Cauchy–Schwarz inequality and the Friedrichs inequality.

4. **Error Indicator**:

   - **Function**: Extracts a pointwise spatial distribution estimate of the error from the Astral loss.
   - **Mechanism**: The error indicator $\|\sigma^{-1/2}(\tilde{F} - \sigma\operatorname{grad}\tilde{\phi})\|^2$ provides an estimate of the local error density.
   - **Key Advantage**: Its spatial correlation with the true error reaches $0.82 \pm 0.04$, compared to only $0.22 \pm 0.09$ for the residual.

### Loss & Training
- Integrals are approximated via Monte Carlo sampling over random subsets of a $64 \times 64$ uniform grid.
- Siren networks with 50–100 hidden neurons and 3–5 layers.
- Lion optimizer with learning rates from $10^{-3}$ to $10^{-4}$ and exponential learning rate decay.
- 50,000 weight updates with a batch size of $16 \times 16$.
- No second-order derivatives are required (unlike residual losses), leading to faster computation.

## Key Experimental Results

### Main Results — Anisotropic Diffusion Equation (averaged over 100 random instances)

| Anisotropy $\epsilon$ | Residual Relative $L_2$ (%) | Astral Relative $L_2$ (%) | Astral Majorant ($\times 10^2$) |
|---|---|---|---|
| 1 | 0.13±0.07 | **0.11±0.05** | 0.13±0.03 |
| 5 | 0.63±0.27 | **0.53±0.19** | 0.09±0.02 |
| 10 | 1.65±0.92 | **0.97±0.57** | 0.11±0.03 |
| 15 | 3.16±1.74 | **2.08±1.24** | 0.12±0.04 |
| 20 | 5.64±3.18 | **3.60±2.18** | 0.13±0.06 |

### Ablation Study — Maxwell Equations Comparison

| Method | Relative $L_2$ (%) | Training Time (s, 100 networks) |
|------|-------------|-------------------|
| Residual (small net) | 5.49±2.35 | 298 |
| Residual (large net) | — | 1176 |
| **Astral (small net)** | **0.45±0.16** | **105** |
| Astral (large net) | — | 481 |

### Key Findings
- **Largest advantage on Maxwell equations**: The relative error is reduced by an order of magnitude (5.49% → 0.45%), with training 3–10× faster.
- **Advantage grows with anisotropy**: At $\epsilon = 20$, Astral's error is 36% lower than that of the residual loss.
- **Upper bounds are generally tight**: The average overestimation factor is 1.5 for anisotropic diffusion, 1.7 for convection-diffusion, and approximately 10 for Maxwell equations.
- **L-shaped domain is a weakness of Astral**: Due to geometric singularities, the residual loss performs better on this problem.
- **Elimination of second-order derivatives** is the primary reason for Astral's faster training.

## Highlights & Insights
- **Training as error estimation**: The most significant feature is that training itself minimizes an upper bound on the error, so a reliable error certificate is obtained automatically upon completion — this is highly valuable for engineering computations.
- **Spatial correlation of the error indicator**: Astral's error indicator accurately localizes regions of high error concentration (correlation 0.82 vs. 0.22 for the residual), enabling adaptive refinement strategies.
- **Avoidance of second-order derivatives**: By introducing an auxiliary field to approximate the flux, Astral requires only first-order derivatives, reducing the computational overhead of automatic differentiation.

## Limitations & Future Work
- The error majorant must be derived manually for each PDE class; the process cannot be automated.
- Performance is inferior to the residual loss on problems with geometric singularities such as the L-shaped domain.
- The auxiliary field increases the number of network parameters and overall training complexity.
- Validation is currently limited to 2D problems; scalability to 3D settings remains unexplored.
- The upper bound overestimates significantly on certain problems (approximately 10× for Maxwell equations), limiting its practical utility in those cases.

## Related Work & Insights
- **vs. Residual Loss**: The residual loss is simple but poorly correlated with the true error; Astral provides a rigorous upper bound and is generally more accurate and faster.
- **vs. Variational Loss**: The variational loss is applicable to diffusion equations but yields considerably lower accuracy and does not extend to non-variational problems such as Maxwell equations.
- **vs. FEM A Posteriori Estimates**: Traditional FEM a posteriori estimates are mesh-dependent; Astral is based on functional estimates and is independent of any discretization method.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Bridging classical a posteriori error estimation with PiNN training is a genuinely novel contribution, although error majorants themselves are well-established in numerical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven PDE types, statistical evaluation over 100 random instances, comparisons across multiple loss functions, and training time analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is presented exceptionally clearly (Figure 1 is immediately informative); theoretical derivations are rigorous and readable.
- **Value**: ⭐⭐⭐⭐ Practically significant for the PiNN community; reliable error estimation is a critical requirement for engineering applications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/scientific_computing/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[ICLR 2026\] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery](empirical_stability_analysis_of_kolmogorov-arnold_networks_in_hard-constrained_r.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](../../NeurIPS2025/scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/scientific_computing/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)

<!-- RELATED:END -->
