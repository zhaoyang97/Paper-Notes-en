---
title: >-
  [Paper Note] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity
description: >-
  [Physics & Scientific Computing] This work establishes empirical scaling laws for single-layer PINNs on representative nonlinear PDEs…
tags:
  - "Physics & Scientific Computing"
date: 2026-05-08
content_hash: 7e43197bf1c9f351
---

# Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity

## Metadata
- **Conference**: NeurIPS 2025
- **arXiv**: [2603.12556](https://arxiv.org/abs/2603.12556)
- **Code**: [GitHub](https://github.com/farischaudhry/pinn-width-vs-nonlinearity)
- **Area**: Medical Imaging
- **Keywords**: Physics-Informed Neural Networks, Scaling Laws, Optimization Pathologies, Spectral Bias, PDE Solving

## TL;DR
This work establishes empirical scaling laws for single-layer PINNs on representative nonlinear PDEs, identifying a dual optimization failure: a width-scaling pathology (error does not decrease with width) and a compound pathology (nonlinearity exacerbates this failure), demonstrating that optimization rather than approximation capacity is the primary bottleneck.

## Background & Motivation

Physics-Informed Neural Networks (PINNs) embed physical equations into the loss function to solve PDEs in a mesh-free manner, representing an important paradigm in computational science. However, the relationship among model capacity, problem complexity, and solution accuracy lacks systematic quantification:

**Theory–Practice Gap**: The Universal Approximation Theorem (UAT) guarantees that single-layer networks possess the expressive power to approximate any continuous function, and Barron space theory suggests that the error should decay as $\mathcal{O}(N^{-1/2})$ with network width. These results, however, only guarantee existence and do not ensure that gradient-based optimization can find good approximations.

**Spectral Bias**: Gradient-based optimization tends to fit low-frequency components first, which becomes increasingly problematic for PDE solutions with stronger high-frequency components.

**Absence of Scaling Laws**: Influential scaling laws have been established in language and vision domains, yet the PINN community lacks an analogous quantitative framework.

**Core Hypotheses**: (i) In practice, the width-scaling exponent $\alpha \neq 0.5$ (baseline pathology); (ii) a separable power law $\text{error} \approx A \cdot N^{-\alpha} \cdot \kappa^\gamma$ is insufficient to describe scaling behavior, as $\alpha$ is itself a function of $\kappa$ (compound pathology).

## Method

### Overall Architecture

Single-layer networks (SLNs) are employed to isolate width effects. A systematic parameter sweep is conducted over three representative nonlinear PDEs, measuring the relationship between error and network width as well as problem difficulty. Scaling laws are then fitted and the mechanisms of optimization failure are analyzed.

### Key Designs

1. **PDE Test Suite and Difficulty Parameter $\kappa$**:

    - **Poisson Equation** (linear baseline): $-u_{xx} = \sin(\pi x)$, no $\kappa$ dependence; used to validate the framework.
    - **KdV Equation** (dispersive): $u_t + \kappa u u_x + u_{xxx} = 0$, where $\kappa = A$ is the soliton amplitude, controlling soliton speed and sharpness.
    - **Sine-Gordon Equation** (hyperbolic/transcendental): $u_{tt} - u_{xx} + \kappa \sin(u) = 0$, where $\kappa$ scales the strength of the nonlinear potential term.
    - **Allen-Cahn Equation** (reaction-diffusion/parabolic): $u_t - Du_{xx} + (u^3 - u) = 0$, with $\kappa = 1/D$; smaller diffusion yields sharper interfaces.

   Across all three nonlinear cases, increasing $\kappa$ strengthens high-frequency components in the true solution, directly challenging the network's spectral bias.

2. **Systematic Parameter Sweep**:

    - Network width: $N \in \{16, 32, 64, 128, 256, 512, 1024\}$
    - Difficulty parameter: 7 log-spaced values per PDE
    - Activation functions: tanh and ReLU
    - Random seeds: 5 seeds per configuration
    - Total experiments per nonlinear PDE: $7 \times 7 \times 2 \times 5 = 490$

3. **Training Setup**: All models are trained with the Adam optimizer at learning rate $10^{-3}$ for 25,000 epochs, with equal-weight ($w=1$) composite loss:
    $\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{bc}} \mathcal{L}_{\text{bc}} + w_{\text{ic}} \mathcal{L}_{\text{ic}}$

### Analysis Methods

- **Univariate Analysis**: For each fixed $\kappa$, fit $\text{error} \approx A N^{-\alpha}$ and analyze $\alpha(\kappa)$.
- **Multivariate Separable Model**: Fit $\text{error} \approx A \cdot N^{-\alpha} \cdot \kappa^\gamma$ to quantify average width and difficulty effects.
- **Non-separable Interaction Model**: Include interaction terms to test whether $\alpha$ depends on $\kappa$.

The error metric is the relative $L_2$ error: $\|\hat{u} - u_{\text{true}}\|_2 / \|u_{\text{true}}\|_2$.

## Key Experimental Results

### Main Results: Separable Power-Law Regression

| PDE | Activation | Width Exponent $\alpha$ (95% CI) | Difficulty Exponent $\gamma$ (95% CI) | log(A) | Adj $R^2$ |
|-----|------|--------------------------|--------------------------|--------|-----------|
| KdV | ReLU | -0.05±0.03** | 0.17±0.04*** | -0.66 | 0.65 |
| KdV | Tanh | 0.00±0.01 | 0.18±0.01*** | -0.27 | 0.51 |
| Sine-Gordon | ReLU | -0.32±0.07*** | 0.28±0.08*** | -3.40 | 0.70 |
| Sine-Gordon | Tanh | -0.14±0.34 | 1.51±0.39*** | -7.30 | 0.52 |
| Allen-Cahn | ReLU | -0.37±0.08*** | -0.44±0.10*** | -3.36 | 0.74 |
| Allen-Cahn | Tanh | -0.02±0.11 | -0.03±0.12 | -6.78 | -0.03 |

### Ablation Study: Per-$\kappa$ Width Scaling Exponents (Sine-Gordon Representative)

| Difficulty $\kappa$ | ReLU $\alpha$ (95% CI) | Tanh $\alpha$ (95% CI) |
|--------------|----------------------|----------------------|
| 0.25 | -0.37±0.14 | -0.45±0.19 |
| 0.50 | -0.41±0.08 | -0.38±0.26 |
| 1.00 | -0.43±0.04 | -0.43±0.20 |
| 2.00 | -0.49±0.10 | -0.48±0.20 |
| 4.00 | -0.40±0.03 | -0.22±0.24 |
| 8.00 | -0.16±0.09 | 0.94±0.70 |
| 16.00 | 0.03±0.16 | 0.03±0.01 |

Fitting each $\kappa$ separately consistently yields $\alpha$ near zero or negative, confirming that the baseline pathology is pervasive across all conditions.

### Key Findings

1. **Width-Scaling Pathology is Universal**: Across all tested PDEs and activation functions, $\alpha$ is consistently near zero or negative; the conventional intuition that "wider = better" does not hold for PINNs.
2. **Catastrophic Failure of ReLU**: On the Poisson equation, the error remains constant at ~1.0 ($\alpha \approx 0.01$), because the second derivative of ReLU is a collection of Dirac delta functions and cannot represent the smooth, continuous derivatives required by the PDE loss.
3. **Nonlinearity is a More Dominant Factor than Width**: Varying $\kappa$ can change the error by several orders of magnitude, whereas varying $N$ typically yields less than one order of magnitude of change.
4. **Activation Functions Determine the Nature of Interactions**: For ReLU, the interaction term is statistically significant across all cases ($\kappa$-dependent stiffness); for tanh, it is not—the width effect of tanh is entirely negligible.
5. **Allen-Cahn Anomaly**: $\gamma$ is negative (ReLU) or statistically insignificant (tanh), suggesting that reaction-diffusion/parabolic PDEs may exhibit qualitatively different failure mechanisms.

## Highlights & Insights

1. **Counterintuitive Core Finding**: "Wider networks perform worse"—directly challenging the prevailing heuristic in deep learning that width benefits optimization.
2. **From Qualitative Description to Quantitative Measurement**: Prior work (e.g., Krishnapriyan et al.) qualitatively identified PINN failure modes; this paper is the first to quantify them precisely using scaling laws.
3. **Separable Models Are Insufficient**: The paper demonstrates that simple power laws cannot capture the coupled width × nonlinearity effects, revealing the intrinsic complexity of the PINN optimization landscape.
4. **Call to Action**: The authors explicitly state that the primary purpose is not to resolve the identified problems, but to advocate for establishing similar scaling studies across broader settings.

## Limitations & Future Work

- Only single-layer networks and the Adam optimizer are considered; the generalizability of findings to multi-layer networks and second-order methods remains unknown.
- Only one-dimensional spatial settings with three PDEs are tested; behavior in higher dimensions and with more complex PDEs may differ.
- The paper does not explore whether known mitigation strategies (Fourier features, adaptive weighting, domain decomposition) can "cure" the identified pathologies.
- The anomalous behavior observed for Allen-Cahn requires deeper theoretical explanation.
- Training epochs are fixed at 25,000; scaling behavior may change under different training budgets.

## Related Work & Insights

- **Krishnapriyan et al. (NeurIPS 2021)**: First systematic characterization of PINN failure modes—optimization rather than expressive power is the bottleneck.
- **Wang et al. (2022)**: Theoretical analysis of gradient flow pathologies and spectral bias.
- **Bonfanti et al. (NeurIPS 2023)**: Standard NTK theory is misleading for nonlinear PDEs.
- **Neural Scaling Laws (Kaplan et al.)**: The paradigm of scaling laws for language models.
- **Insight**: The nature of the PINN optimization landscape is fundamentally different from that of conventional deep learning; general scaling intuitions cannot be straightforwardly transferred.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to establish empirical scaling laws for PINNs and identify the dual pathology.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Systematic sweep is thorough, but limited to single-layer networks, one-dimensional settings, and three PDEs.
- Writing Quality: ⭐⭐⭐⭐⭐ — Argumentation is logically clear, progressing systematically from hypotheses to validation.
- Value: ⭐⭐⭐⭐☆ — Establishes an important quantitative benchmark and methodological template for the PINN community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](../../ICLR2026/physics/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)
- [\[NeurIPS 2025\] Symbolic Regression Is All You Need: From Simulations to Scaling Laws in Binary Neutron Star Mergers](symbolic_regression_is_all_you_need_from_simulations_to_scaling_laws_in_binary_n.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)

</div>

<!-- RELATED:END -->
