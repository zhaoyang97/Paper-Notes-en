---
title: >-
  [Paper Note] Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations
description: >-
  [ICML 2025][Physics & Scientific Computing][Symbolic Regression] This paper proposes the SymPDE framework, which utilizes deep reinforcement learning to directly search for closed-form symbolic solutions to PDEs, bypassing the issues of insufficient numerical accuracy and poor interpretability of PINNs. It achieves a 90% recovery rate on Poisson and heat equations.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "Symbolic Regression"
  - "Partial Differential Equations"
  - "Deep Reinforcement Learning"
  - "Closed-form Solutions"
  - "PINNs"
date: 2026-05-08
content_hash: 3a5f0917a287d9f6
---

# Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations

**Conference**: ICML 2025  
**arXiv**: [2405.14620](https://arxiv.org/abs/2405.14620)  
**Code**: None  
**Area**: Reinforcement Learning / Scientific Computing  
**Keywords**: Symbolic Regression, Partial Differential Equations, Deep Reinforcement Learning, Closed-form Solutions, PINNs

## TL;DR
This paper proposes the SymPDE framework, which utilizes deep reinforcement learning to directly search for closed-form symbolic solutions to PDEs, bypassing the issues of insufficient numerical accuracy and poor interpretability of PINNs. It achieves a 90% recovery rate on Poisson and heat equations.

## Background & Motivation
**Background**: Partial differential equations (PDEs) are widespread in physics, mathematics, and other fields, and solving them is a core problem. Traditional analytical methods (e.g., Green's functions) are almost impractical for non-linear PDEs. Although PINNs utilize deep learning to provide numerical solutions, they are essentially approximations in continuous function spaces.

**Limitations of Prior Work**: (a) PINNs lack sufficient accuracy when fitting high-frequency and sharp-gradient functions, exhibiting large deviations in oscillating or distorted regions; (b) Neural operator methods (e.g., DeepONet, FNO) require a large amount of labeled data; (c) All neural network-based approaches yield numerical solutions, lacking interpretability and demonstrating poor extrapolation capability.

**Key Challenge**: Numerical solutions vs. symbolic solutions—symbolic solutions naturally possess accuracy, interpretability, and extrapolation capabilities, but they suffer from a massive search space and optimization difficulties.

**Limitations of Existing Attempts**: Two-step approaches (the DSR* paradigm), which first obtain a numerical solution using a PINN and then fit a symbolic expression via symbolic regression, suffer from the approximation errors of the numerical solution which can mislead the symbolic regression process.

**Key Insight**: Skip the intermediate numerical solution step and directly search for closed-form solutions satisfying the PDE definition within the symbolic space using reinforcement learning.

**Core Idea**: Use an RNN to generate expression skeletons and optimize constants via BFGS to satisfy PDE constraints, training the RNN using the degree of satisfaction as a reward (risk-seeking policy gradient).

## Method

### Overall Architecture
Input: PDE definition (equation form + boundary/initial conditions + domain) $\to$ Autoregressive generation of symbolic expression trees by RNN $\to$ Extraction of constants from skeletons and optimization via BFGS $\to$ Calculation of MSE as reward $\to$ Update RNN policy $\to$ Repeat until reward $> 0.9999$ $\to$ Output closed-form solution.

### Key Designs

1. **Multi-System PDE Modeling**:

    - Time-independent systems: $\mathcal{F}[\mathbf{x}, u(\mathbf{x}), \nabla u, ..., \nabla^k u] = 0$, with the loss defined as $\mathcal{L}_s = \text{MSE}_\mathcal{F} + \text{MSE}_\mathcal{B}$
    - Spatio-temporal continuous models: $u_t = \mathcal{N}(\mathbf{x},t,\nabla u,...,\nabla^k u)$, adding the initial condition loss $\text{MSE}_\mathcal{I}$
    - Spatio-temporal discrete models: Parameterize time such that the solution at each time step shares the same skeleton but has different parameters $\hat{u}(\mathbf{x}; \vec{\alpha}_t)$, using a Parameteric Neural Network (PNN) to learn the mapping $t \to \vec{\alpha}_t$
    - **Design Motivation**: The discrete-time model reduces the complexity of the search space for spatio-temporally coupled expressions.

2. **RL-based Expression Generation and Optimization**:

    - Use an RNN to autoregressively generate the pre-order traversal sequence of the expression tree.
    - The selection probability of each token is output by softmax, while providing parent and sibling node information to reinforce structural understanding.
    - Each generated complete expression constitutes an episode; constants in the skeleton are optimized using BFGS/Adam $\to$ calculate reward.
    - **Design Motivation**: Formulate PDE solving as an MDP, utilizing RL for automated searching.

3. **Risk-Seeking Policy Gradient**:

    - Reward function: $\mathcal{R}_s(\tau) = \frac{1}{1 + \sqrt{\text{MSE}_\mathcal{F} + \text{MSE}_\mathcal{B}}}$
    - Instead of optimizing average performance, maximize the best-case performance: $J_{\text{risk}}(\theta; \epsilon) \approx \mathbb{E}[\mathcal{R}(\tau) | \mathcal{R}(\tau) \geq \mathcal{R}_\epsilon]$
    - Update the policy using only samples with rewards exceeding the $(1-\epsilon)$-quantile.
    - Incorporate entropy regularization to facilitate exploration.
    - **Design Motivation**: Find the exact solution of the PDE rather than an average-approximated solution.

### Loss & Training
- Constant optimization: BFGS minimizes $\mathcal{L}_s$ or $\mathcal{L}_{s\text{-}t}$
- Policy optimization: Risk-seeking policy gradient + entropy regularization
- Termination condition: $\mathcal{R} > 0.9999$ or reaching the maximum number of episodes

## Key Experimental Results

### Main Results

| Dataset/Benchmark | Metric | SymPDE | DSR* | Gain |
|-------------|------|--------|------|------|
| Nguyen 12 Benchmark Average | $\bar{\mathcal{R}}_s$ | 0.9746 | 0.9144 | +6.6% |
| Nguyen 12 Benchmark Average | Recovery Rate $P_{\text{Re}}$ | 90.0% | 33.3% | +56.7% |
| Periodic Potential Field (Eq.10) | $R^2$ | 1.00 | 0.00 | — |
| Point Charge (Eq.11) | $R^2$ | 1.00 | 0.00 | — |
| Continuous-time Heat Equation (Eq.12) | Expression Correctness | ✓ | ✗ | — |
| Discrete-time Heat Equation (Eq.13) | Relative $\mathcal{L}_2$ Error | 9.84×10⁻⁴ | — | — |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| SymPDE (Continuous Time) | Correct skeleton $x^2 e^{-t}$ | End-to-end discovery of correct spatio-temporally coupled expressions |
| SymPDE (Discrete Time) | Skeleton $c_0 e^{c_1 x^2}$ | Successfully decouple time and space, with PNN fitting the parameters |
| DSR* (PINN+DSR) | Incorrect expression | Minor deviations in PINN are magnified by symbolic regression |
| Pure PINN | $\mathcal{L}_2=0.0283$ | Fair numerical accuracy but lacks interpretability |

### Key Findings
- SymPDE achieves a 100% recovery rate on 10 out of 12 Nguyen benchmarks, while DSR* only achieves it on 4.
- For high-frequency oscillations (sin(5x)) and sharp-gradient functions (1/r), SymPDE achieves perfect recovery while DSR* fails completely.
- The subtle numerical deviations of PINN are significantly amplified by symbolic regression, validating the necessity of an end-to-end framework.

## Highlights & Insights
- **Paradigm Innovation**: Proposes the first paradigm to directly search for closed-form symbolic solutions to PDEs using RL, bypassing intermediate numerical solution steps.
- **Discrete-time Model**: Inspired by FDTD, handles spatio-temporal PDEs with a parameterized expression skeleton, achieving elegant dimensionality reduction.
- **Practical Significance**: Closed-form solutions can extrapolate precisely outside the training domain, which is inherently impossible for numerical methods.

## Limitations & Future Work
- Validated only on Poisson and heat equations; more complex non-linear PDEs (e.g., Navier-Stokes) have not been addressed.
- The search space grows exponentially with the number of variables and operator types; its scalability remains to be validated.
- Requires pre-specifying the set of allowed mathematical operators, which demands domain-specific prior knowledge.

## Related Work & Insights
- Shares the RL framework with the DSR algorithm by Petersen et al. 2020; this work extends it from data fitting to PDE solving.
- PINNs family (Raissi 2019, DeepONet, FNO) provides numerical solution baselines.
- End-to-end symbolic regression methods like X-Net (Li 2024) can serve as alternative skeleton generators.

## Rating
- Novelty: ⭐⭐⭐⭐ First to directly utilize RL for searching closed-form symbolic solutions to PDEs, presenting a highly novel paradigm.
- Experimental Thoroughness: ⭐⭐⭐ The benchmarks are relatively basic, only covering linear PDEs and lacking more challenging test cases.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, and the methodology is systematically presented.
- Value: ⭐⭐⭐⭐ Opens up a new direction for symbolic computing within AI for Science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Riesz Neural Operator for Solving Partial Differential Equations](../../ICLR2026/physics/riesz_neural_operator_for_solving_partial_differential_equations.md)
- [\[ICLR 2026\] Disentangled Representation Learning for Parametric Partial Differential Equations](../../ICLR2026/physics/disentangled_representation_learning_for_parametric_partial_differential_equatio.md)
- [\[ICLR 2026\] Physics-Informed Inference Time Scaling for Solving High-Dimensional Partial Differential Equations via Defect Correction](../../ICLR2026/physics/physics-informed_inference_time_scaling_for_solving_high-dimensional_partial_dif.md)
- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](../../AAAI2026/physics/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)
- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)

</div>

<!-- RELATED:END -->
