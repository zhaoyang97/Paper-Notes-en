---
title: >-
  [Paper Note] GenCP: Towards Generative Modeling Paradigm of Coupled Physics
description: >-
  [ICLR 2026][Image Generation][coupled physics simulation] This paper proposes GenCP, which reformulates coupled multiphysics simulation as a probability density evolution problem. It leverages flow matching to learn conditional velocity fields from decoupled data, and synthesizes coupled solutions at inference time via Lie-Trotter operator splitting—realizing "decoupled training, coupled inference" with theoretically bounded error guarantees.
tags:
  - ICLR 2026
  - Image Generation
  - coupled physics simulation
  - flow matching
  - operator splitting
  - multiphysics
  - decoupled training
date: 2026-05-08
content_hash: c855d87ddee64c5e
---

# GenCP: Towards Generative Modeling Paradigm of Coupled Physics

**Conference**: ICLR 2026
**arXiv**: [2601.19541](https://arxiv.org/abs/2601.19541)
**Code**: [GitHub](https://github.com/AI4Science-WestlakeU/GenCP)
**Area**: Generative Physics Simulation / Flow Matching
**Keywords**: coupled physics simulation, flow matching, operator splitting, multiphysics, decoupled training

## TL;DR

This paper proposes GenCP, which reformulates coupled multiphysics simulation as a probability density evolution problem. It leverages flow matching to learn conditional velocity fields from decoupled data, and synthesizes coupled solutions at inference time via Lie-Trotter operator splitting—realizing "decoupled training, coupled inference" with theoretically bounded error guarantees.

## Background & Motivation

**Background**: Coupled multiphysics simulation (e.g., fluid-structure interaction (FSI), nuclear-thermal coupling) is central to engineering applications. Numerical approaches fall into two categories: monolithic coupling (high accuracy but prohibitively expensive) and partitioned coupling (practical but prone to convergence instability). Surrogate models and neural operators can accelerate simulation, yet most require coupled solution data for training, which is far more costly to obtain than decoupled data (over 5× more expensive).

**Limitations of Prior Work**: (1) Surrogate model approaches approximate coupled solutions via Gauss-Seidel/ADMM iterative inference, but struggle with complex spatiotemporal dynamics (failing to capture high-frequency, high-dimensional, and stochastic behavior); (2) Existing generative approaches either handle single-physics fields only, or learn directly from coupled data, overlooking the challenge of learning coupled physics from decoupled data; (3) M2PDE embeds coupling iterations into each diffusion step but lacks theoretical guarantees.

**Key Challenge**: Acquiring coupled training data in engineering practice is extremely expensive, whereas decoupled data is readily available. The fundamental challenge is: how to learn from decoupled data and generate high-fidelity coupled solutions at inference time?

**Goal**: To develop a framework that learns coupled physics from decoupled training data while ensuring high fidelity, high efficiency, and high reliability (the "3H" criteria).

**Key Insight**: Coupled physics simulation is reformulated as the evolution of probability densities in function space. Flow matching is used to learn conditional velocity fields (decoupled training), and operator splitting synthesizes coupled inference within the flow steps. The theoretical foundation is established via the weak form of the continuity equation and Lie-Trotter splitting.

**Core Idea**: In probability space, operator splitting composes conditionally learned flows into coupled inference, which is physically equivalent to iteratively solving coupled fields in a noisy latent space.

## Method

### Overall Architecture

GenCP consists of two phases. **Training**: two conditional velocity fields $\hat{v}_f$ and $\hat{v}_g$ are learned independently via flow matching on decoupled datasets $\mathcal{D}_f$ and $\mathcal{D}_g$, respectively. **Inference**: via Lie-Trotter operator splitting, the two conditional velocity fields are applied alternately at each flow step, evolving from noise to the coupled solution.

### Key Designs

**Design 1: Probability Density Evolution Perspective**
- **Function**: Reformulates coupled physics simulation as a transport problem for the joint probability measure $\mu_t$ in function space.
- **Mechanism**: The joint state $u=(f,g)$ lives on the product space $\mathcal{U}=\mathcal{F}\times\mathcal{G}$. The evolution of $\mu_t$ is described via the weak continuity equation: $\int_0^1 \int_{\mathcal{U}} (\partial_t\varphi + \langle D\varphi, v\rangle) d\mu_t dt = 0$. Since the weak form is linear in $v$, the decomposition $v = v^{(f)} + v^{(g)}$ follows naturally.
- **Design Motivation**: (1) Empirical measures have no density, making the strong-form continuity equation inapplicable; (2) The divergence operator is ill-defined in infinite-dimensional spaces; (3) The weak form is mathematically well-posed in function space.

**Design 2: Time-Parameterized Linear Interpolation Training**
- **Function**: Constructs flow matching training objectives from decoupled data.
- **Mechanism**: For field $f$: sample $(f_1, \bar{g}) \sim \mathcal{D}_f$ and reference noise $z_f, z_g$; construct linear interpolation $f_t = (1-t)z_f + tf_1$. The instantaneous velocity target is $v_f = f_1 - z_f$. Train $\hat{v}_f(f_t, g_t, t; \theta_f)$ with MSE loss. Field $g$ is handled symmetrically.
- **Design Motivation**: Linear interpolation is the simplest conditional flow path; target velocities can be computed directly from data pairs without solving complex ODEs.

**Design 3: Lie-Trotter Operator Splitting Inference**
- **Function**: At inference time, alternately applies learned conditional velocity fields to synthesize the coupled solution.
- **Mechanism**: Partition $[0,1]$ into $N$ steps with $\tau = 1/N$. At each step: first update $f \leftarrow f + \tau \hat{v}_f(f,g,t)$, then update $g \leftarrow g + \tau \hat{v}_g(f,g,t)$. This is physically equivalent to alternately solving coupled fields in the noise space.
- **Design Motivation**: Lie-Trotter splitting is a classical operator splitting method that converges to the joint flow as $\tau \to 0$, and arises naturally from the decomposition of the weak continuity equation.

**Design 4: Theoretical Error Guarantees**
- **Function**: Proves that the error of GenCP's inference scheme is controllable.
- **Mechanism**: Theorem 3.1: $W_1(\mu_1^{(\tau,learn)}, \mu_1) \leq C_{stab}(\tau + \varepsilon_f + \varepsilon_g)$. Total error is jointly determined by the splitting step size $\tau$ (first-order Lie-Trotter) and the learning errors $\varepsilon_f, \varepsilon_g$.
- **Design Motivation**: Provides reliability guarantees, addressing the lack of theoretical foundations in existing methods such as M2PDE.

### Loss & Training

Training loss: standard flow matching MSE loss
$$\mathcal{L}_f(\theta_f) = \mathbb{E}_{t, (f_1,\bar{g}), z_f, z_g} [\|v_f - \hat{v}_f(f_t, g_t, t; \theta_f)\|^2_\mathcal{F}]$$

$\mathcal{L}_g(\theta_g)$ is defined symmetrically. The two velocity field models are trained independently.

Inference: Lie-Trotter splitting, typically converging in approximately 10 steps.

## Key Experimental Results

### Main Results

**2D Synthetic Distribution Experiment**

| Paradigm | W1↓ | MMD↓ | Energy Distance↓ |
|----------|-----|------|-------------------|
| GenCP (Easy) | 0.4366 | 0.0095 | 0.0411 |
| M2PDE (Easy) | 0.5177 | 0.0141 | 0.0625 |
| GenCP (Complex) | **0.4928** | **0.0053** | **0.0061** |
| M2PDE (Complex) | 25450 | ∞ | 332.4 |

M2PDE completely collapses on complex distributions, while GenCP remains stable.

**Turek-Hron FSI Task (Relative L2 Error)**

| Method | u | v | p | SDF | Inference Time |
|--------|---|---|---|-----|----------------|
| Joint Training | 0.0088 | 0.0344 | 0.0544 | 0.0079 | — |
| M2PDE-FNO* | 0.0590 | 0.2415 | 0.2474 | 0.2482 | 277.2s |
| Surrogate-FNO* | 0.0550 | 0.2257 | 0.2553 | 0.0112 | 93.2s |
| **GenCP-FNO*** | **0.0396** | **0.1678** | **0.1897** | **0.0081** | **19.5s** |

With the FNO* backbone, GenCP reduces mean error by ~26.77% and achieves 14× faster inference.

### Ablation Study

| Backbone | GenCP vs. M2PDE Error Reduction | GenCP vs. Surrogate Error Reduction | Efficiency Gain |
|----------|---------------------------------|--------------------------------------|-----------------|
| FNO* | ~26.77% | Significant | ~14× |
| CNO | ~12.54% | Significant | ~18× |

### Key Findings

1. **Decoupled training → coupled inference is viable**: GenCP trained on conditional distributions successfully recovers the joint distribution, far outperforming M2PDE on complex distributions.
2. **Substantial efficiency advantage**: Accurate coupled solutions are generated in approximately 10 sampling steps, 14–18× faster than M2PDE.
3. **SDF field performance approaches joint training**: With the CNO backbone, GenCP's SDF error (0.0183) approaches that of joint training (0.0079), indicating that coupling information is effectively transmitted through splitting.
4. **Surrogate models' apparently low errors are misleading**: Surrogate models appear to achieve low SDF errors but completely fail to capture the oscillatory bending dynamics of the beam; GenCP is the only method that captures this coupled effect.
5. **M2PDE is severely unstable in complex scenarios**: Iterating to convergence combined with accumulated intermediate estimation errors leads to mode collapse.

## Highlights & Insights

- **Theoretical elegance**: Starting from the weak continuity equation, the velocity field decomposition and Lie-Trotter splitting are derived naturally, yielding a cohesive and rigorous theoretical development.
- **Provable error bounds**: Theorem 3.1 proves that total error is linearly controlled by the splitting step size and learning errors—a level of theoretical rigor rarely seen in AI for Science.
- **Strong practical value**: Decoupled data is over 5× cheaper to obtain than coupled data; GenCP makes large-scale multiphysics simulation feasible.
- **Elegant "coupling in flow" design**: The coupling process is embedded within the flow matching sampling steps rather than applied iteratively post-sampling, yielding a fundamental efficiency advantage.
- **Open-sourced datasets**: FSI and nuclear-thermal coupling datasets are released publicly, advancing the field.

## Limitations & Future Work

1. **First-order splitting accuracy**: Lie-Trotter is a first-order method; Strang splitting (second-order) could be explored for improved accuracy.
2. **Two-field coupling only**: Although generalization to $m$ fields is claimed, experiments only validate the two-field case.
3. **Dependence on flow matching backbone**: The expressive capacity of FNO*/CNO limits ultimate prediction accuracy.
4. **Still requires decoupled simulation data**: Compared to using real experimental data, acquiring decoupled simulation data remains non-trivial.
5. **Error accumulation in long-horizon rollouts**: The paper primarily validates short-horizon prediction (12 steps); error accumulation over long sequences warrants further investigation.

## Related Work & Insights

- **Direct comparison with M2PDE**: M2PDE embeds coupling into each diffusion step but lacks theoretical guarantees; GenCP provides rigorous theoretical grounding via operator splitting.
- **Connection to numerical operator splitting**: Classical tools from numerical analysis (Trotter/Strang splitting) are introduced into generative model inference.
- **Implications for AI for Science**: The probability density evolution perspective may generalize to a broader class of multi-field coupling problems.
- **Distinction from conditional diffusion**: Rather than simply conditioning on one field to generate another, GenCP alternately updates both fields throughout the flow evolution process.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Elegantly combines operator splitting with flow matching, establishing a theoretical paradigm for decoupled training and coupled inference.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers four scenarios (synthetic + two FSI + nuclear-thermal), but limited to two-field coupling; long-horizon rollouts are not validated.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear, though the high mathematical density may be challenging for readers without a strong mathematical background.
- **Value**: ⭐⭐⭐⭐⭐ — Introduces a fundamentally new paradigm for multiphysics simulation with both theoretical and practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [\[CVPR 2026\] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling](../../CVPR2026/image_generation/test-time_instance-specific_parameter_composition_a_new_paradigm_for_adaptive_ge.md)
- [\[ICLR 2026\] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [\[ICLR 2026\] Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek](seek-cad_a_self-refined_generative_modeling_for_3d_parametric_cad_using_local_in.md)

<!-- RELATED:END -->
