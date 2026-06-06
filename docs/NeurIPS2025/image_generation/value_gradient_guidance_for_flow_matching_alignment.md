---
title: >-
  [Paper Note] Value Gradient Guidance for Flow Matching Alignment
description: >-
  [NeurIPS 2025][Image Generation][Flow Matching] This paper proposes VGG-Flow, which leverages the Hamilton-Jacobi-Bellman (HJB) equation from optimal control theory to reformulate flow matching alignment as a gradient ma…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Human Preference Alignment"
  - "Optimal Control"
  - "HJB Equation"
  - "Value Function Gradient"
date: 2026-05-08
content_hash: d01a11fe8cb4e3e1
---

# Value Gradient Guidance for Flow Matching Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2512.05116](https://arxiv.org/abs/2512.05116)  
**Code**: [Project Page](https://vggflow25.github.io)  
**Area**: Flow Matching / Model Alignment
**Keywords**: Flow Matching, Human Preference Alignment, Optimal Control, HJB Equation, Value Function Gradient

## TL;DR

This paper proposes VGG-Flow, which leverages the Hamilton-Jacobi-Bellman (HJB) equation from optimal control theory to reformulate flow matching alignment as a gradient matching task—matching the residual velocity field to the gradient of the value function—enabling efficient reward alignment while preserving the prior distribution.

## Background & Motivation

Flow matching models (e.g., Stable Diffusion 3) represent one of the most powerful approaches for continuous distribution generation, with broad applications in image, video, and 3D object synthesis. Unlike diffusion models, flow matching models employ deterministic ODEs for sampling, resulting in straighter trajectories that are easier to model.

Aligning flow matching models with human preferences (RLHF) poses unique challenges:

**Lack of stochastic flow**: Diffusion models involve stochastic sampling at each step, naturally admitting stochastic optimal control formulations. In contrast, flow matching models follow deterministic ODE trajectories, precluding direct application of diffusion-based alignment methods (e.g., GFlowNet-based fine-tuning).

**Prior preservation**: Directly maximizing reward through the computational graph (e.g., ReFL, DRaFT) only identifies modes of the reward model rather than aligning to the target distribution, often leading to reward hacking and mode collapse.

**Adjoint Matching**, while theoretically principled, requires converting the flow matching ODE into an equivalent SDE and solving an adjoint ODE, incurring substantial computational overhead.

**Key Challenge**: How can flow matching models be aligned efficiently and robustly while maintaining probabilistic correctness?

This paper addresses this question from the perspective of deterministic optimal control and proposes a more efficient alternative.

## Method

### Overall Architecture

VGG-Flow formulates flow matching alignment as a deterministic optimal control problem. The fine-tuning objective is defined as:

$$\min_\theta \mathbb{E}_{x_0 \sim p_0, \dot{x}_t = v_\theta(x_t, t)} \left[\frac{\lambda}{2} \int_0^1 \|\tilde{v}_\theta(x_t, t)\|^2 dt - r(x_1)\right]$$

where $\tilde{v}_\theta = v_\theta - v_{\text{base}}$ denotes the residual velocity field and $\lambda$ is a regularization coefficient. The objective simultaneously maximizes the terminal reward $r(x_1)$ and penalizes deviation from the base model via the cumulative $\ell_2$ cost on the residual velocity field.

### Key Designs

1. **Value Gradient Matching**: Deriving the optimality condition from the HJB equation yields the optimal control law:

$$\tilde{v}^*(x, t) = -\frac{1}{\lambda} \nabla V(x, t)$$

That is, the optimal residual velocity field should align with the negative gradient of the value function. This is the central insight of the method: if the value function gradient $\nabla V(x,t)$ can be accurately estimated, alignment reduces to a straightforward gradient matching problem.

2. **Value Consistency Equation**: Substituting the optimal control law into the HJB equation yields the evolution equation for the value function gradient $g_\phi(x,t) \triangleq \nabla V_\phi(x,t)$:

$$\frac{\partial}{\partial t} g_\phi = [\nabla g_\phi]^T \left(\frac{1}{\lambda} g_\phi - v_{\text{base}}(x,t)\right) - [\nabla v_{\text{base}}(x,t)]^T g_\phi$$

with boundary condition $g_\phi(x, 1) = -\nabla r(x)$. This PDE is efficiently discretized via finite differences.

3. **Forward-looking Parametrization**: Directly solving the above PDE is time-consuming. Inspired by DreamFusion, the authors parametrize the value gradient using the reward gradient of a one-step Euler prediction $\hat{x}_1$ combined with a residual network:

$$g_\phi(x, t) \triangleq -\eta_t \cdot \text{stop-gradient}(\nabla_{x_t} r(\hat{x}_1(x_t, t))) + \nu_\phi(x_t, t)$$

where $\hat{x}_1 = x_t + (1-t) \cdot \text{stop-gradient}(v(x_t, t))$. This parametrization provides a well-initialized starting point (near $t=1$, $\nu_\phi$ should approach zero), accelerating convergence.

### Loss & Training

The total training objective consists of three components:

$$\mathcal{L}_{\text{total}}(\theta, \phi) = \mathcal{L}_{\text{matching}}(\theta) + \mathcal{L}_{\text{consistency}}(\phi) + \alpha \mathcal{L}_{\text{boundary}}(\phi)$$

- **Matching loss** (updates $\theta$): $\mathcal{L}_{\text{matching}} = \mathbb{E}\|\tilde{v}_\theta(x_t, t) + \beta g_\phi(x_t, t)\|^2$
- **Consistency loss** (updates $\phi$): $\mathcal{L}_{\text{consistency}}$ enforces $g_\phi$ to satisfy the HJB gradient equation
- **Boundary loss** (updates $\phi$): $\mathcal{L}_{\text{boundary}} = \mathbb{E}\|g_\phi(x_1, 1) + \nabla r(x_1)\|^2$

The training pipeline proceeds as follows: simulate ODE trajectories → update the value gradient model $g_\phi$ → update the velocity field $v_\theta$. LoRA (rank=8) is applied to the attention layers of SD3, and the value gradient network is a reduced-scale SD-v1.5 U-Net.

## Key Experimental Results

### Aesthetic Score Alignment (400 Fine-tuning Steps)

| Method | Reward↑ | DreamSim Diversity↑ (×10⁻²) | FID↓ |
|---|---|---|---|
| Base (SD3) | 5.99 | 23.12 | 212 |
| VGG-Flow | **8.24** | **22.12** | **375** |
| ReFL | 10.00 | 5.59 | 1338 |
| DRaFT | 9.54 | 7.78 | 1518 |
| Adjoint Matching | 6.87 | 22.34 | 465 |

### Multi-Reward Model Comparison

| Method | HPSv2 Reward↑ | HPSv2 Diversity↑ | PickScore Reward↑ | PickScore Diversity↑ |
|---|---|---|---|---|
| VGG-Flow | **3.86** | **18.40** | **23.21** | **20.93** |
| ReFL | 3.87 | 14.08 | 23.19 | 17.71 |
| DRaFT | 3.76 | 15.05 | 23.00 | 19.03 |
| AM | 3.59 | 14.11 | 22.78 | 19.70 |

### Key Findings

- VGG-Flow achieves the best Pareto frontier in the trade-off between reward and diversity/prior preservation.
- ReFL and DRaFT readily reach reward values above 9 on Aesthetic Score, but at the cost of completely losing the base model prior (FID > 1300).
- At equivalent reward levels, VGG-Flow achieves 3–4× higher DreamSim diversity and 3–4× lower FID.
- Adjoint Matching converges more slowly than VGG-Flow and incurs greater computational overhead (requires 4 GPUs and float32 precision).
- Ablations on temperature $\beta$ show that higher $\beta$ accelerates convergence but degrades diversity and prior preservation; the temporal schedule of $\eta_t$ has limited impact on final performance.

## Highlights & Insights

- Grounding the approach in deterministic optimal control is the key innovation, avoiding the additional overhead of converting the ODE to an SDE (in contrast to Adjoint Matching).
- The forward-looking parametrization exploits the approximate linearity of rectified flows, providing an efficient initialization for the value gradient.
- Analysis of the connection to Pontryagin's Maximum Principle reveals a computational advantage of the HJB approach: amortizing the learning of $\nabla V$ rather than solving the adjoint equation per trajectory.
- The stop-gradient operation is a practical engineering technique, adapted from DreamFusion with theoretical justification.

## Limitations & Future Work

- The method is based on a relaxed objective; the fine-tuned distribution approximates the KL-regularized target distribution well only when $\lambda$ is sufficiently small.
- The use of finite differences and the disabling of second-order gradients introduce unavoidable approximation bias.
- The method inherits the exploration–exploitation trade-off common to standard RL; hyperparameter settings may bias toward mode collapse.
- Better architecture designs, which have been shown to be important in large-model fine-tuning, remain unexplored.

## Related Work & Insights

- The core distinction from Adjoint Matching: AM is based on stochastic optimal control and requires an ODE-to-SDE conversion and adjoint ODE solving; VGG-Flow operates directly on the deterministic ODE.
- ReFL and DRaFT are computational graph truncation methods that lack probabilistic correctness and are prone to reward hacking.
- The application of optimal control to diffusion model alignment is a growing trend; VGG-Flow provides the corresponding solution for flow matching models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of the HJB equation from deterministic optimal control to flow matching alignment; the forward-looking parametrization is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three reward models, multiple ablation studies, Pareto frontier analysis, and convincing experiments on SD3.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear; the discussion connecting to PMP and AM is in-depth.
- **Value**: ⭐⭐⭐⭐⭐ Provides an efficient and practical solution for aligning flow matching models, with direct applicability to large models such as SD3.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](../../ICLR2026/image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)

</div>

<!-- RELATED:END -->
