---
title: >-
  [Paper Note] Training-Free Constrained Generation with Stable Diffusion Models
description: >-
  [NeurIPS 2025 Spotlight][Image Generation][Constrained Generation] This paper proposes a training-free constrained generation method that embeds Proximal Langevin Dynamics into the reverse denoising process of Stable Diffusion. Image-space constraints are backpropagated to the latent space via the decoder, enabling strict constraint satisfaction on generated outputs without retraining.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Image Generation"
  - "Constrained Generation"
  - "Stable Diffusion"
  - "Proximal Mapping"
  - "Latent Space Correction"
  - "Training-Free"
date: 2026-05-08
content_hash: 52a3a0f908d81ce2
---

# Training-Free Constrained Generation with Stable Diffusion Models

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.05625](https://arxiv.org/abs/2502.05625)  
**Code**: [GitHub](https://github.com/RAISELab-atUVA/Constrained-Stable-Diffusion)  
**Area**: Image Generation
**Keywords**: Constrained Generation, Stable Diffusion, Proximal Mapping, Latent Space Correction, Training-Free

## TL;DR

This paper proposes a training-free constrained generation method that embeds Proximal Langevin Dynamics into the reverse denoising process of Stable Diffusion. Image-space constraints are backpropagated to the latent space via the decoder, enabling strict constraint satisfaction on generated outputs without retraining.

## Background & Motivation

Diffusion models deployed in scientific and engineering domains must generate outputs satisfying strict constraints (physical laws, safety standards, design specifications), yet existing approaches exhibit notable limitations:

**Training-time constraints**: Provide only distribution-level constraint compliance, cannot guarantee per-sample satisfaction, and fail to generalize to unseen constraints.

**Inference-time constraints** (e.g., Projected Diffusion): Modify the reverse process directly in image space, **incompatible with latent diffusion models such as Stable Diffusion**, since constraints cannot be expressed directly in the latent space.

**Latent-space variants**: Rely on specialized measurement operators or learned soft penalties, limiting generality.

**Core challenge**: Constraints are defined in image space, whereas the denoising process of Stable Diffusion operates in latent space — a fundamental gap between the two domains.

## Method

### Overall Architecture

At each denoising iteration of Stable Diffusion:
1. **Langevin dynamics step**: Perform the standard denoising update to obtain a pre-correction latent variable $z'_t$.
2. **Proximal mapping step**: Map $z'_t$ to image space via decoder $D$, evaluate constraint violations, compute gradients and backpropagate to the latent space, and iteratively adjust the latent variable until constraints are satisfied.
3. No modification to the score network or decoder is required; no learnable parameters are added.

### Key Designs

1. **Constraint transfer from latent space to image space**: The core insight is that although constraints cannot be directly expressed in latent space, they can be evaluated via the decoder at any stage of denoising. Gradients are backpropagated from image space to latent space via the chain rule: $\nabla_{z_t}g = (\partial D/\partial z_t)^T \cdot \nabla_{x_t}g$. The frozen, differentiable decoder $D$ serves as the bridge.

2. **Proximal Langevin Dynamics**: The projected Langevin dynamics are generalized to a proximal mapping formulation: $\text{prox}_{\lambda g}(z_t) = \arg\min_y \{g(D(y)) + \frac{1}{2\lambda}\|D(y)-D(z_t)\|^2\}$. The proximal operator balances constraint satisfaction against proximity to the updated sample. When the constraint is a set indicator function, it reduces to standard projection; however, the proximal mapping further handles non-smooth regularization, composite penalties, and implicit constraints.

3. **Handling complex constraints**:

    - **Differentiable surrogate models**: Replace constraints that cannot be expressed analytically (e.g., copyright detection) with pretrained classifiers.
    - **Black-box simulator differentiation**: Drawing on Differentiable Perturbed Optimizers (DPO), gradient estimates are obtained via random perturbations and finite differences, enabling non-differentiable physical simulators to participate in optimization. The smoothed function $\bar{\phi}_\nu(x) = \mathbb{E}[\phi(x+\nu\varepsilon)]$ is estimated via Monte Carlo sampling.

### Loss & Training

**Inner minimizer** (proximal mapping solve):
Gradient descent on the proximal objective:
$$z_t^{i+1} = z_t^i - \nabla_{z_t^i}\!\left[g(D(z_t^i)) + \frac{1}{2\lambda}\|D(z_t^i)-D(z_t^0)\|^2\right]$$

**Outer minimizer** (full sampling process):
After each standard denoising update, the inner minimization is iterated until the constraint violation satisfies $g(D(z_t)) < \delta$.

**Convergence guarantees** (convex constraint setting):
- Theorem 4.1: The distance to the feasible set decreases at rate $(1 - 2\beta'\gamma_{t+1})$.
- Theorem 4.2: The KL divergence from the training distribution grows by at most $O(\sum_t \gamma_t)$, becoming negligible as $\gamma_t \to 0$.

## Key Experimental Results

### Main Results

Three application scenarios validate the generality of the method:

| Task | Metric | Ours (Latent) | PDM | Cond | Gain |
|------|--------|--------------|-----|------|------|
| Microstructure generation ($P=30\%$) | FID ↓ | 13.5±3.1 | 30.7±6.8 | **10.8±0.9** | FID vs. PDM −56% |
| Microstructure generation ($P=30\%$) | Violation >10% ↓ | **0%** | **0%** | 68.4% | Perfect constraint satisfaction |
| Metamaterial inverse design | MSE ↓ | **1.4±0.6** | N/A | 7.1±4.5 | MSE reduced by 80% |
| Metamaterial inverse design | Physical invalidity ↓ | **5%** | N/A | 55% | Substantially fewer invalid samples |
| Copyright-safe generation | Constraint satisfaction ↑ | **90%** | 71% | 67% | Highest constraint compliance |
| Copyright-safe generation | FID ↓ | **65.1** | 75.3 | 61.2 | Quality and constraint jointly maintained |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Microstructure $P=50\%$ vs. $P=30\%$ | 0% violation in both | Consistently effective under varying constraint values |
| DPO steps 0→5 | MSE: 179.5→1.2 | Each iteration substantially improves constraint satisfaction |
| Bastek & Kochmann (Prev. SOTA) | MSE: 6.4±4.6 | Proposed method achieves 4.6× improvement |
| High-resolution $1024^2$ | PDM fails | Latent-space approach natively supports high resolution |

### Key Findings

- **Constraint satisfaction and generation quality are jointly achievable**: FID scores remain close to the unconstrained baseline, as theoretically guaranteed by Theorem 4.2.
- **Strong generality**: A single algorithmic framework handles convex constraints (porosity), black-box simulator constraints (stress–strain), and surrogate constraints (copyright detection).
- **DPO iterations can reduce error arbitrarily**: In contrast to the fixed error of baselines.
- **Latent-space approach natively supports high resolution**: PDM fails at $1024^2$ resolution, while the proposed method operates normally.

## Highlights & Insights

- **Elegant core idea**: The frozen decoder serves as a differentiable bridge, seamlessly transferring image-space constraints to the latent space without modifying any network.
- **Solid theoretical guarantees**: Dual guarantees of convergence and distributional fidelity under convex constraints.
- **Black-box simulator integration**: The DPO framework allows any queryable simulator to be incorporated into constrained optimization, substantially broadening the scope of applicability.
- **First work to integrate constrained optimization into the Stable Diffusion sampling process.**

## Limitations & Future Work

- Additional inner optimization iterations at each denoising step increase inference time.
- Relies on differentiability and Lipschitz continuity of the decoder $D$.
- Theoretical guarantees weaken under non-convex constraints; only empirical validation is provided.
- The DPO approach requires multiple simulator queries ($M=10$), which is costly for complex simulators.
- The 10% violation rate in the copyright detection scenario is limited by classifier accuracy.

## Related Work & Insights

- **PDM (Christopher et al.)**: A projected diffusion model operating in image space; the direct predecessor of this work, but incompatible with latent diffusion models.
- **Classifier Guidance**: Distinct from the proposed method — the former provides soft guidance, whereas this work enforces hard constraint satisfaction.
- **DPO (Differentiable Perturbed Optimizer)**: A gradient estimation technique borrowed from differentiable optimization, enabling black-box simulators to be used in optimization.
- The method generalizes to constrained generation in any latent-space generative model (e.g., VAEs, flow models).

## Rating

- Novelty: ⭐⭐⭐⭐ The latent-space constraint transfer idea is concise and effective; DPO integration is a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three diverse application scenarios covering convex, non-convex, and black-box constraints.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; experimental presentation is intuitive.
- Value: ⭐⭐⭐⭐⭐ Addresses a key challenge in constrained generation with Stable Diffusion; high value for engineering and scientific applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](../../CVPR2025/image_generation/enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](../../CVPR2025/image_generation/stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[NeurIPS 2025\] Safe and Stable Control via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[CVPR 2025\] Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI](../../CVPR2025/image_generation/derivative-free_diffusion_manifold-constrained_gradient_for_unified_xai.md)

</div>

<!-- RELATED:END -->
