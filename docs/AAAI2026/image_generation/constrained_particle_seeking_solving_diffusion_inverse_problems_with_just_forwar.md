---
title: >-
  [Paper Note] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes
description: >-
  [AAAI 2026][Image Generation][Diffusion Models] This paper proposes Constrained Particle Seeking (CPS), a gradient-free method for solving diffusion model inverse problems. CPS constructs a locally linear surrogate of th…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Inverse Problems"
  - "Gradient-Free Methods"
  - "Particle Optimization"
  - "Constrained Optimization"
date: 2026-05-08
content_hash: b8bce44a978abf75
---

# Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes

**Conference**: AAAI 2026
**arXiv**: [2603.01837](https://arxiv.org/abs/2603.01837)  
**Code**: [GitHub](https://github.com/deng-ai-lab/CPS)  
**Area**: Diffusion Models / Inverse Problem Solving
**Keywords**: Diffusion Models, Inverse Problems, Gradient-Free Methods, Particle Optimization, Constrained Optimization

## TL;DR

This paper proposes Constrained Particle Seeking (CPS), a gradient-free method for solving diffusion model inverse problems. CPS constructs a locally linear surrogate of the forward process using all candidate particles, then seeks the optimal particle under a hyperspherical constraint within the high-density region of the transition kernel, achieving performance competitive with gradient-based methods.

## Background & Motivation

Inverse problems — recovering the original signal $x$ from indirect noisy observations $y = \mathcal{H}(x) + \eta$ — are a fundamental challenge in computational imaging, medicine, and geophysics. Diffusion models have demonstrated strong prior-based advantages for inverse problem solving, but face a critical limitation:

**Gradient Dependency**: Mainstream methods (DPS, ΠiGDM, DAPS, etc.) require gradient information from the forward observation process to guide sampling. However, in many practical scenarios, the forward model involves highly nonlinear or computationally expensive numerical simulations (e.g., black hole imaging, fluid simulation), making gradient computation difficult or infeasible.

**Limitations of Existing Gradient-Free Methods**:
   - **SCG** (rejection-sampling-like): Samples multiple candidate particles per step but retains only the best one, discarding a large amount of valuable information, resulting in low efficiency.
   - **EnKG** (ensemble Kalman): Maintains thousands of particles in high-dimensional space with large computational overhead; global linearization introduces large errors in early stages.
   - **DPG** (policy gradient): Requires perturbing clean samples with noise, potentially drifting off the data manifold.
   - These methods generally underperform gradient-based approaches.

Core observation: Particles discarded by SCG are not useless — experiments show that the top-2/3 particles, and even bottom particles with negated signs, contain useful guidance information. Information is distributed across the entire particle set and should be fully utilized.

## Method

### Overall Architecture

CPS reformulates the diffusion inverse problem as a constrained optimization problem: at each timestep, find an optimal particle that minimizes the observation alignment objective while remaining within the high-density region of the unconditional prior. This involves three steps:

1. Sample multiple candidate particles from the reverse transition kernel.
2. Fit a locally linear surrogate of the forward process using all particles.
3. Solve for the optimal particle under a hyperspherical constraint and use it as the next state.

### Key Designs

#### 1. Forward Process Surrogate Model

Optimizing the terminal cost requires estimating $\nabla_{x_t} \Phi(\hat{x}_{0|t})$, but the gradient of the forward model $\mathcal{H}(\cdot)$ is unavailable. CPS employs **statistical linearization** to construct a linear surrogate $\mathcal{H}(\hat{x}_{0|t}) \approx \mathbf{A}x_t + \mathbf{b}$ within the local region defined by $p(x_t|x_{t+1})$.

After sampling $n$ particles, the surrogate parameters are computed in closed form:

$$\mathbf{A} = \frac{1}{n\sigma_t^2} \sum_{i=1}^{n} (\mathcal{H}(\hat{x}_{0|t}^i) - \bar{\mathcal{H}})(x_t^i - \mu_t)^\top$$

$$\mathbf{b} = \bar{\mathcal{H}} - \mathbf{A}\mu_t$$

Key advantage: $\mathbf{A}$ is a weighted linear combination of all sampled particles, integrating the full set of candidate information, enabling the discovery of a particle superior to any individual sample.

Distinction from EnKG: EnKG performs global-space linearization, causing large errors in early stages. CPS linearizes only within the local region of $p(x_t|x_{t+1})$; Jensen gap analysis shows that CPS errors remain consistently small.

#### 2. Transition Kernel Constraint (Hyperspherical Constraint)

In high dimensions, the norm of Gaussian samples concentrates on the hypersphere $S^{d-1}(\mu_t, \sigma_t\sqrt{d})$ (mass concentration of the $\chi^d$ distribution). Diffusion models favor inputs on this hypersphere rather than near the mean. The constrained optimization problem is therefore:

$$\min_{x_t} \|y - (\mathbf{A}x_t + \mathbf{b})\|^2 \quad \text{s.t.} \; x_t \in S^{d-1}(\mu_t, \sigma_t\sqrt{d})$$

Solving via Lagrange multipliers, and noting that $\sigma_t$ is typically small so the constraint dominates, yields an asymptotic closed-form solution:

$$x_t^* \approx \mu_t + \sigma_t\sqrt{d} \frac{\mathbf{A}^\top(y - \bar{\mathcal{H}})}{\|\mathbf{A}^\top(y - \bar{\mathcal{H}})\|}$$

This solution can be interpreted as displacing the unconditional mean $\mu_t$ by exactly one "standard radius" along the optimal direction indicated by the surrogate model, ensuring the particle remains in the high-probability-density region of the transition kernel.

#### 3. Restart Strategy for Robustness

In practice, different initial noise realizations may yield varying results. To mitigate error accumulation, a Restart strategy is introduced: at timestep $t$, the sample is re-noised back to $t+1$ following Eq. 2, then re-guided and denoised via CPS, progressively correcting accumulated errors. The outer loop iterates over timesteps $T{-}1$ to $0$; the inner loop executes $N_r$ restarts.

### Loss & Training

CPS is a **training-free** method that directly leverages pretrained diffusion models:
- No fine-tuning or additional training is required.
- Only black-box access to the forward model is needed (only forward outputs, no gradients).
- $n$ candidate particles are sampled per step ($n{=}32$ is generally sufficient in experiments).
- The number of restarts $N_r$ controls the accuracy–efficiency trade-off.

## Key Experimental Results

### Main Results

**Table 1: FFHQ Image Inverse Problems (256×256)**

| Task | Method Type | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| Inpainting (95%) | Gradient | ΠiGDM | 23.13 | 0.774 | 0.231 |
| Inpainting (95%) | Gradient | DAPS | 22.71 | 0.754 | 0.226 |
| Inpainting (95%) | Gradient-Free | SCG | 6.15 | 0.321 | 0.807 |
| Inpainting (95%) | Gradient-Free | EnKG | 22.64 | 0.767 | 0.286 |
| Inpainting (95%) | **Gradient-Free** | **CPS** | **24.90** | **0.794** | **0.161** |
| Super-Res. (×4) | Gradient | DPS | 27.65 | 0.822 | 0.120 |
| Super-Res. (×4) | Gradient-Free | **CPS** | **27.70** | **0.834** | **0.115** |
| JPEG (QF=5) | Gradient | ΠiGDM | 18.25 | 0.593 | 0.304 |
| JPEG (QF=5) | Gradient-Free | **CPS** | **23.23** | **0.784** | **0.223** |

CPS substantially outperforms all gradient-free methods and matches or exceeds gradient-based methods on multiple tasks. In particular, for non-differentiable scenarios such as JPEG restoration, CPS significantly surpasses ΠiGDM.

**Table 2: Black Hole Imaging**

| Observation Ratio | Method | PSNR↑ | BPSNR↑ |
|---|---|---|---|
| 100% | SCG | 25.45 | 30.56 |
| 100% | **CPS** | **25.74** | **31.84** |
| 3% | SCG | 24.65 | 31.28 |
| 3% | **CPS** | **25.03** | **31.81** |

**Table 3: Fluid Data Assimilation**

| Sparsity | Method | Relative L2↓ |
|---|---|---|
| ×2 | EnKG | 0.320 |
| ×2 | **CPS** | **0.295** |
| ×8 | EnKG | 0.821 |
| ×8 | **CPS** | **0.684** |

CPS also demonstrates significant advantages on scientific inverse problems (black hole imaging, fluid assimilation).

### Ablation Study

- **Particle Efficiency**: CPS achieves strong performance with as few as 8 particles. EnKG requires a large number of particles to function, and SCG fails entirely on image problems.
- **Restart Strategy**: Beyond improving overall performance, the restart strategy substantially enhances algorithmic stability, effectively preventing failures caused by poor initial noise realizations.

### Key Findings

1. Discarding particles in SCG is highly wasteful — even the worst particles contain useful guidance information when their direction is negated.
2. The Jensen gap of local linearization is far smaller than that of global linearization (EnKG), validating the superiority of local surrogates.
3. The hyperspherical constraint is critical — it ensures that optimized particles remain in the high-density region of the diffusion prior.
4. In non-differentiable scenarios such as JPEG restoration, gradient-free methods hold a natural advantage.

## Highlights & Insights

- **High Information Utilization**: The conceptual shift from "selecting the best particle" to "synthesizing the optimal particle from all candidate information" drives a substantial performance improvement.
- **Elegant Mathematical Form**: The final closed-form solution is concise and elegant — mean + normalized direction × standard radius.
- **Plug-and-Play**: Training-free with only black-box forward access; directly applicable to various pretrained diffusion models.
- **Cross-Domain Validation**: Validated on both image restoration and scientific problems, demonstrating the generality of the approach.

## Limitations & Future Work

1. Each step requires $n$ forward model evaluations (sampling + surrogate fitting), which still incurs overhead for computationally expensive forward models.
2. The linear surrogate may lack accuracy for highly nonlinear observation models, though localization mitigates this issue.
3. Evaluation is limited to FFHQ and specific scientific datasets; large-scale validation on natural image benchmarks has not been conducted.
4. The restart strategy increases total sampling steps, correspondingly increasing inference time.

## Related Work & Insights

- Compared to gradient-based methods such as DPS, CPS does not require differentiation through $\mathcal{H}$, greatly broadening its applicability.
- Compared to EnKG, local linearization combined with single-particle seeking avoids the overhead of maintaining large particle ensembles and the errors of global linearization.
- The constrained optimization perspective transforms inverse problem solving from "passive selection" to "active seeking," offering insights for improving other particle-based methods.
- The restart strategy is inspired by the RePaint/Restart line of work, and proves highly effective within this framework.

## Rating

- **Novelty**: ★★★★☆ — Hyperspherical constrained particle seeking is a novel gradient-free strategy.
- **Technical Depth**: ★★★★★ — Stochastic control + statistical linearization + Lagrange multipliers + asymptotic analysis; mathematically rigorous.
- **Experimental Thoroughness**: ★★★★☆ — Validated across three categories (images, black holes, fluids) with fair comparisons against gradient-based methods.
- **Writing Quality**: ★★★★☆ — Well-organized with clear experiment-driven explanation of design choices.
- **Value**: ★★★★★ — Training-free, plug-and-play, black-box forward access, open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/image_generation/composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
