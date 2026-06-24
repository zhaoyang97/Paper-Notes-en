---
title: >-
  [Paper Note] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling
description: >-
  [Image Generation] HSO proposes a hierarchical schedule optimizer via a bilevel optimization framework — an upper-level global search for optimal initialization strategies combined with a lower-level local refinement of schedules — achieving training-free SOTA sampling quality under extremely low NFE at a one-time optimization cost of only ~8 seconds.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 54b9e5df035a861e
---

# Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling

## Paper Information

- **Conference**: AAAI 2026
- **arXiv**: [2511.11688](https://arxiv.org/abs/2511.11688)
- **Code**: [https://github.com/chappy0/HSO.git](https://github.com/chappy0/HSO.git)
- **Area**: Diffusion Model Acceleration / Sampling Schedule Optimization
- **Keywords**: Diffusion models, sampling acceleration, schedule optimization, bilevel optimization, training-free, FID, low NFE

## TL;DR

HSO proposes a hierarchical schedule optimizer via a bilevel optimization framework — an upper-level global search for optimal initialization strategies combined with a lower-level local refinement of schedules — achieving training-free SOTA sampling quality under extremely low NFE at a one-time optimization cost of only ~8 seconds.

## Background & Motivation

Diffusion probabilistic models have established new benchmarks in image generation quality, yet the high NFE (number of function evaluations) required by iterative sampling severely impedes real-time deployment. Schedule optimization is a training-free acceleration strategy that maximizes sampling quality by finding the optimal timestep distribution under a fixed small NFE budget.

An ideal schedule optimization method should simultaneously satisfy four core principles:

| Principle | Rule-Based | Perception-Based | Theory-Based | **HSO** |
|-----------|-----------|-----------------|-------------|---------|
| Adaptivity | ✗ | ✓ | ✓ | ✓ |
| Effectiveness | ✗ | ✓ | ✗ | ✓ |
| Practical Robustness | ✓ | ✗ | ✗ | ✓ |
| Computational Efficiency | ✓ | ✗ | ✓ | ✓ |

Limitations of existing paradigms:
- **Rule-based methods** (e.g., EDM schedule): Fixed formulas cannot adapt to different models and NFE settings.
- **Perception-based optimization** (e.g., AutoDiffusion): Requires repeated image generation, resulting in high computational cost (~1.1 days).
- **Theory-based optimization** (e.g., DM-NonUni): Non-convex optimization landscapes are prone to local optima, and robustness is neglected.

## Method

### Overall Architecture: Bilevel Optimization

HSO decomposes the problem of searching for a globally optimal schedule into two tractable subproblems:

**Lower Level: Local Optimization (Schedule Refinement)**
$$\Lambda_{\text{opt}}(\boldsymbol{\psi}) = \arg\min_\Lambda \mathcal{J}_{\text{lower}}(\Lambda | \Lambda_{\text{init}}(\boldsymbol{\psi}))$$

- Starts from the initialization $\Lambda_{\text{init}}$ provided by the upper level
- Employs trust-region constrained optimization
- Guided by the MEP objective function

**Upper Level: Global Search (Initialization Strategy)**
$$\boldsymbol{\psi}^* = \arg\min_\boldsymbol{\psi} \mathcal{F}_{\text{upper}}(\Lambda_{\text{opt}}(\boldsymbol{\psi}))$$

- Searches in a low-dimensional hyperparameter space $\mathbb{R}^3$ with $\boldsymbol{\psi} = (\rho, \tilde{\sigma}_{\min}, \tilde{\sigma}_{\max})$
- Uses population-based evolutionary algorithms such as differential evolution
- Evaluated by the SPF fitness function

**Iterative Alternation**: The upper level proposes a population of candidate strategies → each candidate undergoes lower-level local optimization → evaluated results are fed back to the upper level → the next generation evolves.

### Key Designs

#### 1. Midpoint Error Proxy (MEP)

The core theoretical contribution. Based on the global generation error of the probability flow ODE:

$$x_\epsilon = \frac{\sigma_\epsilon}{\sigma_T}x_T + \sigma_\epsilon \sum_{i=0}^{N-1}\int_{\lambda_i}^{\lambda_{i+1}}e^\lambda f(\lambda)d\lambda$$

**Hybrid midpoint approximation** (Lemma 1): The exactly integrable exponential term $e^\lambda$ is separated from the neural network term $f(\lambda)$; the midpoint approximation is applied only to $f(\lambda)$:

$$\int_{\lambda_i}^{\lambda_{i+1}}e^\lambda f(\lambda)d\lambda \approx f(\lambda_{i+\frac{1}{2}})(e^{\lambda_{i+1}} - e^{\lambda_i})$$

The local truncation error is $O(h^3)$, matching the standard midpoint method, while achieving greater numerical stability.

**Global error bound** (Theorem 1):

$$\|{\tilde{x}_{\epsilon,MEP} - x_0}\| \leq C + \sigma_\epsilon \tilde{\eta} \sum_{i=0}^{N-1} \tilde{\epsilon}_{t(\lambda_{i+\frac{1}{2}})}(e^{\lambda_{i+1}} - e^{\lambda_i})$$

Minimizing the schedule-dependent summation term yields the MEP objective:

$$\mathcal{J}_{\text{MEP}}(\Lambda) = \sum_{i=0}^{N-1} \tilde{\epsilon}_{t(\lambda_{i+\frac{1}{2}})}(e^{\lambda_{i+1}} - e^{\lambda_i})$$

**Advantages of MEP**:
- Solver-agnostic (not bound to specific solvers such as UniPC)
- Linear-time $O(N)$ computation
- Avoids numerical instability by exactly integrating the exponential term

#### 2. Spacing-Penalized Fitness (SPF)

Addresses the ill-conditioned problem of pathologically close timesteps produced by unconstrained optimization:

$$\mathcal{F}_{\text{SPF}}(\psi|N) = \mathcal{J}_{\text{MEP}}(\Lambda_{\text{opt}}) + \gamma L_{\text{penalty}}(\Lambda_{\text{opt}}|N)$$

Penalty term:
$$L_{\text{penalty}} = \sum_{i=0}^{N-1}\max(0, d_{\min}(N) - |t(\lambda_{i+1}) - t(\lambda_i)|)^2$$

$d_{\min}(N)$ adapts to the NFE budget: 0.15 at NFE=4 (enforcing large spacing) and 0.01 at NFE=20 (permitting fine-grained steps).

### Loss & Training

HSO does not involve a training loss; instead it optimizes objective functions. The core consists of the synergistic interaction between the MEP objective (lower level) and the SPF fitness (upper level).

## Experiments

### Experimental Setup

- Model: Stable Diffusion v2.1-base
- Solvers: UniPC, DDIM
- Datasets: LAION-Aesthetics 6.5+ (~30K pairs), MS-COCO 2017 val (30K), ImageNet 512×512 val (50K)
- Metric: FID (Fréchet Inception Distance)
- Search space: $\rho \in [3,16]$, $t_\epsilon \in [0.01,0.03]$, $t_{\max} \in [0.96,1.0]$

### Main Results

**FID Comparison (HSO vs. DM-NonUni)**:

| Dataset | Solver | NFE | DM-NonUni FID | **HSO FID** |
|---------|--------|-----|-------------|-----------|
| LAION | UniPC | 4 | 18.96 | **15.71** |
| LAION | UniPC | 5 | 13.91 | **11.94** |
| LAION | DDIM | 4 | 68.92 | **24.77** |
| LAION | DDIM | 5 | 35.38 | **17.17** |
| MS-COCO | UniPC | 4 | 27.50 | **23.26** |
| MS-COCO | DDIM | 5 | 30.12 | **23.15** |
| ImageNet 512 | UniPC | 4 | 20.75 | **17.20** |
| ImageNet 512 | DDIM | 4 | 41.51 | **19.78** |

Improvements are particularly pronounced on DDIM (LAION: 68.92→24.77 at NFE=4), validating the advantage of MEP's solver-agnostic design.

### Adaptivity Validation

**NFE adaptivity**: The optimal parameters exhibit complex non-monotonic relationships across NFE settings ($\rho^*$ peaks at 12.42 for NFE=8), demonstrating the infeasibility of fixed rule-based methods.

**Model adaptivity**: FID decreases from 37.65 to 18.05 on PixArt-α, validating cross-model generalization.

### Robustness Validation

| Condition | Schedule Example | Min Spacing | FID |
|-----------|----------------|-------------|-----|
| Without penalty | [999, 70, 9, 9] | 0.0 | 165.48 (collapsed) |
| With SPF | [959, 716, 370, 30] | 243.0 | 19.76±0.25 |

Without SPF, "tail clustering" occurs (timesteps collapse to [9, 9]), causing catastrophic FID degradation.

### Ablation Study

| Configuration | UniPC FID | DDIM FID |
|--------------|----------|---------|
| Baseline (DM-NonUni) | 18.07 | 71.57 |
| + Bilevel search | 11.44 | 29.22 |
| + MEP (without bilevel) | 26.33 | 37.27 |
| **Full HSO** | **15.70** | **24.80** |

### Computational Efficiency

| Method | Paradigm | Preparation Cost | NFE | FID |
|--------|---------|-----------------|-----|-----|
| **HSO** | Hierarchical optimization | **~8 sec** | 4 | 15.71 |
| AutoDiffusion | Evolutionary search | ~1.1 days | 4 | 17.86 |
| DM-NonUni | Local optimization | ~1 sec | 4 | 18.96 |
| LCM | Distillation | ~1.33 days | 4 | 11.10 |
| Mean Flows | Flow matching | ~60 days | 1 | 3.43 |

Among training-free methods, HSO achieves optimal performance at an 8-second cost, requiring orders of magnitude less preparation time than training-based methods.

### Key Findings

- The synergistic effect of the bilevel framework is essential: the upper level finds better initialization to help the lower level escape local optima; neither component is dispensable.
- MEP presents a "foreseeable trade-off": it is marginally inferior to solver-specific objectives on certain UniPC tasks, but substantially outperforms on DDIM, demonstrating the value of generality.
- SPF is a necessary safeguard for practical deployment: theoretical optimality does not imply practical optimality, and timestep clustering can lead to catastrophic failure.

## Highlights & Insights

1. **Elegant problem decomposition**: The $N$-dimensional non-convex optimization is decomposed into a 3-dimensional global search plus $N$-dimensional local refinement, overcoming the curse of dimensionality.
2. **Solid theoretical contribution of MEP**: By exactly integrating the exponential term and approximating only the network term, numerical stability is improved while maintaining the same error order.
3. **SPF bridges the gap between theory and practice**: The paper directly identifies the collapse of theoretically optimal schedules in practice and provides an adaptive solution.
4. **8 seconds vs. 1.1 days**: Compared to AutoDiffusion, HSO achieves superior performance with a 4-order-of-magnitude reduction in search cost, offering substantial practical deployment value.

## Limitations & Future Work

- Validation is limited to Stable Diffusion v2.1 and PixArt-α; more recent models (e.g., SD3, FLUX) have not been tested.
- Marginal gains may diminish at NFE ≥ 10, where the benefit of schedule optimization decreases as NFE increases.
- The boundary of the search space $\boldsymbol{\psi}$ still requires manual specification, introducing a degree of heuristic design.
- The paper's classification under `medical_imaging` appears questionable — this work is a general diffusion model acceleration method with no direct connection to medical imaging.
- The linear heuristic for $d_{\min}(N)$ is empirically effective but lacks theoretical justification.

## Related Work & Insights

- **Training-based acceleration**: Knowledge distillation (LCM), consistency models, Mean Flows
- **Training-free acceleration**: Solver design (DPM-Solver, UniPC, DDIM), model-level optimization (quantization/caching)
- **Schedule optimization**: Rule-based (EDM schedule), perception-based (AutoDiffusion, OMS-DPM), theory-based (DM-NonUni)

## Rating

⭐⭐⭐⭐⭐ (5/5)

- Solid theoretical contributions (complete derivation and proof of MEP) with high engineering value (8-second optimization cost)
- Clear systematic analysis framework organized around four design principles
- Experiments cover three large-scale benchmarks, two solvers, and comprehensive validation of adaptivity, robustness, and efficiency
- The bilevel optimization design is concise and elegant; SPF addresses a previously overlooked yet practically critical issue

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](../../CVPR2025/image_generation/easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](../../ICLR2026/image_generation/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)

</div>

<!-- RELATED:END -->
