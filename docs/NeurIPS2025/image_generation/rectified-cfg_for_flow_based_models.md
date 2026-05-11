---
title: >-
  [Paper Note] Rectified-CFG++ for Flow Based Models
description: >-
  [NeurIPS 2025][Image Generation][Classifier-Free Guidance] To address the off-manifold drift caused by standard CFG in Rectified Flow models…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "Rectified Flow"
  - "Text-to-Image Generation"
  - "Predictor-Corrector Sampling"
  - "Flow Models"
date: 2026-05-08
content_hash: 0ab9facc51ca273a
---

# Rectified-CFG++ for Flow Based Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.07631](https://arxiv.org/abs/2510.07631)
**Authors**: Shreshth Saini, Shashank Gupta, Alan C. Bovik (UT Austin)
**Code**: [rectified-cfgpp.github.io](https://rectified-cfgpp.github.io/)
**Area**: Image Generation
**Keywords**: Classifier-Free Guidance, Rectified Flow, Text-to-Image Generation, Predictor-Corrector Sampling, Flow Models

## TL;DR

To address the off-manifold drift caused by standard CFG in Rectified Flow models, this paper proposes Rectified-CFG++—an adaptive predictor-corrector guidance strategy that replaces extrapolative guidance with conditional flow prediction combined with time-scheduled interpolative correction. The method comprehensively outperforms standard CFG on large-scale models including Flux, SD3, SD3.5, and Lumina.

## Background & Motivation

### State of the Field
Classifier-Free Guidance (CFG) is the core technique for controlling conditional generation quality in diffusion models, enhancing text alignment by linearly extrapolating between conditional and unconditional velocity fields. However, Rectified Flow (RF) models employ deterministic ODE integration without stochastic regularization, and the extrapolative nature of CFG causes sampling trajectories to deviate from the learned data manifold, producing visual artifacts such as oversaturation, structural distortion, and text errors.

### Limitations of Prior Work
- **Standard CFG**: Directly applies extrapolative combination $\hat{v}_\omega = (1-\omega)v^u + \omega v^c$ ($\omega \geq 1$) in RF models, pushing trajectories off the manifold $\mathcal{M}_t$
- **CFG++**: A manifold-constrained guidance method designed for diffusion models, not optimized for the geometric structure of RF
- **APG (Analytical Posterior Guidance)**: Partially mitigates artifacts but compromises on detail or geometric accuracy
- **CFG-Zero★**: Provides limited improvement while remaining subject to the fundamental extrapolation problem
- None of the above methods offer **flow-model-specific** theoretical guarantees or geometry-aware design

### Root Cause
The geometric structure of RF models is naturally suited to **interpolation** rather than extrapolation. Designing a sampling strategy that incorporates guidance signals via interpolation—leveraging the deterministic transport paths of conditional flows—can achieve high-quality conditional generation while maintaining manifold consistency.

## Method

### Core Idea: Predictor-Corrector as a Replacement for Extrapolation

The standard CFG update is extrapolative: $x_{t-\Delta t} = x_t + \Delta t(v^u_t + \omega \Delta v^\theta_t)$, where $\Delta v^\theta_t = v^c_t - v^u_t$. This extrapolation, lacking stochastic noise regularization in a deterministic ODE, is prone to divergence.

Rectified-CFG++ replaces this with **three steps**:

### Step 1: Conditional Flow Prediction (Predictor)

A half-step prediction is performed using the pure conditional velocity field $v^c_t$, advancing the sample along the conditional manifold:

$$\tilde{x}_{t-\Delta t/2} \leftarrow x_t + \frac{\Delta t}{2} v^c_t$$

The key insight is that using $v^c_t$ rather than $v^u_t$ or a CFG-mixed velocity anchors the trajectory to the target conditional subspace manifold from the outset, preventing early deviation.

### Step 2: Guidance Difference Correction (Corrector via Guidance Difference)

At the predicted midpoint $\tilde{x}_{t-\Delta t/2}$, both conditional and unconditional velocity fields are evaluated:

$$v^c_{t-\Delta t/2} \leftarrow v_\theta(\tilde{x}_{t-\Delta t/2}, t-\Delta t/2, y)$$
$$v^u_{t-\Delta t/2} \leftarrow v_\theta(\tilde{x}_{t-\Delta t/2}, t-\Delta t/2, \varnothing)$$

Evaluating the guidance difference $\Delta v^\theta_{t-\Delta t/2}$ at the intermediate predicted point is more accurate than evaluating it at the current point $x_t$—particularly when the velocity field changes rapidly.

### Step 3: Interpolative Update

The final effective velocity uses the conditional direction as an anchor, augmented by a time-scheduled guidance correction:

$$\hat{v}_{\lambda t} \leftarrow v^c_t + \alpha(t)(v^c_{t-\Delta t/2} - v^u_{t-\Delta t/2})$$

where the scheduling function is $\alpha(t) = \lambda_{\max}(1-t)^\gamma$, with $\lambda_{\max} > 0, \gamma \geq 0$. A standard ODE update is then performed using $\hat{v}_{\lambda t}$.

### Theoretical Guarantees

**Lemma 3.1 (Guidance Direction Stability)**: Under Lipschitz continuity assumptions, the difference between guidance differentials at the midpoint and at the current point is $O(\Delta t)$:
$$\|\Delta v^\theta_{t-\Delta t/2} - \Delta v^\theta_t(x_t)\| \leq L V_{\max} \Delta t$$

**Proposition 1 (Bounded Single-Step Perturbation)**: The single-step deviation of Rectified-CFG++ from the pure conditional flow is strictly bounded:
$$\|\hat{x}_{t-1} - \tilde{x}_{t-1}\| \leq \alpha(t) B \Delta t$$

This guarantees that the trajectory remains within a bounded tubular neighborhood of the data manifold $\mathcal{M}_t$, where the neighborhood size is controlled by $\alpha(t)$ and the guidance field bound $B$.

### Key Differences from CFG

| Property | Standard CFG | Rectified-CFG++ |
|----------|-------------|-----------------|
| Guidance mode | Extrapolation | Interpolation |
| Reference velocity | Unconditional $v^u_t$ | Conditional $v^c_t$ |
| Guidance evaluation point | Current point $x_t$ | Intermediate predicted point $\tilde{x}_{t-\Delta t/2}$ |
| Manifold preservation | No guarantee; prone to drift | Theoretically guaranteed bounded neighborhood |
| Additional network/training | No | No |

## Key Experimental Results

### Experiment 1: MS-COCO 10K Multi-Model Comprehensive Evaluation

Rectified-CFG++ is comprehensively compared against standard CFG across four mainstream RF models:

| Model | Guidance | FID↓ | CLIP↑ | Aesthetic↑ | ImageReward↑ | PickScore↑ | HPSv2↑ |
|-------|----------|------|-------|-----------|-------------|-----------|--------|
| Lumina | CFG | 26.93 | **0.3511** | **5.8226** | **1.0924** | 0.5867 | 0.2797 |
| Lumina | Rect-CFG++ | **22.49** | 0.3464 | 5.7755 | 0.9611 | **0.6133** | **0.3004** |
| SD3 | CFG | 23.89 | 0.3439 | 5.5465 | 0.9812 | 0.4408 | 0.2751 |
| SD3 | Rect-CFG++ | **23.39** | **0.3471** | **5.6529** | **1.0009** | **0.5591** | **0.2897** |
| SD3.5 | CFG | 20.29 | **0.3506** | 6.155 | 1.0487 | 0.4923 | 0.2933 |
| SD3.5 | Rect-CFG++ | **20.22** | 0.3497 | **6.1651** | **1.0796** | **0.5077** | **0.2946** |
| Flux-dev | CFG | 37.86 | 0.3351 | 4.721 | **1.0528** | 0.3248 | 0.2621 |
| Flux-dev | Rect-CFG++ | **32.23** | **0.3493** | **5.3251** | 0.948 | **0.6752** | **0.2996** |

On Flux-dev, FID drops from 37.86 to 32.23 (a 14.9% reduction) and PickScore nearly doubles from 0.3248 to 0.6752, indicating that standard CFG produces particularly severe artifacts on Flux and that Rectified-CFG++ achieves the largest gains on this model.

### Experiment 2: Guidance Strategy Comparison (MS-COCO 1K, SD3.5)

| Guidance Method | FID↓ | ImageReward↑ | CLIP↑ | HPSv2↑ |
|----------------|------|-------------|-------|--------|
| No guidance | 77.30 | 0.3852 | 0.3260 | 0.2421 |
| CFG | 67.71 | 1.0530 | **0.3515** | 0.2941 |
| CFG-Zero★ | 68.39 | 0.9947 | 0.3458 | 0.2879 |
| APG | 67.23 | 1.0748 | 0.3513 | 0.2935 |
| **Rect-CFG++** | **67.15** | **1.0845** | 0.3506 | **0.2959** |

Rectified-CFG++ achieves the best results on FID, ImageReward, and HPSv2, with CLIP score only marginally below standard CFG.

### T2I-CompBench Compositional Generation Evaluation

| Model | Color↑ | Shape↑ | Texture↑ | Spatial↑ |
|-------|--------|--------|----------|----------|
| Flux CFG | 0.6132 | 0.4152 | 0.5928 | 0.2488 |
| Flux Rect-CFG++ | **0.7728** | **0.5018** | **0.6705** | **0.2790** |
| SD3 CFG | 0.7658 | 0.5698 | 0.7270 | 0.3199 |
| SD3 Rect-CFG++ | **0.8041** | **0.5778** | **0.7362** | **0.3306** |

The Color attribute on Flux improves from 0.6132 to 0.7728 (+26%), demonstrating that the color shift problem of CFG on Flux is effectively corrected.

### Ablation Study: Component Contributions (MS-COCO 1K, SD3.5)

| Configuration | FID↓ | CLIP↑ | HPSv2↑ | Aesthetic↑ |
|--------------|------|-------|--------|-----------|
| Prediction with unconditional velocity | 91.12 | 0.1439 | 0.1870 | 6.1049 |
| Without Predictor | 73.70 | 0.3410 | 0.2969 | 6.1064 |
| Without Corrector | 74.65 | 0.3414 | 0.2975 | 6.1047 |
| **Full Rect-CFG++** | **72.97** | **0.3446** | **0.2995** | **6.1587** |

When the predictor uses the unconditional velocity, CLIP drops sharply to 0.14, confirming that the conditional prediction step is the core of the method.

### Computational Efficiency

Under comparable runtime (SD3.5, 512×512), Rectified-CFG++ achieves FID 74.47 with 20 NFE, while standard CFG reaches only FID 85.82 with 28 NFE. Actual FLOPs are nearly identical, with a runtime difference of approximately 0.04 seconds.

## Highlights & Insights

- **Principled and Elegant**: The core intuition of replacing extrapolation with interpolation is clear; the predictor-corrector framework naturally decouples conditional anchoring from guidance correction without requiring additional networks or training
- **Theoretical Completeness**: Rigorous mathematical proofs of manifold consistency and trajectory boundedness are provided, making this one of the few guidance methods that combines both theoretical and empirical rigor
- **Drop-in Replacement**: Requires no training, no modification of model weights, and negligible additional computation; can directly replace CFG modules in existing RF models
- **Significant Improvement in Text Rendering**: The method performs notably well on in-image text generation, a known weakness of diffusion models
- **Comprehensive Validation**: Covers 4 large-scale models, multiple datasets, 6+ metrics, ablation studies, and a user study, reflecting rigorous experimental design

## Limitations & Future Work

- **One Additional Forward Pass per Step**: The prediction step requires an extra conditional velocity field evaluation; although the total number of steps can be reduced, single-step cost approximately doubles
- **Introduced Hyperparameters**: The paper claims to be "parameter-free beyond guidance scale," yet the scheduling function $\alpha(t) = \lambda_{\max}(1-t)^\gamma$ introduces two additional hyperparameters
- **Text-to-Image Only**: The method is not validated in other RF model applications such as video generation or 3D generation
- **Some Metrics Worse on Lumina**: CLIP, Aesthetic, and ImageReward scores are lower on Lumina compared to standard CFG, indicating the method does not uniformly dominate across all settings
- **Area Misclassification**: This paper addresses generative model sampling methodology and is unrelated to the object detection area

## Related Work & Insights

- **CFG (Ho & Salimans 2022)**: The original extrapolative guidance method; produces severe artifacts in RF models. Rectified-CFG++ fundamentally resolves this by replacing extrapolation with interpolation
- **CFG++ (Chung et al. 2024)**: A manifold-constrained method designed for diffusion SDEs that relies on stochastic regularization and is not applicable to deterministic RF
- **APG (Sadat et al. 2024)**: Analytical posterior guidance that partially alleviates artifacts at the cost of detail; Table 3 shows Rectified-CFG++ outperforms APG on FID, ImageReward, and HPSv2
- **CFG-Zero★**: Reduces early drift by zeroing initial guidance but remains subject to extrapolation effects in later steps; overall performance is inferior to Rectified-CFG++
- **ReCFG**: A related work using guidance differences; Rectified-CFG++ is distinguished by its predictor-corrector framework and midpoint evaluation strategy

## Rating

- Novelty: ⭐⭐⭐⭐ — The core idea of replacing extrapolation with interpolation is simple yet effective; the predictor-corrector framework is well motivated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four models, multiple datasets, 6+ metrics, ablation studies, and a user study; extremely comprehensive
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, figures are informative, and structure is complete
- Value: ⭐⭐⭐⭐ — Strong practical utility as a drop-in replacement with direct contributions to the RF model ecosystem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)

</div>

<!-- RELATED:END -->
