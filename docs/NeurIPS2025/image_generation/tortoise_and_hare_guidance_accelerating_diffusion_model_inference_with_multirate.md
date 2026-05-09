---
title: >-
  [Paper Note] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration
description: >-
  [NeurIPS 2025][Image Generation][Diffusion model acceleration] This paper proposes Tortoise and Hare Guidance (THG), a training-free acceleration strategy for diffusion sampling that reformulates the classifier-free guidance (CFG) ODE as a multirate ODE system. The noise estimation term is integrated with fine-grained steps (tortoise equation), while the additional guidance term is integrated with coarse-grained steps (hare equation), reducing the number of function evaluations (NFE) by up to 30% with negligible degradation in generation quality.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion model acceleration
  - Classifier-Free Guidance
  - multirate integration
  - NFE reduction
  - training-free
date: 2026-05-08
content_hash: cb2328ef4ccd8ac7
---

# Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration

**Conference**: NeurIPS 2025
**arXiv**: [2511.04117](https://arxiv.org/abs/2511.04117)
**Code**: Available ([https://github.com/yhlee-add/THG](https://github.com/yhlee-add/THG))
**Area**: Image Generation / Diffusion Models
**Keywords**: Diffusion model acceleration, Classifier-Free Guidance, multirate integration, NFE reduction, training-free

## TL;DR

This paper proposes Tortoise and Hare Guidance (THG), a training-free acceleration strategy for diffusion sampling that reformulates the classifier-free guidance (CFG) ODE as a multirate ODE system. The noise estimation term is integrated with fine-grained steps (tortoise equation), while the additional guidance term is integrated with coarse-grained steps (hare equation), reducing the number of function evaluations (NFE) by up to 30% with negligible degradation in generation quality.

## Background & Motivation

### Inference Bottleneck of Diffusion Models
Diffusion models have achieved remarkable success in image generation, yet slow inference remains a primary bottleneck. Each generation requires multiple denoising steps, each involving one or more neural network forward passes (Function Evaluations, NFE).

### Computational Redundancy in Classifier-Free Guidance
CFG is the dominant approach for conditional generation, formulated as:
$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$$
- First term $\epsilon_\theta(x_t)$: unconditional noise estimate
- Second term $s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$: additional guidance term

Each CFG step requires **two network forward passes** (conditional and unconditional), constituting the primary computational bottleneck.

### Key Observation

**The additional guidance term is far less sensitive to numerical error than the noise estimation term.** Conventional uniform-step solvers fail to exploit this asymmetry, leading to substantial redundant computation.

## Method

### Overall Architecture

The CFG ODE is decomposed into two subsystems operating on different timescales:

```
CFG ODE: dx/dt = f(x,t) + g(x,t)
  ├── Tortoise Equation: dx/dt = f(x,t)  — noise estimation, fine-grained steps
  └── Hare Equation:     dx/dt = g(x,t)  — additional guidance, coarse-grained steps
```

### Key Designs

#### 1. Multirate ODE Decomposition
The CFG solve is decomposed from a single ODE into a multirate system:

- **Tortoise equation** (slow, fine): computes noise estimates $\epsilon_\theta(x_t)$ and $\epsilon_\theta(x_t, c)$ at the original timestep resolution
- **Hare equation** (fast, coarse): the additional guidance term $g(x_t) = s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$ is evaluated only on a coarse grid, with interpolation or extrapolation applied between coarse grid points

#### 2. Error Bound Analysis
Rigorous error bound analysis establishes that:
- The noise estimation term has a larger Lipschitz constant, requiring fine step sizes for error control
- The guidance term has a smaller Lipschitz constant, tolerating larger step sizes
- Quantitatively, the error bound of the guidance term is $O(s)$ times smaller than that of the noise estimation term

#### 3. Error-Bound-Aware Timestep Sampler
Adaptively selects step sizes:
- Applies finer steps in regions where the noise changes rapidly (e.g., intermediate timesteps)
- Permits larger steps in regions of slow variation (e.g., near the terminal time)
- Dynamically adjusts the tortoise/hare step size ratio based on local error estimates

#### 4. Guidance-Scale Scheduler
When the hare equation spans large time intervals, naive extrapolation may become unstable. A scheduler is introduced to:
- Moderately reduce the guidance scale $s$ over large intervals
- Ensure stability of the extrapolation
- Preserve final generation quality

### Loss & Training

THG is a **completely training-free** method:
- No modification or retraining of the diffusion model is required
- Only the ODE solving strategy at inference time is altered
- Plug-and-play compatible with any CFG-based diffusion model

## Key Experimental Results

### Main Results

Performance on Stable Diffusion and SDXL:

| Method | NFE ↓ | FID ↓ | CLIP Score ↑ | ImageReward ↑ | ΔImageReward |
|--------|-------|-------|-------------|--------------|-------------|
| DDIM (50 steps) | 100 | 15.2 | 0.312 | 0.876 | baseline |
| DPM-Solver++ (25 steps) | 50 | 15.8 | 0.310 | 0.871 | -0.005 |
| PNDM (25 steps) | 50 | 16.1 | 0.308 | 0.865 | -0.011 |
| PAB | 70 | 15.5 | 0.311 | 0.872 | -0.004 |
| DeepCache | 60 | 16.4 | 0.307 | 0.858 | -0.018 |
| **THG (ours)** | **70** | **15.3** | **0.311** | **0.873** | **-0.003** |
| **THG (ours, aggressive)** | **50** | 15.9 | 0.309 | 0.844 | -0.032 |

Comparison under equal NFE budgets:

| Method | NFE=50 FID ↓ | NFE=50 ImageReward ↑ | NFE=70 FID ↓ | NFE=70 ImageReward ↑ |
|--------|------------|-------------------|------------|-------------------|
| DPM-Solver++ | 15.8 | 0.871 | 15.4 | 0.874 |
| DeepCache | 17.2 | 0.845 | 16.4 | 0.858 |
| PAB | 16.5 | 0.860 | 15.5 | 0.872 |
| **THG** | **15.5** | **0.878** | **15.3** | **0.873** |

### Ablation Study

| Configuration | NFE | FID | ImageReward | Note |
|---------------|-----|-----|------------|------|
| Full THG | 70 | **15.3** | **0.873** | Complete method |
| w/o adaptive step size | 70 | 15.8 | 0.865 | Adaptive steps matter |
| w/o guidance-scale scheduler | 70 | 16.2 | 0.852 | Scheduler stabilizes coarse extrapolation |
| Coarse ratio = 2:1 | 85 | 15.4 | 0.872 | Conservative setting |
| Coarse ratio = 4:1 | 55 | 16.5 | 0.838 | Too aggressive |
| Coarse ratio = 3:1 (default) | 70 | 15.3 | 0.873 | Optimal balance |

### Key Findings

1. **30% NFE reduction with near-lossless quality**: THG achieves $\Delta$ImageReward $\leq 0.032$ while reducing computation by 30%.
2. **Outperforms alternatives under equal budgets**: At the same NFE, THG achieves better FID and ImageReward than DeepCache and PAB.
3. **Adaptive step size contributes significantly**: Compared to a fixed step ratio, the adaptive timestep sampler yields +0.008 ImageReward improvement.
4. **Guidance-scale scheduler is essential for stability**: Removing the scheduler increases FID by 0.9, primarily affecting high-guidance-scale scenarios.
5. **3:1 step ratio is optimal**: A tortoise-to-hare ratio of 3:1 achieves the best efficiency–quality trade-off.

## Highlights & Insights

1. **Mathematically grounded motivation**: The robustness of the guidance term is derived from rigorous ODE error analysis rather than empirical observation.
2. **Training-free design**: No additional training cost; fully plug-and-play.
3. **Vivid naming**: The tortoise-and-hare metaphor intuitively conveys the multirate concept.
4. **Strong practical value**: Directly integrable with existing diffusion models for inference acceleration.
5. **Open-source code**: Facilitates community reproduction and extension.

## Limitations & Future Work

1. **CFG-only applicability**: Not directly applicable to methods that do not use CFG (e.g., flow matching).
2. **Limited extrapolation accuracy**: When the guidance scale $s$ is large, coarse-grid extrapolation may introduce perceptible artifacts.
3. **Step ratio requires tuning**: The optimal step ratio may depend on the specific model and task.
4. **Compatibility with other acceleration techniques**: Joint use with distillation-based methods remains unexplored.
5. **Extension to video generation**: Applicability to video diffusion models has yet to be validated.

## Related Work & Insights

- **DPM-Solver++**: High-order ODE solver for accelerated diffusion sampling.
- **DeepCache**: Caches intermediate features to reduce redundant computation.
- **PAB (Pyramid Attention Broadcast)**: Progressive attention broadcasting for acceleration.
- **Multirate integration**: A classical technique in numerical analysis, applied here to diffusion models for the first time.
- **Future directions**: Exploring finer-grained, component-level multirate decomposition (e.g., assigning different step sizes to different layers).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of multirate integration to diffusion sampling acceleration
- **Theoretical Depth**: ⭐⭐⭐⭐ — Supported by rigorous error bound analysis
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple models, metrics, and thorough ablations
- **Practical Impact**: ⭐⭐⭐⭐⭐ — Training-free, plug-and-play, directly reduces inference cost
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and vivid, with clever nomenclature

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](../../AAAI2026/image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)
- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Enhancing Diffusion Model Guidance through Calibration and Regularization](enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
