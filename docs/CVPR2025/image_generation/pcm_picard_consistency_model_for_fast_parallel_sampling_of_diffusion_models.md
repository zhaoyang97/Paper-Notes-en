---
title: >-
  [Paper Note] PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Parallel Sampling] PCM proposes the Picard Consistency Model to accelerate the parallel sampling of diffusion models via Picard iteration. By training a model to directly predict the fixed-point solution and introducing a model switching mechanism to ensure exact convergence, it achieves up to a 2.71x speedup in image generation and robot control tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Parallel Sampling"
  - "Picard Iteration"
  - "Consistency Models"
  - "Diffusion Acceleration"
  - "Exact Convergence"
date: 2026-05-08
content_hash: 2a2f9ecb9dde3a48
---

# PCM: Picard Consistency Model for Fast Parallel Sampling of Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2503.19731](https://arxiv.org/abs/2503.19731)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Parallel Sampling, Picard Iteration, Consistency Models, Diffusion Acceleration, Exact Convergence

## TL;DR

PCM proposes the Picard Consistency Model to accelerate the parallel sampling of diffusion models via Picard iteration. By training a model to directly predict the fixed-point solution and introducing a model switching mechanism to ensure exact convergence, it achieves up to a 2.71x speedup in image generation and robot control tasks.

## Background & Motivation

**Background**: Diffusion models have achieved significant progress in fields such as vision, text, and robotics, but the sequential denoising process leads to slow generation speed. Acceleration approaches generally fall into two categories: (1) reducing the number of sampling steps (e.g., DDIM, Consistency Model, Progressive Distillation), which sacrifices quality or alters the output distribution; and (2) parallelizing computations, such as ParaDiGMS, which achieves parallel sampling based on Picard iteration and guarantees exact convergence.

**Limitations of Prior Work**: Although Picard iteration guarantees convergence to the exact output of the original model, its convergence rate is not guaranteed and can be practically slow. In the early stages of iteration, naive Picard yields extremely poor image quality (e.g., the FID is as high as 257 when k=2), which limits the practical speedup.

**Key Challenge**: The contradiction between guaranteeing exact convergence and the need for fast convergence—modifying model weights can accelerate convergence but alters the output distribution.

**Goal**: To accelerate the convergence rate of Picard iteration while ensuring the final results are completely identical to those of the original model.

**Key Insight**: Drawing an analogy to the Consistency Model concept—the Consistency Model trains a model to directly predict the final output $x_T$ from any point along the denoising trajectory; similarly, Picard iteration also forms a trajectory (in the $\mathbb{R}^{T \times n}$ space), which allows training a model to directly predict the fixed point $X^*$ from any point along the trajectory.

**Core Idea**: Train a "Picard Consistency Model" on the Picard iteration trajectory to accelerate convergence, then use a model switching strategy to smoothly transition back to the original model in the later stages of convergence, ensuring exact convergence.

## Method

### Overall Architecture

First, the original diffusion model is run via Picard iteration to collect a trajectory dataset $\{X^0, X^1, ..., X^K = X^*\}$. Then, PCM is trained: given any point $X^k$ on the trajectory, one Picard iteration $\Phi(X^k; \theta_{PCM})$ should directly jump to the fixed point $X^*$. During inference, model switching is applied: $\theta_{PCM}$ is used in the early stages to accelerate convergence, and a gradual transition to $\theta_{base}$ is performed in the later stages to guarantee exact convergence.

### Key Designs

1. **Picard Consistency Training**:

    - Function: Training the diffusion model to predict the fixed-point solution in Picard iterations.
    - Mechanism: Randomly sampling $X^k$ and the corresponding $X^* = X^K$ from the trajectory dataset, and minimizing the loss $\mathcal{L} = \mathbb{E}_{X \sim \mathcal{D}, k \sim \mathcal{U}[0,K-1]} \alpha(k) D(X^*, \Phi(X^k; \theta_{PCM}))$, where $\alpha(k) = \frac{1}{\sqrt{\text{Var}(k)}}$ is the weighting function. Exponential Moving Average (EMA) updates are used to stabilize training.
    - Design Motivation: The gap between $X^k$ and $X^*$ in the early stages of Picard iteration is large and noisy; the weighting function $\alpha(k)$ mitigates the variance discrepancies across different stages. EMA smooths out the instability of transition points during the training process.

2. **Feature-space Model Switching**:

    - Function: Guaranteeing that PCM eventually converges to an output completely identical to the original model.
    - Mechanism: Utilizing a linear interpolation schedule function $\lambda(k) = \max(0, \min(1, 1 - s \cdot k/K))$ to smoothly transition between the outputs of PCM and the original model: $X^{k+1} = \lambda(k) \cdot \Phi(X^k, \theta_{PCM}) + (1-\lambda(k)) \cdot \Phi(X^k, \theta_{base})$.
    - Design Motivation: PCM alters the weights, resulting in a different convergence point than the original model. Without switching, PCM would converge faster initially but with non-zero final error. Model switching leverages the rapid initial convergence of PCM while guaranteeing exact convergence via the original model in the later stages.

3. **Parameter-space Model Switching with LoRA**:

    - Function: Reducing storage and computational overhead.
    - Mechanism: Only training LoRA parameters $\Delta W$ and tuning the LoRA scale during inference to achieve switching: $h^k = (W_0 + \lambda(k) \Delta W) x^k$. $\lambda(k)$ decays from 1 to 0, achieving a smooth transition from PCM to the original model.
    - Design Motivation: Feature-space switching incurs double the model storage and inference costs; the LoRA approach only requires extra storage for low-rank parameters, and the weight mixing can be pre-computed offline.

### Loss & Training

L2 distance is used as the metric function $D(\cdot, \cdot)$. PCM is initialized from pre-trained weights, with an EMA decay factor of $\mu = 0.999$. The trajectory dataset consists of 500 generated samples, trained for 50 epochs using the Adam optimizer with a learning rate of 1e-4.

## Key Experimental Results

### Main Results (LDM-CelebA, DDIM)

| Method | Sequential Steps | FID↓ | Latency | Speedup |
|------|---------|------|-----|--------|
| Sequential | 18 | 36.09 | 2.83s | 1x |
| Picard | 6 | 36.19 | 1.34s | 2.11x |
| PCM | 6 | 36.67 | 1.87s | 1.51x |
| PCM-LoRA | 6 | 35.97 | 1.34s | 2.11x |
| **Extreme Comparison (k=2)** | | | | |
| Sequential | 2 | 366.92 | 0.32s | - |
| Picard | 2 | 257.83 | 0.44s | - |
| **PCM-LoRA** | **2** | **67.74** | **0.44s** | **-** |

### Ablation Study

| Configuration | Description of Effects |
|------|---------|
| PCM w/o model switching | Fast initially, but fails to converge to the original output in the end |
| PCM w/o EMA | Unstable training; transition points vary with each epoch |
| Feature-space switching | Requires double the inference overhead, but yields more stable convergence |
| LoRA switching | No added inference overhead, but requires selecting an appropriate stiffness |

### Key Findings

- **PCM shows a massive advantage under extremely few steps**: At k=2, PCM-LoRA achieves an FID of 67.74 vs Picard's 257.83, a difference of approximately 4x.
- **After convergence, the output of PCM is completely identical to that of the original model**—model switching successfully ensures exact convergence (validated via qualitative comparison in Fig. 5).
- PCM-LoRA performs closely to the full PCM while incurring no additional inference cost.
- On Stable Diffusion, PCM also maintains a better CLIP score, and PCM-LoRA at k=8 outperforms Picard at k=9.

## Highlights & Insights

- **The "consistency model on Picard trajectories" is a highly elegant analogy**—generalizing the core idea of Consistency Models from denoising trajectories to Picard iteration trajectories is a natural yet previously unexplored extension.
- **Model switching addresses the contradiction between modifying weights and exact convergence**—accelerating the initial convergence with PCM while ensuring final precision with the original model. This "coarse-to-fine" strategy is insightful for many other scenarios.
- **Implementing parameter-space switching with LoRA** not only saves storage but also enables pre-computing the weight mixture, avoiding extra latency. This represents a highly creative application of LoRA in inference optimization.

## Limitations & Future Work

- The requirement of pre-collecting a Picard trajectory dataset increases the upfront cost.
- The stiffness parameter for model switching must be manually tuned, lacking an adaptive strategy.
- Feature-space switching requires double the inference cost, which, though parallelizable, is impractical in resource-constrained scenarios.
- The effectiveness and gains on larger-scale diffusion models (e.g., SDXL, FLUX) remain to be validated.
- Combinations with other acceleration technologies (e.g., step distillation, quantization) can be explored.

## Related Work & Insights

- **vs ParaDiGMS**: ParaDiGMS first applied Picard iteration to parallel sampling of diffusion models, but suffer from slow convergence; PCM significantly accelerates the early stages of convergence through consistency training.
- **vs Consistency Model**: Consistency Model trains on the denoising trajectory $x_t \in \mathbb{R}^n$ to directly predict $x_T$; PCM trains on the Picard trajectory $X \in \mathbb{R}^{T \times n}$ to predict the fixed point $X^*$, which, despite the higher dimensionality, shares a consistent philosophy.
- **vs Progressive Distillation**: Distillation methods alter the model's output distribution and do not guarantee exact convergence; PCM's model switching guarantees that the final output is completely identical to that of the original model.

## Rating

- Novelty: ⭐⭐⭐⭐ The consistency model on Picard trajectories is an elegant new approach, and model switching is also highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across both image generation and robot control fields, though the model scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations with well-explained motivation.
- Value: ⭐⭐⭐ Theoretical contributions are prominent, but the hardware prerequisites for parallel inference limit practical application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](see_further_when_clear_curriculum_consistency_model.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/image_generation/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)

</div>

<!-- RELATED:END -->
