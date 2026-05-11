---
title: >-
  [Paper Note] Inference-Time Diffusion Model Distillation
description: >-
  [ICCV 2025][Image Generation][Diffusion Distillation] This paper proposes Distillation++, an inference-time diffusion distillation framework that leverages a pretrained teacher model during the student model's sampling process to correct its denoising trajectory, significantly narrowing the teacher–student performance gap without requiring additional training data or fine-tuning.
tags:
  - ICCV 2025
  - Image Generation
  - Diffusion Distillation
  - Inference-Time Distillation
  - Score Distillation Sampling
  - Teacher-Guided Sampling
  - Few-Step Generation
date: 2026-05-08
content_hash: 0e3ef3c2c05f690c
---

# Inference-Time Diffusion Model Distillation

**Conference**: ICCV 2025
**arXiv**: [2412.08871](https://arxiv.org/abs/2412.08871)
**Code**: [GitHub](https://github.com/geon-yeong/Distillation-pp)
**Area**: Diffusion Models / Image Generation
**Keywords**: Diffusion Distillation, Inference-Time Distillation, Score Distillation Sampling, Teacher-Guided Sampling, Few-Step Generation

## TL;DR

This paper proposes Distillation++, an inference-time diffusion distillation framework that leverages a pretrained teacher model during the student model's sampling process to correct its denoising trajectory, significantly narrowing the teacher–student performance gap without requiring additional training data or fine-tuning.

## Background & Motivation

Diffusion models generate high-quality images through iterative denoising, but sampling is slow, typically requiring tens to hundreds of NFEs. Distilled models (student models) compress this process to a few steps (e.g., 4 steps) by performing knowledge distillation from pretrained diffusion models (teacher models), greatly accelerating generation.

However, existing distilled models still face two core challenges:

**Teacher–Student Performance Gap**: Distilled models accumulate errors during multi-step sampling, resulting in lower generation quality than the teacher. For instance, Consistency Models do not necessarily improve in quality as NFE increases, since consistency errors accumulate across time intervals.

**Distribution Shift**: Some methods incorporate real training data to bridge the gap, but mismatches between teacher and student data distributions can degrade performance on out-of-distribution (OOD) prompts.

**Lack of Post-Training Options**: Many distilled models directly predict PF-ODE endpoints rather than trajectory tangent directions, making them incompatible with conventional ODE solvers and limiting opportunities for post-training improvement.

Existing work performs distillation exclusively during training. This paper raises a key insight: **can the teacher model continue to guide the student model's sampling at inference time?** This opens up the novel direction of inference-time distillation.

## Method

### Overall Architecture

The core idea of Distillation++ is to introduce teacher model guidance during the student model's sampling process—particularly in the first one or two steps—to correct the denoising trajectory. Concretely, the student's sampling is reformulated as a proximal optimization problem, with a Score Distillation Sampling (SDS) loss as the regularization term.

### Key Designs

1. **SDS Distillation Loss ($\ell_{\text{distill}}$)**:

    - **Function**: Defines the alignment loss between the student's denoising estimate and the teacher model.
    - **Mechanism**: The student's denoising estimate $\hat{x}_0^\theta(t)$ is re-noised to timestep $s$, then denoised by the teacher to obtain $\hat{x}_0^\psi(s)$. The SDS loss simplifies to:
    $\ell_{\text{distill}}(x; \psi, s) = \frac{\bar{\alpha}_s}{1-\bar{\alpha}_s} \|x - \hat{x}_0^\psi(s)\|_2^2$
    - **Design Motivation**: A high-quality denoising estimate should be recoverable by the teacher model even after random perturbation. This builds on the success of the SDS framework in distillation training, extending it to the inference phase.

2. **Inference-Time Teacher-Guided Update Rule**:

    - **Function**: Fuses the student and teacher estimates at each sampling step.
    - **Mechanism**: By leveraging the Decomposed Diffusion Sampling (DDS) framework to bypass intractable Jacobian computations, a concise update formula is obtained:
    $\hat{x}_{\text{new}}^\theta(t) = (1-\lambda)\hat{x}_0^\theta(t) + \lambda \hat{x}_0^\psi(s)$
      where $\lambda$ is the guidance strength. This is equivalent to interpolating between the student and teacher denoising estimates, which is then substituted into the DDIM sampling formula for the next step.
    - **Design Motivation**: This interpolation form resembles the conditional guidance mechanism in CFG, but the guidance direction comes from the teacher model rather than a text condition, hence the term "Teacher Guidance."

3. **Renoising Schedule**:

    - **Function**: Defines the relationship between the teacher evaluation timestep $s$ and the current step $t$.
    - **Mechanism**: A decreasing timestep schedule $s = t - \Delta t$ is adopted instead of the random timesteps used in conventional SDS, simulating the progressive refinement of the reverse diffusion process.
    - **Design Motivation**: Student models typically learn to jump to the endpoint of each sub-interval; applying teacher correction at these endpoints yields the best results. Experiments show that the decreasing schedule outperforms both random and fixed schedules.

### Loss & Training

Distillation++ is a **training-free** framework requiring no gradient updates or fine-tuning. It modifies only the denoising process during inference sampling:

- Teacher guidance is applied only in the first one or two steps, minimizing additional computational overhead.
- A simple constant $\lambda$ is used as the guidance strength.
- Compatible with multiple student models (LCM, DMD2, SDXL-Lightning, etc.) and multiple solvers (Euler, DPM++ 2S Ancestral).

## Key Experimental Results

### Main Results

Quantitative evaluation on MS-COCO 10K with 4-step baseline sampling + 1-step inference-time distillation:

| Model | FID↓ | ImageReward↑ | PickScore↑ |
|------|------|-------------|------------|
| LCM | 20.674 | 0.561 | 0.494 |
| **LCM++** | **20.149** | **0.597** | **0.505** |
| LCM-LoRA | 20.300 | 0.494 | 0.490 |
| **LCM-LoRA++** | **19.815** | **0.522** | **0.510** |
| SDXL-Lightning | 24.506 | 0.787 | 0.496 |
| **Light++** | **23.876** | **0.820** | **0.503** |
| DMD2 | 21.238 | 0.777 | 0.490 |
| **DMD2++** | **20.937** | **0.797** | **0.510** |
| SDXL-Turbo | 18.612 | 0.296 | 0.499 |
| **Turbo++** | **18.481** | **0.310** | **0.501** |

### Ablation Study

| Configuration | FID↓ | ImageReward↑ | Notes |
|------|------|-------------|------|
| DMD2 Baseline | 21.238 | 0.777 | No teacher guidance |
| s=random t | 21.105 | 0.771 | Random timestep |
| s=t | 21.342 | 0.777 | Synchronized timestep |
| **s=t−Δt** | **20.937** | **0.797** | Decreasing timestep (best) |

Computational cost comparison (LCM, 4+1 steps vs. 5 steps vs. 6 steps):

| Metric | 4+1 Steps | 5 Steps | 6 Steps |
|------|-------|-----|-----|
| FID↓ | **20.149** | 20.732 | 21.540 |
| ImageReward↑ | **0.597** | 0.593 | 0.585 |
| Time (s) | 1.987 | 1.996 | 2.250 |

### Key Findings

- Distillation++ **consistently** improves FID, ImageReward, and PickScore across all distillation baselines.
- Adding one teacher evaluation step incurs latency comparable to or less than adding one student sampling step, owing to parallel computation.
- Increasing the number of student sampling steps does not guarantee improvements in semantic alignment or physical plausibility, whereas teacher guidance does.

## Highlights & Insights

- **First Inference-Time Distillation Framework**: Unlike all existing approaches that distill only during training, Distillation++ continuously leverages teacher guidance during sampling, establishing a paradigm of lifelong teacher–student collaboration.
- **Data-Free and Fine-Tuning-Free**: Entirely free from additional training data or parameter updates, serving as a plug-and-play post-training improvement scheme.
- **General Compatibility**: Applicable to diverse distilled model types and solvers.
- **Early-Step Guidance Is Most Effective**: Spatial layout is largely determined in the early sampling steps; guiding only the first one or two steps is sufficient to achieve substantial improvement.

## Limitations & Future Work

- Loading the teacher model (e.g., SDXL) at inference time increases GPU memory requirements.
- Validation is currently limited to image generation; video diffusion distillation is a promising direction for extension.
- A constant $\lambda$ may be suboptimal; timestep-dependent guidance strength warrants further exploration.
- Synergistic sampling with Flow Matching models is worth investigating.

## Related Work & Insights

- Closely related to conditional sampling works such as DreamSampler and CFG++, extending their ideas from conditional guidance to teacher–student distillation guidance.
- The SDS loss is widely used in 3D generation; this paper repurposes it from a training objective to an inference-time optimization target.
- Insight: Inference-time compute is valuable not only for LLMs but also for diffusion models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose the concept of inference-time distillation, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-baseline validation with thorough ablation and computational cost analysis.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear; the simplification from SDS to the interpolation form is elegant.
- Value: ⭐⭐⭐⭐ A practical plug-and-play solution, though the additional teacher model overhead limits deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](../../NeurIPS2025/image_generation/remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](../../ICLR2026/image_generation/large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)

</div>

<!-- RELATED:END -->
