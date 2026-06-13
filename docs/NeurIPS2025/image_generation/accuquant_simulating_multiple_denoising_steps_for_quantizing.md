---
title: >-
  [Paper Note] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Post-training quantization] This paper reveals an error accumulation phenomenon in diffusion model quantization—where quantization errors at each step propagate and amplify into subsequen…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Post-training quantization"
  - "diffusion models"
  - "error accumulation"
  - "multi-step sampling simulation"
  - "O(1) memory optimization"
date: 2026-05-08
content_hash: d54158358c0f19e0
---

# AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.20348](https://arxiv.org/abs/2510.20348)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Post-training quantization, diffusion models, error accumulation, multi-step sampling simulation, O(1) memory optimization

## TL;DR

This paper reveals an error accumulation phenomenon in diffusion model quantization—where quantization errors at each step propagate and amplify into subsequent steps—and proposes explicitly simulating consecutive multi-step denoising during PTQ calibration to jointly optimize quantization parameters, while reducing memory from O(n) to O(1) through a carefully designed objective function.

## Background & Motivation

The unique properties of diffusion model inference pose fundamentally different quantization challenges compared to conventional models:

The sampling process of a diffusion model is inherently iterative—starting from pure noise, it progressively denoises over tens of steps (typically 20–50) to produce a final image, with each step's output serving directly as the next step's input. This sequential dependency implies:

- **Error accumulation**: The quantization error $\delta_t$ at step $t$ not only affects step $t$'s output but is also added as an offset to step $t+1$'s input. Step $t+1$'s quantization introduces its own new error $\delta_{t+1}$ while also incurring additional deviation from the corrupted input. After multiple iterations, small errors can be progressively amplified.
- **Blind spot of existing PTQ methods**: Standard PTQ calibration optimizes each denoising step independently—minimizing the output discrepancy between the full-precision and quantized models for each step in isolation. This mimics the model's *training* paradigm (independent steps) rather than the *sampling* paradigm used at inference (sequential steps).
- **Paradigm mismatch**: The calibration objective is misaligned with the actual use scenario—quantization parameters are optimized for "single-step output error given a clean input," whereas at inference the model receives inputs that already contain accumulated errors.

The paper substantiates the existence and severity of this error accumulation phenomenon through both analytical derivations and empirical experiments, showing the effect is especially pronounced under low-bit (e.g., W4A4) quantization.

## Method

### Overall Architecture

AccuQuant's core modification occurs during PTQ calibration. Rather than optimizing quantization parameters step by step independently, it runs the quantized model through consecutive multi-step denoising (simulating the sampling trajectory), compares the accumulated multi-step outputs against the full-precision model, and jointly optimizes the quantization parameters across all steps in the window.

Calibration procedure:

1. Prepare a small set of calibration data (random noise + optional text prompts).
2. Select a denoising window $[t, t+k]$ (where $k$ is the window size).
3. Run the quantized model through $k$ consecutive denoising steps starting from step $t$.
4. Compare the output at step $t+k$ against the full-precision model's output along the same trajectory.
5. Backpropagate gradients to update the quantization parameters (e.g., scale factors, zero points) within the window.
6. Slide the window to cover the entire denoising trajectory.

### Key Designs

1. **Multi-step simulation calibration**: When determining quantization parameters, the quantized model's consecutive multi-step denoising process is explicitly executed. Starting from a given step, the quantized model denoises for $k$ consecutive steps and its outputs are compared with those of the full-precision model over the same $k$ steps. If the quantization at one step shifts the next step's input, the sequential simulation naturally exposes this cascading effect, allowing calibration to select quantization parameters that better suppress error propagation. This represents a paradigm shift from "mimicking training" to "mimicking sampling"—aligning the calibration objective with the actual inference scenario.

2. **O(1) memory objective function**: A naive implementation of consecutive $k$-step simulation requires storing intermediate activations at every step for backpropagation, resulting in O(k)~O(n) memory, which is entirely infeasible when $n=50$. The paper designs a clever objective function that avoids simultaneously storing activations from all intermediate steps. The core idea is computation rematerialization (analogous to gradient checkpointing), where only the state of the current step needs to be maintained to compute gradients, reducing memory complexity from O(n) to O(1). This optimization makes the method practical for long-step sampling scenarios.

3. **Windowed optimization strategy**: The entire denoising trajectory is divided into multiple overlapping windows, each subjected to multi-step simulation calibration. The window size $k$ is the critical hyperparameter—$k=1$ degenerates to conventional step-wise independent optimization; excessively large $k$ increases computational cost with diminishing returns. Experiments show $k=2$–$4$ to be optimal, capturing the most critical neighbor-step error propagation without incurring excessive computation.

### Loss & Training

The joint multi-step objective minimizes the discrepancy between the quantized model's accumulated outputs and the full-precision model's outputs within the window $[t, t+k]$. The specific form is a weighted sum of per-step output differences within the window, where weights can be uniform or biased toward later steps (as later steps have accumulated more error information).

The standard PTQ pipeline is employed: no original training data is required, and calibration is completed using a small number of randomly generated inputs. The amount of calibration data is comparable to traditional PTQ methods, introducing no additional data requirements.

## Key Experimental Results

### Main Results

Comparisons against existing PTQ methods are conducted across multiple diffusion models and tasks:

**CIFAR-10 unconditional generation (DDPM)**:

| Method | W8A8 FID ↓ | W4A8 FID ↓ | W4A4 FID ↓ |
|--------|-----------|-----------|-----------|
| Full Precision | Baseline | Baseline | Baseline |
| Naive PTQ | Slight degradation | Notable degradation | Severe degradation |
| Q-Diffusion | Moderate improvement | Moderate improvement | Moderate improvement |
| PTQD | Better | Better | Better |
| **AccuQuant** | **Best** | **Best** | **Best** |

**Key observation**: Under W8A8 (small errors), the gap between methods is small, but under W4A8 and W4A4 (large errors, severe accumulation), AccuQuant's advantage progressively widens—consistent with the theoretical prediction that "error accumulation is more severe at lower bit-widths."

The paper further validates consistent improvements on LDM, Stable Diffusion, and other models, as well as on text-guided generation and image editing tasks.

### Ablation Study

**Effect of window size $k$**:

| Window Size | Relative FID Improvement | Calibration Time Increase |
|-------------|--------------------------|--------------------------|
| $k=1$ (degenerates to step-wise) | Baseline | 1× |
| $k=2$ | Significant improvement | ~2× |
| $k=3$ | Further improvement | ~3× |
| $k=4$ | Near saturation | ~4× |
| $k=8$ | Almost no additional gain | ~8× |

$k=2$–$4$ is the optimal range—jointly optimizing the nearest 2–4 steps captures most of the error propagation information.

**O(1) memory vs. naive O(n) implementation**: The two variants show virtually no difference in quantization accuracy (confirming that gradient information can be efficiently recomputed), while the O(1) version achieves a peak memory footprint of only 1/50 that of the naive version under 50-step sampling.

**Different numbers of sampling steps**: AccuQuant consistently outperforms baselines across 20-step, 50-step, and 100-step DDIM/DDPM sampling. The more steps, the more severe the accumulation effect, and the more pronounced AccuQuant's advantage.

### Key Findings

- Quantization errors do accumulate across denoising steps—the paper tracks the L2 distance between each step's output and the full-precision model's output, observing that the distance increases monotonically with step count.
- The accumulation effect is most severe under low-bit quantization—the error growth rate of W4A4 far exceeds that of W8A8, exhibiting an exponential amplification trend.
- The gains from multi-step joint optimization are concentrated in the first few steps—$k=2$ already delivers most of the improvement, with marginal returns diminishing significantly beyond $k=4$.
- The O(1) memory optimization incurs almost no accuracy loss—the equivalent recomputation of gradients is exact and introduces no approximation error.

## Highlights & Insights

- **Problem-driven method design**: Starting from the concise observation that "errors accumulate over iterations," the solution of "making the calibration objective aware of multi-step accumulation effects" follows naturally—the logical chain is clear and compelling.
- **Paradigm shift**: Transitioning PTQ calibration from "mimicking training" (step-wise independent) to "mimicking sampling" (consecutive sequential) is an insightful conceptual reorientation.
- **Engineering contribution of O(n) → O(1)**: This makes the method practically feasible for long-sequence sampling; without it, the approach would remain a theoretically elegant but impractical proposal.
- **Strong generality**: Applicable to diverse diffusion architectures (U-Net, DiT, etc.) and samplers (DDPM, DDIM, etc.) without relying on any specific model structure.

## Limitations & Future Work

1. Multi-step simulation substantially increases calibration time (memory is O(1), but computational cost is $k$-fold).
2. The optimal window size $k$ may need to be tuned for specific model architectures and sampling step counts.
3. Only PTQ is explored—whether combining with quantization-aware training (QAT) yields further improvements remains an open question.
4. Validation on the latest DiT architectures (e.g., SD3, FLUX) and Flow Matching models has not been conducted.
5. The impact of calibration data selection on performance is insufficiently studied—is a representative set of prompts necessary?

## Related Work & Insights

- **Q-Diffusion**: The first PTQ work for diffusion models and a representative of step-wise independent optimization; AccuQuant significantly improves upon this through multi-step simulation.
- **PTQD**: Accounts for the distributional characteristics of quantization noise and applies correction, but still operates step-by-step independently.
- **EfficientDM**: Explores mixed-precision quantization; orthogonal to AccuQuant and potentially complementary.
- **Inspiration**: The error accumulation analysis can generalize to other iterative inference processes—such as quantization error at each decoding step in autoregressive text generation, error propagation in iterative refinement models, and policy quantization in reinforcement learning.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: A problem-driven study grounded in the profound insight of "error accumulation," with a natural and elegant method design. The O(1) memory optimization is a significant engineering contribution. Experiments provide thorough validation across multiple models and tasks. The primary deductions are the lack of validation on the latest generative architectures, and the fact that some FID values in the experimental tables are not reported as absolute numbers, limiting direct comparability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[NeurIPS 2025\] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans](multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](../../ICCV2025/image_generation/fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)

</div>

<!-- RELATED:END -->
