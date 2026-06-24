---
title: >-
  [Paper Note] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment
description: >-
  [CVPR 2025][Image Generation][Flow Matching] Proposes the Diff2Flow framework, which achieves efficient knowledge transfer from pre-trained diffusion models to Flow Matching models through timestep rescaling, interpolation alignment, and velocity field derivation. This achieves performance superior to or on par with the SOTA across multiple tasks such as text-to-image generation and depth estimation with minimal fine-tuning overhead.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Flow Matching"
  - "Diffusion Models"
  - "Knowledge Transfer"
  - "Parameter-Efficient Fine-Tuning"
  - "Trajectory Alignment"
date: 2026-05-08
content_hash: 05a317cd42b65c52
---

# Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment

**Conference**: CVPR 2025  
**arXiv**: [2506.02221](https://arxiv.org/abs/2506.02221)  
**Code**: [https://github.com/CompVis/diff2flow](https://github.com/CompVis/diff2flow)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Flow Matching, Diffusion Models, Knowledge Transfer, Parameter-Efficient Fine-Tuning, Trajectory Alignment

## TL;DR

Proposes the Diff2Flow framework, which achieves efficient knowledge transfer from pre-trained diffusion models to Flow Matching models through timestep rescaling, interpolation alignment, and velocity field derivation. This achieves performance superior to or on par with the SOTA across multiple tasks such as text-to-image generation and depth estimation with minimal fine-tuning overhead.

## Background & Motivation

**Background**: Diffusion models (such as Stable Diffusion) have achieved great success in the field of image generation, while Flow Matching (FM) has attracted attention as an alternative paradigm due to its straighter sampling trajectories and faster inference speeds. Current SOTA FM-based foundation models (e.g., Flux, SDv3) exceed 8B parameters, making fine-tuning extremely costly.

**Limitations of Prior Work**: Existing FM-based large models are impractical to fine-tune due to their massive size, especially under resource-constrained environments. While the Stable Diffusion architecture is efficient and has a well-established ecosystem, its diffusion training paradigm is less efficient in inference compared to FM. Direct application of the FM objective to diffusion models leads to slow convergence and degraded performance due to mismatches in interpolation definitions, timestep scaling, and training objectives.

**Key Challenge**: Although diffusion models and Flow Matching can be unified under the same framework, they are misaligned in three key aspects in practical implementations: (1) Different interpolation formulations—diffusion uses $x_t = \alpha_t x_0 + \sigma_t \epsilon$, while FM uses linear interpolation $x_t = t x_1 + (1-t) x_0$; (2) Different timestep spaces—diffusion uses discrete $[0, T]$, while FM uses continuous $[0, 1]$; (3) Different training objectives—diffusion predicts noise or v, while FM predicts the velocity field.

**Goal**: Efficiently transfer knowledge from pre-trained diffusion models to FM models, endowing them with both the prior knowledge of diffusion models and the inference advantages of FM.

**Key Insight**: The authors observe that diffusion models can still generate reasonable results when inferring on non-integer timesteps, indicating that their internal timestep embeddings constitute a continuous time-space. Based on this observation, an invertible mapping between diffusion and FM can be constructed.

**Core Idea**: By explicitly constructing a timestep mapping $f_t$ and a data point mapping $f_x$, the diffusion trajectory is "warped" into the linear trajectory of FM, and the velocity estimation required by FM is derived from the diffusion model's predictions, thereby achieving seamless knowledge transfer.

## Method

The core idea of Diff2Flow is: instead of forcing the model to "forget" the parameterization of diffusion and relearn velocity prediction, it mathematically and precisely converts the predictions of the diffusion model into the velocity field of FM. The entire method is divided into trajectory traversal and objective unification.

### Overall Architecture

The inputs are a pre-trained diffusion model (e.g., SD 2.1) and training data. During the training phase: for each sample, a linear interpolation $x_{t_{FM}}^{FM}$ is first constructed according to the FM paradigm, then converted back to the data points and timesteps in the diffusion space using the inverse mappings $f_t^{-1}$ and $f_x^{-1}$. These are then fed into the diffusion model to obtain predictions, which are converted to velocity estimates via the objective conversion formula. Finally, the model is trained with the standard FM loss. During the inference phase, Euler step integration is applied after mapping.

### Key Designs

1. **Trajectory Traversal**:

    - **Function**: Establishes a bidirectional, invertible mapping between the diffusion trajectory and the linear trajectory of FM.
    - **Mechanism**: Defines the data point mapping $f_x(x_{t_{DM}}^{DM}) = \frac{1}{\alpha_t + \sigma_t} x_{t_{DM}}^{DM}$ to scale the diffusion interpolation into a linear form, and the timestep mapping $f_t(t_{DM}) = \frac{\alpha_t}{\alpha_t + \sigma_t}$ to map the discrete diffusion timesteps to the continuous $[0,1]$ space of FM. For non-integer timesteps, piecewise linear interpolation is used to extend to the continuous domain. The inverse mapping solves for the diffusion timesteps from the FM timesteps via linear interpolation.
    - **Design Motivation**: In the variance-preserving schedule of diffusion, $\alpha_t^2 + \sigma_t^2 = 1$, which naturally allows the diffusion interpolation to be converted into a linear combination by dividing by $(\alpha_t + \sigma_t)$, aligning it with the FM interpolation form. This mapping ensures consistent boundary conditions—fully aligning the noise and data ends.

2. **Objective Unification**:

    - **Function**: Directly derives the velocity field required by FM from the original predictions of the diffusion model (e.g., v-prediction).
    - **Mechanism**: Taking v-parameterization as an example, the diffusion model predicts $v_\theta = \alpha_t \epsilon - \sigma_t x_0$. Through algebraic derivation, estimates for $\hat{x}_0^{DM}$ and $\hat{x}_T^{DM}$ can be recovered from $v_\theta$, thereby yielding the FM velocity $\mathbf{v}_\theta(x^{FM}, t_{FM}) = (\alpha_t - \sigma_t)(x_{t_{DM}}^{DM} - v_\theta)$. This means the model does not need to "relearn" a new parameterization, but rather performs a deterministic transformation on top of the original predictions.
    - **Design Motivation**: Prior work (such as DepthFM) directly forced the v-parameterized model to predict the velocity field, forcing the model to switch its parameterization, which required longer convergence times and degraded performance. Through objective unification, the model's original knowledge is fully preserved.

3. **Parameter-Efficient Fine-Tuning (LoRA Adaptation)**:

    - **Function**: Completes the diffusion-to-FM translation with minimal parameter updates.
    - **Mechanism**: Uses the LoRA method to update only the low-rank decomposition of the weight matrices $\Delta W = BA$, freezing the main model weights. A key finding is that directly applying the FM objective to the diffusion model renders LoRA ineffective (as parameterization translation requires major scale adjustments), but LoRA works highly efficiently when combined with Diff2Flow's alignment strategy.
    - **Design Motivation**: Full fine-tuning is impractical in computationally constrained scenarios. The alignment of Diff2Flow ensures the model only needs minor parameter adjustments rather than completely relearning, making the low-rank constraint of LoRA no longer a bottleneck.

### Loss & Training

Uses the standard FM loss $\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \|(x_1 - x_0) - \mathbf{v}_\theta(x_t, t)\|^2$. The training pipeline is: sample the FM timestep $t_{FM} \in [0,1]$, construct the FM interpolation, perform inverse mapping to the diffusion space, obtain the model prediction, convert it to a velocity estimate, and optimize via the FM loss using gradient descent. Standard Euler sampling is used during inference.

## Key Experimental Results

### Main Results

| Task | Method | FID ↓ | CLIP ↑ | Aesthetic Score ↑ |
|------|------|-------|--------|----------|
| T2I (SD1.5 continued training) | SD1.5 | 56.77 | 26.34 | 5.32 |
| T2I (SD1.5 continued training) | SD1.5 cont. | 56.36 | 26.33 | 5.90 |
| T2I (SD1.5 continued training) | **Diff2Flow** | **52.80** | **26.54** | **5.99** |

### Ablation Study

| Configuration | Description |
|------|------|
| FM w/o Alignment + Full FT | Slow convergence, can eventually catch up in performance |
| FM w/o Alignment + LoRA | Severe performance degradation, fails to converge to a reasonable level |
| Diff2Flow + Full FT | Converges in ~2.5k steps |
| Diff2Flow + LoRA | Close to Full FT performance, far outperforming naive FM |

### Key Findings

- Under full fine-tuning, naive FM and Diff2Flow eventually converge to similar performance, but Diff2Flow converges significantly faster (~2.5k steps).
- Under LoRA, the gap is even more pronounced: naive FM cannot close the gap with Diff2Flow, showing that alignment is a prerequisite for successful PEFT.
- Diff2Flow naturally resolves the zero-terminal SNR issue of diffusion models—it can correctly generate pure black/white images after converting to FM.
- Converting SD1.5 to FM improves performance on the same generation task, validating the advantage of FM's straighter trajectories.
- Applying Reflow allows SD1.5 to generate images in just 2 steps.

## Highlights & Insights

- The elegance of the method lies in its rigorous mathematical derivation and clean implementation—consisting solely of two invertible mappings and an algebraic transformation, with no extra network components required.
- The finding that diffusion models function normally at non-integer timesteps provides empirical evidence that sinusoidal positional encodings build a continuous time-space.
- Addresses a practical pain point: large FM base models are too expensive to fine-tune, whereas small but strong diffusion models can inherit the benefits of FM via Diff2Flow.
- The method is highly versatile, being compatible with different parameterizations (epsilon/v).

## Limitations & Future Work

- Currently only verified on SD1.5/SD2.1; applicability to larger models (e.g., SDXL) needs further exploration.
- The depth estimation experiment used fewer training samples (74K); behavior on larger-scale data needs research.
- Whether it can be reversed—transferring from FM models to diffusion models—is not discussed in the paper.
- For non-variance-preserving schedulers (e.g., VE schedule), though theoretically applicable, experimental validation is limited.

## Related Work & Insights

- **DepthFM** uses diffusion priors for FM depth estimation but suffers from alignment flaws; this paper provides a systematic solution targeted at this very issue.
- **Reflow** straightens sampling trajectories via paired data; Diff2Flow can directly apply Reflow to diffusion models.
- Complementary to directions like InstaFlow and Consistency Distillation—Diff2Flow focuses on paradigm translation rather than step distillation.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to systematically resolve precise alignment from diffusion to FM |
| Experimental Thoroughness | 4 | Multi-task validation (T2I/depth/reflow), including LoRA ablation |
| Writing Quality | 4 | Clear mathematical derivations, intuitive illustrations |
| Value | 5 | Highly practical, allowing existing diffusion models to directly benefit from FM advantages |
- **Value**: ⭐⭐⭐⭐⭐ Enables seamless transfer of all pre-trained diffusion weights to FM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](../../NeurIPS2025/image_generation/value_gradient_guidance_for_flow_matching_alignment.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](../../ICLR2026/image_generation/densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)

</div>

<!-- RELATED:END -->
