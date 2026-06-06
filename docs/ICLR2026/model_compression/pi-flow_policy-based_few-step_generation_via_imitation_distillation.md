---
title: >-
  [Paper Note] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation
description: >-
  [ICLR 2026][Model Compression][Flow matching distillation] This paper proposes π-Flow, which modifies the output layer of a student flow model to predict a *policy* that generates dynamic flow velocities through multiple…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Flow matching distillation"
  - "few-step generation"
  - "policy learning"
  - "imitation distillation"
  - "DiT"
date: 2026-05-08
content_hash: 2a8004b2643c3dec
---

# π-Flow: Policy-Based Few-Step Generation via Imitation Distillation

**Conference**: ICLR 2026
**arXiv**: [2510.14974](https://arxiv.org/abs/2510.14974)  
**Code**: Available (see paper Comments)  
**Area**: Image Generation / Diffusion Model Distillation
**Keywords**: Flow matching distillation, few-step generation, policy learning, imitation distillation, DiT

## TL;DR

This paper proposes π-Flow, which modifies the output layer of a student flow model to predict a *policy* that generates dynamic flow velocities through multiple sub-steps within a single network evaluation, enabling precise ODE integration. Combined with imitation distillation—matching teacher velocities along the student's own trajectories—the method achieves stable and scalable few-step generation without the quality–diversity trade-off.

## Background & Motivation

Despite their excellent generation quality, diffusion and flow matching models require a large number of inference steps (e.g., 50–1000 ODE steps), severely limiting their practical efficiency. Distillation is the primary approach to accelerating these models.

### Limitations of Prior Work

**Format Mismatch**:
   - Teacher models output **velocity fields**—instantaneous changes along the flow direction.
   - Student models are typically trained to output **denoised data (shortcut predictions)**—directly predicting the final result.
   - This format mismatch complicates and destabilizes the distillation process.

**Quality–Diversity Trade-off**:
   - Methods such as Distribution Matching Distillation (DMD) use KL divergence to match distributions.
   - However, KL divergence tends toward either mode covering or mode seeking, making it difficult to achieve both simultaneously.
   - On large-scale models (FLUX, Qwen-Image), this manifests as a significant drop in diversity.

**Training Instability**:
   - Many methods require online student sample generation, discriminator training, or adversarial training.
   - These strategies increase training complexity and instability.

### Root Cause

Can a distillation method be designed such that the student model retains the same velocity-prediction format as the teacher, thereby:
- Eliminating format mismatch,
- Enabling a simple $\ell_2$ flow matching loss, and
- Compressing the number of steps while preserving both quality and diversity?

## Method

### Overall Architecture

The core innovation of π-Flow lies in redefining the student model's output:

1. **Teacher**: A standard velocity-predicting flow model $v_\theta(x_t, t)$ requiring many ODE steps (e.g., 50).
2. **Student**: The output layer is modified to predict a **policy** rather than a velocity or denoised data directly.
3. **Policy**: A parameterized function—requiring no additional network—that describes how to advance the ODE through multiple sub-steps within a single coarse step.

### Key Designs

1. **Policy Output Layer** → Describes the sub-step velocity field with a small number of parameters → Design motivation: achieve multi-step ODE integration within a single forward pass.

    - Conventional student models perform a single Euler step over the coarse interval $[t_i, t_{i+1}]$.
    - The π-Flow student outputs a policy $\pi$ over the same interval, describing the continuous velocity field within it.
    - The policy can take a simple form such as a polynomial (e.g., linear interpolation $v(s) = a + b \cdot s$).
    - Given the policy parameters, precise ODE integration over multiple sub-steps is performed **without additional network evaluations**.
    - The network only needs to predict policy parameters (e.g., the intercept and slope of a linear segment) rather than per-step velocities.

   Key advantage: one network forward pass → policy parameters → high-precision ODE integration over multiple sub-steps → better trajectory approximation.

2. **Imitation Distillation** → Matches teacher velocities along the student's own trajectories → Design motivation: stabilize training and avoid the quality–diversity trade-off.

    - Conventional distillation trains the student on teacher trajectories.
    - Imitation distillation trains on the **student's own trajectories**:
      1. Use the student policy to generate sub-step points $x_s$.
      2. Query the teacher velocity $v_{\text{teacher}}(x_s, s)$ at these points.
      3. Minimize the $\ell_2$ difference between the student policy's velocity and the teacher velocity at these points.

   This is conceptually equivalent to imitation learning in reinforcement learning:
    - Teacher = expert
    - Student policy = imitator
    - Matching expert behavior under the imitator's own state distribution avoids distribution shift.

3. **Standard Flow Matching Loss** → Maintains format consistency → Design motivation: simplify the training procedure.

   Since the student outputs a parameterized policy over the velocity field, the distillation loss is simply the standard $\ell_2$ flow matching loss:
    $$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \|v_{\text{student}}(x_t, t) - v_{\text{teacher}}(x_t, t)\|_2^2$$

   No adversarial training, distribution matching loss, or additional discriminator is required.

### Loss & Training

- **Loss**: Standard $\ell_2$ velocity matching loss (flow matching loss).
- **Training data**: Sampled from trajectory points generated by the student policy.
- **Not required**: Adversarial training, online generation, discriminators.
- **Scalability**: Fully compatible with standard flow matching training pipelines.

## Key Experimental Results

### Main Results: ImageNet 256×256

| Method | NFE | FID ↓ | Architecture |
|--------|-----|-------|--------------|
| π-Flow | **1** | **2.85** | DiT |
| Best prior 1-NFE (same arch.) | 1 | >2.85 | DiT |
| Teacher (multi-step) | 50+ | ~2.0 | DiT |

π-Flow achieves an FID of 2.85 at 1 NFE, outperforming all known 1-NFE models under the same architecture.

### Large-Scale Model Experiments

| Model | Method | NFE | Quality | Diversity |
|-------|--------|-----|---------|-----------|
| FLUX.1-12B | DMD | 4 | Good | **Significantly degraded** |
| FLUX.1-12B | **π-Flow** | 4 | **Good** | **Preserved at teacher level** |
| Qwen-Image-20B | DMD | 4 | Good | **Significantly degraded** |
| Qwen-Image-20B | **π-Flow** | 4 | **Good** | **Preserved at teacher level** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Direct velocity output (no policy) | Higher FID | Lacks sub-step integration precision |
| Training on teacher trajectories | Higher FID | Suffers from distribution shift |
| Training on student trajectories (imitation distillation) | **Lowest FID** | Avoids distribution shift |
| Linear policy vs. constant policy | Linear is better | Finer sub-step velocity description |
| Increasing number of sub-steps | FID improves | Diminishing returns |

### Key Findings

1. **1-NFE SOTA**: π-Flow achieves an FID of 2.85 on ImageNet 256², setting the best 1-NFE record under the same architecture.
2. **Resolves the quality–diversity trade-off**: On FLUX.1-12B and Qwen-Image-20B (4 NFE), π-Flow preserves teacher-level diversity, whereas DMD suffers significant diversity degradation.
3. **Training stability**: Standard $\ell_2$ loss suffices for stable training without complex training strategies.
4. **Effectiveness of the policy**: Even a simple linear policy substantially improves sub-step ODE integration accuracy.
5. **Imitation distillation outperforms teacher-trajectory distillation**: Training on the student's own trajectories yields better results than training on teacher trajectories.
6. **Scalable to very large models**: Successfully applied to generative models with 12B and 20B parameters.

## Highlights & Insights

1. **An elegant core insight**: The distillation problem is reframed as imitation learning—the student mimics the teacher's *behavior* (velocity) along its own trajectories, rather than trying to match the teacher's *outcomes* (generated data). This perspective shift is the paper's most significant contribution.
2. **Clever policy design**: By predicting policy parameters rather than per-step velocities, the student model obtains a precise description of an entire ODE segment in a single forward pass—incurring almost no additional computational overhead while substantially improving integration accuracy.
3. **Simplicity**: The entire method requires only a standard $\ell_2$ flow matching loss, with no adversarial training, distribution matching, or auxiliary networks.
4. **Addresses DMD's core failure mode**: Preserving diversity on large models (FLUX, Qwen-Image) is the primary failure mode of competing methods such as DMD; π-Flow naturally resolves this by avoiding KL divergence optimization.
5. **Cross-domain connection**: Importing the imitation learning framework from reinforcement learning into generative model distillation represents an insightful cross-domain transfer.

## Limitations & Future Work

1. **Policy expressiveness**: The linear policy used in this work is relatively simple; more expressive policies (e.g., piecewise polynomials or learnable basis functions) may further improve performance.
2. **Computational overhead**: Although sub-step integration requires no additional network evaluations, querying teacher velocities at sub-step points still incurs training-time cost.
3. **Theoretical analysis**: A formal theoretical framework for the convergence and approximation error of imitation distillation is absent.
4. **Non-image modalities**: Validation is limited to image generation; applicability to video, 3D, audio, and other modalities remains unexplored.
5. **Comparison with more baselines**: Large-model experiments primarily compare against DMD; comparisons with other distillation methods (e.g., consistency models) are insufficient.
6. **Adaptive sub-step selection**: The number of sub-steps currently requires manual specification; adaptive selection may yield further improvements.

## Related Work & Insights

### Distillation Methods

- **DMD/DMD2 (Yin et al.)**: Distribution Matching Distillation uses KL divergence for distribution matching but suffers from diversity loss.
- **Consistency Models (Song et al.)**: Achieve few-step generation through consistency training.
- **Progressive Distillation (Salimans & Ho)**: Progressively halve the number of sampling steps.
- **InstaFlow, BOOT**: Other flow model distillation methods.

### Theoretical Connections

- **Imitation learning (DAgger, GAIL)**: The imitation distillation in π-Flow shares a conceptual connection with DAgger's "online aggregation"—both train under the learner's own state distribution.
- **Flow matching (Lipman et al., Liu et al.)**: π-Flow preserves the velocity-prediction format of flow matching.

### Insights for Future Research

1. Format consistency is crucial in distillation—aligning the output format of teacher and student greatly simplifies training.
2. Framing few-step generation as "how to describe an ODE trajectory with a small number of parameters" is more natural than "how to jump directly to the endpoint."
3. Cross-domain idea transfer (RL → generative models) can yield novel design principles.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of a policy output layer and imitation distillation constitutes an entirely new distillation paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Coverage of ImageNet + FLUX + Qwen-Image spans diverse settings; ablation studies are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Concepts are clearly articulated with intuitive naming (π-Flow, policy, imitation distillation).
- Value: ⭐⭐⭐⭐⭐ — Addresses the core quality–diversity trade-off in large-model distillation, with significant implications for accelerating generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](lightmem_lightweight_and_efficient_memory-augmented_generation.md)

</div>

<!-- RELATED:END -->
