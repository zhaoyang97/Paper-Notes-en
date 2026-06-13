---
title: >-
  [Paper Note] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories
description: >-
  [CVPR 2026][Image Generation][flow matching] This paper proposes LeapAlign, which constructs two-step leap trajectories to compress long generation paths into two steps…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "flow matching"
  - "post-training"
  - "reward alignment"
  - "human preference"
  - "diffusion model"
date: 2026-05-08
content_hash: 28d3f9390485913c
---

# LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories

**Conference**: CVPR 2026
**arXiv**: [2604.15311](https://arxiv.org/abs/2604.15311)  
**Code**: [rockeycoss.github.io/leapalign/](https://rockeycoss.github.io/leapalign/)  
**Area**: Image Generation
**Keywords**: flow matching, post-training, reward alignment, human preference, diffusion model

## TL;DR

This paper proposes LeapAlign, which constructs two-step leap trajectories to compress long generation paths into two steps, enabling reward gradients to be directly backpropagated to early generation steps. Combined with trajectory similarity weighting and gradient discounting strategies, LeapAlign achieves efficient post-training alignment of flow matching models.

## Background & Motivation

Aligning flow matching models with human preferences is an important research direction. GRPO-based methods adapted from LLMs introduce substantial stochasticity and variance. Direct gradient methods exploit the differentiability of the flow matching sampling process to backpropagate reward gradients, achieving faster and more stable convergence. However, backpropagation through long trajectories faces two major challenges: (1) prohibitive memory consumption due to long activation chains; and (2) gradient explosion. Consequently, existing methods update only a single step near the final image, failing to update early steps that determine the global structure of the generated image.

## Method

### Overall Architecture

At each iteration, a complete trajectory is sampled from noise to image. Two timesteps $k > j$ are randomly selected to construct a two-step leap trajectory: the first step leaps from $x_k$ to $x_j$, and the second step leaps from $x_j$ to the final $x_0$. The reward is computed on the actual final image, but gradients are backpropagated only through the leap trajectory.

### Key Designs

1. **Leap Trajectory Construction**: Leveraging the single-step leap prediction property of rectified flow matching, $\hat{x}_{j|k} = x_k - (k-j) v_\theta(x_k, k)$, the full multi-step trajectory is compressed into two steps. By randomizing the start and end timesteps $(k, j)$, the method can update arbitrary generation steps, including early steps critical to global structure.

2. **Trajectory Similarity Weighting**: Approximation errors exist between the leap trajectory and the actual multi-step path. Leap trajectories that are more consistent with the true path are assigned higher training weights to improve training efficiency. Similarity is measured by the distance between the leap prediction and the actual intermediate latent code.

3. **Gradient Discounting (Rather Than Truncation)**: DRTune removes nested gradient terms entirely to avoid gradient explosion, thereby discarding cross-timestep dependency information. LeapAlign instead reduces the weight of large gradient terms rather than eliminating them, preserving the learning signal while ensuring stability.

### Loss & Training

The objective is reward maximization, with gradients backpropagated through the two-step leap trajectory. Multiple steps per trajectory can be updated, and memory overhead is constant (only two steps of backpropagation are required).

## Key Experimental Results

### Main Results

Fine-tuning the Flux model and comparing against state-of-the-art methods:

| Metric | DRTune | DanceGRPO | MixGRPO | LeapAlign |
|--------|--------|-----------|---------|-----------|
| HPSv2.1 | Baseline | Moderate | Moderate | **Best** |
| HPSv3 | Baseline | Moderate | Moderate | **Best** |
| PickScore | Baseline | Moderate | Moderate | **Best** |
| GenEval | Baseline | Moderate | Moderate | **Best** |

LeapAlign consistently outperforms both GRPO-based and direct gradient methods across all evaluation metrics.

### Ablation Study

- Updating early steps contributes substantially to improvements in global structure.
- Gradient discounting vs. gradient truncation: the former retains more information and is more stable.
- Trajectory similarity weighting improves convergence speed and final performance.

### Key Findings

- Fine-tuning early steps is critical for improving image layout and composition.
- Two-step trajectories are sufficient to capture effective cross-step gradient information.
- Reward improvement converges noticeably faster than DRTune.

## Highlights & Insights

- The leap trajectory construction reduces memory overhead from $O(T)$ to a constant.
- The "discount rather than truncate" strategy for preserving gradient signals is simple yet effective.
- This work represents the first practical direct gradient update for early steps in flow matching models.

## Limitations & Future Work

- The approximation quality of leap predictions relative to the true path depends on the linearity of the underlying flow matching model.
- The quality of the reward model directly determines alignment effectiveness.
- Generalizability to flow matching applications beyond image generation has not been verified.

## Related Work & Insights

- The leap trajectory technique is applicable to post-training of other long-sequence differentiable sampling processes.
- The gradient discounting strategy provides a reference for other training scenarios prone to gradient explosion.
- The performance gap over GRPO-based methods confirms the advantage of direct gradient approaches in flow matching.

## Rating

8/10 — The method design is concise and effective, addressing the core bottleneck of direct gradient methods, with thorough experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

</div>

<!-- RELATED:END -->
