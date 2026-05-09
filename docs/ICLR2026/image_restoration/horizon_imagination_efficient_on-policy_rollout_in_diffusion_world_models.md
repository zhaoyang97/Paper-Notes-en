---
title: >-
  [Paper Note] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models
description: >-
  [ICLR 2026][Image Restoration][diffusion world models] This paper proposes Horizon Imagination (HI), which samples actions at an intermediate denoising step and processes multiple future frames in parallel, reducing the per-frame computation of on-policy imagination in diffusion world models to less than one full denoising pass while maintaining control performance.
tags:
  - ICLR 2026
  - Image Restoration
  - diffusion world models
  - on-policy rollout
  - reinforcement learning
  - sample efficiency
  - Atari
date: 2026-05-08
content_hash: 54662edd0180645b
---

# Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models

**Conference**: ICLR 2026
**arXiv**: [2602.08032](https://arxiv.org/abs/2602.08032)
**Code**: [https://github.com/leor-c/horizon-imagination](https://github.com/leor-c/horizon-imagination)
**Area**: Image Restoration
**Keywords**: diffusion world models, on-policy rollout, reinforcement learning, sample efficiency, Atari

## TL;DR
This paper proposes Horizon Imagination (HI), which samples actions at an intermediate denoising step and processes multiple future frames in parallel, reducing the per-frame computation of on-policy imagination in diffusion world models to less than one full denoising pass while maintaining control performance.

## Background & Motivation

**Background**: World models learn environment dynamics to generate simulated data. Diffusion world models (e.g., DIAMOND) have attracted attention for their superior generation fidelity, but their multi-step denoising overhead is substantial.

**Limitations of Prior Work**: On-policy imagination requires sampling actions from the current policy after each frame is generated, imposing a strictly sequential dependency that precludes exploitation of diffusion models' parallel denoising capabilities.

**Key Challenge**: Diffusion models offer high generation quality at significant computational cost, and the sequential nature of on-policy RL further amplifies this problem.

**Goal**: Substantially reduce the computational overhead of diffusion world models while preserving the quality of on-policy imagination.

**Key Insight**: The observation that actions can be sampled at intermediate denoising steps, embedding policy selection within the diffusion process.

**Core Idea**: Perform action sampling mid-denoising (rather than after completion), enabling the first half of multi-frame imagination's denoising to proceed fully in parallel.

## Method

### Overall Architecture
The conventional approach follows a serial "full denoising → action sampling → full denoising" pipeline. HI instead initializes noisy latents for $h$ future frames simultaneously, samples actions from the partially denoised intermediate results after $k$ steps ($k < K$ total steps), and then continues denoising the remaining steps conditioned on these actions.

### Key Designs

1. **Intermediate Action Sampling**:

    - **Function**: Determines on-policy actions at an intermediate denoising step.
    - **Mechanism**: Given total denoising steps $K$, actions are sampled at step $k$ from the partially denoised latent $\mathbf{z}^k_t$ via the policy network $\pi(\mathbf{z}^k_t)$. The first $k$ steps require no action conditioning and can be processed in parallel across all frames.
    - **Design Motivation**: Partially denoised frames, though still noisy, already contain sufficient semantic information to support policy decisions.

2. **Parallel Horizon Denoising**:

    - **Function**: Simultaneously denoises latents across multiple future timesteps.
    - **Mechanism**: The $h$ future frames $\{\mathbf{z}^0_{t+1}, \ldots, \mathbf{z}^0_{t+h}\}$ are batched together and fed into the denoising network, amortizing the cost of each forward pass across frames, reducing the total denoising budget to less than one full denoising pass per frame.
    - **Design Motivation**: Denoising networks in diffusion models naturally support batched inference, making parallelization a direct source of computational savings.

3. **Generation Stability Strategy**:

    - **Function**: Prevents generation degradation during long-horizon imagination.
    - **Mechanism**: An action consistency regularization term is introduced to encourage $\pi(\mathbf{z}^k_t) \approx \pi(\mathbf{z}^K_t)$, reducing the distributional shift between intermediate and fully denoised action samples.
    - **Design Motivation**: Sampling directly from noisy states may introduce bias, which is particularly critical in complex 3D environments such as Craftium.

### Loss & Training
The world model is trained with a standard diffusion denoising loss. The policy and value function are optimized via actor-critic on imagined trajectories. Action consistency regularization is applied as an auxiliary loss.

## Key Experimental Results

### Main Results

| Environment | Method | Human-Normalized Score | Denoising Steps per Frame |
|---|---|---|---|
| Atari 100K | DIAMOND | ~1.0x | 10 |
| Atari 100K | HI (k=3) | ~1.0x (matched) | <1 |
| Craftium | DIAMOND | Baseline | 10 |
| Craftium | HI | Matched | <1 |

### Ablation Study

| Configuration | Relative Performance | Notes |
|---|---|---|
| HI full (k=3) | 1.00 | Optimal intermediate sampling point |
| w/o action consistency reg. | 0.93 | Action bias accumulates |
| k=1 (very early sampling) | 0.87 | Insufficient information |
| k=K (sequential) | 1.00 | No efficiency gain |

### Key Findings
- HI reduces per-frame computation to less than one denoising pass while matching control performance.
- There exists a sweet spot for the intermediate sampling point $k$: too early yields insufficient information; too late sacrifices efficiency gains.
- Action consistency regularization is critical for long-horizon rollout stability.

## Highlights & Insights
- **Denoising–Decision Co-design**: Embedding policy selection within the diffusion process enables parallelism between generation and decision-making, representing an important efficiency advance for diffusion world models.
- **Partial Denoising Suffices**: The work demonstrates that policy decisions do not require fully denoised observations.

## Limitations & Future Work
- Validation is limited to discrete action spaces; applicability to continuous control remains unexplored.
- The intermediate sampling point $k$ requires tuning.
- More complex environments may necessitate additional stabilization strategies.

## Related Work & Insights
- **vs. DIAMOND**: HI builds on DIAMOND to improve efficiency by modifying only the imagination procedure, without altering the architecture.
- **vs. IRIS/TWM**: Transformer-based world models do not require multi-step denoising but generally yield lower generation quality than diffusion-based counterparts.

## Rating
- Novelty: ⭐⭐⭐⭐ The intermediate sampling idea is both novel and natural.
- Experimental Thoroughness: ⭐⭐⭐ Coverage is adequate but lacks continuous control evaluation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ Significant practical implications for deploying diffusion world models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [\[AAAI 2026\] RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation](../../AAAI2026/image_restoration/refidiff_progressive_refinement_diffusion_for_efficient_missing_data_imputation.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/image_restoration/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)

<!-- RELATED:END -->
