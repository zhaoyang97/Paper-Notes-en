---
title: >-
  [Paper Note] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models
description: >-
  [CVPR 2026][Image Generation][Unified Multimodal Models] This paper proposes GvU, a self-supervised RL framework (based on GRPO) that leverages the visual understanding branch of a unified multimodal model (UMM) as an intrinsic reward signal. Token-level text-image alignment probabilities are used to iteratively improve T2I generation quality without any external supervision, achieving a 43.3% improvement on GenEval++. Notably, the enhanced generation in turn promotes fine-grained visual understanding.
tags:
  - CVPR 2026
  - Image Generation
  - Unified Multimodal Models
  - Self-Supervised Reinforcement Learning
  - Intrinsic Reward
  - Text-Image Alignment
  - GRPO
  - Understanding-Enhanced Generation
date: 2026-05-08
content_hash: 9349f16f658988fc
---

# Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models

**Conference**: CVPR 2026
**arXiv**: [2603.06043](https://arxiv.org/abs/2603.06043)
**Area**: Image Generation / Unified Multimodal Models
**Keywords**: Unified Multimodal Models, Self-Supervised Reinforcement Learning, Intrinsic Reward, Text-Image Alignment, GRPO, Understanding-Enhanced Generation

## TL;DR
This paper proposes GvU, a self-supervised RL framework (based on GRPO) that leverages the visual understanding branch of a unified multimodal model (UMM) as an intrinsic reward signal. Token-level text-image alignment probabilities are used to iteratively improve T2I generation quality without any external supervision, achieving a 43.3% improvement on GenEval++. Notably, the enhanced generation in turn promotes fine-grained visual understanding.

## Background & Motivation

**State of the Field**: UMMs integrate visual understanding and generation through a shared backbone, theoretically enabling T2I generation with complex instruction following. Representative models include Chameleon, Emu3, Janus, BAGEL, Show-o, and BLIP3-o.

**Core Problem**: UMMs suffer from a severe **understanding-generation capability asymmetry**—the understanding branch typically far outperforms the generation branch. Joint training of both tasks further leads to negative transfer, where optimizing one task degrades the other.

**Limitations of Prior Work**: Conventional RL approaches use image-level external rewards (e.g., ImageReward, PickScore), which are too coarse to capture subtle semantic differences, prone to reward hacking, and dependent on external models.

**Core Insight**: Understanding (image→text) and generation (text→image) are dual tasks. The strong understanding capability already present in UMMs can naturally serve as a "teacher" to evaluate the alignment between self-generated images and the original text, without requiring external supervision.

**Core Idea**: The UMM understanding branch computes token-level conditional probabilities of the original prompt given a generated image, providing fine-grained intrinsic rewards to drive self-supervised RL via GRPO.

## Method

### Overall Architecture
Built upon a UMM with an AR+diffusion head hybrid architecture (X-Omni), GvU comprises three core components: (1) a self-generation data pipeline that forms a closed loop using only text prompts; (2) token-level intrinsic rewards computed by the understanding branch; and (3) self-supervised GRPO RL for iterative optimization of the generation policy.

### Key Designs

1. **Self-Generation Pipeline**:

    - Given a text prompt $T = T_{1:L}$, the generation branch autoregressively produces image tokens $I_{1:L_I}$, which are decoded into pixel images via the diffusion head.
    - The understanding branch receives the generated image and a system instruction, then computes autoregressive conditional probabilities over the original prompt tokens.
    - The entire process requires no external image data or models, forming a fully closed loop.

2. **Token-Level Intrinsic Reward (Core of GvU)**:

    - Given a generated image $I$ and the original prompt $T_{1:L}$, the conditional probability of each token is computed as: $p_\theta(T_j|\mathbf{X}_{j-1}) = \text{Softmax}(\text{Logits}_\theta(\mathbf{X}_{j-1})[T_j])$
    - The overall alignment probability is the geometric mean (to eliminate length bias): $P(T_{1:L}|I) = (\prod_{j=1}^{L} p_\theta(T_j|\mathbf{X}_{j-1}))^{1/L}$
    - **Design Motivation**: Unlike image-level rewards, token-level probabilities provide dense, fine-grained signals capable of distinguishing subtle semantic differences such as color, count, and spatial position.

3. **Self-Supervised GRPO Optimization**:

    - For each prompt, $G$ trajectories are sampled, each receiving reward $R_i = P(T|I_i)$.
    - Group-relative advantage estimation: $A_i = \frac{R_i - \text{mean}(\{R_i\})}{\text{std}(\{R_i\})}$
    - The clipped GRPO objective with KL divergence constraint is maximized, requiring neither a value function nor an external reward model.
    - Training uses LoRA fine-tuning on a set of 50k text prompts.

### Loss & Training
$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(r_i(\theta)A_i, \text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

## Key Experimental Results

### Main Results: GenEval Benchmark

| Model | Single Obj.↑ | Two Obj.↑ | Count↑ | Color↑ | Position↑ | Color Attrib.↑ | Overall↑ |
|------|---------|---------|-------|-------|-------|-----------|----------|
| FLUX.1-dev | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| Janus-Pro | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| BAGEL | 0.99 | 0.94 | 0.80 | 0.87 | 0.64 | 0.63 | 0.81 |
| X-Omni (base) | 1.00 | 0.94 | 0.60 | 0.85 | 0.40 | 0.26 | 0.68 |
| **GvU** | 1.00 | 0.96 | 0.74 | **0.92** | 0.61 | 0.58 | **0.81** |
| **GvU†** | **1.00** | **0.97** | 0.80 | **0.93** | 0.68 | 0.65 | **0.84** |

### Main Results: GenEval++ Benchmark

| Model | Color↑ | Count↑ | Color/Pos↑ | Pos/Count↑ | Pos/Size↑ | Multi-Count↑ | Overall↑ |
|------|--------|--------|-----------|-----------|-----------|-------------|----------|
| FLUX.1-dev | 0.350 | 0.625 | 0.275 | 0.200 | 0.375 | 0.225 | 0.314 |
| BAGEL | 0.325 | 0.600 | 0.325 | 0.250 | 0.475 | 0.375 | 0.371 |
| X-Omni (base) | 0.225 | 0.500 | 0.325 | 0.150 | 0.475 | 0.275 | 0.282 |
| **GvU** | 0.300 | 0.400 | **0.575** | **0.525** | **0.675** | 0.400 | **0.404** |

### Ablation Study: Concurrent Improvement in Understanding (MMT-Bench Fine-Grained Subtasks)

| Model | Overall | Visual Recog.↑ | Visual Halluc.↑ | Halluc. Det.↑ | Commonsense↑ | Domain Know.↑ |
|------|---------|----------|----------|----------|----------|----------|
| Base | 49.76 | 51.21 | 45.57 | 66.25 | 70.0 | 38.46 |
| **GvU** | **49.92** | **52.58** | **50.63** | **68.75** | **75.0** | **42.31** |

### Ablation Study: Weak Base vs. Normal Base

| Base Model | GenEval Gain | Gap Size |
|------|-------------|----------|
| Normal base | 0.68→0.81 (+19.1%) | Smaller |
| **Weak base** | **0.21→0.50 (+138.1%)** | **Larger** |

### Key Findings
- 43.3% improvement on GenEval++ (0.282→0.404), with the most significant gains in composite categories (pos/count, pos/size).
- Intrinsic rewards exhibit **continuous and stable growth** throughout RL training, reflecting a cumulative effect rather than abrupt jumps.
- Enhanced generation **in turn promotes fine-grained understanding**: visual hallucination detection +5.06, commonsense reasoning +5.0.
- Weaker base models with larger understanding-generation gaps benefit more (+138.1% vs. +19.1%), validating the "understanding guides generation" mechanism.
- Removing count/color/region words from prompts leads to a significant drop in intrinsic reward, confirming the reward's sensitivity to fine-grained semantics.

## Highlights & Insights
- **Self-Teaching Paradigm**: The UMM understanding branch serves as "teacher" while the generation branch serves as "student," eliminating the need for external reward models.
- **Token-Level Reward**: Provides substantially finer granularity than image-level rewards, enabling discrimination of subtle semantic differences in color, count, and spatial position.
- **Synergistic Understanding-Generation Enhancement**: The first empirical demonstration that enhanced generation within a UMM can reciprocally improve fine-grained understanding.
- **General Framework**: Applicable to any UMM with an AR+diffusion head hybrid architecture.

## Limitations & Future Work
- The improvement in understanding remains modest (MMT-Bench overall score: +0.16 only); synergistic enhancement warrants further investigation.
- Validation is limited to the X-Omni architecture; generalization experiments on additional UMM architectures are needed.
- Training requires generating multiple samples per prompt (group sampling of size $G$ in GRPO), resulting in non-trivial computational overhead.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose token-level intrinsic rewards + self-supervised RL to bridge the understanding-generation gap in UMMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ GenEval/GenEval++/DPG-Bench + understanding benchmarks + weak base ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and well-motivated design choices.
- Value: ⭐⭐⭐⭐ Open-source RL framework requiring no additional data annotation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)

<!-- RELATED:END -->
