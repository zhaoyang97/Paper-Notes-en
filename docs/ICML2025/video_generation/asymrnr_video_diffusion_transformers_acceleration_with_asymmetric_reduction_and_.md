---
title: >-
  [Paper Note] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration
description: >-
  [ICML 2025][Video Generation][Video Generation Acceleration] Proposes AsymRnR—a training-free video DiT acceleration method. Based on the observation that redundancy levels vary across different attention components (Q/K/V), layers, and denoising steps, it asymmetrically reduces tokens to achieve lossless acceleration.
tags:
  - "ICML 2025"
  - "Video Generation"
  - "Video Generation Acceleration"
  - "Diffusion Transformer"
  - "Token Reduction"
  - "Asymmetric Strategy"
  - "Training-free"
date: 2026-05-08
content_hash: 4fadee0cc34d842d
---

# AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration

**Conference**: ICML 2025  
**arXiv**: [2412.11706](https://arxiv.org/abs/2412.11706)  
**Code**: [https://github.com/wenhao728/AsymRnR](https://github.com/wenhao728/AsymRnR)  
**Area**: Image/Video Restoration  
**Keywords**: Video Generation Acceleration, Diffusion Transformer, Token Reduction, Asymmetric Strategy, Training-free  

## TL;DR
Proposes AsymRnR—a training-free video DiT acceleration method. Based on the observation that redundancy levels vary across different attention components (Q/K/V), layers, and denoising steps, it asymmetrically reduces tokens to achieve lossless acceleration.

## Background & Motivation

**Background**: Video Diffusion Transformers (DiTs) like HunyuanVideo and CogVideoX deliver excellent generation quality, but their computational cost is extremely high (generating a few seconds of video requires minutes or even hours).

**Limitations of Prior Work**: (a) Distillation methods require intensive training; (b) Feature caching methods are architecture-specific; (c) Token Merging (ToMe) uniformly reduces tokens across all components, leading to video distortion and pixelation.

**Key Challenge**: The sensitivity to token reduction varies dramatically across different components—yet existing methods treat them uniformly.

**Goal**: How to efficiently reduce tokens by considering the varying sensitivity of different components?

**Key Insight**: Empirical studies reveal three key observations: (a) Perturbations on Q have a larger impact than those on K/V; (b) Perturbations on shallow layers have a greater impact than those on deep layers; (c) Early denoising steps affect semantics, while late steps affect details.

**Core Idea**: Asymmetric token reduction—substantially reducing K/V while keeping Q intact, with different reduction ratios applied to different layers and steps.

## Method

### Overall Architecture
Before attention computation:
1. Retain all Query tokens.
2. Merge redundant Key/Value tokens based on similarity.
3. Use an asymmetric scheduler to adjust the reduction ratios dynamically based on layer depth and denoising steps.
4. Restore to the original token sequence length after attention computation.

### Key Designs

1. **Asymmetric Q-KV Reduction**:

    - **Function**: Reduces only K/V tokens while preserving all Q tokens.
    - **Mechanism**: Q tokens directly determine the representation of each position in the output, while K/V provides global context, which can tolerate redundancy.
    - **Design Motivation**: Randomly dropping 30% of Q tokens leads to severe quality degradation, whereas dropping 30% of K/V tokens has almost no impact.

2. **Adaptive Reduction Scheduling**:

    - **Function**: Dynamically adjusts K/V reduction ratios based on layer depth and denoising steps.
    - **Mechanism**: Reduces less in shallow layers (highly sensitive to quality) and more in deep layers (higher redundancy); reduces less in early steps (affecting semantics) and more in late steps.
    - **Design Motivation**: Follows Liebig's Law of the Minimum—the overall quality of the system is determined by the most sensitive component.

3. **Matching Cache**:

    - **Function**: Caches and reuses token matching results across denoising steps.
    - **Mechanism**: Token similarity changes slowly between adjacent denoising steps, allowing the reuse of matching indexes.
    - **Design Motivation**: Reduces the computational overhead of the matching algorithm itself.

### Loss & Training
- Completely training-free and plug-and-play.
- Applicable to any video DiT architecture.

## Key Experimental Results

### Main Results

| Model | Method | Speedup | VBench↑ |
|------|------|--------|---------|
| HunyuanVideo | Baseline | 1.0× | 83.2 |
| | ToMe | 1.4× | 79.8 (-3.4) |
| | **AsymRnR** | **1.45×** | **83.5** (+0.3) |
| CogVideoX | Baseline | 1.0× | 81.5 |
| | **AsymRnR** | **1.38×** | **81.8** (+0.3) |

### Ablation Study

| Configuration | VBench | Speedup | Description |
|------|--------|------|------|
| Symmetric Reduction (Same ratio for Q+KV) | 79.8 | 1.4× | Severe distortion |
| **Asymmetric (KV only)** | **83.5** | **1.45×** | Lossless or even improved |
| Uniform Scheduling | 82.1 | 1.4× | Incompatible with layer/step variance |
| Adaptive Scheduling | **83.5** | **1.45×** | Finer control |

### Key Findings
- The asymmetric strategy is not only lossless but can even improve quality (reducing redundant K/V acts as a regularizer).
- Adaptive scheduling yields a +1.4 VBench score improvement over uniform scheduling.
- Consistently effective across 4 SOTA video DiT models.

## Highlights & Insights
- The discovery of the **asymmetric roles of Q and KV** is highly generalizable—it is not only useful for video DiTs but could also inspire inference acceleration across other Transformer architectures.
- The training-free and model-agnostic nature makes the proposed method extremely practical.
- The quality "improvement" suggests that there is redundant "noise" in the K/V tokens of the original models.

## Limitations & Future Work
- The speedup is currently around 1.4-1.5×, which is less than the acceleration factor achieved by step distillation methods.
- The reuse frequency of the matching cache needs to be manually tuned.
- Combination with step distillation or other orthogonal acceleration methods has not been fully explored.

## Related Work & Insights
- **vs ToMe**: Symmetric reduction leads to degradation, while AsymRnR's asymmetric strategy is lossless.
- **vs Feature Caching**: Feature caching is often architecture-dependent, whereas AsymRnR is architecture-agnostic.
- Outlines a broad inspiration for Transformer inference optimization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective of asymmetric token reduction is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 4 SOTA models with detailed VBench metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Clear analysis and intuitive visualizations.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical, training-free acceleration scheme for video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers](riflex_a_free_lunch_for_length_extrapolation_in_video_diffusion_transformers.md)
- [\[ICLR 2026\] Astraea: A Token-wise Acceleration Framework for Video Diffusion Transformers](../../ICLR2026/video_generation/astraea_a_token-wise_acceleration_framework_for_video_diffusion_transformers.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)
- [\[ICLR 2026\] UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers](../../ICLR2026/video_generation/ultravico_breaking_extrapolation_limits_in_video_diffusion_transformers.md)
- [\[ICCV 2025\] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers](../../ICCV2025/video_generation/magicmirror_id-preserved_video_generation_in_video_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
