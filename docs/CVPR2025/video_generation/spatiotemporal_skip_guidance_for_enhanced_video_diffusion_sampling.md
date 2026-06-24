---
title: >-
  [Paper Note] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Models] STG (Spatiotemporal Skip Guidance) proposes to construct an implicit weak model as a degraded version of the original model by selectively skipping spatiotemporal layers of the Transformer for self-perturbed guidance. This improves the generation quality of video diffusion models without additional training, while maintaining sample diversity and motion dynamics, overcoming the fundamental flaw of CFG which causes drop in…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Sampling Guidance"
  - "Layer Skipping"
  - "Training-Free"
  - "CFG Alternative"
date: 2026-05-08
content_hash: adbb63c306c96700
---

# Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling

**Conference**: CVPR 2025  
**arXiv**: [2411.18664](https://arxiv.org/abs/2411.18664)  
**Code**: [https://github.com/junhahyung/STGuidance](https://github.com/junhahyung/STGuidance)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Video Diffusion Models, Sampling Guidance, Layer Skipping, Training-Free, CFG Alternative  

## TL;DR
STG (Spatiotemporal Skip Guidance) proposes to construct an implicit weak model as a degraded version of the original model by selectively skipping spatiotemporal layers of the Transformer for self-perturbed guidance. This improves the generation quality of video diffusion models without additional training, while maintaining sample diversity and motion dynamics, overcoming the fundamental flaw of CFG which causes drop in diversity and dynamics in video generation.

## Background & Motivation

**Background**: Diffusion models have become the mainstream tool for generating high-quality images, videos, and 3D content. CFG (classifier-free guidance) is the most widely used sampling guidance technique, enhancing generation quality by contrasting conditional and unconditional generation. Autoguidance proposes using a weak model as an alternative to the unconditional model to mitigate the issues of CFG, but requires training an additional weak model, which is impractical for large-scale video diffusion models.

**Limitations of Prior Work**: CFG exhibits severe flaws in video generation—while it improves single-frame image quality, it significantly reduces sample diversity and the extent of motion dynamics. This is because the guidance direction of CFG pushes too strongly towards the conditional mode, resulting in converging generation results and static motion. Although Autoguidance can mitigate this issue, training a dedicated weak model for each video model is prohibitively expensive and impractical.

**Key Challenge**: The trade-off between quality improvement and diversity/dynamics—existing guidance methods either sacrifice diversity for quality (CFG) or require additional training overhead (Autoguidance).

**Goal**: (1) Design a training-free guidance method; (2) Improve video quality while maintaining diversity and dynamics; (3) Be applicable to video diffusion models of different architectures.

**Key Insight**: Inspired by Autoguidance, the authors consider whether a weak model can be constructed through self-perturbation instead of training. It is observed that skipping certain layers in a Transformer yields an "aligned, degraded version" of the model, which naturally serves as a weak model.

**Core Idea**: Construct an implicit weak model by selectively skipping spatiotemporal attention or residual blocks, and use the output difference between the original model and the skipped model as the guidance signal, i.e., $\hat{x} = x + s \cdot (x_{full} - x_{skip})$, where $s$ is the guidance scale.

## Method

### Overall Architecture
In each denoising step of the video diffusion model: (1) Perform a forward pass with the full model to get the normal prediction $x_{full}$; (2) Re-run the forward pass with specified spatiotemporal layers skipped to get the degraded prediction $x_{skip}$; (3) Use the difference between the two as the guidance direction, scaled by the guidance strength, and add it to the final prediction. This entire process is embedded in the sampling loop without modifying the model weights.

### Key Designs

1. **Residual Skip**:

    - **Function**: Generate degraded output by skipping entire residual blocks.
    - **Mechanism**: For a residual block $\text{Res}(z_l) = f_l(z_l) + z_l$, residual skip simplifies it to an identity mapping $\text{Res}'(z_l) = z_l$, completely bypassing the non-linear transformation of that layer. This effectively removes all feature modulation capabilities learned by this layer, producing a "simpler" but structurally aligned output.
    - **Design Motivation**: Residual skip introduces a stronger perturbation, making it suitable for scenarios with simple attention layers or shallow models, where it can produce more distinct guidance signals.

2. **Attention Skip**:

    - **Function**: Generate milder degraded output by skipping only self-attention calculations.
    - **Mechanism**: In self-attention $\text{SA}(Q,K,V) = \text{Softmax}(QK^T/\sqrt{d})V = AV$, skip the computation of the attention matrix $A$ and replace it directly with the identity matrix (meaning each token only attends to itself). This preserves most of the network structure while disrupting the spatial and temporal relationships between tokens.
    - **Design Motivation**: It is gentler than residual skip, retaining more model capability while still producing effective degradation signals. For complex large-scale models, the guidance effect of this mild perturbation is more stable.

3. **Spatiotemporal Layer Selection Strategy**:

    - **Function**: Determine which layers to skip for optimal guidance effect.
    - **Mechanism**: Instead of skipping randomly or entirely, specific spatiotemporal attention layers are selectively skipped. Experiments find that skipping middle layers generally outperforms skipping shallow or deep layers. The guidance strength $s$ controls the impact of the perturbation—this hyperparameter needs to be tuned across different models.
    - **Design Motivation**: Different layers encode different levels of information (shallow layers encode low-frequency structure, deep layers encode high-frequency details). Selecting the right layers to skip yields effective quality difference signals without destroying core semantics.

### Loss & Training
STG is entirely training-free and is only introduced during the sampling loop at inference time. The guidance formula is $\hat{x} = x_{full} + s \cdot (x_{full} - x_{skip})$, where the extra forward pass adds approximately 50% inference overhead (similar to CFG).

## Key Experimental Results

### Main Results

| Model | Guidance Method | Quality↑ | Semantic↑ | I.Q.↑ | Dyn.Deg.↑ | T.Flicker↓ |
|------|---------|---------|----------|-------|----------|------------|
| Mochi | CFG | 0.524 | 0.507 | 0.985 | 0.87 | 0.976 |
| Mochi | **STG** | **0.628** | **0.554** | **0.988** | 0.86 | **0.978** |
| Open-Sora | CFG | 0.561 | 0.493 | 0.982 | **0.902** | 0.975 |
| Open-Sora | **STG** | **0.606** | **0.509** | **0.987** | 0.895 | **0.976** |

| Model | Guidance Method | FVD↓ | IS↑ | Quality↑ | Semantic↑ | T.Flicker↓ | Dyn.Deg.↑ |
|------|---------|------|-----|---------|----------|------------|----------|
| SVD | CFG | 151.3 | 38.0 | 0.687 | 0.637 | 0.966 | 0.562 |
| SVD | **STG** | **128.7** | **38.5** | **0.694** | **0.639** | **0.968** | **0.694** |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| STG (Residual Skip) | High | Better for small models, stronger perturbation |
| STG (Attention Skip) | Higher | Better for large models, milder perturbation |
| Skip Shallow Layers | Medium | Affects low-frequency structure |
| Skip Middle Layers | Optimal | Balances semantics and details |
| Skip Deep Layers | Medium-Low | Affects high-frequency details |

### Key Findings
- STG outperforms CFG across all tested video models (Mochi, Open-Sora, SVD), with significant improvements particularly in the Quality and Semantic dimensions.
- The most notable highlight is the preservation of **Dynamic Degree**: SVD+STG improves the dynamic degree from 0.562 to 0.694, whereas CFG tends to degrade motion dynamics.
- Attention Skip generally performs better than Residual Skip due to its gentler and more controllable perturbation.
- User studies indicate that STG is superior to CFG across multiple dimensions such as visual quality, text alignment, and motion naturalness.
- The method is robust to the guidance scale $s$, maintaining good performance across a wide range.

## Highlights & Insights
- **The clever insight of "layer skipping as a weak model"**: Replacing the separate weak model that needs training in Autoguidance with an implicit weak model constructed via layer skipping. This simple yet profound observation transforms an expensive training problem into a zero-cost inference trick.
- **The triple balance of Quality-Diversity-Dynamics**: CFG forces a binary choice between quality and diversity, whereas STG maintains all three simultaneously through gentler guidance. This is particularly crucial for video generation—videos without dynamics are pointless.
- **Strong generalizability**: The method is applicable to various models such as Mochi (DiT architecture), Open-Sora (STDiT architecture), and SVD (UNet architecture), showing that layer skip guidance is a general principle rather than an architecture-specific trick.

## Limitations & Future Work
- The additional forward pass introduces roughly 50% inference time overhead, which might be high for real-time video generation scenarios.
- The choice of optimal layers to skip currently requires manual tuning, lacking an automated selection strategy.
- Theoretical analysis of layer skipping is not yet deep enough—why does skipping middle layers work better than shallow or deep layers?
- Not tested on the latest DiT-based long-form video models (such as CogVideoX).
- Adaptive guidance strategies could be explored, dynamically adjusting which layers to skip or varying the guidance scale across denoising steps.
- Extending this idea to image diffusion models and 3D generation is also a valuable direction.

## Related Work & Insights
- **vs CFG**: CFG uses an unconditional model as a guidance baseline, essentially contrasting conditional and unconditional states. STG uses a degraded version of the original model as a baseline, leading to a gentler and more aligned guidance direction, thus preventing over-compression of diversity.
- **vs Autoguidance (Karras et al. 2024)**: Autoguidance first proposed using a weak model as a replacement for unconditional models, but required additional training. STG's layer skipping strategy perfectly inherits its theoretical advantages while eliminating the training overhead.
- **vs PAG (Perturbed Attention Guidance)**: PAG guides by randomly perturbing attention maps. STG's layer skipping scheme is more structured and produces more predictable degraded outputs, resulting in highly stable guidance quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of constructing a weak model using layer skipping is simple yet highly effective—a overlooked great idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three video models with different architectures, combining quantitative, qualitative, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts and comprehensive experimental presentation.
- Value: ⭐⭐⭐⭐⭐ As a plug-and-play alternative to CFG, it has a direct impact on the entire video generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICML 2026\] Self-Refining Video Sampling](../../ICML2026/video_generation/self-refining_video_sampling.md)
- [\[ICLR 2026\] Realtime Video Frame Interpolation Using One-Step Diffusion Sampling](../../ICLR2026/video_generation/realtime_video_frame_interpolation_using_one-step_diffusion_sampling.md)

</div>

<!-- RELATED:END -->
