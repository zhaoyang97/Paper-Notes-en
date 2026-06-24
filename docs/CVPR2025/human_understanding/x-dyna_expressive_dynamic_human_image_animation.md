---
title: >-
  [Paper Note] X-Dyna: Expressive Dynamic Human Image Animation
description: >-
  [CVPR 2025][Human Understanding][Human Animation] X-Dyna proposes a zero-shot human image animation pipeline based on diffusion models. Through a lightweight Dynamics-Adapter module, it generates realistic human and scene dynamics while maintaining appearance consistency, and introduces S-Face ControlNet to achieve identity-decoupled facial expression transfer.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Animation"
  - "Diffusion Models"
  - "Dynamic Generation"
  - "Expression Control"
  - "Appearance Reference"
date: 2026-05-08
content_hash: 5fa87c39c485184f
---

# X-Dyna: Expressive Dynamic Human Image Animation

**Conference**: CVPR 2025  
**arXiv**: [2501.10021](https://arxiv.org/abs/2501.10021)  
**Code**: [GitHub](https://github.com/bytedance/X-Dyna)  
**Area**: Human Understanding  
**Keywords**: Human Animation, Diffusion Models, Dynamic Generation, Expression Control, Appearance Reference

## TL;DR

X-Dyna proposes a zero-shot human image animation pipeline based on diffusion models. Through a lightweight Dynamics-Adapter module, it generates realistic human and scene dynamics while maintaining appearance consistency, and introduces S-Face ControlNet to achieve identity-decoupled facial expression transfer.

## Background & Motivation

Human video animation aims to drive a single human image to generate a video using the body poses and facial expressions from a driving video, which is widely applied in digital art, social media, and virtual avatars.

Existing methods mainly suffer from the following limitations:
- **ReferenceNet scheme**: Uses a parallel UNet replica to extract appearance features. Although it can effectively transfer appearance information, its strong constraints restrict spatial attention, leading to static backgrounds and stiff human movements.
- **IP-Adapter scheme**: Injects cross-attention layers based on CLIP image embeddings. However, CLIP embeddings struggle to capture detailed appearance information, leading to significant identity loss.
- **Lack of dynamic details**: Existing methods are disadvantageous for generating dynamic textures in terms of both training data and network design, such as fluttering hair, flowing clothes, and natural scene effects like waterfalls.
- **Insufficient facial expression control**: Simplified facial landmark maps lack expressive details and contain identity clues, causing facial identity leakage during cross-identity transfer.

The root cause of these issues is that the appearance reference module imposes excessively strong constraints on spatial attention, suppressing the original dynamic synthesis capability of the diffusion model.

## Method

### Overall Architecture

X-Dyna is built upon the pre-trained Stable Diffusion 1.5 and comprises three core modules: (1) Dynamics-Adapter for appearance reference injection; (2) Pose ControlNet $\mathcal{C}_P$ for body pose control; (3) S-Face ControlNet $\mathcal{C}_F$ for facial expression control. The training employs a mixed-data strategy, utilizing both human motion videos and natural scene videos.

### Key Designs

**Design 1: Dynamics-Adapter — Lightweight Cross-Frame Attention Appearance Injection**

- **Function**: Effectively transfers appearance information of the reference image without compromising the dynamic synthesis capabilities of the diffusion backbone.
- **Mechanism**: Feeds the denoised reference image $I_R$ and the noisy sequence in parallel into shared-weight UNet branches, calculating appearance guidance via a cross-frame attention mechanism. Specifically, the $K$ and $V$ projection layers of the original UNet are used to generate $K_R$ and $V_R$ of the reference image, while a trainable query projection layer replica generates $Q'_i$. The cross-frame attention is computed as $A'_i = \text{softmax}(\frac{Q'_i K_R^\top}{\sqrt{d}}) V_R$ and added to the original self-attention output in a residual manner through a zero-initialized output projection layer $W'_O$.
- **Design Motivation**: ReferenceNet imposes excessively strong constraints on all spatial pixels, leading to a loss of dynamics; Dynamics-Adapter, by minimizing trainable parameters (only the Query projection layer and output projection layer), preserves the spatial-temporal generation capabilities of the diffusion backbone, achieving effective decoupling between appearance control and motion generation.

**Design 2: S-Face ControlNet — Identity-Decoupled Implicit Facial Expression Control**

- **Function**: Achieves precise facial expression transfer in cross-identity driving scenarios, preventing the leakage of identity information from the driving signal.
- **Mechanism**: During training, a pre-trained portrait reenactment network $\mathcal{S}$ (e.g., FaceVid2Vid) is used to transfer facial expressions from the driving frame to a randomly selected subject with different facial attributes, generating face-swapped facial patches as conditional inputs for the auxiliary ControlNet $\mathcal{C}_F$. During inference, the cropped face of the driving video is directly used as input, eliminating the need for a face-swapping network.
- **Design Motivation**: Simplified facial landmark maps contain identity clues (such as face shape), leading to identity leakage during cross-identity transfer. Through the cross-identity training strategy, the ControlNet learns to implicitly extract identity-independent expression features from the synthesized face-swapped images.

**Design 3: Harmonic Data Fusion Training**

- **Function**: Enables the model to simultaneously learn human dynamics and background scene dynamic effects.
- **Mechanism**: Mixes natural scene videos (e.g., waterfalls, fireworks, wind) with real human motion videos during training. For scene videos containing no human figures, the conditional inputs of Pose ControlNet and S-Face ControlNet are left blank.
- **Design Motivation**: Existing methods are primarily trained on human videos with static backgrounds, failing to capture dynamic environmental details. The mixed training strategy allows the model to learn subtle human dynamics while reducing undesired influences on background motion from the ControlNet.

### Loss & Training

Standard DDPM denoising loss is adopted for end-to-end training. Training is performed in stages: first, the Dynamics-Adapter, Pose ControlNet, and motion module are trained for 5 epochs (using mixed data); then, these modules are frozen, and the S-Face ControlNet is trained independently for 2 epochs (using human videos only).

## Key Experimental Results

### Main Results: Dynamic Texture Generation Quality

| Method | FG-DTFVD ↓ | BG-DTFVD ↓ | DTFVD ↓ |
|------|-----------|-----------|---------|
| MagicAnimate | 1.753 | 2.142 | 2.601 |
| Animate-Anyone | 1.789 | 2.034 | 2.310 |
| MagicPose | 1.846 | 1.901 | 2.412 |
| MimicMotion | 2.639 | 3.274 | 3.590 |
| **X-Dyna** | **0.900** | **1.101** | **1.518** |

### Ablation Study: Contributions of Each Module

| Method | FG-DTFVD ↓ | BG-DTFVD ↓ | DTFVD ↓ | Face-Cos ↑ |
|------|-----------|-----------|---------|-----------|
| w/RefNet | 2.137 | 2.694 | 2.823 | 0.466 |
| w/IP-A | 3.738 | 4.702 | 4.851 | 0.292 |
| w/lmk | 0.914 | 1.125 | 1.589 | 0.406 |
| wo/face | 0.912 | 1.098 | 1.550 | 0.442 |
| wo/fusion | 1.301 | 1.467 | 1.652 | 0.495 |
| **X-Dyna** | **0.900** | **1.101** | **1.518** | **0.497** |

### Key Findings

- Dynamics-Adapter reduces DTFVD by approximately 46% compared to ReferenceNet, verifying the advantage of lightweight cross-frame attention in retaining dynamic capabilities.
- In user studies, X-Dyna comprehensively leads in foreground dynamics (3.87 vs. the highest of 2.34), background dynamics (4.26 vs. the highest of 2.78), and identity preservation (4.14).
- The mixed training strategy reduces BG-DTFVD from 1.467 to 1.101, significantly improving the quality of background dynamics.

## Highlights & Insights

1. **Precise Problem Diagnosis**: Accurately identifies that ReferenceNet's excessively strong constraint on spatial attention is the root cause of the lost dynamic details.
2. **Elegant Decoupling Design**: Dynamics-Adapter achieves the decoupling of appearance injection and dynamic generation via a zero-initialized residual mechanism, which is both simple and effective.
3. **Clever Cross-Identity Training Idea**: Synthesizes training data using a face-swapping network, allowing the ControlNet to implicitly learn identity-independent expression features.

## Limitations & Future Work

- When the target pose differs dramatically from the reference human (e.g., extreme scaling), appearance and identity preservation may not be perfect.
- The generation quality of hand poses still needs improvement.
- In the future, Dynamics-Adapter can be applied to more powerful foundation models (such as SVD, SDXL, SD3) and combined with camera trajectories or drag-based control.

## Related Work & Insights

- **ReferenceNet Series** (MagicAnimate, Animate-Anyone, MagicPose): Parallel UNet schemes are effective for appearance preservation but sacrifice dynamic behaviors.
- **AnimateDiff**: A general temporal module that injects temporal consistency into diffusion models.
- **IP-Adapter**: Injects appearance information via CLIP embeddings but suffers from insufficient detail preservation.
- **Insight**: When injecting conditional information into diffusion models, the key challenge lies in finding a balance between the strength of conditional constraints and the generative freedom of the model.

## Rating

⭐⭐⭐⭐ — Presents a clear problem diagnosis and an elegant solution in the field of human animation. The design idea of Dynamics-Adapter is highly generalizable, and the mixed-data training strategy is simple yet effective. The experiments are comprehensive with significant results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](../../ICCV2025/human_understanding/dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)

</div>

<!-- RELATED:END -->
