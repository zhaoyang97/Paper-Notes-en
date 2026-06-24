---
title: >-
  [Paper Note] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation
description: >-
  [CVPR 2025][Image Generation][Portrait Animation] This paper proposes MVPortrait, a two-stage text-guided framework (Text2FLAME + FLAME2Video). By using the FLAME 3D parametric face model as an intermediate representation, it utilizes MotionDM and EmotionDM diffusion models to generate motion and expression parameter sequences, respectively. Subsequently, a multi-view video generation model is employed to transform the rendered FLAME sequences into realistic multi-view portra…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Portrait Animation"
  - "Text-guided"
  - "FLAME"
  - "Multi-view Consistency"
  - "Diffusion Models"
  - "Facial Expression Control"
date: 2026-05-08
content_hash: b1978e8e607d4f99
---

# MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation

**Conference**: CVPR 2025  
**arXiv**: [2503.19383](https://arxiv.org/abs/2503.19383)  
**Code**: Not open-sourced  
**Area**: Image Generation/Portrait Animation  
**Keywords**: Portrait Animation, Text-guided, FLAME, Multi-view Consistency, Diffusion Models, Facial Expression Control

## TL;DR

This paper proposes MVPortrait, a two-stage text-guided framework (Text2FLAME + FLAME2Video). By using the FLAME 3D parametric face model as an intermediate representation, it utilizes MotionDM and EmotionDM diffusion models to generate motion and expression parameter sequences, respectively. Subsequently, a multi-view video generation model is employed to transform the rendered FLAME sequences into realistic multi-view portrait animations. This represents the first approach to achieve controllable portrait animation compatible with text, audio, and video modalities simultaneously.

## Background & Motivation

**Background**: The landscape of portrait animation is predominantly occupied by three types of methods: lip-syncing (e.g., EMO, VASA-1), motion-controlled methods (e.g., FollowYourEmoji, AniPortrait), and 3D-aware face reconstruction (e.g., Portrait4D-v2). Lip-syncing methods focus solely on mouth movements, lacking comprehensive expression representation capabilities; motion-controlled methods rely on landmark maps but tend to omit detailed facial nuances; 3D methods (such as triplane-based ones) suffer from "multi-face" artifacts in profile views.

**Limitations of Prior Work**: (1) Lack of explicit and independent control over both head motion and facial expressions; (2) Inability to generate temporally consistent videos across multiple camera views; (3) Although text-driven portrait animation offers the highest user friendliness, it remains highly under-explored, lacking both dedicated text-to-face-parameter datasets and robust generation frameworks.

**Key Challenge**: Textual descriptions typically bundle both motion (e.g., "shaking head") and expression (e.g., "surprised") together, yet their amplitudes and dynamic patterns vary drastically. Jointly generating them directly yields insufficient control granularity. Meanwhile, preserving multi-view consistency and text alignment represent competing objectives.

**Goal**: To precisely control both facial motion and emotion through textual descriptions while generating temporally and view-consistent multi-view portrait animations.

**Key Insight**: This work leverages the compact parameter space of the FLAME 3D parametric face model (shape, pose, expression) to decouple motion and expression into independent diffusion generation tasks, while directly obtaining multi-view renderings by manipulating head orientation parameters.

**Core Idea**: Using FLAME as a bridge, the portrait animation pipeline is decomposed into two stages: Text $\rightarrow$ FLAME parameter sequences $\rightarrow$ Multi-view videos, achieving decoupled motion/expression control and natural multi-view consistency.

## Method

### Overall Architecture

MVPortrait comprises two stages: In the **Text2FLAME** stage, FLAME shape parameters are first extracted from a reference image using DECA. The text prompt is split into motion and emotion descriptions, which are fed into MotionDM and EmotionDM to generate pose and expression parameter sequences, respectively, which are then combined to form the complete FLAME sequence. In the **FLAME2Video** stage, the FLAME sequence is rendered from multiple views into conditional image sequences. These, alongside the reference image, are input into a multi-view animation generation network (comprising a VAE, a FLAME Encoder, a Reference UNet, and a Denoising UNet) to produce temporally and view-consistent portrait animations.

### Key Designs

1. **Decoupled MotionDM and EmotionDM**:
    - **Function**: Generates the FLAME pose parameter sequence ($f_{\text{pose}} \in \mathbb{R}^{12}$) and expression parameter sequence ($f_{\text{exp}} \in \mathbb{R}^{50}$) separately.
    - **Mechanism**: Two independent Transformer-based diffusion models share the same network architecture (a single-layer encoder-transformer with a latent dimension of 64), conditioned on motion and emotion descriptions, respectively. Adapting the MDM architecture, they progressively denoise using the DDPM framework, while utilizing sliding window smoothing (window size 3) to eliminate jitter caused by raw data noise.
    - **Design Motivation**: The amplitude and motion patterns of rigid head motion and facial micro-expressions are profoundly different. Joint training often compromises accuracy on both. Ablation studies confirm that a joint generation variant leads to erroneous expressions and static head motion, validating the necessity of decoupled training.

2. **View Attention Multi-view Consistency Module**:
    - **Function**: Enables cross-view information sharing within the Denoising UNet, ensuring view consistency across the multi-view animations.
    - **Mechanism**: A view module is inserted after the motion module (temporal attention) to perform view attention (self-attention along the view dimension $m$) on the feature maps from the $\mathbb{R}^{(b \times t \times h \times w) \times m \times c}$ dimension. During feedforward propagation, multi-view features are reshaped into $\mathbb{R}^{(b \times m) \times t \times h \times w \times c}$ to maintain consistency with the single-view generation pipeline.
    - **Design Motivation**: Simply relying on multi-view conditions from FLAME renders is insufficient to guarantee cross-view consistency in the generated video. Ablation studies demonstrate that omitting the view module leads to severe facial artifacts and identity drifts in profile views.

3. **FLAME Encoder Condition Injection**:
    - **Function**: Injects pose and expression cues from the rendered FLAME images into the Denoising UNet.
    - **Mechanism**: Adopting a similar pose guider architecture as AniPortrait, the encoded feature maps of the FLAME renders are added to the noisy latents before being fed into the UNet. Meanwhile, the spatial appearance of the reference image is integrated via key/value concatenation using the spatial attention of the Reference UNet alongside CLIP high-level features adapted through cross-attention.
    - **Design Motivation**: FLAME renders encapsulate the complete 3D information of shape, pose, and expression. Injecting direct feature-level encodings from these renders is far more comprehensive than relying on sparse face landmarks alone, effectively constraining the generated portrait's face shape to remain faithful to the reference image.

### Loss & Training

- **Text2FLAME Stage**: $\mathcal{L}_{DM} = \mathcal{L}_{\text{simple}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}$, where $\mathcal{L}_{\text{simple}}$ represents the standard DDPM reconstruction loss, $\mathcal{L}_{\text{vel}}$ is the visual velocity geometric loss (L2 norm of velocity differences between adjacent frames), and $\lambda_{\text{vel}}=0.5$.
- **FLAME2Video Stage**: Adopts a three-stage training strategy: Phase 1 trains the FLAME Encoder, Reference UNet, and 2D spatial modules; Phase 2 freezes other components to train the motion module independently; Phase 3 isolates the view module for training.

## Key Experimental Results

### Main Results

**Text-Guided Single-View Portrait Animation (CelebV-Text Test Set)**:

| Method | LIQE↑ | FID↓ | CLIPSIM↑ | VideoClip↑ | Variability↑ | MC↑ | EC↑ |
|------|-------|------|----------|------------|-------------|-----|-----|
| AnimateAnything | 4.024 | 34.9 | 0.171 | 0.564 | 0.095 | 1.33 | 1.23 |
| MMVID-interp | 1.541 | 217.4 | 0.175 | 0.517 | 0.092 | 1.67 | 1.56 |
| **MVPortrait** | **4.760** | **28.6** | **0.183** | **0.595** | **0.110** | **2.57** | **2.29** |

**Multi-View Synthesis Comparison**:

| Method | LPIPS↓ | SSIM↑ | ID↑ |
|------|--------|-------|-----|
| Triplanenet | 0.0936 | 0.5974 | 0.7710 |
| Portrait4D-v2 | 0.4468 | 0.5278 | 0.8006 |
| **MVPortrait** | 0.2101 | **0.6224** | **0.8409** |

### Ablation Study

**Ablation of Text2FLAME Stage (CelebV-Text)**:

| Variant | CLIPSIM↑ | VideoClip↑ | MC↑ | EC↑ |
|------|----------|------------|-----|-----|
| No smoothing | 0.169 | 0.586 | 1.88 | 1.22 |
| Larger network | 0.174 | 0.548 | 1.44 | 1.75 |
| Joint training | 0.175 | 0.559 | 1.50 | 1.44 |
| **MVPortrait** | **0.183** | **0.595** | **2.57** | **2.29** |

### Key Findings

- MVPortrait significantly outperforms baselines in subjective ratings of Motion Consistency (MC=2.57) and Emotion Consistency (EC=2.29), indicating that decoupled generation delivers more precise semantic control.
- Removing the sliding window smoothing causes the Variability to artificially spike (to 0.243), yet the MC degrades to 1.88. This proves that high motion variability without smoothing originates from visual jitter rather than coherent motion.
- SSIM=0.6224 and ID=0.8409 validate that using FLAME as an intermediate representation effectively retains identity consistency across multiple views.
- The unified framework also remains highly competitive in video-driven (FLAME-L1=0.196, outperforming FollowYourEmoji and LivePortrait) and audio-driven scenarios.

## Highlights & Insights

1. **FLAME as a unified bridge** is an elegant structural design: a single intermediate representation elegantly addresses multi-modality compatibility (text/audio/video), decoupled emotion/motion generation, and multi-view rendering.
2. **Decoupled over joint generation** yields persuasive experimental findings: the joint generation variant leads to both flawed expressions and static motions, proving that the parameter distributions of these spaces diverge greatly.
3. Text-to-facial-motion defines a **newly paved task direction**. Due to the lack of prior datasets, the authors compiled the CelebV-TF dataset (15k+ text-FLAME pairs) based on CelebV-Text.
4. The three-stage training strategy (2D spatial baseline $\rightarrow$ temporal modeling $\rightarrow$ cross-view modeling) stands as a highly effective engineering practice for scaling dimensional consistencies.

## Limitations & Future Work

- Text annotation quality is inherently bottlenecked by the original video dataset (CelebV-Text), resulting in an imbalanced distribution of facial movements and expression tags.
- The parametric FLAME model has inherent limitations in capturing fine micro-expressions, rendering extremely subtle emotional variations difficult to encode.
- The FVD metric (570.0) is inferior to AnimateAnything (283.0), as larger-amplitude head rotations lead to distribution shifts compared to the ground-truth training distribution.
- The resolution of the 3D parameter space remains limited; FLAME possesses only 5023 vertices, which fails to represent fine facial wrinkles.
- Inference speed (0.45 s/frame for FLAME2Video) remains a barrier for real-time applications.

## Related Work & Insights

- **MDM** [Tevet et al.]: A diffusion model for text-driven body motion generation; this paper adapts its core paradigm to the facial motion parameter space.
- **AniPortrait** [Wei et al.]: Predicts reference and target poses using a 3D facial mesh; this work inherits its FLAME Encoder layout.
- **AnimateDiff** [Guo et al.]: Achieves video generation by inserting temporal attention layers into Stable Diffusion, from which the motion module in this work is directly adapted.
- **Portrait4D-v2** [Deng et al.]: A triplane-based multi-view method that suffers from "multi-face" artifacts, which are gracefully bypassed by using FLAME.
- **Insight**: Parametric models like FLAME can serve as a "structural intermediate representation" in various digital human tasks, decomposing high-dimensional generation challenges into highly controllable generation in low-dimensional parameter spaces.

## Rating

⭐⭐⭐⭐ — Proposes the first text-driven multi-view portrait animation framework. The design of using FLAME as a unified bridge is elegant and practical, supported by meticulous ablation studies on decoupled motion/expression. However, the upper representation limits of FLAME and the inference speed constrain real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](../../ECCV2024/image_generation/livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](../../CVPR2026/image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)

</div>

<!-- RELATED:END -->
