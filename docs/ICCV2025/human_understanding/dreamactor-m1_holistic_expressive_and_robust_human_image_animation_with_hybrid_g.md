---
title: >-
  [Paper Note] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance
description: >-
  [ICCV 2025][Human Understanding][Human Animation] DreamActor-M1 proposes a human image animation framework based on the DiT architecture, achieving fine-grained facial and body control through hybrid control signals comprising implicit facial representations, 3D head spheres, and 3D body skeletons. Combined with complementary appearance guidance and a progressive training strategy, the framework supports multi-scale generation ranging from portrait to full-body.
tags:
  - ICCV 2025
  - Human Understanding
  - Human Animation
  - Diffusion Transformer
  - Hybrid Control Signals
  - Multi-scale Adaptation
  - Long-term Consistency
date: 2026-05-08
content_hash: 9e20a025fc46d067
---

# DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance

**Conference**: ICCV 2025  
**arXiv**: [2504.01724](https://arxiv.org/abs/2504.01724)  
**Code**: None (Project page: [https://grisoon.github.io/DreamActor-M1/](https://grisoon.github.io/DreamActor-M1/))  
**Area**: Human Understanding  
**Keywords**: Human Animation, Diffusion Transformer, Hybrid Control Signals, Multi-scale Adaptation, Long-term Consistency

## TL;DR
DreamActor-M1 proposes a human image animation framework based on the DiT architecture, achieving fine-grained facial and body control through hybrid control signals comprising implicit facial representations, 3D head spheres, and 3D body skeletons. Combined with complementary appearance guidance and a progressive training strategy, the framework supports multi-scale generation ranging from portrait to full-body.

## Background & Motivation
Human image animation is a prominent research direction in video generation, with broad applications in film production, advertising, and gaming. Methods for driving human motion from a single image have achieved significant progress, yet several key challenges remain:

**Insufficient holistic fine-grained control**: Existing methods excel either at facial animation (GAN/NeRF methods limited to portrait regions) or body animation (diffusion methods neglecting fine facial expressions), but cannot precisely control both simultaneously.

**Poor multi-scale adaptability**: Different inputs (portrait/half-body/full-body) differ in information density and focus; existing methods struggle to handle these variations within a unified framework.

**Long-term consistency degradation**: Because long videos cannot be generated in a single pass, visual information from unseen regions of the reference image (e.g., back-side clothing textures) is progressively lost across concatenated segments, leading to inconsistencies.

**Single control signal dilemma**: Skeleton-based control is insufficiently fine-grained for facial regions, while 3DMM parameters cannot govern body motion.

**Key Insight**: Design hybrid control signals to decouple facial expression from body motion, introduce complementary appearance guidance to fill the information gap for unseen regions, and adopt a progressive training strategy to accommodate multi-scale inputs.

## Method

### Overall Architecture
Built upon a pretrained Image-to-Video DiT model (Seaweed) with MMDiT as the backbone. The reference image latent and driving video latent are concatenated and fed into the DiT, where 3D self-attention and spatial cross-attention facilitate reference–video interaction. The core innovation lies in the design and injection of hybrid control signals.

### Key Designs

1. **Hybrid Motion Guidance Signals**:

    - **Implicit Facial Representations**:
        - **Function**: Fine-grained control of facial expressions with simultaneous disentanglement of expression, identity, and head pose.
        - **Mechanism**: Faces are detected and cropped from the driving video, normalized to $F \in \mathbb{R}^{t \times 3 \times 224 \times 224}$, and encoded by a pretrained facial motion encoder $\textbf{E}_f$ into facial motion tokens $M \in \mathbb{R}^{t \times c}$, which are injected via cross-attention within DiT blocks.
        - **Design Motivation**: Compared to facial landmarks, implicit representations capture subtler expression details (e.g., blinks, lip tremors). The facial motion encoder is initialized with identity-agnostic expression features pretrained on large-scale datasets to ensure robustness. An additional audio-driven encoder is trained to support speech-driven lip synchronization.

    - **3D Head Spheres**:
        - **Function**: Independent control of head pose (rotation, position, and scale).
        - **Mechanism**: Camera parameters and rotation angles are extracted via 3D face tracking and rendered as colored spheres projected onto the 2D plane. The sphere position is aligned with the driving head, the size matches the reference head, and the color is determined by the driving head orientation.
        - **Design Motivation**: Complex 3D head motion is abstracted into a simple 2D sphere representation, reducing the model's learning complexity and particularly benefiting the preservation of special head structures in anime/cartoon characters.

    - **3D Body Skeletons**:
        - **Function**: Control of body and hand motion.
        - **Mechanism**: SMPL-X parameters are estimated using 4DHumans and HaMeR; joint projections are extracted to 2D and connected as skeleton lines. During inference, bone length normalization is applied to reconcile body shape differences between the reference and driving subjects.
        - **Design Motivation**: Skeletons are preferred over full-body mesh rendering (e.g., Champ's approach) to avoid strong body-shape constraints, encouraging the model to learn character shape and appearance from the reference image.

   The three signals are combined as follows: body skeletons and head spheres are encoded by a pose encoder and concatenated with noisy video features, while facial motion tokens are injected via cross-attention.

2. **Complementary Appearance Guidance**:

    - **Function**: Provides visual information for unseen regions to maintain long-term consistency.
    - **Mechanism**:
        - During training: three keyframes are selected by sorting on yaw angle (maximum, minimum, and median rotation); for full-body videos, half-body portrait crops are additionally extracted as auxiliary references.
        - During inference, an optional two-stage mode is available: a multi-view sequence is first generated from the single reference image, keyframes are selected as complementary references, and the final video is then generated using multi-reference inference.
    - **Design Motivation**: A single reference image cannot supply appearance information for rotated or side-view scenarios; the multi-reference strategy addresses this information gap by covering multiple viewpoints.

3. **Progressive Training Strategy**:

    - **Function**: Incrementally introduces control signals across three stages.
    - Stage 1 (20k steps): Only 3D skeletons and head spheres are used to establish fundamental human animation capability.
    - Stage 2 (20k steps): Implicit facial representations are introduced; all other parameters are frozen, and only the face motion encoder and face attention layers are trained.
    - Stage 3 (30k steps): All parameters are unfrozen for joint optimization.
    - **Design Motivation**: Avoids learning difficulties caused by simultaneously introducing too many complex signals and ensures a stable and effective training process.

### Loss & Training
Flow Matching is used as the training objective. Training video lengths are randomly sampled between 25 and 121 frames; spatial resolution is resized to $960 \times 640$ area while preserving the original aspect ratio. Training uses 8×H20 GPUs, AdamW optimizer, and a learning rate of $5 \times 10^{-6}$. During inference, each segment consists of 73 frames, with the last frame's latent of the current segment used as the initial latent for the next. The CFG scale is set to 2.5.

## Key Experimental Results

### Main Results: Body Animation Comparison

| Method | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ |
|--------|------|-------|-------|--------|------|
| Animate Anyone | 36.72 | 0.791 | 21.74 | 0.266 | 158.3 |
| Champ | 40.21 | 0.732 | 20.18 | 0.281 | 171.2 |
| MimicMotion | 35.90 | 0.799 | 22.25 | 0.253 | 149.9 |
| DisPose | 33.01 | 0.804 | 21.99 | 0.248 | 144.7 |
| **DreamActor-M1** | **27.27** | **0.821** | **23.93** | **0.206** | **122.0** |

### Main Results: Portrait Animation Comparison

| Method | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ |
|--------|------|-------|-------|--------|------|
| LivePortrait | 31.72 | 0.809 | 24.25 | 0.270 | 147.1 |
| X-Portrait | 30.09 | 0.774 | 22.98 | 0.281 | 150.9 |
| SkyReels-A1 | 30.66 | 0.811 | 24.11 | 0.262 | 133.8 |
| Act-One | 29.84 | 0.817 | 25.07 | 0.259 | 135.2 |
| **DreamActor-M1** | **25.70** | **0.823** | **28.44** | **0.238** | **110.3** |

### Ablation Study

| Configuration | FID↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | Note |
|---------------|------|-------|-------|--------|------|------|
| Single-reference inference | 28.22 | 0.798 | 25.86 | 0.223 | 120.5 | Baseline |
| Multi-reference (pseudo) inference | 26.53 | 0.812 | 26.22 | 0.219 | 116.6 | Multi-reference improves long-term consistency |

Control signal ablation (qualitative):
- Replacing skeletons + spheres with 3D mesh → significant performance degradation; bone length adjustment and the simplified representation prove more effective.
- Replacing implicit facial representations with 3D facial landmarks → loss of fine-grained expression details.

### Key Findings
- DreamActor-M1 comprehensively outperforms the state of the art on both body animation and portrait animation tasks.
- Body animation FID decreases from 33.01 to 27.27 (17.4% improvement); FVD decreases from 144.7 to 122.0 (15.7% improvement).
- Portrait animation PSNR improves from 25.07 to 28.44 (+3.37 dB), indicating highly accurate facial reconstruction.
- Multi-reference inference reduces FVD from 120.5 to 116.6; single-reference inference is already sufficiently strong.
- Implicit facial representations achieve a qualitative leap over 3D landmarks in capturing subtle expressions.
- The framework supports audio-driven lip synchronization (speech-to-lip).

## Highlights & Insights
- The hybrid control signal design is elegant: implicit representations ensure facial precision, spheres simplify head pose learning, and skeletons preserve body shape flexibility.
- Rather than employing an additional ReferenceNet for appearance injection (as in Animate Anyone), reference tokens are directly concatenated with video tokens and interact via self-attention, yielding a cleaner architecture.
- The progressive training strategy effectively prevents learning conflicts arising from simultaneously introducing multiple complex signals.
- The two-stage inference mode of complementary appearance guidance elegantly resolves the consistency problem of unseen regions in long video generation.
- The framework supports independent disentangled control of facial expression and body motion, with facial motion drivable by either video or speech.

## Limitations & Future Work
- Dynamic camera motion control is not supported — the current framework only handles fixed or simple camera setups.
- Physical interactions with environmental objects (e.g., picking up a cup) cannot be generated.
- Bone length adjustment relies on a pretrained image editing model, which is unstable in edge cases and requires multiple iterations of manual selection.
- The dataset is proprietary (500 hours of self-collected video), limiting reproducibility.
- Training cost is high: 8×H20 GPUs for approximately 70k total steps.

## Related Work & Insights
- Represents a comprehensive upgrade over Animate Anyone, MimicMotion, Champ, and similar works: advancing from coarse to fine-grained control.
- Continues the trend of replacing UNet with DiT architectures, as seen in SkyReels-A1, HumanDiT, and others.
- Combines implicit facial representations (e.g., EMOPortraits) with explicit skeleton-based control in a complementary manner.
- Insight: Disentangled design of multimodal control signals is key to achieving holistic animation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The disentangled hybrid control signal design and complementary appearance guidance constitute the core innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive dual-task comparison across body and portrait animation; ablation depth could be further strengthened.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear, though certain details (e.g., audio-driven mode) are not fully elaborated.
- **Value**: ⭐⭐⭐⭐⭐ Significantly surpasses the state of the art in both practicality and quality, representing an important advance in human image animation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](../../CVPR2026/human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[CVPR 2026\] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](../../CVPR2026/human_understanding/sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)

<!-- RELATED:END -->
