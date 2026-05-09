---
title: >-
  [Paper Note] Video Motion Graphs
description: >-
  [ICCV 2025][Image Generation][Video Motion Graphs] Video Motion Graphs proposes a retrieval-augmented generation system for human motion video synthesis. It constructs a motion graph from reference videos and performs conditioned path search to obtain keyframes, then employs HMInterp—a dual-branch diffusion-based frame interpolation model combining skeleton guidance from a Motion Diffusion Model and progressive condition training—to seamlessly connect discontinuous frames. The system supports multiple conditioning signals (music, speech, action labels) and significantly outperforms both generative and retrieval-based baselines in human motion video quality.
tags:
  - ICCV 2025
  - Image Generation
  - Video Motion Graphs
  - Video Frame Interpolation
  - Motion Diffusion Model
  - Retrieval-based Video Generation
  - Human Motion Video
date: 2026-05-08
content_hash: 5f7b914c0df6916c
---

# Video Motion Graphs

**Conference**: ICCV 2025
**arXiv**: [2503.20218](https://arxiv.org/abs/2503.20218)
**Code**: None (Adobe Research)
**Area**: Diffusion Models / Video Generation
**Keywords**: Video Motion Graphs, Video Frame Interpolation, Motion Diffusion Model, Retrieval-based Video Generation, Human Motion Video

## TL;DR

Video Motion Graphs proposes a retrieval-augmented generation system for human motion video synthesis. It constructs a motion graph from reference videos and performs conditioned path search to obtain keyframes, then employs HMInterp—a dual-branch diffusion-based frame interpolation model combining skeleton guidance from a Motion Diffusion Model and progressive condition training—to seamlessly connect discontinuous frames. The system supports multiple conditioning signals (music, speech, action labels) and significantly outperforms both generative and retrieval-based baselines in human motion video quality.

## Background & Motivation

**Background**: Human motion video generation follows two main paradigms: (1) generative methods synthesize all pixels directly from conditional inputs, offering flexibility but susceptible to artifacts such as limb distortion; (2) retrieval-based methods leverage real frames from reference videos to guarantee quality, but require a frame interpolation model to smooth transitions.

**Limitations of Prior Work**:
- Generative methods, even with DiT architectures (e.g., SVD), still struggle to avoid structural errors in hands and faces.
- Existing retrieval-based methods (GVR, TANGO) are designed exclusively for conversational gesture synthesis and rely on linear motion blending, making them unable to handle complex dynamic motions such as dance.
- Linear interpolation provides a reasonable approximation for only 78% of mild gesture motions, but is feasible for only 17% of complex dance motions.

**Key Challenge**: Retrieval-based methods enjoy a natural quality advantage by directly using real frames, but their frame interpolation module becomes the bottleneck—existing approaches cannot handle large-amplitude, nonlinear motions.

**Goal**: Design a general human motion video generation system that supports multiple conditioning inputs (music, speech, action labels) while ensuring both texture quality and motion trajectory accuracy.

**Key Insight**: Replace linear blending with a Motion Diffusion Model to generate motion guidance, and adopt a progressive condition training strategy to resolve identity consistency issues.

**Core Idea**: Dual-branch frame interpolation—a Motion Diffusion Model ensures correct skeletal motion trajectories, while a diffusion-based VFI ensures video texture quality; the two branches are integrated via progressive condition training.

## Method

### Overall Architecture

Video Motion Graphs is a four-stage system:
1. **Graph Initialization**: Represents reference videos as a directed graph, where nodes are frames and edges connect frame pairs that can be smoothly transitioned (based on 3D pose distance thresholds).
2. **Path Search**: Given a conditioning signal (music / speech / label), finds the optimal frame path via dynamic programming or Beam Search.
3. **Frame Interpolation**: Uses HMInterp to smooth discontinuous frame boundaries (generating 12 frames = 0.5s @ 24fps).
4. **Background Recomposition** (optional): Performs foreground separation and background generation for videos with dynamic backgrounds.

### Key Designs

1. **Motion Diffusion Model (MDM)**:

    - **Function**: Generates interpolated 2D joint position sequences between the start and end frames.
    - **Mechanism**: A Transformer-based denoising network redesigned in a UNet style (with skip connections and feature concatenation), fusing features from shallow to deep layers to produce more accurate nonlinear motion interpolation trajectories.
    - **Design Motivation**: Compared to linear blending, MDM handles nonlinear trajectories for highly dynamic actions such as drumming and dancing. Compared to the 8-layer vanilla Transformer used in the original MDM, the UNet structure preserves more motion detail.

2. **Video Frame Interpolation (VFI) Backbone**:

    - **Function**: Generates interpolated video frames conditioned on the start/end frames and the poses generated by MDM.
    - **Mechanism**: Based on an AnimateDiff UNet T2V model, augmented with a ReferenceNet to inject hierarchical appearance features, a Seed Image Guider to inject VAE latents of the start/end frames, and a Pose Guider to inject MDM-generated 2D poses. The CLIP text encoder is replaced with an image encoder.
    - **Improved Reference Decoder**: Based on the ToonCrafter Reference Decoder (initialized from SVD's temporal decoder), low-level latent features are injected during VAE decoding to preserve facial details. Replacing zero-padding with repetition-padding of reference frames improves PSNR by more than 1.0.

3. **Condition Progressive Training**:

    - **Function**: Trains the VFI module in stages—first learning identity conditioning, then pose conditioning.
    - **Mechanism**: Stage 1—Seed Pre-Training trains with image conditioning only for 100k steps to ensure interpolated frames faithfully reflect the reference appearance; Stage 2—Few-Step Pose Finetuning trains with both image and pose conditioning for 8k steps.
    - **Design Motivation**: Joint training on image and pose conditions simultaneously leads to appearance inconsistencies between generated and real frames. Experiments show that reversing the training order (pose before image) or extending pose fine-tuning steps both degrade identity preservation.

### Loss & Training

- The VFI module is trained with v-prediction.
- MDM is trained with $x_0$-prediction.
- The Reference Decoder is trained with MSE + perceptual loss.
- MDM and the Reference Decoder are trained separately and then frozen.

## Key Experimental Results

### Main Results

**Human Motion Video Generation Quality Comparison (Tab. 1)**:

| Method | PSNR↑ | LPIPS↓ | MOVIE↓ | FVD↓ |
|------|-------|--------|--------|------|
| AnimateAnyone | 35.55 | 0.044 | 54.68 | 1.369 |
| MagicPose | 35.64 | 0.048 | 51.97 | 1.277 |
| UniAnimate | 36.75 | 0.042 | 49.89 | 1.090 |
| MimicMotion | 36.30 | 0.047 | 46.84 | 1.078 |
| **Ours (f=32)** | **42.91** | **0.009** | **37.31** | **0.180** |
| Ours (f=64) | 42.75 | 0.010 | 37.53 | 0.213 |
| Ours (f=216) | 39.75 | 0.029 | 39.89 | 0.799 |

Even under the most challenging f=216 setting, the proposed method outperforms all pose-to-video baselines.

**User Study Win Rates (Tab. 2)**:

| Dimension | vs Dance | vs Gesture | vs Action |
|---------|----------|-----------|-----------|
| Texture Quality | 82.10% | 78.38% | 69.12% |
| Cross-modal Alignment | 88.39% | 47.63% | 45.21% |
| Overall Preference | 84.99% | 70.24% | 61.05% |

### Ablation Study

**HMInterp Module Ablation (Tab. 5)**:

| Configuration | PSNR↑ | LPIPS↓ | MOVIE↓ | FVD↓ |
|------|-------|--------|--------|------|
| **HMInterp (s=1)** | **39.53** | **0.034** | **39.18** | **1.210** |
| w/o motion guidance | 39.17 | 0.048 | 41.34 | 1.391 |
| Linear motion guidance | 39.16 | 0.042 | 41.06 | 1.297 |
| w/o Reference Decoder | 37.21 | 0.039 | 49.67 | 1.283 |
| Zero-padding Reference Decoder | 38.13 | 0.034 | 40.11 | 1.221 |

**Progressive Condition Training Ablation (Tab. 6)**:

| Stage 1 | Stage 2 | PSNR↑ | LPIPS↓ | FVD↓ |
|---------|---------|-------|--------|------|
| P (pose-to-video) | - | 35.55 | 0.044 | 1.369 |
| P+SI jointly | - | 36.62 | 0.041 | 1.325 |
| **SI** | **P+SI (8k)** | **37.21** | **0.039** | **1.283** |
| SI | P+SI (30k) | 36.87 | 0.041 | 1.307 |

### Key Findings

- **Motion guidance is critical**: Removing MDM motion guidance degrades LPIPS from 0.034 to 0.048, indicating the model tends to attend to incorrect regions without explicit guidance. MDM-guided nonlinear trajectories substantially outperform linear blending.
- **Reference Decoder improvement**: Repetition-padding of reference frames significantly improves facial and background detail at low resolution (256×256) compared to zero-padding.
- **Necessity of progressive training**: Introducing pose conditioning too early disrupts appearance consistency. Learning "who" before learning "how to move" is essential.
- **Impact of reference video length**: A 100-second reference database already surpasses current generative models; with a 1000-second database, motion diversity approaches that of real videos.

## Highlights & Insights

- **Retrieval-augmented generation paradigm** — Applying the generative model to only a small number of frames while sourcing the majority from real video is highly practical given the current imperfections of generative models. This strategy is transferable to other quality-critical video editing scenarios.
- **Progressive condition training** — A general strategy for resolving identity consistency issues in multi-condition joint training. The principle of learning strong conditions before weak conditions is directly applicable to other pose-to-video or audio-to-video tasks.
- **UNet-style redesign of MDM** — Introducing skip connections into diffusion Transformers is a simple yet effective modification, broadly applicable to generation tasks requiring fine-grained detail preservation.

## Limitations & Future Work

- Handling dynamic backgrounds depends on an additional foreground separation and background generation module, precluding an end-to-end formulation.
- Cross-modal alignment preference in gesture and action label tasks is only approximately 45–47%, indicating room for improvement in conditional alignment beyond dance scenarios.
- The system engineering complexity is high; the four-stage pipeline lacks elegance.
- The path search algorithm for conditioning signals relies on heuristic rules and may not be globally optimal.

## Related Work & Insights

- **vs GVR/TANGO**: These methods handle only linear motion interpolation for conversational gestures and are limited to static backgrounds. Video Motion Graphs extends retrieval-based generation to general human motion via MDM and progressive training.
- **vs DanceAnyBeat**: This generative dance method is strongly preferred by users in texture quality compared to Video Motion Graphs (82.10% preference), highlighting the quality advantage of retrieval-based approaches.
- **vs DCInterp/ACInterp**: Diffusion-based frame interpolation methods using linear motion guidance suffer from degraded frame quality in dynamic scenarios due to incorrect motion trajectories.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of motion graphs and diffusion-based frame interpolation constitutes a novel engineered system; the progressive training strategy offers genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes objective metrics, a user study with 82 participants, multi-task evaluation, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ System description is clear, though the overall pipeline is complex.
- Value: ⭐⭐⭐⭐ Strong practical utility with support for real-time generation and keyframe editing; promising industrial applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents](reducio_generating_1k_video_within_16_seconds_using_extremely_compressed_motion_.md)
- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](tlb-vfi_temporal-aware_latent_brownian_bridge_diffusion_for_video_frame_interpol.md)
- [\[ICCV 2025\] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups](pino_person-interaction_noise_optimization_for_long-duration_and_customizable_mo.md)

</div>

<!-- RELATED:END -->
