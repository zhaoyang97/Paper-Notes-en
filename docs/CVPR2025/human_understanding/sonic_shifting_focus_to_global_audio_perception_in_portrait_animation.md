---
title: >-
  [Paper Note] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
description: >-
  [CVPR 2025][Human Understanding][Audio-driven portrait animation] The Sonic framework is proposed, establishing global audio perception as the core paradigm (rather than relying on visual motion frames). Through three modules—context-enhanced audio learning, a motion-decoupled controller, and time-aware position shift fusion—it achieves high-quality and temporally consistent audio-driven portrait animation generation.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Audio-driven portrait animation"
  - "global audio perception"
  - "temporal consistency"
  - "motion decoupling"
  - "diffusion models"
date: 2026-05-08
content_hash: a76edca96be5421e
---

# Sonic: Shifting Focus to Global Audio Perception in Portrait Animation

**Conference**: CVPR 2025  
**arXiv**: [2411.16331](https://arxiv.org/abs/2411.16331)  
**Code**: [Project Page](https://jixiaozhong.github.io/Sonic/)  
**Area**: Human Understanding/Portrait Animation  
**Keywords**: Audio-driven portrait animation, global audio perception, temporal consistency, motion decoupling, diffusion models

## TL;DR

The Sonic framework is proposed, establishing global audio perception as the core paradigm (rather than relying on visual motion frames). Through three modules—context-enhanced audio learning, a motion-decoupled controller, and time-aware position shift fusion—it achieves high-quality and temporally consistent audio-driven portrait animation generation.

## Background & Motivation

- Talking face animation requires animating a static portrait image based on speech audio, achieving lip synchronization, facial expressions, and head movements.
- Current methods completely decouple audio control and temporal consistency: audio is segmented by timestamps to match each frame, while temporal consistency relies on temporal self-attention or motion frames.
- Segmented audio processing restricts each frame to accessing only neighboring audio information, failing to effectively convert it into optimal motion representations.
- Relying on visual signals such as motion frames to maintain temporal consistency contradicts the essence of being "audio-driven".
- The temporal receptive fields of temporal self-attention and overlapping frame strategies are limited, which may weaken motion diversity and fail to consider audio information.
- There is a need for a purely audio-driven paradigm that utilizes global audio information as the sole prior to control facial motion.

## Method

### Overall Architecture

Sonic is a single-stage framework that takes a single portrait reference image and an audio signal as inputs and outputs a portrait video synchronized with the audio. Focusing on global audio perception, the framework comprises three core modules: (1) Context-enhanced audio learning, which extracts long-range temporal knowledge from audio and injects it into the denoising U-Net via spatial and temporal cross-attentions; (2) Motion-decoupled controller, which decouples head motion and expression motion into independently controllable parameters; (3) Time-aware position shift fusion, which progressively shifts the processing window across denoising timesteps, extending intra-video audio perception to global inter-video audio perception.

### Key Designs

**1. Context-enhanced Audio Learning**

- **Function**: Extract long-range temporal knowledge from audio to drive lip-sync and facial expressions.
- **Mechanism**: Use Whisper-Tiny to extract multi-scale audio features (concatenating the last five stages), where each frame corresponds to 0.2 seconds of audio context. The audio embedding $c_a \in \mathcal{R}^{b \times f \times d \times c}$ is injected into the facial region (restricted by a facial detection bounding box mask) via spatial cross-attention, and into the temporal module via temporal cross-attention. The temporal audio module pools audio features along the temporal dimension to reduce computational complexity.
- **Design Motivation**: Audio contains intonation and speech rate information, which implicitly express priors for expressions and head movements. Unlike AnimateDiff, which only performs visual temporal self-attention, this method introduces audio temporal cross-attention to directly guide motion using audio signals.

**2. Motion-decoupled Controller**

- **Function**: Independently control head translational motion and expression motion to enhance interactivity.
- **Mechanism**: During the training phase, two motion magnitude parameters are calculated from the video: the translation bucket $m_t$ (variance of video frame detection bounding boxes) and the expression bucket $m_e$ (variance of relative keypoints), with a range of $[0, 128]$. These are injected into the ResNet blocks through positional encodings and linear projections. During inference, they can be automatically predicted from the audio and reference image CLIP embeddings, and multiplied by a scale factor $\beta$ (0.5 for mild / 1.0 for moderate / 2.0 for intense).
- **Design Motivation**: Expression motion is strongly correlated with audio, whereas habitual head movement is weakly correlated. Decoupling allows both types of motion to be adjusted independently, also permitting users to customize exaggerated movements.

**3. Time-aware Position Shift Fusion**

- **Function**: Extend intra-video audio perception to global inter-video audio perception, ensuring temporal consistency in long videos.
- **Mechanism**: In each timestep $t$ of the denoising loop, the sliding window processing position is shifted relative to the previous timestep by $\alpha$ frames (e.g., 3 or 7 frames), forcing the model to process audio-video segments from different starting positions during each denoising step. Global connectivity is established progressively through accumulated shifts $\alpha_\Sigma = \alpha_\Sigma + \alpha$. A circular padding strategy is adopted at the end of the sequence.
- **Design Motivation**: Existing methods use overlapping frames or motion frames to maintain temporal consistency, which increases training/inference overhead. Time-aware position shift fusion incurs no extra training cost and introduces no additional inference time from overlapping frames, while naturally using the diffusion model to bridge context across timesteps.

### Loss & Training

- Standard diffusion denoising loss (MSE), based on the Stable Video Diffusion architecture.
- Whisper-Tiny is utilized as the audio encoder (which is more lightweight than the commonly used Wav2Vec).
- Motion bucket parameters are automatically calculated from the training videos and adaptively predicted from the audio and reference image during inference.
- The shift offset $\alpha$ is experimentally set to 3 or 7 frames.

## Key Experimental Results

### Main Results

Comparison on the HDTF dataset (diffusion model-based methods):

| Method | FID↓ | FVD↓ | Sync-C↑ | Sync-D↓ | E-FID↓ | Smooth↑ | Runtime(s)↓ |
|------|------|------|---------|---------|--------|---------|-------------|
| Hallo | 30.18 | 347.36 | 4.06 | 9.55 | 1.79 | 0.9941 | 74.65 |
| Hallo2 | 38.67 | 328.54 | 4.14 | 9.47 | 2.20 | 0.9942 | 45.75 |
| EchoMimic | 33.21 | 384.30 | 2.51 | 10.74 | 1.49 | 0.9934 | 5.45 |
| **Sonic** | **23.45** | **276.32** | **5.12** | **8.89** | **1.34** | **0.9968** | **3.75** |

### Ablation Study

Contributions of each module (HDTF dataset):

| Configuration | FVD↓ | Sync-C↑ | Smooth↑ |
|------|------|---------|---------|
| Baseline (w/o all modules) | 435.2 | 2.83 | 0.9921 |
| + Context-enhanced Audio Learning | 342.1 | 4.56 | 0.9948 |
| + Motion-decoupled Controller | 318.5 | 4.72 | 0.9952 |
| + Time-aware Position Shift Fusion | **276.3** | **5.12** | **0.9968** |

### Key Findings

1. Sonic comprehensively outperforms state-of-the-art (SOTA) methods across all metrics, reducing FID by 22% and improving Sync-C by 25%.
2. Time-aware position shift fusion significantly enhances temporal smoothness ($0.9952 \rightarrow 0.9968$) without introducing inference overhead.
3. The purely audio-driven paradigm (without motion frames) unexpectedly outperforms methods utilizing motion frames, validating the effectiveness of global audio perception.
4. The inference speed is extremely fast (3.75s), far lower than Hallo (74.65s) and AniPortrait (44.03s).
5. The performance is stable for shift offset $\alpha$ in the range of 3 to 7.

## Highlights & Insights

- Paradigm Shift: Abandoning visual auxiliary signals like motion frames and returning to the essence of being "audio-driven" yielded superior results.
- The time-aware position shift fusion strategy is exceptionally clever: it achieves global temporal consistency solely by changing the denoising starting position, without adding training or inference overhead.
- The motion-decoupled design enhances practicality and controllability, allowing users to adjust the animation style via a simple scaling factor.
- Inference is highly efficient, supporting real-time or parallel processing of long videos.

## Limitations & Future Work

- The generalization capability to extreme audio types (e.g., singing, non-verbal sounds) remains to be validated.
- Motion prediction is still based on learned statistical patterns and may lack individual specificity.
- A single reference image limits the robustness of identity preservation; occlusions or extreme angles may lead to artifacts.
- Future work can extend this to full-body animation and multi-person conversation scenarios.
- Integration with LLMs could enable smarter, emotion-driven expression generation.

## Related Work & Insights

- **EMO / Hallo / Loopy**: Methods that rely on motion frames to maintain temporal consistency; Sonic demonstrates that global audio perception can replace motion frames.
- **SadTalker / AniPortrait**: Methods using 3D coefficients as intermediate representations; Sonic's direct end-to-end generation bypasses the precision limitations of such intermediate representations.
- **Stable Video Diffusion**: Provides powerful video priors, upon which Sonic designs its audio-driven modules.
- Insight: In tasks driven by weak cross-modality signals (such as audio-visual), expanding the receptive field of the signals (from local to global) is more effective than adding auxiliary visual conditions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The global audio perception paradigm and position shift fusion strategy are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Fully evaluated on both HDTF and CelebV-HQ benchmarks with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivations are clearly articulated and the algorithm description is precise.
- **Value**: ⭐⭐⭐⭐⭐ — Strong practicality; the paradigm shift serves as a guiding light for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2026\] DeX-Portrait: Disentangled and Expressive Portrait Animation via Explicit and Latent Motion Representations](../../CVPR2026/human_understanding/dex-portrait_disentangled_and_expressive_portrait_animation_via_explicit_and_lat.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)

</div>

<!-- RELATED:END -->
