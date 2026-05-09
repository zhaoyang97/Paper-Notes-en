---
title: >-
  [Paper Note] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer
description: >-
  [ICLR 2026][Audio & Speech][Video-to-Audio] This paper proposes AC-Foley, a reference-audio-guided video-to-audio synthesis framework that achieves fine-grained timbre control, timbre transfer, and zero-shot sound effect generation via two-stage training (acoustic feature learning + temporal adaptation) and multimodal conditional flow matching, significantly outperforming existing methods in audio quality and acoustic fidelity.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - Video-to-Audio
  - Foley Synthesis
  - Reference Audio Control
  - Timbre Transfer
  - Flow Matching
date: 2026-05-08
content_hash: 3232f72c54b59eee
---

# AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer

**Conference**: ICLR 2026
**arXiv**: [2603.15597](https://arxiv.org/abs/2603.15597)
**Code**: None
**Area**: Audio Generation
**Keywords**: Video-to-Audio, Foley Synthesis, Reference Audio Control, Timbre Transfer, Flow Matching

## TL;DR
This paper proposes AC-Foley, a reference-audio-guided video-to-audio synthesis framework that achieves fine-grained timbre control, timbre transfer, and zero-shot sound effect generation via two-stage training (acoustic feature learning + temporal adaptation) and multimodal conditional flow matching, significantly outperforming existing methods in audio quality and acoustic fidelity.

## Background & Motivation

**Background**: Existing V2A methods primarily synthesize audio from text prompts combined with visual information, achieving audio-visual synchronization at the semantic level.

**Limitations of Prior Work**: (a) Dataset granularity gap — training annotations group acoustically distinct sounds (e.g., barks from different dog breeds) under coarse labels; (b) Limitations of text description — language cannot encode fine-grained acoustic features (e.g., "metal impact" cannot distinguish the time-frequency characteristics of a hammer striking an anvil versus a steel chain falling). These constraints prevent text-based control from achieving fine-grained sound effect synthesis.

**Key Challenge**: Foley artists need to synthesize multiple acoustic variants for the same visual action (e.g., footsteps on surfaces of different materials), yet text cannot precisely describe timbral differences, and training data lacks such fine-grained annotations.

**Goal**: To directly control acoustic characteristics using reference audio, bypassing the semantic ambiguity inherent in text.

**Key Insight**: A VAE encodes the reference audio to preserve its complete acoustic signature (rather than using encoders such as CLAP that extract only semantic information), and two-stage training is employed to learn how to adapt the reference timbre to the temporal structure of the video.

**Core Idea**: Replace text with raw audio signals as the control condition; leverage a VAE to preserve timbral features and employ two-stage training to achieve adaptive transfer from reference audio to video temporal dynamics.

## Method

### Overall Architecture
The inputs are a silent video and a reference audio clip (with an optional text prompt). A multimodal Transformer generates audio that is synchronized with the video and preserves the timbral characteristics of the reference audio, operating under a conditional flow matching framework. The three modalities interact through joint training.

### Key Designs

1. **Multimodal Conditional Flow Matching**:

    - **Function**: Extends conditional flow matching to three-modality conditional generation over video, audio, and text.
    - **Mechanism**: The velocity field $v_\theta(t, \mathcal{C}, x_t)$ is guided by multimodal conditions $\mathcal{C} = \{V, A, T\}$. The condition vector $\mathbf{c}$ integrates CLIP visual/text features, Synchformer synchronization features, VAE audio features, and timestep embeddings, modulating Transformer inputs via adaLN.
    - **Design Motivation**: Flow matching offers faster inference than diffusion models, and joint multimodal training allows different control signals to complement one another.

2. **Audio Control Module**:

    - **Function**: Encodes the reference audio using a VAE to preserve complete acoustic features.
    - **Mechanism**: A pretrained VAE encoder maps the reference audio to the latent space (instead of CLAP), and average pooling is applied to extract acoustic features. CLAP captures only semantic-level audio information, whereas the VAE retains full spectral and timbral characteristics.
    - **Design Motivation**: Since text suffers from insufficient semantic granularity, conditioning directly on audio — which preserves waveform-level acoustic information rather than semantic labels — is a principled solution.

3. **Two-Stage Training Strategy**:

    - **Function**: Learns acoustic feature extraction and temporal adaptation in separate stages.
    - **Mechanism**: Stage 1 (Acoustic Feature Learning) trains on overlapping audio-visual clip pairs to establish the ability to extract acoustic features from the reference audio. Stage 2 (Temporal Adaptation) uses non-overlapping audio segments from different temporal positions within the same video as conditions, exploiting intra-video audio self-similarity (e.g., footsteps in the same scene share acoustic characteristics) to force the model to align reference timbral features with the video's temporal structure.
    - **Design Motivation**: The non-overlapping conditioning in Stage 2 is the key design choice — it compels the model to learn "timbre transfer" rather than "waveform copying," resolving the temporal misalignment and audio-visual incoherence caused by naive duplication.

### Loss & Training
The standard conditional flow matching objective (velocity field regression) is used. Each modality's condition is randomly dropped out during training with a certain probability, enabling flexible condition combinations at inference time.

## Key Experimental Results

### Main Results

| Method | FD↓ | KL↓ | MCD↓ | Timbre Fidelity |
|--------|-----|-----|------|-----------------|
| MMAudio (text only) | Baseline | Baseline | Baseline | No control |
| CondFoley | Moderate | Moderate | Moderate | Limited |
| **AC-Foley (audio conditioned)** | **−20%** | **−28%** | **−22%** | **Precise** |
| AC-Foley (no audio condition) | Competitive | Competitive | Competitive | — |

### Ablation Study

| Configuration | Audio Quality | Timbre Fidelity | Notes |
|---------------|---------------|-----------------|-------|
| Full model | Best | Best | Two-stage training + VAE encoding |
| Stage 1 only | Moderate | Temporal misalignment | Lacks temporal adaptation |
| CLAP instead of VAE | Degraded | Timbral detail lost | CLAP captures only semantics |
| No audio condition | Competitive | — | Degenerates to standard V2A |

### Key Findings
- Providing different reference audio clips (e.g., Chihuahua bark vs. large-breed bark) for the same dog video yields acoustically distinct outputs, validating fine-grained control capability.
- Timbre transfer experiments succeed across categories (e.g., transferring donkey vocalizations to a lion video), demonstrating cross-category acoustic feature transfer.
- Zero-shot generation: using a suppressed gunshot as reference audio with a shooting video produces a silenced effect that text prompts are entirely incapable of describing.
- Even without a reference audio input, AC-Foley remains competitive with state-of-the-art V2A methods, indicating that joint multimodal training itself improves baseline generation capability.

## Highlights & Insights
- **A principled bypass of text**: Rather than improving text descriptions, the method uses audio directly as the control signal — "letting sound describe sound" is fundamentally more effective than "letting words describe sound."
- **Elegant two-stage training design**: Exploiting intra-video audio self-similarity to force the model to learn "transfer" rather than "copy" is an ingenious training strategy.
- **VAE vs. CLAP**: Prior work defaults to CLAP for audio encoding, but CLAP is designed for semantic retrieval. Preserving timbre requires lower-level waveform features, making VAE the principled choice.

## Limitations & Future Work
- Reference audio must be provided by the creator, increasing the barrier to use.
- Generation quality may degrade when the reference audio is semantically inconsistent with the video content.
- Two-stage training adds training complexity.
- Flexibility with respect to reference audio length may be constrained by the VAE's processing capacity.

## Related Work & Insights
- **vs. MMAudio**: MMAudio jointly trains on video and text modalities but does not support audio-conditioned control; AC-Foley extends to three modalities and enables precise timbre control.
- **vs. CondFoley**: CondFoley requires equal-length reference audio-video pairs, limiting flexibility; AC-Foley supports variable-length references.
- **vs. MultiFoley**: MultiFoley performs audio continuation/extension and is constrained by the diversity of input audio; AC-Foley performs timbre transfer applicable across different semantic categories.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of replacing text with reference audio control is intuitive yet highly effective; the two-stage training design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three application scenarios are fully covered: fine-grained control, timbre transfer, and zero-shot generation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; application examples are vivid and intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed fine-grained control tool for Foley production practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[ICLR 2026\] PACE: Pretrained Audio Continual Learning](pace_pretrained_audio_continual_learning.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)

</div>

<!-- RELATED:END -->
