---
title: >-
  [Paper Note] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text
description: >-
  [CVPR 2026][Audio & Speech][Video-to-Audio Generation] This paper proposes the Universal Holistic Audio Generation (UniHAGen) task and the OmniSonic framework…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "Video-to-Audio Generation"
  - "Holistic Audio"
  - "Diffusion Models"
  - "Speech Synthesis"
  - "Mixture of Experts"
date: 2026-05-08
content_hash: fb18aa1ad9821017
---

# OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text

**Conference**: CVPR 2026
**arXiv**: [2604.04348](https://arxiv.org/abs/2604.04348)  
**Code**: [https://weiguopian.github.io/OmniSonic_webpage/](https://weiguopian.github.io/OmniSonic_webpage/)  
**Area**: Audio Generation / Multimodal
**Keywords**: Video-to-Audio Generation, Holistic Audio, Diffusion Models, Speech Synthesis, Mixture of Experts

## TL;DR

This paper proposes the Universal Holistic Audio Generation (UniHAGen) task and the OmniSonic framework, which employs a TriAttn-DiT architecture with triple cross-attention and MoE gating to simultaneously generate on-screen environmental sound, off-screen environmental sound, and human speech within a unified audio synthesis pipeline, achieving comprehensive state-of-the-art performance on the newly constructed UniHAGen-Bench.

## Background & Motivation

1. **Background**: Diffusion models have achieved notable progress in audio generation. V2A (Video-to-Audio) methods such as Diff-Foley and MMAudio have continuously improved audio quality and semantic alignment. Joint text-video-to-audio (VT2A) methods such as VinTAGe have begun to consider both on-screen and off-screen sounds simultaneously.

2. **Limitations of Prior Work**: (1) V2A models can only generate sounds corresponding to visible events in the video frame, neglecting off-screen auditory events; (2) VT2A models, while considering both on-screen and off-screen sounds, are limited to environmental sounds and cannot generate human speech; (3) ambient speech generation models (e.g., VoiceLDM) rely solely on text input and lack visual grounding.

3. **Key Challenge**: Real-world auditory scenes are complex—a speaking person in the foreground may be accompanied by birdsong or machinery in the background. No existing model can handle the full combinatorial space of "environmental sound + speech + on/off-screen" within a unified framework.

4. **Goal**: Define a new task, UniHAGen, requiring a model to simultaneously generate a mixture of three sound sources: on-screen environmental sound, off-screen environmental sound, and human speech.

5. **Key Insight**: Decompose the problem into three conditioning streams (on-screen environmental description, off-screen environmental description, and speech transcription), design dedicated triple cross-attention mechanisms to process each independently, and dynamically fuse them via MoE gating.

6. **Core Idea**: Employ TriAttn-DiT triple cross-attention to process on-screen environmental sound, off-screen environmental sound, and speech conditions separately, and use MoE gating to adaptively balance the contribution of each stream, enabling holistic audio generation.

## Method

### Overall Architecture

OmniSonic operates within a Flow Matching diffusion framework, performing denoising in the latent space of an audio VAE. The input conditions consist of four components: video frames (CLIP visual encoder), on-screen environmental sound description (FLAN-T5), off-screen environmental sound description (FLAN-T5), and speech transcription (SpeechT5 + Durator). The core module is TriAttn-DiT, which stacks multiple blocks to predict the velocity field in the audio latent space. During inference, an ODE solver generates the audio latent representation from noise, which is then decoded by the VAE decoder and HiFi-GAN vocoder to recover the waveform.

### Key Designs

1. **TriAttn-DiT Triple Cross-Attention**:

    - Function: Processes the interaction between each of the three conditioning signals (on-screen environment, off-screen environment, speech) and the audio latent representation independently.
    - Mechanism: Visual features $\mathbf{c}_v$ are selectively concatenated with the corresponding condition depending on whether the on-screen environmental description is empty—if non-empty, visual features are concatenated with the on-screen condition; otherwise, they are concatenated with the speech condition. Three independent cross-attention operations are performed: $\mathbf{x}_t^{on} = \text{CA}_{env}(\text{RoPE}(\mathbf{x}_t), \text{RoPE}(\mathbf{c}^{on}_{txt,v}[L_{on}:,:]), \mathbf{c}^{on}_{txt,v})$, with analogous treatment for off-screen and speech streams. RoPE is applied only to visual tokens to encode temporal positional information.
    - Design Motivation: Environmental sounds and speech exhibit vastly different acoustic characteristics; sharing attention layers leads to mutual interference. Separate processing allows each stream to focus on its own semantic alignment.

2. **MoE Gating Fusion Mechanism**:

    - Function: Adaptively balances the contribution weights of the three cross-attention outputs.
    - Mechanism: A representative token is obtained by averaging along the sequence dimension for each of the three conditioning embeddings; these are concatenated and passed through an MLP + Softmax to produce three normalized weights $[\omega^{sp}, \omega^{on}, \omega^{off}]$. The final velocity prediction is obtained by weighted summation: $\mathbf{v}_t = \omega^{sp}\mathbf{x}_t^{sp} + \omega^{on}\mathbf{x}_t^{on} + \omega^{off}\mathbf{x}_t^{off}$
    - Design Motivation: The relative importance of the three sound sources varies across scenarios (e.g., purely environmental scenes vs. speech-dominant scenes), and static weights cannot accommodate this variability.

3. **Frame-Aligned Adaptive Layer Normalization**:

    - Function: Enhances temporal alignment between generated audio and video frames.
    - Mechanism: Visual condition $\mathbf{c}_v$ is projected into the same space as the timestep embedding and added to it to form $\mathbf{c}_{vt}$, which is then upsampled via nearest-neighbor interpolation to the audio temporal resolution, producing per-frame adaLN parameters $[\alpha_1, \beta_1, \gamma_1, \alpha_2, \beta_2, \gamma_2]$.
    - Design Motivation: Per-frame modulation ensures precise alignment between audio features and their corresponding video frames, improving temporal synchronization.

### Loss & Training

The Flow Matching objective is used: $\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}[\|\mathcal{V}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2]$

Training data are synthesized from VGGSound (~195K environmental sound clips), LRS3 (~33K speech videos), and CommonVoice (~1.67M speech recordings), mixed at random SNR levels. FLAN-T5 and the CLIP visual encoder are frozen; SpeechT5 and Durator are trainable.

## Key Experimental Results

### Main Results

Objective evaluation on UniHAGen-Bench (1,003 samples, 3 scenario types):

| Method | FAD↓ | MKL↓ | Mean(AT+AV)↑ | WER↓ | DeSync↓ |
|------|------|------|-------------|------|---------|
| VoiceLDM | 3.58 | 5.74 | 14.03 | 0.15 | 1.25 |
| MMAudio | 5.82 | 5.60 | 17.25 | 1.50 | **0.51** |
| HunyuanVideo-Foley | 6.00 | 5.88 | 16.95 | 1.36 | 0.38 |
| **OmniSonic** | **3.07** | **2.79** | **18.54** | **0.14** | 0.72 |

Subjective MOS evaluation:

| Method | MOS-Q↑ | MOS-EF↑ | MOS-SF↑ | MOS-T↑ |
|------|--------|---------|---------|--------|
| VoiceLDM | 3.13 | 3.40 | 4.05 | 2.54 |
| MMAudio | 3.74 | 3.24 | 1.15 | 3.71 |
| **OmniSonic** | **4.35** | **4.42** | **4.74** | **4.29** |

### Ablation Study

| Configuration | FAD↓ | Mean↑ | WER↓ | DeSync↓ |
|------|------|-------|------|---------|
| OmniSonic (full) | 3.07 | 18.54 | 0.14 | 0.72 |
| w/o MoE Gating | 6.12 | 15.94 | 0.56 | 1.23 |

### Key Findings

- Removing MoE gating doubles FAD from 3.07 to 6.12 and increases WER from 0.14 to 0.56 (4×), demonstrating that the gating mechanism is critical for multi-source balance.
- OmniSonic lags behind MMAudio and HunyuanVideo-Foley on DeSync, as the latter two utilize temporally fine-grained visual features from Synchformer, whereas OmniSonic relies solely on CLIP features.
- MMAudio and HunyuanVideo-Foley achieve MOS-SF (speech fidelity) scores of only 1.15 and 1.17, respectively, indicating an almost complete inability to generate speech; VoiceLDM produces good speech but poor environmental sound (MOS-EF 3.40).
- Qualitative analysis with manual suppression of individual MoE branches shows that suppressing the speech branch eliminates speech generation and suppressing the environmental branch removes background sound, validating the functional specialization of each branch.

## Highlights & Insights

- **Forward-looking task definition**: UniHAGen defines three scenario types spanning "on/off-screen × environmental sound/speech," becoming the first framework to incorporate speech into holistic audio generation and filling an important gap in the field.
- **Elegant TriAttn-DiT design**: The architecture of three independent attention streams combined with a shared MoE gating mechanism ensures both independent processing of each sound source condition and dynamic fusion, avoiding inter-condition interference.
- **Dynamic visual-condition binding**: The decision to bind visual features with environmental descriptions or speech transcription based on whether the on-screen description is empty is a concise and effective design that distinguishes "sound-event-producing objects" from "speaking persons" in the scene.
- The paradigm of multi-stream attention combined with MoE gating is transferable to other generative tasks requiring the handling of multiple heterogeneous conditions.

## Limitations & Future Work

- Temporal synchronization (DeSync) is inferior to methods using Synchformer; introducing finer-grained temporal visual features may improve this.
- Training data consist of synthetic mixtures; the spatial distribution and reverberation characteristics of sound sources in real-world scenes are not modeled.
- Only 10-second audio generation is supported; coherence over longer durations has not been validated.
- Speech quality is strong but has not been rigorously compared against dedicated TTS systems.
- UniHAGen-Bench contains only 1,003 samples, limiting the scale of evaluation.

## Related Work & Insights

- **vs. MMAudio**: MMAudio employs a multimodal DiT to jointly model video and text but targets environmental sounds only; OmniSonic extends this to the speech domain via triple cross-attention while also achieving superior environmental sound quality.
- **vs. VoiceLDM**: VoiceLDM is a purely text-conditioned speech generation system lacking visual grounding; OmniSonic incorporates video conditioning to distinguish between on-screen and off-screen sound sources.
- **vs. VinTAGe**: VinTAGe introduced the concept of "holistic" audio generation but is limited to environmental sounds; OmniSonic achieves true holistic coverage by encompassing both environmental sounds and speech.

## Rating

- Novelty: ⭐⭐⭐⭐ The UniHAGen task definition and TriAttn-DiT architecture are both highly novel; the MoE gating mechanism further enhances the contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive objective and subjective evaluation with ablations validating core components, though the benchmark scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, method descriptions are detailed, and qualitative analysis is thorough.
- Value: ⭐⭐⭐⭐ Fills the gap of unified "environmental sound + speech" generation in the audio generation field, with direct applicability to film and television post-production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ICLR 2026\] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](../../ICLR2026/audio_speech/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)
- [\[ACL 2026\] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](../../ACL2026/audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)

</div>

<!-- RELATED:END -->
