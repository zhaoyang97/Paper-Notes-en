---
title: >-
  [Paper Note] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation
description: >-
  [Image Generation] This paper proposes MAVFlow, a zero-shot audio-visual renderer based on conditional flow matching (CFM), which leverages dual-modal guidance from audio speaker embeddings and visual emotion embeddings to preserve speaker consistency in multilingual AV2AV translation.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: b665e1bf3795fc01
---

# MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation

> **Conference**: ICCV 2025
> **arXiv**: [2503.11026](https://arxiv.org/abs/2503.11026)
> **Code**: [GitHub](https://github.com/Peter-SungwooCho/MAVFlow)
> **Area**: Image Generation
> **Keywords**: audio-visual translation, conditional flow matching, zero-shot, paralinguistic preservation, speaker consistency

## TL;DR

This paper proposes MAVFlow, a zero-shot audio-visual renderer based on conditional flow matching (CFM), which leverages dual-modal guidance from audio speaker embeddings and visual emotion embeddings to preserve speaker consistency in multilingual AV2AV translation.

## Background & Motivation

Audio-visual to audio-visual (AV2AV) translation aims to translate audio-visual content from one language to another while maintaining lip synchronization and speaker identity consistency. Core challenges faced by existing methods:

**Loss of paralinguistic features**: Existing AV2AV methods (e.g., AV2AV, TransFace) primarily focus on translating linguistic content, neglecting paralinguistic features such as speaker identity and emotional expression, resulting in translated speech and facial expressions inconsistent with the original speaker.

**Limitations of unimodal embeddings**: Existing methods employ unimodal embeddings (e.g., d-vectors) independently for audio and visual generation, failing to exploit complementary cross-modal information.

**Lack of advanced conditional generation techniques**: Simple concatenation of speaker embeddings is insufficient to preserve speaker consistency in zero-shot cross-lingual scenarios.

**Core Idea of MAVFlow**: Speaker vocal characteristics and facial information (appearance, emotion) remain consistent across languages. Therefore, **audio speaker embeddings** (global identity) and **visual emotion embeddings** (frame-level dynamics) can serve as guidance signals independent of semantic content, enabling conditional generation that leverages the efficient sampling advantages of OT-CFM.

## Method

### Overall Architecture

MAVFlow comprises four stages (Fig. 2):

1. **AV speech unit translation**: Discrete AV units are extracted using m-AVHuBERT and translated into target-language units via a U2U module.
2. **Duration regulator**: Predicts and expands deduplicated unit durations, interpolating to align with the original audio length.
3. **Multimodal guidance**: Extracts speaker voice embeddings (x-vector) and facial emotion embeddings (EmoFAN).
4. **CFM zero-shot AV renderer**: Integrates linguistic units and paralinguistic guidance to generate mel spectrograms.

### Key Design 1: Multimodal Guidance

**Speaker voice embeddings (global)**: Pretrained speaker encoders are used to extract x-vectors. During training, embeddings are averaged across multiple utterances from the same speaker to obtain speaker-level embeddings:

$$\mathbf{a}_{spk} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{a}_{utt,i}$$

At inference, utterance-level embeddings $\mathbf{a}_{utt,i}$ are used directly to capture fine-grained variation.

**Facial emotion embeddings (frame-level)**: EmoFAN extracts per-frame emotion embeddings $\mathbf{v}_{spk,t} = \text{Emo}(\mathbf{f}_t)$, reflecting dynamically evolving emotional states over time. Key insight: while acoustic features of speech vary across languages (accent, prosody), emotional information remains consistent across languages.

### Key Design 2: OT-CFM Decoder

An optimal transport conditional flow matching (OT-CFM) framework is adopted to learn a conditional vector field from noise to mel spectrograms:

$$\nu_t(\phi_t^{OT}(X_0, X_1) | \theta) = N_\theta(\phi_t^{OT}(X_0, X_1), t; \mathbf{a}_{spk}, \mathbf{v}_{spk,t}, \{\mu_l\}_{1:L}, \tilde{X}_1)$$

where $\phi_t^{OT}(X_0, X_1) = (1-(1-\sigma)t)X_0 + tX_1$ denotes the OT path.

Training objective:

$$\mathcal{L}_{OT-CFM} = \mathbb{E}_{t, p_0, q}\left[\|\omega_t - \nu_t\|^2\right]$$

Multimodal embeddings are integrated as follows: $\mathbf{a}_{spk}$ is added uniformly to all frames (global identity), while $\mathbf{v}_{spk,t}$ is added frame-by-frame (dynamic emotion).

### Duration Regulator

Unlike AV2AV, MAVFlow interpolates the generated audio to align with the length of the original source audio, satisfying the practical constraint of fixed video duration in applications such as film dubbing.

## Key Experimental Results

### Main Results: Zero-Shot Speaker Similarity (Tab. 1, MuAViC Dataset)

| Method | Es-En SS ↑ | Fr-En SS ↑ | It-En SS ↑ | Pt-En SS ↑ |
|--------|-----------|-----------|-----------|-----------|
| 4-Stage Cascade | 0.42 | 0.34 | 0.41 | 0.36 |
| 3-Stage Cascade | 0.42 | 0.35 | 0.41 | 0.35 |
| AV2AV (Direct) | 0.35 | 0.31 | 0.37 | 0.30 |
| **MAVFlow** | **0.49** | **0.51** | **0.53** | **0.48** |

MAVFlow achieves an average speaker similarity improvement of **36%** across all language pairs, with DTW and DTW-SL metrics also ranking best.

### Emotion Evaluation (Tab. 2, CREMA-D Dataset)

| Method | Emo-Acc (%) ↑ | SS ↑ | DTW ↓ |
|--------|-------------|------|-------|
| ASR + YourTTS | 17.52 | 0.40 | 9.02 |
| ASR + XTTS | 28.55 | 0.46 | 11.98 |
| AV2AV | 33.66 | 0.33 | 7.84 |
| **MAVFlow** | **36.46** | 0.39 | **7.30** |

Emotion accuracy improves by +2.8% (vs. AV2AV) and +18.94% (vs. XTTS).

### Translation Quality (Tab. 3, ASR-BLEU)

| Method | Es-En | Fr-En | It-En | Pt-En |
|--------|-------|-------|-------|-------|
| AV2AV | 26.57 | 31.27 | 23.24 | 24.51 |
| **MAVFlow** | **26.97** | **31.33** | **23.43** | **24.97** |

Key finding: dual-modal guidance preserves speaker consistency **without degrading translation quality**, with slight improvements across the board.

### Human Evaluation (Tab. 4, MOS Scores)

| Method | Similarity ↑ | Naturalness ↑ |
|--------|------------|---------------|
| 4-Stage Cascade | 2.81 | 3.29 |
| AV2AV | 3.33 | 3.58 |
| **MAVFlow** | **3.71** | **3.80** |

## Highlights & Insights

1. **Dual-modal guidance strategy**: Audio x-vectors (global identity) and visual emotion embeddings (frame-level dynamics) are complementary, elegantly exploiting the cross-modal sharing property of paralinguistic information.
2. **Advantages of OT-CFM**: Compared to diffusion models, CFM requires fewer sampling steps and is more efficient, making it naturally suitable for integrating multimodal conditions.
3. **Duration alignment** is a critical practical requirement for dubbing applications that prior methods have overlooked.
4. A 36% improvement in speaker consistency represents a substantial practical gain.

## Limitations & Future Work

- Facial generation relies on Wav2Lip; output quality is constrained by the pretrained TFG model.
- Emotion embeddings are derived from the facial emotion recognition model EmoFAN and are therefore subject to its classification accuracy.
- End-to-end system latency and real-time performance are not evaluated.

## Related Work & Insights

- AV2AV translation: AV2AV, TransFace
- Flow matching generation: Voicebox, P-Flow, Matcha-TTS
- Speaker consistency: d-vector, x-vector, ERes2Net

## Rating

- **Novelty**: ★★★★☆ — First application of CFM with dual-modal guidance in AV2AV translation
- **Technical Depth**: ★★★☆☆ — Individual modules are relatively straightforward; innovation lies in the integration strategy
- **Experimental Thoroughness**: ★★★★☆ — Comprehensive multi-dimensional evaluation including human assessment
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated with thorough experimental comparisons

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICML 2025\] FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)

</div>

<!-- RELATED:END -->
