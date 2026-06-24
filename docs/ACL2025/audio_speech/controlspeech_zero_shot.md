---
title: >-
  [Paper Note] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control
description: >-
  [ACL 2025][Audio & Speech][Zero-shot Speech Synthesis] ControlSpeech is the first TTS system to achieve simultaneous and independent zero-shot speaker cloning and zero-shot language style control, addressing the many-to-many style control challenge through decoupled representations in discrete codec spaces and a Style-Mixture Semantic Density (SMSD) module.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Zero-shot Speech Synthesis"
  - "Style-Controllable TTS"
  - "Speaker Cloning"
  - "Discrete Codec Decoupling"
  - "Gaussian Mixture Density Network"
date: 2026-05-08
content_hash: 06369845b8010c54
---

# ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control

**Conference**: ACL 2025  
**arXiv**: [2406.01205](https://arxiv.org/abs/2406.01205)  
**Area**: LLM NLP / Speech Synthesis  
**Keywords**: Zero-shot Speech Synthesis, Style-Controllable TTS, Speaker Cloning, Discrete Codec Decoupling, Gaussian Mixture Density Network

## TL;DR

ControlSpeech is the first TTS system to achieve simultaneous and independent zero-shot speaker cloning and zero-shot language style control, addressing the many-to-many style control challenge through decoupled representations in discrete codec spaces and a Style-Mixture Semantic Density (SMSD) module.

## Background & Motivation

- **Limitations of Prior Zero-shot TTS**: Zero-shot TTS models represented by VALL-E can clone speaker timbres but possess fixed styles, lacking further control or customization.
- **Limitations of Prior Style-Controllable TTS**: Style-controllable models like PromptTTS 2 and InstructTTS can synthesize speech in arbitrary styles but cannot specify the speaker timbre or perform zero-shot speaker cloning.
- **Goal**: Simultaneously and independently control content, timbre, and style (e.g., "speak 'Today is Monday' in a cheerful child style using Donald Trump's voice").
- **Key Challenges**:
  1. Style prompts and speech prompts may be entangled and interfere with each other (e.g., the style in the speech prompt may conflict with the text style description).
  2. The scarcity of large-scale datasets containing both style descriptions and speaker prompts.

## Method

### Overall Architecture

ControlSpeech is fundamentally an encoder-decoder parallel codec generation model consisting of three independent encoders:

1. **Text Encoder**: Converts content text into phonemes and encodes them.
2. **Style Encoder**: Performs word-level encoding on style text using a BERT tokenizer to extract the global [CLS] style representation.
3. **Speech Encoder**: Extracts timbre information using a pre-trained FACodec encoder.

The generation process consists of two stages:
- The first stage generates discrete codec representations via a mask-based parallel decoder (Conformer).
- The second stage fuses timbre embeddings through a conditional normalization layer and feeds them into a pre-trained decoder to generate the final speech.

### Key Designs

#### 1. Codec Decoupling

Speech is decoupled using FACodec (pre-trained on 60,000 hours of data) into:
- **Content Codec Yc**: Semantic content representation.
- **Prosody Codec Yp**: Prosody information.
- **Acoustic Codec Ya**: Acoustic details.
- **Timbre Embedding Yt**: Global timbre vector.

The style codec Ys is obtained by concatenating the prosody and acoustic codecs: Ys = concat(Yp, Ya).

#### 2. Style-Mixture Semantic Density Module (SMSD)

This work discovers and analyzes the **many-to-many problem** in style control for the first time:
- **Many-to-One**: Different textual descriptions can map to the same audio (e.g., "speaking extremely fast" and "very high speech rate" represent the same style).
- **One-to-Many**: A single textual description can map to different intensities of the same style (e.g., "fast" could correspond to speaking rates of 75, 80, or 90).

Key designs of the SMSD module:
- Using a pre-trained BERT to extract global style semantic representations, aligning different descriptions into the same semantic space (addressing many-to-one).
- Modeling the conditional distribution as a mixture of K Gaussian distributions based on **Mixture Density Networks (MDN)**, where different Gaussians correspond to varying degrees of the same style (addressing one-to-many).
- Introducing a **noise perturbation mechanism** to enhance style diversity, supporting four perturbation types (fully factorized, isotropic, cross-cluster isotropic, fixed isotropic), with cross-cluster isotropic performing best in experiments.

#### 3. Confidence-Based Parallel Decoding

Using a mask-based iterative generation approach with a cosine schedule for sampling mask ratios. Discrete acoustic tokens are progressively generated through multiple forward passes, keeping candidate results based on confidence scores.

### Loss & Training

Total Loss: L = L_codec + L_dur + L_SMSD
- L_codec: Cross-entropy loss for codec generation.
- L_dur: Mean squared error loss for duration prediction.
- L_SMSD: Negative log-likelihood of the style mixture distribution.

## Key Experimental Results

### Main Results

**Dataset**: VccmDataset constructed based on TextrolSpeech, featuring fine-grained labels for gender, volume, speed, pitch, and emotion.

**Style Controllability Evaluation (Test Set A, 1500 samples)**:

| Model | Pitch↑ | Speed↑ | Volume↑ | Emotion↑ | WER↓ | MOS-Q↑ |
|------|--------|--------|---------|----------|------|--------|
| GT Codec | 0.954 | 0.885 | 0.977 | 0.758 | 2.6 | 4.25 |
| PromptTTS 2 | 0.867 | 0.785 | 0.825 | 0.406 | 3.1 | 3.83 |
| InstructTTS | 0.849 | 0.761 | 0.822 | 0.412 | 3.0 | 3.81 |
| **ControlSpeech** | **0.833** | **0.829** | **0.894** | **0.557** | **2.9** | **3.91** |

- Reaches optimal performance across volume, speed, and emotion accuracy.
- WER and MOS-Q also outperform all baselines.

**Speaker Cloning Evaluation (Test Set B)**:

| Model | WER↓ | MOS-Q↑ | MOS-S↑ |
|------|------|--------|--------|
| VALL-E | 6.7 | 3.76 | 3.89 |
| MobileSpeech | 4.1 | 3.94 | 4.01 |
| **ControlSpeech** | **3.3** | **3.95** | **3.96** |

**Many-to-Many Style Control Evaluation (Test Set D)**:

| Model | MOS-TS↑ | MOS-SA↑ | MOS-SD↑ |
|------|---------|---------|---------|
| PromptStyle | 3.81 | 3.45 | 3.53 |
| InstructTTS | 3.89 | 3.57 | 3.48 |
| ControlSpeech w/o SMSD | 3.95 | 3.59 | 3.66 |
| **ControlSpeech** | **4.01** | **3.84** | **4.05** |

### Key Findings

1. **Necessity of Decoupling**: Removing decoupling causes Pitch to drop from 0.833 to 0.492, and Speed from 0.829 to 0.517, validating that speech prompts and style prompts indeed interfere with each other.
2. **Effectiveness of SMSD**: Removing SMSD decreases MOS-SA by 0.25 and MOS-SD by 0.39, proving that the SMSD module significantly enhances style accuracy and diversity.
3. Pitch accuracy is slightly lower than some baselines due to the increased difficulty of pitch control when simultaneously handling different speaker timbres and styles.
4. Cross-cluster isotropic noise perturbation achieves the best trade-off between precision and diversity.

## Highlights & Insights

- **First Unified Framework**: Integrates zero-shot speaker cloning and zero-shot style control into a single unified system.
- **Discovery and Resolution of the Many-to-Many Problem**: First to identify and analyze many-to-many relationships in style-controllable TTS, which is fundamentally different from the one-to-many problem in PromptTTS 2.
- **Clever Utilization of Pre-trained Decoupled Space**: By leveraging the decoupled representation space of the large-scale pre-trained FACodec, the model achieves independent control while preserving zero-shot capabilities.
- **Open-source Dataset Contribution**: Releases the VccmDataset, filling the gap of large-scale TTS datasets that contain both style descriptions and speaker prompts.

## Limitations & Future Work

- The field of style-controllable TTS still lacks larger-scale training datasets (e.g., tens of thousands of hours with style descriptions).
- Only the combination of discrete decoupled codecs + non-autoregressive parallel generative models is explored so far; more generative architectures and audio representations can be investigated in the future.
- Pitch control suffers a degradation in accuracy when processing timbre and style simultaneously.

## Related Work & Insights

- **Zero-shot TTS**: VALL-E (autoregressive codec LM), NaturalSpeech 2/3 (continuous/factorized diffusion), VoiceBox (flow matching infilling), MobileSpeech (non-autoregressive parallel generation).
- **Style-Controllable TTS**: PromptTTS/PromptTTS 2 (text prompt style control), InstructTTS (three-stage training to capture style semantics), TextrolSpeech (language model paradigm), AudioBox (unified flow matching model).
- **Key Differences**: Existing zero-shot TTS models cannot control style, and style-controllable TTS models cannot clone speaker timbres. ControlSpeech is the first to achieve both simultaneously.

## Rating

- **Novelty**: ★★★★☆ — Proposes the SMSD module to solve the many-to-many problem, achieving independent control of speaker timbre + style in a unified framework for the first time.
- **Value**: ★★★★☆ — Wide range of application scenarios (audiobooks, customizable virtual assistants) but relies on a specific pre-trained codec.
- **Experimental Thoroughness**: ★★★★☆ — Covers different evaluation dimensions across four test sets, with comprehensive ablation studies, though lacking cross-lingual evaluation.
- **Writing Quality**: ★★★★☆ — Well-defined problem formulation, detailed methods presentation, and rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[ICLR 2026\] FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions](../../ICLR2026/audio_speech/flexivoice_enabling_flexible_style_control_in_zero-shot_tts_with_natural_languag.md)
- [\[ACL 2025\] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)

</div>

<!-- RELATED:END -->
