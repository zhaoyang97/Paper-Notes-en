---
title: >-
  [Paper Note] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization
description: >-
  [NeurIPS 2025][Audio & Speech][Sound Source Localization] This work systematically reveals that AI SSL models suffer from severe visual bias—degrading to near-random performance under audio-visual conflict—and proposes E…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Sound Source Localization"
  - "Modality Bias"
  - "Cross-modal Conflict"
  - "Neuroscience-Inspired"
  - "HRTF"
  - "Cochleagram"
date: 2026-05-08
content_hash: 74b70e4987c33d32
---

# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.11217](https://arxiv.org/abs/2505.11217)  
**Code**: [GitHub](https://github.com/) (publicly available as declared in the paper)  
**Area**: audio_speech / multimodal
**Keywords**: Sound Source Localization, Modality Bias, Cross-modal Conflict, Neuroscience-Inspired, HRTF, Cochleagram

## TL;DR
This work systematically reveals that AI SSL models suffer from severe visual bias—degrading to near-random performance under audio-visual conflict—and proposes EchoPin, a neuroscience-inspired model (HRTF filtering + cochleagram + stereo audio) that substantially outperforms prior methods on AudioCOCO and exhibits a human-like horizontal-over-vertical localization accuracy asymmetry.

## Background & Motivation

1. **Background**: SSL is a fundamental multimodal perception task that associates sounds with their spatial origins in visual scenes. Existing multimodal models (contrastive learning, cross-modal attention, etc.) perform well under standard congruent conditions.
2. **Core Problem**: No prior work has systematically examined AI model behavior under audio-visual conflict—specifically, whether models prioritize audition over vision as humans do, or exhibit visual dominance.
3. **Key Finding**: Humans remain robust under conflict and even audio-only conditions by prioritizing auditory information, whereas AI models are heavily vision-biased and collapse to random-level performance under conflict.

## Method

### Overall Architecture
Three main contributions: (1) AudioCOCO dataset + 6 experimental conditions; (2) human psychophysical experiment baselines; (3) EchoPin neuroscience-inspired model.

### Key Design 1: AudioCOCO Dataset
- 12 sound-producing object categories from MSCOCO, stratified into 3 size tiers by target area ratio (Size1: 0–5%, Size2: 5–15%, Size3: 15–30%)
- Unity 3D simulator + DepthAnything depth estimation: synthesizes **spatially rendered stereo audio** from pixel positions and estimated depth
- Training set: 4,953 images → 9,360 audio-image pairs; test set: 5,500 images → 18,864 pairs
- **6 experimental conditions**: Congruent, ConflictVCue, AbsVCue, AOnly, VOnly, MultiInstLoc

### Key Design 2: EchoPin Model
- **HRTF filtering**: Applies direction-dependent head-related transfer functions from the KEMAR dummy head dataset to simulate direction-specific spectral shaping by the pinna, head, and torso
- **Cochleagram**: An ERB filterbank transforms the HRTF-filtered stereo waveform into a cochleagram (66 channels × 160k time steps × 2 ears), providing a more faithful representation of peripheral auditory processing than mel-spectrograms
- **Dual-encoder architecture**: A 2D CNN dual-stream architecture based on IS3, with independent visual and audio encoders whose outputs are fused for localization
- **Training losses**: Triplet Loss (semantic alignment) + CIoU Loss (spatial alignment)

### Key Design 3: Human Psychophysical Experiment
- 14 participants, 2,100 trials, conducted in a laboratory environment with stereo headphones
- Compared directly against AI models under identical six conditions

## Key Experimental Results

### A-Acc under Congruent Condition (Size2)

| Model | Size1 | Size2 | Size3 |
|-------|-------|-------|-------|
| Random | 1.6% | 9.1% | 19.8% |
| IS3 | 4.8% | 7.9% | 22.4% |
| **EchoPin** | **4.5%** | **24.1%** | **47.1%** |
| Human | 25.7% | 36.4% | 38.6% |

### Key Comparative Findings
- **ConflictVCue**: IS3 degrades to near-random; EchoPin remains significantly above chance
- **AOnly**: Humans can still localize; IS3 essentially fails; EchoPin maintains limited capability
- **VOnly**: AI models achieve above-chance V-Acc without audio → visual bias exposed
- **Mono vs. Stereo**: EchoPin (stereo) outperforms mono by 16.2% on Size3 A-Acc (47.6% vs. 31.4%)
- **Cochleagram vs. Mel**: EchoPin (cochleagram) outperforms EchoPin-S (mel) by 12.4% on Size3

### Human-like Asymmetry
EchoPin exhibits a human-like pattern of horizontal localization accuracy exceeding vertical accuracy, arising from the stronger horizontal-plane spatial cues provided by the binaural stereo + HRTF configuration.

## Highlights & Insights
1. **Systematic modality bias analysis**: First work to quantitatively reveal visual bias in SSL models using six controlled conditions
2. **Neuroscience-driven design**: HRTF + cochleagram combination faithfully emulates the human peripheral auditory system and gives rise to emergent human-like behavior
3. **Human–machine comparison**: Psychophysical experiments provide reliable human baselines
4. **AudioCOCO dataset**: Mitigates shortcut learning problems (e.g., large-object-centered bias) present in existing datasets

## Limitations & Future Work
1. The gap between EchoPin and humans on small targets (Size1) remains large (4.5% vs. 25.7%), indicating insufficient exploitation of weak spatial cues
2. EchoPin is still misled under ConflictVCue conditions—less robust than the human auditory-priority strategy
3. The work is limited to static images + synthesized audio and does not address dynamic sound sources in real-world video

## Related Work & Insights
- **vs. IS3**: IS3 uses mono audio and standard training data, resulting in heavy visual bias; EchoPin addresses this through stereo + HRTF + cochleagram + spatially balanced data
- **vs. CAVP/AVSegformer**: Both exhibit severe visual bias and perform comparably to IS3 under multi-instance conditions
- **vs. ImageBind/LanguageBind**: Large-scale pretrained models also fail to show advantages on SSL tasks

## Implications
- Relevant to the VLM community: multimodal large models may harbor analogous modality biases
- The HRTF + cochleagram auditory frontend is transferable to audio-visual separation, spatial audio synthesis, and robotic auditory perception
- The controlled dataset construction methodology of AudioCOCO is generalizable to other multimodal domains

## Rating
- Novelty: ⭐⭐⭐⭐ Neuroscience-inspired SSL model + systematic modality bias analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Human experiments + multi-model comparison + 6 conditions + ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures
- Value: ⭐⭐⭐⭐ Reveals an important deficiency in multimodal models and provides actionable solutions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](../../ICCV2025/audio_speech/how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [\[NeurIPS 2025\] Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)
- [\[NeurIPS 2025\] Echoes of Humanity: Exploring the Perceived Humanness of AI Music](echoes_of_humanity_exploring_the_perceived_humanness_of_ai_music.md)

</div>

<!-- RELATED:END -->
