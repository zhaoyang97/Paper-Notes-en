---
title: >-
  [Paper Note] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments
description: >-
  [NeurIPS 2025][Audio & Speech][Target speaker extraction] This paper proposes a novel enrollment strategy that encodes target speaker characteristics by contrasting noisy positive enrollments (segments where the target speaker is active) against negative enrollments (segments where the target speaker is silent), achieving state-of-the-art performance on monaural noisy-enrollment target speaker extraction with SI-SNRi exceeding the previous best method by over 2.1 dB.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Target speaker extraction
  - noisy enrollment
  - positive-negative contrastive
  - TF-GridNet
  - two-stage training
date: 2026-05-08
content_hash: 81f5bfd0fd12987d
---

# Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments

**Conference**: NeurIPS 2025
**arXiv**: [2502.16611](https://arxiv.org/abs/2502.16611)
**Code**: [Available](https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll)
**Area**: Audio & Speech
**Keywords**: Target speaker extraction, noisy enrollment, positive-negative contrastive, TF-GridNet, two-stage training

## TL;DR

This paper proposes a novel enrollment strategy that encodes target speaker characteristics by contrasting noisy positive enrollments (segments where the target speaker is active) against negative enrollments (segments where the target speaker is silent), achieving state-of-the-art performance on monaural noisy-enrollment target speaker extraction with SI-SNRi exceeding the previous best method by over 2.1 dB.

## Background & Motivation

Target speaker extraction (TSE) aims to isolate a specific speaker's voice from multi-speaker mixed audio. Existing methods primarily rely on **clean audio samples** as conditioning inputs, which is impractical in real-world scenarios — for instance, at a cocktail party, a user cannot ask a stranger to step away from the noisy environment to record a clean reference.

Prior works attempting noisy enrollment have notable limitations:
- **ADEnet**: Evaluated only on two-speaker mixtures with 38.5% overlap, where enrollment segments are largely clean.
- **TCE**: Requires the user to be a participant in the conversation and does not support extraction from arbitrary mixed audio.
- **LookOnceToHear**: Relies on binaural spatial information (target speaker at 90° azimuth), restricting applicable scenarios.

**Core observation**: In natural conversations, different speakers rarely start and stop speaking in perfect synchrony. This temporal asynchrony can be exploited — by comparing segments where the target speaker is active against segments where the target speaker is silent to perform disambiguation.

## Method

### Overall Architecture

The system comprises two branches: (1) **Encoding branch**: a siamese encoder extracts embeddings from positive/negative enrollment audio, which are compared via an encoder fusion module to derive the target speaker representation; (2) **Extraction branch**: conditioned on the output of the encoding branch, a TF-GridNet backbone extracts the target speaker's voice from the mixed audio. TF-GridNet serves as the backbone for both branches.

### Key Designs

1. **Positive/Negative Enrollment Strategy**: The enrollment input is constructed as a pair consisting of a positive enrollment $a^P$ (target speaker is active) and a negative enrollment $a^N$ (target speaker is silent). The signal model is defined as:

    - $a^M = \sum_{i \in S^{IM} \cup \{t\}} a_i^M + n^M$ (mixed audio)
    - $a^P = \sum_{i \in S^{IE} \cup \{t\}} a_i^P + n^P$ (positive enrollment)
    - $a^N = \sum_{i \in S^{IE}} a_i^N + n^N$ (negative enrollment)

   The target speaker is the **only individual who consistently speaks in all positive enrollments but is absent from the negative enrollments**. Interfering speakers are categorized into four types: negative interferers (NI, present in both positive and negative), positive interferers (PI, present only in portions of the positive enrollment), hybrid interferers (HI), and non-relevant interferers (NRI).

2. **Encoder Fusion Module**: The siamese encoder with shared parameters encodes the positive and negative enrollments separately to obtain $E_{pos} \in \mathbb{R}^{T_{pos} \times D}$ and $E_{neg} \in \mathbb{R}^{T_{neg} \times D}$. The fusion procedure is as follows:

    - Learnable segment embeddings $S_{pos}$ and $S_{neg}$ are added to distinguish the source
    - Concatenation along the time dimension: $E_{concat} = [E_{pos}, E_{neg}]$
    - Processing through two layers of full-band self-attention
    - Truncation of the first $T_{pos}$ frames as output

   **Design Motivation**: The self-attention mechanism naturally enables comparisons among positive enrollment frames (identifying silent segments of positive interferers) and between positive and negative enrollment frames (identifying negative interferers), thereby achieving disambiguation.

3. **Extraction Fusion Module**: The encoder output is downsampled via non-overlapping average pooling (kernel=40) and used as Key/Value in cross-attention, while the output of TF-GridNet in the extraction branch serves as the Query. One fusion module is inserted after each of the first two causal TF-GridNet blocks.

### Loss & Training

**Two-Stage Training Strategy** (one of the core innovations):

- **Stage 1**: Only the siamese encoder and fusion module are trained. Knowledge distillation is applied to align the fused representation with that of a clean encoder:
  $$L_{\text{stage 1}} = \|E_{\text{clean}} - E_{\text{fused}}\|^2$$

- **Stage 2**: The extraction branch is trained using negative SNR as the loss:
  $$L_{\text{stage 2}} = -\text{SNR}(\hat{a}_{tgt}, a_{tgt})$$

**Motivation**: The high variability introduced by noisy enrollments causes extremely slow convergence in end-to-end training (requiring ~600k steps to reach 3 dB SNR). The two-stage strategy achieves this in only 240k steps, reducing optimization steps by 60%.

## Key Experimental Results

### Main Results

**Monaural TSE: 2-speaker mixed audio, 2-speaker enrollment**

| Method | SNRi (dB) | SI-SNRi (dB) | PESQ | STOI | WER |
|--------|-----------|--------------|------|------|-----|
| NMF | 4.24±1.60 | -1.65±3.78 | 1.05 | 0.362 | 0.98 |
| USEF-TFGridnet | 3.42±3.43 | -0.03±5.97 | 1.52 | 0.430 | 0.66 |
| TCE | 8.48±2.34 | 6.67±3.69 | 1.91 | 0.682 | 0.73 |
| **Ours (Monaural)** | **10.14±2.57** | **8.85±3.67** | **2.07** | **0.758** | **0.42** |

### Ablation Study

| Configuration | Steps to Reach 3 dB SNR | Final Performance |
|---------------|------------------------|-------------------|
| End-to-end training | ~600k steps (~125h) | Inferior |
| **Two-stage training** | **~240k steps (~50h)** | **Better** |
| FiLM fusion (replacing cross-attention) | — | Worse across all scenarios |
| **Cross-attention fusion** | — | **Consistently superior** |

### Key Findings

- Compared to TCE, monaural SI-SNRi improves by 2.1+ dB (6.67→8.85) and WER drops from 0.73 to 0.42.
- The model maintains robustness across diverse conditions including 2–4 speakers in enrollment and 2–3 speakers in the mixture.
- Two-stage training reduces the time to reach 3 dB SNR from 125 hours to 50 hours (a 60% reduction).
- Cross-attention fusion consistently outperforms FiLM fusion, as the latter is constrained by fixed embedding dimensions that limit fine-grained encoding capacity.
- The binaural variant outperforms the LookOnceToHear baseline on SNRi/SI-SNRi, though STOI is slightly lower, possibly due to differences in model capacity.

## Highlights & Insights

- **Highly practical enrollment strategy**: Users simply press a button on their phone to indicate whether the target speaker is or is not speaking, with no precise annotation required.
- The **disambiguation assumption based on natural conversational randomness** is both elegant and well-motivated — different speakers are unlikely to start and stop speaking in perfect synchrony.
- The use of **knowledge distillation** in the two-stage training effectively transfers knowledge from a clean encoder to the noisy encoder, decoupling two distinct learning challenges.
- Sharing parameters in the siamese encoder reduces model size while maintaining a consistent feature space.

## Limitations & Future Work

- The approach assumes users can roughly identify when the target speaker is or is not active, which may be difficult in highly noisy or fast-alternating multi-party scenarios.
- Performance degrades when the overlap rate of positive interferers is high (discussed in the paper, though not in depth).
- Training and evaluation are conducted exclusively on synthetic data (LibriSpeech + WHAM!); performance on real-world recordings remains to be validated.
- The TF-GridNet backbone incurs substantial computational cost; exploration of lightweight variants would be valuable.

## Related Work & Insights

- **Relation to LookOnceToHear**: Both address noisy enrollment, but the proposed method does not rely on spatial information, broadening its applicability to monaural settings.
- **Relation to TCE**: TCE requires user participation in the conversation and uses the user's own clean d-vector, imposing stricter constraints.
- **Inspiration**: The positive-negative contrastive enrollment concept may be applicable to related tasks such as video speaker separation and sound source localization.

## Rating

- Novelty: ⭐⭐⭐⭐ (The positive-negative enrollment strategy is novel and practical, though the overall architecture builds upon existing TF-GridNet)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive multi-scenario comparisons and clear ablations, but lacks evaluation on real-world data)
- Writing Quality: ⭐⭐⭐⭐ (Problem formulation is clear and method motivation is well articulated)
- Value: ⭐⭐⭐⭐ (Addresses a critical practical pain point in real-world deployment)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](../../ACL2026/audio_speech/speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

<!-- RELATED:END -->
