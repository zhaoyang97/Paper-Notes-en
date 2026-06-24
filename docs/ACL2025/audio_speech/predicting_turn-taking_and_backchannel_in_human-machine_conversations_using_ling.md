---
title: >-
  [Paper Note] Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals
description: >-
  [ACL 2025][Audio & Speech][turn-taking prediction] Proposes the first end-to-end framework integrating linguistic, acoustic, and visual tri-modal signals to predict turn-taking and backchannel behaviors in conversations. Introduces MM-F2F, a face-to-face conversational dataset of over 210 hours, improving turn-taking F1 by 10% and backchannel F1 by 33%.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "turn-taking prediction"
  - "backchannel"
  - "multi-modal fusion"
  - "face-to-face conversation"
  - "low-rank fusion"
date: 2026-05-08
content_hash: a6a8369502f3aeae
---

# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals

**Conference**: ACL 2025  
**arXiv**: [2505.12654](https://arxiv.org/abs/2505.12654)  
**Code**: [GitHub](https://github.com/Linyx1125/MM-F2F)  
**Area**: Speech  
**Keywords**: turn-taking prediction, backchannel, multi-modal fusion, face-to-face conversation, low-rank fusion

## TL;DR

Proposes the first end-to-end framework integrating linguistic, acoustic, and visual tri-modal signals to predict turn-taking and backchannel behaviors in conversations. Introduces MM-F2F, a face-to-face conversational dataset of over 210 hours, improving turn-taking F1 by 10% and backchannel F1 by 33%.

## Background & Motivation

**Background**: Full-duplex natural spoken dialogue systems must accurately predict when users finish speaking (turn-taking) and when to provide short feedback (backchannels, e.g., "uh-huh", "I see"). Existing systems mostly rely on voice activity detection (VAD) with fixed thresholds or explicit completion signals to trigger responses.

**Limitations of Prior Work**: 
   - Existing datasets primarily cover text and audio modalities, lacking face-to-face conversational data with visual signals.
   - While the EgoCom dataset contains video, it is from a third-person perspective and has partially occluded eyes, making it unsuitable for face-to-face scenarios.
   - Most datasets neglect backchannel annotation.
   - No unified tri-modal (text + audio + video) prediction framework exists for both turn-taking and backchannel behaviors.

**Key Challenge**: In spontaneous human conversations, a speaker's language, intonation, and facial expressions all convey turn-taking cues, but how to effectively fuse these three modalities and support arbitrary modal combinations as input remains unresolved.

**Goal**: Construct a large-scale tri-modal face-to-face dialogue dataset and design an end-to-end prediction framework capable of supporting arbitrary modal combinations.

**Key Insight**: An automated data collection pipeline + privacy desensitization (anonymization) + a flexible multimodal fusion module based on low-rank decomposition + a random modality dropout training strategy.

**Core Idea**: Utilize low-rank tensor fusion combined with a modality selection scheme to enable turn-taking and backchannel prediction supporting any combination of text, audio, and video inputs.

## Method

### Overall Architecture

Two-stage training:
1. **Stage 1**: Train three unimodal encoders (Linguistic/Acoustic/Visual Encoder) separately, each outputting a 256-dimensional feature $\bm{z_k} \in \mathbb{R}^{256}, k \in \{T, A, V\}$.
2. **Stage 2**: Jointly train all modules end-to-end using the multimodal fusion module $F$.

$$\hat{y} = F(E_T(X_T), E_A(X_A), E_V(X_V))$$

### Key Designs

1. **MM-F2F Dataset Construction Pipeline** (5 stages):

    - **Video Collection & Privacy Desensitization**: Collect ~1000 dialogue videos from the web, replace original faces with synthetic faces (selected based on maximum similarity from a pool of 10K+ synthetic faces), perturb voiceprint features (with 20% standard deviation), and crop to retain only the facial region.
    - **Video Transcription**: Segment audio into sentence-level and word-level frames using WhisperX ASR.
    - **Speaker Diarization**: Extract clip embeddings via a ResNet encoder and apply clustering to distinguish between the two speakers.
    - **Active Speaker Detection**: Train a TalkNet model to determine who is talking in each frame.
    - **Annotation**: Label the final word as TURN, label tokens matching a specific vocabulary as BACKCHANNEL, and label others as KEEP.
    - **Final Dataset**: 773 videos, 210+ hours, ~20M frames, 1.5M+ words, 51K turn-taking instances, and 22K backchannel instances.

2. **Flexible Multimodal Fusion Module**: Based on Low-rank Multimodal Fusion (LMF), the weight tensor is decomposed using $r$ low-rank factors:

$$\bm{W} = \sum_{i=1}^{r} \bigotimes_k^K \bm{w_k^{(i)}}$$

$$\bm{h} = \Lambda_k^K \left[\sum_{i=1}^{r} \bm{w_k^{(i)}} \cdot \bm{z_k}\right]$$

3. **Modality Selection Scheme**: Introduces an indicator function $I_k(\bm{x})$. When a modality is missing, it is replaced with an all-ones vector (the identity element for element-wise multiplication), allowing the fusion to degrade naturally into the fusion of the remaining modalities:

$$\bm{h} = I_T(\cdot) \circ I_A(\cdot) \circ I_V(\cdot), \quad I_k(\bm{x}) = \begin{cases} \bm{x} & \text{if modality } k \text{ exists} \\ \bm{1} & \text{otherwise} \end{cases}$$

4. **Random Modality Dropout Training (RMDT)**: During training, a modality is randomly dropped with a small probability to perform fusion with the remaining modalities, enhancing robustness to missing modalities. This enables inference support for all modality combinations from a single training run.

### Loss & Training

- Three-class cross-entropy loss: $L = -\sum_i y^{(i)} \log(\hat{y}^{(i)}), \quad i \in \{\text{Keep}, \text{Turn}, \text{BC}\}$
- Unimodal training stage + end-to-end multimodal training stage.
- Backbone Selection: GPT-2 for text, HuBERT for audio, and VideoMAE for video/facial regions.
- Prediction Head: 3-layer MLP [256, 64, 3], optimized via Adam with a learning rate of $10^{-5}$ for 20 epochs.

## Key Experimental Results

### Backbone Selection Experiments

| Modality | Backbone | Acc | F1-Keep | F1-Turn | F1-BC |
|------|----------|-----|---------|---------|-------|
| Text | BERT | 0.742 | 0.743 | 0.761 | 0.674 |
| Text | **GPT-2** | 0.751 | 0.747 | 0.767 | **0.707** |
| Audio | Wav2Vec2 | 0.730 | 0.715 | 0.726 | 0.779 |
| Audio | **HuBERT** | 0.751 | 0.737 | 0.735 | **0.805** |
| Video | ViT (Single Frame) | 0.473 | 0.535 | 0.470 | 0.271 |
| Video | VideoMAE (Full Frame) | 0.533 | 0.516 | 0.523 | 0.482 |
| Video | **VideoMAE (Face)** | 0.559 | 0.597 | 0.536 | **0.513** |

### Main Results - Modality Ablation

| Modality | Acc | F1-Keep | F1-Turn | F1-BC |
|------|-----|---------|---------|-------|
| Text | 0.751 | 0.747 | 0.767 | 0.707 |
| Audio | 0.751 | 0.737 | 0.735 | 0.805 |
| Video | 0.559 | 0.597 | 0.536 | 0.513 |
| Text+Audio | 0.811 | 0.783 | 0.809 | 0.894 |
| Text+Video | 0.757 | 0.751 | 0.766 | 0.743 |
| Audio+Video | 0.742 | 0.742 | 0.770 | 0.829 |
| **Text+Audio+Video** | **0.823** | **0.806** | **0.811** | **0.906** |

### Fusion Strategy Comparison

| Fusion Method | Acc | F1-Keep | F1-Turn | F1-BC |
|----------|-----|---------|---------|-------|
| Concatenate | 0.771 | 0.764 | 0.774 | 0.784 |
| GMF | 0.807 | 0.791 | 0.795 | 0.889 |
| **Ours (LMF+Selection)** | **0.823** | **0.806** | **0.811** | **0.906** |

### Comparison with SOTA

| Method | Modality | Acc | F1-Keep | F1-Turn | F1-BC |
|------|------|-----|---------|---------|-------|
| TurnGPT | T | 0.645 | 0.745 | 0.420 | - |
| Wang et al. | T+A | 0.737 | 0.742 | 0.739 | 0.680 |
| Kurata et al. | T+A+V | 0.720 | 0.729 | 0.728 | 0.667 |
| **Ours** | T+A+V | **0.823** | **0.806** | **0.811** | **0.906** |

### Effectiveness of RMDT

| Inference Modality | w/o RMDT (Acc / F1-BC) | w/ RMDT (Acc / F1-BC) |
|----------|---------------------|---------------------|
| T+A | 0.552 / 0.017 | **0.816 / 0.896** |
| T+V | 0.423 / 0.005 | **0.760 / 0.747** |
| A+V | 0.433 / 0.041 | **0.765 / 0.845** |

### Key Findings

- Tri-modal fusion outperforms any unimodal or bi-modal baseline across all metrics, reaching 0.906 in backchannel prediction F1-score.
- The acoustic modality contributes the most to backchannel prediction (F1 0.805), likely related to pitch or rhythmic discontinuities.
- Visual signals significantly complement backchannel prediction (0.513 individually, improving the baseline from T+A 0.894 to T+A+V 0.906).
- Focusing exclusively on the facial region yields better results than extracting features from full frames, as background information introduces noise.
- RMDT is crucial: without RMDT, a model trained on tri-modal inputs suffers catastrophic performance drops when evaluated in a bi-modal setting (with F1-BC collapsing from 0.906 to 0.017).
- Compared to SOTA methods, turn-taking F1 improves by approximately 10%, and backchannel F1 increases by over 33%.

## Highlights & Insights

- **Practical and Reproducible Dataset Pipeline**: Automated collection, privacy protection, and annotation from web videos, minimizing manual intervention.
- **Thoughtful Privacy Design**: Employs face synthesis, voice perturbation, and facial cropping, with experiments demonstrating no side effects on dialogue action comprehension.
- **Simple yet Effective RMDT Training**: A single training run covers all modality combinations, leveraging the mathematical elegance of low-rank decomposition in the modality selection scheme.
- **Remarkable Backchannel Prediction**: An F1-score of 0.906 shows that multimodal signals successfully capture subtle feedback cues in human conversations.

## Limitations & Future Work

1. When the speaker's semantics are incomplete during a thinking pause, the model may mistakenly predict a turn-taking action (as seen in Fig. 5’s failure case).
2. The visual modality currently utilizes only facial information, neglecting gestures and body movements.
3. The dataset is built around English conversations, leaving cross-lingual/cross-cultural generalization unverified.
4. VideoMAE processing of 16 frames poses a high computational cost; real-time performance requires further validation.
5. Multi-party dialogues are not considered (the dataset is restricted to dyadic conversations).

## Related Work & Insights

- The VAP model (Ekstedt and Skantze, 2022b) utilizes acoustic signals to predict keep/turn/backchannel; this work extends this paradigm to three modalities.
- The successful experience of LMF (Low-rank Multimodal Fusion, Liu et al.) in multimodal sentiment analysis is transferred here to conversation prediction.
- Provide direct inspiration for building real-time interaction capabilities in full-duplex dialogue systems (such as GPT-4o-level models).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First tri-modal framework and dataset for turn-taking + backchannel prediction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive testing on backbone choices, modality ablation, fusion strategies, RMDT, and comparisons with SOTA.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured and detailed description of the dataset.
- **Value**: ⭐⭐⭐⭐⭐ — Open-source dataset + code, major contribution to full-duplex human-machine interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [\[NeurIPS 2025\] Resounding Acoustic Fields with Reciprocity](../../NeurIPS2025/audio_speech/resounding_acoustic_fields_with_reciprocity.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2025\] Acoustic Individual Identification of White-Faced Capuchin Monkeys Using Joint Multi-Species Embeddings](acoustic_individual_identification_of_white-faced_capuchin_monkeys_using_joint_m.md)
- [\[ACL 2025\] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)

</div>

<!-- RELATED:END -->
