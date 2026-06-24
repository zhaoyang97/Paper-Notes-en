---
title: >-
  [Paper Note] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
description: >-
  [Audio & Speech] Proposes DualTalk, the first unified framework for multi-turn dual-speaker interactive 3D talking head generation that models both speaker and listener behaviors, accompanied by a dual-speaker dialogue dataset containing 50 hours and over 1,000 identities.
tags:
  - "Audio & Speech"
date: 2026-05-08
content_hash: 939f59c75270773c
---

# DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations

| Attribute | Value |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2505.18096](https://arxiv.org/abs/2505.18096) |
| Code | [Project Page](https://ziqiaopeng.github.io/dualtalk) |
| Area | Human Understanding / 3D Talking Head Generation |
| Keywords | dual-speaker, talking head, listener modeling, role transition, 3D face animation |

## TL;DR

Proposes DualTalk, the first unified framework for multi-turn dual-speaker interactive 3D talking head generation that models both speaker and listener behaviors, accompanied by a dual-speaker dialogue dataset containing 50 hours and over 1,000 identities.

## Background & Motivation

### Background

3D talking head generation is an active research area in computer vision, widely applied in customer service, telepresence, education, and entertainment. Existing methods either model speaker behavior only (such as FaceFormer, CodeTalker, SelfTalk) or listener behavior only (such as Learning2Listen), with the two developing independently.

### Limitations of Prior Work

1. **Separated Modeling of Speaker and Listener**: In real conversations, people seamlessly switch between speaking and listening, with their expressions dynamically adjusted based on the partner's feedback. Single-role models fail to capture these interaction dynamics.
2. **Limitation of Audio-Only Driving**: Methods like Audio2Photoreal model based solely on audio, lacking visual feedback from the partner's facial expressions, which prevents adaptive adjustments based on the partner's expressions.
3. **Short-Term Response vs. Continuous Conversation**: Listener models typically generate short, isolated responses (a few seconds) and do not support long-term interactions in multi-turn continuous dialogues.
4. **Lack of Dual-Speaker Interaction Datasets**: Existing 3D facial datasets (such as VOCASET, BIWI) do not contain interaction information. While the L2L dataset includes interactions, it does not support multi-turn dialogues.

### Goal

Define a new task, **multi-turn dual-speaker interactive 3D talking head generation**: given the audio of both speakers and the facial motion of Speaker-A, generate the facial motion of Speaker-B across the entire multi-turn dialogue (covering both speaking and listening states).

### Key Insight & Core Idea

Treat conversation participants as unified entities switching between two states (speaking/listening), modeling both behaviors with a single framework; capture the dynamic coupling between the speaker's verbal-visual signals and the listener's feedback through cross-modal temporal enhancement and a dual-speaker interaction module.

## Method

### Overall Architecture

DualTalk consists of four modules: (1) Dual-Speaker Joint Encoder, which separately encodes audio from both sides and the blendshape of Speaker-A; (2) Cross-Modal Temporal Enhancer, utilizing cross-modal attention + BiLSTM to align temporal features; (3) Dual-Speaker Interaction Module, employing a Transformer encoder-decoder to model dual-speaker dynamics; and (4) Expressive Synthesis Module, conducting adaptive expression modulation + blendshape parameter prediction.

### Key Designs 1: Dual-Speaker Joint Encoder

- **Function**: Encodes the audio of both speakers and the facial motion of Speaker-A into a unified feature space.
- **Mechanism**: Two independent Wav2Vec 2.0 encoders handle the audio $\mathbf{A}_A$, $\mathbf{A}_B$ of Speaker-A and Speaker-B respectively, linearly projecting them to a shared dimension $d$. Concurrently, a two-layer fully connected network with ReLU encodes the 56-dimensional blendshape coefficients $\mathbf{M}_A$ of Speaker-A.
- **Design Motivation**: The audio from both speakers provides speech content and prosody information, while Speaker-A's blendshape provides visual feedback. Separate encoding followed by projection to a shared space facilitates subsequent cross-modal fusion.

### Key Designs 2: Cross-Modal Temporal Enhancer

- **Function**: Aligns the temporal features of audio and facial motion to ensure cross-frame consistency.
- **Mechanism**: First, cross-attention ($Q = \mathbf{Z}_A$, $K = V = \mathbf{M}'_A$) modulates the blendshape features based on the audio; then, a BiLSTM captures contextual temporal dependencies; finally, the original audio features $\mathbf{Z}_A$ are concatenated with the temporally enhanced features $\mathbf{T}$.
- **Design Motivation**: Audio and facial motion have different time scales (audio sampling rate of 16kHz vs. facial motion at 30fps), and cross-modal attention achieves alignment. The bidirectional structure of the BiLSTM ensures that the model utilizes both past and future context, which is crucial for natural facial animation.

### Key Designs 3: Dual-Speaker Interaction Module + Expressive Synthesis

- **Function**: Models the dual-speaker interaction dynamics and generates expressive facial animations.
- **Mechanism**:
    - A Transformer Encoder captures long-range dependencies and complex interaction patterns.
    - Modal Alignment Attention (a biased attention inspired by FaceFormer) aligns temporal information.
    - A Transformer Decoder iteratively refines and generates context-rich representations.
    - Adaptive Expression Modulation: $\mathbf{D}' = \mathbf{D} + \alpha \cdot \sigma(\mathbf{D}\mathbf{W}_m + \mathbf{b}_m)$.
    - A final linear layer maps the features to 56-dimensional blendshape parameters.
- **Design Motivation**: The Transformer architecture is well-suited for modeling long-range interaction relationships in long sequences. Adaptive expression modulation introduces the capability to dynamically adjust expression intensity based on context.
- **Loss & Training**: Regression loss based on blendshape parameters (detailed in the paper's experiments).

## Key Experimental Results

### Dataset Comparison

| Dataset | Duration | Identities | Interaction | Multi-Turn |
|--------|------|--------|------|------|
| VOCASET | 0.5h | 12 | ✗ | ✗ |
| L2L | 72h | 6 | ✓ | ✗ |
| **DualTalk** | **50h** | **1000+** | **✓** | **✓** |

The first large-scale 3D facial dataset with both interaction and multi-turn dialogue, averaging 2.5 turns per conversation.

### Main Results (Speaking Performance Frechet Distance ↓)

| Method | FD-EXP | FD-JAW | FD-POSE |
|------|--------|--------|---------|
| FaceFormer | 34.90 | 5.40 | 8.00 |
| CodeTalker | 48.57 | 6.89 | 10.74 |
| SelfTalk | 35.77 | 5.49 | 8.14 |
| L2L | 24.61 | 3.69 | 7.08 |
| **DualTalk** | **11.14** | **1.90** | **3.83** |

DualTalk leads significantly across all expression/jaw/pose metrics, reducing FD-EXP by 55% (compared to L2L).

### Listener Behavior Performance

| Method | SID-EXP ↑ | SID-JAW ↑ | SID-POSE ↑ |
|------|-----------|-----------|------------|
| FaceFormer | 0.54 | 0.36 | 0.50 |
| L2L | 2.86 | 1.89 | 1.19 |
| **DualTalk** | **3.48** | **2.23** | **1.72** |

The listener responses generated by DualTalk are richer and more diverse (higher SID indicates better diversity).

### Key Findings

- Speaking-only models (FaceFormer, CodeTalker) obtain an SID close to 0, indicating that their generated listener responses are almost static.
- DualTalk maintains coherence during role transitions across turns without sudden jumps.
- DualTalk is also consistently optimal under MSE metrics.
- rPCC (Pearson Correlation Coefficient error) shows that the temporal correlation generated by DualTalk is closest to real conversations.

## Highlights & Insights

1. **Pioneering Task Definition**: First to explicitly propose the "multi-turn dual-speaker interaction" task, filling the research gap of separately modeling speaking and listening.
2. **Unified Framework Design**: Eschews separate training of speaking and listening models, handling role transitions with a single model, which aligns better with actual human conversations.
3. **Dataset Scale & Diversity**: Provides 50 hours of data, over 1,000 identities, dual-channel audio, and multi-turn annotations, establishing a solid foundation for future research.
4. **Significant Performance Gain**: Reducing FD-EXP from 24.61 to 11.14 (L2L $\to$ DualTalk) demonstrates the massive benefit of modeling dual-speaker interactions.

## Limitations & Future Work

1. Facial motion is represented solely by blendshape coefficients, with refinement limited by the expressiveness of the 56-dimensional parameters.
2. The dataset originates from specific dialogue scenarios and may not fully cover all emotions and cultural backgrounds.
3. Speaker-A's real facial motion is required as input, making it inapplicable in audio-only scenarios without visual input.
4. Evaluation metrics for multi-turn dialogues are still incomplete, and how to quantify the "naturalness of role transitions" requires further investigation.

## Related Work & Insights

- **FaceFormer** (CVPR 2022): Transformer-based audio-driven talking head; DualTalk's Modal Alignment Attention is inspired by it.
- **Learning2Listen** (CVPR 2022): A pioneer in listener modeling, but only supports single-turn short responses.
- **Audio2Photoreal** (CVPR 2024): Whole-body dialogue generation, yet relies solely on audio without visual feedback.
- **Insight**: The dual-speaker interaction modeling approach can be extended to full-body movements (gestures, body poses) and multi-person (>2 people) conversation scenarios.

## Rating

⭐⭐⭐⭐ — The new task definition is pioneering, the unified framework design is reasonable, and the accompanying dataset possesses long-term research value. The experimental results are highly convincing, although blendshape expressiveness and evaluation metrics can be further refined.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[ACL 2025\] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](../../ACL2025/audio_speech/childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
