---
title: >-
  [Paper Note] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models
description: >-
  [AAAI 2026][Video Understanding][Audio-Language Models] This paper proposes TimeAudio, which equips large audio-language models (LALMs) with precise temporal grounding and end-to-end long audio understanding capabilities through three key modules: Temporal Markers, Absolute Time-aware Encoding (ATE), and Segment-level Token Merging (SEM). The paper also introduces the FTAR dataset for instruction fine-tuning on fine-grained temporal reasoning.
tags:
  - AAAI 2026
  - Video Understanding
  - Audio-Language Models
  - Temporal Grounding
  - Fine-Grained Audio Understanding
  - Long Audio
  - Token Merging
date: 2026-05-08
content_hash: 0b3e78a2d5e3c5bb
---

# Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.11039](https://arxiv.org/abs/2511.11039)
**Code**: [github](https://github.com/lysanderism/TimeAudio)
**Area**: Video Understanding / Audio Understanding
**Keywords**: Audio-Language Models, Temporal Grounding, Fine-Grained Audio Understanding, Long Audio, Token Merging

## TL;DR

This paper proposes TimeAudio, which equips large audio-language models (LALMs) with precise temporal grounding and end-to-end long audio understanding capabilities through three key modules: Temporal Markers, Absolute Time-aware Encoding (ATE), and Segment-level Token Merging (SEM). The paper also introduces the FTAR dataset for instruction fine-tuning on fine-grained temporal reasoning.

## Background & Motivation

### State of the Field

LALMs such as Qwen2-Audio and SALMONN have demonstrated strong capabilities in audio content understanding and conversational question answering, handling both speech and environmental audio in a unified manner. These models integrate audio encoders with pretrained decoder-based LLMs to enable free-form audio question answering.

### Limitations of Prior Work

Despite significant progress, current LALMs exhibit serious deficiencies in **fine-grained audio understanding**, manifested across three dimensions:

**Inaccurate timestamp prediction**: Existing models struggle to accurately predict event start and end times. For example, Qwen2-Audio achieves a zero-shot Eb-F1 of only 9.8 on dense audio captioning, indicating extremely weak event localization ability.

**Limited long audio processing**: Most models are constrained to short audio inputs and cannot understand long audio end-to-end, suffering from severe hallucinations when faced with longer content.

**Lack of suitable datasets and evaluation metrics**: Existing instruction tuning data and benchmarks focus primarily on general audio understanding, with insufficient support for fine-grained temporal reasoning.

### Root Cause

Audio understanding inherently requires capturing both **semantic content** and **temporal position** simultaneously. However, when existing LALMs directly map audio features to a shared latent space, they **do not explicitly model fine-grained temporal localization information**, making it difficult for the LLM decoder to predict precise timestamps. This constitutes a dual architectural-data deficiency.

### Starting Point

The paper addresses three dimensions simultaneously — **timestamp representation**, **model architecture**, and **data** — by introducing discrete temporal markers to replace continuous numerical regression, incorporating absolute time-aware encoding to enhance temporal perception, designing a token merging mechanism to support long audio, and constructing the FTAR dataset to strengthen temporal reasoning.

## Method

### Overall Architecture

TimeAudio is built upon the SALMONN architecture and consists of four core components:
1. **Sliding Audio Encoder**: Segments long audio into short clips and uses dual BEATs + Whisper encoders to extract environmental sound and speech features respectively.
2. **Window Q-Former**: Projects encoded audio tokens into the language space.
3. **Segment-level Token Merging Module**: Selects important tokens based on attention scores and compresses redundant information.
4. **LLM**: Processes fused audio and text tokens to generate responses.

### Key Designs

#### 1. Temporal Markers

**Problem**: Directly using numeric tokens to predict timestamps causes convergence difficulties for the LLM; relative time tokens (e.g., \<0.2\> \<0.4\> \<0.6\>...) impose a heavy burden on the vocabulary and introduce quantization errors.

**Core Idea**: A two-level discretization scheme of anchor tokens and offset tokens is designed to convert continuous timestamps into discrete temporal token sequences.

- **Anchor token** $\langle a_k \rangle$: represents coarse-grained positions at the integer-second level.
- **Offset token** $\langle f_k \rangle$: represents fine-grained sub-second adjustments.

The key innovation lies in the **initialization strategy** — rather than randomly initializing these new tokens, knowledge is transferred from embeddings of existing numeric tokens:

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle a_0 \rangle)} = [\mathbf{W}_{\text{token}}]_{\text{ID}(0)}$$

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle f_0 \rangle)} = \frac{[\mathbf{W}_{\text{token}}]_{\text{ID}(0)} + [\mathbf{W}_{\text{token}}]_{\text{ID}(\text{.})}}{2}$$

**Design Motivation**: This leverages the LLM's existing numerical understanding to avoid disrupting the pretrained embedding space through random initialization. Only $M=20$ special tokens are added, maintaining precision independent of audio length.

#### 2. Absolute Time-aware Encoding (ATE)

**Problem**: Understanding the temporal ordering of audio events remains challenging, as the diversity of events and speech prosody makes it difficult for models to accurately identify real temporal positions.

**Core Idea**: A learnable absolute time embedding $\mathbf{W}_t$ is constructed to explicitly inject absolute temporal information into each audio segment's features:

$$\hat{\mathbf{W}}_i = \mathbf{W}_i + \mathbf{e}_t(t_i)$$

where $\mathbf{e}_t(t_i) = [\mathbf{W}_t]_{j_i}$ is obtained via one-hot lookup, and $j_i$ is the discretized time index.

**Design Motivation**:
- Preserves the relative positional information of the original sequence embeddings while precisely reflecting each time point's absolute position within the audio.
- Time embeddings are **zero-initialized** to protect the integrity of the pretrained audio encoder in early training.
- The authors argue that absolute time encoding and positional encoding are **orthogonal**, which is also validated experimentally.

#### 3. Segment-level Token Merging (SEM)

**Problem**: Processing long audio incurs large computational overhead; the Q-Former uses a fixed number of queries for alignment, causing token count to grow linearly with audio length.

**Core Idea**: The attention information from the Q-Former is reused to adaptively select and merge tokens without introducing additional computation. This proceeds in two steps:

**Step 1: Attention Token Selection**
- Compute the multi-head attention matrix from the Q-Former: $\mathbf{A} = \text{Softmax}(\mathbf{QK}^T / \sqrt{D})$
- Average across all heads to obtain a unified score matrix.
- Retain the highest-scoring tokens as "attention tokens" (22 tokens retained in experiments).

**Step 2: Clustering-based Token Merging**
- Partition the remaining tokens equally into target tokens and merge candidates.
- Compute similarity using dot products of key vectors: $\mathbf{Sim}(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{k}_i \mathbf{k}_j^T$
- Assign candidate tokens to their most semantically similar centroids and aggregate them into "context tokens" (4 tokens).

**Design Motivation**: Information is dispersed throughout long audio; by adaptively selecting important tokens and aggregating redundant ones, SEM compresses **75%** of redundant information while preserving core semantics.

### Loss & Training

A **two-stage training** strategy is adopted:

**Stage 1: Temporal Token Alignment**
- Continues pretraining from a trained LALM checkpoint.
- Training objective: align fine-grained audio features with temporal information.
- Data: temporal audio grounding, dense audio captioning, and timeline speech summarization.
- Trainable components: LoRA adapter, Window Q-Former, absolute time embeddings, and special text embeddings.
- Frozen components: audio encoder and LLM.
- 10 epochs, learning rate 1e-5.

**Stage 2: Long Audio Instruction Fine-tuning**
- Addresses the semantic misalignment of the Stage 1 model on long audio.
- Fine-tunes the Window Q-Former and LoRA.
- Small-scale instruction data, 5 epochs, learning rate 2e-6.

### FTAR Dataset

A dataset of **260K** audio-text pairs is constructed across three task types:
- **Dense Audio Captioning** (110K): generates event descriptions with timestamps.
- **Temporal Audio Grounding** (100K): localizes time segments given text queries.
- **Timeline Speech Summarization** (42K): summarizes speech content with timelines.
- **Audio Temporal QA** (15K): question answering on counting, duration, and temporal ordering.

## Key Experimental Results

### Main Results

| Task | Metric | TimeAudio | Qwen2-Audio (FTAR FT) | SALMONN-13B (FTAR FT) | Zero-shot Qwen2-Audio |
|------|--------|-----------|----------------------|----------------------|----------------------|
| Dense Audio Captioning | METEOR | 20.4 | **22.4** | 20.2 | 6.7 |
| Dense Audio Captioning | Eb-F1 | **37.4** | 36.5 | 32.0 | 9.8 |
| Dense Audio Captioning | At-F1 | **70.5** | 67.8 | 67.6 | 50.3 |
| Temporal Audio Grounding | R@0.5 | **75.7** | 72.8 | 69.2 | 32.1 |
| Temporal Audio Grounding | R@0.7 | **61.2** | 55.4 | 53.5 | 18.7 |
| Temporal Audio Grounding | mIoU | **57.8** | 51.7 | 50.3 | 20.5 |
| Speech Summarization | ROUGE-1 | **42.4** | 40.0 | 40.2 | 17.4 |
| Speech Summarization | ROUGE-L | **30.8** | 28.5 | 29.8 | 12.3 |
| Speech Summarization | mIoU | **94.2** | 85.2 | 88.2 | 13.3 |

- The largest improvement is on temporal audio grounding: mIoU 57.8 vs. SALMONN-7B's 51.9 (+11.4%).
- Speech summarization mIoU reaches 94.2, surpassing all baselines and the specialized model Hubert-MiniChat.

### Ablation Study

| Configuration | Eb-F1 | At-F1 | R@0.5 | mIoU (Grounding) | ROUGE-1 | mIoU (Summary) | Notes |
|---------------|-------|-------|-------|-----------------|---------|----------------|-------|
| SALMONN (FTAR FT) | 32.4 | 68.0 | 71.4 | 51.9 | 39.5 | 84.3 | Baseline |
| +TM (no init) | 33.5 | 69.2 | 71.7 | 52.3 | 40.0 | 87.5 | TM effective |
| +TM (transfer init) | 36.0 | 70.8 | 72.7 | 54.9 | 41.0 | 88.7 | Init strategy critical |
| +TM+ATE | 37.8 | 71.4 | 73.7 | 56.0 | 41.4 | 90.4 | ATE stacked |
| +TM+ATE+SEM | 37.4 | 70.5 | 75.7 | 57.8 | 42.4 | **94.2** | Full model |

**Token Retention Ratio Ablation**:

| Retention Ratio | ROUGE-1 | ROUGE-L | mIoU |
|----------------|---------|---------|------|
| 0.10 | 24.3 | 18.4 | 72.9 |
| 0.15 | 31.6 | 20.5 | 73.2 |
| 0.20 | 40.2 | 29.7 | 84.8 |
| **0.25** | **42.4** | **30.8** | **94.2** |
| 0.30 | 42.5 | 31.1 | 94.0 |

A ratio of 0.25 is selected as the optimal balance between performance and efficiency.

### Key Findings

1. **The initialization strategy for temporal markers is critical**: Transfer initialization from numeric token embeddings outperforms random initialization by 2.5 points on Eb-F1.
2. **SEM is most impactful on long audio summarization**: mIoU improves from 90.4 to 94.2, though with a slight decrease on dense captioning (Eb-F1 from 37.8 to 37.4), indicating a trade-off.
3. **Absolute time encoding and positional encoding are orthogonal**: The two can be used together without conflict.
4. **Token retention saturates at 0.25**: Further increasing retention yields negligible gains.

## Highlights & Insights

1. **Elegant temporal marker design**: The two-level anchor+offset scheme reduces vocabulary bloat while maintaining precision, and the initialization strategy leverages the LLM's existing numerical understanding.
2. **Zero-overhead token merging**: Reuses existing Q-Former attention scores for token selection without additional modules.
3. **FTAR dataset fills a critical gap**: A 260K-scale temporally-sensitive audio instruction dataset covering three core temporal reasoning tasks.
4. **Novel application of mIoU to speech summarization**: First application of this metric to measure temporal localization accuracy in summarization.

## Limitations & Future Work

1. **METEOR score falls short of fine-tuned Qwen2-Audio**: Likely due to insufficient diversity of audio captioning data during base model pretraining.
2. **SEM has a slight negative effect on dense captioning**: Information compression may discard certain local event details.
3. **Only 7B scale**: Larger models (e.g., 13B) may yield further improvements.
4. **No exploration of streaming or very long audio**: The current system processes at most five 30-second segments (2.5 minutes), while real-world scenarios may require much longer inputs.
5. **Partial reliance on synthetic speech in FTAR**: TTS-synthesized data from CNN/DailyMail introduces a distribution gap relative to natural speech.

## Related Work & Insights

- **Temporal marker design is transferable to video**: Similar designs could be applied to video temporal grounding tasks.
- **Connection to VisionZip in token merging**: The method draws inspiration from visual token compression, validating the feasibility of cross-modal method transfer.
- **Advantages of dual-encoder architecture**: The complementary design of BEATs (environmental audio) + Whisper (speech) is worth referencing in future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The temporal marker design and initialization strategy are innovative, though the overall contribution is incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks, multiple baselines, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ — Fills the gap in temporal reasoning for LALMs; the FTAR dataset has long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)
- [\[NeurIPS 2025\] Video Finetuning Improves Reasoning Between Frames](../../NeurIPS2025/video_understanding/video_finetuning_improves_reasoning_between_frames.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](../../CVPR2026/video_understanding/ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)

</div>

<!-- RELATED:END -->
