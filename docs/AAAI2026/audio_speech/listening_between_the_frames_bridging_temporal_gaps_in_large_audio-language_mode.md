---
title: >-
  [Paper Note] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models
description: >-
  [AAAI 2026][Audio & Speech][Audio-Language Models] This paper proposes TimeAudio, which endows large audio-language models (LALMs) with precise temporal localization capabilities and end-to-end long audio understanding through three key modules: Temporal Markers, Absolute Time-aware Encoding, and Segment-level Token Merging. It also constructs the FTAR dataset for instruction tuning in fine-grained temporal reasoning.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Audio-Language Models"
  - "Temporal Localization"
  - "Fine-grained Audio Understanding"
  - "Long Audio"
  - "Token Merging"
date: 2026-05-08
content_hash: 5cf1a5317eb12cf8
---

# Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models

**Conference**: AAAI 2026  
**arXiv**: [2511.11039](https://arxiv.org/abs/2511.11039)  
**Code**: [github](https://github.com/lysanderism/TimeAudio)  
**Area**: Video Understanding / Audio Understanding  
**Keywords**: Audio-Language Models, Temporal Localization, Fine-grained Audio Understanding, Long Audio, Token Merging

## TL;DR

This paper proposes TimeAudio, which endows large audio-language models (LALMs) with precise temporal localization capabilities and end-to-end long audio understanding through three key modules: Temporal Markers, Absolute Time-aware Encoding, and Segment-level Token Merging. It also constructs the FTAR dataset for instruction tuning in fine-grained temporal reasoning.

## Background & Motivation

### Background

Large Audio-Language Models (LALMs) such as Qwen2-Audio and SALMONN have demonstrated powerful capabilities in audio content understanding and conversational question-answering tasks, enabling the unified processing of speech and environmental sounds. These models achieve free-form audio question-answering by integrating an audio encoder with a pretrained decoder-only LLM.

### Limitations of Prior Work

Despite significant progress, current LALMs exhibit severe deficiencies in **fine-grained audio understanding**, manifested in three aspects:

**Inaccurate timestamp prediction**: Existing models struggle to accurately predict the start and end times of events. For instance, Qwen2-Audio achieves a zero-shot Eb-F1 of only 9.8 on dense audio captioning tasks, indicating extremely weak event localization capabilities.

**Limited long audio processing capacity**: Most models are restricted to short audio inputs, failing to handle end-to-end long audio understanding and suffering from severe hallucination when faced with long audio.

**Lack of suitable datasets and evaluation benchmarks**: Existing instruction-tuning data and evaluation benchmarks primarily focus on general audio understanding, lacking support for fine-grained temporal reasoning.

### Key Challenge

Audio understanding inherently requires capturing both **semantic content** and **temporal localization**. However, when mapping audio features directly to a shared latent space, existing LALMs **do not explicitly model fine-grained temporal localization information**, making it difficult for the LLM decoder to predict precise timestamps. This is a dual architecture-data deficiency.

### Key Insight

Addressing the problem simultaneously from three dimensions—**timestamp representation**, **model architecture**, and **data**: introducing discrete temporal markers to replace continuous numerical regression, incorporating absolute time-aware encoding to enhance temporal perception, designing a token-merging mechanism to support long audio, and constructing the FTAR dataset to strengthen temporal reasoning capabilities.

## Method

### Overall Architecture

Based on the SALMONN architecture, TimeAudio consists of four core components:
1. **Sliding Audio Encoder**: Divides long audio into short segments and extracts environmental sound and speech features using BEATs and Whisper dual encoders, respectively.
2. **Window Q-Former**: Projects encoded audio tokens into the language space.
3. **Segment-level Token Merging (SEM)**: Selectively filters important tokens based on attention scores to compress redundant information.
4. **LLM**: Processes the merged audio-visual tokens and text tokens to generate responses.

### Key Designs

#### 1. Temporal Markers

**Problem**: Directly predicting timestamps with numerical tokens causes convergence difficulties for LLMs, whereas relative temporal tokens (e.g., $\langle 0.2 \rangle$, $\langle 0.4 \rangle$, $\langle 0.6 \rangle$...) impose a heavy burden on the vocabulary and introduce quantization errors.

**Mechanism**: Designing a two-level discretization scheme consisting of anchor tokens and offset tokens, converting continuous timestamps into discrete temporal token sequences.

- **Anchor token** $\langle a_k \rangle$: Represents coarse-grained positions at the integer second level.
- **Offset token** $\langle f_k \rangle$: Represents fine-grained adjustments at the sub-second level.

The key innovation lies in the **initialization strategy**—instead of random initialization, knowledge is transferred from existing numerical token embeddings to these new tokens:

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle a_0 \rangle)} = [\mathbf{W}_{\text{token}}]_{\text{ID}(0)}$$

$$[\mathbf{W}_{\text{token}}]_{\text{ID}(\langle f_0 \rangle)} = \frac{[\mathbf{W}_{\text{token}}]_{\text{ID}(0)} + [\mathbf{W}_{\text{token}}]_{\text{ID}(\text{.})}}{2}$$

**Design Motivation**: Utilizing the LLM's existing capacity for numerical understanding, preventing random initialization from disrupting the pretrained embedding space. This only introduces M=20 special tokens, maintaining precision regardless of audio length.

#### 2. Absolute Time-aware Encoding (ATE)

**Problem**: Understanding the temporal order of audio events remains challenging; the diversity of events and speech prosody makes it difficult for the model to accurately identify real temporal locations.

**Mechanism**: Constructing a learnable absolute time embedding $\mathbf{W}_t$ to explicitly inject absolute time information into the features of each audio segment:

$$\hat{\mathbf{W}}_i = \mathbf{W}_i + \mathbf{e}_t(t_i)$$

where $\mathbf{e}_t(t_i) = [\mathbf{W}_t]_{j_i}$ is obtained via one-hot table lookup, and $j_i$ is the discretized temporal index.

**Design Motivation**:
- Retains the relative positional information of the original sequence embeddings while precisely reflecting the absolute position of each time point within the audio.
- The temporal embeddings are **zero-initialized** to protect the integrity of the pretrained audio encoder in the initial phase of training.
- The authors hypothesize that absolute time-aware encoding and positional encoding are **orthogonal**, which is also verified by experiments.

#### 3. Segment-level Token Merging (SEM)

**Problem**: Processing long audio incurs high computational overhead; Q-Former uses a fixed number of queries for alignment, causing the number of tokens to scale linearly with audio length.

**Mechanism**: Reusing the attention information of Q-Former to adaptively filter and merge tokens without introducing additional computation. This is done in two steps:

**Step 1: Attention Token Selection**
- Compute the multi-head attention matrix in the Q-Former: $\mathbf{A} = \text{Softmax}(\mathbf{QK}^T / \sqrt{D})$
- Average across all heads to obtain a unified score matrix.
- Keep the tokens with the highest scores as "attention tokens" (22 tokens preserved in experiments).

**Step 2: Clustering-based Token Merging**
- Divide the remaining tokens equally into target tokens and merge candidates.
- Calculate similarity using the dot product of key vectors: $\mathbf{Sim}(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{k}_i \mathbf{k}_j^T$
- Assign candidate tokens to the most semantically similar centroids, merging them into "context tokens" (4 tokens).

**Design Motivation**: Information is scattered in long audio. By adaptively selecting important tokens and aggregating redundant ones, **75%** of the redundant information is compressed while preserving the core semantics.

### Loss & Training

A **two-stage training** strategy is adopted:

**Stage 1: Temporal Token Alignment**
- Continue pretraining from a trained LALM checkpoint.
- Training objective: Align fine-grained audio features with temporal information.
- Data: Temporal audio localization, dense audio captioning, and timeline speech summarization.
- Trainable components: LoRA adapter, Window Q-Former, absolute time embeddings, and special text embeddings.
- Frozen components: Audio encoder and LLM.
- 10 epochs, learning rate of 1e-5.

**Stage 2: Long Audio Instruction Tuning**
- Compensate for the semantic misalignment when the Stage 1 model handles long audio.
- Fine-tune Window Q-Former and LoRA.
- A small amount of instruction-following data, 5 epochs, learning rate of 2e-6.

### FTAR Dataset

A dataset containing **260K** audio-text pairs is constructed, covering three major task types:
- **Dense Audio Captioning** (110K): Generating event descriptions with timestamps.
- **Temporal Audio Grounding** (100K): Locating time intervals based on text queries.
- **Timeline Speech Summarization** (42K): Speech content summarization with timelines.
- **Audio Temporal QA** (15K): QA regarding counting, duration, and temporal order.

## Key Experimental Results

### Main Results

| Task | Metric | TimeAudio | Qwen2-Audio (FTAR FT) | SALMONN-13B (FTAR FT) | Zero-shot Qwen2-Audio |
|------|------|-----------|----------------------|---------------------|-------------------|
| Dense Audio Captioning | METEOR | 20.4 | **22.4** | 20.2 | 6.7 |
| Dense Audio Captioning | Eb-F1 | **37.4** | 36.5 | 32.0 | 9.8 |
| Dense Audio Captioning | At-F1 | **70.5** | 67.8 | 67.6 | 50.3 |
| Temporal Audio Grounding | R@0.5 | **75.7** | 72.8 | 69.2 | 32.1 |
| Temporal Audio Grounding | R@0.7 | **61.2** | 55.4 | 53.5 | 18.7 |
| Temporal Audio Grounding | mIoU | **57.8** | 51.7 | 50.3 | 20.5 |
| Speech Summarization | ROUGE-1 | **42.4** | 40.0 | 40.2 | 17.4 |
| Speech Summarization | ROUGE-L | **30.8** | 28.5 | 29.8 | 12.3 |
| Speech Summarization | mIoU | **94.2** | 85.2 | 88.2 | 13.3 |

- The largest improvement is observed in Temporal Audio Grounding: mIoU of 57.8 vs. 51.9 for SALMONN-7B (+11.4%).
- The mIoU for Speech Summarization reaches up to 94.2, outperforming all baselines and specialized models like Hubert-MiniChat.

### Ablation Study

| Configuration | Eb-F1 | At-F1 | R@0.5 | mIoU (Grounding) | ROUGE-1 | mIoU (Summary) | Description |
|------|-------|-------|-------|-----------|---------|-----------|------|
| SALMONN (FTAR FT) | 32.4 | 68.0 | 71.4 | 51.9 | 39.5 | 84.3 | Baseline |
| +TM (Uninitialized) | 33.5 | 69.2 | 71.7 | 52.3 | 40.0 | 87.5 | Temporal Markers are effective |
| +TM (Knowledge Transfer Init) | 36.0 | 70.8 | 72.7 | 54.9 | 41.0 | 88.7 | Initialization strategy is critical |
| +TM+ATE | 37.8 | 71.4 | 73.7 | 56.0 | 41.4 | 90.4 | Absolute time-aware encoding added |
| +TM+ATE+SEM | 37.4 | 70.5 | 75.7 | 57.8 | 42.4 | **94.2** | Full model |

**Ablation of Token Retention Ratio**:

| Retention Ratio | ROUGE-1 | ROUGE-L| mIoU |
|---------|---------|---------|------|
| 0.10 | 24.3 | 18.4 | 72.9 |
| 0.15 | 31.6 | 20.5 | 73.2 |
| 0.20 | 40.2 | 29.7 | 84.8 |
| **0.25** | **42.4** | **30.8** | **94.2** |
| 0.30 | 42.5 | 31.1 | 94.0 |

0.25 is ultimately selected as the balance point between performance and efficiency.

### Key Findings

1. **Initialization strategy of temporal markers is critical**: Transferring embeddings from numerical tokens yields a 2.5-point higher Eb-F1 than random initialization.
2. **SEM is most prominent in long audio summarization**: mIoU increases from 90.4 to 94.2, though there is a minor trade-off as Eb-F1 on dense captioning slightly drops from 37.8 to 37.4.
3. **Absolute time-aware encoding and positional encoding are orthogonal**: The two can be superimposed without conflict.
4. **The token retention ratio reaches saturation at 0.25**: Retaining more tokens yields marginal gains.

## Highlights & Insights

1. **Ingenious Temporal Marker design**: The two-level anchor + offset scheme simultaneously avoids vocabulary expansion and maintains precision, with an initialization strategy leveraging the LLM's existing numerical understanding.
2. **Zero-overhead Token Merging**: Reuses existing Q-Former attention scores to filter tokens without requiring extra modules.
3. **FTAR dataset fills a critical vacancy**: A 260K-scale time-sensitive audio instruction-following dataset covering three core temporal reasoning tasks.
4. **Innovative application of the mIoU metric in speech summarization**: Used for the first time to measure the temporal localization accuracy of summaries.

## Limitations & Future Work

1. **Slightly lower METEOR scores than fine-tuned Qwen2-Audio**: Potentially due to the lack of diverse audio captioning data during the pretraining of the base model.
2. **SEM has a minor negative impact on dense captioning**: Information compression might lead to the loss of certain local event details.
3. **Limited to 7B scale**: Utilizing larger models (e.g., 13B) could yield further improvements.
4. **Lack of exploration on streaming/ultra-long audio**: Currently processes a maximum of five 30-second segments (2.5 minutes), whereas real-world scenarios may require handling longer inputs.
5. **FTAR dataset is partially dependent on synthetic speech**: TTS-synthesized data from CNN/DailyMail exhibits a distribution gap compared to real speech.

## Related Work & Insights

- **Transferability of temporal markers to the video domain**: Similar designs can be applied to video temporal grounding.
- **Connection between token merging and VisionZip**: The method is inspired by visual token compression, validating the feasibility of cross-modal method transfer.
- **Advantages of the dual-encoder architecture**: The complementary dual-encoder design (BEATs for environmental sound + Whisper for speech) is worth emulating.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The temporal markers and initialization strategy are innovative, though the overall work is incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks, multiple baselines, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-elaborated motivation.
- **Value**: ⭐⭐⭐⭐ — Fills a critical gap in LALM temporal reasoning, with the FTAR dataset offering long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](../../ACL2026/audio_speech/temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[CVPR 2026\] AudioStory: Generating Long-Form Narrative Audio with Large Language Models](../../CVPR2026/audio_speech/audiostory_generating_long-form_narrative_audio_with_large_language_models.md)
- [\[ICLR 2026\] OWL: Geometry-Aware Spatial Reasoning for Audio Large Language Models](../../ICLR2026/audio_speech/owl_geometry-aware_spatial_reasoning_for_audio_large_language_models.md)

</div>

<!-- RELATED:END -->
