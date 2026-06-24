---
title: >-
  [Paper Note] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens
description: >-
  [ACL 2025][Audio & Speech][Audio-Visual Speech Recognition] This work proposes MMS-LLaMA, which compresses multimodal speech tokens to only 3.5 per second through three modules: early audio-visual fusion, an AV Q-Former with dynamic query allocation, and a speech rate predictor. It achieves SOTA performance on LRS3 with a 0.72% WER while reducing token usage by 86% and FLOPs by 35.7%.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Audio-Visual Speech Recognition"
  - "LLM-based Speech Recognition"
  - "Token Compression"
  - "Q-Former"
  - "Speech Rate Prediction"
date: 2026-05-08
content_hash: 0e511c98cffab6ef
---

# MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens

**Conference**: ACL 2025  
**arXiv**: [2503.11315](https://arxiv.org/abs/2503.11315)  
**Code**: [https://github.com/JeongHun0716/MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA)  
**Area**: Speech  
**Keywords**: Audio-Visual Speech Recognition, LLM-based Speech Recognition, Token Compression, Q-Former, Speech Rate Prediction

## TL;DR

This work proposes MMS-LLaMA, which compresses multimodal speech tokens to only 3.5 per second through three modules: early audio-visual fusion, an AV Q-Former with dynamic query allocation, and a speech rate predictor. It achieves SOTA performance on LRS3 with a 0.72% WER while reducing token usage by 86% and FLOPs by 35.7%.

## Background & Motivation

**Background**: Audio-Visual Speech Recognition (AVSR) achieves robust speech recognition in noisy environments by combining audio and visual information of lip movements. Recently, LLM-based AVSR systems (such as LLaMA-AVSR) have achieved outstanding results by leveraging the context modeling capabilities of LLMs, with the WER reaching as low as 0.77%.

**Limitations of Prior Work**: LLM-based AVSR systems incur extremely high computational costs — the temporal resolution of multimodal speech tokens is much higher than that of text tokens, forcing the LLM's self-attention mechanism to process a large volume of tokens. For instance, LLaMA-AVSR generates 25 multimodal tokens per second, resulting in a GPU memory usage of 18.2GB and 2.24T FLOPs.

**Key Challenge**: High temporal resolution of speech signals is essential for ensuring recognition accuracy, yet too many tokens impose a heavy computational burden on the LLM. The challenge lies in compressing the number of tokens without losing linguistic information.

**Goal**: Design an efficient multimodal speech LLM framework that retains sufficient linguistic content with minimal token usage, drastically reducing computational costs without sacrificing accuracy.

**Key Insight**: A three-step compression strategy: (1) early fusion to halve the length of the audio-visual sequence; (2) an AV Q-Former that further compresses sequences by dynamically allocating the number of queries based on the input duration; (3) a speech rate predictor that adjusts token allocation according to speech rate, assigning more tokens to fast speech.

**Core Idea**: By using dynamic query allocation and speech rate awareness, multimodal speech tokens can be compressed to only 3.5 per second without sacrificing recognition accuracy.

## Method

### Overall Architecture

MMS-LLaMA consists of the following components:

1. **Visual Encoder** (AV-HuBERT): Extracts visual features $\mathbf{X}_v \in \mathbb{R}^{T_v \times D}$ from lip videos.
2. **Audio Encoder** (Whisper): Extracts audio features $\mathbf{X}_a \in \mathbb{R}^{T_a \times D}$.
3. **Length Adapter**: Aligns the temporal resolution of audio and visual features.
4. **Early AV Fusion Module**: Fuses bimordial sequences into a single sequence, halving the length.
5. **AV Q-Former**: Further compresses the sequence to the level of text tokens via dynamic query allocation.
6. **Speech Rate Predictor**: Adjusts query allocation based on speech rate.
7. **LLM Decoder** (LLaMA 3.2 3B): Predicts text from the compressed multimodal tokens.

### Key Designs

#### 1. Early AV Fusion Module

- **Function**: Fuses audio and visual features before feeding them into the LLM, halving the sequence length.
- **Mechanism**: First uses a length adapter to align the temporal resolution of the two modalities, and then compares three fusion strategies:
    - **Concatenation**: $\mathbf{X}_{av} = [\mathbf{X}'_a; \mathbf{X}_v] \in \mathbb{R}^{T_v \times 2D}$
    - **Addition**: $\mathbf{X}_{av} = \mathbf{X}'_a + \mathbf{X}_v \in \mathbb{R}^{T_v \times D}$
    - **Multimodal Attention**: $\mathbf{X}_{av} = \text{MHCA}(\mathbf{X}_v W_Q, \mathbf{X}'_a W_K, \mathbf{X}'_a W_V)$
- **Design Motivation**: Shifts fusion prior to the LLM to prevent the LLM from processing two sets of high-resolution sequences simultaneously. Experiments show that concatenation performs best under noise conditions (2.4% WER), and is thus adopted.

#### 2. AV Q-Former (Dynamic Query Allocation)

- **Function**: Compresses variable-length audio-visual feature sequences into short sequences matching the textual token scale.
- **Mechanism**: Defines a learnable query sequence $\mathbf{Q} \in \mathbb{R}^{N \times D_q}$, and proportionally allocates the number of queries based on the input duration:
  $$N_{\text{alloc}} = \lfloor f_Q \times \frac{T_v}{F_v} \rfloor$$
  where $f_Q$ is the query frequency (queries per second), and $T_v/F_v$ represents the input duration. The first $N_{\text{alloc}}$ queries are selected to be fed into the Q-Former:
  $$\mathbf{M} = \text{Q-Former}(\mathbf{Q}_{\text{alloc}}; \mathbf{X}_{av})$$
- **Design Motivation**: Traditional Q-Formers with fixed query sizes cannot handle variable-length inputs — wasting computational resources on short inputs and losing information on long ones. The dynamic strategy ensures that token count scales proportionally with the linguistic content. Experiments reveal that performance is maintained even when the query frequency is reduced to 4Hz.

#### 3. Speech Rate Predictor

- **Function**: Further refines token allocation based on the speech rate of each audio segment.
- **Mechanism**: Trains a lightweight predictor to estimate the normalized speech rate $r_s$, modifying the allocation formula to:
  $$N_{\text{alloc}} = \lfloor f_Q \times \frac{T_v}{F_v} \times r_s \rfloor$$
  More tokens are allocated to fast-paced speech, while fewer tokens are assigned to slower speech.
- **Architecture**: A 2-layer Transformer with a 256-dimensional embedding, 4 attention heads, a 1024-dimensional FFN, pretrained on audio-only features with MSE loss.
- **Design Motivation**: Audios of identical duration can contain varying amounts of linguistic content. At a 3Hz query frequency, integrating the speech rate predictor only adds 0.7 tokens/sec but reduces the WER from 0.95% to 0.90%.

### Loss & Training

- **Speech Rate Predictor**: Trained using MSE loss normalized based on the average speech rate of the training set. It is pretrained independently with frozen parameters.
- **Main Model**: Trained with standard CTC / sequence generation loss.
- **QLoRA** is applied to fine-tune the LLM: rank=16, alpha=32, dropout=0.05, optimizing only the Q/K/V/Output projection layers.
- **Training**: Adam optimizer with an initial lr of 1e-4, using a cosine scheduler for 30,000 steps with 0.5k warmup steps.
- **Inference**: Beam search (beam size = 5, temperature = 0.3).
- **Hardware**: 8 x RTX 3090 GPUs.

## Key Experimental Results

### Main Results

**SOTA Comparison on LRS3 Dataset**:

| Method | Decoder | Training Data (h) | WER (Noisy)↓ | WER (Clean)↓ |
|------|--------|-------------|-------------|-------------|
| auto-avsr | Conformer | 3448 | - | 0.9 |
| LP Conformer | LSTM | 100K | 1.9 | 0.9 |
| Whisper-Flamingo | Whisper | 1759 | 5.6 | 0.76 |
| LLaMA-AVSR | LLaMA 3.1 8B | 1759 | - | 0.77 |
| **MMS-LLaMA** | **LLaMA 3.2 3B** | **1759** | **1.9** | **0.72** |

MMS-LLaMA achieves SOTA with a 0.72% WER using a smaller LLM (3B vs 8B), and obtains only 1.9% under noisy environments.

### Ablation Study

**Effect of Step-by-Step Module Integration** (433h training data):

| Method | Tokens/Sec | GPU Memory (GB) | FLOPs (T) | WER↓ |
|------|---------|-------------|----------|------|
| Baseline (LLaMA-AVSR) | 25 | 18.2 | 2.24 | 0.97 |
| + Early AV Fusion | 12.5 | 14.7 | 1.81 | 0.92 |
| + AV Q-Former (freq=3) | 2.8 | 12.2 | 1.42 | 0.95 |
| + Speech Rate Predictor (freq=3) | **3.5** | **12.4** | **1.44** | **0.90** |

The complete pipeline compresses the tokens from 25/sec to 3.5/sec (an 86% reduction), reduces FLOPs from 2.24T to 1.44T (a 35.7% decrease), and lowers GPU memory from 18.2GB to 12.4GB (a 32% reduction), while reducing the WER from 0.97% to 0.90%.

**Comparison of LLM Sizes**:

| LLM | GPU Memory (GB) | FLOPs (T) | WER (Noisy)↓ | WER (Clean)↓ |
|-----|-------------|----------|-------------|-------------|
| LLaMA3.2-1B | 9.8 | 1.19 | 3.11 | 1.11 |
| **LLaMA3.2-3B** | **12.3** | **1.50** | **2.40** | **0.90** |
| LLaMA3.1-8B | 16.7 | 2.17 | 2.61 | 1.02 |

LLaMA3.2-3B performs the best under both clean and noisy conditions, demonstrating that larger models do not necessarily yield better performance.

**Impact of Visual Modality under Various SNR Conditions** (query freq=3):

| Condition | Without Visual | With Visual |
|------|--------|--------|
| Clean (∞ dB) | 1.10 | 0.95 |
| 0 dB | 2.66 | 2.66 |
| -5 dB | 13.54 | **7.44** |

The visual modality plays a vital role under heavy noise, dropping the WER from 13.54% to 7.44% at -5 dB.

### Key Findings

1. **3.5 tokens/sec is Sufficient**: Multimodal speech tokens can be extremely compressed (from 25 to 3.5/sec) with minimal performance degradation.
2. **Highly Effective Early Fusion**: Merely halving the sequence length lowers both the FLOPs and the WER (0.97% to 0.92%).
3. **Stable Gains from Speech Rate Predictor**: Across varying query frequencies, it consistently improves WER by 0.05-0.38%.
4. **3B LLM Outperforms 8B**: For this specific task, the sweet spot for model scale is 3B rather than 8B.
5. **Visual Clues are Crucial in Noise**: Under a -5 dB SNR, the visual modality brings a 45% relative reduction in WER.

## Highlights & Insights

1. **Striking Compression Efficiency**: An 86% token compression rate combined with a 35.7% FLOPs reduction and even improved performance, offering immense practical deployment value.
2. **Novel Query Allocation Strategy**: A dual-regulation mechanism combining dynamic duration and speech rate awareness, which is more robust than a fixed-window Q-Former.
3. **Clear Design Philosophy**: Three modules are structured progressively — first halving the length via fusion, then compressing via the Q-Former, and finally refining through speech rate adjustments.
4. **Visual Speech Rate Predictor**: Auxiliary experiments demonstrate that a vision-only speech rate predictor (relying on lip movements) performs comparably to the audio-based version (0.75% vs 0.72% WER), ensuring usability when audio is unavailable.

## Limitations & Future Work

1. Evaluation is limited to the LRS3 dataset (English TED talks), representing constrained scene and language diversity.
2. While the concatenation strategy is optimal under noise, it is outperformed by multimodal cross-attention under clean conditions (0.90% vs 0.87%). An adaptive fusion strategy could be investigated.
3. The speech rate predictor is trained separately. End-to-end joint training might yield further improvements.
4. The quality of pseudo-labels on VoxCeleb2 might limit the model's generalization to non-TED domains.

## Related Work & Insights

- **LLaMA-AVSR (Cappellazzo et al., 2024)**: Directly serves as the baseline, using a 25 tokens/sec LLaMA 3.1 8B scheme.
- **Whisper-Flamingo (Rouditchenko et al., 2024)**: Integrates a pretrained visual encoder with Whisper.
- **Q-Former (Dai et al., 2023)**: A token compression method popular in the vision-language domain, adapted to audio-visual speech here.
- **auto-avsr (Ma et al., 2023)**: A Conformer-based AVSR SOTA reaching a 0.9% WER using a massive amount of dataset.
- **Insights**: The bottleneck of LLMs in speech tasks is not model capability, but rather computational efficiency. Token compression is the primary challenge; moreover, dynamic allocation is superior to fixed strategies for handling variable-length speech inputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of dynamic query allocation in AV Q-Former and the speech rate predictor is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extensive ablation studies covering fusion strategies, LLM sizes, query frequencies, and SNR levels.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, though some notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐⭐ — Outstanding practical value; the 86% token compression rate is critical for enabling real-world deployments of multimodal speech LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](spark-tts_an_efficient_llm-based_text-to-speech_model_with_single-stream_decoupl.md)
- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](../../NeurIPS2025/audio_speech/mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ACL 2025\] Dialectal Coverage and Generalization in Arabic Speech Recognition](dialectal_coverage_and_generalization_in_arabic_speech_recognition.md)
- [\[ACL 2025\] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder](multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)

</div>

<!-- RELATED:END -->
