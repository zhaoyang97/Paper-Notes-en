---
title: >-
  [Paper Note] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
description: >-
  [AAAI 2026][Audio & Speech][speech tokenization] VARSTok is proposed, which is the first fully dynamic variable-frame-rate speech tokenizer. Through temporal-aware density peak clustering and implicit duration coding, it achieves adaptive token allocation, outperforming fixed-frame-rate baselines while using fewer tokens.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "speech tokenization"
  - "variable frame rate"
  - "density peak clustering"
  - "implicit duration coding"
  - "speech language model"
date: 2026-05-08
content_hash: c75330430d09a250
---

# Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding

**Conference**: AAAI 2026  
**arXiv**: [2509.04685](https://arxiv.org/abs/2509.04685)  
**Code**: [VARSTok](https://zhengrachel.github.io/VARSTok)  
**Area**: Audio & Speech  
**Keywords**: speech tokenization, variable frame rate, density peak clustering, implicit duration coding, speech language model

## TL;DR

VARSTok is proposed, which is the first fully dynamic variable-frame-rate speech tokenizer. Through temporal-aware density peak clustering and implicit duration coding, it achieves adaptive token allocation, outperforming fixed-frame-rate baselines while using fewer tokens.

## Background & Motivation

### Background

**Background**: Existing speech tokenizers (e.g., WavTokenizer, EnCodec) allocate tokens uniformly at a fixed frame rate (e.g., 40Hz, 75Hz), ignoring temporal variations in the information density of speech signals.

### Limitations of Prior Work

**Limitations of Prior Work**: In natural speech, silence and stable vowel regions contain significant redundancy, whereas rapid phonetic transitions and emotionally rich segments exhibit high information density.

### Key Challenge

**Key Challenge**: Fixed frame rates lead to token wastage in redundant regions and under-representation in high-information regions, making it difficult for downstream speech LMs to learn natural prosody.

### Key Insight

**Key Insight**: Existing adaptive compression efforts (e.g., TFC) only switch between a few pre-defined frame rates, which is "pseudo-dynamic" and does not model token duration.

### Goal

**Goal**: How to design a fully dynamic variable-frame-rate acoustic speech tokenizer that can adaptively allocate tokens based on local feature similarity, and can be directly applied to downstream autoregressive speech LMs without an auxiliary duration predictor?

## Method

### Overall Architecture

VARSTok consists of four components: Speech Encoder → Temporal-Aware Density Peak Clustering → VQ Module → Speech Decoder.

1. The Encoder converts the waveform into frame-level embeddings $\mathbf{X} \in \mathbb{R}^{T \times H}$.
2. The clustering module adaptively groups them into $N$ variable-length clusters $\mathcal{C}_1, \dots, \mathcal{C}_N$.
3. Each cluster is mean-pooled and then quantized via VQ (single codebook, $K=4096$).
4. Implicit duration coding encodes both content and duration into a single token ID.
5. During decoding, tokens are expanded according to their duration and fed into the decoder to reconstruct the waveform.

### Key Design 1: Temporal-Aware Density Peak Clustering

The local density $\rho_i$ and peak distance $\delta_i$ for each frame are calculated:

$$\rho_i = \exp\left(\frac{1}{m}\sum_{j \in \text{KNN}(i)} \phi(\mathbf{x}_i, \mathbf{x}_j)\right), \quad \phi(\mathbf{x}_i, \mathbf{x}_j) = \frac{1 + \langle \mathbf{x}_i, \mathbf{x}_j \rangle}{2}$$

The peak score $s_i = \rho_i \cdot \delta_i$ is used, and high-scoring frames serve as cluster seeds. Starting from these seeds, candidates are expanded bilaterally, requiring candidate frames to satisfy:

$$\phi(\mathbf{x}_{i^*}, \mathbf{x}_t) - \beta \cdot s_t > \tau$$

and they must maintain temporal contiguity. The expansion is restricted by a maximum span $S_{\max}$.

### Key Design 2: Implicit Duration Coding

The VQ index $k_n$ and duration $d_n$ are encoded into a single token ID:

$$\text{ID}_n = (d_n - 1) \cdot K + k_n$$

During decoding, these are recovered through division and modulo operations: $d_n = \lfloor \text{ID}_n / K \rfloor + 1$ and $k_n = \text{ID}_n \bmod K$.

This expands the vocabulary size to $K \times S_{\max}$, allowing direct compatibility with autoregressive LMs without requiring an extra duration predictor.

## Key Experimental Results

### Main Results

| Model | Frame Rate (Hz) | Bitrate (kbps) | UTMOS↑ | PESQ↑ | STOI↑ |
|------|---------|--------------|--------|-------|-------|
| WavTokenizer | 75.00 | 0.90 | 4.0247 | 2.4543 | 0.9188 |
| WavTokenizer | 40.00 | 0.48 | 3.6107 | 1.7075 | 0.8652 |
| BigCodec | 40.00 | 0.52 | 3.9802 | 1.8796 | 0.8653 |
| **VARSTok**($\tau=0.8$) | 36.81 | 0.52 | **4.0000** | 1.8887 | 0.8814 |
| **VARSTok**($\tau=0.7$) | 30.95 | 0.43 | **3.8949** | 1.7095 | 0.8601 |

- At 30.95Hz (reducing tokens by **23%** compared to the 40Hz baseline), UTMOS still reaches 3.8949, outperforming the 40Hz WavTokenizer.
- At $\tau=0.8$, UTMOS reaches 4.0000, which is close to the 75Hz WavTokenizer but with less than half the quantity of tokens.
- Downstream TTS: VARSTok ($\tau=0.8$) achieves WER=6.787% (vs WavTokenizer 7.481%) and MOS=4.053 (vs 3.983).
- ARCH semantic evaluation: AudioMNIST F1 improved from 0.4509 to 0.6078 ($\tau=0.7$).
- Inference efficiency: At $\tau=0.6$, RTF is 0.487, achieving a **36%** speedup compared to the baseline.

## Highlights & Insights

- **The first** fully dynamic variable-frame-rate acoustic tokenizer that can be directly integrated into downstream autoregressive speech LMs.
- The implicit duration coding scheme is simple and elegant, encoding both content and duration into a single token without requiring extra modules or training.
- Hyperparameters $\tau$ and $S_{\max}$ provide flexible rate-quality control knobs.
- It also significantly outperforms the fixed-frame-rate baselines on semantic evaluation tasks, indicating that dynamic token allocation learns better representations.

## Limitations & Future Work

- Only trained on LibriTTS (585h), without verification on large-scale data and multilingual scenarios.
- The clustering algorithm is non-differentiable, making it impossible to jointly optimize the segmentation strategy end-to-end.
- When $S_{\max}$ is too large (e.g., 8), quality degrades significantly, indicating limited extreme compression capability.
- Objective speaker similarity decreases slightly as the frame rate drops (although subjective MOS differences are not significant).
- Other audio domains such as music and environmental sound have not been explored.

## Related Work & Insights

| Dimension | VARSTok | TFC | WavTokenizer |
|------|---------|-----|-------------|
| Frame Rate Type | Fully dynamic & continuous | Pseudo-dynamic (3 pre-defined) | Fixed |
| Duration Modeling | Implicit coding | None | None |
| Codebook | Single codebook | Multi-codebook RVQ | Single codebook |
| Downstream LM Adaptation | Direct usage | Requires hierarchical fusion | Direct usage |

### Insights

- The concept of implicit duration coding $(d-1) \cdot K + k$ can be extended to other discretization scenarios that require simultaneous encoding of attributes and content.
- The constraint design of density peak clustering to maintain temporal contiguity can be adapted for adaptive segmentation of video or action sequences.
- The paradigm of variable frame rate + single-token representation may be a key direction for improving the efficiency of speech LMs.

## Rating

⭐⭐⭐⭐ — Highly innovative, demonstrating for the first time that a fully dynamic variable-frame-rate acoustic tokenizer can be directly used for speech LMs, though verification on data scale and generalization is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](../../ACL2025/audio_speech/soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](../../ICLR2026/audio_speech/taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ICLR 2026\] Confident and Adaptive Generative Speech Recognition via Risk Control](../../ICLR2026/audio_speech/confident_and_adaptive_generative_speech_recognition_via_risk_control.md)

</div>

<!-- RELATED:END -->
