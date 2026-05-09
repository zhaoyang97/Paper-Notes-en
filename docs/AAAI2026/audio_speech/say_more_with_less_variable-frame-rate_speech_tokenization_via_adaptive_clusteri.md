---
title: >-
  [Paper Note] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
description: >-
  [AAAI 2026][Audio & Speech][speech tokenization] This paper proposes VARSTok, the first fully dynamic variable-frame-rate speech tokenizer, which achieves adaptive token allocation via temporal-aware density peak clustering and implicit duration coding, surpassing fixed-frame-rate baselines while using fewer tokens.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - speech tokenization
  - variable frame rate
  - density peak clustering
  - implicit duration coding
  - speech language model
date: 2026-05-08
content_hash: be24a19d702133f8
---

# Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding

**Conference**: AAAI 2026
**arXiv**: [2509.04685](https://arxiv.org/abs/2509.04685)
**Code**: [VARSTok](https://zhengrachel.github.io/VARSTok)
**Area**: Audio & Speech
**Keywords**: speech tokenization, variable frame rate, density peak clustering, implicit duration coding, speech language model

## TL;DR

This paper proposes VARSTok, the first fully dynamic variable-frame-rate speech tokenizer, which achieves adaptive token allocation via temporal-aware density peak clustering and implicit duration coding, surpassing fixed-frame-rate baselines while using fewer tokens.

## Background & Motivation

### State of the Field

**Background**: Existing speech tokenizers (e.g., WavTokenizer, EnCodec) uniformly allocate tokens at a fixed frame rate (e.g., 40 Hz, 75 Hz), ignoring the temporal variation in information density of speech signals.

### Limitations of Prior Work

**Limitations of Prior Work**: In natural speech, silence and stable vowel regions exhibit substantial redundancy, whereas segments with rapid phonetic transitions and rich emotional expression carry high information density.

### Root Cause

**Key Challenge**: Fixed frame rates lead to token waste in redundant regions and under-representation in high-information regions, making it difficult for downstream speech LMs to learn natural prosody.

### Prior Solutions

**Prior Solutions**: Existing adaptive compression methods (e.g., TFC) switch among only a few predefined frame rates, constituting "pseudo-dynamic" schemes, and do not model token duration.

### Paper Goals

**Goal**: How to design a fully dynamic variable-frame-rate acoustic speech tokenizer that adaptively allocates tokens based on local feature similarity, and can be directly applied to downstream autoregressive speech LMs without requiring an auxiliary duration predictor?

## Method

### Overall Architecture

VARSTok consists of four components: Speech Encoder → Temporal-Aware Density Peak Clustering → VQ Module → Speech Decoder.

1. The encoder converts the waveform into frame-level embeddings $\mathbf{X} \in \mathbb{R}^{T \times H}$
2. The clustering module adaptively groups frames into $N$ variable-length clusters $\mathcal{C}_1, \dots, \mathcal{C}_N$
3. Each cluster is mean-pooled and quantized via VQ (single codebook, $K=4096$)
4. Implicit duration coding encodes content and duration into a single token ID
5. During decoding, tokens are expanded according to duration and fed into the decoder to reconstruct the waveform

### Key Design 1: Temporal-Aware Density Peak Clustering

The local density $\rho_i$ and peak distance $\delta_i$ of each frame are computed as:

$$\rho_i = \exp\left(\frac{1}{m}\sum_{j \in \text{KNN}(i)} \phi(\mathbf{x}_i, \mathbf{x}_j)\right), \quad \phi(\mathbf{x}_i, \mathbf{x}_j) = \frac{1 + \langle \mathbf{x}_i, \mathbf{x}_j \rangle}{2}$$

The peak score is $s_i = \rho_i \cdot \delta_i$, and frames with high scores serve as cluster seeds. Clusters are expanded bidirectionally from seeds; candidate frames must satisfy:

$$\phi(\mathbf{x}_{i^*}, \mathbf{x}_t) - \beta \cdot s_t > \tau$$

and must maintain temporal contiguity. Expansion is constrained by a maximum span $S_{\max}$.

### Key Design 2: Implicit Duration Coding

The VQ index $k_n$ and duration $d_n$ are encoded into a single token ID:

$$\text{ID}_n = (d_n - 1) \cdot K + k_n$$

During decoding, they are recovered via integer division and modulo: $d_n = \lfloor \text{ID}_n / K \rfloor + 1$, $k_n = \text{ID}_n \bmod K$.

The extended vocabulary size is $K \times S_{\max}$, requiring no additional duration predictor and directly compatible with autoregressive LMs.

## Key Experimental Results

### Main Results

| Model | Frame Rate (Hz) | Bitrate (kbps) | UTMOS↑ | PESQ↑ | STOI↑ |
|-------|----------------|----------------|--------|-------|-------|
| WavTokenizer | 75.00 | 0.90 | 4.0247 | 2.4543 | 0.9188 |
| WavTokenizer | 40.00 | 0.48 | 3.6107 | 1.7075 | 0.8652 |
| BigCodec | 40.00 | 0.52 | 3.9802 | 1.8796 | 0.8653 |
| **VARSTok** (τ=0.8) | 36.81 | 0.52 | **4.0000** | 1.8887 | 0.8814 |
| **VARSTok** (τ=0.7) | 30.95 | 0.43 | **3.8949** | 1.7095 | 0.8601 |

- At 30.95 Hz (reducing tokens by **23%** relative to the 40 Hz baseline), UTMOS still reaches 3.8949, surpassing the 40 Hz WavTokenizer.
- At τ=0.8, UTMOS=4.0000, approaching the 75 Hz WavTokenizer with fewer than half the tokens.
- Downstream TTS: VARSTok (τ=0.8) achieves WER=6.787% (vs. WavTokenizer 7.481%) and MOS=4.053 (vs. 3.983).
- ARCH semantic evaluation: AudioMNIST F1 improves from 0.4509 to 0.6078 (τ=0.7).
- Inference efficiency: at τ=0.6, RTF=0.487, accelerating inference by **36%** over the baseline.

## Highlights & Insights

- **First** fully dynamic variable-frame-rate acoustic tokenizer directly integrable into downstream autoregressive speech LMs.
- The implicit duration coding scheme is elegant and lightweight—encoding content and duration into a single token without extra modules or training.
- Hyperparameters τ and $S_{\max}$ provide flexible rate–quality control knobs.
- Significant improvements over fixed-frame-rate baselines on semantic evaluation tasks indicate that dynamic token allocation yields better learned representations.

## Limitations & Future Work

- Training is conducted solely on LibriTTS (585 h); generalization to large-scale and multilingual settings remains unvalidated.
- The clustering algorithm is non-differentiable, precluding end-to-end joint optimization of the segmentation strategy.
- Quality degrades noticeably when $S_{\max}$ is large (e.g., 8), limiting extreme compression capability.
- Objective speaker similarity declines slightly as frame rate decreases, although subjective MOS differences are not significant.
- Application to other audio domains such as music and environmental sounds has not been explored.

## Related Work & Insights

| Dimension | VARSTok | TFC | WavTokenizer |
|-----------|---------|-----|-------------|
| Frame Rate Type | Fully dynamic & continuous | Pseudo-dynamic (3 predefined rates) | Fixed |
| Duration Modeling | Implicit coding | None | None |
| Codebook | Single codebook | Multi-codebook RVQ | Single codebook |
| Downstream LM Adaptation | Direct use | Requires hierarchical fusion | Direct use |

## Insights

- The implicit duration coding scheme $(d-1) \cdot K + k$ can be generalized to other discretization scenarios requiring simultaneous encoding of attribute and content.
- The temporal contiguity constraint in density peak clustering can be adapted for adaptive segmentation of video and action sequences.
- The paradigm of variable frame rate combined with a single token representation may be a key direction for improving speech LM efficiency.

## Rating

⭐⭐⭐⭐ — Strong novelty; the first work to demonstrate that a fully dynamic variable-frame-rate acoustic tokenizer can be directly applied to speech LMs, though validation at larger data scales and broader generalization settings remains insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[AAAI 2026\] Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[AAAI 2026\] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)

</div>

<!-- RELATED:END -->
