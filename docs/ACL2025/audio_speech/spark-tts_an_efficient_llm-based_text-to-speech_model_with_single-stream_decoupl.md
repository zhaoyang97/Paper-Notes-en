---
title: >-
  [Paper Note] Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens
description: >-
  [ACL 2025][Audio & Speech][text-to-speech] This paper proposes Spark-TTS, an efficient TTS system based on a novel single-stream speech codec, BiCodec, and the Qwen2.5 LLM. By decoupling speech into low-bitrate semantic tokens and fixed-length global tokens, Spark-TTS achieves zero-shot voice cloning and coarse-to-fine attribute control, reaching SOTA intelligibility on Seed-TTS-eval.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "text-to-speech"
  - "speech codec"
  - "LLM-based TTS"
  - "voice cloning"
  - "controllable speech synthesis"
date: 2026-05-08
content_hash: 4fb97c95efa499d7
---

# Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens

**Conference**: ACL 2025  
**arXiv**: [2503.01710](https://arxiv.org/abs/2503.01710)  
**Code**: [GitHub](https://github.com/SparkAudio/Spark-TTS)  
**Area**: Speech  
**Keywords**: text-to-speech, speech codec, LLM-based TTS, voice cloning, controllable speech synthesis

## TL;DR

This paper proposes Spark-TTS, an efficient TTS system based on a novel single-stream speech codec, BiCodec, and the Qwen2.5 LLM. By decoupling speech into low-bitrate semantic tokens and fixed-length global tokens, Spark-TTS achieves zero-shot voice cloning and coarse-to-fine attribute control, reaching SOTA intelligibility on Seed-TTS-eval.

## Background & Motivation

**Background**: LLM-based codec TTS has become the mainstream paradigm for zero-shot TTS. Utilizing large-scale training data and large model architectures, synthesized speech achieves naturalness close to human speech. Representative methods include VALL-E, CosyVoice, Seed-TTS, etc.

**Limitations of Prior Work**: 
   - Existing codec TTS architectures are complex, requiring either dual generative models (e.g., semantic-to-acoustic two-stage pipeline) or parallel multi-stream codebook prediction mechanisms (e.g., group VQ), which deviates from standard text LLM frameworks.
   - Semantic tokens, although compact, lack voice timbre control capabilities, necessitating an additional acoustic feature prediction module.
   - Acoustic tokens rely on complex codebook architectures.
   - Existing systems are primarily limited to reference-audio-driven generation and cannot precisely specify speech attributes (such as precise pitch values) to create new voices.
   - A large number of studies use private datasets, making fair comparisons difficult.

**Key Challenge**: How to achieve high-quality speech reconstruction while maintaining architectural unity with text LLMs (single-stream autoregressive) and supporting flexible speech attribute control?

**Goal**: To design a unified architecture for zero-shot TTS and attribute-controllable speech generation, fully aligned with the standard text LLM paradigm.

**Key Insight**: Design BiCodec to decouple speech into semantic tokens (time-varying linguistic content) and global tokens (time-invariant speaker identity/attributes), and couple this with a Chain-of-Thought generation strategy to achieve coarse-to-fine attribute control.

**Core Idea**: Decompose speech into a single-stream format of semantic tokens + global tokens using BiCodec, allowing a standard text LLM to directly model speech generation.

## Method

### Overall Architecture

The system consists of three parts:
1. **BiCodec**: A single-stream speech codec that encodes speech into semantic tokens (50 TPS) + global tokens (fixed at 32).
2. **Speech LLM**: A decoder-only Transformer based on Qwen2.5-0.5B that unifies text and speech token prediction.
3. **VoxBox Dataset**: A meticulously cleaned and annotated 100K-hour speech dataset.

### Key Designs

1. **BiCodec — Dual Tokenization Architecture**:

    - **Semantic Tokenizer**: Takes the average of layer 11/14/16 features of wav2vec 2.0 (XLSR-53) as input, and passes them through a convolutional encoder $E_s$ + VQ quantizer $Q_s$ to generate 50 TPS semantic tokens $\bm{z}_q$.
    - **Global Tokenizer**: Takes the Mel-spectrogram as input, passes it through an ECAPA-TDNN encoder $E_g$ + Cross-Attention with learnable queries + FSQ quantizer $Q_g$, generating a fixed length (32) of global tokens $\bm{g}_q$.
    - **Decoder**: A convolutional decoder $G$ to reconstruct waveforms from quantified semantic tokens and aggregated global embeddings.

$$\bm{z} = E_s(F(\bm{x})), \quad \bm{g} = E_g(\text{Mel}(\bm{x}))$$
$$\bm{g}_f = \text{CrossAttention}(\bm{g}, \bm{h}), \quad \hat{\bm{x}} = G(\bm{z}_q, A_g(\bm{g}_q))$$

2. **Speech Language Model**: Based on Qwen2.5-0.5B, supporting two generation modes:

    - **Zero-Shot TTS**: Predicts semantic tokens $\bm{o}$ given text prompt $\mathcal{T}$ and reference audio's global tokens $\mathcal{G}$.
    - **Attribute-Controllable Generation (CoT Style)**: Given text and coarse-grained attribute labels (gender/pitch level/speaking speed level) $\mathcal{A}$, the model sequentially predicts fine-grained attribute values $\mathcal{F}$ $\to$ global tokens $\mathcal{G}$ $\to$ semantic tokens $\mathcal{S}$ in a Chain-of-Thought manner.

3. **VoxBox Dataset**:

    - 102.5K hours of speech from 29 open-source datasets, containing 4.7 million audio files.
    - Annotations: Gender (classified via fine-tuned WavLM with 99.4% accuracy), pitch (extracted using PyWorld + Mel-scale percentile bins), and speaking speed (syllables per second (SPS) + percentile bins).
    - Data Cleaning: Original Whisper transcriptions are re-recognized using FunASR; samples with WER > 0.05 are filtered out.

### Loss & Training

- **BiCodec Training**: GAN training (Multi-Period Discriminator + Multi-Band STFT Discriminator), optimized with L1 Mel-reconstruction loss + L1 feature matching loss + VQ codebook loss + commitment loss + wav2vec 2.0 reconstruction loss. Trained for ~800K steps with a batch size of 614.4s of audio.
- **LLM Training**: Negative log-likelihood loss, with joint training on both zero-shot and attribute-controllable targets. Two training samples are constructed per audio clip. Optimized using AdamW for 3 epochs with a batch size of 768.

$$\mathcal{L}_{zst} = -\sum_{t=1}^{T_o} \log P(o_t | \mathcal{T}, \mathcal{G}, \bm{o}_{<t}; \theta_{LM})$$
$$\mathcal{L}_{control} = -\sum_{t=1}^{T_c} \log P(c_t | \mathcal{T}, \mathcal{A}, \bm{c}_{<t}; \theta_{LM})$$

## Key Experimental Results

### BiCodec Reconstruction Performance (LibriSpeech test-clean)

| Model | Bitrate (bps) | STOI↑ | PESQ-WB↑ | UTMOS↑ | SIM↑ |
|------|-----------|-------|----------|--------|------|
| Encodec (8 codebook) | 6000 | 0.94 | 2.75 | 3.07 | 0.89 |
| DAC (12 codebook) | 6000 | 0.95 | 4.01 | 4.00 | 0.98 |
| X-codec2 | 800 | 0.92 | 2.43 | 4.13 | 0.82 |
| StableCodec | 697 | 0.91 | 2.24 | 4.23 | 0.62 |
| **BiCodec** | **650** | **0.92** | **2.51** | **4.18** | **0.80** |

BiCodec achieves SOTA performance across almost all metrics in the low-bitrate range of <1kbps.

### Global Token Length Ablation

| Global Token | STOI | PESQ-WB | UTMOS | SIM |
|-----------|------|---------|-------|-----|
| w/o FSQ | 0.915 | 2.52 | 4.15 | 0.83 |
| GVQ-32 | 0.912 | 2.30 | 4.06 | 0.74 |
| FSQ-8 | 0.916 | 2.41 | 4.16 | 0.74 |
| FSQ-16 | 0.919 | 2.45 | 4.15 | 0.77 |
| **FSQ-32** | **0.922** | **2.51** | **4.18** | **0.80** |

### Zero-Shot TTS (Seed-TTS-eval)

| Model | test-zh CER↓ | test-zh SIM↑ | test-en WER↓ | test-en SIM↑ |
|------|-------------|-------------|-------------|-------------|
| Seed-TTS (Closed-source) | **1.12** | **0.796** | 2.25 | **0.762** |
| CosyVoice2 | 1.45 | 0.748 | 2.57 | 0.652 |
| F5-TTS | 1.56 | 0.741 | **1.83** | 0.647 |
| Llasa-8B-250k | 1.59 | 0.684 | 2.97 | 0.574 |
| **Spark-TTS** | 1.20 | 0.672 | 1.98 | 0.584 |

### Gender Control Accuracy

| Method | Accuracy |
|------|--------|
| VoxInstruct | 82.99% |
| Parler-TTS | 98.12% |
| **Spark-TTS** | **99.77%** |

### Speech Quality (LibriSpeech test-clean UTMOS)

| Method | GT | CosyVoice | CosyVoice2 | Spark-TTS |
|------|-----|-----------|-----------|-----------|
| UTMOS↑ | 4.08 | 4.09 | 4.23 | **4.35** |

### Key Findings

- BiCodec achieves SOTA low-bitrate reconstruction quality at 650 bps (using only 50 TPS semantic tokens and 32 global tokens).
- With only 0.5B parameters and 100K hours of data, Spark-TTS outperforms Llasa-8B (trained on 250K hours of data) in intelligibility.
- The Chinese CER of 1.20% is second only to the closed-source Seed-TTS (1.12%), and the English WER of 1.98% is second only to F5-TTS (1.83%).
- Speaker similarity (SIM) is a weakness (0.672/0.584), which is lower than multi-stage or NAR (non-autoregressive) methods. This is an inherent limitation of single-stage AR methods.
- The confusion matrices for pitch and speaking speed control show high 5-level classification control accuracy (heavily concentrated on the diagonal).
- The synthesized speech quality UTMOS is 4.35, exceeding the ground-truth audio's 4.08, indicating a "beautifying" effect from the model.
- The FSQ + learnable query scheme significantly outperforms the GVQ scheme (SIM: 0.80 vs 0.74).

## Highlights & Insights

- **Elegant and Minimalist Architecture**: BiCodec decomposes speech into semantic (time-varying) and global (time-invariant) tokens. The entire TTS pipeline is fully aligned with standard text LLMs, eliminating the need for flow matching or diffusion models.
- **Ingenious CoT Generation Strategy**: The chain-of-thought reasoning (coarse-grained labels $\to$ fine-grained values $\to$ global tokens $\to$ semantic tokens) enables hierarchical attribute control.
- **VoxBox Dataset Contribution**: The fully open-sourced 100K hours of speech with annotations for gender, pitch, and speed fills a gap in training data for controllable TTS.
- **Efficiency Advantage**: The 0.5B model outperforms 8B baseline models. The low frame rate of 50 TPS substantially reduces sequence lengths.

## Limitations & Future Work

1. Speaker similarity (SIM) is lower than multi-stage methods; speaker variability introduced during AR language model inference remains the primary cause.
2. No explicit decoupling constraints are imposed between global tokens and semantic tokens. Strengthening decoupling through formant/pitch perturbations could be explored.
3. BiCodec is trained on only ~3,000 hours of data. Scaling up training data could further enhance reconstruction quality.
4. Lack of emotional control. Although VoxBox contains voice datasets with emotional expressions, they are not explicitly utilized.
5. Support is limited to English and Chinese; multilingual extension remains to be explored.

## Related Work & Insights

- TiCodec (Ren et al., 2024) is the most similar, but it uses GVQ for global information. BiCodec's use of FSQ + Cross-Attention with learnable queries performs better.
- Llasa (Ye et al., 2025) employs a scheme with FSQ on a single codebook + LLaMA, but lacks separation of global tokens, requiring a larger model (8B) to achieve comparable performance.
- CosyVoice2 requires an additional flow matching step to predict acoustic features, whereas Spark-TTS reconstructs audio directly via the BiCodec decoder, offering a simpler architecture.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The decoupling of semantic and global tokens in BiCodec alongside CoT-controlled generation is elegantly designed and highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments on codec reconstruction, zero-shot TTS, and control, though more analysis on the weakness of speaker similarity is needed.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with a complete system description.
- **Value**: ⭐⭐⭐⭐⭐ — Fully open-sourced code, model, and dataset. The 100K-hour VoxBox dataset is a major contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2026\] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](../../ACL2026/audio_speech/data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](soundwave_less_is_more_for_speech-text_alignment_in_llms.md)

</div>

<!-- RELATED:END -->
