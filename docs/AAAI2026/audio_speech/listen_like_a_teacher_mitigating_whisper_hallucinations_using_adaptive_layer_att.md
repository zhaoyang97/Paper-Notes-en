---
title: >-
  [Paper Note] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation
description: >-
  [AAAI2026][Audio & Speech][Speech Recognition] A two-stage framework is proposed: Adaptive Layer Attention (ALA) fuses multi-layer representations from the Whisper encoder to enhance noise robustness, while Multi-Objective Knowledge Distillation (MOKD) aligns the semantic and attention distributions of a clean-speech teacher with a noisy-speech student — achieving significant reductions in hallucination rate and WER on multilingual noisy ASR benchmarks.
tags:
  - AAAI2026
  - "Audio & Speech"
  - Speech Recognition
  - Whisper
  - Hallucination Mitigation
  - Adaptive Layer Attention
  - Knowledge Distillation
  - Noise Robustness
date: 2026-05-08
content_hash: 2dc55e4cb31af6b2
---

# Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation

**Conference**: AAAI2026
**arXiv**: [2511.14219](https://arxiv.org/abs/2511.14219)
**Authors**: Kumud Tripathi, Aditya Srinivas Menon, Aman Gaurav, Raj Prakash Gohil, Pankaj Wasnik
**Code**: Not released
**Area**: Audio & Speech
**Keywords**: Speech Recognition, Whisper, Hallucination Mitigation, Adaptive Layer Attention, Knowledge Distillation, Noise Robustness

## TL;DR

A two-stage framework is proposed: Adaptive Layer Attention (ALA) fuses multi-layer representations from the Whisper encoder to enhance noise robustness, while Multi-Objective Knowledge Distillation (MOKD) aligns the semantic and attention distributions of a clean-speech teacher with a noisy-speech student — achieving significant reductions in hallucination rate and WER on multilingual noisy ASR benchmarks.

## Background & Motivation

### Hallucination in Whisper

Whisper is an end-to-end ASR model open-sourced by OpenAI that achieves strong performance in multilingual and zero-shot settings. However, it frequently produces hallucinations under noisy or non-speech conditions, generating fluent yet semantically incorrect transcriptions. Such hallucinations often evade standard metrics such as WER and critically undermine the reliability of speech systems. Research indicates that hallucinations primarily stem from misalignment between encoder and decoder internal representations when processing noisy inputs.

### Limitations of Prior Work

Existing work on mitigating Whisper hallucinations has largely focused on downstream processing: pre-processing with voice activity detection (VAD) to filter non-speech segments, post-processing to filter erroneous transcriptions, and data augmentation. However, these approaches do not address the root cause at the level of internal model representations. Direct modifications to the Whisper architecture — reducing hallucinations at the encoder and decoder level — remain largely unexplored.

### Motivation for Multi-Layer Feature Fusion

Each layer of a Transformer encoder captures features at different levels of abstraction (lower layers are more acoustic, higher layers more semantic), yet conventional ASR systems use only the final layer output, discarding the rich information carried by intermediate layers. Under noisy conditions, certain encoder layers may capture corrupted signals, and relying solely on the final layer leads to performance degradation. Adaptively fusing multi-layer representations to improve encoder robustness therefore represents a natural direction for improvement.

## Core Problem

How can ASR hallucinations under noisy conditions be suppressed simultaneously at the representation and attention levels — without altering Whisper's fundamental architecture — by (1) enhancing adaptive fusion of multi-layer encoder representations, and (2) leveraging a clean-speech teacher to guide attention alignment in a noisy-speech student?

## Method

### Overall Architecture

A two-stage architecture: Stage 1 augments the encoder with an Adaptive Layer Attention (ALA) module; Stage 2 applies Multi-Objective Knowledge Distillation (MOKD) on the ALA-enhanced model, using a clean-speech teacher to supervise a noisy-speech student.

### Stage 1: Adaptive Layer Attention (ALA)

**Inter-layer similarity analysis**: Cosine similarities among all encoder layer outputs are computed, revealing that layers naturally cluster into functional blocks. Using Whisper-small's 12 layers as an example: L1–L6 form a low-level acoustic feature block, L7–L11 form a high-level semantic block, and L12 deviates markedly from the others as it is specifically optimized as input to the decoder.

**Block mean pooling**: Layers are partitioned into $K$ blocks $\{B_1, B_2, \ldots, B_K\}$, and a mean representation is computed for each block:

$$r_k = \frac{1}{|B_k|} \sum_{l \in B_k} e_l$$

**Adaptive multi-head attention fusion**: After injecting positional encodings into the block mean representations, the final encoder hidden state serves as the query, and multi-head attention dynamically fuses the block representations:

$$h_t = \text{MHA}(q_t, Z, Z)$$

The output is passed to the decoder via a residual connection and normalization, enabling the model to adaptively select the most informative layer blocks under noisy conditions.

### Stage 2: Multi-Objective Knowledge Distillation (MOKD)

A clean-speech-trained ASR model serves as the teacher, and the ALA-enhanced model trained on noisy speech serves as the student. The total loss comprises four terms:

1. **Encoder cosine similarity loss**: $\mathcal{L}_{\text{Enc\_Cos}} = \sum_{t=1}^T (1 - \cos(e_t^T, e_t^S))$
2. **Decoder cosine similarity loss**: $\mathcal{L}_{\text{Dec\_Cos}} = \sum_{t=1}^T (1 - \cos(d_t^T, d_t^S))$
3. **Decoder cross-attention MSE loss**: $\mathcal{L}_{\text{Dec\_MSE}} = \sum_{t=1}^T \|d_t^T - d_t^S\|_2^2$
4. **Cross-entropy loss**: $\mathcal{L}_{\text{CE}} = -\sum_{t=1}^T \log P_S(y_t)$

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Enc\_Cos}} + \lambda_2 \mathcal{L}_{\text{Dec\_Cos}} + \lambda_3 \mathcal{L}_{\text{Dec\_MSE}} + \lambda_4 \mathcal{L}_{\text{CE}}$$

Grid search determines $\lambda_1=0.8$; all remaining weights are set to 1.0.

## Key Experimental Results

Base model: Whisper-small (W-SS). Datasets: Hindi (Kathbath), Arabic/French (CommonVoice-15), English (LibriSpeech-100). Noise is sourced from the DEMAND database; training SNR ranges from $-8$ to $+4$ dB, and test SNR from $-10$ to $+10$ dB.

### Stage 1: ALA Performance (WER↓ / SeMaScore↑, language average)

| Language | Model | SNR -10 | SNR 0 | Clean | Avg |
|----------|-------|---------|-------|-------|-----|
| Hindi | Baseline-2 | 42.77/0.803 | 18.05/0.937 | 12.77/0.964 | 21.44/0.918 |
| Hindi | W-ALA | **40.74/0.826** | **16.07/0.945** | **11.41/0.967** | **19.64/0.928** |
| English | Baseline-2 | 39.64/0.876 | 7.21/0.971 | 3.44/0.985 | 12.46/0.957 |
| English | W-ALA | **29.68/0.877** | **5.85/0.973** | **3.19/0.987** | **9.68/0.958** |

### Stage 2: MOKD Performance (WER↓ / SeMaScore↑)

| Language | Model | SNR -10 | SNR 0 | Clean | Avg |
|----------|-------|---------|-------|-------|-----|
| Hindi | Baseline-2 | 42.77/0.803 | 18.05/0.937 | 12.77/0.964 | 21.44/0.918 |
| Hindi | W-MOKD | **38.13/0.846** | **14.83/0.958** | **11.23/0.968** | **18.61/0.943** |
| English | Baseline-2 | 39.64/0.876 | 7.21/0.971 | 3.44/0.985 | 12.46/0.957 |
| English | W-MOKD | **26.43/0.905** | **5.72/0.984** | **3.18/0.984** | **8.56/0.969** |

### Efficiency Overhead

| Model | Latency (ms) | RTF | Peak VRAM (GB) |
|-------|-------------|-----|----------------|
| Baseline-2 | 140±10 | 0.021 | 1.5 |
| W-ALA | 152±11 | 0.023 | 2.6 |

ALA adds only 0.98% parameters, with an 8% increase in latency and a 9% increase in RTF.

### Ablation Study: Encoder Fusion Strategy Comparison (Hindi, SNR -10 / Clean)

| Fusion Method | SNR -10 WER/SeMa | Clean WER/SeMa |
|---------------|-----------------|----------------|
| Weighted Sum | 75.85/0.482 | 29.56/0.752 |
| MHA all frozen | 52.73/0.545 | 15.62/0.893 |
| MHA all trainable | 45.12/0.690 | 14.87/0.929 |
| MHA Mean (block mean) | **40.74/0.826** | **11.41/0.967** |

## Highlights & Insights

- **Addressing hallucinations from within the model**: Unlike existing pre/post-processing approaches, this work directly suppresses the root causes of hallucination at both the encoder representation and decoder attention levels.
- **Elegant adaptive layer fusion design**: Inter-layer similarity analysis naturally identifies functional blocks; block mean pooling combined with MHA fusion preserves multi-level information while mitigating interference from noisy layers, at a cost of only 0.98% additional parameters.
- **Comprehensive alignment via multi-objective KD**: Simultaneous alignment of encoder representations, decoder representations, and cross-attention maps proves substantially more effective than single-objective logit distillation.
- **Multilingual generalization**: Consistent improvements across Hindi, Arabic, French, and English demonstrate that the method does not rely on language-specific properties.
- **Interpretable attention analysis**: Block 0 receives an average attention weight of 58.6% under noisy conditions, confirming that the model learns to prioritize noise-robust low-level features.

## Limitations & Future Work

- **Validation limited to Whisper-small**: No experiments are conducted on larger models such as Whisper-medium or Whisper-large, leaving generalizability unclear.
- **Limited noise diversity**: The DEMAND database covers a restricted range of noise types; complex acoustic conditions such as reverberation and far-field recording are not considered.
- **Fixed ALA block partitioning**: The layer grouping strategy is determined via offline similarity analysis and may require different partitioning schemes for different languages or tasks.
- **No comparison with hallucination detection methods**: A joint evaluation with post-processing hallucination detection approaches is absent.

## Related Work & Insights

- **Distil-Whisper** (2023): Reduces parameters and incidentally lowers hallucinations via pseudo-label KD; however, Table 3 of the paper shows that Baseline-3 (Distil-Whisper) performs substantially worse than W-MOKD in multilingual noisy settings.
- **Differential Transformer**: Reduces noisy attention in decoder attention heads via a subtraction operation; this is complementary to ALA's encoder-level layer fusion.
- **MLCA-AVSR**: Fuses audio-visual features through multi-layer cross-attention for robustness enhancement; ALA transfers a similar idea to the internal encoder of single-modality ASR.
- **A2D (Align-to-Distill)**: Attention alignment distillation applied to low-resource NMT; this work extends the paradigm to ASR hallucination suppression.

The inter-layer similarity analysis and block fusion design of ALA can be generalized to other Transformer models (e.g., LLMs, Vision Transformers) for dynamic exploitation of complementary information across layers. The clean-teacher/noisy-student KD paradigm is likewise applicable to other robustness tasks, such as robust NMT and robust image classification. Combining ALA with hallucination detection post-processing may further improve practical utility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The inter-layer similarity analysis and block fusion design of ALA are novel; the multi-objective KD combination is well-motivated, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments across four languages and multiple SNR levels with thorough ablations; validation on larger models and more diverse noise types is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with logically progressive two-stage presentation and rich experimental tables.
- **Value**: ⭐⭐⭐⭐ — Addressing ASR hallucinations from within the model is an important direction; the method is practical, introduces minimal overhead, and holds strong engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] TellWhisper: Tell Whisper Who Speaks When](../../ACL2026/audio_speech/tellwhisper_tell_whisper_who_speaks_when.md)
- [\[AAAI 2026\] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment](multi-granularity_interactive_attention_framework_for_residual_hierarchical_pron.md)
- [\[ICLR 2026\] TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](../../ICLR2026/audio_speech/triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)

<!-- RELATED:END -->
