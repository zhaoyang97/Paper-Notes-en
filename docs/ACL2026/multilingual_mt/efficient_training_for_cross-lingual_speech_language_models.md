---
title: >-
  [Paper Note] Efficient Training for Cross-lingual Speech Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][cross-lingual speech LLM] This paper proposes CSLM, a data-efficient method for training cross-lingual speech LLMs. It introduces a novel alignment strategy to achieve cross…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "cross-lingual speech LLM"
  - "discrete speech tokens"
  - "modality alignment"
  - "chain-of-modality generation"
  - "data-efficient training"
date: 2026-05-08
content_hash: c3841653fa822dcc
---

# Efficient Training for Cross-lingual Speech Language Models

**Conference**: ACL 2026
**arXiv**: [2604.11096](https://arxiv.org/abs/2604.11096)  
**Code**: [https://github.com/ictnlp/CSLM](https://github.com/ictnlp/CSLM)  
**Area**: Multilingual/Translation / Audio & Speech
**Keywords**: cross-lingual speech LLM, discrete speech tokens, modality alignment, chain-of-modality generation, data-efficient training

## TL;DR
This paper proposes CSLM, a data-efficient method for training cross-lingual speech LLMs. It introduces a novel alignment strategy to achieve cross-modal and cross-lingual alignment simultaneously, and presents a speech-text interleaved chain-of-modality generation paradigm to improve quality and reduce latency—without requiring large-scale speech data to extend to new languages.

## Background & Motivation

**Background**: Speech LLMs are emerging to enable more natural human-computer interaction, yet building effective end-to-end speech LLMs remains challenging. Existing approaches include: cascaded ASR+LLM+TTS (suffering from error propagation and high latency), modular encoder+LLM methods (with weak speech generation capability), and unified modeling based on discrete speech tokens (e.g., SpeechGPT, GLM-4-Voice).

**Limitations of Prior Work**: (1) Speech data is extremely scarce relative to text, especially for certain languages; (2) existing unified modeling approaches (e.g., GLM-4-Voice, Moshi) require massive amounts of training data; (3) extending speech LLMs to more languages faces the dual challenge of data scarcity and training difficulty; (4) existing chain-of-modality generation (TQ→full TA→full SA) incurs high latency.

**Key Challenge**: Building unified multilingual multimodal representations typically requires large amounts of data, yet speech data is severely insufficient for many languages. The core challenge is how to achieve cross-lingual and cross-modal alignment simultaneously with limited data.

**Goal**: Design a data-efficient training method that achieves cross-modal and cross-lingual alignment using limited speech data while remaining easily extensible to new languages.

**Key Insight**: Use the text modality as a "bridge" for cross-lingual alignment—within a single language, cross-modal speech-text alignment is achieved via ASR/TTS data; cross-lingual alignment is achieved via machine translation data (text-to-text). This eliminates the need for cross-lingual speech-speech paired data.

**Core Idea**: Introduce a speech-text interleaved chain-of-modality generation paradigm, in which the model alternately generates short text chunks and their corresponding speech chunks (TQ→TA→SA→TA→SA…). Compared to the full chain-of-modality format (TQ→full TA→full SA), this achieves finer-grained modality alignment and lower latency.

## Method

### Overall Architecture
CSLM consists of three components: (1) a CosyVoice speech tokenizer (vocabulary size 4096, 25 Hz) that converts speech into discrete tokens; (2) a joint speech-text LLM with a merged vocabulary; and (3) a speech decoder (flow matching model + HiFi-GAN vocoder). Training follows a two-stage paradigm: continual pre-training followed by supervised fine-tuning.

### Key Designs

1. **Cross-Modal and Cross-Lingual Alignment Strategy**

    - **Function**: Simultaneously achieve speech-text alignment and cross-lingual alignment with limited data.
    - **Mechanism**: Within a single language, cross-modal alignment is established via ASR data (speech→text) and TTS data (text→speech). Cross-lingual alignment is established via machine translation data (Chinese↔English text). The text modality serves as a bridge—once each language's speech is aligned to text, cross-lingual speech alignment is achieved indirectly through inter-text translation alignment. Monolingual instruction data is also incorporated to prevent degradation of text capabilities.
    - **Design Motivation**: Obtaining cross-lingual speech-speech paired data directly is difficult, whereas ASR/TTS paired data and translation data are relatively easy to acquire. Bridging through text avoids the dependency on cross-lingual speech data.

2. **Speech-Text Interleaved Chain-of-Modality Generation**

    - **Function**: Enable finer-grained modality alignment during fine-tuning while reducing inference latency.
    - **Mechanism**: The original chain-of-modality format (TQ→full TA→full SA) is replaced by an interleaved format: the model generates a short text response chunk, immediately generates the corresponding speech chunk, and repeats until completion, i.e., TQ→TA→SA→TA→SA…. A CTC aligner is used to construct interleaved training data from existing speech-text pairs—the optimal alignment path is found via CTC dynamic programming, $\pi^* = \arg\max_\pi \prod_t P(\pi_t|\mathbf{h}_t)$, yielding token-level time boundaries, which are then segmented into chunks (7 words) at punctuation boundaries.
    - **Design Motivation**: Full chain-of-modality generation must wait for the complete text before beginning speech generation, resulting in high latency. Interleaved generation achieves temporal overlap between generation and playback—while one speech chunk is being played, the model is already generating subsequent content. Chunk-level text-speech interleaving is more stable than word-level interleaving.

3. **Language Extensibility Design**

    - **Function**: Enable the training method to be easily extended to new languages with minimal data and training overhead.
    - **Mechanism**: Integrating a new language into CSLM requires only (1) speech-text paired data (for cross-modal alignment) and (2) translation data (for cross-lingual alignment). Discrete speech tokens are language-agnostic, and the CosyVoice tokenizer natively supports multiple languages.
    - **Design Motivation**: Data requirements are minimized—large-scale monolingual speech data or cross-lingual speech paired data for the target language are not required; only ASR/TTS pairs and translation data are needed. This makes extension to low-resource languages feasible.

### Loss & Training
Two-stage training: (1) **Continual pre-training**—starting from an instruction-tuned LLM, the speech vocabulary is merged and mixed training is conducted on ASR/TTS/MT/monolingual instruction data to obtain CSLM-base; (2) **Supervised fine-tuning**—CSLM-SFT is trained on text instruction and speech dialogue data using the interleaved chain-of-modality format. Consecutive repeated speech tokens are merged before being fed to the LLM to improve efficiency.

## Key Experimental Results

### Main Results

| Task | Model | English | Chinese |
|------|-------|---------|---------|
| ASR (WER↓) | Whisper-large-v3 | 2.5 | 9.3 |
| ASR | GLM-4-Voice | 2.8 | 2.5 |
| ASR | CSLM-SFT | 9.8 | 9.0 |
| TTS (WER↓) | CosyVoice-SFT | 3.4 | — |
| TTS | GLM-4-Voice | 4.7 | — |
| TTS | CSLM-SFT | **3.8** | — |
| TTS (LibriTTS) | CSLM-SFT | **2.9** | — |

### Ablation Study

| Configuration | Effect | Notes |
|--------------|--------|-------|
| Full chain-of-modality | High latency | TQ→full TA→full SA |
| Interleaved chain-of-modality | Low latency, better quality | TQ→TA→SA→TA→SA… |
| w/o cross-lingual alignment | Poor cross-lingual performance | Lacks translation data bridging |
| w/o cross-modal alignment | Poor speech quality | Lacks ASR/TTS training |

### Key Findings
- CSLM achieves TTS quality comparable to or exceeding dedicated TTS systems (CosyVoice) while simultaneously supporting dialogue and cross-lingual capabilities.
- Interleaved chain-of-modality significantly reduces latency—the model generates subsequent content while the previous audio chunk is still being played.
- CSLM achieves comparable performance to GLM-4-Voice using substantially less speech data (the latter relies on massive-scale data).
- Chunk-level interleaved data constructed by the CTC aligner is more stable than word-level interleaving.
- ASR performance falls short of dedicated ASR models (Whisper), but is sufficient for conversational scenarios.

## Highlights & Insights
- **Text as a cross-lingual bridge**: The approach cleverly leverages the rich resources of the text modality to bridge speech across languages, eliminating the dependency on cross-lingual speech paired data. This idea generalizes to all multilingual multimodal systems.
- **Latency optimization via interleaved chain-of-modality**: Alternating text and speech generation to achieve generation-playback overlap is a practical and elegant solution for latency reduction.
- **CTC aligner for training data construction**: The CTC module of existing ASR models is used to obtain precise speech-text alignment, enabling automatic construction of interleaved training data without manual annotation.

## Limitations & Future Work
- ASR performance is significantly weaker than dedicated Whisper models, indicating that unified modeling still lags behind on comprehension tasks.
- Evaluation is limited to Chinese-English bilingual settings; extension to more languages has not been tested.
- System performance is directly constrained by the speech tokenizer (CosyVoice); replacing the tokenizer may yield further improvements.
- The chunk size for interleaved generation (7 words) is selected manually; adaptive chunk segmentation strategies merit further exploration.

## Related Work & Insights
- **vs. GLM-4-Voice**: GLM-4-Voice is the first Chinese-English bilingual speech LLM but requires massive data. CSLM achieves comparable results with substantially less data.
- **vs. SPIRIT LM / Moshi**: Unified modeling approaches that require large amounts of speech data. CSLM's efficient alignment strategy substantially reduces data requirements.
- **vs. LLaMA-Omni**: A modular approach (encoder+LLM+TTS) with limited speech quality and diversity. CSLM provides more natural speech through unified discrete-token modeling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The interleaved chain-of-modality and text-bridged alignment strategy are novel and practical designs.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers multiple tasks but is limited to bilingual evaluation; comparisons of data scale lack sufficient detail.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clearly presented; visualizations of the alignment strategy aid understanding.
- **Value**: ⭐⭐⭐⭐ Provides a viable training pathway for speech LLMs in low-resource language settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[AAAI 2026\] GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](../../AAAI2026/multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)

</div>

<!-- RELATED:END -->
