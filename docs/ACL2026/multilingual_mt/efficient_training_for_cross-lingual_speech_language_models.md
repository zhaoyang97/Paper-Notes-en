---
title: >-
  [Paper Note] Efficient Training for Cross-lingual Speech Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Cross-lingual speech LLM] This paper proposes CSLM, a data-efficient training method for cross-lingual speech LLMs. It achieves cross-modality and cross-lingual alignment th…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Cross-lingual speech LLM"
  - "discrete speech tokens"
  - "modality alignment"
  - "interleaved chain-of-modality generation"
  - "data-efficient training"
date: 2026-05-08
content_hash: 0262c4b3557d9497
---

# Efficient Training for Cross-lingual Speech Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.11096](https://arxiv.org/abs/2604.11096)  
**Code**: [https://github.com/ictnlp/CSLM](https://github.com/ictnlp/CSLM)  
**Area**: Multilingual/Translation / Audio Speech  
**Keywords**: Cross-lingual speech LLM, discrete speech tokens, modality alignment, interleaved chain-of-modality generation, data-efficient training

## TL;DR
This paper proposes CSLM, a data-efficient training method for cross-lingual speech LLMs. It achieves cross-modality and cross-lingual alignment through a novel alignment strategy and introduces interleaved speech-text chain-of-modality generation to enhance quality and reduce latency, enabling expansion to new languages without large-scale speech data.

## Background & Motivation

**Background**: Speech LLMs are emerging to enable more natural human-computer interaction, but building effective end-to-end models remains challenging. Existing methods include cascaded ASR+LLM+TTS (suffering from error accumulation and high latency), modular encoder+LLM approaches (limited speech generation capability), and unified modeling based on discrete speech tokens (e.g., SpeechGPT, GLM-4-Voice).

**Limitations of Prior Work**: (1) Speech data is extremely scarce compared to text, especially for certain languages; (2) Existing unified modeling methods (e.g., GLM-4-Voice, Moshi) require massive amounts of training data; (3) Expanding speech LLMs to more languages faces the double challenge of data scarcity and training difficulty; (4) Existing chain-of-modality generation (TQ → full TA → full SA) results in high latency.

**Key Challenge**: Constructing a unified multilingual multimodal representation usually requires vast amounts of data, yet speech data is severely lacking for many languages. Simultaneously achieving cross-lingual and cross-modal alignment with limited data is the core challenge.

**Goal**: Design a data-efficient training method that achieves both cross-modality and cross-lingual alignment using limited speech data while ensuring good language scalability.

**Key Insight**: Leverage the text modality as a "bridge" for cross-lingual alignment—performing cross-modal alignment between speech and text within a single language via ASR/TTS data, and cross-lingual alignment through machine translation (text-to-text) data. This eliminates the need for parallel cross-lingual speech-to-speech data.

**Core Idea**: Design an "interleaved speech-text chain-of-modality" generation approach where the model alternates between generating short text chunks and corresponding speech chunks (TQ → TA → SA → TA → SA...). This provides finer-grained modality alignment and lower latency than full chain-of-modality generation (TQ → full TA → full SA).

## Method

### Overall Architecture
CSLM consists of three components: (1) A CosyVoice speech tokenizer (4096 vocabulary, 25Hz) that converts speech into discrete tokens; (2) A joint speech-text LLM (merging speech and text vocabularies); (3) A speech decoder (flow-matching model + HiFi-GAN vocoder). Training follows a two-stage paradigm: continual pre-training and supervised fine-tuning.

### Key Designs

1. **Cross-modal + Cross-lingual Alignment Strategy**:

    - **Function**: Achieve speech-text and cross-lingual alignment simultaneously with limited data.
    - **Mechanism**: Within a language, cross-modal alignment is achieved via ASR data (speech → text) and TTS data (text → speech). Cross-lingual alignment is achieved through machine translation data (CN ↔ EN text). Text serves as the bridge—once speech and text are aligned within each language, indirect cross-lingual speech alignment is realized through inter-language text translation. Monolingual instruction data is also used to prevent text capability degradation.
    - **Design Motivation**: Paired cross-lingual speech-to-speech data is hard to obtain, whereas ASR/TTS and translation data are more accessible. Bridging through text avoids reliance on cross-lingual speech data.

2. **Interleaved Speech-text Chain-of-modality Generation**:

    - **Function**: Enable finer-grained modality alignment during fine-tuning while reducing inference latency.
    - **Mechanism**: The original chain-of-modality (TQ → full TA → full SA) is replaced with an interleaved format: the model generates a small text response chunk and immediately generates the corresponding speech chunk, repeating until completion (TQ → TA → SA → TA → SA...). A CTC aligner builds interleaved data from existing speech-text pairs by finding the optimal alignment path $\pi^* = \arg\max_\pi \prod_t P(\pi_t|\mathbf{h}_t)$ to obtain token-level time boundaries and splitting at punctuation based on chunk size (7 words).
    - **Design Motivation**: Full chain-of-modality requires the entire text to be generated before speech starts, causing high latency. Interleaved generation allows overlapping of generation and playback—while the current speech chunk plays, the model generates subsequent content. Chunk-level interleaving produces fewer errors than word-level interleaving.

3. **Language Scalability Design**:

    - **Function**: Ensure the training method is easily extensible to new languages in terms of data volume and training difficulty.
    - **Mechanism**: As long as the target language has (1) paired speech-text data (for modality alignment) and (2) translation data (for language alignment), it can be integrated into CSLM. Discrete tokens are language-independent, and the CosyVoice tokenizer itself supports multiple languages.
    - **Design Motivation**: Minimizing data requirements—no large-scale target-language monolingual speech data or cross-lingual speech-to-speech pairs are needed. This allows scaling to low-resource languages.

### Loss & Training
Two-stage training: (1) Continual Pre-training—starting from an instruction-finetuned LLM, the speech vocabulary is merged, and the model is trained on a mix of ASR/TTS/MT/monolingual instruction data to obtain CSLM-base; (2) Supervised Fine-tuning—training on text instructions and speech dialogue data to obtain CSLM-SFT using the interleaved format. Consecutive identical speech tokens are merged before entering the LLM for efficiency.

## Key Experimental Results

### Main Results

| Task | Model | English | Chinese |
|------|------|------|------|
| ASR (WER↓) | Whisper-large-v3 | 2.5 | 9.3 |
| ASR | GLM-4-Voice | 2.8 | 2.5 |
| ASR | CSLM-SFT | 9.8 | 9.0 |
| TTS (WER↓) | CosyVoice-SFT | 3.4 | — |
| TTS | GLM-4-Voice | 4.7 | — |
| TTS | CSLM-SFT | **3.8** | — |
| TTS (LibriTTS) | CSLM-SFT | **2.9** | — |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full chain-of-modality | High latency | TQ → full TA → full SA |
| Interleaved chain-of-modality | Low latency, better quality | TQ → TA → SA → TA → SA... |
| w/o cross-lingual alignment | Poor cross-lingual tasks | Lacks translation data bridging |
| w/o modality alignment | Poor speech quality | Lacks ASR/TTS training |

### Key Findings
- CSLM approaches or exceeds dedicated TTS systems (CosyVoice) in speech quality while possessing dialogue and cross-lingual capabilities.
- Interleaved chain-of-modality significantly reduces latency by overlapping generation and audio playback.
- CSLM achieves comparable performance to GLM-4-Voice despite using significantly less speech data.
- Chunk-level interleaved data constructed via the CTC aligner is more stable than word-level interleaving.
- ASR performance is inferior to dedicated models (Whisper) but remains sufficient for dialogue scenarios.

## Highlights & Insights
- **Text as a Cross-lingual Bridge**: Cleverly leverages rich text resources to bridge speech across different languages, avoiding the need for parallel cross-lingual speech data. This concept is valuable for all multilingual multimodal systems.
- **Latency Optimization via Interleaved Generation**: Optimizing latency through alternating text and speech generation to achieve generation-playback overlap is a practical and elegant solution.
- **Data Construction via CTC Aligner**: Using the CTC module of existing ASR models to obtain precise speech-text alignment allows for automatic construction of interleaved training data, avoiding manual alignment efforts.

## Limitations & Future Work
- ASR performance is significantly weaker than dedicated Whisper models, suggesting unified modeling still has a gap in understanding tasks.
- Experiments were only conducted on Chinese-English; the scalability to more languages remains untested.
- The performance of the speech tokenizer (CosyVoice) directly impacts the system; replacing it might yield improvements.
- The chunk size for interleaved generation (7 words) was selected manually; adaptive chunk partitioning could be explored.

## Related Work & Insights
- **vs GLM-4-Voice**: GLM-4-Voice is the first CN-EN speech LLM but requires massive data. CSLM achieves comparable results with much less data.
- **vs SPIRIT LM / Moshi**: Unified modeling methods that require large speech datasets. CSLM's efficient alignment strategy significantly reduces data requirements.
- **vs LLaMA-Omni**: Modular approaches (Encoder+LLM+TTS) are limited in speech quality and diversity. CSLM provides more natural speech through unified modeling with discrete tokens.

## Rating
- Novelty: ⭐⭐⭐⭐ Interleaved chain-of-modality and text-bridged alignment are novel and practical designs.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple tasks but only includes two languages, and data scale comparisons are not sufficiently detailed.
- Writing Quality: ⭐⭐⭐⭐ Clear framework; visualization of alignment strategies aids understanding.
- Value: ⭐⭐⭐⭐ Provides a feasible training path for speech LLMs in low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)

</div>

<!-- RELATED:END -->
