---
title: >-
  [Paper Note] Recent Advances in Speech Language Models: A Survey
description: >-
  [ACL 2025][LLM (Other)][speech language model] The first comprehensive survey on Speech Language Models (SpeechLMs), systematically tracing the evolution from "ASR+LLM+TTS" cascaded architectures to end-to-end speech language models. It proposes a taxonomy categorized by three key components (speech tokenizer / language model / vocoder) and training strategies, and covers downstream capabilities, evaluation metrics, challenges, and future directions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "speech language model"
  - "end-to-end speech"
  - "speech tokenizer"
  - "vocoder"
  - "survey"
date: 2026-05-08
content_hash: d9e38b3cb1fda894
---

# Recent Advances in Speech Language Models: A Survey

**Conference**: ACL 2025  
**arXiv**: [2410.03751](https://arxiv.org/abs/2410.03751)  
**Code**: [GitHub](https://github.com/dreamtheater123/Awesome-SpeechLM-Survey)  
**Area**: Speech/LLM  
**Keywords**: speech language model, end-to-end speech, speech tokenizer, vocoder, survey

## TL;DR
The first comprehensive survey on Speech Language Models (SpeechLMs), systematically tracing the evolution from "ASR+LLM+TTS" cascaded architectures to end-to-end speech language models. It proposes a taxonomy categorized by three key components (speech tokenizer / language model / vocoder) and training strategies, and covers downstream capabilities, evaluation metrics, challenges, and future directions.

## Background & Motivation

**Background**: LLMs perform exceptionally in text interaction, but natural human-computer interaction relies on speech. The traditional "ASR+LLM+TTS" three-stage cascade, though intuitive, suffers from three primary issues: (a) information loss (paralinguistic information such as tone/emotion is lost in text); (b) high latency (due to the three stages running serially); (c) cascaded error accumulation (ASR errors propagate to the LLM and subsequently to the TTS).

**Limitations of Prior Work**: There is a lack of a systematic survey in the SpeechLM field. Existing surveys either focus on traditional speech technologies (SLU/SSL) or pay attention to speech as a subset of multimodal LLMs, lacking a holistic overview centered on "end-to-end speech language models".

**Key Challenge**: Despite the rapid development of SpeechLMs (such as GPT-4o voice, Moshi, etc.), the research community lacks a systematic understanding of their architectural choices, training strategies, and capability boundaries.

**Goal**: To provide the first comprehensive survey of the SpeechLM domain, covering architectural components, training schemes, capability taxonomies, and evaluation systems.

## Method

### Formal Definition of SpeechLM
SpeechLM is an autoregressive foundational model that directly processes and generates speech sequences $\mathbf{M}^{\text{out}} = \text{SpeechLM}(\mathbf{M}^{\text{in}}; \theta)$, where $\mathbf{M}$ can be speech, text, or interleaved multimodal sequences.

### Three Core Components

1. **Speech Tokenizer**
    - **Function**: Converts continuous audio waveforms into discrete tokens for autoregressive language modeling.
    - **Three types**:
     - **Semantic tokenizer**: e.g., HuBERT/wav2vec 2.0 + k-means quantization; extracts semantic features but loses paralinguistic details.
     - **Acoustic tokenizer**: e.g., EnCodec/SoundStream; uses RVQ (Residual Vector Quantization) to preserve acoustic details (timbre/pitch), though semantics may be diluted.
     - **Hybrid tokenizer**: combines both approaches (e.g., SpeechTokenizer decoupling semantic and acoustic layers), capturing both semantics and paralinguistics.
    - Key trade-off: Semantic tokens represent high-level abstraction beneficial for understanding, while acoustic tokens represent low-level details beneficial for generation.

2. **Language Model (LM Backbone)**
    - **Function**: Performs next-token prediction on speech tokens, acting as the core "brain".
    - **Integration Approaches**:
     - **Direct modeling**: Pre-training a decoder-only Transformer on speech tokens (e.g., GSLM, AudioPaLM).
     - **Adapting existing TextLMs**: Freezing the LLM and attaching a speech adapter (e.g., Qwen-Audio, SALMONN).
     - **Joint training**: Mixed training on both text and speech tokens (e.g., Spirit-LM interleaving text/speech tokens).
    - Multi-stream generation: Single-stream autoregressive vs. multi-stream parallel decoding (e.g., VALL-E uses a 2-stage approach: AR to generate coarse tokens $\rightarrow$ NAR to complete fine tokens).

3. **Vocoder**
    - **Function**: Converts tokens or representations output by the LM into audio waveforms.
    - **Main Methods**:
     - **HiFi-GAN family**: Direct conversion from mel-spectrogram/tokens to waveform (fast).
     - **Diffusion models**: e.g., DiffWave, high-quality but slow.
     - **Token decoder**: e.g., EnCodec decoder directly converts RVQ tokens into waveforms.

### Taxonomy of Training Schemes

| Stage | Method | Representative Works |
|------|------|---------|
| Pre-training | Speech continuation (next-token prediction on speech) | GSLM, AudioLM |
| Pre-training | Joint speech-text pre-training | Spirit-LM, SpeechGPT |
| Alignment | Multi-task training for ASR/TTS | Whisper, Qwen-Audio |
| Alignment | Interleaved speech-text token training | Spectron, LauraGPT |
| Fine-tuning | Instruction tuning + RLHF alignment | GPT-4o, some closed-source models |

## Key Experimental Results

### Comparison of Representative SpeechLMs

| Model | Speech Tokenizer | LM | Vocoder | Capabilities |
|------|-----------------|----|---------|----|
| GSLM | HuBERT+kmeans | Transformer | code-HiFiGAN | Speech continuation |
| AudioLM | w2v-BERT+SoundStream | Transformer | SoundStream | Speech generation |
| VALL-E | EnCodec | AR+NAR Transformer | EnCodec dec. | Zero-shot TTS |
| SpeechGPT | HuBERT+kmeans | LLaMA | code-HiFiGAN | Dialogue |
| Spirit-LM | HuBERT+pitch/style | LLaMA | HiFi-GAN | Interleaved Text + Speech |
| Qwen-Audio | Whisper encoder | Qwen-7B | - | Understanding (no generation) |

### SpeechLM Capability Classification

| Capability Category | Specific Tasks | Description |
|---------|---------|------|
| Speech understanding | ASR, SLU, emotion recognition | Foundational capabilities |
| Speech generation | TTS, voice cloning, speech editing | Core generation |
| Conversational interaction | Speech dialogue, real-time interruption | GPT-4o-level capability |
| Paralinguistics | Emotional expression, speaking style control | Distinguishing SpeechLM from ASR+LLM+TTS |
| Multilingual | Cross-lingual speech translation | Extension capabilities |

### Key Findings
- **Semantic vs. Acoustic tokenizer is a core design choice**: Understanding tasks prefer semantic tokens, whereas generation tasks require acoustic tokens. Blending both approach paradigms is the current trend.
- **Adapting existing TextLMs is more practical than training from scratch**: Freezing the LLM and employing an adapter achieves the optimal balance between resource efficiency and performance.
- **Real-time interaction remains an open challenge**: The latency of current SpeechLMs (especially AR decoding) falls short of meeting real-time conversational requirements.
- **Unified evaluation framework is lacking**: Diverse works employ different metrics (e.g., WER, MOS, PESQ, speaker similarity), lacking a unified benchmark.

## Highlights & Insights
- **First comprehensive survey in the SpeechLM domain**: Provides a timely and systematic review following the surge of interest sparked by GPT-4o voice.
- **Clear three-component taxonomy**: The decomposition framework of tokenizer $\rightarrow$ LM $\rightarrow$ vocoder makes complex architectures easy to understand.
- **Accurate summarization of the "three major flaws of ASR+LLM+TTS"**: Information loss, high latency, and cascaded error propagation—providing a clear rationale for the existence of SpeechLMs.
- **The hybrid tokenizer direction**: Decoupling semantic and acoustic layers (e.g., SpeechTokenizer) offers an elegant solution to resolve the tension between "understanding vs. generation".

## Limitations & Future Work
- **Extremely rapid domain advancement**: Implementation details of closed-source systems like GPT-4o and Moshi remain publically unknown, leading to potential omissions in this review.
- **Lack of quantitative comparison**: A systematic comparison of different SpeechLMs on a unified benchmark is missing (each work utilizes distinct datasets and evaluation metrics).
- **Insufficient safety and ethical discussion**: Risks such as voice cloning abuse and deepfake speech detection are not deeply investigated.
- **Exclusion of wider multimodality**: The text focuses solely on speech and text, without considering broader multimodal LLMs of speech+vision.

## Related Work & Insights
- **vs. Whisper (Radford et al., 2023)**: Whisper is an encoder-only understanding model, whereas this SpeechLM survey encompasses the full paradigm of both understanding and generation.
- **vs. AudioLM (Borsos et al., 2023)**: AudioLM represents an early attempt in SpeechLMs, while this survey encompasses the rapid evolution that succeeded it.
- **vs. Multimodal LM surveys (Zhang et al., 2024)**: Multimodal surveys span across vision, audio, and text, whereas this work focuses specifically on an in-depth analysis of the speech modality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First survey of SpeechLMs; taxonomic system is highly valuable.
- **Experimental Thoroughness**: ⭐⭐ Pure survey with no original experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear categorization system and rich diagrams (especially the taxonomy tree in Figure 4).
- **Value**: ⭐⭐⭐⭐⭐ Provides a highly demanded systematic reference for the rapidly evolving SpeechLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)

</div>

<!-- RELATED:END -->
