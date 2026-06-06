---
title: >-
  [Paper Note] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling
description: >-
  [AAAI 2026][Audio & Speech][speech large language model] This paper proposes DualSpeechLM, a framework that extracts high-level semantic tokens via an understanding-driven speech tokenizer (USTokenizer) as LLM input and…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "speech large language model"
  - "dual token modeling"
  - "speech understanding and generation"
  - "speech tokenizer"
  - "unified framework"
date: 2026-05-08
content_hash: bf1a8e2a45d4273e
---

# DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling

**Conference**: AAAI 2026
**arXiv**: [2508.08961](https://arxiv.org/abs/2508.08961)  
**Code**: [https://github.com/lavendery/UUG](https://github.com/lavendery/UUG)  
**Area**: Audio & Speech / Speech Large Language Models
**Keywords**: speech large language model, dual token modeling, speech understanding and generation, speech tokenizer, unified framework

## TL;DR

This paper proposes DualSpeechLM, a framework that extracts high-level semantic tokens via an understanding-driven speech tokenizer (USTokenizer) as LLM input and uses acoustic tokens as output, jointly optimizing speech understanding and generation within a single end-to-end framework.

## Background & Motivation

**Background**: Speech large language models (Speech LLMs) built upon text LLMs have advanced rapidly in recent years, encompassing understanding-oriented models (QwenAudio, SALMONN) and generation-oriented models (SEED-TTS, UniAudio). Unified understanding-and-generation approaches (SpeechGPT, Moshi, Mini-Omni2) are also being actively explored.

**Limitations of Prior Work**:
   - **Data dependency**: Adapting text LLMs into unified speech LLMs requires large amounts of paired data due to the substantial modality gap between speech and text (SpeechGPT requires 70K hours; SpiritLM requires 570K hours).
   - **Task conflict**: Generation tasks require rich acoustic details (prosody, emotion, speaker characteristics), whereas understanding tasks require high-level semantic features. A single token type cannot serve both objectives well—acoustic tokens degrade understanding performance, while semantic tokens degrade generation quality.

**Key Challenge**: A single token type cannot simultaneously satisfy the distinct information requirements of understanding (semantics-oriented) and generation (acoustics-oriented); improving one objective typically harms the other.

**Goal**: Achieve mutual benefit between speech understanding and generation under low-resource data conditions, rather than a zero-sum trade-off.

**Key Insight**: Innovations are proposed along two dimensions—speech tokenization and language modeling—by designing an understanding-driven tokenizer and a dual-token modeling framework.

**Core Idea**: High-level semantic tokens (USTokens) are used as input to reduce the modality alignment difficulty and enhance understanding, while acoustic tokens are used as output to preserve acoustic details for high-quality generation. Both are jointly trained within a unified end-to-end framework.

## Method

### Overall Architecture

DualSpeechLM consists of two core modules:

1. **USTokenizer**: Extracts understanding-driven tokens from speech that are aligned with the semantic space of a text LLM.
2. **DualSpeechLM main framework**: A dual-token LLM that takes USTokens as input and produces acoustic tokens as output.

### Key Designs

1. **Understanding-Driven Speech Tokenizer (USTokenizer)**:

    - Architecture: pretrained Whisper encoder → downsampling encoder → vector quantization (VQ, single codebook) → upsampling decoder.
    - **Key innovation**: An Adapter module projects VQ-quantized vectors into the input space of a frozen text LLM; the semantic content of the tokens is optimized through backpropagation from understanding tasks.
    - Training loss: $\mathcal{L}_{\text{USTokenizer}} = \alpha \cdot \mathcal{L}_{\text{commit}} + \beta \cdot \mathcal{L}_{\text{Under}} + \gamma \cdot \mathcal{L}_{\text{reconstruction}}$
    - The understanding loss $\mathcal{L}_{\text{Under}}$ is the autoregressive generation likelihood of the text LLM given speech input, so token optimization is directly guided by the semantic space of the text LLM.
    - Unlike prior semantic tokenizers based on SSL quantization (HuBERT) or ASR intermediate-layer quantization (CosyVoice), USTokenizer is explicitly aligned with the semantic capability of the text LLM, substantially reducing the modality alignment burden.

2. **Dual-Token Modeling Architecture**:

    - **Input side**: USTokens provide high-level semantic information and are fed directly into the text LLM.
    - **Output side**: Rather than directly generating USTokens (which lack acoustic detail), the **AcousticGPT** module converts LLM hidden states into acoustic tokens.
    - AcousticGPT is integrated within the text LLM and trained jointly, forming an end-to-end pipeline.
    - Understanding path: speech → USTokens → LLM → text output.
    - Generation path: (prompt + USTokens) → LLM predicts target USTokens → AcousticGPT produces acoustic tokens → waveform.

3. **Semantic Supervision Loss**:

    - An auxiliary supervision signal on intermediate USToken prediction is added to the generation path to prevent the LLM from losing semantic information.
    - This serves as a regularization mechanism to stabilize joint dual-token training.

4. **Chain-of-Condition (CoC) Strategy**:

    - Rather than generating acoustic tokens directly from input USTokens in a single step, the LLM first autoregressively generates target USTokens, which are then used to produce acoustic tokens.
    - Conceptually analogous to Chain-of-Thought reasoning but applied to speech generation, providing more stable intermediate conditioning.

### Loss & Training

- USTokenizer: commitment loss + understanding loss + reconstruction loss.
- DualSpeechLM: cross-entropy for the understanding branch; acoustic token prediction loss + semantic supervision loss for the generation branch.
- Only 4.5K hours of training data are used (compared to 570K hours for SpiritLM).
- Built upon Phi3.5-3B with LoRA fine-tuning rather than full-parameter training.

## Key Experimental Results

### Main Results

**Understanding performance** (WER↓ lower is better):

| Model | LLM | Training Data | ASR-Clean | ASR-Other | SQA (b4↑/gs↑) |
|------|-----|---------|-----------|-----------|--------------|
| SpeechGPT | LLaMA-7B | 70K hrs | 42.73 | 78.54 | 3.58/40 |
| SpiritLM | LLaMA-7B | 570K hrs | 6.0 | 11.0 | — |
| Baseline-Acoustic | Phi3.5-3B | 4.5K hrs | 36.52 | 80.06 | 17.68/76 |
| Baseline-Semantic | Phi3.5-3B | 4.5K hrs | 5.70 | 14.32 | 42.01/85 |
| **DualSpeechLM (USToken)** | Phi3.5-3B | 4.5K hrs | **4.22** | **9.71** | **44.38/88** |

**Generation performance** (TTS, SIM↑/WER↓/DNSMOS↑):

| Model | Clean | Other |
|------|-------|-------|
| Baseline-Acoustic | 0.88/22.11/3.76 | 0.87/26.38/3.69 |
| Baseline-Semantic | 0.80/21.72/3.29 | 0.81/22.32/3.26 |
| **DualSpeechLM (USToken)** | **0.90/9.25/3.86** | **0.88/9.88/3.82** |

### Ablation Study

**Data proportion experiments (key finding)**:
- Baseline models: increasing generation data degrades understanding performance, and increasing understanding data degrades generation performance (task conflict).
- DualSpeechLM: increasing data in either direction simultaneously improves performance on both dimensions (mutual benefit).

**Token type comparison**:
- DualSpeechLM + HuBERT tokens: improvements in both understanding and generation, but limited.
- DualSpeechLM + USTokens: substantial gains in both understanding and generation, validating the core contribution of USTokens.

### Key Findings

- Using only 4.5K hours of data, the model surpasses SpiritLM trained on 570K hours, demonstrating that USTokens substantially reduce data requirements for modality alignment.
- The dual-token design successfully breaks the zero-sum trade-off between understanding and generation, enabling positive mutual reinforcement.
- USTokens significantly outperform HuBERT tokens on both understanding and generation tasks.

## Highlights & Insights

- Decoupling input tokens from output tokens is a concise yet profound design insight: understanding and generation have fundamentally different information granularity requirements, and constraining both to a single token type is an unnecessary restriction.
- USTokenizer leverages the understanding capability of the text LLM to guide speech token learning in reverse, constituting an elegant form of cross-modal knowledge distillation.
- Exceeding prior methods with only 1% of the training data (4.5K vs. 570K hours) represents a remarkable improvement in data efficiency.

## Limitations & Future Work

- Built upon Phi3.5-3B, a relatively small LLM; scalability to larger models has not been verified.
- USTokenizer remains dependent on the output quality of the Whisper encoder.
- Acoustic tokens are produced by WavTokenizer (single codebook); multi-codebook schemes may further improve generation quality.
- Evaluation is limited to English data; multilingual generalization remains unexplored.
- The CoC strategy introduces additional inference latency due to the sequential generation of USTokens followed by acoustic tokens.

## Related Work & Insights

- SpeechGPT / SpiritLM: unified models using HuBERT tokens, but requiring an additional Mel-to-waveform stage.
- Moshi: a real-time dialogue model employing multi-codebook acoustic tokens.
- Qwen2.5-Omni: uses continuous Whisper features rather than discrete tokens.
- Inspiration: the dual-token paradigm is potentially generalizable to vision-language models, using high-level visual tokens for understanding and pixel-level tokens for generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the dual-token decoupling design and the understanding-driven tokenizer represent clear and compelling innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Bidirectional evaluation across understanding and generation, with convincing data-proportion ablations.
- Writing Quality: ⭐⭐⭐⭐ Intuitive figures and well-structured progressive argumentation.
- Value: ⭐⭐⭐⭐⭐ Provides an elegant and data-efficient paradigm for unified speech large language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](../../ACL2026/audio_speech/do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](../../ICCV2025/audio_speech/understanding_co-speech_gestures_in-the-wild.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](../../ICLR2026/audio_speech/pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)

</div>

<!-- RELATED:END -->
