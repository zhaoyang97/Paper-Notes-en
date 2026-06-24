---
title: >-
  [Paper Note] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling
description: >-
  [AAAI 2026][Audio & Speech][Speech Large Language Models] This paper proposes the DualSpeechLM framework, which leverages an understanding-driven speech tokenizer (USTokenizer) to extract high-level semantic tokens as LLM inputs and acoustic tokens as outputs. This approach simultaneously optimizes speech understanding and generation capabilities within a unified, end-to-end framework.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Speech Large Language Models"
  - "Dual Speech Token Modeling"
  - "Speech Understanding and Generation"
  - "Speech Tokenizer"
  - "Unified Framework"
date: 2026-05-08
content_hash: 2c188c5290edbf6f
---

# DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling

**Conference**: AAAI 2026  
**arXiv**: [2508.08961](https://arxiv.org/abs/2508.08961)  
**Code**: [https://github.com/lavendery/UUG](https://github.com/lavendery/UUG)  
**Area**: Audio & Speech / Speech Large Language Models  
**Keywords**: Speech Large Language Models, Dual Speech Token Modeling, Speech Understanding and Generation, Speech Tokenizer, Unified Framework

## TL;DR

This paper proposes the DualSpeechLM framework, which leverages an understanding-driven speech tokenizer (USTokenizer) to extract high-level semantic tokens as LLM inputs and acoustic tokens as outputs. This approach simultaneously optimizes speech understanding and generation capabilities within a unified, end-to-end framework.

## Background & Motivation

**Background**: In recent years, speech large language models (Speech LLMs) extended from text LLMs have flourished. These include understanding-oriented models (such as QwenAudio and SALMONN) and generation-oriented models (such as SEED-TTS and UniAudio). Unified understanding and generation efforts (e.g., SpeechGPT, Moshi, Mini-Omni2) are also under active exploration.

**Limitations of Prior Work**:
   - **Data Dependency**: Due to the massive modality gap between speech and text, adapting text LLMs to unified speech LLMs requires a vast amount of paired data (e.g., SpeechGPT requires 70K hours, and SpiritLM requires 570K hours).
   - **Task Conflict**: Generation tasks require rich acoustic details (such as prosody, emotion, and speaker characteristics), while understanding tasks require high-level semantic features. Representing both aspects with a single type of token is challenging: using acoustic tokens yields poor understanding, while using semantic tokens leads to low-quality generation.

**Key Challenge**: A single token type cannot meet the distinct informational demands of understanding (which favors semantics) and generation (which favors acoustics). Enhancing one often degrades the performance of the other.

**Goal**: To achieve mutual synergy rather than mutual conflict between speech understanding and generation under small-scale data scenarios.

**Key Insight**: Innovations are proposed from two dimensions: speech tokenization and language modeling. This involves designing an understanding-driven tokenizer and a dual-token modeling framework.

**Core Idea**: Use high-level semantic tokens (USTokens) as inputs to ease the modality alignment difficulty and enhance understanding, and use acoustic tokens as outputs to preserve acoustic details for high-quality generation. Both token types are jointly trained within a unified end-to-end framework.

## Method

### Overall Architecture

DualSpeechLM consists of two core modules:

1. **USTokenizer**: Extracts understanding-driven tokens aligned with the semantic space of the text LLM from speech.
2. **DualSpeechLM Main Framework**: A dual-token LLM that takes USTokens as inputs and produces acoustic tokens as outputs.

### Key Designs

1. **Understanding-Driven Speech Tokenizer (USTokenizer)**:

    - Architecture: Pre-trained Whisper encoder $\rightarrow$ Downsampling Encoder $\rightarrow$ Vector Quantization (VQ, single codebook) $\rightarrow$ Upsampling Decoder.
    - **Key Designs**: An Adapter module is integrated to project VQ quantized vectors into the input space of a frozen text LLM. The semantic content of the tokens is optimized through backpropagation from understanding tasks.
    - Training Loss: $$\mathcal{L}_{\text{USTokenizer}} = \alpha \cdot \mathcal{L}_{\text{commit}} + \beta \cdot \mathcal{L}_{\text{Under}} + \gamma \cdot \mathcal{L}_{\text{reconstruction}}$$
    - Here, the understanding loss $\mathcal{L}_{\text{Under}}$ is the autoregressive generation likelihood of the text LLM given the speech input. Consequently, token optimization is directly guided by the semantic space of the text LLM.
    - Unlike prior semantic tokenizers based on self-supervised learning (SSL) quantization (e.g., HuBERT) or ASR middle-layer quantization (e.g., CosyVoice), USTokenizer is explicitly aligned with the semantic capabilities of the text LLM, thereby significantly reducing the difficulty of modality alignment.

2. **Dual-Token Modeling Architecture**:

    - **Input Side**: USToken provides high-level semantic information and directly feeds into the text LLM.
    - **Output Side**: Instead of directly outputting USTokens (due to their lack of acoustic details), the **AcousticGPT** module converts the latent states of the LLM into acoustic tokens.
    - AcousticGPT is integrated inside the text LLM and jointly trained, forming an end-to-end pipeline.
    - Understanding Path: Speech $\rightarrow$ USToken $\rightarrow$ LLM $\rightarrow$ Text Output.
    - Generation Path: (Prompt + USToken) $\rightarrow$ LLM predicts objective USToken $\rightarrow$ AcousticGPT generates acoustic token $\rightarrow$ Waveform.

3. **Semantic Supervision Loss**:

    - Superimposed supervision on intermediate USToken predictions is incorporated into the generation path to prevent the LLM from "forgetting" semantic details.
    - This serves as a regularizer to stabilize dual-token joint training.

4. **Chain-of-Condition (CoC) Strategy**:

    - During generation tasks, rather than generating acoustic tokens directly from the input USToken in a single step, the LLM is guided to progressively generate target USTokens first, and then generate acoustic tokens based on them.
    - This mirrors the Chain-of-Thought concept but is applied to speech generation to provide more stable intermediate conditioning.

### Loss & Training

- USTokenizer: commitment loss + understanding loss + reconstruction loss.
- DualSpeechLM: Cross-entropy is employed for the understanding branch, and acoustic token prediction loss + semantic supervision loss are utilized for the generation branch.
- Only 4.5K hours of training data are used (compared to 570K hours for SpiritLM).
- Built on Phi3.5-3B, using LoRA fine-tuning instead of full-parameter fine-tuning.

## Key Experimental Results

### Main Results

**Understanding Capability** (WER↓, lower is better):

| Model | LLM | Training Data | ASR-Clean | ASR-Other | SQA (b4↑/gs↑) |
|------|-----|---------|-----------|-----------|--------------|
| SpeechGPT | LLaMA-7B | 70K hrs | 42.73 | 78.54 | 3.58/40 |
| SpiritLM | LLaMA-7B | 570K hrs | 6.0 | 11.0 | — |
| Baseline-Acoustic | Phi3.5-3B | 4.5K hrs | 36.52 | 80.06 | 17.68/76 |
| Baseline-Semantic | Phi3.5-3B | 4.5K hrs | 5.70 | 14.32 | 42.01/85 |
| **Ours (USToken)** | Phi3.5-3B | 4.5K hrs | **4.22** | **9.71** | **44.38/88** |

**Generation Capability** (TTS, SIM↑/WER↓/DNSMOS↑):

| Model | Clean | Other |
|------|-------|-------|
| Baseline-Acoustic | 0.88/22.11/3.76 | 0.87/26.38/3.69 |
| Baseline-Semantic | 0.80/21.72/3.29 | 0.81/22.32/3.26 |
| **Ours (USToken)** | **0.90/9.25/3.86** | **0.88/9.88/3.82** |

### Ablation Study

**Data Ratio Experiment** (Key Finding):
- Baseline Model: Increasing generation data degrades understanding performance, and increasing understanding data degrades generation performance (task conflict).
- DualSpeechLM: Increasing data in either direction simultaneously improves performance in both directions (mutual synergy).

**Token Type Comparison**:
- DualSpeechLM + HuBERT token: Understanding and generation show limited improvements.
- DualSpeechLM + USToken: Both understanding and generation are substantially enhanced, validating the core contribution of USToken.

### Key Findings

- Outperforming SpiritLM (trained on 570K hours) using only 4.5K hours of data demonstrates that USToken markedly mitigates the data requirement for modality alignment.
- The dual-token design successfully resolves the zero-sum game between understanding and generation, establishing a positive mutual synergy.
- USToken is significantly superior to HuBERT token in both understanding and generation.

## Highlights & Insights

- Decoupling "input tokens" from "output tokens" yields a simple yet profound architectural insight: understanding and generation inherently demand different levels of information granularity, and restricting them to a single token type is an unnecessary constraint.
- USTokenizer backpropagates guidance from the understanding capacity of the text LLM to drive speech token learning, formulating an elegant cross-modality knowledge distillation paradigm.
- Achieving superior results to prior methods with only 1% of the data (4.5K vs 570K hours) highlights an astonishing improvement in data efficiency.

## Limitations & Future Work

- The framework is evaluated on a relatively small LLM (Phi3.5-3B) and has not yet been validated on larger-scale models.
- USTokenizer remains dependent on the output quality of the pre-trained Whisper encoder.
- The acoustic tokens rely on WavTokenizer (single codebook); a multi-codebook scheme might further improve generation quality.
- Evaluations are limited to English data, leaving multilingual generalization unexplored.
- The CoC strategy introduces additional inference latency, as it requires generating USTokens before generating acoustic tokens.

## Related Work & Insights

- SpeechGPT / SpiritLM: Unified models utilizing HuBERT tokens, which, however, necessitate an extra trade-off phase (e.g., Mel $\rightarrow$ Waveform).
- Moshi: A real-time conversational model that leverages multi-codebook acoustic tokens.
- Qwen2.5-Omni: Leverages continuous Whisper features rather than discrete tokens.
- Insight: The dual-token concept can be naturally extended to vision-language models (e.g., using high-level semantic tokens for understanding and pixel-level tokens for generation).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Decoupled dual-token design and understanding-driven tokenizer represent clean and impactful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprising evaluations across understanding and generation, alongside persuasive data-ratio ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Highly intuitive diagrams with a clear developmental flow.
- Value: ⭐⭐⭐⭐⭐ Establishes an elegant and highly efficient paradigm for unified speech large language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](../../ICML2025/audio_speech/ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)
- [\[ICLR 2026\] UALM: Unified Audio Language Model for Understanding, Generation and Reasoning](../../ICLR2026/audio_speech/ualm_unified_audio_language_model_for_understanding_generation_and_reasoning.md)
- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](../../ACL2026/audio_speech/unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)

</div>

<!-- RELATED:END -->
