---
title: >-
  [Paper Note] Distilling an End-to-End Voice Assistant Without Instruction Training Data
description: >-
  [ACL 2025][Audio & Speech][Speech Large Language Models] This work proposes DiVA (Distilled Voice Assistant), which performs cross-modal distillation by utilizing the text LLM's responses to transcriptions as self-supervised signals. This approach trains an end-to-end speech LLM without any speech instruction training data. With only 3.5k hours of ASR data, the model generalizes to spoken QA, classification, and translation tasks, outperforming Qwen 2 Audio (which uses over 1…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Speech Large Language Models"
  - "Cross-modal Distillation"
  - "End-to-end Voice Assistant"
  - "Knowledge Transfer"
  - "Whisper"
date: 2026-05-08
content_hash: 39a60c642fe68974
---

# Distilling an End-to-End Voice Assistant Without Instruction Training Data

**Conference**: ACL 2025  
**arXiv**: [2410.02678](https://arxiv.org/abs/2410.02678)  
**Code**: [github](https://github.com/google-research/google-research/tree/master/DiVA)  
**Area**: Speech/Audio  
**Keywords**: Speech Large Language Models, Cross-modal Distillation, End-to-end Voice Assistant, Knowledge Transfer, Whisper

## TL;DR

This work proposes DiVA (Distilled Voice Assistant), which performs cross-modal distillation by utilizing the text LLM's responses to transcriptions as self-supervised signals. This approach trains an end-to-end speech LLM without any speech instruction training data. With only 3.5k hours of ASR data, the model generalizes to spoken QA, classification, and translation tasks, outperforming Qwen 2 Audio (which uses over 100x more training compute) with a 72% win rate in user preference tests.

## Background & Motivation

Extending the capabilities of LLMs to the speech modality holds significant value: speech is a more natural interaction mode, and directly processing speech preserves intonation, rhythm, and accent that are typically lost in ASR systems, while reducing annotation costs and latency.

Current Speech LLMs primarily employ **large-scale multi-task supervised fine-tuning (SFT)**, but face several severe challenges:

**The "forgetting" issue**: Models tend to lose the pre-existing capabilities of the text LLM after SFT training. This cannot be entirely avoided even when freezing LLM weights.

**Data bottleneck**: Preventing forgetting requires a vast amount of diverse, annotated speech instruction data, which is currently extremely scarce.

**Under-representation**: Existing speech data often originates from a small cohort of speakers, exacerbating biases in speech processing.

**Difficulty in open-sourcing**: Training details of state-of-the-art Speech LLMs are often proprietary, making them difficult to replicate.

However, the speech community has accumulated vast amounts of **ASR data** (e.g., CommonVoice, LibriSpeech). The key challenge is that SFT cannot train a general-purpose voice assistant solely using ASR data. The authors propose a fundamentally different paradigm: utilizing distillation to train a generalizable Speech LLM using nothing but ASR data.

## Method

### Overall Architecture

DiVA is initialized with three pre-trained components:
- **Audio Encoder**: Whisper-Large-v3 encoder (extracts audio features)
- **Audio-to-Text Feature Aligner**: A Q-Former initialized from the Whisper decoder (aggregates high-granularity audio features into embeddings compatible with the LLM)
- **Text Decoder**: Llama 3 (with frozen weights, responsible for reasoning and generation)

The training process utilizes only audio and transcript pairs from ASR data, optimizing the audio encoder and Q-Former via two distillation losses.

### Key Designs

1. **Initializing Q-Former from the Whisper Decoder**: Prior works discard the Whisper decoder and train a Q-Former from scratch to align audio and text. However, the Whisper decoder itself is already trained for ASR—its cross-attention mechanism has learned to map audio embeddings to discrete text tokens. The authors directly reuse the Whisper decoder's K and V cross-attention weights to initialize the Q-Former, replacing the input only with static query tokens $Q$. This substantially reduces training costs and achieves the "best of both worlds"—leveraging the adaptive dimensionality reduction of a Q-Former while eliminating the huge overhead of training from scratch.

2. **Cross-modal Token Alignment Loss ($L_{con}$)**: Feeding audio through the audio encoder yields $Q$ audio tokens, while the corresponding transcription is embedded as $N$ text tokens (where typically $Q > N$). The L2 distance between the text embeddings and the **last $N$** tokens of the audio embeddings is minimized: $L_{con} = \sum_{n=0}^{N} |t_n^{text} - t_{Q-N+n}^{audio}|_2$. The "last $N$" tokens are chosen because the Whisper decoder employs causal attention, letting the final tokens attend to all preceding ones, which propagates gradient signals throughout the sequence. The extra $Q-N$ tokens provide bandwidth to transmit non-textual information like intonation.

3. **Output Distribution Distillation Loss (Core Innovation)**: The objective is to make the model's response distribution to audio input match its response distribution to text input, implementing cross-modal context distillation. For KL divergence minimization, the authors prove a key lemma: when the teacher and student share the output embedding matrix $O$ (the frozen Llama output layer), **minimizing the L2 distance of hidden states is a subset of minimizing the KL divergence**. This not only yields smoother gradients but also significantly reduces computation compared to KL divergence since the LLM vocabulary is much larger than the hidden dimension. In practice, only the hidden state of the first predicted token is contrasted during training, as a single token probability already encodes substantial information.

4. **Synergy of the Two Losses**: Using only the distillation loss yields a reasonably capable model (stronger in QA than SFT baselines), but it tends to ignore text instructions (e.g., ignoring the target language during translation). Using only the token alignment loss yields poor results (incoherent generation). However, combining both losses significantly improves instruction-following capability (increasing language identification accuracy from 1.4% to 74%).

### Loss & Training

- **Training Data**: Only CommonVoice 17 English subset, 3.5k hours, 93,725 speakers
- **Training Volume**: 4300 steps, batch size 512, approximately 2 epochs
- **Hardware**: TPU v4-256, completed in about 12 hours
- **Optimizer**: AdamW, lr=5e-5, cosine decay
- **Freezing Strategy**: Llama 3 weights are fully frozen

## Key Experimental Results

### Main Results - Spoken QA

| Model | HeySquad (PANDA) | SDQA (PANDA) |
|------|-----------------|--------------|
| SALMONN | Low | Low |
| Qwen Audio Chat | Low (30% direct transcription) | Low |
| Qwen 2 Audio | Medium (4% ignore instructions) | Medium |
| **DiVA** | **Highest (+5 PANDA, significant P<0.05)** | **Highest** |

DiVA significantly outperforms all baselines on both QA benchmarks (by at least +10%), and is the only model that 100% follows instructions instead of transcribing the questions.

### Main Results - Sentiment Classification

| Model | MELD (F1) | IEMOCAPS (F1) |
|------|-----------|---------------|
| SALMONN | Low (biased towards Neutral) | Low |
| Qwen Audio | Low (>90% predicting Sadness) | Low |
| Qwen 2 Audio | Low | Low |
| **DiVA** | **Significantly the Best** | **Significantly the Best** |

DiVA significantly outperforms all baselines without any sentiment-supervised training, demonstrating that the distillation process implicitly learns the correlation between text and speech sentiment.

### Main Results - Speech Translation (7 Languages)

| Model | Strongest Languages | Weakest Languages |
|------|---------|---------|
| Qwen Audio | Chinese, Japanese | - |
| Qwen 2 Audio | Arabic, German, Indonesian | - |
| **DiVA** | **Turkish, Tamil** | Chinese, Japanese |

The translation results are mixed. DiVA performs weaker than baselines on Chinese and Japanese because Llama 3 itself is biased toward outputting Romanized spellings for these two languages.

### User Preference Study

| Comparison | DiVA Win Rate | User Preference |
|------|---------|---------|
| DiVA vs Qwen 2 Audio | **72%** | 41 out of 53 users (77%) preferred DiVA |

DiVA won with an overwhelming advantage in 522 pairwise evaluations, validating that standard benchmarks may not fully reflect actual user preferences.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Full DiVA (Distillation + Alignment) | Best QA, Good Translation | Synergy of both losses |
| Distillation Loss Only | QA outperforms SFT baselines | Translation is nearly zero (language ID only 1.4%) |
| Token Alignment Loss Only | Incoherent generation | Classification is near-random |

### Key Findings

- Generalizing to spoken QA, classification, and translation tasks using only ASR data proves the feasibility of cross-modal distillation.
- User preferences do not perfectly align with benchmark results—72% of users prefer DiVA despite it not being optimal across all benchmark tasks.
- DiVA achieves superior user satisfaction while requiring over 100x less training compute.
- Distillation inherits the pitfalls of the text LLM (e.g., Llama's tendency to output Latin transliterations for Chinese/Japanese, or assuming text is always humorous).
- Reusing the Whisper decoder is a key engineering innovation, bypassing the massive overhead of training a Q-Former from scratch.

## Highlights & Insights

- **Paradigm Innovation**: Replacing SFT with distillation addresses the dual challenges of "forgetting" and "data scarcity," charting a new path for Speech LLM training.
- **Extreme Efficiency**: Outstanding performance is achieved with only 3.5k hours of data and 12 hours of training, outpacing Qwen 2 Audio which utilizes over 50k hours and DPO.
- **Mathematical Elegance**: The proof showing that the L2 distance is a subset of KL divergence minimization when sharing the output matrix simplifies complex KL distillation into straightforward L2 regression.
- **Whisper Decoder Reuse**: A component completely overlooked by prior works becomes a critical source for initialization.
- **Open Research**: Uses entirely open-source data and code, releasing full training code rather than just inference weights.

## Limitations & Future Work

- Inherits the limitations of Llama 3 (e.g., transliteration bias in Chinese/Japanese translation, misjudgment of humor/sarcasm).
- Pure prosody understanding remains weak—sarcasm and humor detection perform close to random guessing.
- Trained exclusively on English CommonVoice; multi-lingual scenarios are not fully validated.
- The "last $N$" alignment strategy in the token alignment loss may not be optimal.
- Distillation in the speech generation (TTS) direction remains unexplored.
- The user study is small-scale (53 participants, 522 evaluations), leaving room for larger-scale validation.

## Related Work & Insights

- Establishes a direct contrast with SALMONN and the Qwen Audio series, showcasing a fundamentally different training paradigm.
- Extension of context distillation to cross-modal applications—from text to speech, which could generalize to image, video, and other modalities.
- The Q-Former initialization strategy can inspire other cross-modal alignment research.
- Delivers an elegant solution to the general problem of "how to efficiently transfer LLM capabilities to a new modality."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Paradigm-shifting innovation of using cross-modal distillation instead of SFT, combined with the clever reuse of the Whisper decoder.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers spoken QA, classification, translation, user studies, and ablation studies, although translation results are mixed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, elegant mathematical derivation, tight structure, and excellent readability.
- **Value**: ⭐⭐⭐⭐⭐ Charts a highly efficient and open path forward for the Speech LLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)

</div>

<!-- RELATED:END -->
