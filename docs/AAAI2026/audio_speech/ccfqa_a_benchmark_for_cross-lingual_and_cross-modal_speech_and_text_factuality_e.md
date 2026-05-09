---
title: >-
  [Paper Note] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
description: >-
  [AAAI2026][Audio & Speech][factuality evaluation] This paper introduces CCFQA—the first cross-lingual and cross-modal factuality benchmark covering 8 languages with 14,400 fully parallel speech-text factual QA samples. It supports four task settings (QA/XQA/SQA/XSQA), systematically revealing factual inconsistencies in existing MLLMs under language and modality switching. The paper also proposes LLM-SQA, which bridges via English with only 5-shot examples to achieve cross-lingual spoken QA transfer, attaining an F1 of 51.4 on XSQA—surpassing GPT-4o-mini-Audio (45.7).
tags:
  - AAAI2026
  - "Audio & Speech"
  - factuality evaluation
  - multilingual benchmark
  - spoken question answering
  - cross-lingual consistency
  - multimodal LLM
date: 2026-05-08
content_hash: 2739e74e17414741
---

# CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation

**Conference**: AAAI2026
**arXiv**: [2508.07295](https://arxiv.org/abs/2508.07295)
**Code**: [yxduir/ccfqa](https://github.com/yxduir/ccfqa)
**Area**: Audio & Speech
**Keywords**: factuality evaluation, multilingual benchmark, spoken question answering, cross-lingual consistency, multimodal LLM

## TL;DR

This paper introduces CCFQA—the first cross-lingual and cross-modal factuality benchmark covering 8 languages with 14,400 fully parallel speech-text factual QA samples. It supports four task settings (QA/XQA/SQA/XSQA), systematically revealing factual inconsistencies in existing MLLMs under language and modality switching. The paper also proposes LLM-SQA, which bridges via English with only 5-shot examples to achieve cross-lingual spoken QA transfer, attaining an F1 of 51.4 on XSQA—surpassing GPT-4o-mini-Audio (45.7).

## Background & Motivation

**State of the Field**: Multimodal large language models (MLLMs) have been widely deployed in multilingual settings, yet evaluation of their factual reliability lags severely behind. Existing factuality benchmarks (SimpleQA, TruthfulQA, HaluEval) primarily target English text; the multilingual benchmark KoLasSimpleQA covers 9 languages but is text-only. Speech-side benchmarks (SD-QA, VoiceBench, SpeechIQ) are limited to a single speech modality and are mostly English-only.

**Limitations of Prior Work**: When the same factual question is posed in different languages (cross-lingual inconsistency) or presented in different modalities (cross-modal inconsistency), MLLMs frequently produce contradictory answers. For example, questions answered correctly by the GPT-4o series in English text may be answered incorrectly when posed in Japanese speech. Yet no benchmark with fully parallel data exists to systematically measure such inconsistencies.

**Root Cause**: The combinatorial explosion of multilingual × multimodal evaluation space poses a fundamental challenge—8 languages × 2 modalities × cross-lingual combinations requires up to 128,000 API calls to fully evaluate a single MLLM, and constructing high-quality parallel data is extremely costly, requiring native-speaker recordings for each language.

**Paper Goals**: To construct the first factuality evaluation benchmark simultaneously covering cross-lingual and cross-modal dimensions, and to propose a low-cost transfer approach that leverages the strong factual reasoning capability of English LLMs.

**Starting Point**: Existing English QA datasets (MKQA + MOOCCubeX) → GPT-4.1 translation + human verification → native-speaker recordings + ASR quality control, yielding high-quality parallel data. On the model side, English bridging + 5-shot transfer overcomes the language barrier.

**Core Idea**: Fully parallel speech-text multilingual data + English-bridged few-shot transfer = systematic diagnosis and mitigation of cross-lingual and cross-modal factual inconsistencies in MLLMs.

## Method

### Overall Architecture

CCFQA encompasses two orthogonal components: **benchmark construction** (diagnosing the problem) and the **LLM-SQA model** (mitigating the problem). The benchmark generates 14,400 parallel samples through three stages: cross-lingual data construction → cross-modal data construction → quality control. LLM-SQA employs curriculum pre-training (ASR → SRT → SQA) followed by 5-shot cross-lingual transfer, leveraging English factual knowledge to serve non-English spoken QA.

### Key Design 1: Cross-Lingual Data Construction — Rigorous Filtering + Machine Translation + Human Verification

Data sources are drawn from MKQA (open-domain QA) and MOOCCubeX (educational domain), spanning 4 major categories (humanities, social sciences, natural sciences, applied sciences) across 20 sub-domains. Key filtering criteria exclude ambiguous questions (e.g., "Who is the current prime minister?"), sensitive content (PII/offensive language), factually incorrect QA pairs, and culturally dependent questions (e.g., "What is the legal marriage age?"). Filtered English QA pairs are translated into 7 target languages (Chinese, French, Japanese, Korean, Russian, Spanish, and Cantonese) using GPT-4.1. Chinese, English, and Japanese undergo human verification; remaining languages are validated via back-translation and review to ensure accuracy.

### Key Design 2: Cross-Modal Data Construction — Native-Speaker Recording + Iterative ASR Quality Control

Native-speaker volunteers (gender-balanced) are recruited for all 8 languages and record audio following guidelines for clarity, naturalness, steady pace, and neutral intonation. Low-volume samples receive audio enhancement, after which Whisper-large-v3 is applied for ASR transcription. WER/CER is computed against the reference text, and audio exceeding the WER threshold is flagged as anomalous and re-recorded. This iterative pipeline of "record → ASR check → re-record" ensures precise speech-text alignment. Final speech quality: English WER 3.2%, Chinese CER 6.8%.

### Key Design 3: LLM-SQA — Curriculum Learning + English-Bridged 5-Shot Transfer

Model architecture: frozen Whisper speech encoder (~635M) + trainable Q-Former adapter (80 queries, dim=768) + MLP (~80.5M total) + GemmaX2-9B LLM (~9.2B). The LLM is fine-tuned with LoRA ($r=16, \alpha=32$), training only ~8.9M parameters.

Training employs a three-stage curriculum:
1. **ASR stage**: Train speech recognition on FLEURS to establish speech-text alignment.
2. **SRT stage**: Train speech recognition + translation to learn cross-lingual mapping.
3. **SQA stage**: Two steps—(a) supervised fine-tuning on ~3,000 synthetic English speech-text pairs to learn QA task structure; (b) 5-shot cross-lingual transfer for each target language.

Core Mechanism: Via special instruction tokens (e.g., `<|qa|><|fra|>`), the model internally converts non-English spoken questions into English for factual reasoning, then translates answers back into the target language. This leverages the LLM's strongest knowledge base in English while minimizing the need for non-English annotated data.

### Loss & Training

All stages use standard autoregressive cross-entropy loss. ASR and SRT are trained on FLEURS; SQA uses synthetic English data plus 5-shot target-language data. Training runs on 4×A100 (80GB) GPUs for one week. Optimizer: AdamW; peak lr = $1 \times 10^{-4}$; 1,000-step warmup + linear decay.

## Key Experimental Results

### Main Results: Four Task Settings (F1 / LLM Acc)

| Model | Text QA Avg | Text XQA Avg | Speech SQA Avg | Speech XSQA Avg |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 63.9 / 64.4 | 59.7 / 62.2 | — | — |
| GPT-4o-mini-Audio | — | — | 47.7 / 40.4 | 45.7 / 38.6 |
| Phi-4-Multimodal | 18.0 / 13.8 | 18.0 / 15.3 | 18.5 / 22.0 | 21.0 / 5.7 |
| Qwen2-Audio | 27.9 / 30.6 | 24.2 / 19.2 | 27.7 / 17.0 | 24.1 / 10.7 |
| Qwen2.5-Omni-3B | 20.6 / 11.8 | — | 34.9 / 20.9 | 29.8 / 17.1 |
| Qwen2.5-Omni-7B | 46.5 / 38.2 | 42.1 / 34.5 | 44.0 / 33.2 | 38.5 / 29.5 |
| **LLM-SQA (Ours)** | — | — | **52.0 / 40.3** | **51.4 / 39.7** |

### Cross-Lingual and Cross-Modal Consistency (%, higher is more consistent)

| Model | Cross-Lingual Consistency | Cross-Modal Consistency |
|------|:---:|:---:|
| GPT-4o-mini | 96.6 / 95.5 | 62.7 / 62.1 |
| Phi-4-Multimodal | 90.8 / 25.9 | 62.7 / 37.5 |
| Qwen2-Audio | 62.4 / 62.9 | 55.6 / 56.0 |
| Qwen2.5-Omni-3B | 75.4 / 81.8 | 56.5 / 52.0 |
| Qwen2.5-Omni-7B | 90.3 / 87.2 | 90.3 / 85.5 |

### Effect of Speech Duration on SQA Accuracy (LLM Acc)

| Model | 0–5s | 5–10s | 10–30s | Avg |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini-Audio | 38.6 | 42.6 | 40.1 | 40.4 |
| LLM-SQA (Ours) | 40.3 | 40.8 | 36.1 | 40.3 |
| Qwen2.5-Omni-7B | 32.1 | 34.9 | 29.0 | 33.2 |
| Qwen2-Audio | 14.7 | 19.5 | 18.6 | 17.0 |
| Phi-4-Multimodal | 17.7 | 26.4 | 27.4 | 22.0 |

### Key Findings

- LLM-SQA outperforms all open-source baselines on both SQA and XSQA, and its XSQA F1 (51.4) substantially surpasses GPT-4o-mini-Audio (45.7), demonstrating the effectiveness of the 5-shot transfer strategy.
- Cross-lingual challenge: Performance drops significantly for most models from QA to XQA; GPT-4o-mini is the most stable (96.6% consistency).
- Cross-modal challenge: A pervasive "modality gap" exists from text to speech; Qwen2.5-Omni-7B achieves the highest cross-modal consistency (90.3%), benefiting from its Omni architectural design.
- High F1 but low LLM Acc → hallucination (fluent but factually incorrect responses); low F1 but high LLM Acc → instruction-following issues (correct knowledge but non-compliant format).
- By language: English, French, and Spanish perform best; Korean and Cantonese perform worst, reflecting pretraining data distribution bias.
- Speech quality: English WER is 3.2%; some languages exhibit higher WER (French 13.8%, Russian 18.2%, Cantonese 16.8%) due to proper nouns.

## Highlights & Insights

- **Filling an evaluation gap**: CCFQA is the only fully parallel factuality benchmark simultaneously supporting cross-lingual (8 languages) and cross-modal (text + speech) evaluation. A complete evaluation requires 128K API calls, representing an unprecedented evaluation scope.
- **Strong diagnostic power**: The two orthogonal dimensions of cross-lingual and cross-modal consistency precisely localize model weaknesses—e.g., Phi-4-Multimodal's cross-lingual LLM Acc consistency of only 25.9% exposes deficiencies in its multilingual token design.
- **Extremely low-cost transfer**: LLM-SQA achieves cross-lingual spoken QA transfer with only 5-shot examples, using English as a universal interface for factual knowledge—a conceptually simple yet highly effective approach.
- **Decoupled analysis of F1 and LLM Acc**: Reveals two distinct failure modes—hallucination (high F1, low Acc) and instruction-following failure (low F1, high Acc)—providing a new diagnostic tool for model analysis.
- **High-quality speech data**: The native-speaker recording pipeline with iterative ASR quality control ensures evaluation reliability.

## Limitations & Future Work

- Language coverage of 8 languages remains limited, lacking important low-resource languages such as Arabic and Hindi, as well as tonal languages beyond Cantonese.
- Speech data consists of read-aloud recordings in controlled laboratory conditions, not covering accented speech, noisy environments, or natural conversational styles, potentially overestimating real-world model performance.
- The English-bridging strategy in LLM-SQA introduces English-centric bias; for languages with large linguistic distance from English (e.g., Korean), transfer effectiveness may plateau.
- Benchmark scale (1,000 test pairs × 8 languages) results in limited sample sizes in certain sub-domains.
- Evaluation is restricted to factual QA; more complex tasks such as speech reasoning and speech summarization are not addressed.
- End-to-end multilingual speech training (as opposed to English bridging) is not explored, leaving open whether it might perform better at scale.

## Related Work & Insights

- **vs. SimpleQA / Chinese SimpleQA**: Text-only, monolingual. CCFQA extends to 8 languages + speech modality, enabling cross-modal factual consistency evaluation for the first time.
- **vs. KoLasSimpleQA**: Covers 9 languages but is text-only with non-parallel QA. CCFQA's fully parallel design enables precise quantification of cross-lingual consistency.
- **vs. VoiceBench / SpeechIQ**: English-only speech, non-open-ended answers. CCFQA supports open-ended spoken QA across 8 languages.
- **vs. SD-QA**: Dialect spoken QA in 5 languages, but does not support cross-lingual evaluation. CCFQA adds XQA/XSQA evaluation dimensions.

| Benchmark | # Languages | Modality | Size | Supported Tasks |
|------|:---:|:---:|:---:|:---:|
| SimpleQA | 1 | Text | 4,326 | QA |
| KoLasSimpleQA | 9 | Text | 2,147 | QA |
| VoiceBench | 1 | Speech | 5,783 | SQA |
| SD-QA | 5 | Speech | 11,109 | SQA |
| **CCFQA** | **8** | **Text + Speech** | **14,400** | **QA/XQA/SQA/XSQA** |

**Insights**: The English-bridging strategy proves effective for factual knowledge transfer and is generalizable to other knowledge-intensive multilingual tasks. The curriculum learning approach (ASR → SRT → SQA) is transferable to multimodal alignment scenarios. The "modality gap" phenomenon warrants deeper investigation—why does speech input systematically degrade factual accuracy?

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: CCFQA is the first cross-lingual and cross-modal factuality evaluation benchmark, filling an important gap. The benchmark construction pipeline is rigorous (GPT translation + human verification + native-speaker recording + ASR quality control), the evaluation dimensions are comprehensive (4 task settings × 2 metrics × 8 languages), and the LLM-SQA 5-shot transfer approach is concise and effective. The main limitations are the restricted language coverage and speech scenario diversity, and the model-side contribution (English bridging) is incremental rather than fundamentally novel.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

<!-- RELATED:END -->
