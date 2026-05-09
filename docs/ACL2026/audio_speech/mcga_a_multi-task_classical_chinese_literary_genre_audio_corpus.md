---
title: >-
  [Paper Note] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
description: >-
  [ACL 2026][Audio & Speech][classical literary speech corpus] This paper introduces MCGA, the first large-scale (119 hours, 22,000 samples) fully copyright-cleared audio corpus for classical Chinese literature, spanning five major literary genres (Fu, Shi, Wen, Ci, Qu) and six speech tasks (ASR/S2TT/SEC/SQA/SU/SR). An evaluation of 10 multimodal large models reveals substantial deficiencies in current models' ability to understand classical literary speech.
tags:
  - ACL 2026
  - "Audio & Speech"
  - classical literary speech corpus
  - multimodal large language models
  - speech emotion analysis
  - cross-modal consistency
  - classical Chinese literary studies
date: 2026-05-08
content_hash: 00509eb6cf74dea2
---

# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus

**Conference**: ACL 2026
**arXiv**: [2601.09270](https://arxiv.org/abs/2601.09270)
**Code**: [https://github.com/yxduir/MCGA](https://github.com/yxduir/MCGA)
**Area**: Speech & Natural Language Processing / Classical Chinese Literature
**Keywords**: classical literary speech corpus, multimodal large language models, speech emotion analysis, cross-modal consistency, classical Chinese literary studies

## TL;DR

This paper introduces MCGA, the first large-scale (119 hours, 22,000 samples) fully copyright-cleared audio corpus for classical Chinese literature, spanning five major literary genres (Fu, Shi, Wen, Ci, Qu) and six speech tasks (ASR/S2TT/SEC/SQA/SU/SR). An evaluation of 10 multimodal large models reveals substantial deficiencies in current models' ability to understand classical literary speech.

## Background & Motivation

**Background**: The rapid development of multimodal large language models (MLLMs) has opened new possibilities for classical Chinese studies (CCS). However, existing research focuses primarily on text (ACLUE, WenMind, etc.) and visual (Oracle-Bench, MCS-Bench, etc.) modalities, leaving the speech dimension of classical literature virtually unexplored. The absence of high-quality domain-specific audio corpora makes it impossible to systematically evaluate or improve MLLM capabilities in classical Chinese speech understanding.

**Limitations of Prior Work**: (1) Most existing Chinese cultural datasets cover only text or image modalities, with no parallel classical literary speech data; (2) the few resources addressing Chinese speech target modern Mandarin and cannot capture the rhetorical devices, literary allusions, and prosodic features unique to classical literature; (3) copyright issues have persistently hindered the construction of open-source CCS audio datasets, as recitation audio found online is typically rights-restricted and cannot be freely distributed for research.

**Key Challenge**: MLLMs have acquired strong text and visual understanding capabilities, yet the infrastructure for evaluating classical Chinese speech understanding is entirely absent. Without a speech corpus, evaluation is impossible; without evaluation, progress in this domain cannot be driven.

**Goal**: To construct a multi-genre, multi-task, fully copyright-cleared audio corpus for classical Chinese literature, establish a systematic evaluation framework, and comprehensively assess current MLLM capabilities in classical literary speech understanding.

**Key Insight**: The corpus is organized along two axes — genre diversity, covering the five most historically significant literary genres in Chinese literature (Fu, Shi, Wen, Ci, Qu), and task diversity, comprising a six-level progressive task hierarchy ranging from foundational (ASR) to advanced (speech reasoning, SR).

**Core Idea**: Twenty-eight native speakers are recruited to manually record all audio with copyright assignment agreements. LLMs generate question–answer pairs that are validated through a triple-verification process to ensure quality, yielding a parallel corpus supporting six speech tasks and four text tasks.

## Method

### Overall Architecture

MCGA is constructed in three stages: (1) *Data collection and preprocessing* — classical literary texts and their pinyin transcriptions are collected from the public domain, cleaned, and segmented to keep individual recordings within 30 seconds; (2) *Human recording* — 28 native speakers record all texts under unified guidelines, with quality control conducted through two rounds of inspection (MLLM-based and manual); (3) *Text data construction* — DeepSeek-V3.2 generates multi-task question–answer pairs grounded in the full literary context of each segment, followed by triple verification using DeepSeek-V3.2, GPT-5-mini, and Gemini-3-Flash to filter invalid samples.

### Key Designs

1. **Six-Level Progressive Task Hierarchy**
    - Function: Covers the full spectrum of speech understanding capabilities, from low-level perception to high-level reasoning.
    - Mechanism: Six core speech tasks are designed — ASR (automatic speech recognition), S2TT (speech-to-text translation), SEC (speech emotion characterization), SQA (spoken question answering), SU (speech understanding), and SR (speech reasoning). ASR tests basic transcription; S2TT requires cross-lingual translation from classical Chinese to modern English; SEC requires identification of speaker characteristics and sentence-level emotion analysis; SQA is open-ended factual question answering; SU tests comprehension based on speech content; SR requires external knowledge for inference. The parallel text data additionally supports four text tasks: MT, QA, LU, and LR.
    - Design Motivation: A single task cannot comprehensively assess MLLM capabilities in classical literary understanding. The progressive task design enables precise localization of model bottlenecks at different cognitive levels.

2. **Emotion Characterization Fidelity Metric (ECF)**
    - Function: Provides fine-grained automatic evaluation for classical literary speech emotion characterization.
    - Mechanism: ECF comprises three sub-metrics — ECF-P (person recognition, 0–2, assessing accuracy of age and gender identification, with 1 point deducted per error); ECF-G (global emotional tone, 0–3, evaluating richness and accuracy of the overall emotional atmosphere description); and ECF-S (sentence-level emotion fidelity, 0–5, assessing transcription quality and sentence-level emotion analysis, with 1 point deducted per emotional error and a score of 0 for hallucinations). The final score is normalized to a 100-point scale.
    - Design Motivation: Existing speech emotion evaluation metrics primarily target emotion classification (e.g., happy/sad) in modern speech and cannot capture the complex emotional layers characteristic of classical literary recitation — such as the intertwined reluctance, equanimity, and ambition in a farewell poem.

3. **Cross-Modal Consistency Metric (CMC)**
    - Function: Quantifies the consistency of MLLM performance across speech and text input modalities.
    - Mechanism: $CMC = \frac{1}{3}\left(\frac{SQA}{QA} + \frac{SU}{LU} + \frac{SR}{LR}\right) \times 100$, computed as the average ratio of scores on the three speech tasks (SQA/SU/SR) to their corresponding text counterparts (QA/LU/LR). A CMC value approaching 100 indicates that the model's speech understanding ability is consistent with its text understanding ability.
    - Design Motivation: An ideal MLLM should produce consistent answers whether given speech or text input. CMC reveals whether a model genuinely "understands" spoken content or merely relies on text-channel capabilities.

### Loss & Training

Fine-tuning experiments use Qwen2.5-Omni-7B as the base model, applying LoRA ($r=8, \alpha=32$) trained for 3 epochs on the MCGA training set with the AdamW optimizer at a learning rate of $1 \times 10^{-4}$, on 4 A100 GPUs.

## Key Experimental Results

### Main Results

| Model | ASR (CER↓) | S2TT (LLM-B↑) | SEC (ECF↑) | SQA (F1↑) | SU (Acc↑) | SR (Acc↑) | Total↑ |
|---|---|---|---|---|---|---|---|
| GPT-4o-mini-Audio | 20.6 | 43.5 | 5.7 | 30.6 | 74.8 | 70.2 | 304.2 |
| Gemini-3-Flash | 6.1 | 74.0 | 54.0 | 48.7 | 86.6 | 83.7 | 440.9 |
| Qwen2.5-Omni-7B | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 | 380.7 |
| Qwen3-Omni-30B | 4.4 | 67.6 | 58.4 | 51.5 | 86.9 | 82.9 | 442.9 |
| Step-Audio-2-mini | 9.9 | 41.9 | 36.8 | 45.2 | 80.5 | 80.4 | 374.9 |
| Phi-4-Multimodal | 59.6 | 27.5 | 12.7 | 24.5 | 50.6 | 54.4 | 210.1 |

Qwen3-Omni achieves the highest total score (442.9), leading on ASR, SEC, SQA, and SU; Gemini-3-Flash performs best on S2TT and SR, reflecting the advantage of closed-source models in English generation and reasoning.

| Model | Shi CER | Ci CER | Qu CER | Fu CER | Wen CER |
|---|---|---|---|---|---|
| Qwen3-Omni-30B | 3.8 | 2.8 | 4.1 | 6.2 | 4.3 |
| Qwen2.5-Omni-7B | 9.9 | 7.5 | 8.9 | 14.8 | 8.8 |
| Qwen-Omni-MCGA (fine-tuned) | 2.8 | 3.1 | 7.8 | 5.3 | 4.1 |

### Ablation Study

| Configuration | ASR CER↓ | S2TT↑ | SEC↑ | SQA↑ | SU↑ | SR↑ |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-7B (original) | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 |
| Qwen-Omni-MCGA (fine-tuned) | — | — | — | — | — | — |

The fine-tuned Qwen-Omni-MCGA surpasses the 30B-parameter Qwen3-Omni on ASR for Shi and Wen (CER 2.8 vs. 3.8), demonstrating the high value of MCGA as a training resource.

### Key Findings

- **Fu is the most challenging genre**: All models yield the highest CER on Fu, attributed to its elaborate rhetoric, dense allusions, and abundant modal particles.
- **SEC is the most challenging task**: Even the strongest model, Qwen3-Omni, scores only 58.4 on SEC; GPT-4o-mini-Audio scores just 5.7 due to safety protocols that cause it to refuse emotion analysis requests.
- **High data consistency**: CER differences across training/validation/test splits are only 0.1 (Qwen3-Omni), confirming the effectiveness of recording quality control.
- **Open-source models match closed-source**: Qwen3-Omni's total score (442.9) surpasses Gemini-3-Flash (440.9), indicating that open-source models have reached a competitive level in the classical Chinese domain.
- **Large gains from fine-tuning small models**: Qwen2.5-Omni (7B) fine-tuned on MCGA outperforms Qwen3-Omni (30B) on ASR for several genres.

## Highlights & Insights

- **Filling a domain gap**: MCGA is the first large-scale, fully copyright-cleared audio corpus specifically targeting classical Chinese literature, genuinely resolving the absence of audio data in this field. All 22,000 recordings are accompanied by signed copyright assignment agreements from the original speakers, definitively addressing the intellectual property challenges of open-source speech datasets.
- **Elegant ECF metric design**: Decomposing speech emotion evaluation into three levels — person recognition, global emotional tone, and sentence-level fidelity — accommodates the particularities of classical literary recitation while remaining operationally tractable for automatic evaluation.
- **Insight afforded by CMC**: Measuring cross-modal consistency via the ratio of speech-to-text task scores clearly exposes the phenomenon of models "relying on text-channel capabilities rather than genuinely understanding speech."
- **Genre-level analysis**: The finding that Fu is the most difficult genre and Ci the easiest provides directional guidance for future genre-specific model optimization.

## Limitations & Future Work

- The corpus contains only standard Mandarin recordings and does not cover dialectal recitation or traditional performance forms such as classical chanting.
- SEC evaluation relies on an LLM judge (DeepSeek API); automatic evaluation of subjective emotional judgments remains an open problem.
- Fine-tuning experiments are validated only on Qwen2.5-Omni-7B and do not cover other base models.
- Cached file truncation prevented complete retrieval of detailed SQA/SU/SR analysis and specific CMC experimental results.
- Future work may extend the corpus to richer audio forms such as classical literary chanting and traditional opera.

## Related Work & Insights

- **vs. ACLUE/WenMind**: These benchmarks cover only the text modality; MCGA is the first to extend classical literary evaluation to the speech dimension.
- **vs. MCS-Bench/Oracle-Bench**: These multimodal benchmarks focus on text + vision; MCGA fills the gap in text + speech evaluation.
- **vs. LibriSpeech/Common Voice**: General-purpose speech datasets target modern languages and cannot handle the allusions, rhetoric, and prosodic features of classical Chinese.
- **vs. CII-Bench**: CII-Bench addresses text–image understanding of Chinese cultural common knowledge; MCGA focuses on deep speech understanding and emotion analysis of classical literature.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale, fully copyright-cleared audio corpus for classical Chinese literature, filling a clear domain gap
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 10 models, 6 tasks, and 5 genres, with rich analytical dimensions
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous metric definitions, and thorough data presentation
- Value: ⭐⭐⭐⭐ Significant contribution to advancing digital classical literary research and MLLM speech capability evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)

</div>

<!-- RELATED:END -->
