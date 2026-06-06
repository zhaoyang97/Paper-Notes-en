---
title: >-
  [Paper Note] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
description: >-
  [ACL 2026][Audio & Speech][Classical Chinese literary speech corpus] This paper constructs MCGA, the first large-scale (119 hours, 22,000 samples) copyright-cleared audio corpus for Classical Chinese literature. It cover…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Classical Chinese literary speech corpus"
  - "multimodal large language models"
  - "speech emotion analysis"
  - "cross-modal consistency"
  - "Classical Chinese studies"
date: 2026-05-08
content_hash: a7362e3a4cf59ba4
---

# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus

**Conference**: ACL 2026  
**arXiv**: [2601.09270](https://arxiv.org/abs/2601.09270)  
**Code**: [https://github.com/yxduir/MCGA](https://github.com/yxduir/MCGA)  
**Area**: Speech and Natural Language Processing / Classical Chinese Literature  
**Keywords**: Classical Chinese literary speech corpus, multimodal large language models, speech emotion analysis, cross-modal consistency, Classical Chinese studies

## TL;DR

This paper constructs MCGA, the first large-scale (119 hours, 22,000 samples) copyright-cleared audio corpus for Classical Chinese literature. It covers five genres (Fu, Poetry, Prose, Ci, and Qu) and six speech tasks (ASR/S2TT/SEC/SQA/SU/SR). Evaluations of 10 multimodal large language models (MLLMs) reveal significant deficiencies in current models regarding the comprehension of Classical Chinese speech.

## Background & Motivation

**Background**: The rapid development of Multimodal Large Language Models (MLLM) has brought new possibilities to Classical Chinese Studies (CCS). However, existing research primarily focuses on text (ACLUE, WenMind, etc.) and visual (Oracle-Bench, MCS-Bench, etc.) modalities, while the speech dimension of classical literature remains nearly blank. The absence of high-quality, domain-specific audio corpora prevents systematic evaluation and improvement of MLLM capabilities in understanding classical Chinese speech.

**Limitations of Prior Work**: (1) Existing Chinese cultural datasets mostly involve text or image modalities without parallel classical literature speech data; (2) Rare resources involving Chinese speech are mainly oriented toward modern Chinese and fail to cover the unique rhetoric, allusions, and prosodic features of classical literature; (3) Copyright issues have hindered the construction of open-source CCS audio datasets, as recitations available online are often restricted for research distribution.

**Key Challenge**: While MLLMs possess strong text and visual understanding capabilities, the evaluation infrastructure for classical Chinese speech comprehension is entirely missing. Without a speech corpus, evaluation is impossible, which in turn stalls model progress in this field.

**Goal**: To construct a copyright-cleared, multi-genre, and multi-task Classical Chinese audio corpus and establish a systematic evaluation framework to comprehensively assess the current capabilities of MLLMs in classical literature speech understanding.

**Key Insight**: Approaching the problem from two dimensions: genre diversity and task diversity. In terms of genre, it covers the five most important forms in Chinese literary history (Fu, Poetry, Prose, Ci, and Qu). In terms of tasks, it designs a six-level progressive task system ranging from basic perception (ASR) to advanced reasoning (SR).

**Core Idea**: Recruit 28 native speakers to manually record all audio with copyright transfers, utilize LLMs to generate Q&A pairs with triple-verification to ensure quality, and build a parallel corpus supporting 6 speech tasks and 4 text tasks.

## Method

### Overall Architecture

The construction of the MCGA corpus consists of three stages: (1) Data collection and preprocessing—collecting public domain classical texts and Pinyin, cleaning them, and segmenting the text to limit recording duration to under 30 seconds; (2) Manual recording—28 native speakers recorded all texts according to unified specifications, followed by two rounds of quality control via MLLM and human inspection; (3) Text data construction—utilizing DeepSeek-V3.2 to generate multi-task Q&A pairs based on the full literary context of each segment, followed by triple-verification (DeepSeek-V3.2, GPT-5-mini, and Gemini-3-Flash) to filter invalid samples.

### Key Designs

1. **Six-level Progressive Task System**:
    - **Function**: Covers the full spectrum of speech understanding from low-level perception to high-level reasoning.
    - **Mechanism**: Designs six core speech tasks—ASR (Automatic Speech Recognition), S2TT (Speech-to-Text Translation), SEC (Speech Emotion Characterization), SQA (Spoken Question Answering), SU (Speech Understanding), and SR (Speech Reasoning). ASR tests basic transcription; S2TT requires cross-lingual translation from ancient Chinese to modern English; SEC requires the model to identify speaker characteristics and perform sentence-by-sentence emotion analysis; SQA is open-ended factual Q&A; SU and SR test content-based understanding and external knowledge-based reasoning, respectively. Parallel text data also supports MT, QA, LU, and LR tasks.
    - **Design Motivation**: A single task cannot comprehensively evaluate the classical literature understanding of MLLMs. The progressive design allows for precise pinpointing of model bottlenecks at different cognitive levels.

2. **Emotion Characterization Fidelity (ECF)**:
    - **Function**: Provides a fine-grained automatic evaluation metric for classical literature speech emotion characterization tasks.
    - **Mechanism**: ECF consists of three sub-metrics—ECF-P (Person identification, 0-2 points, detecting age and gender accuracy); ECF-G (Global sentiment tone, 0-3 points, assessing the richness and accuracy of the overall emotional atmosphere); and ECF-S (Sentence-level fidelity, 0-5 points, assessing transcription and sentiment analysis quality per sentence). The final score is normalized to a 100-point scale.
    - **Design Motivation**: Existing metrics focus on modern speech emotion classification (e.g., happy/sad) and fail to capture the complex emotional layers in classical recitations—such as the mixture of reluctance, open-mindedness, and ambition in a parting poem.

3. **Cross-Modal Consistency (CMC)**:
    - **Function**: Quantifies the consistency of MLLM performance between speech and text input modalities.
    - **Mechanism**: $CMC = \frac{1}{3}\left(\frac{SQA}{QA} + \frac{SU}{LU} + \frac{SR}{LR}\right) \times 100$. It represents the average ratio of scores between the three speech tasks (SQA/SU/SR) and their corresponding text tasks (QA/LU/LR). A CMC value closer to 100 indicates that the model's speech understanding is more consistent with its text understanding.
    - **Design Motivation**: An ideal MLLM should provide consistent answers regardless of speech or text input. CMC reveals whether a model truly "understands" speech content or merely relies on its text-channel capabilities.

### Loss & Training

Fine-tuning experiments used Qwen2.5-Omni-7B as the base model, employing LoRA ($r=8, \alpha=32$) trained for 3 epochs on the MCGA training set. The AdamW optimizer was used with a learning rate of $1 \times 10^{-4}$ on 4 A100 GPUs.

## Key Experimental Results

### Main Results

| Model | ASR (CER↓) | S2TT (LLM-B↑) | SEC (ECF↑) | SQA (F1↑) | SU (Acc↑) | SR (Acc↑) | Total Score↑ |
|------|-----------|---------------|------------|----------|----------|----------|------|
| GPT-4o-mini-Audio | 20.6 | 43.5 | 5.7 | 30.6 | 74.8 | 70.2 | 304.2 |
| Gemini-3-Flash | 6.1 | 74.0 | 54.0 | 48.7 | 86.6 | 83.7 | 440.9 |
| Qwen2.5-Omni-7B | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 | 380.7 |
| Qwen3-Omni-30B | 4.4 | 67.6 | 58.4 | 51.5 | 86.9 | 82.9 | 442.9 |
| Step-Audio-2-mini | 9.9 | 41.9 | 36.8 | 45.2 | 80.5 | 80.4 | 374.9 |
| Phi-4-Multimodal | 59.6 | 27.5 | 12.7 | 24.5 | 50.6 | 54.4 | 210.1 |

Qwen3-Omni achieved the highest total score (442.9), leading in ASR, SEC, SQA, and SU. Gemini-3-Flash performed best in S2TT and SR, reflecting the strengths of closed-source models in English generation and reasoning.

| Model | Poetry CER | Ci CER | Qu CER | Fu CER | Prose CER |
|------|--------|--------|--------|--------|--------|
| Qwen3-Omni-30B | 3.8 | 2.8 | 4.1 | 6.2 | 4.3 |
| Qwen2.5-Omni-7B | 9.9 | 7.5 | 8.9 | 14.8 | 8.8 |
| **Ours** (Qwen-Omni-MCGA) | 2.8 | 3.1 | 7.8 | 5.3 | 4.1 |

### Ablation Study

| Configuration | ASR CER↓ | S2TT↑ | SEC↑ | SQA↑ | SU↑ | SR↑ |
|------|---------|-------|------|------|-----|-----|
| Qwen2.5-Omni-7B (Original) | 10.1 | 49.7 | 37.0 | 43.5 | 81.3 | 79.3 |
| **Ours** (Qwen-Omni-MCGA) | — | — | — | — | — | — |

The fine-tuned Qwen-Omni-MCGA outperformed the 30B parameter Qwen3-Omni in ASR for Poetry and Prose (CER 2.8 vs 3.8), proving the high value of MCGA as a training resource.

### Key Findings

- **Fu is the most difficult genre**: All models showed the highest CER on Fu, due to its ornate rhetoric, frequent allusions, and numerous modal particles.
- **SEC is the hardest task**: Even the strongest Qwen3-Omni scored only 58.4 in SEC; GPT-4o-mini-Audio scored only 5.7 due to safety protocols refusing sentiment analysis requests.
- **High data consistency**: The CER difference between train/val/test sets was only 0.1 (Qwen3-Omni), validating the effectiveness of recording quality control.
- **Open-source models bridging the gap**: Qwen3-Omni's total score (442.9) surpassed Gemini-3-Flash (440.9), indicating that open-source models have reached competitive levels in the Classical Chinese domain.
- **Significant gains for small models**: After fine-tuning on MCGA, the 7B Qwen2.5-Omni surpassed the 30B Qwen3-Omni in ASR for certain genres.

## Highlights & Insights

- **Filling the domain gap**: MCGA is the first large-scale, copyright-cleared audio corpus specifically for Classical Chinese literature, moving the field from zero to one in speech data availability. All 22,000 audio clips involve signed copyright transfers, resolving intellectual property dilemmas for open-source speech datasets.
- **Sophisticated ECF metric**: Decomposing speech emotion evaluation into person identification, global tone, and sentence-level fidelity accommodates the specifics of classical recitation while remaining operationally feasible for automatic evaluation.
- **Insight from the CMC metric**: Measuring cross-modal consistency through speech-to-text task ratios clearly exposes models that "depend on the text channel rather than truly understanding speech."
- **Analysis by genre**: The discovery that "Fu is the hardest while Ci is the easiest" provides direction for future model optimizations targeting specific genres.

## Limitations & Future Work

- The corpus currently only includes standard Mandarin recordings and does not cover traditional performance forms like dialect recitations or chanting.
- SEC evaluation relies on LLM judges (DeepSeek API); automatic evaluation of subjective emotional judgment remains an open problem.
- Training experiments were only validated on Qwen2.5-Omni-7B, without covering fine-tuning effects for other base models.
- Future work could extend to richer audio forms such as classical chanting and traditional opera.

## Related Work & Insights

- **vs ACLUE/WenMind**: These benchmarks only cover the text modality; MCGA extends classical literature evaluation to the speech dimension for the first time.
- **vs MCS-Bench/Oracle-Bench**: These multimodal benchmarks focus on text+vision; MCGA fills the text+speech gap.
- **vs LibriSpeech/Common Voice**: General speech datasets target modern languages and cannot handle the allusions, rhetoric, and phonological features of classical Chinese.
- **vs CII-Bench**: While CII-Bench focuses on image-text understanding of Chinese cultural common sense, MCGA focuses on deep speech understanding and sentiment analysis of classical literature.

## Rating

- Novelty: ⭐⭐⭐⭐ First copyright-cleared audio corpus for Classical Chinese; fills a clear gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 10 models, 6 tasks, and 5 genres.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous metric definitions, and sufficient data presentation.
- Value: ⭐⭐⭐⭐ Significant for driving digitized classical literature research and MLLM speech capability evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[NeurIPS 2025\] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](../../NeurIPS2025/audio_speech/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)

</div>

<!-- RELATED:END -->
