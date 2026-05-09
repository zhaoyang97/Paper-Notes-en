---
title: >-
  [Paper Note] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning
description: >-
  [AAAI2026][Audio & Speech][Emotion Recognition in Conversation] This paper proposes PRC-Emo, a framework that integrates explicit/implicit emotion prompting, a dedicated retrieval database, and curriculum learning to enhance LLM performance on Emotion Recognition in Conversation (ERC), achieving state-of-the-art results on the IEMOCAP and MELD benchmarks.
tags:
  - AAAI2026
  - "Audio & Speech"
  - Emotion Recognition in Conversation
  - Prompt Engineering
  - Retrieval-Augmented Generation
  - Curriculum Learning
  - LLM Fine-tuning
date: 2026-05-08
content_hash: b1d48b0a3368e693
---

# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning

**Conference**: AAAI2026
**arXiv**: [2511.07061](https://arxiv.org/abs/2511.07061)
**Code**: [LiXinran6/PRC-Emo](https://github.com/LiXinran6/PRC-Emo)
**Area**: Audio & Speech
**Keywords**: Emotion Recognition in Conversation, Prompt Engineering, Retrieval-Augmented Generation, Curriculum Learning, LLM Fine-tuning

## TL;DR

This paper proposes PRC-Emo, a framework that integrates explicit/implicit emotion prompting, a dedicated retrieval database, and curriculum learning to enhance LLM performance on Emotion Recognition in Conversation (ERC), achieving state-of-the-art results on the IEMOCAP and MELD benchmarks.

## Background & Motivation

Emotion Recognition in Conversation (ERC) is a key task for enabling natural human-computer interaction. With the rise of LLMs, generative-architecture-based ERC methods (e.g., InstructERC, BiosERC) have substantially surpassed traditional discriminative models (RNN, GNN, PLM-based). However, existing LLM-based ERC approaches still suffer from three core limitations:

1. **Prompt design neglects implicit emotion**: Most prior work predicts emotion solely from surface-level text, without distinguishing between the emotion a speaker *expresses* (explicit) and the emotion the speaker *actually feels* (implicit). For instance, a person may sound optimistic verbally while feeling anxious internally.
2. **Limited quality of demonstration retrieval databases**: Existing methods construct demonstration databases using only the training set, resulting in limited scenario coverage and restricted generalization.
3. **Coarse training strategies**: There is a lack of systematic modeling of sample difficulty, with insufficient handling of class imbalance and complex emotion transitions.

## Core Problem

How to design a unified training framework that enables LLMs to simultaneously understand explicit and implicit emotions in conversations, and to improve emotion recognition accuracy under class-imbalanced settings through high-quality retrieval and progressive training strategies?

## Method

### Overall Architecture: PRC-Emo

PRC-Emo comprises three core modules: **P**rompt engineering, **R**etrieval, and **C**urriculum learning. The overall pipeline is divided into two stages: external knowledge extraction and emotion label prediction.

### 1. External Knowledge Extraction (Prompt Engineering)

For each conversation, an external LLM (Qwen3-14B) is used to generate three types of auxiliary knowledge:

- **Explicit Emotion Interpretation**: Analyzes the emotions the speaker directly expresses through language, based on the dialogue history.
- **Implicit Emotion Interpretation**: Infers the speaker's underlying emotional state that may not be directly expressed.
- **Speaker Characteristic**: Summarizes the speaker's personality and behavioral patterns based on the full conversation.

This knowledge is injected as structured text into the downstream emotion prediction prompt, guiding the model to attend to multi-level emotional expression.

### 2. Retrieval Template Module

The final emotion prediction prompt consists of five components:

- **Instruction**: Defines the model role and task objective.
- **Historical Content**: Dialogue context within a history window $w$, along with speaker information.
- **External Knowledge**: Explicit/implicit emotion interpretations and speaker characteristics extracted in the previous step.
- **Demonstration Retrieval**: Top-3 most similar demonstration pairs retrieved from the retrieval database using SBERT cosine similarity.
- **Label Statement**: Constrains the output to a predefined set of emotion labels.

**ERC-dedicated demonstration retrieval database**: This work is the first to construct a dedicated retrieval database for ERC, comprising 36,712 utterances:

| Source | Count |
|--------|-------|
| Custom dataset (GPT-4o generated + human verified) | 14,009 |
| IEMOCAP training set | 5,163 |
| MELD training set | 9,989 |
| EmoryNLP training set | 7,551 |

The custom dataset covers six domains—medical, workplace, education, family, social, and entertainment—and employs a "label masking + dual independent annotation" strategy for quality assurance, with three rounds of generation-filtering iteration to achieve class balance.

### 3. Curriculum Learning

**Dialogue difficulty metric**: A difficulty function based on weighted emotion shift frequency is proposed, accounting for emotion transitions both within the same speaker and across different speakers (prior methods only consider same-speaker transitions).

- Emotion similarity is computed on a two-dimensional arousal-valence affective circumplex: $s_{ij} = \max(\cos(\theta_{ij}), 0)$
- Weighted emotion shift: $N^{WES} = k \times s_{ij} + b$
- Dialogue difficulty: $DIF(c_i) = \frac{WES_{same}(c_i) + WES_{diff}(c_i) + N_{sp}(c_i)}{N_u(c_i) + N_{sp}(c_i)}$

where $WES_{same}$ and $WES_{diff}$ denote the total weighted emotion shifts within the same speaker and across speakers, respectively, $N_{sp}$ is the number of speakers (smoothing factor), and $N_u$ is the total number of utterances.

**Training scheduler**: Data is partitioned into $n$ buckets ordered by difficulty (easy → hard). Training begins with only the easy buckets, progressively incorporating harder ones, and finally continues on the full dataset.

### Fine-tuning

LoRA-based efficient fine-tuning is adopted. Qwen2.5-7B-Instruct is used for IEMOCAP and Qwen3-8B for MELD. All experiments are conducted on a single NVIDIA 4090D GPU.

## Key Experimental Results

### Main Results (Weighted F1)

| Method | IEMOCAP | MELD |
|--------|---------|------|
| InstructERC | 71.39 | 69.15 |
| BiosERC | 71.19 | 69.83 |
| **PRC-Emo (Ours)** | **71.95** | **70.44** |

Compared to the best baseline (InstructERC / BiosERC), PRC-Emo achieves gains of +0.76% on IEMOCAP and +0.61% on MELD.

### Ablation Study

| Configuration | IEMOCAP | MELD |
|---------------|---------|------|
| PRC-Emo (full) | 71.95 | 70.44 |
| w/o Curriculum | 71.52 (↓0.43) | 70.07 (↓0.37) |
| w/o Retrieval + Curriculum | 70.74 (↓1.21) | 69.62 (↓0.82) |
| w/o Prompt + Retrieval + Curriculum | 68.54 (↓3.41) | 68.72 (↓1.72) |

The Prompt module contributes the most (removing it causes a 2.20-point drop on IEMOCAP), demonstrating that explicit/implicit emotion interpretation is critical for the model's understanding of emotional states.

### Prompt Design Ablation

Removing implicit emotion interpretation (w/o I + R) leads to a 1.68-point drop on IEMOCAP, validating the importance of implicit emotion modeling.

## Highlights & Insights

1. **Dual-channel explicit + implicit emotion interpretation**: This work is the first to systematically distinguish and model explicit and implicit emotions in ERC, understanding speaker states from both "surface expression" and "inner feeling" perspectives—a conceptually clear and practically meaningful contribution.
2. **First ERC-dedicated demonstration retrieval database**: A 36K-utterance database integrating multiple datasets, LLM generation, and human verification across six life domains, significantly improving few-shot retrieval generalization.
3. **Cross-speaker emotion transition modeling**: The curriculum learning difficulty metric accounts for emotion shifts both within and across speakers, providing a more comprehensive measure of dialogue complexity than prior methods.
4. **Engineering accessibility**: Training requires only a single 4090D GPU, and the code is publicly available.

## Limitations & Future Work

1. **Text-only modality**: Despite the incorporation of rich textual auxiliary knowledge, ERC is inherently a multimodal task (vocal tone, facial expression); the absence of acoustic/visual information is a notable limitation.
2. **Dependence on external LLM**: Emotion interpretation generation relies on Qwen3-14B, incurring high inference costs, with interpretation quality bounded by the external model's capability.
3. **Limited dataset scale**: Validation is restricted to two English datasets—IEMOCAP (dyadic conversations) and MELD (multi-party conversations)—leaving applicability to other languages and larger-scale datasets unexplored.
4. **Annotation cost of retrieval database**: The dual-annotation and three-round iterative filtering approach ensures quality but entails high costs when extending to new domains or languages.
5. **Curriculum learning hyperparameter sensitivity**: The number of buckets $n$ and the linear transformation parameters $k, b$ require dataset-specific tuning, lacking an adaptive mechanism.

## Related Work & Insights

- **vs InstructERC**: InstructERC was the first to introduce a generative architecture for ERC, but its retrieval database relies solely on the training set and does not distinguish explicit from implicit emotion. PRC-Emo offers substantive improvements in both prompt design and retrieval database quality.
- **vs BiosERC**: BiosERC enriches prompts with speaker background information; PRC-Emo further incorporates explicit/implicit emotion interpretation and curriculum learning.
- **vs HybridCL / LSDGNN**: These methods consider only same-speaker emotion transitions in curriculum learning difficulty design; PRC-Emo extends this to the cross-speaker dimension, providing a more comprehensive measure of dialogue complexity.

Methodologically, the combination of **retrieval augmentation + curriculum learning** is transferable to other conversational understanding tasks (intent recognition, dialogue summarization, etc.). The **explicit/implicit dual-channel** paradigm—analogous to separating surface semantics from deep semantics—can be generalized to tasks requiring understanding of implied meaning, such as sarcasm detection and stance detection. The retrieval database construction methodology (LLM generation + human verification + iterative filtering) offers a reusable paradigm for data augmentation in low-resource settings.

## Rating

- **Novelty**: 3.5/5 — Individual modules are not entirely novel, but the combination and the systematic modeling of explicit/implicit emotion introduce meaningful originality.
- **Experimental Thoroughness**: 4/5 — Ablations are detailed, but evaluation is limited to two datasets with no cross-lingual or cross-domain validation.
- **Writing Quality**: 4/5 — Well-structured, with rich figures and tables and well-motivated problem formulation.
- **Value**: 3.5/5 — A solid engineering contribution to the ERC field; the retrieval database has reuse value, though the performance gains are modest.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](../../ICLR2026/audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)

<!-- RELATED:END -->
