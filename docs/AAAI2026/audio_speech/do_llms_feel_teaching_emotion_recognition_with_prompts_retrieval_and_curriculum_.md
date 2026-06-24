---
title: >-
  [Paper Note] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning
description: >-
  [AAAI2026][Audio & Speech][Emotion Recognition in Conversation] This work proposes the PRC-Emo framework, which integrates explicit/implicit emotion prompts, a dedicated retrieval repository, and curriculum learning strategies to comprehensively enhance LLM performance on the Emotion Recognition in Conversation (ERC) task, achieving state-of-the-art (SOTA) results on the IEMOCAP and MELD benchmarks.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Emotion Recognition in Conversation"
  - "Prompt Engineering"
  - "Retrieval-Augmented Generation"
  - "Curriculum Learning"
  - "LLM Fine-tuning"
date: 2026-05-08
content_hash: 52258c7ed29f3fb1
---

# Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning

**Conference**: AAAI2026  
**arXiv**: [2511.07061](https://arxiv.org/abs/2511.07061)  
**Code**: [LiXinran6/PRC-Emo](https://github.com/LiXinran6/PRC-Emo)  
**Area**: Audio and Speech  
**Keywords**: Emotion Recognition in Conversation, Prompt Engineering, Retrieval-Augmented Generation, Curriculum Learning, LLM Fine-tuning

## TL;DR

This work proposes the PRC-Emo framework, which integrates explicit/implicit emotion prompts, a dedicated retrieval repository, and curriculum learning strategies to comprehensively enhance LLM performance on the Emotion Recognition in Conversation (ERC) task, achieving state-of-the-art (SOTA) results on the IEMOCAP and MELD benchmarks.

## Background & Motivation

Emotion Recognition in Conversation (ERC) is a critical task for achieving natural human-computer interaction. With the rise of LLMs, generative framework-based ERC methods (such as InstructERC and BiosERC) have substantially outperformed traditional discriminative models (RNNs, GNNs, and PLM series). However, existing LLM-ERC methods still suffer from three core limitations:

1. **Prompt designs ignore implicit emotions**: Most existing works predict emotions based solely on the literal meaning of text, failing to distinguish between the speaker's "expressed emotion" (explicit) and "truly felt emotion" (implicit). For example, a person might sound optimistic verbally while feeling anxious internally.
2. **Limited quality of demonstration retrieval repositories**: The demonstration repositories in existing methods only use training sets, resulting in a single coverage scenario and limited generalization capability.
3. **Coarse training strategies**: There is a lack of systematic modeling for sample difficulty, resulting in inadequate handling of class imbalance and complex emotional transitions.

## Core Problem

How can a unified training framework be designed to enable LLMs to simultaneously comprehend both explicit and implicit emotions in conversation, while leveraging high-quality retrieval and progressive training strategies to improve emotion accuracy in class-imbalanced scenarios?

## Method

### Overall Architecture: PRC-Emo

PRC-Emo comprises three core modules: **P**rompt (prompt engineering), **R**etrieval (demonstration retrieval), and **C**urriculum learning. The overall framework is divided into two phases: external knowledge extraction and emotion label prediction.

### 1. External Supplementary Knowledge Extraction (Prompt Engineering)

For each dialogue turn, an external LLM (Qwen3-14B) is utilized to generate three types of auxiliary knowledge:

- **Explicit Emotion Interpretation**: Analyzing the emotion directly expressed by the speaker through language based on the historical context.
- **Implicit Emotion Interpretation**: Inferring the internal emotional state that the speaker might truly feel but does not directly express.
- **Speaker Characteristic**: Summarizing the speaker's personality and behavioral patterns based on the complete dialogue.

These knowledge components are injected as structured text into the subsequent emotion prediction prompt, guiding the model to focus on multi-layered emotional expressions.

### 2. Retrieval Template Module

The final emotion prediction prompt consists of five parts:

- **Instruction**: Defining the model's role and task objective.
- **Historical Content**: The dialogue context and speaker information within a history window $w$.
- **External Knowledge**: The explicit/implicit emotion interpretations and speaker characteristics extracted in the previous step.
- **Demonstration Retrieval**: Retrieving the top-3 most similar demonstration pairs from the retrieval repository using SBERT cosine similarity.
- **Label Statement**: Constraining the output to a pre-defined set of emotion labels.

**Dedicated Demonstration Retrieval Repository for ERC**: This work constructs the first dedicated retrieval repository for ERC, containing 36,712 utterances:

| Source | Count |
|---|---|
| Self-built dataset (GPT-4o generation + manual verification) | 14,009 |
| IEMOCAP training set | 5,163 |
| MELD training set | 9,989 |
| EmoryNLP training set | 7,551 |

The self-built dataset covers six major scenarios: medical, workplace, education, family, social, and entertainment. It employs a "label masking + two-annotator independent annotation" strategy to ensure quality, with three rounds of generation-filtering iteration to achieve class balance.

### 3. Curriculum Learning

**Dialogue Difficulty Metric**: A difficulty function based on weighted emotion shift frequency is proposed, which simultaneously considers emotional changes within the same speaker and across different speakers (prior methods only considered the same speaker).

- Emotion similarity is calculated based on a 2D arousal-valence emotion wheel: $s_{ij} = \max(\cos(\theta_{ij}), 0)$
- Weighted emotion shift: $N^{WES} = k \times s_{ij} + b$
- Dialogue difficulty: $DIF(c_i) = \frac{WES_{same}(c_i) + WES_{diff}(c_i) + N_{sp}(c_i)}{N_u(c_i) + N_{sp}(c_i)}$

where $WES_{same}$ and $WES_{diff}$ represent the sum of weighted emotion shifts for the same speaker and across different speakers, respectively, $N_{sp}$ denotes the number of speakers (serving as a smoothing factor), and $N_u$ is the total number of utterances.

**Training Scheduler**: Data is partitioned into $n$ buckets based on difficulty (easy $\rightarrow$ hard). In the early training stages, only easy buckets are utilized, with harder buckets progressively introduced, eventually training continuously on the entire dataset.

### Fine-tuning Method

Efficient fine-tuning is conducted based on LoRA. Qwen2.5-7B-Instruct is used for IEMOCAP, and Qwen3-8B is used for MELD. All experiments are conducted on a single NVIDIA 4090D GPU.

## Key Experimental Results

### Main Results (Weighted F1)

| Method | IEMOCAP | MELD |
|------|---------|------|
| InstructERC | 71.39 | 69.15 |
| BiosERC | 71.19 | 69.83 |
| **PRC-Emo (Ours)** | **71.95** | **70.44** |

Compared to the best baselines, InstructERC / BiosERC, the performance increases by +0.76% on IEMOCAP and +0.61% on MELD.

### Ablation Study

| Configuration | IEMOCAP | MELD |
|------|---------|------|
| PRC-Emo (Full) | 71.95 | 70.44 |
| w/o Curriculum | 71.52 (↓0.43) | 70.07 (↓0.37) |
| w/o Retrieval + Curriculum | 70.74 (↓1.21) | 69.62 (↓0.82) |
| w/o Prompt + Retrieval + Curriculum | 68.54 (↓3.41) | 68.72 (↓1.72) |

The Prompt module contributes the most (with a drop of 2.20 on IEMOCAP after removal), indicating that explicit/implicit emotion interpretation is crucial for the model to comprehend emotional states.

### Prompt Design Ablation

Removing the implicit emotion interpretation (w/o I + R) leads to a drop of 1.68 on IEMOCAP, verifying the importance of implicit emotion modeling.

## Highlights & Insights

1. **Explicit + Implicit Dual-Channel Emotion Interpretation**: This work systematically distinguishes and models explicit and implicit emotions in ERC for the first time, comprehending speaker states from both "surface expression" and "internal feelings" levels, which is a clear and practically significant approach.
2. **First Dedicated Demonstration Retrieval Repository for ERC**: A 36K-utterance repository is built by fusing multiple datasets, LLM generation, and manual verification, spanning six major real-life scenarios and significantly enhancing few-shot retrieval generalization capability.
3. **Cross-Speaker Emotion Shift Modeling**: In the difficulty design of curriculum learning, emotion shifts are modeled both within the same speaker and across different speakers, offering a more comprehensive assessment than prior methods.
4. **Engineering Friendly**: Training can be completed on a single 4090D GPU, and the code has been open-sourced.

## Limitations & Future Work

1. **Text-Only Modality**: Despite incorporating rich text-based auxiliary knowledge, ERC is inherently a multimodal task (involving voice, tone, and facial expressions). The lack of acoustic/visual information is a notable limitation.
2. **Dependency on External LLMs**: The generation of emotion interpretations relies on Qwen3-14B, which incurs relatively high inference costs, and the interpretation quality is bounded by the external model's capabilities.
3. **Limited Dataset Scale**: Validation is only performed on two English datasets, IEMOCAP (dyadic conversations) and MELD (multi-party conversations). Its applicability to Chinese/other languages and larger-scale datasets remains to be investigated.
4. **Human Cost of the Retrieval Repository**: While the dual-person annotation and three-round iterative filtering ensure high quality, scaling to new domains or languages incurs substantial costs.
5. **Hyperparameter Sensitivity in Curriculum Learning**: Hyperparameters such as the number of buckets $n$ and linear transformation arguments $k, b$ must be tuned manually for different datasets, indicating a lack of an adaptive mechanism.

## Related Work & Insights

- **vs InstructERC**: InstructERC first introduced generative architectures to ERC, but its retrieval repository utilizes only the training set, and it fails to distinguish between explicit and implicit emotions. PRC-Emo achieves substantial improvements in both prompt design and retrieval repository quality.
- **vs BiosERC**: BiosERC introduces speaker background information to enrich prompts, on top of which PRC-Emo further incorporates explicit/implicit emotion interpretations and curriculum learning.
- **vs HybridCL / LSDGNN**: These methods only consider same-speaker emotion shifts in their curriculum learning difficulty design, whereas PRC-Emo extends this directly to cross-speaker dimensions to evaluate dialogue complexity more comprehensively.

## Insights & Connections

- The combination of **Retrieval-Augmentation + Curriculum Learning** can be transferred to other dialogue understanding tasks (e.g., intent recognition, dialogue summarization).
- The **explicit/implicit dual-channel** concept is analogous to separating "surface semantics vs. deep semantics," which can be generalized to tasks requiring understanding of implied meanings, such as sarcasm detection and stance detection.
- The construction methodology of the retrieval repository (LLM generation + manual verification + multi-round iteration) provides a reusable paradigm for data augmentation in low-resource scenarios.

## Rating

- Novelty: 3.5/5 (While the individual modules are not entirely novel on their own, the combination method and the systematic modeling of explicit/implicit emotions show originality.)
- Experimental Thoroughness: 4/5 (Ablation studies are detailed, but evaluations are limited to two datasets, lacking cross-lingual or cross-scenario validation.)
- Writing Quality: 4/5 (Clear structure, rich tables and figures, and well-articulated motivation.)
- Value: 3.5/5 (Solid engineering improvements in the ERC domain; the retrieval repository is of reuse value, but the performance gain is relatively minor.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICLR 2026\] Learnable Fractional Superlets with a Spectro-Temporal Emotion Encoder for Speech Emotion Recognition](../../ICLR2026/audio_speech/learnable_fractional_superlets_with_a_spectro-temporal_emotion_encoder_for_speec.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

</div>

<!-- RELATED:END -->
