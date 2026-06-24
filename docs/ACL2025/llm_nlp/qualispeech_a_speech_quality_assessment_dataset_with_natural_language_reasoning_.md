---
title: >-
  [Paper Note] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning
description: >-
  [ACL2025][LLM (Other)][Speech Quality Assessment] This paper proposes QualiSpeech, the first speech quality assessment dataset featuring 11-dimensional annotations and detailed natural language reasoning descriptions, along with an accompanying evaluation benchmark. It demonstrates that fine-tuned audio LLMs can generate detailed descriptions of noise and distortion, and highlights the potential of reasoning-enhanced quality assessment.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Speech Quality Assessment"
  - "Natural Language Description"
  - "Audio Large Language Model"
  - "Low-level Speech Perception"
  - "Reasoning"
date: 2026-05-08
content_hash: 0e6033ef057640a1
---

# QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning

**Conference**: ACL2025  
**arXiv**: [2503.20290](https://arxiv.org/abs/2503.20290)  
**Code**: [HuggingFace: tsinghua-ee/QualiSpeech](https://huggingface.co/datasets/tsinghua-ee/QualiSpeech)  
**Area**: others (Speech Quality Assessment)  
**Keywords**: Speech Quality Assessment, Natural Language Description, Audio Large Language Model, Low-level Speech Perception, Reasoning  

## TL;DR

This paper proposes QualiSpeech, the first speech quality assessment dataset featuring 11-dimensional annotations and detailed natural language reasoning descriptions, along with an accompanying evaluation benchmark. It demonstrates that fine-tuned audio LLMs can generate detailed descriptions of noise and distortion, and highlights the potential of reasoning-enhanced quality assessment.

## Background & Motivation

Speech quality assessment is a fundamental task for evaluating speech synthesis systems and communication network distortions. Current mainstream methods primarily focus on **MOS score prediction**—generating a single numerical score representing perceived quality. While numerical scores are convenient for comparison, they fail to reveal the underlying reasons behind the scores and lack specific diagnostic quality analysis.

Natural language descriptions provide much richer feedback. For instance, "there is electrical-sounding distortion noise between 1.5 and 2.5 seconds" contains far more diagnostic information than a simple distortion score. However, existing datasets lack the comprehensive annotations required to support such natural language-based assessments.

Meanwhile, with the rapid development of audio LLMs (such as SALMONN, Qwen-Audio), natural language speech quality assessment has become technically viable. Nonetheless, low-level speech perception tasks remain largely overlooked in the training and evaluation of these models.

## Method

### Dataset Construction

#### Data Sources

The QualiSpeech training set contains **10,558** samples with a balanced distribution of sources:

- **Synthetic Speech**:
    - BVCC dataset (samples from Blizzard/VCC challenges over the years)
    - 10 state-of-the-art open-source TTS models (ChatTTS, XTTS v2, CosyVoice, F5-TTS, E2 TTS, OpenVoice V1/V2, Parler-TTS Mini/Large, VoiceCraft-830M)
    - Each TTS model generates 72 samples, using sentences sourced from the SOMOS corpus (covering 10 domains, including dialogue, news, Wikipedia, etc.)
    - 20% of synthetic data is mixed with noise (noise sources from the DNS Challenge, SNR $0-15\text{dB}$)

- **Real Speech**:
    - NISQA dataset (simulated and live communication network distortions)
    - GigaSpeech (audiobooks, podcasts, YouTube recordings)

#### Annotation Process (Three-step Approach)

**Step 1: Listening Test Annotation**

Detailed annotations of 11 low-level speech features:
- **7 numerical ratings** (on a 5-point scale): noise, distortion, speech rate, continuity, listening effort, naturalness, overall quality
- **4 specific descriptions**: noise (type + timestamps), distortion (type + timestamps), unnatural pauses (timestamps), vocal characteristics (age/gender/intonation)

**Step 2: GPT Generation of Natural Language Descriptions**

All annotated dimensions are input into GPT-4o-mini to generate descriptions in a Chain-of-Thought (CoT) format: first conducting a dimension-by-dimension analysis of low-level features, and then synthesizing them into an overall quality assessment.

**Step 3: Human Rectification**

Annotators review and correct:
- Inconsistencies between descriptions and annotations
- GPT hallucinations or unsupported assertions
- Missing dimensions (by adding them manually)
- Coherence of the reasoning logic

### QualiSpeech Benchmark

A multiple-choice benchmark is established covering 7 low-level speech understanding dimensions. Audio LLMs are required to select the most appropriate score for speech samples, evaluating their low-level speech perception capabilities.

### Evaluation Metrics

- **Numerical Ratings**: PCC (Pearson Correlation Coefficient)
- **Specific Descriptions**:
    - Precision/Recall: Capability to detect the presence of noise/distortion
    - Correlation Score generated by GPT: Measuring overall description relevance
    - IoU: Intersection over Union of predicted time intervals and ground truth
- **Natural Language Descriptions**: First extract dimensional information using GPT, then apply the corresponding metrics

## Experimental Results

### Performance of Open-Source Audio LLMs on Benchmark

| Model | Noise PCC | Distortion PCC | Overall PCC |
|------|-----------|----------------|-------------|
| SALMONN-7B | 0.003 | 0.013 | 0.084 |
| SALMONN-13B | 0.001 | 0.002 | 0.100 |
| Qwen-Audio-Chat | 0.014 | -0.003 | 0.250 |
| Qwen2-Audio-7B | -0.048 | 0.056 | 0.112 |
| WavLLM | -0.021 | -0.069 | 0.071 |

**Conclusion**: Existing open-source audio LLMs almost completely fail at low-level speech quality assessment. SALMONN-7B predicts the same score for over 80% of the samples, exhibiting severe numerical preference bias.

### Fine-Tuned Models Learning Low-Level Features

Fine-tuning based on SALMONN-7B (Whisper + BEATs encoders, Vicuna backbone, with only Q-former + LoRA trained):

| Training Strategy | Noise PCC | Distortion PCC | Overall PCC |
|---------|-----------|----------------|-------------|
| basic | 0.721 | 0.553 | 0.597 |
| balance | 0.696 | 0.547 | 0.600 |
| joint | 0.693 | 0.595 | 0.636 |
| **joint + balance** | **0.696** | **0.614** | **0.660** |

- Noise PCC reaches $\sim0.7$, indicating that the model can distinguish between semantic and non-semantic components.
- Joint training improves dimensions such as continuity without significant performance degradation.
- **Highlights in specific descriptions**: IoU of noise/distortion reaches $\sim0.8$, showing that the model can accurately localize time segments; vocal gender classification accuracy reaches 98%.

### Natural Language Description Learning

| Description Format | Noise PCC | Distortion PCC | Overall PCC |
|---------|-----------|----------------|-------------|
| revised concise | 0.656 | 0.579 | 0.630 |
| concise with num | **0.703** | 0.571 | 0.622 |
| concise | 0.642 | 0.559 | 0.582 |
| detailed | 0.686 | 0.518 | 0.572 |

- Audio LLMs can indeed generate paragraph-level natural language speech quality evaluations.
- Incorporating numerical scores into descriptions improves overall output quality.
- Correcting GPT hallucinations is critical for high-quality evaluation.
- Description length has minimal impact on performance.

### Exploration of Reasoning Capabilities

| Model | Overall Prediction Accuracy |
|------|-------------------|
| Vicuna-v1.5-7B (Reasoning based on ground-truth features) | 0.28 |
| GPT-4o-mini (Reasoning based on ground-truth features) | **0.46** |

- Fine-tuned models fail to improve overall score prediction through reasoning in natural language reasoning scenarios.
- However, GPT-4o-mini outperforms all fine-tuned models when reasoning is based on ground-truth low-level features.
- This suggests that reasoning capability is bounded by the capacity of the LLM backbone itself, rather than being a limitation of the methodological framework.

### Key Findings

1. Fine-tuned models generate noise/distortion descriptions with high temporal precision (IoU $\sim0.8$).
2. Joint training on multiple dimensions does not produce optimization conflicts.
3. Training solely on one data type leads to poor generalization on other types.
4. Fusing multi-source data yields comprehensive performance improvements.

## Highlights & Insights

1. **First Speech Quality Assessment Dataset with Natural Language**: Fills a crucial gap by transitioning speech quality assessment from numerical metrics to natural language descriptions.
2. **Comprehensive 11-Dimensional Annotation**: More thorough than NISQA (4 dimensions), and integrates both numerical ratings and descriptive annotations.
3. **Unified Evaluation of Synthetic & Real Speech**: Prior studies typically treated synthetic and real speech separately, whereas QualiSpeech unifies them for the first time.
4. **Pragmatic and Efficient Three-Step Annotation**: The GPT generation + human correction pipeline perfectly balances annotation cost and data quality.
5. **Revealing Reasoning Potential**: While current model reasoning capabilities are limited, using more powerful LLM backbones holds promise for reasoning-based quality assessment.

## Limitations & Future Work

1. Each sample is evaluated by only a single annotator (MOS typically requires multiple evaluators).
2. Certain speech dimensions and diverse data sources are not yet covered.
3. Fine-tuned models are constrained by the reasoning capability of the LLM backbone (Vicuna-7B), failing to fully unleash the potential of natural language reasoning.
4. The dataset is English-only and does not cover other languages.

## Related Work & Insights

- **Speech Quality Assessment Datasets**: BVCC (synthetic speech MOS), NISQA (communication network distortion + multidimensional ratings), SOMOS (sentence-level assessment).
- **Audio LLMs**: SALMONN, Qwen-Audio, WavLLM, etc., perform exceptionally well on high-level language understanding but disregard low-level perception.
- **LLMs for Speech Quality Assessment**: Existing works still primarily focus on MOS prediction and do not fully exploit the natural language generation capabilities of LLMs.

## Rating

⭐⭐⭐⭐ — The dataset design is comprehensive and innovative, filling an important gap. The experiments systematically validate both the capabilities and limits of audio LLMs in understanding low-level speech features. The main drawbacks are that the reasoning capability of the fine-tuned model is constrained by a weaker backbone, and there is still room to improve the data quality of single-annotator labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] GAMEBoT: Transparent Assessment of LLM Reasoning in Games](gamebot_transparent_assessment_of_llm_reasoning_in_games.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] TEXT2ARCH: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions](../../ICLR2026/llm_nlp/text2arch_a_dataset_for_generating_scientific_architecture_diagrams_from_natural.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)

</div>

<!-- RELATED:END -->
