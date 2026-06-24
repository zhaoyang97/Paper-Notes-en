---
title: >-
  [Paper Note] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks
description: >-
  [ACL2025][Spoken Language Understanding] This paper proposes the ENDow framework, which systematically analyzes the impact of ASR transcription noise on downstream NLU tasks for the first time. By evaluating task models under different noise levels and categories using a configurable pipeline, the authors find that named entities are the most critical word type and that models can tolerate a certain degree of noise.
tags:
  - "ACL2025"
  - "Spoken Language Understanding"
  - "ASR Noise"
  - "Transcript Cleaning"
  - "WER"
  - "Framework"
date: 2026-05-08
content_hash: da1b8bb5ebf69c14
---

# Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks

**Conference**: ACL2025  
**arXiv**: [2502.13645](https://arxiv.org/abs/2502.13645)  
**Code**: [ENDow](https://github.com/OriShapira/ENDow)  
**Area**: Spoken Language Understanding / ASR Noise Analysis  
**Keywords**: Spoken Language Understanding, ASR Noise, Transcript Cleaning, WER, Framework  

## TL;DR

This paper proposes the ENDow framework, which systematically analyzes the impact of ASR transcription noise on downstream NLU tasks for the first time. By evaluating task models under different noise levels and categories using a configurable pipeline, the authors find that named entities are the most critical word type and that models can tolerate a certain degree of noise.

## Background & Motivation

- **ASR Noise Propagation**: Speech transcribed by ASR systems inevitably introduces errors, which propagate to downstream NLU tasks (such as dialogue summarization, question answering, etc.).
- **Lack of Systematic Analysis**: Existing studies focus only on specific tasks and scenarios for noise impact analysis, lacking a general and configurable evaluation framework.
- **Limitations of the WER Metric**: WER only measures the quantity of errors without distinguishing error types (e.g., noun errors vs. adverb errors), nor does it predict downstream task performance.
- **Discrepancies Across Tasks and Models**: Different downstream tasks have various sensitivities to noise, and different LLMs behave differently across noise levels.
- **Core Motivation**: There is a need for a flexible framework to systematically evaluate the impact of ASR noise on any SLU pipeline, supporting both quantitative analysis and qualitative insights.

## Method

### Overall Architecture (ENDow Pipeline)

The ENDow framework comprises five configurable components, forming a complete evaluation pipeline:

1. **TTS Model**: Converts reference transcripts into audio (used to control the experimental baseline or supplement missing audio).
2. **Acoustic Noise**: Applies degradation to audio at $k$ intensity levels (reverberation + background noise with increasing SNR levels).
3. **ASR System**: Transcribes the noisy audio across all levels, yielding $k+1$ sets of transcripts (including the one from clean audio).
4. **Transcript Cleaning**: Uses $m$ cleaning techniques to partially repair each set of transcripts (controlling for noise types).
5. **Downstream Task Model**: Executes and evaluates the target task on all transcript variants.

Ultimately, $(k+1) \times (m+1)$ transcript variants of different noise levels and types are generated.

### Noise Tolerance Point (NTP)

The NTP is defined as the lowest WER value $w^t_j$ such that:

$$f_j^{lower}(0) = f_j^{upper}(w_j^t)$$

This represents the point where the task score becomes statistically significantly ($p < 0.05$) lower than the noise-free score for the first time, indicating that noise has caused a perceptible impact on model performance.

### Cleaning Effectiveness Score (CES)

Measures the effectiveness of cleaning technique $j$:

$$CES_j = \frac{1}{k+1}\sum_{i=0}^{k} e_{i_j}, \quad e_{i_j} = \frac{\delta s_{i_j}}{\sqrt{\Delta w_{i_j} + \epsilon}}$$

where:
- $\delta s_{i_j} = (s_{i0} - s_{i_j})/s$ is the relative change in the task score.
- $\Delta w_{i_j} = w_{i0} - w_{i_j}$ is the change in WER (the "effort").
- A square root transformation is used to diminish the effect of large WER changes at high noise levels.

The CES jointly evaluates two objectives: maximizing task score improvement with minimal cleaning "effort".

### Analysis Dimensions

The framework supports three types of analysis:
1. **Model Performance vs. Noise**: AUC compares overall noise tolerance, and NTP localizes the tolerance threshold.
2. **Cross-Model Comparison**: Compares the relative performance of different models across various noise intervals on the same plot.
3. **Cleaning Technique Comparison**: Evaluates the effectiveness of repairing different word categories via CES and curve shifts.

## Key Experimental Results

### Experimental Setup

**Three SLU Tasks**:

| Task | Dataset | Type | Granularity | Evaluation Metric |
|------|--------|------|------|----------|
| Summarization | QMSum | Generative | Full text | Pairwise Ranking, ROUGE |
| Question Answering | QAConv | Extractive | Full text | Fuzzy Match, EM, F1 |
| Dialogue Act Classification | MRDA | Classification | Sentence | macro-F1, Accuracy |

**Four LLMs**: Mistral-7B, Llama3-8B, Llama3.1-8B, GPT-4o-mini (all evaluated in a zero-shot setting)

**Configurations**:
- TTS: tortoise-tts
- Noise: Five SNR levels of reverberation + office background noise (yielding 7 transcript sets, including reference and clean audio)
- ASR: Whisper small
- Seven cleaning techniques: Repairing nouns/verbs/adjectives/adverbs/content words/non-content words/named entities

### Model Performance across Noise Levels

**Summarization Task** (QMSum):
- Models can tolerate noise up to approximately WER = 0.2 (NTP lies between 0.07 and 0.30).
- GPT-4o-mini performs best under low noise, but other models outperform it as the noise level increases.
- The differences in AUC among models are not statistically significant ($p < 0.05$).

**Question Answering Task** (QAConv):
- GPT and Llama3.1 significantly outperform the other two models (the latter are limited by smaller context windows, requiring chunk-based processing).
- The same phenomenon is observed: GPT leads under low noise, while Llama3.1 overtakes it under high noise.

**Dialogue Act Classification** (MRDA):
- The NTP is high, but this is attributed to the overall poor performance of the models rather than noise having no impact.
- The zero-shot classification performance of all models is far below that of task-specific models.

### Comparison of Cleaning Techniques (CES Ranking)

| Rank | Summarization (GPT) | QA (GPT) | Dialogue Classification (GPT) |
|------|-----------|-----------|---------------|
| 1 | Named Entities 0.499 | Named Entities 0.311 | Named Entities 0.735 |
| 2 | Content Words 0.479 | Nouns 0.211 | Adjectives 0.290 |
| 3 | Nouns 0.305 | Content Words 0.202 | Non-Content Words 0.285 |
| 4 | Non-Content Words 0.181 | Non-Content Words 0.133 | Content Words 0.212 |
| 5 | Adjectives 0.135 | Adjectives 0.120 | Nouns 0.203 |
| 6 | Verbs 0.073 | Verbs 0.090 | Verbs 0.158 |
| 7 | Adverbs -0.023 | Adverbs 0.071 | Adverbs 0.107 |

### Key Findings

1. **Named Entities are Crucial**: In both summarization and QA tasks, repairing named entities is the most efficient cleaning strategy.
2. **Verbs and Adverbs are Less Important**: Surprisingly, repairing verbs and adverbs offers minimal task performance gains; sometimes adverbs are even counterproductive.
3. **Noise Type Matters More Than Quantity**: Transcripts with WER = 0.4 but with content words repaired exhibit far better summarization quality than transcripts of equivalent WER without targeted repairs.
4. **Model Performance Rankings Shift with Noise**: GPT-4o-mini performs best under low-noise conditions, but is surpassed by other models under high-noise levels.
5. **Significant Task Differences**: In dialogue classification tasks, the importance of non-content words (function words) is higher than in document-level tasks.
6. **Existence of a "Diminishing Returns" Threshold**: Past a certain noise level, the marginal benefit of further noise reduction is negligible.

## Highlights & Insights

1. **Pioneering General SLU Evaluation Framework**: ENDow is the first framework applicable to any task, dataset, and model for evaluating the impact of ASR noise.
2. **Two New Metrics (NTP and CES)**: They quantify the noise tolerance threshold and the efficacy of cleaning techniques, offering high practical value.
3. **Challenging the Adequacy of WER**: The experimental results thoroughly demonstrate the insufficiency of WER as an evaluation metric, as different types of errors under the same WER lead to completely different downstream impacts.
4. **Applicability to Non-SLU Datasets**: Via the TTS module, the framework supports using any text dataset for SLU analysis.
5. **Core Role of Named Entities**: This provides clear direction for ASR system design, suggesting that transcribing named entities accurately should be prioritized.
6. **Model Robustness Ranking Reversal**: Reveals the complementary characteristics of different models under varying noise conditions, which has practical reference value for deployment.

## Limitations & Future Work

1. **Limited Noise Types**: Only reverberation + background noise were used, failing to cover scenarios like dialects, overlapping speakers, or microphone discrepancies.
2. **Language Constraint**: Experiments are restricted to English; SLU in other languages may exhibit different noise sensitivity patterns.
3. **Cleaning Techniques Depend on Reference**: The cleaning simulation in the experiments requires reference transcripts to localize specific word types, which is not an unsupervised method applicable in practical systems.
4. **Dependency on Evaluation Metrics**: Analytical conclusions are influenced by the choice of evaluation metrics; different metrics may yield different results.
5. **Limited Model Selection**: Only four LLMs were evaluated, some of which were constrained by context window sizes, requiring chunk-based processing.
6. **Bias Introduced by TTS**: Using TTS synthetic audio as the experimental baseline might introduce distribution discrepancies compared to natural speech, which could affect the conclusions.

## Related Work & Insights

- **ASR Transcript Correction**: Spelling correction, disfluency removal, punctuation restoration, and general error correction methods.
- **Noise-Robust NLU**: Methods that train models using noisy transcripts to enhance robustness.
- **Noise Impact Analysis**: Prior research analyzing the effects of transcription noise on specific tasks (summarization, QA, classification).
- **ASR-GLUE**: Analyzes the SLU capability of GPT series on short transcribed texts, but does not cover full-length long dialogues.
- **Studies on the WER Metric**: Prior work indicating that WER fails to capture discrepancies in error types or predict downstream performance.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The framework design is systematic, and the NTP and CES metrics are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive experimental matrix covering 3 tasks × 4 models × 7 noise levels × 7 cleaning techniques.
- **Value**: ⭐⭐⭐⭐ Offers direct guidelines for SLU system design and ASR optimization strategies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear system description and in-depth analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)
- [\[ACL 2025\] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)
- [\[ACL 2025\] ChuLo: Chunk-Level Key Information Representation for Long Document Understanding](chulo_chunk-level_key_information_representation_for_long_document_understanding.md)
- [\[ACL 2025\] DeepRTL2: A Versatile Model for RTL-Related Tasks](deeprtl2_a_versatile_model_for_rtl-related_tasks.md)
- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)

</div>

<!-- RELATED:END -->
