---
title: >-
  [Paper Note] StressTest: Can YOUR Speech LM Handle the Stress?
description: >-
  [ACL 2026][Audio & Speech][Sentence Stress] This paper proposes StressTest, a benchmark for evaluating the ability of speech language models (SLMs) to understand the meaning conveyed by sentence stress. Evaluations revea…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Sentence Stress"
  - "Speech Language Models"
  - "Prosody Understanding"
  - "Benchmark"
  - "Synthetic Data"
date: 2026-05-08
content_hash: ce64355ec7fc63de
---

# StressTest: Can YOUR Speech LM Handle the Stress?

**Conference**: ACL 2026
**arXiv**: [2505.22765](https://arxiv.org/abs/2505.22765)  
**Code**: [Project Page](https://pages.cs.huji.ac.il/adiyoss-lab/stresstest)  
**Area**: Speech Understanding
**Keywords**: Sentence Stress, Speech Language Models, Prosody Understanding, Benchmark, Synthetic Data

## TL;DR

This paper proposes StressTest, a benchmark for evaluating the ability of speech language models (SLMs) to understand the meaning conveyed by sentence stress. Evaluations reveal that existing models are nearly incapable of inferring speaker intent from stress patterns. A synthetic data pipeline, Stress-17k, is introduced, and the resulting fine-tuned model, StresSLM, substantially outperforms state-of-the-art models on both stress detection and stress reasoning tasks.

## Background & Motivation

**Background**: Speech language models (e.g., GPT-4o-audio, Gemini 2.5 Pro, Qwen2Audio) are now capable of directly processing audio for reasoning, bypassing traditional ASR cascade pipelines and leveraging paralinguistic information.

**Limitations of Prior Work**: Sentence stress is a critical prosodic element — the same sentence "I didn't say she stole the money" can convey entirely different meanings depending on which word is stressed — yet it has been almost entirely overlooked in the evaluation and development of SLMs. Existing benchmarks focus on speech recognition, emotion detection, and similar tasks, with no coverage of stress understanding.

**Key Challenge**: Understanding sentence stress requires a model not only to process *what* is said but also *how* it is said, demanding deep integration of prosodic cues (pitch, loudness, duration) with semantic reasoning — a capability absent in current SLMs.

**Goal**: To construct a benchmark for stress understanding, identify the capability gaps of state-of-the-art SLMs, and train a model with stress understanding ability using synthetic data.

**Key Insight**: A dual-task evaluation framework is designed — Sentence Stress Detection (SSD) and Sentence Stress Reasoning (SSR) — along with a complete pipeline encompassing synthetic data generation, validation, and multi-task training.

**Core Idea**: A pipeline combining LLM-generated stress-marked text, TTS-synthesized stressed speech, and automatic validation filtering is used to create training data, enabling fine-tuned SLMs to generalize to stress understanding in real recordings.

## Method

### Overall Architecture

The framework consists of two components: (1) the StressTest benchmark — sentences recorded by professional actors (each with at least two stress patterns and corresponding meanings), supplemented by StressPresso, a post-annotated subset derived from the Expresso dataset; and (2) the Stress-17k training pipeline — LLM-generated stress-marked text → TTS-synthesized stressed speech → WhiStress-based validation and filtering → four training task definitions → fine-tuned Qwen2Audio yielding StresSLM.

### Key Designs

1. **Dual-Task Benchmark Design (SSD + SSR)**:

    - Function: Comprehensively evaluate models' stress perception and reasoning capabilities.
    - Mechanism: SSD (Sentence Stress Detection) provides the model with audio and a transcript and requires it to identify which words are emphasized; SSR (Sentence Stress Reasoning) provides only audio and requires the model to select the correct meaning from two candidates. SSR is a novel task, while SSD aligns with prior work.
    - Design Motivation: Detecting stress is a prerequisite for understanding its meaning; the two tasks provide complementary evaluation.

2. **Synthetic Data Generation Pipeline (Stress-17k)**:

    - Function: Create sufficiently diverse and high-quality training data.
    - Mechanism: (a) *Text generation*: CrewAI + GPT-4o generate sentences whose meaning changes with stress, stratified by domain, topic, and sentence type; (b) *Speech synthesis*: OpenAI TTS synthesizes speech with asterisk-marked stressed words, producing one male and one female recording per stress pattern; (c) *Stress validation*: WhiStress automatically detects actual stress positions and filters erroneous samples; (d) *Four training tasks*: stress detection, end-to-end reasoning, detailed reasoning (with explanation), and cascaded reasoning (stress detection followed by reasoning).
    - Design Motivation: Not all sentences are suitable for stress-variant evaluation, necessitating targeted generation. TTS synthesis enables large-scale creation but introduces stress errors; the validation step ensures data quality.

3. **Two-Stage Training Strategy**:

    - Function: Balance stress-specific tasks with preservation of original capabilities.
    - Mechanism: Stage 1 fine-tunes on the full Stress-17k (including unvalidated data) for one epoch to establish basic competence; Stage 2 fine-tunes on the high-quality validated subset for one epoch to refine performance. ASR (LibriLight) and emotion recognition (MELD) samples are mixed in to prevent forgetting.
    - Design Motivation: The staged curriculum balances data quantity and quality; auxiliary tasks prevent catastrophic forgetting.

## Key Experimental Results

### Main Results (SSR Accuracy)

| Model | StressTest | StressPresso |
|-------|-----------|-------------|
| Human (majority vote) | 96.0 | 96.0 |
| StresSLM (ours) | **86.2** | **87.6** |
| Gemini 2.5 Pro | 77.5 | 72.7 |
| GPT-4o-audio | 68.8 | 64.8 |
| Qwen3-Omni-30B | 64.6 | 64.8 |
| Qwen2Audio-7B | 53.2 | 51.4 |
| SALMONN | 55.9 | 52.4 |
| Cascade (WhiStress→GPT-4o) | 83.4 | 79.7 |

### SSD Detection Performance (F1)

| Model | StressTest | StressPresso |
|-------|-----------|-------------|
| StresSLM | **86.9** | **80.6** |
| Gemini 2.5 Pro | 48.5 | 40.7 |
| GPT-4o-audio | 46.1 | 36.9 |
| WhiStress (specialized model) | 88.3 | 83.5 |

### Key Findings
- Existing SLMs perform near chance on stress reasoning (most scoring 50–55%); Gemini 2.5 Pro is the only model exceeding 70%.
- StresSLM (7B) surpasses all evaluated SLMs on SSR, including GPT-4o and Gemini 2.5 Pro, as well as the cascade baseline.
- Models trained on synthetic data generalize to real recordings (87.6% on StressPresso).
- End-to-end processing outperforms the cascade approach, as direct audio handling avoids the loss of stress information.
- StresSLM exhibits negligible degradation on original ASR and SER tasks.

## Highlights & Insights
- **Addressing an Important Gap**: Sentence stress is linguistically fundamental yet entirely overlooked in SLM evaluation; this work provides the first systematic assessment.
- **Elegant Synthetic Data Pipeline**: The fully automated pipeline of LLM generation + TTS synthesis + automatic validation is replicable for studying other prosodic features.
- **Strong Evidence for End-to-End over Cascade**: Direct audio processing demonstrates a clear advantage in stress understanding.
- **Small Model Outperforms Large Models**: StresSLM (7B) surpasses GPT-4o and Gemini 2.5 Pro, demonstrating the value of targeted training data.

## Limitations & Future Work
- **English-Only Evaluation**: Sentence stress functions differently across languages; cross-lingual extension is needed.
- **Synthetic Speech Training**: Although generalization to real recordings is strong, a gap between TTS and natural speech remains.
- **Narrow Prosodic Focus**: Other prosodic features (intonation, pauses, rhythm) are not addressed.
- Future directions: multilingual extension, natural speech training data, and more complex prosodic understanding tasks.

## Related Work & Insights
- **vs. WhiStress**: A specialized model for stress detection only; this work extends it with stress reasoning capability.
- **vs. VocalBench/URO-Bench**: These benchmarks evaluate expressive capabilities of SLMs but do not address stress understanding.
- **vs. Cascade Approach**: ASR + stress detection + LLM reasoning; this work demonstrates the superiority of the end-to-end approach.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose sentence stress reasoning as a task and benchmark; synthetic data pipeline is innovative and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8+ SLMs, multiple input configurations, human evaluation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; method description is complete.
- Value: ⭐⭐⭐⭐⭐ Opens a new research direction in stress understanding with substantial contributions to SLM evaluation and training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[NeurIPS 2025\] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](../../NeurIPS2025/audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](computational_narrative_understanding_for_expressive_text-to-speech.md)

</div>

<!-- RELATED:END -->
