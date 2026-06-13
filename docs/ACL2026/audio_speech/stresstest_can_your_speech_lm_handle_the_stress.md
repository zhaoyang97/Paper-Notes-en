---
title: >-
  [Paper Note] StressTest: Can YOUR Speech LM Handle the Stress?
description: >-
  [ACL 2026][Audio & Speech][Sentence stress] This paper proposes the StressTest benchmark to evaluate the ability of Speech Language Models (SLMs) to understand the meaning of sentence stress. The study reveals that exist…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Sentence stress"
  - "Speech language models"
  - "Prosody understanding"
  - "Benchmark"
  - "Synthetic data"
date: 2026-05-08
content_hash: 7449f398c8bfa68b
---

# StressTest: Can YOUR Speech LM Handle the Stress?

**Conference**: ACL 2026  
**arXiv**: [2505.22765](https://arxiv.org/abs/2505.22765)  
**Code**: [Project Page](https://pages.cs.huji.ac.il/adiyoss-lab/stresstest)  
**Area**: Speech Understanding  
**Keywords**: Sentence stress, Speech language models, Prosody understanding, Benchmark, Synthetic data

## TL;DR

This paper proposes the StressTest benchmark to evaluate the ability of Speech Language Models (SLMs) to understand the meaning of sentence stress. The study reveals that existing models are nearly unable to reason speaker intent based on stress patterns. StresSLM, trained through the Stress-17k synthetic data pipeline, significantly outperforms state-of-the-art models in stress detection and reasoning tasks.

## Background & Motivation

**Background**: Speech Language Models (such as GPT-4o-audio, Gemini 2.5 Pro, Qwen2Audio, etc.) can directly process audio for reasoning, bypassing traditional ASR-cascaded pipelines to leverage paralinguistic information.

**Limitations of Prior Work**: Sentence stress is a critical element of prosody—the same sentence "I didn't say she stole the money" can express completely different meanings depending on the stress position. However, it is almost entirely ignored in the evaluation and development of SLMs. Existing benchmarks focus on speech recognition and emotion detection, lacking evaluations for stress understanding.

**Key Challenge**: Understanding sentence stress requires the model to not only identify "what was said" but also "how it was said." This necessitates a deep integration of prosodic cues (pitch, loudness, duration) and semantic reasoning, a capability currently lacking in existing SLMs.

**Goal**: To build a stress understanding benchmark, evaluate the capability gaps of cutting-edge SLMs, and train a model possessing stress understanding capabilities through synthetic data.

**Key Insight**: Designing a dual-task evaluation (SSD + SSR) and constructing a full pipeline encompassing synthetic data generation, validation, and multi-task training.

**Core Idea**: Creating training data via a pipeline involving LLM-generated stress text, TTS-synthesized stress speech, and automated verification filtering. This enables fine-tuned SLMs to generalize to stress understanding in real recordings.

## Method

### Overall Architecture

The framework consists of two parts: (1) The StressTest benchmark—comprising sentences recorded by professional actors (each sentence has at least two stress patterns and corresponding meanings), and the StressPresso supplementary set post-annotated from the Expresso dataset; (2) The Stress-17k training pipeline—generating stressed text via LLM $\rightarrow$ synthesizing stressed speech via TTS $\rightarrow$ filtering via WhiStress validation $\rightarrow$ defining four training tasks, ultimately fine-tuning Qwen2Audio to obtain StresSLM.

### Key Designs

1.  **Dual-Task Benchmark Design (SSD + SSR)**:
    *   **Function**: To comprehensively evaluate the model's stress perception and reasoning capabilities.
    *   **Mechanism**: SSD (Sentence Stress Detection) requires the model to identify which words are emphasized given the audio and transcript. SSR (Sentence Stress Reasoning) requires the model to select the correct meaning from two possible options given only the audio. SSR is a novel task, while SSD aligns with prior research.
    *   **Design Motivation**: Detecting stress is a prerequisite for understanding its meaning; the two tasks provide complementary evaluations.

2.  **Synthetic Data Generation Pipeline (Stress-17k)**:
    *   **Function**: To create sufficiently diverse and high-quality training data.
    *   **Mechanism**: (a) Text generation: Using CrewAI and GPT-4o to generate sentences whose meanings change with stress across different domains and themes; (b) Speech synthesis: OpenAI TTS synthesizes speech with stressed words marked by asterisks, generating male and female versions for each stress pattern; (c) Stress validation: Using WhiStress to automatically detect actual stress positions and filter out incorrect samples; (d) Four training tasks: Stress detection, end-to-end reasoning, detailed reasoning (with explanations), and cascaded reasoning (detection followed by reasoning).
    *   **Design Motivation**: Not all sentences are suitable for stress variant evaluation, necessitating specialized generation. TTS synthesis allows for large-scale creation, but verification steps are required to ensure data quality.

3.  **Phased Training Strategy**:
    *   **Function**: To balance stress tasks with original model capabilities.
    *   **Mechanism**: The first phase involves fine-tuning for one epoch on the full Stress-17k (including unvalidated data) to establish base capabilities. The second phase involves fine-tuning for another epoch on the high-quality validated subset for refinement. ASR (LibriLight) and sentiment recognition (MELD) samples are mixed in to prevent catastrophic forgetting.
    *   **Design Motivation**: Phased curriculum training accounts for both data volume and quality, while auxiliary tasks maintain overall model stability.

## Key Experimental Results

### Main Results (SSR Accuracy)

| Model | StressTest | StressPresso |
| :--- | :--- | :--- |
| Human (Majority Vote) | 96.0 | 96.0 |
| **Ours (StresSLM)** | **86.2** | **87.6** |
| Gemini 2.5 Pro | 77.5 | 72.7 |
| GPT-4o-audio | 68.8 | 64.8 |
| Qwen3-Omni-30B | 64.6 | 64.8 |
| Qwen2Audio-7B | 53.2 | 51.4 |
| SALMONN | 55.9 | 52.4 |
| Cascade (WhiStress $\rightarrow$ GPT-4o) | 83.4 | 79.7 |

### SSD Detection Performance (F1)

| Model | StressTest | StressPresso |
| :--- | :--- | :--- |
| **Ours (StresSLM)** | **86.9** | **80.6** |
| Gemini 2.5 Pro | 48.5 | 40.7 |
| GPT-4o-audio | 46.1 | 36.9 |
| WhiStress (Specialized Model) | 88.3 | 83.5 |

### Key Findings
*   Existing SLMs perform near random chance on stress reasoning (mostly between 50-55%), with Gemini 2.5 Pro being the only model to exceed 70%.
*   StresSLM (7B) outperforms all SLMs including GPT-4o and Gemini 2.5 Pro on SSR, and also exceeds the cascaded approach.
*   Models trained on synthetic data generalize well to real recordings (87.6% on StressPresso).
*   The end-to-end approach outperforms the cascaded approach—direct audio processing avoids the loss of prosodic information.
*   StresSLM shows almost no degradation in original ASR and SER tasks.

## Highlights & Insights
*   **Filling a Critical Gap**: Sentence stress is vital in linguistics but has been ignored in SLM evaluation; this paper provides the first systematic assessment.
*   **Clever Synthetic Pipeline**: The fully automated pipeline (LLM generation + TTS synthesis + automated validation) is reproducible for researching other prosodic features.
*   **Evidence for E2E superiority**: Demonstrates the advantages of direct audio processing for stress understanding over cascaded methods.
*   **Small Models Outperform Large Models**: The 7B StresSLM surpassing GPT-4o and Gemini 2.5 Pro highlights the value of specialized training data.

## Limitations & Future Work
*   **Limited to English**: Stress functions differently in other languages, requiring cross-lingual expansion.
*   **Synthetic Training**: While generalization to real speech is good, there remains a gap between TTS and natural speech.
*   **Focus on Sentence Stress**: The study does not cover other prosodic features such as intonation, pauses, or rhythm.
*   **Future Directions**: Expansion to multilingual datasets, natural speech training data, and more complex prosodic understanding tasks.

## Related Work & Insights
*   **vs WhiStress**: WhiStress is a specialized model for detection only; this work adds reasoning capabilities on top of it.
*   **vs VocalBench/URO-Bench**: These evaluate SLM expressive capabilities but do not touch upon stress understanding.
*   **vs Cascaded Approaches**: Compared to ASR + stress detection + LLM reasoning, this paper proves that end-to-end models are superior.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Proposes the first sentence stress reasoning task and benchmark; the synthetic pipeline is innovative and practical.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8+ SLMs, various input settings, and includes human evaluation and ablation studies.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation and complete methodological description.
*   Value: ⭐⭐⭐⭐⭐ Establishes a research direction for stress understanding, providing a substantial push for SLM evaluation and training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/audio_speech/diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[NeurIPS 2025\] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](../../NeurIPS2025/audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ICCV 2025\] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation](../../ICCV2025/audio_speech/align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)

</div>

<!-- RELATED:END -->
