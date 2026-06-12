---
title: >-
  [Paper Note] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models
description: >-
  [ICLR2026][Audio & Speech][Speech Language Model] This paper proposes EchoMind, the first multi-level interrelated benchmark for empathetic dialogue…
tags:
  - "ICLR2026"
  - "Audio & Speech"
  - "Speech Language Model"
  - "Empathetic Dialogue"
  - "benchmark"
  - "Vocal Cue"
  - "Evaluation"
date: 2026-05-08
content_hash: 0ace4b2877e121b4
---

# EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models

**Conference**: ICLR2026
**arXiv**: [2510.22758](https://arxiv.org/abs/2510.22758)  
**Code**: [Project Page](https://hlt-cuhksz.github.io/EchoMind/)  
**Area**: Audio & Speech
**Keywords**: Speech Language Model, Empathetic Dialogue, benchmark, Vocal Cue, Evaluation

## TL;DR

This paper proposes EchoMind, the first multi-level interrelated benchmark for empathetic dialogue, which systematically evaluates Speech Language Models' ability to perceive non-verbal acoustic cues and generate empathetic responses through a cognitive pipeline of Understanding → Reasoning → Conversation.

## Background & Motivation

Speech Language Models (SLMs) have achieved remarkable progress in spoken language understanding and are widely deployed in intelligent assistants and affective companionship systems. However, effective dialogue requires not only understanding *what is said*, but also perceiving *who is speaking*, *how they are speaking*, and *in what context*. Non-verbal acoustic cues—including prosody, emotion, physiological vocal signals, and environmental sounds—are critical for natural and emotionally resonant communication.

Existing benchmarks suffer from three major limitations: (1) they typically evaluate only a single capability (understanding, reasoning, or conversation) without joint cross-capability assessment; (2) tasks lack shared context, preventing the study of inter-level dependencies; and (3) empathy is rarely evaluated directly, which hinders the development of emotionally intelligent SLMs.

## Core Problem

Can current SLMs genuinely perceive non-lexical acoustic cues in speech (e.g., prosody, emotion, environmental sounds) and produce empathetic responses that are consistent with the emotional and contextual information conveyed?

## Method

### Empathy-Oriented Evaluation Framework

EchoMind structures acoustic cues into 3 coarse-grained dimensions and 12 fine-grained categories, covering 39 specific acoustic attributes:

- **Speaker Information**: gender (male/female), age (child/elderly)
- **Paralinguistic Information**: physiological state (hoarse/breathy/vocal fatigue/sobbing), emotion (6 categories), volume (shouting/whispering), speech rate (fast/slow), non-verbal expressions (coughing/sighing/laughter/yawning/moaning)
- **Environmental Information**: weather (wind/thunderstorm/rain), location (beach/basketball court/bus/subway), background voices, sudden events (alarm/ringing/horn), others (music/dog barking)

### Semantically Neutral Scripts + Controlled Acoustic Variants

A key design choice is the use of semantically neutral dialogue scripts that contain no explicit emotional or contextual cues. Each script is presented in three vocal style variants (target expression, alternative expression, neutral expression), allowing acoustic cues to be evaluated independently of lexical content. All tasks share the same scripts, enabling cross-level correlation analysis.

A total of 1,137 high-quality scripts are retained after GPT-4o generation followed by three rounds of human review. Audio synthesis adopts differentiated strategies by difficulty: speaker information uses Doubao TTS; paralinguistic cues employ a multi-method combination (Doubao conversational TTS, YouTube voice cloning, GPT-4o-mini-TTS); environmental sounds are mixed in from AudioCaps background audio.

### Three-Level Cognitive Task Pipeline

Modeling the cognitive process of human empathetic conversation, the benchmark defines a progressively structured task hierarchy:

**Level 1 — Understanding**:

- Content Understanding: ASR task requiring speech transcription under expressive and environmentally noisy conditions (3,356 instances)
- Sound Understanding: multiple-choice questions comprising 1 coarse-grained and 7 fine-grained sub-tasks (4,576 questions) for acoustic cue identification

**Level 2 — Reasoning**:

- Integrative Reasoning: 10 categories of multiple-choice questions (4,747 questions) requiring higher-order reasoning that integrates both linguistic content and acoustic features, including tasks such as personalized recommendation matching, antecedent event inference, and empathetic response selection

**Level 3 — Conversation**:

- Open-domain response generation (3,356 instances) evaluating models' ability to produce contextually coherent, socially appropriate, and empathetic replies

### Multi-Dimensional Evaluation System

- **Objective text-level evaluation**: BLEU, ROUGE-L, METEOR, BERTScore
- **Subjective text-level evaluation** (GPT-4o scoring, 5-point scale): Contextual Fit (CCtxFit), Response Naturalness (CRespNat), Colloquial Degree (CColloqDeg), Speech Information Relevance (CSpeechRel)
- **Audio-level evaluation**: NISQA/UTMOS for audio quality, EmoAlign for emotional alignment, Vocal Empathy Score (VES) assessed by Gemini-2.5-Pro for vocal empathy in generated responses
- An EchoMind-Human variant (491 scripts, 1,453 human recordings) is also provided to compare the effects of real versus synthetic speech

## Key Experimental Results

Twelve state-of-the-art SLMs are evaluated (1 closed-source GPT-4o-Audio + 11 open-source models):

| Key Finding | Data |
|---|---|
| Open-source models with Sound Understanding accuracy >60% | Only 3, including Audio-Flamingo3 and Qwen2.5-Omni-7B |
| Open-source models with Reasoning accuracy >60% | Only 1 (DeSTA2.5-Audio) |
| Highest CSpeechRel (speech cue utilization) | GPT-4o-Audio at 3.42; no model exceeds 4.0 |
| Highest VES (vocal empathy) | GPT-4o-Audio at 3.34 |
| CSpeechRel gain in upper-bound experiment | Step-Audio +1.10, GPT-4o-Audio +1.03 |
| Arena win rate | GPT-4o-Audio 42% > Step-Audio 34% > Qwen2.5-Omni-7B 28% |
| Human recordings vs. TTS | Human speech is more challenging across all levels, with the largest gap at the conversation level |

Three in-depth research questions (RQs) are investigated:

1. **Prompt Sensitivity**: 7 out of 12 models achieve higher CSpeechRel with enhanced prompts, but some models perform better without prompts, exposing deficiencies in instruction-following capability.
2. **Speech Source Effect**: Human recordings are more challenging to process than TTS, as real acoustic variability and subtle prosodic nuances pose greater difficulty.
3. **Upper Bound of Empathetic Response**: All models improve when provided with ideal acoustic cue information, yet a significant performance gap remains.

## Highlights & Insights

- **First multi-level interrelated evaluation**: The hierarchical Understanding → Reasoning → Conversation design, with all tasks sharing the same scripts, enables cross-level correlation analysis—a feature unique among comparable benchmarks.
- **Semantically neutral script design**: Scripts contain no emotion-laden words, strictly isolating the contribution of acoustic cues and genuinely testing models' perception of *how* something is said.
- **Comprehensive coverage of 39 acoustic attributes**: Spanning speaker, paralinguistic, and environmental dimensions—far exceeding the scope of existing benchmarks.
- **Dual-layer text + audio evaluation**: Both content empathy and vocal empathy are assessed, combining objective metrics with subjective scoring (model-as-judge and human evaluation).
- **Reveals a core bottleneck**: No model exceeds a CSpeechRel score of 4.0, demonstrating a systematic deficiency in acoustic cue utilization among current SLMs.

## Limitations & Future Work

- Dialogue scripts are LLM-generated and, despite human review, may still carry inherent biases; future work could incorporate authentic human-to-human conversations.
- The majority of audio is TTS-synthesized; although a human-recorded variant exists, its scale is limited (491 scripts), providing insufficient coverage of real-world scenarios.
- Only single-turn dialogues are evaluated; multi-turn empathy maintenance and evolution are not considered.
- Evaluation relies heavily on model-as-judge scoring (e.g., GPT-4o), with relatively low inter-rater agreement on fine-grained dimensions such as CSpeechRel (Spearman 0.64).
- Only English is covered; empathetic dialogue evaluation in other languages is not addressed.

## Related Work & Insights

Key distinctions from existing benchmarks (see Table 1 in the paper):

- **SD-Eval / VoxDialog / EChat-eval**: Evaluate only conversation; do not cover understanding or reasoning, and tasks are not interrelated.
- **AIR-Bench / SAKURA / MMAU**: Evaluate only understanding or reasoning; do not include conversation.
- **URO-Bench**: Covers understanding + reasoning + conversation, but tasks are not correlated (Corr. = ✗) and only a single expression style is supported.
- **EchoMind**: The only benchmark that simultaneously satisfies multi-level tasks (understanding + reasoning + conversation), inter-task correlation (Corr. = ✓), multiple expression style variants, and environmental sound support.

Additional insights:

- Current SLMs approach human-level performance at understanding *what is said*, yet exhibit a substantial gap in understanding *emotion, tone, and environment*—pointing to the next critical improvement frontier for SLMs.
- Upper-bound experiments demonstrate that empathy quality improves substantially when acoustic cues are perfectly recognized, indicating that the bottleneck lies in perception rather than generation.
- The hierarchical evaluation framework (perception → reasoning → generation) is generalizable to other multimodal evaluation scenarios, such as video dialogue and multimodal sentiment analysis.
- Prompt sensitivity analysis suggests that carefully designed system prompts can substantially improve empathetic response quality in deployment settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — First multi-level interrelated empathy evaluation benchmark; the semantically neutral script design combined with controlled acoustic variants is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 12 models, multi-dimensional evaluation, human evaluation validation, and three in-depth research questions; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables, though some notation definitions are scattered.
- Value: ⭐⭐⭐⭐ — Reveals systematic bottlenecks in SLM acoustic cue utilization, providing important guidance for the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](../../ICML2026/audio_speech/multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ACL 2026\] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models](../../ACL2026/audio_speech/s2s-arena_evaluating_paralinguistic_instruction_following_in_speech-to-speech_mo.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](../../ACL2026/audio_speech/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)

</div>

<!-- RELATED:END -->
