---
title: >-
  [Paper Note] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction
description: >-
  [ICLR 2026][Social Computing][Turing Test] This work conducts the first speech-based Turing test on 9 state-of-the-art speech-to-speech (S2S) dialogue systems, collecting 2,968 human judgments. Results show that all systems fail the test (pass rates of 7%–31%). The primary bottlenecks lie not in semantic understanding but in paralinguistic features, emotional expression, and conversational persona. The study also introduces an 18-dimensional fine-grained evaluation framework and an interpretable AI judge model.
tags:
  - ICLR 2026
  - Social Computing
  - Turing Test
  - spoken dialogue
  - human-likeness
  - S2S systems
  - fine-grained evaluation
date: 2026-05-08
content_hash: 7dbfa1770a9de5c9
---

# Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction

**Conference**: ICLR 2026
**arXiv**: [2602.24080](https://arxiv.org/abs/2602.24080)
**Code**: [GitHub](https://github.com/Carbohydrate1001/Turing-Test)
**Area**: Social Computing
**Keywords**: Turing Test, spoken dialogue, human-likeness, S2S systems, fine-grained evaluation

## TL;DR
This work conducts the first speech-based Turing test on 9 state-of-the-art speech-to-speech (S2S) dialogue systems, collecting 2,968 human judgments. Results show that all systems fail the test (pass rates of 7%–31%). The primary bottlenecks lie not in semantic understanding but in paralinguistic features, emotional expression, and conversational persona. The study also introduces an 18-dimensional fine-grained evaluation framework and an interpretable AI judge model.

## Background & Motivation

**Background**: S2S systems (e.g., GPT-4o, Gemini-2.5-Pro) are rapidly advancing, enabling direct voice-based interaction. Existing evaluations focus primarily on speech comprehension and reasoning tasks, while the question of whether these systems converse in a human-like manner remains largely unaddressed.

**Limitations of Prior Work**: (1) Text-based Turing tests are ill-suited for speech, as they do not account for acoustic naturalness and emotional expression; (2) existing speech benchmarks evaluate task-oriented capabilities (e.g., ASR, emotion recognition) rather than human-likeness; (3) no standardized methodology exists for assessing human-likeness in S2S systems.

**Key Challenge**: High scores on task-oriented benchmarks do not imply human-like speech—models may approach human-level understanding while exhibiting clearly machine-like expressive styles.

**Key Insight**: Directly applying the Turing test—having human judges determine whether they are speaking with a human or a machine—and using an 18-dimensional diagnostic framework to identify the reasons for failure.

## Method

### Overall Architecture
Dialogue recordings of 28 volunteers interacting with 9 S2S systems, captured in a professional studio → human judgments collected via a gamified online platform → 17-dimensional evaluation framework applied to diagnose failure causes → interpretable AI judge model trained.

### Key Designs

1. **Dialogue Data Construction (3 Categories)**:

    - Human–Machine (H-M): 28 participants × 9 systems × 10 topics; 3 interaction strategies designed to minimize identity leakage
    - Human–Human (H-H): sourced from public datasets and volunteer recordings, matched to the topic distribution
    - Pseudo-Human (PH): TTS-synthesized dialogues (to increase test difficulty)

2. **18-Dimensional Human-Likeness Taxonomy**:

    - Semantic-pragmatic: memory consistency, logical coherence, pragmatic appropriateness
    - Non-physiological paralinguistics: rhythm, intonation, stress, disfluency (hesitations, fillers)
    - Physiological paralinguistics: breath sounds, pronunciation accuracy
    - Mechanical persona: excessive affirmation, apology tendency, formal written-style language
    - Emotional expression: textual emotion, acoustic emotion

3. **Interpretable AI Judge**:

    - Function: human-annotated 18-dimensional scores → input to a regularized linear classifier → human/machine classification
    - Design Motivation: 9 off-the-shelf AI models used as judges performed poorly (42–63% accuracy), necessitating a purpose-trained model
    - Novelty: linear model weights directly reflect the discriminative contribution of each dimension

## Key Experimental Results

### Turing Test Results

| System | Pass Rate (Judged as Human) | Notes |
|---|---|---|
| Human–Human dialogue | 70–87% | Upper-bound reference |
| GPT-4o | ~20% | Well below 50% |
| Gemini-2.5-Pro | ~25% | Well below 50% |
| Best-performing S2S | 31% | Still well below 50% |
| Pseudo-Human (TTS) | 40–60% | Outperforms S2S systems |
| **Pass threshold (50%)** | **No system passed** | — |

### 18-Dimensional Diagnosis

| Dimension Category | Human | S2S | Gap |
|---|---|---|---|
| Memory consistency | High | **Near human** | Small |
| Logical coherence | High | **Near human** | Small |
| Pronunciation accuracy | High | **High** | Small |
| Rhythm / Intonation | Natural | **Mechanical** | Large |
| Emotional expression | Rich | **Monotonic** | Large |
| Conversational persona | Natural | **Excessively affirmative / apologetic** | Large |

### Key Findings
- Semantic understanding has approached human-level performance—logical coherence and memory consistency are no longer bottlenecks.
- The primary bottleneck lies in paralinguistics: overly regular rhythm, absence of hesitation and breath sounds, and unnatural stress patterns.
- Acoustic emotion scores are markedly lower than textual emotion scores, indicating that even when text conveys emotion, TTS fails to render it acoustically.
- S2S systems underperform TTS-based pseudo-human dialogues, suggesting the problem extends beyond speech synthesis to conversational strategy (e.g., excessive affirmation).
- Judges with greater AI experience achieve higher accuracy (78.8% vs. 64.2%).

## Highlights & Insights
- **A rigorous return to the Turing Test**: Rather than a toy experiment, this study employs 2,968 large-scale judgments with methodological rigor (professional recording, interaction strategy control, three-way dialogue comparison).
- **"Semantics is solved; expression is the bottleneck"**: This finding is highly instructive—future S2S improvements should target paralinguistics and emotional expression rather than stronger NLU.
- **The problem of excessive affirmation and apology**: The "sycophantic persona" of current models renders them immediately identifiable as machines—a side effect of over-alignment during fine-tuning.

## Limitations & Future Work
- Only 10 topics were tested; more diverse scenarios may yield different conclusions.
- Dialogue duration of 20–60 seconds is relatively short; issues may be more pronounced in longer conversations.
- Pseudo-human TTS dialogues used scripted content rather than genuine S2S interaction.
- The human judge sample may be skewed toward younger and more technically oriented populations.

## Related Work & Insights
- **vs. Text-based Turing Test (Jones et al.)**: This work presents the first speech-to-speech Turing test, involving considerably more complex dimensions.
- **vs. VoiceBench**: VoiceBench evaluates task-oriented capabilities, whereas this work assesses human-likeness—the two perspectives are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First S2S Turing test + 18-dimensional diagnostic framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 systems, 28 participants, 2,968 judgments, both human and AI judges
- Writing Quality: ⭐⭐⭐⭐ Rigorous study design with compelling conclusions
- Value: ⭐⭐⭐⭐⭐ Establishes a standard for human-likeness evaluation of S2S systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](../../ACL2026/social_computing/explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/social_computing/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](../../ACL2026/social_computing/persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)

</div>

<!-- RELATED:END -->
