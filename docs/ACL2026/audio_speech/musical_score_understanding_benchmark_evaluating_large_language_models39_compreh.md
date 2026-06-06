---
title: >-
  [Paper Note] MSU-Bench: Musical Score Understanding Benchmark
description: >-
  [ACL 2026][Audio & Speech][Musical score understanding] MSU-Bench is the first human-annotated benchmark for full musical score understanding, comprising 1…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Musical score understanding"
  - "music information retrieval"
  - "ABC notation"
  - "multimodal benchmark"
  - "hallucination"
date: 2026-05-08
content_hash: 66152d493bc5c2e1
---

# MSU-Bench: Musical Score Understanding Benchmark

**Conference**: ACL 2026  
**arXiv**: [2511.20697](https://arxiv.org/abs/2511.20697)  
**Code**: [https://github.com/Congren-Dai/MSU-Bench](https://github.com/Congren-Dai/MSU-Bench)  
**Area**: Multimodal / Music Understanding  
**Keywords**: Musical score understanding, music information retrieval, ABC notation, multimodal benchmark, hallucination

## TL;DR

MSU-Bench is the first human-annotated benchmark for full musical score understanding, comprising 1,800 generative QA pairs across 150 works with four difficulty levels. Evaluations reveal severe deficiencies in LLMs/VLMs regarding score localization and hallucination, while text input via ABC notation significantly mitigates these issues.

## Background & Motivation

**Background**: LLMs and VLMs have demonstrated powerful capabilities in natural language processing, yet their reasoning abilities regarding full musical scores remain under-explored. Existing music understanding benchmarks are often limited to fragments, short excerpts, or multiple-choice formats, and predominantly focus on monophonic music.

**Limitations of Prior Work**: VLMs face two persistent challenges when processing full scores: (1) Localization failure: models often fail to correctly identify measure positions, which is a prerequisite for answering high-level questions on harmony, texture, etc.; (2) Hallucination: models generate content not grounded in the score, and localization errors exacerbate these hallucinations.

**Key Challenge**: Full musical score understanding requires integrated reasoning across pitch, rhythm, harmony, and large-scale structures, but existing benchmarks fail to systematically evaluate this comprehensive ability.

**Goal**: (1) Construct a full musical score understanding benchmark covering four difficulty levels; (2) Support dual-modality evaluation for text (ABC notation) and vision (PDF); (3) Systematically evaluate the capabilities of mainstream LLMs/VLMs.

**Key Insight**: ABC notation, as a structured text format, explicitly encodes measure structures, pitch, and rhythm, providing an LLM-friendly score representation that can significantly alleviate localization and hallucination issues.

**Core Idea**: Systematically evaluate score understanding using a four-level hierarchy (starting information → notation and notes → chords and harmony → texture and form), utilizing ABC as an upper-bound modality for symbol-to-theory reasoning.

## Method

### Overall Architecture

MSU-Bench includes 150 full scores (Bach, Beethoven, Chopin, Debussy, etc.) and 1,800 human-annotated QA pairs. It supports two evaluation modalities: ABC notation (Text → LLM) and PDF scores (Image → VLM). The four difficulty levels progress from basic recognition to advanced analysis.

### Key Designs

1.  **Four-level Hierarchical Evaluation Framework**:
    - **Function**: Systematically evaluate musical score understanding from basic to advanced levels.
    - **Mechanism**: Level 1 Starting Information (key signature, time signature, tempo) → Level 2 Notation and Notes (notes in specific measures, articulation) → Level 3 Chords and Harmony (chord recognition, tonal analysis) → Level 4 Texture and Form (thematic motives, structural forms). Each work contains 3 questions per level.
    - **Design Motivation**: Reflect the pedagogical levels of undergraduate musicology curricula; models capable of answering these questions can serve as teaching assistants.

2.  **Dual-Modality Evaluation**:
    - **Function**: Compare differences between text and vision modalities in score understanding.
    - **Mechanism**: ABC notation is provided to LLMs as text input, while PDF format is provided to VLMs as image input. ABC explicitly encodes measure structures, eliminating VLM localization issues. Evaluation utilizes LLM-as-a-judge (GPT-4o) for automated scoring.
    - **Design Motivation**: Quantify the "modality gap" and understand the impact of visual OCR difficulties on score understanding.

3.  **Joint vs. Sequential Questioning**:
    - **Function**: Explore the impact of question presentation on performance.
    - **Mechanism**: Comparison of two strategies—Joint Questioning (all four levels of questions provided simultaneously) vs. Sequential Questioning (level-by-level). Joint questioning was found to yield better performance, suggesting models can utilize reasoning relationships between levels.
    - **Design Motivation**: Provide guidance on optimal prompting strategies for practical applications.

### Loss & Training

The benchmark itself does not involve training. Fine-tuning experiments use standard Supervised Fine-Tuning (SFT) on the MSU-Bench training set.

## Key Experimental Results

### Main Results

**Zero-shot Evaluation (Partial)**

| Model | Modality | L1 | L2 | L3 | L4 | Avg |
|-------|----------|----|----|----|----|-----|
| GPT-4o | ABC | High | Med | Low | Low | — |
| GPT-4o | PDF | Med | Low | Low | Low | — |
| Qwen3-72B | ABC | Higher | — | — | — | — |
| Gemma-3 | PDF | Low | — | — | — | — |

### Key Findings

- **Significant Modality Gap**: ABC text input consistently and significantly outperforms PDF visual input, confirming that localization difficulty is a core bottleneck for VLMs.
- Fine-tuning significantly improves performance in both modalities without compromising general knowledge.
- Joint questioning outperforms sequential questioning, indicating models can leverage hierarchical reasoning.
- Level 1 (basic information) is the easiest, while Level 3-4 (harmony, form) are the most difficult.
- Even the strongest models perform poorly at Level 4, indicating that advanced music analysis remains a major challenge.

## Highlights & Insights

- The design concept of structured evaluation based on pedagogical hierarchies in music understanding is clear and practical.
- The discovery of ABC notation as a "silver bullet for localization issues" has direct application value—suggesting that future music AI should prioritize solving OCR and measure localization.
- This is the first understanding benchmark to cover full scores (including polyphonic music), filling a significant gap in the field.

## Limitations & Future Work

- Only covers Western classical music, excluding styles such as jazz and pop.
- QA pairs are human-annotated, which limits the scale due to annotation costs.
- ABC notation itself has limitations and is less expressive than MusicXML.
- Evaluation relies on an LLM judge; the accuracy of assessing professional musical terminology remains to be verified.

## Related Work & Insights

- **vs. Existing Music NLP Benchmarks**: Existing benchmarks focus on fragments or multiple-choice questions; MSU-Bench is the first to evaluate generative understanding of full scores.
- **vs. OMR Systems**: OMR focuses on recognition, while MSU-Bench focuses on understanding and reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First full score understanding benchmark, interdisciplinary (Musicology + NLP).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 15+ model evaluations, dual-modality, and fine-tuning experiments, though lacking human evaluation comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-motivated.
- **Value**: ⭐⭐⭐⭐ Provides much-needed evaluation infrastructure for music AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ICML 2026\] PhaLar: Phasors for Learned Musical Audio Representations](../../ICML2026/audio_speech/phalar_phasors_for_learned_musical_audio_representations.md)

</div>

<!-- RELATED:END -->
