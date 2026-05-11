---
title: >-
  [Paper Note] MSU-Bench: Musical Score Understanding Benchmark
description: >-
  [ACL 2026][Audio & Speech][musical score understanding] MSU-Bench is the first human-annotated benchmark for complete musical score understanding, comprising 1…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "musical score understanding"
  - "music information retrieval"
  - "ABC notation"
  - "multimodal benchmark"
  - "hallucination"
date: 2026-05-08
content_hash: bd8bd466a14b8163
---

# MSU-Bench: Musical Score Understanding Benchmark

**Conference**: ACL 2026
**arXiv**: [2511.20697](https://arxiv.org/abs/2511.20697)
**Code**: [https://github.com/Congren-Dai/MSU-Bench](https://github.com/Congren-Dai/MSU-Bench)
**Area**: Multimodal / Music Understanding
**Keywords**: musical score understanding, music information retrieval, ABC notation, multimodal benchmark, hallucination

## TL;DR

MSU-Bench is the first human-annotated benchmark for complete musical score understanding, comprising 1,800 generative QA pairs from 150 pieces across four difficulty levels. Evaluation reveals severe deficiencies in LLM/VLM localization and hallucination, while text-based ABC notation input substantially mitigates these issues.

## Background & Motivation

**Background**: LLMs and VLMs have demonstrated strong capabilities in natural language processing, yet their ability to reason over complete musical scores remains largely unexplored. Existing music understanding benchmarks are typically limited to fragments, short excerpts, or multiple-choice formats, and mostly focus on monophonic music.

**Limitations of Prior Work**: VLMs face two persistent challenges when processing complete scores — (1) localization failure: models frequently fail to correctly identify measure positions, which is a prerequisite for answering higher-level questions about harmony and texture; (2) hallucination: models generate content not grounded in the score, a problem exacerbated by localization errors.

**Key Challenge**: Complete musical score understanding requires integrated reasoning over pitch, rhythm, harmony, and large-scale structure, yet existing benchmarks do not systematically evaluate this composite capability.

**Goal**: (1) Construct a complete musical score understanding benchmark covering four levels of difficulty; (2) support dual-modality evaluation via text (ABC notation) and vision (PDF); (3) systematically assess the capabilities of mainstream LLMs/VLMs.

**Key Insight**: ABC notation, as a structured text format, explicitly encodes measure structure, pitch, rhythm, and other information, providing an LLM-friendly score representation that can substantially alleviate localization and hallucination problems.

**Core Idea**: Systematically evaluate score understanding through a four-level hierarchy (header information → notation and notes → chords and harmony → texture and form), with ABC notation serving as the upper-bound modality for symbol-to-theory reasoning.

## Method

### Overall Architecture

MSU-Bench contains 150 complete musical scores (Bach, Beethoven, Chopin, Debussy, etc.) and 1,800 human-annotated QA pairs. It supports two evaluation modalities: ABC notation (text → LLM) and PDF scores (image → VLM). Four difficulty levels progress from basic recognition to advanced analysis.

### Key Designs

1. **Four-Level Hierarchical Evaluation Framework**:

    - Function: Systematically evaluate score understanding capabilities from basic to advanced levels.
    - Mechanism: Level 1 — header information (key signature, time signature, tempo) → Level 2 — notation and notes (notes and articulations in specific measures) → Level 3 — chords and harmony (chord identification, tonal analysis) → Level 4 — texture and form (thematic motives, formal structure). Each piece has 3 questions per level.
    - Design Motivation: Reflects the pedagogical hierarchy of undergraduate music theory curricula; a model capable of answering these questions could serve as a teaching assistant.

2. **Dual-Modality Evaluation**:

    - Function: Compare differences between text and visual modalities in score understanding.
    - Mechanism: ABC notation is provided as text input to LLMs; PDF format is provided as image input to VLMs. ABC notation explicitly encodes measure structure, eliminating VLM localization issues. Evaluation employs LLM-as-judge (GPT-4o) for automatic scoring.
    - Design Motivation: Quantify the "modality gap" and understand the impact of visual OCR difficulty on score comprehension.

3. **Joint vs. Sequential Questioning**:

    - Function: Investigate the effect of question presentation strategy on performance.
    - Mechanism: Two strategies are compared — joint questioning (all four levels presented at once) vs. sequential questioning (level-by-level). Joint questioning yields better performance, suggesting models can leverage inter-level reasoning relationships.
    - Design Motivation: Provide guidance on optimal prompting strategies for practical use.

### Loss & Training

The benchmark itself does not involve training. Fine-tuning experiments employ standard supervised fine-tuning (SFT) on the MSU-Bench training split.

## Key Experimental Results

### Main Results

**Zero-Shot Evaluation (Selected)**

| Model | Modality | L1 | L2 | L3 | L4 | Avg |
|-------|----------|----|----|----|----|-----|
| GPT-4o | ABC | High | Medium | Low | Low | — |
| GPT-4o | PDF | Medium | Low | Low | Low | — |
| Qwen3-72B | ABC | Relatively High | — | — | — | — |
| Gemma-3 | PDF | Low | — | — | — | — |

### Key Findings

- **Significant modality gap**: ABC text input consistently and substantially outperforms PDF visual input, confirming that localization difficulty is the core bottleneck for VLMs.
- Fine-tuning substantially improves performance on both modalities without degrading general knowledge.
- Joint questioning outperforms sequential questioning, indicating that models can exploit hierarchical reasoning.
- Level 1 (basic information) is the easiest; Levels 3–4 (harmony, form) are the most difficult.
- Even the strongest models perform poorly on Level 4, demonstrating that advanced musical analysis remains a major challenge.

## Highlights & Insights

- The design philosophy of structuring music understanding evaluation according to pedagogical levels is both clear and practically motivated.
- The finding that ABC notation serves as a "silver bullet" for localization problems has direct application value — suggesting that future music AI systems should prioritize OCR and measure localization.
- This is the first understanding benchmark covering complete musical scores (including polyphonic works), filling an important gap in the field.

## Limitations & Future Work

- Coverage is limited to Western classical music; genres such as jazz and popular music are not included.
- QA pairs are human-annotated, and annotation costs constrain the scale of the benchmark.
- ABC notation itself has limitations and is less expressive than MusicXML.
- Evaluation relies on an LLM judge, and the accuracy of judgments on specialized music terminology requires further validation.

## Related Work & Insights

- **vs. existing music NLP benchmarks**: Prior benchmarks focus on fragments or multiple-choice questions; MSU-Bench is the first to evaluate generative understanding of complete musical scores.
- **vs. OMR systems**: OMR focuses on recognition, whereas MSU-Bench focuses on understanding and reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark for complete musical score understanding; interdisciplinary (musicology + NLP).
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation covers 15+ models, dual modalities, and fine-tuning experiments, though human evaluation comparisons are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated contributions.
- Value: ⭐⭐⭐⭐ Provides much-needed evaluation infrastructure for music AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)

</div>

<!-- RELATED:END -->
