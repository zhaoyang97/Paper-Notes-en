---
title: >-
  [Paper Note] MSU-Bench: Musical Score Understanding Benchmark
description: >-
  [ACL 2026][Audio & Speech][Score understanding] MSU-Bench is the first human-annotated benchmark for full musical score understanding, comprising 1,800 generative QA pairs from 150 works across four difficulty levels. Evaluations reveal significant deficiencies in LLMs/VLMs regarding score localization and hallucinations, while text input via ABC notation significantly mitigates these issues.
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Score understanding"
  - "Music Information Retrieval"
  - "ABC notation"
  - "Multimodal benchmark"
  - "Hallucination"
date: 2026-05-08
content_hash: 05d2feadf59c7638
---

# MSU-Bench: Musical Score Understanding Benchmark

**Conference**: ACL 2026  
**arXiv**: [2511.20697](https://arxiv.org/abs/2511.20697)  
**Code**: [https://github.com/Congren-Dai/MSU-Bench](https://github.com/Congren-Dai/MSU-Bench)  
**Area**: Multimodal / Music Understanding  
**Keywords**: Score understanding, Music Information Retrieval, ABC notation, Multimodal benchmark, Hallucination

## TL;DR

MSU-Bench is the first human-annotated benchmark for full musical score understanding, comprising 1,800 generative QA pairs from 150 works across four difficulty levels. Evaluations reveal significant deficiencies in LLMs/VLMs regarding score localization and hallucinations, while text input via ABC notation significantly mitigates these issues.

## Background & Motivation

**Background**: LLMs and VLMs đã have demonstrated powerful capabilities in natural language processing, but their reasoning abilities on full musical scores remain under-explored. Existing music understanding benchmarks are often limited to snippets, short excerpts, or multiple-choice formats, mostly focusing on monophonic music.

**Limitations of Prior Work**: VLMs face two persistent challenges when processing full scores: (1) Localization failure: models frequently fail to correctly identify measure positions, which is a prerequisite for answering high-level questions about harmony or texture; (2) Hallucination: models generate content not grounded in the score, and localization errors exacerbate these hallucinations.

**Key Challenge**: Full score understanding requires integrating reasoning across pitch, rhythm, harmony, and large-scale structures, but existing benchmarks fail to systematically evaluate this integrated capability.

**Goal**: (1) Construct a full score understanding benchmark covering four difficulty levels; (2) Support dual-modality evaluation using both text (ABC notation) and vision (PDF); (3) Systematically evaluate the capabilities of mainstream LLMs/VLMs.

**Key Insight**: ABC notation, as a structured text format, explicitly encodes measure structures, pitches, and rhythms. It provides an LLM-friendly score representation that can significantly alleviate localization and hallucination issues.

**Core Idea**: Systematically evaluate score understanding using a four-level hierarchy (Starting Information $\rightarrow$ Notation & Notes $\rightarrow$ Chords & Harmony $\rightarrow$ Texture & Form), with ABC notation serving as the upper-bound modality for symbolic-to-theoretical reasoning.

## Method

### Overall Architecture

MSU-Bench addresses the problem of "how to systematically evaluate the understanding of full scores (rather than snippets or multiple-choice questions) by LLMs/VLMs." It collects 150 full scores (Bach, Beethoven, Chopin, Debussy, etc.) and human-annotates 1,800 generative QA pairs. Questions progress through four difficulty levels from basic identification to advanced analysis. The same score is fed in two modalities: ABC notation (Text $\rightarrow$ LLM) and PDF scores (Image $\rightarrow$ VLM). Finally, an LLM-as-judge automatically scores the generated answers to quantify localization, hallucination, and modality gaps.

### Key Designs

**1. Four-level Hierarchical Evaluation Framework: Graduated from Identification to Analysis**

Full score understanding requires integrating reasoning across pitch, rhythm, harmony, and large-scale structure; a single difficulty level cannot characterize where a model fails. This work divides questions into four levels: Level 1 Starting Information (key signature, time signature, tempo) $\rightarrow$ Level 2 Notation and Notes (notes in specific measures, articulations) $\rightarrow$ Level 3 Chords and Harmony (chord identification, tonal analysis) $\rightarrow$ Level 4 Texture and Form (thematic motives, formal structure). Each work includes 3 questions per level. This gradient corresponds to the pedagogical levels of undergraduate musicology, identifying model boundaries and implying that high-performing models can serve as educational assistants.

**2. Dual-Modality Evaluation: Quantifying the Modality Gap between ABC and PDF**

The primary bottleneck for VLMs in processing full scores is the failure of measure localization, which amplifies hallucinations. This study feeds the same score to LLMs via ABC notation and to VLMs via PDF images. ABC notation explicitly encodes measure structures, pitch, and rhythm, bypassing the VLM's visual OCR and localization challenges. It serves as the upper bound for symbolic-to-theoretical reasoning. Comparing the two paths quantifies the "modality gap," exposing how localization difficulties in visual input hinder score understanding.

**3. Joint vs. Sequential Questioning: Testing if Hierarchical Reasoning can be Activated by Prompting**

Natural reasoning dependencies exist between the four levels (e.g., harmony and form analysis rely on basic identification). The questioning method may influence model performance. This study compares two strategies: Joint questioning (presenting all four levels at once) and Sequential questioning (level-by-level). Results show joint questioning performs better, suggesting models can leverage inter-level relationships within a single reasoning pass. This finding identifies the optimal prompting strategy for practical use.

### Loss & Training

The benchmark itself does not involve training. Fine-tuning experiments were conducted using standard Supervised Fine-Tuning (SFT) on the MSU-Bench training set. Results show significant improvements in both modalities without compromising the models' general knowledge.

## Key Experimental Results

### Main Results

**Zero-shot Evaluation (Partial)**

| Model | Modality | L1 | L2 | L3 | L4 | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | ABC | High | Mid | Low | Low | — |
| GPT-4o | PDF | Mid | Low | Low | Low | — |
| Qwen3-72B | ABC | Relative High | — | — | — | — |
| Gemma-3 | PDF | Low | — | — | — | — |

### Key Findings

- **Significant Modality Gap**: ABC text input consistently and significantly outperforms PDF visual input, confirming that localization difficulty is the core bottleneck for VLMs.
- Fine-tuning significantly improves performance in both modalities without damaging general knowledge.
- Joint questioning outperforms sequential questioning, indicating models can utilize hierarchical reasoning.
- Level 1 (basic information) is the easiest, while Level 3-4 (harmony, form) are the most difficult.
- Even the strongest models perform poorly on Level 4, indicating that advanced musical analysis remains a major challenge.

## Highlights & Insights

- The design philosophy of structuring music understanding according to pedagogical levels is clear and practical.
- The discovery of ABC notation as a "silver bullet for localization" has direct application value, suggesting that future music AI should prioritize solving OCR and measure localization.
- This is the first understanding benchmark to cover full scores (including polyphony), filling a significant gap in the field.

## Limitations & Future Work

- Covers only Western classical music; does not include genres like jazz or pop.
- QA pairs are human-annotated, which limits scale due to cost.
- ABC notation has limitations and is less rich than MusicXML.
- Evaluation uses an LLM judge; the accuracy of judging professional musical terminology remains to be verified.

## Related Work & Insights

- **vs. Existing Music NLP Benchmarks**: Existing benchmarks focus on snippets or multiple-choice; MSU-Bench evaluates generative understanding of full scores for the first time.
- **vs. OMR Systems**: OMR focuses on recognition, while MSU-Bench focuses on understanding and reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First full score understanding benchmark, interdisciplinary (Musicology + NLP).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation of 15+ models, dual modalities, and fine-tuning experiments, though lacking human evaluation comparison.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-justified motivation.
- **Value**: ⭐⭐⭐⭐ Provides essential evaluation infrastructure for music AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Phun-Bench: Evaluating LLMs on Phonological Understanding in Chinese](phun-bench_evaluating_llms_on_phonological_understanding_in_chinese.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ICML 2026\] PhaLar: Phasors for Learned Musical Audio Representations](../../ICML2026/audio_speech/phalar_phasors_for_learned_musical_audio_representations.md)

</div>

<!-- RELATED:END -->
