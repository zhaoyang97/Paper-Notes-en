---
title: >-
  [Paper Note] MSU-Bench: Musical Score Understanding Benchmark
description: >-
  [ACL 2026][Audio & Speech][Paper Note] MSU-Bench is the first human-annotated benchmark for full musical score understanding, containing 1,800 generative QA pairs from 150 works across four difficulty levels. Evaluation reveals significant deficiencies in LLMs/VLMs regarding score localization and hallucination, while text input via ABC notation significant
tags:
  - ACL 2026
  - Audio & Speech
date: 2026-05-08
content_hash: 42de352aef3735f8
---
# MSU-Bench: Musical Score Understanding Benchmark

**Conference**: ACL 2026  
**arXiv**: [2511.20697](https://arxiv.org/abs/2511.20697)  
**Code**: [https://github.com/Congren-Dai/MSU-Bench](https://github.com/Congren-Dai/MSU-Bench)  
**Area**: Multimodal / Music Understanding  
**Keywords**: Musical score understanding, Music Information Retrieval, ABC notation, Multimodal benchmark, Hallucination

## TL;DR

MSU-Bench is the first human-annotated benchmark for full musical score understanding, containing 1,800 generative QA pairs from 150 works across four difficulty levels. Evaluation reveals significant deficiencies in LLMs/VLMs regarding score localization and hallucination, while text input via ABC notation significantly mitigates these issues.

## Background & Motivation

**Background**: LLMs and VLMs have demonstrated powerful capabilities in natural language processing, but their reasoning abilities regarding full musical scores remain under-explored. Existing music understanding benchmarks are often limited to fragments, short excerpts, or multiple-choice formats, and predominantly focus on monophonic music.

**Limitations of Prior Work**: VLMs face two persistent challenges when processing full scores—(1) Localization failure: models often fail to identify measure positions correctly, which is a prerequisite for answering advanced questions about harmony and texture; (2) Hallucination: models generate content not grounded in the score, with localization errors exacerbating these hallucinations.

**Key Challenge**: Full score understanding requires integrating reasoning across pitch, rhythm, harmony, and large-scale structures, yet existing benchmarks fail to systematically evaluate this comprehensive capability.

**Goal**: (1) Construct a full musical score understanding benchmark covering four difficulty levels; (2) Support dual-modality evaluation for text (ABC notation) and vision (PDF); (3) Systematically evaluate the capabilities of mainstream LLMs/VLMs.

**Key Insight**: ABC notation, as a structured text format, explicitly encodes measure structures, pitch, and rhythm. It provides an LLM-friendly score representation that can significantly alleviate localization and hallucination issues.

**Core Idea**: Systemic evaluation of score understanding using a four-level hierarchical structure (Starting Information → Notation & Notes → Chords & Harmony → Texture & Form), with ABC notation serving as the upper-bound modality for symbolic-to-theoretic reasoning.

## Method

### Overall Architecture

MSU-Bench aims to address the systemic evaluation of LLM/VLM understanding of full musical scores (rather than fragments or multiple-choice questions). It includes 150 full scores (Bach, Beethoven, Chopin, Debussy, etc.) with 1,800 human-annotated generative QA pairs. Questions progress through four levels of difficulty, from basic recognition to advanced analysis. The same score is fed into the model in two modalities—ABC notation (Text → LLM) and PDF scores (Image → VLM). Finally, an LLM-as-judge automatically scores the generated answers to quantify issues such as localization, hallucination, and modality gaps.

### Key Designs

**1. Four-level Hierarchical Evaluation Framework: Gradual Difficulty from Recognition to Analysis**

Full score understanding requires integrating reasoning across pitch, rhythm, harmony, and structure. A single difficulty level cannot characterize where a model fails. This work divides questions into four levels: Level 1 Starting Information (key signature, time signature, tempo) → Level 2 Notation and Notes (notes in specific measures, articulations) → Level 3 Chords and Harmony (chord recognition, tonal analysis) → Level 4 Texture and Form (thematic motifs, formal structure), with three questions per work for each level. This gradient corresponds to undergraduate musicology curriculum levels, allowing the identification of model capability boundaries and implying that models performing well could serve as educational assistants.

**2. Dual-modality Evaluation: Quantifying Modality Gaps via ABC vs. PDF Comparison**

The primary bottleneck for VLMs in processing full scores is the failure of measure localization, which amplifies hallucinations. This study feeds the same score to LLMs via ABC notation and to VLMs via PDF images. ABC notation explicitly encodes measure structure, pitch, and rhythm, bypassing the VLM's visual OCR and localization challenges. Thus, it serves as the upper bound for symbolic-to-theoretic reasoning. Comparing the two paths quantifies the "modality gap," highlighting how localization difficulties in visual inputs drag down overall score understanding.

**3. Joint vs. Sequential Prompting: Testing if Hierarchical Reasoning is Activated by Questioning Style**

Natural reasoning dependencies exist between the four levels (advanced harmony and formal analysis presuppose basic recognition). This work compares two strategies: Joint Prompting (all four levels of questions provided at once) and Sequential Prompting (level-by-level). Results show that Joint Prompting performs better, suggesting models can utilize cross-level correlations within a single reasoning pass. This finding also informs optimal prompting strategies for practical use.

### Loss & Training

The benchmark itself does not involve training. Fine-tuning experiments utilized standard Supervised Fine-Tuning (SFT) on the MSU-Bench training set. Results show significant improvements in both modalities without compromising the models' general knowledge.

## Key Experimental Results

### Main Results

**Zero-shot Evaluation (Partial)**

| Model | Modality | L1 | L2 | L3 | L4 | Avg |
|------|------|----|----|----|----|-----|
| GPT-4o | ABC | High | Mid | Low | Low | — |
| GPT-4o | PDF | Mid | Low | Low | Low | — |
| Qwen3-72B | ABC | Higher | — | — | — | — |
| Gemma-3 | PDF | Low | — | — | — | — |

### Key Findings

- **Significant Modality Gap**: ABC text input consistently and significantly outperforms PDF visual input, confirming that localization difficulty is the core bottleneck for VLMs.
- Fine-tuning significantly improves performance in both modalities without harming general knowledge.
- Joint Prompting outperforms Sequential Prompting, indicating models can utilize hierarchical reasoning.
- Level 1 (Basic Info) is the easiest, while Levels 3-4 (Harmony, Form) are the most challenging.
- Even the strongest models perform poorly at Level 4, indicating that advanced music analysis remains a major challenge.

## Highlights & Insights

- The concept of structured evaluation based on pedagogical levels is clear and practical.
- The discovery of ABC notation as a "silver bullet for localization" has direct application value, suggesting future music AI should prioritize OCR and measure localization.
- This is the first understanding benchmark to cover full scores (including polyphony), filling a critical gap.

## Limitations & Future Work

- Only covers Western classical music; lacks genres like jazz and pop.
- QA pairs are human-annotated, which limits scale due to annotation costs.
- ABC notation itself has limitations and is less rich than MusicXML.
- Evaluation relies on an LLM judge; the accuracy of assessments concerning technical music terminology requires further verification.

## Related Work & Insights

- **vs. Existing Music NLP Benchmarks**: Existing benchmarks focus on fragments or multiple-choice questions; MSU-Bench evaluates generative understanding of full scores for the first time.
- **vs. OMR Systems**: OMR focuses on recognition, whereas MSU-Bench focuses on understanding and reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ First full score understanding benchmark, interdisciplinary (Musicology + NLP).
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation of 15+ models, dual modalities, and fine-tuning experiments, though lacking human evaluation comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-founded motivation.
- Value: ⭐⭐⭐⭐ Provides much-needed evaluation infrastructure for music AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[CVPR 2026\] AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](../../CVPR2026/audio_speech/amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u.md)
- [\[ICML 2026\] PhaLar: Phasors for Learned Musical Audio Representations](../../ICML2026/audio_speech/phalar_phasors_for_learned_musical_audio_representations.md)

</div>

<!-- RELATED:END -->
