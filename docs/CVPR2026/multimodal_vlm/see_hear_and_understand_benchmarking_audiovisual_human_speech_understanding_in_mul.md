---
title: >-
  [Paper Note] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][multimodal benchmark] This paper introduces AV-SpeakerBench, a benchmark comprising 3,212 speaker-centric audiovisual reasoning multiple-choice questions…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "multimodal benchmark"
  - "speaker-centric reasoning"
  - "audiovisual fusion"
  - "large language model evaluation"
  - "audiovisual understanding"
date: 2026-05-08
content_hash: b4e0726e88e66726
---

# See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2512.02231](https://arxiv.org/abs/2512.02231)  
**Code**: [https://plnguyen2908.github.io/AV-SpeakerBench-project-page/](https://plnguyen2908.github.io/AV-SpeakerBench-project-page/)  
**Area**: Multimodal VLM / Audiovisual Understanding
**Keywords**: multimodal benchmark, speaker-centric reasoning, audiovisual fusion, large language model evaluation, audiovisual understanding

## TL;DR

This paper introduces AV-SpeakerBench, a benchmark comprising 3,212 speaker-centric audiovisual reasoning multiple-choice questions, which systematically evaluates multimodal large language models on fine-grained audiovisual fusion capabilities—specifically, *who is speaking, what was said, and when*—revealing a gap of over 20% between the strongest current models and human performance.

## Background & Motivation

Multimodal large language models (MLLMs) are expanding from image-text understanding toward unified audio-video-language comprehension. However, existing video benchmarks suffer from a critical flaw: most questions can be answered using visual information alone (e.g., "How many people are in the video?"), with virtually no dependence on audio. Even the few benchmarks that include audio only address coarse-grained sound event classification (e.g., "male/female voice"), and fail to assess whether a model truly understands **who said what**.

Audiovisual speaker perception is a long-standing research problem encompassing speaker detection, recognition, and speech localization. Yet existing datasets rely on closed-set labels or frame-level annotations, rendering them incompatible with the open-ended language evaluation paradigm of MLLMs.

The core problem is: **Can current MLLMs accurately associate the people seen in video with the speech they hear?** This requires cross-modal temporal reasoning—not only identifying a speaker's appearance and voice, but aligning "who said what and when" along the temporal axis.

AV-SpeakerBench is built on three design principles: (1) the speaker, rather than the scene, is the central unit of reasoning; (2) fusion-driven question design—audiovisual dependencies are embedded into the semantic structure of questions themselves; (3) expert manual annotation ensures temporal precision and cross-modal validity.

## Method

### Overall Architecture

AV-SpeakerBench is an evaluation benchmark containing 3,212 four-choice multiple-choice questions, spanning 2,051 video clips (5–30 seconds) and 12 task types. Videos are sourced from YouTube, including movie clips, interviews, and podcasts, featuring rich multi-speaker dialogue scenarios. All annotations are produced by experienced researchers through multiple rounds of review.

### Key Designs

1. **Speaker-Centric Task Taxonomy (12 Task Types)**:

    - **Function**: Comprehensively evaluates speaker-centric audiovisual reasoning across multiple dimensions.
    - **Mechanism**: Tasks are grouped into three categories—speaker-related (speaker detection, identification, and counting; e.g., "When did the person in the black T-shirt say 'what's up'?"), vision-related (visual attribute recognition, activity recognition, and counting, requiring audio assistance), and audio-related (duration, pitch, speech rate, intensity, and speech counting, requiring visual assistance). Every question is designed such that integrating both auditory and visual information is necessary for a correct answer.
    - **Design Motivation**: Existing benchmarks either evaluate vision in isolation or assess audio at a coarse granularity, and cannot test genuine fusion capability. The 12 task types cover a spectrum of abilities ranging from recognition and counting to temporal localization.

2. **Fusion-Driven Question Semantics**:

    - **Function**: Ensures that every question requires cross-modal reasoning at the semantic level.
    - **Mechanism**: Audiovisual dependencies are encoded within the question text and answer options through: (a) linking spoken phrases to visible identities (e.g., "after the person in the grey shirt finished speaking..."); (b) using visual events to anchor speech (e.g., "what did she say before drinking water?") or using speech to anchor visual events (e.g., "how many people were visible when he said 'we're not cool'?"); (c) integrated reasoning in multi-speaker scenarios (e.g., "from when the man in the grey shirt shook his finger until the end of the video, how many times was 'red line' mentioned by everyone?").
    - **Design Motivation**: Simple "what was said in the video" questions can be solved by pure audio transcription combined with an LLM. Interweaving visual identity, temporal anchors, and speech content within the question semantics compels models to perform genuine cross-modal alignment.

3. **Expert-Driven Quality Control Pipeline**:

    - **Function**: Ensures temporal precision and cross-modal validity for every question.
    - **Mechanism**: Annotation proceeds in three stages: (a) annotators select 5–30-second clips from full-length videos satisfying task requirements (multiple speakers, meaningful conversational dynamics); (b) questions and distractor options are composed following detailed task guidelines, with distractors drawn from entities, actions, and speech events within the same clip; (c) multi-stage review (initial review by an independent researcher → language model polishing → final review by at least two additional researchers), filtering out ambiguous, inconsistent, or unimodally solvable questions.
    - **Design Motivation**: Crowdsourced annotation struggles to guarantee quality for cross-modal reasoning questions; expert annotation, though costly, ensures that every question genuinely requires audiovisual fusion.

### Loss & Training

AV-SpeakerBench is a purely evaluative benchmark and involves no model training. Evaluation is conducted using multiple-choice accuracy, reported separately across all 12 task types.

## Key Experimental Results

### Main Results

| Model | Parameters | Overall Accuracy | vs. Human (93.74%) |
|-------|-----------|-----------------|-------------------|
| **Gemini 2.5 Pro (Thinking)** | — | **73.04** | −20.70 |
| Gemini 2.5 Flash (Thinking) | — | 67.84 | −25.90 |
| Gemini 2.0 Flash | — | 53.21 | −40.53 |
| **Qwen3-Omni** | 30B | **54.14** | −39.60 |
| Qwen2.5-Omni | 7B | 46.64 | −47.10 |
| Phi-4 Multimodal | 5.6B | 38.45 | −55.29 |
| VITA-1.5 | 7B | 36.27 | −57.47 |
| Video-LLaMA2 | 7B | 37.67 | −56.07 |
| PandaGPT | 7B | 22.88 | −70.86 |

### Ablation Study (Modality Ablation)

| Model | Vision Only | Audio + Vision | Audio Gain | Notes |
|-------|------------|---------------|-----------|-------|
| Gemini 2.5 Pro | ~60% | 73.04 | **+10–20%** | Consistently benefits from audio |
| Qwen3-Omni 30B | ~52% | 54.14 | **+2% or negative** | Weak audiovisual fusion |

### Key Findings

- Human accuracy is 93.74%, while the strongest model achieves 73.04%—a gap exceeding 20 percentage points—demonstrating that speaker-centric audiovisual reasoning remains a fundamental challenge.
- Gemini 2.5 Pro's advantage stems primarily from stronger audiovisual fusion (audio consistently yields 10–20% gains) rather than superior visual perception.
- Qwen3-Omni 30B approaches Gemini 2.0 Flash, yet shows limited or even negative improvement upon adding audio, indicating that fusion capability is the bottleneck for open-source models.
- Error analysis identifies audio perception and temporal reasoning as the primary sources of failure.
- Earlier open-source models (Video-LLaMA, PandaGPT, Unified-IO 2) perform near random chance despite claiming audiovisual support.

## Highlights & Insights

- **Benchmark design philosophy**: Rather than simply annotating question-answer pairs, the paper "hard-codes" cross-modal dependencies into question semantics, rendering unimodal shortcuts ineffective. This *fusion-driven design* paradigm is generalizable to other multimodal benchmarks.
- **Modality ablation reveals the essential gap**: The performance gap between Gemini and Qwen lies not in visual perception but in fusion capability, providing clear guidance for the open-source community on where improvement is needed.
- **Fine-grained decomposition across 12 task types**: The multi-dimensional evaluation—spanning detection and identification through pitch and speech rate—offers substantially greater diagnostic value than a single aggregate "audiovisual understanding" score.

## Limitations & Future Work

- Evaluation is limited to multiple-choice accuracy, leaving open-ended responses and conversational interaction—more naturalistic interaction formats—unaddressed.
- Video sources are predominantly English, and cross-lingual generalizability is untested.
- Certain tasks (e.g., pitch comparison) have limited practical demand in real-world applications.
- The dataset scale of 3,212 questions is insufficient for training purposes; it serves exclusively as an evaluation resource.
- The paper does not analyze whether models answer questions via lip reading rather than genuine audio processing.

## Related Work & Insights

- **vs. Video-MME**: 900 videos / 2,700 questions, but most are solvable through vision alone; AV-SpeakerBench enforces audiovisual fusion.
- **vs. AVQA**: Evaluates audiovisual matching but is not speaker-centric and focuses on non-speech sound events.
- **vs. WorldSense**: Involves audiovisual content but focuses on scene-level understanding (musical style, sound summarization) rather than speaker-level reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first benchmark to systematically evaluate speaker-centric audiovisual reasoning, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 6 Gemini variants and 12 open-source models, with in-depth modality ablation and error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Design principles are articulated clearly, and task examples are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Identifies the core bottleneck in audiovisual fusion for current MLLMs, providing clear direction for the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)
- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->
