---
title: >-
  [Paper Note] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs
description: >-
  [ACL 2026][Audio & Speech][Music Audio-Visual Question Answering] As the first comprehensive survey of the Music Audio-Visual Question Answering (Music AVQA) field, this paper systematically analyzes dataset evolution and method design, demonstrating that specialized input processing, spatiotemporal architectural design, and music domain knowledge are essential for this task, and that general-purpose multimodal models are insufficient to address the unique challenges of music performance understanding.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Music Audio-Visual Question Answering
  - Spatiotemporal Reasoning
  - Multimodal Design
  - Domain Specialization
  - Survey
date: 2026-05-08
content_hash: 040636e4504704ff
---

# Music Audio-Visual Question Answering Requires Specialized Multimodal Designs

**Conference**: ACL 2026
**arXiv**: [2505.20638](https://arxiv.org/abs/2505.20638)
**Code**: [https://github.com/WenhaoYou1/Survey4MusicAVQA](https://github.com/WenhaoYou1/Survey4MusicAVQA)
**Area**: Multimodal / Music Understanding
**Keywords**: Music Audio-Visual Question Answering, Spatiotemporal Reasoning, Multimodal Design, Domain Specialization, Survey

## TL;DR

As the first comprehensive survey of the Music Audio-Visual Question Answering (Music AVQA) field, this paper systematically analyzes dataset evolution and method design, demonstrating that specialized input processing, spatiotemporal architectural design, and music domain knowledge are essential for this task, and that general-purpose multimodal models are insufficient to address the unique challenges of music performance understanding.

## Background & Motivation

**State of the Field**: Multimodal large language models have achieved remarkable progress on general audio-visual understanding tasks. Music AVQA, as a specialized subfield, requires fine-grained spatiotemporal reasoning and cross-modal correspondence over dense, continuous audio-visual signals in music performance videos.

**Limitations of Prior Work**: Music AVQA differs fundamentally from general AVQA in several respects: (1) musical audio signals are continuous and multi-layered (multiple instruments playing simultaneously), unlike the discrete and sparse sound events in general scenarios; (2) precise temporal alignment is required, as a performer's visual actions and their acoustic output are temporally misaligned; (3) domain-specific knowledge is necessary, including instrument recognition, music theory (rhythm, harmony), and performance conventions; (4) questions involve quantifying subjective attributes (e.g., "more rhythmic," "more melodic").

**Root Cause**: The broad design of general-purpose multimodal models is insufficient to address the unique complexity of the music domain, which demands specialized spatiotemporal design, input processing, and music priors.

**Paper Goals**: (1) Systematically analyze the evolution of Music AVQA datasets; (2) Comparatively examine the design characteristics of various methods; (3) Identify effective design patterns and propose future directions.

**Starting Point**: Analysis of which design choices correlate with strong performance, examined along three dimensions: input processing, encoder selection, and spatiotemporal architectural design.

**Core Idea**: Music AVQA requires three levels of specialization — specialized input processing (audio-visual feature extraction), specialized architecture (explicit spatiotemporal modeling), and specialized knowledge (music prior integration).

## Method

### Overall Architecture

This is a survey paper that systematically analyzes Music AVQA datasets (MUSIC-AVQA → v2.0 → MUSIC-AVQA-R) and over 30 methods. Starting from five question types (existence / counting / localization / comparison / temporal) and four performance scenarios (solo / homogeneous ensemble / heterogeneous ensemble / culturally distinctive ensemble), it provides a systematic comparison of design choices across methods.

### Key Designs

1. **Dataset Evolution Analysis**:

    - Function: Traces the development of Music AVQA datasets from biased to balanced designs.
    - Mechanism: MUSIC-AVQA (9,288 videos, 45,867 QA pairs) → v2.0 (10,518 videos, 54,000 QA pairs, with answer distribution bias corrected) → MUSIC-AVQA-R (expanded to 211,572 questions, introducing robustness evaluation and head/tail sample distinction).
    - Design Motivation: Dataset bias and limitations directly affect the reliability of model evaluation.

2. **Method Design Dimension Analysis**:

    - Function: Identifies design patterns correlated with strong performance.
    - Mechanism: Methods are analyzed along three dimensions — (a) input encoder selection: comparing visual encoders such as CNN/ViT/CLIP and audio encoders such as VGGish/HTS-AT/AST; (b) spatiotemporal architecture: distinguishing methods with explicit spatiotemporal design (e.g., Amuse, AVST, LAST-Att) from those without, with the former consistently achieving better performance; (c) music prior integration: analyzing the contribution of domain-specific modules such as beat detection and instrument classification.
    - Design Motivation: To provide empirically supported design guidelines for researchers.

3. **Proposed Future Directions**:

    - Function: Guides the future development of Music AVQA research.
    - Mechanism: (a) Integrating music theory priors (rhythmic analysis, harmony theory) into model design; (b) developing finer-grained spatiotemporal attention mechanisms; (c) leveraging pretrained music models for transfer learning; (d) constructing larger-scale and more diverse datasets.
    - Design Motivation: Current methods still have substantial room for improvement, particularly in comparative and temporal reasoning tasks that require deep musical understanding.

### Loss & Training

This is a survey paper and does not involve specific training strategies.

## Key Experimental Results

### Main Results

**Performance Comparison of Methods on MUSIC-AVQA Benchmark (Partial)**

| Method | Spatiotemporal Design | Avg Acc | Comparison Qs | Temporal Qs |
|--------|----------------------|---------|---------------|-------------|
| AVST (2022) | ✓ | Baseline | — | — |
| Amuse (2024) | ✓ | SOTA | Strong | Strong |
| GPT-4o | × | Moderate | Weak | Weak |
| General MLLM Methods | × | Below specialized methods | Weak | Weak |

### Key Findings

- Methods with explicit spatiotemporal design consistently outperform those without.
- General-purpose MLLMs (e.g., GPT-4o) perform worse than specialized methods on Music AVQA.
- Dataset bias is a key reason for the inflated performance of early methods — the rebalancing in v2.0 exposed the true weaknesses of these models.
- Robustness evaluation (MUSIC-AVQA-R) reveals significant performance degradation on tail samples.

## Highlights & Insights

- The first comprehensive survey of Music AVQA, providing a systematic overview of the field.
- The argument that "general-purpose models are insufficient and specialization is required" is well-supported empirically, offering clear guidance for future research directions.
- The detailed analysis of dataset bias issues serves as a valuable reference for multimodal benchmark research more broadly.

## Limitations & Future Work

- As a survey, the paper makes no contribution of a novel method.
- The analysis is primarily a secondary synthesis of published results, lacking fair comparisons under a unified experimental platform.
- Music AVQA datasets remain limited to relatively simple question types; deeper musical analysis tasks (e.g., harmonic progression, formal structure analysis) have yet to be addressed.

## Related Work & Insights

- **vs. General AVQA Surveys**: This is the first survey to focus specifically on audio-visual question answering in the music domain, revealing the limitations of general methods when applied to music.
- **vs. Music Information Retrieval Surveys**: By approaching the problem from a QA task perspective, this work supplements the multimodal reasoning viewpoint that is absent from traditional MIR research.

## Rating

- Novelty: ⭐⭐⭐ Survey work with limited novelty, though its value as the first in the field fills an important gap.
- Experimental Thoroughness: ⭐⭐⭐ Systematically organizes existing results, but lacks new experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and systematically analyzed.
- Value: ⭐⭐⭐⭐ Provides Music AVQA researchers with a comprehensive introductory guide and design reference.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)

<!-- RELATED:END -->
