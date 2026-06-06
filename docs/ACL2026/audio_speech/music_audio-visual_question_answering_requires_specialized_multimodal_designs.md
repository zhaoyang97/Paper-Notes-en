---
title: >-
  [Paper Note] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs
description: >-
  [ACL 2026][Audio & Speech][Music Audio-Visual Question Answering] As the first comprehensive survey in the Music Audio-Visual Question Answering (Music AVQA) field…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Music Audio-Visual Question Answering"
  - "Spatio-Temporal Reasoning"
  - "Multimodal Design"
  - "Domain Specialization"
  - "Survey"
date: 2026-05-08
content_hash: b1aef22b50b28579
---

# Music Audio-Visual Question Answering Requires Specialized Multimodal Designs

**Conference**: ACL 2026  
**arXiv**: [2505.20638](https://arxiv.org/abs/2505.20638)  
**Code**: [https://github.com/WenhaoYou1/Survey4MusicAVQA](https://github.com/WenhaoYou1/Survey4MusicAVQA)  
**Area**: Multimodal / Music Understanding  
**Keywords**: Music Audio-Visual Question Answering, Spatio-Temporal Reasoning, Multimodal Design, Domain Specialization, Survey

## TL;DR

As the first comprehensive survey in the Music Audio-Visual Question Answering (Music AVQA) field, this paper systematically analyzes dataset evolution and methodological designs. It demonstrates that specialized input processing, spatio-temporal architectures, and music domain knowledge are critical for this task, as general multimodal models are insufficient for the unique challenges of music performance.

## Background & Motivation

**Background**: Multimodal large language models have made significant progress in general audio-visual understanding tasks. As a specialized subfield, Music AVQA requires fine-grained spatio-temporal reasoning and cross-modal correspondence of dense, continuous audio-visual signals in music performance videos.

**Limitations of Prior Work**: Fundamental differences exist between Music AVQA and general AVQA: (1) Music audio signals are continuous and multi-layered (multiple instruments playing simultaneously), unlike discrete, sparse sound events in general scenes; (2) Precise temporal alignment is required—there is often a temporal offset between a performer's visual movements and the sound output; (3) Domain-specific knowledge such as instrument identification, music theory (rhythm, harmony), and performance conventions is necessary; (4) Questions often involve the quantification of subjective attributes ("more rhythmic," "more melodic").

**Key Challenge**: The broad designs of general multimodal models cannot fully address the unique complexity of the music domain—specialized spatio-temporal designs, input processing, and music priors are required.

**Goal**: (1) Systematically analyze the evolution of Music AVQA datasets; (2) Comparatively analyze structural design features of various methods; (3) Identify effective design patterns and propose future directions.

**Key Insight**: Analysis is conducted across three dimensions: input processing, encoder selection, and spatio-temporal architecture design to determine which designs correlate empirically with strong performance.

**Core Idea**: Music AVQA requires three layers of specialization: specialized input processing (audio-visual feature extraction), specialized architecture (explicit spatio-temporal modeling), and specialized knowledge (integration of music priors).

## Method

### Overall Architecture

This is a survey paper that systematically analyzes the datasets (MUSIC-AVQA → v2.0 → MUSIC-AVQA-R) and 30+ methods in the Music AVQA field. Starting from five question types (existential/counting/location/comparative/temporal) and four performance scenarios (solo/homogeneous ensemble/heterogeneous ensemble/cultural ensemble), it systematically compares the design choices of various methods.

### Key Designs

1.  **Dataset Evolution Analysis**:
    *   **Function**: Tracks the development of Music AVQA datasets from biased to balanced.
    *   **Mechanism**: MUSIC-AVQA (9,288 videos, 45,867 QA) → v2.0 (10,518 videos, 54,000 QA, fixing answer distribution bias) → MUSIC-AVQA-R (expanded to 211,572 questions, introducing robustness evaluation and head/tail sample differentiation).
    *   **Design Motivation**: Biases and limitations in datasets directly affect the reliability of model evaluation.

2.  **Methodological Design Dimension Analysis**:
    *   **Function**: Identifies design patterns associated with high performance.
    *   **Mechanism**: Analysis across three dimensions: (a) Input encoder selection: comparing visual encoders like CNN/ViT/CLIP and audio encoders like VGGish/HTS-AT/AST; (b) Spatio-temporal architecture: distinguishing between methods with explicit spatio-temporal designs (e.g., Amuse, AVST, LAST-Att) and those without, where the former show better performance consistency; (c) Music prior integration: analyzing the contribution of domain-specific modules like beat detection and instrument classification.
    *   **Design Motivation**: Provides empirically supported design guidelines for researchers.

3.  **Future Directions**:
    *   **Function**: Guides the direction of Music AVQA research.
    *   **Mechanism**: (a) Integrating music theory priors (rhythm analysis, harmony theory) into model design; (b) Developing finer-grained spatio-temporal attention mechanisms; (c) Leveraging pre-trained music models for transfer learning; (d) Constructing larger and more diverse datasets.
    *   **Design Motivation**: Current methods still have significant room for improvement, particularly in comparative and temporal reasoning requiring deep music understanding.

### Loss & Training

A survey paper; does not involve specific training strategies.

## Key Experimental Results

### Main Results

**Performance Comparison of Baseline Methods on MUSIC-AVQA (Partial)**

| Method | Spatio-temporal Design | Avg Acc | Comparative Questions | Temporal Questions |
| :--- | :--- | :--- | :--- | :--- |
| AVST (2022) | ✓ | Baseline | — | — |
| Amuse (2024) | ✓ | SOTA | Strong | Strong |
| GPT-4o | × | Moderate | Weak | Weak |
| General MLLMs | × | Below Specialized | Weak | Weak |

### Key Findings

*   Methods with explicit spatio-temporal designs consistently outperform those without.
*   General MLLMs (e.g., GPT-4o) perform worse on Music AVQA than specialized designs.
*   Dataset bias was a major reason for the inflated performance of early methods—the balancing in v2.0 exposed true model weaknesses.
*   Robustness evaluation (MUSIC-AVQA-R) reveals significant model degradation on tail samples.

## Highlights & Insights

*   The first comprehensive survey of Music AVQA, systematically organizing the landscape of the field.
*   The argument that "general models are insufficient and specialization is required" is well-supported by empirical evidence—providing clear guidance for future research.
*   Detailed analysis of dataset bias issues provides valuable lessons for all multimodal benchmark research.

## Limitations & Future Work

*   As a survey, it lacks contributions of new methods.
*   The analysis is primarily based on secondary organization of published results, lacking fair comparisons on a unified experimental platform.
*   Music AVQA datasets are still limited to relatively simple question types; true music analysis (e.g., harmonic progression, musical form analysis) has not yet been addressed.

## Related Work & Insights

*   **vs. General AVQA Surveys**: Focuses specifically on the music domain for the first time, revealing the limitations of general methods in music contexts.
*   **vs. Music Information Retrieval (MIR) Surveys**: Approaches from a QA task perspective, complementing the multimodal reasoning viewpoint missing in traditional MIR research.

## Rating

*   Novelty: ⭐⭐⭐ Survey work; novelty is limited, but holds value as the first in the field to fill the gap.
*   Experimental Thoroughness: ⭐⭐⭐ Systematically organizes existing results but lacks new experiments.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure and systematic analysis.
*   Value: ⭐⭐⭐⭐ Provides a comprehensive introductory guide and design instructions for Music AVQA researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)

</div>

<!-- RELATED:END -->
