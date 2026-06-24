---
title: >-
  [Paper Note] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs
description: >-
  [ACL 2026][Audio & Speech][Music Audio-Visual Question Answering] As the first comprehensive survey in the Music Audio-Visual Question Answering (Music AVQA) field, this paper systematically analyzes dataset evolution and methodological designs. It demonstrates that specialized input processing, spatio-temporal architectural design, and music domain knowledge are essential for this task, as general multimodal models are insufficient for the unique challenges of musical perfor…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Music Audio-Visual Question Answering"
  - "Spatio-temporal Reasoning"
  - "Multimodal Design"
  - "Domain Specialization"
  - "Survey"
date: 2026-05-08
content_hash: cdca57cff4801fdf
---

# Music Audio-Visual Question Answering Requires Specialized Multimodal Designs

**Conference**: ACL 2026  
**arXiv**: [2505.20638](https://arxiv.org/abs/2505.20638)  
**Code**: [https://github.com/WenhaoYou1/Survey4MusicAVQA](https://github.com/WenhaoYou1/Survey4MusicAVQA)  
**Area**: Multimodal / Music Understanding  
**Keywords**: Music Audio-Visual Question Answering, Spatio-temporal Reasoning, Multimodal Design, Domain Specialization, Survey

## TL;DR

As the first comprehensive survey in the Music Audio-Visual Question Answering (Music AVQA) field, this paper systematically analyzes dataset evolution and methodological designs. It demonstrates that specialized input processing, spatio-temporal architectural design, and music domain knowledge are essential for this task, as general multimodal models are insufficient for the unique challenges of musical performance.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have achieved significant progress in general audio-visual understanding tasks. As a specialized subfield, Music AVQA requires fine-grained spatio-temporal reasoning and cross-modal correspondence for dense, continuous audio-visual signals in music performance videos.

**Limitations of Prior Work**: Fundamental differences exist between Music AVQA and general AVQA: (1) Musical audio signals are continuous and multi-layered (multiple instruments playing simultaneously), unlike discrete and sparse sound events in general scenarios; (2) Precise temporal alignment is required—visual actions of performers often have temporal misalignments with the sound output; (3) Domain-specific knowledge such as instrument identification, music theory (rhythm, harmony), and performance conventions is necessary; (4) Questions involve the quantification of subjective attributes ("more rhythmic," "more melodic").

**Key Challenge**: The broad designs of general multimodal models cannot adequately address the unique complexities of the music domain, necessitating specialized spatio-temporal designs, input processing, and music priors.

**Goal**: (1) Systematically analyze the evolution of Music AVQA datasets; (2) Conduct a comparative analysis of design features across various methods; (3) Identify effective design patterns and propose future research directions.

**Key Insight**: This paper analyzes which designs are empirically correlated with strong performance across three dimensions: input processing, encoder selection, and spatio-temporal architectural design.

**Core Idea**: Music AVQA requires three layers of specialization: specialized input processing (audio-visual feature extraction), specialized architecture (explicit spatio-temporal modeling), and specialized knowledge (integration of music priors).

## Method

### Overall Architecture

This paper presents the first comprehensive survey in the Music AVQA field, addressing "why general multimodal models are insufficient and what specialized designs are required for the music domain." The analysis is structured along two axes: the evolution of datasets (MUSIC-AVQA → v2.0 → MUSIC-AVQA-R) and more than 30 methodologies. Evaluation is anchored on five question types (presence / counting / localization / comparison / temporal) and four performance scenarios (solo / homogeneous ensemble / heterogeneous ensemble / cultural ensemble) to compare input processing, encoder, and spatio-temporal architecture choices. Finally, design patterns associated with superior performance are summarized to guide future research.

### Key Designs

**1. Dataset Evolution Analysis: Tracking the Transition from Bias to Balance and Robustness**

Dataset bias can directly cause distortions in model evaluation. Therefore, this survey clarifies the evolution of three generations of benchmarks: MUSIC-AVQA (9288 videos, 45867 QA) serves as the starting point; v2.0 (10518 videos, 54000 QA) contributes by correcting answer distribution biases, exposing the inflated performance of early methods; MUSIC-AVQA-R further expands to 211,572 questions, introducing robustness evaluation and distinguishing between head/tail samples to reveal model degradation on long-tail data. This trajectory indicates that progress in Music AVQA is largely a progress in "evaluation reliability."

**2. Methodological Design Dimension Analysis: Identifying Truly Effective Designs**

To provide evidence-based design guidelines, the survey decomposes 30+ methods across three dimensions. In input encoder selection, it compares the suitability of visual encoders (CNN / ViT / CLIP) and audio encoders (VGGish / HTS-AT / AST) for dense, multi-layered audio. Regarding spatio-temporal architecture, methods are categorized into those with explicit spatio-temporal designs (e.g., Amuse, AVST, LAST-Att) and those without. Results show that the former consistently perform better on comparison and temporal questions requiring time alignment. For music prior integration, the specific contributions of domain modules, such as beat detection and instrument classification, are analyzed. This decomposition supports the core argument that specialized designs outperform general stacking.

**3. Future Directions: Toward Deeper Music Understanding and Finer Spatio-Temporal Modeling**

Based on the analysis, the survey points out that current methods still have significant room for improvement in comparison and temporal reasoning tasks requiring deep musical understanding. Four actionable directions are proposed: explicitly integrating music theory priors (rhythm analysis, harmony theory) into models; developing finer-grained spatio-temporal attention mechanisms to align visual actions with sound output; utilizing pre-trained music models for transfer learning; and constructing larger-scale, more diverse datasets covering more performance scenarios and cultures.

## Key Experimental Results

### Main Results

**Performance Comparison of Baseline Methods on MUSIC-AVQA (Partial)**

| Method | Spatio-temporal Design | Avg Acc | Comparison Questions | Temporal Questions |
|------|---------|---------|-----------|-----------|
| AVST (2022) | ✓ | Baseline | — | — |
| Amuse (2024) | ✓ | SOTA | Strong | Strong |
| GPT-4o | × | Medium | Weak | Weak |
| General MLLM | × | Lower than specialized | Weak | Weak |

### Key Findings

- Methods with explicit spatio-temporal designs consistently outperform those without.
- General MLLMs (e.g., GPT-4o) underperform compared to specialized designs in Music AVQA.
- Data bias was a significant reason for the inflated performance of early methods; the balancing in v2.0 exposed the true weaknesses of models.
- Robustness evaluation (MUSIC-AVQA-R) reveals that models significantly degrade on tail samples.

## Highlights & Insights

- The first comprehensive survey of Music AVQA, providing a systematic overview of the field.
- The argument that "general models are insufficient and specialization is required" is well-supported by empirical evidence, offering clear guidance for future research.
- The detailed analysis of dataset bias provides valuable lessons for benchmark research across all multimodal domains.

## Limitations & Future Work

- As a survey, it lacks a new methodological contribution.
- The analysis is primarily based on secondary processing of published results, lacking a fair comparison on a unified experimental platform.
- Music AVQA datasets remain limited to relatively simple question types; complex music analysis (e.g., harmonic progression, musical form analysis) has not yet been addressed.

## Related Work & Insights

- **vs General AVQA Surveys**: Specifically focuses on the music domain for the first time, revealing the limitations of general methods in musical contexts.
- **vs Music Information Retrieval (MIR) Surveys**: Approaches from a QA task perspective, supplementing the multimodal reasoning perspective missing in traditional MIR research.

## Rating

- Novelty: ⭐⭐⭐ Survey work with limited novelty, but possesses value as the first in the field to fill the gap.
- Experimental Thoroughness: ⭐⭐⭐ Systematically organizes existing results but lacks new experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and systematically analyzed.
- Value: ⭐⭐⭐⭐ Provides a comprehensive introductory guide and design instructions for Music AVQA researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ACL 2025\] Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering](../../ACL2025/audio_speech/sparsify_music_avqa.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)

</div>

<!-- RELATED:END -->
