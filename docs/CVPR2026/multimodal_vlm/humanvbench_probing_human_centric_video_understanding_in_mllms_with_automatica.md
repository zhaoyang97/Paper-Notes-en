---
title: >-
  [Paper Note] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks
description: >-
  [CVPR 2026][Multimodal VLM][Video benchmark] This paper introduces HumanVBench, a human-centric video understanding benchmark comprising 16 fine-grained tasks…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Video benchmark"
  - "human-centric video understanding"
  - "multimodal large language models"
  - "emotion perception"
  - "speech-visual alignment"
date: 2026-05-08
content_hash: 33694116b7b99929
---

# HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks

**Conference**: CVPR 2026
**arXiv**: [2412.17574](https://arxiv.org/abs/2412.17574)  
**Code**: [https://github.com/datajuicer/data-juicer/tree/HumanVBench](https://github.com/datajuicer/data-juicer/tree/HumanVBench)  
**Area**: Multimodal VLM / Video Understanding
**Keywords**: Video benchmark, human-centric video understanding, multimodal large language models, emotion perception, speech-visual alignment

## TL;DR

This paper introduces HumanVBench, a human-centric video understanding benchmark comprising 16 fine-grained tasks, accompanied by two automated pipelines (video annotation and distractor-aware QA synthesis). Evaluation of 30 mainstream video MLLMs reveals critical deficiencies in current models regarding nuanced emotion perception and speech-visual alignment.

## Background & Motivation

Multimodal large language models (MLLMs) have expanded from text to images and video, with video-oriented MLLMs attracting increasing attention for their potential to emulate human visual perception. Whether these models truly achieve human-like understanding—particularly in complex human-centric scenarios—remains an open question.

Existing MLLM benchmarks suffer from three core limitations: (1) mainstream benchmarks (e.g., Video-MME) focus on general video understanding and lack structured, fine-grained evaluation of human-centric perceptual capabilities; (2) emotion understanding datasets (e.g., VEATIC) rely on discrete emotion classification with fixed categories, missing multi-dimensional tasks such as emotional dynamics and cross-subject intensity comparison; (3) speech-visual synchronization—audio-visual mismatches easily detected by humans—is frequently overlooked in evaluation.

These fundamental perceptual skills (emotion, behavior, identity recognition, audio-visual alignment) are prerequisites for higher-order human-relevant reasoning tasks (narrative reasoning, intent inference, social intelligence). Yet existing benchmarks either conflate perception with higher-order reasoning or depend on extensive manual annotation that is difficult to scale.

- **Core Idea**: Two automated pipelines are designed—a video annotation pipeline leveraging 20+ SOTA data processing operators to generate dense multimodal annotations, and a QA synthesis pipeline that produces semantically deceptive distractors via multi-model ensemble and model error mining—thereby constructing a high-quality, scalable human-centric video benchmark with minimal human labor.

## Method

### Overall Architecture

HumanVBench is constructed in three stages: (1) videos featuring human subjects are collected from Pexels and MF2, followed by scene segmentation; (2) a human-centric video annotation pipeline extracts multimodal annotations spanning visual, auditory, and holistic event-level cues; (3) a distractor-aware QA synthesis pipeline generates multiple-choice questions, followed by human verification and answer-leakage post-processing. The final benchmark contains 2,475 question instances covering 16 tasks.

### Key Designs

1. **Human-Centric Video Annotation Pipeline**:

    - **Function**: Automatically extracts dense human-centric multimodal annotations from raw videos.
    - **Mechanism**: A modular operator chain is constructed—`video_human_tracks_extraction_mapper` links detected faces and bodies across frames via cross-frame overlap thresholds to produce reliable subject trajectories; `human_demographics_mapper` infers demographic attributes (age, gender, etc.) from face crops; `video_human_description_mapper` and `video_facial_description_mapper` use MLLMs to describe appearance/pose and facial expression dynamics respectively (cropping ensures descriptions are free from background interference); on the audio side, `active_speaker_detection_mapper` (fusing audio-visual cues to localize speakers), `asr_mapper` (speech-to-text), and `speech_emotion_recognition_mapper` (speech emotion detection) are employed. The pipeline is built on the Data-Juicer framework.
    - **Design Motivation**: Conventional benchmarks relying on extensive manual annotation cannot scale to in-the-wild video data; by leveraging SOTA task-specific models for automatic annotation, human effort is reduced to verifying approximately 25% of cases.

2. **Distractor-Aware QA Synthesis Pipeline**:

    - **Function**: Automatically generates high-quality multiple-choice questions with semantically deceptive distractors.
    - **Mechanism**: Four stages are involved—(1) videos are filtered according to task-specific criteria; (2) target subjects are marked with red bounding boxes in "annotated videos" fed to a Video-MLLM to generate preliminary descriptions, from which task-relevant attributes are extracted and distribution-balanced; (3) multi-model ensemble answer selection—multiple MLLMs (Gemini, VideoLLaMA3, ShareGPT4Video) each produce candidate answers, the correct answer is selected via preference voting, and **model error responses are repurposed as distractors** rather than being arbitrarily constructed, ensuring distractors reflect typical model failure patterns; (4) human verification, where annotators confirm options in 75% of cases and rewrite them in the remaining 25%.
    - **Design Motivation**: Repurposing model errors as distractors is the core methodological innovation—it ensures distractors are genuinely deceptive (reflecting the most common model mistakes) rather than randomly generated options based on semantic similarity, as in conventional approaches.

3. **16 Fine-Grained Task Design**:

    - **Function**: Comprehensively covers the foundational perceptual layer of human-centric video understanding.
    - **Mechanism**: Tasks are divided into two categories based on observability. **Intrinsic Emotion** (4 tasks): Emotion Recognition (ER), Emotion Temporal Analysis (ETA), Attitude Recognition (AT), Emotion Intensity Comparison (EIC). **Extrinsic Behavior** is further divided into three sub-categories—Subject Identification (4 tasks: Text-to-Human T2H, Human-to-Text H2T, Human Counting HC, Appearance Time Detection ATD), Behavior Analysis (4 tasks: Behavior Temporal Analysis BTA, Behavior Causal Analysis BCA, Action at Specified Time AST, Time of Specific Action TSA), and Speech-Visual Alignment (4 tasks: Audio-Visual Speaker Matching AVSM, Active Speaker Detection ASD, Audio-Visual Alignment Detection AVAD, Speech Content Matching SCM).
    - **Design Motivation**: Existing benchmarks conflate basic perception and higher-order reasoning, making it impossible to pinpoint specific perceptual deficiencies. By focusing exclusively on the foundational perceptual layer, this work provides a clear capability baseline for subsequent higher-order reasoning evaluation.

### Loss & Training

HumanVBench is an evaluation benchmark and does not involve model training. Answer leakage mitigation: models are tested without visual input and QA instances that are answered correctly with high frequency under this condition (approximately 6%) are removed, ensuring visual information is required. Annotation reliability: Cohen's Kappa of 0.8833 is achieved between two independent annotators on 240 randomly sampled questions.

## Key Experimental Results

### Main Results

| Model | Modality | Emotion Perception | Subject ID | Behavior Analysis | 12-Task Avg. | Speech-Visual | 16-Task Avg. |
|--------|------|------|----------|------|------|------|------|
| Random Guess | - | 24.4 | 25.2 | 22.9 | 24.2 | 31.2 | 25.9 |
| Qwen-VL3 (7B) | V | 43.2 | 67.6 | 54.3 | 55.0 | 48.3 | 53.4 |
| VideoLLaMA3 (7B) | V | 39.7 | 68.5 | 55.8 | 54.7 | 45.0 | 52.3 |
| Qwen2.5-Omni (7B) | V+A | 35.5 | 44.5 | 38.3 | 39.4 | 54.6 | 43.2 |
| GPT-4o | V | 33.6 | 50.9 | 62.1 | 48.9 | - | - |
| Gemini-2.5-Pro | V+A | 52.9 | 83.5 | 70.7 | 69.0 | 86.5 | **73.4** |
| **Human** | - | **84.6** | **88.5** | **87.0** | **86.7** | **94.4** | **88.6** |

### Ablation Study (Annotation Quality)

| Configuration | Proportion | Description |
|------|---------|------|
| Annotators confirm existing options | 75% | Automatically generated options are sufficient |
| Annotators rewrite correct answers | 25% | Human correction required |
| Cohen's Kappa (IAA) | 0.8833 | High annotation consistency |
| Answer leakage removal rate | ~6% | Questions answerable without visual input |

### Key Findings

- **Emotion perception is the most critical weakness**: Even Gemini-2.5-Pro (best model, 52.9%) falls far below human-level performance (84.6%), with a gap exceeding 30 percentage points.
- **GPT-4o performs unexpectedly poorly**: It scores below multiple open-source models on emotion understanding and several subject identification tasks, with an overall 12-task average of 48.9% surpassed by Qwen-VL3's 55.0%.
- **Speech-visual alignment reveals a catastrophic gap**: Nearly all open-source audio-visual models perform near chance level on AVAD and SCM tasks, indicating that current models lack precise lip-reading capabilities. Gemini series models are the only exception.
- **Speaker emotion recognition is harder**: Emotion recognition accuracy during speaking is consistently 2–4 percentage points lower than on the full dataset, due to greater facial expression complexity during speech.
- **The open-source vs. proprietary gap is narrowing**: Qwen3-VL approaches commercial model performance on visual tasks.

## Highlights & Insights

- **Repurposing model errors as distractors** represents a significant methodological innovation in benchmark construction—conventional practice generates random distractors, whereas this work leverages multi-model ensemble errors as distractors, enabling the benchmark to naturally distinguish genuine capability differences. The insight is that these distractors correspond precisely to the mistakes models are most prone to making.
- **The 16-task taxonomy** makes a structural contribution to video understanding evaluation—the systematic categorization from intrinsic emotion to extrinsic behavior, and from unimodal to cross-modal tasks, provides a clear capability map for subsequent work.
- **The operator-based annotation pipeline design** is transferable to benchmark construction in other domains—the paradigm of using SOTA models as automatic annotators with humans serving only as verifiers substantially reduces benchmark construction costs.

## Limitations & Future Work

- Videos are primarily sourced from Pexels (royalty-free), and scene diversity may be limited compared to real-world social media or surveillance footage.
- Emotion annotation relies on facial expressions and speech, neglecting contextual cues (e.g., the influence of event context on emotional state).
- Only multiple-choice question format is evaluated; open-ended response evaluation (potentially more reflective of real-world application needs) is not considered.
- The subject tracking operator may fail in heavily occluded scenes, potentially degrading downstream annotation quality.

## Related Work & Insights

- **vs. Video-MME**: Video-MME is a general-purpose video benchmark; HumanVBench focuses on the human-centric dimension, and the two are complementary.
- **vs. Social-IQ**: Social-IQ conflates basic perception with higher-order reasoning; HumanVBench focuses exclusively on the foundational perceptual layer, providing a capability baseline for higher-order evaluation.
- **vs. VEATIC**: VEATIC provides only discrete emotion classification; HumanVBench extends to multi-dimensional tasks including temporal emotion analysis and intensity comparison.

## Rating

- Novelty: ⭐⭐⭐⭐ The 16 fine-grained human-centric task taxonomy and model-error-driven distractor generation both constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 30 models spanning open-source/proprietary and visual/audio-visual dimensions, with rigorous annotation quality validation.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, task classification is systematic, and result analysis is in-depth.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in human-centric video understanding evaluation, reveals critical deficiencies in current models, and provides important guidance for the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[AAAI 2026\] Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh](../../AAAI2026/multimodal_vlm/towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)

</div>

<!-- RELATED:END -->
