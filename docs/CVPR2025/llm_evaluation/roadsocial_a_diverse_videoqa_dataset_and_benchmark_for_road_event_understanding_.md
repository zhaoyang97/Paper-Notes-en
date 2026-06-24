---
title: >-
  [Paper Note] RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives
description: >-
  [CVPR 2025][LLM Evaluation][VideoQA] This paper proposes RoadSocial, a large-scale and diverse VideoQA dataset sourced from social media (consisting of 13.2K videos and 260K QA pairs) that covers multi-regional and multi-perspective road event scenarios globally. Through a semi-automatic annotation framework and 12 categories of QA tasks, the paper systematically evaluates the road event understanding capabilities of 18 Video LLMs.
tags:
  - "CVPR 2025"
  - "LLM Evaluation"
  - "VideoQA"
  - "Road Events"
  - "Social Media"
  - "Dataset"
  - "Video LLM"
date: 2026-05-08
content_hash: 685d69f494302b7f
---

# RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: VideoQA, Road Events, Social Media, Dataset, Video LLM

## TL;DR

This paper proposes RoadSocial, a large-scale and diverse VideoQA dataset sourced from social media (consisting of 13.2K videos and 260K QA pairs) that covers multi-regional and multi-perspective road event scenarios globally. Through a semi-automatic annotation framework and 12 categories of QA tasks, the paper systematically evaluates the road event understanding capabilities of 18 Video LLMs.

## Background & Motivation

### Background

**Background**: Video Question Answering (VideoQA) is a crucial task for evaluating the comprehension capabilities of video-language models. In the domains of autonomous driving and traffic safety, accurately understanding road events (such as accidents, traffic violations, and weather impacts) is vital for safety systems.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Regional and perspective bias: Existing road event datasets (such as BDD100K, Drive-LM) primarily originate from in-car cameras in specific regions, failing to reflect the variability and diversity of global road conditions. (2) Annotation bias: Existing datasets rely on expert annotation, resulting in limited event types and high annotation costs. (3) Lack of social context: Road event videos on social media contain rich comments and narratives, but existing datasets fail to utilize this contextual information. (4) Limited QA task variety: Most datasets only consider simple event descriptions or causal reasoning, lacking deep understanding tasks such as spatial reasoning, temporal localization, and social impact analysis.

**Key Challenge**: The complexity and diversity of road events require large-scale, highly diverse data for training and evaluating models. However, manual annotation is extremely costly, and existing data sources (such as dashcams) feature single perspectives.

**Goal**: How to construct a large-scale road event VideoQA dataset that covers global locations, multi-view perspectives, and rich social contexts?

**Key Insight**: Sourcing road event videos from social media platforms (such as YouTube) and utilizing Text LLMs and Video LLMs to construct a semi-automatic annotation pipeline to generate high-quality QA pairs at scale.

**Core Idea**: Sourcing videos from social media combined with LLM-based semi-automatic annotation to construct the largest and most diverse road event VideoQA dataset globally.

## Method

### Overall Architecture

The construction pipeline of RoadSocial: (1) Data Collection: Search and filter road event videos from social media platforms using keywords (e.g., accident, road rage, traffic violation) to ensure regional and perspective diversity. (2) Metadata Extraction: Collect social media metadata, including video titles, descriptions, and comments. (3) Semi-automatic QA Generation: Utilize Video LLMs to comprehend video content, and Text LLMs to generate QA pairs covering 12 task categories by integrating social metadata. (4) Quality Control: Perform manual review and filtering to guarantee the accuracy and diversity of QA pairs.

### Key Designs

1. **Semi-automatic Annotation Framework (LLM-Powered Annotation)**:
    - Function: Economically and efficiently generate large-scale, high-quality QA pairs.
    - Mechanism: A two-stage annotation pipeline. Stage 1: Video LLMs (e.g., GPT-4V, VideoChat) process the videos to generate event descriptions, temporal timestamps, and element identification. Stage 2: Text LLMs (e.g., GPT-4) generate structured QA pairs based on video descriptions and social media comments, covering 12 task categories such as event recognition, causal reasoning, spatial relations, and temporal localization. Finally, human verification is conducted to filter out low-quality samples.
    - Design Motivation: Pure manual annotation of 260K QA pairs is prohibitively expensive, whereas fully automated generation yields uncontrollable quality. The two-stage LLM pipeline plus human verification strikes a balance between scale and quality.

2. **Design of 12 Challenging QA Tasks**:
    - Function: Comprehensively evaluate the depth of Video LLMs' understanding of road events.
    - Mechanism: Design 12 QA tasks of varying difficulty and types: event type recognition, event description, causal reasoning, participant identification, spatial relations, temporal localization, traffic rule judgment, weather impact analysis, sentiment analysis (from comments), severity assessment, risk prediction, and social impact analysis. Each task category is associated with corresponding evaluation metrics.
    - Design Motivation: Existing benchmarks only consider 2-3 types of simple QA tasks, which fail to comprehensively evaluate the model's ability to understand road scenarios.

3. **Diversity Guarantee Mechanism**:
    - Function: Ensure diversity in regions, perspectives, and event types within the dataset.
    - Mechanism: (1) Regional diversity: Covers multiple regions globally, including Asia, Europe, North America, and South America, featuring diverse traffic rules and road conditions. (2) Perspective diversity: Includes various perspectives such as fixed CCTV monitoring, handheld shooting, dashcams, and drone aerial photography. (3) Event type balance: Ensures sufficient samples for all types of events, such as accidents, violations, severe weather, and road conflicts.
    - Design Motivation: Models trained on single-source datasets generalize poorly to other scenarios.

## Key Experimental Results

### Key Findings

- Dataset scale: 13.2K videos, 14M frames, 414K social comments, 674 labels, 260K QA pairs.
- Comprehensive evaluation of 18 Video LLMs (including GPT-4V, VideoChat2, Video-LLaVA, etc.).
- All models perform the worst on causal reasoning and temporal localization tasks, indicating that these capabilities remain bottlenecks.
- The inclusion of social media metadata improves event understanding accuracy by approximately 5-10%.
- Closed-source models (GPT-4V) still significantly outpace open-source models, with a gap of approximately 15-20%.
- Evaluations on multi-perspective samples show that models perform worst on the drone perspective.

## Highlights & Insights

- **Innovative Data Source**: This work is the first to systematically utilize social media as a data source for road events, which naturally introduces diversity and social context.
- **Comprehensive Evaluation**: The design of 12 QA task categories covers various dimensions of road event understanding.
- **Replicable Annotation Pipeline**: The semi-automatic framework can be borrowed for constructing VideoQA datasets in other domains.

## Limitations & Future Work

- Social media video quality varies significantly (resolution, stability), which may affect the fairness of model evaluation.
- Automatic annotation by LLMs may still introduce systematic biases.
- Privacy issues: Videos from social media require anonymization/redaction.
- Future work can extend the dataset to multi-language annotations and more social media platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](../../ICCV2025/llm_evaluation/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)
- [\[ICLR 2026\] VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding](../../ICLR2026/llm_evaluation/videojudge_bootstrapping_enables_scalable_supervision_of_mllm-as-a-judge_for_vid.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](../../ACL2025/llm_evaluation/culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](../../ACL2025/llm_evaluation/sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](../../ACL2025/llm_evaluation/belarusian_glue.md)

</div>

<!-- RELATED:END -->
