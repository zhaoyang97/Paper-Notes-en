---
title: >-
  [Paper Note] EgoLife: Towards Egocentric Life Assistant
description: >-
  [CVPR 2025][Video Understanding][Egocentric Vision] Releases the EgoLife dataset (300 hours of egocentric multimodal videos of 6 participants co-living for a week) and the EgoLifeQA benchmark, and proposes the EgoButler system (EgoGPT + EgoRAG) to explore construction pathways for ultra-long context egocentric life assistants.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Egocentric Vision"
  - "Life Assistant"
  - "Long-Context QA"
  - "Multimodal LLM"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 85039e8ba5244df6
---

# EgoLife: Towards Egocentric Life Assistant

**Conference**: CVPR 2025  
**arXiv**: [2503.03803](https://arxiv.org/abs/2503.03803)  
**Code**: [https://egolife-ai.github.io/](https://egolife-ai.github.io/) (project page available)  
**Area**: Video Understanding  
**Keywords**: Egocentric Vision, Life Assistant, Long-Context QA, Multimodal LLM, Retrieval-Augmented Generation

## TL;DR

Releases the EgoLife dataset (300 hours of egocentric multimodal videos of 6 participants co-living for a week) and the EgoLifeQA benchmark, and proposes the EgoButler system (EgoGPT + EgoRAG) to explore construction pathways for ultra-long context egocentric life assistants.

## Background & Motivation

Building an AI life speech assistant capable of understanding long-term behavioral patterns and complex social interactions of users is one of the ultimate goals of egocentric vision. Existing datasets and methods face significant gaps:

1. **Insufficient dataset dimensions**: Epic-Kitchen focuses on kitchen scenes, while Ego4D, despite its large scale (3,670 hours), mainly consists of single-person short clips (averaging 22.8 minutes per clip). Both lack multi-person social interactions and ultra-long-term (day/week-level) behavioral recordings.
2. **Lack of long-context capabilities**: The maximum context scope of evidence in existing benchmarks (e.g., EgoSchema, EgoPlan-Bench) does not exceed 2 hours, failing to evaluate memory retrieval and habit analysis across multiple days or weeks.
3. **Insufficient multimodal integration**: Joint understanding of video, audio, and speech transcriptions is critical in egocentric scenarios, yet few existing models possess the capability to process both vision and audio simultaneously.

## Method

### Overall Architecture

The EgoLife project comprises three major contributions: (1) **The EgoLife Dataset**—multimodal egocentric recordings of 6 participants co-living for a week, capturing over 8 hours per person daily, complemented by 15 third-person cameras and 2 millimeter-wave radars; (2) **The EgoLifeQA Benchmark**—3,000 long-context QAs designed for life assistants across five categories (Entity Log, Event Recall, Habit Insight, Relationship Map, Task Master); (3) **The EgoButler System**—composed of EgoGPT for clip-level understanding and EgoRAG for long-context question answering.

### Key Designs

1. **EgoGPT—Egocentric Audio-Visual Language Model**:
    - **Function**: Multimodal (visual + audio) dense captioning and QA for 30-second video segments.
    - **Mechanism**: Based on LLaVA-OneVision (7B), an audio branch is added—encoding audio using Whisper Large v3 and training an audio projection module. The EgoIT-99K dataset (compiled from 9 classic egocentric video datasets, featuring 99K QA pairs and 43 hours of video) is constructed for fine-tuning. For personalization, additional fine-tuning is performed on Day-1 data of EgoLife, enabling the model to learn participant identities.
    - **Design Motivation**: General-purpose VLMs (e.g., GPT-4o, Gemini) lack specialized understanding of egocentric perspectives and identity recognition capabilities. Fine-tuning on egocentric data coupled with personalized training can bridge this gap.

2. **EgoRAG—Retrieval-Augmented Long-Context QA**:
    - **Function**: Answers questions requiring a time span of days or weeks.
    - **Mechanism**: Constructs a hierarchical memory bank $M = \{(c_i, d_i, t_i)\}_{i=1}^N$, containing clip features $c_i$, textual descriptions $d_i$, and multi-granularity temporal summaries $t_i$ (hour-level, day-level). During QA, relevant time windows are first localized via high-level summaries, and then the top-k clips within those windows are retrieved using a relevance score $s_i = \text{Similarity}(q, c_i) + \lambda \text{Similarity}(q, d_i)$, which are finally fed into the LLM to generate answers.
    - **Design Motivation**: No existing VLM can directly ingest 40+ hours of video. Hierarchical retrieval (day $\rightarrow$ hour $\rightarrow$ clip) achieves reasoning over ultra-long content while maintaining efficiency, mimicking the hierarchical structures of human memory retrieval.

3. **EgoLifeQA Annotation Pipeline**:
    - **Function**: High-quality long-context QA benchmark.
    - **Mechanism**: First, GPT-4o is used to generate approximately 100K candidate questions based on audio-visual captions. Human annotators then filter out less than 1% high-quality questions (requiring evidence from at least 5 minutes prior). This results in 500 meticulously revised QAs per person, totaling 3,000. Five question types are covered: EntityLog (item tracking), EventRecall (event recall), HabitInsight (habit analysis), RelationMap (interpersonal relations), and TaskMaster (task reminders).
    - **Design Motivation**: 67% of the questions require retracing a context span exceeding 2 hours, which is an evaluation dimension entirely unaddressed by existing benchmarks.

### Loss & Training

EgoGPT is fine-tuned using the standard autoregressive language modeling loss. The training process consists of two stages: (1) training the audio projection module on LibriSpeech to align the audio and language spaces; (2) performing final stage fine-tuning with EgoIT-99K on top of LLaVA-OneVision. The personalized version undergoes additional fine-tuning on the EgoLife Day-1 data.

## Key Experimental Results

### Main Results (EgoGPT Performance on Egocentric Benchmarks)

| Model | Params | EgoSchema | EgoPlan | EgoThink |
|------|------|-----------|---------|----------|
| GPT-4o | — | 72.2 | 32.8 | 65.5 |
| Qwen2-VL | 7B | 66.7 | 34.3 | 59.3 |
| LLaVA-OV | 7B | 60.1 | 30.7 | 54.2 |
| **EgoGPT (EgoIT)** | **7B** | **73.2** | **32.4** | **61.7** |
| **EgoGPT (+D1)** | **7B** | **75.4** | **33.4** | **61.4** |

### EgoRAG Ablation Study (QA Accuracy across Different Evidence Context Spans)

| Model | <2h | 2h-6h | 6h-24h | >24h |
|------|-----|-------|--------|------|
| Gemini-1.5-Pro | 27.9 | 14.8 | 25.0 | 18.4 |
| EgoGPT | 28.2 | 29.1 | 26.8 | 25.0 |
| **EgoGPT+EgoRAG** | **27.2** | **35.7** | **38.9** | **35.4** |

### Key Findings

- EgoGPT achieves 75.4 on EgoSchema, outperforming GPT-4o (72.2), which demonstrates the effectiveness of egocentric domain fine-tuning.
- EgoRAG yields substantial improvements on long-context queries: the QA accuracy for questions spanning >24 hours increases from 25.0 to 35.4 (+42%), validating the necessity of the hierarchical retrieval strategy.
- Caption quality remains a critical bottleneck for EgoButler's performance—the QA accuracy using human audio-visual annotations (45.5) significantly exceeds that of EgoGPT-generated captions (36.0), indicating a 26% margin for improvement.
- Joint audio-visual understanding outperforms single modalities (33.1 vs 31.2 for vision-only, and 27.2 for audio-only), though the standalone contribution of audio remains limited.
- Personalization (Day-1 fine-tuning) steadily improves all metrics, but poses overfitting risks (e.g., misidentifying individuals if they change clothes from the blue outfit worn on Day-1).

## Highlights & Insights

- **Pioneering Dataset**: The first week-long, multi-person, multimodal, multi-view egocentric lifestyle dataset that fills the gap in research on ultra-long-term behaviors and social interactions.
- **Practical Value of EgoLifeQA**: Five question types directly address the core demands of a life assistant: object tracking, event recall, habit analysis, relationship mapping, and task reminders.
- **Simple Effectiveness of EgoRAG**: A straightforward hierarchical retrieval mechanism (day $\rightarrow$ hour $\rightarrow$ clip) significantly boosts ultra-long context QA performance without requiring complex reasoning chains.
- **Candid Failure Analysis**: The paper clearly identifies three bottlenecks of EgoGPT—insufficient speech emotion understanding, overfitting in identity recognition, and the lack of error-correcting mechanisms in single-pass retrieval.

## Limitations & Future Work

- EgoLifeQA is currently evaluated only on Jake's 500 questions, leaving the full suite of 3,000 questions unutilized.
- The single-pass retrieval mechanism of EgoRAG lacks multi-step reasoning capabilities, failing to handle complex questions that require reasoning chains.
- Identity recognition heavily relies on Day-1 fine-tuning, making it vulnerable to appearance changes (e.g., changing clothes).
- The dataset is predominantly in Chinese, and its multilingual extensibility remains to be validated.
- The current framework operates offline, leaving a substantial gap toward real-time personal assistants.

## Related Work & Insights

- Ego4D (3,670 hours) laid the foundation for egocentric vision, but EgoLife represents a completely new exploratory direction in terms of multi-person interactions and ultra-long-term dimensions.
- EgoExo4D provides multi-view paired videos, whereas EgoLife emphasizes natural daily scenarios (rather than skilled task execution).
- Migrating RAG techniques from NLP to video understanding, the design of the hierarchical memory bank is inspired by episodic memory hierarchies in human cognition.
- The proposed "Vision-Audio Caption" pipeline (initial annotation $\rightarrow$ GPT merging $\rightarrow$ GPT enrichment $\rightarrow$ human verification) provides a reusable paradigm for large-scale video annotation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering contributions in both dataset and benchmark, bridging the research gap in ultra-long-term egocentric understanding.
- **Experimental Thoroughness**: ⭐⭐⭐ System functions are fully validated, but the evaluation scope remains somewhat limited (evaluated on only 1/6 of the participants' QA data).
- **Writing Quality**: ⭐⭐⭐⭐ Highly complete structure with rich plots and tables, though it demands patience due to the high information density typical of large-project papers.
- **Value**: ⭐⭐⭐⭐⭐ The combined contribution of dataset + benchmark + system will propel the entire research field of egocentric AI assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](lion-fs_fast_slow_video-language_thinker_as_online_video_assistant.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[CVPR 2025\] FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video](frame_floor-aligned_representation_for_avatar_motion_from_egocentric_video.md)

</div>

<!-- RELATED:END -->
