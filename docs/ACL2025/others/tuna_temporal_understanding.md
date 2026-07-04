---
title: >-
  [Paper Note] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos
description: >-
  [ACL 2025] Tuna constructs a fine-grained, multi-dimensional annotated dataset of 1,000 temporally dense short videos, along with two evaluation tasks: captioning (event splitting → matching → relationship classification) and temporal question answering. This systematically exposes the weaknesses of current video LMMs in dynamic temporal understanding.
tags:
  - "ACL 2025"
date: 2026-05-08
content_hash: abd9a07e5e92463c
---

# Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos

**Conference**: ACL 2025  
**arXiv**: [2505.20124](https://arxiv.org/abs/2505.20124)  
**Code**: [https://friedrichor.github.io/projects/TUNA](https://friedrichor.github.io/projects/TUNA)  
**Authors**: Fanheng Kong, Jingyuan Zhang, Hongzhi Zhang et al. (Northeastern University + Kuaishou)

## TL;DR

Tuna constructs a fine-grained, multi-dimensional annotated dataset of 1,000 temporally dense short videos, along with two evaluation tasks: captioning (event splitting → matching → relationship classification) and temporal question answering. This systematically exposes the weaknesses of current video LMMs in dynamic temporal understanding.

## Background & Motivation

- **Video ≠ Static Image Stack**: The core of video lies in the temporal dimension—camera movement, scene transitions, subject actions, and object attributes evolve dynamically over time. However, existing benchmarks often evaluate these attributes in isolation or focus only on specific aspects (e.g., action only).
- **Long-Video Bias**: Benchmarks like Video-MME and MLVU favor long-video evaluation, coupling temporal understanding with long-context modeling, which makes it difficult to attribute performance bottlenecks.
- **Unreliable Caption Evaluation**: N-gram metrics lack semantic consistency, and direct scoring by LLMs is uninterpretable. Existing event-level methods (such as DREAM-1K) focus only on action events while ignoring the camera and scenes.
- **Core Problem**: The lack of an **all-inclusive, temporal-oriented, and interpretable** evaluation benchmark for short video understanding.

## Method

### Overall Architecture

Tuna consists of two components:

1. **Tuna-1K Dataset**: 1,000 high-quality short videos (averaging 14.5 seconds) annotated by humans with hierarchical temporal descriptions (global caption → event sequence → fine-grained visual elements + types/weights).
2. **Tuna Benchmark**:
    - **Tuna-cap** (Captioning Task): An automated evaluation pipeline that assesses the correctness and completeness of temporally dense captions.
    - **Tuna-mcq** (Multiple-Choice Q&A Task): 1,432 multiple-choice questions where each question requires the entire video context to answer.

### Key Design 1: Multi-Dimensional Visual Element Annotation System

Each event in the video is decomposed into multiple **visual elements**, where each element is annotated with:
- **Type** $t \in \{\text{camera}, \text{scene}, \text{action}, \text{attribute}\}$
- **Weight** $w \in \{1, 2, 3\}$ (importance)

This fine-grained decomposition allows the evaluation to be reported separately by dimensions (camera/scene/action/attribute) and visual features (high-dynamic/low-dynamic/multi-scene/multi-subject), achieving **interpretable diagnostic analysis**. The data spans 10 sources (academic datasets + web videos), covering 12 domains.

### Key Design 2: Tuna-cap Three-Stage Evaluation Pipeline

The caption evaluation consists of three steps:
1. **Event Splitting**: Splits model-generated captions into an event sequence $G = [g_1, ..., g_k]$.
2. **Event Matching**: Matches each candidate event with reference events while enforcing temporal consistency $id_1 \leq id_2 \leq ... \leq id_k$. Invalid events that violate temporal consistency are discarded.
3. **Relationship Classification**: Classifies each visual element in the matched event pairs using GPT-4o into *entailment*, *lack*, or *contradiction*.

The metric computation incorporates element weights $w_{ij}$:
- **Precision**: The weighted proportion of correctly described elements (excluding *lack*).
- **Recall**: The weighted proportion of correctly described elements out of all reference elements.
- Correlation with human judgment (Kendall τ=57.2, Spearman ρ=76.7) significantly outperforms METEOR, BERTScore, etc.

### Key Design 3: Temporally Indispensable MCQ Generation

The Q&A generation workflow consists of:
1. Leveraging LMMs' own "visual misjudgments" as **error-prone points**.
2. Generating multiple-choice questions based on 10 task types (camera movement, transitions, scene description, action recognition, action sequence, etc.).
3. **Temporally Indispensable Filtering**: Excluding questions that can be answered using a single frame, ensuring that multi-frame temporal understanding is mandatory.

## Key Experimental Results

### Captioning Task (Tuna-cap)

| Model | Camera F1 | Scene F1 | Action F1 | Attribute F1 | Overall F1 |
|------|---------|---------|---------|---------|---------|
| GPT-4o | 61.3 | **66.4** | **48.0** | **57.8** | **58.5** |
| MiniCPM-V-2.6 (8B) | **56.0** | 60.6 | 38.8 | 50.2 | 51.7 |
| LLaVA-Video-7B | 50.4 | 58.9 | 37.8 | 53.1 | 51.0 |
| InternVL2-76B | 53.9 | 61.4 | 41.2 | 50.9 | 51.9 |
| Qwen2-VL-72B | 54.0 | 52.8 | 42.6 | 48.5 | 51.7 |

- The state-of-the-art GPT-4o achieves an overall F1 of only 58.5% and a Recall of only 48.2%, indicating that a large number of visual elements are ignored or misdescribed.
- **Weakest on Action Description**: All models perform the worst on the Action dimension, with Tarsier-34B being the sole exception.
- **Multi-Subject Videos are the Most Difficult**: All models exhibit their worst performance in the multi-subject category.

### MCQ Task (Tuna-mcq)

| Model | Camera Motion | Scene Description | Action Sequence | Overall Acc |
|------|----------|----------|----------|----------|
| GPT-4o | 50.4 | **79.6** | **60.5** | **56.2** |
| Qwen2-VL-7B | 41.0 | 66.7 | 52.8 | 51.3 |
| LLaVA-Video-7B | 39.1 | 59.3 | 52.4 | 50.6 |
| InternVL2-8B | 41.0 | 66.7 | 50.5 | 48.4 |

- **Camera Motion** perception is the biggest shortfall (GPT-4o achieves only 50.4%).
- Scene description performance is acceptable, while action sequence understanding still has significant room for improvement.

## Highlights & Insights

1. **All-Inclusive Coverage**: The first video benchmark to simultaneously evaluate four-dimensional temporal dynamics—camera, scene, action, and attribute—filling the gap left by previous works that ignored camera motion and scene transitions.
2. **Interpretable Evaluation**: The event splitting → matching → relationship classification pipeline of Tuna-cap is more reliable than direct LLM scoring, and its correlation with human judgment significantly outperforms traditional metrics.
3. **High Diagnostic Value**: Multi-angle analysis by dimension, visual feature, and complexity provides clear directions for model improvement (e.g., action description, multi-subject understanding).
4. **Short Video Focus**: Averaging 14.5 seconds, it decouples temporal understanding from long-context modeling, making performance attribution highly precise.

## Limitations & Future Work

1. **Evaluation Pipeline Dependency on GPT-4o**: Event splitting, matching, and relationship classification all rely on GPT-4o, leading to high cost and API dependency issues.
2. **Limited Data Scale**: Consisting of 1,000 videos and 1,432 Q&A questions, the dataset size is relatively small and may not cover all video understanding scenarios.
3. **Short Video Limitation**: Averages 14.5 seconds and does not address temporal understanding evaluation in long-video scenarios.
4. **Domain Bias**: Although annotations cover 12 domains, the distribution across these domains is not necessarily balanced.

## Related Work & Insights

- **Video Q&A Benchmarks**: NExT-QA, EgoSchema, MVBench, Video-MME—each has its own focus but lacks all-inclusive temporal evaluation.
- **Video Captioning Benchmarks**: DREAM-1K (action event-level), VDC (multi-dimensional but not temporally oriented).
- **Multi-Task Benchmarks**: TempCompass, E.T.Bench, TemporalBench—evaluate temporal aspects but do not cover camera/scene features.
- **Video LMMs**: LLaVA-Video, Qwen2-VL, InternVL2, etc., serve as the main evaluation targets.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Rating | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)

</div>

<!-- RELATED:END -->
