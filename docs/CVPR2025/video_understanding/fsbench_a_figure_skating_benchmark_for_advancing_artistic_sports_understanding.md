---
title: >-
  [Paper Note] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding
description: >-
  [CVPR 2025][Video Understanding][Figure Skating] This work proposes FSAnno/FSBench, the first fine-grained, multimodal, and multi-level benchmark dataset for figure skating. It covers a comprehensive chain of tasks from prior knowledge testing and individual action recognition/evaluation/commentary to overall performance evaluation/commentary, revealing significant deficiencies in existing MLLMs regarding artistic sports understanding.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Figure Skating"
  - "Sports Understanding Benchmark"
  - "Artistic Sports"
  - "Multimodal Large Language Models"
  - "Action Quality Assessment"
date: 2026-05-08
content_hash: 3b0b51b1b34df0b5
---

# FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding

**Conference**: CVPR 2025  
**arXiv**: [2504.19514](https://arxiv.org/abs/2504.19514)  
**Code**: None (dataset available)  
**Area**: Video Understanding  
**Keywords**: Figure Skating, Sports Understanding Benchmark, Artistic Sports, Multimodal Large Language Models, Action Quality Assessment

## TL;DR

This work proposes FSAnno/FSBench, the first fine-grained, multimodal, and multi-level benchmark dataset for figure skating. It covers a comprehensive chain of tasks from prior knowledge testing and individual action recognition/evaluation/commentary to overall performance evaluation/commentary, revealing significant deficiencies in existing MLLMs regarding artistic sports understanding.

## Background & Motivation

Figure skating is celebrated as an "art on ice," where evaluations depend not only on technical difficulty (jumps, spins) but also on artistic expression (rhythm, flow, emotional expression). Existing datasets and studies exhibit three key limitations:

1. **Single-task Focus**: Existing datasets (such as FSD-10 for action recognition, FisV for scoring, and MCFS for temporal segmentation) operate in isolation and fail to link individual actions with overall performance.
2. **Neglect of Artistry**: Mainstream sports understanding research focuses on tactical and strategic analysis of ball games (e.g., SportQA, SPORTU), almost entirely ignoring the unique evaluation dimensions of artistic sports like figure skating—such as emotional expression, musical interpretation, and movement fluidity.
3. **Single Modality**: For long videos, MLLMs struggle to focus on local details, while for short videos, they struggle with long sequences. Skeleton data alone lacks shape and appearance information, which is insufficient to depict subtle movements like spins and transitions.

## Method

### Overall Architecture

The contributions of this work are divided into three parts:
- **FSAnno**: A large-scale training and validation dataset containing multimodal and multi-granularity annotations for 783 complete figure skating performances.
- **FSBench**: An independent evaluation benchmark, divided into FSBench-Text (multiple-choice questions + explanations) and FSBench-Motion (multimodal QA pairs).
- **SkateLLM**: A domain-fine-tuned model based on MotionGPT, used to validate the training utility of the dataset.

### Key Designs

1. **Multi-level Task System**:
    - Function: Evaluates the system's depth of understanding of figure skating from simple to complex, and from local to global.
    - Mechanism: Divided into three levels—Prior Knowledge Testing (4,200+ multiple-choice questions on rules and event information), Individual Action Level (action recognition, single-action evaluation, single-action commentary), and Full Performance Level (temporal segmentation, performance evaluation, performance commentary).
    - Design Motivation: Simulates the evaluation process of real judges—first identifying and scoring each technical element, and then comprehensively evaluating the overall performance.

2. **Multimodal Data Acquisition and Privacy Protection**:
    - Function: Provides identity-agnostic multimodal data to ensure fair evaluation.
    - Mechanism: Extracts 3D motion data using 4DHumans and estimates skeleton data using HRNet from raw competition videos, resulting in de-identified motion representations. Meanwhile, raw RGB video links are preserved for community use.
    - Design Motivation: MLLMs might "cheat" by recognizing the athletes' appearance or the competition venue. Using motion and skeleton modalities enables a fairer evaluation of how well models understand the actions themselves.

3. **Multi-source Annotation Strategy**:
    - Function: Provides objective, multi-dimensional, and fine-grained annotations.
    - Mechanism: Annotations are derived from three authoritative sources: official judging cards (GOE scores, multi-dimensional TES/PCS scores), video content (temporal segmentation, action categories), and audio commentary (transcribed with Whisper and aligned with actions via timestamps). For overall performance evaluation, an LLM is used to integrate scattered commentator remarks.
    - Design Motivation: Leverages judging cards to ensure annotation accuracy, utilizes commentator remarks for artistic descriptions, and performs multi-source fusion to achieve comprehensive coverage of both technical and artistic aspects.

### Loss & Training

SkateLLM undergoes instruction fine-tuning based on MotionGPT:
- Uses general motion understanding datasets (e.g., HumanML3D) combined with custom figure skating description data.
- Employs GPT-4 to generate figure skating descriptions based on templates, adding positive artistic evaluations for high GOEs and constructive criticism for low GOEs.
- Two-stage training: Vision-text alignment pre-training followed by LoRA instruction fine-tuning to prevent catastrophic forgetting.
- Employs cross-entropy loss for token prediction.

## Key Experimental Results

### LLM Prior Knowledge Test

| Model | Event Info Acc. | Rules Acc. | Trivia Acc. |
|------|-------------|---------|-------------|
| GPT-4 | **73.0%** | **78.0%** | **87.9%** |
| GPT-3.5-turbo | 59.0% | 64.8% | 72.7% |
| LLaVA 13B | 47.0% | 51.4% | 63.6% |

### Action Description Generation (AutoDQ Metrics)

| Method | F1 (↑) | Recall (↑) | Precision (↑) |
|------|--------|-----------|-------------|
| SkateLLM | **38.0** | **58.3** | **61.6** |
| Motion-GPT | 7.1 | 3.8 | 27.5 |

### Key Findings

1. **Existing MLLMs struggle significantly with figure skating understanding**: Even GPT-4 only achieves 78% (comparable to "All-Star" level) on the rules test, whereas open-source models perform poorly.
2. **Direct application of Motion-GPT to figure skating yields poor results** (F1 score of only 7.1), but fine-tuning on FSAnno boosts SkateLLM's F1 to 38.0, demonstrating the critical importance of domain-specific data for professional sports understanding.
3. SkateLLM's precision is substantially higher than its recall because differences in figure skating action categories are highly subtle (e.g., varying number of spin revolutions). While the model recognizes macro-categories, it struggles to distinguish fine-grained subcategories.

## Highlights & Insights

- **Fills a critical gap**: This is the first figure skating understanding benchmark to concurrently emphasize both technical proficiency and artistry. It is currently the only dataset that includes both action evaluation and commentary annotations (compared in Table 2).
- **Pragmatic task design**: Simulates the multi-role cognitive understanding progression of "Audience → Judge → Commentator."
- **AutoDQ evaluation metric**: More suitable than traditional metrics like BLEU/METEOR for measuring semantic accuracy in technical descriptions, achieved by extracting and matching key events.
- **Importance of negative samples**: Unlike prior datasets from top-tier events lacking error cases, FSAnno spans multiple competition levels and includes a substantial number of negative GOE samples.

## Limitations & Future Work

- Current experiments predominantly focus on motion-based methods; a comprehensive evaluation of video-based MLLMs remains for future work.
- SkateLLM has only been evaluated on basic action description tasks; more complex tasks, such as scoring and full commentary generation, are yet to be thoroughly explored.
- The extraction quality of motion data remains dependent on 4DHumans/HRNet, which may introduce errors in complex action scenarios (e.g., rapid spins, partner/multi-person coordination).
- The dataset is currently limited to singles skating. Understanding team coordination in pairs skating and ice dance represents a more formidable challenge.

## Related Work & Insights

- **Distinction from SportQA/SPORTU**: While existing benchmarks center on tactical understanding of ball sports, FSBench serves as the first benchmark tailored for artistic sports.
- **Evolution over traditional figure skating datasets**: Transitions from single-task evaluation (e.g., FSD-10 for recognition only, FisV for scoring only) to holistic, multi-task, end-to-end evaluation.
- **Implications for future research**: The benchmark construction approach can be extended to other judged sports like gymnastics, ballet, and diving.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic evaluation benchmark for artistic sports, featuring deeply designed tasks.
- Experimental Thoroughness: ⭐⭐⭐ Relatively sparse experiments; only prior knowledge testing and basic action description are demonstrated, with evaluations for more complex tasks currently missing.
- Writing Quality: ⭐⭐⭐ Complete structure but limited text length; extensive details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Pioneers a new pathway for artistic sports understanding, presenting a dataset of long-lasting value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](seriesbench_a_benchmark_for_narrative-driven_drama_series_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[ICLR 2026\] QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning and Proactive Response](../../ICLR2026/video_understanding/querystream_advancing_streaming_video_understanding_with_query-aware_pruning_and.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)

</div>

<!-- RELATED:END -->
