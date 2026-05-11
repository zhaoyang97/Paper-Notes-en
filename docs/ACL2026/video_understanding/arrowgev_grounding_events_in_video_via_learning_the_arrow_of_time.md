---
title: >-
  [Paper Note] ArrowGEV: Grounding Events in Video via Learning the Arrow of Time
description: >-
  [ACL 2026][Video Understanding][Video event grounding] This paper proposes ArrowGEV, a reinforcement learning framework inspired by the physics concept of "arrow of time…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video event grounding"
  - "temporal directionality"
  - "reinforcement learning"
  - "vision-language models"
  - "temporal understanding"
date: 2026-05-08
content_hash: fce11dfb4392778c
---

# ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

**Conference**: ACL 2026  
**arXiv**: [2601.06559](https://arxiv.org/abs/2601.06559)  
**Code**: [Available](https://arxiv.org/abs/2601.06559) (Code / Model / Data all public)  
**Area**: Video Understanding  
**Keywords**: Video event grounding, temporal directionality, reinforcement learning, vision-language models, temporal understanding

## TL;DR

This paper proposes ArrowGEV, a reinforcement learning framework inspired by the physics concept of "arrow of time," which models temporal directionality in videos by distinguishing between temporally sensitive and insensitive events, improving VLM event localization accuracy and temporal understanding capabilities.

## Background & Motivation

**State of the Field**: Grounding Events in Video (GEV) is a fundamental task in video analysis. In recent years, VLMs have become the mainstream approach due to their end-to-end reasoning capabilities, achieving event localization through large-scale timestamp annotation training, temporal token embedding, or video segmentation adaptation.

**Limitations of Prior Work**: Existing methods only align events with timestamps on forward videos, ignoring the intrinsic temporal structure and directionality of events. Experiments show that VLMs cannot distinguish semantic changes of events in forward vs. reversed videos—for example, "picking up a cup" becomes "putting down a cup" when reversed, but models still incorrectly localize the original event in reversed videos.

**Root Cause**: VLMs overfit to text timestamps rather than video semantics, lacking understanding of event temporal directionality, resulting in poor generalization on tasks requiring temporal reasoning.

**Paper Goals**: Improve VLM event localization accuracy and temporal structure understanding by explicitly modeling temporal directionality.

**Starting Point**: Drawing from the physics concept of "arrow of time," events are classified into temporally sensitive (reversal changes semantics) and temporally insensitive (reversal invariant) categories, with differentiated reward signals designed accordingly.

**Core Idea**: Use reversed videos as additional training signals—penalize localization in reversed videos for temporally sensitive events, and enforce forward-reverse consistency for temporally insensitive events.

## Method

### Overall Architecture

Based on the GRPO reinforcement learning framework, taking forward and reversed videos as input, computing differentiated rewards based on event categories. After training, VLMs can not only accurately localize events in forward videos but also understand temporal structure to enhance robustness.

### Key Designs

1. **Event Temporal Directionality Classification**:

    - Function: Classify events into temporally sensitive and insensitive categories
    - Mechanism: Use LLM reasoning to determine event category $c(q) \in \{\text{sensitive}, \text{insensitive}\}$, e.g., "opening a door" is temporally sensitive (reversal becomes "closing door"), while "ball on table" is temporally insensitive
    - Design Motivation: Different event types exhibit different semantic changes under temporal reversal, requiring differentiated handling

2. **Temporal Directionality Reward Modeling**:

    - Function: Unified reward function combining localization accuracy and temporal directionality
    - Mechanism: $r_{\text{grounding}} = r_{\text{acc}} + \lambda \cdot r_{\text{temp}}$, where $r_{\text{acc}}$ evaluates forward localization accuracy using tIoU, $r_{\text{temp}}$ rewards forward-reverse consistency ($S_c$) for insensitive events and discrepancy ($1-S_c$) for sensitive events
    - Design Motivation: Jointly optimize localization accuracy and temporal direction understanding in a unified framework

3. **Difficulty-Aware Training Strategy**:

    - Function: Dynamically adjust sample weights and training data distribution
    - Mechanism: Weight adjustment $w_i = \exp((1 - \text{avg\_tIoU})/\tau)$ focuses the model on difficult samples; dynamic curriculum filtering removes mastered samples (worst IoU > $\eta=0.7$) at the end of each epoch
    - Design Motivation: Samples gradually become easier during training, requiring dynamic maintenance of learning signal strength

### Loss & Training

Final reward $r_{\text{final}} = r_{\text{grounding}} + r_{\text{form}}(o)$, where $r_{\text{form}}$ is format reward requiring output template `<think>...</think><answer>$t_s$ to $t_e$</answer>`. Based on Qwen2.5-VL-7B-Instruct, 2 FPS sampling.

## Key Experimental Results

### Main Results

| Method | Charades-STA R1@0.5 | ActivityNet R1@0.5 | TVGBench R1@0.5 |
|------|-------------------|-------------------|-----------------|
| Gemini-2.5-Pro | 25.5 | 31.9 | 25.7 |
| GPT-5 | 18.3 | 33.0 | 18.8 |
| TimeSuite* | 67.1 | - | - |
| ArrowGEV (Ours) | **Significant improvement** | **Significant improvement** | **Significant improvement** |

### TDD Metric (Temporal Directionality Understanding)

Introduces Temporal Directionality Discrepancy (TDD) metric: $\text{TDD}(m) = \frac{R1@m(\text{fwd}) - R1@m(\text{rev})}{R1@m(\text{fwd})}$. For temporally sensitive events, TDD should approach 1 (able to distinguish forward/reverse); for temporally insensitive events, TDD should approach 0 (forward-reverse consistent).

### Key Findings

- ArrowGEV significantly improves localization accuracy across all three GEV benchmarks
- Substantially improves VLM understanding of temporal directionality (TDD metric)
- Also shows improvements on OOD general video understanding and reasoning tasks (TempCompass, MVBench, VideoMME, etc.)
- Temporally sensitive events constitute a significant proportion in common benchmarks, especially Charades-STA

## Highlights & Insights

- Introducing the "arrow of time" concept from physics to video understanding offers a novel and intuitively clear perspective
- Leverages reversed videos as "free" training signals without requiring additional annotations
- Proposes TDD metric, first quantitative evaluation of model understanding of event temporal directionality
- Difficulty-aware training strategy (weight adjustment + curriculum filtering) effectively maintains learning efficiency

## Limitations & Future Work

- Event classification relies on LLM reasoning, potentially introducing classification noise
- Validated only on 7B model; effects on larger models remain to be explored
- 2 FPS video sampling rate may be insufficient to capture rapid events
- Future work could explore more fine-grained temporal directionality modeling

## Related Work & Insights

- GRPO / DeepSeek-R1: Foundation for RL training paradigm
- TimeSuite / ChatVTG: Supervised learning approaches for GEV task
- Self-supervised learning related to temporal directionality (shuffle-and-learn, order prediction)
- Temporal directionality as a fundamental inductive bias for video understanding is a promising direction

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Physics-inspired temporal directionality modeling with unique perspective
- Experimental Thoroughness: ⭐⭐⭐⭐ Three GEV benchmarks + six general benchmarks, thorough ablations
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, convincing pilot study
- Value: ⭐⭐⭐⭐ Reveals VLM deficiencies in temporal directionality understanding, proposes effective solution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[CVPR 2026\] Envisioning the Future, One Step at a Time](../../CVPR2026/video_understanding/envisioning_the_future_one_step_at_a_time.md)
- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](../../ICCV2025/video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](../../CVPR2026/video_understanding/cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
