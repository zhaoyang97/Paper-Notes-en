---
title: >-
  [Paper Note] ArrowGEV: Grounding Events in Video via Learning the Arrow of Time
description: >-
  [ACL 2026][Video Understanding][Grounding Event in Video] ArrowGEV is proposed, a reinforcement learning framework inspired by the "arrow of time" in physics…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Grounding Event in Video"
  - "Temporal Directionality"
  - "Reinforcement Learning"
  - "Vision-Language Models"
  - "Temporal Understanding"
date: 2026-05-08
content_hash: 441d07576b2c4076
---

# ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.06559](https://arxiv.org/abs/2601.06559)  
**Code**: [Available](https://arxiv.org/abs/2601.06559) (Code / Model / Data are all public)  
**Area**: Video Understanding  
**Keywords**: Grounding Event in Video, Temporal Directionality, Reinforcement Learning, Vision-Language Models, Temporal Understanding

## TL;DR

ArrowGEV is proposed, a reinforcement learning framework inspired by the "arrow of time" in physics, which models temporal directionality in videos by distinguishing between time-sensitive and time-insensitive events to improve the event grounding accuracy and temporal understanding of VLMs.

## Background & Motivation

**Background**: Grounding Event in Video (GEV) is a fundamental task in video analysis. Recently, VLMs have become the mainstream approach due to their end-to-end reasoning capabilities, achieving event localization through large-scale timestamp annotation training, temporal token embeddings, or video segmentation adaptation.

**Limitations of Prior Work**: Existing methods only align events with timestamps in forward-playing videos, ignoring the intrinsic temporal structure and directionality of events. Experiments show that VLMs fail to distinguish semantic changes in events between forward and reversed videos—for example, "picking up a cup" becomes "putting down a cup" when reversed, yet models still incorrectly locate the original event in the reversed video.

**Key Challenge**: VLMs overfit text timestamps rather than video semantics and lack an understanding of event temporal directionality, leading to insufficient generalization in tasks requiring temporal reasoning.

**Goal**: To improve VLM event grounding accuracy and temporal structure understanding by explicitly modeling temporal directionality.

**Key Insight**: Drawing from the "arrow of time" concept in physics, events are categorized into two types: time-sensitive (reversal changes semantics) and time-insensitive (invariant to reversal), with differentiated reward signals designed for each.

**Core Idea**: Use reversed videos as additional training signals—penalize localization in reversed videos for time-sensitive events, and enforce consistency between forward and reversed videos for time-insensitive events.

## Method

### Overall Architecture

Based on the GRPO reinforcement learning framework, the model takes both forward and reversed videos as input and calculates differentiated rewards according to the event category. After training, the VLM can not only accurately locate events in forward videos but also understand temporal structures to enhance robustness.

### Key Designs

1.  **Event Temporal Directionality Classification**:
    - **Function**: Categorizes events into time-sensitive and time-insensitive types.
    - **Mechanism**: Use LLM reasoning to determine the event category $c(q) \in \{\text{sensitive}, \text{insensitive}\}$. For instance, "opening a door" is time-sensitive (reversing it makes it "closing a door"), while "a ball on the table" is time-insensitive.
    - **Design Motivation**: Different types of events exhibit different semantic changes under time reversal, requiring differentiated processing.

2.  **Temporal Directionality Reward Modeling**:
    - **Function**: A unified reward function combining grounding accuracy and temporal directionality.
    - **Mechanism**: $r_{\text{grounding}} = r_{\text{acc}} + \lambda \cdot r_{\text{temp}}$, where $r_{\text{acc}}$ evaluates forward grounding accuracy using tIoU, and $r_{\text{temp}}$ rewards consistency ($S_c$) for insensitive events and discrepancy ($1-S_c$) for sensitive events.
    - **Design Motivation**: Simultaneously optimize grounding accuracy and temporal direction understanding within a unified framework.

3.  **Difficulty-Aware Training Strategy**:
    - **Function**: Dynamically adjusts sample weights and training data distribution.
    - **Mechanism**: Weight adjustment $w_i = \exp((1 - \text{avg\_tIoU})/\tau)$ makes the model focus on difficult samples; dynamic curriculum filtering removes mastered samples (worst IoU > $\eta=0.7$) at the end of each epoch.
    - **Design Motivation**: Samples gradually become easier during training, necessitating the dynamic maintenance of learning signal strength.

### Loss & Training

The final reward is $r_{\text{final}} = r_{\text{grounding}} + r_{\text{form}}(o)$, where $r_{\text{form}}$ is a format reward requiring the output to follow the `<think>...</think><answer>$t_s$ to $t_e$</answer>` template. Based on Qwen2.5-VL-7B-Instruct, with 2 FPS sampling.

## Key Experimental Results

### Main Results

| Method | Charades-STA R1@0.5 | ActivityNet R1@0.5 | TVGBench R1@0.5 |
| :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | 25.5 | 31.9 | 25.7 |
| GPT-5 | 18.3 | 33.0 | 18.8 |
| TimeSuite* | 67.1 | - | - |
| ArrowGEV (Ours) | **Significant Gain** | **Significant Gain** | **Significant Gain** |

### TDD Indicators (Temporal Directionality Understanding)

The Temporal Directionality Discrepancy (TDD) metric is introduced: $\text{TDD}(m) = \frac{R1@m(\text{fwd}) - R1@m(\text{rev})}{R1@m(\text{fwd})}$. For time-sensitive events, TDD should be close to 1 (able to distinguish forward/backward), while for time-insensitive events, TDD should be close to 0 (consistent across directions).

### Key Findings

- ArrowGEV significantly improves grounding accuracy across three GEV benchmarks.
- It substantially improves VLM understanding of temporal directionality (TDD metric).
- Gains are also observed in OOD general video understanding and reasoning tasks (TempCompass, MVBench, VideoMME, etc.).
- Time-sensitive events account for a significant proportion of common benchmarks, particularly in Charades-STA.

## Highlights & Insights

- The "arrow of time" concept from physics is introduced to video understanding, providing a novel and intuitive perspective.
- Reversed videos are utilized as "free" training signals without requiring additional labels.
- The TDD metric is proposed to quantitatively evaluate the model's understanding of event temporal directionality for the first time.
- The difficulty-aware training strategy (weight adjustment + curriculum filtering) effectively maintains learning efficiency.

## Limitations & Future Work

- Event classification relies on LLM reasoning, which may introduce classification noise.
- Validated only on a 7B model; performance on larger models remains to be explored.
- The video sampling rate of 2 FPS might be insufficient to capture fast-paced events.
- Future work could explore finer-grained temporal directionality modeling.

## Related Work & Insights

- GRPO / DeepSeek-R1: Foundation for the RL training paradigm.
- TimeSuite / ChatVTG: Supervised learning methods for the GEV task.
- Self-supervised learning related to temporal directionality (shuffle-and-learn, order prediction).
- Using temporal directionality as a fundamental inductive bias for video understanding is a promising direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Unique perspective using physics-inspired temporal directionality modeling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three GEV benchmarks + six general benchmarks, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and convincing pilot study.
- **Value**: ⭐⭐⭐⭐ Reveals defects in VLM temporal directionality understanding and proposes an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](../../AAAI2026/video_understanding/learning_time_in_static_classifiers.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[CVPR 2026\] Envisioning the Future, One Step at a Time](../../CVPR2026/video_understanding/envisioning_the_future_one_step_at_a_time.md)
- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](../../ICCV2025/video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)

</div>

<!-- RELATED:END -->
