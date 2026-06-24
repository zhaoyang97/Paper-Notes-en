---
title: >-
  [Paper Note] On the Consistency of Video Large Language Models in Temporal Comprehension
description: >-
  [CVPR 2025][Video Understanding][Video Large Language Models] This work systematically investigates the prediction consistency of Video Large Language Models (Video-LLMs) in temporal comprehension. It reveals that current models exhibit extremely poor consistency (near-random levels) under probes such as query rephrasing, temporal shifting, and self-verification. To address this, the Event Temporal Verification Tuning (VTune) method is proposed…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Large Language Models"
  - "Temporal Comprehension Consistency"
  - "Video Temporal Grounding"
  - "Robustness Evaluation"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: e0ec3c70855809d3
---

# On the Consistency of Video Large Language Models in Temporal Comprehension

**Conference**: CVPR 2025  
**arXiv**: [2411.12951](https://arxiv.org/abs/2411.12951)  
**Code**: [github](https://github.com/minjoong507/Consistency-of-Video-LLM)  
**Area**: Video Understanding / Temporal Grounding  
**Keywords**: Video Large Language Models, Temporal Comprehension Consistency, Video Temporal Grounding, Robustness Evaluation, Instruction Tuning

## TL;DR

This work systematically investigates the prediction consistency of Video Large Language Models (Video-LLMs) in temporal comprehension. It reveals that current models exhibit extremely poor consistency (near-random levels) under probes such as query rephrasing, temporal shifting, and self-verification. To address this, the Event Temporal Verification Tuning (VTune) method is proposed, which explicitly incorporates consistency to significantly improve both grounding and consistency performance.

## Background & Motivation

Video Large Language Models (Video-LLMs) can localize video segments corresponding to language queries (temporal grounding), but the **robustness and trustworthiness** of this temporal comprehension capability have not been fully explored. The core question is: once a model localizes a video segment, are its subsequent responses consistent with this initial localization?

Practical observations indicate that existing Video-LLMs exhibit highly inconsistent performance in self-verification. When asked "whether the event occurred within the time interval you just predicted," the models frequently generate contradictory responses. This inconsistency suggests that the temporal comprehension of these models may not stem from genuine video content understanding but rather from a reliance on language priors.

Although existing "temporal-aware" models (such as TimeChat and VTimeLLM) have improved in grounding metrics, their progress on consistency is limited. Conventional prompting (CoT, description prompting) and instruction tuning can enhance localization performance, but their improvements to consistency remain unstable.

Core Motivation: **A training approach that explicitly considers consistency is required** to keep the model logically consistent across localization, rephrased localization, and self-verification.

## Method

### Overall Architecture

This work consists of two parts: (1) **A consistency evaluation framework**: constructing the Charades-CON and ActivityNet-CON datasets, designing four consistency probes (rephrased grounding, shifted grounding, holistic verification, and compositional verification), and evaluating ten models; (2) **The VTune method**: an extended instruction tuning strategy that reformulates the grounding task as a verification process, explicitly training the model to recognize and correct content and temporal variations.

### Key Designs

**1. Consistency Probes**

- **Function**: Multi-dimensional detection of the robustness of temporal comprehension in models.
- **Mechanism**: Four probes are defined: (a) **Rephrased Grounding (R-Ground)** evaluates consistency in localization (using IoU) after the query is rephrased; (b) **Shifted Grounding (S-Ground)** tests whether the model can update its predictions when video contents are temporally shifted; (c) **Holistic Verification (H-Verify)** prompts the model to verify if a query occurred within its predicted interval (requiring a "Yes" response), while using mismatched queries to test for "No" responses; (d) **Compositional Verification (C-Verify)** verifies whether the model can confirm individual components (subject/action/relationship) within the query.
- **Design Motivation**: A single grounding metric cannot reflect the reliability of comprehension. A model might guess the correct timestamp using language priors without truly understanding the content; consistency probes can expose this "spurious understanding."

**2. Dataset Construction (Charades-CON / ActivityNet-CON)**

- **Function**: Providing structured consistency evaluation data.
- **Mechanism**: Sampling 500 videos each from Charades-STA and ActivityNet-Captions, with GPT-4o-mini used to generate aligned, misaligned, and compositional queries. Aligned queries are generated via synonym substitution, active/passive voice conversion, and word order modification, while misaligned queries introduce subtle modifications to key components. Human evaluation indicates that 92.2% of the query sentences are of "Well-matched" quality.
- **Design Motivation**: High-quality rephrasings and misleading queries are required to objectively measure consistency, instead of relying on the model's own generation capability.

**3. Event Temporal Verification Tuning (VTune)**

- **Function**: Explicitly improving model consistency through verification-based instruction tuning.
- **Mechanism**: Moving beyond simple "timestamp prediction" instruction tuning, VTune trains models on three verification tasks: (a) **Alignment confirmation**: verifying that a rephrased query matches the correct time segment; (b) **Content change identification**: detecting misaligned queries and pinpointing the mismatched content; (c) **Temporal change identification**: detecting temporal shifts and correcting timestamps. Verification training data is constructed based on the Charades-STA and ActivityNet-Captions training sets.
- **Design Motivation**: The token likelihood objective of standard instruction tuning does not directly optimize consistency. VTune forces the model to construct faithful mappings between queries and visual content by explicitly requiring it to handle aligned and misaligned variants.

### Loss & Training

- Standard LLM training objective (next-token prediction) is used.
- VTune augments original grounding training data with verification task data.
- Training data consists of three types: grounding instructions (G), event change verification (E), and temporal change verification (T).
- Effectiveness is validated on two models: Video-LLaMA and TimeChat.

## Key Experimental Results

### Main Results

Consistency evaluation of 10 models (Charades-CON, relative consistency score %):

| Model | Ground | R-Ground | S-Ground | H-Verify | C-Verify |
|------|--------|----------|----------|----------|----------|
| Video-LLaVA | 9.4 | 80.8% | 30.3% | 52.8% | 50.0% |
| TimeChat | 30.5 | 82.1% | 18.5% | 45.9% | 51.2% |
| VTimeLLM | 27.3 | 83.2% | 26.9% | 43.7% | 49.8% |
| GPT-4o | 28.5 | 74.3% | 32.8% | **62.4%** | **71.3%** |
| Gemini 1.5 | **34.6** | **85.7%** | **71.7%** | 65.8% | 70.8% |

### Ablation Study

VTune vs. other methods (TimeChat on Charades-CON):

| Method | Ground | S-Ground | H-Verify | C-Verify |
|------|--------|----------|----------|----------|
| TimeChat (Original) | 30.5 | 5.6 | 14.0 | 15.6 |
| + CoT prompting | 28.7 | 7.1 | 13.5 | 14.4 |
| + Desc prompting | 33.3 | 7.3 | 19.9 | 20.6 |
| + Instruction Tuning | 55.8 | 10.5 | 16.7 | 25.7 |
| + **VTune** | **76.2** | **36.2** | **44.8** | **42.4** |

### Key Findings

1. **Verification consistency of open-source models is close to random level** (around 50%): models can localize but cannot reliably verify their own predictions.
2. **Temporal-aware models fail to significantly improve consistency**: VTimeLLM achieves 7.3% higher grounding than Video-LLaMA, but its verification consistency is only 1.6% higher.
3. **Shifted Grounding exposes model reliance on language priors**: most models are insensitive to video temporal shifts, with predictions remaining largely unchanged.
4. **VTune comprehensively outperforms prompting and standard instruction tuning**: on TimeChat, grounding performance increases from 30.5 to 76.2, and H-Verify increases from 14.0 to 44.8.
5. **Closed-source models (GPT-4o, Gemini) show significantly better consistency than open-source models**, though room for improvement remains.

## Highlights & Insights

1. **Evaluating temporal comprehension from a "consistency" perspective is novel and profound**: it reveals a blind spot in existing metrics—high grounding scores do not equate to genuine comprehension.
2. **The findings from S-Ground experiments are the most impactful**: models are almost entirely insensitive to temporal shifts in video content, strongly suggesting reliance on language priors rather than visual understanding.
3. **The verification-based reformulation of VTune is highly inspiring**: integrating "prediction + verification" into training forces models to establish bidirectional comprehension.

## Limitations & Future Work

1. While VTune significantly improves consistency, the absolute level remains moderate (around 55-60% relative consistency in H-Verify).
2. Evaluation relies on GPT-4o-mini to judge the correctness of verification responses, introducing potential automation bias.
3. VTune has only been validated on two models; its effectiveness on larger-scale models remains unexplored.
4. Direct integration of consistency into the training objective function could be explored, rather than relying solely on data augmentation.

## Related Work & Insights

- **TimeChat / VTimeLLM**: Temporal-aware Video-LLMs, which this paper reveals to lack consistency.
- **Chain-of-Thought prompting**: Demonstrates unstable improvements in video temporal tasks, with effects varying across models.
- **LLM Consistency in NLP**: This work extends consistency analysis to the domain of video temporal comprehension.
- **DETR-style Temporal Grounding Models**: Task-specific models might exhibit more stable consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Evaluating Video-LLM temporal comprehension from a consistency perspective is an original and significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 models, 2 datasets, 4 probes, multiple solution comparisons, ablation studies, and human verification.
- **Writing Quality**: ⭐⭐⭐⭐ — Precise problem definition and in-depth experimental analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Holds significant cautionary value for the Video-LLM research community, and VTune offers a viable direction for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

</div>

<!-- RELATED:END -->
