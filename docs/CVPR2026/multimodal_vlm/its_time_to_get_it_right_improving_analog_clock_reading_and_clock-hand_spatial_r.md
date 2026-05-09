---
title: >-
  [Paper Note] It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Analog clock reading] This paper reveals that state-of-the-art VLMs still fail to reliably read analog clocks in real-world scenes (zero-shot accuracy below 10%), and proposes TickTockVQA, a real-world dataset of 12K images, along with a Swap-DPO fine-tuning framework that improves Llama-3.2-11B's time-reading accuracy from 1.43% to 46.22%.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Analog clock reading
  - spatial reasoning
  - VLM
  - DPO
  - spatiotemporal understanding
date: 2026-05-08
content_hash: 23c1d36d6c037bd0
---

# It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.08011](https://arxiv.org/abs/2603.08011)
**Code**: [https://it-s-time-to-get-it-right.github.io/](https://it-s-time-to-get-it-right.github.io/)
**Area**: Multimodal VLM
**Keywords**: Analog clock reading, spatial reasoning, VLM, DPO, spatiotemporal understanding

## TL;DR
This paper reveals that state-of-the-art VLMs still fail to reliably read analog clocks in real-world scenes (zero-shot accuracy below 10%), and proposes TickTockVQA, a real-world dataset of 12K images, along with a Swap-DPO fine-tuning framework that improves Llama-3.2-11B's time-reading accuracy from 1.43% to 46.22%.

## Background & Motivation
**State of the Field**: VLMs continue to advance on complex multimodal reasoning tasks, yet reading analog clocks remains surprisingly difficult for them.

**Limitations of Prior Work**: (a) Existing clock datasets are predominantly synthetic, with uniform styles and limited background variation that do not reflect real-world complexity; (b) VLMs frequently confuse the hour and minute hands, lacking fine-grained spatial-temporal reasoning capabilities.

**Root Cause**: Clock reading requires jointly localizing hands, recognizing their orientations, interpreting angular configurations, and mapping them to discrete time values — a compact yet spatially demanding task.

**Paper Goals**: (a) Address the lack of high-quality real-world clock datasets; (b) correct the spatial reasoning deficiency in VLMs that leads to hour/minute hand confusion.

**Starting Point**: Replace synthetic training data with real-world scene data, and apply DPO alignment to teach models to distinguish between the hour and minute hands.

**Core Idea**: Swap-DPO — construct preference pairs by swapping the hour and minute hand readings, enabling the model to explicitly learn correct hand-role assignments.

## Method

### Overall Architecture
TickTockVQA provides real-world training data → SFT establishes basic clock-reading capability → Swap-DPO further aligns the model's hand-discrimination ability.

### Key Designs

1. **TickTockVQA Dataset**:

    - **Function**: Provides the first large-scale real-world analog clock VQA dataset.
    - **Mechanism**: Collects 12K+ real clock images from multiple sources including COCO, SBU, VG, ImageNet, OID, CC12M, and movie frames; manually annotates hours, minutes, and inferable AM/PM; removes over-represented canonical times such as 10:10.
    - **Data Diversity**: Covers seven clock types (wall clocks, tower clocks, wristwatches, alarm clocks, post-box clocks, etc.) with indoor/outdoor/inverted/partially-occluded variants, and three dial styles: Arabic numerals, Roman numerals, and unmarked dials.

2. **Swap-DPO**:

    - **Function**: Uses preference learning to teach the model to distinguish between the hour and minute hands.
    - **Mechanism**: Constructs preference pairs where the correct time reading serves as the *chosen* response, and a reading in which the hour and minute hand values are swapped serves as the *rejected* response. This directly targets the most common VLM failure mode — hour/minute confusion.
    - **Design Motivation**: SFT teaches the model to "read" time but cannot reinforce discrimination of the spatial roles of each hand. DPO uses contrastive negative samples to explicitly signal that swapping is incorrect.

3. **Evaluation Design (Swap-Equivalence Evaluation)**:

    - **Function**: Provides a fairer evaluation by accepting both the original and swap-equivalent time interpretations where applicable.
    - **Mechanism**: Accounts for inherently ambiguous readings (e.g., 12:00 vs. 00:00) by considering both interpretations during evaluation.

### Loss & Training
- **SFT stage**: Standard supervised fine-tuning to establish basic clock-reading ability.
- **DPO stage**: Swap-DPO alignment applied on top of the SFT checkpoint.

## Key Experimental Results

### Zero-Shot Baselines (TickTockVQA Test Set)

| Model | Hour Acc. | Minute Acc. | Full Time Acc. | MAE↓ |
|-------|-----------|-------------|----------------|------|
| SpatialVLM-3B | 12.51 | 6.44 | 1.05 | 161.68 |
| Llama-3.2-11B | 11.51 | 8.58 | 1.43 | 156.96 |
| Qwen2.5-VL-7B | 17.65 | 22.44 | 6.04 | 148.62 |
| It's About Time | 28.95 | 25.00 | 18.54 | 135.15 |

### Post-Training Results (Llama-3.2-11B)

| Stage | Hour Acc. | Full Time Acc. | MAE↓ |
|-------|-----------|----------------|------|
| Zero-shot | 11.51 | 1.43 | 156.96 |
| +SFT (TickTockVQA) | Improved | Improved | Substantially reduced |
| +Swap-DPO | **Best** | **46.22** | **Best** |

### Training Data Comparison

| Training Data | Type | Qwen2.5-VL-7B MAE↓ |
|---------------|------|---------------------|
| SynClock | OpenCV synthetic | Worse |
| CtrlClock | Diffusion synthetic | Moderate |
| **TickTockVQA** | **Real-world** | **99.9 (Best)** |

### Key Findings
- Zero-shot clock-reading accuracy is extremely low across all SOTA VLMs; GPT-5, Claude Sonnet 4.5, and Gemini-2.5 Pro also make errors.
- Real-world data substantially outperforms synthetic data for training, confirming that the domain gap is a critical bottleneck.
- Swap-DPO yields substantial gains over SFT alone; Llama-3.2-11B full time accuracy improves from 1.43% to 46.22% (+44.81 pp).
- Hour-hand accuracy is generally higher than minute-hand accuracy, but Swap-DPO yields significant improvements for both.

## Highlights & Insights
- **Counterintuitive finding**: Such an ostensibly "simple" task exposes a major weakness in SOTA VLMs, revealing substantive deficiencies in spatial reasoning.
- **Swap-DPO targets the root cause precisely**: Rather than improving spatial reasoning in general, it constructs negative samples that specifically address the hour/minute confusion failure mode.
- TickTockVQA at 12K images is currently the largest and most diverse real-world clock benchmark.
- Analog clock reading serves as a compact and informative testbed for evaluating fine-grained spatial-temporal reasoning in VLMs.

## Limitations & Future Work
- Although 46.22% accuracy represents a substantial improvement over the baseline, a considerable gap to human-level performance remains.
- Swap-DPO addresses only hour/minute confusion; other error types (e.g., misrecognition of numeral styles) require additional strategies.
- Dataset annotation was performed by the authors rather than a professional annotation team; while cross-validation was conducted, quality assurance at scale warrants attention.
- The ability to read time continuously from video has not been evaluated.

## Related Work & Insights
- **vs. It's About Time**: The prior work relies on synthetic data; this paper demonstrates the necessity of real-world data and contributes the largest real-world clock dataset to date.
- **vs. SpatialVLM**: SpatialVLM targets general spatial understanding, whereas this paper reveals fine-grained gaps in spatial reasoning within a specific task.
- Framing clock reading as a "compact spatial reasoning testbed" is an interesting positioning, analogous to the role of the Winograd Schema in NLU evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ — Swap-DPO is an elegantly targeted design; the task is simple yet the insight is deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-model comparison, training data ablation, and synthetic vs. real-world analysis.
- Writing Quality: ⭐⭐⭐⭐ — Detailed dataset statistics and clear argumentation.
- Value: ⭐⭐⭐⭐ — Exposes a spatial reasoning weakness in VLMs; both the dataset and method have reuse potential.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

<!-- RELATED:END -->
