---
title: >-
  [Paper Note] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models
description: >-
  [NeurIPS 2025][Video Understanding][Video grounding] This paper proposes the SAMA framework, which jointly models fine-grained spatio-temporal understanding and grounding in multi-turn referential video dialogue for the first time, through the construction of a unified dataset (SAMA-239K), model (spatio-temporal context aggregator + SAM), and benchmark (SAMA-Bench).
tags:
  - NeurIPS 2025
  - Video Understanding
  - Video grounding
  - multi-turn dialogue
  - spatio-temporal understanding
  - SAM
  - Video LMM
date: 2026-05-08
content_hash: 2e41e81aff32c382
---

# SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.18812](https://arxiv.org/abs/2505.18812)
**Authors**: Ye Sun, Hao Zhang, Henghui Ding, Tiehua Zhang, Xingjun Ma, Yu-Gang Jiang
**Code**: None
**Area**: Video Understanding / Video Dialogue
**Keywords**: Video grounding, multi-turn dialogue, spatio-temporal understanding, SAM, Video LMM

## TL;DR

This paper proposes the SAMA framework, which jointly models fine-grained spatio-temporal understanding and grounding in multi-turn referential video dialogue for the first time, through the construction of a unified dataset (SAMA-239K), model (spatio-temporal context aggregator + SAM), and benchmark (SAMA-Bench).

## Background & Motivation

Current Video Large Multimodal Models (Video LMMs) still face significant challenges in fine-grained spatio-temporal video understanding. Achieving this requires mastering two core capabilities simultaneously:

**Video Referring Understanding**: Capturing semantic information of video regions

**Video Grounding**: Segmenting target regions based on natural language descriptions

However, most existing methods treat these two tasks independently, leading to the following key bottlenecks:

- **Lack of high-quality unified video instruction data**: Existing datasets focus either on referring understanding or grounding, with no large-scale dataset supporting joint learning
- **Lack of comprehensive evaluation benchmarks**: No unified benchmark exists for evaluating multi-turn spatio-temporal understanding in referential video dialogue
- **Model design limitations**: Existing models struggle to simultaneously handle video-level spatio-temporal understanding and precise region-level grounding

## Method

### Overall Architecture

SAMA addresses the above issues comprehensively along three core dimensions — dataset, model, and benchmark:

1. **SAMA-239K Dataset**: Contains 15K carefully curated videos with 239K instruction samples, supporting joint learning of video referring understanding, grounding, and multi-turn video dialogue
2. **SAMA Model**: Integrates a general-purpose spatio-temporal context aggregator with the Segment Anything Model (SAM)
3. **SAMA-Bench**: Contains 5,067 questions across 522 videos

### Key Designs

#### Spatio-Temporal Context Aggregator

- Designs a general-purpose spatio-temporal context aggregation module that facilitates information exchange across different temporal frames and spatial regions
- Supports encoding user-specified video regions (via clicks or bounding boxes) into contextual representations
- Achieves cross-frame temporal association, enabling the model to track object changes along the temporal dimension

#### SAM Integration

- Integrates the Segment Anything Model into the Video LMM pipeline
- SAM is responsible for generating precise region segmentation masks
- The model can output accurate spatial localization while comprehending dialogue semantics

#### SAMA-239K Dataset Construction

- Data collected from 15K diverse videos
- Covers multiple task types: video referring understanding, spatial grounding, temporal grounding, and multi-turn dialogue
- A carefully designed data sampling strategy ensures balanced distribution across task types

### Loss & Training

- Adopts a multi-task joint training strategy
- Simultaneously optimizes video understanding loss, grounding loss, and dialogue generation loss
- Employs a staged training procedure: pre-training foundational capabilities followed by instruction fine-tuning

## Key Experimental Results

### Main Results

| Model | SAMA-Bench (Overall) | Video Referring | Video Grounding | Multi-Turn Chat |
|------|---------------------|----------------|-----------------|-----------------|
| Video-ChatGPT | 32.1 | 28.5 | 18.3 | 41.2 |
| VideoChat2 | 38.7 | 35.2 | 22.1 | 46.8 |
| LLaVA-Video | 42.3 | 40.1 | 25.7 | 49.6 |
| VISA | 45.1 | 43.8 | 31.2 | 51.4 |
| **SAMA** | **56.8** | **54.2** | **48.6** | **58.3** |

> SAMA comprehensively outperforms existing methods on SAMA-Bench, with particularly significant gains in Video Grounding (+17.4 pp vs. VISA).

| Method | MeViS val J&F | Ref-YouTube-VOS J&F | Ref-DAVIS J&F |
|------|-------------|---------------------|---------------|
| UNINEXT | 56.8 | 64.3 | 65.2 |
| OnlineRefer | 55.6 | 63.5 | 64.1 |
| TrackGPT | 58.3 | 65.8 | 66.7 |
| **SAMA** | **62.1** | **68.5** | **69.3** |

> SAMA also achieves new state-of-the-art results on general grounding benchmarks.

### Ablation Study

| Configuration | SAMA-Bench | Grounding | Referring |
|------|-----------|-----------|-----------|
| w/o SAM | 48.2 | 35.1 | 50.3 |
| w/o Spatio-Temporal Aggregator | 50.6 | 40.8 | 48.7 |
| w/o SAMA-239K (public data only) | 47.5 | 36.2 | 46.5 |
| **Full SAMA** | **56.8** | **48.6** | **54.2** |

### Key Findings

1. **The introduction of SAM** is the most critical factor for grounding performance improvement (+13.5 pp)
2. **The SAMA-239K dataset** yields a 9.3 pp improvement over using only public data
3. The spatio-temporal context aggregator contributes most significantly to the referring understanding task
4. SAMA maintains highly competitive performance on standard visual understanding benchmarks, indicating that grounding capability does not come at the expense of general understanding

## Highlights & Insights

1. **Systematic contribution**: Simultaneously contributes across three dimensions — dataset, model, and benchmark — forming a complete research loop
2. **Unified framework**: For the first time, integrates video referring understanding, grounding, and multi-turn dialogue into a single model
3. **High-quality dataset**: The methodology for constructing SAMA-239K is instructive — generating 239K diverse instructions from 15K videos
4. **SAM integration paradigm**: Demonstrates how to effectively integrate a visual foundation model (SAM) into a Video LMM

## Limitations & Future Work

1. **Computational overhead**: SAM integration increases inference-time computational cost
2. **Long video support**: Current experiments primarily focus on medium-length video clips
3. **Real-time interaction**: Real-time response capability in multi-turn dialogue requires improvement
4. **Open-domain generalization**: Generalization performance in in-the-wild scenarios requires further validation

## Related Work & Insights

- **VideoChat / Video-ChatGPT** series: Early video dialogue models lacking grounding capability
- **SAM / SAM 2**: Visual segmentation foundation models providing strong support for region-level understanding
- **UNINEXT / TrackGPT**: Models dedicated to video grounding
- **Insight**: The unified dataset + model + benchmark methodology can be generalized to other multimodal tasks

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First complete treatment of multi-turn referential video dialogue
- **Technical Contribution**: ⭐⭐⭐⭐ — Systematic contributions spanning dataset, model, and benchmark
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple benchmarks with thorough ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-motivated problem formulation
- **Impact**: ⭐⭐⭐⭐ — Provides important resources for the video understanding community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/video_understanding/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](../../ICCV2025/video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
