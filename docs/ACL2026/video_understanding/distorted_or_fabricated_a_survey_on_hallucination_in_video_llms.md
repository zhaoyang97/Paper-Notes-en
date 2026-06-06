---
title: >-
  [Paper Note] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
description: >-
  [ACL 2026][Video Understanding][Video LLM Hallucination] This paper presents the first systematic taxonomy of hallucination phenomena in Video Large Language Models (Vid-LLMs)…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video LLM Hallucination"
  - "Dynamic Distortion"
  - "Content Fabrication"
  - "Spatiotemporal Reasoning"
  - "Multimodal"
date: 2026-05-08
content_hash: 4c42f0388c9bbd1b
---

# Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.12944](https://arxiv.org/abs/2604.12944)  
**Code**: [GitHub](https://github.com/hukcc/Awesome-Video-Hallucination)  
**Area**: Video Understanding / Survey  
**Keywords**: Video LLM Hallucination, Dynamic Distortion, Content Fabrication, Spatiotemporal Reasoning, Multimodal

## TL;DR
This paper presents the first systematic taxonomy of hallucination phenomena in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven classification framework that distinguishes between "dynamic distortion" (spatiotemporal relationship and reference consistency errors) and "content fabrication" (driven by statistical priors and audio-visual conflicts), while surveying evaluation benchmarks, mitigation strategies, and root cause analysis.

## Background & Motivation

**State of the Field**: Video Large Language Models have made progress in tasks such as action recognition and temporal reasoning, but the hallucination problem—generating plausible outputs that contradict video content—remains pervasive. While hallucinations in image VLMs have been extensively studied, the temporal structure, motion dynamics, and audio-visual integration in videos make the problem significantly more complex.

**Limitations of Prior Work**: Existing multimodal hallucination surveys (Sahoo et al., Bai et al.) only briefly mention video hallucinations, lacking structural or causal analysis. Image hallucination taxonomies (objects, attributes, relations) cannot be directly transferred to video—video-specific temporal errors (such as event ordering mistakes and action frequency miscounting) and cross-segment reference inconsistencies require a dedicated classification framework.

**Root Cause**: The root causes of video hallucinations differ from images—dynamic distortions stem from limited temporal representation capacity, while content fabrication arises from insufficient visual grounding—yet existing mitigation strategies are mostly transferred from image hallucination work and not designed for video-specific characteristics.

**Paper Goals**: Establish the first mechanism-driven taxonomy for video hallucinations, comprehensively review evaluation benchmarks and mitigation methods, analyze root causes, and point out future directions.

**Starting Point**: A two-way division based on the criterion "whether visual evidence exists"—dynamic distortion (visual evidence exists but spatiotemporal relationships are incorrectly modeled) vs. content fabrication (no visual evidence, outputs driven by priors).

**Core Idea**: A dichotomy of video hallucinations—Distorted (distorting existing content) vs. Fabricated (fabricating non-existent content).

## Method

### Overall Architecture
The taxonomy has two layers and four categories:
- **Dynamic Distortion**: (1) Spatiotemporal dynamic errors (event ordering, duration, frequency); (2) Reference inconsistency (character confusion, scene confusion)
- **Content Fabrication**: (3) Context-driven fabrication (object-action co-occurrence priors, scene-event priors); (4) Audio-visual conflicts (audio-dominant action inference, audio-dominant emotion inference)

### Key Designs

1. **Mechanism-Driven Taxonomy**:

    - Function: Provides an actionable diagnostic framework for video hallucinations
    - Mechanism: Uses "whether visual evidence exists" as the primary criterion and "error mechanism" as the secondary criterion. Provides a decision checklist (Figure 3): Does the output have corresponding visual evidence? → Yes: Check if spatiotemporal relationships are correct → Spatiotemporally correct but reference consistency error? → No visual evidence: Prior-driven or audio-driven?
    - Design Motivation: Uses observable failure modes rather than input attributes (such as video length, domain) as classification axes—the same failure mode can occur across input settings, and using input attributes would fragment structurally identical failures

2. **Root Cause Analysis and Future Direction Mapping**:

    - Function: Maps hallucination types to root causes, guiding mitigation strategy design
    - Mechanism: The root cause of dynamic distortion is limited temporal encoding (lack of fine-grained motion cues) + weak long-range memory and poor temporal localization in long videos. The root cause of content fabrication is insufficient visual grounding, causing pretraining priors or dominant audio signals to override visual evidence. This leads to future directions including motion-aware visual encoders and counterfactual training strategies
    - Design Motivation: Mitigation strategies should align with root causes—strengthen temporal representation for dynamic distortion, strengthen visual grounding for content fabrication

3. **Systematic Review of Evaluation Benchmarks**:

    - Function: Comprehensive coverage of existing video hallucination benchmarks, categorized by hallucination type
    - Mechanism: Organizes 15+ benchmarks by four hallucination types (spatiotemporal dynamics, reference inconsistency, contextual fabrication, audio-visual conflicts), annotating each benchmark's video length, domain, evaluation format, whether it contains specialized baselines, and SOTA performance. Highlights coverage gaps (e.g., only 3 benchmarks for audio-visual conflicts)
    - Design Motivation: Helps researchers quickly find benchmarks matching their research direction

### Loss & Training
This is a survey paper and does not involve specific model training.

## Key Experimental Results

### Main Results

| Hallucination Type | Representative Benchmark | SOTA Performance | Note |
|---------|---------------|----------|------|
| Spatiotemporal Dynamics | VidHalluc (CVPR'25) | GPT-4o: 81.2% | Action ordering/duration |
| Spatiotemporal Dynamics | HAVEN | Valley-Eagle: 61.3% | Frequency miscounting |
| Reference Inconsistency | EGOILLUSION (EMNLP'25) | Gemini-Pro: 59.4% | Character confusion |
| Reference Inconsistency | ELV-Halluc | Gemini2.5-Flash: 53.1% | Long video scene confusion |
| Contextual Fabrication | FactVC (EMNLP'23) | - | Object-action co-occurrence priors |
| Audio-Visual Conflicts | - | - | Least benchmarked type |

### Ablation Study
This is a survey paper with no ablation experiments.

### Key Findings
- Spatiotemporal dynamic errors are already prevalent in short videos, and the problem is more severe in long videos (reference inconsistency and long-range memory failures)
- The root cause of content fabrication is overly strong statistical priors during pretraining—models generate outputs based on co-occurrence statistics even when visual input does not support them
- Audio-visual conflicts are the most overlooked type, with very few benchmarks and mitigation strategies
- SOTA models (such as GPT-4o) achieve only ~80% on the best benchmarks, indicating video hallucinations are far from solved

## Highlights & Insights
- **The Distorted vs. Fabricated dichotomy** is simple yet powerful—directly corresponding to two fundamentally different failure modes: "evidence exists but reasoning is wrong" and "no evidence but prior fills in"
- The survey structure is clear, with a complete logical chain from taxonomy → evaluation → mitigation → root causes → future directions
- Points out that audio-visual conflicts are an important future direction—as multimodal models integrate more modalities, cross-modal conflict resolution will become increasingly critical

## Limitations & Future Work
- The survey focuses on "detection and classification" of hallucinations, with insufficient depth in mechanistic analysis of "why Transformers are weak in temporal encoding"
- Lacks quantitative comparison of different mitigation strategies
- The operability of the taxonomy awaits validation through actual annotation experiments
- Limited literature on audio-visual conflicts, with limited discussion depth

## Related Work & Insights
- **vs. Image VLM Hallucination Surveys**: Image hallucinations focus on object/attribute/relation errors, this paper focuses on video-specific temporal and cross-modal errors
- **vs. MLLM Hallucination Survey (Sahoo et al.)**: They only briefly mention video, this paper provides in-depth taxonomy and root cause analysis
- **vs. Specific Benchmark Papers**: This paper unifies scattered benchmarks into a single classification framework

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated survey on video hallucinations with clear taxonomy
- Experimental Thoroughness: ⭐⭐⭐ Survey paper without experiments, but comprehensive benchmark coverage
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured hierarchy, practical decision checklist design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](../../AAAI2026/video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)

</div>

<!-- RELATED:END -->
