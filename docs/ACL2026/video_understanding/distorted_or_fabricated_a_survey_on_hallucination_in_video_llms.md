---
title: >-
  [Paper Note] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
description: >-
  [ACL 2026][Video Understanding][Video LLM Hallucination] This paper presents the first systematic classification of hallucination phenomena in Video Large Language Models (Vid-LLMs)…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video LLM Hallucination"
  - "Dynamic Distortion"
  - "Content Fabrication"
  - "Spatio-temporal Reasoning"
  - "Multimodality"
date: 2026-05-08
content_hash: a91c97e555e93827
---

# Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.12944](https://arxiv.org/abs/2604.12944)  
**Code**: [GitHub](https://github.com/hukcc/Awesome-Video-Hallucination)  
**Area**: Video Understanding / Survey  
**Keywords**: Video LLM Hallucination, Dynamic Distortion, Content Fabrication, Spatio-temporal Reasoning, Multimodality

## TL;DR
This paper presents the first systematic classification of hallucination phenomena in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven taxonomy consisting of "Dynamic Distortion" (errors in spatio-temporal relations and reference consistency) and "Content Fabrication" (driven by statistical priors and audio-visual conflicts), while surveying evaluation benchmarks, mitigation strategies, and root cause analyses.

## Background & Motivation

**Background**: Vid-LLMs have made significant progress in tasks such as action recognition and temporal reasoning; however, the hallucination problem—generating plausible but contradictory outputs relative to video content—remains pervasive. While hallucinations in image VLMs have been extensively studied, the temporal structure, motion dynamics, and audio-visual integration of video introduce greater complexity.

**Limitations of Prior Work**: Existing multimodal hallucination surveys (Sahoo et al., Bai et al.) only briefly mention video hallucinations and lack structural or causal analysis. Taxonomies for image hallucinations (objects, attributes, relations) cannot be directly transferred to video—specific temporal errors (e.g., event ordering mistakes, action frequency miscounting) and cross-segment reference inconsistencies require a specialized classification framework.

**Key Challenge**: The root causes of video hallucinations differ from those in images—Dynamic Distortion stems from limited temporal representation capabilities, while Content Fabrication arises from insufficient visual grounding. However, most existing mitigation strategies are transferred from image hallucinations and are not designed for video-specific characteristics.

**Goal**: To establish the first mechanism-driven taxonomy for video hallucinations, comprehensively review evaluation benchmarks and mitigation methods, analyze root causes, and identify future directions.

**Key Insight**: The classification is based on the criterion of "whether visual evidence exists"—partitioning hallucinations into Dynamic Distortion (visual evidence exists but spatio-temporal relations are modeled incorrectly) vs. Content Fabrication (no visual evidence, output is driven by priors).

**Core Idea**: A dichotomy of video hallucinations—Distorted (distorting existing content) vs. Fabricated (fabricating non-existent content).

## Method

### Overall Architecture
The taxonomy is structured into two layers and four categories:
- **Dynamic Distortion**: (1) Spatio-temporal dynamic errors (event ordering, duration, frequency); (2) Reference inconsistency (character confusion, scene confusion).
- **Content Fabrication**: (3) Context-driven fabrication (object-action co-occurrence priors, scene-event priors); (4) Audio-visual conflict (audio-dominated action inference, audio-dominated emotion inference).

### Key Designs

1. **Mechanism-Driven Taxonomy**:
    - Function: Provides an actionable diagnostic framework for video hallucinations.
    - Mechanism: Uses "existence of visual evidence" as the primary criterion and "error mechanism" as the secondary criterion. A decision checklist (Figure 3) is provided: Does the output have corresponding visual evidence? -> Yes: Check if spatio-temporal relations are correct -> Spatio-temporally correct but reference inconsistent? -> No visual evidence: Is it prior-driven or audio-driven?
    - Design Motivation: Uses observable failure modes rather than input attributes (e.g., video length, domain) as classification axes—identical failure modes can appear across input settings, and using input attributes would split structurally identical failures.

2. **Root Cause Analysis and Future Direction Mapping**:
    - Function: Maps hallucination types to root causes to guide the design of mitigation strategies.
    - Mechanism: The root cause of Dynamic Distortion is limited temporal encoding (lack of fine-grained motion cues) + weak long-range memory and poor temporal localization in long videos. The root cause of Content Fabrication is insufficient visual grounding, allowing pre-training priors or dominant audio signals to override visual evidence. Future directions identified include motion-aware visual encoders and counterfactual training strategies.
    - Design Motivation: Mitigation strategies should align with root causes—strengthening temporal representation for Dynamic Distortion and strengthening visual grounding for Content Fabrication.

3. **Systematic Review of Evaluation Benchmarks**:
    - Function: Comprehensively covers existing video hallucination benchmarks, categorized by hallucination type.
    - Mechanism: Organizes 15+ benchmarks into four hallucination types (Spatio-temporal Dynamics, Reference Inconsistency, Contextual Fabrication, Audio-visual Conflict), recording video length, domain, evaluation format, presence of specialized baselines, and SOTA performance for each. It highlights coverage gaps (e.g., only 3 benchmarks for audio-visual conflict).
    - Design Motivation: Assists researchers in quickly locating benchmarks that match their research direction.

### Loss & Training
As this is a survey paper, it does not involve specific model training.

## Key Experimental Results

### Main Results

| Hallucination Type | Representative Benchmark | SOTA Performance | Description |
| :--- | :--- | :--- | :--- |
| Spatio-temporal Dynamics | VidHalluc (CVPR'25) | GPT-4o: 81.2% | Action sequence/duration |
| Spatio-temporal Dynamics | HAVEN | Valley-Eagle: 61.3% | Frequency miscounting |
| Reference Inconsistency | EGOILLUSION (EMNLP'25) | Gemini-Pro: 59.4% | Character confusion |
| Reference Inconsistency | ELV-Halluc | Gemini2.5-Flash: 53.1% | Scene confusion in long videos |
| Contextual Fabrication | FactVC (EMNLP'23) | - | Object-action co-occurrence priors |
| Audio-visual Conflict | - | - | Category with fewest benchmarks |

### Ablation Study
This is a survey; no ablation experiments were performed.

### Key Findings
- Spatio-temporal dynamic errors are prevalent in short videos and become more severe in long videos (reference inconsistency and long-range memory failure).
- The root cause of Content Fabrication is excessively strong statistical priors from the pre-training phase—models generate outputs based on co-occurrence statistics even when visual input does not support them.
- Audio-visual conflict is the most neglected type, with very few benchmarks and mitigation strategies.
- SOTA models (e.g., GPT-4o) achieve only ~80% on the best benchmarks, indicating that video hallucination is far from resolved.

## Highlights & Insights
- The **Distorted vs. Fabricated dichotomy** is concise and powerful—it directly corresponds to two fundamentally different failure modes: "evidence exists but reasoning is wrong" and "no evidence but priors fill the gap."
- The survey structure is clear, with a complete logical chain from classification -> evaluation -> mitigation -> root causes -> future directions.
- It identifies audio-visual conflict as a critical future direction—as multimodal models integrate more modalities, resolving cross-modal conflicts will become increasingly vital.

## Limitations & Future Work
- The survey focuses on the "detection and classification" of hallucinations, with insufficient depth in analyzing the mechanisms of "why Transformers are weak at temporal encoding."
- Lack of quantitative comparison between different mitigation strategies.
- The operability of the taxonomy requires validation through actual annotation experiments.
- Literature on audio-visual conflict is sparse, limiting the depth of discussion in that section.

## Related Work & Insights
- **vs. Image VLM Hallucination Surveys**: While image hallucinations focus on object/attribute/relation errors, this paper focuses on video-specific temporal and cross-modal errors.
- **vs. MLLM Hallucination Surveys (Sahoo et al.)**: They only briefly mention video; this paper provides in-depth classification and root cause analysis.
- **vs. Specific Benchmark Papers**: This paper unifies scattered benchmarks into a single classification framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First specialized survey on video hallucination with a clear taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ Survey paper with no experiments, but comprehensive benchmark coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with a practical decision checklist design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[ACL 2026\] Confidence Estimation for LLMs in Multi-turn Interactions](confidence_estimation_for_llms_in_multi-turn_interactions.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)

</div>

<!-- RELATED:END -->
