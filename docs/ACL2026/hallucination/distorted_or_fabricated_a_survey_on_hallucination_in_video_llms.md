---
title: >-
  [Paper Note] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
description: >-
  [ACL 2026][Hallucination Detection][Paper Note] This paper presents the first systematic taxonomy of hallucinations in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven classification system of "Dynamic Distortion" (errors in spatio-temporal relations and reference consistency) and "Content Fabrication" (statistical prior-driven and audio-visual c
tags:
  - ACL 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: 94a6bf18a7eec8ce
---
# Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.12944](https://arxiv.org/abs/2604.12944)  
**Code**: [GitHub](https://github.com/hukcc/Awesome-Video-Hallucination)  
**Area**: Hallucination Detection  
**Keywords**: Video LLM Hallucination, Dynamic Distortion, Content Fabrication, Spatio-temporal Reasoning, Multimodality

## TL;DR
This paper presents the first systematic taxonomy of hallucinations in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven classification system of "Dynamic Distortion" (errors in spatio-temporal relations and reference consistency) and "Content Fabrication" (statistical prior-driven and audio-visual conflicts), while surveying evaluation benchmarks, mitigation strategies, and root cause analysis.

## Background & Motivation

**Background**: Video Large Language Models have made progress in tasks such as action recognition and temporal reasoning, but the problem of hallucination—generating outputs that appear plausible but contradict the video content—remains prevalent. While hallucinations in image VLMs have been extensively studied, the temporal structure, motion dynamics, and audio-visual integration of video make the problem more complex.

**Limitations of Prior Work**: Existing multimodal hallucination surveys (Sahoo et al., Bai et al.) only briefly mention video hallucinations and lack structural or causal analysis. Taxonomies for image hallucination (objects, attributes, relations) cannot be directly migrated to video—video-specific temporal errors (e.g., event ordering errors, action frequency miscounting) and cross-segment reference inconsistencies require a dedicated classification framework.

**Key Challenge**: The root causes of video hallucinations differ from those of images—dynamic distortion stems from limited temporal representation capabilities, while content fabrication arises from insufficient visual grounding. However, most existing mitigation strategies are migrated from image hallucinations and are not designed for video characteristics.

**Goal**: Establish the first mechanism-driven taxonomy for video hallucinations, comprehensively review evaluation benchmarks and mitigation methods, analyze root causes, and point out future directions.

**Key Insight**: Categorize based on the criterion "whether visual evidence exists"—Dynamic Distortion (visual evidence is present but spatio-temporal relations are modeled incorrectly) vs. Content Fabrication (no visual evidence, output is driven by priors).

**Core Idea**: A dichotomy of video hallucination—Distorted (warping existing content) vs. Fabricated (inventing non-existent content).

## Method

### Overall Architecture
The classification system is divided into two layers and four categories:
- **Dynamic Distortion**: (1) Spatio-temporal dynamic errors (event ordering, duration, frequency); (2) Reference inconsistency (character confusion, scene confusion)
- **Content Fabrication**: (3) Context-driven fabrication (object-action co-occurrence priors, scene-event priors); (4) Audio-visual conflicts (audio-dominant action inference, audio-dominant emotion inference)

### Key Designs

**1. Mechanism-driven Taxonomy: Using "failure modes" instead of "input attributes" as the classification axis**

If video hallucinations were classified by input attributes such as video length or domain, failures with the same structure would be artificially separated. This paper instead uses "observable failure mechanisms" as the axis to establish an operational diagnostic framework: the primary criterion is "whether the output has corresponding visual evidence," and the secondary criterion is the "error mechanism." It provides a decision checklist (Figure 3)—does the output have corresponding visual evidence? If so, check if spatio-temporal relations are correct or if reference consistency is flawed despite correct spatio-temporal modeling; if there is no visual evidence, further determine if it is prior-driven or audio-driven. Since the same failure mode can appear across different input settings, classifying by failure mechanism groups them together, allowing diagnosis to be based on actionable criteria.

**2. Mapping Root Causes to Future Directions: Aligning mitigation strategies with causes rather than symptoms**

Most existing mitigation strategies are directly migrated from image hallucinations and do not target video-specific root causes. This paper maps each type of hallucination to its fundamental cause: the root cause of dynamic distortion is limited temporal encoding (lack of fine-grained motion cues) coupled with weak long-range memory and poor temporal localization in long videos; the root cause of content fabrication is insufficient visual grounding, allowing pre-training priors or dominant audio signals to override visual evidence. This leads to directions aligned with root causes—enhancing temporal representation for dynamic distortion (e.g., motion-aware visual encoders) and strengthening visual grounding for content fabrication (e.g., counterfactual training strategies) to avoid "treating the symptoms but not the disease."

**3. Systematic Survey of Evaluation Benchmarks: Reorganizing scattered benchmarks by hallucination type to expose coverage gaps**

Existing benchmarks are scattered and use different metrics, making it difficult for researchers to find evaluations that match their focus. This paper reorganizes over 15 benchmarks according to four hallucination types (spatio-temporal dynamics, reference inconsistency, context fabrication, audio-visual conflict) and labels each benchmark with video length, domain, evaluation format, whether it includes specific baselines, and SOTA performance. Through this organization, coverage gaps become clear—for example, there are only 3 benchmarks for audio-visual conflict, directly highlighting a severely neglected research direction.

### Loss & Training
This is a survey paper and does not involve specific model training.

## Key Experimental Results

### Main Results

| Hallucination Type | Representative Benchmark | SOTA Performance | Description |
| :--- | :--- | :--- | :--- |
| Spatio-temporal Dynamics | VidHalluc (CVPR'25) | GPT-4o: 81.2% | Action order/duration |
| Spatio-temporal Dynamics | HAVEN | Valley-Eagle: 61.3% | Frequency miscounting |
| Reference Inconsistency | EGOILLUSION (EMNLP'25) | Gemini-Pro: 59.4% | Character confusion |
| Reference Inconsistency | ELV-Halluc | Gemini2.5-Flash: 53.1% | Scene confusion in long videos |
| Context Fabrication | FactVC (EMNLP'23) | - | Object-action co-occurrence prior |
| Audio-visual Conflict | - | - | Category with the fewest benchmarks |

### Ablation Study
This paper is a survey and does not include ablation studies.

### Key Findings
- Spatio-temporal dynamic errors are already prevalent in short videos, and the problem becomes more severe in long videos (reference inconsistency and long-range memory failures).
- The root cause of content fabrication is the excessive strength of statistical priors during the pre-training phase—models still generate outputs based on co-occurrence statistics even when visual input does not support them.
- Audio-visual conflict is the most neglected type, with very few benchmarks and mitigation strategies.
- SOTA models (e.g., GPT-4o) only achieve ~80% on the best benchmarks, indicating that video hallucination is far from resolved.

## Highlights & Insights
- **The dichotomy of Distorted vs. Fabricated** is simple and powerful—it directly corresponds to two fundamentally different failure modes: "evidence exists but reasoning is wrong" and "no evidence but the prior fills in the gaps."
- The survey structure is clear, with a complete logical chain from taxonomy $\rightarrow$ evaluation $\rightarrow$ mitigation $\rightarrow$ root causes $\rightarrow$ future directions.
- It identifies audio-visual conflict as an important future direction—as multimodal models integrate more modalities, resolving cross-modal conflicts will become increasingly critical.

## Limitations & Future Work
- The survey focuses on "detection and classification" of hallucinations, but the mechanism analysis of "why Transformers are weak at temporal encoding" is not deep enough.
- It lacks a quantitative comparison of different mitigation strategies.
- The operability of the taxonomy needs to be verified through actual annotation experiments.
- There are few publications on audio-visual conflict, leading to limited depth in that discussion.

## Related Work & Insights
- **vs. Image VLM Hallucination Surveys**: Image hallucination focuses on object/attribute/relation errors, whereas this paper focuses on video-specific temporal and cross-modal errors.
- **vs. MLLM Hallucination Surveys (Sahoo et al.)**: They only briefly mention video, while this paper provides in-depth taxonomy and root cause analysis.
- **vs. Specific Benchmark Papers**: This paper unifies scattered benchmarks into a single taxonomic framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated survey on video hallucination with a clear taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ Survey paper with no experiments, but comprehensive benchmark coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Structurally well-organized with a practical decision checklist design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unstitching the Chimera: Frame-Level Risk and Train-Free Mitigation for Video Hallucination](../../CVPR2026/hallucination/unstitching_the_chimera_frame-level_risk_and_train-free_mitigation_for_video_hal.md)
- [\[CVPR 2026\] ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Video Understanding](../../CVPR2026/hallucination/elv-halluc_benchmarking_semantic_aggregation_hallucinations_in_video_understandi.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[CVPR 2026\] SEASON: Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding](../../CVPR2026/hallucination/season_mitigating_temporal_hallucination_in_video_large_language_models_via_self.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)

</div>

<!-- RELATED:END -->
