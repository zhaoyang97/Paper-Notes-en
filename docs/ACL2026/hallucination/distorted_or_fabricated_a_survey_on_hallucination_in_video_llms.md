---
title: >-
  [Paper Note] Distorted or Fabricated? A Survey on Hallucination in Video LLMs
description: >-
  [ACL 2026][Hallucination Detection][Paper Note] This paper provides the first systematic classification of hallucinations in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven taxonomy comprising "Dynamic Distortion" (errors in spatiotemporal relations and reference consistency) and "Content Fabrication" (driven by statistical priors and audio-visu
tags:
  - ACL 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: ff43f9f195b014da
---
# Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.12944](https://arxiv.org/abs/2604.12944)  
**Code**: [GitHub](https://github.com/hukcc/Awesome-Video-Hallucination)  
**Area**: Hallucination Detection  
**Keywords**: Video LLM Hallucination, Dynamic Distortion, Content Fabrication, Spatiotemporal Reasoning, Multimodality

## TL;DR
This paper provides the first systematic classification of hallucinations in Video Large Language Models (Vid-LLMs), proposing a mechanism-driven taxonomy comprising "Dynamic Distortion" (errors in spatiotemporal relations and reference consistency) and "Content Fabrication" (driven by statistical priors and audio-visual conflicts), while surveying evaluation benchmarks, mitigation strategies, and root causes.

## Background & Motivation

**Background**: Although Vid-LLMs have progressed in action recognition and temporal reasoning, hallucination—generating plausible but contradictory outputs relative to video content—remains pervasive. While hallucinations in image VLMs are well-studied, the temporal structure, motion dynamics, and audio-visual integration of video introduce further complexities.

**Limitations of Prior Work**: Existing multimodal hallucination surveys (Sahoo et al., Bai et al.) mention video hallucination only briefly and lack structural or causal analysis. Image hallucination taxonomies (objects, attributes, relations) do not transfer directly to video, as video-specific temporal errors (e.g., event ordering, action frequency miscounting) and inconsistent cross-segment references require specialized frameworks.

**Key Challenge**: The root causes of video hallucination differ from those in images—Dynamic Distortion stems from limited temporal representation capabilities, while Content Fabrication arises from insufficient visual grounding. However, most current mitigation strategies are adapted from image hallucinations without specific design for video characteristics.

**Goal**: Establish the first mechanism-driven taxonomy for video hallucination, comprehensively review evaluation benchmarks and mitigation methods, analyze root causes, and identify future directions.

**Key Insight**: Categorization is based on the criterion of "whether visual evidence exists"—Dynamic Distortion (visual evidence exists but spatiotemporal relations are modeled incorrectly) vs. Content Fabrication (no visual evidence, output is driven by priors).

**Core Idea**: A dichotomy of video hallucination—Distorted (distorting existing content) vs. Fabricated (fabricating non-existent content).

## Method

### Overall Architecture
The taxonomy is structured into two levels and four categories:
- **Dynamic Distortion**: (1) Spatiotemporal dynamic errors (event ordering, duration, frequency); (2) Reference inconsistency (character confusion, scene confusion).
- **Content Fabrication**: (3) Context-driven fabrication (object-action co-occurrence priors, scene-event priors); (4) Audio-visual conflicts (audio-dominant action inference, audio-dominant emotion inference).

### Key Designs

**1. Mechanism-Driven Taxonomy: Using "Failure Modes" instead of "Input Attributes" as Classification Axes**

Classifying video hallucinations by input attributes like video length or domain would split structurally identical failures. This paper instead uses "observable failure mechanisms" as the axis to establish an operational diagnostic framework. The primary criterion is "whether the output has corresponding visual evidence," and the secondary criterion is the "error mechanism." It provides a decision flowchart (Figure 3)—if visual evidence exists, spatiotemporal relations and reference consistency are checked; if no visual evidence exists, the system distinguishes between prior-driven and audio-driven causes. This allows for grouping identical failure modes across different settings into actionable diagnostic criteria.

**2. Root Cause Analysis and Future Direction Mapping: Aligning Mitigation Strategies with Causes Rather than Symptoms**

Current mitigation strategies are mostly direct transfers from image hallucinations and do not address video-specific root causes. This paper maps each hallucination type to its fundamental cause: Dynamic Distortion is rooted in limited temporal encoding (lack of fine-grained motion cues), weak long-range memory, and poor temporal localization in long videos. Content Fabrication stems from insufficient visual grounding, where pre-training priors or dominant audio signals override visual evidence. This leads to targeted directions—strengthening temporal representations (e.g., motion-aware visual encoders) for Dynamic Distortion, and enhancing visual grounding (e.g., counterfactual training strategies) for Content Fabrication.

**3. Systematic Survey of Evaluation Benchmarks: Reorganizing Scattered Benchmarks by Hallucination Type to Reveal Gaps**

Existing benchmarks are scattered with inconsistent metrics. This paper reorganizes 15+ benchmarks according to the four hallucination types (spatiotemporal dynamics, reference inconsistency, context fabrication, audio-visual conflict) and labels each with video length, domain, evaluation format, and SOTA performance. This reorganization highlights coverage gaps—for instance, only 3 benchmarks exist for audio-visual conflicts, pointing to a significantly neglected research direction.

### Loss & Training
This is a survey paper and does not involve specific model training.

## Key Experimental Results

### Main Results

| Hallucination Type | Representative Benchmark | SOTA Performance | Description |
| :--- | :--- | :--- | :--- |
| Spatiotemporal Dynamics | VidHalluc (CVPR'25) | GPT-4o: 81.2% | Action sequence/duration |
| Spatiotemporal Dynamics | HAVEN | Valley-Eagle: 61.3% | Frequency miscounting |
| Reference Inconsistency | EGOILLUSION (EMNLP'25) | Gemini-Pro: 59.4% | Character confusion |
| Reference Inconsistency | ELV-Halluc | Gemini2.5-Flash: 53.1% | Long video scene confusion |
| Context Fabrication | FactVC (EMNLP'23) | - | Object-action co-occurrence priors |
| Audio-visual Conflict | - | - | Type with the fewest benchmarks |

### Ablation Study
As this is a survey, no ablation studies were performed.

### Key Findings
- Spatiotemporal dynamic errors are prevalent in short videos and worsen in long videos (reference inconsistency and long-range memory failure).
- The root cause of Content Fabrication is excessively strong statistical priors from the pre-training phase—models generate outputs based on co-occurrence statistics even when visual input does not support them.
- Audio-visual conflict is the most neglected type, with minimal benchmarks and mitigation strategies.
- SOTA models (e.g., GPT-4o) achieve only ~80% on the best benchmarks, indicating that video hallucination is far from resolved.

## Highlights & Insights
- The **Distorted vs. Fabricated** dichotomy is concise and powerful—it maps directly to two fundamentally different failure modes: "evidence exists but reasoning is flawed" and "no evidence exists but the prior takes over."
- The survey structure is logically complete, moving from taxonomy → evaluation → mitigation → root causes → future directions.
- It identifies audio-visual conflict as a critical future direction; as multimodal models integrate more modalities, resolving cross-modal conflicts will become increasingly vital.

## Limitations & Future Work
- The survey focuses on "detection and classification," with less in-depth mechanistic analysis of "why Transformers are weak at temporal encoding."
- Lack of quantitative comparison between different mitigation strategies.
- The operability of the taxonomy requires validation through actual annotation experiments.
- Limited literature on audio-visual conflicts restricts the depth of discussion in that section.

## Related Work & Insights
- **vs. Image VLM Hallucination Surveys**: Image hallucinations focus on object/attribute/relation errors, whereas this paper focuses on video-specific temporal and cross-modal errors.
- **vs. MLLM Hallucination Surveys (Sahoo et al.)**: They only briefly mention video; this paper provides in-depth classification and root cause analysis.
- **vs. Specific Benchmark Papers**: This paper unifies scattered benchmarks into a single taxonomic framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First specialized survey on video hallucination with a clear taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ Survey paper with no experiments, but comprehensive benchmark coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with a practical diagnostic checklist design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[AAAI 2026\] Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs](../../AAAI2026/hallucination/does_less_hallucination_mean_less_creativity_an_empirical_investigation_in_llms.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/hallucination/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)

</div>

<!-- RELATED:END -->
