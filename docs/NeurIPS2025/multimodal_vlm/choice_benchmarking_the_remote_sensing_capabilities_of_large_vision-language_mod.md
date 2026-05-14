---
title: >-
  [Paper Note] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Remote Sensing] This paper proposes CHOICE, a large-scale multi-level VLM benchmark for the remote sensing domain, comprising 10…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Remote Sensing"
  - "VLM Benchmark"
  - "Multi-level Evaluation"
  - "Multiple Choice Questions"
  - "Visual Reasoning"
date: 2026-05-08
content_hash: c508229e29cbf72e
---

# CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2411.18145](https://arxiv.org/abs/2411.18145)
**Code**: [GitHub](https://github.com/xiaoan-whu/CHOICE)
**Area**: Multimodal VLM
**Keywords**: Remote Sensing, VLM Benchmark, Multi-level Evaluation, Multiple Choice Questions, Visual Reasoning

## TL;DR

This paper proposes CHOICE, a large-scale multi-level VLM benchmark for the remote sensing domain, comprising 10,507 newly collected questions spanning 2 top-level dimensions, 6 sub-dimensions, and 23 leaf tasks across perception and reasoning, enabling the first systematic and objective evaluation of VLM remote sensing capabilities.

## Background & Motivation

VLM evaluation in remote sensing faces three core problems:

**Fragmented evaluation scope**: Existing evaluations rely on individual datasets (e.g., UCM-Caption covers only image captioning; DIOR-RSVG covers only visual grounding), lacking a unified multi-dimensional evaluation framework.

**Coarse-grained benchmarks**: While LHRS-Bench and VRSBench offer multi-task evaluation, their dimensions are coarse, sample sizes are small, and pixel-level and multi-temporal tasks are absent.

**Data leakage**: Many benchmarks reuse publicly available datasets such as DOTA and DIOR, which may have been included in VLM training, compromising evaluation objectivity.

CHOICE addresses these issues by collecting novel remote sensing imagery from 50 cities worldwide, adopting a multiple-choice format to eliminate scoring bias, and constructing a hierarchical taxonomy of 23 leaf tasks.

## Method

### Overall Architecture

CHOICE employs a three-level capability taxonomy:

- **L-1 (2 dimensions)**: Perception and Reasoning
- **L-2 (6 sub-dimensions)**: Image-Level Comprehension (ILC), Single-Instance Identification (SII), Cross-Instance Discrimination (CID), Attribute Reasoning (AttR), Assessment Reasoning (AssR), and Commonsense Reasoning (CSR)
- **L-3 (23 leaf tasks)**: Full coverage from scene classification to disaster identification

### Key Designs

**Data coverage**: Images are sourced from 50 randomly selected cities across 6 continents (drawn from the top 1,000 cities in the Oxford Economics GCI), using multi-source satellite data including Landsat-8, Sentinel-1/2, and Google Earth Engine, with spatial resolutions ranging from 0.1 m/pixel to 30 m/pixel.

**Three question construction strategies**:

1. **Label-driven construction**: Predefined labels (e.g., scene categories, seasonal labels) are used to collect corresponding images from Google Earth Engine; distractors are sampled from other class labels.
2. **Foundation model-driven construction**: Visual foundation models extract instance-level attributes (rotated bounding box coordinates, colors, etc.), which are verified manually before constructing fine-grained questions.
3. **Human–machine collaborative construction**: Human annotators write precise descriptions; GPT-4 generates distractor options, which are subsequently verified by human reviewers.

**Evaluation strategy**:

- LLM-based VLMs: directly output option A/B/C/D; accuracy is computed accordingly.
- CLIP-based VLMs: questions and options are converted into declarative statements, and image–text similarity is computed.
- Visual Grounding: IoU > 0.5 is considered correct.

### Quality Control

Each question is formatted as $P_i = [Q_i, C_i, I_i, L_i]$, where $Q_i$ denotes the question, $C_i$ denotes $n$ answer options ($2 \leq n \leq 4$), $I_i$ denotes the remote sensing image, and $L_i$ denotes the ground-truth label. Twelve master's and doctoral students with backgrounds in remote sensing or computer vision were recruited to participate in quality assurance.

## Key Experimental Results

### Main Results

**L-2 dimension evaluation results**:

| Model | ILC | SII | CID | AttR | AssR | CSR |
|------|-----|-----|-----|------|------|-----|
| GPT-4o-2024-11-20 | 0.845 | 0.616 | **0.591** | **0.536** | 0.277 | **0.900** |
| GPT-4o-mini | 0.800 | 0.588 | 0.448 | 0.494 | **0.474** | 0.876 |
| Gemini-1.5-Pro | **0.867** | 0.585 | — | — | — | — |
| Qwen2-VL-70B | — | — | — | — | — | — |
| GeoChat (RSVLM) | — | — | — | — | — | — |

### Ablation Study

**Key comparison dimensions**:

| Analysis Dimension | Finding |
|---------|------|
| General VLMs vs. RSVLMs | RSVLMs outperform on tasks requiring specialized remote sensing knowledge, but underperform on tasks requiring general knowledge integration |
| Open-source vs. closed-source | Qwen2-VL-70B and InternVL2-40B match or surpass GPT-4o on certain tasks |
| ILC vs. SII difficulty | Image-level comprehension accuracy reaches ~80%+, while single-instance fine-grained perception accuracy is significantly lower |
| Perception vs. Reasoning | AssR (Assessment Reasoning) is the weakest dimension across all models; GPT-4o-mini achieves only 0.474 |

### Key Findings

1. **RSVLMs show no consistent advantage**: Remote sensing-specific VLMs perform better on tasks requiring domain knowledge but neglect general knowledge integration, resulting in lower overall performance compared to general VLMs.
2. **Fine-grained perception and reasoning are core challenges**: Reasoning tasks involving complex scenes, social attributes, and remote sensing-specific features remain highly challenging for all VLMs.
3. **Open-source VLMs can substitute closed-source ones**: The latest open-source models demonstrate strong potential on remote sensing tasks, matching or even surpassing GPT-4o.

## Highlights & Insights

- **Newly collected data ensures objectivity**: The complete avoidance of publicly available datasets fundamentally eliminates data leakage concerns.
- **Hierarchical taxonomy with 23 leaf tasks**: Far exceeds the dimensional coverage of existing benchmarks (LHRS-Bench: 11 dimensions; VRSBench: 3 dimensions).
- **Global coverage across 50 cities**: Addresses intra-region inter-class variation in remote sensing imagery.
- **Multiple-choice format**: Eliminates metric inconsistency and subjectivity inherent in free-text evaluation.
- First benchmark to incorporate pixel-level tasks (RES) and multi-temporal analysis (change detection), filling critical gaps in existing benchmarks.

## Limitations & Future Work

- Evaluation focuses on optical remote sensing; multi-modal remote sensing data such as SAR and LiDAR are not covered.
- The multiple-choice format, while objective, limits assessment of models' generative capabilities.
- Some tasks (e.g., Visual Grounding) have relatively small sample sizes (600 questions), which may limit statistical significance.
- The analysis focuses solely on final answer accuracy, without examining models' reasoning processes.

## Related Work & Insights

- **MMBench / MMStar**: General-purpose VLM benchmarks whose hierarchical evaluation design inspired CHOICE's three-level taxonomy.
- **LHRS-Bot / GeoChat**: Remote sensing-specific VLMs; CHOICE reveals their deficiencies in general knowledge integration.
- **CLIP / RemoteCLIP**: CHOICE designs a dedicated CLIP evaluation strategy that converts questions into text–image matching tasks.
- Insight: The next development direction for remote sensing VLMs should be strengthening general reasoning capabilities while preserving domain-specific expertise.

## Rating

- ⭐ Novelty: 4/5 — First remote sensing VLM hierarchical benchmark with newly collected, globally distributed data
- ⭐ Experimental Thoroughness: 5/5 — Large-scale evaluation covering 24 models, 23 tasks, and 10,507 questions
- ⭐ Writing Quality: 4/5 — Clear taxonomy and rich figures and tables
- ⭐ Value: 4/5 — Provides a much-needed standardized evaluation tool for the remote sensing VLM community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[NeurIPS 2025\] ChartMuseum: Testing Chart Visual Reasoning in Large Vision-Language Models](chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)

</div>

<!-- RELATED:END -->
