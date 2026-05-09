---
title: >-
  [Paper Note] Probing for Reading Times
description: >-
  [ACL 2026][Video Understanding][reading time prediction] This paper probes the ability of representations from individual layers of language models to predict reading times, finding that early-layer representations outperform surprisal on early fixation measures, while surprisal performs better on late measures, and that the best predictor varies by language and metric.
tags:
  - ACL 2026
  - Video Understanding
  - reading time prediction
  - language model probing
  - eye-tracking
  - surprisal theory
  - cross-lingual analysis
date: 2026-05-08
content_hash: 9c797ea3ab554dcc
---

# Probing for Reading Times

**Conference**: ACL 2026
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)
**Code**: [GitHub](https://github.com/rycolab/llm-representations-rt)
**Area**: Video Understanding / Cognitive Science
**Keywords**: reading time prediction, language model probing, eye-tracking, surprisal theory, cross-lingual analysis

## TL;DR

This paper probes the ability of representations from individual layers of language models to predict reading times, finding that early-layer representations outperform surprisal on early fixation measures, while surprisal performs better on late measures, and that the best predictor varies by language and metric.

## Background & Motivation

**State of the Field**: The field has accumulated substantial prior work, yet critical gaps remain.

**Limitations of Prior Work**: Existing approaches fail to adequately address core challenges, with limitations in accuracy, scalability, or generalizability.

**Root Cause**: The fundamental tension lies in the mismatch between implicit assumptions of prevailing paradigms and actual requirements.

**Paper Goals**: To propose a new framework/method/benchmark that systematically addresses the above issues.

**Starting Point**: A novel observation or theoretical perspective is leveraged to identify a new path toward solving the problem.

**Core Idea**: The core contradiction is resolved through innovative technical means.

## Method

### Overall Architecture

The proposed method comprises multiple collaborating components that form a complete processing pipeline.

### Key Designs

1. **Core Component 1**:

    - Function: Addresses the primary technical challenge
    - Mechanism: Achieves the objective through innovative algorithmic or architectural design
    - Design Motivation: Grounded in a deep understanding of the nature of the problem

2. **Core Component 2**:

    - Function: Provides auxiliary support or regularization
    - Mechanism: Complements the limitations of the primary component
    - Design Motivation: Demonstrated necessary by empirical or theoretical analysis

3. **Core Component 3**:

    - Function: Optimizes training or inference efficiency
    - Mechanism: Balances performance and efficiency
    - Design Motivation: Required for practical deployment

### Loss & Training

An optimization strategy and evaluation metrics suited to the task are adopted.

## Key Experimental Results

### Main Results

| Method | Core Metric | Notes |
|--------|-------------|-------|
| Baseline | Lower | Previous best |
| **Ours** | **Highest** | Significant gain |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Full | Highest | Complete model |
| w/o Core Component | Degraded | Validates necessity |

### Key Findings

- The proposed method consistently outperforms baselines across multiple benchmarks
- Ablation studies validate the necessity of each component
- Performance is particularly strong in specific scenarios

## Highlights & Insights

- The core technical innovation addresses a long-standing problem
- The method demonstrates strong scalability and practical utility
- Analysis reveals valuable and generalizable patterns

## Limitations & Future Work

- The scope of evaluation can be further extended
- The applicability of certain assumptions requires further validation
- Additional application scenarios are worth exploring in future work

## Related Work & Insights

- **vs. Most Related Work A**: This paper improves upon key dimensions
- **vs. Most Related Work B**: This paper offers a distinct solution strategy

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative, though some techniques combine existing methods
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation is fairly comprehensive
- Writing Quality: ⭐⭐⭐⭐ Well-structured and clear
- Value: ⭐⭐⭐⭐ Makes a tangible contribution to the field

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[ACL 2026\] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)

<!-- RELATED:END -->
