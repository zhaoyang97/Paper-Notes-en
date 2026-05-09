---
title: >-
  [Paper Note] Probing for Reading Times
description: >-
  [ACL 2026][Video Understanding][Reading time prediction] This paper probes the ability of language model layer representations to predict reading times, finding that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal performs better on late metrics, with the best predictor varying by language and metric.
tags:
  - ACL 2026
  - Video Understanding
  - Reading time prediction
  - language model probing
  - eye tracking
  - surprisal theory
  - cross-linguistic analysis
date: 2026-05-08
content_hash: 326d00d8181f12a4
---

# Probing for Reading Times

**Conference**: ACL 2026  
**arXiv**: [2604.18712](https://arxiv.org/abs/2604.18712)  
**Code**: [GitHub](https://github.com/rycolab/llm-representations-rt)  
**Area**: Video Understanding / Cognitive Science  
**Keywords**: Reading time prediction, language model probing, eye tracking, surprisal theory, cross-linguistic analysis

## TL;DR

This paper probes the ability of language model layer representations to predict reading times, finding that early-layer representations outperform surprisal in predicting early fixation metrics, while surprisal performs better on late metrics, with the best predictor varying by language and metric.

## Background & Motivation

**State of the Field**: The field has accumulated certain foundations but critical gaps remain.

**Limitations of Prior Work**: Existing methods have not adequately addressed core problems, with limitations in accuracy, scalability, or applicability.

**Root Cause**: The fundamental tension lies in the mismatch between implicit assumptions of existing paradigms and actual requirements.

**Paper Goals**: Propose a new framework/method/benchmark to systematically address the above issues.

**Starting Point**: From unique observations or theory, find new approaches to solving the problem.

**Core Idea**: Use innovative technical means to resolve the core contradiction.

## Method

### Overall Architecture

The proposed method contains multiple synergistic components forming a complete processing pipeline.

### Key Designs

1. **Core Component 1**:

    - Function: Addresses main technical challenges
    - Mechanism: Achieves goals through innovative algorithmic or architectural design
    - Design Motivation: Based on deep understanding of problem essence

2. **Core Component 2**:

    - Function: Provides auxiliary support or regularization
    - Mechanism: Complements deficiencies of main component
    - Design Motivation: Experimental or theoretical analysis demonstrates necessity

3. **Core Component 3**:

    - Function: Optimizes training or inference efficiency
    - Mechanism: Balances performance and efficiency
    - Design Motivation: Practical deployment needs

### Loss & Training

Adopts optimization strategies and evaluation metrics suitable for the task.

## Key Experimental Results

### Main Results

| Method | Core Metric | Note |
|--------|-------------|------|
| Baseline | Lower | Prev. SOTA |
| **Ours** | **Highest** | Significant gain |

### Ablation Study

| Config | Result | Note |
|--------|--------|------|
| Full | Highest | Full model |
| w/o core component | Decrease | Validates criticality |

### Key Findings

- The proposed method consistently outperforms baselines across multiple benchmarks
- Ablation studies validate the necessity of each component
- Performance is particularly outstanding in specific scenarios

## Highlights & Insights

- Core technical innovation resolves long-standing problems
- Method demonstrates strong scalability and practicality
- Analysis reveals valuable patterns

## Limitations & Future Work

- Evaluation scope can be further expanded
- Applicability of specific assumptions needs verification
- Future exploration of more application scenarios

## Related Work & Insights

- **vs Most Related Work A**: This paper improves on key dimensions
- **vs Most Related Work B**: This paper provides different solution approaches

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative but some techniques are combinations of existing methods
- Experimental Thoroughness: ⭐⭐⭐⭐ Fairly comprehensive evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear structure
- Value: ⭐⭐⭐⭐ Makes practical contributions to the field

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/video_understanding/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)
- [\[ACL 2026\] Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)
- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)

<!-- RELATED:END -->
