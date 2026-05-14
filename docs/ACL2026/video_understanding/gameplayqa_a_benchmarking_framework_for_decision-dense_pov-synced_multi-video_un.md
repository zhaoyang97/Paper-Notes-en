---
title: >-
  [Paper Note] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents
description: >-
  [ACL 2026][Video Understanding][Video QA] This paper presents GameplayQA, an end-to-end benchmarking framework built on multi-player 3D game videos. Through dense timeline annotation (1.22 labels/second) and a structured…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video QA"
  - "Multi-View Understanding"
  - "Game AI"
  - "Hallucination Diagnosis"
  - "Multi-Agent Perception"
date: 2026-05-08
content_hash: c365fb04558d2d31
---

# GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents

**Conference**: ACL 2026
**arXiv**: [2603.24329](https://arxiv.org/abs/2603.24329)
**Code**: [Project Page](https://hats-ict.github.io/gameplayqa/)
**Area**: Video Understanding
**Keywords**: Video QA, Multi-View Understanding, Game AI, Hallucination Diagnosis, Multi-Agent Perception

## TL;DR

This paper presents GameplayQA, an end-to-end benchmarking framework built on multi-player 3D game videos. Through dense timeline annotation (1.22 labels/second) and a structured distractor taxonomy, it systematically evaluates multimodal large language models (MLLMs) on perception and reasoning in decision-dense, multi-view synchronized scenarios, revealing a substantial gap between frontier models and human performance.

## Background & Motivation

**State of the Field**: MLLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments (e.g., robotics, virtual worlds), demanding capabilities such as rapid state-change perception, action attribution recognition, and concurrent multi-agent behavior reasoning.

**Limitations of Prior Work**: Existing video understanding benchmarks suffer from three critical shortcomings: (1) they lack embodiment and agent grounding, predominantly featuring slow-paced passive observation videos that fail to test high-frequency state transitions and decision-dense scenarios; (2) they are unable to diagnose hallucination types, offering only global performance metrics without fine-grained localization of model failures (temporal misjudgment? object fabrication? agent confusion?); (3) they lack multi-video understanding evaluation, almost exclusively focusing on single viewpoints.

**Root Cause**: Agent perception requires simultaneously tracking self-state (Self), modeling other agents' behaviors (Other), and perceiving environmental changes (World), yet existing benchmark annotation and evaluation frameworks cannot cover this multi-level, multi-view cognitive demand.

**Paper Goals**: Construct an end-to-end benchmarking framework capable of evaluating foundational perceptual abilities of models in decision-dense 3D environments, while providing diagnosable error analysis.

**Starting Point**: Multi-player 3D games are leveraged as a "cognitive sandbox"—with high determinism in states and outcomes and fast decision rhythms, they are naturally suited for evaluating agent perception.

**Core Idea**: Design an annotation system around the Self–Other–World ternary entity decomposition, combined with a compositional template-based QA generation algorithm and a structured distractor taxonomy, to enable multi-level diagnosable evaluation ranging from basic perception to cross-video reasoning.

## Method

### Overall Architecture

The GameplayQA framework comprises five stages: (1) collecting synchronized multi-view videos from 9 multi-player 3D games; (2) performing dense multi-track timeline annotation across 6 entity types (SA/SS/OA/OS/WO/WE) at a density of 1.22 labels/second; (3) generating distractors with negative labels to elicit hallucinations; (4) producing QA pairs from annotations via a compositional template algorithm, initially yielding 400K candidate pairs, downsampled to 4K, and refined to 2,365 pairs after quality assurance; (5) supporting model evaluation and fine-grained hallucination analysis.

### Key Designs

1. **Self–Other–World Ternary Entity Annotation System**:

    - **Function**: Provides a structured annotation framework for perception in 3D multi-agent environments.
    - **Mechanism**: Observable events are classified along two axes—entity type (Self/Other/World) and temporal attribute (action/state for agents; object/event for the environment)—yielding 6 primitive label types (SA/SS/OA/OS/WO/WE). Each type forms an independent annotation track, and temporal overlap between tracks is permitted to capture concurrent events.
    - **Design Motivation**: Directly corresponds to the three core requirements in multi-agent reinforcement learning—dense state-action tracking, other-agent modeling, and environment perception—endowing evaluation results with clear diagnostic meaning.

2. **Three-Level Cognitive Complexity Question Taxonomy**:

    - **Function**: Progressively evaluates model capability from basic perception to complex reasoning.
    - **Mechanism**: L1 (Single-Reference Perception) tests basic action/state/object recognition; L2 (Temporal Reasoning) requires cross-entity association, temporal localization, absence recognition, ordering, and intent inference; L3 (Cross-Video Understanding) demands referencing, ordering, and viewpoint identification across synchronized multi-view videos. A total of 15 task categories are defined.
    - **Design Motivation**: Mirrors the progressive complexity of agent cognition—from "what was seen" to "when it occurred" to "how observations across different viewpoints relate."

3. **Structured Distractor Taxonomy**:

    - **Function**: Enables diagnosable analysis of model hallucinations.
    - **Mechanism**: Incorrect answer choices are categorized by their relationship to the correct answer into: lexical distractors (textual variants), scene distractors (plausible but non-occurring events), temporal distractors (events occurring outside the query time window), agent distractors (agent attribution swaps), and cross-video distractors (events from other viewpoints).
    - **Design Motivation**: Conventional benchmarks only indicate that "the model answered incorrectly," whereas structured distractors precisely localize failure patterns (temporal localization error vs. agent confusion vs. semantic misunderstanding), providing clear directions for model improvement.

### Quality Assurance

A two-stage process is employed: first, questions answerable without visual understanding are removed via language-prior filtering (blind filtering); then, 120 uniformly sampled questions undergo human evaluation, with approximately 8% flagged as defective and removed.

## Key Experimental Results

### Main Results

| Model | Overall | L1 Single-Ref | L2 Temporal | L3 Cross-Video |
|-------|---------|--------------|-------------|----------------|
| Human | 80.5 | ~84% | ~77% | ~89% |
| Gemini 2.5 Pro | 71.3 | ~63% | ~60% | ~77% |
| GPT-5 | 67.0 | ~67% | ~64% | ~62% |
| Gemini 3 Flash | 68.2 | ~64% | ~62% | ~63% |
| Qwen3 VL 235B | 63.8 | ~67% | ~62% | ~49% |
| Claude 4.5 Sonnet | 51.3 | ~62% | ~51% | ~42% |

### Ablation Study

| Configuration | Overall | L1 | L2 | L3 |
|--------------|---------|-----|-----|-----|
| Full video (baseline) | 62.7 | 67.2 | 61.9 | 60.6 |
| No video | 29.4 | 36.0 | 29.1 | 24.2 |
| Random single frame | 41.7 | 52.9 | 40.9 | 33.7 |
| Shuffled frames | 54.8 | 63.1 | 52.6 | 53.4 |

### Key Findings
- Accuracy consistently declines across all models as cognitive level increases: L1 (61.2%) → L2 (56.0%) → L3 (49.4%), validating the effectiveness of the three-level taxonomy.
- The two most challenging tasks are occurrence counting (OccCnt, 36.5%) and cross-video ordering (X-VOrd, 38.8%), indicating that precise temporal tracking represents a fundamental weakness of current models.
- Other-agent-related categories (OA: 54.0%, OS: 55.4%) are approximately 8 percentage points harder than world-object categories (WO: 62.0%).
- Cross-video and temporal distractors account for the most errors, while scene distractors are the easiest—models handle static visual inputs better than temporal and cross-video reasoning.
- Fast-paced shooter games (CS2, Battlefield) yield the highest error rates, while slow-paced exploration games are comparatively easier.

## Highlights & Insights
- **High Diagnostic Power**: The structured distractor taxonomy is the paper's most significant contribution, transforming "the model answered incorrectly" into "why the model answered incorrectly," providing clear guidance for improvement.
- **Framework Rather Than Static Dataset**: This is not merely a benchmark but a complete end-to-end pipeline encompassing annotation protocols, QA generation algorithms, and error analysis, extensible to new games and new domains.
- **Well-Designed Cognitive Hierarchy**: The progressive complexity of L1→L2→L3 effectively distinguishes different capability dimensions, exposing systematic weaknesses in temporal reasoning and multi-view understanding.
- **Synchronized Multi-View**: This is the first benchmark in the game domain to provide synchronized multi-POV video QA, filling a gap in multi-video understanding evaluation.

## Limitations & Future Work
- **Limited Data Scale**: With only 2,365 QA pairs and 100 videos, the dataset is comparatively small relative to some large-scale benchmarks.
- **Domain Bias Toward Games**: The data predominantly derives from competitive 3D games; generalization to other domains (robotics, autonomous driving) requires further validation.
- **Annotation Error Propagation**: Despite human verification following automated annotation generation, approximately 8% quality issues remain.
- Future directions include expansion to more game genres and non-game domains, incorporation of open-ended QA, and evaluation of models' active exploration capabilities.

## Related Work & Insights
- **vs. MarioQA**: Pioneered game-domain video QA but is limited to 2D platformers; GameplayQA extends to 3D multi-player games with multi-view support.
- **vs. Ego4D/EgoSchema**: Focuses on egocentric video understanding but lacks the multi-agent and multi-view dimensions.
- **vs. MVU-Eval**: Supports multi-video understanding but is not oriented toward agent scenarios, lacking decision density and diagnostic capability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Self–Other–World ternary decomposition and structured distractor taxonomy are novel designs that fill the gap in multi-view game video QA.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 15+ frontier models with ablation studies and multi-dimensional error analysis, though data scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Framework design is clear, figures and tables are rich, and the overall structure is well-organized.
- **Value**: ⭐⭐⭐⭐ Provides a practical diagnostic tool for multi-agent perception evaluation, with implications for embodied AI and world model research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking](../../ICLR2026/video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[AAAI 2026\] UVLM: Benchmarking Video Language Model for Underwater World Understanding](../../AAAI2026/video_understanding/uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)

</div>

<!-- RELATED:END -->
