---
title: >-
  [Paper Note] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents
description: >-
  [ACL 2026][Video Understanding][Video Question Answering] GameplayQA is proposed, an end-to-end benchmarking framework based on multiplayer 3D game videos. Through dense timeline annotations (1.22 labels/sec) and a struc…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video Question Answering"
  - "Multi-view Understanding"
  - "Game AI"
  - "Hallucination Diagnosis"
  - "Multi-agent Perception"
date: 2026-05-08
content_hash: 3eb7bbd807855a3d
---

# GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents

**Conference**: ACL 2026  
**arXiv**: [2603.24329](https://arxiv.org/abs/2603.24329)  
**Code**: [Project Page](https://hats-ict.github.io/gameplayqa/)  
**Area**: Video Understanding  
**Keywords**: Video Question Answering, Multi-view Understanding, Game AI, Hallucination Diagnosis, Multi-agent Perception

## TL;DR

GameplayQA is proposed, an end-to-end benchmarking framework based on multiplayer 3D game videos. Through dense timeline annotations (1.22 labels/sec) and a structured distractor taxonomy, the framework systematically evaluates the perception and reasoning capabilities of Multimodal Large Language Models (MLLMs) in decision-dense, multi-view synchronized scenarios, revealing a significant gap between frontier models and human performance.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) are being widely deployed as perception backbones for autonomous agents in 3D environments (e.g., robotics, virtual worlds). This requires models to possess capabilities such as rapid state change perception, action attribution identification, and concurrent multi-agent behavior reasoning.

**Limitations of Prior Work**: Current video understanding benchmarks suffer from three key deficiencies: (1) Lack of embodiment and agent grounding, mostly consisting of slow-paced passive observation videos that fail to test high-frequency state transitions and decision-dense scenarios; (2) Non-diagnosable hallucination types, providing only global performance metrics without fine-grained localization of model failure causes (temporal misjudgment? object fabrication? character confusion?); (3) Lack of multi-video understanding evaluation, focusing almost entirely on a single perspective.

**Key Challenge**: Agent perception necessitates simultaneous tracking of the agent's own state (Self), modeling of other agents' behaviors (Other), and perceiving environmental changes (World). Existing benchmark annotation and evaluation systems cannot cover these multi-layered, multi-perspective cognitive requirements.

**Goal**: Construct an end-to-end benchmarking framework capable of evaluating grounded perception capabilities in decision-dense 3D environments while providing diagnosable error analysis.

**Key Insight**: Utilize multiplayer 3D games as "cognitive sandboxes"—where states and outcomes are highly deterministic and decision-making is fast-paced—making them naturally suitable for evaluating agent perception.

**Core Idea**: Design an annotation system around the Self–Other–World triad, combined with compositional template-based QA generation and a structured distractor taxonomy to achieve multi-layered diagnosable evaluation from basic perception to cross-video reasoning.

## Method

### Overall Architecture

The GameplayQA framework consists of five stages: (1) Collecting synchronized multi-view videos from 9 multiplayer 3D games; (2) Performing dense multi-track timeline annotation across 6 entity types (SA/SS/OA/OS/WO/WE), with an annotation density of 1.22 labels/sec; (3) Generating distractors containing negative labels to induce hallucinations; (4) Generating QA pairs from annotations via a compositional template algorithm, initially producing 400k candidates downsampled to 4k, resulting in 2,365 pairs after quality assurance; (5) Supporting model evaluation and fine-grained hallucination analysis.

### Key Designs

1.  **Self–Other–World Triad Annotation System**:
    - **Function**: Provides a structured annotation framework for perception in 3D multi-agent environments.
    - **Mechanism**: Observably events are categorized along two axes: Entity (Self/Other/World) and Temporal Attributes (Action/State for agents, Object/Event for the environment), forming 6 primitive label types (SA/SS/OA/OS/WO/WE). Each type serves as an independent annotation track, allowing temporal overlaps between tracks to support concurrent event capture.
    - **Design Motivation**: Directly corresponds to the three core requirements in multi-agent reinforcement learning—dense state-action tracking, opponent modeling, and environmental awareness—ensuring that evaluation results have clear diagnostic significance.

2.  **Three-tier Cognitive Complexity Categorization**:
    - **Function**: Progressively evaluates model capabilities from basic perception to complex reasoning.
    - **Mechanism**: L1 (Single-reference Perception) tests basic action/state/object recognition; L2 (Temporal Reasoning) requires cross-entity association, temporal localization, absence identification, sequencing, and intent inference; L3 (Cross-video Understanding) requires reference, ordering, and perspective identification across synchronized multi-views. There are 15 task categories in total.
    - **Design Motivation**: Simulates the progressive complexity of agent cognition—from "what is seen" to "when it happens" and then to "how things seen from different perspectives relate."

3.  **Structured Distractor Taxonomy**:
    - **Function**: Enables diagnosable analysis of model hallucinations.
    - **Mechanism**: Categorizes incorrect options based on their relationship with the correct answer: Lexical distractors (textual variants), Scene distractors (plausible events that did not occur), Temporal distractors (events occurring outside the query window), Actor distractors (agent attribution swaps), and Cross-video distractors (events from other perspectives).
    - **Design Motivation**: Traditional benchmarks only indicate "the model was wrong," whereas a structured distractor taxonomy precisely locates failure modes (temporal localization error vs. character confusion vs. semantic misunderstanding), providing clear directions for model improvement.

### Quality Assurance

A two-stage process is adopted: first, blind filtering is performed via language priors to remove questions answerable without visual understanding; then, human evaluation is conducted on 120 uniformly sampled questions, where approximately 8% of the questions were marked as flawed and removed.

## Key Experimental Results

### Main Results

| Model | Overall | L1 Single-ref | L2 Temporal | L3 Multi-video |
| :--- | :--- | :--- | :--- | :--- |
| Human | 80.5 | ~84% | ~77% | ~89% |
| Gemini 2.5 Pro | 71.3 | ~63% | ~60% | ~77% |
| GPT-5 | 67.0 | ~67% | ~64% | ~62% |
| Gemini 3 Flash | 68.2 | ~64% | ~62% | ~63% |
| Qwen3 VL 235B | 63.8 | ~67% | ~62% | ~49% |
| Claude 4.5 Sonnet | 51.3 | ~62% | ~51% | ~42% |

### Ablation Study

| Configuration | Overall | L1 | L2 | L3 |
| :--- | :--- | :--- | :--- | :--- |
| Full Video (Baseline) | 62.7 | 67.2 | 61.9 | 60.6 |
| No Video | 29.4 | 36.0 | 29.1 | 24.2 |
| Random Frame | 41.7 | 52.9 | 40.9 | 33.7 |
| Shuffled Frames | 54.8 | 63.1 | 52.6 | 53.4 |

### Key Findings
- Model accuracy consistently declines as the cognitive level rises: L1 ($61.2\%$) → L2 ($56.0\%$) → L3 ($49.4\%$), validating the effectiveness of the three-tier classification.
- The two most difficult tasks are Occurrence Counting (OccCnt, $36.5\%$) and Cross-video Ordering (X-VOrd, $38.8\%$), indicating that precise temporal tracking is a fundamental weakness of current models.
- Tasks related to Other agents (OA: $54.0\%$, OS: $55.4\%$) are approximately 8 percentage points harder than World objects (WO: $62.0\%$).
- Cross-video and temporal distractors cause the most errors, while scene distractors are the easiest—models perform better at processing static visual input than temporal and cross-video reasoning.
- Fast-paced shooter games (CS2, Battlefield) have the highest error rates, while slow-paced exploration games are easier.

## Highlights & Insights
- **Highly Diagnosable**: The structured distractor taxonomy is the most significant highlight, transforming "the model is wrong" into "why the model is wrong," providing clear guidance for improvement.
- **Framework over Static Dataset**: This is not just a benchmark but a complete end-to-end pipeline including annotation protocols, QA generation algorithms, and error analysis, extensible to new games and domains.
- **Reasonable Cognitive Hierarchy**: The progressive complexity of L1→L2→L3 effectively differentiates between different capability dimensions, revealing systematic weaknesses in temporal reasoning and multi-view understanding.
- **Multi-view Synchronization**: The first benchmark to provide synchronized multi-POV video QA in the gaming domain, filling the gap in multi-video understanding evaluation.

## Limitations & Future Work
- **Small Data Scale**: With only 2,365 QA pairs and 100 videos, it is limited compared to some large-scale benchmarks.
- **Game Domain Bias**: Data primarily comes from competitive 3D games; generalization to other domains (robotics, autonomous driving) requires validation.
- **Annotation Error Propagation**: Automated annotation generation followed by human verification still leaves approximately $8\%$ quality issues.
- **Future Directions**: Extend to more game genres and non-gaming domains, introduce open-ended QA, and add evaluation of active exploration by models.

## Related Work & Insights
- **vs MarioQA**: Pioneered video QA in the gaming domain but was limited to 2D platformers; GameplayQA extends to 3D multiplayer games and supports multi-view.
- **vs Ego4D/EgoSchema**: Focuses on first-person video understanding but lacks multi-agent and multi-view dimensions.
- **vs MVU-Eval**: Supports multi-video understanding but is not oriented toward agent scenarios and lacks decision density and diagnosability.

## Rating
- Novelty: ⭐⭐⭐⭐ The Self-Other-World triad and structured distractor taxonomy are novel, filling the gap in multi-view game video QA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 15+ frontier models with ablation studies and multi-dimensional error analysis, though data scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear framework design, rich diagrams, and distinct hierarchy.
- Value: ⭐⭐⭐⭐ Provides a practical diagnostic tool for multi-agent perception evaluation, offering inspiration for embodied AI and world model research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4d_bench_benchmarking_multimodal_llms_for_4d_object_understanding.md)
- [\[AAAI 2026\] UVLM: Benchmarking Video Language Model for Underwater World Understanding](../../AAAI2026/video_understanding/uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)
- [\[ACL 2026\] TRACE: Evidence Localization-based Multi-Video Event Understanding and Statement Generation](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)

</div>

<!-- RELATED:END -->
