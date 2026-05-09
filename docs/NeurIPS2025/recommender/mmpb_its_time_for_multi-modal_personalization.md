---
title: >-
  [Paper Note] MMPB: It's Time for Multi-Modal Personalization
description: >-
  [NeurIPS 2025][Recommender Systems][VLM] This paper introduces MMPB, the first VLM personalization evaluation benchmark, comprising 111 personalizable concepts, 10k+ image-text QA pairs, and 15 task types. Evaluation of 23 VLMs reveals that even the strongest model, GPT-4o, performs poorly on personalization tasks, exposing critical limitations in preference reasoning, visual cue utilization, and conflicts between safety alignment and personalization.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - VLM
  - Personalization
  - benchmark
  - visual question answering
  - Cold-start
date: 2026-05-08
content_hash: dd9e7c7af97092a5
---

# MMPB: It's Time for Multi-Modal Personalization

**Conference**: NeurIPS 2025
**arXiv**: [2509.22820](https://arxiv.org/abs/2509.22820)
**Code**: [https://aidaslab.github.io/MMPB](https://aidaslab.github.io/MMPB) (project page)
**Area**: Recommender Systems
**Keywords**: VLM, Personalization, benchmark, visual question answering, Cold-start

## TL;DR
This paper introduces MMPB, the first VLM personalization evaluation benchmark, comprising 111 personalizable concepts, 10k+ image-text QA pairs, and 15 task types. Evaluation of 23 VLMs reveals that even the strongest model, GPT-4o, performs poorly on personalization tasks, exposing critical limitations in preference reasoning, visual cue utilization, and conflicts between safety alignment and personalization.

## Background & Motivation

**State of the Field**: VLMs (e.g., GPT-4o, LLaVA) are widely used for general visual question answering, but follow a one-size-fits-all paradigm—responding identically to all users without adapting to individual identity, preferences, or history.

**Limitations of Prior Work**: (a) Existing VQA benchmarks focus solely on general knowledge (commonsense, science, etc.) and do not evaluate personalization capabilities; (b) Prior personalization works (e.g., MyVLM, Yo'LLaVA) are small-scale (29–95 concepts), unsystematic, and exclude preference reasoning; (c) A unified evaluation framework and cold-start setting are absent.

**Root Cause**: Strong performance on general tasks does not imply effectiveness in personalized scenarios. Personalization requires models to understand user-specific visual concepts and preferences—capabilities not covered by general-purpose training.

**Paper Goals**: To establish a comprehensive and systematic evaluation benchmark for VLM personalization.

**Starting Point**: Defining four core attributes of personalization (Awareness, Appropriateness, Coherency, Persistency) and designing corresponding task types and evaluation protocols.

**Core Idea**: To reveal the true state and primary bottlenecks of VLM personalization capabilities through systematic benchmarking.

## Method

### Overall Architecture
MMPB evaluation proceeds in three stages:
1. **Concept Injection**: Personalizable concepts are injected into VLMs via reference images or textual descriptions.
2. **Multi-turn Dialogue**: Concept retention is tested through general conversation.
3. **Personalized Query**: The model is tested on new images to assess whether injected concepts can be applied.

### Key Designs

1. **Concept Taxonomy**:

    - 111 concepts across 4 categories: persons, animals, objects, and characters.
    - Each concept includes 5 reference images and 4-level textual descriptions (simple / moderate / detailed / extended).
    - Person-category concepts are additionally equipped with preference information: 5 major domains × 6 sub-domains = 30 preference sub-domains.

2. **Task Types**:

    - **Awareness**: Whether a concept is recognized in positive images, distinguishing single-entity vs. multi-entity settings.
    - **Appropriateness**: Whether irrelevant concepts are correctly suppressed in negative images; animal categories further distinguish same-species vs. different-species.
    - **Coherency**: Whether responses are generated consistently with the concept (4-option MCQ).
    - **Persistency**: Concept retention tested through multi-turn dialogue.
    - 3 task types × 5 concept scenarios = 15 evaluation tasks.

3. **Quality Control**:

    - To prevent *concept-only solvability*: at least one distractor is plausible for the concept but inconsistent with the image.
    - To prevent *image-only solvability*: at least one distractor is plausible for the image but inconsistent with the concept.
    - Options are shuffled to mitigate positional bias.
    - Each question is annotated by at least 3 annotators and retained only upon majority agreement.

### Evaluation Protocol
- Cold-start setting: only moderate-level textual descriptions or 2 reference images are provided.
- Two dialogue settings: 0-turn and 10-turn.
- Evaluation metric: overall accuracy.

## Key Experimental Results

### Main Results — Personalization Evaluation of 23 VLMs

| Model | Awareness | Appropriateness | Coherency | Overall |
|-------|-----------|----------------|-----------|---------|
| GPT-4o | Upper-mid | Good | Poor | ~60% |
| Claude-Sonnet | Mid | Good | Poor | ~55% |
| InternVL2.5-78B | Mid | Mid | Mid | ~50% |
| LLaVA-NeXT | Low | Low | Low | ~40% |

### Ablation Study

| Configuration | Key Finding |
|---------------|------------|
| Text vs. image injection | 1 image ≈ 3 textual keywords, indicating models struggle to leverage visual cues |
| 0-turn vs. 10-turn dialogue | Performance degrades significantly after 10 turns; intermediate concepts are prone to forgetting |
| Simple vs. detailed description | More detailed descriptions do not consistently improve performance; models may be adversely affected by long contexts |

### Key Findings
- **Even GPT-4o struggles on preference reasoning tasks**, which require abductive reasoning capabilities.
- **Safety alignment in closed-source models impedes personalization**: responses involving persons are frequently refused.
- **VLMs fail to effectively leverage visual cues for personalization**: the small performance gap between image- and text-based injection suggests shallow visual understanding.
- **Mid-sequence forgetting occurs in multi-turn dialogue**: concepts injected at intermediate positions are most susceptible to being forgotten.
- **Personalization bias**: models exhibit significantly weaker personalization for certain concept types (e.g., persons) compared to others.

## Highlights & Insights
- **First systematic VLM personalization benchmark**: fills a critical gap in existing benchmarks, with a scale (111 concepts + 10k QA pairs + 15 task types) far surpassing prior work.
- **Reveals the conflict between safety alignment and personalization**: closed-source models refuse person-related personalization for safety reasons, representing an important policy-capability trade-off.
- **Four-level textual description design**: elegantly enables future research to explore optimal personalization strategies at varying levels of granularity.
- **Rigorous quality control**: the anti-concept-only and anti-image-only design ensures the benchmark genuinely measures multimodal personalization reasoning.

## Limitations & Future Work
- **Static concept assumption**: temporal changes in concepts (e.g., evolving user appearance or preference drift) are not considered.
- **Fixed cold-start setting**: only 2 reference images / moderate-level descriptions are tested; the effect of larger injection volumes remains unexplored.
- **MCQ-only format**: despite the possibility of converting to open-ended evaluation, the current benchmark is limited to multiple-choice questions.
- **Future directions**: incorporating concept temporality; exploring post-hoc fine-tuning using all 5 reference images; extending to additional modalities.

## Related Work & Insights
- **vs. MyVLM/Yo'LLaVA**: These works contain only 29–40 concepts and exclude preference reasoning; MMPB comprehensively surpasses them.
- **vs. MC-LLaVA**: Covers 95 concepts but lacks systematic evaluation, preference modeling, and multi-turn testing.
- **vs. general VQA benchmarks**: ScienceQA, MMBench, and similar benchmarks do not evaluate personalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First comprehensive VLM personalization benchmark, filling a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 models + 15 task types + multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rigorous formalization, in-depth analysis.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to advancing VLM personalization research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[NeurIPS 2025\] EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)
- [\[NeurIPS 2025\] The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process](the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)

<!-- RELATED:END -->
