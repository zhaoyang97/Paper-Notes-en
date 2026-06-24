---
title: >-
  [Paper Note] PCoKG: Personality-aware Commonsense Reasoning with Debate
description: >-
  [AAAI 2026][Graph Learning][Commonsense knowledge graph] This work constructs PCoKG, the first large-scale personality-aware commonsense knowledge graph comprising 521,316 quadruples $(e, p, r, t)$ (event–personality–reasoning dimension–tail), generated via LLM role-playing combined with a multi-agent debate mechanism to produce high-quality personality-differentiated inferences. Experiments validate that MBTI personality information enhances commonsense reasoning and persona…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Commonsense knowledge graph"
  - "personality-aware reasoning"
  - "MBTI"
  - "LLM role-playing"
  - "multi-agent debate"
date: 2026-05-08
content_hash: 87cc2ef24214464a
---

# PCoKG: Personality-aware Commonsense Reasoning with Debate

**Conference**: AAAI 2026
**arXiv**: [2601.06234](https://arxiv.org/abs/2601.06234)  
**Code**: [https://github.com/silverbeats/pcs_v2](https://github.com/silverbeats/pcs_v2)  
**Area**: Graph Learning / Commonsense Reasoning
**Keywords**: Commonsense knowledge graph, personality-aware reasoning, MBTI, LLM role-playing, multi-agent debate

## TL;DR

This work constructs PCoKG, the first large-scale personality-aware commonsense knowledge graph comprising 521,316 quadruples $(e, p, r, t)$ (event–personality–reasoning dimension–tail), generated via LLM role-playing combined with a multi-agent debate mechanism to produce high-quality personality-differentiated inferences. Experiments validate that MBTI personality information enhances commonsense reasoning and personalized dialogue generation.

## Background & Motivation

### State of the Field
Commonsense reasoning is a central challenge in machine intelligence. Existing commonsense knowledge bases such as ATOMIC organize knowledge as $(e, r, t)$ triples—event $e$, reasoning dimension $r$, and tail $t$. Models such as COMET perform inference over ATOMIC and have been applied to emotion recognition and empathetic dialogue generation.

### Limitations of Prior Work

**Individual differences, and in particular the influence of personality traits on reasoning, are systematically ignored.** ATOMIC assumes that all individuals respond identically to the same event; in practice, however:
- Introverts respond to social interactions very differently from extroverts.
- Intuitive (N) types tend toward abstract reasoning, whereas Sensing (S) types prefer concrete expression.
- Different personality types form distinct plans, emotional responses, and preparatory behaviors toward the same event.

For example, given the event "PersonX makes any money":
- ISFJ: plans to save some and spend it on family and friends.
- INTP: delves into the theoretical aspects to discover underlying principles.

Inferences produced by existing COMET models are plausible yet **overly generic**, lacking personalization.

### Core Idea
The paper extends triples to **quadruples** $(e, p, r, t)$ by introducing MBTI-based personality information $p$, and constructs a large-scale personality-aware commonsense knowledge graph via an LLM role-playing and debate framework, thereby avoiding the high cost of recruiting annotators with specific personality profiles.

## Method

### Overall Architecture

PCoKG construction proceeds in two stages:
1. **Event and reasoning dimension acquisition**: Events and reasoning dimension pairs likely to elicit personality-differentiated inferences are filtered from ATOMIC2020.
2. **Personality-aware inference generation**: High-quality inferences are generated through an LLM role-playing and debate mechanism.

### Key Designs

#### 1. **Event and Reasoning Dimension Acquisition**: Three-Evaluator Filtering

**Objective**: Not all events elicit personality-differentiated inferences; the pipeline must identify event–dimension pairs with genuine personality discriminability.

**Procedure**:
- Events are extracted from ATOMIC2020; grammatically ill-formed instances are filtered using `language_tool_python`, yielding 19,184 well-formed events.
- Three LLMs (Deepseek-R1, Qwen-Turbo, Doubao-1.6-Seed) serve as evaluators.
- Nine evaluation criteria are defined, corresponding to nine reasoning dimensions:
    - **xIntent** (motivation): Does the event elicit notably different intrinsic motivations across MBTI types?
    - **xWant** (plans): Do different MBTI types form different plans or intentions toward the event?
    - **xEffect** (effects): Does the event produce different psychological or behavioral effects across MBTI types?
    - **xReact** (emotional reaction), **xNeed** (preparation), **xAttr** (self-narration)
    - **oReact** (others' emotions), **oWant** (others' intentions), **oEffect** (effects on others)
- Each $(e, r)$ pair is scored 1–10 by all three evaluators; only pairs receiving scores **≥ 6 from all three** are retained.
- This yields 95,783 $(e, r)$ pairs covering 15,227 events.

**Design Motivation**: Multi-evaluator consensus filtering ensures data quality—a pair is retained only when all three models agree that the event exhibits significant personality discriminability.

#### 2. **Personality-Aware Inference Generation — Role-Playing and Debate**: Quality Assurance

**Role-playing generation**:
- Sampling follows the global population distribution across 16 MBTI types.
- Role-playing prompts instruct the LLM to simulate a specific MBTI type and produce inferences.
- Reasoning dimensions are expressed as clear natural-language descriptions to improve model comprehension.

**Multi-agent debate mechanism** (core quality assurance):

Three roles are defined:
- **Proponent**: argues that the generated inference is consistent with the target MBTI type and provides supporting evidence.
- **Opponent**: challenges the consistency of the inference with the expected personality type.
- **Judge**: evaluates both arguments and renders a final verdict.

**Debate procedure** (Algorithm 2):
1. Initialization → the LLM generates inference $t$ given $(e, r, p)$.
2. Multi-round debate: Proponent defends → Opponent challenges → repeat.
3. The Judge evaluates and decides:
    - If the inference is acceptable → return $t$.
    - If not → the Judge provides feedback and the model iteratively revises.
4. At most `max_generate_times` retries are allowed.

**Design Motivation**: Pure role-playing generation may not fully conform to the target personality. The debate mechanism ensures personality consistency and output quality through multi-perspective scrutiny and iterative feedback.

#### 3. **Dataset Scale and Statistics**

| Metric | Value |
|--------|-------|
| Dataset size | 521,316 quadruples |
| Number of events | 15,077 |
| Average event length | 4.79 words |
| Average inference length | 8.75 words |
| MBTI types | All 16 covered |
| Reasoning dimensions | 9 |

### Loss & Training

**Downstream application training**:
- Data split: 99 : 0.5 : 0.5 (train : validation : test), grouped by event.
- Backbone models: Qwen3-0.6B, LLaMA3-1B, MiniCPM4-0.5B.
- **Full-parameter fine-tuning (PCoKGM)**: reasoning dimensions and personality type are concatenated into the input as natural-language prompts.
- **Comparison baseline (COMET)**: reasoning dimensions and personality type are encoded as special tokens.
- Training setup: 4 × 3090 GPUs, batch size 8 per GPU, gradient accumulation over 4 steps, cosine learning rate schedule, 1 epoch, validation every 300 steps, early stopping.

## Key Experimental Results

### Dataset Validation

**1. Readability–Personality Correlation Analysis**:
- ESFP (Flesch score 77.7) and ESTP (74.0) employ direct, concrete language.
- INTJ (37.0) and INTP (39.6) use more complex, abstract language.
- These findings align with MBTI theory: T/N types prefer logical and abstract reasoning, while F/S types prefer emotionally accessible expression.

**2. Adjusted Mutual Information (AMI) Analysis**:

| Reasoning Dimension | AMI | AMI (shuffled) | Note |
|--------------------|-----|----------------|------|
| xAttr (self-narration) | **0.512** | −0.000027 | Strongest personality association |
| xReact (emotional reaction) | 0.256 | −0.000009 | Strong association |
| xIntent (motivation) | 0.240 | −0.000033 | Strong association |
| xWant (plans) | 0.238 | 0.000021 | Strong association |
| oReact (others' emotions) | 0.115 | −0.000044 | Weaker but significant |

AMI values across all dimensions are significantly higher than the shuffled baseline (Mann–Whitney U test, $p < 0.01$); dimensions related to self-perception show the strongest association with MBTI.

**3. Human Evaluation**:
- Three psychology graduate students evaluate 1,440 samples.
- Coherence: 1.78/2.0; Naturalness: 1.71/2.0; Personality consistency: 1.63/2.0.
- Fleiss' Kappa = 0.57 (moderate to substantial agreement).

### Main Results

| Model | B-4 | R-1 | R-2 | R-L |
|-------|-----|-----|-----|-----|
| DeepSeek-R1 (1-shot) | 2.67 | 14.45 | 1.89 | 13.44 |
| GPT-o4-mini (1-shot) | 5.38 | 15.34 | 2.09 | 14.28 |
| COMET-LLaMA3 | 12.58 | 30.51 | 12.77 | 28.91 |
| COMET-Qwen3 | 10.09 | 26.31 | 9.26 | 25.00 |
| **PCoKGM-LLaMA3** | 13.73 | 32.09 | 14.31 | 30.53 |
| **PCoKGM-Qwen3** | 14.08 | 32.68 | 14.78 | 31.07 |
| **PCoKGM-MiniCPM4** | **14.50** | **32.99** | **15.27** | **31.38** |

PCoKGM consistently outperforms both COMET and 1-shot inference with large LLMs, demonstrating the superiority of personality information and the natural-language prompt encoding scheme.

### Ablation Study

| Configuration (LLaMA3) | B-4 | R-1 | R-2 | R-L | Note |
|------------------------|-----|-----|-----|-----|------|
| PCoKGM (full) | **13.73** | **32.09** | **14.31** | **30.53** | Full model |
| w/o MBTI | 10.16 | 25.59 | 9.36 | 24.51 | **Largest drop**; personality is central |
| w/o select | 11.25 | 27.92 | 10.59 | 26.49 | Event filtering is effective |
| w/o debate | 12.09 | 29.45 | 12.04 | 28.08 | Debate mechanism improves quality |
| w/o select & debate | 10.66 | 26.00 | 9.62 | 24.72 | Both are complementary; removing both yields the worst performance |

Consistent ablation trends across all three backbone models confirm the robustness of these conclusions.

### Key Findings

1. **MBTI information is the most critical factor**: removing MBTI causes the largest performance drop (B-4: 13.73 → 10.16), confirming that personality traits serve as an anchor for structured reasoning.
2. **Natural language > special tokens**: PCoKGM's natural-language prompt encoding of personality and reasoning dimensions outperforms COMET's special-token encoding.
3. **Performance scales with model size**: across the Qwen3, LLaMA3, and MiniCPM4 families, larger models consistently perform better on PCoKG tasks.
4. **Effective for dialogue generation**: on the SPC dataset, PCoKGM-augmented dialogue generation comprehensively outperforms COMET-augmented and commonsense-free baselines.
5. **LLM zero/few-shot generation is insufficient**: 1-shot inference with large LLMs falls far short of fine-tuned small models, validating the necessity of the debate framework for refining outputs during construction.

## Highlights & Insights

1. **First large-scale personality-aware commonsense knowledge graph**: 521K quadruples, 16 MBTI types, and 9 reasoning dimensions, filling the gap at the intersection of personality and commonsense reasoning.
2. **Quality assurance via multi-agent debate**: the Proponent–Opponent–Judge triangular debate framework is more reliable than single-pass generation; the feedback loop systematically improves generation quality beyond individual sample refinement.
3. **Intriguing phenomenon revealed by readability analysis**: LLMs in role-playing mode genuinely produce language-style differences consistent with MBTI theory (INTJ: complex and abstract; ESFP: direct and concrete), suggesting that LLMs have internalized implicit personality–language associations.
4. **Transferability of the dataset construction pipeline**: although built around the MBTI framework, the pipeline design can be adapted to other personality theories or character attributes.
5. **Deeper insight from AMI analysis**: self-perception dimensions (xAttr, AMI = 0.512) exhibit a stronger association with personality than other-perception dimensions (oReact, AMI = 0.115)—an independently interesting psychological finding.

## Limitations & Future Work

1. **Only personality traits are considered**; other factors influencing reasoning, such as gender, occupation, and cultural background, are not addressed.
2. **The psychometric validity of MBTI is contested** (test–retest reliability is unstable), and a knowledge graph built upon it may inherit this limitation.
3. **Personality consistency received the lowest human evaluation score (1.63/2.0)**, indicating that LLM role-playing still has room to improve in precisely matching specific personality types.
4. **Evaluation metrics are primarily lexical overlap-based** (BLEU, ROUGE) and do not assess personality consistency at the semantic level.
5. **Future directions**: integrating additional demographic attributes, exploring more psychometrically robust personality models such as the Big Five, introducing semantic evaluation metrics, and constructing multilingual versions.

## Related Work & Insights

- **ATOMIC/COMET** provide the foundational event-inference framework; PCoKG extends it by adding a personality dimension.
- Yang et al. (2024) constructed a Chinese personalized commonsense reasoning dataset but relied on manual annotation, limiting its scale.
- The debate framework design is transferable to **any scenario requiring LLMs to generate high-quality structured data**.
- PCoKG has direct application value for **personalized dialogue systems, affective computing, and user modeling**.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The intersection of personality and commonsense reasoning is a novel direction; the debate + role-playing pipeline is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-faceted dataset validation (readability / AMI / human evaluation) + model experiments + ablation + scale analysis + downstream application.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured overall, though some passages are verbose.
- **Value**: ⭐⭐⭐⭐ — Open-source dataset and code with a reusable pipeline, though the contested validity of MBTI limits direct application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs](../../ACL2026/graph_learning/cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)

</div>

<!-- RELATED:END -->
