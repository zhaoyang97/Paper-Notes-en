---
title: >-
  [Paper Note] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction
description: >-
  [AAAI 2026][LLM (Other)][LLM Evaluation] This paper systematically quantifies the reliability degradation of LLMs in multi-turn conversations through three deterministically evaluable representative tasks—instruction following, tool selection, and entity extraction—revealing failure modes such as instruction drift, intent confusion, and context overwriting in extended dialogues.
tags:
  - "AAAI 2026"
  - "LLM (Other)"
  - "LLM Evaluation"
  - "Multi-Turn Dialogue"
  - "Reliability"
  - "Instruction Following"
  - "Tool Selection"
date: 2026-05-08
content_hash: 301b211cf79874e4
---

# Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction

**Conference**: AAAI 2026
**arXiv**: [2603.01423](https://arxiv.org/abs/2603.01423)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: LLM Evaluation, Multi-Turn Dialogue, Reliability, Instruction Following, Tool Selection

## TL;DR

This paper systematically quantifies the reliability degradation of LLMs in multi-turn conversations through three deterministically evaluable representative tasks—instruction following, tool selection, and entity extraction—revealing failure modes such as instruction drift, intent confusion, and context overwriting in extended dialogues.

## Background & Motivation

### State of the Field
LLMs are increasingly deployed in real-world applications where users engage in extended, mixed-topic multi-turn conversations. Prior work has demonstrated significant performance degradation in multi-turn settings, including the "lost in the middle" effect and declining instruction consistency.

### Limitations of Prior Work

**Insufficient research on multi-turn reliability**: Existing benchmarks (Multi-IF, StructFlowBench, MINT, etc.) address certain aspects of conversational robustness but tend to focus on abstract challenges or rely on subjective judgments, making it difficult to objectively assess specific behaviors in practical deployments.

**Unclear reliability gap between single-turn and multi-turn settings**: Strong single-turn performance does not guarantee multi-turn reliability, yet systematic paired comparisons are lacking.

**Failure modes not systematically analyzed**: Key failure modes observed in real deployments—such as instruction drift, intent confusion, and context overwriting—lack quantitative investigation.

### Starting Point
Three compact, deterministically pass/fail evaluable tasks are designed, each with paired single-turn and multi-turn versions, to isolate and quantify the reliability degradation introduced by multi-turn dialogue. The tasks directly reflect core requirements of practical assistant systems.

## Method

### Overall Architecture

Three evaluation tasks × two settings (single-turn / multi-turn) = 6 conditions, with approximately 600 evaluation instances. Reliability degradation is quantified by comparing accuracy differences between single-turn and multi-turn settings. All tasks are designed for deterministic pass/fail evaluation to avoid subjectivity.

### Key Designs

#### 1. **Instruction Following**

**Task Definition**: A global constraint is specified at the start of the conversation (e.g., "always answer in at most 5 sentences"); the dialogue continues for several turns on unrelated topics before a question deliberately designed to elicit a long response is posed.

**Single-Turn Setting**: Constraint + question provided directly.
**Multi-Turn Setting**: 5–15 turns of unrelated small talk are inserted after the constraint, followed by the target question.

**Evaluation Criterion**: Whether the response satisfies the constraint of no more than 5 sentences; exceeding this results in failure.

**Design Motivation**: Simulates real-world scenarios where chatbots must continuously comply with formatting rules (e.g., staying concise, avoiding specific vocabulary).

#### 2. **Tool Selection**

**Task Definition**: The model must select the correct tool from a fixed set [Weather, News, Calculator, Stock, Recipe, Dictionary] for each request.

**Single-Turn Setting**: A user query directly maps to a single tool.
**Multi-Turn Setting**: Tool requests spanning multiple topics are interleaved throughout the conversation (6–16 turns, 2–6 different tools selected at random).

**Evaluation Criterion**: Whether the selected tool matches the ground-truth tool.

**Design Motivation**: Reflects intent classification in digital assistants and request routing in multi-agent systems.

#### 3. **Entity Extraction**

**Task Definition**: Extract the final structured information (date, time, party size) for a restaurant reservation.

**Single-Turn Setting**: The reservation request is stated directly.
**Multi-Turn Setting**: Realistic complications are introduced—changes of mind, intermediate chit-chat, and multiple mentions of other reservations.

**Evaluation Criterion**: Whether all three slots exactly match the ground-truth values.

**Design Motivation**: Simulates practical requirements for tool parameter extraction (e.g., calendar or booking APIs).

### Data Generation

- Dialogues are synthesized using GPT-5, with controlled conversation length, topic transition count, and modification frequency.
- Ground-truth labels are automatically derived during generation and verified via human sampling.
- Approximately 100 dialogues per task across multiple conditions.

### Evaluated Models

**Commercial LLMs**: GPT-4o, GPT-4o-mini, Gemini-2.5-Flash
**Open-Source SLMs**: Qwen3-4B/8B/32B, Ministral-8B, Mistral-small-24B, Gemma-3-12B
All models are decoded at temperature 0 to ensure deterministic outputs.

## Key Experimental Results

### Main Results

| Model | Instruction Following (S/M) | Tool Selection (S/M) | Entity Extraction (S/M) |
|---|---|---|---|
| GPT-4o | 96→**63** (-33) | 100→99 (-1) | 100→86 (-14) |
| GPT-4o-mini | 93→**24** (-69) | 100→93 (-7) | 96→84 (-12) |
| Gemini-2.5-Flash | 96→**89** (-7) | 100→97 (-3) | 100→89 (-11) |
| Gemma-3-12B | 92→**33** (-59) | 100→98 (-2) | 92→79 (-13) |
| Qwen3-8B | 83→**27** (-56) | 100→89 (-11) | 98→88 (-10) |
| Qwen3-32B | 92→**54** (-38) | 100→**47** (-53) | 100→89 (-11) |
| Ministral-8B | 27→**11** (-16) | 99→**37** (-62) | 100→88 (-12) |

### Ablation Study

**By Dialogue Length (Instruction Following)**:

| Turns | 5 | 6 | 7 | 8 | 9 | 11 |
|---|---|---|---|---|---|---|
| Accuracy | 0.40 | 0.28 | 0.38 | 0.15 | 0.29 | 0.25 |

**By Number of Tools (Tool Selection)**:

| Number of Tools | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Accuracy | **0.98** | 0.82 | 0.74 | 0.64 | 0.71 |

**By Scenario Type (Entity Extraction)**:

| Scenario Type | Date | Time | Party Size | All Match |
|---|---|---|---|---|
| Change in Mind | 0.91 | 0.96 | 0.99 | **0.85** |
| Intermediate Chit-Chat | 0.91 | 0.97 | 0.97 | 0.86 |
| Multiple Mention | 0.94 | 0.98 | 0.99 | **0.91** |

### Key Findings

1. **Instruction following exhibits the most severe degradation**: Even GPT-4o drops from 96% to 63%, with smaller models faring far worse (GPT-4o-mini: 93%→24%). This is not purely a long-context problem—accuracy does not decrease monotonically with turn count (recovering to 96% at turn 10) but is instead tied to specific distractor content.

2. **Tool selection shows a bimodal pattern**: Commercial LLMs are nearly unaffected (GPT-4o: 100%→99%), whereas smaller models collapse under mixed multi-tool conditions (Qwen3-32B: 100%→47%; Ministral-8B: 99%→37%). Accuracy decreases substantially as the number of available tools increases.

3. **Entity extraction is most robust**: Because the target fields are structured short phrases or numbers (date, time, party size), models can capture final slot values with less ambiguity. Nevertheless, the date slot is consistently the weakest, reflecting difficulty in temporal tracking.

4. **Three core failure modes**:
    - **Instruction drift**: Global constraints are forgotten after multiple turns.
    - **Intent confusion**: Models over-rely on recent context and reuse the previous tool (e.g., selecting Stock again when transitioning to a Weather request).
    - **Context overwriting**: Nearby mentions interfere with working memory, overwriting already correctly updated slot values.

## Highlights & Insights

1. **Elegant task design**: The three tasks each represent a distinct category of core requirements (maintaining global constraints, request routing, state tracking), all with clear pass/fail criteria.
2. **Paired single-turn/multi-turn design**: Precisely isolates the effect of "dialogue extension" as an independent variable.
3. **Degradation is not caused by length per se**: Accuracy on instruction following bears no monotonic relationship with dialogue length; rather, it is associated with specific contextual conflicts and competing demands.
4. **Capacity-dependent fragility**: Larger models (GPT-4o, Gemini-2.5-Flash, Qwen3-32B) degrade substantially less than smaller models, yet even large models exhibit significant degradation on instruction following.

## Limitations & Future Work

1. **Small dataset scale**: Approximately 100 dialogues per task and roughly 600 total evaluation instances limit statistical power.
2. **English only**: Multi-turn reliability in multilingual settings may be worse.
3. **Entity extraction task is relatively simple**: Target fields are explicit numbers or short phrases; degradation would likely be more pronounced if richer contextual understanding were required (e.g., mapping "the pizza with pineapple" to "Hawaiian pizza").
4. **Tasks are somewhat artificial**: Although designed to simulate real-world scenarios, synthetic dialogues still differ from authentic user behavior.
5. **No remediation proposed**: The paper primarily diagnoses problems without proposing solutions.

## Related Work & Insights

- **Laban et al. (2025)**: Demonstrates significant LLM performance degradation in multi-turn analysis.
- **Liu et al. (2023)**: Discovery of the "lost in the middle" effect.
- **Multi-IF / StructFlowBench / MINT**: Precursor work on multi-turn evaluation benchmarks.
- **McNemar's test**: Used to confirm the statistical significance of single-turn vs. multi-turn performance gaps.

## Rating

- Novelty: ⭐⭐⭐ — Task design has practical value, but the core finding (multi-turn degradation) is not unexpected.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 8 models × 3 tasks × analysis by length/complexity + qualitative case analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-defined conclusions, intuitive figures.
- Value: ⭐⭐⭐⭐ — Provides a practical evaluation framework for assessing LLM reliability in multi-turn deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Confidence Estimation for LLMs in Multi-turn Interactions](../../ACL2026/llm_nlp/confidence_estimation_for_llms_in_multi-turn_interactions.md)
- [\[ACL 2025\] Quantifying Semantic Emergence in Language Models](../../ACL2025/llm_nlp/quantifying_semantic_emergence_in_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](../../ACL2026/llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](../../ACL2025/llm_nlp/controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)

</div>

<!-- RELATED:END -->
