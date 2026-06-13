---
title: >-
  [Paper Note] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs
description: >-
  [ACL 2026][LLM/NLP][Reasoning Models] This paper identifies a "benign self-reading" pattern in reasoning LLMs (such as DeepSeek-R1) during quantitative reasoning—where answer tokens exhibit forward centroid drift (advanc…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Reasoning Models"
  - "Self-Reading Patterns"
  - "Attention Analysis"
  - "Activation Steering"
  - "Quantitative Reasoning"
date: 2026-05-08
content_hash: bc6e4ade214599d5
---

# How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19149](https://arxiv.org/abs/2604.19149)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Reasoning Models, Self-Reading Patterns, Attention Analysis, Activation Steering, Quantitative Reasoning

## TL;DR

This paper identifies a "benign self-reading" pattern in reasoning LLMs (such as DeepSeek-R1) during quantitative reasoning—where answer tokens exhibit forward centroid drift (advancing along the reasoning chain) and concentration on semantic anchors (repeatedly revisiting key steps). This pattern strongly correlates with correctness. Based on this, a training-free activation steering method driven by SRQ (Self-Reading Quality) is proposed, improving accuracy by up to 2.6% across multiple benchmarks.

## Background & Motivation

**Background**: Reasoning LLMs (e.g., DeepSeek-R1, GPT-5, Gemini 3) generate reasoning traces (delimited by `</think>`) before producing an answer. Activation steering has been proven effective in controlling reasoning traces, such as compressing redundant outputs or guiding verification and backtracking.

**Limitations of Prior Work**: Existing research primarily focuses on shaping the reasoning traces themselves, while how answer tokens "read" and integrate these traces to produce reliable outputs remains unclear. Navigating thousands of tokens of reasoning to utilize key information amidst noise is a critical challenge for answer tokens.

**Key Challenge**: Reasoning traces contain both critical steps and exploratory trials or redundancies. Answer tokens require "selective reading"—yet the mechanism behind this and its relationship with correctness remain unknown.

**Goal**: (1) Understand how answer tokens read reasoning traces; (2) Establish a correlation between self-reading patterns and correctness; (3) Utilize self-reading quality signals for training-free guidance.

**Key Insight**: By analyzing the attention distribution of answer tokens over reasoning tokens, the trajectory of attention centroids and focus points reveals the model's "reading strategy."

**Core Idea**: Benign self-reading acts as a behavioral signature of internal certainty—the model has committed to a solution path and relies on a few key reasoning steps as evidence for generating the answer. Forward drift of attention centroids reflects "control" (advancing along the selected branch), while persistent focusing on semantic anchors reflects "monitoring" (repeatedly verifying evidence).

## Method

### Overall Architecture

The method consists of three stages: (1) Self-reading analysis—analyzing answer-reasoning attention patterns across three reasoning LLMs on GSM8K to identify benign features; (2) SRQ score design—quantifying self-reading quality from geometric (process structure of control strategies) and semantic (content anchoring of monitoring quality) dimensions; (3) Activation steering—constructing steering vectors using samples with different SRQ scores and injecting them during inference to promote benign self-reading.

### Key Designs

1.  **Identification of Benign Self-Reading Patterns**:
    *   **Function**: Discovering structured patterns in how answer tokens read reasoning traces.
    *   **Mechanism**: The weighted average position (centroid) of the attention distribution for each answer token over reasoning tokens is calculated, normalized to [0,1]. In correct samples, the centroid trajectory follows a clear diagonal pattern—the focus shifts forward along the reasoning chain as the answer progresses. Simultaneously, attention concentrates on "semantic anchors" (constraints, plans, reflections, final conclusions). In incorrect samples, attention is scattered and irregular.
    *   **Design Motivation**: Reasoning tokens perform "object-level" computation, while answer tokens perform "meta-level" operations—control (advancing the reading focus) and monitoring (reviewing evidence). This aligns with metacognition frameworks in classical cognitive theory (Nelson 1990, Koriat 1997).

2.  **SRQ (Self-Reading Quality) Score**:
    *   **Function**: Quantifying the quality of self-reading patterns for sample selection and steering vector construction.
    *   **Mechanism**: The geometric dimension measures the forward progression and smoothness of centroid trajectories. The semantic dimension measures focus on critical semantic steps. These dimensions are combined into the SRQ score.
    *   **Design Motivation**: Geometric metrics alone might select "smooth but semantically meaningless" samples, while semantic metrics alone might select "correctly anchored but structurally chaotic" samples. Combining them ensures high-quality self-reading.

3.  **SRQ-Driven Activation Steering**:
    *   **Function**: Promoting better self-reading patterns during inference without additional training.
    *   **Mechanism**: Steering vectors are built from the activation differences between high-SRQ and low-SRQ samples. During inference, these vectors are added to hidden states to push the model away from chaotic reading and toward ordered reading. This requires no parameter updates.
    *   **Design Motivation**: Benign self-reading correlates with correctness (manual verification shows 159/171 correct samples exhibit this pattern); thus, encouraging this behavior should improve performance.

### Loss & Training

Entirely training-free. Steering vectors are extracted from activation differences of contrastive samples and added to hidden states at target layers during inference.

## Key Experimental Results

### Main Results

**Accuracy improvement from SRQ steering across multiple benchmarks**

| Model | Benchmark | Baseline Acc | + SRQ Steering | Gain |
| :--- | :--- | :--- | :--- | :--- |
| R1-Distill-Llama-8B | GSM8K | ~82% | ~84.6% | +2.6% |
| R1-Distill-Qwen-7B | GSM8K | ~83% | ~85% | +2% |
| Qwen3-4B-Thinking | GSM8K | ~80% | ~81.5% | +1.5% |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Geometric Only | Small Gain | Lacks semantic anchoring signals |
| Semantic Only | Moderate Gain | Lacks process structure signals |
| **Geometric + Semantic** | **Optimal** | Both dimensions are complementary |

**Manual Annotation Verification (200 samples)**

| Type | Count | Description |
| :--- | :--- | :--- |
| Correct + Benign | 159/171 correct | 93% of correct samples show benign self-reading |
| Incorrect + Benign | 3/26 incorrect | Only 12% of incorrect samples show benign self-reading |
| Balanced Subset (50+50) | 48 Correct/Benign vs 46 Incorrect/Non-Benign | Consistent trends |

### Key Findings

*   Benign self-reading patterns are nearly universal in correct samples (93%) and rare in incorrect ones (12%).
*   Aggregating attention maps across 100 correct samples still yields a clear diagonal ridge, proving it is a stable systematic behavior.
*   SRQ steering consistently improves accuracy without parameter modification, validating the causal link between self-reading patterns and correctness.
*   Geometric and semantic dimensions are complementary; neither is as effective alone as when combined.

## Highlights & Insights

*   The discovery of "self-reading" behavior in reasoning LLMs and its link to correctness is a significant contribution to understanding internal mechanisms.
*   The introduction of a metacognitive framework (control + monitoring) provides a theoretical foundation from cognitive science to explain LLM behavior.
*   SRQ-driven activation steering demonstrates a complete loop from mechanistic understanding to practical application.

## Limitations & Future Work

*   Validated only on quantitative reasoning; applicability to logical or commonsense reasoning is unknown.
*   Accuracy gains are relatively modest (up to 2.6%).
*   Identification of semantic anchors may be task-specific.
*   Future work could explore guiding models to learn better self-reading patterns during the training phase.

## Related Work & Insights

*   **vs Venhoff et al. (2025)**: Approaches focus on steering verification and backtracking within traces; this work focuses on reading behavior during the answer stage.
*   **vs Azizi et al. (2025)**: Focuses on trace compression; this work focuses on how answers utilize reasoning content.
*   **vs Zhang et al. (2025)**: Confirms the existence of answer-reasoning attention links; this work provides deep analysis of their structural patterns and functional significance.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of answer token self-reading in reasoning LLMs with cognitive science depth.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Tested across three models with manual validation, though restricted in task variety.
*   Writing Quality: ⭐⭐⭐⭐⭐ Excellent visualization, deep analysis, and appropriate cognitive analogies.
*   Value: ⭐⭐⭐⭐ Provides a new analytical perspective and practical tools for improving reasoning LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)

</div>

<!-- RELATED:END -->
