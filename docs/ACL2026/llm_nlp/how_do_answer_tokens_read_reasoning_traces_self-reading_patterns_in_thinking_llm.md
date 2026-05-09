---
title: >-
  [Paper Note] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs
description: >-
  [ACL 2026][LLM/NLP][reasoning models] This paper identifies a "benign self-reading" pattern in reasoning LLMs (e.g., DeepSeek-R1) during quantitative reasoning: answer tokens' attention over reasoning traces exhibits forward drift (progressively advancing along the reasoning chain) and semantic anchor concentration (repeatedly revisiting key steps), and this pattern strongly correlates with correctness. Building on this finding, the authors propose a training-free activation steering method driven by Self-Reading Quality (SRQ) scores, achieving accuracy improvements of up to 2.6% across multiple benchmarks.
tags:
  - ACL 2026
  - LLM/NLP
  - reasoning models
  - self-reading patterns
  - attention analysis
  - activation steering
  - quantitative reasoning
date: 2026-05-08
content_hash: 4b1b453f837cb17e
---

# How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs

**Conference**: ACL 2026
**arXiv**: [2604.19149](https://arxiv.org/abs/2604.19149)
**Code**: None
**Area**: LLM/NLP
**Keywords**: reasoning models, self-reading patterns, attention analysis, activation steering, quantitative reasoning

## TL;DR

This paper identifies a "benign self-reading" pattern in reasoning LLMs (e.g., DeepSeek-R1) during quantitative reasoning: answer tokens' attention over reasoning traces exhibits forward drift (progressively advancing along the reasoning chain) and semantic anchor concentration (repeatedly revisiting key steps), and this pattern strongly correlates with correctness. Building on this finding, the authors propose a training-free activation steering method driven by Self-Reading Quality (SRQ) scores, achieving accuracy improvements of up to 2.6% across multiple benchmarks.

## Background & Motivation

**Background**: Reasoning LLMs (e.g., DeepSeek-R1, GPT-5, Gemini 3) generate reasoning traces before producing answers (delimited by `</think>`). Activation steering has been shown to control the behavior of reasoning traces, such as compressing redundant output and inducing verification and backtracking.

**Limitations of Prior Work**: Existing work primarily focuses on shaping the reasoning traces themselves, leaving unclear how answer tokens "read" and integrate reasoning traces to produce reliable outputs. How answer tokens navigate noise and exploit critical information in reasoning chains spanning thousands of tokens remains an open question.

**Key Challenge**: Reasoning traces contain both critical reasoning steps and exploratory attempts alongside redundant content. Answer tokens must engage in "selective reading," yet it remains unknown how the model achieves this or how reading patterns relate to correctness.

**Goal**: (1) Understand how answer tokens read reasoning traces; (2) establish the association between self-reading patterns and correctness; (3) leverage self-reading quality signals for training-free steering.

**Key Insight**: Analyze the attention distribution of answer tokens over reasoning tokens—trajectories and concentration points of the attention centroid reveal the model's "reading strategy."

**Core Idea**: Benign self-reading constitutes a behavioral signature of internal certainty: the model has committed to a solution path and relies on a small number of key reasoning steps as evidence for answer generation. Forward drift of the attention centroid reflects "control" (advancing along the selected branch), while sustained focus on semantic anchors reflects "monitoring" (repeatedly verifying evidence).

## Method

### Overall Architecture

Three phases: (1) Self-reading analysis—analyze answer-reasoning attention patterns of three reasoning LLMs on GSM8K to identify benign self-reading characteristics; (2) SRQ score design—quantify self-reading quality along geometric dimensions (process structure of the control strategy) and semantic dimensions (content anchoring of monitoring quality); (3) Activation steering—construct steering vectors from activation differences between high- and low-SRQ samples and inject them into hidden states at inference time to promote benign self-reading.

### Key Designs

1. **Benign Self-Reading Pattern Identification**:

    - **Function**: Discover structured patterns in how answer tokens read reasoning traces.
    - **Mechanism**: Compute the weighted average position (centroid) of each answer token's attention distribution over reasoning tokens, normalized to $[0, 1]$. In correct samples, the centroid trajectory exhibits a clear diagonal pattern—as answer generation progresses, the reading focus advances along the reasoning chain. Simultaneously, attention repeatedly concentrates on "semantic anchors" (problem constraints, solution plans, reflections, and final conclusions). In incorrect samples, attention is scattered and irregular.
    - **Design Motivation**: Reasoning tokens implement "object-level" computation, while answer tokens implement "meta-level" operations—control (advancing the reading focus) and monitoring (reviewing evidence). This aligns with the metacognitive framework of classical cognitive theory (Nelson 1990; Koriat 1997).

2. **SRQ (Self-Reading Quality) Score**:

    - **Function**: Quantify the quality of self-reading patterns for sample selection and steering vector construction.
    - **Mechanism**: Geometric dimension—measures the degree of forward advancement and smoothness of the attention centroid trajectory (i.e., whether it progresses along the diagonal). Semantic dimension—measures whether attention concentrates on critical semantic steps (constraints, plans, conclusions). The two dimensions are combined into a unified SRQ score.
    - **Design Motivation**: Using the geometric dimension alone may select samples that are "smooth but semantically meaningless," while using the semantic dimension alone may select samples that are "correctly anchored but procedurally disordered." The two dimensions are complementary, ensuring that genuinely high-quality self-reading samples are selected.

3. **SRQ-Driven Activation Steering**:

    - **Function**: Training-free promotion of better self-reading patterns at inference time.
    - **Mechanism**: High-SRQ and low-SRQ samples are selected; activation differences at intermediate layers are extracted to construct steering vectors. At inference time, the steering vectors are added to hidden states at target layers to steer the model away from disordered reading and toward structured reading. No model parameters are modified and no additional training is required.
    - **Design Motivation**: Benign self-reading is associated with correctness (human annotation confirms 159 out of 171 correct samples exhibit benign self-reading); therefore, steering the model toward benign self-reading patterns should improve accuracy.

### Loss & Training

Entirely training-free. Steering vectors are extracted from activation differences of contrastive samples and added to hidden states at target layers during inference.

## Key Experimental Results

### Main Results

**Accuracy improvements from SRQ steering across multiple benchmarks**

| Model | Benchmark | Baseline Accuracy | + SRQ Steering | Gain |
|---|---|---|---|---|
| R1-Distill-Llama-8B | GSM8K | ~82% | ~84.6% | +2.6% |
| R1-Distill-Qwen-7B | GSM8K | ~83% | ~85% | +2% |
| Qwen3-4B-Thinking | GSM8K | ~80% | ~81.5% | +1.5% |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Geometric dimension only | Smaller gain | Missing semantic anchoring signal |
| Semantic dimension only | Moderate gain | Missing process structure signal |
| **Geometric + Semantic** | **Optimal** | Two dimensions are complementary |

**Human Annotation Validation (200 samples)**

| Type | Count | Notes |
|---|---|---|
| Correct + benign self-reading | 159/171 correct | 93% of correct samples exhibit benign self-reading |
| Incorrect + benign self-reading | 3/26 incorrect | Only 12% of incorrect samples exhibit benign self-reading |
| Balanced subset (50+50) | 48 correct with benign vs. 46 incorrect without | Trend consistent |

### Key Findings

- Benign self-reading patterns are nearly universal among correct samples (93%) and rare among incorrect samples (12%).
- Aggregating attention maps from 100 correct samples still yields a clear diagonal ridge, confirming stable and systematic behavior.
- SRQ steering consistently improves accuracy without modifying model parameters, validating a causal association between self-reading patterns and correctness.
- The geometric and semantic dimensions are complementary—either dimension alone is less effective than the combination.

## Highlights & Insights

- Identifying the "self-reading" behavior of reasoning LLMs and its association with correctness constitutes a significant contribution to understanding LLM internal mechanisms.
- The introduction of a metacognitive framework (control + monitoring) provides a cognitive-scientific theoretical basis for interpreting LLM behavior.
- SRQ-driven activation steering demonstrates a complete pipeline from mechanistic understanding to practical application.

## Limitations & Future Work

- Validation is limited to quantitative reasoning tasks; applicability to other reasoning types (logical, commonsense) remains unexplored.
- Accuracy improvements are modest (up to 2.6%).
- Identification of semantic anchors may be task-specific.
- Future work could explore how to guide models to learn better self-reading patterns during training.

## Related Work & Insights

- **vs. Venhoff et al. (2025)**: Steers verification and backtracking behavior within reasoning traces; this work steers reading behavior during the answer phase.
- **vs. Azizi et al. (2025)**: Focuses on steering to compress reasoning length; this work examines how answers leverage reasoning.
- **vs. Zhang et al. (2025)**: Confirms the existence of answer-reasoning attention links; this work provides in-depth analysis of their structural patterns and functional significance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of self-reading behavior in answer tokens of reasoning LLMs; conceptually novel with cognitive-scientific depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, human annotation, and activation steering validation, though the task scope is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Visualizations are compelling, analysis is thorough, and cognitive analogies are apt.
- Value: ⭐⭐⭐⭐ Provides a new analytical perspective and practical tool for understanding and improving reasoning LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](../../ICLR2026/llm_nlp/gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)

</div>

<!-- RELATED:END -->
