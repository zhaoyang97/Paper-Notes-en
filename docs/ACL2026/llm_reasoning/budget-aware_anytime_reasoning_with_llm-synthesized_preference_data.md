---
title: >-
  [Paper Note] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data
description: >-
  [ACL 2026][LLM Reasoning][budget-aware reasoning] This paper proposes a budget-aware anytime reasoning framework and an Anytime Index metric to quantify the quality-efficiency trade-off of LLM reasoning under limited tok…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "budget-aware reasoning"
  - "Anytime Index"
  - "preference data prompting"
  - "test-time scaling"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: fb33fa8a4d9a265c
---

# Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data

**Conference**: ACL 2026
**arXiv**: [2601.11038](https://arxiv.org/abs/2601.11038)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: budget-aware reasoning, Anytime Index, preference data prompting, test-time scaling, reasoning efficiency

## TL;DR

This paper proposes a budget-aware anytime reasoning framework and an Anytime Index metric to quantify the quality-efficiency trade-off of LLM reasoning under limited token budgets. It further introduces Preference Data Prompting (PDP), a test-time self-improvement method based on LLM-synthesized preference data, achieving substantial improvements in both intermediate and final solution quality across planning, mathematics, and science QA tasks.

## Background & Motivation

**State of the Field**: LLMs have demonstrated strong reasoning capabilities through methods such as Chain-of-Thought (CoT) and Tree-of-Thoughts. Test-time scaling has emerged as an important technique for improving reasoning performance; however, existing approaches typically assume unlimited computational resources and evaluate only final answer quality.

**Limitations of Prior Work**: (1) Many practical scenarios impose strict computational or latency budget constraints, where even partial solutions are more useful than none (e.g., an incomplete but feasible travel itinerary); (2) existing methods lack a principled way to evaluate how reasoning quality evolves as token count increases; (3) budget-aware techniques such as BRPO focus on "when to stop thinking" rather than "how to think better under constraints."

**Root Cause**: Real-world reasoning tasks require producing optimal intermediate solutions within a limited budget, yet current evaluation and optimization frameworks focus exclusively on final answers, neglecting the efficiency of the reasoning trajectory.

**Paper Goals**: (1) Establish a framework and metric for evaluating LLM reasoning efficiency across varying token budgets; (2) provide a method to improve the quality of budget-aware reasoning.

**Starting Point**: Drawing on the classical AI concept of anytime algorithms, the paper treats reasoning as a process of monotonically improving solution quality as the token budget increases.

**Core Idea**: Reasoning efficiency is quantified by truncating reasoning trajectories and evaluating solution quality at each checkpoint. Preference data is constructed from comparisons among the model's own reasoning outputs and used as in-context examples to improve intermediate solution quality.

## Method

### Overall Architecture

The framework comprises two components: (1) **Evaluation Framework** — for each task, $N$ CoT trajectories are sampled and truncated at a series of token-budget checkpoints $b_1, b_2, \ldots, b_n$; the model is re-prompted with the truncated reasoning to generate a final answer, and the Anytime Index is computed; (2) **Preference Data Prompting (PDP)** — the model generates multiple reasoning trajectories at a fixed budget, trajectory pairs that lead to higher- versus lower-quality intermediate solutions are identified as preference pairs, and these pairs are provided as in-context examples at inference time.

### Key Designs

1. **Anytime Index Metric**:

    - Function: Quantifies reasoning efficiency across different token budgets.
    - Mechanism: Defines $Q_t^* = \max_{i \leq t} Q_i$ as the best quality score achieved up to budget $b_t$. The Anytime Index computes the area under the quality curve via the trapezoidal rule and normalizes it: $\text{AI} = \frac{\sum_{t=1}^{T-1} \frac{Q_t^* + Q_{t+1}^*}{2} \cdot (b_{t+1} - b_t)}{(b_T - b_1) \cdot Q_{\max}}$, with values in $[0, 1]$. A higher value indicates that the model reaches high-quality solutions more rapidly.
    - Design Motivation: Distinguishes "fast-thinking" from "slow-thinking" models — two models with identical final scores may differ substantially if one achieves high quality at a much smaller budget.

2. **Preference Data Prompting (PDP)**:

    - Function: Improves intermediate solution quality at inference time without additional training.
    - Mechanism: (a) Multiple reasoning trajectories are generated for the same task at a fixed token budget; (b) trajectory pairs leading to higher- versus lower-quality intermediate solutions are identified as preference pairs (winner vs. loser); (c) these pairs are provided as in-context examples at inference time. PDP(+) uses only positive examples, while PDP uses both positive and negative examples.
    - Design Motivation: Enables the model to learn from comparisons among its own reasoning outputs without human supervision; as a purely inference-time method, it is applicable to any LLM.

3. **Evaluation Pipeline Design**:

    - Function: Standardizes the evaluation procedure for anytime reasoning.
    - Mechanism: $N$ complete CoT trajectories are sampled per task (up to 4,096 tokens for NaturalPlan; up to 16,384 tokens for AIME/GPQA). Reasoning is truncated at predefined checkpoints, and the truncated prefix is used to re-prompt the model for a final answer. Task-specific quality metrics are applied (constraint satisfaction rate for planning; accuracy for mathematics and QA).
    - Design Motivation: Simulates real-world scenarios in which reasoning is interrupted early, evaluating a model's ability to produce optimal outputs under limited computation.

### Loss & Training

PDP is a purely inference-time method and involves no model training. Preference data is generated automatically through multiple sampling runs and quality comparisons using the model itself.

## Key Experimental Results

### Main Results

**Grok-3 Results**

| Method | NaturalPlan Final | AIME Final | GPQA Final | Overall Final |
|--------|-------------------|------------|------------|---------------|
| Base | 74.7 | 24.0 | 69.8 | 56.2 |
| LEAP | 87.9 | 22.8 | 69.3 | 60.0 |
| PDP | **90.2** | **24.9** | 69.7 | **61.6** |

**Grok-3-mini Results**

| Method | NaturalPlan Final | AIME Final | GPQA Final | Overall Final |
|--------|-------------------|------------|------------|---------------|
| Base | 81.5 | 80.6 | 99.3 | 87.1 |
| PDP | **90.7** | **100.0** | 98.9 | **96.5** |

### Ablation Study

- PDP yields consistent improvements on the Anytime Index as well (e.g., Grok-3-mini improves from 85.4 to 88.7).
- Gains from PDP are more pronounced on reasoning-specialized models (e.g., Grok-3-mini) than on non-reasoning models.
- Using both positive and negative preference pairs (PDP) generally outperforms using positive examples alone (PDP(+)), confirming the value of contrastive information from negative examples.

### Key Findings

- Different model families exhibit markedly distinct reasoning efficiency profiles as measured by the Anytime Index.
- Reasoning-specialized models (e.g., Grok-3-mini) produce high-quality solutions at earlier budget points, resulting in higher Anytime Index scores.
- PDP yields consistent improvements across three task types of diverse nature, validating the generality of the method.
- The Anytime Index reveals efficiency differences among models that cannot be detected by final accuracy alone.

## Highlights & Insights

- The Anytime Index constitutes an important complement to existing LLM reasoning evaluation, filling the gap in "quality trajectory" assessment.
- As a purely inference-time method, PDP improves reasoning efficiency across multiple model families without any training.
- Experiments span multiple model families including Grok, GPT, and LLaMA, lending broad generalizability to the conclusions.
- The concept of anytime reasoning is successfully transferred from classical AI to the LLM domain.

## Limitations & Future Work

- PDP requires generating multiple additional trajectories at inference time to construct preference data, incurring additional inference overhead.
- The quality of preference data depends on the diversity of the model's own samples.
- The choice of checkpoints for the Anytime Index may influence evaluation results.
- Future work could explore using PDP preference data for fine-tuning rather than solely for in-context learning.

## Related Work & Insights

- PDP is complementary to BRPO (budget-aware reasoning policy optimization): BRPO addresses when to stop thinking, while PDP addresses how to reason better within constraints.
- Compared to self-improvement methods such as LEAP, PDP is more specifically designed for budget-constrained settings.
- The Anytime Index can serve as a standard evaluation tool for future research on reasoning efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The Anytime Index concept is original, and PDP is practically useful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple model families, tasks, and metrics.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly defined and the experiments are well organized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)

</div>

<!-- RELATED:END -->
