---
title: >-
  [Paper Note] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data
description: >-
  [ACL 2026][LLM Reasoning][Budget-aware reasoning] This paper proposes a budget-aware anytime reasoning framework and the Anytime Index metric to quantify the trade-off between reasoning quality and efficiency of LLMs und…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Budget-aware reasoning"
  - "Anytime Index"
  - "preference data prompting"
  - "test-time scaling"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: e770306aa07e19e9
---

# Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.11038](https://arxiv.org/abs/2601.11038)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Budget-aware reasoning, Anytime Index, preference data prompting, test-time scaling, reasoning efficiency

## TL;DR

This paper proposes a budget-aware anytime reasoning framework and the Anytime Index metric to quantify the trade-off between reasoning quality and efficiency of LLMs under limited token budgets. It also designs a reasoning-time self-improvement method (PDP) based on LLM-synthesized preference data, which significantly improves the quality of intermediate and final solutions in planning, mathematics, and science QA tasks.

## Background & Motivation

**Background**: LLMs have demonstrated powerful reasoning capabilities through methods such as Chain-of-Thought (CoT) and Tree-of-Thoughts. Test-time scaling has become an important means to improve reasoning performance, but existing methods usually assume unlimited computational resources and only evaluate the final answer quality.

**Limitations of Prior Work**: (1) Many real-world scenarios face strict computational or latency budget constraints, where even partial solutions are more useful than no solution (e.g., an incomplete but feasible travel plan); (2) Existing methods lack a principled way to evaluate the trajectory of reasoning quality as tokens grow; (3) Budget-aware techniques (such as BRPO) focus on "when to stop thinking" but not on "how to think better under constraints."

**Key Challenge**: Real-world reasoning tasks require producing optimal intermediate solutions within a finite budget, but current evaluation and optimization frameworks only focus on final answers, ignoring the efficiency of the reasoning trajectory.

**Goal**: (1) Establish a framework and metrics to evaluate LLM reasoning efficiency under different token budgets; (2) Provide a method to improve the quality of budget-aware reasoning.

**Key Insight**: Borrowing the concept of anytime algorithms from classical AI, reasoning is viewed as a quality-increasing process as the token budget increases.

**Core Idea**: Quantify reasoning efficiency by truncating reasoning trajectories and evaluating solution quality at various checkpoints, and use self-generated reasoning comparisons to construct preference data as in-context examples to improve the quality of intermediate solutions.

## Method

### Overall Architecture

The framework is divided into two parts: (1) **Evaluation Framework**—sample $N$ CoT trajectories for each task, truncate them at a series of token budget checkpoints $b_1, b_2, \ldots, b_n$, re-prompt the model to generate final answers based on the truncated reasoning, and calculate the Anytime Index; (2) **Preference Data Prompting (PDP)**—the model generates multiple reasoning trajectories at a fixed budget, identifies trajectory pairs leading to higher-quality intermediate solutions as preference pairs, and uses them as in-context examples during inference.

### Key Designs

1.  **Anytime Index Metric**:

    - **Function**: Quantifies the reasoning efficiency of the model under different token budgets.
    - **Mechanism**: Define $Q_t^* = \max_{i \leq t} Q_i$ as the optimal quality score up to budget $b_t$. Anytime Index uses the trapezoidal rule to calculate the Area Under the quality Curve and normalizes it: $\text{AI} = \frac{\sum_{t=1}^{T-1} \frac{Q_t^* + Q_{t+1}^*}{2} \cdot (b_{t+1} - b_t)}{(b_T - b_1) \cdot Q_{\max}}$, with a range of $[0, 1]$. A higher value indicates the model approaches a high-quality solution faster.
    - **Design Motivation**: Distinguishes between "fast thinking" and "slow thinking" models—two models may have the same final score, but if one reaches high quality at a small budget, its Anytime Index is higher.

2.  **Preference Data Prompting (PDP)**:

    - **Function**: Improves intermediate solution quality during reasoning without additional training.
    - **Mechanism**: (a) Generate multiple reasoning trajectories for the same task at a fixed token budget; (b) identify trajectory pairs that lead to higher/lower quality intermediate solutions to form preference pairs (winner vs. loser); (c) use preference pairs as in-context examples provided to the model during reasoning. PDP(+) use only positive examples, while PDP uses both positive and negative examples.
    - **Design Motivation**: Allows the model to learn from its own reasoning comparisons without human supervision; as a reasoning-time method, it can be applied to any LLM.

3.  **Evaluation Pipeline Design**:

    - **Function**: Standardizes the evaluation process of anytime reasoning.
    - **Mechanism**: Sample $N$ full CoT trajectories for each task (up to 4096 tokens for NaturalPlan, up to 16384 tokens for AIME/GPQA). Truncate reasoning at preset checkpoints and use the truncated reasoning as a prefix to re-prompt the model to generate answers. Use task-specific quality metrics (constraint satisfaction rate for planning, accuracy for math/QA).
    - **Design Motivation**: Simulates real-world scenarios where reasoning is interrupted early, evaluating the model's ability to output optimal results under limited computation.

### Loss & Training

PDP is a pure reasoning-time method and does not involve model training. Preference data is automatically generated through the model's own multiple sampling and quality comparisons.

## Key Experimental Results

### Main Results

**Grok-3 Results**

| Method | NaturalPlan Final | AIME Final | GPQA Final | Overall Final |
|--------|-------------------|------------|------------|---------------|
| Base   | 74.7              | 24.0       | 69.8       | 56.2          |
| LEAP   | 87.9              | 22.8       | 69.3       | 60.0          |
| PDP    | **90.2**          | **24.9**   | 69.7       | **61.6**      |

**Grok-3-mini Results**

| Method | NaturalPlan Final | AIME Final | GPQA Final | Overall Final |
|--------|-------------------|------------|------------|---------------|
| Base   | 81.5              | 80.6       | 99.3       | 87.1          |
| PDP    | **90.7**          | **100.0**  | 98.9       | **96.5**      |

### Ablation Study

- PDP also brings consistent improvements in Anytime Index (e.g., Grok-3-mini improved from 85.4 to 88.7).
- The improvement of PDP is more significant on reasoning-oriented models (e.g., Grok-3-mini) than on non-reasoning models.
- Positive and negative preference pairs (PDP) are generally better than positive examples only (PDP(+)), indicating that the contrastive information of negative examples is valuable.

### Key Findings

- Different model families show distinct reasoning efficiency characteristics on Anytime Index.
- Reasoning models (e.g., Grok-3-mini) can produce high-quality solutions at earlier budget points, resulting in a higher Anytime Index.
- PDP brings consistent improvements across three different types of tasks, verifying the generalizability of the method.
- Anytime Index reveals efficiency differences between models that cannot be discovered through final accuracy alone.

## Highlights & Insights

- Anytime Index is an important supplement to LLM reasoning evaluation, filling the gap in "quality trajectory" assessment.
- PDP, as a pure reasoning-time method, improves the reasoning efficiency of various models without training.
- The experiment covers multiple model families such as Grok, GPT, and LLaMA, and the conclusions are widely applicable.
- The concept of "anytime reasoning" has been successfully migrated from classical AI to the LLM field.

## Limitations & Future Work

- PDP requires generating multiple extra trajectories during inference to construct preference data, which increases inference overhead.
- The quality of preference data depends on the model's own sampling diversity.
- The checkpoint settings of Anytime Index may affect the evaluation results.
- Future work could explore using PDP preference data for fine-tuning rather than just for in-context learning.

## Related Work & Insights

- Complementary to BRPO (Budget-aware Reasoning Performance Optimization): BRPO focuses on when to stop, while PDP focuses on how to reason better under constraints.
- Compared with self-improvement methods like LEAP, PDP is more specifically designed for budget-constrained scenarios.
- Anytime Index can serve as a standard evaluation tool for future reasoning efficiency research.

## Rating

- Novelty: ⭐⭐⭐⭐ The Anytime Index concept is novel, and the PDP method is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple model families, multiple tasks, and multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly defined and the experiments are well-organized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](coact_co-active_llm_preference_learning_with_human-ai_synergy.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)

</div>

<!-- RELATED:END -->
