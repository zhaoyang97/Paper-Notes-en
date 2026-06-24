---
title: >-
  [Paper Note] The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning
description: >-
  [NeurIPS 2025 Workshop on Efficient Reasoning][Reasoning][Overthinking] This paper demonstrates that selecting the shortest solution in Best-of-N sampling for reasoning models is a simple yet counterintuitive and effective heuristic, achieving performance comparable to self-consistency at significantly lower token cost. The underlying mechanism exploits a systematic bias in reasoning models between a "conventional mode" and an "overthinking mode."
tags:
  - "NeurIPS 2025 Workshop on Efficient Reasoning"
  - "Reasoning"
  - "Overthinking"
  - "Shortest-Answer Heuristic"
  - "Best-of-N"
  - "Self-Consistency"
  - "Reasoning Models"
date: 2026-05-08
content_hash: cb167ba1a0d4d98e
---

# The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning

**Conference**: NeurIPS 2025 Workshop on Efficient Reasoning  
**arXiv**: [2510.21067](https://arxiv.org/abs/2510.21067)  
**Code**: None  
**Area**: LLM Reasoning / Test-Time Compute  
**Keywords**: Overthinking, Shortest-Answer Heuristic, Best-of-N, Self-Consistency, Reasoning Models

## TL;DR

This paper demonstrates that selecting the shortest solution in Best-of-N sampling for reasoning models is a simple yet counterintuitive and effective heuristic, achieving performance comparable to self-consistency at significantly lower token cost. The underlying mechanism exploits a systematic bias in reasoning models between a "conventional mode" and an "overthinking mode."

## Background & Motivation

**Background**: Reasoning models (e.g., DeepSeek-R1, Grok-3-mini) substantially improve performance on complex tasks such as mathematics and code through long chain-of-thought (CoT). Parallel test-time compute (Best-of-N) further boosts accuracy by sampling $N$ solutions and selecting the best. Self-consistency is the most widely used heuristic, selecting the most frequently occurring answer.

**Limitations of Prior Work**: Self-consistency requires at least $N \geq 3$ solutions to form a majority vote; it is inapplicable to tasks such as code generation where outputs are not directly comparable; all $N$ solutions must be fully generated, incurring substantial token overhead. Moreover, prior work has identified an "overthinking" phenomenon in reasoning models—generating excessive unnecessary tokens on simple problems, wasting computational resources.

**Key Challenge**: How can one efficiently select a high-quality answer from multiple candidates without relying on complex scoring mechanisms or auxiliary reward models? Existing approaches either require comparable outputs (self-consistency) or require training dedicated verifiers (reward models), entailing high complexity and computational cost.

**Goal**: (1) Provide a simpler and more general Best-of-N selection heuristic; (2) explain why the shortest solution tends to be the correct one; (3) reduce the token cost of parallel inference.

**Key Insight**: Reasoning models develop an implicit strategy during RL training—when the model lacks confidence in a solution's correctness, it dilutes negative rewards through "padding reasoning" (since standard RL algorithms normalize rewards per token), causing incorrect and uncertain solutions to be systematically longer. Selecting the shortest solution is equivalent to selecting the most confident one.

**Core Idea**: Correct solutions in reasoning models tend to be shorter; selecting the shortest solution avoids the tail distribution of overthinking, achieving a Pareto improvement.

## Method

### Overall Architecture

The method is extremely simple: sample $N$ solutions in parallel for the same problem ($N=5$) and select the one with the fewest tokens as the final answer. In parallel inference settings, once the first solution is complete, all remaining unfinished candidates are terminated (since they are necessarily longer), enabling early stopping to save tokens.

### Key Designs

1. **Two-Regime Hypothesis**:

    - Function: Explains why the shortest-solution heuristic is effective.
    - Mechanism: Reasoning models operate in two regimes during solution generation—the "conventional regime," where the model is confident and produces compact, direct reasoning chains that tend to be correct; and the "overthinking regime," where the model is uncertain and extends its output through repeated reasoning, self-correction, and hedging expressions, tending to produce incorrect solutions. The token-count distribution exhibits bimodality or heavy right-skew—conventional-regime solutions concentrate in the shorter region, while the overthinking regime forms a long tail.
    - Design Motivation: This hypothesis provides a unified explanation for multiple observations: (1) correct solutions are on average shorter than incorrect ones; (2) longer solutions exhibit higher density of uncertainty markers; (3) embedding distances cease to grow beyond a critical point.

2. **Critical Point Analysis**:

    - Function: Identifies the location where the overthinking regime begins to dominate.
    - Mechanism: All solutions are sorted by token count, and the mode of the token-count distribution is identified as the critical point—the peak of the conventional-regime distribution and the inflection point beyond which the proportion of overthinking solutions rises significantly. Uncertainty marker frequency and embedding distance trends are analyzed on either side of the critical point, revealing a clear trend break: before the critical point, uncertainty increases monotonically with length; after it, the trend breaks down.
    - Design Motivation: Critical point analysis provides quantifiable empirical support for the two-regime hypothesis.

3. **Pareto Improvement via Early Stopping**:

    - Function: Further reduces computation in parallel inference.
    - Mechanism: In synchronous token-generation settings, all remaining candidates are terminated as soon as the shortest solution completes. Compared to self-consistency, which must wait for all $N$ solutions to finish, the shortest-solution heuristic only requires waiting for the fastest-completing candidate. It yields meaningful gains at $N=2$, whereas self-consistency requires $N \geq 3$, making it well-suited for cost-sensitive scenarios.
    - Design Motivation: Pareto curve analysis clearly demonstrates that, under the same token budget, the shortest-solution heuristic achieves higher or comparable accuracy relative to self-consistency.

## Key Experimental Results

### Main Results ($N=5$, 400 AIME problems + LiveCodeBench v5)

| Model | Method | AIME Accuracy | LiveCodeBench Accuracy |
|-------|--------|--------------|------------------------|
| DeepSeek-R1 | Single-sample mean | 85.0% | 76.5% |
| DeepSeek-R1 | Shortest solution | **89.0%** | **79.2%** |
| DeepSeek-R1 | Self-consistency | 89.2% | *N/A* |
| DeepSeek-R1 | Longest solution | 78.2% | 76.5% |
| Qwen3-32B | Single-sample mean | 89.5% | 78.6% |
| Qwen3-32B | Shortest solution | **92.5%** | **79.5%** |
| Qwen3-32B | Self-consistency | 93.0% | *N/A* |
| Qwen3-32B | Longest solution | 85.5% | 76.8% |

### Ablation Study (Uncertainty Marker Density)

| Model | % of cases where longer solutions have higher uncertainty (AIME) | LiveCodeBench |
|-------|------------------------------------------------------------------|---------------|
| DeepSeek-R1 | 67.0% | 67.5% |
| Grok-3-mini | 67.4% | 63.7% |
| Qwen3-32B | 58.2% | 65.8% |

### Key Findings

- **Shortest solution ≈ self-consistency at substantially lower compute cost**: The gap on AIME is under 1%; on LiveCodeBench, self-consistency is inapplicable due to incomparable outputs, while the shortest-solution heuristic remains effective.
- **Selecting the longest solution performs worse than single sampling**: This further validates the overthinking hypothesis—longer solutions are systematically less accurate.
- **Significant improvement already at $N=2$**: The shortest-solution heuristic can discriminate within a pair of candidates, whereas self-consistency requires at least $N=3$.
- **Trend break at the critical point**: Uncertainty marker density and embedding distance exhibit a clear trend change near the mode, supporting the two-regime hypothesis.
- **Token distribution of shortest solutions is more concentrated**: The peak position is the same as for longest solutions, but the long tail is absent, indicating that the heuristic operates by truncating the overthinking tail.

## Highlights & Insights

- **Triumph of minimalist methodology**: The entire method can be implemented in a single line of code (`argmin(lengths)`), yet matches the performance of complex self-consistency approaches and is applicable to a broader range of tasks (e.g., code generation with incomparable outputs). This exemplifies the principle that "good heuristics stem from deep understanding of problem structure."
- **Insightful mechanistic account of overthinking's training origins**: Overthinking is attributed to per-token normalization of negative rewards in GRPO/PPO—models learn to "pad" outputs under uncertainty to dilute penalties. This mechanistic explanation is more principled than simply noting that models generate redundant tokens.
- **Practical value of early stopping**: In parallel inference deployment, the shortest-solution heuristic naturally supports early termination, offering direct engineering value for latency-sensitive applications such as real-time coding assistants.

## Limitations & Future Work

- Validation is limited to mathematics (AIME) and code (LiveCodeBench); the relationship between length and correctness in natural language reasoning, commonsense reasoning, and other tasks remains unexplored.
- Synchronous parallel generation (all candidates start simultaneously) is assumed; the early stopping mechanism requires adaptation for asynchronous settings.
- Critical point identification relies on mode estimation, which may be unreliable for irregular distributions.
- Combining the shortest-solution heuristic with other selection strategies (e.g., reward models) is not explored.
- As a workshop paper, ablation studies are limited in scope (e.g., systematic analysis across different temperatures and values of $N$).

## Related Work & Insights

- **vs. Self-consistency (Wang et al. 2023)**: Self-consistency selects answers by majority vote, requiring comparable outputs and at least 3 samples; the shortest-solution heuristic is more general and effective at $N=2$.
- **vs. Chen et al. 2025 ("Don't Think That Much")**: That work focuses on overthinking waste on simple problems; this paper complements it by characterizing the systematic length–correctness bias induced by overthinking on difficult problems.
- **vs. Reward Model approaches (Zhang et al. 2025)**: Methods such as Gen-RM require training dedicated verifiers; the shortest-solution heuristic requires no additional models.

## Rating

- Novelty: ⭐⭐⭐⭐ Counterintuitive simple heuristic + theoretical account via the two-regime overthinking hypothesis
- Experimental Thoroughness: ⭐⭐⭐ Workshop paper; task coverage and ablations are limited
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure; Pareto curve figures are highly intuitive
- Value: ⭐⭐⭐⭐ Directly applicable to reasoning model deployment; inspires reward design improvements in RL training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)

</div>

<!-- RELATED:END -->
