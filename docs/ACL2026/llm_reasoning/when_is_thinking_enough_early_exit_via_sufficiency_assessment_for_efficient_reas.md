---
title: >-
  [Paper Note] When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Reasoning Efficiency] The DTSR framework is proposed, which detects "reflection signals" (e.g., Wait, Alternatively) during the reasoning process and triggers a self-assessment of "sufficiency"…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reasoning Efficiency"
  - "Early Exit Strategy"
  - "Overthinking"
  - "Metacognition"
  - "Chain-of-Thought (CoT)"
date: 2026-05-08
content_hash: 92b5e0ce571042d2
---

# When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.06787](https://arxiv.org/abs/2604.06787)  
**Code**: To be confirmed  
**Area**: LLM Reasoning  
**Keywords**: Reasoning Efficiency, Early Exit Strategy, Overthinking, Metacognition, Chain-of-Thought (CoT)

## TL;DR

The DTSR framework is proposed, which detects "reflection signals" (e.g., Wait, Alternatively) during the reasoning process and triggers a self-assessment of "sufficiency" to decide whether to terminate reasoning early. This achieves a 28.9%–34.9% reduction in reasoning length on Qwen3 series models with almost no loss in accuracy.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) such as DeepSeek-R1 and Qwen3 have made significant progress in complex reasoning tasks through long Chain-of-Thought (CoT). However, this introduces serious reasoning redundancy—models repeatedly verify and explore alternative paths even for simple problems.

**Limitations of Prior Work**: Existing early exit methods rely on hand-crafted exit criteria—Dynasor-CoT uses consecutive answer consistency (but still requires extra tokens for verification after the correct answer appears), and DEER uses the entropy of intermediate answers as a confidence indicator. These methods face two fundamental issues: (1) Reasoning models exhibit overconfidence, maintaining high confidence even when answers are wrong, making confidence-based judgments unreliable; (2) They are only applicable to short-answer tasks and are infeasible for long-answer scenarios like code generation and open-ended QA.

**Key Challenge**: How to judge whether the reasoning process is "enough" without relying on answer confidence? A method is needed to evaluate the reasoning process itself rather than the correctness of the answer.

**Goal**: Design a universal and reliable early exit framework that determines the exit timing by assessing the sufficiency of the reasoning chain rather than the confidence of the answer.

**Key Insight**: Borrow from human metacognition—humans do not frequently produce intermediate answers to judge whether to stop thinking; instead, they internally evaluate whether the "current thoughts are sufficient to support the final conclusion."

**Core Idea**: Trigger a sufficiency check at reflection signals (e.g., "Wait", "Let me check") of the reasoning model, allowing the model to evaluate the sufficiency of the current reasoning chain from a third-person perspective to reach a final answer.

## Method

### Overall Architecture

DTSR operates in two stages: (1) Reflection Signal Monitoring—detects specific reflection trigger words (e.g., "Wait", "Alternatively", "But") during the generation process, signaling positions where the model is about to begin redundant validation or backtracking; (2) Thought Sufficiency Check—upon detecting a reflection signal, the original question and the current reasoning chain are input into a sufficiency evaluation template, and the model outputs a sufficiency score from 0-100. If the score exceeds a threshold $\tau$ (default 100), `</think>` is appended to terminate reasoning and output the final answer; otherwise, reasoning continues to the next reflection signal. To avoid frequent checks, a minimum token interval $k$ (default 64) is set.

### Key Designs

1.  **Construction of the Reflection Signal Set**:

    - **Function**: Identify potential exit candidates during reasoning.
    - **Mechanism**: By analyzing reasoning trajectories of Qwen3-32B and finding the optimal early exit points (the earliest positions where the model can already answer correctly), it was found that these points are often followed by explicit self-reflection behaviors. A set of reflection signals was constructed, including keywords like "Wait", "Alternatively", "But wait", and "Let me check".
    - **Design Motivation**: Reflection signals mark the boundary where the model shifts from "reasoning" to "verification"—validation after a correct answer has been generated is usually redundant, making these positions natural candidates for early exit.

2.  **Thought Sufficiency Check (Third-Person Self-Evaluation)**:

    - **Function**: Determine if the current reasoning chain is sufficient to reach the correct answer at a reflection signal.
    - **Mechanism**: The question $Q$ and current reasoning chain $T$ are combined into a sufficiency assessment prompt. The model evaluates the sufficiency of the reasoning chain from a "third-person" perspective—evaluating whether the process is complete enough to derive the correct answer rather than assessing "is my answer correct". It outputs a scalar score $s \in [0, 100]$, and terminates if $s \geq \tau$.
    - **Design Motivation**: Evaluating the sufficiency of the reasoning process is more reliable than directly evaluating answer confidence—overconfidence primarily occurs in "self" evaluations, whereas third-person assessment of the reasoning process is more objective. Experiments show third-person assessment accuracy is significantly higher than first-person.

3.  **Minimum Token Interval Control**:

    - **Function**: Avoid redundant checking overhead when reflection signals are dense.
    - **Mechanism**: A requirement of at least $k$ tokens (default 64) between two sufficiency checks; otherwise, the current reflection signal is skipped. $k = 64$ provides optimal reasoning length and latency; $k < 64$ increases check overhead, while $k > 64$ causes the model to miss optimal exit points.
    - **Design Motivation**: Reflection signals often appear consecutively (e.g., "Wait, but let me check"); checking every time creates excessive computational redundancy.

### Loss & Training

DTSR is a training-free method and requires no additional training. The sufficiency check step is inserted only during inference.

## Key Experimental Results

### Main Results

| Method | Model | Overall Acc | Overall Tok | Reduction |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla | Qwen3-8B | 81.9 | 6510 | - |
| DEER | Qwen3-8B | 79.3 | 4532 | -30.4% |
| **DTSR** | Qwen3-8B | **81.0** | **4428** | **-32.0%** |
| Vanilla | Qwen3-14B | 84.4 | 5761 | - |
| DEER | Qwen3-14B | 83.0 | 4367 | -24.2% |
| **DTSR** | Qwen3-14B | **84.8** | **3748** | **-34.9%** |
| Vanilla | Qwen3-32B | 84.7 | 5638 | - |
| DEER | Qwen3-32B | 83.1 | 4318 | -23.4% |
| **DTSR** | Qwen3-32B | **84.6** | **4010** | **-28.9%** |

### Ablation Study

| Configuration | Key Result | Description |
| :--- | :--- | :--- |
| $k=16$ | Latency increases, length similar | Checks are too frequent |
| $k=64$ | Optimal balance point | Default setting |
| $k=256$ | Length increases | Misses optimal exit points |
| $\tau=50$ | Acc drops significantly | Terminates reasoning too early |
| $\tau=100$ | Optimal | High-confidence termination |
| 1st Person | Acc drops | More severe overconfidence |

### Key Findings
- On Qwen3-14B, DTSR even improved accuracy (84.8 vs 84.4), suggesting that removing redundant reasoning can improve results.
- For programming tasks (LiveCodeBench), reasoning length reduction exceeded 50%, as redundant verification is more severe in coding.
- DTSR's inference latency is lower than DEER (MATH-500: 1.9s vs 4.2s) because DEER requires full decoding of intermediate answers for each check, while DTSR only generates a single score.
- The third-person evaluation paradigm (evaluating the reasoning process rather than self-assessment) is significantly superior to the first-person, validating the hypothesis that "overconfidence stems from self-assessment."

## Highlights & Insights

- **Redefining early exit from a metacognitive perspective**: Instead of judging "is the answer right," judging "is the reasoning enough"—this shift in perspective bypasses the overconfidence problem and is more universal (not limited to tasks with standard answers).
- **Reflection signals as natural exit candidates**: Utilizing the reasoning model's own behavioral patterns (reflection trigger words) avoids the need for fixed-interval checks and better aligns with the model's generation logic.
- **Discovery of Third-Person vs. First-Person Evaluation**: Models are more accurate at evaluating the reasoning of others than their own, a finding with broader implications for the field of LLM self-evaluation.

## Limitations & Future Work

- Only validated on the Qwen3 series; reflection signal patterns in other reasoning models (DeepSeek-R1, o1, etc.) may differ.
- The sufficiency check itself requires extra computation—while overall latency decreases, the check overhead might offset savings in some scenarios (where $k$ is very small).
- The optimal value of threshold $\tau$ might change with task difficulty—keeping it fixed at 100 may not be flexible enough.
- Integration with training-based methods has not been explored; the sufficiency check module might perform better if specifically trained.

## Related Work & Insights

- **vs DEER**: Methods based on intermediate answer entropy are affected by overconfidence and limited to short-answer tasks; DTSR evaluates the reasoning process instead of the answer, making it more universal and reliable.
- **vs Dynasor-CoT**: Still requires extra verification tokens after consecutive answer consistency and cannot reach the optimal exit point; DTSR can exit earlier by evaluating directly at reflection signals.
- **vs NoWAIT**: Reduces redundancy by masking reflection tokens but disrupts the model's natural reasoning ability; DTSR preserves full reasoning ability and terminates only at appropriate moments.

## Rating

- Novelty: ⭐⭐⭐⭐ The metacognitive perspective and third-person evaluation are innovative, though the overall framework is intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three model sizes, six datasets, and multi-dimensional ablations provide sufficient evidence.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly described with in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ Practical value for reasoning efficiency; the training-free approach is easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](drp_distilled_reasoning_pruning_with_skill-aware_step_decomposition_for_efficien.md)
- [\[ACL 2026\] Reinforced Efficient Reasoning via Semantically Diverse Exploration](reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)
- [\[ICLR 2026\] Agentified Assessment of Logical Reasoning Agents](../../ICLR2026/llm_reasoning/agentified_assessment_of_logical_reasoning_agents.md)

</div>

<!-- RELATED:END -->
