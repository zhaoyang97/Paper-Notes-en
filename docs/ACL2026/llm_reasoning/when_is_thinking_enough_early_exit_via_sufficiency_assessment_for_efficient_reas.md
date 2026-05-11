---
title: >-
  [Paper Note] When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Reasoning efficiency] This paper proposes the DTSR framework, which detects "reflection signals" (e.g., *Wait*…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reasoning efficiency"
  - "early exit"
  - "overthinking"
  - "metacognition"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 76dbb2017da820d1
---

# When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.06787](https://arxiv.org/abs/2604.06787)
**Code**: To be confirmed
**Area**: LLM Reasoning
**Keywords**: Reasoning efficiency, early exit, overthinking, metacognition, chain-of-thought

## TL;DR

This paper proposes the DTSR framework, which detects "reflection signals" (e.g., *Wait*, *Alternatively*) during the reasoning process and triggers a self-assessment of the current reasoning chain's "sufficiency" at those positions to determine whether to exit early. DTSR achieves 28.9%–34.9% reasoning length reduction on the Qwen3 model series with negligible accuracy loss.

## Background & Motivation

**State of the Field**: Large reasoning models (LRMs) such as DeepSeek-R1 and Qwen3 have achieved remarkable progress on complex reasoning tasks through long chain-of-thought (CoT), but this comes at the cost of severe reasoning redundancy—even simple problems trigger repeated verification and exploration of alternatives.

**Limitations of Prior Work**: Existing early-exit methods rely on manually designed exit criteria. Dynasor-CoT uses consecutive answer consistency as the exit signal (yet still requires extra tokens for verification after the correct answer appears), while DEER uses the entropy of intermediate answers as a confidence indicator. These approaches share two fundamental problems: (1) reasoning models exhibit overconfidence—confidence remains high even when answers are incorrect, rendering confidence-based judgments unreliable; (2) they are only applicable to tasks with short-answer formats and cannot generalize to code generation or open-ended QA.

**Root Cause**: How can one determine whether the reasoning process is already "sufficient" without relying on answer confidence? A method that evaluates the reasoning process itself, rather than answer correctness, is needed.

**Paper Goals**: Design a general and reliable early-exit framework that decides the exit timing by assessing the sufficiency of the reasoning chain rather than the confidence of the answer.

**Starting Point**: Drawing on human metacognition—humans do not frequently produce intermediate answers to judge when to stop thinking; instead, they internally assess whether "the current line of thought is sufficient to support the final conclusion."

**Core Idea**: Trigger a sufficiency check at reflection signals in the reasoning process (e.g., "Wait," "Let me check"), prompting the model to evaluate from a third-person perspective whether the current reasoning chain is sufficient to derive the final answer.

## Method

### Overall Architecture

DTSR operates in two stages: (1) **Reflection signal monitoring**—detecting specific reflection trigger words (e.g., "Wait," "Alternatively," "But") during generation, as these positions signal that the model is about to begin redundant verification or backtracking; (2) **Thought sufficiency checking**—upon detecting a reflection signal, the original question and the current reasoning chain are fed into a sufficiency assessment template, prompting the model to output a sufficiency score $s \in [0, 100]$. If $s$ exceeds a threshold $\tau$ (default 100), `</think>` is appended to terminate reasoning and the final answer is generated; otherwise, reasoning continues to the next reflection signal. A minimum token interval $k$ (default 64) is enforced between consecutive sufficiency checks to avoid frequent triggering.

### Key Designs

1. **Construction of the Reflection Signal Set**

    - **Function**: Identify candidate exit points during the reasoning process.
    - **Mechanism**: By analyzing the reasoning trajectories of Qwen3-32B to locate optimal early-exit points (the earliest positions at which the model can already answer correctly), it is observed that explicit self-reflective behaviors follow these points. This motivates the construction of a reflection signal set containing keywords such as "Wait," "Alternatively," "But wait," and "Let me check."
    - **Design Motivation**: Reflection signals mark the boundary at which the model transitions from "reasoning" to "verification." Verification that occurs after a correct answer has already been produced is typically redundant, making these positions natural early-exit candidates.

2. **Thought Sufficiency Check (Third-Person Self-Assessment)**

    - **Function**: Determine at each reflection signal whether the current reasoning chain is sufficient to produce a correct answer.
    - **Mechanism**: Question $Q$ and the current reasoning chain $T$ are composed into a sufficiency assessment prompt, asking the model to evaluate the reasoning chain from a "third-person" perspective—that is, the model assesses not "is my answer correct?" but "is this reasoning process complete enough to derive the correct answer?" The output is a scalar score $s \in [0, 100]$; the process terminates when $s \geq \tau$.
    - **Design Motivation**: Evaluating the sufficiency of the reasoning process is more reliable than directly assessing answer confidence. Overconfidence manifests primarily in "self" evaluation, whereas third-person assessment of the reasoning process is considerably more objective. Experiments confirm that third-person assessment achieves significantly higher accuracy than first-person assessment.

3. **Minimum Token Interval Control**

    - **Function**: Avoid redundant check overhead when reflection signals appear in rapid succession.
    - **Mechanism**: At least $k$ tokens (default 64) must be generated between two consecutive sufficiency checks; otherwise, the current reflection signal is skipped. $k = 64$ yields the optimal trade-off between reasoning length and latency; $k < 64$ increases check overhead, while $k > 64$ causes missed optimal exit points.
    - **Design Motivation**: Reflection signals frequently appear consecutively (e.g., "Wait, but let me check"), and checking at every occurrence would incur substantial redundant computation.

### Loss & Training

DTSR is a training-free method requiring no additional training. The sufficiency checking step is inserted solely at inference time.

## Key Experimental Results

### Main Results

| Method | Model | Overall Acc | Overall Tok | Length Reduction |
|--------|-------|------------|------------|-----------------|
| Vanilla | Qwen3-8B | 81.9 | 6510 | — |
| DEER | Qwen3-8B | 79.3 | 4532 | −30.4% |
| **DTSR** | Qwen3-8B | **81.0** | **4428** | **−32.0%** |
| Vanilla | Qwen3-14B | 84.4 | 5761 | — |
| DEER | Qwen3-14B | 83.0 | 4367 | −24.2% |
| **DTSR** | Qwen3-14B | **84.8** | **3748** | **−34.9%** |
| Vanilla | Qwen3-32B | 84.7 | 5638 | — |
| DEER | Qwen3-32B | 83.1 | 4318 | −23.4% |
| **DTSR** | Qwen3-32B | **84.6** | **4010** | **−28.9%** |

### Ablation Study

| Configuration | Key Result | Note |
|---------------|-----------|------|
| $k=16$ | Increased latency, comparable length | Checks too frequent |
| $k=64$ | Optimal trade-off | Default setting |
| $k=256$ | Increased length | Misses optimal exit points |
| $\tau=50$ | Significant accuracy drop | Premature termination |
| $\tau=100$ | Optimal | High-confidence termination |
| First-person assessment | Accuracy drop | Overconfidence more severe |

### Key Findings

- On Qwen3-14B, DTSR even improves accuracy (84.8 vs. 84.4), indicating that removing redundant reasoning can actually benefit performance.
- On coding tasks (LiveCodeBench), reasoning length is reduced by over 50%, as redundant verification is particularly prevalent in code generation.
- DTSR incurs lower inference latency than DEER (MATH-500: 1.9 s vs. 4.2 s), since DEER requires fully decoding an intermediate answer at each check, whereas DTSR only needs to generate a single score.
- The third-person assessment paradigm (evaluating the reasoning process rather than self-assessment) substantially outperforms first-person assessment, validating the authors' hypothesis that overconfidence originates from self-evaluation.

## Highlights & Insights

- **Reframing the early-exit problem from a metacognitive perspective**: The question shifts from "is the answer correct?" to "is the reasoning sufficient?"—this perspective change circumvents the overconfidence problem and is more general (not restricted to tasks with ground-truth answers).
- **Reflection signals as natural exit candidates**: Leveraging the behavioral patterns of reasoning models (reflection trigger words) eliminates the need for fixed-interval checking mechanisms, aligning more naturally with the model's generation dynamics.
- **Third-person vs. first-person assessment**: Models evaluate others' reasoning more accurately than their own, a finding with broader implications for the LLM self-evaluation literature.

## Limitations & Future Work

- Validation is limited to the Qwen3 model series; reflection signal patterns may differ in other reasoning models (e.g., DeepSeek-R1, o1).
- The sufficiency check itself incurs additional computation—although overall latency decreases, check overhead may offset the savings in scenarios with very small $k$.
- The optimal value of $\tau$ may vary with task difficulty; fixing it at 100 may lack flexibility.
- Integration with training-based methods remains unexplored; a trained sufficiency checking module could potentially yield further improvements.

## Related Work & Insights

- **vs. DEER**: DEER relies on intermediate answer entropy, is susceptible to overconfidence, and is limited to short-answer tasks. DTSR evaluates the reasoning process rather than the answer, making it more general and reliable.
- **vs. Dynasor-CoT**: Dynasor-CoT requires additional verification tokens even after consecutive answer consistency is achieved, preventing it from reaching the optimal exit point. DTSR directly assesses sufficiency at reflection signals, enabling earlier termination.
- **vs. NoWAIT**: NoWAIT reduces redundancy by masking reflection tokens, but disrupts the model's natural reasoning capability. DTSR preserves full reasoning capacity and only terminates at appropriate moments.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The metacognitive perspective and third-person assessment are original, though the overall framework is relatively intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three model scales, six datasets, and multi-dimensional ablations provide solid empirical support.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and method are clearly described; experimental analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Practically valuable for reasoning efficiency; the training-free approach facilitates easy deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Reinforced Efficient Reasoning via Semantically Diverse Exploration](reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)

</div>

<!-- RELATED:END -->
