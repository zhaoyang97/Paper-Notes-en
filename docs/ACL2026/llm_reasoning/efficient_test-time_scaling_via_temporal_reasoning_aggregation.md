---
title: >-
  [Paper Note] Efficient Test-Time Scaling via Temporal Reasoning Aggregation
description: >-
  [ACL 2026][LLM Reasoning][Test-time scaling] The TRACE framework is proposed to determine reasoning convergence by aggregating two complementary signals—multi-step answer consistency and confidence trajectories—within a…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Test-time scaling"
  - "early exit strategies"
  - "reasoning convergence"
  - "multi-step aggregation"
  - "overthinking"
date: 2026-05-08
content_hash: 69d3e34ae88a22d6
---

# Efficient Test-Time Scaling via Temporal Reasoning Aggregation

**Conference**: ACL 2026  
**arXiv**: [2604.17304](https://arxiv.org/abs/2604.17304)  
**Code**: [https://github.com/qianfantianyuzhouzhou/TRACE](https://github.com/qianfantianyuzhouzhou/TRACE)  
**Area**: LLM Inference Efficiency  
**Keywords**: Test-time scaling, early exit strategies, reasoning convergence, multi-step aggregation, overthinking

## TL;DR

The TRACE framework is proposed to determine reasoning convergence by aggregating two complementary signals—multi-step answer consistency and confidence trajectories—within a sliding window. This enables training-free dynamic early exit, reducing token usage by 25-30% with only a 1-2% drop in accuracy.

## Background & Motivation

**Background**: Test-time scaling enhances LLM reasoning performance by increasing inference-time computation (extending chains of thought or searching multiple paths). However, this leads to significant unnecessary token generation—models often continue reasoning after reaching the correct answer (the overthinking phenomenon).

**Limitations of Prior Work**: Existing dynamic early exit methods primarily rely on single-step confidence signals to decide when to terminate reasoning. Research indicates that single-step confidence is unreliable in multi-step reasoning, as it reflects single-step certainty rather than stability across steps. For instance, a model may assign high confidence to an incorrect intermediate step, leading to premature termination.

**Key Challenge**: Premature termination leads to incorrect outputs, while late termination wastes resources. Single-step confidence fails to distinguish "true reasoning convergence" from "transient high-confidence erroneous steps." Reasoning convergence is inherently a temporal phenomenon that requires stability signals across multiple steps.

**Goal**: Design an early exit strategy based on multi-step evidence aggregation to provide a more reliable judgment of reasoning convergence than single-step confidence.

**Key Insight**: Inspired by self-consistency methods—where multiple reasoning paths yielding the same answer suggest correctness—this approach generalizes the idea from multiple samplings to multiple steps within a single inference.

**Core Idea**: Simultaneously track two complementary signals within a sliding window: (1) Answer Consistency—whether the predicted answer remains consistent over multiple steps; (2) Confidence Trajectory—whether confidence evolves stably over time. These are combined to judge if reasoning has truly converged.

## Method

### Overall Architecture

TRACE maintains a sliding window of size $k$ during the autoregressive inference process, covering the most recent $k$ reasoning steps. At each step, TRACE calculates the Answer Consistency Score (ACS) and Confidence Trajectory Score (CTS) within the window. These are combined into a weighted unified stability score; if it exceeds a threshold $\tau$, inference is terminated. TRACE is training-free and can be directly applied to off-the-shelf LLMs.

### Key Designs

1. **Answer Consistency Score (ACS)**:

    - **Function**: Measures the persistence of the predicted answer across recent reasoning steps.
    - **Mechanism**: At each reasoning step, a candidate final answer is induced from the current context using a lightweight auxiliary prompt. ACS is defined as the frequency of the candidate answer $a$ within the sliding window: $\text{ACS}(a) = \text{count}(a) / k$. When reasoning converges, the correct answer appears consistently, resulting in a high ACS.
    - **Design Motivation**: Inspired by self-consistency, the multi-step persistence of an answer is a strong signal of reasoning convergence.

2. **Confidence Trajectory Score (CTS)**:

    - **Function**: Tracks the temporal evolution of model confidence.
    - **Mechanism**: At each step, the confidence $c = 1 - \frac{1}{n}\sum_j \tilde{H}(p_j)$ is calculated using normalized entropy $\tilde{H}$ for the candidate answer. CTS is defined as the average confidence of candidate answer $a$ over the steps where it appeared: $\text{CTS}(a) = \frac{1}{\text{count}(a)}\sum_{t \in \mathcal{T}(a)} c_t$. CTS distinguishes between sustained high confidence (convergence signal) and sporadic high confidence (noise).
    - **Design Motivation**: While single-step confidence is unreliable, an answer consistently receiving high confidence across multiple steps is more likely to represent true convergence.

3. **Joint Early Exit Decision**:

    - **Function**: Synthesizes both signals to make a reliable termination judgment.
    - **Mechanism**: A unified stability score $S(a) = \alpha \cdot \text{ACS}(a) + (1-\alpha) \cdot \text{CTS}(a)$ is computed, and the candidate answer $a^\star$ with the highest score is selected. Inference terminates and outputs $a^\star$ when $S(a^\star) > \tau$. The parameter $\alpha$ controls the relative contribution of each signal.
    - **Design Motivation**: ACS and CTS provide complementary perspectives—consistency of the answer versus model certainty. Their combination is more robust than either signal alone.

### Loss & Training

TRACE is a training-free inference-time method applied directly to existing models. It was evaluated on Qwen3-8B and DeepSeek-R1-Distill-Llama-8B using benchmarks including OlympiadBench, MATH500, AIME24, AMC23, and AIME25. The hyperparameter $\alpha$ and threshold $\tau$ are tuned via a validation set.

## Key Experimental Results

### Main Results

| Method | Avg. Accuracy | Avg. Token Consumption Rate | Description |
|------|----------|----------------|------|
| Vanilla (Full Inference) | Baseline | 100% | Complete reasoning |
| Single-step Confidence | -11% | ~60% | Severe accuracy drop due to premature exit |
| TRACE | -1~2% | 70-75% | Optimal trade-off |

### Ablation Study

| Signal Combination | Effect | Description |
|---------|------|------|
| ACS Only | Moderate | Consistent answers but potentially low confidence |
| CTS Only | Moderate | High confidence but potentially inconsistent answers |
| ACS + CTS | Optimal | Joint judgment via complementary signals |

### Key Findings

- Single-step confidence early exit achieves only 0.44 average accuracy on difficult benchmarks (vs. 0.55 for full inference), confirming the unreliability of single-step signals.
- TRACE reduces token usage by 25-30% with only a 1-2% accuracy drop, significantly outperforming existing dynamic reasoning methods.
- Compared to the strongest early exit baselines, TRACE improves average accuracy by 2-4 points at comparable or lower token budgets.
- The choice of sliding window size $k$ affects sensitivity—too small lead to unstable signals, while too large introduces detection latency.

## Highlights & Insights

- **The paradigm shift from "single-step judgment" to "multi-step aggregation" is compelling**: The case in Figure 1 clearly demonstrates how single-step high confidence can mislead early exits, whereas TRACE avoids misjudgment by observing multi-step consistency.
- **Clever answer induction design**: Inducing candidate answers at each step using lightweight prompts allows for real-time monitoring of internal reasoning states without interrupting the process.
- **Strong practicality of training-free plug-and-play**: Reducing inference costs without requiring model modifications or additional training is highly valuable.

## Limitations & Future Work

- Inducing an answer at every step requires an additional forward pass, introducing some overhead.
- Applicability to non-mathematical reasoning tasks (e.g., code generation, natural language inference) has not been fully verified.
- Threshold $\tau$ and window size $k$ require tuning on a validation set, as settings may vary across datasets.
- TRACE may misjudge scenarios where reasoning truly requires a long chain rather than being a case of overthinking.

## Related Work & Insights

- **vs. Single-step Confidence Early Exit**: Single-step signals are systematically unreliable in multi-step reasoning; TRACE provides more stable convergence signals through temporal aggregation.
- **vs. RL Methods (e.g., Length Penalty Training)**: RL methods require additional training and are sensitive to reward design, whereas TRACE is more flexible as a training-free application.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-step aggregation approach is natural and logical; the ACS+CTS design is clever without being overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, covering 5 math benchmarks, 2 models, various baselines, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly established via experiments, and the method description is concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[ACL 2026\] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)

</div>

<!-- RELATED:END -->
