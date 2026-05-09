---
title: >-
  [Paper Note] Efficient Test-Time Scaling via Temporal Reasoning Aggregation
description: >-
  [ACL 2026][LLM Inference][To be added] To be supplemented after reading the paper in depth
tags:
  - ACL 2026
  - LLM Inference
  - To be added
date: 2026-05-08
content_hash: 69e7d89a44da4d52
---
# Efficient Test-Time Scaling via Temporal Reasoning Aggregation

**Conference**: ACL 2026
**arXiv**: [2604.17304](https://arxiv.org/abs/2604.17304)
**Code**: [https://github.com/qianfantianyuzhouzhou/TRACE](https://github.com/qianfantianyuzhouzhou/TRACE)
**Area**: LLM Reasoning Efficiency
**Keywords**: Test-time scaling, early exit, reasoning convergence, multi-step aggregation, overthinking

## TL;DR

This paper proposes TRACE, a framework that determines reasoning convergence by aggregating two complementary signals within a sliding window — answer consistency across steps and confidence trajectory over time — enabling training-free dynamic early exit that reduces token usage by 25–30% with only a 1–2% accuracy drop.

## Background & Motivation

**State of the Field**: Test-time scaling improves LLM reasoning performance by increasing inference-time computation (extending chain-of-thought or searching multiple paths). However, this leads to substantial unnecessary token generation — models frequently continue reasoning after having already arrived at the correct answer (the overthinking phenomenon).

**Limitations of Prior Work**: Existing dynamic early-exit methods primarily rely on single-step confidence signals to decide when to terminate reasoning. Research has shown that single-step confidence is unreliable in multi-step reasoning — it reflects the certainty of a single step rather than stability across steps. For instance, a model may assign high confidence to an incorrect intermediate step, triggering premature termination.

**Root Cause**: Terminating too early yields incorrect outputs, while terminating too late wastes resources. Single-step confidence cannot distinguish between genuine reasoning convergence and transiently high-confidence erroneous steps. Reasoning convergence is inherently a temporal phenomenon that requires stability signals spanning multiple steps.

**Paper Goals**: To design an early-exit strategy based on multi-step evidence aggregation that provides more reliable convergence judgments than single-step confidence.

**Starting Point**: Inspired by self-consistency — if multiple reasoning paths yield the same answer, that answer is more likely correct. This intuition is extended from multi-path sampling to multiple steps within a single reasoning trajectory.

**Core Idea**: Two complementary signals are tracked simultaneously within a sliding window: (1) answer consistency — whether the predicted answer remains stable across multiple steps; and (2) confidence trajectory — whether confidence evolves stably over time. Both signals are jointly used to determine whether reasoning has genuinely converged.

## Method

### Overall Architecture

TRACE maintains a sliding window of size $k$ covering the most recent $k$ reasoning steps during autoregressive inference. At each step, TRACE computes an Answer Consistency Score (ACS) and a Confidence Trajectory Score (CTS) within the window, combines them with a weighted sum into a unified stability score, and terminates reasoning when this score exceeds threshold $\tau$. No additional training is required; TRACE can be directly applied to off-the-shelf LLMs.

### Key Designs

1. **Answer Consistency Score (ACS)**:

    - **Function**: Measures the persistence of the predicted answer across recent reasoning steps.
    - **Mechanism**: At each reasoning step, a lightweight auxiliary prompt elicits a candidate final answer from the current reasoning context. ACS is defined as the frequency of candidate answer $a$ within the sliding window: $\text{ACS}(a) = \text{count}(a) / k$. When reasoning converges, the correct answer recurs across consecutive steps, producing a high ACS.
    - **Design Motivation**: Inspired by self-consistency, cross-step persistence of an answer is a strong signal of reasoning convergence.

2. **Confidence Trajectory Score (CTS)**:

    - **Function**: Tracks the temporal evolution of model confidence.
    - **Mechanism**: At each step, the confidence of the candidate answer is computed using normalized entropy $\tilde{H}$ as $c = 1 - \frac{1}{n}\sum_j \tilde{H}(p_j)$. CTS is then defined as the average confidence of candidate answer $a$ over the steps at which it appears: $\text{CTS}(a) = \frac{1}{\text{count}(a)}\sum_{t \in \mathcal{T}(a)} c_t$. CTS distinguishes persistently high confidence (a convergence signal) from sporadically high confidence (noise).
    - **Design Motivation**: Single-step confidence is unreliable, but an answer that receives high confidence across multiple steps is more likely to reflect genuine convergence.

3. **Joint Early-Exit Decision**:

    - **Function**: Integrates both signals to make a reliable termination judgment.
    - **Mechanism**: The unified stability score is $S(a) = \alpha \cdot \text{ACS}(a) + (1-\alpha) \cdot \text{CTS}(a)$. The candidate answer with the highest score is selected as $a^\star$. Reasoning terminates and $a^\star$ is output when $S(a^\star) > \tau$. The parameter $\alpha$ controls the relative contribution of each signal.
    - **Design Motivation**: ACS and CTS provide complementary perspectives — ACS captures whether answers are consistent, while CTS captures whether the model is confident. Their joint judgment is more robust than either signal alone.

### Loss & Training

TRACE is a training-free inference-time method applied directly to off-the-shelf models. Evaluation is conducted on Qwen3-8B and DeepSeek-R1-Distill-Llama-8B across the benchmarks OlympiadBench, MATH500, AIME24, AMC23, and AIME25. Hyperparameters $\alpha$ and threshold $\tau$ are tuned on a validation set.

## Key Experimental Results

### Main Results

| Method | Avg. Accuracy | Avg. Token Usage | Notes |
|--------|--------------|-----------------|-------|
| Vanilla (full inference) | baseline | 100% | Complete reasoning |
| Single-step confidence early exit | −11% | ~60% | Severe degradation from premature termination |
| TRACE | −1~2% | 70–75% | Optimal trade-off |

### Ablation Study

| Signal Combination | Performance | Notes |
|-------------------|-------------|-------|
| ACS only | Moderate | Answers consistent but confidence may be low |
| CTS only | Moderate | Confidence high but answers may be inconsistent |
| ACS + CTS | Best | Complementary signals jointly decide |

### Key Findings

- Single-step confidence early exit achieves an overall accuracy of only 0.44 on hard benchmarks (vs. 0.55 for full inference), confirming the unreliability of single-step signals.
- TRACE reduces token usage by 25–30% with only a 1–2% accuracy drop, significantly outperforming existing dynamic reasoning methods.
- Compared to the strongest early-exit baseline, TRACE improves average accuracy by 2–4 points under comparable or lower token budgets.
- The choice of sliding window size $k$ affects sensitivity — too small yields unstable signals, while too large introduces detection latency.

## Highlights & Insights

- **The paradigm shift from single-step judgment to multi-step aggregation is compelling**: Figure 1 clearly illustrates how single-step high confidence can mislead early exit, while TRACE avoids such errors by observing cross-step consistency.
- **The answer elicitation design is elegant**: A lightweight prompt elicits candidate answers at each step without interrupting the reasoning process, enabling real-time monitoring of intermediate reasoning states.
- **The training-free, plug-and-play nature is highly practical**: No model modification or additional training is required, directly reducing inference cost.

## Limitations & Future Work

- Answer elicitation at each step requires additional forward passes, introducing non-trivial overhead.
- Applicability to non-mathematical reasoning tasks (e.g., code generation, natural language inference) has not been thoroughly validated.
- Threshold $\tau$ and window size $k$ require tuning on a validation set and may need different configurations across datasets.
- When reasoning genuinely requires long chains (rather than exhibiting overthinking), TRACE may incorrectly judge the process as having converged.

## Related Work & Insights

- **vs. single-step confidence early exit**: Single-step signals are systematically unreliable in multi-step reasoning; TRACE provides more stable convergence signals through temporal aggregation.
- **vs. RL-based methods (e.g., length-penalty training)**: RL methods require additional training and are sensitive to reward design, whereas TRACE is training-free and directly applicable, offering greater flexibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-step aggregation idea is natural and well-motivated; the ACS+CTS design is elegant, though not technically complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five mathematical benchmarks, two models, multiple baselines, and detailed ablations — very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly established through experiments; method description is concise.
**Code**: To be confirmed
**Area**: llm_reasoning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)

<!-- RELATED:END -->
