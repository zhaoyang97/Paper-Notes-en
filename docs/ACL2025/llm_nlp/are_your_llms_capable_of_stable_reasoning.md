---
title: >-
  [Paper Note] Are Your LLMs Capable of Stable Reasoning?
description: >-
  [ACL 2025][LLM (Other)][G-Pass@k] Proposes the G-Pass@k evaluation metric and the LiveMathBench dynamic benchmark to comprehensively evaluate the reasoning capabilities of LLMs from two dimensions: "performance ceiling" and "stability", revealing that current LLMs still have significant room for improvement in reasoning consistency.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "G-Pass@k"
  - "Reasoning Stability"
  - "LiveMathBench"
  - "Evaluation Metrics"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 40ca6ac7031e0caa
---

# Are Your LLMs Capable of Stable Reasoning?

**Conference**: ACL 2025  
**arXiv**: [2412.13147](https://arxiv.org/abs/2412.13147)  
**Code**: [https://github.com/open-compass/GPassK](https://github.com/open-compass/GPassK)  
**Area**: LLM Evaluation / Mathematical Reasoning  
**Keywords**: G-Pass@k, Reasoning Stability, LiveMathBench, Evaluation Metrics, Mathematical Reasoning

## TL;DR

Proposes the G-Pass@k evaluation metric and the LiveMathBench dynamic benchmark to comprehensively evaluate the reasoning capabilities of LLMs from two dimensions: "performance ceiling" and "stability", revealing that current LLMs still have significant room for improvement in reasoning consistency.

## Background & Motivation

Current LLMs have made significant progress in complex tasks such as mathematical reasoning, but **there is a notable gap between benchmark scores and real-world performance**. The authors argue that the root of this gap lies in the inadequacy of existing evaluation protocols and metrics, which fail to fully capture the true capabilities of LLMs.

The Key Challenge of existing evaluation methods lies in:

**Greedy decoding only considers the "best single response"**: Pass@1 only reflects the single-trial performance under greedy decoding, failing to measure whether the model can "stably" produce the correct answer. A model might occasionally get the correct answer but exhibit highly inconsistent performance across multiple samplings.

**Pass@k only considers "at least one correct answer"**: The traditional Pass@k measures the probability of getting at least one correct answer out of $k$ samplings, which only reflects the model's performance ceiling (potential) while completely ignoring stability.

**Static benchmarks pose a risk of data leakage**: Classic benchmarks like MATH and GSM8K have been widely used, and models may have encountered similar questions in their training data, leading to artificially inflated evaluation results.

Core Idea: **A truly reliable reasoning model must not only "be capable of answering correctly" but also "answer correctly consistently"**. Thus, a metric capable of quantifying both the performance ceiling and stability simultaneously is required.

## Method

### Overall Architecture

The contribution of this paper consists of two parts: (1) the G-Pass@k evaluation metric; and (2) the LiveMathBench dynamic benchmark. The workflow is as follows: sample $n$ responses for each question $\rightarrow$ count the number of correct responses $c$ $\rightarrow$ compute performance under different stability requirements using the G-Pass@k suite of metrics.

### Key Designs

1. **G-Pass@k 基础指标**:

    - Function: Measures the probability that all $k$ responses randomly selected from $n$ samplings are correct.
    - Mechanism: Computed using the hypergeometric distribution. Let the total number of samples be $n$ and the number of correct responses be $c$, then:
    $$\text{G-Pass@}k = \mathbb{E}_{\text{Questions}}\left[\frac{\binom{c}{k}}{\binom{n}{k}}\right]$$
    - Design Motivation: Regresses to Pass@1 when $k=1$, and reflects stability when requiring all $k$ trials to be correct. However, a single value of $k$ is still not comprehensive enough.

2. **G-Pass@$k_\tau$ 带阈值的指标**:

    - Function: Measures the probability of getting at least $\lceil\tau \cdot k\rceil$ correct responses out of $k$ samples.
    - Mechanism: Introduces a threshold parameter $\tau \in [0, 1]$ to control the criterion of "success":
    $$\text{G-Pass@}k_\tau = \mathbb{E}_{\text{Questions}}\left[\sum_{j=\lceil\tau \cdot k\rceil}^{c}\frac{\binom{c}{j}\cdot\binom{n-c}{k-j}}{\binom{n}{k}}\right]$$
    - Design Motivation: When $\tau=0$, it is equivalent to the traditional Pass@k (at least one correct response), and when $\tau = 1$, it requires all responses to be correct. Tuning $\tau$ allows continuous trade-offs between "performance ceiling" and "perfect stability".

3. **mG-Pass@k 综合指标**:

    - Function: Computes the integrated mean of G-Pass@$k_\tau$ over the interval $\tau \in [0.5, 1.0]$ to evaluate overall performance with a single number.
    - Mechanism:
    $$\text{mG-Pass@}k = 2\int_{0.5}^{1.0}\text{G-Pass@}k_\tau \, d\tau = \frac{2}{k}\sum_{i=\lceil 0.5k\rceil+1}^{k}\text{G-Pass@}\frac{i}{k}$$
    - Design Motivation: The interval $[0.5, 1.0]$ is selected because it is more effective at distinguishing models—most models can pass at low thresholds, yielding little discriminative power. mG-Pass@k provides a concise single-value metric.

4. **LiveMathBench 动态基准**:

    - Function: Provides continuously updated, challenging math competition problems to mitigate data leakage.
    - Mechanism: Collects new problems from contemporary math competitions and updates them periodically (e.g., v202412, v202505) to ensure that the questions have not been seen in the models' training data.
    - Design Motivation: To address the issues of scoreboard optimization and data leakage associated with static benchmarks.

### Evaluation Method

Qwen2.5-72B-Instruct is used as the judge model to evaluate answer correctness, and a lightweight LiveMath-Judge model is also released to reduce evaluation costs.

## Key Experimental Results

### Main Results (LiveMathBench-202412)

| Model | Pass@1 | G-Pass@16₀ | G-Pass@16₀.₅ | G-Pass@16₁ | mG-Pass@16 |
|------|--------|-----------|-------------|-----------|-----------|
| Llama-3.1-8B-Inst | 24.0 | 18.2 | 11.3 | 4.5 | 10.4 |
| Qwen2.5-7B-Inst | 37.0 | 36.5 | 27.2 | 16.0 | 25.8 |
| Claude-3.5-Sonnet | 46.7 | 44.1 | 36.2 | 26.6 | 35.3 |
| Qwen2.5-Math-7B-Inst | 68.4 | 44.1 | 38.3 | 28.1 | 36.6 |
| QwQ-32B-Preview | 72.7 | 74.9 | 65.8 | 40.1 | 61.2 |
| OpenAI-o1-mini | 74.1 | 76.3 | 67.3 | 48.3 | 64.8 |
| DeepSeek-R1 | 81.1 | 83.6 | 79.1 | 69.5 | 77.6 |
| OpenAI-o3-mini | 84.7 | 85.7 | 78.8 | 65.3 | 76.8 |

### Ablation Study (AIME2025)

| Model | Pass@1 | mG-Pass@16 | Description |
|------|--------|-----------|------|
| Qwen2.5-Math-72B-Inst | 13.3 | 13.3 | Specialized math models exhibit higher stability |
| DeepSeek-Distill-Qwen-7B | 46.7 | 36.1 | Distilled models have a large gap between Pass@1 and mG, indicating lower stability |
| DeepSeek-R1 | 66.7 | 42.5 | The strongest model still has a 24% stability gap |
| OpenAI-o3-mini | 53.3 | 43.6 | o1-like models are relatively stable |

### Key Findings

- **Pass@1 may disconnect from stability**: Qwen2.5-Math-7B scores as high as 68.4 on Pass@1, but its mG-Pass@16 is only 36.6, illustrating that a high Pass@1 does not necessarily indicate stable reasoning.
- **o1-like reasoning models (🏀) are overall more stable**: The gap between Pass@1 and mG-Pass is smaller for DeepSeek-R1 and o3-mini.
- **Specialized mathematical models (🏐) are not necessarily stable**: They might overfit to specific problem-solving schemas.
- **Discrepancies are more pronounced on the Hard subset**: Most models score close to 0 on G-Pass@16₁.₀ in LiveMathBench-Hard.

## Highlights & Insights

- The core value of G-Pass@k lies in **using a curve instead of a single point** to evaluate models, providing a complete profile from "best-case scenario" to "worst-case scenario".
- mG-Pass@k offers a practical, single-value metric that is directly applicable to leaderboards.
- The dynamic update mechanism of LiveMathBench effectively counters data leakage and deserves extension to other fields.
- The formula design based on the hypergeometric distribution is simple and elegant, requiring no extra parameter fitting.
- The complete evaluation framework has been open-sourced, with support for OpenCompass and LightEval.

## Limitations & Future Work

- Currently only validated on mathematical reasoning, without expansion to other tasks requiring stability, such as code generation or logical reasoning.
- The dataset size of LiveMathBench is limited (at the level of ~100 problems), which may not be large enough.
- Only the phenomenon (instability) is analyzed; the underlying mechanistic reasons for **why** models are unstable have not been deeply explored.
- The choice of $k$ and $n$ affects the results, and the paper does not fully discuss the optimal settings.
- Inference hyperparameters such as sampling temperature also affect stability, but their interactive effects have not been systematically studied.

## Related Work & Insights

- **vs Pass@k (Chen et al., 2021)**: Pass@k only measures "at least one success", whereas G-Pass@k achieves continuous evaluation from lenient to strict via the threshold $\tau$.
- **vs Greedy/Majority Voting**: Greedy decoding evaluates a single trial, and Majority Voting takes the most frequent result, while G-Pass@k offers fine-grained analysis.
- **vs LiveCodeBench**: Similarly coordinates dynamic updates to prevent data leakage, but G-Pass@k offers richer assessment dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ While simple, the metric design is motivated by a clear concept, offering a significant complement to existing evaluation dimensions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers a wide array of models across multiple benchmarks, with sufficient data volume.
- Writing Quality: ⭐⭐⭐⭐ Well-formulated and clearly presented, with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Reveals a neglected evaluation dimension—"stability"—which holds valuable reference import for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Stepwise Reasoning Disruption Attack of LLMs](seed_stepwise_reasoning_disruption_attack.md)
- [\[ACL 2025\] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](sarft_roleplay_safety.md)
- [\[ACL 2025\] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery](learning_from_litigation_graphs_and_llms_for_retrieval_and_reasoning_in_ediscove.md)
- [\[ACL 2025\] IPO: Your Language Model is Secretly a Preference Classifier](ipo_your_language_model_is_secretly_a_preference_classifier.md)

</div>

<!-- RELATED:END -->
