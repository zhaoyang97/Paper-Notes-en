---
title: >-
  [Paper Note] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases
description: >-
  [ACL 2026][LLM Evaluation][Reasoning Model Judgment] This paper systematically compares the performance of reasoning models versus standard LLMs as judges. It finds that while reasoning models exhibit superior accuracy, evaluation instruction following, and attack robustness, they remain susceptible to surface-level quality biases. The authors propose PlanJudge, a prompt-only strategy to mitigate these biases.
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Reasoning Model Judgment"
  - "Evaluation Biases"
  - "PlanJudge"
  - "RewardBench"
  - "BiasBench"
date: 2026-05-08
content_hash: 93607bed71b419d3
---

# Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases

**Conference**: ACL 2026  
**arXiv**: [2601.03630](https://arxiv.org/abs/2601.03630)  
**Code**: https://github.com/HuihuiChyan/LRM-Judge  
**Area**: LLM Security / LLM Evaluation / LLM-as-a-Judge  
**Keywords**: Reasoning Model Judgment, Evaluation Biases, PlanJudge, RewardBench, BiasBench

## TL;DR
This paper systematically compares the performance of reasoning models versus standard LLMs as judges. It finds that while reasoning models exhibit superior accuracy, evaluation instruction following, and attack robustness, they remain susceptible to surface-level quality biases. The authors propose PlanJudge, a prompt-only strategy to mitigate these biases.

## Background & Motivation
**Background**: As open-ended generation tasks increase, traditional metrics such as BLEU and ROUGE fail to capture LLM output quality. LLM-as-a-Judge has become a mainstream evaluation solution, where researchers use strong models to compare or score responses to replace expensive human review.

**Limitations of Prior Work**: Judge models themselves are prone to errors. Existing research indicates that LLM-as-a-Judge is affected by biases related to position, length, style, specificity, and format. Meanwhile, Large Reasoning Models (LRMs) have demonstrated superior performance in math and coding tasks via long-form reasoning and self-correction, but their suitability as judges has not been systematically established.

**Key Challenge**: Reasoning models might be more effective at complex judgments, but they could also be more easily induced by surface features due to "overthinking" or over-reliance on explicit criteria. Determining their superiority requires a comprehensive analysis beyond accuracy on reward benchmarks, including instruction following, attack robustness, and bias robustness.

**Goal**: Under a controlled setting where "reasoning" is the only variant, this work compares four pairs of reasoning and non-reasoning models as judges and designs a lightweight strategy to mitigate biases.

**Key Insight**: The authors select reasoning/instruct variants from the same model families (e.g., DeepSeek-V3 vs. DeepSeek-R1, Qwen2.5-32B-Instruct vs. QwQ-32B, Qwen3 instruct vs. thinking mode) to isolate the impact of the "reasoning process" itself.

**Core Idea**: Reasoning models are indeed stronger judges, but they must be prompted to formulate clear evaluation plans before executing judgment to reduce over-preference for surface signals like length and specificity.

## Method
The paper consists of two parts. The first is a systematic empirical comparison examining four dimensions of Large Reasoning Models as judges (LRM-as-a-Judge): general evaluation accuracy, evaluation instruction following, robustness to prompt injection, and robustness to evaluation biases. The second part introduces PlanJudge: a method that requires the judge to generate or receive a fine-grained evaluation plan before completing comparisons, without training new models.

### Overall Architecture
Experiments were conducted using four model pairs across RewardBench, JudgeBench, Helpsteer2-trivial, RobustJudge, BiasBench, and LLMBar. Helpsteer2-trivial is a new dataset constructed by the authors where Response A is overall superior, but Response B is better in a specific dimension. A competent judge should select A under an overall prompt but switch to B under a specific prompt; the Reversal Rate is used to measure evaluation instruction following.

Subsequently, the authors integrated PlanJudge into the same set of judges. PlanJudge decomposes evaluation into "planning" and "execution": first defining evaluation dimensions, priorities, and precautions based on the task, then prompting the model to judge according to the plan. Plans can be heuristic-based, self-synthesized, or a combination of both.

### Key Designs

**1. Controlled reasoning vs. non-reasoning comparison: Isolating the "reasoning process" as the sole variable**

To determine if reasoning models are better judges, the experiment avoids confounding factors such as parameter count or pre-training data by comparing instruct and thinking variants within the same family (e.g., DeepSeek-V3 vs. R1). This ensures that performance gaps can be cleanly attributed to the reasoning process itself.

**2. Helpsteer2-trivial and Reversal Rate: Quantifying instruction following in evaluation contexts**

In evaluation scenarios, a judge may understand the overall quality but fail to ignore extraneous dimensions when instructed to "only evaluate helpfulness." Helpsteer2-trivial contains samples where Response A is better overall while Response B is better in one specific dimension. The Reversal Rate (RR) measures how often the model correctly switches its selection based on the specific prompt.

**3. PlanJudge Two-stage Evaluation: Plan first, judge second**

Reasoning models inherently check criteria, but ambiguous criteria can lead them to mistake surface signals (length, detail) for quality. PlanJudge splits the process: first writing out which dimensions to evaluate and their priorities, then comparing the two responses strictly according to this plan. Plans originate from three sources: heuristic-based (human rules), self-synthesized (model-generated for the sample), or combined. Explicit planning anchors the model's attention to the required standards before reasoning begins.

### Loss & Training
PlanJudge is a prompt-only method requiring no additional fine-tuning, reward models, or external resources. Its cost stems from longer reasoning and evaluation prompts, making it easily deployable as an evaluation protocol compared to methods requiring judge training.

## Key Experimental Results

### Main Results
In terms of general evaluation accuracy, reasoning variants mostly outperform instruct variants, with the Qwen series showing significant gains on JudgeBench. DeepSeek-R1 is stronger than V3 on RewardBench but lower on JudgeBench, which the authors attribute to hallucination issues in knowledge-based tasks.

| Model Pair | RewardBench | JudgeBench | Conclusion |
|--------|-------------|------------|------|
| DeepSeek-V3 | 89.74 | 84.19 | Standard model is more stable on JudgeBench |
| DeepSeek-R1 | 91.18 | 80.48 | Stronger on RewardBench; exceptions in knowledge judgment |
| Qwen2.5-32B-Instruct | 89.31 | 60.40 | Non-reasoning version is significantly weaker on JudgeBench |
| QwQ-32B | 91.05 | 79.75 | Reasoning brings substantial improvement |
| Qwen3-30B-A3B-Instruct | 89.88 | 74.00 | Instruct baseline is relatively strong |
| Qwen3-30B-A3B-Thinking | 92.01 | 83.87 | Reasoning is superior across both metrics |
| Qwen3-Next-80B-A3B-Instruct | 88.96 | 79.45 | Instruct version is stable |
| Qwen3-Next-80B-A3B-Thinking | 92.90 | 82.42 | Reasoning continues to improve |

Regarding evaluation instruction following, reasoning versions generally exhibit higher Reversal Rates, suggesting that long-form reasoning enhances the model's ability to cross-check evaluation dimensions.

| Model | OriACC | RR | Observation |
|------|--------|----|------|
| DeepSeek-V3 | 78.22 | 87.80 | Overall judgment is accurate, but dimension switching is weaker |
| DeepSeek-R1 | 73.61 | 95.24 | RR is significantly higher |
| Qwen2.5-32B-Instruct | 71.13 | 83.19 | Insufficient dimension switching in instruct version |
| QwQ-32B | 76.49 | 91.11 | Reasoning improves instruction following |
| Qwen3-30B-A3B-Instruct | 72.78 | 95.67 | Inherently strong |
| Qwen3-30B-A3B-Thinking | 78.14 | 97.44 | One of the best performers across both metrics |
| Qwen3-Next-80B-A3B-Instruct | 75.88 | 82.50 | Lower RR |
| Qwen3-Next-80B-A3B-Thinking | 77.94 | 91.18 | Significant improvement after reasoning |

### Ablation Study
On bias robustness, the results are more varied. LRMs are generally stronger on LLMBar due to better identification of explicit instruction misalignment, but they are more affected by surface quality (length, specificity) on BiasBench.

| Model | BiasBench | LLMBar | Explanation |
|------|-----------|--------|------|
| DeepSeek-V3 | 81.25 | 76.49 | Balanced bias robustness |
| DeepSeek-R1 | 65.00 | 79.00 | Stronger surface bias, but identifies explicit mismatch |
| Qwen2.5-32B-Instruct | 82.50 | 67.71 | High BiasBench, weaker LLMBar |
| QwQ-32B | 67.50 | 79.31 | Reasoning improves LLMBar, degrades BiasBench |
| Qwen3-30B-Instruct | 81.25 | 59.25 | Weaker LLMBar |
| Qwen3-30B-Thinking | 77.50 | 83.07 | Reasoning significantly improves LLMBar |
| Qwen3-Next-Instruct | 80.00 | 64.55 | Lower bias in instruct version |
| Qwen3-Next-Thinking | 75.00 | 77.55 | LLMBar improves after reasoning |

The PlanJudge combined strategy significantly improves BiasBench performance while typically maintaining or improving RewardBench/LLMBar scores.

| Model | Method | RewardBench | BiasBench | LLMBar |
|------|------|-------------|-----------|--------|
| DeepSeek-V3 | Original | 89.70 | 81.25 | 76.49 |
| DeepSeek-V3 | Combined | 93.07 | 98.75 | 86.83 |
| DeepSeek-R1 | Original | 91.10 | 65.00 | 79.00 |
| DeepSeek-R1 | Combined | 92.47 | 97.50 | 86.21 |
| Qwen2.5-32B | Original | 89.30 | 82.50 | 67.71 |
| Qwen2.5-32B | Combined | 89.68 | 93.59 | 75.55 |
| QwQ-32B | Original | 91.00 | 67.50 | 79.31 |
| QwQ-32B | Combined | 93.13 | 95.00 | 83.07 |

### Key Findings
- LRM-as-a-Judge generally outperforms standard LLM judges, particularly in reasoning-intensive coding, math, and complex judgment tasks.
- Reasoning models exhibit stronger evaluation instruction following, contrasting with some findings in general instruction following where reasoning models appear "stubborn."
- Reasoning models are more stable against prompt injection attacks due to checking task boundaries during reasoning.
- However, reasoning models still prefer responses that appear more specific, longer, or more organized, even if these traits do not represent actual quality.
- The core value of PlanJudge is not simply making the model "think more," but defining "how it should think" before reasoning begins.

## Highlights & Insights
- The paper decomposes judge capability into four dimensions, avoiding single-metric conclusions like "a higher benchmark score means a better judge."
- Helpsteer2-trivial is a practical contribution that converts evaluation instruction following into a measurable Reversal Rate.
- PlanJudge is simple but effective, addressing the root cause: ambiguous evaluation rubrics.
- It serves as a reminder that reasoning chains are not inherently reliable and can amplify existing evaluation preferences; therefore, the reasoning process must be constrained by structured criteria.

## Limitations & Future Work
- Model coverage is intentionally limited to families with clear reasoning/non-reasoning pairs; conclusions for LLaMA-based or closed-source o-series models require further verification.
- Use of only one or two benchmarks per evaluation dimension may still be affected by dataset design biases.
- PlanJudge increases reasoning cost and latency, requiring a trade-off in high-throughput automatic evaluation scenarios.
- Bias types are primarily from BiasBench/LLMBar; domain, cultural, and linguistic biases in real-world evaluation are not fully covered.
- Future work could investigate automated plan validation and the combination of PlanJudge with calibration or human-in-the-loop auditing.

## Related Work & Insights
- **vs. Empirical studies of LLM-as-a-Judge**: While previous work focused on human-agreement of judges like GPT-4, this work specifically compares the reasoning mode.
- **vs. BiasBench / LLMBar**: These benchmarks provide bias diagnostics; this paper discovers that reasoning affects different biases differently, precluding a simple "reasoning is better/worse" conclusion.
- **vs. Training-based judge improvements**: Unlike methods requiring fine-tuning, PlanJudge is lightweight and relies on the model’s inherent planning execution.
- **Insight**: When using LLM judges in research, evaluation rubrics should be formatted as explicit plans to improve evaluation reproducibility.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The systematic comparison of reasoning judges is timely, and PlanJudge is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers four judge capabilities across multiple model pairs, though benchmark dimensions could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Clear conclusions and information-dense tables.
- Value: ⭐⭐⭐⭐⭐ Directly guides model selection, bias control, and evaluation protocol design for LLM-as-a-Judge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge](mm-judgebias_a_benchmark_for_evaluating_compositional_biases_in_mllm-as-a-judge.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](../../ICML2026/llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)

</div>

<!-- RELATED:END -->
