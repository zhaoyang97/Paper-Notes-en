---
title: >-
  [Paper Note] Scaling Evaluation-Time Compute with Reasoning Models as Evaluators
description: >-
  [ACL2026][LLM Reasoning][evaluation-time compute] This paper extends test-time scaling from "answer generation" to "answer evaluation," finding that allowing reasoning models to generate more reasoning tokens during eval…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "evaluation-time compute"
  - "reasoning evaluator"
  - "process evaluation"
  - "outcome evaluation"
  - "Best-of-N"
date: 2026-05-08
content_hash: 05556d812b8a4c70
---

# Scaling Evaluation-Time Compute with Reasoning Models as Evaluators

**Conference**: ACL2026  
**arXiv**: [2503.19877](https://arxiv.org/abs/2503.19877)  
**Code**: None (Cache not provided)  
**Area**: LLM Reasoning / LLM Evaluation / Test-time Scaling  
**Keywords**: evaluation-time compute, reasoning evaluator, process evaluation, outcome evaluation, Best-of-N

## TL;DR
This paper extends test-time scaling from "answer generation" to "answer evaluation," finding that allowing reasoning models to generate more reasoning tokens during evaluation, perform step-by-step process checks, and combine outcome/process scores enables them to outperform trained PRMs/ORMs in ProcessBench and Best-of-N reranking.

## Background & Motivation
**Background**: The reasoning capabilities of LLMs have significantly benefited from test-time compute, such as allowing models to generate longer Chain-of-Thought (CoT), performing self-verification, or multiple sampling. Simultaneously, model evaluators have become increasingly important, as they judge answer correctness, identify errors in reasoning chains, and help generators select better outputs in Best-of-N or search-based reasoning.

**Limitations of Prior Work**: Mainstream evaluators are mostly ORMs or PRMs that directly predict rewards/scores. They usually require specialized training and are prone to reward over-optimization on out-of-distribution tasks. While generative evaluators can output CoT, many fine-tuned evaluators produce short CoT that lacks the self-correction, backtracking, and edge-case analysis seen in reasoning models.

**Key Challenge**: If generative models become stronger by "thinking longer," can evaluation models also become stronger by "evaluating longer"? Existing work often spends compute budgets on sampling more candidate answers without systematically comparing the trade-offs between "generating more candidates" and "evaluating candidates more thoroughly."

**Goal**: The authors investigate evaluation-time scaling: using off-the-shelf reasoning models as evaluators, forcing them to generate reasoning at both outcome and process levels. The study examines if evaluation quality improves monotonically with reasoning tokens and whether this stronger evaluation improves the problem-solving performance of generators.

**Key Insight**: The paper categorizes reasoning evaluators into outcome evaluators and process evaluators. The former judges the correctness of the full answer, while the latter evaluates each reasoning segment step-by-step, aggregating process scores into an overall score.

**Core Idea**: Transfer the problem-solving strategies of reasoning models to the evaluation process, allowing the evaluator to consume more evaluation-time compute through step-by-step checks, self-consistency, and outcome/process fusion.

## Method
The method does not involve training new models but constructs an inference-time evaluation recipe. Given a question $x_i$ and a candidate answer $y_i$, the reasoning evaluator first generates a CoT and then outputs a binary judgment. Scores are derived from the logits of "1/0" tokens. Process evaluation splits the answer into multiple steps, evaluates each step, and aggregates them.

### Overall Architecture
The pipeline serves two purposes. The first is capability testing of the evaluator itself: in ProcessBench, the model must identify the first erroneous segment in a solution. The second is enhancing the generator: the generator first samples multiple candidate answers for each question, then the evaluator scores these candidates, selecting the highest-scored one as the Best-of-N output. The paper compares reasoning-based Best-of-8 against direct evaluator Best-of-64 under a fixed approximate compute budget.

### Key Designs
1. **Reasoning Outcome Evaluator**:
	- **Function**: Judges whether the full candidate answer is correct and provides a global score.
	- **Mechanism**: Prompts the reasoning model to read $(x_i, y_i)$, first outputting a CoT $c_i$, and then a judgment $j_i$. If the model is required to output "1" for correct and "0" for incorrect, softmax is applied to the corresponding token logits: $s_i = e^{\ell(j_i=1)} / (e^{\ell(j_i=0)} + e^{\ell(j_i=1)})$.
	- **Design Motivation**: Outcome evaluation has a complete perspective, suitable for capturing whether the final answer is reasonable, but it might overlook fine-grained mistakes in intermediate steps.

2. **Reasoning Process Evaluator**:
	- **Function**: Step-by-step inspection of each reasoning segment in the candidate answer to locate the first error step.
	- **Mechanism**: For the $k$-th step, the evaluator sees the problem and the first $k$ steps, generating a specific CoT and judgment for $y_{ik}$. All step judgments are converted into a global score via an aggregation function. The paper prefers multi-step process evaluation over single-pass evaluation of all steps.
	- **Design Motivation**: Step-by-step checking naturally scales evaluation-time compute and forces the model to perform more detailed verification of each local reasoning component, which is less likely to miss errors than a single CoT covering all steps.

3. **Model-based Splitting and Outcome/Process Fusion**:
	- **Function**: Adapts process evaluation to irregularly structured answers and fuses complementary signals from outcome and process levels.
	- **Mechanism**: When answers lack clear line breaks or contain code, $M_{split}$ is used to insert `[SPLIT]` for step division. For process score aggregation, the paper finds `mean_logit` performs better than the commonly used `min`. The final score uses $s_{final}=\alpha s_{outcome}+(1-\alpha)s_{process}$, with $\alpha=0.5$ in main experiments.
	- **Design Motivation**: Process evaluators have high precision but low recall, while outcome evaluators are more holistic but potentially coarse; interpolation balances both perspectives.

### Loss & Training
No new training losses are proposed; the core lies in inference-time strategies. ProcessBench uses F1 to measure the accuracy of predicting the first error segment; for Best-of-N, LeetCode uses pass@1, and the other 6 benchmarks use accuracy. To ensure a fair compute budget, direct ORM/PRMs use Best-of-64, while reasoning evaluators use Best-of-8 due to higher per-evaluation cost. Candidates are sourced from Eurus-2-SFT, Llama3.1-70B-Instruct, and Qwen2.5-7B-Instruct, totaling 4,680 instances and 299,520 responses across 7 benchmarks.

## Key Experimental Results

### Main Results
On ProcessBench, both multi-step process evaluation and self-consistency improve evaluator F1. Notably, a non-specialized 32B reasoning model can outperform a trained 72B PRM.

| Evaluator | Setting | Avg. F1 | Key Comparison |
|-----------|---------|---------|----------------|
| Qwen2.5-Math-PRM-7B | Direct PRM | 73.5 | Trained 7B PRM |
| Qwen2.5-Math-PRM-72B | Direct PRM | 78.3 | Trained 72B PRM |
| DeepSeek-R1-Distill-Qwen-32B | Single-step reasoning | 75.5 | Single CoT insufficient to beat 72B PRM |
| DeepSeek-R1-Distill-Qwen-32B | Multi-step process | 78.6 | Slightly higher than 72B PRM |
| QwQ-32B | Multi-step process | 79.3 | Higher than 72B PRM |
| DeepSeek-R1-Distill-Qwen-32B | Multi-step + self-consistency | 82.8 | Significantly exceeds previous SOTA |
| QwQ-32B | Multi-step + self-consistency | 82.0 | Also exceeds 72B PRM |

### Ablation Study
In Best-of-N, the reasoning evaluator seeing only 8 candidates approaches or exceeds the performance of a direct evaluator seeing 64 candidates. Values in the table represent average scores across 7 benchmarks and 3 generators.

| Evaluator | N=1 | N=2 | N=4 | N=8 | N=64 / Note |
|-----------|-----|-----|-----|-----|-------------|
| Skywork-Reward-Gemma-2-27B-v0.2 | 38.2 | 41.8 | 43.4 | 44.8 | N=64 is 45.4 |
| Qwen2.5-Math-PRM-72B | 38.2 | 42.9 | 45.4 | 48.2 | N=64 is 50.6 |
| DeepSeek-R1-Distill-Qwen-32B outcome | 38.2 | 43.9 | 47.7 | 51.1 | reasoning evaluator only to N=8 |
| DeepSeek-R1-Distill-Qwen-32B process | 38.2 | 43.6 | 46.9 | 50.3 | process alone near 72B PRM N=64 |
| DeepSeek-R1-Distill-Qwen-32B process + outcome | 38.2 | 44.4 | 48.5 | 52.0 | Exceeds Qwen2.5-Math-PRM-72B N=64 |

### Key Findings
- Multi-step process evaluation is more effective than self-consistency: e.g., QwQ-32B multi-step yields 79.3 vs. 76.8 for self-consistency; DeepSeek-R1-Distill-Qwen-32B multi-step yields 78.6 vs. 77.8.
- Combining multi-step with self-consistency further improves performance: DeepSeek-R1-Distill-Qwen-7B improves from 54.5 (single-step) to 73.7 (multi-step + self-consistency).
- Under fixed budgets, reasoning evaluator using Best-of-8 outperforms direct evaluator using Best-of-64 by 4.30 to 6.63 percentage points.
- Process evaluation is more conservative: analysis shows reasoning process evaluators have higher precision and lower recall; when it judges all steps to be correct, the final answer is highly likely to be correct (false positive rates of 3.8% and 3.5%).

## Highlights & Insights
- The core insight is concise: evaluation is a reasoning task and should thus benefit from test-time scaling. While previous work focused reasoning budgets on "generating more answers," this proves "evaluating fewer answers more thoroughly" can be more efficient.
- The gains from multi-step process evaluation do not come solely from model size, but from decomposing the task into local checkpoints, allowing the reasoning model's self-check and back-tracking capabilities to function.
- The complementarity between outcome and process levels is crucial. Outcome looks at the final answer, while process looks at local reasoning; interpolation reduces bias from a single evaluative perspective.
- Insights for code tasks: Traditional PRMs are mostly trained on mathematical processes and generalize poorly to code output regarding splitting and reward; reasoning evaluators can bridge this by reading code logic and boundary conditions.

## Limitations & Future Work
- Evaluator implementation was primarily tested on ProcessBench, which, despite high label quality, focuses mainly on error localization in mathematical reasoning chains.
- Best-of-N experiments focus on math and code since these tasks are easily verifiable and align with reasoning model strengths; non-verifiable tasks like creative or scientific writing have not been systematically evaluated.
- The authors did not test some newer or stronger closed-source/large reasoning models due to their recency, high hardware costs for 70B-class reasoning evaluators, and API budget constraints (e.g., OpenAI o1, Gemini 2.5, Claude 3).
- Using reasoning evaluators increases the cost per candidate evaluation. Real-world systems must decide whether to use the full process + outcome workflow based on latency, cost, and task value.

## Related Work & Insights
- **vs ORM / PRM**: ORM/PRMs output scores directly with low cost but require training and are prone to over-optimization; reasoning evaluators trade inference time for generalization and robust evaluation without training new reward heads.
- **vs Self-Consistency**: Self-consistency asks the same global judgment multiple times, whereas multi-step process evaluation decomposes the answer for step-by-step verification. Experiments show the latter is more effective under similar budgets.
- **vs Best-of-N Sampling Scaling**: Traditional test-time scaling favors generating more candidates; this paper shows that investing in stronger evaluation for fewer candidates can be more effective than blindly increasing $N$.
- **Insights**: For agent planning, code generation, math competitions, and scientific reasoning systems, verifiers can be upgraded from lightweight scorers to "reasoning reviewers with a budget," with evaluation depth determined dynamically by task risk.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Concept is straightforward but targets a new dimension of test-time scaling, clearly positioning evaluation-time compute.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ ProcessBench, Best-of-N, and various model families/ablations are well-covered, though non-math/code tasks need expansion.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological formulas are clear, findings directly support claims, and appendix results are comprehensive.
- Value: ⭐⭐⭐⭐⭐ Direct reference value for verifiers, reward models, Best-of-N, and budget allocation in reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
