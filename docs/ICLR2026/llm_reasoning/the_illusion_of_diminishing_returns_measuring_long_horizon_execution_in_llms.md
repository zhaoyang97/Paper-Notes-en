---
title: >-
  [Paper Note] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
description: >-
  [ICLR 2026][LLM Reasoning][Chain-of-Thought] The paper reveals that short-task benchmarks provide an illusion of "diminishing returns"—marginal gains in single-step accuracy are amplified exponentially in long-horizon tasks. It identifies the "self-conditioning effect" (where a model's own errors increase the probability of subsequent errors), which thinking mode
tags:
  - ICLR 2026
  - LLM Reasoning
  - Chain-of-Thought
  - diminishing returns
date: 2026-05-08
content_hash: 0b98843cb6368a69
---
# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs

**Conference**: ICLR 2026  
**arXiv**: [2509.09677](https://arxiv.org/abs/2509.09677)  
**Code**: Yes  
**Area**: LLM Reasoning  
**Keywords**: Long-horizon execution, self-conditioning, Chain-of-Thought, scaling, diminishing returns  

## TL;DR
The paper reveals that short-task benchmarks provide an illusion of "diminishing returns"—marginal gains in single-step accuracy are amplified exponentially in long-horizon tasks. It identifies the "self-conditioning effect" (where a model's own errors increase the probability of subsequent errors), which thinking models can mitigate. Notably, GPT-5 thinking can execute tasks exceeding 2100 steps.

## Background & Motivation

**Background**: LLMs continue to progress on complex reasoning benchmarks, yet they fail when simple tasks are extended in length. This has been interpreted as a "fundamental flaw in reasoning capability" or evidence that "thinking is merely an illusion."

**Limitations of Prior Work**: The causes of long-task failures are often conflated—is it a failure of planning or execution? Existing analyses do not isolate execution capability.

**Key Challenge**: Marginal improvements on short-task benchmarks appear to be diminishing. However, this may underestimate actual capability growth: an increase in single-step accuracy from 95% to 99% translates to a success rate jump from 0.6% to 36.6% in a 100-step task.

**Goal**: (a) Quantify the long-horizon execution capabilities of LLMs; (b) Identify the root causes of execution failure; (c) Analyze the impact of scale and inference-time compute on long-horizon execution.

**Key Insight**: By explicitly providing knowledge and plans, the study isolates and measures "execution" capability itself.

**Core Idea**: Failures in long tasks for LLMs are primarily due to execution errors rather than insufficient reasoning. Furthermore, a "self-conditioning" effect exists where errors within the context increase the likelihood of subsequent mistakes.

## Method

### Overall Architecture

This is a purely evaluative study that does not train any models. The core is a controlled "long-horizon execution" measurement protocol. Conventional reasoning benchmarks typically mix planning (deciding what to search and in what order), knowledge, and execution (following the plan step-by-step). When a model fails a long task, it is unclear which component failed. This paper strips away the first two by directly feeding the required knowledge and a complete plan to the model, leaving "strict step-by-step execution" as the sole variable. The study then observes how step-wise accuracy and overall completion rates decay as the task length increases. The work revolves around three pillars: isolating execution capability, uncovering the "self-conditioning" failure mode through counterfactual experiments, and using a mathematical formula to explain why "diminishing returns" on short tasks are illusory.

### Key Designs

**1. Execution Capability Isolation: Distinguishing Planning from Constant Execution**

Standard benchmarks conflate planning, knowledge, and execution. This paper fixates the former two using a "key-value dictionary" abstraction: knowledge is a fixed dictionary of "five-letter English words $\rightarrow$ integers (range $[-99, 99]$)" placed in the context. The plan tells the model exactly which keys to look up in each round. The model only needs to perform a mechanical "retrieve-then-compose" operation—looking up integers and adding them to a running sum $S_t = S_{t-1} + \sum_{i=1}^{K} \mathcal{D}[k_{t,i}]$. Task length is controlled by the product of rounds $T$ and per-round complexity $K$. This setup allows failures to be cleanly attributed to the accumulation of execution errors.

**2. Self-Conditioning Effect Detection: Observing the Impact of Prior Errors**

The authors discovered a counter-intuitive phenomenon: while humans often become more proficient at repetitive tasks, LLMs are more likely to fail if they see their own previous errors in the context. To disentangle this from the difficulty of long contexts, a counterfactual experiment was designed. The model's dialogue history is artificially manipulated to inject false outputs at a specified error rate, and the accuracy of the 100th round is measured. If accuracy drops even when the history is "cured" to a 0% error rate, the issue is the long context; if the accuracy drops as the historical error rate increases, the self-conditioning effect is confirmed. Results show both factors are present, with a "cascade of errors" occurring when the historical error rate reaches 20%. Notably, thinking models are less affected, suggesting RL training may bias models toward "getting the current step right" rather than just completing the context.

**3. Horizon Length Metric: Explaining the Illusion of Diminishing Returns**

To quantify the relationship between single-step capability and task length, the authors define the task horizon length $H_s(p) = \lceil \frac{\ln(s)}{\ln(p)} \rceil$ under the assumptions of independent, constant single-step accuracy $p$ and that any single error leads to task failure ($s$ is the target total success rate). Since $H(p) \propto 1/\ln(p)$ is a hyperbolic function, the curve steepens sharply as $p$ approaches 1. A single-step accuracy increase from 95% to 99% results in the success rate for a 100-step task jumping from 0.6% to 36.6%. Thus, the "diminishing returns" observed on short benchmarks are merely an artifact of the metric used.

## Key Experimental Results

### Main Results (Maximum Execution Steps for Frontier Thinking Models)

| Model | Max Execution Steps | Note |
|-------|----------------------|------|
| GPT-5 (Horizon) thinking | **>2100** | Far exceeds all competitors |
| Claude-4 Sonnet thinking | 432 | Second place |
| DeepSeek-R1 (thinking) | >100 | Thinking provides significant help |
| DeepSeek-V3 (no thinking) | <4 | Nearly unable to execute without thinking |

### Self-Conditioning Effect

| History Error Rate | Subsequent Step Accuracy Change | Note |
|--------------------|---------------------------------|------|
| 0% | Baseline | Normal accuracy |
| 5% | Significant Decrease | Self-conditioning begins |
| 20% | Sharp Decline | Error avalanche |

### Key Findings
- **Diminishing returns in single-step accuracy are an illusion**: Minor gains in high-accuracy regimes amplify exponentially in long-horizon tasks.
- **Self-Conditioning Effect**: Unlike humans who improve with practice, LLMs are more prone to error after observing their own mistakes. Thinking models are largely immune to this.
- **Scaling yields massive returns in execution**: Even if small models have near-perfect single-step accuracy, larger models perform significantly better on long tasks.
- **Thinking fundamentally improves execution**: DeepSeek-V3 cannot execute 4 steps, while R1 can execute 100+ steps.
- The 2100+ step execution capability of GPT-5 thinking marks a qualitative leap in LLM long-horizon execution.

## Highlights & Insights
- The argument that **"diminishing returns are an illusion"** is highly provocative: the hyperbolic growth of $H(p) \propto 1/\ln(p)$ implies that the value of moving from 99% to 99.5% is far greater than moving from 90% to 95%. This fundamentally changes the perception of ROI in scaling.
- The **Self-Conditioning Effect** is a novel discovery that cannot be solved by simple scaling; it requires the "thinking" mechanism. This has major implications for Agent system design, suggesting a need to clean or isolate error histories during execution.
- Attributing long-task failure to **"execution"** rather than **"reasoning"** is a crucial perspective that helps guide future research directions.

## Limitations & Future Work
- The experimental tasks are relatively simple (controlled environment); real-world execution failures may involve more complex factors.
- The mechanism behind self-conditioning (e.g., attention patterns, training data distribution) remains unanalyzed.
- Evaluation of "thinking" is limited to closed-source models, preventing internal analysis of how thinking mitigates self-conditioning.

## Related Work & Insights
- **vs Shojaee et al. (Illusion of Thinking)**: They claimed thinking models' failure on long tasks proves reasoning is an illusion; this paper counters that it is a failure of execution, not reasoning.
- **vs Mirzadeh et al.**: They posited that LLMs cannot truly reason; this paper distinguishes between reasoning and execution.
- **Implications for Agent Systems**: There is a need to design agent frameworks that manage execution history and avoid self-conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the self-conditioning effect and the diminishing returns illusion is highly impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous controlled experiments and comprehensive evaluation of frontier models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and concise mathematical analysis.
- Value: ⭐⭐⭐⭐⭐ Fundamentally influences scaling investment decisions and agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](../../ACL2026/llm_reasoning/sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICLR 2026\] R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](r-horizon_how_far_can_your_large_reasoning_model_really_go_in_breadth_and_depth.md)
- [\[ICLR 2026\] Long Chain-of-Thought Reasoning Across Languages](long_chain-of-thought_reasoning_across_languages.md)
- [\[ICLR 2026\] When More Is Less: Understanding Chain-of-Thought Length in LLMs](when_more_is_less_understanding_chain-of-thought_length_in_llms.md)

</div>

<!-- RELATED:END -->
