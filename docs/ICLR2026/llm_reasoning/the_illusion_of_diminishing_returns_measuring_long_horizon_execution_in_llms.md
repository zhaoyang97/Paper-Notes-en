---
title: >-
  [Paper Note] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs
description: >-
  [ICLR 2026][LLM Reasoning][long-horizon execution] This paper reveals that short-task benchmarks create an illusion of diminishing returns — marginal gains in per-step accuracy are amplified exponentially in long-horizon…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "long-horizon execution"
  - "self-conditioning"
  - "chain-of-thought"
  - "scaling"
  - "diminishing returns"
date: 2026-05-08
content_hash: d6a903094fe757e8
---

# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs

**Conference**: ICLR 2026
**arXiv**: [2509.09677](https://arxiv.org/abs/2509.09677)
**Code**: Available
**Area**: LLM Reasoning
**Keywords**: long-horizon execution, self-conditioning, chain-of-thought, scaling, diminishing returns

## TL;DR
This paper reveals that short-task benchmarks create an illusion of diminishing returns — marginal gains in per-step accuracy are amplified exponentially in long-horizon tasks. It identifies a "self-conditioning effect" in LLMs (whereby prior errors increase the probability of subsequent errors), shows that thinking models mitigate this effect, and demonstrates that GPT-5 thinking can execute tasks exceeding 2,100 steps.

## Background & Motivation

### State of the Field

**Background**: LLMs continue to improve on complex reasoning benchmarks, yet consistently fail when simple tasks are extended over longer horizons. This has been interpreted as a "fundamental deficiency in reasoning ability" or evidence that "thinking is merely an illusion."

**Limitations of Prior Work**: The causes of long-horizon failure are conflated — is the failure due to planning or execution? Existing analyses do not isolate execution capability.

**Key Challenge**: Marginal improvements on short-task benchmarks appear to be diminishing, yet this may systematically underestimate actual capability gains — a per-step accuracy improvement from 95% to 99% translates to a task success rate increase from 0.6% to 36.6% on a 100-step task.

**Goal**: (a) Quantify the long-horizon execution capability of LLMs; (b) identify the root causes of execution failure; (c) analyze the effect of scale and inference-time computation on long-horizon execution.

**Key Insight**: By explicitly providing knowledge and plans to models, the paper isolates and measures "execution" capability itself.

**Core Idea**: Long-horizon task failures in LLMs are primarily attributable to execution errors rather than insufficient reasoning. Furthermore, a "self-conditioning" phenomenon exists, whereby errors in the context increase the probability of subsequent errors.

## Method

### Overall Architecture
A controlled experimental design explicitly provides the knowledge and execution plan required for each task (eliminating the need for planning and knowledge retrieval), varies task length (steps × rounds), and measures per-step accuracy and task completion rate.

### Key Designs

1. **Execution Capability Isolation**: Models are given complete plans and knowledge, purely measuring the ability to "execute according to plan." This distinguishes the evaluation from reasoning benchmarks that simultaneously test planning, knowledge, and execution.

2. **Self-Conditioning Effect Detection**: The historical error rate in the context is controlled, and its causal influence on subsequent step accuracy is observed. Results show that as the error rate increases from 0% to 20%, subsequent accuracy drops sharply.

3. **Horizon Length Formula**: $H_s(p) = \lceil \frac{\ln(s)}{\ln(p)} \rceil$, which illustrates the hyperbolic relationship between per-step accuracy $p$ and achievable task length $H$ — small improvements in the high-accuracy regime yield explosive increases in achievable task length.

### Loss & Training
No model training is performed; this is a purely evaluative study.

## Key Experimental Results

### Main Results (Maximum Single-Round Execution Steps for Frontier Thinking Models)

| Model | Max Execution Steps | Notes |
|-------|-------------------|-------|
| GPT-5 (Horizon) thinking | **>2100** | Far exceeds all competitors |
| Claude-4 Sonnet thinking | 432 | Second place |
| DeepSeek-R1 (thinking) | >100 | Thinking provides significant benefit |
| DeepSeek-V3 (no thinking) | <4 | Almost no execution without thinking |

### Self-Conditioning Effect

| Historical Error Rate | Change in Subsequent Step Accuracy | Notes |
|----------------------|-----------------------------------|-------|
| 0% | Baseline | Normal accuracy |
| 5% | Significant drop | Self-conditioning begins |
| 20% | Sharp drop | Error cascade |

### Key Findings
- **Diminishing per-step accuracy is an illusion**: Small improvements in the high-accuracy regime are amplified exponentially in long-horizon tasks.
- **Self-conditioning effect**: Unlike humans who improve with practice, models become more error-prone upon encountering their own prior mistakes. Thinking models are immune to this effect.
- **Scaling yields large gains in execution**: Even when small models achieve near-perfect per-step accuracy, large models perform significantly better on long-horizon tasks.
- **Thinking fundamentally improves execution**: DeepSeek-V3 cannot execute 4 steps, whereas R1 can execute 100+.
- GPT-5 thinking's 2,100+ step execution capability marks a qualitative leap in long-horizon execution for LLMs.

## Highlights & Insights
- The argument that **"diminishing returns is an illusion"** is highly illuminating: the hyperbolic growth of $H(p) \propto 1/\ln(p)$ implies that an improvement from 99% to 99.5% delivers far greater value than one from 90% to 95%. This fundamentally reframes the perceived return on investment in scaling.
- The **self-conditioning effect** is a novel finding that cannot be resolved through simple scaling — only thinking can remedy it. This has significant implications for agent system design: error histories must be cleaned or isolated during execution.
- Attributing long-horizon task failure to "execution" rather than "reasoning" is an important reframing that helps direct research efforts more accurately.

## Limitations & Future Work
- Experimental tasks are relatively simple (controlled settings); real-world task execution failures likely involve additional factors.
- The mechanism underlying self-conditioning is not analyzed (is it attributable to attention patterns, training data distribution, or other causes?).
- Only closed-source thinking models are evaluated, precluding analysis of how thinking remedies self-conditioning.

## Related Work & Insights
- **vs. Shojaee et al. (Illusion of Thinking)**: That work claims that thinking model failures on long tasks indicate reasoning is an illusion; this paper refutes that interpretation, arguing such failures reflect execution failure rather than reasoning failure.
- **vs. Mirzadeh et al.**: That work contends LLMs cannot truly reason; this paper distinguishes between reasoning and execution.
- Implications for agent systems: agent frameworks need to be designed to manage execution history and avoid self-conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the self-conditioning effect and the illusion of diminishing returns is highly impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Controlled experimental design is rigorous; frontier model evaluation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Argumentation is logically clear; mathematical analysis is concise and compelling.
- Value: ⭐⭐⭐⭐⭐ Has fundamental implications for LLM scaling investment decisions and agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ACL 2026\] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](../../ACL2026/llm_reasoning/fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)

</div>

<!-- RELATED:END -->
