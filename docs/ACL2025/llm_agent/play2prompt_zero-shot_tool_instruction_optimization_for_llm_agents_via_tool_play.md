---
title: >-
  [Paper Note] Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play
description: >-
  [ACL 2025][LLM Agent][tool learning] Proposes Play2Prompt, which enables LLMs to autonomously "play" with tools (exploring input-output behaviors) to generate tool-use examples and optimize tool documentation in a zero-shot manner, significantly enhancing the tool-calling capabilities of LLM agents without requiring any annotated data.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "tool learning"
  - "zero-shot"
  - "prompt optimization"
  - "beam search"
  - "self-reflection"
date: 2026-05-08
content_hash: 7f65e9e1466df80e
---

# Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play

**Conference**: ACL 2025  
**arXiv**: [2503.14432](https://arxiv.org/abs/2503.14432)  
**Code**: [GitHub](https://github.com/wfangtw/play2prompt)  
**Area**: Agent  
**Keywords**: tool learning, zero-shot, prompt optimization, beam search, self-reflection

## TL;DR
Proposes Play2Prompt, which enables LLMs to autonomously "play" with tools (exploring input-output behaviors) to generate tool-use examples and optimize tool documentation in a zero-shot manner, significantly enhancing the tool-calling capabilities of LLM agents without requiring any annotated data.

## Background & Motivation
**Background**: LLM agent frameworks solve complex tasks by calling external tools. The dynamic integration of new tools primarily relies on prompting (tool documentation + usage examples), and the quality of tool calls heavily depends on the completeness of the documentation and the quality of the examples.

**Limitations of Prior Work**: In real-world scenarios, user-provided tool documentation is often incomplete or noisy, and lacks usage examples. Existing automatic prompt optimization methods still require annotated data for validation, making them inapplicable in true zero-shot settings.

**Key Challenge**: No usage examples $\rightarrow$ unable to optimize documentation $\rightarrow$ unable to generate high-quality examples. This is a chicken-and-egg problem.

**Goal**: Automatically generate high-quality tool usage examples and optimize tool documentation under zero-annotation conditions.

**Key Insight**: Simulate human trial-and-error behavior—firstly "play" with tools to explore their behaviors (forward: exploring legitimate calls $\rightarrow$ backward: constructing corresponding queries), and then use the generated examples as a validation set to optimize the documentation.

**Core Idea**: First "play" with tools to acquire knowledge of input-output behaviors, then generate examples in a backward manner, and finally use the example set to validate and optimize the documentation—achieving the entire process with zero annotations.

## Method

### Overall Architecture
Play2Prompt consists of two steps: **Step 1** (Tool Usage Example Generation): Explore valid tool calls through a trial-and-error process and backward-generate query-answer pairs. **Step 2** (Documentation Optimization): Use the example set from Step 1 as a validation set to optimize the documentation via beam search. Both steps utilize a beam search framework integrated with self-reflection.

### Key Designs

1. **Backward Example Generation (Core Innovation)**:

    - **Function**: Generate tool usage examples $E = (x, F_k, I_k, y)$ in a zero-shot manner.
    - **Mechanism**: **Backward generation**—first explore valid calling parameters $I_k$ (through rejection sampling + self-reflection) and observe the tool output, then construct the query $x$ and answer $y$, rather than the traditional approach of formulating a query before calling the tool.
    - **Design Motivation**: When documentation is incomplete, forward calling is highly prone to failure. Starting backward from valid calls guarantees the executability of the examples.

2. **Adversarial Reward Design**:

    - **Function**: Evaluate the quality of the generated examples.
    - **Mechanism**: $R^{(i)} = R_q^{(i)} + \lambda R_e^{(i)}$, where $R_q$ evaluates the example quality (clarity, consistency), and $R_e = -\mathcal{P}\{\mathcal{M}_T(x; \mathcal{D}_0, \varnothing); y, F_k, I_k\}$ **encourages difficult examples** (those the task LLM fails to answer correctly).
    - **Design Motivation**: Similar to active learning, difficult examples are more beneficial for helping the LLM learn tool usage. Example generation (seeking maximum difficulty) and documentation optimization (seeking maximum accuracy) form an adversarial objective.

3. **Documentation Optimization (Step 2)**:

    - **Function**: Rewrite tool documentation using beam search + self-reflection.
    - **Mechanism**: Use the example set generated in Step 1 as a validation set. In each iteration, feedback tool execution errors of the task LLM to the generator LLM, guiding it to correct ambiguities and omissions in the documentation.
    - **Reward**: $R^{(i)} = \mathbb{E}_{(x,F,I,y) \in \mathcal{E}}[\mathcal{P}\{\mathcal{M}_T(x; \mathcal{D}^{(i)}, \varnothing); y, F, I\}]$

### Loss & Training
- No parameter training is involved—purely prompt optimization.
- Beam search retains top-reward samples at each step, with self-reflection guiding the direction of improvement.

## Key Experimental Results

### Main Results
Berkeley Function-Calling Leaderboard (Executable):

| Model | Prompting | +Play2Prompt | Gain |
|------|----------|-------------|------|
| LLaMA-8B | 85.9% | **92.9%** | +7.0 |
| LLaMA-70B | 90.8% | Improved | Significant |
| GPT-4o | baseline | **Improved** | Consistent |

### Ablation Study

| Configuration | Result | Description |
|------|------|------|
| Generate examples only (No doc optimization) | Improved but underperforms the full version | The two steps are complementary |
| Optimize doc only (No examples) | Limited improvement | Examples serve as both ICL and validation set |
| Forward example generation (Non-backward) | High rate of invalid calls | Backward generation is key |
| Remove difficult example preference | Performance degradation | Adversarial reward is effective |

### Key Findings
- **Backward generation is a critical design**: Constructing examples from legitimate calls avoids calling failures caused by incomplete documentation.
- **Both open-source and proprietary models benefit**: Consistent improvements are observed across LLaMA-8B/70B and GPT-4o.
- **Single-tool examples are sufficient**: Although only single-tool usage examples are generated, the task LLM generalizes effectively to multi-tool composition tasks.
- **The adversarial relationship between documentation optimization and example generation** enhances the overall performance.

## Highlights & Insights
- **The metaphor of "playing with tools" is highly intuitive**: It simulates human exploratory behavior when facing new tools—trying them out first, observing the results, and then understanding how they work. This trial-and-error mindset can be generalized to any scenario requiring the understanding of API behaviors.
- **Elegant solution to the zero-annotation chicken-and-egg problem**: It breaks the deadlock of "no examples mean no optimization" through backward generation.
- **Adversarial reward** borrows from active learning ideas: Selecting the most challenging examples for the task LLM maximizes learning efficiency.

## Limitations & Future Work
- **Dependency on tool executability**: It requires executing actual tools to obtain feedback, which is inapplicable to non-executable tools or those with side effects.
- **High computational cost**: The nested combination of beam search × rejection sampling × self-reflection incurs a substantial number of LLM calls.
- **Only single-tool examples are generated**: Although the task LLM generalizes to multiple tools, directly generating multi-tool compound examples could be more effective.
- **Future directions**: (1) Designing efficient search methods to reduce LLM calls; (2) Supporting non-executable tools (e.g., through document reasoning); (3) Multi-tool combination example generation.

## Related Work & Insights
- **vs ToolBench/ToolLLM**: These methods require extensive annotated tool-use data for training, whereas Play2Prompt is fully zero-shot.
- **vs Automatic Prompt Optimization (DSPy, RLPROMPT)**: These methods require a predefined validation set, whereas Play2Prompt generates its own validation set.
- **vs Tool Documentation Enhancement**: Prior works only optimize documentation or only generate examples. Play2Prompt jointly optimizes both with an adversarial objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both backward example generation and adversarial optimization are novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two real-world benchmarks (BFCL + StableToolBench) with multi-model validation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework descriptions and an intuitive Figure 1.
- Value: ⭐⭐⭐⭐⭐ Resolves the core pain points of tool integration in real-world scenarios, possessing high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[ECCV 2024\] Agent3D-Zero: An Agent for Zero-shot 3D Understanding](../../ECCV2024/llm_agent/agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](../../ACL2026/llm_agent/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
