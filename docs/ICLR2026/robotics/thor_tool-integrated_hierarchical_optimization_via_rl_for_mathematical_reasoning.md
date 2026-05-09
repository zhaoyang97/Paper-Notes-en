---
title: >-
  [Paper Note] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning
description: >-
  [ICLR 2026][Robotics][tool-integrated reasoning] This paper proposes THOR, a framework that systematically addresses three core challenges in tool-integrated mathematical reasoning for LLMs—data construction, fine-grained optimization, and inference enhancement—through three complementary components: the TIRGen data construction pipeline, hierarchical reinforcement learning (joint episode-level and step-level optimization), and a self-correction inference mechanism. THOR achieves state-of-the-art performance among models of comparable scale on benchmarks including MATH500 and AIME.
tags:
  - ICLR 2026
  - Robotics
  - tool-integrated reasoning
  - hierarchical RL
  - GRPO
  - code generation
  - self-correction
  - mathematical reasoning
date: 2026-05-08
content_hash: 92442fb7efbd74aa
---

# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning

**Conference**: ICLR 2026
**arXiv**: [2509.13761](https://arxiv.org/abs/2509.13761)
**Code**: [GitHub](https://github.com/JingMog/THOR)
**Area**: Robotics
**Keywords**: tool-integrated reasoning, hierarchical RL, GRPO, code generation, self-correction, mathematical reasoning

## TL;DR

This paper proposes THOR, a framework that systematically addresses three core challenges in tool-integrated mathematical reasoning for LLMs—data construction, fine-grained optimization, and inference enhancement—through three complementary components: the TIRGen data construction pipeline, hierarchical reinforcement learning (joint episode-level and step-level optimization), and a self-correction inference mechanism. THOR achieves state-of-the-art performance among models of comparable scale on benchmarks including MATH500 and AIME.

## Background & Motivation

1. As probabilistic next-token predictors, LLMs are inherently limited in high-precision numerical computation, equation solving, and symbolic manipulation, where sampling errors accumulate across multi-step calculations.

2. Tool-Integrated Reasoning (TIR) is an effective paradigm for mitigating this limitation, but faces three core challenges:

    - **Data construction difficulty**: Synthesizing data via prompting external large models such as GPT-4o introduces style mismatch issues and yields poor results for reasoning models (e.g., DeepSeek-R1); rule-injection methods such as START suffer from imprecise insertion points, leading to redundant code invocations.
    - **Coarse optimization granularity**: Existing RL methods (Agent-R, ToRL, ReTool) perform only episode-level optimization, using final answer correctness as the sole reward signal—this causes severe sparse reward problems in long reasoning chains, leaving intermediate code steps without fine-grained updates.
    - **Lack of error correction during inference**: Single-pass reasoning ignores immediate feedback from tool execution; when code execution fails, the model should backtrack and correct rather than proceed blindly.

3. SFT-based approaches (Toolformer, AIMO-2) require large amounts of high-quality demonstration data and generalize poorly.

4. **Core Insight**: The execution success rate of intermediate tool calls is a strong predictor of final answer correctness, providing a natural and dense reward signal for step-level optimization.

## Method

### Overall Architecture

THOR comprises three complementary components forming a complete pipeline:

1. **TIRGen Data Construction Pipeline**: A Generator-Refiner collaboration that automatically generates policy-aligned TIR training data $\mathcal{D}_{SFT}$.

2. **Hierarchical RL Training**: Cold Start SFT → joint training with episode-level optimization (final answer reward) and step-level optimization (code execution reward).

3. **Self-Correction Inference Mechanism**: During inference, tool execution feedback is leveraged to backtrack and regenerate failed steps.

The entire process is formalized as a think-act-observe loop: given a problem $q$, the model generates an alternating sequence $\tau=(r^1, a^1, o^1, \ldots, r^n)$, where $r^t$ denotes a natural language reasoning step, $a^t$ a code action, and $o^t$ the execution result.

### Key Designs

1. **TIRGen Data Construction Pipeline (Generator-Refiner Framework)**

    - The Generator produces natural language reasoning steps (with a per-step maximum length $L_{step}$), preserving the model's original reasoning style.
    - The Refiner evaluates whether each step can be converted to code execution (numerical computation, equation solving, etc.), extracts the pure logical reasoning portion $r_{logic}^t$, and converts it to executable Python code $a^t$.
    - Code is executed in a sandboxed environment to obtain observation $o^t$, which replaces the original computed result.
    - **Policy alignment advantage**: The Refiner observes only the single reasoning step (not the full problem or answer), so the generated data is naturally aligned with the Generator's policy distribution, avoiding performance degradation from out-of-distribution data.
    - **Reduced dependence on large models**: The Generator handles high-level mathematical reasoning, while the Refiner requires only basic instruction following and code generation capability; this task decomposition reduces the need for very large models.
    - **Multi-stage filtering**: Format consistency checks → code quality filtering (requiring sympy/numpy calls or control flow) → difficulty and invocation-count balanced sampling → exclusion of simple problems solvable by pure CoT.

2. **Hierarchical Reinforcement Learning (Joint Episode- and Step-Level Optimization)**

    - **Cold Start Phase**: SFT on $\mathcal{D}_{SFT}$ establishes foundational tool-calling patterns, enabling the model to master the think-act-observe format.
    - **Episode-Level Optimization**: Employs the GRPO algorithm with final answer correctness as reward ($\mathcal{R}_i = 1$ for correct, $0$ for incorrect); trajectories with execution failures are filtered to avoid improper penalization due to environment issues. Advantage function: $A_i = \frac{\mathcal{R}_i - \text{mean}(\mathcal{R})}{\text{std}(\mathcal{R})}$.
    - **Step-Level Optimization**: Targets code steps with execution failures via backtracking—the erroneous reasoning step $r^t$ is split into a prefix $r_{pre}^t$ and suffix $r_{suf}^t$ (of length $L_{suf}$); the prefix context is retained while the suffix and code action are regenerated, using code execution success rate as the step-level reward.
    - Positive samples are additionally reinforced by introducing a VAPO-style NLL loss $\mathcal{L}_{NLL}$ with weighting coefficient $\alpha$ to directly reinforce the likelihood of samples with positive advantage.

3. **Self-Correction Inference Mechanism**

    - During inference, if code action $a^t$ fails to execute, a backtracking mechanism is triggered.
    - The reasoning step $r^t$ is split into $r_{pre}^t$ and $r_{suf}^t$; based on the history up to $r_{pre}^t$, the model regenerates a revised reasoning suffix $\hat{r}_{suf}^t$ and corrected action $\hat{a}^t$.
    - Up to $N_{corr}$ retries are allowed; each retry regenerates only the suffix rather than the entire trajectory, incurring minimal additional computational overhead.

### Loss & Training

The total training loss combines episode-level and step-level objectives:

$$\mathcal{L}(\theta) = \mathcal{L}_{\pi_\theta}^{epis}(\theta) + \mathcal{L}_{\pi_\theta}^{step}(\theta)$$

Both levels adopt GRPO's clipped surrogate objective augmented with a VAPO-style NLL loss:

- Episode level: $\mathcal{L}^{epis}$ uses final answer correctness as reward; an indicator function $I(s_{i,t})$ ensures gradients are computed only over model-generated tokens (not external executor outputs).
- Step level: $\mathcal{L}^{step}$ uses code execution success rate as reward; each sample contains a single think-act-observe cycle to ensure dense rewards.

Training procedure: Cold Start SFT → joint episode- and step-level RL optimization.

## Key Experimental Results

### Datasets and Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| MATH500 | Mathematical reasoning | 500 problems spanning 7 difficulty levels |
| AIME 2024 | Competition mathematics | American Invitational Mathematics Examination 2024 |
| AIME 2025 | Competition mathematics | American Invitational Mathematics Examination 2025 |
| AMC | Competition mathematics | American Mathematics Competition |
| Minerva Math | Mathematical reasoning | STEM mathematics problem set |
| Olympiad Bench | Competition mathematics | Olympiad mathematics benchmark |
| HumanEval | Code generation | Function-level code generation |
| MBPP | Code generation | Basic programming problems |
| LiveCodeBench | Code generation | Continuously updated competitive programming problems |

### Baselines

| Model | Parameters | Type | Tool Use |
|-------|-----------|------|----------|
| QwQ-32B | 32B | Reasoning model | ✗ |
| DeepSeek-R1-Distill-32B | 32B | Distilled reasoning model | ✗ |
| ToRL-Qwen-32B | 32B | RL + tool calling | ✓ |
| ReTool-32B | 32B | RL + tool calling | ✓ |
| START-32B | 32B | Rule injection + tool calling | ✓ |
| DeepSeek-R1-7B | 7B | Distilled reasoning model | ✗ |
| STILL-3-1.5B | 1.5B | Small reasoning model | ✗ |
| AIMO-2 | 32B | SFT + tool calling | ✓ |

### Main Results (Math Benchmarks, Avg@4)

| Model | MATH500 | AIME24 | AIME25 | AMC | Minerva | OlympiadBench | Avg |
|-------|---------|--------|--------|-----|---------|---------------|-----|
| QwQ-32B | 96.4 | 79.2 | 72.0 | - | - | - | - |
| DeepSeek-R1-Distill-32B | 94.3 | 72.6 | 60.2 | - | - | - | - |
| ToRL-Qwen-32B | 95.6 | 75.2 | 67.3 | - | - | - | - |
| ReTool-32B | 95.2 | 72.5 | 67.5 | - | - | - | - |
| **THOR-32B** | **97.5** | **82.2** | **76.0** | - | - | - | **Best** |

**Key Findings**:

1. ⭐⭐⭐ **THOR comprehensively outperforms all models at the same scale**: achieving 97.5% on MATH500, 82.2% on AIME24, and 76.0% on AIME25, surpassing all same-scale baselines.

2. ⭐⭐⭐ **Strong cross-architecture generalization**: THOR is effective on both reasoning models (QwQ-32B backbone) and non-reasoning models (Qwen-2.5-32B backbone), and scales across parameter sizes (1.5B/7B/32B).

3. ⭐⭐ **Concurrent improvements on code benchmarks**: Positive gains are also observed on HumanEval, MBPP, and LiveCodeBench, indicating that step-level code optimization enhances general code generation capability.

4. ⭐⭐ **Low inference overhead**: The self-correction mechanism is triggered only upon execution failure and regenerates only the suffix rather than the entire trajectory, yielding average inference token counts lower than ToRL and ReTool.

### Ablation Study

| Configuration | MATH500 | AIME24 | AIME25 | Trend |
|---------------|---------|--------|--------|-------|
| Full THOR | **97.5** | **82.2** | **76.0** | Complete framework |
| − Step-level RL | 96.8 | 78.5 | 72.7 | ↓ Notable degradation |
| − Self-Correction | 96.4 | 80.3 | 73.5 | ↓ Inference decline |
| − TIRGen (replaced with GPT-4o data) | 95.9 | 76.8 | 70.3 | ↓ Significant degradation |
| Cold Start SFT only | 94.2 | 68.3 | 58.7 | ↓ Severe degradation |

**Key Findings**:

1. ⭐⭐⭐ **Step-level RL contributes most**: Removing it reduces AIME24 by 3.7 points and AIME25 by 3.3 points, validating the effectiveness of intermediate code execution success rate as a dense reward signal.

2. ⭐⭐ **TIRGen data alignment is critical**: Replacing with GPT-4o synthesized data reduces AIME25 by 5.7 points, demonstrating that policy-aligned training data substantially outperforms out-of-distribution data.

3. ⭐⭐ **Self-correction is most impactful on hard problems**: It contributes a 2.5-point gain on AIME25 (the most challenging benchmark), as harder problems are more prone to code execution failures requiring correction.

### Key Statistical Validation

The paper validates its core insight—that intermediate tool call success rate is strongly correlated with final answer correctness: trajectories in which all code calls succeed exhibit substantially higher final answer accuracy than those containing execution failures, providing theoretical grounding for step-level optimization.

## Highlights & Insights

1. ⭐⭐⭐ **Systematic framework design**: The three stages of data, training, and inference are mutually complementary; TIRGen addresses data quality, hierarchical RL addresses optimization, and self-correction addresses inference, forming a complete closed loop.
2. ⭐⭐⭐ **Dense rewards via hierarchical RL**: Leveraging code execution success rate as a step-level reward signal effectively mitigates the sparse reward problem in long reasoning chains.
3. ⭐⭐ **Generator-Refiner data construction**: The design in which the Refiner observes only a single step rather than the full problem elegantly achieves policy alignment while reducing dependence on external large models.
4. ⭐⭐ **Low-overhead self-correction**: Regenerating only the suffix rather than the entire trajectory incurs minimal computational cost.
5. ⭐ **Broad model applicability**: Effective across reasoning and non-reasoning models and multiple parameter scales.

## Limitations & Future Work

1. ⭐⭐ **Single tool type**: The current framework supports only a Python code executor and has not been extended to other tools such as symbolic solvers (Mathematica) or search engines.
2. ⭐⭐ **Fixed backtracking strategy**: The suffix length $L_{suf}$ is a fixed hyperparameter that does not adaptively adjust based on error type—too long wastes computation, while too short may fail to address the root cause of errors.
3. ⭐ **Restriction to mathematics**: The framework has not been evaluated on broader reasoning tasks such as scientific or logical reasoning.
4. ⭐ **Limited self-correction retries**: The $N_{corr}$ retry limit may still fail to repair certain systematic errors, and no mechanism exists for abandoning the current reasoning path and switching strategies.

## Summary

THOR is an end-to-end framework for enhancing tool-integrated mathematical reasoning. Its core contributions are: (1) the TIRGen pipeline, which generates high-quality TIR training data aligned with the policy model's distribution; (2) hierarchical RL, which jointly optimizes at the episode level (answer reward) and step level (code execution reward) to effectively alleviate sparse rewards; and (3) a self-correction mechanism that leverages tool feedback for low-overhead online error correction. The framework achieves state-of-the-art performance among same-scale models across multiple mathematical and code benchmarks, demonstrating a systematic solution for tool-integrated reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](on_entropy_control_in_llm-rl_algorithms.md)
- [\[ICLR 2026\] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)
- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[AAAI 2026\] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](../../AAAI2026/robotics/h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)
- [\[AAAI 2026\] Sketch-HARP: Hierarchical Autoregressive Sketch Generation for Flexible Stroke-Level Drawing Manipulation](../../AAAI2026/robotics/generating_sketches_in_a_hierarchical_auto-regressive_proces.md)

</div>

<!-- RELATED:END -->
