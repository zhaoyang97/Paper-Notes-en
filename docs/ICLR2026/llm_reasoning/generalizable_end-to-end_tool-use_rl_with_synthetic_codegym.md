---
title: >-
  [Paper Note] Generalizable End-to-End Tool-Use RL with Synthetic CodeGym
description: >-
  [ICLR2026][LLM Reasoning][tool-use] This paper proposes CodeGym, a framework that automatically converts programming problems into multi-turn interactive tool-use environments for reinforcement learning training of LLM a…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "tool-use"
  - "reinforcement-learning"
  - "LLM agent"
  - "synthetic environment"
  - "code-based training"
date: 2026-05-08
content_hash: e2e93bdf6609ea5a
---

# Generalizable End-to-End Tool-Use RL with Synthetic CodeGym

**Conference**: ICLR2026
**arXiv**: [2509.17325](https://arxiv.org/abs/2509.17325)  
**Code**: [StigLidu/CodeGym](https://github.com/StigLidu/CodeGym)  
**Area**: LLM Reasoning
**Keywords**: tool-use, reinforcement-learning, LLM agent, synthetic environment, code-based training

## TL;DR

This paper proposes CodeGym, a framework that automatically converts programming problems into multi-turn interactive tool-use environments for reinforcement learning training of LLM agents, achieving significant out-of-distribution generalization gains (e.g., +8.7 points on τ-Bench for Qwen2.5-32B).

## Background & Motivation

Tool-augmented large language model (LLM) agents extend their capabilities by invoking external tools (databases, search engines, code executors, etc.). However, existing training paradigms face two major bottlenecks:

1. **Limitations of SFT**: Supervised fine-tuning relies on static trajectories following hand-crafted patterns with limited coverage of environment and task configurations, resulting in poor generalization to novel tools or unseen workflows.
2. **Limitations of RL**: Existing RL training environments target only narrow tasks (e.g., code debugging assistants, information retrieval), limiting the potential of RL to promote generalization.

Core insight: **Code inherently encodes strict execution logic that is structurally analogous to real-world workflows**. For instance, a loop-until-condition code pattern mirrors iterative approval processes in practice. Programming problems therefore serve as an ideal foundation for constructing diverse tool-use training environments.

## Core Problem

How to build a **scalable and generalizable RL environment** that enables LLM agents to acquire tool-use skills through active exploration and interaction that transfer to real-world tasks?

## Method

### 1. CodeGym Environment Generation Pipeline

The pipeline consists of two stages:

**Gym Synthesis**:

- Given a programming problem and its solution code, reusable **atomic functions/logic** are extracted as callable tools.
- Tools may be standalone functions, computational utilities, or common code snippets (e.g., loop bodies).
- An LLM generates precise documentation (functionality, parameters, examples) for each tool; examples are withheld during training to encourage exploratory learning.
- Each environment is modeled as a POMDP: $\mathcal{E} = \langle \mathcal{S}, \mathcal{A}, T, R, \mathcal{O} \rangle$
- The action space includes general-purpose functions (Observe, Done) and domain-specific tools.
- Rewards are sparse and binary: +1 for a correct answer, 0 otherwise.

**Gym Verification**:

- Synthetic unit test inputs are generated to cover multiple difficulty levels and edge cases.
- The original programming solution is used to produce corresponding unit test outputs (ground truth).
- A **pass@K strategy** ($K=10$) is adopted: 10 candidate solution functions are generated, and an environment is considered solvable if any one passes all unit tests.
- Verified unit tests serve as task configurations for RL training.

### 2. Quality Control

A two-tier filtering mechanism ensures training data quality:

- **Tool-call complexity**: Task configurations with fewer than 10 or more than 256 tool calls are filtered out; each environment must contain at least 4 distinct tools.
- **Difficulty filtering**: Qwen2.5-32B-Instruct evaluates each configuration 4 times; only configurations with a pass rate $\leq 25\%$ are retained.

Final dataset: **13k environments, 80k+ task configurations**, averaging 6.52 tools and 44.07 steps to completion.

### 3. Difficulty Augmentation

To address the possibility that long-CoT models may bypass tool use through pure reasoning, task configurations are augmented at environment initialization to increase the difficulty of reasoning-only solutions.

### 4. Training Framework

- GRPO algorithm with batch size $512 \times 8$.
- Decoupled distributed rollout framework: CPU-side environment servers and GPU-side rollout workers.
- **Trial-then-Overwrite mechanism**: environment states are serialized and tool calls are executed in subprocesses; successful calls commit the state, while failed calls roll back and return error messages, ensuring training robustness.
- A maximum tool-call limit $T_{\max}$ is imposed to prevent infinite loops.

## Key Experimental Results

### Models and Settings

- Short-CoT: Qwen2.5 series (7B/14B/32B/72B)
- Long-CoT: QwQ-32B
- Training steps: models $\leq$32B saturate at ~100 steps; 72B at ~50 steps.

### OOD Benchmark Results (Core Highlights)

| Model | τ-airline | τ-retail | τ²-bench | ALFWorld | ZebraLogic | MMLU-Pro | Avg. |
|-------|-----------|----------|----------|----------|------------|----------|------|
| Qwen2.5-32B-Instruct | 26.8 | 41.4 | 24.7 | 66.8 | 24.2 | 70.0 | 42.3 |
| Qwen2.5-32B-CodeGym | **31.2** (+4.4) | **54.4** (+13.0) | **30.7** (+6.0) | **80.8** (+14.0) | **29.0** (+4.8) | **71.2** (+1.2) | **49.6** (+7.3) |
| QwQ-32B | 37.6 | 37.7 | 26.1 | 62.4 | 79.9 | 81.4 | 54.2 |
| QwQ-32B-CodeGym | **43.2** (+5.6) | **43.0** (+5.3) | **30.7** (+4.6) | **64.4** (+2.0) | 76.6 (−3.3) | 81.4 | **56.6** (+2.4) |

Key findings:

- **Larger models benefit more**: the 32B model achieves an average gain of +7.3 versus only +2.8 for 7B, suggesting generalization rather than memorization in larger models.
- **Tool-call counts increase during training**: the average number of tool calls made by the agent rises steadily and approaches the oracle, indicating that the agent learns more complete workflows.
- **Limitations of small models**: the 7B model produces the most tool calls, but many constitute repetitive failure-retry loops, exposing insufficient error diagnosis capability in smaller models.

### RL vs. SFT Comparison

- Oracle-SFT and Distillation-SFT perform adequately in-domain but exhibit **notable degradation** on OOD tasks.
- RL training is critical for generalization and cannot be substituted by SFT.

### Filtering Ablation

- No filtering (CodeGym-Full): OOD average 46.2 (+3.9)
- With filtering (CodeGym-Filter): OOD average 49.6 (+7.3), nearly doubling the gain.

## Highlights & Insights

1. **Elegant conceptualization**: leveraging the structural similarity between code execution logic and real-world workflows to convert programming problems into general agent training environments is both novel and intuitive.
2. **Complete pipeline**: the system forms a closed loop from data collection, environment synthesis, verification, and quality control to distributed RL training.
3. **Significant OOD generalization**: substantial improvements are achieved on tasks semantically unrelated to the training environments (e.g., retail customer service, household navigation).
4. **Scalable data support**: 13k environments and 80k+ task configurations far exceed existing agent training works in scale.
5. **Compelling qualitative analysis**: post-training agents demonstrate stronger multi-step planning before acting (illustrated via ALFWorld examples).

## Limitations & Future Work

1. **Environment diversity constrained by programming problems**: despite rich code logic, environments involving visual, physical, or other non-textual modalities are absent.
2. **Slight degradation in long-CoT reasoning**: QwQ-32B drops 3.3 points on ZebraLogic, suggesting a potential conflict between tool-use training and reasoning ability that warrants joint optimization.
3. **Limited benefit for small models**: the 7B model gains only 2.8 points and exhibits repetitive invocation issues, raising questions about the framework's effectiveness for smaller models.
4. **Sparse reward signal**: only binary rewards at the final answer are provided; the absence of process rewards may limit learning efficiency on long-horizon tasks.
5. **Multi-agent collaboration unexplored**: all experiments operate in a single-agent setting.

## Related Work & Insights

| Dimension | CodeGym | SWE-Gym | BrowseComp-Plus | ToolBench |
|-----------|---------|---------|-----------------|-----------|
| # Environments | 13k | Small | Small | Large-scale dataset |
| Interactive | ✅ Multi-turn | ✅ Code debugging | ✅ Web search | ❌ Static data |
| Generality | High (code → general tool-use) | Low (code only) | Low (search only) | Medium |
| RL Support | ✅ Full GRPO training | Limited | Limited | ❌ |
| Verifiable Reward | ✅ Unit tests | ✅ | Partial | ❌ |

The paradigm of **transferring code-derived capabilities to general agent skills** offers broad inspiration, analogous to the idea that "pretraining on code improves reasoning," extended here to interactive agent training. This work aligns with the RLVR (Reinforcement Learning with Verifiable Reward) trend and validates the effectiveness of verifiable rewards in agent training. Integration with process reward models could introduce finer-grained supervision signals for long-horizon tool-use sequences. The work also offers insights for agent benchmark design: environment diversity and complexity directly influence post-training generalization.

## Rating
- **Novelty**: 8/10 — The idea of converting programming problems into interactive agent environments is novel, though the core techniques (GRPO, POMDP) are established.
- **Experimental Thoroughness**: 9/10 — Multi-scale models, multi-dimensional OOD evaluation, ablation studies, and qualitative analysis are all provided.
- **Writing Quality**: 8/10 — Clear structure, rich figures and tables, and naturally motivated problem statement.
- **Value**: 8/10 — Provides a scalable general environment generation solution for agent training with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](../../CVPR2026/llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)
- [\[ICML 2026\] Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents](../../ICML2026/llm_reasoning/diversity_over_frequency_rethinking_tool_use_in_visual_chain-of-thought_agents.md)
- [\[ICML 2026\] MOSAIC: Learning When to Act or Refuse — Guarding Agentic Reasoning Models for Safe Multi-step Tool Use](../../ICML2026/llm_reasoning/learning_when_to_act_or_refuse_guarding_agentic_reasoning_models_for_safe_multi-.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
