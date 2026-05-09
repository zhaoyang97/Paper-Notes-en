---
title: >-
  [Paper Note] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling
description: >-
  [ACL 2026][LLM Alignment][Reward modeling] This paper proposes Plan-RewardBench, a trajectory-level preference benchmark targeting complex tool-augmented scenarios, designed to evaluate the ability of reward models to distinguish superior from inferior agent trajectories across multi-step planning, tool usage, and error recovery settings.
tags:
  - ACL 2026
  - LLM Alignment
  - Reward modeling
  - agent evaluation
  - trajectory-level preference
  - tool calling
  - planning benchmark
date: 2026-05-08
content_hash: ff4a8e3fa15e0b80
---

# Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling

**Conference**: ACL 2026
**arXiv**: [2604.08178](https://arxiv.org/abs/2604.08178)
**Code**: Unavailable (pending corporate approval for release)
**Area**: LLM Alignment
**Keywords**: Reward modeling, agent evaluation, trajectory-level preference, tool calling, planning benchmark

## TL;DR

This paper proposes Plan-RewardBench, a trajectory-level preference benchmark targeting complex tool-augmented scenarios, designed to evaluate the ability of reward models to distinguish superior from inferior agent trajectories across multi-step planning, tool usage, and error recovery settings.

## Background & Motivation

**State of the Field**: Large language models have evolved from passive conversational systems into agentic systems capable of autonomously invoking tools and performing complex reasoning, with behavioral scope expanding from single-turn responses to complete trajectories comprising user input, reasoning, tool execution, and environmental feedback.

**Limitations of Prior Work**: Existing RM benchmarks (e.g., RewardBench, RM-Bench) focus primarily on response-level preference evaluation, covering only limited dimensions such as helpfulness and safety, and are restricted to short-context scenarios. Tool-calling benchmarks (e.g., FC-RewardBench) validate only atomic action correctness while neglecting the evaluation of long-horizon planning behavior.

**Root Cause**: Agentic systems are inherently multi-turn in nature, yet existing benchmarks fail to assess the judgment capacity of reward models over long-horizon, multi-step trajectories, particularly on critical dimensions such as planning consistency, error recovery, and refusal quality.

**Paper Goals**: To construct a trajectory-level preference benchmark that systematically evaluates the ability of reward models to assess planning logic and tool-use faithfulness in complex tool-integrated scenarios.

**Starting Point**: Building on MCP tool registries and real execution environments, hard negative pairs that are "difficult to distinguish" are constructed through multi-model natural sampling, rule-based perturbation, and minimal edits.

**Core Idea**: Elevate RM evaluation from the response level to the trajectory level, covering four scenario families: safety refusal, tool irrelevance, complex planning, and robust error recovery.

## Method

### Overall Architecture

Plan-RewardBench formulates the task as pairwise trajectory preference judgment: given a tool environment $\mathcal{T}$, multi-turn user interactions, and two candidate trajectories $(\tau_A, \tau_B)$, a reward model determines which trajectory is superior. Three evaluation protocols are supported: DRM/GRM training preference, inference-time best-of-N reranking, and DPO-style optimization.

### Key Designs

1. **Four Scenario Family Design**:

    - Function: Covers the core challenge dimensions of agentic systems
    - Mechanism: Safety Refusal, Tool-Irrelevance, Complex Planning, Robust Recovery
    - Design Motivation: Existing benchmarks cover only a single dimension, whereas real-world agents must perform correctly across diverse scenario types

2. **Multi-Source Hard Negative Construction**:

    - Function: Generate deceptive rejection trajectories
    - Mechanism: 70% natural sampling + 22% minimal-edit perturbation + 8% rule injection, with length and format biases controlled to isolate semantic failures
    - Design Motivation: Simple negatives can be identified via surface cues (length/format); it is therefore necessary to construct negatives that are "nearly correct but semantically flawed"

3. **Multi-Judge Annotation with Human Auditing**:

    - Function: Ensure reliability of preference labels
    - Mechanism: $K=3$ LLM judges with median scoring + meta-review + human auditing (Cohen's $\kappa \in [0.71, 0.86]$)
    - Design Motivation: A single judge is prone to bias; multi-judge consensus combined with human validation ensures annotation quality

### Data Construction Pipeline

Tasks and tool environments are sourced from Toucan/MCP. Natural trajectories are obtained via multi-model, multi-parameter sampling using Qwen-Agent and OpenAI-Agent. Negative samples are then constructed through rule-based and minimal-edit methods. Finally, preference pairs are assembled through multi-judge scoring and human auditing.

## Key Experimental Results

### Main Results

| Model Type | Representative Model | Evaluation Protocol | Characteristics |
|---------|---------|---------|------|
| Discriminative RM | Inf-ORM-Llama3.1-70B | Pointwise scoring → select higher score | Evaluates each trajectory independently |
| Generative RM | Skywork-o1, etc. | Generative scoring | Evaluates via generation process |
| LLM-as-Judge | GPT-o3, Claude, etc. | Pairwise comparison | Directly compares two trajectories |

### Dataset Statistics

| Scenario | Pairs | Avg. Tokens (Chosen/Rejected) | Max Tokens |
|------|-----|---------------------------|-----------|
| Tool-Irrelevance | 275 | 1,363 / 1,358 | ~5K |
| Planning-Multi (Hard) | 73 | 6,523 / 6,554 | ~17K |
| Robust Recovery | 361 | 4,545 / 4,462 | ~29K |
| Safety Refusal | 51 | 1,219 / 2,233 | ~11K |

### Key Findings

- All three evaluator types (discriminative, generative, LLM-as-Judge) exhibit substantial performance degradation on long-horizon trajectories
- Tool-grounding hallucination (claiming tool usage without actual invocation) is the most prevalent failure mode in complex planning
- In the safety refusal scenario, delayed refusal (partial execution followed by refusal) is the primary source of confusion
- Blind retry is the most common error pattern in robust recovery

## Highlights & Insights

- This work is the first to systematically elevate RM evaluation from the response level to the agent trajectory level, filling a critical gap in agent alignment evaluation
- The hard negative construction methodology can serve as a general blueprint for building agent planning preference training data
- Human auditing results (Cohen's $\kappa > 0.7$) validate the reliability of the annotation pipeline
- All mainstream RMs are found to face significant challenges on long-horizon trajectories, highlighting the necessity of specialized training

## Limitations & Future Work

- Only text modality is covered; multimodal agent scenarios are not considered
- Data scale is constrained by the high cost of quality annotation
- The safety refusal scenario contains relatively few samples (51 pairs), limiting statistical significance
- Future work may extend to multimodal, longer-horizon, and more complex tool-chain scenarios

## Related Work & Insights

- RewardBench series (Lambert et al., 2025): Foundation for response-level RM evaluation
- AgentRewardBench (Lù et al., 2025): Web agent trajectory evaluation, but not in tool-augmented settings
- FC-RewardBench (Agarwal et al., 2025): Tool-calling correctness evaluation, limited to single-turn
- This work may inspire future research in the direction of RL-from-agent-feedback

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First trajectory-level preference benchmark targeting tool-augmented agents
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple model types with human auditing validation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with systematic scenario categorization
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in agent RM evaluation

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](../../NeurIPS2025/llm_alignment/provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[ACL 2026\] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](../../ICLR2026/llm_alignment/chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

<!-- RELATED:END -->
