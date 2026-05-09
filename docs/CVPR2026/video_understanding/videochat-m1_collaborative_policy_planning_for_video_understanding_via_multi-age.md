---
title: >-
  [Paper Note] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
description: >-
  [CVPR2026][Video Understanding][Multi-agent systems] This paper proposes VideoChat-M1, which replaces conventional fixed tool-calling strategies with Collaborative Policy Planning (CPP) and Multi-Agent Reinforcement Learning (MARL). Multiple policy agents dynamically generate, execute, and communicate tool-invocation plans, achieving state-of-the-art results on 8 video understanding benchmarks—surpassing Gemini 2.5 Pro by 3.6% and GPT-4o by 15.6% on LongVideoBench.
tags:
  - CVPR2026
  - Video Understanding
  - Multi-agent systems
  - multi-agent reinforcement learning
  - collaborative policy planning
  - video question answering
  - tool invocation
date: 2026-05-08
content_hash: 574b4a30e9333511
---

# VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning

**Conference**: CVPR2026
**arXiv**: [2511.19524](https://arxiv.org/abs/2511.19524)
**Code**: To be confirmed
**Area**: Video Understanding
**Keywords**: Multi-agent systems, multi-agent reinforcement learning, collaborative policy planning, video question answering, tool invocation

## TL;DR

This paper proposes VideoChat-M1, which replaces conventional fixed tool-calling strategies with Collaborative Policy Planning (CPP) and Multi-Agent Reinforcement Learning (MARL). Multiple policy agents dynamically generate, execute, and communicate tool-invocation plans, achieving state-of-the-art results on 8 video understanding benchmarks—surpassing Gemini 2.5 Pro by 3.6% and GPT-4o by 15.6% on LongVideoBench.

## Background & Motivation

- **Background**: Multimodal large language models (MLLMs) perform well on short videos but struggle with long-horizon videos that involve complex temporal and spatial structures. Agent-based frameworks that invoke specialized tools to extract key video cues—without feeding massive frame sequences directly into MLLMs—have demonstrated superior potential over end-to-end models.

- **Limitations of Prior Work**: Existing multi-agent video understanding frameworks rely on static, predefined tool-invocation rules that cannot adaptively discover diverse cues, limiting perception and reasoning over complex videos. Single-agent or fixed-strategy approaches cannot simultaneously handle perception, retrieval, and summarization across multiple temporal scales. Furthermore, existing multi-agent systems (e.g., CAMEL, MetaGPT) depend on engineered logic and fixed roles, lacking joint training mechanisms tailored to video-multimodal tasks. Prior multi-agent RL methods are largely confined to unimodal text tasks, neglecting the temporal and perceptual challenges unique to video.

- **Goal**: Introduce a learnable, collaborative policy planning framework that enables multiple heterogeneous agents to jointly optimize tool-invocation strategies through multi-agent reinforcement learning for video understanding.

## Method

### Overall Architecture: Collaborative Policy Planning (CPP)

VideoChat-M1 consists of a set of policy agents $\mathcal{G}=\{\mathcal{G}_i\}$, a set of video perception tools $\mathcal{T}=\{\mathcal{T}_j\}$, and a shared memory buffer $\mathcal{M}=\{\mathcal{M}_i\}$. The CPP paradigm comprises three core phases:

**Phase 1: Policy Generation**
Each agent independently generates its own tool-invocation policy based on the user query $\mathcal{Q}$ and the available tool set $\mathcal{T}$:
$$\mathcal{P}_i = \mathcal{G}_i(\mathcal{Q}, \mathcal{T})$$
The policy is an ordered tool-invocation plan $\mathcal{P}_i = \{\mathcal{P}_{i,1} \to \mathcal{P}_{i,2} \to \ldots \to \mathcal{P}_{i,N}\}$.

**Phase 2: Policy Execution**
Each agent executes tool calls step by step according to its policy:
$$\mathcal{A}_{i,n} = \mathcal{P}_{i,n}(\mathcal{V}, \mathcal{T}, \mathcal{A}_{i,n-1})$$
At each step, the agent selects the appropriate tool to analyze the video conditioned on the intermediate result of the previous step.

**Phase 3: Policy Communication**
After each execution step, all agents write their intermediate results to the shared memory $\mathcal{M}$. Each agent then decides whether to revise its subsequent policy based on its own plan and the team's shared memory:
$$\mathcal{P}'_i = \mathcal{G}_i(\mathcal{Q}, \mathcal{T}, \mathcal{M}, \mathcal{P}_i)$$
If the current policy remains optimal, it is kept unchanged; otherwise, subsequent steps are revised. Communication and execution alternate across multiple iterations, allowing each agent to continuously refine its strategy using the team's accumulated intermediate results.

Final answer aggregation varies by task type: multiple-choice questions are decided by majority voting; open-ended questions and temporal grounding tasks are summarized by the best-performing team model (Qwen3-8B).

### Key Designs: Multi-Agent Reinforcement Learning (MARL)

**Policy SFT Phase**: GPT-4o and DeepSeek-R1 are used to automatically annotate a high-quality policy plan dataset, filtered by two criteria: (1) the plan produces the correct answer, and (2) it can be executed successfully in a single pass without modification. Each agent is fine-tuned with cross-entropy loss to acquire basic policy generation capability.

**MARL Phase**: All agents are jointly optimized via GRPO with three reward signals.

## Loss & Training

- **Outcome reward $\mathcal{R}_{res}$**: Positive reward for correct answers; negative penalty for incorrect answers.
- **Format reward $\mathcal{R}_{format}$**: Reward for well-formed output (parseable plans, valid tool calls); penalty for format violations.
- **Collaboration reward $\mathcal{R}_{col}$**: GPT-4o evaluates each agent's intermediate planning trajectory on plan feasibility, tool-call appropriateness, and step management, yielding a binary reward (1/0); trajectories exceeding 5 tool calls receive a strong penalty.

The total reward is $\mathcal{R} = \mathcal{R}_{res} + \mathcal{R}_{format} + \mathcal{R}_{col}$, and model parameters are optimized using the GRPO objective. Agent Dropout is applied during training, randomly sampling DAG communication topologies at each step to improve generalization.

## Key Experimental Results

### Main Results

State-of-the-art results are achieved across 8 benchmarks and 4 task categories with a total team parameter count of 37B:

| Task | Benchmark | VideoChat-M1 | Comparison Model | Gain |
|------|-----------|:---:|------|:---:|
| Long-video QA | LongVideoBench | **82.3** | Gemini 2.5 Pro (78.7) | +3.6 |
| Long-video QA | LongVideoBench | **82.3** | GPT-4o (66.7) | +15.6 |
| Video reasoning | Video-Holmes | **60.5** | GPT-4o (42.0) | +18.5 |
| Video reasoning | VideoMMMU | **80.0** | Qwen3-VL-235B (74.7) | +5.3 |
| Spatial intelligence | VSIBench Avg | **71.9** | Gemini 1.5 Pro (45.4) | +26.5 |
| Temporal grounding | Charades-STA | **67.7** | Eagle-2.5-8B (65.9) | +1.8 |

### Efficiency Comparison

VideoChat-M1 uses on average only 69.9 frames (12%–18% of competing models) and 19.8 s inference time (9%–22% of competing models), yet achieves comprehensively superior performance. Compared to GPT-4o (384 frames / 153.6 s) and Gemini 1.5 Pro (568 frames / 227.2 s), VideoChat-M1 achieves highly efficient frame sampling through intelligent tool invocation, obtaining better results with less than one-fifth of the computational resources.

### Implementation Details

Training uses 8×A100 80G GPUs, with a learning rate of 1e-6 for SFT and 1e-7 for MARL. SFT runs for 1 epoch (batch size 32); MARL reaches peak performance at 200 steps (4 rollouts, batch size 8). The agent team comprises four heterogeneous models—Qwen2.5-3B, Qwen2.5-7B, Qwen3-4B, and Qwen3-8B—totaling approximately 37B parameters.

### Ablation Study

- **Number of agents**: Performance improves steadily from 1 to 4 agents, saturating beyond 4.
- **Architectural diversity is critical**: The heterogeneous agent group (Qwen2.5-3B/7B + Qwen3-4B/8B) outperforms homogeneous groups; structural redundancy reduces discussion diversity.
- **Contribution of MARL components**: Removing $\mathcal{R}_{col}$ drops performance by ~1 point; removing $\mathcal{R}_{format}$ has a similar effect; removing Agent Dropout causes the largest drop (~2 points), making it the most critical regularization component.
- **SFT and MARL are complementary**: SFT alone yields +6.6, MARL alone yields +10.9, and their combination achieves the peak (60.5 / 82.3), confirming that both initialization priors and emergent collaboration are indispensable.
- **LoRA approaches full fine-tuning**: LoRA updates only ~2% of parameters yet achieves performance only marginally below full fine-tuning (59.4 vs. 60.5), offering a lightweight deployment option.
- **Majority voting is optimal**: Majority voting (60.5 / 82.3) > agent decision (60.2 / 81.6) > top-score selection (59.9 / 81.2).
- **Outperforms closed-source agent teams**: The trained 37B team substantially surpasses an untrained 4×GPT-4o team (+7.8 / +9.4) and a 4×DeepSeek-R1 team (+8.7 / +10.9), demonstrating that collaborative fine-tuning injects task-specific coordination patterns that zero-shot reasoning cannot discover.

## Highlights & Insights

- **First multi-agent policy learning framework for video understanding**: Replacing fixed tool-invocation strategies with learnable collaborative policy planning represents a paradigm shift in this direction.
- **Elegant three-phase CPP design**: The generate→execute→communicate iterative loop enables agents to dynamically revise strategies and fully exploit team intermediate information, offering greater flexibility than static role assignment.
- **Exceptional efficiency**: Only 69.9 frames and 19.8 s inference time are required, far below GPT-4o and similar models, while achieving comprehensively superior results.
- **37B parameters rivaling 235B**: On benchmarks such as VideoMMMU, the collaboration of four small models matches Qwen3-VL-235B.
- **Highly thorough ablation**: Covers agent count, composition, diversity, reward components, training strategy, and decision mechanisms across multiple dimensions.

## Limitations & Future Work

- **Collaboration reward depends on GPT-4o evaluation**: The collaboration reward for intermediate processes uses GPT-4o as an external evaluator, introducing additional API costs and evaluation bias, which limits scalability in large-scale training.
- **Tool set scope insufficiently discussed**: The paper does not detail the specific composition of $\mathcal{T}$ or its extensibility, and the mechanism for integrating new tools is unclear.
- **High training cost**: The pipeline requires SFT data annotation via GPT-4o and DeepSeek-R1 followed by MARL, resulting in a complex workflow that depends heavily on strong closed-source models.
- **Validation limited to QA-style tasks**: Although four task categories are covered, all are in the form of question answering or grounding; generalization to video generation, editing, or summarization remains unknown.
- **Inter-agent communication overhead**: All agents share intermediate results after each execution step; communication cost grows with the number of agents and steps, which may also explain why performance saturates beyond four agents.

## Related Work & Insights

- **Single-agent video tool invocation**: VideoAgent, VideoChat-Flash, and InternVideo2.5 enhance video understanding via a single agent with retrieval/search tools, but their strategies are fixed and non-learnable.
- **Training-free multi-agent frameworks**: LVAgent and VCA adopt static collaboration with fixed role assignments, lacking adaptivity and being constrained by predefined rules.
- **Agent + RL training**: VideoChat-R1/R1.5 apply RL to train the reasoning capability of a single agent; this work is the first to extend RL to joint multi-agent training for optimizing inter-agent collaboration.
- **Multi-agent RL (text domain)**: CAMEL and MetaGPT are confined to multi-agent collaboration in purely textual domains; this work introduces multi-agent RL to vision-language multimodal tasks for the first time.
- **Video RAG methods**: VideoRAG and ReAgent-V improve long-video understanding via retrieval augmentation but lack learnable collaborative mechanisms.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First trainable multi-agent collaborative policy learning framework for video understanding; the CPP+MARL paradigm is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 8 benchmarks across 4 task categories; ablation covers agent composition, rewards, training strategies, and decision mechanisms.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical formulation, and intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ — The 37B multi-agent team surpasses GPT-4o and Gemini 2.5 Pro, demonstrating the remarkable potential of small-model collaboration.

<!-- end of note -->

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[CVPR 2026\] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)
- [\[CVPR 2026\] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)
- [\[CVPR 2026\] TrajTok: Learning Trajectory Tokens Enhances Video Understanding](trajtok_learning_trajectory_tokens_enables_better_video_understanding.md)

<!-- RELATED:END -->
