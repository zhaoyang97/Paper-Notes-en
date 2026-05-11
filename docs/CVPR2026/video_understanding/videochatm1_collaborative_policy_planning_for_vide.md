---
title: >-
  [Paper Note] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
description: >-
  [CVPR 2026][Video Understanding][Multi-Agent] VideoChat-M1 proposes the Collaborative Policy Planning (CPP) paradigm and a Multi-Agent Reinforcement Learning (MARL) training framework, enabling four heterogeneous VLM agents to dynamically generate and update tool-calling policies for video understanding. It surpasses Gemini 2.5 Pro by 3.6% and GPT-4o by 15.6% on LongVideoBench.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-Agent
  - Collaborative Policy Planning
  - MARL
  - GRPO
date: 2026-05-08
content_hash: 3d5750b6d6f51671
---

# VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2511.19524](https://arxiv.org/abs/2511.19524)
**Code**: None
**Area**: Video Understanding
**Keywords**: Multi-Agent, Collaborative Policy Planning, MARL, GRPO, Video Understanding

## TL;DR

VideoChat-M1 proposes the Collaborative Policy Planning (CPP) paradigm and a Multi-Agent Reinforcement Learning (MARL) training framework, enabling four heterogeneous VLM agents to dynamically generate and update tool-calling policies for video understanding. It surpasses Gemini 2.5 Pro by 3.6% and GPT-4o by 15.6% on LongVideoBench.

## Background & Motivation

**Limitations of Prior Work**: Existing agent-based video understanding frameworks universally employ **static, non-learnable tool-calling policies** with predefined tool selection orders that do not adapt dynamically to video content or question type, limiting the discovery and exploitation of diverse cues in spatiotemporally complex videos.

**Single-Agent Bottleneck**: A single agent struggles to simultaneously handle perception, retrieval, and synthesis. Even when equipped with retrieval, memory, and search tools, its general-purpose design constrains effective integration and reasoning.

**Deficiencies of Training-Free Multi-Agent Approaches**: Methods such as LVAgent rely solely on static role assignments and fixed textual logic, lacking trainable collaborative policies and thus unable to adaptively adjust collaboration patterns through learning. Furthermore, existing RL methods are confined to the unimodal text domain and cannot address the temporal and perceptual challenges inherent to video.

**Core Problem**: How can multiple agents dynamically generate, execute, and coordinate tool-calling policies to handle complex video understanding tasks? And how can multiple heterogeneous agents be jointly trained to learn effective collaboration?

## Method

### Overall Architecture

VideoChat-M1 consists of two core components: the **Collaborative Policy Planning (CPP)** inference paradigm and the **Multi-Agent Reinforcement Learning (MARL)** training framework.

The system comprises four heterogeneous policy agents (Qwen3-8B, Qwen3-4B, Qwen2.5-7B, Qwen2.5-3B; ~37B total parameters), a set of video perception tools $\mathcal{T}$ (including Global Sampling, Video Retrieval, Image Retrieval, Rough/Fine Browser, Spatial Tool, and Grounding Tool, totaling 7 tools), and a shared memory buffer $\mathcal{M}$.

Inference pipeline: user query $\mathcal{Q}$ + video $\mathcal{V}$ → each agent independently generates a tool-calling plan → tools are executed step by step with intermediate results exchanged via shared memory → each agent decides whether to update subsequent plans based on peer information → after multiple iterations, each agent aggregates its answer → majority voting (multiple-choice) or designated-agent summarization (open-ended) yields the final answer.

### Key Designs

1. **Collaborative Policy Planning (CPP)**:

    - **Function**: Enables multiple agents to collaborate dynamically in generating and updating tool-calling policies.
    - **Mechanism**: Policy planning is decomposed into three alternating iterative phases — **Policy Generation** (each agent autonomously generates an initial tool-calling sequence $\mathcal{P}_i = \{\mathcal{P}_{i,1} \to \mathcal{P}_{i,2} \to \cdots \to \mathcal{P}_{i,N}\}$ based on the question), **Policy Execution** (tools are called step by step to obtain video cues $\mathcal{A}_{i,n} = \mathcal{P}_{i,n}(\mathcal{V}, \mathcal{T}, \mathcal{A}_{i,n-1})$), and **Policy Communication** (after each execution step, intermediate results are written to shared memory, and each agent consults team information to decide whether to revise subsequent plans $\mathcal{P}'_i = \mathcal{G}_i(\mathcal{Q}, \mathcal{T}, \mathcal{M}, \mathcal{P}_i)$).
    - **Design Motivation**: Fixed policies cannot cover the diversity of video content. By having multiple agents generate diverse strategies and dynamically updating them through communication, richer video cues can be extracted from different perspectives.

2. **Multi-Agent Reinforcement Learning (MARL)**:

    - **Function**: Jointly optimizes the collaborative policies of multiple agents; this is the first joint multi-agent RL training framework for video understanding.
    - **Mechanism**: Training proceeds in two stages — **Policy SFT** (GPT-4o and DeepSeek-R1 are used to automatically annotate high-quality policy data; samples with correct answers and plans requiring no revision are selected for SFT warm-up of each agent); **Joint GRPO Optimization** (three reward signals are designed: outcome reward $\mathcal{R}_{res}$, format reward $\mathcal{R}_{format}$, and collaboration reward $\mathcal{R}_{col}$; GRPO jointly optimizes all agents).
    - **Design Motivation**: Zero-shot inference cannot discover effective coordination patterns (even a GPT-4o ensemble scores only 56.2 vs. VideoChat-M1's 60.5 on Video-Holmes). RL training is necessary to inject task-specific collaborative behaviors.

3. **Agent Dropout**:

    - **Function**: A regularization mechanism during training to prevent over-co-adaptation among agents.
    - **Mechanism**: At each training step, a DAG is randomly sampled from the fully connected agent graph as the communication topology, analogous to dropout in neural networks.
    - **Design Motivation**: Fixed communication topologies cause agents to become overly reliant on specific teammates. Randomizing the topology forces agents to develop more robust and flexible communication strategies. Ablation studies confirm this is "the most important regularizer" (removing it costs 2 points).

### Loss & Training

**SFT Stage**: Cross-entropy loss maximizing the likelihood of generating ground-truth policy plans. Learning rate: 1e-6; batch size: 32; 1 epoch.

**MARL Stage**: GRPO objective comprising a reward-seeking term and KL divergence regularization:

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{k=1}^{K} \frac{\pi_\theta(o_k)}{\pi_{\theta_{\text{old}}}(o_k)} \cdot A_R^{(k)} - \beta \, D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

Three reward signals: $\mathcal{R}_{res}$ (positive reward for correct answers, negative penalty for incorrect ones), $\mathcal{R}_{format}$ (whether tool calls are valid and executable), and $\mathcal{R}_{col}$ (GPT-4o evaluates the quality of intermediate collaborative trajectories as a binary reward; a strong penalty is applied when more than 5 tool calls are made). Learning rate: 1e-7; 4 rollouts; batch size: 8; optimal performance is reached in only 200 steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | VideoChat-M1 (37B) | GPT-4o | Gemini 2.5 Pro | Qwen3-VL-235B | Best Agent Method |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| LongVideoBench | Acc | **82.3** | 66.7 | 78.7 | - | 71.6 (DeepVideoDiscovery) |
| Video-MME (Avg) | Acc | **83.2** | 71.9 | 84.3 | 79.2 | 75.7 (VideoRAG-72B) |
| MLVU (M-avg/G-avg) | Acc | **84.2/76.7** | 70.3/65.3 | - | - | 72.9/73.1 (VideoRAG-72B) |
| VideoMMMU | Acc | **83.4** | 61.2 | 83.6 | 74.7 | 76.2 (VideoChat-A1) |
| Video-Holmes | Acc | **60.5** | 42.0 | 45.7 | - | - |
| MMR-V (CoT) | Acc | **5.92** | 5.80 | - | - | - |
| VSIBench (Avg) | Acc | **71.9** | 34.0 | - | 62.6 | - |
| Charades-STA | mIOU | **67.7** | - | - | 64.8 | 65.9 (Eagle-2.5) |

### Efficiency Comparison

| Model | Avg. Frames | Inference Time | LongVideoBench | Video-MME |
|------|:---:|:---:|:---:|:---:|
| Qwen2-VL-72B | 568 | 90.5s | 55.6 | 71.2 |
| GPT-4o | 384 | 153.6s | 66.7 | 71.9 |
| Gemini-1.5-Pro | 568 | 227.2s | 64.0 | 75.0 |
| **VideoChat-M1** | **69.9** | **19.8s** | **82.3** | **83.2** |

### Ablation Study

**Number and Combination of Agents (Table 3)**:

| No. of Agents | Combination | Video-Holmes | LongVideoBench |
|:---:|------|:---:|:---:|
| 1 | Qwen3-8B | 31.2 | 61.9 |
| 2 | Qwen3-4B + Qwen3-8B | 43.5 | 67.9 |
| 3 | Qwen2.5-7B + Qwen3-4B + Qwen3-8B | 55.9 | 78.9 |
| 4 | All 4 heterogeneous agents | **60.5** | **82.3** |

**Heterogeneous vs. Homogeneous Agents (Table 4)**:

| Configuration | Video-Holmes | LongVideoBench |
|------|:---:|:---:|
| 4× Qwen2.5-3B (homogeneous) | 55.8 | 79.2 |
| 2× Qwen2.5-3B + 2× Qwen2.5-7B | 55.9 | 79.3 |
| 2× Qwen3-4B + 2× Qwen3-8B | 58.8 | 80.9 |
| Fully heterogeneous (4 distinct models) | **60.5** | **82.3** |

**MARL Component Ablation (Table 6)**:

| $\mathcal{R}_{format}$ | $\mathcal{R}_{col}$ | $\mathcal{R}_{res}$ | Agent Dropout | Video-Holmes | LongVideoBench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✓ | ✗ | ✓ | 32.4 | 63.8 |
| ✓ | ✗ | ✓ | ✓ | 59.4 | 81.1 |
| ✗ | ✓ | ✓ | ✓ | 60.2 | 82.0 |
| ✓ | ✓ | ✓ | ✗ | 58.5 | 79.9 |
| ✓ | ✓ | ✓ | ✓ | **60.5** | **82.3** |

**SFT and MARL Are Both Necessary (Table 7)**:

| SFT | MARL | Video-Holmes | LongVideoBench |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 52.1 | 69.3 |
| ✓ | ✗ | 55.2 | 75.9 |
| ✗ | ✓ | 57.9 | 80.2 |
| ✓ | ✓ | **60.5** | **82.3** |

**vs. Closed-Source LLM Agent Ensembles (Table 5)**:

| Agent Ensemble | Video-Holmes | LongVideoBench |
|----------|:---:|:---:|
| 4× GPT-4o | 52.7 | 72.9 |
| 4× DeepSeek-R1 | 51.8 | 71.4 |
| 2× GPT-4o + 2× DeepSeek-R1 | 56.2 | 75.9 |
| **VideoChat-M1 (37B)** | **60.5** | **82.3** |

### Key Findings

- The outcome reward $\mathcal{R}_{res}$ is the most critical signal in MARL; removing it causes Video-Holmes to drop sharply from 60.5 to 32.4.
- Agent Dropout is "the most important regularizer"; removing it causes LongVideoBench to drop by 2.4 points (79.9 vs. 82.3).
- The heterogeneous agent ensemble (4 distinct architectures) significantly outperforms homogeneous ensembles (e.g., 4× identical models), as architectural diversity promotes discussion diversity.
- Even running the CPP pipeline with a GPT-4o ensemble yields only 56.2/75.9, far below MARL-trained VideoChat-M1 (60.5/82.3), demonstrating that zero-shot inference cannot discover effective coordination patterns.
- Majority voting (60.5/82.3) > designated-agent decision (60.2/81.6) > score-based selection (59.9/81.2).
- Only 69.9 frames and 19.8s of inference time are required, far exceeding the efficiency of feeding large numbers of frames to a single model.

## Highlights & Insights

- **Novel CPP Paradigm**: The generate→execute→communicate iterative loop is elegant and natural, allowing agents to dynamically revise strategies based on peer information — a clear departure from prior fixed-policy or training-free multi-agent schemes.
- **First Joint Multi-Agent RL Framework for Video Understanding**: Three complementary reward signals (outcome, format, collaboration) are designed, with the collaboration reward evaluating intermediate process quality (a process reward) rather than solely the final answer.
- **Elegant Regularization via Agent Dropout**: Randomizing the communication topology to prevent co-adaptation is inspired by neural network dropout — simple yet effective.
- **Exceptional Efficiency**: With ~37B total parameters distributed across four small models plus tool models, VideoChat-M1 uses only 69.9 frames and 19.8s inference time, matching or surpassing 235B-scale models and GPT-4o across multiple benchmarks.

## Limitations & Future Work

- Parallel inference across four models plus multiple tool models entails **high deployment complexity**, posing substantial engineering challenges in practical applications.
- The collaboration reward relies on GPT-4o as an evaluator, incurring **non-trivial training costs** and introducing an external dependency.
- The current tool set is fixed at 7 predefined tools; **autonomous tool discovery or creation by agents** has not been explored.
- For simple video questions, the multi-agent framework may constitute **over-engineering**.
- Performance saturates when scaling homogeneous agents beyond four, indicating **limited scalability** in that direction.

## Related Work & Insights

- **vs. VideoChat-A1/VideoRAG**: These methods employ a single agent or training-free multi-agent setups with fixed policies; VideoChat-M1's CPP paradigm makes policies dynamic and learnable.
- **vs. Video-R1/VideoChat-R1**: These apply RL to optimize single-model reasoning; VideoChat-M1 is the first joint multi-agent RL training framework.
- **vs. GPT-4o/Gemini**: Even running the CPP pipeline with a GPT-4o ensemble (56.2/75.9) falls far short of MARL-trained VideoChat-M1 (60.5/82.3), confirming that MARL injects coordination patterns that zero-shot inference cannot discover.
- **Insights**: The multi-agent collaboration + RL paradigm is transferable to other multimodal tasks; the idea of randomizing communication topology via Agent Dropout offers reference value for other multi-agent systems; the tool set in CPP could be made extensible, enabling agents to learn to compose new tools during RL training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First multi-agent RL framework for video understanding; both the CPP paradigm and MARL training are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 8 benchmarks × 4 task types, with extensive ablations covering agent count, heterogeneity, reward components, training stages, and decision mechanisms.
- **Writing Quality**: ⭐⭐⭐⭐ — Overall clear with coherent method-experiment logic; tables are numerous but information-dense.
- **Value**: ⭐⭐⭐⭐⭐ — A 37B-parameter system surpassing GPT-4o and 235B-scale models, establishing a new paradigm for multi-agent video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[CVPR 2026\] CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [\[CVPR 2026\] TrajTok: Learning Trajectory Tokens Enhances Video Understanding](trajtok_learning_trajectory_tokens_enables_better_video_understanding.md)

</div>

<!-- RELATED:END -->
