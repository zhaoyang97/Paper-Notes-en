---
title: >-
  [Paper Note] M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining
description: >-
  [ICLR 2026][LLM Agent][GUI Agent] This paper proposes M2-Miner, the first MCTS-based automated data mining framework for mobile GUI agents. Through the collaboration of three agents—InferAgent, OrchestraAgent, and JudgeAgent—combined with an intent recycling strategy and progressive model-in-the-loop training, M2-Miner generates SOTA-quality data at 1/18 the cost of human annotation.
tags:
  - ICLR 2026
  - LLM Agent
  - GUI Agent
  - Data Mining
  - Monte Carlo Tree Search
  - Multi-Agent Collaboration
  - Intent Recycling
date: 2026-05-08
content_hash: f51a0015aedf9d7a
---

# M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining

**Conference**: ICLR 2026
**arXiv**: [2602.05429](https://arxiv.org/abs/2602.05429)
**Code**: Coming soon
**Area**: LLM Agent
**Keywords**: GUI Agent, Data Mining, Monte Carlo Tree Search, Multi-Agent Collaboration, Intent Recycling

## TL;DR

This paper proposes M2-Miner, the first MCTS-based automated data mining framework for mobile GUI agents. Through the collaboration of three agents—InferAgent, OrchestraAgent, and JudgeAgent—combined with an intent recycling strategy and progressive model-in-the-loop training, M2-Miner generates SOTA-quality data at 1/18 the cost of human annotation.

## Background & Motivation

Training GUI agents (agents that manipulate graphical interfaces to fulfill user intents) relies heavily on high-quality intent-trajectory data. Existing datasets face three major challenges:

1. **High construction cost**: Manual annotation of each data sample takes hours (e.g., Android Control's 88k images cost ~$31,662).
2. **Poor data quality**: Both manually annotated and automatically mined data frequently contain redundant steps, ambiguous intent descriptions, and biased action paths.
3. **Low data richness**: Existing datasets record only a single successful path (flat structure), with monotonic intents and insufficient descriptive information.

Limitations of existing automatic mining approaches:

- **AgentQ**: MCTS-based but limited to HTML-parsed web environments; does not support mobile visual scenarios.
- **OS-Genesis**: Rule-based step-by-step interaction with inverse task synthesis; lacks structured exploration.
- Directly applying vanilla MCTS to GUI data mining is highly inefficient: random expansion and rollout-based reward computation are prohibitively expensive.

## Method

### Overall Architecture

M2-Miner is built on the four-phase MCTS cycle (Selection → Expansion → Simulation → Backpropagation) and introduces a three-agent collaborative framework to enhance the expansion and simulation phases.

The **intent-trajectory tree** is formalized as $\mathcal{T} = (\mathcal{V}, \mathcal{A}, \mathcal{P}, \mathcal{I})$:

- $\mathcal{V}$: node set, where each node corresponds to a GUI state
- $\mathcal{A}$: executable action set (tap, swipe, type)
- $\mathcal{P}$: edge set (state transition relations)
- $\mathcal{I}$: user intent set

Each node $v$ is defined as a quintuple $(\mathit{img}_v, \mathit{meta}_v, Q_v, N_v, \mathit{stat}_v)$.

### Key Designs

**Three-Agent Collaborative Framework**:

1. **InferAgent**: During the expansion phase, infers the action most likely to achieve the target intent. Multiple MLLMs generate $K$ candidate actions to ensure diversity. Historical actions are included in the prompt to prevent duplicate generation.

2. **OrchestraAgent**: Merges equivalent actions and ranks them by confidence. A multiple-choice query format is used across $K-1$ queries to produce a ranked action queue, with ranked actions assigned decreasing initial UCT values.

3. **JudgeAgent**: Replaces costly rollout simulation. Analyzes the GUI screenshot of newly expanded nodes, assesses task completion status, and computes rewards:

$$r_{\text{intermediate}} = \frac{\exp(logits_{\text{valid}})}{\exp(logits_{\text{valid}}) + \exp(logits_{\text{invalid}})}$$

Softmax normalization is applied to the MLLM output logits to produce reward values in $[0,1]$. Node $Q$-value update:

$$Q_i = \frac{Q_{i-1} \times N_{i-1} + R_i}{N_{i-1} + 1}, \quad N_i = N_{i-1} + 1$$

**Intent Recycling Strategy**:

Core observation: Beyond the successful path for the original intent, other paths in the MCTS tree may correspond to valuable new intents. For example, in a map application where the original intent is to query a route, an accidental tap on a "ride-hailing" button produces an additional ride-hailing intent trajectory.

Procedure:

1. For each completed tree, consider all paths from the root to each node.
2. Use an MLLM-based intent recycling filter to evaluate trajectory quality.
3. For trajectories passing the filter, use an MLLM to generate matching new intents.
4. JudgeAgent verifies whether the terminal node of the trajectory corresponds to a successful state.

### Loss & Training

**Progressive Model-in-the-Loop Training**:

- **Warm-up phase**: Train InferAgent and JudgeAgent on public datasets to establish baseline capabilities.
- **Stage 1 (Basic intents)**: Collect screenshots from popular app homepages → generate common service intents → conditional rewriting for expansion → mine trajectories → train.
- **Stage 2 (Complex intents)**: Add conditions and functional combinations to Stage 1 intents (e.g., "book hotel" → "book hotel + book flight") → mine → train.
- **Stage 3 (Recycled intents)**: Apply intent recycling to all historical trees → enrich trajectories → train.

Base models: InferAgent and JudgeAgent use Qwen2.5-VL-7B; OrchestraAgent uses Qwen2.5-VL-72B.

## Key Experimental Results

### Main Results: GUI Agent Performance

| Model | AC-Low TP/SR | AC-High TP/SR | AITZ TP/SR | CAGUI TP/SR |
|------|:-:|:-:|:-:|:-:|
| GPT-4o | 74.3/19.4 | 66.3/20.8 | 70.0/35.3 | 3.67/3.67 |
| Qwen2.5-VL-7B | 94.1/85.0 | 75.1/62.9 | 78.4/54.6 | 74.2/55.2 |
| UI-TARS-7B* | 98.0/90.8 | 83.7/72.5 | 80.4/65.8 | 88.6/70.0 |
| OS-Genesis-7B | 90.7/74.2 | 66.2/44.5 | 20.0/8.5 | 38.1/14.5 |
| GUI-Owl-7B | 93.8/90.0 | 81.5/72.8 | 78.9/65.1 | 80.0/59.2 |
| **M2-Miner-7B** | **97.5/93.5** | 81.8/**72.9** | **81.3/69.4** | **88.8/70.2** |

M2-Miner-7B achieves SOTA on nearly all benchmarks, and consistently surpasses UI-TARS-7B—which relies on large-scale proprietary data—on the SR metric.

### Data Efficiency Comparison

| Dataset | Images | Automated | RL-Ready | Cost (USD) | Cost/Image |
|------|:-:|:-:|:-:|:-:|:-:|
| Android Control | 88k | ✗ | ✗ | $31,662 | $0.36 |
| AMEX | 38k | ✗ | ✗ | $13,680 | $0.36 |
| GUI-Odyssey | 119k | ✗ | ✗ | $42,816 | $0.36 |
| **M2-Miner-Agent** | 20k | **✓** | **✓** | **$466** | **$0.02** |

The per-image cost is only $0.02, which is **1/18** that of manual annotation.

### Ablation Study

**Multi-Agent Framework Efficiency**:

Compared to vanilla MCTS using only InferAgent, M2-Miner's efficiency gain grows exponentially with task complexity, reaching a **64× speedup** at task length 9.

**Model-in-the-Loop Training**:

| Stage | TP | SR |
|------|:-:|:-:|
| Warm-up | 85.0 | 64.2 |
| +Stage 1 (Basic intents) | 86.5 | 67.3 |
| +Stage 2 (Complex intents) | 87.6 | 69.1 |
| +Stage 3 (Recycled intents) | 88.2 | 69.9 |

**Data Structure Ablation**:

| Setting | TP | SR |
|------|:-:|:-:|
| Actions only | 85.2 | 66.8 |
| Actions + descriptions | 88.2 | 69.9 |
| Actions + descriptions + preferences | **88.8** | **70.2** |

### Key Findings

1. **Automatically mined data outperforms human annotation in quality**: The DQA of M2-Miner data exceeds that of Android Control and AITZ.
2. **Intent recycling substantially enriches diversity**: t-SNE visualization shows that recycled intents cover a broader semantic space.
3. **Descriptions and preference data provide additional value**: Semantic descriptions and preference signals retained in the MCTS tree contribute positively to training.
4. **Generalization to unseen scenarios**: On CAGUI, where no training data was available, M2-Miner improves the SR of Qwen2.5-VL-7B from 55.2% to 70.2%.

## Highlights & Insights

- **First application of MCTS to mobile GUI data mining**: The tree structure comprehensively records the exploration process, offering far richer information than flat-structured datasets.
- **Process reward replaces outcome reward**: JudgeAgent's intermediate node reward design avoids expensive rollouts, which is the key to efficiency gains.
- **Elegance of intent recycling**: Transforming "failed paths" during MCTS exploration into valuable data is a particularly elegant insight.
- **Exceptional cost-effectiveness**: A model trained on data mined for $466 outperforms one trained on $42,816 worth of human-annotated data.
- **Natural acquisition of preference data**: The MCTS tree inherently contains positive examples (successful paths) and negative examples (failed branches), enabling direct construction of preference pairs.

## Limitations & Future Work

1. Validation is currently limited to mobile environments; extension to web and desktop settings requires additional adaptation.
2. OrchestraAgent relies on a 72B model; although mining costs are lower than manual annotation, API expenses remain non-trivial.
3. The quality of intent recycling depends on the MLLM's intent generation capability, which may introduce inaccurate intent descriptions.
4. The progressive training strategy requires multiple rounds of mining-training iterations, making the overall pipeline time-consuming.
5. Evaluation is primarily based on static benchmarks; assessment in real-world user scenarios is lacking.

## Related Work & Insights

- **Improvements over AgentQ**: AgentQ also employs MCTS but is limited to web HTML environments with low efficiency under vanilla MCTS. M2-Miner extends the approach to mobile visual scenarios and substantially improves efficiency via multi-agent collaboration.
- **Distinction from OS-Genesis**: OS-Genesis relies on unstructured rule-based exploration with inverse task synthesis, whereas M2-Miner uses MCTS tree structure with forward intent-driven exploration, yielding higher data quality.
- **Implications for data flywheels**: The progressive training strategy demonstrates a positive feedback loop of "data → model → better data → better model," serving as a general paradigm for automated data generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First MCTS + multi-agent framework for GUI data mining; the intent recycling strategy is highly original.
- **Technical Depth**: ⭐⭐⭐⭐ — The three-agent design is well-motivated; replacing rollouts with process rewards is theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five benchmarks, cost analysis, multi-dimensional ablations, and quality evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear figures.
- **Value**: ⭐⭐⭐⭐⭐ — Achieves SOTA at minimal cost; directly advances the GUI agent community.
- **Overall Rating**: ⭐⭐⭐⭐⭐ (9/10)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](efficient_agent_training_for_computer_use.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)

<!-- RELATED:END -->
