---
title: >-
  [Paper Note] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution
description: >-
  [ACL 2026][LLM Agent][Agent Training] CoEvolve proposes an **agent-data mutual evolution framework** that extracts three types of weakness signals (forgetting, boundary…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agent Training"
  - "Data Synthesis"
  - "Co-evolution"
  - "Forgetting Signals"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 5a5c186bde6c5df6
---

# CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution

**Conference**: ACL 2026  
**arXiv**: [2604.15840](https://arxiv.org/abs/2604.15840)  
**Code**: [https://github.com/AMAP-ML/CoEvolve](https://github.com/AMAP-ML/CoEvolve)  
**Area**: LLM Agent  
**Keywords**: Agent Training, Data Synthesis, Co-evolution, Forgetting Signals, Reinforcement Learning

## TL;DR
CoEvolve proposes an **agent-data mutual evolution framework** that extracts three types of weakness signals (forgetting, boundary, and rare) from training trajectories. These signals guide the LLM to perform targeted environment re-exploration and task synthesis, allowing the training data distribution to adapt dynamically to the agent's capabilities, resulting in absolute improvements of 19-23% on AppWorld and BFCL.

## Background & Motivation

**Background**: LLM Agents are typically trained in interactive environments via RL, but the source of training data is a core bottleneck—relying either on human expert trajectories (expensive, limited coverage) or static data synthesized by LLMs (lacks feedback, unable to adapt to agent evolution).

**Limitations of Prior Work**: (1) Human expert trajectories are "static snapshots" that fail to cover real-world long-tail variants (e.g., failure when a button label changes from "Book Now" to "Reserve Now"); (2) Although LLM-synthesized data reduces human dependence, it is based on random exploration with shallow and incomplete environment coverage; (3) Crucially, synthesized data is static and cannot adjust as agent capabilities evolve—skills already mastered by the agent are over-trained while weaknesses are ignored.

**Key Challenge**: Agent capabilities change continuously, but the training data distribution remains fixed—the lack of closed-loop feedback leads to low training efficiency and an inability to achieve continuous improvement.

**Goal**: Design a framework without human supervision where the training data distribution dynamically adjusts according to the agent's evolving weaknesses, achieving a closed loop of "agent improvement → discovery of new weaknesses → targeted data synthesis → further agent improvement."

**Key Insight**: Utilize trajectory replay signals during training (forgetting, boundary, and rare patterns) to identify specific agent weaknesses, using these as conditions to guide the LLM in directional environment exploration.

**Core Idea**: Extract weakness signals from RL training rollout trajectories, conditionally guide the LLM to re-explore the environment, synthesize new tasks targeting these weaknesses, and update the training distribution to form an agent-data mutual evolution loop.

## Method

### Overall Architecture
A three-stage closed loop: (1) **Training + Signal Extraction**: Use GRPO to train the agent and extract three types of signals (forgetting, boundary, and rare) from rollout trajectories; (2) **Signal-Guided Re-exploration**: Provide signal trajectories to the LLM for reflection, generating structured exploration contexts to guide the LLM in discovering new interaction patterns in the environment; (3) **Task Synthesis & Verification**: Abstract discovered interactions into executable tasks, which are added to the training set after environmental verification to update the data distribution.

### Key Designs

1. **Extraction of Three Weakness Signals**:
    - **Function**: Systematically identify specific agent weaknesses from training trajectories.
    - **Mechanism**: (1) **Forgetting Signals**: Detected via a sliding window—if a task succeeded in the last $W$ iterations but currently fails ($\exists s_i \geq 0.5$ and $s_{\text{now}} < 0.5$), it indicates the agent has "forgotten" a previously learned capability; (2) **Boundary Signals**: A single training iteration for the same task contains both successful and failed trajectories among $K$ samples, indicating the agent is at a decision boundary with unstable behavior; (3) **Rare Signals**: Action patterns with a frequency below a threshold ($c_p/N < \theta/100$) but occurring more than 0 times, indicating systematically under-explored interaction patterns in the environment.
    - **Design Motivation**: The three signals capture complementary weaknesses: forgetting = capability regression, boundary = instability, rare = insufficient exploration. Signal-driven data synthesis is more efficient than random generation.

2. **Signal-Guided Environment Re-exploration**:
    - **Function**: Use weakness signals to guide the LLM toward targeted environment exploration.
    - **Mechanism**: Provide failed trajectories annotated with signals (including task descriptions, action sequences, and environment feedback) to the LLM, requiring it to reflect on the failure causes and generate structured exploration contexts (describing where and how it failed/was unstable). This context then conditions the LLM to interact with the real environment to discover new interaction patterns and task variants.
    - **Design Motivation**: Unlike random exploration, signal-conditioned exploration focuses on the agent's current weakness regions, significantly improving exploration efficiency.

3. **Task Synthesis and Environmental Verification**:
    - **Function**: Transform interactions discovered during exploration into executable training tasks.
    - **Mechanism**: Abstract new interaction patterns found during re-exploration into task descriptions, perform execution verification in the environment (to ensure executability), and add verified tasks to the training set $\mathcal{D}_{t+1}$. The entire process is automated without human supervision.
    - **Design Motivation**: Environmental verification ensures the executability of synthesized tasks (avoiding hallucinated tasks), while task abstraction ensures reusability.

### Loss & Training
Train the agent using GRPO, sampling $K$ trajectories for each task and calculating policy gradients based on relative advantages within the group, with KL regularization to prevent deviation from the reference policy. Signal extraction, re-exploration, and task synthesis are performed after each training iteration.

## Key Experimental Results

### Main Results

| Model | AppWorld-TestN TGC | AppWorld-TestC TGC | BFCL Multi-turn | Average Gain |
|--------|------|------|------|------|
| Qwen2.5-7B + CoEvolve | 27.98 (+26.79) | 8.39 (+7.67) | 61.50 (+48.00) | **+19.43%** |
| Qwen3-4B + CoEvolve | 35.71 (+19.04) | 17.03 (+9.12) | 63.00 (+36.50) | **+15.58%** |
| Qwen3-30B-A3B + CoEvolve | 54.76 (+23.21) | 31.65 (+11.75) | 63.00 (+19.50) | **+18.14%** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Forgetting Signal Only | Effective but incomplete | Captures only capability regression |
| Boundary Signal Only | Effective but incomplete | Captures only unstable behavior |
| Rare Signal Only | Effective but incomplete | Captures only insufficient exploration |
| Combined Signals | **Optimal** | Comprehensive coverage of complementary weaknesses |
| No Env. Verification | Significant drop | Hallucinated tasks introduce noise |

### Key Findings
- CoEvolve transforms Qwen2.5-7B from almost unusable (1.19%) to a competitive level (27.98%), showing a massive improvement.
- On BFCL, Qwen2.5-7B+CoEvolve reaches 61.50%, even surpassing GPT-4 (54.00%), demonstrating that data quality can compensate for gaps in model scale.
- Qwen3-30B-A3B+CoEvolve reaches 54.76% on AppWorld-TestN, approaching Claude-Sonnet-4.5 (73.81%).
- The three types of signals are complementary—using any single type is less effective than using them in combination.

## Highlights & Insights
- **"Forgetting signals" as a data selection criterion** is the most clever design in this paper: borrowing the concept of forgetting events from curriculum learning and applying it to guide data synthesis rather than just data selection. This idea is transferable to any training scenario requiring dynamic data distribution adjustment.
- The **closed-loop design** (training → identifying weaknesses → synthesizing data → retraining) is more fundamental than simple data augmentation—it allows the training distribution and model capabilities to co-evolve, acting as a form of adaptive curriculum learning.
- The result where a 7B model surpasses GPT-4 on BFCL is very striking, strongly proving that "targeted data" is more valuable than "large amounts of random data."

## Limitations & Future Work
- It requires interaction with real environments for verification, limiting it to scenarios with executable environments (e.g., API calls, web navigation), making it difficult to generalize to open-domain tasks.
- Hyperparameters for signal extraction (sliding window size $W$, rare threshold $\theta$) may need adjustment for different environments.
- The re-exploration stage relies on a strong LLM (for reflection and exploration), which introduces additional computational costs.
- There is no direct comparison with other adaptive curriculum learning methods.

## Related Work & Insights
- **vs Static Synthetic Data (Ye et al., 2024; Ding et al., 2024)**: The latter generates data offline in a one-off manner, while CoEvolve continuously evolves the data distribution via closed-loop feedback.
- **vs Self-Play/Self-Improvement**: The latter usually performs trajectory optimization on a fixed query set, whereas CoEvolve discovers entirely new tasks and environment states beyond just rewriting existing data.

## Rating
- Novelty: ⭐⭐⭐⭐ The closed-loop framework for agent-data mutual evolution is a novel paradigm, and using forgetting signals for data synthesis is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes multiple models (7B/4B/30B), multiple benchmarks (AppWorld/BFCL), detailed ablations, and comparisons with closed-source models.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the methodology flowchart is intuitive, though signal extraction formulas could be further streamlined.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)
- [\[ACL 2026\] WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)
- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)

</div>

<!-- RELATED:END -->
