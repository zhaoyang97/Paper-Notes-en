---
title: >-
  [Paper Note] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution
description: >-
  [ACL 2026][LLM Agent][agent training] CoEvolve proposes an **agent-data co-evolution framework** that extracts three types of weakness signals—forgetting, boundary, and rare—from training trajectories, guiding LLMs to perform targeted environment re-exploration and task synthesis. This allows the training data distribution to dynamically adapt to the agent's evolving capabilities, yielding absolute improvements of 19–23% on AppWorld and BFCL.
tags:
  - ACL 2026
  - LLM Agent
  - agent training
  - data synthesis
  - co-evolution
  - forgetting signals
  - reinforcement learning
date: 2026-05-08
content_hash: 321a1d5ebed635db
---

# CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution

**Conference**: ACL 2026
**arXiv**: [2604.15840](https://arxiv.org/abs/2604.15840)
**Code**: [https://github.com/AMAP-ML/CoEvolve](https://github.com/AMAP-ML/CoEvolve)
**Area**: LLM Agent
**Keywords**: agent training, data synthesis, co-evolution, forgetting signals, reinforcement learning

## TL;DR
CoEvolve proposes an **agent-data co-evolution framework** that extracts three types of weakness signals—forgetting, boundary, and rare—from training trajectories, guiding LLMs to perform targeted environment re-exploration and task synthesis. This allows the training data distribution to dynamically adapt to the agent's evolving capabilities, yielding absolute improvements of 19–23% on AppWorld and BFCL.

## Background & Motivation

**Background**: LLM agents are typically trained via RL in interactive environments, but the source of training data remains a core bottleneck—either relying on costly human expert trajectories with limited coverage, or using LLMs to synthesize static data that lacks feedback and cannot adapt to agent evolution.

**Limitations of Prior Work**: (1) Human expert trajectories are "static snapshots" that fail to cover real-world long-tail variants (e.g., a button label change from "Book Now" to "Reserve Now" causes failure); (2) LLM-synthesized data reduces human dependency but relies on random exploration, resulting in shallow and incomplete environment coverage; (3) More critically, synthesized data is static and cannot adjust as the agent's capabilities evolve—skills already mastered are over-trained while weaknesses are neglected.

**Key Challenge**: The agent's capabilities continuously change, yet the training data distribution remains fixed—the absence of closed-loop feedback renders training inefficient and prevents sustained improvement.

**Goal**: Design a framework requiring no human supervision that dynamically adjusts the training data distribution according to the agent's evolving weaknesses, realizing a closed loop of "agent improves → new weaknesses discovered → targeted data synthesized → agent improves further."

**Key Insight**: Leverage trajectory replay signals from the training process (forgetting, boundary, and rare patterns) to identify specific agent weaknesses, and use these signals to guide directed LLM environment exploration.

**Core Idea**: Extract weakness signals from RL training rollout trajectories, conditionally guide LLMs to re-explore the environment, synthesize new weakness-targeted tasks, update the training distribution, and form a closed agent-data co-evolution loop.

## Method

### Overall Architecture
A three-stage closed loop: (1) **Training + Signal Extraction**: Train the agent with GRPO and extract forgetting, boundary, and rare signals from rollout trajectories; (2) **Signal-Guided Re-Exploration**: Provide signal-annotated trajectories to an LLM for reflection, generate structured exploration contexts, and guide the LLM to discover new interaction patterns in the real environment; (3) **Task Synthesis and Verification**: Abstract newly discovered interactions into executable tasks, validate them in the environment, and add verified tasks to the training set to update the data distribution.

### Key Designs

1. **Three-Category Weakness Signal Extraction**:

    - Function: Systematically identify specific agent weaknesses from training trajectories.
    - Mechanism: (1) **Forgetting signals**: Sliding-window detection—if there exists a success within the most recent $W$ attempts but the current attempt fails ($\exists s_i \geq 0.5$ and $s_{\text{now}} < 0.5$), the agent has "forgotten" a previously acquired capability; (2) **Boundary signals**: Within a single training step, $K$ trajectories sampled for the same task contain both successes and failures, indicating the agent is near the decision boundary and exhibits unstable behavior; (3) **Rare signals**: An action pattern whose frequency falls below a threshold ($c_p/N < \theta/100$) but occurs more than zero times, indicating a systematically under-explored interaction pattern in the environment.
    - Design Motivation: The three signal types capture complementary weaknesses—forgetting = capability regression, boundary = instability, rare = insufficient exploration. Signal-driven data synthesis is more efficient than random generation.

2. **Signal-Guided Environment Re-Exploration**:

    - Function: Use weakness signals to guide LLMs toward targeted environment exploration.
    - Mechanism: Signal-annotated failure trajectories (including task descriptions, action sequences, and environment feedback) are provided to an LLM, which is prompted to reflect on failure causes and generate structured exploration contexts (describing where and how failures or instabilities occur). These contexts then condition LLM interaction with the real environment to discover new interaction patterns and task variants.
    - Design Motivation: Unlike random exploration, signal-conditioned exploration focuses on the agent's current weakness regions, substantially improving exploration efficiency.

3. **Task Synthesis and Environment Verification**:

    - Function: Convert exploration-discovered interactions into executable training tasks.
    - Mechanism: New interaction patterns discovered during LLM re-exploration are abstracted into task descriptions, executed in the environment for validation (ensuring executability), and upon passing verification, added to the training set $\mathcal{D}_{t+1}$. The entire process requires no human supervision—exploration, synthesis, and verification are fully automated.
    - Design Motivation: Environment verification ensures the executability of synthesized tasks (avoiding hallucinated tasks), while task abstraction ensures reusability.

### Loss & Training
GRPO is used to train the agent. For each task, $K$ trajectories are sampled; policy gradients are computed based on within-group relative advantages, with KL regularization to prevent excessive deviation from the reference policy. Signal extraction, re-exploration, and task synthesis are executed after each training iteration.

## Key Experimental Results

### Main Results

| Model | AppWorld-TestN TGC | AppWorld-TestC TGC | BFCL Multi-turn | Avg. Gain |
|--------|------|------|------|------|
| Qwen2.5-7B + CoEvolve | 27.98 (+26.79) | 8.39 (+7.67) | 61.50 (+48.00) | **+19.43%** |
| Qwen3-4B + CoEvolve | 35.71 (+19.04) | 17.03 (+9.12) | 63.00 (+36.50) | **+15.58%** |
| Qwen3-30B-A3B + CoEvolve | 54.76 (+23.21) | 31.65 (+11.75) | 63.00 (+19.50) | **+18.14%** |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Forgetting signal only | Effective but incomplete | Captures only capability regression |
| Boundary signal only | Effective but incomplete | Captures only unstable behavior |
| Rare signal only | Effective but incomplete | Captures only insufficient exploration |
| All three signals combined | **Best** | Complementary weaknesses fully covered |
| Without environment verification | Significant drop | Hallucinated tasks introduce noise |

### Key Findings
- CoEvolve elevates Qwen2.5-7B from near-unusable (1.19%) to a moderate level (27.98%), a remarkable margin of improvement.
- On BFCL, Qwen2.5-7B + CoEvolve achieves 61.50%, surpassing GPT-4 (54.00%), demonstrating that data quality can compensate for model scale differences.
- Qwen3-30B-A3B + CoEvolve reaches 54.76% on AppWorld-TestN, approaching Claude-Sonnet-4.5 (73.81%).
- The three signal types are complementary—using any single type alone is inferior to using all three jointly.

## Highlights & Insights
- **Forgetting signals as a data synthesis criterion** is the most elegant design in this work: borrowing the concept of forgetting events from curriculum learning and repurposing it to guide data synthesis rather than data selection. This idea is transferable to any training scenario requiring dynamic data distribution adjustment.
- The **closed-loop design** (train → identify weaknesses → synthesize data → retrain) is more fundamental than plain data augmentation—it allows the training distribution and model capabilities to co-evolve, constituting a form of adaptive curriculum learning.
- The result of a 7B model surpassing GPT-4 on BFCL is striking and strongly demonstrates that "targeted data" is more valuable than "large volumes of random data."

## Limitations & Future Work
- Real environment interaction is required for verification, limiting applicability to scenarios with executable environments (e.g., API calls, web navigation); generalization to open-domain tasks is difficult.
- Hyperparameters for signal extraction (sliding window size $W$, rarity threshold $\theta$) may require environment-specific tuning.
- The re-exploration stage depends on a capable LLM (for reflection and exploration), which itself introduces additional computational cost.
- No direct comparison is made with other adaptive curriculum learning methods.

## Related Work & Insights
- **vs. Static synthetic data (Ye et al., 2024; Ding et al., 2024)**: The latter generates data offline in a one-shot manner; CoEvolve continuously evolves the data distribution through closed-loop feedback.
- **vs. Self-Play/Self-Improve**: The latter typically performs trajectory optimization over a fixed query set; CoEvolve discovers entirely new tasks and environment states, not limited to rewriting existing data.

## Rating
- Novelty: ⭐⭐⭐⭐ The closed-loop agent-data co-evolution framework represents a novel paradigm; using forgetting signals to guide data synthesis is a clever idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple model scales (7B/4B/30B), multiple benchmarks (AppWorld/BFCL), detailed ablations, and comparison with closed-source models.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method pipeline diagram is intuitive, though the signal extraction formulas could be streamlined.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](../../NeurIPS2025/llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](../../AAAI2026/llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)

<!-- RELATED:END -->
