---
title: >-
  [Paper Note] Efficient Agent Training for Computer Use
description: >-
  [ICLR 2026][LLM Agent][computer use agent] PC Agent-E uses only 312 human-annotated Windows operation trajectories. Via the proposed Trajectory Boost method, Claude 3.7 Sonnet synthesizes diverse alternative action decisions at each timestep. The resulting fine-tuned Qwen2.5-VL-72B achieves a 141% relative improvement on WindowsAgentArena-V2, even surpassing the teacher model Claude 3.7 Sonnet by 10%.
tags:
  - ICLR 2026
  - LLM Agent
  - computer use agent
  - trajectory augmentation
  - data efficiency
  - GUI agent
  - SFT
date: 2026-05-08
content_hash: 6dbe960f1f94afe9
---

# Efficient Agent Training for Computer Use

**Conference**: ICLR 2026
**arXiv**: [2505.13909](https://arxiv.org/abs/2505.13909)
**Code**: [https://github.com/GAIR-NLP/PC-Agent-E](https://github.com/GAIR-NLP/PC-Agent-E)
**Area**: Agent
**Keywords**: computer use agent, trajectory augmentation, data efficiency, GUI agent, SFT

## TL;DR
PC Agent-E uses only 312 human-annotated Windows operation trajectories. Via the proposed Trajectory Boost method, Claude 3.7 Sonnet synthesizes diverse alternative action decisions at each timestep. The resulting fine-tuned Qwen2.5-VL-72B achieves a 141% relative improvement on WindowsAgentArena-V2, even surpassing the teacher model Claude 3.7 Sonnet by 10%.

## Background & Motivation

**Background**: Computer use agents represent a critical direction in current AI research, with the goal of enabling models to operate computers through GUIs (clicking, typing, navigating) as humans do. Mainstream approaches fall into two categories: modular multi-agent workflows and native agent models. The latter (e.g., Claude Computer Use, OpenAI Operator) has become the dominant paradigm due to its flexibility and optimizability.

**Limitations of Prior Work**: Open-source models lag far behind closed-source systems (e.g., Claude 3.7 Sonnet) on computer use tasks, with the core bottleneck being the **extreme scarcity of high-quality trajectory data**. Existing data synthesis methods either rely on large-scale human annotation or sample complete trajectories end-to-end from stronger models. The latter suffers from error accumulation and is slow due to the need for online interaction with VM environments.

**Key Challenge**: How can maximal computer use capability be obtained from minimal human annotation? Training directly on human trajectories yields limited results (single-path supervision), while end-to-end distillation is inefficient and produces unstable quality (900 hours vs. 3 hours).

**Goal**: (a) How can a small amount of human-annotated data be utilized efficiently? (b) How can error accumulation in end-to-end distillation be avoided? (c) How can open-source models reach closed-source performance levels?

**Key Insight**: Inspired by data synthesis techniques from reasoning models such as DeepSeek-R1, the authors observe that computer use tasks naturally admit multiple valid execution paths — at any given timestep, several reasonable action choices may exist. Human trajectories can therefore serve as environment snapshots, allowing a strong model to synthesize alternative actions at each step without requiring online environment interaction.

**Core Idea**: Use environment snapshots from human trajectories as anchors, and have frontier models synthesize diverse action decisions offline at each step to augment trajectory data, enabling data-efficient training.

## Method

### Overall Architecture
PC Agent-E is a four-stage efficient training framework: (1) **Trajectory Collection** — 312 Windows operation trajectories are annotated by humans; (2) **Thought Completion** — implicit reasoning processes are reconstructed for human actions; (3) **Trajectory Boost** — 9 alternative action decisions are synthesized offline at each timestep; (4) **Agent Training** — Qwen2.5-VL-72B is trained on the augmented dataset of 27K samples. The input consists of screenshots, task descriptions, and action history; the output follows the ReAct paradigm as thought–action pairs.

### Key Designs

1. **Trajectory Collection**:

    - **Function**: Uses the PC Tracker tool to record 312 task completion trajectories from two annotators operating Windows.
    - **Mechanism**: Recordings include task descriptions, screenshot sequences, and human keyboard/mouse actions. Annotators may discard unsatisfactory trajectories or revise task descriptions. The action space $\mathcal{A}$ comprises 11 operation types: click, right click, double click, drag, scroll, press key, hotkey, type text, wait, finish, and fail.
    - **Design Motivation**: Human task completion is naturally correct, eliminating the need for additional verification. Two annotators can complete the labeling in a single day at minimal cost. Data decontamination is performed using 13-gram overlap $< 0$ and semantic similarity $< 0.85$.

2. **Thought Completion**:

    - **Function**: Reconstructs the implicit reasoning process for each step of the raw human trajectories.
    - **Mechanism**: For each action step, Claude 3.7 Sonnet is provided with the task description, the history of previous actions along with their reconstructed thoughts, the current action, and the corresponding screenshot, and is prompted to generate the underlying reasoning. This process is iterative — the context for later steps includes reconstructed thoughts from all preceding steps.
    - **Design Motivation**: Raw human trajectories contain only actions without accompanying reasoning. Training a ReAct-style agent requires thought–action pairs. Completing the thought process yields more complete trajectories and provides the necessary historical context for the subsequent Trajectory Boost stage.

3. **Trajectory Boost**:

    - **Function**: Synthesizes 9 alternative action decisions at each timestep of the human trajectories.
    - **Mechanism**: Each timestep constitutes an environment snapshot $\langle T, o_k, h_k \rangle$ (task description, observation screenshot, historical context), which is fed into Claude 3.7 Sonnet to sample 9 single-step action decisions $(t'_k, a'_k)$ in parallel. The result is a **Traj Tree**: the human trajectory forms the trunk, and the synthesized actions form the leaf nodes.
    - **Design Motivation**: This is the central contribution of the paper. Compared to end-to-end distillation, Trajectory Boost offers three key advantages: (a) it avoids error accumulation, since the environment state at each step is anchored by the human trajectory and cannot drift; (b) it is **offline**, requiring no interaction with a real environment and naturally parallelizable, achieving a 300× speedup (3 hours vs. 900 hours); (c) it leverages both the **authenticity** of human trajectories and the **diversity** of frontier models.

### Loss & Training
- Training is based on Qwen2.5-VL-72B with a standard SFT loss.
- Each action node on the Traj Tree (both human and synthesized) is converted into an independent training sample.
- The training sample format directly corresponds to the inference-time scaffold: input is screenshot + task description + history; output is thought + action.
- Historical context for all synthesized nodes consists only of preceding steps from the trunk (human trajectory), ensuring consistency.
- The 312 trajectories yield 27K training samples in total, with image resolution 1280×720 and context length 8192.

## Key Experimental Results

### Main Results

| Model | LibreOffice | Chrome | Edge | System | VS Code | VLC | Utils | Total |
|-------|-------------|--------|------|--------|---------|-----|-------|-------|
| GPT-4o | 0.0 | 5.9 | 0.0 | 8.3 | 0.0 | 0.0 | 0.0 | 2.1 |
| Qwen2.5-VL-72B | 0.0 | 34.7 | 15.4 | 20.8 | 26.3 | 7.6 | 16.7 | 14.9 |
| UI-TARS-72B-DPO | 0.0 | 40.6 | 38.5 | 58.3 | 36.8 | 7.6 | 25.0 | 26.2 |
| Claude 3.7 Sonnet | 2.4 | 46.5 | 61.5 | 54.2 | 52.6 | 29.0 | 16.7 | 32.6 |
| Claude 3.7 (thinking) | 2.4 | 64.1 | 46.2 | 66.7 | 52.6 | 21.9 | 25.0 | 35.4 |
| **PC Agent-E** | **4.8** | **64.1** | 46.2 | 50.0 | **57.9** | **35.7** | **33.3** | **36.0** |

PC Agent-E achieves a 141% relative improvement over Qwen2.5-VL-72B and surpasses Claude 3.7 Sonnet by 10%.

### Ablation Study

| Method | Data Size | WindowsAgentArena-V2 (%) | Notes |
|--------|-----------|--------------------------|-------|
| Base (Qwen2.5-VL-72B) | 0 | 14.9 | Baseline |
| Human only (s=1) | 2.7K | 17.2 | Human trajectories only |
| Direct Distillation (s=10) | 3120 traj | ~28 | End-to-end distillation |
| Trajectory Boost (s=10) | 27K | **36.0** | Ours |

### Key Findings
- **Trajectory Boost substantially outperforms human data alone**: Increasing the scaling factor from 1 to 10 raises performance from 17.2 to 36.0, whereas training on human trajectories alone reaches only 17.2.
- **Substantial advantage over direct distillation**: At comparable data scales, Trajectory Boost outperforms Direct Distillation by approximately 8 percentage points while being 300× more time-efficient (3h vs. 900h).
- **Cross-platform generalization**: On OSWorld (Linux), PC Agent-E achieves a 34% relative improvement (4.4→10.9%) despite being trained exclusively on Windows data.
- **Gains are primarily in planning capability**: Qualitative analysis shows that the trained model produces longer reasoning chains with markedly improved self-correction and verification behavior, while knowledge and grounding capabilities show limited improvement.
- **Infeasible Hacking phenomenon**: Weaker models score higher on infeasible tasks (Qwen 86.7% vs. PC Agent-E 63.3%), revealing a vulnerability in current evaluation protocols.

## Highlights & Insights
- **Single-step offline synthesis vs. end-to-end online distillation**: This is a particularly elegant insight — computer use tasks naturally admit multiple valid paths at each step. Anchoring environment states with human trajectories and synthesizing alternative actions step-by-step avoids the error accumulation inherent in multi-step distillation while achieving a 300× speedup.
- **Extreme data efficiency**: 312 trajectories → 27K samples → surpassing the teacher model. This demonstrates that high-quality, diverse supervision is more valuable than large-scale low-quality data.
- **WindowsAgentArena-V2 evaluation improvements**: The paper addresses issues including evaluation dependencies, infeasible hacking, and VM state instability, constituting an independent contribution to the community.
- **Transferability of the Traj Tree structure**: The proposed paradigm is applicable to any sequential decision-making task grounded in environment snapshots (e.g., web navigation, mobile GUI, robotics), as long as multiple valid paths exist at each step.

## Limitations & Future Work
- **Limited coverage due to only 312 training trajectories**: Data is concentrated in common applications such as Chrome and system settings; performance on complex scenarios such as LibreOffice remains weak (4.8%).
- **Image history not utilized**: Only the current screenshot is used during inference; past screenshots are excluded. The authors acknowledge that incorporating image history may be beneficial.
- **Knowledge and grounding bottlenecks remain unresolved**: Gains are primarily attributable to improved planning; tasks requiring domain-specific software knowledge (e.g., VLC features) or precise element localization show limited improvement.
- **Synthesized actions are not validated in real environments**: Actions generated by Trajectory Boost are plausible but unexecuted, and may include actions that cannot be successfully performed in practice.
- **Only SFT is applied; RL is not explored**: Incorporating reinforcement learning (e.g., GRPO with environment rewards) may yield further improvements.

## Related Work & Insights
- **vs. UI-TARS**: UI-TARS is trained on large-scale multi-step trajectory data; PC Agent-E demonstrates that intelligent augmentation of a very small dataset can surpass large-scale data approaches.
- **vs. Direct Distillation**: End-to-end distillation requires online interaction, suffers from error accumulation, and is 300× slower; Trajectory Boost is offline, parallelizable, and yields higher quality.
- **vs. Self-Play/Self-Improvement**: Self-improvement methods require the model itself to possess sufficiently strong capabilities. PC Agent-E cleverly uses human trajectories as a foundation, avoiding the cold-start problem.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Trajectory Boost idea is clean and elegant, though it is conceptually straightforward — leveraging human trajectories and a strong model for single-step synthesis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple baselines, complete ablations, cross-platform generalization, test-time scaling, and qualitative analysis are all included.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Structure is clear, figures are well-designed, and the motivation is developed fluently.
- **Value**: ⭐⭐⭐⭐ The work offers important reference value for data-efficient training of GUI agents; the 300× speedup and teacher-surpassing results are impressive.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[ICLR 2026\] M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent.md)

<!-- RELATED:END -->
