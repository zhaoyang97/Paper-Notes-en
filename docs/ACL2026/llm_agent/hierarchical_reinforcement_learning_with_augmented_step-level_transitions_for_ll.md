---
title: >-
  [Paper Note] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents
description: >-
  [ACL 2026][LLM Agent][Hierarchical Reinforcement Learning] This paper proposes STEP-HRL, which iteratively compresses interaction histories into compact text summaries through a local progress module. This allows high-le…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Hierarchical Reinforcement Learning"
  - "Step-level transitions"
  - "Local progress"
  - "Token efficiency"
  - "Offline RL"
date: 2026-05-08
content_hash: 0076f728e09cae66
---

# Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents

**Conference**: ACL 2026  
**arXiv**: [2604.05808](https://arxiv.org/abs/2604.05808)  
**Code**: [GitHub](https://github.com/TonyStark042/STEP-HRL)  
**Area**: LLM Agent / Hierarchical Reinforcement Learning  
**Keywords**: Hierarchical Reinforcement Learning, Step-level transitions, Local progress, Token efficiency, Offline RL

## TL;DR

This paper proposes STEP-HRL, which iteratively compresses interaction histories into compact text summaries through a local progress module. This allows high-level and low-level policies to make decisions based on step-level transitions rather than full histories, significantly improving performance and generalization on ScienceWorld and ALFWorld while reducing token usage.

## Background & Motivation

**Background**: LLM agents have shown powerful capabilities in interactive decision-making tasks. Reinforcement Learning (RL) provides a principled mechanism for improving agents by optimizing policies through environment interaction and reward feedback. Existing LLM agents commonly adopt a "history-conditioned" paradigm, where policies are conditioned on increasingly long historical sequences.

**Limitations of Prior Work**: (1) The quadratic complexity of the attention mechanism makes inference with long histories computationally expensive; (2) Unfiltered historical accumulation contains redundant or irrelevant information that can obscure critical decision signals; (3) Existing HRL methods, despite introducing temporal abstraction, still condition high-level and low-level policies on cumulative history, inheriting the long-context dependency issue.

**Key Challenge**: Long-history conditioning is a modeling choice rather than a necessity for RL—conflating long-term decision-making with long context introduces unnecessary computational burden and reasoning noise.

**Goal**: To design a progress-based rather than history-based HRL framework that enables policies to make decisions based only on step-level transitions.

**Key Insight**: The sequence of completed subtasks naturally forms a compact summary of global progress; the remaining challenge is how to compactly represent the local interaction history within each subtask.

**Core Idea**: Introduce a local progress policy $\pi_\theta^p$ that iteratively compresses the interaction history within a subtask into a compact text representation. The low-level policy conditions only on the current subtask, local progress, and current observation, eliminating reliance on the full history.

## Method

### Overall Architecture

STEP-HRL consists of three policies with shared parameters: (1) **High-level policy** $\pi_\theta^h$ generates the next subtask based on task instructions, completed subtasks, the final progress of the previous subtask, and the current observation; (2) **Low-level policy** $\pi_\theta^l$ generates primitive actions based on the current subtask, local progress, and current observation; (3) **Local progress policy** $\pi_\theta^p$ updates the local progress based on the current subtask, the previous action, the current observation, and the previous step's progress. Training consists of two stages: Behavior Cloning initialization followed by step-level offline RL optimization.

### Key Designs

1.  **Local Progress Module**:
    - **Function**: Iteratively compresses the growing interaction history within a subtask into a fixed-size text summary.
    - **Mechanism**: $p_t^k \sim \pi_\theta^p(\cdot | g_k, a_{t-1}^k, o_t^k, p_{t-1}^k)$. Each step receives the previous progress, the last action, and the current observation to selectively extract subtask-relevant information and output an updated compact progress summary. Initialized as $p_0^k = \varnothing$.
    - **Design Motivation**: Unlike simple history truncation, local progress is selective—it retains only subtask-relevant information and discards redundancy.

2.  **Step-Level Transition Construction**:
    - **Function**: Enables both low-level and high-level policies to make decisions based on constant-sized inputs.
    - **Mechanism**: Low-level step transitions are $(o_t^k, p_t^k, a_t^k, \hat{r}_t^k, o_{t+1}^k, p_{t+1}^k)$; high-level step transitions are $(\hat{p}_{k-1}, o_0^k, g_k, R_k, \hat{p}_k, o_0^{k+1})$, where $\hat{p}_k$ is the final local progress at the end of subtask $g_k$.
    - **Design Motivation**: Step-level transitions are Markovian—decisions can be made without backtracking through the full history.

3.  **Step-Level Offline RL (IQL-based)**:
    - **Function**: Further optimizes policies after Behavior Cloning initialization.
    - **Mechanism**: Based on the Implicit Q-Learning framework, the three policies share parameters but are equipped with independent critic networks (utterance-level V and Q). Expectile regression is used to learn value functions, and advantage-weighted regression optimizes the policy. Low-level uses intrinsic rewards (subtask completion = 1), and high-level uses extrinsic environment rewards.
    - **Design Motivation**: Behavior Cloning only mimics experts, while offline RL can discover superior policies; step-level transitions make value estimation in RL more stable.

### Loss & Training

The Behavior Cloning stage utilizes an autoregressive cross-entropy loss. The offline RL stage involves joint optimization: Q-function TD regression loss + value function expectile loss + policy advantage-weighted loss. All three policies share LLM parameters to promote cross-level knowledge transfer.

## Key Experimental Results

### Main Results

**ScienceWorld (30 scientific task families)**

| Method | Total Score | Token Usage | Generalization (Unseen Variants) |
| :--- | :--- | :--- | :--- |
| ReAct | 32.1 | High | Low |
| GLIDER (HRL) | 48.2 | High | Medium |
| STEP-HRL (BC only) | 52.7 | **Low** | Medium |
| **STEP-HRL (BC + RL)** | **57.3** | **Low** | **High** |

### Ablation Study

| Configuration | ScienceWorld | ALFWorld |
| :--- | :--- | :--- |
| W/O Local Progress (Full History) | 44.8 | 62.3 |
| Fixed Window Truncation | 47.2 | 65.1 |
| **Local Progress (STEP-HRL)** | **57.3** | **78.4** |

### Key Findings

- STEP-HRL in the Behavior Cloning stage alone surpasses existing HRL baselines (52.7 vs 48.2), validating the effectiveness of step-level transitions.
- Offline RL provides an additional 4.6 percentage point improvement, proving that step-level transitions make RL optimization more efficient.
- The local progress module improves performance by 10.1 percentage points compared to fixed-window truncation—selective information retention is far superior to simple truncation.
- Shared parameters among the three policies reduce training and inference overhead while promoting cross-level knowledge transfer.

## Highlights & Insights

- The core insight that "long-term decision-making $\neq$ long context" is profound—step-level transitions demonstrate that information compression can replace history accumulation.
- Local progress acts as an information bottleneck, naturally achieving attention focus and noise filtering.
- The design of shared parameters across the three policies strikes a good balance between efficiency and performance.

## Limitations & Future Work

- The quality of local progress depends on the LLM's summarization ability—weaker LLMs might generate low-quality progress.
- Subtask decomposition and progress labeling for expert demonstrations were generated by DeepSeek, which may inherit its biases.
- Evaluation was limited to text-based environments (ScienceWorld, ALFWorld); applicability to visual or multimodal environments remains unknown.
- Offline RL is limited by the quality and diversity of the collected data.

## Related Work & Insights

- **vs GLIDER**: GLIDER uses HRL but still conditions on full history; STEP-HRL eliminates history dependency through local progress.
- **vs ReAct**: ReAct interleaves reasoning and acting but lacks a hierarchical structure; STEP-HRL adds hierarchical abstraction and step-level optimization.
- **vs Decision Transformer**: DT treats decision-making as sequence prediction requiring full trajectories; STEP-HRL requires only step-level transitions.

## Rating

- Novelty: ⭐⭐⭐⭐ The HRL design combining step-level transitions and local progress is novel and sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, detailed ablations, token analysis, and generalization assessment.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and complete methodological derivation.
- Value: ⭐⭐⭐⭐ Provides a more efficient framework for long-term decision-making in LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Verified Critical Step Optimization for LLM Agents](verified_critical_step_optimization_for_llm_agents.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)
- [\[ACL 2026\] Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning](temp-r1_a_unified_autonomous_agent_for_complex_temporal_kgqa_via_reverse_curricu.md)
- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)

</div>

<!-- RELATED:END -->
