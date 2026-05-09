---
title: >-
  [Paper Note] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents
description: >-
  [ACL 2026][LLM Agent][Hierarchical Reinforcement Learning] This paper proposes STEP-HRL, which introduces a local progress module to iteratively compress interaction history within each subtask into compact textual summaries, enabling both high-level and low-level policies to make decisions based solely on single-step transitions rather than full histories. The approach achieves significant performance and generalization gains on ScienceWorld and ALFWorld while reducing token usage.
tags:
  - ACL 2026
  - LLM Agent
  - Hierarchical Reinforcement Learning
  - Step-Level Transitions
  - Local Progress
  - Token Efficiency
  - Offline RL
date: 2026-05-08
content_hash: c5bd44624b4a9c1b
---

# Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents

**Conference**: ACL 2026
**arXiv**: [2604.05808](https://arxiv.org/abs/2604.05808)
**Code**: [GitHub](https://github.com/TonyStark042/STEP-HRL)
**Area**: LLM Agent / Hierarchical Reinforcement Learning
**Keywords**: Hierarchical Reinforcement Learning, Step-Level Transitions, Local Progress, Token Efficiency, Offline RL

## TL;DR

This paper proposes STEP-HRL, which introduces a local progress module to iteratively compress interaction history within each subtask into compact textual summaries, enabling both high-level and low-level policies to make decisions based solely on single-step transitions rather than full histories. The approach achieves significant performance and generalization gains on ScienceWorld and ALFWorld while reducing token usage.

## Background & Motivation

**Background**: LLM agents have demonstrated strong capabilities in interactive decision-making tasks. Reinforcement learning provides a principled mechanism for improving agents through environment interaction and reward feedback. Existing LLM agents predominantly adopt a "history-conditioned" paradigm, where policies condition on ever-growing historical sequences.

**Limitations of Prior Work**: (1) The quadratic complexity of attention mechanisms makes reasoning over long histories computationally expensive; (2) unfiltered history accumulates redundant or irrelevant information that may obscure decision-critical signals; (3) existing HRL methods introduce temporal abstraction but still condition both high- and low-level policies on cumulative histories, inheriting the long-context dependency problem.

**Key Challenge**: History conditioning is a modeling choice rather than a necessity of RL — conflating long-horizon decision-making with long-context reasoning introduces unnecessary computational overhead and reasoning noise.

**Goal**: Design a progress-based rather than history-based HRL framework in which policies rely solely on single-step transitions for decision-making.

**Key Insight**: The sequence of completed subtasks naturally constitutes a compact summary of global progress; the remaining challenge is how to compactly represent the local interaction history within each subtask.

**Core Idea**: Introduce a local progress policy $\pi_\theta^p$ that iteratively compresses the intra-subtask interaction history into a compact textual representation at each step. The low-level policy conditions only on the current subtask, local progress, and current observation, eliminating dependence on the full history.

## Method

### Overall Architecture

STEP-HRL comprises three policies sharing parameters: (1) the **high-level policy** $\pi_\theta^h$ generates the next subtask conditioned on the task instruction, completed subtasks, the final progress of the previous subtask, and the current observation; (2) the **low-level policy** $\pi_\theta^l$ generates primitive actions conditioned on the current subtask, local progress, and current observation; (3) the **local progress policy** $\pi_\theta^p$ updates local progress conditioned on the current subtask, previous action, current observation, and prior-step progress. Training proceeds in two stages: behavioral cloning initialization → step-level offline RL optimization.

### Key Designs

1. **Local Progress Policy**:

    - **Function**: Iteratively compresses the growing intra-subtask interaction history into a fixed-size textual summary.
    - **Mechanism**: $p_t^k \sim \pi_\theta^p(\cdot | g_k, a_{t-1}^k, o_t^k, p_{t-1}^k)$. At each step, the policy receives the previous progress, the last action, and the current observation, selectively extracts subtask-relevant information, and outputs an updated compact progress summary. Initialized as empty: $p_0^k = \varnothing$.
    - **Design Motivation**: Unlike simple history truncation, local progress is selective — retaining only subtask-relevant information and discarding redundancy.

2. **Step-Level Transition Construction**:

    - **Function**: Enables both low-level and high-level policies to make decisions based on constant-size inputs.
    - **Mechanism**: The low-level step transition is $(o_t^k, p_t^k, a_t^k, \hat{r}_t^k, o_{t+1}^k, p_{t+1}^k)$; the high-level step transition is $(\hat{p}_{k-1}, o_0^k, g_k, R_k, \hat{p}_k, o_0^{k+1})$, where $\hat{p}_k$ denotes the final local progress upon completion of subtask $g_k$.
    - **Design Motivation**: Step-level transitions are Markovian — decisions can be made without backtracking through the full history.

3. **Step-Level Offline RL (IQL-based)**:

    - **Function**: Further optimizes policies after behavioral cloning initialization.
    - **Mechanism**: Built on the Implicit Q-Learning framework, the three policies share parameters but are each equipped with independent critic networks (utterance-level $V$ and $Q$). Value functions are learned via expectile regression, and policies are optimized via advantage-weighted regression. The low-level policy uses intrinsic rewards (subtask completion = 1); the high-level policy uses extrinsic environment rewards.
    - **Design Motivation**: Behavioral cloning merely imitates experts, whereas offline RL can discover superior policies; step-level transitions yield more stable value estimates for RL.

### Loss & Training

The behavioral cloning stage employs autoregressive cross-entropy loss. The offline RL stage jointly optimizes: a Q-function TD regression loss, a value function expectile loss, and an advantage-weighted policy loss. All three policies share LLM parameters to facilitate cross-level knowledge transfer.

## Key Experimental Results

### Main Results

**ScienceWorld (30 science task families)**

| Method | Total Score | Token Usage | Generalization (Unseen Variants) |
|--------|-------------|-------------|----------------------------------|
| ReAct | 32.1 | High | Low |
| GLIDER (HRL) | 48.2 | High | Medium |
| STEP-HRL (BC only) | 52.7 | **Low** | Medium |
| **STEP-HRL (BC + RL)** | **57.3** | **Low** | **High** |

### Ablation Study

| Configuration | ScienceWorld | ALFWorld |
|---------------|-------------|---------|
| No local progress (full history) | 44.8 | 62.3 |
| Fixed-window truncation | 47.2 | 65.1 |
| **Local progress (STEP-HRL)** | **57.3** | **78.4** |

### Key Findings

- STEP-HRL with behavioral cloning alone already surpasses existing HRL baselines (52.7 vs. 48.2), validating the effectiveness of step-level transitions per se.
- Offline RL provides a further 4.6-point improvement, demonstrating that step-level transitions enable more efficient RL optimization.
- The local progress module outperforms fixed-window truncation by 10.1 points — selective information retention is substantially superior to naive truncation.
- Parameter sharing across the three policies reduces training and inference overhead while promoting cross-level knowledge transfer.

## Highlights & Insights

- The core insight that "long-horizon decision-making ≠ long-context reasoning" is profound — step-level transitions demonstrate that information compression can substitute for history accumulation.
- Local progress functions as an information bottleneck, naturally achieving attention focusing and noise filtering.
- The parameter-sharing design across three policies strikes a favorable balance between efficiency and performance.

## Limitations & Future Work

- The quality of local progress depends on the LLM's summarization capability — weaker LLMs may produce low-quality progress representations.
- Subtask decomposition and progress annotations in expert demonstrations are generated by DeepSeek, which may introduce its inherent biases.
- Evaluation is limited to text-based environments (ScienceWorld, ALFWorld); applicability to visual or multimodal environments remains unexplored.
- Offline RL is constrained by the quality and diversity of collected data.

## Related Work & Insights

- **vs. GLIDER**: GLIDER employs HRL but still conditions on full histories; STEP-HRL eliminates historical dependence through local progress.
- **vs. ReAct**: ReAct interleaves reasoning and acting but lacks hierarchical structure; STEP-HRL adds hierarchical abstraction and step-level optimization.
- **vs. Decision Transformer**: DT reformulates decision-making as sequence prediction and requires complete trajectories; STEP-HRL requires only single-step transitions.

## Rating

- Novelty: ⭐⭐⭐⭐ The HRL design combining step-level transitions and local progress is novel and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, detailed ablations, token analysis, and generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear and method derivation is complete.
- Value: ⭐⭐⭐⭐ Provides a more efficient framework for long-horizon decision-making in LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)
- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)

</div>

<!-- RELATED:END -->
