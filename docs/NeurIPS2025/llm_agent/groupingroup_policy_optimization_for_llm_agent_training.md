---
title: >-
  [Paper Note] Group-in-Group Policy Optimization for LLM Agent Training
description: >-
  [NeurIPS 2025][LLM Agent][GiGPO] GiGPO introduces step-level grouping nested within the episode-level grouping of GRPO by leveraging recurring environment states across trajectories as anchor states, enabling fine-grained credit assignment without additional rollouts or a critic model. It outperforms GRPO by >12% on ALFWorld and >9% on WebShop.
tags:
  - NeurIPS 2025
  - LLM Agent
  - GiGPO
  - credit assignment
  - anchor state grouping
  - multi-turn agent
  - GRPO
date: 2026-05-08
content_hash: 1b56078abc4c4fac
---

# Group-in-Group Policy Optimization for LLM Agent Training

**Conference**: NeurIPS 2025
**arXiv**: [2505.10978](https://arxiv.org/abs/2505.10978)
**Code**: [https://github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
**Area**: LLM Agent / Reinforcement Learning
**Keywords**: GiGPO, credit assignment, anchor state grouping, multi-turn agent, GRPO

## TL;DR
GiGPO introduces step-level grouping nested within the episode-level grouping of GRPO by leveraging recurring environment states across trajectories as anchor states, enabling fine-grained credit assignment without additional rollouts or a critic model. It outperforms GRPO by >12% on ALFWorld and >9% on WebShop.

## Background & Motivation

**Background**: Group-based RL methods (GRPO, RLOO) have achieved strong results on single-turn tasks such as mathematical reasoning and code generation. However, these methods treat entire trajectories as atomic units when computing advantages, making it impossible to distinguish the contribution of individual steps within a trajectory.

**Limitations of Prior Work**: LLM agent interactions span dozens of steps and tens of thousands of tokens (e.g., up to 50 steps and 20k+ tokens in ALFWorld), with rewards typically provided only at the end of an episode. GRPO assigns identical advantages to all tokens within an episode, treating good and bad steps equally. While PPO supports step-level advantage estimation, it requires an additional critic network with substantial memory overhead.

**Key Challenge**: The most straightforward approach to step-level credit assignment — rolling out multiple actions from each state to form a contrastive group — requires a large number of additional LLM forward passes and is computationally prohibitive.

**Goal**: To introduce fine-grained step-level credit assignment for multi-turn agent training while retaining the critic-free, memory-efficient, and stable convergence properties of group-based RL.

**Key Insight**: A key observation is that among $N$ trajectories sampled for the same task from the same initial state, many environment states recur naturally (e.g., revisiting the same webpage, returning to the same room). These repeated states can form step-level contrastive groups at no additional cost.

**Core Idea**: Recurring environment states across trajectories are used as anchor states. A hashmap retroactively constructs step-level groups, enabling a two-level "group-in-group" advantage estimation without any additional rollouts.

## Method

### Overall Architecture
GiGPO employs a two-level advantage estimation scheme: (1) an episode-level macro advantage $A_E$ — standard inter-trajectory comparison as in GRPO; and (2) a step-level micro advantage $A_S$ — inter-step comparison constructed via anchor state grouping. The final advantage is a weighted sum: $A = A_E + \omega \cdot A_S$.

### Key Designs

1. **Episode Relative Advantage $A_E$**

   - **Function**: Captures the overall quality of a trajectory.
   - **Mechanism**: Normalizes the total return $R(\tau_i) = \sum_t r_t^{(i)}$ across $N$ trajectories: $A_E(\tau_i) = \frac{R(\tau_i) - \text{mean}}{F_{\text{norm}}}$.
   - **Design Motivation**: Provides a stable global training signal that encourages the policy to develop coherent trajectory-level behavior.

2. **Anchor State Grouping**

   - **Function**: Constructs step-level contrastive groups at zero additional cost.
   - **Mechanism**: Identifies the set $\mathcal{U}$ of unique environment states appearing across all trajectories. For each state $\tilde{s}$, all (action, return) pairs departing from that state are collected to form a step-level group $G_S(\tilde{s})$. Implementation requires only lightweight hashmap key matching without triggering any additional LLM inference.
   - **Design Motivation**: Agents naturally revisit states during exploration (e.g., returning to the same webpage, room, or search result). These natural recurrences provide free data for step-level comparison.

3. **Step Relative Advantage $A_S$**

   - **Function**: Evaluates the relative quality of different actions taken from the same state.
   - **Mechanism**: Computes the discounted return $R_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}$ for each action within a step-level group, followed by within-group normalization. For example, in WebShop, among actions taken from the same search result page, clicking on the correct item and successfully completing the purchase receives the highest $A_S$.
   - **Design Motivation**: Using discounted returns rather than immediate rewards $r_t$ captures the long-term consequences of actions.

4. **Similarity-based Grouping (Extension)**

   - **Function**: Handles scenarios where exact state matching is infeasible (e.g., QA tasks where search results differ slightly), using longest common subsequence similarity >0.9 for approximate grouping.

### Loss & Training
- Standard PPO-clip objective with KL regularization; advantage replaced by $A = A_E + \omega \cdot A_S$.
- $\omega = 1$ (no tuning required), $\gamma = 0.95$, rollout group size $N = 8$.
- $F_{\text{norm}}$ is either std (standard GRPO) or 1 (unbiased RLOO estimate), depending on the task.
- Implemented on the veRL framework with step-wise multi-turn rollout to avoid context explosion.

## Key Experimental Results

### Main Results

| Method | ALFWorld (7B) | WebShop Score (7B) | WebShop Succ (7B) |
|---|---|---|---|
| GPT-4o | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 60.3 | 42.5 | 35.9 |
| ReAct (7B) | 31.2 | 46.2 | 19.5 |
| PPO (with critic, 7B) | 80.4 | 81.4 | 68.7 |
| GRPO (7B) | 77.6 | 79.3 | 66.1 |
| **GiGPO w/o std (7B)** | **90.2** | **86.2** | **75.2** |
| **GiGPO w/ std (7B)** | **90.8** | **84.4** | **72.8** |

GiGPO surpasses GRPO by 12.6% on ALFWorld and 9.1% on WebShop (success rate), while also outperforming PPO, which requires an additional critic.

### Ablation Study

| Configuration | ALFWorld (1.5B) | WebShop Succ (1.5B) |
|---|---|---|
| GiGPO full (w/o std) | 86.1 | 67.4 |
| w/o $A_S$ (episode-level only) | significant drop | significant drop |
| w/o $A_E$ (step-level only) | large drop | large drop |

Both levels of advantage are indispensable. Removing $A_E$ eliminates the global signal, leading to incoherent policies; removing $A_S$ causes the most severe degradation on complex tasks (Cool, Pick2, WebShop).

### Key Findings
- **High step-level group coverage**: During training, fewer than 35% of states appear only once; over 65% recur across trajectories, providing ample data for anchor state grouping.
- **Interpretable training dynamics**: The group size distribution is initially right-skewed (many repeated looping states) and converges toward 6–8 (≈ group size) as training progresses, indicating that the agent learns to avoid loops.
- **Negligible computational overhead**: Anchor state grouping (hashmap) takes only 0.01s/iter; step advantage computation takes 0.53s/iter — less than 0.002% of total training time.
- **Orthogonal and composable**: GiGPO combined with DAPO's dynamic sampling achieves 75.0% on WebShop, surpassing standalone DAPO (66.1%).
- **Effective for VLM agents**: Consistent improvements are also observed on visual tasks (Sokoban and EZPoints).

## Highlights & Insights
- **"Free lunch" design**: Repeated states across rollout trajectories already exist in the collected data. GiGPO simply recovers these free signals via a hashmap, requiring no additional inference. This idea generalizes to any sequential decision-making problem where state revisitation occurs.
- **Elegant balance of hierarchical advantages**: $A_E$ provides direction (how good is this trajectory overall?), while $A_S$ provides granularity (how good is this specific action?). Both are necessary. This "global + local" credit assignment paradigm proves more effective than purely trajectory-level or purely step-level approaches.

## Limitations & Future Work
- Relies on exact or high-similarity state matching, which may be difficult in highly stochastic or continuous state spaces where anchor states are scarce.
- In the extreme case of no state repetition, GiGPO degrades gracefully to standard GRPO — a safe fallback rather than a failure mode.
- The optimal value of $\omega$ may vary across tasks, although experiments show robustness within the range $[0.4, 1.2]$.

## Related Work & Insights
- **vs. GRPO/RLOO**: These methods treat entire multi-turn trajectories as single-turn responses, losing inter-step discrimination. GiGPO restores this at no additional cost.
- **vs. PPO**: PPO performs step-level advantage estimation via a critic network, incurring extra memory and training cost. GiGPO replaces the neural network (critic) with a data structure (hashmap), achieving comparable or superior performance.
- **vs. ArCHer / AgentQ**: These methods use additional value networks or MCTS for step-level credit assignment at high computational cost. Anchor state grouping is an elegant low-cost alternative.
- **vs. RAGEN**: RAGEN concatenates entire interaction histories as episode-level responses for standard GRPO, which does not scale to long-horizon tasks. GiGPO's step-wise design is more scalable.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Anchor state grouping is a remarkably simple yet effective idea that resolves the fundamental limitation of group-based RL in step-level credit assignment at zero cost.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Experiments cover three model scales (1.5B/3B/7B), four task categories (ALFWorld/WebShop/QA/VLM), with comprehensive ablations, training dynamics analysis, and computational overhead profiling.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figure 3's illustration of step-level groups in WebShop is highly intuitive; the narrative from motivation to method to experiments is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ The open-source verl-agent framework and plug-in design fully compatible with existing group-based RL methods make this work directly impactful for LLM agent training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[NeurIPS 2025\] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](btlui_blinkthinklink_reasoning_model_for_gui_agent.md)
- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)

<!-- RELATED:END -->
