---
title: >-
  [Paper Note] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning
description: >-
  [ACL 2026][LLM Reasoning][agentic RL] HISR utilizes GPT-4o to segment agent trajectories into segments aligned with sub-goals. Subsequently…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "agentic RL"
  - "segmental process reward"
  - "hindsight model"
  - "credit assignment"
  - "PPO"
date: 2026-05-08
content_hash: d355dd0f4ef081e6
---

# HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2603.18683](https://arxiv.org/abs/2603.18683)  
**Code**: Coming soon  
**Area**: LLM Agent / Multi-turn Reinforcement Learning / Process Reward  
**Keywords**: agentic RL, segmental process reward, hindsight model, credit assignment, PPO

## TL;DR
HISR utilizes GPT-4o to segment agent trajectories into segments aligned with sub-goals. Subsequently, a hindsight model and a policy model calculate an importance score for each segment via their likelihood ratio to modulate segment-level process rewards. This approach achieves more reliable credit assignment on Alfworld, Virtualhome, and Webshop, with an average score increase of over 5 compared to SPA.

## Background & Motivation

**Background**: Enabling LLMs to act as agents for solving multi-turn decision-making tasks (household chores, online shopping, etc.) relies on multi-turn RL (PPO/GRPO). The reward model is central: (a) outcome RM (a single scalar at the end of the trajectory); (b) using MCTS or GPT-4 to label turn-level pseudo process rewards; (c) indirect supervision of turn-level credit using outcomes, as seen in SPA.

**Limitations of Prior Work**:
- Outcome RM: Under long horizons, "delayed rewards" are difficult to propagate to early actions, leading to near-random credit assignment.
- Turn-level pseudo labels: MCTS/GPT-4 labeling is expensive and prone to noise.
- Unlabeled turn-level: Completely ignores action importance, making process rewards "unfocused."
- **Key Challenge**: All the above assign rewards by "turn," but a sub-goal often spans multiple turns (e.g., "find the item first, then clean it"). Turn-level granularity is too fine, fragmenting actions belonging to the same sub-goal.

**Key Challenge**: The trilemma of reward granularity vs. labeling cost vs. signal focus—the finer the granularity, the less focused it is, while the coarser it is, the less it propagates.

**Goal**: (1) Increase granularity from turn-level to sub-goal segments; (2) Introduce action importance signals to segments without additional manual process labels.

**Key Insight**: Drawing from Hindsight Credit Assignment (Harutyunyan 2019)—the rationality of an action is better reflected by "looking back" after knowing the trajectory outcome than by "predicting forward."

**Core Idea**: Utilize GPT-4o for segmenting and a segment-level RM for segment rewards; then use the token-level likelihood ratio between a hindsight model and the policy model to aggregate importance scores $\hat z_s$ for each segment. These are multiplied into the rewards to obtain a corrected reward that is "sub-goal aligned + importance-amplified."

## Method

### Overall Architecture
Three stages:
1. **Behavior Cloning + Trajectory Collection**: SFT on expert trajectories $D_{bc}$ to obtain a reference policy $\pi_{ref}$; then execute $\pi_{ref}$ in the environment for $N$ rollouts, filtering failed/repetitive samples to get $D_{ct} = \{(\tau_{i,j}, R_{i,j})\}$.
2. **Auxiliary Model Construction**: Use GPT-4o to partition each trajectory into segments $\tau^s = \{s_1, \dots, s_n\}$, and train a Segmental Process RM (SPRM) and a hindsight model $\pi_{hind}$.
3. **PPO Training**: Use SPRM to predict segment-level reward $\hat R$ and the hindsight/policy likelihood ratio to calculate segment importance $\hat z_s$. Their product is normalized to get $\hat R_{him}$, which drives PPO after being combined with grounding rewards.

### Key Designs

1. **Segmental Process Reward Model (SPRM)**:

    - Function: Decomposes the trajectory scalar outcome $R$ into continuous sub-goal level rewards.
    - Mechanism: An MLP is attached to the final hidden layer of $\pi_{ref}$, outputting $r_i = W_2(\text{SiLU}(W_1 h_i))$ at each segment's end token. Using $\mathcal{L}_{sprm}(\tau^s) = (R - \sum_{i=1}^n r_i)^2$, the outcome is fitted as the sum of segment contributions, which is equivalent to learning a "task progress estimator."
    - Design Motivation: Setting the granularity specifically to segments aligned with sub-goals avoids being too fragmented at the turn-level while eliminating the need for GPT-4 to provide pseudo-labels—SPRM learns directly from the outcome.

2. **Hindsight Model and Likelihood Ratio Importance**:

    - Function: Assigns a "hindsight importance" score to each action.
    - Mechanism: A hindsight model $\pi_{hind}$ is trained via a masked language modeling style on $\pi_{ref}$: masking the response $a_k$ of each turn and requiring the model to reconstruct $a_k$ given the rest of the trajectory (including future tokens). Then defines the token-level ratio $r(a_k^j) = \pi_{hind}(a_k^j|o, a_{<k}, a_{>k}, a_k^{<j}) / \pi_{policy}(a_k^j|o_{\le k}, a_{<k}, a_k^{<j})$, which is aggregated into action importance $z(a_k) = \exp(\frac{1}{\beta |a_k|} \sum_j \log r(a_k^j))$. Intuition: If $z(a_k) > 1$, it indicates the agent is more likely to choose this action when looking back with hindsight, making the action "critical" to success. Turn-level $z(a_k)$ within the same segment are summed to get segment importance $z(s_i)$.
    - Design Motivation: No additional process labels are required; process information is captured solely through the likelihood variance between two models, equivalent to applying hindsight credit assignment to token-level LLM agents.

3. **Hindsight-modulated Reward + Grounding Reward**:

    - Function: Amplitudes/scales segment rewards based on importance and ensures action executability.
    - Mechanism: The normalized modulated reward is $\hat R_{him} = \frac{\hat R \odot \hat z_s}{\|\hat R \odot \hat z_s\|}$. A grounding reward $\hat r^g$ (1 if action is valid, 0 otherwise) is added: $\hat r^{fuse} = (1-\alpha) \hat r^{him} + \alpha \hat r^g$. This $\hat r^{fuse}$ is fed into PPO's GAE, calculating advantage as $\delta_t = \hat r_t^{fuse} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.
    - Design Motivation: $\hat R_{him}$ captures "correctness" while $\hat r^g$ captures "executability," making them complementary; without grounding, agents tend to generate hallucinated actions.

### Loss & Training
- BC Phase: NLL is calculated only on thought-action tokens, skipping observation tokens to improve training stability.
- SPRM Training: Uses only segment-level MSE on the single outcome label, requiring no turn-by-turn labeling.
- PPO Phase: Clip-objective $\mathcal{L}_{clip}(\theta) = \mathbb{E}_t[\min(\frac{\pi_\theta}{\pi_{\theta_{old}}} \hat A_t^{fuse}, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon) \hat A_t^{fuse})]$ with GAE-based advantage.

## Key Experimental Results

### Main Results
Evaluated against 12+ baselines on Alfworld (6 sub-tasks: PICK/CLEAN/HEAT/COOL/LOOK/PICK2), Virtualhome, and Webshop.

| Method | Type | Alfworld-Avg | Virtualhome | Webshop |
|------|------|--------------|-------------|---------|
| GPT-4o | PE | 48.0 | 20.8 | 23.7 |
| Gemini2.5pro | PE | 60.3 | 31.7 | 35.9 |
| SFT | BC | 73.1 | 51.8 | 62.0 |
| DPO | BC | 76.1 | 52.8 | 62.6 |
| PPO | RL | 73.9 | 51.0 | 62.1 |
| GRPO | RL | 73.4 | 51.2 | 61.8 |
| RAGEN | RL | 75.4 | 52.1 | 63.0 |
| PRM4A | RL | 73.9 | — | — |
| SPA (Prev. SOTA) | RL | 79.1 | 53.4 | 64.1 |
| **Ours (HISR)** | RL | **83.6** | **59.1** | **69.1** |

HISR sets new SOTA performance on all three benchmarks: Alfworld average +4.5, Virtualhome +5.7, and Webshop +5.0; specifically achieving 100% completion on the LOOK sub-task.

### Ablation Study

| Config | Alfworld-Avg | Virtualhome | Webshop | Description |
|------|--------------|-------------|---------|-------------|
| HISR (full) | 83.6 | 59.1 | 69.1 | Full method |
| w/o HIM | 80.6 | 55.1 | 63.7 | SPRM segment reward only |
| w/o SPR | 82.1 | 57.9 | 69.1 | Turn-level progress estimation |
| w/o BOTH | 87.5 (single task) | — | — | Degenerates to outcome reward |

### Key Findings
- The HIM module contributes the most: Removing it leads to a 5+ drop on Webshop, proving the marginal benefit of hindsight over segmentation alone.
- Segmentation is essential: Removing SPR drops the score by 1.2 on Virtualhome; the two modules are complementary.
- HISR shows the most significant gains in long-horizon sub-tasks (e.g., PICK2, requiring multi-step combinations, where PICK2 increased from 58.8 with SPA to 82.4), confirming the effectiveness of granularity + importance correction for long-chain reasoning.

## Highlights & Insights
- **Token-level LLM Hindsight**: The classical hindsight credit assignment concept (inferring action importance knowing the outcome) is implemented naturally via likelihood ratios between two language models, with virtually no new hyperparameters.
- **Label-free Importance**: This is the core difference from MCTS labeling methods like PRM4A; it is much more engineering-friendly.
- **Alignment with Sub-goals**: Aligning reward granularity with task structure (e.g., household chores/shopping tasks with inherent sub-goals) serves as a simple yet effective inductive bias for LLM agent RL.

## Limitations & Future Work
- Relies on GPT-4o for segmenting; transitioning to new tasks without a GPT-4-friendly structure might require resetting segmentation strategies.
- The hindsight and policy models share the same origin (both from $\pi_{ref}$), potentially amplifying similar biases.
- $\beta$ and $\alpha$ are hyperparameters; the paper lacks a detailed sensitivity analysis for these.
- Labeled "Work in progress," indicating some experiments and analyses are still undergoing iteration.

## Related Work & Insights
- **vs SPA (Wang 2025a)**: Both use outcomes to supervise processes, but SPA operates at the turn-level while HISR operates at the segment-level with added hindsight modulation, outperforming it across all benchmarks.
- **vs PRM4A**: PRM4A requires MCTS to provide pseudo-labels, whereas HISR is completely label-free.
- **vs RAGEN**: RAGEN focuses on exploration/data diversity, while HISR focuses on credit assignment; the two could be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ Hindsight credit assignment implemented via likelihood ratios in LLM agent RL for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 benchmarks + 12 baselines + complete ablations, though lacking hyperparameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Formulas and Figure 2 flowchart clearly explain the dual-model coordination.
- Value: ⭐⭐⭐⭐ A plug-and-play reward enhancement solution for any long-horizon LLM agent RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](mtr-bench_a_comprehensive_benchmark_for_multi-turn_reasoning_evaluation.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ACL 2026\] CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards](csrp_chain-of-thought_reasoning_for_chinese_text_correction_via_reinforcement_le.md)
- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)

</div>

<!-- RELATED:END -->
