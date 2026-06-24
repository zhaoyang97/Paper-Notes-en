---
title: >-
  [Paper Note] AgentRM: Enhancing Agent Generalization with Reward Modeling
description: >-
  [ACL 2025][LLM Alignment][agent] AgentRM is proposed, a generalizable reward model constructed via explicit, implicit, and LLM-as-judge approaches. It guides policy models using test-time search (Best-of-N / Beam Search), achieving an average improvement of 8.8 points across 9 agent tasks and outperforming the best generalist agent by 4.0 points.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "agent"
  - "reward model"
  - "generalization"
  - "test-time search"
  - "MCTS"
  - "Best-of-N"
date: 2026-05-08
content_hash: c6b3616f87762d3a
---

# AgentRM: Enhancing Agent Generalization with Reward Modeling

**Conference**: ACL 2025  
**arXiv**: [2502.18407](https://arxiv.org/abs/2502.18407)  
**Code**: -  
**Area**: LLM Alignment  
**Keywords**: agent, reward model, generalization, test-time search, MCTS, Best-of-N  

## TL;DR

AgentRM is proposed, a generalizable reward model constructed via explicit, implicit, and LLM-as-judge approaches. It guides policy models using test-time search (Best-of-N / Beam Search), achieving an average improvement of 8.8 points across 9 agent tasks and outperforming the best generalist agent by 4.0 points.

## Background & Motivation

- **Limitations of Prior Work**: LLM-based agents perform well on tasks seen during training but exhibit poor generalization on unseen tasks. Prior works fine-tune policy models by expanding the diversity of training tasks, but policy fine-tuning inflates the probability of seen action tokens while suppressing unseen actions, leading to performance degradation on held-out tasks.
- **Key Findings**: Fine-tuning reward models is more robust than directly fine-tuning policy models—fine-tuning a policy model on a single task only improves performance on that task while degrading other tasks (positive diagonal), whereas fine-tuning a reward model on a single task improves performance on most unseen tasks.
- **Key Analysis**: The regression training objective of the reward function is inherently insensitive to the specific distribution of action tokens, thus avoiding the over-bias toward the action space of training tasks often caused by policy fine-tuning.
- **Ours**: AgentRM is proposed to systematically study three reward modeling methods and guide policy models to make better decisions during test-time via Best-of-N sampling and step-level Beam Search.

## Method

### Overall Architecture

The workflow of AgentRM consists of four steps: (1) **Behavior Cloning**: SFT on expert trajectories to obtain the initial policy model; (2) **Search Tree Construction**: construct MCTS search trees in the training task environments using the SFT policy model; (3) **Reward Model Training**: extract state-reward pairs from search trees to train the generalizable reward model; (4) **Test-Time Search**: guide the policy model on unseen tasks using the reward model (Best-of-N or Beam Search).

### Key Designs

- **Explicit Reward Modeling (Explicit RM)**: Construct search trees using MCTS heuristic search, estimate the Q-value $V(s_t)$ of each state through Monte Carlo simulations, and train a language model with a value head to minimize the MSE loss $\mathcal{L}(\theta) = \frac{1}{N}\sum_{t=1}^{N}(\hat{V}(s_t) - V(s_t))^2$. The search tree employs UCB for node selection, action merging to reduce redundant exploration, and simulation node caching to accelerate search.
- **Implicit Reward Modeling (Implicit RM)**: Based on the DPO paradigm, derive step-level process rewards $r_\theta^t = \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ from models trained on outcome rewards, without requiring explicit step-level reward annotations.
- **Step-Level Beam Search**: Sample $W_1 \times W_2$ candidate actions at each step, retain top-$W_1$ actions based on the reward model's score, then expand $W_2$ actions for each of the retained states, iterating until termination.

### Loss & Training

- Explicit RM: MSE loss to learn Q-values.
- Implicit RM: MSE loss to fit the progress scalar rewards provided by the environment.
- Policy Model SFT: Standard autoregressive cross-entropy loss.

## Key Experimental Results

### Main Results (Comparison with Generalist Agents, LLaMA-3-8B Policy Model)

| Method | Web | Embodied | Text Game | Tool | Overall |
|------|-----|----------|-----------|------|------|
| GPT-4o | 57.7 | 73.6 | 59.9 | 49.7 | 65.9 |
| AgentGym | 68.5* | 62.2* | 28.5 | 55.3* | 59.3* |
| Greedy Search | 57.8 | 50.6 | 37.4 | 56.6 | 52.7 |
| Best-of-5 (Explicit RM) | 62.4 | 62.7 | 47.8 | 68.7 | **61.5** |
| Beam Search (Explicit RM) | 64.4 | 65.1 | 47.5 | 64.0 | **63.3** |

*\* Indicates tasks seen during training*

### Ablation Study

| Dimensional Analysis | Key Findings |
|----------|----------|
| Three RM Comparison | Explicit RM is the best (+8.8), followed by Implicit RM (+2.0), while LLM-as-Judge leads to degradation (-0.6) |
| Robustness Testing | Under 5 types of perturbations on Alfworld, AgentGym drops by 25.6, Agent-FLAN drops by 30.3, while AgentRM only drops by 2.1, achieving the lowest standard deviation |
| Weak-to-Strong Generalization | Direct application of the RM trained with LLaMA-3-8B samples to LLaMA-3-70B yields a 12.6-point gain |
| Training Data Scaling | Only 4K states are needed to outperform LLM-as-Judge (57.6 vs 52.1), with performance growing log-linearly with data size |
| State Representation Ablation | Primarily relies on action tokens; removing thought and observation tokens simultaneously results in a 3.2-point performance drop |

### Key Findings

1. Explicit RM is consistently optimal across all settings, and Beam Search yields further improvements (63.3 overall vs 61.5 for Best-of-5).
2. AgentRM demonstrates remarkable weak-to-strong generalization—RMs trained on weak model (8B) samples deliver greater gains on strong models (70B) (+12.6 vs +8.8).
3. Prior generalist agents (AgentGym, Agent-FLAN) exhibit severe overfitting—simple action perturbations trigger performance collapses (up to -30.3), whereas AgentRM remains stable.
4. On specialized tasks, AgentRM + Beam Search outperforms the best specialized agent (QLASS) by 11.4 points.

## Highlights & Insights

- Reveals the core insight that "fine-tuning a reward model is more robust to generalization than fine-tuning a policy model," supported by clear experimental visualizations.
- The discovery of weak-to-strong generalization is highly practical—leveraging weak model experience to enhance the decision quality of large models.
- Systematically compares the effectiveness of three reward modeling paradigms in agent scenarios, filling a literature gap.
- Perturbation tests reveal that prior generalist agents are essentially "memorizing" instead of "understanding" tasks.

## Limitations & Future Work

- MCTS search tree construction requires interaction with environments, making it inapplicable to real-world environments that cannot be reset.
- Training data is sourced from only 3 held-in tasks (Webshop, Alfworld, Sciworld), with limited task diversity.
- The improvements of Implicit RM and LLM-as-Judge are relatively small; the potential of these two methods in agent scenarios may require further investigation.
- The reward model is trained solely using LLaMA-3-8B, leaving benefits on larger-scale RMs unexplored.
- The computational cost of Beam Search along with $W_1 \times W_2$ is significantly higher than Best-of-N, requiring trade-offs for real deployment.

## Related Work & Insights

- **Specialized Agents**: SPIN, NAT, ETO, StepAgent, QLASS—each trained individually per task, lacking cross-task generalization.
- **Generalist Agents**: Agent-FLAN, AgentGym, AgentGen—fine-tune policy models through multi-task learning but still suffer from overfitting.
- **Reward Modeling**: Process Reward Model (established in mathematical reasoning), DPO implicit rewards, LLM-as-Judge—first systematically applied to agent tasks.
- **Test-Time Search**: Best-of-N, Beam Search, MCTS—transferring from mathematical reasoning to agent decision-making.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Practicality | 5 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Overall Rating | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ACL 2025\] A Dual-Mind Framework for Strategic and Expressive Negotiation Agent](a_dual-mind_framework_for_strategic_and_expressive_negotiation_agent.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)
- [\[ACL 2025\] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings](focalpo_enhancing_preference_optimizing_by_focusing_on_correct_preference_rankin.md)
- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)

</div>

<!-- RELATED:END -->
