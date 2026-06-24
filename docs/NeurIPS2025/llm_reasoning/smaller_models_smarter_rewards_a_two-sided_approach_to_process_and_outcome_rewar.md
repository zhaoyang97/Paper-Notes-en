---
title: >-
  [Paper Note] Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards
description: >-
  [NeurIPS 2025 (Workshop: Foundations of Reasoning in Language Models)][Reasoning][Reward Model] The final layer of Phi-4 family small models (3.8B/14B) is replaced with a regression head and fine-tuned, enabling them to serve simultaneously as ORM (outcome reward model) and PRM (process reward model). On code generation tasks, selecting the optimal rollout yields 20%+ improvements in pass@k.
tags:
  - "NeurIPS 2025 (Workshop: Foundations of Reasoning in Language Models)"
  - "Reasoning"
  - "Reward Model"
  - "Process Reward"
  - "Outcome Reward"
  - "Code Generation"
  - "Small Models"
date: 2026-05-08
content_hash: c62e26cbb23563a4
---

# Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards

**Conference**: NeurIPS 2025 (Workshop: Foundations of Reasoning in Language Models)  
**arXiv**: [2510.23083](https://arxiv.org/abs/2510.23083)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Reward Model, Process Reward, Outcome Reward, Code Generation, Small Models

## TL;DR
The final layer of Phi-4 family small models (3.8B/14B) is replaced with a regression head and fine-tuned, enabling them to serve simultaneously as ORM (outcome reward model) and PRM (process reward model). On code generation tasks, selecting the optimal rollout yields 20%+ improvements in pass@k.

## Background & Motivation

**Background**: Reasoning models (o1, R1) improve reasoning capability through long-chain CoT, with training relying on reward models to distinguish good from bad reasoning trajectories. Reward models fall into two categories: ORM (scoring only the final outcome) and PRM (scoring intermediate steps progressively).

**Limitations of Prior Work**: ORM signals are sparse and suffer from delayed credit assignment—providing only a single terminal score with no guidance for intermediate search steps. PRM requires expensive annotation of intermediate steps (typically manual). Furthermore, existing reward models are generally based on large models (tens of billions of parameters), incurring high deployment costs.

**Key Challenge**: Can a small model (<15B) simultaneously fulfill the roles of both ORM and PRM—evaluating the correctness of complete rollouts while also estimating the success probability of intermediate states during generation?

**Goal**: To verify whether small LMs from the Phi-4 family can become effective reward models through simple architectural modifications (regression head replacement + SFT), serving both ORM and PRM roles simultaneously.

**Key Insight**: A forking tokens strategy is employed to construct training data—identifying the lowest-probability "forking point" tokens during generation, branching from these positions to produce multiple rollouts, and using test cases to determine correctness, thereby obtaining token-level state-value labels.

**Core Idea**: A Phi-4 decoder architecture combined with a regression head and sigmoid activation is trained on forking-point code generation data, enabling the small model to estimate success probability at each token position.

## Method

### Overall Architecture

The input consists of a programming problem and (partially) generated code; the output is a success probability estimate $v_i = \mathbb{E}[J(\mathcal{G}(s_1, ..., s_i))]$ at each token position—i.e., the probability of obtaining a correct answer by continuing to sample from the current prefix. The model can score complete rollouts (ORM) as well as intermediate states during generation (PRM).

### Key Designs

1. **Forking Tokens Data Construction**:

    - Function: Generates labeled training data from the APPS programming dataset.
    - Mechanism: Phi-4-mini generates one primary rollout per problem; the 6 lowest-probability token positions ("forking tokens") are identified; 6 new rollouts are branched from each position, yielding 36 rollouts per problem. Test case execution determines correctness.
    - Design Motivation: Low-probability tokens represent critical decision points in reasoning—either genuine ambiguity where multiple viable paths exist, or positions where the model is likely to err. Branching from these points produces diverse correct/incorrect samples that are more informative than random independent sampling.

2. **Value Head Architecture**:

    - Function: Converts a pretrained LM into a per-token value estimator.
    - Mechanism: The original classification head (next-token prediction) is replaced by a single-output linear regression layer followed by sigmoid activation, trained with BCE loss. Only the last 12 layers and the regression head are fine-tuned.
    - Design Motivation: The decoder-only architecture inherently enforces causal constraints—each position's prediction is based solely on preceding tokens—which precisely satisfies the requirements of state-value estimation.

3. **Balanced Dataset Strategy**:

    - Function: Over-samples each problem individually to balance correct and incorrect rollout counts.
    - Mechanism: The raw data contains varying correct/incorrect ratios per problem (correlated with difficulty); imbalance may cause the model to learn a "problem difficulty" shortcut rather than genuinely assessing code quality.
    - Design Motivation: Prevents the model from predicting correctness based on shallow features of the problem description (length, complexity, etc.), forcing it to attend to the actual reasoning and code content.

### Loss & Training
- Models: Phi-4-mini (3.8B) and Phi-4 (14B)
- Training for 2 epochs only, learning rate $1 \times 10^{-4}$, BCE loss
- Training set: 3,984 problems (~110K rollouts); test set: 465 problems

## Key Experimental Results

### Main Results (ORM Capability — Optimal Rollout Selection)

| Metric | Baseline | 3.8B Best | 14B Best | Gain |
|--------|----------|-----------|----------|------|
| Pass@1 (best 1 of 3) | 45% | 52% | **55%** | +22.2% |
| Pass@3 (best 3 of 10) | 65% | 73% | **78%** | +20.0% |

### Classification Accuracy Comparison

| Configuration | 3.8B balanced | 14B balanced | 14B imbalanced |
|---------------|---------------|--------------|----------------|
| Overall accuracy (balanced test) | 55.3% | **65.8%** | 60.5% |
| Predicted correct → truly correct | 58.5% | **67.3%** | 60.1% |
| Predicted incorrect → truly incorrect | 53.9% | **64.5%** | 60.9% |

### Key Findings
- **Model scale is critical**: The 14B model substantially outperforms the 3.8B model across all metrics, indicating that reward modeling places considerable demands on model capacity.
- **Balanced training is beneficial**: Balanced training yields notably better results on the balanced test set (65.8% vs. 60.5%), though the gap narrows on the rollout ranking task.
- **PRM capability**: The model requires approximately 50% of the code tokens before outperforming random guessing, suggesting it can detect errors in code but may be unable to assess the quality of individual reasoning steps.
- **Confidence increases with generation progress**: On correct rollouts, the model's success probability estimates rise progressively with token count, consistent with ground-truth trends.

## Highlights & Insights
- **Minimal architectural modification**: Replacing only the final layer and fine-tuning the last 12 layers suffices to transform a general-purpose LM into a dual ORM+PRM reward model, requiring only 2 training epochs.
- **Forking tokens data construction**: Leveraging the model's own token probabilities to identify critical decision points for branching is more efficient than random sampling.
- **Viability of small models**: The 14B Phi-4 demonstrates meaningful reward modeling capability, providing evidence for low-cost deployment of reward models.

## Limitations & Future Work
- **Limited data scale**: 36 rollouts per problem remains sparse, and ground-truth state values are approximate; the effect of larger branching factors remains unexplored.
- **Cold-start problem**: The current approach requires branching from a primary rollout; problems for which the model cannot generate any correct solution are excluded.
- **Evaluation limited to APPS**: Generalizability is unknown, as no evaluation is conducted on other tasks such as mathematical reasoning.
- **Limited PRM capability**: Effectiveness requires 50% of tokens, limiting utility for early-stage guidance (e.g., early pruning in tree search).
- **Distribution shift**: Training uses rollouts from the base policy, but the policy changes after post-training, potentially degrading value estimate accuracy.

## Related Work & Insights
- **vs. OVM (Yu et al.)**: OVM also employs a decoder-based value model but evaluates on mathematical tasks at sentence-level granularity; this work operates at token-level granularity on code tasks and includes a detailed analysis of PRM capability.
- **vs. Lightman et al.**: They find PRM superior to ORM, but their ORM does not evaluate intermediate steps; the value head approach proposed here inherently supports intermediate-step evaluation.
- **vs. PRIME**: PRIME uses LLM-as-judge (implicit reward model) to circumvent PRM annotation costs; this work automatically obtains labels via forking tokens and test cases, but is restricted to executable code.

## Rating
- Novelty: ⭐⭐⭐⭐ The value head architecture is not novel, but the forking tokens data construction and the unified ORM+PRM framing offer some originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analysis is detailed (balanced/imbalanced, ORM/PRM, per-percentile), but limited to a single dataset and two model scales, with no direct comparison against other reward modeling methods.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper's reasoning is clear, notation is well-defined, and research questions (RQ1/RQ2) are logically organized.
- Value: ⭐⭐⭐⭐ As a workshop paper, it validates the feasibility of small-model reward modeling, though the overall contribution is primarily empirical with limited depth and breadth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](../../ACL2026/llm_reasoning/process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ICML 2026\] Verifying Meta-Awareness via Predictive Rewards in Reasoning Models](../../ICML2026/llm_reasoning/verifying_meta-awareness_via_predictive_rewards_in_reasoning_models.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](../../ACL2026/llm_reasoning/hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[ICLR 2026\] Agentic Reinforcement Learning with Implicit Step Rewards](../../ICLR2026/llm_reasoning/agentic_reinforcement_learning_with_implicit_step_rewards.md)
- [\[ICML 2026\] Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](../../ICML2026/llm_reasoning/prioritize_the_process_not_just_the_outcome_rewarding_latent_thought_trajectorie.md)

</div>

<!-- RELATED:END -->
