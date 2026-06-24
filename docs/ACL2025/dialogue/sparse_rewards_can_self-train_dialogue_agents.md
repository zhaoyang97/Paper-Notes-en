---
title: >-
  [Paper Note] Sparse Rewards Can Self-Train Dialogue Agents
description: >-
  [ACL 2025 (Findings)][Dialogue Systems][Self-training] This paper proposes JOSH (Juxtaposed Outcomes for Simulation Harvesting), a self-alignment algorithm that enables LLM dialogue agents to autonomously improve their performance through simulated environments with sparse rewards without external human feedback. Additionally, the ToolWOZ sparse-reward tool-calling simulation environment is constructed for validation.
tags:
  - "ACL 2025 (Findings)"
  - "Dialogue Systems"
  - "Self-training"
  - "Sparse rewards"
  - "Dialogue agent"
  - "Tool use"
  - "Simulated environment"
date: 2026-05-08
content_hash: 67138af9ce6042bc
---

# Sparse Rewards Can Self-Train Dialogue Agents

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2409.04617](https://arxiv.org/abs/2409.04617)  
**Code**: [GitHub](https://github.com/asappresearch/josh-llm-simulation-training)  
**Area**: Others  
**Keywords**: Self-training, Sparse rewards, Dialogue agent, Tool use, Simulated environment  

## TL;DR

This paper proposes JOSH (Juxtaposed Outcomes for Simulation Harvesting), a self-alignment algorithm that enables LLM dialogue agents to autonomously improve their performance through simulated environments with sparse rewards without external human feedback. Additionally, the ToolWOZ sparse-reward tool-calling simulation environment is constructed for validation.

## Background & Motivation

**Background**: Advances in LLM agents in multi-turn dialogue tasks mainly rely on two pillars: supervised fine-tuning (SFT) and high-quality human feedback (RLHF/DPO). These methods train models using manually annotated dialogue examples or preference data.

**Limitations of Prior Work**: As foundational LLM capabilities continue to enhance, acquiring meaningful human feedback becomes increasingly difficult and expensive. In certain specialized domains (e.g., complex tool use, API orchestration), foundational LLMs may already approach or even exceed the capability of average human annotators, making traditional human feedback-driven methods impractical. Furthermore, manually annotating dialogue data (especially multi-turn dialogues with tool use) is extremely costly and difficult to ensure consistency.

**Key Challenge**: Models require large amounts of high-quality training signals to improve, but the cost and quality of acquiring human feedback are becoming bottlenecks. Can models generate their own training data to achieve self-improvement?

**Goal**: Design a self-improvement paradigm that requires no external human feedback, enabling LLM agents to autonomously improve their multi-turn dialogue and tool use capabilities through interaction with simulated environments.

**Key Insight**: It is observed that many dialogue tasks can define clear success/failure conditions (e.g., whether a booking has been completed, or whether information was correctly queried). Although such sparse binary reward signals are simple, they are sufficient to guide the model to learn correct behaviors. The key lies in how to efficiently utilize these sparse signals.

**Core Idea**: Utilize beam search-style simulated interactions to generate multiple dialogue trajectories, filter out successful trajectories as positive samples using sparse rewards for self-training, and thereby continuously improve the model without relying on human feedback.

## Method

### Overall Architecture

The workflow of JOSH is as follows: (1) The LLM agent engages in multi-turn dialogues with a User Simulator in a simulated environment while exploring multiple parallel trajectories (beam search); (2) The environment provides a sparse reward (success/failure) based on predefined goal conditions; (3) Successful trajectories are collected as training data; (4) The LLM is fine-tuned using these self-generated high-quality data. The entire process requires no human intervention.

### Key Designs

1. **JOSH Search Algorithm (Juxtaposed Outcomes for Simulation Harvesting)**:

    - **Function**: Efficiently extract successful action trajectories from simulated dialogues.
    - **Mechanism**: In each step of dialogue interaction, JOSH does not generate a single agent response. Instead, it generates multiple candidate responses in parallel (similar to beam search), with each candidate forming an independent dialogue branch. After the dialogue ends, the environment provides a sparse binary reward (goal achieved = 1, failed = 0). JOSH retains all successful dialogue branches as positive samples. Increasing the beam size can enhance the probability of finding successful trajectories, allowing sufficient positive samples to be collected even when the initial capability of the model is weak.
    - **Design Motivation**: Traditional self-play methods are inefficient in LLM scenarios because failures dominate. JOSH significantly improves the efficiency of finding successful cases through parallel search.

2. **ToolWOZ Simulation Environment**:

    - **Function**: Provide a standardized tool-calling dialogue simulator with sparse rewards.
    - **Mechanism**: Built upon the classic MultiWOZ dialogue dataset, it is transformed into an interactive tool-calling simulation environment. The environment consists of a user simulator (which provides dialogue goals and responses), tool/API interfaces (e.g., hotel reservations, restaurant queries), and a reward function (which checks whether all subtasks are correctly completed). The reward strategy is sparse—only a global success/failure signal is given at the very end of the dialogue, with no feedback provided for intermediate steps.
    - **Design Motivation**: Existing dialogue training benchmarks often rely on dense stepwise feedback, which is not realistic. Sparse rewards are closer to real-world deployments (where users express satisfaction/dissatisfaction only at the end), and learning from such signals has more practical value.

3. **Preference Annotation and LoRA Fine-Tuning Strategy**:

    - **Function**: Convert the discovered successful and failed trajectories into model update signals.
    - **Mechanism**: Dialogue trajectories collected by JOSH naturally form preference pairs—successful and failed branches originating from the same starting point. These preference pairs are used to fine-tune the model via preference optimization algorithms such as DPO/KTO. Simultaneously, LoRA is adopted for parameter-efficient fine-tuning to prevent the degradation of general capabilities caused by full-parameter updates.
    - **Design Motivation**: Directly using SFT only leverages positive samples, whereas preference optimization can learn from both positive and negative samples, utilizing information more fully.

### Loss & Training

KTO (Kahneman-Tversky Optimization) loss is adopted for preference learning, combined with parameter-efficient fine-tuning using LoRA. The training data is entirely automatically generated by JOSH without human annotation. Multi-turn iteration: Generation $\rightarrow$ Filtering $\rightarrow$ Training $\rightarrow$ Regeneration, forming a self-improvement loop.

## Key Experimental Results

### Main Results

| Model/Method | ToolWOZ Success Rate | $\tau$-bench Success Rate | MT-Bench Score | Notes |
|---|---|---|---|---|
| GPT-4o-mini (baseline) | Baseline | Baseline | Baseline | Without self-training |
| GPT-4o-mini + JOSH | Significant Gain | Significant Gain | Maintained | Frontier models also benefit |
| LLaMA-3-8B (baseline) | Low | Low | Baseline | Weak initial capability of small model |
| LLaMA-3-8B + JOSH | Massive Gain | Massive Gain | Maintained | Most pronounced improvement |
| LLaMA-3-8B + SFT only | Moderate Gain | Moderate Gain | Slightly Decreased | Limited effectiveness using only positive samples |

### Ablation Study

| Configuration | ToolWOZ Success Rate | Notes |
|---|---|---|
| JOSH + KTO | Best | Full scheme |
| JOSH + SFT (Positive Only) | Second Best | Does not utilize negative samples |
| Without JOSH (Direct Generation) | Poor | Low beam size fails to find sufficient positive samples |
| Beam size = 2 | Low | Insufficient search space |
| Beam size = 8 | Best | Sufficient parallel exploration |
| Without LoRA (Full Parameter) | Decreased | Overfitting + general capability degradation |

### Key Findings

- The performance gain of JOSH on smaller models (LLaMA-3-8B) is much larger than on larger models (GPT-4o-mini), as smaller models have more room for improvement.
- Beam size significantly impacts performance, with size=8 being the sweet spot balancing effectiveness and efficiency.
- The MT-Bench score of the models remains unchanged after JOSH training, indicating that the improvement in tool-use capability does not come at the expense of general dialogue capabilities.
- Cross-dataset generalization on $\tau$-bench (another tool-use benchmark) validates the generalizability of the proposed method.

## Highlights & Insights

- **Self-improvement without Human Feedback**: JOSH breaks the reliance on human annotations, achieving continuous progress through self-play within a simulated environment. This is particularly valuable considering the rising costs of human feedback.
- **Efficient Utilization of Sparse Rewards**: Learning can be guided solely by binary success/failure signals, avoiding the need to design complex reward functions. This "environment as teacher" paradigm can be transferred to other task-oriented dialogue scenarios.
- **Cross-Benchmark Generalization**: Strategies trained on ToolWOZ can be directly transferred to $\tau$-bench, indicating that JOSH learns generic tool-calling capabilities rather than environment-specific shortcuts.

## Limitations & Future Work

- ToolWOZ is built upon MultiWOZ, which limits its domain to booking scenarios such as hotels and restaurants; open-domain dialogue tasks would require new environments.
- Sparse rewards assume that dialogue goals can be explicitly defined and automatically evaluated, which does not apply to open-domain chit-chat.
- The beam search process of JOSH incurs high computational costs, especially when using larger models, being several times that of standard inference.
- Currently, only the effect of single-round JOSH iteration has been validated; it remains unclear whether multi-round self-improvement has an upper limit or suffers from degradation.
- In the future, JOSH can be explored for extension to more complex agent scenarios (e.g., code generation, browser interaction).

## Related Work & Insights

- **vs Self-Play (AlphaGo style)**: AlphaGo-style self-play requires a perfect environment model. JOSH relaxes this requirement, needing only a simulator with sparse rewards.
- **vs RLHF/DPO**: Traditional preference learning relies on human-annotated preference pairs, whereas JOSH automatically generates preference pairs through a simulated environment, completely eliminating manual labor.
- **vs ReST/STaR**: These self-training methods mainly target reasoning tasks (e.g., mathematics). JOSH is the first to systematically extend them to multi-turn tool-calling dialogues.
- This work provides a reusable framework for agent self-training (the JOSH repository is open-sourced), which can serve as a baseline for subsequent agent training research.

## Rating

- Novelty: ⭐⭐⭐⭐ Self-training combined with sparse rewards is a novel combination for dialogue agents, although the individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models and benchmarks with detailed ablations, though multi-round iteration analysis is lacking.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the construction process of ToolWOZ is detailed.
- Value: ⭐⭐⭐⭐ Provides a practical solution for agent improvement without human feedback, and the open-source release is thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/dialogue/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](../../ACL2026/dialogue/dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ICML 2025\] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](../../ICML2025/dialogue/position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)
- [\[ICLR 2026\] Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings](../../ICLR2026/dialogue/dropping_just_a_handful_of_preferences_can_change_top_large_language_model_ranki.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)

</div>

<!-- RELATED:END -->
