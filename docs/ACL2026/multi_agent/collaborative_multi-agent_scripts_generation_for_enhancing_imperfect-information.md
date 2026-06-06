---
title: >-
  [Paper Note] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games
description: >-
  [ACL 2026][Multi-Agent][Imperfect Information Reasoning] A collaborative multi-agent framework is proposed to automatically generate high-quality murder mystery game scripts and training data. Through a two-stage trainin…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Imperfect Information Reasoning"
  - "Murder Mystery Game"
  - "Multi-Agent Data Generation"
  - "Vision-Language Model"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: c736fab097837529
---

# Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games

**Conference**: ACL 2026  
**arXiv**: [2604.11741](https://arxiv.org/abs/2604.11741)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Imperfect Information Reasoning, Murder Mystery Game, Multi-Agent Data Generation, Vision-Language Model, Reinforcement Learning

## TL;DR
A collaborative multi-agent framework is proposed to automatically generate high-quality murder mystery game scripts and training data. Through a two-stage training strategy (CoT Fine-tuning + GRPO Reinforcement Learning with ScoreAgent reward shaping), the framework enhances VLM multi-hop reasoning under imperfect information, significantly improving narrative reasoning, fact extraction, and deception resistance on WhodunitBench.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) excel in perception tasks but still degrade in complex multi-hop reasoning involving imperfect information, deception, and multi-player social interactions. Murder Mystery games, as a form of social reasoning, require players to infer hidden truths based on partial clues, making them an ideal testbed for such reasoning.

**Limitations of Prior Work**: (1) The murder mystery domain lacks large-scale, high-quality datasets for fine-tuning and evaluation. (2) Manual production of high-quality scripts is costly and difficult to scale. (3) Existing VLMs struggle with role consistency (murderers needing to deceive, innocents needing to cooperate) and multimodal multi-hop reasoning (combining text and visual clues). (4) Role-playing and interactive discussions lack standard answers, making pure SFT insufficient for training these behaviors.

**Key Challenge**: VLMs need to perform reliable reasoning in imperfect and deceptive information environments but lack appropriate training data and methodologies.

**Goal**: (1) Construct a scalable multi-agent data synthesis framework. (2) Design a two-stage training strategy suitable for reasoning under imperfect information.

**Key Insight**: Strong LLMs (e.g., Gemini 2.5 Pro) are utilized as Agents to collaboratively generate game scripts, and an Agent-monitored training strategy is then used to enhance the target VLM.

**Core Idea**: A collaborative pipeline of Generation Agents (Story Outline → Character Scripts → Clues → Dialogue → QA) and Evaluation Agents (Quality Control + Reward Shaping) constructs the training data, followed by two-stage training (SFT + GRPO with ScoreAgent) to enhance the VLM.

## Method

### Overall Architecture
The framework consists of two main modules: (1) Data Generation Module: Multiple specialized Agents (OutlineAgent, CharacterAgent, ClueAgent, RoleplayAgent, QaAgent, CriticAgent) collaborate to generate scripts and training data; (2) Model Enhancement Module: Stage 1 SFT establishes basic reasoning capabilities → Stage 2 GRPO Reinforcement Learning optimizes role-specific behaviors under ScoreAgent monitoring.

### Key Designs

1. **Multi-Agent Script Generation Framework**:
    - **Function**: Automatically generates diverse, high-quality murder mystery scripts and training data.
    - **Mechanism**: A pipeline of six specialized agents: OutlineAgent constructs the crime narrative (motives + secrets) → CharacterAgent refines daily actions and interactions → CriticAgent evaluates and provides feedback across four dimensions: plot complexity, character development, difficulty, and logical consistency → ClueAgent generates multimodal clues (visual + text) → RoleplayAgent simulates multi-round dialogues → QaAgent generates reasoning chains and QA pairs ranging from single-hop to multi-hop.
    - **Design Motivation**: Generating an entire script with a single model often leads to logical inconsistencies. Specialized division of labor combined with CriticAgent feedback ensures script quality.

2. **ScoreAgent-Monitored GRPO Reinforcement Learning**:
    - **Function**: Optimizes VLM role consistency and reasoning quality.
    - **Mechanism**: Tailored reward functions are designed for different data types. **Non-verifiable data** (Self-introductions, Discussions): ScoreAgent (LLM-as-Judge) scores role consistency. For discussions, $S_{\text{choice}}$ is added (1 point for questioning suspects, 0.5 for others, 0 for self). **Verifiable data** (QA): A weighted combination of answer correctness, format correctness, and clue matching accuracy.
    - **Design Motivation**: SFT establishes base capabilities but cannot handle role-playing behaviors without gold labels. GRPO utilizes ScoreAgent evaluations to distinguish between superior and inferior role-playing performances.

3. **Reasoning Chain Generation under Imperfect Information**:
    - **Function**: Provides reasoning examples under conditions of incomplete information for training.
    - **Mechanism**: Automatically generates reasoning chains based on incomplete information—players only see their own clues and public info, requiring multi-hop inference. This contrasts with traditional CoT which assumes full information.
    - **Design Motivation**: Traditional reasoning data assumes complete information, whereas the core challenge of murder mysteries is imperfect information and deception.

## Key Experimental Results

### Main Results (WhodunitBench)

| Method | MMR | CMD | RP | DM | LSU | TIU | MIU |
|------|-----|-----|----|----|-----|-----|-----|
| GPT-4V | 58.75 | 26.43 | 6.43 | 24.2% | 92.40 | 51.88 | 69.25 |
| Gemini-1.5-Pro | 57.39 | 19.20 | 7.22 | 16.9% | - | - | - |
| Qwen2.5-VL-3B | baseline | - | - | - | - | - | - |
| **Qwen2.5-VL-3B + Ours** | **Significant Gain** | **Gain** | **Gain** | **Gain** | **Gain** | **Gain** | **Gain** |

### Ablation Study

| Configuration | Description |
|------|------|
| SFT Only | Establishes basic reasoning but poor role consistency |
| SFT + RL w/o ScoreAgent | Inaccurate reward signals, limited improvement |
| **SFT + ScoreAgent GRPO** | Improvement in both role consistency and reasoning quality |

### Key Findings
- **The multi-agent framework successfully generates diverse and logically consistent data**, with the CriticAgent feedback mechanism significantly enhancing script quality.
- **Two-stage training is consistently effective** across both 3B and 7B model scales.
- **ScoreAgent's role-specific reward design** enables the model to learn distinct behavior patterns for murderers versus innocents.
- **GRPO provides particularly significant improvements for role-playing behaviors**, where SFT shows limited effectiveness for tasks without standard answers.
- **Low-score examples exhibit clear characteristics**: off-topic remarks, self-contradiction, and premature identity exposure.

## Highlights & Insights
- **Modeling murder mystery as a reasoning training platform for VLMs** is an ingenious task choice, covering challenges such as imperfect information, deception detection, multi-hop reasoning, and multimodal integration.
- **The differentiated reward design of ScoreAgent** (different functions for verifiable vs. non-verifiable data) is a practical solution that avoids training a separate reward model for tasks without gold-standard answers.
- **Scalability of the data generation framework**: By adding or adjusting specialized Agents, the framework can be adapted to other game-theoretic tasks such as Werewolf or courtroom simulations.

## Limitations & Future Work
- WhodunitBench contains only 50 scripts, leading to a limited evaluation scale.
- Script generation quality depends on Gemini 2.5 Pro, which results in high costs.
- Role-playing evaluation still relies primarily on LLM-as-Judge, which is subjective.
- Real-world multi-player interaction training between multiple VLMs has not been explored.
- Visual clues are currently simple and do not involve complex scene understanding (e.g., surveillance video analysis).
- The diversity of training data is limited by the creativity of the generation Agents.

## Related Work & Insights
- **vs. WhodunitBench (Xie et al., 2024)**: WhodunitBench provides an evaluation platform but lacks data. This paper provides a data generation framework and training methodology.
- **vs. AgentInstruct / MATRIX**: While those focus on general synthetic data, this work concentrates on structured data generation for imperfect information game scenarios.
- **vs. Reason-RFT / SRPO**: Unlike general reasoning enhancement methods, the ScoreAgent design here is specialized for role consistency.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing murder mystery as a VLM reasoning training scenario is novel; the multi-agent framework design is thorough.
- Experimental Thoroughness: ⭐⭐⭐ WhodunitBench scale is limited, and some specific numerical data are incomplete.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, though the length is substantial.
- Value: ⭐⭐⭐⭐ Makes unique contributions to VLM reasoning training under imperfect information.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](../../ICML2026/multi_agent/systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[ACL 2026\] when identity skews debate anonymization for bias-reduced multi-agent reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)

</div>

<!-- RELATED:END -->
