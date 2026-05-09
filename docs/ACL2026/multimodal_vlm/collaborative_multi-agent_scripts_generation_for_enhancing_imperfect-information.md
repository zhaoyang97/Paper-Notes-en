---
title: >-
  [Paper Note] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games
description: >-
  [ACL 2026][Multimodal VLM][imperfect-information reasoning] This paper proposes a collaborative multi-agent framework for automatically generating high-quality murder mystery game scripts and training data. Through a two-stage training strategy (CoT fine-tuning + GRPO reinforcement learning with ScoreAgent reward shaping), it enhances VLM multi-hop reasoning under imperfect information, achieving significant improvements on WhodunitBench in narrative reasoning, fact extraction, and deception resistance.
tags:
  - ACL 2026
  - Multimodal VLM
  - imperfect-information reasoning
  - murder mystery games
  - multi-agent data generation
  - vision-language models
  - reinforcement learning
date: 2026-05-08
content_hash: e8dced13c77d952f
---

# Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games

**Conference**: ACL 2026
**arXiv**: [2604.11741](https://arxiv.org/abs/2604.11741)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: imperfect-information reasoning, murder mystery games, multi-agent data generation, vision-language models, reinforcement learning

## TL;DR
This paper proposes a collaborative multi-agent framework for automatically generating high-quality murder mystery game scripts and training data. Through a two-stage training strategy (CoT fine-tuning + GRPO reinforcement learning with ScoreAgent reward shaping), it enhances VLM multi-hop reasoning under imperfect information, achieving significant improvements on WhodunitBench in narrative reasoning, fact extraction, and deception resistance.

## Background & Motivation

**Background**: Vision-language models (VLMs) excel at perception tasks but degrade on complex multi-hop reasoning involving imperfect information, deception, and multi-player social interaction. Murder mystery games, which require players to infer hidden truths from partial clues, serve as an ideal testbed for studying such reasoning.

**Limitations of Prior Work**: (1) Large-scale, high-quality datasets for fine-tuning and evaluating VLMs in murder mystery scenarios are lacking; (2) Manual production of high-quality murder mystery scripts is costly and difficult to scale; (3) Existing VLMs perform poorly in role consistency (murderers must deceive; innocents must cooperate) and multimodal multi-hop reasoning (combining textual and visual clues); (4) Role-playing and interactive discussion lack ground-truth answers, making pure SFT insufficient for training such behaviors.

**Key Challenge**: VLMs must perform reliable reasoning in environments with incomplete and deceptive information, yet appropriate training data and training methodologies are absent.

**Goal**: (1) Build a scalable multi-agent data synthesis framework; (2) Design a two-stage training strategy suited to imperfect-information reasoning.

**Key Insight**: Powerful LLMs (Gemini 2.5 Pro) are employed as collaborative agents to generate game scripts, followed by an agent-supervised training strategy to enhance the target VLM.

**Core Idea**: Generation agents (story outline → character scripts → clues → dialogues → QA) and evaluation agents (quality control + reward shaping) collaboratively construct training data; a two-stage training pipeline (SFT + GRPO with ScoreAgent) enhances the VLM.

## Method

### Overall Architecture
Two main modules: (1) **Data Generation Module** — multiple specialized agents (OutlineAgent, CharacterAgent, ClueAgent, RoleplayAgent, QaAgent, CriticAgent) collaboratively generate murder mystery scripts and training data; (2) **Model Enhancement Module** — Stage 1 SFT establishes foundational reasoning ability → Stage 2 GRPO reinforcement learning optimizes role-specific behaviors under ScoreAgent supervision.

### Key Designs

1. **Multi-Agent Script Generation Framework**:

    - **Function**: Automatically generates diverse, high-quality murder mystery game scripts and training data.
    - **Mechanism**: A pipeline of six specialized agents: OutlineAgent constructs the crime-day narrative (motives + secrets) → CharacterAgent elaborates each character's daily actions and interactions → CriticAgent evaluates and provides feedback along four dimensions (plot complexity, character development, difficulty, logical consistency) → ClueAgent generates multimodal clues (visual + textual) → RoleplayAgent simulates multi-turn dialogues → QaAgent generates reasoning chains and QA pairs ranging from single-hop to multi-hop.
    - **Design Motivation**: A single model generating an entire script is prone to logical inconsistencies. Specialized role division combined with CriticAgent feedback ensures script quality.

2. **ScoreAgent-Supervised GRPO Reinforcement Learning**:

    - **Function**: Optimizes VLM role consistency and reasoning quality.
    - **Mechanism**: Different reward functions are designed for different types of training data. For **non-verifiable data** (self-introduction, discussion): a ScoreAgent (LLM-as-Judge) scores role consistency; discussion additionally incorporates $S_{\text{choice}}$ (querying a suspect yields 1 point, querying another player yields 0.5 points, querying oneself yields 0 points). For **verifiable data** (QA): a weighted combination of answer correctness, format correctness, and clue-matching correctness.
    - **Design Motivation**: SFT can establish basic capabilities but cannot handle role-playing behaviors lacking ground-truth answers. GRPO leverages ScoreAgent evaluations to distinguish good from poor role-playing performance.

3. **Reasoning Chain Generation under Imperfect Information**:

    - **Function**: Provides training examples of reasoning under incomplete information conditions.
    - **Mechanism**: Reasoning chains are automatically generated based on incomplete information — players can only observe their own clues and public information, from which multi-hop reasoning must be performed. This contrasts with conventional CoT, which assumes complete information.
    - **Design Motivation**: Traditional reasoning data assumes complete information, whereas the core challenge of murder mystery games is precisely the incompleteness of information and the presence of deception.

## Key Experimental Results

### Main Results (WhodunitBench)

| Method | MMR | CMD | RP | DM | LSU | TIU | MIU |
|--------|-----|-----|----|----|-----|-----|-----|
| GPT-4V | 58.75 | 26.43 | 6.43 | 24.2% | 92.40 | 51.88 | 69.25 |
| Gemini-1.5-Pro | 57.39 | 19.20 | 7.22 | 16.9% | — | — | — |
| Qwen2.5-VL-3B | baseline | — | — | — | — | — | — |
| **Qwen2.5-VL-3B + Ours** | **significant gain** | **gain** | **gain** | **gain** | **gain** | **gain** | **gain** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| SFT only | Establishes basic reasoning but poor role consistency |
| SFT + RL without ScoreAgent | Inaccurate reward signals; limited improvement |
| **SFT + ScoreAgent GRPO** | Dual improvement in role consistency and reasoning quality |

### Key Findings
- **The multi-agent framework successfully generates diverse, logically consistent murder mystery data**; the CriticAgent feedback mechanism substantially improves script quality.
- **The two-stage training is consistently effective at both the 3B and 7B scales.**
- **The role-specific reward design of ScoreAgent** enables the model to learn distinct behavioral patterns for murderers and innocents.
- **GRPO yields especially pronounced improvements in role-playing behavior** — SFT is limited in training behaviors without ground-truth answers.
- **Characteristics of low-scoring samples are clear**: topic deviation, self-contradiction, and premature identity disclosure.

## Highlights & Insights
- **Modeling murder mystery games as a reasoning training platform for VLMs** is a clever task selection — it subsumes multiple challenges including imperfect information, deception detection, multi-hop reasoning, and multimodal integration.
- **The differentiated reward design of ScoreAgent** (distinct reward functions for verifiable vs. non-verifiable data) is a practical solution that avoids training a separate reward model for tasks without ground-truth answers.
- **Scalability of the data generation framework**: by adding or adjusting specialized agents, the framework can be adapted to other game-theoretic tasks (e.g., Werewolf, courtroom simulation).

## Limitations & Future Work
- WhodunitBench contains only 50 scripts, limiting evaluation scale.
- The quality of generated scripts depends on Gemini 2.5 Pro's capabilities, incurring high computational cost.
- Role-playing evaluation still relies primarily on LLM-as-Judge, introducing subjectivity.
- Real multi-player interaction training among multiple VLMs has not been explored.
- Visual clues are currently simple and do not involve complex scene understanding (e.g., surveillance video analysis).
- The diversity of training data is constrained by the creativity of the generation agents.

## Related Work & Insights
- **vs. WhodunitBench (Xie et al., 2024)**: WhodunitBench provides an evaluation platform but lacks sufficient data. This work contributes a data generation framework and training methodology.
- **vs. AgentInstruct / MATRIX**: These works focus on general synthetic data generation; this paper specializes in structured data generation for imperfect-information game scenarios.
- **vs. Reason-RFT / SRPO**: These are general reasoning enhancement methods; the ScoreAgent design in this work is tailored specifically for role consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using murder mystery games as a VLM reasoning training scenario is novel; the multi-agent data generation framework is thoughtfully designed.
- **Experimental Thoroughness**: ⭐⭐⭐ WhodunitBench is limited in scale; specific numerical results are insufficiently complete.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear, though the paper is lengthy.
- **Value**: ⭐⭐⭐⭐ Makes a distinctive contribution to VLM reasoning training under imperfect information.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/multimodal_vlm/comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](../../AAAI2026/multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)

<!-- RELATED:END -->
