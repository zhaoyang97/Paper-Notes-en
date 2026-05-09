---
title: >-
  [Paper Note] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents
description: >-
  [CVPR 2026][Multimodal VLM][Theory of Mind] MindPower proposes a robot-centric Theory-of-Mind (ToM) reasoning framework that organizes perception → belief → desire → intention → decision → action into a six-layer reasoning hierarchy, and optimizes reasoning consistency via Mind-Reward (based on GRPO), surpassing GPT-4o by 12.77% on decision-making and 12.49% on action generation.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Theory of Mind
  - BDI Reasoning
  - Embodied Agent
  - Mind-Reward
  - GRPO
date: 2026-05-08
content_hash: 7f08e52f6c599c05
---

# MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents

**Conference**: CVPR 2026  
**arXiv**: [2511.23055](https://arxiv.org/abs/2511.23055)  
**Code**: [zhangdaxia22.github.io/MindPower/](https://zhangdaxia22.github.io/MindPower/) (Benchmark)  
**Area**: Multimodal VLM  
**Keywords**: Theory of Mind, BDI Reasoning, Embodied Agent, Mind-Reward, GRPO

## TL;DR

MindPower proposes a robot-centric Theory-of-Mind (ToM) reasoning framework that organizes perception → belief → desire → intention → decision → action into a six-layer reasoning hierarchy, and optimizes reasoning consistency via Mind-Reward (based on GRPO), surpassing GPT-4o by 12.77% on decision-making and 12.49% on action generation.

## Background & Motivation

**State of the Field**: The embodied agent field is advancing rapidly — PaLM-E, RoboBench, and Smart-Help have realized task decomposition and execution. VLMs (GPT-4o, Gemini, Qwen-VL) perform well at the perception level but remain weak at inferring human intent and providing proactive assistance. Existing ToM benchmarks (MuMA-ToM, MMToM-QA) only evaluate inference of mental states of characters observed in videos.

**Limitations of Prior Work**: (1) Existing VLM-based agents can only execute explicit instructions and lack the ability to infer human beliefs, desires, and intentions; (2) existing ToM benchmarks adopt a "role-centric" perspective — inferring the mental states of video characters without involving the agent's own viewpoint or requiring the generation of decisions and actions; (3) VLMs at the perception level are susceptible to scene bias (e.g., predicting "cleaning" upon observing a kitchen rather than reasoning about actual intent).

**Root Cause**: An agent must understand "what others are thinking" in order to offer proactive assistance, yet must simultaneously reason from its own perspective — e.g., "I know the apple is actually in the refrigerator, even though Alice believes it is on the table." Neither existing benchmarks nor methods have established this dual-perspective reasoning loop.

**Paper Goals**: Enable embodied agents to infer human mental states (beliefs, desires, intentions) from their own perspective and thereby produce proactive decisions and actions.

**Starting Point**: Systematically introduce the cognitive-science BDI (Belief-Desire-Intention) framework into embodied agents, constructing a three-level six-layer continuous reasoning hierarchy, and optimize reasoning consistency through a structured reward function (Mind-Reward) via reinforcement learning.

**Core Idea**: Connect perception to action via a robot-centric BDI reasoning hierarchy of three levels and six layers, and optimize the consistency of the reasoning chain through GRPO using an atomic-action-matching Mind-Reward.

## Method

### Overall Architecture

MindPower consists of three components: (1) **MindPower Benchmark** — 590 household scenarios (VirtualHome + ThreeDWorld) with two tasks (false-belief correction and implicit goal inference); (2) **MindPower Reasoning Hierarchy** — a three-level six-layer reasoning structure; (3) **Mind-Reward + GRPO** — two-stage training (SFT cold-start + GRPO reinforcement). The base model is Qwen2.5-VL-7B.

### Key Designs

1. **MindPower Reasoning Hierarchy (Three-Level Six-Layer Structure)**:
    - **Function**: Formalizes the embodied decision-making process as a continuous reasoning chain from perception to action.
    - **Mechanism**:
        - Level-1 **Perception** `<Perception>`: Observes the environment and human behavior to answer "what is happening now."
        - Level-2 **Mind Reasoning**: `<Belief>` (infers both the agent's own and the human's beliefs, including second-order beliefs — "I think Alice believes the apple is on the table") → `<Desire>` (identifies the assistance goal — "what help does Alice need") → `<Intention>` (forms a specific action intention).
        - Level-3 **Decision and Action**: `<Decision>` (selects a plan) → `<Action>` (outputs an atomic operation sequence such as `walk(fridge), open(fridge), pick(apple)`).
    - **Design Motivation**: Existing VLMs rely on one-step decision-making without intermediate reasoning. The BDI hierarchy ensures that every decision is grounded in traceable belief-desire-intention support, improving interpretability and consistency.

2. **Robot-Centric Perspective (vs. Role-Centric)**:
    - **Function**: Requires the agent to simultaneously infer its own beliefs and the human's beliefs, forming a closed dual-perspective reasoning loop.
    - **Mechanism**: In the false-belief correction task — the agent observes that an object has been moved (Stage 2); when the human returns to search for it (Stage 3), the agent must reason that "Alice believes the apple is on the table (false belief)" + "I know the apple is actually in the refrigerator (agent's own belief)" → "I should retrieve the apple from the refrigerator for Alice."
    - **Design Motivation**: Existing benchmarks such as MuMA-ToM and MMToM-QA are limited to multiple-choice questions about character mental states and do not involve the agent's own perspective. Genuine collaboration requires the agent to simultaneously maintain mental models of itself and others.

3. **Mind-Reward (Atomic Action Matching Reward)**:
    - **Function**: Designs a structured reward function to drive GRPO optimization and ensure consistency across the reasoning chain from perception to action.
    - **Mechanism**: Atomic action sequences are extracted from each reasoning layer's output via an LLM (Qwen3-Max), and three alignment metrics are computed: atomic accuracy (ROUGE-1), local consistency (ROUGE-2), and global consistency (ROUGE-L). $R_{Mind} = \alpha_1 R_{atomic} + \alpha_2 R_{local} + \alpha_3 R_{global}$, supplemented by a Format-Reward to ensure structural completeness of the hierarchy.
    - **Design Motivation**: The reasoning layers are sequential — temporal and logical dependencies exist from perception to action. Process-level rewards better ensure the quality of intermediate reasoning steps than evaluating only the final output.

### Loss & Training

- Two-stage training: (1) SFT cold-start (5 epochs) to establish basic reasoning capability; (2) GRPO reinforcement (400 iterations, 8 generated samples per step) using Mind-Reward + Format-Reward.
- GRPO updates the policy via intra-group relative advantage: $A_i = (R_i - \text{mean}(\{R_j\})) / \text{std}(\{R_j\})$.
- Training is conducted on a single H800 GPU with Qwen2.5-VL-7B as the base model.

## Key Experimental Results

### Main Results

| Method | Decision (S) | Action SR | Action AC | BPC |
|------|-------------|-----------|-----------|-----|
| GPT-4o (image) | 34.35 | 1.82 | 2.91 | 8.05 |
| Gemini-2.5 Pro | 33.87 | 2.08 | 2.54 | 8.56 |
| Video-R1 (best open-source) | 30.33 | 1.43 | 1.72 | 6.45 |
| Qwen2.5-VL-7B (base) | 26.56 | 0.29 | 0.22 | 6.07 |
| **Ours (SFT+Mind-Reward)** | **47.12** | **11.75** | **15.40** | **8.87** |
| Human Baseline | 56.66 | 19.37 | 26.26 | 8.19 |

### Ablation Study

| Training Configuration | Action AC | Decision (S) | BPC |
|----------|-----------|-------------|-----|
| Qwen2.5-VL-7B (no training) | 0.22 | 26.56 | 6.07 |
| Mind-Reward only (no SFT) | 0.40 | - | - |
| SFT only (no RL) | 10.48 | 42.35 | 8.32 |
| **SFT + Mind-Reward** | **15.40** | **47.12** | **8.87** |

| Reasoning Strategy (GPT-4o) | Decision | Action AC |
|-------------------|----------|-----------|
| Direct output (no reasoning) | 33.11 | 0.82 |
| Standard CoT (`<think>`) | 29.46 | 0.90 |
| **MindPower Hierarchy** | **34.35** | **2.91** |

### Key Findings

- SFT alone yields substantial improvements (Action AC: 0.22→10.48), demonstrating the intrinsic effectiveness of the BDI reasoning hierarchy structure.
- RL further improves performance by approximately 5 points over SFT (10.48→15.40), but RL without SFT is nearly ineffective (0.40).
- The MindPower Hierarchy significantly outperforms standard CoT (Decision +4.89%) — structured BDI reasoning is more effective than generic "thinking."
- Open-source VLMs exhibit severe deficiency in robot-centric perspective and are easily misled by scene bias (e.g., kitchen → cleaning, bedroom → tidying).
- A significant gap remains relative to the human baseline (Decision: 47.12 vs. 56.66; Action: 15.40 vs. 26.26).

## Highlights & Insights

- The systematic introduction of the cognitive-science BDI framework into embodied agents produces an interpretable reasoning chain in which every decision is grounded in traceable belief support.
- The robot-centric perspective is the core innovation — the agent not only infers others' mental states but also explicitly models its own beliefs, enabling second-order reasoning.
- Mind-Reward decomposes reasoning quality into atomic-, local-, and global-level consistency evaluations, offering more controllability than black-box LLM scoring.
- The two task designs are insightful: false-belief correction (the agent detects that an object has been moved) and implicit goal inference (inferring needs from search behavior).

## Limitations & Future Work

- The dataset contains only 590 scenarios, all sourced from simulators (VirtualHome + ThreeDWorld), limiting scene diversity.
- The action space is coarse (high-level atomic operations such as `walk(fridge)`), with no coverage of low-level motion control.
- Mind-Reward depends on Qwen3-Max for atomic action extraction, introducing an additional LLM dependency.
- Whether automated open-ended evaluation metrics (BERTScore, ROUGE) truly reflect reasoning quality remains questionable.
- Evaluation is limited to the 7B model; performance at larger scales has not been verified.

## Related Work & Insights

- **MuMA-ToM / MMToM-QA**: Limited to multiple-choice inference of character mental states; MindPower requires complete BDI reasoning from the agent's own perspective together with action generation.
- **Smart-Help / AToM-Bot**: Address human-robot collaborative assistance but lack explicit mental reasoning; MindPower explicitly models the detection and correction of belief inconsistencies.
- **Video-R1 / VideoChat-R1**: Apply RL training for video understanding but do not involve ToM reasoning or embodied decision-making.
- **Insights**: The BDI reasoning hierarchy can be generalized as a "structured CoT" for other tasks requiring inference of others' intentions; the process decomposition and atomic matching approach of Mind-Reward provides a valuable reference for designing other process-level rewards.

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: Robot-centric ToM combined with the BDI reasoning hierarchy represents an entirely new perspective and a cross-disciplinary innovation at the intersection of cognitive science and AI.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Comparisons against multiple closed-source and open-source VLMs, a human baseline, and detailed ablations are provided, though the dataset scale is limited.
- ⭐⭐⭐⭐ **Writing Quality**: Concepts are clearly articulated with well-organized structure; the three-level six-layer formalization is easy to follow.
- ⭐⭐⭐⭐ **Value**: Endowing embodied agents with ToM capability is an important research direction; practical deployment remains distant but the direction is well-defined.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)

<!-- RELATED:END -->
