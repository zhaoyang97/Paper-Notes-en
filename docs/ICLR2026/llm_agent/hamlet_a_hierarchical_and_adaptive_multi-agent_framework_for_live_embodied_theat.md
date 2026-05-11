---
title: >-
  [Paper Note] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre
description: >-
  [ICLR 2026][LLM Agent][Multi-agent framework] This paper proposes HAMLET, a multi-agent framework that decouples AI theatrical creation and live performance into an offline planning phase and an online performance phase.…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Multi-agent framework"
  - "theatrical performance"
  - "perception and decision-making"
  - "interactive narrative"
date: 2026-05-08
content_hash: 12e995cae489924f
---

# HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre

**Conference**: ICLR 2026
**arXiv**: [2507.15518](https://arxiv.org/abs/2507.15518)
**Code**: [https://github.com/HAMLET-2025/HAMLET](https://github.com/HAMLET-2025/HAMLET)
**Area**: LLM Agent / Interactive Narrative
**Keywords**: Multi-agent framework, theatrical performance, LLM Agent, perception and decision-making, interactive narrative

## TL;DR

This paper proposes HAMLET, a multi-agent framework that decouples AI theatrical creation and live performance into an offline planning phase and an online performance phase. Through a narrative blueprint, a Perceive And Decide (PAD) module, and a hierarchical control system, HAMLET enables an AI theatre experience characterized by proactivity, physical environment interaction, and improvisational freedom.

## Background & Motivation

### State of the Field
Creating immersive interactive theatrical experiences has been a long-standing goal in interactive narrative research. The emergence of LLMs has opened new pathways toward this goal; however, existing LLM-driven theatre generation methods suffer from three critical limitations:

**Lack of proactivity**: AI agents typically wait passively for instructions and cannot make independent decisions.

**Inability to interact with the physical environment**: Character actions do not affect the stage environment, reducing drama to abstract dialogue.

**Dependence on detailed user input**: Methods require complete story outlines or detailed guiding passages, limiting flexibility.

### Root Cause
The core challenge lies in a paradigm shift from passive response to active narrative guidance — AI actors must be capable of autonomous decision-making, cooperation or conflict in open-ended scenarios, and proactively driving plot development. This represents the concrete instantiation of Agentic AI principles within theatrical performance.

## Method

### Overall Architecture

The HAMLET framework is decoupled into two main phases:

**Phase 1: Offline Planning**
- Input: an arbitrary theme or a complete literary work
- A multi-agent collaborative workflow generates a structured Narrative Blueprint

**Phase 2: Online Performance**
- Input: the Narrative Blueprint
- A hierarchical control system executes the blueprint, manages real-time interactions, and processes environmental feedback

### Key Designs

1. **Offline Planning: Four-Agent Collaborative Workflow**

    - **Actor Designer**: Generates character profiles from user input, including static attributes (background, personality) and dynamic attributes (initial goals, core relationships); can query external knowledge bases.
    - **Plot Designer**: Drafts a preliminary narrative based on the theme and characters.
    - **Reviewer**: Evaluates the plausibility of character settings, clarity of motivations, and inter-character relationships.
    - **Director**: Responsible for final structured processing, performing the following key steps:
        - Dividing the narrative into Acts and Scenes
        - Creating environment interaction elements (prop lists per scene)
        - Defining Narrative Points — each with explicit completion flags and outcomes
        - **Reverse planning**: generating the ending node first, then constructing preceding nodes in reverse order to prevent plot drift

2. **Online Performance: Beat-Driven Improvisation System**

   Performance unit hierarchy: **Act** → **Scene + Narrative Point** → **Beat**

    - A Beat represents one valid interaction step (a character taking a valid action).
    - Character decisions are driven by a **dual-goal system**: public flags for the current Narrative Point and private individual goals.
    - **Multiple trajectories** are permitted between two Narrative Points, affording a high degree of improvisational freedom.

3. **PAD Module (Perceive And Decide)**

   Designed on the basis of Kahneman's dual-process theory, integrating intuitive and reflective reasoning:

   **Dual-perspective input**:
    - Internal state (subjective): Persona + subjective relationships + Memory + Goal
    - External stimuli (objective): environment description + character list + dialogue history + interactable objects

   **Decision outputs**:
    - FAST: rapid intuitive response (System 1)
    - SLOW: deliberate analytical response (System 2)
    - SILENCE: silence / inaction
    - Potential actions (structured action triples generated via tool calls: subject–verb–object)

   The PAD module is an 8B fine-tuned model capable of simulating the dual-system structure of human cognition.

4. **Environment Interaction: Narrator Arbitration System**

   A Narrator Agent is designed to arbitrate all physical interactions:
    - When a character attempts a physical action, the Narrator assesses feasibility based on environmental state and physical rules.
    - Success: confirms the action, updates the environmental state, and broadcasts an objective description.
    - Failure: rules the action as failed and provides a logical explanation.

5. **Hierarchical Control: Three-Agent System**

    - **Planner**: Pre-designs multi-trajectory plans and decomposes narrative flags into executable Beat sequences.
    - **Transfer**: Periodically checks whether Narrative Point flags are satisfied, advances to the next point, and manages character entrances and exits.
    - **Advancer**: If the plot stalls beyond a time threshold, guides relevant characters to advance the narrative.

### Evaluation Framework

Three-dimensional evaluation:
- **Character Performance**: character consistency, emotional expression
- **Narrative Quality**: plot coherence, structural integrity
- **Interaction Experience**: naturalness of environmental interaction, sense of immersion

A HAMLETJudge (8B critic model) is trained, with GPT-4o used as a baseline for win-rate comparison.

## Key Experimental Results

### Main Results: Multi-Model Evaluation Leaderboard

| Model | Avg. Score (EN) | Avg. Score (ZH) | Overall |
|-------|----------------|----------------|---------|
| Claude-4-sonnet-Thinking | 78.98 | 79.92 | **79.45** |
| Claude-4-sonnet | 76.92 | 79.68 | 78.30 |
| Qwen3-32B-Thinking | 69.10 | 78.59 | 73.85 |
| OpenAI-o3 | 69.45 | 77.89 | 73.67 |
| Qwen3-235B-A22B-Thinking | 70.74 | 75.92 | 73.33 |
| DeepSeek-R1-0528 | 66.58 | 79.37 | 72.98 |
| Qwen3-235B-A22B | 69.65 | 72.76 | 71.21 |
| Llama-3.1-8B | 35.51 | 33.83 | 34.67 |

### Dataset Composition

| Source | Count | Description |
|--------|-------|-------------|
| Chinese literary classics | 25 | Literary excerpts |
| English literary classics | 25 | Literary excerpts |
| Custom themes | 50 | Spanning 10 distinct themes |
| **Total** | **100 cases** | |

### Key Findings
- Reasoning-oriented models (e.g., Claude-4-sonnet-Thinking) achieve the best overall performance, though the advantage is less pronounced than expected.
- Chinese-language performances consistently outperform English-language ones, possibly because Chinese literary works align more closely with the framework's design.
- Small models (e.g., Llama-3.1-8B) perform significantly worse on theatrical tasks.
- The PAD module (8B) achieves state-of-the-art performance on decision-making tasks.
- HAMLETJudge (8B) provides a cost-effective and reliable evaluation alternative.

## Highlights & Insights

1. **End-to-end AI theatre pipeline**: A complete framework from thematic input to real-time live performance, filling a systemic gap in the AI theatre domain.
2. **Cognitive-theoretic foundation of the PAD module**: Integrating System 1 and System 2 into AI actor decision-making based on Kahneman's dual-process theory produces more human-like responses.
3. **Reverse planning strategy**: The Director first determines the ending and then constructs preceding plot points in reverse, effectively preventing narrative drift — a clever design in the context of interactive storytelling.
4. **Beat-driven multi-trajectory improvisation**: Allowing multiple trajectories between two Narrative Points strikes a balance between structured narrative and free improvisation.
5. **Physical environment interaction**: The Narrator arbitration system moves the drama beyond pure dialogue, enhancing embodiment and immersion.

## Limitations & Future Work

- Evaluation relies primarily on LLM-as-Judge (GPT-4o and HAMLETJudge), with no large-scale human evaluation.
- The evaluation dataset of 100 cases is limited in scale.
- The current framework supports text-based theatre only, without addressing multimodal dimensions (speech, vision, motion capture).
- The PAD module is an 8B fine-tuned model that may introduce latency in real-time performance settings.
- Interactive experiences involving human players are not sufficiently evaluated in the paper.
- Maintaining consistency in long-form (multi-act) performances may pose challenges due to long-context limitations.

## Related Work & Insights

- **Dramatron** (Mirowski et al., 2023): Employs a hierarchical approach separating planning from generation, but does not support real-time performance.
- **CoSER** (Wang et al., 2025): Scales up the number of characters but lacks holistic theatrical performance evaluation.
- **CharacterEval** (Tu et al., 2024): Provides multi-turn, multi-dimensional dialogue scoring, but is limited to two-character scenarios.
- **Kahneman's dual-process theory**: The cognitive science foundation underlying the PAD module.
- Insight: The balanced design of hierarchical control and improvisational freedom holds reference value for other agent systems, including game NPCs and virtual assistants.

## Rating
- Novelty: ⭐⭐⭐⭐ — Comprehensive and novel framework design; the PAD module is grounded in cognitive theory.
- Experimental Thoroughness: ⭐⭐⭐ — The leaderboard evaluation is valuable but lacks human evaluation and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear and well-illustrated, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ — Makes a significant contribution to interactive narrative and AI theatre research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](../../ACL2026/llm_agent/hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](../../CVPR2026/llm_agent/nerfify_multiagent_nerf_paper_to_code.md)
- [\[ACL 2026\] MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](../../ACL2026/llm_agent/mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)

</div>

<!-- RELATED:END -->
