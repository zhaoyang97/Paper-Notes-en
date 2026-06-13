---
title: >-
  [Paper Note] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
description: >-
  [NeurIPS 2025 (Datasets and Benchmarks Track)][Robotics][Multi-agent cooperation] This paper introduces VIKI-Bench, the first hierarchical benchmark for embodied multi-agent cooperation…
tags:
  - "NeurIPS 2025 (Datasets and Benchmarks Track)"
  - "Robotics"
  - "Multi-agent cooperation"
  - "VIKI-Bench"
  - "VLM fine-tuning"
  - "Chain-of-Thought"
  - "multi-level rewards"
  - "heterogeneous robots"
date: 2026-05-08
content_hash: 618be097e57747f8
---

# VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning

**Conference**: NeurIPS 2025 (Datasets and Benchmarks Track)  
**arXiv**: [2506.09049](https://arxiv.org/abs/2506.09049)  
**Code**: Available (Project Page)  
**Area**: Reinforcement Learning / Embodied Multi-Agent Systems  
**Keywords**: Multi-agent cooperation, VIKI-Bench, VLM fine-tuning, Chain-of-Thought, multi-level rewards, heterogeneous robots

## TL;DR
This paper introduces VIKI-Bench, the first hierarchical benchmark for embodied multi-agent cooperation, comprising three evaluation levels—agent activation, task planning, and trajectory perception—and proposes VIKI-R, a two-stage training framework combining CoT-supervised fine-tuning with multi-level reward reinforcement learning. The framework achieves significant improvements over baselines across diverse robot morphologies and multi-view visual observations, with combinatorial coordination patterns emerging during the RL stage.

## Background & Motivation

**Background**: Coordinating multiple embodied agents to accomplish cooperative tasks in dynamic environments is a core challenge in AI. Recent work has leveraged LLMs for multi-agent planning, and a growing body of research has begun exploring VLMs for visually grounded multi-agent cooperation.

**Limitations of Prior Work**: (1) Existing VLM-based multi-agent approaches support only a single robot morphology, lacking support for heterogeneous agents—robots of different types with distinct capabilities. (2) No systematic evaluation benchmark exists: current benchmarks either omit multi-agent settings, visual reasoning, or heterogeneous morphologies. (3) Pure LLM methods lack grounded visual perception, while pure RL methods lack high-level semantic reasoning.

**Key Challenge**: Effective multi-agent cooperation simultaneously requires visual perception (understanding the current scene), semantic reasoning (planning tasks), and cooperative strategy (role assignment and conflict avoidance). Existing methods typically address only one or two of these aspects.

**Goal**: (1) Construct a structured benchmark that systematically evaluates the full pipeline of multi-agent cooperation, from perception to planning to execution. (2) Develop a unified framework that leverages VLMs' visual reasoning capabilities and further optimizes cooperative strategies via RL.

**Key Insight**: The multi-agent cooperation problem is decomposed into three hierarchical levels—activation, planning, and perception—each independently evaluable, together forming a complete pyramid of cooperative capabilities.

**Core Idea**: Address visually grounded cooperation among heterogeneous embodied agents through a hierarchical benchmark and a two-stage VLM-RL framework that first learns structured reasoning formats and then optimizes cooperative strategies via RL.

## Method

### Overall Architecture
VIKI-Bench defines three evaluation levels: **Level 1 – Agent Activation** (determining which agents should participate in a given task), **Level 2 – Task Planning** (generating cooperative action plans from visual observations), and **Level 3 – Trajectory Perception** (understanding agent execution trajectories and states from multi-view observations). Each level incorporates diverse robot morphologies (ground, aerial, manipulation), multi-view visual inputs, and structured supervision signals.

VIKI-R trains in two stages: **Stage 1 – CoT Supervised Fine-Tuning** (fine-tuning a pretrained VLM on Chain-of-Thought annotated demonstration data to learn reasoning formats and basic cooperative strategies); **Stage 2 – Multi-Level Reward RL** (further optimizing the VLM via reinforcement learning with multi-level reward signals to acquire superior cooperative behaviors).

### Key Designs

1. **Hierarchical Evaluation Framework (VIKI-Bench)**:

    - **Function**: Provides full-pipeline evaluation of cooperative capabilities, from low-level perception to high-level planning.
    - **Mechanism**: Level 1 tests agent selection (given a task description and a list of available agents, output the subset to activate); Level 2 tests task allocation and action sequence generation (given a visual scene and task objective, output an action plan for each agent); Level 3 tests comprehension of execution processes (given multi-view video, answer questions about agent behaviors and states).
    - **Design Motivation**: Decomposing complex multi-agent cooperation into independently evaluable sub-capabilities avoids the bottleneck-localization problem inherent in end-to-end evaluation. Supporting heterogeneous morphologies and multi-view inputs makes the benchmark more representative of real-world scenarios than existing alternatives.

2. **CoT-Annotated Demonstration Fine-Tuning (Stage 1)**:

    - **Function**: Enables the VLM to learn structured cooperative reasoning patterns.
    - **Mechanism**: A large collection of multi-agent cooperation demonstrations is gathered, each annotated with Chain-of-Thought reasoning (scene analysis → sub-task identification → agent capability assessment → task assignment → action sequence generation). The pretrained VLM is then fine-tuned on these annotated demonstrations via supervised learning.
    - **Design Motivation**: Although pretrained VLMs possess visual understanding and reasoning capabilities, they lack the structured output format required for multi-agent cooperation. CoT fine-tuning simultaneously teaches the model the appropriate output format and conveys foundational cooperative knowledge, such as capability matching and task decomposition.

3. **Multi-Level Reward Reinforcement Learning (Stage 2)**:

    - **Function**: Further optimizes cooperative strategies beyond the CoT fine-tuning stage via RL.
    - **Mechanism**: Multi-level reward signals are designed in correspondence with the three evaluation levels—correctness reward for agent activation, soundness reward for task planning, and accuracy reward for trajectory prediction. An RL algorithm jointly optimizes VLM performance across all levels.
    - **Design Motivation**: CoT fine-tuning is constrained by the coverage of demonstration data, whereas RL enables the model to explore cooperative strategies not present in demonstrations. Multi-level rewards prevent the model from optimizing one level at the expense of others. Experiments show that combinatorial coordination patterns emerge during RL that are absent from CoT demonstrations.

### Loss & Training
Stage 1 employs standard sequence-to-sequence cross-entropy loss for fine-tuning on CoT-annotated data. Stage 2 uses RL training driven by multi-level reward signals, where rewards for each level are computed independently and combined as a weighted sum.

## Key Experimental Results

### Main Results

| Method | Level 1 Activation | Level 2 Planning | Level 3 Perception | Overall |
|--------|-------------------|-----------------|-------------------|---------|
| GPT-4V (zero-shot) | Baseline | Baseline | Baseline | Baseline |
| LLaVA | Weak | Weak | Weak | Below GPT-4V |
| VIKI-R (CoT only) | Significant gain | Significant gain | Significant gain | Large margin over baseline |
| **VIKI-R (CoT+RL)** | **Best** | **Best** | **Best** | **Best across all levels** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full VIKI-R (CoT+RL) | Best | Complete two-stage framework |
| CoT fine-tuning only | Second best | Lacks RL-driven strategy exploration |
| RL only (no CoT pre-training) | Poor | Lacks structured reasoning prior |
| Single-level reward RL | Partial degradation | Optimizing one level degrades others |

### Key Findings
- **Both stages are indispensable**: CoT fine-tuning provides the reasoning foundation but yields limited strategy diversity; RL alone fails to converge to good strategies due to the absence of a structured output prior.
- **RL elicits emergent combinatorial coordination**: After RL training, the model exhibits cooperative patterns absent from CoT demonstrations—such as dynamic role switching among heterogeneous agents and pipeline-parallel execution of multi-step tasks—which supervised learning alone cannot produce.
- **Heterogeneous morphologies pose a critical challenge**: Ground, aerial, and manipulation robots have distinct action spaces and capability boundaries; the model must understand the characteristics of each morphology to allocate tasks effectively.

## Highlights & Insights
- **The hierarchical evaluation design** is notably systematic: decomposing the vague notion of "multi-agent cooperative capability" into three quantifiable levels—activation, planning, and perception—facilitates targeted diagnosis and provides clear research targets. This decomposition strategy is transferable to benchmark design for other complex AI tasks.
- **The two-stage CoT-to-RL training paradigm** elegantly combines the stability of supervised learning with the exploratory power of RL: CoT fine-tuning instills output format and basic cooperative strategies, while RL breaks through the ceiling imposed by demonstration data. This establishes a general "imitate then surpass" training paradigm.
- **The emergent coordination during RL** is the most compelling finding: the emergence of combinatorial cooperative patterns demonstrates that RL combined with VLMs not only learns behaviors explicitly rewarded by the signal, but also develops implicit coordination capabilities.

## Limitations & Future Work
- As a Datasets and Benchmarks Track paper, the technical methodology is relatively standard—CoT SFT followed by RL is an established paradigm—with the primary contribution lying in benchmark construction.
- The task scenarios in the benchmark may be limited in scope; real-world multi-agent cooperation involves greater uncertainty, including fault recovery and dynamic environmental changes.
- In-depth comparisons with non-VLM approaches (e.g., conventional MARL methods) are absent, making it difficult to quantify the advantage conferred by VLM-based visual reasoning.
- Computational costs are not discussed in detail; the expense of RL training on large-scale VLMs may limit practical applicability.

## Related Work & Insights
- **vs. LLM-based multi-agent systems (e.g., CAMEL, AutoGen)**: These methods rely on text-only LLMs for planning and lack visual perception. VIKI-R integrates visual inputs into the decision loop via VLMs.
- **vs. conventional MARL**: Traditional approaches require extensive environment interactions and manual reward engineering; VIKI-R leverages pretrained VLM knowledge and CoT reasoning to substantially reduce interaction requirements.
- **vs. environments such as RoboTHOR/Habitat**: These are single-agent simulation environments; VIKI-Bench is specifically designed for multi-agent cooperation with hierarchical evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First hierarchical visual reasoning benchmark targeting heterogeneous embodied multi-agent cooperation, with a strong benchmark contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-level, multi-baseline comparisons and ablation analyses are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the hierarchical design is easy to follow.
- Value: ⭐⭐⭐⭐ Provides a standardized evaluation platform and baseline method for embodied multi-agent cooperation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](../../ICLR2026/robotics/distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
