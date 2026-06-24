---
title: >-
  [Paper Note] See and Think: Embodied Agent in Virtual Environment
description: >-
  [ECCV 2024][Robotics][Embodied Agent] This paper proposes STEVE, an open-world embodied agent in Minecraft based on three major components: visual perception, language instruction, and code action. By fine-tuning LLaMA-2 on the STEVE-21K dataset and combining it with a visual encoder and a skill database, STEVE significantly outperforms existing methods in tech tree unlocking and block search tasks.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Embodied Agent"
  - "Minecraft"
  - "Multi-modal LLM"
  - "Open-world"
  - "Skill Retrieval"
date: 2026-05-08
content_hash: c9c2d364eae6255c
---

# See and Think: Embodied Agent in Virtual Environment

**Conference**: ECCV 2024  
**arXiv**: [2311.15209](https://arxiv.org/abs/2311.15209)  
**Code**: None  
**Area**: Robotics  
**Keywords**: Embodied Agent, Minecraft, Multi-modal LLM, Open-world, Skill Retrieval

## TL;DR

This paper proposes STEVE, an open-world embodied agent in Minecraft based on three major components: visual perception, language instruction, and code action. By fine-tuning LLaMA-2 on the STEVE-21K dataset and combining it with a visual encoder and a skill database, STEVE significantly outperforms existing methods in tech tree unlocking and block search tasks.

## Background & Motivation

**Background**: Open-world embodied agent research (e.g., in the Minecraft environment) has become an important testing ground for AI. Recently, LLM-driven agents (such as Voyager, DEPS) have demonstrated powerful planning capabilities.

**Limitations of Prior Work**: Current LLM-driven Minecraft agents primarily rely on text interaction, lacking visual perception capabilities. They tend to generate unpredictable outputs, relying heavily on carefully crafted prompt engineering; furthermore, text cannot naturally convey visual information such as synthesis recipes.

**Key Challenge**: There is a need for a multimodal framework that can process visual inputs and textual reasoning simultaneously, and translate high-level planning into executable code.

**Goal**: To build a comprehensive Minecraft embodied agent that combines vision, language, and code.

**Key Insight**: Decompose the problem into three modules—visual perception (what to see), language instruction (how to plan), and code action (how to execute)—and construct a dedicated dataset, STEVE-21K, to support training.

**Core Idea**: Build a complete open-world embodied agent by perceiving the environment through a visual encoder, reasoning and decomposing tasks with an LLM fine-tuned for the Minecraft domain, and retrieving execution code from a skill database.

## Method

### Overall Architecture

STEVE is an LLM-based multimodal autonomous system that receives the visual state $X^v$, agent state $X^s$, and task description $X^t$, and outputs executable code actions $\mathbf{a}^c$. The overall formulation is: $\mathbf{a}^c = \mathcal{F}(X^v, X^s, X^t) = \mathcal{A}^c(\mathcal{I}^l(\mathcal{P}^v(X^v, X^s, X^t)))$.

### Key Designs

1. **Vision Perception 视觉感知 $\mathcal{P}^v$**

    - **Function**: Encodes the visual state (image/video), agent state (HP/inventory), and task description into a unified token representation.
    - **Mechanism**: EfficientFormerV2-S0 is used as the visual encoder to encode images into $n$ visual tokens of dimension $d$, which are concatenated with the state and task tokens processed by the text tokenizer.
    - **Design Motivation**: The textual context in Minecraft is insufficient to convey the visual characteristics of blocks and entities, necessitating direct visual perception.

2. **Language Instruction 语言指令 $\mathcal{I}^l$**

    - **Function**: Responsible for iterative reasoning and decomposing complex tasks into manageable steps.
    - **Mechanism**: Comprises four independent LLM sub-modules: Planner (high-level planning), Critic (evaluation and feedback), Curriculum (progressive learning), and Describer (information summarization). These are based on STEVE-7B/13B (fine-tuned from LLaMA-2) and possess Minecraft domain expertise.
    - **Design Motivation**: A single LLM struggles to perform planning, evaluation, and learning simultaneously; collaborative role-playing is more effective.

3. **Code Action 代码动作 $\mathcal{A}^c$**

    - **Function**: Translates language instructions into executable code in Minecraft.
    - **Mechanism**: Based on skill database retrieval. Instructions are encoded into query vectors, which are matched with skill-code pairs in the database using cosine similarity, containing 210 skills across 8 categories.
    - **Design Motivation**: Code execution is more reliable than direct control, and skill database retrieval is more stable than generating code directly via LLMs.

4. **Curriculum Learning with Memory（带记忆的课程学习）**

    - **Function**: Progressively learns tasks from simple to complex, accumulating experience into memory.
    - **Mechanism**: Performs curriculum tasks first to allow agent exploration, storing successful experiences. Uses the Chain of Summarization method to compress overly long memories, enabling gradient-free in-context lifelong learning.
    - **Design Motivation**: Open-world tasks require progressive learning, and memory can reuse experiences to improve efficiency.

### Loss & Training

- **Two-Stage Training**:
    - Stage 1 (Offline Warm-up): Fine-tunes LLaMA-2 using LoRA on the STEVE-21K dataset containing 20K single-round QA pairs.
    - Stage 2 (Online Fine-tuning): Simultaneously trains the visual encoder and fine-tunes the LLM under simulation, utilizing instructions generated by an Expert LLM (GPT-4) as ground truth.
- **Visual Encoder Training**: Class/block/entity labels are retrieved within the FOV using Ray Tracing, and context data from successful runs are collected after 5,000 simulations.
- **Loss Function**: Negative log-likelihood objective: $\mathcal{L}(\theta) = -\sum_{j=1}^{L} \log \mathcal{F}_\theta(Y_j | X^v, \hat{Y}_{1:j-1})$

## Key Experimental Results

### Main Results

Tech Tree Unlocking Task Comparison (lower iteration count is better, 3/3 indicates full success in three runs):

| Method | Wooden Tool | Stone Tool | Iron Tool | Diamond Tool |
|------|------------|------------|-----------|-------------|
| AutoGPT | 92±72 (3/3) | 94±72 (3/3) | 135±103 (3/3) | N/A (0/3) |
| Voyager | 6±2 (3/3) | 11±2 (3/3) | 21±7 (3/3) | 102 (1/3) |
| **STEVE** | **4±1 (3/3)** | **8±1 (3/3)** | **15±2 (3/3)** | **106±12 (3/3)** |

Continuous Block Search Task:

| Method | Avg. Iterations↓ | Avg. Diamonds Found↑ |
|------|-------------|---------------|
| AutoGPT | N/A | 7 |
| Voyager | 35 | 26 |
| **STEVE** | **14** | **67** |

### Ablation Study

Ablation study (tech tree task):

| Method | Wooden Tool | Stone Tool | Iron Tool | Diamond Tool |
|------|------------|------------|-----------|-------------|
| w/o vision unit | 11±5 (3/3) | 27±5 (3/3) | 46±11 (3/3) | 158 (1/3) |
| STEVE (GPT-4) | 6±2 (3/3) | 10±1 (3/3) | 14±3 (3/3) | 89±9 (3/3) |
| STEVE (Ours-13B) | 4±1 (3/3) | 8±1 (3/3) | 15±2 (3/3) | 106±12 (3/3) |

### Key Findings

- STEVE is faster than all methods on simple tasks (wooden/stone tools), including the GPT-4 based version.
- The visual unit is critical for complex tasks like the Diamond Tool (the success rate drops from 3/3 to 1/3 upon removal).
- STEVE-13B scores 8.12 in QA, outperforming GPT-4's 8.04, which demonstrates the value of domain-specific fine-tuning.
- Visual perception increases search efficiency by 2.5x compared to Voyager.

## Highlights & Insights

- **Complete Trinity Framework**: The see-think-act paradigm design is clear and natural, with a well-defined division of labor among modules.
- **Dataset Construction**: STEVE-21K contains vision-environment pairs, QA pairs, and skill-code pairs, providing a comprehensive resource for Minecraft AI training.
- **Smaller Models Outperforming Larger Models**: Domain-fine-tuned STEVE-13B outperforms the general-purpose GPT-4 in Minecraft knowledge.
- **Curriculum Learning + Memory**: The gradient-free, in-context lifelong learning paradigm is highly practical.

## Limitations & Future Work

- Evaluated only in the Minecraft environment; generalizability to other open-world scenarios remains unverified.
- The skill database is manual-crafted with 210 skills, which limits its scalability.
- The visual encoder is relatively simple (EfficientFormerV2-S0); stronger vision concept models were not utilized.
- It is still less efficient than the Voyager + GPT-4 combination on the Diamond Tool task.
- Relies on GPT-4 to generate ground truth for online fine-tuning, which incurs high costs.

## Related Work & Insights

- Similar positioning to Voyager but emphasizes multimodal inputs, filling the visual perception blank in Minecraft AI.
- The skill retrieval concept in Code Action is similar to tool learning in HuggingGPT, but more tailored to Minecraft.
- Insights from DEPS (multi-step reasoning) and GITM (structured actions) are both reflected in STEVE.
- Insight: The combination of domain-specific fine-tuning, visual perception, and structured execution is an effective paradigm for building embodied agents.

## Rating

- ⭐⭐⭐ Novelty: The framework is an integration of existing components; technical novelty of individual modules is limited.
- ⭐⭐⭐ Experimental Thoroughness: The number of baselines is small (only AutoGPT and Voyager), and more ablation studies are needed.
- ⭐⭐⭐ Writing Quality: Generally readable, but mathematical notations are somewhat redundant.
- ⭐⭐⭐⭐ Value: The STEVE-21K dataset and the complete framework design are valuable resources for the Minecraft AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[NeurIPS 2025\] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](../../NeurIPS2025/robotics/viki-r_coordinating_embodied_multi-agent_cooperation_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](../../NeurIPS2025/robotics/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[ECCV 2024\] ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments](realfred_an_embodied_instruction_following_benchmark_in_photo-realistic_environm.md)
- [\[CVPR 2025\] Hearing Anywhere in Any Environment](../../CVPR2025/robotics/hearing_anywhere_in_any_environment.md)

</div>

<!-- RELATED:END -->
