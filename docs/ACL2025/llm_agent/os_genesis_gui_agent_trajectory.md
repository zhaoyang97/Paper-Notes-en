---
title: >-
  [Paper Note] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
description: >-
  [ACL 2025][LLM Agent][GUI Agent] This work proposes OS-Genesis, an interaction-driven GUI Agent trajectory synthesis pipeline. By allowing the agent to explore and interact with the environment first, followed by deriving tasks in reverse (Reverse Task Synthesis), and combined with a Trajectory Reward Model (TRM) for quality filtering, it generates high-quality, diverse training trajectories, nearly doubling the performance on AndroidWorld.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Trajectory Synthesis"
  - "Reverse Task Synthesis"
  - "VLM"
  - "Reward Model"
date: 2026-05-08
content_hash: 13ec842d774f1a66
---

# OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis

**Conference**: ACL 2025  
**arXiv**: [2412.19723](https://arxiv.org/abs/2412.19723)  
**Code**: [OS-Genesis Homepage](https://os-genesis.github.io/)  
**Area**: GUI Agent / Data Synthesis  
**Keywords**: GUI Agent, Trajectory Synthesis, Reverse Task Synthesis, VLM, Reward Model  

## TL;DR

This work proposes OS-Genesis, an interaction-driven GUI Agent trajectory synthesis pipeline. By allowing the agent to explore and interact with the environment first, followed by deriving tasks in reverse (Reverse Task Synthesis), and combined with a Trajectory Reward Model (TRM) for quality filtering, it generates high-quality, diverse training trajectories, nearly doubling the performance on AndroidWorld.

## Background & Motivation

**Key Challenge:** VLM-based GUI agents require high-quality trajectory data for training. However, existing data collection methods face severe bottlenecks: manual annotation is expensive and inefficient, while task-driven synthesis from pre-defined tasks is limited in data diversity and quality.

**Limitations of Prior Work:** (1) Manual collection requires annotators to log complete trajectories and manually pre-define high-level tasks, which is high-cost and limited in scale; (2) task-driven model synthesis heavily relies on pre-defined high-level tasks, restricting scalability and diversity; (3) intermediate step errors or task goal mismatches lead to incomplete or semantically incoherent synthetic trajectories.

**Core Idea:** Simulating how humans learn GUI interaction—exploring application capabilities first (interaction-driven) and then reversely synthesizing meaningful tasks from the executed operations. This paradigm naturally bridges the gap between abstract instructions and the dynamic characteristics of GUIs.

## Method

### Overall Architecture

OS-Genesis consists of three core phases: (1) **Interaction-Driven Capability Discovery**—traversing UI elements in an online environment without human intervention; (2) **Reverse Task Synthesis**—deriving low-level instructions from collected interaction triplets, then synthesizing them into high-level tasks; (3) **Trajectory Reward Model (TRM)**—conducting graded quality assessment and weighted sampling of synthetic trajectories for training.

### Key Designs

1. **Interaction-Driven Capability Discovery:** Realizes rule-based UI element traversal (CLICK, TYPE, SCROLL) in Android emulators and Chrome browsers. GPT-4o is leveraged to generate contextually appropriate inputs only during input box interactions. A large number of triplets $\langle s_{pre}, a, s_{post} \rangle$ (screenshots before and after interaction + executed action) are collected.

2. **Reverse Task Synthesis:** A two-level generation process. (a) Low-level: GPT-4o is employed to derive atomic operation descriptions $\tau_{low}$ (e.g., "click the dropdown menu to display options") from each triplet; (b) High-level: Low-level tasks are mapped to broader user intentions $\tau_{high}$ (e.g., "regularize app settings"). These high-level instructions are then used to drive GPT-4o execution in the environment to collect complete trajectories.

3. **Trajectory Reward Model (TRM):** Unlike traditional binary labeler functions (keep/discard), TRM outputs a graded reward (1-5) for each trajectory, evaluating both completion and coherence. During training, trajectories are sampled with probability $P(g_i) = R_i / \sum_{k=1}^N R_k$, permitting incomplete but partially valuable trajectories to contribute to training.

### Loss & Training

Two complementary SFT objectives:
- **Planning Training:** $\mathcal{L}_1 = -\sum \log(p_\theta(\ell | s, h_i, c) \cdot p_\theta(a | s, h_i, c, \ell))$, predicting both low-level instructions and actions.
- **Action Training:** $\mathcal{L}_2 = -\sum \log p_\theta(a | s, c, \ell)$, predicting the action given the low-level instruction.

## Key Experimental Results

### Main Results (AndroidWorld Success Rate)

| Base Model | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|---------|-----------|-------------|---------------|----------------|
| GPT-4o (M3A) | 23.70 | — | — | — |
| InternVL2-4B | 0.00 | 4.02 | 7.14 | **15.18** |
| InternVL2-8B | 2.23 | 4.46 | 5.36 | **16.96** |
| Qwen2-VL-7B | 0.89 | 6.25 | 9.82 | **17.41** |

### WebArena Success Rate

| Base Model | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|---------|-----------|-------------|---------------|----------------|
| InternVL2-4B | 0.00 | 4.98 | 5.81 | **7.88** |
| InternVL2-8B | 0.00 | 4.56 | 7.05 | **9.96** |
| Qwen2-VL-7B | 7.47 | 7.05 | 5.39 | **10.79** |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Data Diversity | OS-Genesis achieves the highest cosine distance in both instruction and trajectory dimensions, outperforming human data in trajectory diversity. |
| TRM vs Labeler | Graded TRM sampling outperforms binary labeler filtering, particularly for high-level planning tasks. |
| Data Scale | Performance scales with the volume of data, initiating saturation around 1K trajectories. |
| Comparison with Human Data | Instruction quality from OS-Genesis is even superior to human-written designs (as predefined tasks might mismatch dynamic environments). |

### Key Findings

- OS-Genesis boosts the success rate of Qwen2-VL-7B on AndroidWorld from 9.82% to 17.41%, nearly doubling execution performance while using only 1K trajectories (compared to 1.5K for Self-Instruct).
- Under OOD evaluations on AndroidControl (where only 20 out of 833 apps were included in synthetic training data), OS-Genesis demonstrates strong generalization capabilities.
- While human-written instructions exhibit high diversity, their corresponding trajectory diversity is low, as humans tend to reuse familiar operational paths. OS-Genesis achieves high diversity across both dimensions.

## Highlights & Insights

- Shifts the paradigm of GUI trajectory construction from "task-driven" to "interaction-driven", significantly enhancing data diversity and quality.
- The intuition of reverse task synthesis is clear and elegant—exploring first and reverse-engineering tasks after, which naturally aligns with the dynamic nature of GUI environments.
- Graded evaluation in TRM avoids data waste caused by simply discarding incomplete trajectories.

## Limitations & Future Work

- The capability discovery phase relies on GPT-4o for text generation in input fields and reverse task synthesis, introducing high execution costs.
- Absolute performance on WebArena still lags significantly behind GPT-4o Zero-Shot (16.25%), indicating that there is room for improvement in composite Web tasks using synthesized data.
- Primarily supports CLICK, TYPE, and SCROLL actions, failing to cover more complex interactive operations such as dragging and gestures.
- Data scale increases suffer from saturation, bounded by the intrinsic limitations of the base VLM and the quality of trajectories executed by GPT-4o.

## Related Work & Insights

- **GUI Agent Data:** AndroidControl (Li et al., 2024) provides human-annotated mobile trajectories; AgentTrek (Lai et al., 2024) utilizes predefined tasks to drive trajectory synthesis.
- **GUI Agent Systems:** M3A (Rawles et al., 2024) is a GPT-4o-based Android agent; CogAgent (Hong et al., 2024) is fine-tuned based on a VLM.
- **Reverse Task Synthesis Conceptualization:** Unlike Self-Instruct (Wang et al., 2023), which generates tasks directly from an LLM, OS-Genesis reverses synthesized tasks from real-world environmental interactions, aligning closer with actual GUI capabilities.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Utility | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Overall Rating | 8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)

</div>

<!-- RELATED:END -->
